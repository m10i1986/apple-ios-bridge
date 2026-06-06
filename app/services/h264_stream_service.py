"""H.264 fragmented MP4 broadcaster for iOS simulators.

Pipeline (single process, no external ffmpeg):
    xcrun simctl io <udid> recordVideo --codec=h264 --force <fifo>
        -> (FIFO, MOV container with an H.264 elementary stream)
    PyAV demux(MOV from FIFO) -> PyAV mux(fragmented MP4 to a capturing
    file-like) -> top-level MP4 boxes parsed and broadcast to WS clients.

This mirrors the proven IdbStreamService recordVideo pattern, which decodes
the very same simctl FIFO via PyAV reliably on this host.  The crucial detail
is that libav opens the FIFO read side natively in a reader THREAD (blocking
O_RDONLY) and probes the MOV container; simctl --force then connects the write
side.  Using an in-process PyAV reader (instead of a separate ffmpeg process)
avoids the FIFO inode race that made the ffmpeg-subprocess approach fail with
"Interrupted system call".

The packets are remuxed (bitstream copy, no transcode) into fragmented MP4 so
browsers/Electron can play them via Media Source Extensions:
    - Init segment: ftyp + moov  (cached, re-sent to each new client)
    - Media segments: moof + mdat  (broadcast to all connected clients)

This module is fully additive and does not modify existing capture paths
(VideoService / IdbStreamService / FastWebRTCService).
"""

import asyncio
import os
import signal
import subprocess
import threading
import time
from typing import Dict, Optional, Set

import av # type: ignore

from app.core.logging import logger


# Default MSE codec hint sent to clients.  simctl recordVideo --codec=h264
# typically produces High@4.0 or similar on modern simulators; this string is
# permissive enough for most browsers and Electron's Chromium build.
DEFAULT_MSE_MIME = 'video/mp4; codecs="avc1.640032"'

_FRAG_DURATION_US = 100_000  # 100ms fragments for low latency


class _Fmp4Writer:
    """Minimal write-only, non-seekable file-like for the PyAV MP4 muxer.

    Deliberately exposes NO ``seek``/``tell`` so libav treats the target as a
    non-seekable stream and emits a fragmented layout (moov first).  Every
    chunk libav writes is forwarded to the service's fMP4 box parser.
    """

    def __init__(self, service: "H264StreamService") -> None:
        self._service = service

    def write(self, data) -> int:
        b = bytes(data)
        self._service._feed_bytes(b)
        return len(b)

    def flush(self) -> None:
        pass


class H264StreamService:
    """Per-UDID fMP4 broadcaster.  Instances are managed by the WS handler."""

    START_TIMEOUT: float = 10.0  # seconds to wait for the init (moov) segment

    def __init__(self, udid: str) -> None:
        self.udid = udid

        self._simctl: Optional[subprocess.Popen] = None
        self._fifo_path: Optional[str] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

        # PyAV containers (managed by the reader thread).
        self._in_container = None
        self._out_container = None

        # fMP4 box-parser state (fed by the muxer's write callback).
        self._parse_buffer = b""
        self._init_accumulator = b""
        self._init_collected = False
        self._media_accumulator = b""

        self._lock = threading.Lock()
        self._init_segment: Optional[bytes] = None  # ftyp(+optional)+moov
        self._init_ready = threading.Event()

        self._clients_lock = threading.Lock()
        self._clients: Set = set()  # type: ignore[type-arg]
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop) -> bool:
        if self._running:
            return True
        self._loop = loop

        # Kill any stale simctl recordVideo processes for this UDID before
        # starting a new one.  Stale processes hold the recording lock, causing
        # the new simctl to wait indefinitely even with --force.
        try:
            result = subprocess.run(
                ["pkill", "-SIGINT", "-f",
                 f"simctl io {self.udid} recordVideo"],
                capture_output=True, timeout=3,
            )
            if result.returncode == 0:
                logger.info(
                    f"H264StreamService: killed stale simctl recording "
                    f"for {self.udid}, waiting for release..."
                )
                time.sleep(0.8)  # allow the lock to be released
        except Exception:
            pass

        # Create a named FIFO.  The reader thread (started next) calls
        # av.open(FIFO_PATH) so libav opens the read side natively (blocking
        # O_RDONLY) and probes the MOV container — the proven IdbStreamService
        # pattern.  simctl --force then connects the write side.
        fifo_path = f"/tmp/ios_bridge_h264_{self.udid}_{os.getpid()}.pipe"
        try:
            if os.path.exists(fifo_path):
                os.unlink(fifo_path)
            os.mkfifo(fifo_path, mode=0o600)
        except Exception as e:
            logger.error(f"H264StreamService: FIFO creation failed for {self.udid}: {e}")
            return False
        self._fifo_path = fifo_path

        self._running = True

        # Start the PyAV reader thread FIRST.  It blocks in av.open() until
        # simctl connects the FIFO write side, then remuxes packets into
        # fragmented MP4 (see _reader_loop).
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        # Start simctl recordVideo (writer).  --force overwrites a pre-existing
        # path; its write-side open rendezvous with the reader's av.open above.
        try:
            self._simctl = subprocess.Popen(
                ["xcrun", "simctl", "io", self.udid,
                 "recordVideo", "--codec=h264", "--force", fifo_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            logger.error(f"H264StreamService: simctl launch failed for {self.udid}: {e}")
            self.stop()
            return False

        threading.Thread(
            target=self._drain_stderr,
            args=(self._simctl.stderr, "simctl"),
            daemon=True,
        ).start()

        # Wait for the init segment so the first client gets it immediately.
        # Poll periodically to detect early process/reader exit and fail fast.
        deadline = time.monotonic() + self.START_TIMEOUT
        while not self._init_ready.is_set():
            if self._init_ready.wait(timeout=1.0):
                break
            if time.monotonic() >= deadline:
                logger.warning(
                    f"H264StreamService: init segment timeout for {self.udid} "
                    f"(simctl_exit={self._simctl.poll()})"
                )
                self.stop()
                return False
            simctl_rc = self._simctl.poll()
            reader_alive = self._reader_thread.is_alive()
            if simctl_rc is not None or not reader_alive:
                logger.error(
                    f"H264StreamService: process exited early for {self.udid} "
                    f"(simctl_exit={simctl_rc}, reader_alive={reader_alive})"
                )
                self.stop()
                return False

        logger.info(f"✅ H264StreamService started for {self.udid}")
        return True

    def stop(self) -> None:
        self._running = False
        self._cleanup_processes()
        if self._reader_thread:
            self._reader_thread.join(timeout=3)
            self._reader_thread = None
        logger.info(f"H264StreamService stopped for {self.udid}")

    def _cleanup_processes(self) -> None:
        # simctl needs SIGINT to release the host recording lock cleanly
        # (same constraint as IdbStreamService).
        if self._simctl:
            try:
                if self._simctl.poll() is None:
                    self._simctl.send_signal(signal.SIGINT)
                    try:
                        self._simctl.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._simctl.terminate()
                        self._simctl.wait(timeout=3)
            except Exception:
                try: self._simctl.kill()
                except Exception: pass
            self._simctl = None

        # Closing the input container unblocks the reader thread's demux loop.
        if self._out_container is not None:
            try:
                self._out_container.close()
            except Exception:
                pass
            self._out_container = None

        if self._in_container is not None:
            try:
                self._in_container.close()
            except Exception:
                pass
            self._in_container = None

        if self._fifo_path:
            try:
                os.unlink(self._fifo_path)
            except Exception:
                pass
            self._fifo_path = None

    # ------------------------------------------------------------------
    # Reader / broadcaster
    # ------------------------------------------------------------------

    def _drain_stderr(self, stream, name: str) -> None:
        try:
            for line in stream:
                text = line.decode(errors="replace").strip()
                if text:
                    logger.info(f"H264Stream {name}[{self.udid}]: {text}")
        except Exception:
            pass

    def _reader_loop(self) -> None:
        """Demux the simctl MOV (FIFO) with PyAV and remux to fragmented MP4.

        av.open(fifo_path) blocks on the FIFO read-side open until simctl
        connects the write side, then probes the MOV container — exactly the
        proven IdbStreamService path.  Packets are bitstream-copied (no
        transcode) into a fragmented MP4 muxer whose output is captured by
        ``_Fmp4Writer`` and parsed into MSE init/media segments.
        """
        try:
            self._in_container = av.open(
                self._fifo_path,
                mode="r",
                format="mov",
                options={
                    "fflags": "+nobuffer+discardcorrupt+igndts",
                    "flags": "+low_delay",
                    "probesize": "5000000",
                    "analyzeduration": "5000000",
                },
            )
            in_stream = self._in_container.streams.video[0]

            self._out_container = av.open(
                _Fmp4Writer(self),
                mode="w",
                format="mp4",
                options={
                    "movflags": "frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
                    "frag_duration": str(_FRAG_DURATION_US),
                },
            )
            out_stream = self._out_container.add_stream(template=in_stream)

            for packet in self._in_container.demux(in_stream):
                if not self._running:
                    break
                if packet.dts is None:
                    continue  # flush/incomplete packet
                packet.stream = out_stream
                self._out_container.mux(packet)
        except Exception as e:
            logger.error(f"H264StreamService reader error for {self.udid}: {e}")
        finally:
            logger.info(f"H264StreamService reader exited for {self.udid}")

    def _feed_bytes(self, chunk: bytes) -> None:
        """Parse fragmented-MP4 bytes emitted by the muxer into MSE segments.

        Called from the muxer's write callback (reader thread).  Caches the
        init segment (ftyp..moov) and broadcasts each moof+mdat media segment.
        """
        if not chunk:
            return
        self._parse_buffer += chunk

        while len(self._parse_buffer) >= 8:
            size = int.from_bytes(self._parse_buffer[:4], "big")
            box_type = self._parse_buffer[4:8].decode("ascii", errors="replace")

            # size==1 (64-bit) / size==0 (to EOF) do not occur in fragmented
            # MP4 with these flags, but guard against corruption anyway.
            if size < 8 or size > 64 * 1024 * 1024:
                logger.warning(
                    f"H264Stream[{self.udid}]: invalid box size {size} "
                    f"({box_type!r}); resyncing"
                )
                self._parse_buffer = b""
                return
            if size > len(self._parse_buffer):
                return  # need more data

            box_data = self._parse_buffer[:size]
            self._parse_buffer = self._parse_buffer[size:]

            if not self._init_collected:
                self._init_accumulator += box_data
                if box_type == "moov":
                    with self._lock:
                        self._init_segment = self._init_accumulator
                    self._init_collected = True
                    self._init_accumulator = b""
                    self._init_ready.set()
            else:
                # Pair each moof with its following mdat so MSE gets a complete
                # fragment in a single appendBuffer call.
                if box_type == "moof":
                    if self._media_accumulator:
                        self._broadcast(self._media_accumulator)
                    self._media_accumulator = box_data
                elif box_type == "mdat":
                    self._media_accumulator += box_data
                    self._broadcast(self._media_accumulator)
                    self._media_accumulator = b""
                else:
                    # styp / sidx / etc.  Prepend to the next pair or pass through.
                    if self._media_accumulator:
                        self._media_accumulator = box_data + self._media_accumulator
                    else:
                        self._broadcast(box_data)

    def _broadcast(self, data: bytes) -> None:
        if not self._loop or not data:
            return
        with self._clients_lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                asyncio.run_coroutine_threadsafe(ws.send_bytes(data), self._loop)
            except Exception as e:
                logger.debug(f"H264 broadcast error for {self.udid}: {e}")

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    async def add_client(self, websocket) -> bool:
        """Register a client and send the cached init segment to it."""
        with self._lock:
            init = self._init_segment
        if not init:
            return False
        try:
            await websocket.send_bytes(init)
        except Exception as e:
            logger.warning(f"H264 send init failed for {self.udid}: {e}")
            return False
        with self._clients_lock:
            self._clients.add(websocket)
        return True

    def remove_client(self, websocket) -> int:
        """Unregister a client. Returns remaining client count."""
        with self._clients_lock:
            self._clients.discard(websocket)
            return len(self._clients)

    @property
    def mime_type(self) -> str:
        return DEFAULT_MSE_MIME

    @property
    def is_running(self) -> bool:
        return self._running and (self._simctl is not None and self._simctl.poll() is None)


# ----------------------------------------------------------------------
# Per-UDID registry (kept here so the WS handler stays minimal)
# ----------------------------------------------------------------------

_registry_lock = threading.Lock()
_registry: Dict[str, H264StreamService] = {}


def get_or_start_service(udid: str, loop: asyncio.AbstractEventLoop, max_retries: int = 2) -> Optional[H264StreamService]:
    """Return a running service for the UDID, creating one if needed.

    Retries up to *max_retries* times on start failure (e.g. simctl warm-up
    delay causing the init-segment timeout to fire on the first attempt).
    """
    for attempt in range(max_retries + 1):
        with _registry_lock:
            svc = _registry.get(udid)
            if svc is not None and svc.is_running:
                return svc
            if svc is not None:
                # Stale entry; clean up before recreating.
                svc.stop()
                _registry.pop(udid, None)
            svc = H264StreamService(udid)
        if svc.start(loop):
            with _registry_lock:
                _registry[udid] = svc
            return svc
        if attempt < max_retries:
            logger.info(
                f"H264StreamService: retrying start for {udid} "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(2.0)
    return None


def release_service_if_idle(udid: str) -> None:
    """Stop and remove the service if no clients remain."""
    with _registry_lock:
        svc = _registry.get(udid)
        if svc is None:
            return
        with svc._clients_lock:
            if svc._clients:
                return
        _registry.pop(udid, None)
    svc.stop()
