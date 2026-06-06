"""H.264 fragmented MP4 broadcaster for iOS simulators.

Pipeline:
    xcrun simctl io <udid> recordVideo --codec=h264 --force <fifo>
        -> (FIFO, MOV container with H.264 elementary stream)
    ffmpeg -i <fifo> -c copy -movflags +frag_keyframe+empty_moov+default_base_moof
           -f mp4 pipe:1
        -> (fragmented MP4 on stdout; remux only, no transcode)

The reader thread parses top-level MP4 boxes:
    - Init segment: ftyp + moov  (cached, re-sent to each new client)
    - Media segments: moof + mdat  (broadcast to all connected clients)

This module is fully additive and does not modify existing capture paths
(VideoService / IdbStreamService / FastWebRTCService).
"""

import asyncio
import os
import shutil
import signal
import subprocess
import threading
from typing import Dict, Optional, Set

from app.core.logging import logger


def _resolve_ffmpeg() -> str:
    """Return the ffmpeg executable path.

    Preference order:
    1. System PATH (``which ffmpeg``)
    2. imageio-ffmpeg bundled binary
    3. Bare ``'ffmpeg'`` (raises clear error at runtime if missing)
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg  # type: ignore[import]
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return "ffmpeg"


_FFMPEG_EXE = _resolve_ffmpeg()


# Default MSE codec hint sent to clients.  simctl recordVideo --codec=h264
# typically produces High@4.0 or similar on modern simulators; this string is
# permissive enough for most browsers and Electron's Chromium build.
DEFAULT_MSE_MIME = 'video/mp4; codecs="avc1.640032"'

_FRAG_DURATION_US = 100_000  # 100ms fragments for low latency
_READ_CHUNK = 65536


class H264StreamService:
    """Per-UDID fMP4 broadcaster.  Instances are managed by the WS handler."""

    START_TIMEOUT: float = 8.0  # seconds to wait for the init (moov) segment

    def __init__(self, udid: str) -> None:
        self.udid = udid

        self._simctl: Optional[subprocess.Popen] = None
        self._ffmpeg: Optional[subprocess.Popen] = None
        self._fifo_path: Optional[str] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

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

        fifo_path = f"/tmp/ios_bridge_h264_{self.udid}_{os.getpid()}.pipe"
        try:
            if os.path.exists(fifo_path):
                os.unlink(fifo_path)
            os.mkfifo(fifo_path, mode=0o600)
        except Exception as e:
            logger.error(f"H264StreamService: FIFO creation failed for {self.udid}: {e}")
            return False
        self._fifo_path = fifo_path

        # Start ffmpeg (reader).  open(FIFO, O_RDONLY) blocks until simctl
        # opens the write side.  Popen returns immediately regardless.
        try:
            self._ffmpeg = subprocess.Popen(
                [
                    _FFMPEG_EXE,
                    "-loglevel", "error",
                    "-fflags", "+nobuffer+discardcorrupt+igndts",
                    "-flags", "+low_delay",
                    "-probesize", "5000000",
                    "-analyzeduration", "5000000",
                    "-i", fifo_path,
                    "-c:v", "copy",
                    "-an",
                    "-movflags", "+frag_keyframe+empty_moov+default_base_moof+omit_tfhd_offset",
                    "-frag_duration", str(_FRAG_DURATION_US),
                    "-f", "mp4",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception as e:
            logger.error(f"H264StreamService: ffmpeg launch failed for {self.udid}: {e}")
            self._cleanup_processes()
            return False

        # Start simctl (writer).  Blocks open() until ffmpeg's reader is ready.
        try:
            self._simctl = subprocess.Popen(
                ["xcrun", "simctl", "io", self.udid,
                 "recordVideo", "--codec=h264", "--force", fifo_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            logger.error(f"H264StreamService: simctl launch failed for {self.udid}: {e}")
            self._cleanup_processes()
            return False

        self._running = True

        # Start reader and stderr-drain threads
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        threading.Thread(
            target=self._drain_stderr,
            args=(self._simctl.stderr, "simctl"),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._drain_stderr,
            args=(self._ffmpeg.stderr, "ffmpeg"),
            daemon=True,
        ).start()

        # Wait for the init segment so the first client gets it immediately.
        if not self._init_ready.wait(timeout=self.START_TIMEOUT):
            logger.warning(
                f"H264StreamService: init segment timeout for {self.udid} "
                f"(simctl_exit={self._simctl.poll()}, ffmpeg_exit={self._ffmpeg.poll()})"
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

        if self._ffmpeg:
            try:
                if self._ffmpeg.poll() is None:
                    self._ffmpeg.terminate()
                    try:
                        self._ffmpeg.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._ffmpeg.kill()
                        self._ffmpeg.wait(timeout=2)
            except Exception:
                pass
            self._ffmpeg = None

        if self._fifo_path:
            try: os.unlink(self._fifo_path)
            except Exception: pass
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
        """Parse top-level MP4 boxes from ffmpeg stdout and broadcast."""
        try:
            assert self._ffmpeg and self._ffmpeg.stdout
            stdout = self._ffmpeg.stdout

            buffer = b""
            init_accumulator = b""
            init_collected = False
            # Accumulate a media segment as moof + mdat so MSE gets a complete
            # fragment in a single appendBuffer call.
            media_accumulator = b""

            while self._running:
                chunk = stdout.read(_READ_CHUNK)
                if not chunk:
                    break
                buffer += chunk

                while len(buffer) >= 8:
                    size = int.from_bytes(buffer[:4], "big")
                    box_type = buffer[4:8].decode("ascii", errors="replace")

                    # size==1 means 64-bit large size follows; size==0 means
                    # extends to EOF.  Neither occurs in fragmented MP4
                    # produced by ffmpeg with these flags, but guard anyway.
                    if size < 8 or size > 64 * 1024 * 1024:
                        logger.warning(
                            f"H264Stream[{self.udid}]: invalid box size {size} "
                            f"({box_type!r}); resyncing"
                        )
                        buffer = b""
                        break
                    if size > len(buffer):
                        break  # need more data

                    box_data = buffer[:size]
                    buffer = buffer[size:]

                    if not init_collected:
                        init_accumulator += box_data
                        if box_type == "moov":
                            with self._lock:
                                self._init_segment = init_accumulator
                            init_collected = True
                            init_accumulator = b""
                            self._init_ready.set()
                    else:
                        # Pair each moof with its following mdat before sending
                        if box_type == "moof":
                            # Flush any orphaned mdat-less leftover (shouldn't
                            # happen but be defensive)
                            if media_accumulator:
                                self._broadcast(media_accumulator)
                            media_accumulator = box_data
                        elif box_type == "mdat":
                            media_accumulator += box_data
                            self._broadcast(media_accumulator)
                            media_accumulator = b""
                        else:
                            # styp / sidx / etc.  Pass through standalone.
                            if media_accumulator:
                                # Prepend to the next pair (e.g. styp before moof)
                                media_accumulator = box_data + media_accumulator
                            else:
                                self._broadcast(box_data)
        except Exception as e:
            logger.error(f"H264StreamService reader error for {self.udid}: {e}")
        finally:
            logger.info(f"H264StreamService reader exited for {self.udid}")

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
        return self._running and (self._ffmpeg is not None and self._ffmpeg.poll() is None)


# ----------------------------------------------------------------------
# Per-UDID registry (kept here so the WS handler stays minimal)
# ----------------------------------------------------------------------

_registry_lock = threading.Lock()
_registry: Dict[str, H264StreamService] = {}


def get_or_start_service(udid: str, loop: asyncio.AbstractEventLoop) -> Optional[H264StreamService]:
    """Return a running service for the UDID, creating one if needed."""
    with _registry_lock:
        svc = _registry.get(udid)
        if svc is not None and svc.is_running:
            return svc
        if svc is not None:
            # Stale entry; clean up before recreating.
            svc.stop()
            _registry.pop(udid, None)
        svc = H264StreamService(udid)
    if not svc.start(loop):
        return None
    with _registry_lock:
        _registry[udid] = svc
    return svc


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
