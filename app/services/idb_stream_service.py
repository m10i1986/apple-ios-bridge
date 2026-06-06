"""Fast video stream service for iOS simulators.

Capture priority:
1. xcrun simctl io <udid> recordVideo --codec=h264 <fifo> + PyAV decode  (native ~30fps)
2. xcrun simctl io <udid> screenshot <tmpfile> loop                       (~10fps)
3. idb screenshot --udid <udid> /dev/stdout loop                         (~5fps)

Note: xcrun simctl io screenshot treats "-" as a literal filename on some macOS
versions, not stdout, so the screenshot stdout variant is not used.
"""

import base64
import fcntl
import io
import os
import subprocess
import tempfile
import threading
import time
from typing import Optional, Dict, Tuple

import av
import numpy as np
from PIL import Image

from app.config.settings import settings
from app.core.logging import logger

# JPEG stream markers (kept for compatibility)
_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"
_READ_CHUNK = 65536  # 64 KB

# Capture mode constants
_MODE_RECORD      = "recordVideo"
_MODE_SIMCTL_FILE = "simctl_file"
_MODE_IDB         = "idb"


class IdbStreamService:
    """Fast video stream for iOS simulators.

    Tries recordVideo (FIFO + PyAV) first for native frame rate,
    then falls back to screenshot capture loops.
    """

    # If recordVideo mode: seconds to wait for "Recording started" + first frame
    RECORD_START_TIMEOUT: float = 6.0
    # If screenshot mode: seconds to wait for first frame
    FIRST_FRAME_TIMEOUT: float = 8.0

    def __init__(self, udid: str, fps: int = None) -> None:
        self.udid = udid
        self.fps = fps or settings.DEFAULT_VIDEO_FPS

        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_width: int = 0
        self._latest_height: int = 0
        self._frame_count: int = 0
        self._start_time: float = 0.0

        self._mode: Optional[str] = None
        self._fifo_path: Optional[str] = None
        self._av_container = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _try_record_mode(self) -> bool:
        """Try recordVideo + PyAV via a named FIFO. Returns True on success.

        Race-condition-free sequence:
          1. Open FIFO read end with O_NONBLOCK so it succeeds immediately
             (no blocking wait for a writer).  This ensures the read end exists
             before recordVideo tries to open the write end.
          2. Switch back to blocking mode via fcntl so PyAV reads block normally.
          3. Launch recordVideo – write-side open succeeds because read end is
             already established (even for O_WRONLY|O_NONBLOCK writes).
          4. Pass the pre-opened fd to PyAV in a background thread and wait for
             the first decoded frame as the success criterion.
        """
        fifo_path = f"/tmp/ios_bridge_{self.udid}_{os.getpid()}.pipe"
        try:
            if os.path.exists(fifo_path):
                os.unlink(fifo_path)
            os.mkfifo(fifo_path, mode=0o600)
        except Exception as e:
            logger.info(f"IdbStreamService: FIFO creation failed for {self.udid}: {e}")
            return False

        # Open read end immediately with O_NONBLOCK so it does not block waiting
        # for a writer.  recordVideo's write-side open will then succeed at once.
        try:
            rd_fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
            fl = fcntl.fcntl(rd_fd, fcntl.F_GETFL)
            fcntl.fcntl(rd_fd, fcntl.F_SETFL, fl & ~os.O_NONBLOCK)  # restore blocking
        except Exception as e:
            logger.info(f"IdbStreamService: FIFO read-open failed for {self.udid}: {e}")
            try: os.unlink(fifo_path)
            except Exception: pass
            return False

        # Launch recordVideo – read end is already open so write open succeeds.
        try:
            proc = subprocess.Popen(
                ["xcrun", "simctl", "io", self.udid,
                 "recordVideo", "--codec=h264", "--force", fifo_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            logger.info(f"IdbStreamService: recordVideo launch failed for {self.udid}: {e}")
            try: os.close(rd_fd)
            except Exception: pass
            try: os.unlink(fifo_path)
            except Exception: pass
            return False

        # Drain stderr so the pipe does not block recordVideo.
        def _drain_stderr():
            try:
                for line in proc.stderr:
                    text = line.decode(errors="replace").strip()
                    logger.debug(f"simctl recordVideo: {text}")
            except Exception:
                pass

        threading.Thread(target=_drain_stderr, daemon=True).start()

        # Open PyAV using the pre-opened fd (wrapped as a file object).
        av_result: Dict = {}
        av_ready = threading.Event()

        def _open_av():
            file_obj = None
            try:
                file_obj = os.fdopen(rd_fd, "rb")
                container = av.open(
                    file_obj,
                    options={
                        "fflags": "+nobuffer+discardcorrupt",
                        "flags": "+low_delay",
                        "probesize": "131072",
                        "analyzeduration": "1000000",
                    },
                )
                for frame in container.decode(video=0):
                    av_result["container"] = container
                    av_result["first_frame"] = frame
                    break
            except Exception as e:
                av_result["error"] = e
                logger.info(f"IdbStreamService: PyAV open error for {self.udid}: {e}")
                if file_obj is not None:
                    try: file_obj.close()
                    except Exception: pass
                else:
                    try: os.close(rd_fd)
                    except Exception: pass
            finally:
                av_ready.set()

        av_thread = threading.Thread(target=_open_av, daemon=True)
        av_thread.start()

        # Wait for first decoded frame (primary success criterion).
        if not av_ready.wait(timeout=self.RECORD_START_TIMEOUT):
            logger.info(
                f"IdbStreamService: recordVideo first-frame timeout for {self.udid}"
                f" (proc_exit={proc.poll()})"
            )
            proc.terminate()
            try: os.unlink(fifo_path)
            except Exception: pass
            return False

        if "error" in av_result or "container" not in av_result:
            logger.info(
                f"IdbStreamService: recordVideo PyAV failed for {self.udid}:"
                f" {av_result.get('error')}"
            )
            proc.terminate()
            try: os.unlink(fifo_path)
            except Exception: pass
            return False

        if proc.poll() is not None:
            logger.info(f"IdbStreamService: recordVideo exited early for {self.udid}")
            try: os.unlink(fifo_path)
            except Exception: pass
            return False

        first_frame = av_result["first_frame"]
        self._store_av_frame(first_frame)

        self._process = proc
        self._fifo_path = fifo_path
        self._av_container = av_result["container"]
        logger.info(f"✅ IdbStreamService: recordVideo+PyAV available for {self.udid}")
        return True

    def _probe_screenshot(self) -> Optional[str]:
        """Probe which screenshot method is available. Returns mode string or None."""

        # 1) xcrun simctl io screenshot <tmpfile>
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            r = subprocess.run(
                ["xcrun", "simctl", "io", self.udid, "screenshot", tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=4.0,
            )
            if r.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                os.unlink(tmp_path)
                logger.info(f"✅ IdbStreamService: simctl screenshot available for {self.udid}")
                return _MODE_SIMCTL_FILE
            try: os.unlink(tmp_path)
            except Exception: pass
        except Exception as e:
            logger.debug(f"simctl screenshot probe failed for {self.udid}: {e}")

        # 2) idb screenshot /dev/stdout
        try:
            r = subprocess.run(
                ["idb", "screenshot", "--udid", self.udid, "/dev/stdout"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=4.0,
            )
            if r.returncode == 0 and r.stdout:
                logger.info(f"✅ IdbStreamService: idb screenshot available for {self.udid}")
                return _MODE_IDB
        except Exception as e:
            logger.debug(f"idb screenshot probe failed for {self.udid}: {e}")

        return None

    def start(self) -> bool:
        """Start the stream. Tries recordVideo first, then screenshot loop."""
        if self._running:
            return True

        # Try recordVideo + PyAV (best quality and frame rate)
        if self._try_record_mode():
            self._mode = _MODE_RECORD
        else:
            # Fall back to screenshot loop
            mode = self._probe_screenshot()
            if mode is None:
                logger.warning(f"❌ IdbStreamService: no capture method available for {self.udid}")
                return False
            self._mode = mode

        self._running = True
        self._start_time = time.monotonic()

        target = (
            self._reader_loop_record
            if self._mode == _MODE_RECORD
            else self._reader_loop_screenshot
        )
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()
        logger.info(f"✅ IdbStreamService started for {self.udid} @ {self.fps}fps (mode={self._mode})")
        return True

    def stop(self) -> None:
        """Stop the stream and clean up resources."""
        self._running = False

        if self._av_container:
            try:
                self._av_container.close()
            except Exception:
                pass
            self._av_container = None

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try: self._process.kill()
                except Exception: pass
            self._process = None

        if self._fifo_path:
            try: os.unlink(self._fifo_path)
            except Exception: pass
            self._fifo_path = None

        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

        logger.info(f"🛑 IdbStreamService stopped for {self.udid}")

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def first_frame_timed_out(self) -> bool:
        """True if started but no frame arrived within the timeout."""
        if self._frame_count > 0:
            return False
        if self._start_time == 0.0:
            return False
        timeout = (
            self.RECORD_START_TIMEOUT
            if self._mode == _MODE_RECORD
            else self.FIRST_FRAME_TIMEOUT
        )
        return time.monotonic() - self._start_time > timeout

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def get_latest_frame_b64(self) -> Optional[Dict]:
        """Return the latest frame as base64-encoded JPEG with dimensions."""
        with self._lock:
            if self._latest_jpeg is None:
                return None
            jpeg = self._latest_jpeg
            w, h = self._latest_width, self._latest_height
        return {
            "data": base64.b64encode(jpeg).decode("utf-8"),
            "pixel_width": w,
            "pixel_height": h,
        }

    def get_latest_frame_ndarray(self) -> Optional[Tuple[np.ndarray, int, int]]:
        """Return the latest frame as (rgb_ndarray, width, height)."""
        with self._lock:
            if self._latest_jpeg is None:
                return None
            jpeg = self._latest_jpeg
            w, h = self._latest_width, self._latest_height
        try:
            with Image.open(io.BytesIO(jpeg)) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return np.array(img, dtype=np.uint8), img.width, img.height
        except Exception as e:
            logger.debug(f"IdbStreamService ndarray decode error: {e}")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_av_frame(self, frame: "av.VideoFrame") -> None:
        """Convert an av.VideoFrame to JPEG and store it."""
        try:
            img_array = frame.to_ndarray(format="rgb24")
            h, w = img_array.shape[:2]
            with Image.fromarray(img_array) as img:
                buf = io.BytesIO()
                img.save(buf, format="JPEG",
                         quality=settings.STREAM_JPEG_QUALITY,
                         optimize=False)
                jpeg = buf.getvalue()
            with self._lock:
                self._latest_jpeg = jpeg
                self._latest_width = w
                self._latest_height = h
                self._frame_count += 1
        except Exception as e:
            logger.debug(f"IdbStreamService av frame store error: {e}")

    def _capture_screenshot_raw(self) -> Optional[bytes]:
        """Capture one screenshot PNG using the probed screenshot mode."""
        try:
            if self._mode == _MODE_SIMCTL_FILE:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    r = subprocess.run(
                        ["xcrun", "simctl", "io", self.udid, "screenshot", tmp_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=settings.SCREENSHOT_TIMEOUT,
                    )
                    if r.returncode == 0 and os.path.exists(tmp_path):
                        with open(tmp_path, "rb") as f:
                            return f.read()
                finally:
                    try: os.unlink(tmp_path)
                    except Exception: pass

            elif self._mode == _MODE_IDB:
                r = subprocess.run(
                    ["idb", "screenshot", "--udid", self.udid, "/dev/stdout"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=settings.SCREENSHOT_TIMEOUT,
                )
                if r.returncode == 0 and r.stdout:
                    return r.stdout

        except subprocess.TimeoutExpired:
            logger.debug(f"IdbStreamService screenshot timeout for {self.udid}")
        except Exception as e:
            logger.debug(f"IdbStreamService screenshot error for {self.udid}: {e}")
        return None

    # ------------------------------------------------------------------
    # Reader loops
    # ------------------------------------------------------------------

    def _reader_loop_record(self) -> None:
        """Background thread: decode av frames from recordVideo FIFO."""
        logger.info(f"🎬 IdbStreamService recordVideo reader started for {self.udid}")

        last_log = time.monotonic()
        local_count = 0

        try:
            # _av_container already has the first frame consumed in probe.
            # Continue decoding from the same container.
            for frame in self._av_container.decode(video=0):
                if not self._running:
                    break

                self._store_av_frame(frame)
                local_count += 1

                now = time.monotonic()
                if now - last_log >= 15.0:
                    fps_actual = local_count / (now - last_log)
                    with self._lock:
                        w, h = self._latest_width, self._latest_height
                    logger.info(
                        f"📊 IdbStream {self.udid}: {fps_actual:.1f}fps "
                        f"(recordVideo), size={w}x{h}, total={self._frame_count}"
                    )
                    last_log = now
                    local_count = 0

        except Exception as e:
            if self._running:
                logger.error(f"IdbStreamService recordVideo reader error for {self.udid}: {e}")
        finally:
            self._running = False
            logger.info(f"🛑 IdbStreamService reader stopped for {self.udid}")

    def _reader_loop_screenshot(self) -> None:
        """Background thread: capture screenshots at target FPS."""
        logger.info(f"🎬 IdbStreamService screenshot reader started for {self.udid}")

        last_log = time.monotonic()
        local_count = 0
        next_frame_time = time.monotonic()

        try:
            while self._running:
                now = time.monotonic()
                frame_interval = 1.0 / max(1, self.fps)

                if now < next_frame_time:
                    sleep_time = next_frame_time - now
                    if sleep_time > 0.001:
                        time.sleep(sleep_time)
                    continue

                raw = self._capture_screenshot_raw()
                if raw:
                    try:
                        with Image.open(io.BytesIO(raw)) as img:
                            w, h = img.width, img.height
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG",
                                     quality=settings.STREAM_JPEG_QUALITY,
                                     optimize=False)
                            jpeg = buf.getvalue()
                        with self._lock:
                            self._latest_jpeg = jpeg
                            self._latest_width = w
                            self._latest_height = h
                            self._frame_count += 1
                        local_count += 1
                    except Exception as e:
                        logger.debug(f"IdbStreamService screenshot decode error: {e}")

                next_frame_time += frame_interval
                if next_frame_time < time.monotonic():
                    next_frame_time = time.monotonic() + frame_interval

                now = time.monotonic()
                if now - last_log >= 15.0:
                    fps_actual = local_count / (now - last_log)
                    with self._lock:
                        w, h = self._latest_width, self._latest_height
                    logger.info(
                        f"📊 IdbStream {self.udid}: {fps_actual:.1f}fps "
                        f"(screenshot), size={w}x{h}, total={self._frame_count}"
                    )
                    last_log = now
                    local_count = 0

        except Exception as e:
            logger.error(f"IdbStreamService screenshot reader error for {self.udid}: {e}")
        finally:
            self._running = False
            logger.info(f"🛑 IdbStreamService reader stopped for {self.udid}")
