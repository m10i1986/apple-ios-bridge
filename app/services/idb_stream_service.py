"""Persistent idb video-stream subprocess with MJPEG frame parsing.

Runs `idb video-stream --udid <udid> --format mjpeg --fps <fps>` as a
long-lived process and parses JPEG frames from its stdout.

Consumers call:
- get_latest_frame_b64()      → {"data": base64, "pixel_width": w, "pixel_height": h}
- get_latest_frame_ndarray()  → (np.ndarray[H,W,3], width, height)

This eliminates per-frame subprocess fork overhead that exists when using
`idb screenshot` for every frame.
"""

import base64
import io
import subprocess
import threading
import time
from typing import Optional, Dict, Tuple

import numpy as np
from PIL import Image

from app.config.settings import settings
from app.core.logging import logger

# JPEG stream markers
_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"
_READ_CHUNK = 65536  # 64 KB


class IdbStreamService:
    """Persistent idb video-stream MJPEG reader."""

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Start the idb video-stream subprocess and reader thread."""
        if self._running:
            return True

        try:
            cmd = [
                "idb", "video-stream",
                "--udid", self.udid,
                "--format", "mjpeg",
                "--fps", str(self.fps),
            ]
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            time.sleep(0.5)

            if self._process.poll() is not None:
                err = self._process.stderr.read().decode(errors="replace")
                logger.warning(
                    f"❌ idb video-stream exited immediately for {self.udid}: {err}"
                )
                self._process = None
                return False

            self._running = True
            self._thread = threading.Thread(
                target=self._reader_loop, daemon=True
            )
            self._thread.start()
            logger.info(
                f"✅ IdbStreamService started for {self.udid} @ {self.fps}fps"
            )
            return True

        except Exception as e:
            logger.error(f"❌ IdbStreamService start error for {self.udid}: {e}")
            self._process = None
            return False

    def stop(self) -> None:
        """Stop the subprocess and reader thread."""
        self._running = False

        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

        logger.info(f"🛑 IdbStreamService stopped for {self.udid}")

    @property
    def is_running(self) -> bool:
        return (
            self._running
            and self._process is not None
            and self._process.poll() is None
        )

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def get_latest_frame_b64(self) -> Optional[Dict]:
        """Return the latest frame as base64-encoded JPEG with dimensions.

        Returns None if no frame has been received yet.
        """
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
        """Return the latest frame as (rgb_ndarray, width, height).

        Returns None if no frame has been received yet.
        """
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
    # Internal reader loop
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Background thread: read stdout and parse MJPEG frames."""
        logger.info(f"🎬 IdbStreamService reader started for {self.udid}")

        buf = b""
        last_log = time.monotonic()
        local_count = 0

        try:
            while self._running and self._process and self._process.poll() is None:
                chunk = self._process.stdout.read(_READ_CHUNK)
                if not chunk:
                    break

                buf += chunk

                # Parse all complete JPEG frames currently in buffer
                while True:
                    soi = buf.find(_SOI)
                    if soi == -1:
                        buf = b""
                        break

                    eoi = buf.find(_EOI, soi + 2)
                    if eoi == -1:
                        # Incomplete frame – retain from SOI onward
                        buf = buf[soi:]
                        break

                    eoi += 2  # include the two EOI bytes
                    jpeg = buf[soi:eoi]
                    buf = buf[eoi:]

                    # Decode dimensions (fast: read header only)
                    try:
                        with Image.open(io.BytesIO(jpeg)) as img:
                            w, h = img.width, img.height
                    except Exception:
                        continue

                    with self._lock:
                        self._latest_jpeg = jpeg
                        self._latest_width = w
                        self._latest_height = h
                        self._frame_count += 1

                    local_count += 1
                    now = time.monotonic()
                    if now - last_log >= 15.0:
                        fps = local_count / (now - last_log)
                        logger.info(
                            f"📊 IdbStream {self.udid}: {fps:.1f}fps, "
                            f"size={w}x{h}, total={self._frame_count}"
                        )
                        last_log = now
                        local_count = 0

        except Exception as e:
            logger.error(f"IdbStreamService reader error for {self.udid}: {e}")
        finally:
            self._running = False
            logger.info(f"🛑 IdbStreamService reader stopped for {self.udid}")
