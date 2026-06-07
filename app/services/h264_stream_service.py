"""Screenshot-based screen streaming for iOS simulators.

Background: on this simulator host neither ``simctl recordVideo`` nor
``idb video-stream`` produces a usable real-time stream.  ``simctl recordVideo``
buffers the whole movie and only writes it on SIGINT (so nothing streams while
recording), and ``idb video-stream`` yields zero bytes on this environment
(iOS 17.x simulator, x86_64).  The only reliable real-time capture is

    idb screenshot --udid <udid> -

which returns exactly one PNG on stdout per invocation.  This service runs that
capture in a tight background loop and broadcasts each frame to all registered
WebSocket clients as a JSON message carrying a base64-encoded image.

The output image format is selectable per stream:
    - "jpeg" (default): smaller, lossy, lower latency / bandwidth.
    - "png":            lossless, larger.

This module keeps the public surface used by the ``video_h264`` WebSocket
handler (``get_or_start_service`` / ``release_service_if_idle`` and the
``add_client`` / ``remove_client`` instance methods) so the endpoint wiring is
unchanged.
"""

import asyncio
import base64
import io
import json
import subprocess
import threading
import time
from typing import Dict, Optional, Set

from PIL import Image

from app.config.settings import settings
from app.core.logging import logger


VALID_FORMATS = ("jpeg", "png")
DEFAULT_FORMAT = "jpeg"


class ScreenStreamService:
    """Per-UDID screenshot-loop broadcaster.

    A background thread captures ``idb screenshot -`` frames, scales/encodes
    each to the configured format, and pushes them to every registered WS
    client.  One instance per UDID is shared by all clients via the registry
    below.
    """

    START_TIMEOUT: float = 10.0      # seconds to wait for the first frame
    CAPTURE_INTERVAL: float = 0.05   # minimum seconds between capture attempts
    MAX_CONSECUTIVE_FAILURES: int = 10

    def __init__(self, udid: str, fmt: str = DEFAULT_FORMAT) -> None:
        self.udid = udid
        self.fmt = fmt if fmt in VALID_FORMATS else DEFAULT_FORMAT

        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._first_frame = threading.Event()

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
        self._running = True

        self._reader_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._reader_thread.start()

        # Wait for the first successful frame so the first client gets data.
        deadline = time.monotonic() + self.START_TIMEOUT
        while not self._first_frame.is_set():
            if self._first_frame.wait(timeout=1.0):
                break
            if time.monotonic() >= deadline:
                logger.warning(
                    f"ScreenStreamService: first-frame timeout for {self.udid}"
                )
                self.stop()
                return False
            if self._reader_thread is None or not self._reader_thread.is_alive():
                logger.error(
                    f"ScreenStreamService: capture thread died for {self.udid}"
                )
                self.stop()
                return False

        logger.info(
            f"✅ ScreenStreamService started for {self.udid} (format={self.fmt})"
        )
        return True

    def stop(self) -> None:
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=3)
            self._reader_thread = None
        logger.info(f"ScreenStreamService stopped for {self.udid}")

    @property
    def is_running(self) -> bool:
        return (
            self._running
            and self._reader_thread is not None
            and self._reader_thread.is_alive()
        )

    # ------------------------------------------------------------------
    # Capture / encode
    # ------------------------------------------------------------------

    def _capture_once(self) -> Optional[bytes]:
        """Run ``idb screenshot -`` once and return raw PNG bytes (or None)."""
        try:
            r = subprocess.run(
                ["idb", "screenshot", "--udid", self.udid, "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=settings.SCREENSHOT_TIMEOUT,
            )
            if r.returncode == 0 and r.stdout:
                return r.stdout
        except subprocess.TimeoutExpired:
            logger.debug(f"ScreenStreamService: capture timeout for {self.udid}")
        except Exception as e:
            logger.debug(f"ScreenStreamService: capture error for {self.udid}: {e}")
        return None

    def _encode(self, png_bytes: bytes) -> Optional[Dict]:
        """Scale per stream settings and encode to the configured format."""
        try:
            with Image.open(io.BytesIO(png_bytes)) as img:
                if self.fmt == "jpeg":
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                elif img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA")

                orig_w, orig_h = img.width, img.height
                target_w, target_h = orig_w, orig_h

                scale = settings.STREAM_SCALE_FACTOR
                if 0 < scale < 1.0:
                    target_w = max(1, int(orig_w * scale))
                    target_h = max(1, int(orig_h * scale))

                max_w = settings.STREAM_MAX_WIDTH
                if max_w > 0 and target_w > max_w:
                    ratio = max_w / target_w
                    target_w = max_w
                    target_h = max(1, int(target_h * ratio))

                max_h = settings.STREAM_MAX_HEIGHT
                if max_h > 0 and target_h > max_h:
                    ratio = max_h / target_h
                    target_h = max_h
                    target_w = max(1, int(target_w * ratio))

                if (target_w, target_h) != (orig_w, orig_h):
                    img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

                out = io.BytesIO()
                if self.fmt == "jpeg":
                    img.save(
                        out,
                        format="JPEG",
                        quality=settings.STREAM_JPEG_QUALITY,
                        optimize=False,
                    )
                else:
                    # compress_level=1: fastest PNG encode (low latency).
                    img.save(out, format="PNG", compress_level=1)

                return {
                    "data": base64.b64encode(out.getvalue()).decode("utf-8"),
                    "pixel_width": target_w,
                    "pixel_height": target_h,
                }
        except Exception as e:
            logger.debug(f"ScreenStreamService: encode error for {self.udid}: {e}")
            return None

    def _capture_loop(self) -> None:
        consecutive_failures = 0
        while self._running:
            t0 = time.monotonic()

            png = self._capture_once()
            if not png:
                consecutive_failures += 1
                if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        f"ScreenStreamService: too many capture failures "
                        f"for {self.udid}, stopping"
                    )
                    break
                time.sleep(0.2)
                continue
            consecutive_failures = 0

            frame = self._encode(png)
            if frame is None:
                continue

            if not self._first_frame.is_set():
                self._first_frame.set()
            self._broadcast(frame)

            # Pace the loop so we don't spin faster than CAPTURE_INTERVAL.
            elapsed = time.monotonic() - t0
            if elapsed < self.CAPTURE_INTERVAL:
                time.sleep(self.CAPTURE_INTERVAL - elapsed)

        logger.info(f"ScreenStreamService capture loop exited for {self.udid}")

    # ------------------------------------------------------------------
    # Broadcast / client management
    # ------------------------------------------------------------------

    def _broadcast(self, frame: Dict) -> None:
        if not self._loop:
            return
        payload = json.dumps({
            "type": "frame",
            "format": self.fmt,
            "data": frame["data"],
            "pixel_width": frame["pixel_width"],
            "pixel_height": frame["pixel_height"],
        })
        with self._clients_lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                asyncio.run_coroutine_threadsafe(ws.send_text(payload), self._loop)
            except Exception as e:
                logger.debug(f"ScreenStream broadcast error for {self.udid}: {e}")

    async def add_client(self, websocket) -> bool:
        """Register a client.  Requires the first frame to be available."""
        if not self._first_frame.is_set():
            return False
        with self._clients_lock:
            self._clients.add(websocket)
        return True

    def remove_client(self, websocket) -> int:
        """Unregister a client.  Returns remaining client count."""
        with self._clients_lock:
            self._clients.discard(websocket)
            return len(self._clients)


# ----------------------------------------------------------------------
# Per-UDID registry (kept here so the WS handler stays minimal)
# ----------------------------------------------------------------------

_registry_lock = threading.Lock()
_registry: Dict[str, ScreenStreamService] = {}


def get_or_start_service(
    udid: str,
    loop: asyncio.AbstractEventLoop,
    fmt: str = DEFAULT_FORMAT,
    max_retries: int = 2,
) -> Optional[ScreenStreamService]:
    """Return a running service for the UDID, creating one if needed.

    If an existing service uses a different image format, it is restarted with
    the requested format.  Retries up to *max_retries* times on start failure.
    """
    fmt = fmt if fmt in VALID_FORMATS else DEFAULT_FORMAT
    for attempt in range(max_retries + 1):
        with _registry_lock:
            svc = _registry.get(udid)
            if svc is not None and svc.is_running and svc.fmt == fmt:
                return svc
            if svc is not None:
                # Stale entry or format change; clean up before recreating.
                svc.stop()
                _registry.pop(udid, None)
            svc = ScreenStreamService(udid, fmt)
        if svc.start(loop):
            with _registry_lock:
                _registry[udid] = svc
            return svc
        if attempt < max_retries:
            logger.info(
                f"ScreenStreamService: retrying start for {udid} "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(1.0)
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
