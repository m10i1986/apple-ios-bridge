import subprocess
import os
import base64
from PIL import Image
import io
from typing import Optional, Dict
from app.config.settings import settings
from app.core.logging import logger
from app.utils.image_utils import ImageUtils
from app.utils.system_utils import SystemUtils

class ScreenshotService:
    """Service for screenshot capture with dynamic UDID support"""

    def __init__(self, udid: Optional[str] = None):
        self.udid = udid

    def set_udid(self, udid: str):
        """Set the UDID for this service instance"""
        self.udid = udid

    def capture_screenshot(
        self,
        quality: int = None,
        scale_factor: float = None,
        max_width: int = None,
        max_height: int = None,
    ) -> Optional[Dict[str, any]]:
        """Capture device screenshot.

        Resolution is adjusted by scale_factor first, then clamped to
        max_width / max_height while preserving the aspect ratio.
        Falls back to temp-file approach when stdout pipe is unavailable.
        """
        if not self.udid:
            logger.error("No UDID set for screenshot capture")
            return None

        if quality is None:
            quality = settings.STREAM_JPEG_QUALITY
        if scale_factor is None:
            scale_factor = settings.STREAM_SCALE_FACTOR
        if max_width is None:
            max_width = settings.STREAM_MAX_WIDTH
        if max_height is None:
            max_height = settings.STREAM_MAX_HEIGHT

        image_bytes: Optional[bytes] = None

        # --- Fast path: pipe PNG via /dev/stdout (no temp file I/O) ---
        try:
            cmd = ["idb", "screenshot", "--udid", self.udid, "/dev/stdout"]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=settings.SCREENSHOT_TIMEOUT,
            )
            if result.returncode == 0 and result.stdout:
                image_bytes = result.stdout
        except subprocess.TimeoutExpired:
            logger.debug(f"Screenshot stdout timeout for UDID: {self.udid}")
        except Exception as e:
            logger.debug(f"Screenshot stdout error for UDID {self.udid}: {e}")

        # --- Slow path: temp file fallback ---
        if not image_bytes:
            try:
                with SystemUtils.create_temp_file('.png') as tmp:
                    cmd2 = ["idb", "screenshot", "--udid", self.udid, tmp.name]
                    r2 = subprocess.run(
                        cmd2,
                        capture_output=True,
                        timeout=settings.SCREENSHOT_TIMEOUT,
                    )
                    if r2.returncode == 0 and os.path.exists(tmp.name):
                        with open(tmp.name, 'rb') as f:
                            image_bytes = f.read()
                        SystemUtils.cleanup_temp_file(tmp.name)
            except subprocess.TimeoutExpired:
                logger.debug(f"Screenshot temp-file timeout for UDID: {self.udid}")
            except Exception as e:
                logger.error(f"Screenshot temp-file error for UDID {self.udid}: {e}")

        if not image_bytes:
            return None

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                orig_w, orig_h = img.width, img.height
                target_w, target_h = orig_w, orig_h

                # 1. Scale factor
                if 0 < scale_factor < 1.0:
                    target_w = max(1, int(orig_w * scale_factor))
                    target_h = max(1, int(orig_h * scale_factor))

                # 2. Clamp to max_width
                if max_width > 0 and target_w > max_width:
                    ratio = max_width / target_w
                    target_w = max_width
                    target_h = max(1, int(target_h * ratio))

                # 3. Clamp to max_height
                if max_height > 0 and target_h > max_height:
                    ratio = max_height / target_h
                    target_h = max_height
                    target_w = max(1, int(target_w * ratio))

                if target_w != orig_w or target_h != orig_h:
                    # BILINEAR is faster than LANCZOS and acceptable for streaming
                    img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

                output = io.BytesIO()
                # optimize=False skips Huffman-table optimisation for lower latency
                img.save(output, format='JPEG', quality=quality, optimize=False)
                image_data = output.getvalue()

            return {
                "data": base64.b64encode(image_data).decode('utf-8'),
                "pixel_width": target_w,
                "pixel_height": target_h,
            }

        except Exception as e:
            logger.error(f"Screenshot encoding error for UDID {self.udid}: {e}")

        return None

    def capture_ultra_fast_screenshot(self) -> Optional[Dict[str, any]]:
        """Ultra-fast screenshot for real-time streaming"""
        return self.capture_screenshot()

    def capture_high_quality_screenshot(self) -> Optional[Dict[str, any]]:
        """High-quality screenshot (full resolution, high JPEG quality)"""
        return self.capture_screenshot(
            quality=settings.WEBRTC_HIGH_QUALITY,
            scale_factor=1.0,
            max_width=0,
            max_height=0,
        )
