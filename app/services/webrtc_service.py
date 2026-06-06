import threading
import time
import asyncio
import uuid
import base64
import io
from typing import Dict, Optional
from queue import Queue, Empty
import av
import numpy as np
from PIL import Image
from fractions import Fraction
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate

from app.core.logging import logger
from app.config.settings import settings
from app.services.screenshot_service import ScreenshotService

class SimpleVideoTrack(VideoStreamTrack):
    """Stable video track with consistent frame timing to prevent flickering"""

    def __init__(self, service, target_fps=None):
        if target_fps is None:
            target_fps = settings.WEBRTC_FPS
        super().__init__()
        self.service = service
        self.frame_count = 0
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.start_time = time.time()
        self.last_frame = None
        logger.info(f"🎬 VideoTrack initialized: {target_fps}fps, interval: {self.frame_interval:.3f}s")

    async def recv(self):
        """Generate stable video frames with consistent timing"""
        # Calculate target timestamp for this frame
        target_time = self.start_time + (self.frame_count * self.frame_interval)
        current_time = time.time()

        # Wait until it's time for the next frame
        wait_time = target_time - current_time
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        # Try to get fresh frame
        frame = await self.service.get_next_frame()

        # If no new frame available, reuse last frame to prevent flickering
        if frame is None and self.last_frame is not None:
            # Create new frame from last frame's data
            frame_array = self.last_frame.to_ndarray(format='rgb24')
            frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
            logger.debug(f"🔄 Reusing last frame for WebRTC stability (frame {self.frame_count})")
        elif frame is None:
            # Only create placeholder if we have no previous frame
            frame = av.VideoFrame.from_ndarray(
                np.zeros((390, 844, 3), dtype=np.uint8),  # Fixed dimensions
                format='rgb24'
            )
            logger.debug(f"🖼️  Created placeholder frame {self.frame_count}")
        else:
            # Store this frame's data for potential reuse
            self.last_frame = frame

        # Set consistent WebRTC timing
        frame.pts = self.frame_count
        frame.time_base = Fraction(1, self.target_fps)

        self.frame_count += 1
        return frame

class SimpleWebRTCService:
    """Simple WebRTC service using optimized screenshots for proven reliability"""

    def __init__(self, udid: Optional[str] = None):
        self.udid = udid

        # WebRTC state
        self.peer_connections: Dict[str, RTCPeerConnection] = {}
        self.stream_active = False
        self.video_queue = Queue(maxsize=2)  # Minimal buffer to prevent stale frames

        # Stream processing
        self.frame_thread = None
        self.stream_lock = threading.Lock()

        # Quality settings optimized for stability
        self.quality_preset = "high"
        self.target_fps = settings.WEBRTC_FPS

    def set_udid(self, udid: str):
        """Set the UDID for this service instance"""
        self.udid = udid
        logger.info(f"🎯 SimpleWebRTC UDID set to: {udid}")

    def start_video_stream(self, quality: str = "high", fps: int = None) -> bool:
        """Start optimized screenshot-based streaming for WebRTC"""
        if fps is None:
            fps = settings.WEBRTC_FPS
        if not self.udid:
            logger.error("❌ No UDID set for WebRTC streaming")
            return False

        with self.stream_lock:
            if self.stream_active:
                logger.info(f"✅ WebRTC stream already active for {self.udid}")
                return True

            self.quality_preset = quality
            self.target_fps = fps

            try:
                logger.info(f"🚀 Starting WebRTC stream for {self.udid} at {fps}fps, quality: {quality}")

                self.stream_active = True

                # Start frame generation thread
                self.frame_thread = threading.Thread(
                    target=self._generate_frames,
                    daemon=True
                )
                self.frame_thread.start()

                logger.info(f"✅ WebRTC stream started for {self.udid}")
                return True

            except Exception as e:
                logger.error(f"❌ Failed to start WebRTC stream for {self.udid}: {e}")
                import traceback
                logger.error(f"   Traceback: {traceback.format_exc()}")
                return False

    def _generate_frames(self):
        """Generate stable frames for WebRTC with consistent timing"""
        logger.info(f"🎬 Starting stable WebRTC frame generation for {self.udid} @ {self.target_fps}fps...")

        try:
            screenshot_service = ScreenshotService(self.udid)

            frame_count = 0
            last_log_time = time.time()
            next_frame_time = time.time()

            while self.stream_active:
                # Re-read settings each iteration for runtime FPS changes
                target_fps = max(1, self.target_fps)
                frame_interval = 1.0 / target_fps
                current_time = time.time()

                # Wait for precise frame timing
                if current_time < next_frame_time:
                    sleep_time = next_frame_time - current_time
                    if sleep_time > 0.001:  # Only sleep if worth it
                        time.sleep(sleep_time)
                    continue

                try:
                    # Capture screenshot based on quality
                    if self.quality_preset in ["ultra", "high"]:
                        screenshot_data = screenshot_service.capture_high_quality_screenshot()
                    else:
                        screenshot_data = screenshot_service.capture_ultra_fast_screenshot()

                    if screenshot_data and "data" in screenshot_data:
                        # Decode and process image
                        image_bytes = base64.b64decode(screenshot_data["data"])

                        with Image.open(io.BytesIO(image_bytes)) as img:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')

                            # Consistent resolution for stability
                            if self.quality_preset == "ultra":
                                target_size = (468, 1014)  # 2x device logical resolution
                            elif self.quality_preset == "high":
                                target_size = (390, 844)   # 1.67x device logical resolution
                            else:
                                target_size = (294, 639)   # Device logical resolution

                            # Resize with BILINEAR (faster than LANCZOS, sufficient for streaming)
                            img = img.resize(target_size, Image.Resampling.BILINEAR)

                            # Convert to numpy array
                            img_array = np.array(img, dtype=np.uint8)

                            # Create AV frame
                            rgb_frame = av.VideoFrame.from_ndarray(img_array, format='rgb24')

                            # Manage queue for consistent flow
                            while not self.video_queue.empty():
                                try:
                                    self.video_queue.get_nowait()  # Remove old frames
                                except:
                                    break

                            # Add new frame
                            try:
                                self.video_queue.put_nowait(rgb_frame)
                                frame_count += 1
                            except:
                                logger.debug("Queue full, frame dropped")

                    # Set next frame time
                    next_frame_time += frame_interval

                    # Prevent time drift
                    if next_frame_time < current_time:
                        next_frame_time = current_time + frame_interval

                    # Periodic logging
                    if current_time - last_log_time >= 15.0:
                        actual_fps = frame_count / (current_time - last_log_time) if (current_time - last_log_time) > 0 else 0
                        logger.info(f"📊 WebRTC {self.udid}: {frame_count} frames, {actual_fps:.1f}fps actual, queue: {self.video_queue.qsize()}")
                        last_log_time = current_time
                        frame_count = 0

                except Exception as e:
                    logger.debug(f"Frame generation error: {e}")
                    next_frame_time += frame_interval  # Keep timing consistent even on errors

        except Exception as e:
            logger.error(f"WebRTC frame generation critical error for {self.udid}: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
        finally:
            logger.info(f"🛑 WebRTC frame generation stopped for {self.udid}")

    async def get_next_frame(self):
        """Get next frame from video queue with timeout"""
        try:
            # Shorter timeout to avoid frame staleness
            frame = self.video_queue.get(timeout=0.05)
            return frame
        except Empty:
            return None

    def stop_video_stream(self):
        """Stop video streaming and cleanup"""
        logger.info(f"🛑 Stopping WebRTC stream for {self.udid}")

        with self.stream_lock:
            self.stream_active = False

            # Clear frame queue
            while not self.video_queue.empty():
                try:
                    self.video_queue.get_nowait()
                except Empty:
                    break

        # Close all peer connections
        connections_to_close = list(self.peer_connections.items())
        self.peer_connections.clear()

        for connection_id, pc in connections_to_close:
            try:
                asyncio.create_task(pc.close())
            except Exception as e:
                logger.debug(f"Error closing connection {connection_id}: {e}")

    async def create_peer_connection(self) -> tuple[str, RTCPeerConnection]:
        """Create new WebRTC peer connection"""
        if not self.stream_active:
            if not self.start_video_stream(self.quality_preset, self.target_fps):
                raise Exception("Failed to start WebRTC stream")

        connection_id = str(uuid.uuid4())
        pc = RTCPeerConnection()

        # Add video track with stable FPS
        video_track = SimpleVideoTrack(self, target_fps=self.target_fps)
        pc.addTrack(video_track)
        logger.info(f"📹 Added video track with {self.target_fps}fps for {self.udid}")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():  # noqa: F841
            logger.info(f"🔗 WebRTC connection state for {self.udid}: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                self.remove_connection(connection_id)

        self.peer_connections[connection_id] = pc
        logger.info(f"🤝 Created WebRTC peer connection: {connection_id} for {self.udid}")
        return connection_id, pc

    async def handle_offer(self, pc: RTCPeerConnection, offer_data: Dict) -> Dict:
        """Handle WebRTC offer"""
        logger.info(f"📤 Handling WebRTC offer for {self.udid}")

        await pc.setRemoteDescription(RTCSessionDescription(
            sdp=offer_data["sdp"],
            type=offer_data["type"]
        ))

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        logger.info(f"📥 WebRTC answer created for {self.udid}")

        return {
            "type": "answer",
            "sdp": pc.localDescription.sdp
        }

    async def handle_ice_candidate(self, pc: RTCPeerConnection, candidate_data: Dict):
        """Handle ICE candidate"""
        logger.debug(f"🧊 Handling ICE candidate for {self.udid}")
        candidate_info = candidate_data.get("candidate")
        if candidate_info:
            candidate = RTCIceCandidate(
                candidate=candidate_info.get("candidate"),
                sdpMid=candidate_info.get("sdpMid"),
                sdpMLineIndex=candidate_info.get("sdpMLineIndex")
            )
            await pc.addIceCandidate(candidate)

    def remove_connection(self, connection_id: str):
        """Remove connection and cleanup if no more connections"""
        if connection_id in self.peer_connections:
            try:
                del self.peer_connections[connection_id]
                logger.info(f"🗑️  Removed WebRTC connection: {connection_id}")
            except KeyError:
                pass

        # Stop stream if no more connections
        if not self.peer_connections:
            self.stop_video_stream()

    def set_quality(self, quality: str) -> Dict:
        """Set streaming quality preset"""
        valid_qualities = ["low", "medium", "high", "ultra"]
        if quality not in valid_qualities:
            return {"success": False, "error": f"Invalid quality. Must be one of: {valid_qualities}"}

        old_quality = self.quality_preset
        self.quality_preset = quality

        logger.info(f"🎚️  Quality changed from {old_quality} to {quality} for {self.udid}")
        return {"success": True, "quality": quality}

    def set_fps(self, fps: int) -> Dict:
        """Set target FPS"""
        if fps < 10 or fps > 120:
            return {"success": False, "error": "FPS must be between 10 and 120"}

        old_fps = self.target_fps
        self.target_fps = fps

        logger.info(f"📊 FPS changed from {old_fps} to {fps} for {self.udid}")
        return {"success": True, "fps": fps}

    def get_status(self) -> Dict:
        """Get service status"""
        return {
            "stream_active": self.stream_active,
            "connections": len(self.peer_connections),
            "quality": self.quality_preset,
            "fps": self.target_fps,
            "queue_size": self.video_queue.qsize(),
            "udid": self.udid
        }
