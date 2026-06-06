"""WebSocket handler for H.264 fragmented MP4 streaming (MSE compatible).

Wire format:
    1. On connect: server sends a JSON text message
         {"type": "init", "mime": "video/mp4; codecs=\"...\""}
       so the client can call MediaSource.addSourceBuffer() with the right mime.
    2. Then: server sends the init segment (ftyp + moov) as a binary frame.
    3. Then: server sends media segments (moof + mdat) as binary frames.

The client is expected to feed every binary frame into the SourceBuffer
unchanged.  This handler does not consume any messages from the client.
"""

import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect

from app.core.logging import logger
from app.services.h264_stream_service import (
    get_or_start_service,
    release_service_if_idle,
)


class VideoH264WebSocket:
    """Per-connection H.264 fMP4 WebSocket handler.

    The underlying capture pipeline (simctl recordVideo + ffmpeg remux) is
    shared across all clients for the same UDID via the service registry in
    h264_stream_service.
    """

    async def handle_connection(self, websocket: WebSocket, udid: str) -> None:
        loop = asyncio.get_event_loop()

        svc = await asyncio.to_thread(get_or_start_service, udid, loop)
        if svc is None:
            logger.error(f"H264 WS: failed to start stream service for {udid}")
            await websocket.close(code=1011, reason="Failed to start H264 stream")
            return

        try:
            # 1) Send codec descriptor
            await websocket.send_text(json.dumps({
                "type": "init",
                "mime": svc.mime_type,
            }))

            # 2) Send cached init segment + register as client for media frames
            ok = await svc.add_client(websocket)
            if not ok:
                logger.warning(f"H264 WS: init segment unavailable for {udid}")
                await websocket.close(code=1011, reason="Init segment unavailable")
                return

            logger.info(f"H264 WS client added for {udid}")

            # 3) Idle: keep the connection open until the client disconnects.
            #    We don't expect text from the client; ignore anything that comes.
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"H264 WS error for {udid}: {e}")
        finally:
            try:
                remaining = svc.remove_client(websocket)
                logger.info(
                    f"H264 WS client removed for {udid} (remaining={remaining})"
                )
                if remaining == 0:
                    await asyncio.to_thread(release_service_if_idle, udid)
            except Exception as e:
                logger.warning(f"H264 WS cleanup error for {udid}: {e}")
