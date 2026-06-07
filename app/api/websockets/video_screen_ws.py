"""WebSocket handler for screenshot-based screen streaming (PNG/JPEG).

Wire format (all messages are JSON text frames):
    1. On connect: server sends
         {"type": "init", "format": "jpeg" | "webp"}
       so the client knows how to decode the image data that follows.
    2. Then: for every captured frame, the server sends
         {"type": "frame", "format": "...", "data": "<base64>",
          "pixel_width": <int>, "pixel_height": <int>}
       where "data" is a base64-encoded JPEG or WEBP image.

The client decodes each frame and draws it onto a canvas.  This handler does
not consume any messages from the client.

The output image format is selectable via the ``?format=jpeg|webp`` query
parameter (defaults to "jpeg").
"""

import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect

from app.core.logging import logger
from app.services.screen_stream_service import (
    VALID_FORMATS,
    DEFAULT_FORMAT,
    get_or_start_service,
    release_service_if_idle,
)


class VideoScreenWebSocket:
    """Per-connection screen-stream WebSocket handler.

    The underlying capture pipeline (``idb screenshot`` loop) is shared across
    all clients for the same UDID via the service registry in
    screen_stream_service.
    """

    async def handle_connection(self, websocket: WebSocket, udid: str) -> None:
        loop = asyncio.get_event_loop()

        # Resolve requested image format from the query string.
        fmt = DEFAULT_FORMAT
        try:
            requested = websocket.query_params.get("format", DEFAULT_FORMAT)
            requested = (requested or DEFAULT_FORMAT).lower()
            if requested in VALID_FORMATS:
                fmt = requested
        except Exception:
            fmt = DEFAULT_FORMAT

        svc = await asyncio.to_thread(get_or_start_service, udid, loop, fmt)
        if svc is None:
            logger.error(f"Screen WS: failed to start stream service for {udid}")
            await websocket.close(code=1011, reason="Failed to start screen stream")
            return

        try:
            # 1) Tell the client which image format the frames use.
            await websocket.send_text(json.dumps({
                "type": "init",
                "format": svc.fmt,
            }))

            # 2) Register as a client; we get a queue the capture thread feeds.
            queue = await svc.add_client(websocket)
            if queue is None:
                logger.warning(f"Screen WS: no frame available yet for {udid}")
                await websocket.close(code=1011, reason="No frame available")
                return

            logger.info(f"Screen WS client added for {udid} (format={svc.fmt})")

            # 3) Drain the queue and forward each frame to the client.  When the
            #    client disconnects, send_text raises and we fall through to
            #    cleanup.
            while True:
                payload = await queue.get()
                await websocket.send_text(payload)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Screen WS error for {udid}: {e}")
        finally:
            try:
                remaining = svc.remove_client(websocket)
                logger.info(
                    f"Screen WS client removed for {udid} (remaining={remaining})"
                )
                if remaining == 0:
                    await asyncio.to_thread(release_service_if_idle, udid)
            except Exception as e:
                logger.warning(f"Screen WS cleanup error for {udid}: {e}")
