from typing import Optional
import os

from dotenv import load_dotenv

# .env ファイルが存在すれば読み込む（環境変数が既に設定済みの場合は上書きしない）
load_dotenv()

class Settings:
    """Application settings"""

    # Authentication
    AUTH_ENABLED: bool = os.environ.get("AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
    AUTH_USERNAME: str = os.environ.get("AUTH_USERNAME", "admin")
    AUTH_PASSWORD: str = os.environ.get("AUTH_PASSWORD", "")
    AUTH_REALM: str = os.environ.get("AUTH_REALM", "iOS Remote Control")

    # Device Configuration - Remove hardcoded UDID
    # UDID will be provided by session management

    # Video Configuration
    DEFAULT_VIDEO_FPS: int = int(os.environ.get("DEFAULT_VIDEO_FPS", "30"))
    WEBRTC_FPS: int = int(os.environ.get("WEBRTC_FPS", "30"))
    VIDEO_QUEUE_SIZE: int = 3
    WEBRTC_QUEUE_SIZE: int = 2

    # Connection Management
    MAX_CONNECTIONS_PER_SESSION: int = 10
    MAX_CONNECTIONS_PER_MINUTE: int = 20
    CONNECTION_CLEANUP_INTERVAL: int = 30

    # Resource Management
    MAX_MEMORY_MB: int = 2048
    SERVICE_IDLE_TIMEOUT: int = 300  # 5 minutes
    MEMORY_CHECK_INTERVAL: int = 30

    # Quality Settings
    DEFAULT_JPEG_QUALITY: int = 80
    WEBRTC_HIGH_QUALITY: int = 95

    # Streaming Resolution & Quality (runtime-adjustable via API)
    STREAM_SCALE_FACTOR: float = float(os.environ.get("STREAM_SCALE_FACTOR", "1.0"))  # 0.1-1.0
    STREAM_MAX_WIDTH: int = int(os.environ.get("STREAM_MAX_WIDTH", "0"))    # 0 = unlimited
    STREAM_MAX_HEIGHT: int = int(os.environ.get("STREAM_MAX_HEIGHT", "0"))  # 0 = unlimited
    STREAM_JPEG_QUALITY: int = int(os.environ.get("STREAM_JPEG_QUALITY", "75"))  # 10-95

    # Timeouts
    SCREENSHOT_TIMEOUT: float = 0.5
    TAP_TIMEOUT: float = 2.0
    SWIPE_TIMEOUT: float = 3.0
    TEXT_TIMEOUT: float = 5.0

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    STATIC_DIR: str = "static"
    TEMP_DIR: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"

settings = Settings()
