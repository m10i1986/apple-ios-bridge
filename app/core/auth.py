import hashlib
import secrets
import time
from typing import Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config.settings import settings
from app.core.logging import logger

# WebSocket パスは認証対象外とする
_EXCLUDED_PATH_PREFIXES = ("/ws/",)

# ノンスの有効期限（秒）
_NONCE_EXPIRY_SECONDS: int = 300


class DigestAuthMiddleware(BaseHTTPMiddleware):
    """RFC 7616 準拠の HTTP Digest 認証ミドルウェア（SHA-256）。
    WebSocket パス（/ws/）はスキップする。
    """

    def __init__(self, app, realm: str = "iOS Remote Control") -> None:
        super().__init__(app)
        self.realm = realm
        # nonce -> 発行時刻（Unix time）
        self._nonce_store: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # ノンス管理
    # ------------------------------------------------------------------

    def _generate_nonce(self) -> str:
        nonce = secrets.token_hex(32)
        self._nonce_store[nonce] = time.monotonic()
        self._cleanup_expired_nonces()
        return nonce

    def _cleanup_expired_nonces(self) -> None:
        now = time.monotonic()
        expired = [n for n, t in self._nonce_store.items() if now - t > _NONCE_EXPIRY_SECONDS]
        for n in expired:
            del self._nonce_store[n]

    def _is_valid_nonce(self, nonce: str) -> bool:
        if nonce not in self._nonce_store:
            return False
        if time.monotonic() - self._nonce_store[nonce] > _NONCE_EXPIRY_SECONDS:
            del self._nonce_store[nonce]
            return False
        return True

    # ------------------------------------------------------------------
    # レスポンス生成
    # ------------------------------------------------------------------

    def _unauthorized_response(self, stale: bool = False) -> Response:
        nonce = self._generate_nonce()
        stale_param = ", stale=TRUE" if stale else ""
        www_auth = (
            f'Digest realm="{self.realm}", '
            f'algorithm=SHA-256, '
            f'nonce="{nonce}", '
            f'qop="auth"'
            f"{stale_param}"
        )
        return Response(
            content="Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": www_auth},
        )

    # ------------------------------------------------------------------
    # ヘッダー解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_digest_header(auth_header: str) -> Optional[Dict[str, str]]:
        """Authorization ヘッダーを解析して dict を返す。"""
        if not auth_header.startswith("Digest "):
            return None
        params: Dict[str, str] = {}
        for part in auth_header[7:].split(","):
            part = part.strip()
            if "=" in part:
                key, _, value = part.partition("=")
                params[key.strip()] = value.strip().strip('"')
        return params

    # ------------------------------------------------------------------
    # Digest 検証
    # ------------------------------------------------------------------

    def _verify_digest(self, params: Dict[str, str], method: str) -> bool:
        required = {"username", "realm", "nonce", "uri", "nc", "cnonce", "qop", "response"}
        if not required.issubset(params.keys()):
            return False

        # ユーザー名確認（タイミング攻撃対策に compare_digest は使用できないが
        # 先にノンス・HA1 検証を通過させないことで情報漏洩を防ぐ）
        if params["username"] != settings.AUTH_USERNAME:
            return False

        if not self._is_valid_nonce(params["nonce"]):
            return False

        # HA1 = SHA-256(username:realm:password)
        ha1 = hashlib.sha256(
            f"{settings.AUTH_USERNAME}:{self.realm}:{settings.AUTH_PASSWORD}".encode()
        ).hexdigest()

        # HA2 = SHA-256(method:uri)
        ha2 = hashlib.sha256(
            f"{method}:{params['uri']}".encode()
        ).hexdigest()

        # response = SHA-256(HA1:nonce:nc:cnonce:qop:HA2)
        expected = hashlib.sha256(
            f"{ha1}:{params['nonce']}:{params['nc']}:{params['cnonce']}:{params['qop']}:{ha2}".encode()
        ).hexdigest()

        return secrets.compare_digest(expected, params["response"])

    # ------------------------------------------------------------------
    # ミドルウェアメイン処理
    # ------------------------------------------------------------------

    async def dispatch(self, request: Request, call_next) -> Response:
        # Digest 認証が無効な場合は素通し
        if not settings.AUTH_ENABLED:
            return await call_next(request)

        # 除外パス（WebSocket）は素通し
        path = request.url.path
        if any(path.startswith(prefix) for prefix in _EXCLUDED_PATH_PREFIXES):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            return self._unauthorized_response()

        params = self._parse_digest_header(auth_header)
        if params is None:
            return self._unauthorized_response()

        # ノンス期限切れ → stale=TRUE で再チャレンジを促す
        nonce = params.get("nonce", "")
        if nonce and not self._is_valid_nonce(nonce):
            return self._unauthorized_response(stale=True)

        if not self._verify_digest(params, request.method):
            logger.warning("Digest auth failed for user=%s path=%s", params.get("username"), path)
            return self._unauthorized_response()

        return await call_next(request)
