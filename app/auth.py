import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException
from jose import jwt
from jose.exceptions import JWTError
from starlette.requests import Request


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    s = str(data or "")
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def _now_ts() -> int:
    return int(time.time())


@dataclass
class UserPrincipal:
    user_id: str
    email: Optional[str]
    session_expires_at: int


class AuthConfig:
    def __init__(self) -> None:
        self.domain = str(os.getenv("AUTH0_DOMAIN", "")).strip()
        self.client_id = str(os.getenv("AUTH0_CLIENT_ID", "")).strip()
        self.client_secret = str(os.getenv("AUTH0_CLIENT_SECRET", "")).strip()
        self.public_base_url = str(os.getenv("PUBLIC_BASE_URL", "")).strip().rstrip("/")
        self.cookie_secret = str(os.getenv("AUTH_COOKIE_SECRET", "")).strip()
        self.connection_name = str(os.getenv("AUTH0_CONNECTION_NAME", "email")).strip() or "email"
        self.session_ttl_days = max(1, int(str(os.getenv("SESSION_TTL_DAYS", "3")).strip() or "3"))
        secure_raw = str(os.getenv("AUTH_COOKIE_SECURE", "")).strip().lower()
        if secure_raw in {"1", "true", "yes"}:
            self.cookie_secure = True
        elif secure_raw in {"0", "false", "no"}:
            self.cookie_secure = False
        else:
            self.cookie_secure = self.public_base_url.startswith("https://")

    @property
    def configured(self) -> bool:
        return all(
            [
                self.domain,
                self.client_id,
                self.client_secret,
                self.public_base_url,
                self.cookie_secret,
            ]
        )

    @property
    def issuer(self) -> str:
        return f"https://{self.domain}/"

    @property
    def authorize_url(self) -> str:
        return f"https://{self.domain}/authorize"

    @property
    def token_url(self) -> str:
        return f"https://{self.domain}/oauth/token"

    @property
    def jwks_url(self) -> str:
        return f"https://{self.domain}/.well-known/jwks.json"

    @property
    def callback_url(self) -> str:
        return f"{self.public_base_url}/auth/callback"

    @property
    def logout_url(self) -> str:
        rt = urllib.parse.quote(f"{self.public_base_url}/", safe="")
        return (
            f"https://{self.domain}/v2/logout?client_id={urllib.parse.quote(self.client_id, safe='')}"
            f"&returnTo={rt}"
        )


class AuthService:
    SESSION_COOKIE_NAME = "sb_session"
    STATE_COOKIE_NAME = "sb_auth_state"

    def __init__(self, cfg: AuthConfig) -> None:
        self.cfg = cfg
        self._jwks_cache: Dict[str, Any] = {"fetched_at": 0.0, "keys": []}

    def ensure_configured(self) -> None:
        if not self.cfg.configured:
            raise HTTPException(status_code=500, detail="Auth is not configured.")

    def _sign_blob(self, payload_b64: str) -> str:
        sig = hmac.new(
            self.cfg.cookie_secret.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return _b64url_encode(sig)

    def sign_json(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        b64 = _b64url_encode(raw)
        return f"{b64}.{self._sign_blob(b64)}"

    def verify_json(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            parts = str(token or "").split(".")
            if len(parts) != 2:
                return None
            b64, sig = parts
            expected = self._sign_blob(b64)
            if not hmac.compare_digest(sig, expected):
                return None
            payload = json.loads(_b64url_decode(b64).decode("utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def build_authorize_url(self, state: str, nonce: str) -> str:
        q = {
            "response_type": "code",
            "client_id": self.cfg.client_id,
            "redirect_uri": self.cfg.callback_url,
            "scope": "openid profile email",
            "state": state,
            "nonce": nonce,
            "connection": self.cfg.connection_name,
        }
        return f"{self.cfg.authorize_url}?{urllib.parse.urlencode(q)}"

    def make_state_cookie_value(self, state: str, nonce: str) -> str:
        return self.sign_json(
            {
                "state": state,
                "nonce": nonce,
                "iat": _now_ts(),
                "exp": _now_ts() + 600,
            }
        )

    def parse_state_cookie(self, token: str) -> Optional[Dict[str, Any]]:
        payload = self.verify_json(token)
        if not payload:
            return None
        if int(payload.get("exp", 0) or 0) < _now_ts():
            return None
        return payload

    def build_session_cookie_value(self, user_id: str, email: Optional[str]) -> str:
        iat = _now_ts()
        exp = iat + (self.cfg.session_ttl_days * 86400)
        return self.sign_json(
            {
                "user_id": str(user_id or "").strip(),
                "email": str(email or "").strip() or None,
                "iat": iat,
                "exp": exp,
            }
        )

    def parse_session_cookie(self, token: str) -> Optional[UserPrincipal]:
        payload = self.verify_json(token)
        if not payload:
            return None
        user_id = str(payload.get("user_id", "")).strip()
        if not user_id:
            return None
        exp = int(payload.get("exp", 0) or 0)
        if exp < _now_ts():
            return None
        email = str(payload.get("email", "")).strip() or None
        return UserPrincipal(user_id=user_id, email=email, session_expires_at=exp)

    def mint_state_nonce(self) -> tuple[str, str]:
        return secrets.token_urlsafe(24), secrets.token_urlsafe(24)

    def _fetch_jwks(self) -> Dict[str, Any]:
        now = time.time()
        if (now - float(self._jwks_cache.get("fetched_at", 0.0) or 0.0)) < 300 and self._jwks_cache.get("keys"):
            return {"keys": self._jwks_cache.get("keys", [])}
        with httpx.Client(timeout=10.0) as client:
            rsp = client.get(self.cfg.jwks_url)
            rsp.raise_for_status()
            jwks = rsp.json()
            if not isinstance(jwks, dict):
                raise RuntimeError("Invalid JWKS response.")
            self._jwks_cache = {"fetched_at": now, "keys": jwks.get("keys", [])}
            return jwks

    def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.cfg.client_id,
            "client_secret": self.cfg.client_secret,
            "code": code,
            "redirect_uri": self.cfg.callback_url,
        }
        with httpx.Client(timeout=15.0) as client:
            rsp = client.post(self.cfg.token_url, json=payload)
            rsp.raise_for_status()
            out = rsp.json()
            if not isinstance(out, dict):
                raise RuntimeError("Invalid token response.")
            return out

    def validate_id_token(self, id_token: str, nonce: str) -> Dict[str, Any]:
        self.ensure_configured()
        try:
            headers = jwt.get_unverified_header(id_token)
            kid = str(headers.get("kid", "")).strip()
            if not kid:
                raise RuntimeError("Missing kid.")
            jwks = self._fetch_jwks()
            key = None
            for k in jwks.get("keys", []) if isinstance(jwks.get("keys", []), list) else []:
                if isinstance(k, dict) and str(k.get("kid", "")).strip() == kid:
                    key = k
                    break
            if not key:
                raise RuntimeError("Signing key not found.")
            claims = jwt.decode(
                id_token,
                key,
                algorithms=[str(key.get("alg", "RS256"))],
                audience=self.cfg.client_id,
                issuer=self.cfg.issuer,
            )
            if str(claims.get("nonce", "")).strip() != str(nonce or "").strip():
                raise RuntimeError("Nonce mismatch.")
            return claims if isinstance(claims, dict) else {}
        except (JWTError, RuntimeError) as exc:
            raise HTTPException(status_code=401, detail=f"Invalid ID token: {exc}")


def get_current_user_from_request(request: Request, auth: AuthService) -> Optional[UserPrincipal]:
    raw = request.cookies.get(AuthService.SESSION_COOKIE_NAME, "")
    if not raw:
        return None
    return auth.parse_session_cookie(raw)
