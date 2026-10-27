from __future__ import annotations

import hashlib
import secrets
import time
from collections import OrderedDict


def password_hash(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt),
                            n=32768, r=8, p=3, maxmem=64 * 1024 * 1024, dklen=32)
    return "scrypt-v1$" + salt + "$" + digest.hex()


def verify_password(password: str, stored: str) -> bool:
    version, salt, _ = stored.split("$")
    return version == "scrypt-v1" and secrets.compare_digest(password_hash(password, salt), stored)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class LoginLimiter:
    """Bounded process-local limiter; the gateway intentionally uses one worker."""
    def __init__(self):
        self.attempts: OrderedDict[str, list[float]] = OrderedDict()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        values = [t for t in self.attempts.pop(key, []) if now - t < 300]
        allowed = len(values) < 12
        if allowed:
            values.append(now)
        self.attempts[key] = values
        while len(self.attempts) > 2048:
            self.attempts.popitem(last=False)
        return allowed
