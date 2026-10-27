from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ChatConfig:
    database: Path
    upstream_url: str
    upstream_key: str = field(repr=False)
    invite_code: str = field(repr=False)
    public_origin: str
    method: str = "BudgetTree-ASD"
    max_new_tokens: int = 1024
    timeout_seconds: float = 600

    def __post_init__(self):
        if len(self.upstream_key) < 24 or len(self.invite_code) < 16:
            raise ValueError("Use a random upstream key (24+ chars) and invitation code (16+ chars)")
        for value in (self.public_origin, self.upstream_url):
            parsed = urlsplit(value)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
                raise ValueError("Expected an HTTP(S) origin without credentials, query or path")
            if parsed.scheme == "http":
                try:
                    address = ipaddress.ip_address(parsed.hostname)
                except ValueError as error:
                    raise ValueError("Plain HTTP is restricted to loopback or Tailscale IPv4") from error
                if not (address.is_loopback or address in ipaddress.ip_network("100.64.0.0/10")):
                    raise ValueError("Use HTTPS outside loopback/Tailscale")
        if self.method not in {"AR", "DFlash", "BudgetTree", "ASD", "BudgetTree-ASD"}:
            raise ValueError("Unsupported chat method")
        if not 1 <= self.max_new_tokens <= 4096 or not 1 <= self.timeout_seconds <= 3600:
            raise ValueError("Invalid generation limits")

    @property
    def secure_cookie(self):
        return self.public_origin.startswith("https://")

    @classmethod
    def from_env(cls):
        return cls(database=Path(os.environ["MEMORYDRAFT_CHAT_DB"]),
                   upstream_url=os.environ["MEMORYDRAFT_UPSTREAM_URL"].rstrip("/"),
                   upstream_key=os.environ["MEMORYDRAFT_API_KEY"],
                   invite_code=os.environ["MEMORYDRAFT_INVITE_CODE"],
                   public_origin=os.environ["MEMORYDRAFT_PUBLIC_ORIGIN"].rstrip("/"),
                   method=os.environ.get("MEMORYDRAFT_CHAT_METHOD", "BudgetTree-ASD"),
                   max_new_tokens=int(os.environ.get("MEMORYDRAFT_CHAT_MAX_TOKENS", "1024")))
