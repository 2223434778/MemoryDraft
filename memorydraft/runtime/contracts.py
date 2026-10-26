from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


class RequestCancelled(RuntimeError):
    """Cooperative cancellation; no partial result may be committed to memory."""


@dataclass(frozen=True)
class GenerationRequest:
    id: str
    user_id: str
    order: int
    messages: tuple[dict[str, str], ...]
    max_new_tokens: int = 128

    def __post_init__(self):
        if not self.id or not self.user_id or type(self.order) is not int or self.order < 0:
            raise ValueError("Nonempty request/user IDs and a nonnegative integer order are required")
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be an integer in [1, 4096]")
        if not self.messages or len(self.messages) > 128:
            raise ValueError("Expected between 1 and 128 messages")
        for message in self.messages:
            if set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"}:
                raise ValueError("Messages require role and content; supported roles: system/user/assistant")
            if not isinstance(message["content"], str):
                raise ValueError("Message content must be text")
        if self.messages[-1]["role"] != "user":
            raise ValueError("Requests must end in a user message")
        if sum(len(m["content"]) for m in self.messages) > 262144:
            raise ValueError("Message text exceeds the 256K character admission limit")

    @classmethod
    def from_dict(cls, value: dict) -> GenerationRequest:
        allowed = {"id", "user_id", "order", "messages", "max_new_tokens"}
        if set(value) - allowed:
            raise ValueError("Unknown request fields: " + ", ".join(sorted(set(value) - allowed)))
        return cls(**{**value, "messages": tuple(dict(m) for m in value["messages"])})


@dataclass
class GenerationResult:
    request_id: str
    method: str
    token_ids: list[int]
    text: str
    prompt_tokens: int
    prompt_sha256: str
    snapshot_id: str
    latency_seconds: float
    ttft_seconds: float | None
    finish_reason: str
    peak_gpu_bytes: int = 0
    counters: dict[str, int | float] = field(default_factory=dict)
    execution_profile: str = "eager"
    approximate: bool = False
    asd: dict | None = None
    draft_adapter_sha256: str | None = None
    tree: dict | None = None


TokenCallback = Callable[[list[int]], None]
StopCallback = Callable[[], bool]


class Engine(Protocol):
    methods: tuple[str, ...]

    def generate(self, request: GenerationRequest, method: str, *,
                 on_tokens: TokenCallback | None = None,
                 should_stop: StopCallback | None = None,
                 profile: bool = False) -> GenerationResult: ...

    def describe(self) -> dict: ...

    def close(self) -> None: ...
