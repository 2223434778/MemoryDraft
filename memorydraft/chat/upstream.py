from __future__ import annotations

import asyncio
import json

import anyio


class ChatStopped(Exception):
    pass


async def until_stopped(awaitable, signal):
    reading = asyncio.ensure_future(awaitable)
    stopping = asyncio.create_task(signal.wait())
    try:
        await asyncio.wait((reading, stopping), return_when=asyncio.FIRST_COMPLETED)
        if signal.is_set():
            raise ChatStopped()
        return reading.result()
    finally:
        reading.cancel()
        stopping.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(reading, stopping, return_exceptions=True)


class UpstreamError(Exception):
    def __init__(self, message, code=502):
        self.code = code
        super().__init__(message)


async def fit_context(client, messages, max_new_tokens):
    messages = messages[-121:]
    dropped = 0
    while True:
        response = await client.post("/v1/token-count", json={"messages": messages, "max_new_tokens": max_new_tokens})
        if response.status_code != 200:
            raise UpstreamError("模型服务暂时不可用，请稍后重试")
        counts = response.json()
        if not all(type(counts.get(k)) is int for k in ("input_tokens", "input_limit")):
            raise UpstreamError("模型服务返回了无效的上下文信息")
        if counts["input_tokens"] <= counts["input_limit"]:
            return messages, dropped, counts["input_tokens"]
        if len(messages) <= 1:
            raise UpstreamError("这条消息超过模型上下文限制，请缩短后重试", 422)
        messages = messages[2:]
        dropped += 1


async def events(client, body):
    async with client.stream("POST", "/v1/generate", json=body) as response:
        if response.status_code != 200:
            message = "模型队列已满，请稍后重试" if response.status_code == 429 else "模型服务暂时不可用，请稍后重试"
            raise UpstreamError(message, 429 if response.status_code == 429 else 502)
        if not response.headers.get("content-type", "").startswith("text/event-stream"):
            raise UpstreamError("模型服务没有返回流式响应")
        name, data, size = "message", [], 0
        async for line in response.aiter_lines():
            size += len(line)
            if size > 1024 * 1024:
                raise UpstreamError("模型响应过大")
            if not line:
                if data:
                    try:
                        value = json.loads("\n".join(data))
                    except ValueError as error:
                        raise UpstreamError("模型返回了无效的数据") from error
                    if not isinstance(value, dict):
                        raise UpstreamError("模型返回了无效的数据")
                    yield name, value
                name, data, size = "message", [], 0
            elif line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            raise UpstreamError("模型响应提前中断")
