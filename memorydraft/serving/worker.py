from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import queue
import threading
import time
import uuid

from memorydraft.runtime.contracts import GenerationRequest, GenerationResult, RequestCancelled


class QueueFull(RuntimeError):
    pass


@dataclass(eq=False)
class Job:
    request: GenerationRequest
    method: str
    timeout_seconds: float
    stream: bool
    event_capacity: int
    key: str = field(default_factory=lambda: uuid.uuid4().hex)
    submitted: float = field(default_factory=time.perf_counter)
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: GenerationResult | None = None
    error: Exception | None = None
    started: float | None = None
    finished: float | None = None

    def __post_init__(self):
        self.events: queue.Queue[list[int]] = queue.Queue(self.event_capacity)

    def should_stop(self):
        return self.cancelled.is_set() or time.perf_counter() >= self.submitted + self.timeout_seconds

    def emit(self, tokens):
        if not self.stream:
            return
        while not self.should_stop():
            try:
                self.events.put(tokens, timeout=0.05)
                return
            except queue.Full:
                continue
        raise RequestCancelled("Stream cancelled or timed out")

    @property
    def queue_seconds(self):
        return (self.started or self.finished or time.perf_counter()) - self.submitted


class InferenceWorker:
    """Only this thread owns model execution; cancellation never releases it early."""

    def __init__(self, engine, *, queue_capacity=8, event_capacity=32):
        if queue_capacity < 1 or event_capacity < 1:
            raise ValueError("Queue and event capacities must be positive")
        self.engine, self.queue_capacity, self.event_capacity = engine, queue_capacity, event_capacity
        self.pending: deque[Job] = deque()
        self.active: Job | None = None
        self.closing = False
        self.condition = threading.Condition()
        self.counts = dict(submitted=0, completed=0, failed=0, cancelled=0, timed_out=0, rejected=0, output_tokens=0)
        self.thread = threading.Thread(target=self._run, name="memorydraft-inference", daemon=True)
        self.thread.start()

    def submit(self, request, method, *, timeout_seconds=60.0, stream=False):
        if method not in self.engine.methods:
            raise ValueError("Unsupported method: " + method)
        if not 0 < timeout_seconds <= 3600:
            raise ValueError("timeout_seconds must be in (0, 3600]")
        with self.condition:
            if self.closing:
                raise RuntimeError("Worker is shutting down")
            # Expired waiting jobs release admission capacity even during a long prefill.
            for old in list(self.pending):
                if old.should_stop():
                    self.pending.remove(old)
                    old.error = RequestCancelled("Request expired in queue")
                    self._finish(old)
            if len(self.pending) >= self.queue_capacity:
                self.counts["rejected"] += 1
                raise QueueFull("Inference queue is full")
            job = Job(request, method, timeout_seconds, stream, self.event_capacity)
            self.pending.append(job)
            self.counts["submitted"] += 1
            self.condition.notify()
            return job

    def cancel(self, job):
        with self.condition:
            if job.done.is_set():
                return
            job.cancelled.set()
            if job in self.pending:
                self.pending.remove(job)
                job.error = RequestCancelled("Request cancelled in queue")
                self._finish(job)

    def _finish(self, job):
        job.finished = time.perf_counter()
        if isinstance(job.error, RequestCancelled):
            timed_out = job.finished >= job.submitted + job.timeout_seconds
            self.counts["timed_out" if timed_out else "cancelled"] += 1
        elif job.error:
            self.counts["failed"] += 1
        else:
            self.counts["completed"] += 1
            self.counts["output_tokens"] += len(job.result.token_ids)
        job.done.set()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending or self.closing)
                if self.closing and not self.pending:
                    return
                job = self.pending.popleft()
                self.active = job
                job.started = time.perf_counter()
            try:
                if job.should_stop():
                    raise RequestCancelled("Request expired before execution")
                result = self.engine.generate(job.request, job.method, on_tokens=job.emit, should_stop=job.should_stop)
                if job.should_stop():
                    raise RequestCancelled("Request expired during execution")
                job.result = result
            except Exception as error:
                # Tracebacks retain generate() locals, including GPU KV tensors.
                # The HTTP layer needs the error category, not live model frames.
                error.__traceback__ = None
                error.__context__ = None
                error.__cause__ = None
                job.error = error
            finally:
                with self.condition:
                    self._finish(job)
                    self.active = None

    def metrics(self):
        with self.condition:
            return {**self.counts, "running": int(self.active is not None), "queued": len(self.pending),
                    "queue_capacity": self.queue_capacity, "event_capacity": self.event_capacity,
                    "closing": self.closing}

    def close(self, timeout=120.0):
        with self.condition:
            self.closing = True
            for job in list(self.pending):
                self.cancel(job)
            if self.active:
                self.cancel(self.active)
            self.condition.notify_all()
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise RuntimeError("Inference did not stop at a cooperative boundary before shutdown timeout")
        self.engine.close()
