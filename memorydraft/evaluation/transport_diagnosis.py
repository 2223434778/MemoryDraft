"""Post-hoc candidate audits at captured online decisions; never a retrieval policy."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from memorydraft.retrieval.transport import AlignedReuseMatcher, AlignmentConfig, edit_score, transport_score


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def proposal_dict(proposal):
    return json.loads(json.dumps(asdict(proposal))) if proposal is not None else None


def accepted_prefix(tokens, remaining):
    count = 0
    for actual, expected in zip(tokens, remaining):
        if actual != expected:
            break
        count += 1
    return count


def failure_class(raw, eligible, pool, selected, proposed, *, long_prefix=8):
    """Eight tokens is a diagnostic stratum, not an estimated break-even length."""
    if raw < long_prefix:
        return "no_long_raw_span"
    if eligible < long_prefix:
        return "ineligible_endpoint"
    if pool < long_prefix:
        return "shortlist_miss"
    if not proposed:
        return "threshold_rejection"
    if selected < long_prefix:
        return "ranking_miss"
    if selected < pool:
        return "long_but_suboptimal"
    return "long_pool_best"


class DecodeRecorder:
    """Wrap one decode in the standalone diagnostic, without access to gold tokens."""

    def __init__(self, decode):
        self.decode = decode
        self.calls, self.rounds = [], []
        self.index = None

    def __call__(self, *args, **kwargs):
        matcher = kwargs.get("matcher")
        if matcher is None:
            raise ValueError("Diagnostic expects a retrieval matcher")
        self.index = matcher.index
        original = matcher.propose

        def capture(confirmed, max_tokens):
            before = dict(getattr(matcher, "stats", {}))
            call = dict(confirmed=list(confirmed), max_tokens=max_tokens,
                        blocked=matcher.blocked, calls_before=before.get("alignment_calls", 0))
            result = original(confirmed, max_tokens)
            call.update(proposal=proposal_dict(result),
                        alignment_called=getattr(matcher, "stats", {}).get("alignment_calls", 0)
                        > call["calls_before"])
            self.calls.append(call)
            return result

        matcher.propose = capture
        try:
            result = self.decode(*args, **kwargs)
            self.rounds = [dict(row) for row in result.rounds]
            return result
        finally:
            matcher.propose = original


def inspect_calls(recorder, reference, tokenizer, *, method, config=AlignmentConfig()):
    """Run only after strict output validation; references never enter the decoder."""
    index, rows = recorder.index, []
    if index is None:
        raise ValueError("Missing captured index")
    for call in recorder.calls:
        confirmed = call["confirmed"]
        position = len(confirmed)
        if confirmed != reference[:position]:
            raise ValueError("Captured prefix differs from the frozen Target reference")
        proposal = call["proposal"]
        row = dict(generated_before=position, prefix_sha256=digest(confirmed),
                   max_tokens=call["max_tokens"], calls_before=call["calls_before"],
                   alignment_called=call["alignment_called"], proposal=proposal)
        if not call["alignment_called"]:
            row["reason"] = ("suffix" if proposal is not None else "short_or_budget_limited")
            rows.append(row)
            continue
        matcher = AlignedReuseMatcher(index, method=method, config=config)
        matcher.blocked = call["blocked"]
        matcher.stats["alignment_calls"] = call["calls_before"]
        replay = matcher.propose(confirmed, call["max_tokens"])
        if proposal_dict(replay) != proposal:
            raise ValueError("Offline replay changed the actual online proposal")
        remaining = reference[position:]
        prefix = tuple(confirmed[-config.window:])
        pool = [p for p in matcher.shortlist(confirmed)
                if matcher.blocked != (tuple(confirmed[-index.max_match:]), p)]
        candidates = []
        for p in pool:
            lower, end, source = matcher.bounds[p]
            history = index.tokens[max(lower, p - len(prefix)):p]
            continuation = index.tokens[p:min(p + call["max_tokens"], end)]
            ot_score = transport_score(prefix, history, config)
            candidates.append(dict(position=p, source=source,
                                   ot_score=ot_score if abs(ot_score) != float("inf") else None,
                                   edit_score=edit_score(prefix, history),
                                   accepted=accepted_prefix(continuation, remaining),
                                   continuation=continuation,
                                   exact_tail=_tail(prefix, history)))
        raw_best = eligible_best = 0
        raw_position = eligible_position = None
        for start, end in zip(index.starts, index.ends):
            for p in range(start, end):
                count = accepted_prefix(index.tokens[p:min(p + call["max_tokens"], end)], remaining)
                if count > raw_best:
                    raw_best, raw_position = count, p
                if (p >= start + config.min_context
                        and matcher.blocked != (tuple(confirmed[-index.max_match:]), p)
                        and count > eligible_best):
                    eligible_best, eligible_position = count, p
        pool_best = max((c["accepted"] for c in candidates), default=0)
        selected = accepted_prefix(proposal["tokens"], remaining) if proposal else 0
        matching_rounds = [r for r in recorder.rounds if r["generated_before"] == position]
        if len(matching_rounds) != 1:
            raise ValueError("Missing or duplicate actual decode round")
        actual = matching_rounds[0]
        if proposal and (actual["route"] != "R" or actual["lookup"] != method or actual["accepted"] != selected):
            raise ValueError("Gold prefix audit disagrees with actual Target verification")
        row.update(reason="alignment", raw_best=raw_best, raw_position=raw_position,
                   eligible_best=eligible_best, eligible_position=eligible_position,
                   pool_best=pool_best, selected_accepted=selected, candidates=candidates,
                   classification=failure_class(raw_best, eligible_best, pool_best, selected, proposal is not None),
                   actual_round=actual,
                   prefix_tail=tokenizer.decode(confirmed[-16:], skip_special_tokens=False),
                   expected_next=tokenizer.decode(remaining[:16], skip_special_tokens=False),
                   selected_next=tokenizer.decode(proposal["tokens"][:16], skip_special_tokens=False) if proposal else None)
        rows.append(row)
    return rows


def _tail(left, right):
    count = 0
    for a, b in zip(reversed(left), reversed(right)):
        if a != b:
            break
        count += 1
    return count


def first_divergence(left, right):
    """Only the first differing round has a shared prior decode route/state."""
    fields = ("route", "generated_before", "proposed", "accepted", "source")
    for a, b in zip(left, right):
        if any(a[field] != b[field] for field in fields):
            return dict(left=a, right=b, same_prefix=a["generated_before"] == b["generated_before"])
    return None
