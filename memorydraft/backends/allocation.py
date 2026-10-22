"""Finite real compute actions; selectors see only pre-verification history."""
from __future__ import annotations

ACTIONS = ("native", "half", "verify-half", "target")
FEATURES = ("generated", "prompt_tokens", "recent_advance", "last_advance")


def action_spec(name, native):
    if name not in ACTIONS or type(native) is not int or native < 2:
        raise ValueError("Unknown allocation action or native block")
    half = max(2, native // 2)
    return {"native": (native, native-1), "half": (half, half-1),
            "verify-half": (native, half-1), "target": (0, 0)}[name]


def features(generated, history, prompt_tokens):
    advances = [r["advance"] for r in history]
    return dict(generated=generated, prompt_tokens=prompt_tokens,
                recent_advance=sum(advances)/len(advances) if advances else 1.,
                last_advance=advances[-1] if advances else 1.)


class ForkAction:
    """Pay the chosen current action and its actual strict future, keeping KV live."""
    def __init__(self, source, index, action, synchronize, now):
        if action not in ACTIONS or not 0 <= index < len(source):
            raise ValueError("Invalid allocation fork")
        self.source, self.index, self.action = source, index, action
        self.synchronize, self.now = synchronize, now
        self.started = None

    def __call__(self, step, generated, history):
        if step == self.index:
            expected = self.source[step]
            if generated != expected["generated_before"]:
                raise ValueError("Allocation fork missed its source frontier")
            for previous, original in zip(history, self.source[max(0, step-4):step]):
                if any(previous[k] != original[k] for k in ("generated_before", "accepted", "advance", "proposed", "pending_before", "draft_cache_before")):
                    raise ValueError("Allocation fork changed its past KV trajectory")
            self.synchronize()
            self.started = self.now()
            return self.action
        return "native"


def predict(model, values):
    if model["feature"] is None:
        return model["left"]
    return model["left"] if values[model["feature"]] <= model["threshold"] else model["right"]


def fit_stump(samples, actions=ACTIONS):
    """One split, request-balanced costs; thresholds come only from training data."""
    if not samples:
        return dict(feature=None, threshold=None, left=actions[0], right=actions[0])
    counts = {q: sum(s["request_id"] == q for s in samples) for q in {s["request_id"] for s in samples}}
    cost = lambda items, action: sum(s["costs"][action]/counts[s["request_id"]] for s in items)
    fixed = min(actions, key=lambda a: cost(samples, a))
    best_cost = cost(samples, fixed)
    model = dict(feature=None, threshold=None, left=fixed, right=fixed)
    for feature in FEATURES:
        values = sorted({s["features"][feature] for s in samples})
        for x, y in zip(values, values[1:]):
            threshold = (x+y)/2
            left = [s for s in samples if s["features"][feature] <= threshold]
            right = [s for s in samples if s["features"][feature] > threshold]
            if min(len(left), len(right)) < 3:
                continue
            a, b = (min(actions, key=lambda a: cost(items, a)) for items in (left, right))
            total = cost(left, a)+cost(right, b)
            if total < best_cost-1e-12:
                best_cost = total
                model = dict(feature=feature, threshold=threshold, left=a, right=b)
    return model
