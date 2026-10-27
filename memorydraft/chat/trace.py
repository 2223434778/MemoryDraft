"""Keep a bounded, request-owned record for the process inspector."""
from __future__ import annotations

from collections import OrderedDict
import json

from memorydraft.chat.upstream import UpstreamError
from memorydraft.runtime.contracts import MAX_OUTPUT_TOKENS


class TraceHistory:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.rounds = OrderedDict()
        self.total = 0

    def add(self, value):
        if not isinstance(value, dict):
            raise UpstreamError("模型返回了无效的过程事件")
        number, phase = value.get("round"), value.get("phase")
        if (value.get("version") != 1 or type(number) is not int or not 1 <= number <= MAX_OUTPUT_TOKENS
                or phase not in {"draft", "verified"} or len(json.dumps(value)) > 65536):
            raise UpstreamError("模型返回了无效的过程事件")
        if phase == "draft":
            proposal, tree = value.get("proposal"), value.get("tree")
            if not self.tokens(proposal):
                raise UpstreamError("候选长度超过可视化限制")
            if tree is not None:
                if not isinstance(tree, dict):
                    raise UpstreamError("候选树结构无效")
                parents, depths, tokens = tree.get("parents", []), tree.get("depths", []), tree.get("tokens", [])
                if (not isinstance(parents, list) or not isinstance(depths, list) or not self.tokens(tokens)
                        or not 1 <= len(parents) <= 128 or len(depths) != len(parents) or len(tokens) != len(parents)
                        or any(type(d) is not int for d in depths)
                        or parents[0] != -1 or depths[0] != 0 or any(type(p) is not int or not 0 <= p < i
                            or depths[i] != depths[p] + 1 for i, p in enumerate(parents) if i)):
                    raise UpstreamError("候选树结构无效")
        else:
            draft = self.rounds.get(number, {}).get("draft")
            committed, correction = value.get("committed"), value.get("correction")
            accepted, path, relaxed = value.get("accepted"), value.get("selected_path"), value.get("relaxed")
            if (not draft or type(accepted) is not int or not 0 <= accepted <= 128
                    or not self.tokens(committed) or not self.tokens(correction) or len(correction) > 1
                    or len(committed) != accepted + len(correction)
                    or not isinstance(relaxed, list) or any(type(i) is not int for i in relaxed)):
                raise UpstreamError("验证结果无效")
            tree = draft.get("tree")
            if tree:
                if (not isinstance(path, list) or len(path) != accepted + 1 or path[0] != 0
                        or any(type(i) is not int or not 0 <= i < len(tree["parents"]) for i in path)
                        or any(tree["parents"][i] != p for p, i in zip(path, path[1:]))
                        or any(i not in path[1:] for i in relaxed)):
                    raise UpstreamError("验证路径无效")
            elif path is not None or accepted > len(draft["proposal"]) or any(not 0 <= i < accepted for i in relaxed):
                raise UpstreamError("验证路径无效")
        row = self.rounds.setdefault(number, {})
        row[phase] = value
        self.total = max(self.total, number)
        while len(self.rounds) > self.capacity:
            self.rounds.popitem(last=False)

    def export(self):
        return {"version": 1, "round_count": self.total, "truncated": self.total > len(self.rounds),
                "rounds": list(self.rounds.values())}

    @staticmethod
    def tokens(value):
        return (isinstance(value, list) and len(value) <= 128 and all(isinstance(t, dict)
                and type(t.get("id")) is int and t["id"] >= 0 and isinstance(t.get("text"), str)
                and len(t["text"]) <= 80 for t in value))
