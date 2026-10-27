"""Bounded, contextual text previews and UTF-16 ranges for an inline inspector."""
from __future__ import annotations


def common_prefix(left, right):
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return left[:length]


def utf16_length(text):
    return len(text.encode("utf-16-le")) // 2


class InlinePresentation:
    def __init__(self, engine):
        self.tokenizer = engine.tokenizer
        self.candidate = []

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True).rstrip("\ufffd")

    def present(self, event, confirmed):
        before = self.decode(confirmed)
        if event["phase"] == "draft":
            tree = event.get("tree")
            if tree:
                parents = set(tree["parents"][1:])
                leaf = max((i for i in range(len(tree["parents"])) if i not in parents),
                           key=lambda i: (tree["depths"][i], -i))
                path = []
                while leaf > 0:
                    path.insert(0, tree["token_ids"][leaf])
                    leaf = tree["parents"][leaf]
                self.candidate = path
            else:
                self.candidate = event["proposal_ids"]
            future = self.decode([*confirmed, *self.candidate])
            return {"anchor": utf16_length(before),
                    "candidate": future[len(common_prefix(before, future)):][:80]}

        committed = event["committed_ids"]
        accepted = committed[:event["accepted"]]
        after_accepted = self.decode([*confirmed, *accepted])
        after = self.decode([*confirmed, *committed])
        common = common_prefix(before, after)
        # A character completed by a Target byte is blue, rather than attributing
        # the whole Unicode character to an incomplete accepted byte fragment.
        aligned = after.startswith(after_accepted) and after_accepted.startswith(common)
        accepted_end = utf16_length(after_accepted) if aligned else utf16_length(common)
        shared = 0
        for a, b in zip(self.candidate, accepted):
            if a != b:
                break
            shared += 1
        candidate_text = self.decode([*confirmed, *self.candidate])
        shared_text = self.decode([*confirmed, *self.candidate[:shared]])
        rejected = candidate_text[len(common_prefix(shared_text, candidate_text)):] if shared < len(self.candidate) else ""
        return {"start": utf16_length(common), "accepted_end": accepted_end, "end": utf16_length(after),
                "accepted": after_accepted[len(common):][:80] if aligned else "",
                "target": after[len(after_accepted) if aligned else len(common):][:32],
                "rejected": rejected[:64]}
