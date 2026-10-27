"""Small, JSON-only descriptions of real committed/draft token events."""
from __future__ import annotations


def payload(engine, value: dict) -> dict:
    tokenizer = engine.tokenizer

    def items(ids):
        if len(ids) > 128:
            raise ValueError("Speculation event exceeds the visualization node limit")
        result = []
        special = set(getattr(tokenizer, "all_special_ids", ()))
        for token in ids:
            text = tokenizer.decode([token], skip_special_tokens=True)
            if token in special:
                text = "⟨结束⟩" if token in getattr(engine, "stops", ()) else "⟨特殊词元⟩"
            elif not text or "\ufffd" in text:
                text = "#" + str(token)
            else:
                text = text.replace("\n", "↵").replace("\t", "⇥").replace(" ", "␠")
            result.append({"id": int(token), "text": text[:40]})
        return result

    result = {"version": 1, **value}
    if value["phase"] == "draft":
        result["proposal"] = items(value["proposal_ids"])
        tree = value.get("tree")
        if tree is not None:
            result["tree"] = {**tree, "tokens": items(tree["token_ids"])}
    else:
        result["committed"] = items(value["committed_ids"])
        result["correction"] = items(value["correction_ids"])
    return result
