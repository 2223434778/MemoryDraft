"""CPU-only post-analysis: can static R propose the same gains from frozen Target history?"""
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from memorydraft.evaluation.history_alignment import common_prefix
from memorydraft.evaluation.replay import read_jsonl
from memorydraft.memory.store import MemoryStore
from memorydraft.retrieval.suffix import SuffixArrayIndex


if __name__ == "__main__":
    base = Path("results/h-causal")
    provenance = json.loads((base / "provenance.json").read_text())
    if provenance["status"] != "complete":
        raise RuntimeError("Run this CPU audit only after both primary diagnostics finish")
    config = provenance["config"]
    tokenizer = AutoTokenizer.from_pretrained(config["target"]["model"], trust_remote_code=False)
    requests = {r["id"]: r for r in read_jsonl("data/research/memory-test.jsonl")}
    original = read_jsonl(base / "alignment-prefix.jsonl")
    groups = {}
    for row in original:
        groups.setdefault(row["request_id"], []).append(row)
    evidence = []
    with MemoryStore(base / "teacher.db") as store:
        store.import_jsonl(base / "teacher-history/history.jsonl")
        for rid, group in groups.items():
            request = requests[rid]
            prompt = tokenizer.apply_chat_template(request["messages"], add_generation_prompt=True,
                                                    **config["template_options"])
            reference = [r["reference_ids"][0] for r in group]
            index = SuffixArrayIndex(store.snapshot(request["user_id"], request["order"]), tokenizer)
            for row in group:
                # Search only the confirmed prefix; answer suffix is used after retrieval for scoring.
                proposal, source, length = index.match(prompt + reference[:row["position"]], row["budget"])
                accepted = common_prefix(proposal, row["reference_ids"])
                evidence.append(dict(request_id=rid, position=row["position"], r_teacher_accepted=accepted,
                                     teacher_r_match_length=length, r_source_id=source,
                                     h_teacher=row["lengths"]["H_teacher"], h_manual=row["lengths"]["H_manual"]))
    improved = [r for r in evidence if r["h_teacher"] > r["h_manual"]]
    summary = dict(purpose="Post-test CPU proposal-quality audit only; full teacher snapshot static R, not an online speed measurement or dynamic lookup.",
                   diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   alignment_input_sha256=hashlib.sha256((base / "alignment-prefix.jsonl").read_bytes()).hexdigest(),
                   teacher_history_sha256=hashlib.sha256((base / "teacher-history/history.jsonl").read_bytes()).hexdigest(),
                   positions=len(evidence), h_improved_positions=len(improved),
                   improved_positions_static_r_covers_h=sum(r["r_teacher_accepted"] >= r["h_teacher"] for r in improved),
                   per_request={})
    for rid in groups:
        group = [r for r in evidence if r["request_id"] == rid]
        weak = [r for r in group if r["teacher_r_match_length"] < 3]
        summary["per_request"][rid] = dict(mean_h_teacher=sum(r["h_teacher"] for r in group) / len(group),
                                           mean_static_r_teacher=sum(r["r_teacher_accepted"] for r in group) / len(group),
                                           teacher_r_weak_positions=len(weak),
                                           weak_mean_h_teacher_minus_manual=sum(r["h_teacher"] - r["h_manual"] for r in weak) / len(weak) if weak else None)
    (base / "teacher-r-overlap.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
