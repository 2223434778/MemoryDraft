"""Project-excluded, common-state cost decisions over previously observed outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memorydraft.evaluation.benefit_probe import calibration, policy_data, protected
from memorydraft.evaluation.matrix_probe import observed_rows
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.retrieval.benefit import BenefitPolicy
from memorydraft.runtime.benchmark import source_digest
from scripts.diagnose_transport_online import write
from scripts.run_runtime_migration import append


ROOT = Path(__file__).resolve().parents[1]


def decisions(root, costs):
    samples = calibration(root)
    rows, predictions = observed_rows(root), []
    for frame, sample in zip(read_rows(root / "results/ot-matrix/offline/frames.jsonl"), samples):
        draft = next(c["draft_tokens"] for r in rows if r["request_id"] == frame["request_id"]
                     for c in r["calls"] if c["generated"] == frame["generated"]
                     and c["draft_oracle_accepted"] == frame["draft_accepted"]
                     and all(item["tokens"][:item["prefix_length"]] == c["draft_tokens"][:item["prefix_length"]]
                             for item in frame["candidates"]))
        policy = BenefitPolicy(**policy_data(samples, costs, sample["project"]))
        value = policy.predict(frame["candidates"], draft, 32, 512)
        if sample["project"] in value["projects"]:
            raise ValueError("Project leakage")
        choice = value["choice"]
        accepted = min(sample["outcomes"][choice["route"]][0], choice["length"])
        predictions.append(dict(request_id=sample["request_id"], project_id=sample["project"], generated=sample["generated"],
                                **value, actual_accepted=accepted, predicted_accepted=choice["advance"] - 1,
                                edit_accepted=sample["outcomes"]["edit"][0], ot_accepted=sample["outcomes"]["ot"][0],
                                draft_accepted=sample["outcomes"]["draft"][0]))
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    costs = json.loads(args.costs.read_text())
    rows = decisions(ROOT, costs)
    for row in rows:
        append(args.output / "rows.jsonl", row)
    summary = dict(states=len(rows), ot_calls=sum(r["choice"]["route"] == "ot" for r in rows),
                   short=sum(r["choice"]["length"] < 32 and r["choice"]["route"] != "draft" for r in rows),
                   draft=sum(r["choice"]["route"] == "draft" for r in rows),
                   actual_accepted=sum(r["actual_accepted"] for r in rows),
                   predicted_accepted=sum(r["predicted_accepted"] for r in rows),
                   mean_absolute_error=sum(abs(r["actual_accepted"] - r["predicted_accepted"]) for r in rows) / len(rows))
    write(args.output / "summary.json", summary)
    write(args.output / "provenance.json", dict(status="complete", source=source_digest(),
          scope="Project-excluded development replay; all answers previously observed; context512 proxy only",
          original_sha256=protected(ROOT), original_unchanged=True, costs_sha256=file_hash(args.costs),
          runner_sha256=file_hash(Path(__file__)), protocol_sha256=file_hash(ROOT / "docs/OT_BENEFIT_PROTOCOL.md")))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
