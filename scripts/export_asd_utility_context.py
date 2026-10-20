"""Decode frozen public source states for qualitative review; no model forward."""
import argparse
from pathlib import Path

from transformers import AutoTokenizer

from scripts.diagnose_transport_online import write
from scripts.run_asd_utility_causal_probe import CONFIG, selected_states, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = []
    for q, source in sources().items():
        for s in selected_states(source):
            p, e = s["position"], source["events"][s["block"]]
            rows.append(dict(request_id=q, position=p, candidate_text=tokenizer.decode([s["candidate"]]),
                             recovery_text=tokenizer.decode([s["recovery"]]),
                             context_before=tokenizer.decode(source["token_ids"][max(0, p - 55):p]),
                             source_after=tokenizer.decode(source["token_ids"][p:p + 45]),
                             block_accepted=e["accepted"], block_strict=e["strict_accepted"],
                             block_proposed=len(e["mismatch"]), run_gate_accepts=s["regret"] / s["run"] <= CONFIG.gate))
    write(args.output, rows)


if __name__ == "__main__":
    main()
