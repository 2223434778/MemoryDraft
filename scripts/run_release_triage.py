"""Target-only feasibility pass; no automatic timing or prompt search."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from memorydraft.evaluation.release_triage import answer_event, load_dataset, messages, request_text, sha
from memorydraft.memory.store import MemoryStore
from scripts.run_reuse_pilot import append, digest, save


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/research/release-triage"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/release-triage-feasibility")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing feasibility evidence")
    projects, _, manifest = load_dataset(DATA)
    args.output.mkdir(parents=True)
    provenance = dict(status="started", stage="target_only", torch=torch.__version__, transformers=transformers.__version__,
                      source=json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")),
                      data_sha256=sha((DATA / "manifest.json").read_bytes()), data_manifest=manifest,
                      dtype="float32", attention="eager", budget=768, threads=4)
    save(args.output / "provenance.json", provenance)
    try:
        if transformers.__version__ != "4.57.3":
            raise ValueError("Use the isolated audited .venv-dflash")
        config = json.loads((ROOT / "configs/a6000-qwen3.json").read_text(encoding="utf-8"))
        provenance["target"] = config["target"]
        provenance["gpu"] = torch.cuda.get_device_name(0)
        # Verify the same pinned Target weights already used by prior experiments, before measuring requests.
        assets = json.loads((ROOT / "configs/qwen3-download-manifest.json").read_text(encoding="utf-8"))
        target_assets = [item for item in assets if item["destination"] == "qwen3-8b"]
        if len(target_assets) != 5 or any(item["hf_revision"] != config["target"]["revision"] for item in target_assets):
            raise ValueError("Unexpected pinned Target shards/revision")
        provenance["target_asset_manifest"] = target_assets
        for item in target_assets:
            path = Path(config["target"]["model"]) / item["filename"]
            if path.stat().st_size != item["size"]:
                raise ValueError("Target asset size mismatch")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                    raise ValueError("Target asset mismatch: " + item["filename"])
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(config["target"]["model"], local_files_only=True)
        preflight = [dict(project_id=p["id"], turn=t,
            input_tokens=len(tokenizer.apply_chat_template(messages(request_text(p, t)), add_generation_prompt=True,
                                                           enable_thinking=False)))
            for p in projects for t in range(3)]
        save(args.output / "preflight.json", preflight)
        if any(row["input_tokens"] > 6144 for row in preflight):
            raise ValueError("Input exceeds predeclared budget; preserve all cases and revise before any inference")
        target = AutoModelForCausalLM.from_pretrained(config["target"]["model"], local_files_only=True,
            torch_dtype=torch.float32, attn_implementation="eager").to("cuda:0").eval()
        stops = target.generation_config.eos_token_id
        stops = [stops] if isinstance(stops, int) else stops
        provenance["load_seconds"] = time.perf_counter() - load_started
        provenance["eos_token_ids"] = stops
        warm = tokenizer.apply_chat_template(messages("Reply with exactly: ready"), enable_thinking=False,
                                             add_generation_prompt=True, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            target.generate(warm, attention_mask=torch.ones_like(warm), do_sample=False, use_model_defaults=False,
                            max_new_tokens=16, eos_token_id=stops, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        save(args.output / "projects.json", projects)
        with MemoryStore(args.output / "history.db") as store, torch.inference_mode():
            for project in projects:
                for turn in range(3):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    text, order = request_text(project, turn), 10 + 10 * turn
                    snapshot = store.snapshot(project["id"], order)
                    ids = tokenizer.apply_chat_template(messages(text), enable_thinking=False,
                                    add_generation_prompt=True, return_tensors="pt").to("cuda:0")
                    output = target.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                            use_model_defaults=False, max_new_tokens=768, use_cache=True,
                            eos_token_id=stops, pad_token_id=tokenizer.pad_token_id)
                    torch.cuda.synchronize()
                    tokens = output[0, ids.shape[1]:].tolist()
                    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
                    store.append([answer_event(project, turn, decoded)])
                    row = dict(project_id=project["id"], turn=turn, order=order, method="AR", token_ids=tokens,
                        text=decoded, input_tokens=ids.shape[1], prompt_token_ids=ids[0].tolist(),
                        prompt_sha256=digest(ids[0].tolist()), request_sha256=sha(text.encode()),
                        snapshot_id=snapshot.snapshot_id,
                        history_records=[dict(record_id=r.record_id, order=r.order, text_sha256=sha(r.text.encode())) for r in snapshot.records],
                        budget=768, budget_hit=len(tokens) >= 768, eos=bool(tokens and tokens[-1] in stops),
                        latency_seconds=time.perf_counter() - started)
                    append(args.output / "references.jsonl", row)
                    print("complete", project["id"], turn, "tokens", len(tokens), "eos", row["eos"], flush=True)
        provenance.update(status="complete", rows=18)
    except Exception as error:
        provenance.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(args.output / "provenance.json", provenance)


if __name__ == "__main__":
    main()
