"""Audited DFlash + legal history reuse: strict integration smoke and development pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from memorydraft.backends.dflash_reuse import ReuseConfig, generate_reuse
from memorydraft.evaluation.reuse_cases import confirmation_projects, development_projects, messages
from memorydraft.memory.store import MemoryEvent, MemoryStore, Snapshot
from memorydraft.retrieval.reuse import ReuseMatcher
from memorydraft.retrieval.suffix import SuffixArrayIndex


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("AR", "DFlash", "Global", "Cursor", "Context")
CONFIRMATION_METHODS = ("DFlash", "Adapted", "Global", "Context")


def save(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def append(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def summarize(rows, methods=METHODS):
    result = {}
    groups = ["all", *dict.fromkeys(row["scenario"] for row in rows)]
    for scenario in groups:
        selected = [row for row in rows if scenario == "all" or row["scenario"] == scenario]
        times = {method: sum(row["latency_seconds"] for row in selected if row["method"] == method)
                 for method in methods}
        result[scenario] = dict(times=times, rows=len(selected), projects=len({r["project_id"] for r in selected}),
                                vs_dflash={m: times["DFlash"] / t for m, t in times.items() if t},
                                correct=all(row["correct"] for row in selected))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    out = args.output or ROOT / ("results/reuse-confirmation" if args.confirmation else "results/reuse-pilot")
    if out.exists():
        raise RuntimeError("Preserve existing reuse evidence; choose a new output directory")
    out.mkdir(parents=True)
    methods = CONFIRMATION_METHODS if args.confirmation else METHODS
    repeats = 3 if args.confirmation else 1
    provenance = dict(status="started", transformers=transformers.__version__, torch=torch.__version__,
                      dtype="float32", attention="eager", stage="confirmation" if args.confirmation else "development",
                      repeats=repeats, methods=methods,
                      source=json.loads((ROOT / "SOURCE_MANIFEST.json").read_text()), config=asdict(ReuseConfig()))
    save(out / "provenance.json", provenance)
    try:
        if transformers.__version__ != "4.57.3":
            raise RuntimeError("Use the isolated audited .venv-dflash")
        manifest = json.loads((ROOT / "configs/dflash-manifest.json").read_text())
        directory = ROOT / "models/dflash-qwen3-8b"
        for item in manifest["code"] + [manifest["weight"]]:
            with (directory / item["filename"]).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != item["sha256"]:
                raise RuntimeError("DFlash checksum mismatch: " + item["filename"])
        provenance["dflash_manifest"] = manifest
        model_config = json.loads((ROOT / "configs/a6000-qwen3.json").read_text())
        provenance["target"] = model_config["target"]
        provenance["gpu"] = torch.cuda.get_device_name(0)
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        tokenizer = AutoTokenizer.from_pretrained(model_config["target"]["model"], local_files_only=True)
        target = AutoModelForCausalLM.from_pretrained(model_config["target"]["model"], local_files_only=True,
                                                    torch_dtype=torch.float32, attn_implementation="eager").to("cuda:0").eval()
        draft = AutoModel.from_pretrained(directory, trust_remote_code=True, local_files_only=True,
                                         torch_dtype=torch.float32, attn_implementation="eager").to("cuda:0").eval()
        eos = target.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        config = ReuseConfig()

        @torch.inference_mode()
        def generate(text, method, budget, store=None, user="warm", order=1, routes=(), stops=None):
            stops = eos if stops is None else stops
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            snapshot = store.snapshot(user, order) if store else Snapshot(user, order, ())
            snapshot_hash = snapshot.snapshot_id
            input_ids = tokenizer.apply_chat_template(messages(text), add_generation_prompt=True,
                                                       enable_thinking=False, return_tensors="pt").to("cuda:0")
            prompt_hash = digest(input_ids[0].tolist())
            index, matcher, detail = None, None, None
            if method in {"Global", "Cursor", "Context", "Mixed"}:
                resource = Snapshot(user, order, ()) if method == "Context" else snapshot
                index = SuffixArrayIndex(resource, tokenizer, min_match=3, max_match=32, prompt_text=text)
                matcher = ReuseMatcher(index, cursor=method != "Global" and not args.confirmation)
            if method == "AR":
                output = target.generate(input_ids, attention_mask=torch.ones_like(input_ids), do_sample=False,
                                         use_model_defaults=False, max_new_tokens=budget, use_cache=True,
                                         pad_token_id=tokenizer.pad_token_id, eos_token_id=stops)
            elif method == "DFlash":
                output = draft.spec_generate(target=target, input_ids=input_ids, max_new_tokens=budget,
                                             stop_token_ids=stops, temperature=0.)
            else:
                detail = generate_reuse(target, draft, input_ids, max_new_tokens=budget,
                                         stop_token_ids=stops, matcher=matcher, config=config,
                                         route_sequence=routes)
                output = detail.sequences
            torch.cuda.synchronize()
            tokens = output[0, input_ids.shape[1]:].tolist()
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            latency = time.perf_counter() - started
            return dict(method=method, token_ids=tokens, text=decoded, latency_seconds=latency,
                        snapshot_id=snapshot_hash, prompt_sha256=prompt_hash, budget=budget,
                        budget_hit=len(tokens) >= budget, eos=bool(tokens and tokens[-1] in stops),
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(), index_bytes=index.bytes_used if index else 0,
                        lookups=matcher.lookups if matcher else {}, rounds=detail.rounds if detail else [],
                        pending_peak=detail.feature_peak_tokens if detail else None,
                        buffer_flushes=detail.buffer_flushes if detail else 0)

        def compare(row, reference, label):
            row["correct"] = row["token_ids"] == reference["token_ids"] and row["prompt_sha256"] == reference["prompt_sha256"]
            if not row["correct"]:
                save(out / "failure.json", dict(label=label, row=row, reference=reference))
                raise RuntimeError("Strict greedy mismatch: " + label)

        for method in ("AR", "DFlash", "Adapted"):
            generate("Repeat exactly: A quick brown fox.", method, 16)
        smoke_text = "Repeat exactly, without extra text:\n" + "alpha bravo charlie delta echo foxtrot.\n" * 5
        with MemoryStore(out / "smoke.db") as smoke_store:
            smoke_store.append([MemoryEvent("smoke", "past", 0, "alpha bravo charlie delta echo foxtrot.\n" * 5)])
            for case in range(0 if args.confirmation else 8):
                budget = (1, 2, 31, 48, 64, 64, 64, 48)[case]
                stops = eos
                if case == 7:
                    first = generate(smoke_text, "AR", 1)
                    stops = [first["token_ids"][0]]
                reference = generate(smoke_text, "AR", budget, smoke_store, "smoke", 10, stops=stops)
                reference.update(case=case, stage="smoke_reference")
                append(out / "smoke-references.jsonl", reference)
                for method in ("DFlash", "Adapted", "Mixed"):
                    routes = (("R", "R", "D"), ("D", "R", "D"), ("R", "D", "R", "R"))[max(0, case - 4) % 3]
                    row = generate(smoke_text, method, budget, smoke_store, "smoke", 10,
                                   routes=routes if case in (4, 5, 6) else (), stops=stops)
                    compare(row, reference, f"smoke-{case}/{method}")
                    row.update(case=case, stage="smoke")
                    append(out / "smoke.jsonl", row)
                print("smoke", case, "correct", flush=True)
        provenance["smoke_complete"] = not args.confirmation
        if args.confirmation:
            previous = json.loads((ROOT / "results/reuse-pilot/provenance.json").read_text())
            if previous["status"] != "complete" or not previous.get("smoke_complete"):
                raise RuntimeError("Complete the development integration run first")
            provenance["integration_reference_source"] = previous["source"]["source_sha256"]
        save(out / "provenance.json", provenance)
        if args.smoke_only:
            provenance["status"] = "smoke_complete"
            return
        projects = confirmation_projects() if args.confirmation else development_projects()
        save(out / "projects.json", projects)
        rows = []
        with MemoryStore(out / "history.db") as store:
            # Freeze every project's past answer before generating any current reference.
            for project in projects:
                row = generate(project["past"], "AR", 768)
                row.update(project_id=project["id"], order=1)
                append(out / "past-generations.jsonl", row)
                store.append([MemoryEvent(project["id"], "artifact", 1, row["text"], project_id=project["id"])])
            initial = [asdict(record) for project in projects for record in store.snapshot(project["id"], 10).records]
            save(out / "initial-history.json", initial)
            for project in projects:
                for turn, text in enumerate(project["currents"]):
                    order = 10 + 10 * turn
                    budget = project.get("budget", 384)
                    reference = generate(text, "AR", budget, store, project["id"], order)
                    reference.update(project_id=project["id"], turn=turn, order=order)
                    append(out / "references.jsonl", reference)
                    for repeat in range(repeats):
                        shuffled = list(methods)
                        random.Random(20261018 + int(project["id"].rsplit("-", 1)[-1]) + 17 * turn + 101 * repeat).shuffle(shuffled)
                        for method in shuffled:
                            row = generate(text, method, budget, store, project["id"], order)
                            compare(row, reference, f"{project['id']}/{turn}/{repeat}/{method}")
                            if row["snapshot_id"] != reference["snapshot_id"]:
                                raise RuntimeError("Methods used different legal snapshots")
                            row.update(project_id=project["id"], scenario=project["scenario"], turn=turn, order=order,
                                       repeat=repeat, prompt_contains_old_artifact=project["prompt_contains_old_artifact"])
                            rows.append(row)
                            append(out / "replay.jsonl", row)
                    # Shared finalized answer becomes legal only for a later request.
                    store.append([MemoryEvent(project["id"], "artifact", order + 1, reference["text"],
                                              project_id=project["id"])])
                    print("complete", project["id"], turn, flush=True)
                    save(out / "summary.json", summarize(rows, methods))
        provenance["status"] = "complete"
        provenance["rows"] = len(rows)
    except Exception as error:
        provenance.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(out / "provenance.json", provenance)


if __name__ == "__main__":
    main()
