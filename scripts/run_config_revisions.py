"""Full-cost, strict greedy replay of frozen public CI file revisions."""
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
import yaml

from memorydraft.backends.dflash_reuse import ReuseConfig, generate_reuse
from memorydraft.evaluation.config_revisions import (METHODS_BY_MODE, answer_event, load_projects,
                                                     messages, methods_for_split, quality, request_text, working_event)
from memorydraft.memory.store import MemoryStore, Snapshot
from memorydraft.retrieval.arctic import ArcticMatcher
from memorydraft.retrieval.reuse import ReuseMatcher
from memorydraft.retrieval.suffix import SuffixArrayIndex
from scripts.run_reuse_pilot import append, digest, save


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/research/config-revisions"


def summaries(rows, methods_by_mode=METHODS_BY_MODE):
    result = {}
    for mode, methods in methods_by_mode.items():
        group = [r for r in rows if r["mode"] == mode]
        times = {m: sum(r["latency_seconds"] for r in group if r["method"] == m) for m in methods}
        result[mode] = dict(times=times, rows=len(group), correct=all(r["correct"] for r in group),
                            vs_dflash={m: times["DFlash"] / t for m, t in times.items() if t})
    return result


def verify_assets():
    dflash = json.loads((ROOT / "configs/dflash-manifest.json").read_text())
    suffix = json.loads((ROOT / "configs/suffix-manifest.json").read_text())
    for directory, entries in ((ROOT / "models/dflash-qwen3-8b", dflash["code"] + [dflash["weight"]]),
                               (ROOT / "models/arctic-suffix-audit", suffix["files"])):
        for entry in entries:
            with (directory / entry["filename"]).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                    raise ValueError("Audited asset hash mismatch: " + entry["filename"])
    binaries = list((ROOT / "models/arctic-suffix-audit/arctic_inference/suffix_decoding").glob("_C*.so"))
    if len(binaries) != 1:
        raise ValueError("Build exactly one pinned SuffixDecoding C++ component")
    return dict(dflash=dflash, suffix=suffix,
                suffix_binary_sha256=hashlib.sha256(binaries[0].read_bytes()).hexdigest())


def development_reference(directory: Path, data_sha256: str):
    previous = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (previous["status"] != "complete" or previous.get("rows") != 90
            or previous["data_sha256"] != data_sha256
            or any(summary[mode]["rows"] != count or not summary[mode]["correct"]
                   for mode, count in (("full", 72), ("diff", 18)))):
        raise ValueError("Complete and audit the unchanged development replay before confirmation")
    return dict(source_sha256=previous["source"]["source_sha256"], data_sha256=data_sha256, rows=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "confirmation"), default="development")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contract-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--contract-probe", action="store_true")
    args = parser.parse_args()
    if args.contract_probe and (args.split != "development" or args.contract_version != 2):
        raise ValueError("The one contract probe uses only development failures with explicit contract v2")
    out = args.output or ROOT / ("results/config-revisions-contract-probe" if args.contract_probe else
                                 "results/config-revisions-" + args.split)
    if out.exists():
        raise RuntimeError("Preserve existing evidence; select a new output path")
    projects = load_projects(DATA, args.split)
    if args.contract_probe:
        projects = [p for p in projects if p["id"] in {"astral-sh--ruff", "psf--requests"}]
    repeats = 1 if args.split == "development" else 3
    selected_methods = methods_for_split(args.split)
    if args.contract_probe:
        selected_methods = {"full": ("DFlash", "Global")}
    out.mkdir(parents=True)
    (out / "stores").mkdir()
    provenance = dict(status="started", split=args.split, repeats=repeats, methods=selected_methods, quality_version=2,
                      contract_version=args.contract_version, contract_probe=args.contract_probe,
                      config=asdict(ReuseConfig()), budget_full=3072, budget_diff=768,
                      source=json.loads((ROOT / "SOURCE_MANIFEST.json").read_text()),
                      data_sha256=hashlib.sha256((DATA / "projects.json").read_bytes()).hexdigest(),
                      transformers=transformers.__version__, torch=torch.__version__, yaml=yaml.__version__,
                      dtype="float32", attention="eager", threads=4)
    save(out / "provenance.json", provenance)
    stores, rows = {}, []
    try:
        if transformers.__version__ != "4.57.3":
            raise ValueError("Use isolated .venv-dflash")
        if args.split == "confirmation" or args.contract_probe:
            provenance["development_reference"] = development_reference(ROOT / "results/config-revisions-development",
                                                                         provenance["data_sha256"])
        provenance["assets"] = verify_assets()
        provenance["gpu"] = torch.cuda.get_device_name(0)
        config = json.loads((ROOT / "configs/a6000-qwen3.json").read_text())
        provenance["target"] = config["target"]
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(config["target"]["model"], local_files_only=True)
        # Check sizes without using expected outputs to select or drop any task.
        preflight = [dict(project_id=p["id"], turn=t,
                          input_tokens=len(tokenizer.apply_chat_template(messages(request_text(p, t, contract_version=args.contract_version)),
                                add_generation_prompt=True, enable_thinking=False)),
                          expected_tokens=len(tokenizer.encode(p["turns"][t]["expected"], add_special_tokens=False)))
                     for p in projects for t in range(1 if args.contract_probe else 2)]
        save(out / "preflight.json", preflight)
        if any(r["input_tokens"] > 6144 or r["expected_tokens"] >= 3072 for r in preflight):
            raise ValueError("Predeclared budget insufficient; revise protocol before inference, never drop cases")
        target = AutoModelForCausalLM.from_pretrained(config["target"]["model"], local_files_only=True,
                    torch_dtype=torch.float32, attn_implementation="eager").to("cuda:0").eval()
        draft = AutoModel.from_pretrained(ROOT / "models/dflash-qwen3-8b", trust_remote_code=True,
                    local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager").to("cuda:0").eval()
        provenance["load_seconds"] = time.perf_counter() - load_start
        eos = target.generation_config.eos_token_id
        stops = [eos] if isinstance(eos, int) else eos
        warm = tokenizer.apply_chat_template(messages("Reply with exactly: ready"), add_generation_prompt=True,
                                             enable_thinking=False, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            target.generate(warm, attention_mask=torch.ones_like(warm), do_sample=False, use_model_defaults=False,
                            max_new_tokens=16, eos_token_id=stops, pad_token_id=tokenizer.pad_token_id)
            draft.spec_generate(target=target, input_ids=warm, max_new_tokens=16, stop_token_ids=stops, temperature=0.)
            generate_reuse(target, draft, warm, max_new_tokens=16, stop_token_ids=stops)
        torch.cuda.synchronize()
        save(out / "projects.json", projects)
        save(out / "provenance.json", provenance)

        @torch.inference_mode()
        def generate(project, turn, mode, method, repeat, reference=None):
            budget = 3072 if mode == "full" else 768
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            text = request_text(project, turn, mode, contract_version=args.contract_version)
            identity = (project["id"], mode, method, repeat)
            if identity not in stores:
                stores[identity] = MemoryStore(out / "stores" / ("-".join(map(str, identity)) + ".db"))
            store = stores[identity]
            event = working_event(project, mode, turn)
            store.append([event])
            order = 10 + 10 * turn
            snapshot = store.snapshot(event.user_id, order)
            snapshot_id = snapshot.snapshot_id
            ids = tokenizer.apply_chat_template(messages(text), add_generation_prompt=True,
                                    enable_thinking=False, return_tensors="pt").to("cuda:0")
            prompt_hash = digest(ids[0].tolist())
            matcher, detail, index_bytes = None, None, 0
            if method in {"Global", "Context"}:
                resource = Snapshot(event.user_id, order, ()) if method == "Context" else snapshot
                index = SuffixArrayIndex(resource, tokenizer, min_match=3, max_match=32, prompt_text=text,
                                         include_user_records=True)
                matcher, index_bytes = ReuseMatcher(index, cursor=False), index.bytes_used
            elif method == "Arctic":
                matcher = ArcticMatcher(snapshot, tokenizer, text, directory=ROOT / "models/arctic-suffix-audit")
                index_bytes = matcher.bytes_used
            if method in {"AR", "Reference"}:
                output = target.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                            use_model_defaults=False, max_new_tokens=budget, use_cache=True,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=stops)
            elif method == "DFlash":
                output = draft.spec_generate(target=target, input_ids=ids, max_new_tokens=budget,
                                              stop_token_ids=stops, temperature=0.)
            else:
                detail = generate_reuse(target, draft, ids, max_new_tokens=budget,
                                         stop_token_ids=stops, matcher=matcher)
                output = detail.sequences
            torch.cuda.synchronize()
            tokens = output[0, ids.shape[1]:].tolist()
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            correct = reference is None or (tokens == reference["token_ids"]
                        and prompt_hash == reference["prompt_sha256"] and snapshot_id == reference["snapshot_id"])
            if not correct:
                save(out / "failure.json", dict(project_id=project["id"], turn=turn, mode=mode, method=method,
                     token_ids=tokens, reference=reference, prompt_sha256=prompt_hash, snapshot_id=snapshot_id))
                raise RuntimeError("Strict greedy/prompt/snapshot mismatch")
            write_started = time.perf_counter()
            store.append([answer_event(project, mode, turn, decoded)])
            write_seconds = time.perf_counter() - write_started
            latency = time.perf_counter() - started
            return dict(project_id=project["id"], turn=turn, order=order, mode=mode, method=method, repeat=repeat,
                        token_ids=tokens, text=decoded, prompt_sha256=prompt_hash, snapshot_id=snapshot_id,
                        correct=correct, latency_seconds=latency, write_seconds=write_seconds,
                        input_tokens=ids.shape[1], budget=budget, budget_hit=len(tokens) >= budget,
                        eos=bool(tokens and tokens[-1] in stops), index_bytes=index_bytes,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(), rounds=detail.rounds if detail else [],
                        lookups=matcher.lookups if matcher else {}, pending_peak=detail.feature_peak_tokens if detail else None,
                        buffer_flushes=detail.buffer_flushes if detail else 0)

        for i, project in enumerate(projects):
            for mode, methods in selected_methods.items():
                for turn in range(2 if mode == "full" and not args.contract_probe else 1):
                    reference = generate(project, turn, mode, "Reference", 0)
                    reference["quality"] = quality(reference["text"], project, turn, mode)
                    append(out / "references.jsonl", reference)
                    for repeat in range(repeats):
                        shuffled = list(methods)
                        random.Random(20261018 + i * 37 + turn * 17 + repeat * 101 + (mode == "diff") * 997).shuffle(shuffled)
                        for method in shuffled:
                            row = generate(project, turn, mode, method, repeat, reference)
                            rows.append(row)
                            append(out / "replay.jsonl", row)
                    save(out / "summary.json", summaries(rows, selected_methods))
                    print("complete", project["id"], mode, turn, "quality", reference["quality"], flush=True)
        provenance.update(status="complete", rows=len(rows))
    except Exception as error:
        provenance.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for store in stores.values():
            store.db.close()
        save(out / "provenance.json", provenance)


if __name__ == "__main__":
    main()
