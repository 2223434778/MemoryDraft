"""One bounded, project-separated H distillation experiment on the existing backend."""
import gc
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, GenerationConfig

from memorydraft.backends.transformers import TransformersBackend
from memorydraft.drafters.candidates import IndependentDraft
from memorydraft.evaluation.distillation import projects, response_sample, short_history, task_family
from memorydraft.evaluation.history_alignment import paired_delta
from memorydraft.evaluation.prefixes import common_prefix
from memorydraft.evaluation.replay import replay
from memorydraft.memory.store import MemoryEvent, MemoryStore, Snapshot
from memorydraft.retrieval.pool import LexicalEncoder
from memorydraft.retrieval.suffix import SuffixArrayIndex
from scripts.run_h_causal import save, sha


root = Path(__file__).resolve().parents[1]
out = root / "results/h-training"
adapter = root / "models/h-training-adapter"
UNRELATED = "A quiet garden contains orchids, ferns and moss near a pond."


def write_rows(path, rows):
    Path(path).write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))


def teacher_labels(backend, requests, store):
    result = []
    for request in requests:
        history = short_history(backend, request, store.snapshot(request["user_id"], request["order"]))
        answer = backend.generate(request["messages"], store.snapshot(request["user_id"], request["order"]),
                                  method="AR", max_new_tokens=128)
        result.append(dict(request_id=request["id"], token_ids=answer.token_ids, text=answer.text,
                           selected_history=history,
                           target_prompt_sha256=answer.target_prompt_sha256, seconds=answer.latency_seconds,
                           hit_budget=len(answer.token_ids) == 128))
    return result


def samples(backend, requests, labels, store):
    index = {row["request_id"]: row for row in labels}
    result = []
    for request in requests:
        history = index[request["id"]]["selected_history"]
        for condition, text in (("empty", ""), ("relevant", history), ("unrelated", UNRELATED)):
            result.append((request["id"], condition, response_sample(backend, request["messages"],
                                                                    index[request["id"]]["token_ids"], text)))
    return result


def train(backend, training, development):
    import peft
    from peft import LoraConfig, TaskType, get_peft_model
    if peft.__version__ != "0.11.1":
        raise RuntimeError("This pilot pins PEFT 0.11.1")
    if adapter.exists():
        raise RuntimeError("Preserve existing adapter")
    torch.manual_seed(20261017)
    model = get_peft_model(backend.draft, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
                           target_modules=["q_proj", "v_proj"], bias="none", task_type=TaskType.CAUSAL_LM))
    model.config.use_cache = False
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    metrics, best = [], float("inf")
    started = time.perf_counter()
    for epoch in range(4):
        model.train()
        order = list(range(len(training)))
        random.Random(20261017 + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for step, index in enumerate(order):
            batch = {key: value.to(model.device) for key, value in training[index][2].items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss_sum += loss.item()
            (loss / 4).backward()
            if (step + 1) % 4 == 0 or step + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        weighted, count = 0., 0
        with torch.no_grad():
            for _, _, item in development:
                batch = {key: value.to(model.device) for key, value in item.items()}
                n = int((batch["labels"][:, 1:] != -100).sum())
                weighted += model(**batch).loss.item() * n
                count += n
        value = weighted / count
        metrics.append(dict(epoch=epoch + 1, train_loss=loss_sum / len(order), dev_response_loss=value))
        if value < best:
            best = value
            model.save_pretrained(adapter)
        save(out / "training-metrics.json", dict(epochs=metrics, trainable_parameters=sum(p.numel() for p in params),
                                                 seconds=time.perf_counter() - started, selected_by="dev_response_loss"))
        print("epoch", json.dumps(metrics[-1]), flush=True)
    del optimizer, params
    model.load_adapter(adapter, adapter_name="selected")
    model.set_adapter("selected")
    trained = model.merge_and_unload()
    trained.config.use_cache = True
    trained.eval()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return trained


def score(backend, requests, labels, store, models, path, block=2, stride=1):
    index = {row["request_id"]: row for row in labels}
    rows = []
    with path.open("w", encoding="utf-8") as stream:
        for request in requests:
            snapshot = store.snapshot(request["user_id"], request["order"])
            text = short_history(backend, request, snapshot)
            prefixes = dict(D_native=backend.prompt(request["messages"]), D_format=backend.prompt(request["messages"], ""),
                            H=backend.prompt(request["messages"], text), H_unrelated=backend.prompt(request["messages"], UNRELATED))
            reference = index[request["id"]]
            target = backend.prompt(request["messages"])
            config = GenerationConfig(do_sample=False, max_length=target.shape[1] + len(reference["token_ids"]) + 1,
                                      eos_token_id=backend.target.generation_config.eos_token_id,
                                      pad_token_id=backend.tokenizer.pad_token_id, use_cache=True)
            providers = {label + "_" + name: IndependentDraft(model, prefix, target.shape[1], config, block)
                         for label, model in models.items() for name, prefix in prefixes.items()}
            retrieval = SuffixArrayIndex(snapshot, backend.tokenizer)
            for position in range(0, len(reference["token_ids"]), stride):
                confirmed = torch.tensor([reference["token_ids"][:position]], device=backend.draft.device, dtype=torch.long)
                expected = reference["token_ids"][position:position + block]
                lengths, proposals = {}, {}
                for name, provider in providers.items():
                    proposal = provider.propose(confirmed)[0, :len(expected)].tolist()
                    proposals[name], lengths[name] = proposal, common_prefix(proposal, expected)
                    provider.reconcile(position)
                _, _, match = retrieval.match(target[0].tolist() + reference["token_ids"][:position], len(expected))
                row = dict(request_id=request["id"], scenario=request["scenario"], position=position,
                           lengths=lengths, proposal_ids=proposals, reference_ids=expected, r_match_length=match,
                           snapshot_id=snapshot.snapshot_id, target_prompt_sha256=reference["target_prompt_sha256"])
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                rows.append(row)
    return {"trained_H_minus_D": paired_delta(rows, "trained_H", "trained_D_format"),
            "base_H_minus_D": paired_delta(rows, "base_H", "base_D_format"),
            "trained_H_minus_unrelated": paired_delta(rows, "trained_H", "trained_H_unrelated")}


class Variants:
    def __init__(self, backend, models, block, gate=()):
        self.backend, self.models, self.block, self.gate = backend, models, block, set(gate)
        self.tokenizer = backend.tokenizer

    def generate(self, messages, snapshot, *, method, **kwargs):
        if method == "AR":
            return self.backend.generate(messages, snapshot, method="AR", **kwargs)
        variant, route = method.split("_", 1)
        self.backend.draft = self.models[{"BASE": "base", "TRAIN": "trained"}[variant]]
        actual = {"D": "D_native", "Dformat": "D", "H": "H", "off": "RHD", "on": "RHD", "gated": "RHD"}[route]
        gate_start = time.perf_counter()
        if route in {"off", "on", "gated"}:
            kwargs["enable_history"] = route == "on" or (route == "gated" and task_family(messages) in self.gate)
        gate_seconds = time.perf_counter() - gate_start
        kwargs.update(block_size=self.block.get(route, 2), top_k=1, token_budget=128, history_budget=128)
        result = self.backend.generate(messages, snapshot, method=actual, **kwargs)
        result.latency_seconds += gate_seconds
        if result.ttft_seconds is not None:
            result.ttft_seconds += gate_seconds
        result.stage_seconds["gate_host"] = gate_seconds
        result.method = method
        return result


def main(resume_assets=False):
    if out.exists() and not resume_assets:
        raise RuntimeError("Preserve existing experiment")
    if resume_assets and (not (out / "history.jsonl").exists() or adapter.exists() or (out / "test-labels.jsonl").exists()):
        raise RuntimeError("Resume only the frozen asset stage, before adapter training or test labels")
    out.mkdir(parents=True, exist_ok=resume_assets)
    if resume_assets:
        save(out / "asset-stage-provenance.json", json.loads((out / "provenance.json").read_text()))
    provenance = dict(status="started", protocol_sha256=sha(root / "docs/H_TRAINING_PROTOCOL.md"),
                      seed=20261017, counts=dict(train=24, dev=8, test=8), online_scope="synthetic pilot")
    try:
        manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
        if any(sha(root / path) != digest for path, digest in manifest["files"].items()):
            raise RuntimeError("Source manifest mismatch")
        provenance["source_manifest"] = manifest
        config = json.loads((root / "configs/a6000-qwen3.json").read_text())
        config.update(dtype="float32", attention="eager")
        provenance["config"] = config
        provenance.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                          weight_verification=json.loads((root / "models/qwen3-weights-verified.json").read_text()),
                          training_config=dict(peft="0.11.1", rank=8, alpha=16, dropout=0, modules=["q_proj", "v_proj"],
                                               learning_rate=1e-4, epochs=4, accumulation=4))
        save(out / "provenance.json", provenance)
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        backend = TransformersBackend.load(config)
        provenance["tokenizer_fingerprint"] = backend.fingerprint
        events = []
        # Every past answer is generated before any current answer label exists.
        past_evidence = []
        if resume_assets:
            from memorydraft.evaluation.replay import read_jsonl
            events = [MemoryEvent(**row) for row in read_jsonl(out / "history.jsonl")]
            past_evidence = read_jsonl(out / "past-generations.jsonl")
            expected = [project["past"] for split in ("train", "dev", "test") for project in projects(split)]
            if [row["request"] for row in past_evidence] != expected:
                raise ValueError("Frozen past task construction changed")
            originals = {(event.user_id, event.record_id, event.order): event.text for event in events}
            if len(events) != 80 or any(originals[(row["request"]["user_id"], row["request"]["user_id"] + "-old", 1)] != row["text"] for row in past_evidence):
                raise ValueError("Frozen past history changed")
            provenance["resumed_frozen_asset_sha256"] = sha(out / "history.jsonl")
        for split in (() if resume_assets else ("train", "dev", "test")):
            for project in projects(split):
                past = project["past"]
                answer = backend.generate(past["messages"], Snapshot(past["user_id"], 0, ()), method="AR", max_new_tokens=192)
                events.extend([MemoryEvent(past["user_id"], past["user_id"] + "-old", 1, answer.text),
                               MemoryEvent(past["user_id"], past["user_id"] + "-other", 2, UNRELATED)])
                past_evidence.append(dict(request=past, token_ids=answer.token_ids, text=answer.text,
                                          hit_budget=len(answer.token_ids) == 192, seconds=answer.latency_seconds))
        if not resume_assets:
            write_rows(out / "past-generations.jsonl", past_evidence)
            write_rows(out / "history.jsonl", [event.__dict__ for event in events])
        provenance["frozen_history_sha256"] = sha(out / "history.jsonl")
        print("40 past answers frozen", flush=True)
        with MemoryStore(out / "history.db") as store:
            store.append(events)
            requests = {split: [p["request"] for p in projects(split)] for split in ("train", "dev")}
            labels = {split: teacher_labels(backend, values, store) for split, values in requests.items()}
            for split in requests:
                write_rows(out / f"{split}-requests.jsonl", requests[split])
                write_rows(out / f"{split}-labels.jsonl", labels[split])
            trained = train(backend, samples(backend, requests["train"], labels["train"], store),
                            samples(backend, requests["dev"], labels["dev"], store))
            provenance["adapter_files_sha256"] = {p.name: sha(p) for p in adapter.iterdir() if p.is_file()}
            provenance["training_complete_before_test_labels"] = True
            save(out / "provenance.json", provenance)
            original = AutoModelForCausalLM.from_pretrained(config["draft"]["model"], revision=config["draft"]["revision"],
                         torch_dtype=torch.float32, attn_implementation="eager").to("cuda:0").eval()
            for model in (original, trained):
                model.generation_config.num_assistant_tokens_schedule = "constant"
                model.generation_config.assistant_confidence_threshold = 0.
            models = dict(base=original, trained=trained)
            backend.draft = trained
            # Only development data chooses lengths and eligibility; no test labels loaded yet.
            dev_summaries = {}
            for block in (2, 4):
                wrapper = Variants(backend, models, dict(D=block, H=block, Dformat=block))
                dev_summaries[block] = replay(wrapper, store, requests["dev"], ["AR", "TRAIN_D", "TRAIN_Dformat", "TRAIN_H"],
                                               out / f"dev-k{block}.jsonl", encoder=LexicalEncoder(), max_new_tokens=128)
            best_d = min((2, 4), key=lambda k: dev_summaries[k]["TRAIN_D"]["total_seconds"])
            best_h = min((2, 4), key=lambda k: dev_summaries[k]["TRAIN_H"]["total_seconds"])
            selection = dict(D=best_d, Dformat=best_h, H=best_h, off=best_d, on=best_d, gated=best_d)
            # Route integration uses one shared block size; its off/on control is symmetric.
            score(backend, requests["dev"], labels["dev"], store, models, out / "dev-prefixes.jsonl", block=best_d, stride=4)
            wrapper = Variants(backend, models, selection)
            dev_router = replay(wrapper, store, requests["dev"], ["AR", "TRAIN_off", "TRAIN_on"],
                                out / "dev-router.jsonl", encoder=LexicalEncoder(), max_new_tokens=128)
            from memorydraft.evaluation.replay import read_jsonl
            prefix_rows, router_rows = read_jsonl(out / "dev-prefixes.jsonl"), read_jsonl(out / "dev-router.jsonl")
            gate = []
            for family in sorted({request["scenario"] for request in requests["dev"]}):
                group = [row for row in prefix_rows if row["scenario"] == family]
                quality_gain = sum(row["lengths"]["trained_H"] - row["lengths"]["trained_D_format"] for row in group)
                times = {method: sum(row["latency_seconds"] for row in router_rows if row["scenario"] == family and row["method"] == method)
                         for method in ("TRAIN_off", "TRAIN_on")}
                if quality_gain > 0 and times["TRAIN_on"] < times["TRAIN_off"]:
                    gate.append(family)
            save(out / "dev-selection.json", dict(blocks=selection, gate=gate, summaries=dev_summaries,
                                                  router_summary=dev_router, quality_stride=4,
                                                  gate_rule="positive H-D_format quality and faster matched router in development family"))
            test_requests = [p["request"] for p in projects("test")]
            test_labels = teacher_labels(backend, test_requests, store)
            write_rows(out / "test-requests.jsonl", test_requests)
            write_rows(out / "test-labels.jsonl", test_labels)
            summary = score(backend, test_requests, test_labels, store, models, out / "test-prefixes.jsonl", block=best_d)
            save(out / "quality-summary.json", summary)
            print("Held-out quality", json.dumps(summary), flush=True)
            wrapper = Variants(backend, models, selection, gate)
            methods = ["AR", "BASE_D", "BASE_H", "TRAIN_D", "TRAIN_H", "TRAIN_off", "TRAIN_on"]
            if gate:
                methods.append("TRAIN_gated")
            result = replay(wrapper, store, test_requests, methods, out / "test-replay.jsonl", repeats=3,
                            encoder=LexicalEncoder(), max_new_tokens=128)
            save(out / "test-summary.json", result)
        provenance["status"] = "complete"
        print("Training pilot complete", json.dumps(result), flush=True)
    except Exception as error:
        provenance.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(out / "provenance.json", provenance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-assets", action="store_true")
    main(parser.parse_args().resume_assets)
