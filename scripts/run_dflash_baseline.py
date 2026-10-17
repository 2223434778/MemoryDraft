"""Use audited official DFlash spec_generate in an isolated Transformers environment."""
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


root = Path(__file__).resolve().parents[1]
out = root / "results/dflash-baseline"


def save(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def read_rows(path):
    path = Path(path)
    data = path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def main():
    if out.exists():
        raise RuntimeError("Preserve existing DFlash evidence")
    out.mkdir(parents=True)
    provenance = dict(status="started", dtype="float32", attention="eager", transformers=transformers.__version__,
                      torch=torch.__version__, block=16, max_new_tokens=128, repeats=3)
    try:
        if transformers.__version__ != "4.57.3":
            raise RuntimeError("Use isolated Transformers 4.57.3")
        manifest = json.loads((root / "configs/dflash-manifest.json").read_text())
        model_dir = root / "models/dflash-qwen3-8b"
        for item in manifest["code"] + [manifest["weight"]]:
            with (model_dir / item["filename"]).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise RuntimeError("DFlash asset checksum mismatch: " + item["filename"])
        provenance["dflash_manifest"] = manifest
        provenance["source_manifest"] = json.loads((root / "SOURCE_MANIFEST.json").read_text())
        config = json.loads((root / "configs/a6000-qwen3.json").read_text())
        provenance["target"] = config["target"]
        provenance["gpu"] = torch.cuda.get_device_name(0)
        save(out / "provenance.json", provenance)
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        tokenizer = AutoTokenizer.from_pretrained(config["target"]["model"], local_files_only=True)
        target = AutoModelForCausalLM.from_pretrained(config["target"]["model"], torch_dtype=torch.float32,
                                                    attn_implementation="eager", local_files_only=True).to("cuda:0").eval()
        draft = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32,
                                         attn_implementation="eager", local_files_only=True).to("cuda:0").eval()
        eos = target.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos

        @torch.inference_mode()
        def generate(messages, method, budget):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                                return_tensors="pt").to("cuda:0")
            if method == "AR":
                result = target.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                         use_model_defaults=False, max_new_tokens=budget, use_cache=True,
                                         pad_token_id=tokenizer.pad_token_id, eos_token_id=eos)
            else:
                result = draft.spec_generate(target=target, input_ids=ids, max_new_tokens=budget,
                                             stop_token_ids=eos, temperature=0.)
            torch.cuda.synchronize()
            tokens = result[0, ids.shape[1]:].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            return dict(method=method, token_ids=tokens, text=text, latency_seconds=time.perf_counter()-started,
                        target_prompt_sha256=hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),
                        peak_gpu_bytes=torch.cuda.max_memory_allocated())

        warm = [dict(role="user", content="Repeat exactly: A quick brown fox.")]
        for method in ("AR", "DFlash"):
            generate(warm, method, 16)
        requests = read_rows(root / "data/research/memory-test.jsonl")
        public = read_rows(root / "data/research/specbench-test.jsonl")
        categories = dict.fromkeys(row["scenario"] for row in public)
        requests += [next(row for row in public if row["scenario"] == category) for category in categories]
        previous = {}
        for stage in ("memory-test", "public-test"):
            for row in read_rows(root / f"results/qwen3-pilot/{stage}/replay-k2.jsonl"):
                if row["method"] == "AR":
                    previous[row["request_id"]] = row
        all_rows = []
        with (out / "replay.jsonl").open("w", encoding="utf-8") as stream:
            for request in requests:
                reference = generate(request["messages"], "AR", 128)
                old = previous[request["id"]]
                old_match = reference["token_ids"] == old["token_ids"] and reference["target_prompt_sha256"] == old["target_prompt_sha256"]
                if not old_match:
                    save(out / "old-target-difference.json", dict(request_id=request["id"], current=reference, old=old))
                    raise RuntimeError("Isolated environment differs from the fixed old Target reference")
                for repeat in range(3):
                    methods = ["AR", "DFlash"]
                    random.Random(20261017 + repeat).shuffle(methods)
                    for method in methods:
                        row = generate(request["messages"], method, 128)
                        row.update(request_id=request["id"], repeat=repeat, scenario=request["scenario"],
                                   correct=row["token_ids"] == reference["token_ids"], snapshot_id="none-used")
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        all_rows.append(row)
                        if not row["correct"]:
                            save(out / "failure.json", dict(reference=reference, result=row))
                            raise RuntimeError("DFlash strict greedy mismatch; evidence saved")
                print("complete", request["id"], flush=True)
        times = {method: sum(row["latency_seconds"] for row in all_rows if row["method"] == method) for method in ("AR", "DFlash")}
        save(out / "summary.json", dict(times=times, speedup=times["AR"]/times["DFlash"], rows=len(all_rows)))
        provenance["status"] = "complete"
    except Exception as error:
        provenance.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(out / "provenance.json", provenance)


if __name__ == "__main__":
    main()
