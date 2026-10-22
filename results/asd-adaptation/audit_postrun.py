"""Rebuild source/provenance, train/eval separation and preserved evidence proof."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from memorydraft.evaluation.adaptation import SOURCE_ROUNDS, prefix_digest
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.run_asd_adaptation_collect import questions
from scripts.write_asd_adaptation_report import analyze

root = Path.cwd()
base = root/"results/asd-adaptation"
freeze = "4d7ac6eca2ab9970e9f71314fc77b9b9de79e140"
sha = lambda b: hashlib.sha256(b).hexdigest()
git = lambda *args: subprocess.check_output(["git", *args], cwd=root)


def verify_source(manifest, expected):
    assert manifest["commit"] == expected
    assert sha(json.dumps(manifest["files"], sort_keys=True).encode()) == manifest["source_sha256"]
    entries = []
    for raw in git("ls-tree", "-rz", expected).split(b"\0"):
        if raw:
            meta, name = raw.split(b"\t", 1)
            entries.append((name.decode(), meta.split()[2].decode()))
    data = subprocess.check_output(["git", "cat-file", "--batch"], cwd=root,
                    input="".join(oid+"\n" for _, oid in entries).encode())
    position = 0
    for name, oid in entries:
        end = data.index(b"\n", position)
        header = data[position:end].split()
        assert header[:2] == [oid.encode(), b"blob"]
        size = int(header[2])
        assert sha(data[end+1:end+1+size]) == manifest["files"][name], name
        position = end+size+2
    assert len(entries) == len(manifest["files"]) == 1584
    return len(entries)


manifest = json.loads((base/"ASD_ADAPTATION_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
execution = json.loads((base/"execution-verification.json").read_text(encoding="utf-8"))
assert execution["status"] == "complete" and execution["commit"] == freeze
assert execution["phases"] == ["collect", "train", "evaluate"]
assert execution["files_verified_before"] == execution["files_verified_after"] == verify_source(manifest, freeze)
failed = base/"failed-peft-preflight"
assert verify_source(json.loads((failed/"ASD_ADAPTATION_SOURCE_MANIFEST.json").read_text()), "9bb81bed254761efd4cef30d2845a59cfa71126d") == 1584
failure_info = json.loads((failed/"execution-verification.json").read_text())
assert failure_info["status"] == "failed" and failure_info["phases"] == []
frozen = ["memorydraft/evaluation/adaptation.py", "scripts/prepare_asd_adaptation.py",
          "scripts/run_asd_adaptation_collect.py", "scripts/run_asd_adaptation_train.py", "scripts/run_asd_adaptation_eval.py",
          "scripts/run_asd_adaptation_pipeline.py", "tests/test_adaptation.py", "docs/ASD_ADAPTATION_PROTOCOL.md",
          "data/research/asd-adaptation/questions.json", "data/research/asd-adaptation/manifest.json",
          "data/research/asd-adaptation/GSM8K_LICENSE.txt"]
for n in frozen:
    assert (root/n).read_bytes().replace(b"\r\n", b"\n") == git("show", freeze+":"+n), n
analysis_commit = "f8e68035f6ae3f01e8e0fa531260a248e03fe7f8"
analysis_files = ["scripts/write_asd_adaptation_report.py", "tests/test_asd_adaptation_report.py"]
for n in analysis_files:
    assert (root/n).read_bytes().replace(b"\r\n", b"\n") == git("show", analysis_commit+":"+n), n
examples, sources, protected, rows, infos = [], {}, {}, [], []
for phase in ("collect-0", "collect-1", "train-ordinary", "train-asd-prefix", "eval-0", "eval-1"):
    p = json.loads((base/phase/"provenance.json").read_text(encoding="utf-8"))
    assert p["status"] == "complete" and p["original_unchanged"]
    for name, h in p["original_sha256"].items():
        assert name not in protected or protected[name] == h
        protected[name] = h
    if phase.startswith("collect"):
        examples.extend(read_rows(base/phase/"examples.jsonl"))
        for r in read_rows(base/phase/"sources.jsonl"):
            key = r["request_id"], r["source_policy"]
            assert key not in sources
            sources[key] = r
    if phase.startswith("train"):
        assert p["updates"] == 128 and p["accumulation"] == 2
        assert p["teacher_parameter_versions_unchanged"] and p["teacher_gradients_absent"] and p["draft_base_versions_unchanged"]
        assert p["first_grad_norm"] > 0 and 0 < p["peak_gpu_bytes"] <= 46*2**30
        curve = list(read_rows(base/phase/"metrics.jsonl"))
        assert [r["update"] for r in curve] == [0, 32, 64, 96, 128]
        best = min(curve[1:], key=lambda r: r["calibration"]["loss"])
        assert best["update"] == p["selected_step"] and best["calibration"]["loss"] == p["selected_calibration_loss"]
    if phase.startswith("eval"):
        assert p["base_restored"] and p["teacher_parameter_versions_unchanged"] and p["split"] == "development"
        rows.extend(read_rows(base/phase/"rows.jsonl"))
        infos.append(p)
expected_ids = {r["id"] for r in questions()}
assert {q for q, _ in sources} == expected_ids and len(sources) == 80
pair_keys = {}
for r in examples:
    q, arm, index = r["request_id"], r["source_policy"], r["source_round"]
    source = sources[q, arm]
    assert r["prefix_sha256"] == prefix_digest(r["prefix_ids"])
    assert r["generated_before"] == source["rounds"][index]["generated_before"]
    assert r["prefix_ids"][source["prompt_tokens"]:] == source["token_ids"][:r["generated_before"]]
    assert r["context_tokens"] == len(r["prefix_ids"])-1 and 1 <= len(r["teacher_tokens"]) <= 15
    if arm == "asd-prefix":
        assert r["past_spent"] == source["events"][index-1]["spent"]
        assert r["past_relaxed"] == sum(x["relaxed"] for x in source["events"][:index])
    arms = pair_keys.setdefault((q, index), {})
    assert arm not in arms
    arms[arm] = r
assert all(set(v) == {"ordinary", "asd-prefix"} for v in pair_keys.values())
expected_pairs = {(q, index) for q in expected_ids for index in SOURCE_ROUNDS
                  if all(index < len(sources[q, a]["rounds"]) for a in ("ordinary", "asd-prefix"))}
assert set(pair_keys) == expected_pairs
assert not expected_ids & {r["request_id"] for r in rows} and len(rows) == 540
for row in rows:
    for r in row["rounds"]:
        frontier = row["prompt_tokens"]+r["generated_before"]-1
        assert r["draft_cache_before"]+r["pending_before"] == frontier
        assert r["target_cache_after"] == frontier+r["advance"]
legacy = []
for name, h in protected.items():
    data = (root/name).read_bytes()
    if sha(data) != h:
        assert name in {"results/research-audit/model_preflight.json", "results/research-audit/pilot_summary.json"}
        assert sha(data.replace(b"\r\n", b"\n")) == h, name
        legacy.append(name)
prior = []
for name, commit in (("asd-suite-stage1", "dcfc01d"), ("asd-block-preflight", "2a4ab50"), ("asd-block-suite", "d72f6b6"),
                     ("asd-execution", "1e56ecb"), ("asd-cpu", "38ce3b3"), ("asd-compile", "4f3aa2a"),
                     ("asd-rmsnorm", "841e0e5"), ("asd-rmsnorm-suite", "e958532"), ("asd-rmsnorm-validation", "9b8176e"),
                     ("rmsnorm-runtime", "7991557"), ("asd-allocation", "1e95580"), ("asd-candidate-pilot", "70c2d74")):
    pub = json.loads((root/f"results/{name}/publication.json").read_text())
    for n, h in pub["files"].items():
        assert sha(git("show", commit+":"+n)) == h, n
    prior.append(dict(name=name, commit=git("rev-parse", commit).decode().strip(), files=len(pub["files"])))
refs = {r:git("rev-parse", r).decode().strip() for r in ("main", "codex/infra-interview-v0.3", "infra-v0.3-interview^{}")}
assert set(refs.values()) == {"27e36199f8a65acdb9a9c42ccb744ddfa9a9402b"}
archives = []
for path, dest in ((root/"artifacts/asd-adaptation-results.zip", base), (root/"artifacts/asd-adaptation-failed-peft-results.zip", failed)):
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            assert (dest/n).read_bytes() == z.read(n), n
    archives.append(dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes())))
saved = json.loads((base/"analysis.json").read_text(encoding="utf-8"))
rebuilt = analyze(rows)
assert {k:saved[k] for k in rebuilt} == rebuilt
result = dict(status="complete", source_commit=freeze, source_files=1584, failed_source_verified=True,
    frozen_files=frozen, analysis_commit=analysis_commit, analysis_files=analysis_files,
    protected_results=len(protected), legacy_crlf_normalized=legacy, prior_publications=prior,
    infra_refs=refs, source_generations=len(sources), examples=len(examples), paired_states=len(pair_keys),
    evaluation_generations=len(rows), training_evaluation_ID_disjoint=True,
    captured_source_frontiers_and_eval_KV_verified=True, all_evaluation_safe=saved["all_safe"], archives=archives)
(base/"postrun-audit.json").write_bytes((json.dumps(result, indent=2)+"\n").encode())
print(json.dumps(result))
