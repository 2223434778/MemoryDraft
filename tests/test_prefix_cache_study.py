from copy import deepcopy
import json
from pathlib import Path

import pytest

from memorydraft.evaluation.long_context import load_data
from memorydraft.evaluation.prefix_cache import analyze, compare


def fixture():
    item = dict(task_id="q",level=512,prompt_tokens=500,prompt_sha256="p",request=dict(id="r"))
    base = dict(request_id="r",prompt_tokens=500,prompt_sha256="p",execution_profile="rmsnorm-inductor",
        latency_seconds=3.,ttft_seconds=1.,token_ids=[7]*128,text="#### 7",finish_reason="eos",
        snapshot_id="snapshot",peak_gpu_bytes=1,draft_adapter_sha256=None)
    rows = []
    for method in ("DFlash","ASD"):
        for stage in ("natural","prefix"):
            for state in ("off","cold","hit"):
                counters = dict(rounds=10,compile_graphs_before=4,compile_graphs_after=4)
                if state != "off":counters.update(prefix_cache_hit=int(state == "hit"),
                    prefix_cache_prefix_tokens=400,prefix_cache_bytes=200)
                rows.append(dict(base,task_id="q",level=512,method=method,stage=stage,cache_state=state,
                    approximate=method == "ASD",asd=dict(budget=8,spent=1,gate=.25,cap=2) if method == "ASD" else None,counters=counters))
    return rows,[item]


def test_cache_analysis_keeps_natural_prefix_and_cold_cost_separate():
    rows,items = fixture()
    for r in rows:
        if r["cache_state"] == "cold":r["latency_seconds"] = 4.
        if r["cache_state"] == "hit":r.update(latency_seconds=2.,ttft_seconds=.1)
    result = analyze(rows,items)
    assert result["all_outputs_and_decisions_equal"]
    data = result["summary"]["512/ASD/prefix"]
    assert data["all_full_128"] and data["cold_plus_one_hit_ratio"] == 1
    assert data["hit_ratio"] == 2/3


def test_output_and_round_differences_remain_visible_and_are_not_speedup_samples():
    rows,items = fixture()
    rows[-1]["token_ids"] = [8]*100
    rows[-1]["counters"]["rounds"] += 1
    result = analyze(rows,items)
    assert not result["all_outputs_and_decisions_equal"]
    assert result["differences"][0]["changed"] == ["token_ids","rounds"]
    assert result["summary"]["512/ASD/prefix"]["same_output_and_decisions"] == 0


def test_spending_drift_is_measured_without_changing_acceptance_budget():
    rows,items = fixture()
    before = deepcopy(rows[-1])
    rows[-1]["asd"]["spent"] += 1e-6
    result = analyze(rows,items)
    assert result["all_outputs_and_decisions_equal"] and result["max_spent_abs_delta"] > 0
    assert not compare(before,rows[-1])["spent_bitwise_equal"]
    rows[-1]["asd"]["spent"] = 8.00000001
    with pytest.raises(ValueError,match="budget"):analyze(rows,items)


@pytest.mark.parametrize("bad",["missing","duplicate","input","hit","bytes","compile"])
def test_invalid_cache_matrix_is_rejected(bad):
    rows,items = fixture()
    if bad == "missing":rows.pop()
    elif bad == "duplicate":rows[-1] = deepcopy(rows[0])
    elif bad == "input":rows[-1]["prompt_sha256"] = "changed"
    elif bad == "hit":rows[-1]["counters"]["prefix_cache_hit"] = 0
    elif bad == "bytes":rows[-1]["counters"]["prefix_cache_bytes"] = 3*2**30
    elif bad == "compile":rows[-1]["counters"]["compile_graphs_after"] = 5
    with pytest.raises(ValueError):analyze(rows,items)


def test_frozen_public_registry_cannot_include_final_questions():
    root = Path(__file__).resolve().parents[1]
    tasks,items,_ = load_data(root)
    registry = json.loads((root/"data/workloads/asd-prefix-cache/public-prefixes.json").read_text())
    assert registry["visibility"] == "public" and len(registry["prefixes"]) == 5
    from transformers import AutoTokenizer
    if not (root/"models/qwen3-8b/tokenizer.json").exists():pytest.skip("Prepared local tokenizer absent")
    tokenizer = AutoTokenizer.from_pretrained(root/"models/qwen3-8b",local_files_only=True,trust_remote_code=False)
    from scripts.prepare_public_prefixes import build
    from memorydraft.runtime.config import RuntimeConfig
    assert build(items,tokenizer,RuntimeConfig.load(root/"configs/runtime-dflash-asd.json")) == registry
    for p in registry["prefixes"]:
        text = tokenizer.decode(p["token_ids"])
        assert all(task["question"] not in text for task in tasks)
        assert all(not r.startswith("gsm8k:") for r in p["demonstration_ids"])
