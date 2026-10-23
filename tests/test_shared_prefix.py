from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from memorydraft.runtime.prefix_cache import PrefixCacheSpec
from scripts.prepare_shared_prefix import PARENT_SHA, build
from scripts.run_asd_prefix_cache import http_checks
from test_prefix_cache import cache, registry
from test_runtime import engine


def data():
    return dict(schema=1,visibility="public",prefixes=[
        dict(id="short",token_ids=[3],levels=[512],source="fixture"),
        dict(id="a",token_ids=[3,4,5],levels=[4096],source="fixture"),
        dict(id="b",token_ids=[3,4,6],levels=[4096],source="fixture")])


def test_common_prefix_is_exact_and_other_entries_and_identity_are_preserved():
    value = data()
    before = deepcopy(value)
    result = build(value,"a"*64)
    assert value == before
    assert result["prefixes"][0] == before["prefixes"][0]
    common = result["prefixes"][-1]
    assert common["token_ids"] == [3,4] and common["parent_prefix_ids"] == ["a","b"]
    assert common["parent_registry_sha256"] == "a"*64
    assert result["derivation"]["parent_registry_sha256"] == "a"*64


@pytest.mark.parametrize("bad",["private","no_shared","one","invalid_token"])
def test_invalid_coalescing_is_rejected(bad):
    value = data()
    if bad == "private":value["visibility"] = "private"
    elif bad == "no_shared":value["prefixes"][-1]["token_ids"] = [7]
    elif bad == "one":value["prefixes"].pop()
    elif bad == "invalid_token":value["prefixes"][-1]["token_ids"] = [True]
    with pytest.raises(ValueError):build(value,"a"*64)


@torch.inference_mode()
def test_alternating_suffixes_reuse_shared_state_with_same_byte_budget(engine,tmp_path):
    spec = registry(engine,tmp_path,prefixes=((3,4,5),(3,4,6)))
    value = json.loads(Path(spec.path).read_bytes())
    for p in value["prefixes"]:p["levels"] = [4096]
    raw = json.dumps(value).encode()
    Path(spec.path).write_bytes(raw)
    spec = replace(spec,sha256=hashlib.sha256(raw).hexdigest())
    def run(obj,tokens):
        ids = torch.tensor([tokens])
        return obj.prefill(engine.target,ids,torch.arange(len(tokens)).unsqueeze(0),[0,1])
    probe = cache(engine,spec)
    run(probe,[3,4,5,7])
    budget = probe.bytes
    whole = cache(engine,replace(spec,max_bytes=budget))
    expected = []
    for tokens in ([3,4,5,7],[3,4,6,8],[3,4,5,9]):
        expected.append(run(whole,tokens))
        assert whole.last["prefix_cache_hit"] == 0
    assert whole.last["prefix_cache_evictions"] == 1
    shared_path = tmp_path/"shared.json"
    raw = json.dumps(build(value,spec.sha256)).encode()
    shared_path.write_bytes(raw)
    shared = cache(engine,PrefixCacheSpec(str(shared_path),hashlib.sha256(raw).hexdigest(),max_bytes=budget))
    for i,tokens in enumerate(([3,4,5,7],[3,4,6,8],[3,4,5,9])):
        logits,features,kv = run(shared,tokens)
        torch.testing.assert_close(logits,expected[i][0],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(features,expected[i][1],atol=1e-6,rtol=1e-5)
        assert torch.equal(logits.argmax(-1),expected[i][0].argmax(-1))
        assert kv.get_seq_length() == features.shape[1] == 4
        assert shared.last["prefix_cache_hit"] == (i > 0)
        assert shared.last["prefix_cache_evictions"] == 0
    assert shared.bytes < budget and list(shared.entries) == [(3,4)]
    assert next(iter(shared.entries.values())).features.shape[1] == 2


def test_real_registry_is_reproducible_and_leaves_all_requests_unchanged():
    root = Path(__file__).resolve().parents[1]
    parent = (root/"data/workloads/asd-prefix-cache/public-prefixes.json").read_bytes()
    assert hashlib.sha256(parent).hexdigest() == PARENT_SHA
    value = build(json.loads(parent),PARENT_SHA)
    actual = (root/"data/workloads/asd-shared-prefix/public-prefixes.json").read_bytes()
    assert (json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode() == actual
    assert [len(p["token_ids"]) for p in value["prefixes"]] == [459,1867,323,3934]
    assert len(value["prefixes"][-1]["token_ids"]) == 3934
    from memorydraft.evaluation.long_context import load_data
    from memorydraft.runtime.config import RuntimeConfig
    _,items,_ = load_data(root)
    config = RuntimeConfig.load(root/"configs/runtime-dflash-asd-shared-prefix.json")
    assert hashlib.sha256(actual).hexdigest() == config.prefix_cache["sha256"]
    assert config.prefix_cache["max_bytes"] == 2*2**30 and len(items) == 24


@pytest.mark.parametrize("states",[(),("cache","cache"),("other",)])
def test_http_load_modes_fail_before_starting_any_service(states):
    with pytest.raises(ValueError,match="states"):
        http_checks(None,[],{},None,load_states=states)
