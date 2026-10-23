from dataclasses import asdict, replace
import hashlib
import json

import pytest
import torch

from memorydraft.runtime.config import RuntimeConfig
from memorydraft.runtime.contracts import RequestCancelled
from memorydraft.runtime.engine import InferenceEngine
from memorydraft.runtime.prefix_cache import PrefixCacheSpec, PublicPrefixCache, tokenizer_sha256
from test_runtime import engine, request


def registry(engine, tmp_path, prefixes=((3,4),(3,5)), **changes):
    value = dict(schema=1,visibility="public",target_revision=engine.config.target_revision,
        tokenizer_sha256=tokenizer_sha256(engine.tokenizer),template_options=engine.config.template_options,
        prefixes=[dict(id=str(i),source="public test fixture",token_ids=list(p)) for i,p in enumerate(prefixes)])
    value.update(changes)
    path = tmp_path/"public.json"
    path.write_bytes(json.dumps(value).encode())
    return PrefixCacheSpec(str(path),hashlib.sha256(path.read_bytes()).hexdigest())


def cache(engine, spec):
    return PublicPrefixCache(spec,target=engine.target,tokenizer=engine.tokenizer,
            target_revision=engine.config.target_revision,template_options=engine.config.template_options,
            layers=engine.draft.target_layer_ids,max_input_tokens=engine.config.max_input_tokens)


@pytest.mark.parametrize("method",["DFlash","ASD"])
def test_cold_and_hit_match_uncached_real_model_and_stream(engine,tmp_path,method):
    spec = registry(engine,tmp_path)
    config = replace(engine.config,backend="dflash",draft="fixture",asd=dict(budget=8,gate=.25,cap=2))
    plain = InferenceEngine(engine.target,engine.tokenizer,config,engine.draft)
    expected = plain.generate(request(),method)
    cached = InferenceEngine(engine.target,engine.tokenizer,replace(config,prefix_cache=asdict(spec)),engine.draft)
    cold = cached.generate(request(),method)
    hit = cached.generate(request(user_id="another",id="other"),method)
    assert expected.token_ids == cold.token_ids == hit.token_ids
    assert expected.prompt_sha256 == cold.prompt_sha256 == hit.prompt_sha256
    assert expected.counters["rounds"] == cold.counters["rounds"] == hit.counters["rounds"]
    assert cold.counters["prefix_cache_hit"] == 0
    assert cold.counters["prefix_cache_filled_bytes"] > 0
    assert hit.counters["prefix_cache_hit"] == 1
    assert hit.counters["prefix_cache_reused_tokens"] == 2
    assert hit.snapshot_id != cold.snapshot_id
    commits = []
    streamed = cached.generate(request(),method,on_tokens=commits.extend)
    assert commits == streamed.token_ids == expected.token_ids
    cached.close()
    assert cached._prefix_cache.bytes == 0


@torch.inference_mode()
def test_prefill_features_positions_and_no_shared_mutable_storage(engine,tmp_path):
    obj = cache(engine,registry(engine,tmp_path))
    ids = torch.tensor([[3,4,5,6,7]])
    positions = torch.arange(5).unsqueeze(0)
    logits,features,first = obj.prefill(engine.target,ids,positions,[0,1])
    stored = next(iter(obj.entries.values()))
    before = [t.clone() for pair in stored.kv for t in pair]
    logits2,features2,second = obj.prefill(engine.target,ids,positions,[0,1])
    torch.testing.assert_close(logits,logits2,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(features,features2,rtol=1e-5,atol=1e-6)
    assert first.get_seq_length() == second.get_seq_length() == 5
    assert stored.features.shape[1] == 2
    assert all(k.shape[-2] == v.shape[-2] == 2 for k,v in stored.kv)
    for k,v in second.to_legacy_cache():k.zero_();v.zero_()
    features2.zero_()
    second.crop(1)
    assert all(torch.equal(t,b) for t,b in zip([t for pair in stored.kv for t in pair],before))
    assert torch.equal(stored.features,features[:,:2])


def test_private_nonmatching_or_exact_length_prefix_does_not_fill(engine,tmp_path):
    obj = cache(engine,registry(engine,tmp_path))
    for tokens in ([3,4],[7,8,9]):
        ids = torch.tensor([tokens])
        obj.prefill(engine.target,ids,torch.arange(len(tokens)).unsqueeze(0),[0,1])
        assert obj.last["prefix_cache_prefix_tokens"] == 0
        assert not obj.entries and obj.bytes == 0


def test_lru_and_byte_limits_and_cancel_preserve_public_entry(engine,tmp_path):
    spec = replace(registry(engine,tmp_path),max_entries=1)
    obj = cache(engine,spec)
    def run(tokens):
        ids = torch.tensor([tokens])
        return obj.prefill(engine.target,ids,torch.arange(len(tokens)).unsqueeze(0),[0,1])
    run([3,4,6])
    size = obj.bytes
    run([3,5,6])
    assert obj.last["prefix_cache_evictions"] == 1
    assert list(obj.entries) == [(3,5)] and obj.bytes == size
    with pytest.raises(RequestCancelled):
        obj.prefill(engine.target,torch.tensor([[3,5,6]]),torch.arange(3).unsqueeze(0),[0,1],lambda:True)
    assert obj.bytes == size
    small = cache(engine,replace(spec,max_bytes=size-1))
    small.prefill(engine.target,torch.tensor([[3,4,6]]),torch.arange(3).unsqueeze(0),[0,1])
    assert small.bytes == 0 and not small.entries


def test_cancel_after_hit_and_configuration_change_release_engine(engine,tmp_path):
    spec = registry(engine,tmp_path)
    config = replace(engine.config,backend="dflash",draft="fixture",prefix_cache=asdict(spec))
    obj = InferenceEngine(engine.target,engine.tokenizer,config,engine.draft)
    expected = obj.generate(request(),"DFlash")
    seen = []
    with pytest.raises(RequestCancelled):
        obj.generate(request(),"DFlash",on_tokens=seen.extend,should_stop=lambda:bool(seen))
    assert not obj.lock.locked()
    assert obj.generate(request(),"DFlash").token_ids == expected.token_ids
    obj.config = replace(config,prefix_cache={})
    with pytest.raises(ValueError,match="Reload"):
        obj.generate(request(),"DFlash")
    assert not obj.lock.locked()
    obj.close()


@pytest.mark.parametrize("changes",[dict(visibility="private"),dict(target_revision="wrong"),
    dict(tokenizer_sha256="0"*64),dict(template_options={}),dict(prefixes=[])])
def test_registry_identity_and_allowlist_fail_closed(engine,tmp_path,changes):
    with pytest.raises(ValueError):cache(engine,registry(engine,tmp_path,**changes))


def test_registry_hash_and_binding_fail_closed(engine,tmp_path):
    spec = registry(engine,tmp_path)
    with pytest.raises(ValueError,match="checksum"):cache(engine,replace(spec,sha256="0"*64))
    obj = cache(engine,spec)
    with pytest.raises(ValueError,match="bound"):
        obj.prefill(engine.target,torch.tensor([[3,4,5]]),torch.arange(3).unsqueeze(0),[1])


def test_prefix_config_paths_limits_and_default(tmp_path):
    assert RuntimeConfig(target="test").prefix_cache == {}
    path = tmp_path/"config.json"
    path.write_text(json.dumps(dict(backend="dflash",target="target",draft="draft",
        prefix_cache=dict(path="public.json",sha256="a"*64))))
    assert RuntimeConfig.load(path).prefix_cache["path"] == str(tmp_path/"public.json")
    for change in (dict(max_bytes=True),dict(max_entries=0),dict(sha256="invalid")):
        value = dict(path="x",sha256="a"*64)
        value.update(change)
        with pytest.raises(ValueError):PrefixCacheSpec(**value)
    with pytest.raises(ValueError,match="dflash"):
        RuntimeConfig(target="x",prefix_cache=dict(path="public.json",sha256="a"*64))


def test_registry_validation_does_not_query_expensive_vocab_size_per_token(engine,tmp_path):
    spec = registry(engine,tmp_path,prefixes=(tuple([3]*1000),))
    class Tokenizer:
        calls = 0
        backend_tokenizer = engine.tokenizer.backend_tokenizer
        chat_template = engine.tokenizer.chat_template
        def __len__(self):
            self.calls += 1
            if self.calls > 2:raise AssertionError("Per-token vocabulary-size lookup")
            return len(engine.tokenizer)
    tokenizer = Tokenizer()
    obj = PublicPrefixCache(spec,target=engine.target,tokenizer=tokenizer,target_revision=engine.config.target_revision,
            template_options=engine.config.template_options,layers=[0,1],max_input_tokens=8192)
    assert len(obj.prefixes[0][0]) == 1000 and tokenizer.calls <= 2
