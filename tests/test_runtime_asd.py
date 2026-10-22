from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json

from fastapi.testclient import TestClient
import pytest
from safetensors.torch import save_file
import torch

from memorydraft.evaluation.adaptation import AdapterBank, RANK
from memorydraft.runtime.benchmark import benchmark, check_regression, compare
from memorydraft.runtime.config import RuntimeConfig
from memorydraft.runtime.contracts import RequestCancelled
from memorydraft.runtime.demo import DemoEngine
from memorydraft.runtime.draft_adapter import DraftAdapterSpec, merge_draft_adapter
from memorydraft.runtime.engine import InferenceEngine
from memorydraft.serving.app import create_app
from scripts.prepare_asd_runtime_adapter import prepare
from test_adaptation import ToyDraft
from test_runtime import engine, request


@pytest.fixture
def asd_engine(engine):
    config = replace(engine.config, backend="dflash", draft="fixture", asd=dict(budget=8,gate=.25,cap=2))
    engine.close()
    result = InferenceEngine(engine.target,engine.tokenizer,config,engine.draft)
    try:
        yield result
    finally:
        result.close()


def adapter(tmp_path, draft, *, extra=False):
    torch.manual_seed(51)
    weights = AdapterBank(draft).original
    state = {"base_model.model."+n+suffix:torch.randn(shape)*.01 for n,w in weights.items()
             for suffix,shape in ((".lora_A.weight",(RANK,w.shape[1])),(".lora_B.weight",(w.shape[0],RANK)))}
    if extra:
        state["unexpected"] = torch.zeros(1)
    path = tmp_path/"adapter.safetensors"
    save_file(state,str(path))
    return DraftAdapterSpec(str(path),hashlib.sha256(path.read_bytes()).hexdigest()),state


def test_asd_is_opt_in_with_fixed_runtime_policy():
    assert not RuntimeConfig(target="fixture").asd
    for backend in ("demo","transformers"):
        with pytest.raises(ValueError,match="dflash"):
            RuntimeConfig(backend=backend,target="fixture",asd=dict(budget=8))
    for changes in (dict(budget=True),dict(budget=float("nan")),dict(gate=-1),dict(cap=1.5),dict(value="exact-run")):
        with pytest.raises(ValueError):
            RuntimeConfig(backend="dflash",target="fixture",draft="fixture",asd=changes)


@pytest.mark.parametrize("value",[[],None,dict(path="file.safetensors"),dict(path="file.safetensors",sha256="wrong")])
def test_invalid_adapter_configuration_fails_explicitly(value):
    with pytest.raises(ValueError):
        RuntimeConfig(backend="dflash",target="fixture",draft="fixture",draft_adapter=value)


def test_adapter_paths_resolve_relative_to_config_file(tmp_path):
    path = tmp_path/"config.json"
    path.write_text(json.dumps(dict(backend="dflash",target="target",draft="draft",asd=dict(budget=8),
                        draft_adapter=dict(path="weights/model.safetensors",sha256="a"*64))))
    result = RuntimeConfig.load(path)
    assert result.draft_adapter["path"] == str(tmp_path/"weights/model.safetensors")
    assert RuntimeConfig(**asdict(result)) == result


def test_checked_startup_merge_matches_frozen_bank_and_rejects_second_merge(tmp_path):
    draft = ToyDraft()
    expected = deepcopy(draft)
    spec,state = adapter(tmp_path,draft)
    AdapterBank(expected).select(state)
    info = merge_draft_adapter(draft,spec)
    assert info["sha256"] == spec.sha256 and info["application"] == "startup_dense_merge"
    assert all(torch.equal(p,expected.state_dict()[n]) for n,p in draft.state_dict().items())
    with pytest.raises(ValueError,match="fresh base"):
        merge_draft_adapter(draft,spec)


@pytest.mark.parametrize("bad",["hash","extra_tensor","dtype"])
def test_failed_adapter_validation_does_not_mutate_draft(tmp_path,bad):
    draft = ToyDraft().double() if bad == "dtype" else ToyDraft()
    before = {n:p.clone() for n,p in draft.state_dict().items()}
    spec,_ = adapter(tmp_path,draft,extra=bad == "extra_tensor")
    if bad == "hash":
        spec = replace(spec,sha256="0"*64)
    with pytest.raises(ValueError):
        merge_draft_adapter(draft,spec)
    assert all(torch.equal(p,draft.state_dict()[n]) for n,p in before.items())


def test_adapter_preparation_is_local_checked_and_idempotent(tmp_path):
    spec,_ = adapter(tmp_path,ToyDraft())
    config = tmp_path/"runtime.json"
    destination = tmp_path/"prepared.safetensors"
    config.write_text(json.dumps(dict(backend="dflash",target="fixture",draft="fixture",
                       draft_adapter=dict(path=str(destination),sha256=spec.sha256))))
    assert prepare(tmp_path/"adapter.safetensors",config) == destination
    assert prepare(tmp_path/"adapter.safetensors",config) == destination
    assert destination.read_bytes() == (tmp_path/"adapter.safetensors").read_bytes()


def test_request_budget_is_fresh_stream_is_exact_and_cancel_recovers(asd_engine,monkeypatch):
    from memorydraft.runtime import engine as module
    native,seen = module.generate_reuse,[]
    def inspect(*args,**kwargs):
        policy = kwargs["verifier"]
        assert policy.spent == policy.relaxed == policy.blocks == 0
        assert kwargs["incremental_stop"]
        seen.append(policy)
        return native(*args,**kwargs)
    monkeypatch.setattr(module,"generate_reuse",inspect)
    tokens = []
    baseline = asd_engine.generate(request(),"ASD",on_tokens=tokens.extend)
    assert tokens == baseline.token_ids
    assert baseline.approximate and 0 <= baseline.asd["spent"] <= 8
    assert baseline.asd["gate"] == .25 and baseline.asd["cap"] == 2
    tokens.clear()
    with pytest.raises(RequestCancelled):
        asd_engine.generate(request(),"ASD",on_tokens=tokens.extend,should_stop=lambda:len(tokens)>3)
    recovered = asd_engine.generate(request(),"ASD")
    assert recovered.token_ids == baseline.token_ids and recovered.asd == baseline.asd
    assert len(seen) == len({id(p) for p in seen}) == 3
    assert not asd_engine.lock.locked()


def test_asd_does_not_change_strict_methods_or_comparison_contract(asd_engine):
    ar = asd_engine.generate(request(),"AR")
    strict = asd_engine.generate(request(),"DFlash")
    assert compare(strict,ar) and not strict.approximate and strict.asd is None
    approximate = asd_engine.generate(request(),"ASD")
    assert not compare(approximate,ar)


@pytest.mark.parametrize("field",["asd","draft_adapter"])
def test_policy_or_adapter_mutation_requires_reload(asd_engine,field):
    value = dict(budget=4,gate=.25,cap=2) if field == "asd" else dict(path="fixture.safetensors",sha256="a"*64)
    asd_engine.config = replace(asd_engine.config,**{field:value})
    with pytest.raises(ValueError,match="Reload"):
        asd_engine.generate(request(),"ASD")
    assert not asd_engine.lock.locked()


def test_engine_merges_adapter_once_and_tags_draft_results(engine,tmp_path):
    engine.close()
    engine.draft.q_proj = torch.nn.Linear(32,32,bias=False)
    engine.draft.v_proj = torch.nn.Linear(32,32,bias=False)
    spec,_ = adapter(tmp_path,engine.draft)
    config = replace(engine.config,backend="dflash",draft="fixture",asd=dict(budget=8),draft_adapter=asdict(spec))
    loaded = InferenceEngine(engine.target,engine.tokenizer,config,engine.draft)
    try:
        before = {n:p.clone() for n,p in loaded.draft.state_dict().items()}
        for method in ("ASD","DFlash","ASD"):
            result = loaded.generate(request(),method)
            assert result.draft_adapter_sha256 == spec.sha256
        assert loaded.generate(request(),"AR").draft_adapter_sha256 is None
        assert all(torch.equal(p,loaded.draft.state_dict()[n]) for n,p in before.items())
        assert loaded.describe()["draft_adapter"]["sha256"] == spec.sha256
    finally:
        loaded.close()


def test_http_json_sse_exposes_approximate_metadata_and_stream_contract():
    class TransportFixture(DemoEngine):
        methods = ("AR","ASD")
        def describe(self):
            return {**super().describe(),"approximate_methods":["ASD"]}
        def generate(self,req,method,**kwargs):
            row = super().generate(req,"AR",**kwargs)
            row.method, row.approximate = method, method == "ASD"
            row.asd = dict(budget=8,spent=1,gate=.25,cap=2) if row.approximate else None
            return row
    req = request()
    body = {**asdict(req),"messages":list(req.messages),"method":"ASD"}
    with TestClient(create_app(TransportFixture)) as client:
        assert client.get("/healthz").json()["approximate_methods"] == ["ASD"]
        first = client.post("/v1/generate",json=body)
        assert first.status_code == 200
        streamed = client.post("/v1/generate",json={**body,"stream":True})
        assert streamed.status_code == 200
        events = [(lines[0][7:],json.loads(lines[1][6:])) for b in streamed.text.split("\n\n") if b
                  for lines in [b.splitlines()]]
        assert events[0][1]["approximate"] is True
        complete = events[-1][1]
        assert complete["approximate"] and complete["asd"] == first.json()["asd"]
        assert [t for name,data in events if name == "tokens" for t in data["token_ids"]] == complete["token_ids"] == first.json()["token_ids"]


def test_strict_benchmark_rejects_asd_before_creating_artifacts(asd_engine,tmp_path):
    out = tmp_path/"benchmark"
    with pytest.raises(ValueError,match="approximate"):
        benchmark(asd_engine,[request()],["ASD"],out)
    assert not out.exists()


def test_regression_rejects_different_adapter_identities(tmp_path):
    class FixtureEngine(DemoEngine):
        def __init__(self,sha):self.sha = sha
        def describe(self):
            return dict(synthetic=False,backend="fixture",torch="fixture",transformers="fixture",gpu="fixture",
                        device="fixture",dtype="fixture",target_revision="fixture",draft_adapter=dict(sha256=self.sha))
    for name,sha in (("first","a"*64),("second","b"*64)):
        benchmark(FixtureEngine(sha),[request()],["AR"],tmp_path/name,repeats=1,warmup=False)
    with pytest.raises(ValueError,match="draft_adapter"):
        check_regression(tmp_path/"first",tmp_path/"second")
