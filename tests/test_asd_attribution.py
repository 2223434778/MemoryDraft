from dataclasses import asdict
import json
import random

import pytest

from memorydraft.backends.asd import ASDConfig, ASDVerifier
from memorydraft.evaluation.asd_budget import request_for, score
from scripts.prepare_asd_attribution import IDS, ROOT, calibrate, load_policies, load_tasks, scalar_decision
from scripts.run_asd_budget_probe import BudgetHook, POLICIES
from test_asd import scores


def test_new_tasks_are_disjoint_source_order_and_references_pass():
    records=load_tasks(ROOT/"data/research/asd-attribution/records.json")
    assert [r["id"] for r in records]==IDS
    old=json.loads((ROOT/"data/research/asd-budget/records.json").read_text(encoding="utf-8"))
    assert not {r["id"] for r in records}&{r["id"] for r in old["records"]}
    for r in records:
        assert score(r,r.get("solution",r["answer"]))["correct"]
        assert request_for(r).messages[0]["content"].endswith(r.get("question",r.get("problem")))


def test_control_is_frozen_from_old_states_only():
    a=calibrate()
    frozen=json.loads((ROOT/"data/research/asd-attribution/calibration.json").read_text(encoding="utf-8"))
    assert a==frozen and a["selected_gate"]==.0625
    policies=load_policies(ROOT/"configs/asd-attribution-policies.json")
    assert {m:c for m,c in policies.items() if m!="ASD-Conservative"}==POLICIES
    assert asdict(policies["ASD-Conservative"])=={**asdict(POLICIES["ASD"]),"gate":.0625}


def test_scalar_oracle_matches_tensor_on_same_state_including_eos():
    rng=random.Random(879)
    for _ in range(100):
        gaps=[rng.choice([-1,.1,.5,1.,3.,10.]) for _ in range(rng.randrange(2,16))]
        tokens,logits=scores(gaps)
        config=ASDConfig(budget=8,gate=rng.choice([.25,.0625]),value=rng.choice(["remaining","exact-run"]),
                         selection=rng.choice(["longest","utility"]))
        policy=ASDVerifier(config,record=True)
        policy.spent=rng.choice([0.,3.,7.5])
        spent=policy.spent
        eos=rng.choice([False,True])
        decision=policy.decide(tokens,logits,stop_token_ids=[1] if eos else [])
        expected=scalar_decision(policy.events[0],config,spent,1 if eos else None)
        assert asdict(decision)=={k:v for k,v in expected.items() if k!="feasible_accepted"}


def test_hook_uses_control_configuration_and_fresh_ledger():
    from types import SimpleNamespace
    policies=load_policies(ROOT/"configs/asd-attribution-policies.json")
    def generate(**kwargs):
        policy=kwargs["verifier"]
        assert policy.config.gate==.0625 and policy.spent==0
        policy.decide(*scores([.5,-1,-1]))
        return SimpleNamespace(rounds=[dict(generated_before=1)])
    hook=BudgetHook(generate,"ASD-Conservative",True,policies)
    hook()
    hook()


def test_changed_control_or_task_selection_rejected(tmp_path):
    path=tmp_path/"policies.json"
    data=json.loads((ROOT/"configs/asd-attribution-policies.json").read_text(encoding="utf-8"))
    data["policies"]["ASD-Conservative"]["value"]="exact-run"
    path.write_text(json.dumps(data),encoding="utf-8")
    with pytest.raises(ValueError):load_policies(path)
    data=json.loads((ROOT/"data/research/asd-attribution/records.json").read_text(encoding="utf-8"))
    data["records"].reverse()
    path.write_text(json.dumps(data),encoding="utf-8")
    with pytest.raises(ValueError):load_tasks(path)
