"""Freeze new tasks and a conservative ASD control using only published development states."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from memorydraft.backends.asd import ASDConfig
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from scripts.run_asd_budget_probe import POLICIES

ROOT = Path(__file__).resolve().parents[1]
GATES = (.01, .02, .04, .0625, .1, .125, .1875, .25)
IDS = [*(f"gsm8k:{i}" for i in range(24, 32)), *(f"math500:{i}" for i in range(8, 24))]


def scalar_decision(event, config, spent, stop_length=None):
    """Independent scalar oracle on ONE frozen state; this is not a rollout."""
    mismatch, regrets = event["mismatch"], event["regrets"]
    k = len(mismatch)
    values = [k-i if config.value == "remaining" else
              next((j-i for j in range(i+1,k) if mismatch[j]), k-i) for i in range(k)]
    strict = next((i for i,m in enumerate(mismatch) if m), k)
    cost, exceptions, ceiling = 0., 0, 0
    candidates = [(0., 0., 0)]
    stop = k+1 if stop_length is None else stop_length
    baseline = min(strict, stop)
    for i,(m,r,v) in enumerate(zip(mismatch,regrets,values)):
        if m:
            if (config.budget == 0 or config.gate == 0 or config.cap == 0 or
                    r/v > config.gate or exceptions+1 > config.cap or cost+r > max(0.,config.budget-spent)):
                break
            cost += r
            exceptions += 1
        ceiling = i+1
        candidates.append((min(ceiling+1,stop)-min(baseline+1,stop)-config.regret_price*cost, -cost, ceiling))
        if ceiling == stop:
            break
    if config.selection == "utility":
        a = max(c for c in candidates if c[2] >= baseline)[2]
    else:
        a = ceiling
    return dict(accepted=a, strict_accepted=strict, relaxed=sum(mismatch[:a]),
                regret=sum(r for m,r in zip(mismatch[:a],regrets[:a]) if m), feasible_accepted=ceiling)


def calibrate():
    paths = [ROOT/f"results/asd-budget/{b}/mechanism/rows.jsonl.gz" for b in ("gsm8k","math500")]
    states = []
    for path in paths:
        for row in read_rows(path.with_suffix("")):
            if row["method"] == "ASD-Run":
                # Exclude the terminal block so no unseen proposal EOS can affect calibration.
                states.extend((event, event["spent"]-event["regret"]) for event in row["events"][:-1])
    run = [scalar_decision(e,POLICIES["ASD-Run"],s) for e,s in states]
    target_cost = sum(d["regret"] for d in run)
    target_count = sum(d["relaxed"] for d in run)
    candidates = []
    for gate in GATES:
        decisions = [scalar_decision(e,replace(POLICIES["ASD"],gate=gate),s) for e,s in states]
        cost, count = sum(d["regret"] for d in decisions), sum(d["relaxed"] for d in decisions)
        candidates.append(dict(gate=gate,regret=cost,exceptions=count,
                               cost_distance=abs(cost-target_cost),exception_distance=abs(count-target_count)))
    best = min(candidates,key=lambda c:(c["cost_distance"],c["exception_distance"],c["gate"]))
    return dict(schema=1,development_files={p.relative_to(ROOT).as_posix():file_hash(p) for p in paths},
                development="Published GSM8K 8..23 and MATH-500 0..7 ASD-Run states, excluding terminal blocks",
                objective="Match aggregate charged regret on identical states and identical available ledger; tie by exception distance then smaller gate",
                limitation="Borrowed Run ledgers and states; shadow decisions are not an on-policy control rollout or speed estimate",
                states=len(states),target_regret=target_cost,target_exceptions=target_count,
                candidates=candidates,selected_gate=best["gate"])


def load_tasks(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if data["schema"] != 1 or [r["id"] for r in data["records"]] != IDS:
        raise ValueError("Changed frozen ASD attribution tasks")
    return data["records"]


def load_policies(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    policies={m:ASDConfig(**c) for m,c in data["policies"].items()}
    expected={**POLICIES,"ASD-Conservative":replace(POLICIES["ASD"],gate=data["gate"])}
    if data["schema"] != 1 or policies != expected or data["gate"] not in GATES:
        raise ValueError("Changed frozen ASD attribution controls")
    return policies


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm-source",type=Path,required=True)
    parser.add_argument("--math-source",type=Path,required=True)
    args=parser.parse_args()
    old=json.loads((ROOT/"data/research/asd-budget/records.json").read_text(encoding="utf-8"))
    if file_hash(args.gsm_source) != "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14" or file_hash(args.math_source) != old["sources"]["math500"]["source_sha256"]:
        raise ValueError("Public source bytes do not match pinned versions")
    gsm=[json.loads(s) for s in args.gsm_source.read_text(encoding="utf-8").splitlines()]
    math=[json.loads(s) for s in args.math_source.read_text(encoding="utf-8").splitlines()]
    records=[dict(id=f"gsm8k:{i}",benchmark="gsm8k",**gsm[i]) for i in range(24,32)]
    records.extend(dict(id=f"math500:{i}",benchmark="math500",**math[i]) for i in range(8,24))
    old["sources"]["gsm8k"]["source_sha256"]=file_hash(args.gsm_source)
    write(ROOT/"data/research/asd-attribution/records.json",dict(schema=1,sources=old["sources"],records=records,
          selection="Unseen source-order GSM8K test 24..31 and MATH-500 8..23; no filtering by answers, length or outcomes"))
    calibration=calibrate()
    write(ROOT/"data/research/asd-attribution/calibration.json",calibration)
    policies={**POLICIES,"ASD-Conservative":replace(POLICIES["ASD"],gate=calibration["selected_gate"])}
    write(ROOT/"configs/asd-attribution-policies.json",dict(schema=1,gate=calibration["selected_gate"],
          policies={m:asdict(c) for m,c in policies.items()}))
    print(json.dumps(calibration,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
