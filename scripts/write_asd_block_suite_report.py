"""Question-paired, per-family analysis of the frozen full longer-block suite."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_suite import BENCHMARKS
from memorydraft.evaluation.runtime_migration import read_rows
from scripts import write_asd_block_report as preflight_report
from scripts.diagnose_transport_online import write
from scripts.prepare_asd_attribution import scalar_decision
from scripts.run_asd_block_probe import ARMS, selected_records
from scripts.run_asd_block_suite import ALL_ARMS, REPEATS, previous_rows, records_and_positions
from scripts.run_asd_budget_probe import POLICIES
from scripts.write_asd_suite_report import assessed, paired

ROOT=Path(__file__).resolve().parents[1]
PAIRS=tuple(((m,a),(m,b)) for m in POLICIES for a,b in ((24,16),(32,16),(32,24)))+tuple(
    ((m,size),(ref,16 if ref=="AR" else size)) for size in (16,24,32)
    for m,ref in (("DFlash","AR"),("ASD","DFlash"),("ASD-Run","ASD"),("ASD-Utility","ASD"),
                  ("ASD-Run","DFlash"),("ASD-Utility","DFlash")))+tuple(
    (("ASD-Utility",size),("ASD",16)) for size in (24,32))


def label(arm):
    return arm[0] if arm[0]=="AR" else arm[0]+str(arm[1])


def audit_events(row,stops):
    events=row["events"];rounds=row["rounds"];config=POLICIES[row["method"]]
    if len(events)!=len(rounds):raise ValueError("Missing full mechanism events")
    before=0.;relaxed=0;shortened=0;foregone=0
    for e,rd in zip(events,rounds):
        proposal=e["proposal_token_ids"];top=e["target_top_token_ids"];k=len(proposal);a=e["accepted"]
        if not (k==len(top)==len(e["mismatch"])==len(e["regrets"])==len(e["values"])==rd["proposed"] and 0<=a<=k):
            raise ValueError("Invalid full mechanism dimensions")
        if e["mismatch"]!=[x!=y for x,y in zip(proposal,top)]:raise ValueError("Changed proposal/top identities")
        values=[k-i if config.value=="remaining" else next((j-i for j in range(i+1,k) if e["mismatch"][j]),k-i) for i in range(k)]
        if e["values"]!=values or any(not math.isfinite(x) or x<0 for x in e["regrets"]):
            raise ValueError("Invalid full mechanism values or regrets")
        stop=next((i+1 for i,t in enumerate(proposal) if t in stops),k+1)
        d=scalar_decision(e,config,before,stop_length=stop)
        if any(d[f]!=e[f] for f in ("accepted","strict_accepted","relaxed","feasible_accepted")) or abs(d["regret"]-e["regret"])>1e-8:
            raise ValueError("Full scalar decision mismatch")
        if (rd["accepted"]!=a or rd["strict_accepted"]!=e["strict_accepted"] or rd["relaxed_tokens"]!=e["relaxed"]
                or rd["regret"]!=e["regret"]):raise ValueError("Changed mechanism round")
        committed=row["token_ids"][rd["generated_before"]:rd["generated_before"]+a]
        if committed!=proposal[:a]:raise ValueError("Changed committed proposal IDs")
        before+=e["regret"];relaxed+=e["relaxed"]
        if not math.isfinite(e["spent"]) or abs(e["spent"]-before)>1e-8 or before>config.budget+1e-8:
            raise ValueError("Full mechanism ledger discontinuity")
        shortened+=a<e["feasible_accepted"];foregone+=e["feasible_accepted"]-a
    summary=row["asd"]
    if (summary["blocks"]!=len(events) or summary["relaxed_tokens"]!=relaxed or abs(summary["spent"]-before)>1e-8
            or summary["shortened_blocks"]!=shortened or summary["foregone_proposal_tokens"]!=foregone):
        raise ValueError("Changed full mechanism final summary")
    return len(events)


def audit(rows,mechanism,records,positions,previous,stops):
    expected={(r["id"],m,size,stage,rep) for r in records for m,size in ALL_ARMS
              for stage,reps in REPEATS.items() for rep in range(reps)}
    seen={}
    for r in rows:
        key=r["request_id"],r["method"],r["experimental_block_size"],r["stage"],r["repeat"]
        if key not in expected or key in seen:raise ValueError("Changed full block matrix identity")
        if not math.isfinite(r["latency_seconds"]) or r["latency_seconds"]<=0:raise ValueError("Invalid timed latency")
        seen[key]=r
    if set(seen)!=expected:raise ValueError("Incomplete full block matrix")
    # Reuse the preflight audit for identities, grading, EOS, policy budget,
    # native reproduction and strict AR checks. Its matrix has fewer repeats.
    subset=[r for r in rows if r["repeat"]<(1 if r["stage"]=="natural" or r["method"]=="AR" else 2)]
    detail=preflight_report.audit(subset,records,previous,stops)
    for r in rows:
        first=seen[r["request_id"],r["method"],r["experimental_block_size"],r["stage"],0]
        fields=("token_ids","asd","rounds","prompt_sha256","snapshot_id","finish_reason","max_new_tokens","training_block_size","benchmark")
        if any(r[k]!=first[k] for k in fields) or r.get("quality")!=first.get("quality"):
            raise ValueError("Changed full block repeat or quality")
    expected_mechanism={(r["id"],m,size) for r in records if positions[r["id"]]<2 for m,size in ARMS}
    observed=set();blocks=0
    for r in mechanism:
        key=r["request_id"],r["method"],r["experimental_block_size"]
        if key not in expected_mechanism or key in observed:raise ValueError("Changed full mechanism coverage")
        observed.add(key);natural=seen[*key,"natural",0]
        for k in ("token_ids","asd","quality","prompt_sha256","snapshot_id","finish_reason","training_block_size","benchmark"):
            if r[k]!=natural[k]:raise ValueError("Mechanism changed full timed trajectory")
        if "latency_seconds" in r:raise ValueError("Instrumented time leaked into timed rows")
        blocks+=audit_events(r,stops)
    if observed!=expected_mechanism:raise ValueError("Incomplete full mechanism coverage")
    detail.update(rows=len(rows),native_b16_cases=sum(r["experimental_block_size"]==16 for r in rows),
                  mechanism_rows=len(mechanism),independent_mechanism_blocks=blocks)
    return detail


def aggregate(rows):
    entries=preflight_report.merge(rows)
    for r in rows:
        if r["repeat"]!=0:continue
        item=entries[r["request_id"],r["method"],r["experimental_block_size"],r["stage"]]
        seconds=item["repeat_seconds"];quartiles=statistics.quantiles(seconds,n=4,method="inclusive")
        item.update(q25=quartiles[0],q75=quartiles[2],iqr=quartiles[2]-quartiles[0],spread=max(seconds)/min(seconds)-1,
                    target_forwards=item["forwards"],complete128=item["tokens"]==128,
                    relaxed=r["asd"]["relaxed_tokens"] if r["asd"] else 0,
                    useful_accepted=sum(max(0,min(rd["accepted"],item["tokens"]-rd["generated_before"])) for rd in r["rounds"]),
                    verified_positions=sum(rd["proposed"]+1 for rd in r["rounds"]),
                    draft_noise_positions=r["experimental_block_size"]*sum(rd["draft_forward"] for rd in r["rounds"]))
    return entries


def estimate(values,seed,*,minimum=1):
    if not values:return None
    if len(values)<minimum:
        return dict(n=len(values),ratio=math.exp(statistics.mean(math.log(v) for v in values)),ci95=None,
                    slower=sum(v>1 for v in values),faster=sum(v<1 for v in values))
    return paired(values,seed)


def compare(entries,ids,arm,ref,seed):
    natural=[];prefix=[];complete=[];correct=[];length=[];density=[];cost=[];changes=[];unknown=0
    per_question=[]
    for q in ids:
        a,z=entries[q,*arm,"natural"],entries[q,*ref,"natural"]
        ratio=a["seconds"]/z["seconds"];natural.append(ratio)
        length.append(a["tokens"]/z["tokens"])
        density.append((a["forwards"]/a["tokens"])/(z["forwards"]/z["tokens"]))
        cost.append((a["seconds"]/a["forwards"])/(z["seconds"]/z["forwards"]))
        if assessed(a) and assessed(z):changes.append(int(a["quality"]["correct"])-int(z["quality"]["correct"]))
        else:unknown+=1
        if a["finish_reason"]==z["finish_reason"]=="eos":
            complete.append(ratio)
            if a["quality"]["correct"] and z["quality"]["correct"]:correct.append(ratio)
        p,r=entries[q,*arm,"prefix"],entries[q,*ref,"prefix"]
        fixed=p["seconds"]/r["seconds"] if p["complete128"] and r["complete128"] else None
        if fixed is not None:prefix.append(fixed)
        per_question.append(dict(request_id=q,natural_ratio=ratio,prefix_equal128_ratio=fixed))
    gm=lambda values:math.exp(statistics.mean(math.log(v) for v in values))
    return dict(natural=estimate(natural,seed+":natural",minimum=5),prefix_equal128=estimate(prefix,seed+":prefix",minimum=5),
                common_eos=estimate(complete,seed+":eos",minimum=5),common_correct_eos=estimate(correct,seed+":correct",minimum=5),
                length_ratio=gm(length),target_density_ratio=gm(density),cost_per_forward_ratio=gm(cost),
                quality_improved=sum(x>0 for x in changes),quality_lost=sum(x<0 for x in changes),quality_unassessed=unknown,
                per_question=per_question)


def summarize(entries,records):
    overlap={r["id"] for r in selected_records()};tables={};comparisons={}
    for b in BENCHMARKS:
        ids=[r["id"] for r in records if r["benchmark"]==b]
        if not ids:continue
        tables[b]={};comparisons[b]={}
        for arm in ALL_ARMS:
            items=[entries[q,*arm,"natural"] for q in ids]
            seconds=sum(r["seconds"] for r in items);tokens=sum(r["tokens"] for r in items)
            tables[b][label(arm)]=dict(tasks=len(items),seconds=seconds,tokens=tokens,tps=tokens/seconds,
                correct=sum(r["quality"]["correct"] for r in items),truncated=sum(r["finish_reason"]!="eos" for r in items),
                forwards=sum(r["forwards"] for r in items),regret=sum(r["regret"] for r in items),
                relaxed=sum(r["relaxed"] for r in items),useful_accepted=sum(r["useful_accepted"] for r in items),
                verified_positions=sum(r["verified_positions"] for r in items),draft_noise_positions=sum(r["draft_noise_positions"] for r in items),
                median_repeat_spread=statistics.median(r["spread"] for r in items),
                secondary_correct=sum(r["quality"].get("secondary_correct",r["quality"]["correct"]) for r in items),
                needs_review=[r["request_id"] for r in items if r["quality"].get("needs_review")],
                restricted_unassessed=[r["request_id"] for r in items if not assessed(r)],
                f1_mean=statistics.mean(r["quality"]["f1"] for r in items) if b=="squad" else None)
        for arm,ref in PAIRS:
            name=label(arm)+"/"+label(ref);seed="full-block:"+b+":"+name
            c=compare(entries,ids,arm,ref,seed)
            for group,test in (("preflight_overlap",lambda q:q in overlap),("other_development",lambda q:q not in overlap)):
                subset=[q for q in ids if test(q)]
                c[group]=compare(entries,subset,arm,ref,seed+":"+group) if subset else None
            comparisons[b][name]=c
    return dict(methods=tables,comparisons=comparisons)


def interval(value):
    if value is None:return "无共同样本"
    ci=value["ci95"]
    return (f"{value['ratio']:.4f} [{ci[0]:.4f},{ci[1]:.4f}]/{value['n']}" if ci else
            f"{value['ratio']:.4f}/{value['n']}（仅描述）")


def main():
    base=ROOT/"results/asd-block-suite"
    execution=json.loads((base/"execution-verification.json").read_text())
    if execution["status"]!="complete" or execution["files_verified_before"]!=execution["files_verified_after"]:
        raise ValueError("Incomplete full block execution")
    rows=[];mechanism=[];stops=None
    for i in (0,1):
        directory=base/f"part-{i}";p=json.loads((directory/"provenance.json").read_text())
        if p["status"]!="complete" or not p["original_unchanged"] or p["split"]!="development":raise ValueError("Changed full block worker")
        if stops is not None and stops!=p["stop_token_ids"]:raise ValueError("Changed full stop IDs")
        stops=p["stop_token_ids"];rows.extend(read_rows(directory/"rows.jsonl"));mechanism.extend(read_rows(directory/"mechanism.jsonl"))
    records,positions=records_and_positions()
    checked=audit(rows,mechanism,records,positions,previous_rows(),stops)
    entries=aggregate(rows);analysis=summarize(entries,records);analysis.update(audit=checked)
    write(base/"analysis.json",analysis);write(base/"per-case.json",list(entries.values()))
    lines=["# 真实16/24/32块：完整48开发题", "", "六类各8题；四原策略×三长度及AR，同题同卡交错，自然4096三重复/遵守EOS前128两重复。3120计时、144另计机制。仍是已观察开发集，不是新的48题独立验证；48验证及96保留尚未生成模型答案。", "",
           "各题先合并计时重复，以题为单位配对几何平均/5000 bootstrap。自然错答与截断不删；共同EOS/正确EOS仅作诊断。少于5题不展示区间。长度外推与近似接受均可能改变轨迹，不能叫无损或预测准确率提升。", ""]
    for b in BENCHMARKS:
        lines += ["## "+b,"","|方法|自然秒（题中位和）|token|TPS|主正确|截断|调用|regret|重复极差中位|","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name,t in analysis["methods"][b].items():
            lines.append(f"|{name}|{t['seconds']:.4f}|{t['tokens']}|{t['tps']:.2f}|{t['correct']}/{t['tasks']}|{t['truncated']}|{t['forwards']}|{t['regret']:.3f}|{100*t['median_repeat_spread']:.2f}%|")
        lines += ["","|比较|自然比[95%]/题数|等128比[95%]/题数|长度比|调用密度比|摊销成本比|质量升/降/未评|另7开发题自然比|","|---|---|---|---:|---:|---:|---|---|"]
        for name,c in analysis["comparisons"][b].items():
            other=c["other_development"]
            lines.append(f"|{name}|{interval(c['natural'])}|{interval(c['prefix_equal128'])}|{c['length_ratio']:.4f}|{c['target_density_ratio']:.4f}|{c['cost_per_forward_ratio']:.4f}|{c['quality_improved']}/{c['quality_lost']}/{c['quality_unassessed']}|{interval(other['natural']) if other else '无'}|")
        lines += ["", "辅助质量、需复核及受限未评估：`"+json.dumps({m:{k:t[k] for k in ("secondary_correct","needs_review","restricted_unassessed","f1_mean")} for m,t in analysis["methods"][b].items()},ensure_ascii=False)+"`。", ""]
    diffs=[r for r in checked["numeric"] if not r["exact"]]
    lines += ["## 审计与解释边界","",f"计时{checked['rows']}行，16块含AR重现{checked['native_b16_cases']}行；严格D三长度自然/128共{len(checked['numeric'])}对，逐ID不同{len(diffs)}对。自然/前缀在裁短块之前分歧{len(checked['prefix_disagreements'])}项。",
              f"机制{checked['mechanism_rows']}条、独立标量检查{checked['independent_mechanism_blocks']}块；实际proposal/top IDs和EOS可用，不枚举缺失位置。trace不包含完整词表logits或最后bonus的logit，不能据此重算全部原始浮点分数。",
              "", "逐题及几何平均时延比=长度比×调用密度比×每调用摊销成本比；后者含prefill/Draft/全部在线成本，不是纯Target内核计时或因果分解。有效接受裁掉EOS后未输出proposal；验证位置和完整Draft噪声位置为操作计数，不等于FLOPs。",
              "", "原六题预检重叠与另外42题开发迁移、共同EOS/正确EOS诊断及每题正负方向均在analysis.json；不跨任务混成一个主分数，不按最小区间选择赢家。", "",
              "重建：`.venv\\Scripts\\python.exe -m scripts.write_asd_block_suite_report`。协议ASD_BLOCK_SUITE_PROTOCOL.md，原始结果results/asd-block-suite。", ""]
    (ROOT/"docs/ASD_BLOCK_SUITE_RESULTS.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(rows=len(rows),mechanism=len(mechanism),numeric_differences=len(diffs))))


if __name__=="__main__":main()
