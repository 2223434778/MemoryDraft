"""Cross-task audit and paired, question-level uncertainty without method selection."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from memorydraft.evaluation.asd_suite import BENCHMARKS, load_suite, score
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts.run_asd_suite import METHODS, prefix_agreement

ROOT=Path(__file__).resolve().parents[1]
PAIRS=(("DFlash","AR"),("ASD","DFlash"),("ASD-Run","ASD"),("ASD-Utility","ASD"),
       ("ASD-Run","DFlash"),("ASD-Utility","DFlash"))


def paired(ratios,seed):
    if not ratios:
        return None
    if any(x<=0 or not math.isfinite(x) for x in ratios):
        raise ValueError("Invalid paired time ratio")
    logs=[math.log(x) for x in ratios]
    rng=random.Random(seed)
    samples=sorted(math.exp(sum(rng.choice(logs) for _ in logs)/len(logs)) for _ in range(5000))
    return dict(n=len(logs),ratio=math.exp(statistics.mean(logs)),ci95=[samples[125],samples[4874]],
                slower=sum(x>1 for x in ratios),faster=sum(x<1 for x in ratios))


def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[r["request_id"],r["method"],r["stage"]].append(r)
    entries={}
    for (q,m,stage),items in groups.items():
        item=items[0]
        seconds=statistics.median(r["latency_seconds"] for r in items)
        values=[r["latency_seconds"] for r in items]
        n=len(item["token_ids"])
        entries[q,m,stage]=dict(request_id=q,benchmark=item["benchmark"],method=m,seconds=seconds,tokens=n,
                               finish_reason=item["finish_reason"],quality=item.get("quality"),
                               target_forwards=n if m=="AR" else len(item["rounds"])+1,
                               spread=max(values)/min(values)-1,complete128=n==128,
                               regret=item["asd"]["spent"] if item["asd"] else 0,
                               relaxed=item["asd"]["relaxed_tokens"] if item["asd"] else 0)
    return entries


def summarize(entries):
    table,comparisons={},{}
    for b in BENCHMARKS:
        table[b]={}; comparisons[b]={}
        for m in METHODS:
            selected=[r for (q,method,stage),r in entries.items() if r["benchmark"]==b and method==m and stage=="natural"]
            if not selected:
                continue
            seconds=sum(r["seconds"] for r in selected); tokens=sum(r["tokens"] for r in selected)
            table[b][m]=dict(tasks=len(selected),seconds=seconds,tokens=tokens,tps=tokens/seconds,
                correct=sum(r["quality"]["correct"] for r in selected),truncated=sum(r["finish_reason"]!="eos" for r in selected),
                target_forwards=sum(r["target_forwards"] for r in selected),regret=sum(r["regret"] for r in selected),
                repeat_spread_median=statistics.median(r["spread"] for r in selected),
                secondary_correct=sum(r["quality"].get("secondary_correct",r["quality"]["correct"]) for r in selected),
                needs_review=[r["request_id"] for r in selected if r["quality"].get("needs_review")],
                f1_mean=statistics.mean(r["quality"]["f1"] for r in selected) if b=="squad" else None)
        ids=sorted({q for (q,m,s),r in entries.items() if r["benchmark"]==b and s=="natural"})
        for m,ref in PAIRS:
            natural=[]; prefix=[]; quality_changes=[]; length_ratios=[]; density_ratios=[]
            for q in ids:
                a,z=entries[q,m,"natural"],entries[q,ref,"natural"]
                natural.append(a["seconds"]/z["seconds"])
                length_ratios.append(a["tokens"]/z["tokens"])
                density_ratios.append((a["target_forwards"]/a["tokens"])/(z["target_forwards"]/z["tokens"]))
                quality_changes.append(int(a["quality"]["correct"])-int(z["quality"]["correct"]))
                ap,zp=entries[q,m,"prefix"],entries[q,ref,"prefix"]
                if ap["complete128"] and zp["complete128"]:
                    prefix.append(ap["seconds"]/zp["seconds"])
            comparisons[b][m+"/"+ref]=dict(natural=paired(natural,f"suite:natural:{b}:{m}:{ref}"),
                prefix_equal128=paired(prefix,f"suite:prefix:{b}:{m}:{ref}"),
                length_ratio_geomean=math.exp(statistics.mean(math.log(x) for x in length_ratios)),
                target_density_ratio_geomean=math.exp(statistics.mean(math.log(x) for x in density_ratios)),
                quality_improved=sum(x>0 for x in quality_changes),quality_lost=sum(x<0 for x in quality_changes))
    return dict(methods=table,comparisons=comparisons)


def audit(rows,mechanism,records,stops):
    seen={}
    for r in rows:
        key=r["request_id"],r["method"],r["stage"],r["repeat"]
        if key in seen:
            raise ValueError("Duplicate suite row")
        seen[key]=r
        if r["request_id"] not in records or r["method"] not in METHODS:
            raise ValueError("Unexpected suite identity")
        limit=4096 if r["stage"]=="natural" else 128
        if not 0<len(r["token_ids"])<=limit or r["max_new_tokens"]!=limit or r["latency_seconds"]<=0:
            raise ValueError("Invalid suite generation")
        if (r["finish_reason"]=="eos")!=(r["token_ids"][-1] in stops):
            raise ValueError("Invalid EOS")
        if r["asd"] and r["asd"]["spent"]>r["asd"]["budget"]+1e-9:
            raise ValueError("Exceeded request regret budget")
    expected={(q,m,stage,rep) for q in records for m in METHODS
              for stage,reps in (("natural",3),("prefix",2)) for rep in range(reps)}
    if set(seen)!=expected:
        raise ValueError("Incomplete suite matrix")
    numeric=[]
    for q,record in records.items():
        identity=None
        for m in METHODS:
            for stage,reps in (("natural",3),("prefix",2)):
                first=seen[q,m,stage,0]
                if identity is None:
                    identity=first["prompt_sha256"],first["snapshot_id"]
                for repeat in range(reps):
                    r=seen[q,m,stage,repeat]
                    if (r["prompt_sha256"],r["snapshot_id"])!=identity:
                        raise ValueError("Changed target input/history")
                    if (r["token_ids"],r["asd"],r["rounds"])!=(first["token_ids"],first["asd"],first["rounds"]):
                        raise ValueError("Non-deterministic suite repeats")
                if stage=="natural" and first["quality"]!=score(record,first["text"]):
                    raise ValueError("Changed suite quality scoring")
                if stage=="prefix":
                    checked=prefix_agreement(seen[q,m,"natural",0],first)
                    if checked!=first["prefix_agreement"]:
                        raise ValueError("Changed prefix agreement audit")
            if m=="DFlash":
                for stage in ("natural","prefix"):
                    a,z=seen[q,m,stage,0]["token_ids"],seen[q,"AR",stage,0]["token_ids"]
                    common=next((i for i,(x,y) in enumerate(zip(a,z)) if x!=y),min(len(a),len(z)))
                    numeric.append(dict(request_id=q,benchmark=record["benchmark"],stage=stage,exact=a==z,common_prefix=common))
        if any(r.get("regret",0) for r in seen[q,"DFlash","natural",0]["rounds"]):
            raise ValueError("Strict method spent approximate regret")
    events=set()
    for r in mechanism:
        key=r["request_id"],r["method"]
        if key in events or r["method"]=="AR":
            raise ValueError("Duplicate or invalid mechanism row")
        events.add(key)
        natural=seen[*key,"natural",0]
        if (r["token_ids"],r["asd"],r["quality"])!=(natural["token_ids"],natural["asd"],natural["quality"]):
            raise ValueError("Mechanism changed timed trajectory")
        if len(r["events"])!=len(r["rounds"]):
            raise ValueError("Incomplete mechanism events")
    if len(mechanism)!=48:
        raise ValueError("Missing frozen mechanism rows")
    return numeric


def main():
    base=ROOT/"results/asd-suite-stage1"
    info=json.loads((base/"execution-verification.json").read_text(encoding="utf-8"))
    if info["status"]!="complete" or info["files_verified_before"]!=info["files_verified_after"]:
        raise ValueError("Incomplete source audit")
    rows=[]; mechanism=[]; stops=None
    for i in (0,1):
        directory=base/f"part-{i}"; p=json.loads((directory/"provenance.json").read_text(encoding="utf-8"))
        if p["status"]!="complete" or not p["original_unchanged"] or p["split"]!="development":
            raise ValueError("Incomplete or changed worker")
        if stops is not None and stops!=p["stop_token_ids"]:
            raise ValueError("Changed stop tokens")
        stops=p["stop_token_ids"]
        rows.extend(read_rows(directory/"rows.jsonl")); mechanism.extend(read_rows(directory/"mechanism.jsonl"))
    records={r["id"]:r for r in load_suite(ROOT/"data/research/asd-suite")}
    numeric=audit(rows,mechanism,records,stops)
    entries=aggregate(rows); analysis=summarize(entries)
    analysis.update(rows=len(rows),mechanism_rows=len(mechanism),questions=len(records),numeric=numeric,
                    prefix_disagreements=[dict(request_id=r["request_id"],method=r["method"],**r["prefix_agreement"])
                                          for r in rows if r["stage"]=="prefix" and r["repeat"]==0 and not r["prefix_agreement"]["agrees"]])
    write(base/"analysis.json",analysis)
    write(base/"per-question.json",list(entries.values()))
    lines=["# 跨任务基线第一阶段", "", "48个新开发请求、6类各8题，五方法自然3重复/遵守EOS的前128两重复；验证48及保留96尚未生成模型答案。重复不算新题。", "",
           "FP32/eager、batch1、greedy、固定Target输入。自然上限4096；所有错答/截断保留。仅建立基线，不用本轮直接否决整个优化方向。", ""]
    for b in BENCHMARKS:
        lines += ["## "+b,"","|方法|自然秒（逐题中位数之和）|token|TPS|主正确|截断|Target调用|重复极差中位|","|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m,t in analysis["methods"][b].items():
            lines.append(f"|{m}|{t['seconds']:.4f}|{t['tokens']}|{t['tps']:.2f}|{t['correct']}/{t['tasks']}|{t['truncated']}|{t['target_forwards']}|{t['repeat_spread_median']*100:.2f}%|")
        lines += ["","|方法/参照|自然时延比[题bootstrap95%]|自然更慢题|长度比|调用密度比|等128时延比[95%]/题数|质量提升/下降|","|---|---|---:|---:|---:|---|---|"]
        for name,c in analysis["comparisons"][b].items():
            n=c["natural"]; p=c["prefix_equal128"]
            fixed=(f"{p['ratio']:.4f} [{p['ci95'][0]:.4f},{p['ci95'][1]:.4f}]/{p['n']}" if p else "无共同等128样本")
            lines.append(f"|{name}|{n['ratio']:.4f} [{n['ci95'][0]:.4f},{n['ci95'][1]:.4f}]|{n['slower']}/{n['n']}|{c['length_ratio_geomean']:.4f}|{c['target_density_ratio_geomean']:.4f}|{fixed}|{c['quality_improved']}/{c['quality_lost']}|")
        if b=="math500":
            lines += ["", "MATH有限标量等价后的正确数与需复核ID：`"+json.dumps({m:dict(secondary=t["secondary_correct"],review=t["needs_review"]) for m,t in analysis["methods"][b].items()},ensure_ascii=False)+"`。主评分不改。"]
        if b=="squad":
            lines += ["", "SQuAD平均token F1：`"+json.dumps({m:t["f1_mean"] for m,t in analysis["methods"][b].items()})+"`。"]
        lines.append("")
    differences=[r for r in numeric if not r["exact"]]
    lines += ["## 数值与边界", "", f"严格D与AR完整自然/128共{len(numeric)}次检查，不一致{len(differences)}次。首分歧详情见analysis.json；不放宽接受条件或删除请求。",
              f"自然/前缀在最后裁短块之前的不一致{len(analysis['prefix_disagreements'])}项，见analysis.json。",
              "", "时延比为逐题配对几何平均，95%区间按题bootstrap5000次；总秒/TPS是另一种汇总口径，不能混为一个加速比。八题/类别仍是开发样本，不足以证明质量非劣或不存在收益。",
              "", "SQuAD短输出不强制续写，没有共同128样本时不发布固定工作量速度。MATH主评分失败中非标量答案进入待复核清单；代码用提供测试，不代表全部隐藏边界。JSON明确是合成控制。",
              "", "本阶段不选择新策略、不消费验证/保留模型结果。后续按ASD_RESEARCH_ROADMAP.md逐项登记实验，先调查跨场景机会再冻结方法。",
              "", "重建：`.venv\\Scripts\\python.exe -m scripts.write_asd_suite_report`。原始结果`results/asd-suite-stage1`，协议ASD_SUITE_PROTOCOL.md。", ""]
    (ROOT/"docs/ASD_SUITE_RESULTS.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(questions=len(records),rows=len(rows),mechanism=len(mechanism),numeric_differences=len(differences))))


if __name__=="__main__":
    main()
