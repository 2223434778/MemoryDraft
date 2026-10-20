"""Audit unseen ASD trajectories and separate length, round density and conservative gating."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

from memorydraft.evaluation.asd_budget import score
from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from scripts.prepare_asd_attribution import ROOT, calibrate, load_policies, load_tasks, scalar_decision
from scripts.run_asd_budget_probe import compare_previous
from scripts.write_asd_budget_report import audit_events

PARTS={"math-a":tuple(f"math500:{i}" for i in range(8,16)),
       "math-b":tuple(f"math500:{i}" for i in range(16,24)),
       "gsm-a":tuple(f"gsm8k:{i}" for i in range(24,28)),
       "gsm-b":tuple(f"gsm8k:{i}" for i in range(28,32))}
POLICY_PATH=ROOT/"configs/asd-attribution-policies.json"
DATA_PATH=ROOT/"data/research/asd-attribution/records.json"
PROTOCOL_PATH=ROOT/"docs/ASD_ATTRIBUTION_PROTOCOL.md"


def close(a,b):
    return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8)


def audit_frozen(info,manifest):
    expected={n:h for n,h in manifest["files"].items() if n.startswith("results/") and not n.startswith("results/raw/")}
    if info["original_sha256"]!=expected:
        raise ValueError("Changed protected evidence coverage")
    for name,sha in expected.items():
        if file_hash(ROOT/name)!=sha:
            # Preserve the same two legacy Windows checkout exceptions as earlier audits.
            if (name not in {"results/research-audit/model_preflight.json","results/research-audit/pilot_summary.json"}
                    or hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()!=sha):
                raise ValueError("Published evidence changed")
    files={n.removeprefix("memorydraft/"):h for n,h in manifest["files"].items()
           if n.startswith("memorydraft/") and n.endswith((".py",".json",".txt"))}
    if (info["source"]["files"]!=files or info["source"]["sha256"]!=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
            or info["runner_sha256"]!=manifest["files"]["scripts/run_asd_budget_probe.py"]):
        raise ValueError("Changed frozen runtime source")


def audit_decisions(row,config):
    audit_events(row)
    spent=0.
    inferred=[]
    for n,event in enumerate(row["events"]):
        stop=None
        if n==len(row["events"])-1 and row["finish_reason"]=="eos":
            distance=len(row["token_ids"])-event["generated_before"]
            if 1<=distance<=event["accepted"]:
                stop=distance
        decision=scalar_decision(event,config,spent,stop)
        candidates=[(stop,decision)]
        # Old traces omit proposal IDs: a rejected proposal EOS can clip the
        # feasible ceiling even in a nonterminal block. A ceiling equal to the
        # unclipped one can also remove utility's bonus at that boundary.
        ceiling=event['feasible_accepted']
        if stop is None and ceiling>event['accepted']:
            candidates.append((ceiling,scalar_decision(event,config,spent,ceiling)))
        matches=[(s,d) for s,d in candidates if all(close(value,event[key]) for key,value in d.items())]
        if not matches:
            raise ValueError("Scalar and recorded verifier decisions differ")
        if matches[0][0] is not None and stop is None:
            inferred.append(dict(event=n,unaccepted_proposal_eos=matches[0][0]))
        mismatch=event["mismatch"]
        expected_values=[len(mismatch)-i if config.value=="remaining" else
                         next((j-i for j in range(i+1,len(mismatch)) if mismatch[j]),len(mismatch)-i)
                         for i in range(len(mismatch))]
        if event["values"]!=expected_values:
            raise ValueError("Incorrect suffix values")
        spent+=event["regret"]
    return inferred


def totals(rows,methods):
    result={}
    for m in methods:
        chosen=[r for r in rows if r["method"]==m]
        seconds=sum(r["latency_seconds"] for r in chosen)
        tokens=sum(len(r["token_ids"]) for r in chosen)
        rounds=sum(r["counters"]["rounds"] for r in chosen)
        result[m]=dict(seconds=seconds,tokens=tokens,tps=tokens/seconds if seconds else 0.,
                       timed_rounds=rounds,rounds_per_token=rounds/tokens if tokens else 0.,
                       seconds_per_round=seconds/rounds if rounds else 0.)
    return result


def decompose(base,new):
    if min(base["tokens"],new["tokens"],base["timed_rounds"],new["timed_rounds"])<=0:
        raise ValueError("Cannot decompose empty generation")
    factors=dict(length_ratio=new["tokens"]/base["tokens"],
                 round_density_ratio=new["rounds_per_token"]/base["rounds_per_token"],
                 round_cost_ratio=new["seconds_per_round"]/base["seconds_per_round"],
                 latency_ratio=new["seconds"]/base["seconds"])
    if not close(factors["length_ratio"]*factors["round_density_ratio"]*factors["round_cost_ratio"],factors["latency_ratio"]):
        raise ValueError("Invalid latency factorization")
    return factors


def paired_interval(requests,ids,method,base):
    rng=random.Random(20261020)
    values=[]
    for _ in range(2000):
        sample=rng.choices(ids,k=len(ids))
        values.append(sum(requests[q][method]["seconds"] for q in sample)/
                      sum(requests[q][base]["seconds"] for q in sample))
    values.sort()
    return [values[49],values[1949]]


def same_states(rows,policies):
    result={}
    names=("ASD","ASD-Run","ASD-Conservative")
    # Both alternatives are stricter than original ASD. On its nonterminal
    # states they cannot cross an unrecorded, unaccepted proposal EOS.
    for source in ("ASD",):
        selected=[r for r in rows if r["method"]==source]
        states=[(e,e["spent"]-e["regret"]) for r in selected for e in r["events"][:-1]]
        decisions={m:[scalar_decision(e,policies[m],spent) for e,spent in states] for m in names}
        for i,(event,_) in enumerate(states):
            if (decisions['ASD'][i]['accepted']!=event['accepted'] or
                    any(ds[i]['accepted']>event['feasible_accepted'] for ds in decisions.values())):
                raise ValueError('Counterfactual prefix can cross an unobserved EOS boundary')
        table={}
        for m,values in decisions.items():
            paid=[d for d in values if d["relaxed"]]
            gain=sum(max(0,d["accepted"]-d["strict_accepted"]) for d in values)
            cost=sum(d["regret"] for d in values)
            low=[d for d in paid if d["accepted"]-d["strict_accepted"]<=2]
            table[m]=dict(regret=cost,exceptions=sum(d["relaxed"] for d in values),paid_blocks=len(paid),
                          local_extra_proposals=gain,low_gain_paid_blocks=len(low),
                          low_gain_regret=sum(d["regret"] for d in low),regret_per_local_extra=cost/gain if gain else None)
        pairs={}
        for control in ("ASD","ASD-Conservative"):
            pairs[control]=dict(equal=sum(a["accepted"]==b["accepted"] for a,b in zip(decisions["ASD-Run"],decisions[control])),
                               run_more=sum(a["accepted"]>b["accepted"] for a,b in zip(decisions["ASD-Run"],decisions[control])),
                               control_more=sum(a["accepted"]<b["accepted"] for a,b in zip(decisions["ASD-Run"],decisions[control])))
        result[source]=dict(states=len(states),methods=table,pairs=pairs)
    return result


def quality_review(source,requests,methods):
    path=source/"quality-audit.json"
    if not path.exists():return None
    review=json.loads(path.read_text(encoding="utf-8"))
    for part,sha in review["mechanism_sha256"].items():
        if file_hash(source/part/"mechanism/rows.jsonl.gz")!=sha:
            raise ValueError("Quality audit refers to different raw output")
    counts={b:{m:t["correct"] for m,t in table.items()} for b,table in methods.items()}
    seen=set()
    for item in review["equivalent_answers"]:
        key=item["request_id"],item["method"]
        q,m=key
        original=requests[q][m]["quality"]
        if (key in seen or original["correct"] or original["parsed_answer"]!=item["parsed_answer"]
                or original["expected_answer"]!=item["expected_answer"] or not item["rationale"]):
            raise ValueError("Invalid separate answer equivalence audit")
        seen.add(key)
        counts[q.split(":")[0]][m]+=1
    return dict(primary_unchanged=True,correct=counts,review=review)


def analyze(source):
    policies=load_policies(POLICY_PATH)
    records={r["id"]:r for r in load_tasks(DATA_PATH)}
    calibration=json.loads((ROOT/"data/research/asd-attribution/calibration.json").read_text(encoding="utf-8"))
    if calibration!=calibrate():
        raise ValueError("Development calibration changed")
    manifest=json.loads((source/"ASD_ATTRIBUTION_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    execution=json.loads((source/"execution-verification.json").read_text(encoding="utf-8"))
    for path in (DATA_PATH,POLICY_PATH,PROTOCOL_PATH):
        if manifest["files"][path.relative_to(ROOT).as_posix()]!=file_hash(path):
            raise ValueError("Current inputs differ from frozen execution")
    digest=hashlib.sha256(json.dumps(manifest["files"],sort_keys=True).encode()).hexdigest()
    if (execution["status"]!="complete" or execution["commit"]!=manifest["commit"]
            or execution["source_sha256"]!=digest or manifest["source_sha256"]!=digest
            or execution["files_verified_before"]!=len(manifest["files"])
            or execution["files_verified_after"]!=len(manifest["files"])):
        raise ValueError("Missing frozen execution verification")
    mechanism,natural,prefix,infos={},[],[],{}
    inferred_eos=[]
    for part,ids in PARTS.items():
        directory=source/part
        info=json.loads((directory/"provenance.json").read_text(encoding="utf-8"))
        if (info["status"]!="complete" or not info["original_unchanged"] or info["task_ids"]!=list(ids)
                or info["max_new_tokens"]!=2048 or info["prefix_tokens"]!=128
                or info["data_sha256"]!=file_hash(DATA_PATH) or info["policies_sha256"]!=file_hash(POLICY_PATH)
                or info["protocol_sha256"]!=file_hash(PROTOCOL_PATH)
                or info["policies"]!={m:asdict(c) for m,c in policies.items()}):
            raise ValueError("Incomplete probe or changed frozen inputs")
        audit_frozen(info,manifest)
        infos[part]=info
        refs=json.loads((directory/"references.json").read_text(encoding="utf-8"))
        for stage,repeats in (("mechanism",1),("natural",2),("prefix",2)):
            rows=read_rows(directory/stage/"rows.jsonl")
            expected={(q,m,i) for q in ids for m in policies for i in range(repeats)}
            if len(rows)!=len(expected) or {(r["request_id"],r["method"],r["repeat"]) for r in rows}!=expected:
                raise ValueError("Missing or duplicate trajectory")
            for row in rows:
                q,m=row["request_id"],row["method"]
                ref=refs[q]
                if (row["prompt_sha256"],row["snapshot_id"])!=(ref["prompt_sha256"],ref["snapshot_id"]):
                    raise ValueError("Changed Target input or isolated history")
                if row["quality"]!=score(records[q],row["text"]):
                    raise ValueError("Changed primary quality score")
                state=row["asd"]
                if state["budget"]!=policies[m].budget or not 0<=state["spent"]<=policies[m].budget+1e-8:
                    raise ValueError("Changed or exceeded budget")
                if stage!="prefix":
                    identity=row["token_ids"]==ref["token_ids"]
                    if row["strict_identity"]!=identity or m=="DFlash" and not identity:
                        raise ValueError("Incorrect strict reference identity")
                if stage=="mechanism":
                    inferred_eos.extend(dict(request_id=q,method=m,**e) for e in audit_decisions(row,policies[m]))
                    mechanism[q,m]=row
                elif stage=="natural":
                    anchor=mechanism[q,m]
                    if row["token_ids"]!=anchor["token_ids"] or m!="DFlash" and state!=anchor["asd"]:
                        raise ValueError("Timed and mechanism trajectories differ")
                    natural.append(row)
                else:
                    if len(row["token_ids"])>128 or row["strict_identity"] is not None:
                        raise ValueError("Invalid bounded prefix")
                    if row["full_prefix"]!=compare_previous(SimpleNamespace(**mechanism[q,m]),row):
                        raise ValueError("Incorrect bounded prefix audit")
                    prefix.append(row)
    if len({i["source"]["sha256"] for i in infos.values()})!=1:
        raise ValueError("Different runtime packages")
    first={(r["request_id"],r["method"]):r for r in prefix if r["repeat"]==0}
    if any((r["token_ids"],r["asd"],r["rounds"])!=(first[r["request_id"],r["method"]]["token_ids"],
                first[r["request_id"],r["method"]]["asd"],first[r["request_id"],r["method"]]["rounds"]) for r in prefix):
        raise ValueError("Repeated prefix trajectories differ")
    output,comparisons,prefixes,states,termination={},{},{},{},{}
    requests={}
    for q in records:
        requests[q]={}
        for m in policies:
            r=mechanism[q,m]
            requests[q][m]=dict(quality=r["quality"],tokens=len(r["token_ids"]),finish_reason=r["finish_reason"],
                                rounds=r["counters"]["rounds"],asd=r["asd"],
                                seconds=sum(t["latency_seconds"] for t in natural if (t["request_id"],t["method"])==(q,m)))
    for benchmark in ("gsm8k","math500"):
        ids=[q for q,r in records.items() if r["benchmark"]==benchmark]
        selected=[r for r in mechanism.values() if r["request_id"] in ids]
        table=totals([r for r in natural if r["request_id"] in ids],policies)
        for m in policies:
            chosen=[r for r in selected if r["method"]==m]
            table[m].update(correct=sum(r["quality"]["correct"] for r in chosen),requests=len(ids),
                            truncated=sum(r["finish_reason"]=="length" for r in chosen),
                            regret=sum(r["asd"]["spent"] for r in chosen),exceptions=sum(r["asd"]["relaxed_tokens"] for r in chosen),
                            shortened=sum(r["asd"]["shortened_blocks"] for r in chosen))
        output[benchmark]=table
        comparisons[benchmark]={}
        for m in policies:
            if m=="DFlash":continue
            comparisons[benchmark][m]={}
            for base in ("ASD","ASD-Conservative","DFlash"):
                comparisons[benchmark][m][base]=dict(**decompose(table[base],table[m]),
                    tps_ratio=table[m]["tps"]/table[base]["tps"],
                    paired_question_bootstrap_latency_ratio_95=paired_interval(requests,ids,m,base),
                    quality_losses=[q for q in ids if requests[q][base]["quality"]["correct"] and not requests[q][m]["quality"]["correct"]],
                    quality_gains=[q for q in ids if not requests[q][base]["quality"]["correct"] and requests[q][m]["quality"]["correct"]],
                    slower_requests=[q for q in ids if requests[q][m]["seconds"]>requests[q][base]["seconds"]])
        equal=[q for q in ids if all(len(first[q,m]["token_ids"])==128 for m in policies)]
        prefixes[benchmark]=dict(equal_work_ids=equal,
            all=totals([r for r in prefix if r["request_id"] in ids],policies),
            equal_work=totals([r for r in prefix if r["request_id"] in equal],policies))
        states[benchmark]=same_states(selected,policies)
        groups={"all_eos":[],"all_length":[],"mixed":[]}
        for q in ids:
            reasons={r["finish_reason"] for r in requests[q].values()}
            name="all_eos" if reasons=={"eos"} else "all_length" if reasons=={"length"} else "mixed"
            groups[name].append(q)
        termination[benchmark]={g:dict(ids=qs,methods=totals([r for r in natural if r["request_id"] in qs],policies))
                                for g,qs in groups.items()}
    return dict(methods=output,comparisons=comparisons,prefixes=prefixes,same_states=states,requests=requests,
                inferred_unaccepted_eos=inferred_eos,
                quality_review=quality_review(source,requests,output),
                termination=termination,
                calibration=calibration,execution_commit=manifest["commit"],runtime_sha256=next(iter(infos.values()))["source"]["sha256"],
                grader_sha256=file_hash(ROOT/"memorydraft/evaluation/asd_budget.py")),infos


def render(a,infos):
    lines=["# ASD 新题与收益来源验证", "", "固定8道新GSM8K、16道新MATH-500；120机制+240完整自然计时+240遵守EOS的128上限前缀计时，共600生成。",
           "完整上限2048；Qwen3-8B+DFlash、FP32/eager、greedy、非思考。同题五组同卡交错，两次计时重复。全部在线成本计入，加载/预热/插桩另计。",
           "", "Conservative只将原ASD固定gate收紧至0.0625；旧开发状态做近似regret校准，新题不调参、不保证实际预算相等。旧trace缺少未接受EOS位置的限制详见文末审计修订。四个旧策略不改。", ""]
    for b in a["methods"]:
        c=a["comparisons"][b]["ASD-Run"]["ASD"]
        d=a["comparisons"][b]["ASD-Run"]["DFlash"]
        control=a["comparisons"][b]["ASD-Run"]["ASD-Conservative"]
        lines.append(f"{b}：Run相对原ASD时延{(c['latency_ratio']-1)*100:+.2f}%、自然TPS{(c['tps_ratio']-1)*100:+.2f}%；输出长度{(c['length_ratio']-1)*100:+.2f}%、每token轮数{(c['round_density_ratio']-1)*100:+.2f}%。相对严格DFlash时延{(d['latency_ratio']-1)*100:+.2f}%、自然TPS{(d['tps_ratio']-1)*100:+.2f}%；相对保守ASD时延{(control['latency_ratio']-1)*100:+.2f}%、自然TPS{(control['tps_ratio']-1)*100:+.2f}%。")
    lines += ["", "自然时延更低可以来自推理路径/长度变化；本报告用严格DFlash、保守门槛及等token前缀检查是否还有执行效率增量。下列全题结果包含错答和截断，质量与终止状态始终单列。", ""]
    c=a['comparisons']['math500']['ASD-Run']['ASD-Conservative']
    table=a['methods']['math500']
    prefix=a['prefixes']['math500']['equal_work']
    prefix_change=(prefix['ASD-Run']['seconds']/prefix['ASD']['seconds']-1)*100
    lines += [f"MATH归因重点：Run/Conservative实际总regret为{table['ASD-Run']['regret']:.3f}/{table['ASD-Conservative']['regret']:.3f}，例外{table['ASD-Run']['exceptions']}/{table['ASD-Conservative']['exceptions']}。Run相对这个保守对照的输出长度变化{(c['length_ratio']-1)*100:+.2f}%，自然TPS变化{(c['tps_ratio']-1)*100:+.2f}%；共同128token前缀相对原ASD时延{prefix_change:+.2f}%。因此，完整输出路径变化与验证执行效率需要分别解释，不能仅用全请求自然时延认定实际后缀信息带来独有提速。", "",
              "旧三题补全曾同时表现为输出减少8.58%、每token轮数减少8.22%、每轮成本约+0.32%。那是三个已观察的长题诊断；新16题的轮密度结果不同，不能把旧局部优势直接推广，也不需要改写旧结果。两轮推理package相同，当前标量/张量决策与重复轨迹审计通过。", ""]
    lines += ["本轮判断：没有复现Run/Utility相对原ASD的稳定每token吞吐增量。新MATH上自然时延更低伴随解题路径与质量变化，Run/ASD、Run/保守ASD的跨题时延比区间均包含1；GSM上Run退化，Utility基本持平。保留局部选择器作为实验资产，不把本轮包装成通用后缀提速或稳定质量保证。", "",
              "后续建议：停止在本批新题扫描门槛、预算和价格。若继续研究，先在旧开发集建立原ASD的门槛/预算速度—质量对照，再用独立任务判断实际后缀信息是否带来额外价值；当前不优先增加更复杂的预测器或扩大同类GPU试验。", ""]
    for b,table in a["methods"].items():
        lines += [f"## {b} 完整答案", "", "| 方法 | 秒（两重复） | token（两重复） | 自然TPS | 主评分正确 | 截断 | 每100token轮 | regret | 例外 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m,t in table.items():
            lines.append(f"| {m} | {t['seconds']:.6f} | {t['tokens']} | {t['tps']:.3f} | {t['correct']}/{t['requests']} | {t['truncated']} | {t['rounds_per_token']*100:.3f} | {t['regret']:.3f} | {t['exceptions']} |")
        lines += ["", "| 方法/参照 | 时延变化 | TPS变化 | 长度比 | 轮密度比 | 每轮成本比 | 时延比95%区间 |", "|---|---:|---:|---:|---:|---:|---|"]
        for m in ("ASD-Run","ASD-Utility","ASD-Conservative"):
            for base in (("ASD","ASD-Conservative","DFlash") if m!="ASD-Conservative" else ("ASD","DFlash")):
                c=a["comparisons"][b][m][base]
                lo,hi=c['paired_question_bootstrap_latency_ratio_95']
                lines.append(f"| {m}/{base} | {(c['latency_ratio']-1)*100:+.2f}% | {(c['tps_ratio']-1)*100:+.2f}% | {c['length_ratio']:.4f} | {c['round_density_ratio']:.4f} | {c['round_cost_ratio']:.4f} | [{lo:.4f}, {hi:.4f}] |")
        lines += ["", "`时延比=长度比×轮密度比×每轮成本比`严格成立，但这些因素来自不同实现轨迹，不是可独立相加的因果效应。每轮成本含prefill摊销及全部在线开销。", ""]
        for m in ("ASD-Run","ASD-Utility","ASD-Conservative"):
            c=a['comparisons'][b][m]['ASD']
            lines.append(f"- {m}相对原ASD：逐题时延退化{len(c['slower_requests'])}/{table[m]['requests']}题；主评分损失 `{', '.join(c['quality_losses']) or '无'}`，提升 `{', '.join(c['quality_gains']) or '无'}`。")
        lines.append("")
        lines += ["区间按题配对重采样2000次，两次重复先合并为同一题；只是小切片不确定性诊断，不提供质量非劣性保证。", "",
                  "| 五组终止状态 | 题数 | Run/ASD时延变化 | Utility/ASD时延变化 | Conservative/ASD时延变化 |", "|---|---:|---:|---:|---:|"]
        for group,g in a['termination'][b].items():
            if not g['ids']:continue
            base=g['methods']['ASD']['seconds']
            changes=[(g['methods'][m]['seconds']/base-1)*100 for m in ('ASD-Run','ASD-Utility','ASD-Conservative')]
            lines.append(f"| {group} | {len(g['ids'])} | {changes[0]:+.2f}% | {changes[1]:+.2f}% | {changes[2]:+.2f}% |")
        lines += ["", "终止状态分组只作事后诊断，全题表始终是主结果；完整结束不等于答案正确。", ""]
        p=a["prefixes"][b]
        lines += [f"128前缀五组全部达到128token的共同集合：{len(p['equal_work_ids'])}/{table['ASD']['requests']}题；ID `{', '.join(p['equal_work_ids'])}`。未达到者仍保留在analysis.json的all表中。", "",
                  "| 方法 | 等token前缀秒 | token | TPS | 相对原ASD时延 |", "|---|---:|---:|---:|---:|"]
        for m,t in p["equal_work"].items():
            base=p["equal_work"]["ASD"]["seconds"]
            change=(t["seconds"]/base-1)*100 if base else 0
            lines.append(f"| {m} | {t['seconds']:.6f} | {t['tokens']} | {t['tps']:.3f} | {change:+.2f}% |")
        s=a["same_states"][b]["ASD"]
        lines += ["", f"在原ASD实际非终止轨迹的{s['states']}个共同候选块、相同剩余账本上离线重算：", "",
                  "| 选择器 | regret | 例外 | 局部额外proposal | 付费且局部增量≤2的块 | 该部分regret |", "|---|---:|---:|---:|---:|---:|"]
        for m,t in s["methods"].items():
            lines.append(f"| {m} | {t['regret']:.3f} | {t['exceptions']} | {t['local_extra_proposals']} | {t['low_gain_paid_blocks']} | {t['low_gain_regret']:.3f} |")
        for base,c in s["pairs"].items():
            lines.append(f"- Run与{base}：相同{c['equal']}块、Run多接{c['run_more']}块、{base}多接{c['control_more']}块。")
        lines += ["", "各块独立借用源轨迹账本，阴影支出相加可能超过每题预算×题数，不能当作替代策略的实际总消耗。局部额外proposal不等于全请求节约token/验证轮；离线反事实不推进模型状态。", "",
                  "逐题完整结果：", "", "| 请求 | 方法 | 秒 | token | 轮 | 正确 | 结束 |", "|---|---|---:|---:|---:|---|---|"]
        for q,rows in a["requests"].items():
            if not q.startswith(b+":"):continue
            for m,r in rows.items():
                lines.append(f"| {q} | {m} | {r['seconds']:.6f} | {r['tokens']} | {r['rounds']} | {r['quality']['correct']} | {r['finish_reason']} |")
    lines += ["", "## 主评分未通过的输出", "", "主指标沿用保守boxed规范化精确匹配，不宣称一般符号等价；数值题使用冻结GSM解析器。以下全部保留，不将格式假失败视为真实推理错误：", ""]
    for q,rows in a["requests"].items():
        for m,r in rows.items():
            if not r["quality"]["correct"]:
                lines.append(f"- {q}/{m}: {json.dumps(r['quality'],ensure_ascii=False,sort_keys=True)}")
    if a["quality_review"]:
        review=a["quality_review"]
        lines += ["", "## 单列答案等价审计", "", "原始rows和主评分均未改动。人工核对新MATH主评分失败答案，math500:23五组的`5`与参考`x=5`等价；其他数值错误、负的相位参数和未完成输出不提升评分。详情及原始机制SHA见quality-audit.json。", "",
                  "| 方法 | MATH保守主评分 | MATH答案等价审计后 |", "|---|---:|---:|"]
        for m,t in a["methods"]["math500"].items():
            lines.append(f"| {m} | {t['correct']}/16 | {review['correct']['math500'][m]}/16 |")
    lines += ["", "旧gsm8k:21的Utility真实错误、旧math500:7的Run退化继续保留；本轮不修正旧结论。短前缀样本是完整新题的补充诊断，不是新的独立问题。",
              "", "离线审计修订：原trace没有保存未接受proposal的EOS位置。本轮gsm8k:30/Utility有一块拒绝EOS后继续生成，EOS裁剪可从记录的可行上限反推；审计兼容此情况，不更改原推理/账本/计时。",
              "同状态反事实限定为原ASD的非终止块：Run与Conservative的门槛都更严，可选前缀必在原ASD实际可行上限内，不会跨越未知EOS。原协议拟报告五条源轨迹，此处因日志可识别性收窄为这一条，其他源不作无界后缀收益统计。",
              "旧开发Run日志同样缺少未接受EOS标记，因此原门槛校准是近似启发式，不能视作精确EOS裁剪后的支出匹配。保持预先冻结gate0.0625，正式比较以新题实际账本为准。",
              "", f"冻结执行commit `{a['execution_commit']}`；runtime package `{a['runtime_sha256']}`。源码与旧发布证据执行前后SHA一致。",
              "重建：`python -m scripts.write_asd_attribution_report`。配置/数据/协议/源码SHA、逐题原始输出、账本、分片命令和服务器测试见results/asd-attribution。", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"results/asd-attribution")
    parser.add_argument("--report",type=Path,default=ROOT/"docs/ASD_ATTRIBUTION_RESULTS.md")
    args=parser.parse_args()
    a,infos=analyze(args.source)
    (args.source/"analysis.json").write_bytes((json.dumps(a,ensure_ascii=False,indent=2)+"\n").encode())
    args.report.write_bytes(render(a,infos).encode())
    print(json.dumps({b:{m:dict(latency_change_percent=(c['ASD']['latency_ratio']-1)*100,
                               tps_change_percent=(c['ASD']['tps_ratio']-1)*100)
                        for m,c in table.items() if m in ('ASD-Run','ASD-Utility','ASD-Conservative')}
                      for b,table in a['comparisons'].items()},ensure_ascii=False))


if __name__=="__main__":
    main()
