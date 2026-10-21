"""Audit the frozen 228-run longer-draft preflight; do not select winners."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_suite import score
from memorydraft.evaluation.runtime_migration import read_rows
from scripts.diagnose_transport_online import write
from scripts.run_asd_block_probe import ARMS, check_native, prefix_check, selected_records
from scripts.run_asd_budget_probe import POLICIES

ROOT=Path(__file__).resolve().parents[1]


def audit(rows,records,previous,stops):
    records={r["id"]:r for r in records}
    expected={(q,m,size,stage,rep) for q in records for m,size in ARMS
              for stage,reps in (("natural",1),("prefix",2)) for rep in range(reps)}
    expected|={(q,"AR",16,stage,0) for q in records for stage in ("natural","prefix")}
    seen={}
    for r in rows:
        key=r["request_id"],r["method"],r["experimental_block_size"],r["stage"],r["repeat"]
        if key in seen or key not in expected:raise ValueError("Changed block matrix identity")
        seen[key]=r
        limit=4096 if r["stage"]=="natural" else 128
        if (r["training_block_size"]!=16 or r["max_new_tokens"]!=limit or r["latency_seconds"]<=0
                or not 0<len(r["token_ids"])<=limit or r["benchmark"]!=records[r["request_id"]]["benchmark"]):
            raise ValueError("Invalid block row")
        if (r["finish_reason"]=="eos")!=(r["token_ids"][-1] in stops):raise ValueError("Invalid block EOS")
        if r["asd"] and r["asd"]["spent"]>r["asd"]["budget"]+1e-9:raise ValueError("Exceeded block budget")
        if r["method"]!="AR":
            cfg=POLICIES[r["method"]]
            if r["asd"]["budget"]!=cfg.budget or (r["method"]=="DFlash" and r["asd"]["spent"]!=0):
                raise ValueError("Changed block policy budget")
            if any(not 0<=rd["accepted"]<=rd["proposed"]<=r["experimental_block_size"]-1 for rd in r["rounds"]):
                raise ValueError("Invalid block proposal dimensions")
        if r["experimental_block_size"]==16:check_native(r,previous[r["request_id"],r["method"],r["stage"]])
        if r["stage"]=="natural" and r["quality"]!=score(records[r["request_id"]],r["text"]):
            raise ValueError("Changed block scoring")
    if set(seen)!=expected:raise ValueError("Incomplete block matrix")
    numeric=[];disagreements=[]
    for q in records:
        identity=seen[q,"AR",16,"natural",0]
        for m,size in (*ARMS,("AR",16)):
            natural=seen[q,m,size,"natural",0]
            prefix=seen[q,m,size,"prefix",0]
            for stage,reps in (("natural",1),("prefix",1 if m=="AR" else 2)):
                first=seen[q,m,size,stage,0]
                for rep in range(reps):
                    r=seen[q,m,size,stage,rep]
                    if (r["prompt_sha256"],r["snapshot_id"])!=(identity["prompt_sha256"],identity["snapshot_id"]):
                        raise ValueError("Changed block Target input")
                    if (r["token_ids"],r["asd"],r["rounds"])!=(first["token_ids"],first["asd"],first["rounds"]):
                        raise ValueError("Changed block repeat")
                if m=="DFlash":
                    a,z=first["token_ids"],seen[q,"AR",16,stage,0]["token_ids"]
                    common=next((i for i,(x,y) in enumerate(zip(a,z)) if x!=y),min(len(a),len(z)))
                    numeric.append(dict(request_id=q,size=size,stage=stage,exact=a==z,common_prefix=common))
            checked=prefix_check(natural,prefix,size)
            if checked!=prefix["prefix_agreement"]:raise ValueError("Changed block prefix audit")
            if not checked["agrees"]:disagreements.append(dict(request_id=q,method=m,size=size,**checked))
    return dict(rows=len(rows),questions=len(records),native_b16_cases=sum(r["experimental_block_size"]==16 for r in rows),
                numeric=numeric,prefix_disagreements=disagreements)


def merge(rows):
    groups=defaultdict(list)
    for r in rows:groups[r["request_id"],r["method"],r["experimental_block_size"],r["stage"]].append(r)
    entries={}
    for key,items in groups.items():
        r=items[0];seconds=[v["latency_seconds"] for v in items]
        entries[key]=dict(request_id=r["request_id"],benchmark=r["benchmark"],method=r["method"],size=r["experimental_block_size"],
                          stage=r["stage"],seconds=statistics.median(seconds),repeat_seconds=seconds,
                          tokens=len(r["token_ids"]),finish_reason=r["finish_reason"],quality=r.get("quality"),
                          forwards=len(r["token_ids"]) if r["method"]=="AR" else 1+len(r["rounds"]),
                          regret=r["asd"]["spent"] if r["asd"] else 0.)
    return entries


def main():
    base=ROOT/"results/asd-block-preflight"
    info=json.loads((base/"execution-verification.json").read_text())
    if info["status"]!="complete" or info["files_verified_before"]!=info["files_verified_after"]:
        raise ValueError("Incomplete block execution")
    rows=[];stops=None
    for i in (0,1):
        directory=base/f"part-{i}";p=json.loads((directory/"provenance.json").read_text())
        if p["status"]!="complete" or not p["original_unchanged"]:raise ValueError("Incomplete block worker")
        if stops is not None and stops!=p["stop_token_ids"]:raise ValueError("Changed block stop IDs")
        stops=p["stop_token_ids"];rows.extend(read_rows(directory/"rows.jsonl"))
    previous={}
    for i in (0,1):
        for r in read_rows(ROOT/f"results/asd-suite-stage1/part-{i}/rows.jsonl"):
            if r["repeat"]==0:previous[r["request_id"],r["method"],r["stage"]]=r
    records=selected_records();checked=audit(rows,records,previous,stops);entries=merge(rows)
    write(base/"analysis.json",checked);write(base/"per-case.json",list(entries.values()))
    lines=["# 更长真实Draft预检", "", "六类各第一道开发题；训练块16的同一权重真实起草16/24/32，四策略保留同长度对照。自然各一次/遵守EOS前128各两次，AR各一次，共228计时。", "",
           "这不是48题确认实验，每类只有1题；自然时间无重复稳定性结论，不作质量非劣或方法否决，不选择赢家或扫参数。", ""]
    lines += ["## 成本观察（开发预检）","","两道数学请求均达到128。下表24/16为同策略同题比较，负数表示更快；自然只计时一次，不能视为稳定收益。","",
              "|请求|策略|自然时延变化|等128时延变化|","|---|---|---:|---:|"]
    for q in ("gsm8k:885","math500:329"):
        for m in ("DFlash","ASD","ASD-Run","ASD-Utility"):
            a,z=entries[q,m,24,"natural"],entries[q,m,16,"natural"]
            p,r=entries[q,m,24,"prefix"],entries[q,m,16,"prefix"]
            lines.append(f"|{q}|{m}|{100*(a['seconds']/z['seconds']-1):+.2f}%|{100*(p['seconds']/r['seconds']-1):+.2f}%|")
    lines += ["", "更长块已能正确执行；严格D在本批没有显示长块自然时延收益，部分请求验证调用反而增加。Utility24在两道数学请求的等128时间下降，基础ASD则一升一降；这提供完整48题检查的理由，不构成泛化结论。长块同时改变Draft预测与近似门控，可行域与路径效应尚未因果分离，不能说这些收益就是预测更准。", ""]
    for record in records:
        q=record["id"]
        lines += ["## "+q,"","|方法|块|自然秒|token|正确|结束|Target调用|regret|前缀中位秒|前缀实际token|","|---|---:|---:|---:|---|---|---:|---:|---:|---:|"]
        for m,size in (("AR",16),*ARMS):
            a=entries[q,m,size,"natural"];p=entries[q,m,size,"prefix"]
            lines.append(f"|{m}|{size}|{a['seconds']:.5f}|{a['tokens']}|{a['quality']['correct']}|{a['finish_reason']}|{a['forwards']}|{a['regret']:.5f}|{p['seconds']:.5f}|{p['tokens']}|")
        lines.append("")
    diffs=[r for r in checked["numeric"] if not r["exact"]]
    lines += ["## 审计与边界","",f"完整矩阵{len(rows)}行；16块（包括AR及前缀重复）与上阶段{checked['native_b16_cases']}行重现。严格D三长度自然/前128共36对，与同轮AR不一致{len(diffs)}对；自然/前缀在裁短块之前不一致{len(checked['prefix_disagreements'])}项。详情analysis.json。",
              "", "各实际Draft长度及剩余长度裁短可改变近似门控/后续轨迹；原请求预算不重置。24/32能运行不表示在该长度训练过，不表示预测变准。短答案遵守EOS；不足128不算固定工作量吞吐。",
              "", "协议ASD_BLOCK_PROTOCOL.md；重建`.venv\\Scripts\\python.exe -m scripts.write_asd_block_report`。源码/旧结果SHA、真实模型结构测试和原始记录见results/asd-block-preflight。", ""]
    (ROOT/"docs/ASD_BLOCK_RESULTS.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(rows=len(rows),numeric_differences=len(diffs))))


if __name__=="__main__":main()
