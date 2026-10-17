"""Rebuild the Target-history H switch report from paired artifacts, without a GPU."""
import gzip
import json
import statistics
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from memorydraft.evaluation.replay import summarize
from scripts.write_h_causal_report import relative_time_interval
from scripts.write_research_report import speed_interval


METHODS = ("AR", "RHD_off", "RHD_on", "RHD_on_cached")


def rows(path):
    path = Path(path)
    raw = path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def validate_rows(data, snapshots, references, repeats=3, token_limit=128):
    expected = {(rid, repeat, method) for rid in snapshots for repeat in range(repeats) for method in METHODS}
    keys = [(row["request_id"], row["repeat"], row["method"]) for row in data]
    if len(keys) != len(set(keys)) or set(keys) != expected or set(snapshots) != set(references):
        raise ValueError("Incomplete, duplicate or unexpected paired rows")
    for row in data:
        rid = row["request_id"]
        reference = references[rid]
        if not row["correct"] or row["token_ids"] != reference["token_ids"][:token_limit]:
            raise ValueError("Output differs from frozen Target reference")
        if row["target_prompt_sha256"] != reference["target_prompt_sha256"]:
            raise ValueError("Target prompt changed")
        if row["snapshot_id"] != snapshots[rid]["snapshot_id"]:
            raise ValueError("Snapshot differs from frozen Target history")
        if row["method"] == "RHD_off" and any(event["route"] == "H" for event in row["rounds"]):
            raise ValueError("H-off activated H")


def draft_rounds(value):
    return sum(value["routes"][route]["rounds"] for route in ("H", "D"))


def verification_rounds(value):
    return sum(route["rounds"] for route in value["routes"].values())


def same_prefix_audit(data, prefix_rows):
    index = {(row["request_id"], row["position"]): row for row in prefix_rows}
    totals = dict(rounds=0, proposed=0, online_h=0, selected_h=0, native_d=0)
    for row in data:
        if row["method"] != "RHD_on":
            continue
        position = 0
        for event in row["rounds"]:
            if event["route"] == "H":
                prefix = index[(row["request_id"], position)]
                count = event["proposed"]
                if (prefix["target_prompt_sha256"] != row["target_prompt_sha256"] or count > prefix["budget"] or
                        prefix["reference_ids"][:count] != row["token_ids"][position:position + count]):
                    raise ValueError("Offline prefix reference differs from the online H position")
                totals["rounds"] += 1
                totals["proposed"] += count
                totals["online_h"] += event["accepted"]
                totals["selected_h"] += min(prefix["lengths"]["H_teacher"], count)
                totals["native_d"] += min(prefix["lengths"]["D_native"], count)
            position += event["committed"]
        if position != len(row["token_ids"]):
            raise ValueError("Committed round lengths do not reconstruct the Target output")
    return totals


def main():
    base = root / "results/h-teacher-router"
    provenance = json.loads((base / "provenance.json").read_text(encoding="utf-8"))
    if provenance["status"] != "complete" or not provenance["history_reused_without_regeneration"]:
        raise ValueError("Experiment incomplete or frozen history not reused")
    parent = json.loads((root / "results/h-causal/provenance.json").read_text(encoding="utf-8"))
    if provenance["frozen_teacher_history_sha256"] != parent["frozen_teacher_history_sha256"]:
        raise ValueError("Teacher history differs from the source experiment")
    data = rows(base / "replay.jsonl")
    snapshots = json.loads((base / "snapshots.json").read_text(encoding="utf-8"))
    references = {row["request_id"]: row for row in rows(root / "results/h-causal/router-replay.jsonl") if row["method"] == "AR"}
    validate_rows(data, snapshots, references)
    validate_rows(rows(base / "smoke.jsonl"), {key: snapshots[key] for key in list(snapshots)[:2]},
                  {key: references[key] for key in list(snapshots)[:2]}, repeats=1, token_limit=16)
    summary = summarize(data)
    if summary != json.loads((base / "summary.json").read_text(encoding="utf-8")):
        raise ValueError("Saved summary disagrees with replay")
    off, on, cached = (summary[method] for method in METHODS[1:])
    on_change = on["total_seconds"] / off["total_seconds"] - 1
    cached_change = cached["total_seconds"] / off["total_seconds"] - 1
    lines = ["# 同一 Target 历史下的 H 开关验证", "",
             "2026-10-17。由 `scripts/write_h_teacher_router_report.py` 从完整配对记录重建。协议先于 GPU 运行提交，见 [H_TEACHER_ROUTER_PROTOCOL](H_TEACHER_ROUTER_PROTOCOL.md)。这是原六个合成项目的追加诊断，不是新的独立保留集。", "",
             f"相对 H-off，H-on 总耗时变化 **{on_change:+.2%}**，缓存 H-on **{cached_change:+.2%}**；正值表示更慢，负值表示更快。下面的配对时间比和请求级区间用于判断本组是否出现净收益。", "",
             "## 对照条件", "",
             "R 和 H 共享上一轮冻结的 Target 过去回答。R 使用完整合法快照的静态 token 接续索引及原动态 lookup，H 使用请求级语义检索；R 不受 H 的 top-k 限制。H-off/on 复用同一 RHD 路由，D/AR 探测、EMA、冷却、驻留及 D 原生上下文相同。H-on 增加的检索、探测、prefill、起草与追赶全部计费。",
             "固定 Qwen3-8B/0.6B、单卡 A6000 GPU0、FP32/eager、非思考 greedy、k=2、输出上限 128、历史预算 512，六请求三次交错重复。smoke 与正式计时的向量缓存各自从空开始。每次查询在线编码，历史向量可跨合法快照复用相同文本。",
             f"冻结历史 SHA-256：`{provenance['frozen_teacher_history_sha256']}`。本轮直接校验并复用，未重新生成；构造成本已在上一轮记录。当前请求、Target prompt 与输出均和上一轮 AR 核对。", "",
             "## 端到端时间", "",
             "| 方法 | 次数 | 总秒数 | AR/方法 | 请求 bootstrap 95% 区间 | TTFT 中位数 ms | L1 累计秒数 | PyTorch 峰值 GiB |",
             "|---|---:|---:|---:|---|---:|---:|---:|"]
    for method in METHODS:
        value = summary[method]
        selected = [row for row in data if row["method"] == method]
        low, high = speed_interval(data, method)
        ttft = statistics.median(row["ttft_seconds"] for row in selected) * 1000
        l1 = sum(row["stage_seconds"]["l1_host"] for row in selected)
        peak = max(max(row["peak_gpu_bytes"].values()) for row in selected) / 2**30
        lines.append(f"| {method} | {value['requests']} | {value['total_seconds']:.3f} | {value['speedup_vs_ar']:.3f}× | [{low:.3f}, {high:.3f}] | {ttft:.1f} | {l1:.3f} | {peak:.2f} |")
    lines += ["", "| 配对比较 | baseline/方法 | 请求 bootstrap 95% 区间 |", "|---|---:|---|"]
    for baseline, method in (("RHD_off", "RHD_on"), ("RHD_off", "RHD_on_cached"), ("RHD_on", "RHD_on_cached")):
        ratio = summary[baseline]["total_seconds"] / summary[method]["total_seconds"]
        low, high = relative_time_interval(data, baseline, method)
        lines.append(f"| {baseline}/{method} | {ratio:.3f}× | [{low:.3f}, {high:.3f}] |")
    lines += ["", "大于 1 表示分母方法更快。按请求成组 bootstrap 2000 次，保留其三次重复。六项目共享少量模板，区间仅供探索性判断；token 和重复不是独立自然用户样本。显存是包含驻留 T/S 的 PyTorch 分配峰值，CPU 峰值内存未测。", "",
              "| 访问 | H-off/H-on | H-off/缓存 H-on |", "|---|---:|---:|"]
    for label, repeats in (("首次 repeat=0", {0}), ("后续 repeat=1,2", {1, 2})):
        times = {method: sum(row["latency_seconds"] for row in data if row["method"] == method and row["repeat"] in repeats)
                 for method in METHODS[1:]}
        lines.append(f"| {label} | {times['RHD_off']/times['RHD_on']:.3f}× | {times['RHD_off']/times['RHD_on_cached']:.3f}× |")
    lines += ["", "## 验证与起草成本", "",
              "| 方法 | R/H/D/AR 轮数 | 验证轮数 | H+D 起草轮数 | H 接受/提议 | H prefill/追赶输入 token |",
              "|---|---|---:|---:|---:|---:|"]
    for method in METHODS[1:]:
        value = summary[method]
        routes = value["routes"]
        counts = "/".join(str(routes[route]["rounds"]) for route in ("R", "H", "D", "AR"))
        h = routes["H"]
        tokens = sum(event.get("draft_prefill_or_catchup_tokens", 0) for row in data if row["method"] == method
                     for event in row["rounds"] if event["route"] == "H")
        lines.append(f"| {method} | {counts} | {verification_rounds(value)} | {draft_rounds(value)} | {h['accepted']}/{h['proposed']} | {tokens} |")
    lines += ["", f"H-on 相对 H-off 的验证轮数变化为 **{verification_rounds(on)-verification_rounds(off):+d}**，H+D 起草轮数变化为 **{draft_rounds(on)-draft_rounds(off):+d}**。缓存 H-on 对应为 **{verification_rounds(cached)-verification_rounds(off):+d} / {draft_rounds(cached)-draft_rounds(off):+d}**。",
              "H 路线的接受比例不是额外收益：它可能替代原本 R/D 已能接受的接续。prefill/追赶输入 token 是工作量代理；host 分阶段计时受 CUDA 异步影响，不能相加解释为精确 GPU kernel 成本。AR 没有候选器轮次事件，因此不报告其验证轮数为零。", "",
              "## 逐项目与重复", "",
              "| 项目 | 场景 | H-off/H-on | H-off/缓存 H-on | on−off 验证轮数 | on−off 起草轮数 |",
              "|---|---|---:|---:|---:|---:|"]
    for rid in snapshots:
        selected = [row for row in data if row["request_id"] == rid]
        group = summarize(selected)
        a, b, c = (group[method] for method in METHODS[1:])
        lines.append(f"| {rid} | {selected[0]['scenario']} | {a['total_seconds']/b['total_seconds']:.3f}× | {a['total_seconds']/c['total_seconds']:.3f}× | {verification_rounds(b)-verification_rounds(a):+d} | {draft_rounds(b)-draft_rounds(a):+d} |")
    lines += ["", "| repeat | H-off/H-on | H-off/缓存 H-on |", "|---|---:|---:|"]
    for repeat in range(3):
        group = summarize([row for row in data if row["repeat"] == repeat])
        lines.append(f"| {repeat} | {group['RHD_off']['total_seconds']/group['RHD_on']['total_seconds']:.3f}× | {group['RHD_off']['total_seconds']/group['RHD_on_cached']['total_seconds']:.3f}× |")
    by_key = {(row["request_id"], row["repeat"], row["method"]): row for row in data}
    fields = ("route", "proposed", "accepted", "committed", "source_id", "match_length")
    signatures = lambda row: [tuple(event.get(field) for field in fields) for event in row["rounds"]]
    identical = sum(signatures(row) == signatures(by_key[(row["request_id"], row["repeat"], "RHD_on_cached")])
                    for row in data if row["method"] == "RHD_on")
    lines += ["", f"缓存与未缓存 H-on 的路线、来源、接受计数序列相同的配对为 {identical}/18；不据此声称所有候选 token 逐一相同。", "",
              "## 已选 H 位置的同前缀质量检查", ""]
    audit = same_prefix_audit(data, rows(root / "results/h-causal/alignment-prefix.jsonl"))
    lines += [f"将本轮已选 H 的 {audit['rounds']} 轮起点与上一轮完整前缀评分对齐。实际在线 H 接受 **{audit['online_h']}/{audit['proposed']}**；相同前缀处，旧评分中的 D_native 候选可接受 **{audit['native_d']}/{audit['proposed']}**，人工只选一条 Target 历史的 H_teacher 可接受 **{audit['selected_h']}/{audit['proposed']}**。",
              f"实际 H 与该离线 D 评分总量只差 **{audit['online_h']-audit['native_d']:+d} token**。此处重复仍是同一组请求，不能当作独立质量样本。D 的数字是已保存候选的离线评分，未测替换这些轮次后整条路由的输出步进与延迟；也不能将局部差值当作全系统增益。",
              "离线 H_teacher 只含指定旧记录，在线 H 使用原自动检索上下文，因此不是同一 H prompt 的复验；这个比较也不能单独识别多历史干扰或某个检索器的因果效果。它检查的是实际被路由选中的前缀上，是否已有很大的候选质量优势。", "",
              "## 本轮判断", "",
              "更换为同一 Target 的过去回答后，本组数据仍没有得到 H 的端到端增量收益。复制和代码修改的验证轮数没有减少；文档修改增加 18 轮验证，新任务减少 9 轮但增加 12 轮小模型起草。两种 H-on 在六个项目上均慢于 H-off。",
              "当前证据将问题收敛到 H 的额外候选质量、与 R/D 的重叠，以及包含探测和追赶的启用成本之间的关系。单独改善历史来源或向量缓存不足以让这套固定策略获益；尚不能区分新的触发策略、专门训练或候选长度优化中哪一种能解决它。", "",
              "## 正确性、复现与解释边界", "",
              f"共 {len(data)} 次完整计时和 8 次 smoke，全部输出与原 AR 对齐；Target prompt 与各请求冻结快照核验通过，结束后历史未变。正式缓存初始状态、最终统计和每次访问保存在 provenance/cache-visits。",
              "在线快照、检索、索引、prefill、起草、验证、追赶、缓存维护与输出解码均计入。模型加载、公共预热、独立 AR 参考及这次实验前已完成的历史构造不计入请求延迟。",
              "这是对当前实现、冻结块长和既有六项目的功能开关验证。原人工历史批次可作背景，不能把跨批时间差解释为纯历史来源的因果效应。结果不证明真实长期用户分布的普遍收益或无效，也未验证新的 H 触发策略、训练、成熟 B6/B7、采样或并发。", "",
              f"实际 GPU 运行源码清单 SHA-256：`{provenance['source_manifest']['source_sha256']}`。结果目录 `results/h-teacher-router/`；完整命令见协议；离线重建：`python scripts/write_h_teacher_router_report.py`。", ""]
    log = base / "tests-final.log"
    if log.exists():
        lines += ["## 最终回归", "", "```text", log.read_text(encoding="utf-8").strip(), "```", ""]
    (root / "docs/H_TEACHER_ROUTER_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote docs/H_TEACHER_ROUTER_RESULTS.md")


if __name__ == "__main__":
    main()
