"""Rebuild the two H diagnostic results from complete, paired artifacts; no GPU required."""
import gzip
import json
import statistics
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scripts.write_research_report import speed_interval


base = root / "results/h-causal"


def rows(name):
    path = base / name
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        with gzip.open(str(path) + ".gz", "rt", encoding="utf-8") as stream:
            text = stream.read()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def relative_time_interval(data, baseline, method):
    paired = [dict(r, method="AR" if r["method"] == baseline else method)
              for r in data if r["method"] in {baseline, method}]
    return speed_interval(paired, method)


if __name__ == "__main__":
    provenance = json.loads((base / "provenance.json").read_text(encoding="utf-8"))
    if provenance["status"] != "complete" or not provenance["current_requests_read_after_teacher_freeze"]:
        raise ValueError("Causal experiment incomplete or teacher history not frozen first")
    source = json.loads((base / "alignment-summary.json").read_text(encoding="utf-8"))
    prefix = rows("alignment-prefix.jsonl")
    conditions = source["conditions"]
    generations = json.loads((base / "teacher-history/generations.json").read_text(encoding="utf-8"))
    lines = ["# H 追加实验：过去回答来源与相同路由下的 H 开关", "",
             "2026-10-17。此报告由 `scripts/write_h_causal_report.py` 从完整结果重建。沿用第二轮六个合成项目，属于事后机制/工程诊断，不是新的独立保留集。协议见 `docs/H_CAUSAL_PROTOCOL.md`。",
             "固定 Qwen3-8B Target / Qwen3-0.6B S、单卡 A6000、FP32/eager、非思考 greedy；当前输出上限 128、k=2、历史预算 512。Target 输入及严格验证规则不变，H/D 共用 S 权重、隔离 KV。", "",
             "## 实验一：过去回答来源", "",
             "仅从原始 order=1 历史构造模拟过去的 lower()/ten-minutes 请求（order=0），先用同一 Target AR 生成过去回答并冻结为 order=1，再读取当前测试请求（order=10）。没有用当前答案构造过去，没有训练或更新权重。干扰记录保持原样。",
             f"共生成 {len(generations)} 条旧回答，Target 生成累计 {provenance['teacher_generation_seconds']:.3f}s，建资产墙钟 {provenance['teacher_build_wall_seconds']:.3f}s；此为一次性模拟历史构造成本，不能当作已测在线速度收益。达到 256 token 上限的旧回答有 {sum(g['hit_output_budget'] for g in generations)} 条。",
             f"冻结历史 SHA-256：`{provenance['frozen_teacher_history_sha256']}`；过去任务 SHA-256：`{provenance['frozen_past_requests_sha256']}`。", "",
             "人工预选相同合法记录，排除自动检索召回变化。全前缀逐位置比较，先对每个请求求均值，再等权汇总；bootstrap 按请求重采样。", "",
             "| 条件 | 请求平均连续接受长度 | 相对 D_format 增量 | 增量请求 bootstrap 95% 区间 | 首位命中 |", "|---|---:|---:|---|---:|"]
    for name, value in conditions.items():
        low, high = value["delta_bootstrap95_request"]
        lines.append(f"| {name} | {value['mean_accepted_length_per_request']:.3f} | {value['mean_delta_vs_format_per_request']:+.3f} | [{low:+.3f}, {high:+.3f}] | {value['survival_by_position']['1']:.2%} |")
    lines += ["", "首位命中按位置计数，与请求等权均值的权重不同。H_empty 与 D_format 提议逐一相等。matched 条件将双方历史截到较短者的 token 数；截断也会丢失内容，只用于长度敏感性检查，不等于完美隔离文风。", "",
              "| 直接配对比较 | 平均增量 | 请求 bootstrap 95% 区间 | 原人工 R 弱匹配位置平均增量 |", "|---|---:|---|---:|"]
    for key, label in (("teacher_minus_manual", "H_teacher − H_manual"),
                       ("teacher_minus_manual_matched", "等长度 H_teacher − H_manual")):
        value = source[key]
        low, high = value["bootstrap95_request"]
        lines.append(f"| {label} | {value['mean_delta']:+.3f} | [{low:+.3f}, {high:+.3f}] | {value['weak_r_mean_delta']:+.3f} |")
    lines += ["", "R 弱匹配固定为原人工历史索引的后缀长度 <3，所有来源共用同一分层；不覆盖动态 prompt lookup。弱匹配列按该子集位置平均。", "",
              "| 项目 | 场景 | 位置数 | 人工/Target 历史 token 数 | Teacher−Manual 平均增量 |", "|---|---|---:|---:|---:|"]
    for rid in dict.fromkeys(r["request_id"] for r in prefix):
        group = [r for r in prefix if r["request_id"] == rid]
        first = group[0]
        payload = first["history_payload_tokens"]
        delta = source["teacher_minus_manual"]["per_request"][rid]
        lines.append(f"| {rid} | {first['scenario']} | {len(group)} | {payload['H_manual']}/{payload['H_teacher']} | {delta:+.3f} |")
    lines += ["", f"共 {len(prefix)} 个位置；新 AR 参考 token 和 Target prompt hash 已与第二轮计时 AR 核对一致。两种历史来自相同任务事实，但生成方式、措辞和长度可能一起变化，因此主比较衡量整体历史来源效果，不能只归因为某个格式特征。", "",
              "## 实验二：相同路由规则的 H 开关", "",
              "继续使用原人工历史。RHD_off/on 共享 R 索引、D 原生上下文和 RHD 路由代码：同样的 R 冷却、D/AR 两轮探测、EMA、最小驻留；on 额外允许 H 并计入其两轮探测、检索、prefill 和追赶成本。旧 RD 无相同 AR 选择逻辑，仅作为辅助基线。",
              "RHD_on_cached 是同一个 H-on 的缓存优化控制，缓存初始为空；查询每次在线编码，首次历史编码计入在线成本。AR 与所有方法交错运行，三次重复。预热/模型加载与离线评分参考不计入请求延迟；在线快照、检索、索引、起草、验证、缓存维护、输出解码均计入。", ""]
    data = rows("router-replay.jsonl")
    summary = json.loads((base / "router-summary.json").read_text(encoding="utf-8"))
    lines += ["| 方法 | 次数 | 总秒数 | AR/方法 | 请求 bootstrap 95% 区间 | TTFT 中位数 ms | L1 累计秒数 |", "|---|---:|---:|---:|---|---:|---:|"]
    for method, value in summary.items():
        low, high = speed_interval(data, method)
        selected = [r for r in data if r["method"] == method]
        ttft = statistics.median(r["ttft_seconds"] for r in selected) * 1000
        retrieval = sum(r["stage_seconds"]["l1_host"] for r in selected)
        lines.append(f"| {method} | {value['requests']} | {value['total_seconds']:.3f} | {value['speedup_vs_ar']:.3f}× | [{low:.3f}, {high:.3f}] | {ttft:.1f} | {retrieval:.3f} |")
    lines += ["", "| 比较 | baseline/方法 | 请求 bootstrap 95% 区间 |", "|---|---:|---|"]
    for baseline, method in (("RHD_off", "RHD_on"), ("RHD_off", "RHD_on_cached"),
                             ("RHD_on", "RHD_on_cached"), ("RD", "RHD_off")):
        ratio = summary[baseline]["total_seconds"] / summary[method]["total_seconds"]
        low, high = relative_time_interval(data, baseline, method)
        lines.append(f"| {baseline}/{method} | {ratio:.3f}× | [{low:.3f}, {high:.3f}] |")
    lines += ["", "大于 1 表示分母方法更快。H-on 相对 H-off 才是主要功能开关估计；包含探测和成本反馈导致的路线变化，不能单独归因为 H 提议质量。", "",
              "| 方法 | R/H/D/AR 轮数 | 全部验证轮数 | H 接受/提议 token | H 首次设置 host 秒数 | H 首次/追赶输入 token 数 |", "|---|---|---:|---:|---:|---:|"]
    for method, value in summary.items():
        routes = value["routes"]
        counts = "/".join(str(routes[r]["rounds"]) for r in ("R", "H", "D", "AR"))
        h = routes["H"]
        events = [e for r in data if r["method"] == method for e in r["rounds"] if e["route"] == "H"]
        setup = sum(e.get("setup_host_seconds", 0.) for e in events)
        tokens = sum(e.get("draft_prefill_or_catchup_tokens", 0) for e in events)
        lines.append(f"| {method} | {counts} | {sum(v['rounds'] for v in routes.values())} | {h['accepted']}/{h['proposed']} | {setup:.4f} | {tokens} |")
    if summary["RHD_off"]["routes"]["H"]["rounds"]:
        raise ValueError("H-off activated H")
    lines += ["", "H 设置列是 host 包装器初始化，GPU prefill/追赶在生成路径内计入总延迟；不能将该 host 数值解释为完整 prefill 耗时。H-off 的 H 路线必须为零。路线的起始前缀不同，H/D 在线汇总接受率不作同前缀因果比较。", "",
              "| 场景 | AR/RHD_off | AR/RHD_on | AR/RHD_on_cached |", "|---|---:|---:|---:|"]
    for scenario in dict.fromkeys(r["scenario"] for r in data):
        selected = [r for r in data if r["scenario"] == scenario]
        ar = sum(r["latency_seconds"] for r in selected if r["method"] == "AR")
        ratios = [ar / sum(r["latency_seconds"] for r in selected if r["method"] == m)
                  for m in ("RHD_off", "RHD_on", "RHD_on_cached")]
        lines.append("| " + scenario + " | " + " | ".join(f"{v:.3f}×" for v in ratios) + " |")
    lines += ["", f"共 {len(data)} 次完整配对计时方法运行，另有两请求 smoke（6 次），不将重复或 token 视作独立自然用户样本。所有输出 token、Target prompt hash 和 snapshot 配对通过。六个合成项目共享少量模板，区间仅为探索性统计。", "",
              "## 复现与边界", "",
              "代码与版本清单、模型验证记录、输入哈希见 `results/h-causal/provenance.json`。模拟过去的 Target 回答及生成输入均为公开合成内容；不是自然用户日志。原始结果保留，JSONL 可无损 gzip，压缩前后哈希由发布工具记录。",
              "无需 GPU 重建本报告：`python scripts/write_h_causal_report.py`。完整 GPU 命令为协议中的 `scripts/run_h_causal.py`，脚本拒绝覆盖已有输出目录。成熟 B6/B7、真实长期数据、采样与并发不属于本轮已完成项。", ""]
    overlap_path = base / "teacher-r-overlap.json"
    if overlap_path.exists():
        overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
        lines += ["## 来源实验后的 CPU 互补性审计", "",
                  "此处额外使用同一冻结 Target 历史建立静态 R 索引，在相同已确认前缀查询后再用参考评分；不跑 GPU，不测速度。它不是第二项路由实验的 R 索引，也不将更换 R 历史与 H 开关混在一起。",
                  f"H_teacher 优于 H_manual 的 {overlap['h_improved_positions']} 个位置中，现有静态 R 检索器用 Target 历史已能提出至少同样长的正确接续的位置有 {overlap['improved_positions_static_r_covers_h']} 个。此覆盖只针对具体静态索引候选，不含动态 lookup，也不是所有历史接续的 oracle 上界。", "",
                  "| 项目 | H_teacher 平均接受长度 | 静态 R_teacher 平均接受长度 | R_teacher 弱匹配位置数 | 弱匹配处 H_teacher−H_manual |", "|---|---:|---:|---:|---:|"]
        for rid, value in overlap["per_request"].items():
            weak = value["weak_mean_h_teacher_minus_manual"]
            weak_text = f"{weak:+.3f}" if weak is not None else "无"
            lines.append(f"| {rid} | {value['mean_h_teacher']:.3f} | {value['mean_static_r_teacher']:.3f} | {value['teacher_r_weak_positions']} | {weak_text} |")
        lines += ["", "这些均为位置等权质量指标。若 H 增益主要可由同来源 R 直接复制，改进历史来源首先证明了历史语料对齐的价值，不能独自证明 H 在 R 之外的必要性。", ""]
    q = source["teacher_minus_manual"]
    low, high = q["bootstrap95_request"]
    ratios = {m: summary["RHD_off"]["total_seconds"] / summary[m]["total_seconds"]
              for m in ("RHD_on", "RHD_on_cached")}
    lines[5:5] = [f"主结果：Target 过去历史相对人工过去历史的连续接受长度增量为 **{q['mean_delta']:+.3f} token**，请求 bootstrap 95% 区间 **[{low:+.3f}, {high:+.3f}]**。相同路由下，H-off/H-on 总时间比为 **{ratios['RHD_on']:.3f}×**，H-off/缓存 H-on 为 **{ratios['RHD_on_cached']:.3f}×**（大于 1 才是启用 H 更快）。", "",
                   "两项实验分开控制变量：来源实验的 Target 历史没有用于路由实验。因此不能把来源实验的质量提升和路由表直接拼成已测的 Target-history RHD 加速。", ""]
    (root / "docs/H_CAUSAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote docs/H_CAUSAL_RESULTS.md")
