"""Summarize measured artifacts without fabricating missing experiments."""
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
lines = ["# GPU pilot 实测报告", "", "2026-10-16。A6000 GPU0，Target Qwen2.5-1.5B-Instruct、H/D 共用普通 Qwen2.5-0.5B-Instruct，FP32/eager，batch=1，greedy，最多 64 新 token、候选块长 8。模型/分词器/embedding 版本见对应 provenance。",
         "", "13 个请求来自 3 个合成项目流及空历史对照，不是真实用户长期分布。单次 pilot 未作统计显著性或通用性能声明。所有方法固定 Target 输入；同步端到端耗时包括在线快照、检索、prefill、起草、验证和缓存维护。", ""]
for directory, label in [("validation", "词法诊断检索：初版路由"), ("ar-fallback", "词法诊断检索：增加 AR 成本退路"), ("semantic", "真正语义检索：最终路由")]:
    path = root / "results" / directory / "replay.jsonl.summary.json"
    lines.extend([f"## {label}", ""])
    if not path.exists():
        lines.extend(["尚未完成；不提供性能数字。", ""])
        continue
    summary = json.loads(path.read_text(encoding="utf-8"))
    lines.extend(["| 方法 | 请求数 | 总时间（秒） | AR 时间 / 方法时间 | token 一致 |", "|---|---:|---:|---:|---|"])
    for method in ("AR", "D", "R", "RD", "H", "RHD"):
        if method not in summary:
            continue
        row = summary[method]
        lines.append(f"| {method} | {row['requests']} | {row['total_seconds']:.3f} | {row['speedup_vs_ar']:.3f} | {'通过' if row['all_correct'] else '失败'} |")
    lines.extend(["", "大于 1 才表示相对 AR 加速；小于 1 表示变慢。跨表是不同运行，不把它们混成一次配对统计。", ""])
    if "D" in summary and "H" in summary:
        d, h = summary["D"]["routes"]["D"], summary["H"]["routes"]["H"]
        lines.extend([f"D 接受 {d['accepted']}/{d['proposed']} 个候选 token；H 接受 {h['accepted']}/{h['proposed']}。接受数不等于最终提交数，后者还含 Target 纠正或 bonus。", ""])
    if "RHD" in summary:
        lines.extend(["完整路由实际事件：" + "，".join(f"{route}={stats['rounds']} 轮" for route, stats in summary["RHD"]["routes"].items()) + "。", ""])

lines.extend(["## H 历史条件消融", "", "仅比较同一组 5 个受控请求；Target 输入始终相同，无关历史由公开的合成花园描述按原 token 长度生成，不使用其他用户私有历史。", "", "| 历史条件 | D 总时间 | H 总时间 | D 接受/提议 | H 接受/提议 | 输出一致 |", "|---|---:|---:|---:|---:|---|"])
ids = {json.loads(line)["id"] for line in (root / "data/synthetic/h_cases.jsonl").read_text().splitlines()}
for directory, label in [("semantic", "原有历史"), ("ablation-empty", "空历史"), ("ablation-unrelated", "等 token 无关历史")]:
    path = root / "results" / directory / "replay.jsonl"
    if not path.exists():
        continue
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    group = {m: [r for r in rows if r["request_id"] in ids and r["method"] == m] for m in ("D", "H")}
    if any(len(group[m]) != len(ids) for m in group):
        continue
    stats = {m: {key: sum(r["route_stats"][m][key] for r in group[m]) for key in ("accepted", "proposed")} for m in group}
    times = {m: sum(r["latency_seconds"] for r in group[m]) for m in group}
    correct = all(r["correct"] for subset in group.values() for r in subset)
    lines.append(f"| {label} | {times['D']:.3f} | {times['H']:.3f} | {stats['D']['accepted']}/{stats['D']['proposed']} | {stats['H']['accepted']}/{stats['H']['proposed']} | {'通过' if correct else '失败'} |")

lines.extend(["", "## 结论与限制", "", "本轮小模型组合上，未证明 H 的净收益或完整系统加速。普通 Draft 的串行生成与 Python 调度开销相对小 Target 较大；额外历史没有稳定提高接受量。AR 退路可以减少损失，但关闭 H 的轮次不能作为 H 研究成功。先保留这些负结果，再评估匹配的更强 Target/成熟 Draft，避免扩大训练或堆叠机制。",
             "", "BF16/SDPA 初测在 docs-3 第 54 个 token 发生分歧；同请求 FP32/eager 一致。诊断保存于 `results/diagnostics/precision.json`。不放宽匹配规则，也不把失败运行计入有效无损统计。FP32/eager 的通过只限已测请求，不保证所有硬件/内核/任意输入位级等价。",
             "", "B6/B7 尚未运行：当前 Target 没有已核验的匹配 EAGLE 权重。已确认官方 Qwen2-7B ↔ EAGLE-Qwen2-7B 组合，但它需要另一套 Target 和特征状态适配，不能冒充当前模型的强基线。采样/并发/服务化未实现。正式研究还需要应用事实记忆设置 B、真实合法历史、更长输出、重复/交错运行、轻量 trace 及 CPU 内存/成本分析。",
             "", "## 复现", "", "在服务器项目根目录执行 `OMP_NUM_THREADS=4 .venv/bin/python scripts/run_a6000_suite.py`。源代码包由 `python scripts/bundle_source.py` 构建，运行 provenance 包含 SHA-256 源码清单。原始初版日志没有源码清单，其配置与模型版本仍保留。", ""])
(root / "docs/EXPERIMENTS.md").write_text("\n".join(lines), encoding="utf-8")
print(root / "docs/EXPERIMENTS.md")
