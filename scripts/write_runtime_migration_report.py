"""Audit saved migration rows and rebuild the report without model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import METHODS, audit_rows, frozen_inputs, input_hashes, read_rows
from scripts.write_reuse_report import ratio_interval


ROOT = Path(__file__).resolve().parents[1]


def rebuild(source, original, output):
    info = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    if info["status"] != "complete" or info["fresh_ar_correct"] != 8 or not info["original_inputs_unchanged"]:
        raise ValueError("Migration run or fresh AR validation incomplete")
    if input_hashes(original) != info["original_input_sha256"]:
        raise ValueError("Original published confirmation bytes changed")
    tasks, _ = frozen_inputs(original)
    rows = read_rows(source / "rows.jsonl")
    summary = audit_rows(rows, tasks)
    if summary != json.loads((source / "summary.json").read_text(encoding="utf-8")):
        raise ValueError("Saved summary differs from audited rows")
    fresh_rows = read_rows(source / "fresh-ar.jsonl")
    references = {row["request_id"]: row for row in fresh_rows}
    if len(references) != len(tasks) or len(fresh_rows) != len(tasks):
        raise ValueError("Fresh AR references missing or duplicated")
    for task in tasks:
        reference = references[task.request.id]
        if (reference["token_ids"] != task.reference["token_ids"]
                or reference["prompt_sha256"] != task.reference["prompt_sha256"]
                or reference["snapshot_id"] != task.snapshot_id):
            raise ValueError("Fresh AR differs from frozen reference")
    resources = json.loads((source / "resources.json").read_text(encoding="utf-8"))
    if ({r["request_id"] for r in resources} != {t.request.id for t in tasks}
            or len(resources) != len(tasks) or any(r["legacy"] != r["aligned"] for r in resources)):
        raise ValueError("Aligned candidate resources differ")
    keyed = {(r["request_id"], r["repeat"], r["method"]): r for r in rows}
    counter_keys = ("rounds", "draft_rounds", "reuse_rounds", "reuse_accepted_tokens")

    def different_counters(left, right):
        differences = []
        for task in tasks:
            for repeat in range(3):
                a, b = (keyed[task.request.id, repeat, method] for method in (left, right))
                if any(a["counters"][key] != b["counters"][key] for key in counter_keys):
                    differences.append({"request_id": task.request.id, "repeat": repeat})
        return differences

    comparisons = (("Legacy-D", "Legacy-R"), ("Runtime-D", "Runtime-R"), ("Runtime-D", "Runtime-R-aligned"),
                   ("Runtime-D", "Legacy-D"), ("Runtime-R-aligned", "Legacy-R"), ("Runtime-R", "Runtime-R-aligned"))
    times = summary["times"]
    ratios = {f"{left}/{right}": {"ratio": times[left] / times[right],
                                  "project_bootstrap_95": list(ratio_interval(rows, left, right))}
              for left, right in comparisons}
    diagnostic = {"comparisons": ratios, "engineering_slowdown_budget": .10,
                  "d_budget_passed": times["Runtime-D"] / times["Legacy-D"] <= 1.10,
                  "r_aligned_budget_passed": times["Runtime-R-aligned"] / times["Legacy-R"] <= 1.10,
                  "legacy_to_aligned_counter_differences": different_counters("Legacy-R", "Runtime-R-aligned"),
                  "default_to_aligned_counter_differences": different_counters("Runtime-R", "Runtime-R-aligned"),
                  "aligned_resources_equal": len(resources),
                  "fresh_ar_seconds_separate": sum(r["latency_seconds"] for r in references.values())}
    (source / "diagnostic.json").write_bytes((json.dumps(diagnostic, indent=2) + "\n").encode())
    gain = times["Runtime-D"] / times["Runtime-R"]
    gain_ci = ratios["Runtime-D/Runtime-R"]["project_bootstrap_95"]
    lines = ["# 原正结果到infra运行时的迁移验证", "", "2026-10-18。运行前[协议](RUNTIME_MIGRATION_PROTOCOL.md)。工程回归验证，非新的独立研究确认。", "",
             ("**本批旧工作负载在新版生产入口中仍有R净收益。**" if gain_ci[0] > 1 else
              "**本批尚未确认新版生产入口保留了稳定R净收益。**"),
             f"新版D/R总时间比为{gain:.3f}×，延迟变化为{(times['Runtime-R']/times['Runtime-D']-1):+.1%}。",
             "", "## 同批端到端计时", "",
             "原4项目、每项目两轮，五种配置各3重复，共120次计时；全部tokens、Target prompt及合法快照匹配原参考。8个新AR完整生成也与原参考一致，另计时间。", "",
             "| 配置 | 计时次数 | 总秒数 | R轮数 | Draft轮数 | R接受token |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        c = summary["counters"][method]
        lines.append(f"| {method} | 24 | {times[method]:.6f} | {c['reuse_rounds']} | {c['draft_rounds']} | {c['reuse_accepted_tokens']} |")
    lines += ["", "Legacy-D/R使用原解码循环；Runtime-D/R使用未修改的v0.3生产入口；aligned组仅将候选资源中的当前文本对齐为user文本，Target输入始终不变。", "",
              "| 比较（分子总耗时/分母） | 比值 | 项目bootstrap95%区间 |", "|---|---:|---|" ]
    for label, values in ratios.items():
        low, high = values["project_bootstrap_95"]
        lines.append(f"| {label} | {values['ratio']:.4f} | [{low:.4f}, {high:.4f}] |")
    lines += ["", "前3项是D/R速度比，超过1表示R更快；后3项是迁移或资源对齐的耗时比，超过1表示分子更慢。区间以4项目为重采样单位、2000次，重复不是独立请求，不泛化生产流量。", "",
              "## 回归与资源控制", "",
              f"Runtime-D相对Legacy-D的10%工程退化门槛：{'通过' if diagnostic['d_budget_passed'] else '未通过'}；Runtime-R-aligned相对Legacy-R：{'通过' if diagnostic['r_aligned_budget_passed'] else '未通过'}。工程预算不等于统计等价性证明。",
              f"旧版R与对齐新版R逐请求/重复的R/D轮数、总轮数及接受量差异：{len(diagnostic['legacy_to_aligned_counter_differences'])}/24；新版默认R与对齐组差异：{len(diagnostic['default_to_aligned_counter_differences'])}/24。这里比较的是计数，不声称逐轮候选轨迹逐字相同。",
              f"对齐资源的tokens、记录边界及来源指纹{len(resources)}/{len(tasks)}相同。新版默认额外包含system文本，实际资源差异和大小见resources.json；不能把其影响与纯调用包装开销混为一谈。",
              "", "## 为什么此前infra例子变慢", "",
              "此前v0.3验收用了两个不同的短请求，不能与旧配置修订直接判断版本回归。原文重复例的Target在行末插入两个空格，历史候选为普通换行，token切分不同。DFlash为5轮；R+DFlash仍需5轮Draft，另加3轮R且只接受1token。无命中解释题仅约0.5%额外耗时，重复题约53.5%，混合总计约10.6%。这些数字来自原infra验收记录，本轮没有重跑或改写它。",
              "", "本轮沿用旧配置修订数据，不改变匹配长度、候选长度、冷却、Target格式或接受规则，也未加自适应退化保护。原先已知事实展开可解释复用机会的归因限制仍成立；保住收益不重新证明长期历史必要性或方法新颖性。", "",
              "## 成本与来源", "",
              "相同A6000 GPU0、相同Target/DFlash权重、FP32/eager、TF4.57.3、4线程，方法在请求/重复内随机交错。所有在线快照、tokenization、冷索引、起草、验证、KV维护、同步及解码计入；模型加载、独立预热、资源核对和新AR单列。旧入口复用SQLite连接，新入口每请求开关连接属于被测工程成本。无HTTP、无profiler。",
              f"新AR另计{diagnostic['fresh_ar_seconds_separate']:.3f}s。原确认的96条记录先通过原audit，原文件运行前后SHA全部不变。",
              "", f"原确认运行源码SHA：`{info['original_runtime_source']}`。",
              f"本轮package源码SHA：`{info['source']['sha256']}`。",
              f"本轮执行脚本SHA：`{info['runner_sha256']}`。",
              "", "旧解码与旧索引源码分别从固定Git提交恢复，逐文件SHA严格匹配原运行清单。最初将索引差异误判为换行，恢复失败后查明为新增可选user记录开关，在任何GPU运行前修正协议并重新冻结；本次实际运行使用正确原代码。",
              "", "## 复算", "", "```bash", "python -m scripts.write_runtime_migration_report", "```", "",
              "公开小型证据在results/runtime-migration/；完整源码来源、测试XML及发布哈希分别保存。模型、数据库和大型trace不提交。", ""]
    output.write_bytes("\n".join(lines).encode())
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/runtime-migration")
    parser.add_argument("--original", type=Path, default=ROOT / "results/reuse-confirmation")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/RUNTIME_MIGRATION_RESULTS.md")
    args = parser.parse_args()
    print(json.dumps(rebuild(args.source, args.original, args.output), indent=2))


if __name__ == "__main__":
    main()
