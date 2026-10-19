"""Verify saved calibration predictions and rebuild the development report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from memorydraft.evaluation.runtime_migration import check_result, read_rows
from memorydraft.evaluation.splice_probe import inputs, input_digest
from memorydraft.retrieval.acceptance import AcceptancePredictor
from scripts.collect_acceptance_probe import METHODS, protected
from scripts.evaluate_acceptance_probe import evaluate
from scripts.write_benefit_report import equivalent

ROOT = Path(__file__).resolve().parents[1]


def build(source):
    manifest = json.loads((source / "source-manifest.json").read_text())
    info = json.loads((source / "collection/provenance.json").read_text())
    tasks, events = inputs(ROOT)
    files = {n.removeprefix("memorydraft/"): h for n, h in manifest["files"].items()
             if n.startswith("memorydraft/") and n.endswith((".py", ".json", ".txt"))}
    if (info["status"] != "complete" or not info["original_unchanged"] or info["original_sha256"] != protected(ROOT)
            or info["source"]["files"] != files or info["source"]["sha256"] != hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
            or info["runner_sha256"] != manifest["files"]["scripts/collect_acceptance_probe.py"]
            or info["protocol_sha256"] != manifest["files"]["docs/ACCEPTANCE_PROTOCOL.md"]
            or info["inputs_sha256"] != input_digest(tasks, events)):
        raise ValueError("Changed acceptance execution provenance")
    rows = read_rows(source / "collection/rows.jsonl")
    expected = {(t.request.id, method) for t in tasks for method in METHODS}
    if len(rows) != 48 or {(r["request_id"], r["method"]) for r in rows} != expected:
        raise ValueError("Changed strict generation coverage")
    task_map = {t.request.id: t for t in tasks}
    old = {(r["request_id"], r["method"]): r for r in read_rows(ROOT / "results/ot-matrix/mechanism/rows.jsonl")}
    matched = 0
    for r in rows:
        check_result(SimpleNamespace(**r), task_map[r["request_id"]])
        prior = old.get((r["request_id"], r["method"]))
        if prior is not None:
            if any(v != r["counters"][k] for k, v in prior["counters"].items()
                   if k != "alignment_seconds" and not k.startswith("matrix_")):
                raise ValueError("Changed original decoding path")
            matched += 1
    if matched != info["previous_paths_checked"] or matched != 24:
        raise ValueError("Changed path comparison coverage")
    states = read_rows(source / "collection/states.jsonl")
    saved = {p.stem: json.loads(p.read_text()) for p in (source / "evaluation").glob("*.json")}
    models = {p: {name: AcceptancePredictor(**m) for name, m in values.items()} for p, values in saved["models"].items()}
    # Reuse executed coefficients. Re-fitting with another SciPy/BLAS version is
    # a separate optimizer reproducibility check, not a report rendering step.
    actual = json.loads(json.dumps(evaluate(states, calibrated=models)))
    if not equivalent(actual, saved):
        raise ValueError("Changed saved prediction evidence")
    summary = saved["summary"]
    if (info["states"], info["candidates"], info["projects"]) != tuple(summary[k] for k in ("states", "candidates", "projects")):
        raise ValueError("Changed diagnostic counts")
    lines = ["# 接受长度预测：条件生存与 Draft 置信度", "",
             "2026-10-19。固定16个已观察请求，三个旧拼接路径共48/48严格匹配原AR token、Target prompt和合法记忆快照；24/24旧matrix路径计数相同。只读诊断未改变生成路径。",
             "", f"去重后{summary['states']}个Draft状态、{summary['candidates']}个shadow候选、{summary['projects']}个项目；每折排除整个当前项目。shadow候选并非线上实际提议，重复候选与截断位置不是独立样本。",
             "", "预测拆成 Draft 前缀生存与边界通过后的历史条件生存，预计接受量为生存曲线之和；候选分别建特征。训练只使用其他项目事后验证标签，不向历史库加入答案。无置信度消融与新模型共用数据和训练规则。",
             "", "## 原14状态、原动作、原长度", "",
             "| 预测器 | MAE（token） | 平均偏差 | Brier |", "|---|---:|---:|---:|"]
    for name in ("old", "no_confidence", "confidence"):
        m = summary["paired"][name]
        lines.append(f"| {name} | {m['error']:.4f} | {m['bias']:+.4f} | {m['brier']:.6f} |")
    gain = 1 - summary["paired"]["confidence"]["error"] / summary["paired"]["old"]["error"]
    interval = summary["project_bootstrap_95"]
    lines += ["", f"新模型MAE相对旧模型变化：降低{gain:.2%}。项目等权MAE改善{summary['project_mean_mae_improvement']:.4f} token，5000次项目bootstrap 95%区间[{interval[0]:.4f}, {interval[1]:.4f}]。这是同批开发项目的不确定性描述，不是独立测试证明。",
              "", "| 请求/位置 | 原动作 | 实际接受 | 旧预测 | 无置信度 | 有置信度 |", "|---|---|---:|---:|---:|---:|"]
    for r in saved["paired"]:
        lines.append(f"| {r['request_id']}/{r['generated']} | {r['route']}/{r['length']} | {r['old']['actual']} | {r['old']['predicted']:.2f} | {r['no_confidence']['predicted']:.2f} | {r['confidence']['predicted']:.2f} |")
    lines += ["", "## 全部状态的诊断指标", "", "这部分覆盖不同于原14帧，仅比较两个新模型，不能用作旧模型的同口径优劣结论。", "",
              "| 范围 | 模型 | MAE | Brier |", "|---|---|---:|---:|"]
    for scope, values in summary["broad"].items():
        for name, m in values.items():
            lines.append(f"| {scope} | {name} | {m['error']:.4f} | {m['brier']:.6f} |")
    lines += ["", "## Edit与TailOT同状态候选比较", "", "下表是离线预测/实际接受，不是新路由运行结果。", "",
              "| 请求/位置 | Edit预测/实际 | TailOT预测/实际 |", "|---|---:|---:|"]
    for r in saved["ordering"]:
        a, b = (r["options"][m] for m in ("Splice-Edit", "Splice-TailOT"))
        lines.append(f"| {r['request_id']}/{r['generated']} | {a['confidence']:.2f}/{a['actual']} | {b['confidence']:.2f}/{b['actual']} |")
    full = [r for r in saved["paired"] if r["old"]["actual"] == 32]
    lines += ["", "## 原因定位与下一步", "",
              "前缀风险信号有效：payment/11的严重高估由32降至5.57（实际3）；shipment/14由30.89降至8.03（实际7），shipment/22由30.56降至7.34（实际6）。这不是单纯减少所有预测值：全部56个Draft状态的MAE也从无置信度5.04降至2.12。",
              "", "两个关键候选排序恢复：payment/15预测TailOT 17.59 > Edit 8.52，实际32 > 7；shipment/22预测Edit 14.43 > TailOT 7.34，实际32 > 6。这里只比较已生成的候选，不能推导调用OT之前已知道其候选，亦未证明在线门控能加速。",
              "", f"剩余问题是长接续被低估：原14动作中{len(full)}个实际接受32的动作，新模型全部预测低于32，平均仅{sum(r['confidence']['predicted'] for r in full)/len(full):.2f}。整体偏差由旧模型+4.18变成−7.50；并非完成了可靠校准。",
              "", "原因的可证实部分是信息范围：Draft置信度最多覆盖原15个位置；长度16–32的历史尾部没有对应Draft概率，只能依赖其他项目条件生存统计。训练仅56状态/69候选，候选之间高度相关。正则、跨项目差异与尾部特征不足可能共同导致保守估计，本轮没有独立消融区分三者的因果贡献，不能断言只要多采样就能解决。",
              "", "建议保留廉价Draft前缀风险与候选独立预测；下一步先用独立项目检验风险排序，暂不让不准的长尾绝对值控制长度。若继续长度校准，应专门补充边界已经通过后的历史长尾成功/失败覆盖，并单独检验条件生存；不得在本批上整体抬高预测或改阈值凑通过。需要在线应用时再计入特征/同步成本、端到端验证轮与后续suffix机会。"]
    lines += ["", "## 预先冻结的门槛", "", "```json", json.dumps(summary["gate"], indent=2), "```", "",
              "未将该预测器接入默认收益路由。即使某些误差降低，也不能直接推导加速：特征计算有在线成本，动态长度会改变后续suffix命中与验证轮数。训练与评估任务全部曾被观察，跨项目排除不能替代独立数据。",
              "", "首次启动漏建模型缓存软链接，在load_engine阶段停止、没有生成。保留pipeline-load-failed.log；补齐链接后以同一冻结源码完成本轮。没有根据本轮结果调整特征、正则、阈值或选择新的实验范围。",
              "", f"冻结执行提交`{manifest['commit']}`；package SHA `{info['source']['sha256']}`。默认suffix、原收益门控实验及独立infra版本保留。重建使用已执行系数，统计浮点复算容差1e-12，token/动作/路径检查仍为精确比较。",
              "", "```bash", "python -m scripts.write_acceptance_report", "```", ""]
    return "\n".join(lines), saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/acceptance")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/ACCEPTANCE_RESULTS.md")
    args = parser.parse_args()
    report, _ = build(args.source)
    args.report.write_text(report, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
