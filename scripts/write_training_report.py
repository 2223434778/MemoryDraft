"""Rebuild H distillation and DFlash findings, including incomplete/failed runs."""
import gzip
import json
import statistics
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from scripts.write_h_causal_report import relative_time_interval
from scripts.write_research_report import speed_interval
from memorydraft.evaluation.replay import summarize
from memorydraft.memory.store import MemoryEvent, MemoryStore


def rows(path):
    path = Path(path)
    raw = path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def paired_table(data, summary):
    lines = ["| 方法 | 次数 | 总秒数 | AR/方法 | 请求 bootstrap 95% 区间 |", "|---|---:|---:|---:|---|"]
    for method, value in summary.items():
        low, high = speed_interval(data, method)
        lines.append(f"| {method} | {value['requests']} | {value['total_seconds']:.3f} | {value['speedup_vs_ar']:.3f}× | [{low:.3f}, {high:.3f}] |")
    return lines


def validate_training_rows(data, labels):
    reference = {row["request_id"]: row for row in labels}
    keys = [(row["request_id"], row["repeat"], row["method"]) for row in data]
    methods = {row["method"] for row in data}
    expected = {(rid, repeat, method) for rid in reference for repeat in range(3) for method in methods}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate final training replay")
    for row in data:
        ref = reference[row["request_id"]]
        if not row["correct"] or row["token_ids"] != ref["token_ids"] or row["target_prompt_sha256"] != ref["target_prompt_sha256"]:
            raise ValueError("Replay differs from frozen held-out Target labels")


def main():
    base = root / "results/h-training"
    provenance = json.loads((base / "provenance.json").read_text(encoding="utf-8"))
    lines = ["# H 小规模蒸馏与 DFlash 基线", "",
             "2026-10-17。协议：[H_TRAINING_PROTOCOL](H_TRAINING_PROTOCOL.md)。该试验固定 Target，复用原验证器、PEFT 和官方 DFlash；未将模型训练收益直接归为历史收益。", "",
             "## H 试验状态", "", f"状态：`{provenance['status']}`。"]
    if provenance["status"] != "complete":
        lines += ["", f"保存的错误：`{provenance.get('error_type', '')}: {provenance.get('error', '')}`。不从不完整记录报告正式加速。"]
    else:
        if not provenance["training_complete_before_test_labels"]:
            raise ValueError("Test labels were available before training freeze")
        metrics = json.loads((base / "training-metrics.json").read_text())
        selection = json.loads((base / "dev-selection.json").read_text())
        quality = json.loads((base / "quality-summary.json").read_text())
        data = rows(base / "test-replay.jsonl")
        labels = rows(base / "test-labels.jsonl")
        validate_training_rows(data, labels)
        summary = json.loads((base / "test-summary.json").read_text())
        if summary != summarize(data):
            raise ValueError("Saved training summary differs from paired replay")
        requests = {split: rows(base / f"{split}-requests.jsonl") for split in ("train", "dev", "test")}
        users = {split: {row["user_id"] for row in values} for split, values in requests.items()}
        if users["train"] & users["dev"] or users["test"] & (users["train"] | users["dev"]):
            raise ValueError("Training, development and test users overlap")
        with MemoryStore(":memory:") as memory:
            memory.append([MemoryEvent(**row) for row in rows(base / "history.jsonl")])
            snapshots = {request["id"]: memory.snapshot(request["user_id"], request["order"])
                         for request in requests["test"]}
            if any(row["snapshot_id"] != snapshots[row["request_id"]].snapshot_id for row in data):
                raise ValueError("Replay snapshot differs from published causal history")
        past = rows(base / "past-generations.jsonl")
        train_labels = rows(base / "train-labels.jsonl")
        dev_labels = rows(base / "dev-labels.jsonl")
        lines += ["", "24 训练项目、8 开发项目、8 测试项目，四类结构化任务，用户/项目 ID 分离。各家族内部仍共享模板；不是自然长期对话或未见任务家族的泛化测试。旧六项目未用于训练/选择。",
                  "先冻结 40 条 Target 过去回答，历史 order=1、当前请求 order=10。训练用 24 个当前 Target 输出，开发用 8 个输出；测试当前标签在权重与开发选择冻结后生成。训练标签不写入历史。",
                  f"LoRA rank8/alpha16/q_proj+v_proj，四轮、lr=1e-4、batch1/累积4。每个训练项目含相关、空及无关历史三条件，共72样本。可训练参数 **{metrics['trainable_parameters']:,}**，训练墙钟 **{metrics['seconds']:.3f}s**。这是响应 token 硬标签蒸馏，不是完整 DistillSpec on-policy/KL 复现。", "",
                  "| 轮次 | 训练 loss（样本均值） | 开发响应 loss（token 加权） |", "|---|---:|---:|"]
        for epoch in metrics["epochs"]:
            lines.append(f"| {epoch['epoch']} | {epoch['train_loss']:.4f} | {epoch['dev_response_loss']:.4f} |")
        lines += ["", f"历史生成累计 {sum(row['seconds'] for row in past):.3f}s，训练/开发标签生成 {sum(row['seconds'] for row in train_labels+dev_labels):.3f}s，测试参考生成 {sum(row['seconds'] for row in labels):.3f}s，均为离线准备/评分成本。历史达到192 token预算 {sum(row['hit_budget'] for row in past)}/40 条；当前测试达到128 token预算 {sum(row['hit_budget'] for row in labels)}/8 条。参考轨迹按固定预算评估，不代表所有回答都完整结束。初始资产任务中断后废弃的未保存标注计算不包含于上述已保存计数。"]
        lines += ["", f"按最低开发响应 loss 保存 adapter，再合并到独立 S。原始/训练后各自的 D/H 共用同一模型实例；Target 未更新。开发选择块长：`{selection['blocks']}`。独立 H/D 可用各自块长，但 RHD off/on 使用同一块长。", "",
                  "## 同前缀历史增量", "",
                  "| 对照 | 请求平均连续接受长度差 | 请求 bootstrap 95% 区间 | 静态 R 弱匹配位置差 |",
                  "|---|---:|---|---:|"]
        for name, value in quality.items():
            low, high = value["bootstrap95_request"]
            weak = value["weak_r_mean_delta"]
            text = f"{weak:+.3f}" if weak is not None else "无弱匹配位置"
            lines.append(f"| {name} | {value['mean_delta']:+.3f} | [{low:+.3f}, {high:+.3f}] | {text} |")
        prefix = rows(base / "test-prefixes.jsonl")
        lines += ["", "| 条件 | 项目等权平均接受长度 | 首 token 命中（位置等权） |", "|---|---:|---:|"]
        request_ids = {row["request_id"] for row in prefix}
        for condition in prefix[0]["lengths"]:
            means = [statistics.mean(row["lengths"][condition] for row in prefix if row["request_id"] == rid) for rid in request_ids]
            first = sum(row["lengths"][condition] > 0 for row in prefix) / len(prefix)
            lines.append(f"| {condition} | {statistics.mean(means):.3f} | {first:.2%} |")
        lines += ["", f"全部 {len(prefix)} 个已确认前缀均评分，同一 Target 参考、同一候选长度；H 使用词法 top-1、128-token 历史，D_format 对应空参考指令。无关控制使用固定短片段，未配平历史 token 数，不能单独隔离相关性和长度。R 弱匹配为该冻结历史静态索引后缀长度<3，不包括动态 prompt lookup。区间按项目重采样，重复与 token 不作独立样本。", "",
                  "## 完整在线成本", ""]
        lines += paired_table(data, summary)
        lines += ["", "BASE 为未训练 Draft，TRAIN 为同一个 LoRA 合并后的 Draft。D 是官方原生 D，H 是历史条件化独立起草；off/on 是同一 RHD 路由的 H 开关。原始/训练后 S 同时驻留仅用于交错公平对照，模型选择/加载不属于每个请求的在线模型切换。", "",
                  "| 直接比较 | baseline/方法 | 请求 bootstrap 95% 区间 |", "|---|---:|---|"]
        for baseline, method in (("BASE_D", "BASE_H"), ("BASE_D", "TRAIN_D"), ("TRAIN_D", "TRAIN_H"), ("TRAIN_off", "TRAIN_on")):
            low, high = relative_time_interval(data, baseline, method)
            ratio = summary[baseline]["total_seconds"] / summary[method]["total_seconds"]
            lines.append(f"| {baseline}/{method} | {ratio:.3f}× | [{low:.3f}, {high:.3f}] |")
        lines += ["", "| 方法 | R/H/D/AR 轮数 | 验证轮数 | H+D 起草轮数 | H 接受/提议 |", "|---|---|---:|---:|---:|"]
        for method, value in summary.items():
            if method == "AR":
                continue
            routes = value["routes"]
            counts = "/".join(str(routes[key]["rounds"]) for key in ("R", "H", "D", "AR"))
            n = sum(route["rounds"] for route in routes.values())
            s = routes["H"]["rounds"] + routes["D"]["rounds"]
            lines.append(f"| {method} | {counts} | {n} | {s} | {routes['H']['accepted']}/{routes['H']['proposed']} |")
        gate = selection["gate"]
        lines += ["", f"开发门控选择的任务类：`{gate}`。选择同时要求该类开发 H-D_format 质量正增量和同路由 on 比 off 更快。"]
        dev_prefix = rows(base / "dev-prefixes.jsonl")
        dev_router = rows(base / "dev-router.jsonl")
        lines += ["", "| 开发任务类 | 稀疏位置平均 H-D_format | TRAIN_off/TRAIN_on |", "|---|---:|---:|"]
        for family in sorted({row["scenario"] for row in dev_prefix}):
            group = [row for row in dev_prefix if row["scenario"] == family]
            delta = statistics.mean(row["lengths"]["trained_H"]-row["lengths"]["trained_D_format"] for row in group)
            times = {method: sum(row["latency_seconds"] for row in dev_router if row["scenario"] == family and row["method"] == method)
                     for method in ("TRAIN_off", "TRAIN_on")}
            lines.append(f"| {family} | {delta:+.3f} | {times['TRAIN_off']/times['TRAIN_on']:.3f}× |")
        lines += ["", "开发质量按预先固定 stride=4 评分，可能遗漏局部机会，门控只是一项保守小试验；最终测试使用全部前缀，不将稀疏结果替代全前缀结论。"]
        if not gate:
            lines += ["没有任务类通过；因此未重复测量等价于 off 的 gated 方法，明确记为门控关闭所有 H，不算 H 系统成功。"]
        lines += ["", "在线快照、词法检索、完整 R 索引、历史 prefill、起草、验证、追赶和解码均计入；模型加载、训练、过去资产构造与独立 AR 参考另行记录。峰值包含两套 S 的驻留，不能当作单套部署显存。", "",
                  "| 测试任务类 | TRAIN_off/TRAIN_on | TRAIN_D/TRAIN_H |", "|---|---:|---:|"]
        for family in sorted({row["scenario"] for row in data}):
            totals = {method: sum(row["latency_seconds"] for row in data if row["method"] == method and row["scenario"] == family)
                      for method in summary}
            lines.append(f"| {family} | {totals['TRAIN_off']/totals['TRAIN_on']:.3f}× | {totals['TRAIN_D']/totals['TRAIN_H']:.3f}× |")
        lines += ["", f"{len(data)} 次最终配对计时全部与冻结 Target 一致；训练后 D 的改善属于蒸馏收益，H 必须另看同权重质量和 off/on 增量。", "",
                  f"训练运行源码清单 SHA-256：`{provenance['source_manifest']['source_sha256']}`。adapter 文件 SHA 保存在 provenance；权重不入 Git。"]
        ratio = summary["TRAIN_D"]["total_seconds"] / summary["TRAIN_H"]["total_seconds"]
        router = summary["TRAIN_off"]["total_seconds"] / summary["TRAIN_on"]["total_seconds"]
        lines += ["", "## 本轮解释", "",
                  f"普通未训练 S 的 H 在这组新任务、短词法历史和k4组合上有正向候选增量；蒸馏后同权重 H−D_format 为 {quality['trained_H_minus_D']['mean_delta']:+.3f}，区间跨零。训练后独立 D/H 时间比为 {ratio:.3f}，同路由 off/on 为 {router:.3f}，两者区间均跨1，未证明训练后 H 的稳定独立收益。",
                  "本轮不将结果概括为历史永远无效：未训练 H 的质量和时间均有正信号；但对齐训练使空历史 D 追上 H，新增历史的价值需要在强基线上重新证明。旧六项目到新数据同时改变了任务、历史选择和候选长度，不能单独归因于某一个改动。"]
    dflash = root / "results/dflash-baseline"
    if (dflash / "provenance.json").exists():
        p = json.loads((dflash / "provenance.json").read_text())
        lines += ["", "## 独立 DFlash 基线", "",
                  f"状态：`{p['status']}`。Target 为同一固定 Qwen3-8B，独立 Transformers {p['transformers']}、torch {p['torch']}，FP32/eager；使用已审计官方模型快照的 spec_generate。权重、代码、许可和 SHA 在 dflash manifest/provenance。"]
        if p["status"] == "complete":
            ds = json.loads((dflash / "summary.json").read_text())
            dr = rows(dflash / "replay.jsonl")
            low, high = speed_interval(dr, "DFlash")
            lines += [f"六个旧合成请求加六个公开请求、三重复，共 {ds['rows']} 次；所有 token 与同环境 AR 及旧 AR 参考一致。AR={ds['times']['AR']:.3f}s，DFlash={ds['times']['DFlash']:.3f}s，加速 **{ds['speedup']:.3f}×**，请求 bootstrap 95% **[{low:.3f},{high:.3f}]**。"]
            lines += ["", "| 子集 | AR 秒 | DFlash 秒 | AR/DFlash | 请求 bootstrap 95% 区间 |", "|---|---:|---:|---:|---|"]
            for label, synthetic in (("六合成请求", True), ("六公开请求", False)):
                group = [row for row in dr if row["scenario"].startswith("synthetic") == synthetic]
                a = sum(row["latency_seconds"] for row in group if row["method"] == "AR")
                b = sum(row["latency_seconds"] for row in group if row["method"] == "DFlash")
                low, high = speed_interval(group, "DFlash")
                lines.append(f"| {label} | {a:.3f} | {b:.3f} | {a/b:.3f}× | [{low:.3f},{high:.3f}] |")
            lines += ["", "官方 spec_generate 返回完整输出，本轮未额外插入计时回调测量 TTFT。独立环境的 AR 自身也需与旧 AR 对齐，不能把后端行为变化当作起草收益。"]
        else:
            lines += [f"错误：`{p.get('error_type', '')}: {p.get('error', '')}`。保存失败现场，不将未通过或不完整运行汇总为无损加速。"]
    lines += ["", "## 复现与边界", "",
              "命令与环境见 PROGRESS；`python scripts/write_training_report.py` 可从公开压缩记录离线重建。DFlash 是独立成熟起草对照，尚未集成 R 或记忆，不属于 H 收益。DFlash 与本轮训练后 H 使用不同测试集合，不能直接用两表加速比相除判断优劣。",
              "数据规模小且共享模板，只用于一次预声明机制小试验。候选质量、训练收益、H 增量和后端变化分开解释；不放宽 Target 接受规则，不从测试反馈追加训练或阈值选择。", ""]
    log = base / "tests-final.log"
    if log.exists():
        lines += ["## 服务器最终回归", "", "```text", log.read_text(encoding="utf-8").strip(), "```", ""]
    (root / "docs/H_TRAINING_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote docs/H_TRAINING_RESULTS.md")


if __name__ == "__main__":
    main()
