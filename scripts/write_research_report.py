"""Build the next-stage report from paired, completed runs, including failures."""
import gzip
import json
import random
from pathlib import Path


root = Path(__file__).resolve().parents[1]
base = root / "results/qwen3-pilot"


def load_rows(stage, block):
    path = base / stage / f"replay-k{block}.jsonl"
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        with gzip.open(str(path) + ".gz", "rt", encoding="utf-8") as stream:
            text = stream.read()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def speed_interval(rows, method):
    ar = {(r["request_id"], r["repeat"]): r for r in rows if r["method"] == "AR"}
    selected = [r for r in rows if r["method"] == method]
    keys = [(r["request_id"], r["repeat"]) for r in selected]
    if len(keys) != len(set(keys)) or set(keys) != set(ar):
        raise ValueError("Incomplete paired stage")
    groups = {}
    for row in selected:
        ref = ar[(row["request_id"], row["repeat"])]
        if not row["correct"] or row["token_ids"] != ref["token_ids"]:
            raise ValueError("Cannot report a mismatched run as speedup")
        if row.get("target_prompt_sha256") != ref.get("target_prompt_sha256"):
            raise ValueError("Target prompt changed")
        if row.get("snapshot_id") != ref.get("snapshot_id"):
            raise ValueError("History snapshot changed")
        pair = groups.setdefault(row["request_id"], [0.0, 0.0])
        pair[0] += ref["latency_seconds"]
        pair[1] += row["latency_seconds"]
    values = list(groups.values())
    rng = random.Random(20261016)
    estimates = []
    for _ in range(2000):
        sample = rng.choices(values, k=len(values))
        estimates.append(sum(x[0] for x in sample) / sum(x[1] for x in sample))
    estimates.sort()
    return estimates[49], estimates[1949]


if __name__ == "__main__":
    selection = json.loads((base / "selection.json").read_text())
    block = selection["block"]
    lines = ["# 第二轮 GPU 实验：D/R 基线与同前缀 H 诊断", "",
             "本报告由 `scripts/write_research_report.py` 从完整配对结果生成。单卡 A6000 GPU0，Qwen3-8B Target、Qwen3-0.6B 独立 S，H/D 共用权重、隔离 KV。",
             f"固定配置：`{selection['profile']}`，候选块长 **{block}**，测试输出上限 128 token。Target 使用相同非思考模板和严格 greedy 验证。", "",
             "## 开发集选择", "", "| 块长 | 原生 D 相对 AR |", "|---|---:|"]
    for speed, k in sorted(selection["dev_choices"], key=lambda x: x[1]):
        lines.append(f"| {k} | {speed:.3f}× |")
    lines += ["", "选择仅使用开发请求；测试配置冻结后未按测试结果调整。预热、权重/encoder 加载与独立 AR 评分参考不计入请求延迟；请求快照、tokenization、检索、索引构建、起草、验证、缓存维护和输出解码均计入。AR 参与随机运行顺序。", ""]
    total = 0
    for stage, title in [("memory-test", "合成历史测试项目"), ("public-test", "公开 Spec-Bench 测试切分")]:
        provenance = json.loads((base / stage / "provenance.json").read_text())
        if provenance["status"] != "complete":
            raise ValueError("Stage not complete: " + stage)
        rows = load_rows(stage, block)
        total += len(rows)
        summary = json.loads((base / stage / f"summary-k{block}.json").read_text())
        lines += [f"## {title}", "", "| 方法 | 配对运行数 | 总时间（秒） | AR/方法 | 请求成组 bootstrap 95% 区间 |", "|---|---:|---:|---:|---|"]
        for method, value in summary.items():
            low, high = speed_interval(rows, method)
            lines.append(f"| {method} | {value['requests']} | {value['total_seconds']:.3f} | {value['speedup_vs_ar']:.3f}× | [{low:.3f}, {high:.3f}] |")
        lines += ["", "区间按请求整体重采样，保留其三次重复；不把重复运行或 token 当作独立用户样本。大于 1 表示快于 AR。", ""]
        if stage == "public-test":
            lines += ["公开任务没有对应账户历史，因此该表 R 主要检查当前上下文动态复用的适用边界；它不是带通用静态语料库的官方 REST 复现。", ""]
        if stage == "memory-test":
            lines += ["| 场景 | D_native 加速 | R 加速 | H 加速 | RHD 加速 |", "|---|---:|---:|---:|---:|"]
            for scenario in sorted({r["scenario"] for r in rows}):
                subset = [r for r in rows if r["scenario"] == scenario]
                ar_time = sum(r["latency_seconds"] for r in subset if r["method"] == "AR")
                ratios = [ar_time / sum(r["latency_seconds"] for r in subset if r["method"] == method)
                          for method in ("D_native", "R", "H", "RHD")]
                lines.append("| " + scenario + " | " + " | ".join(f"{r:.3f}×" for r in ratios) + " |")
            lines += ["", "R 包含静态历史索引与当前已确认上下文的动态 prompt lookup。其总加速不能单独归因于跨会话记忆；PLD 是输入复用控制。D 是带空 reference 的同指令对照，D_native 是实际原生性能基线；RD/RHD 的 D 路径使用原生上下文。", ""]
    lines += [f"本轮以上两个测试切分共 **{total} 次计时方法运行**，不是 {total} 个独立自然请求。开发校准、失败运行与同前缀诊断不计入此数。", "",
              "## H：相同前缀下的候选质量", ""]
    prefix = json.loads((base / "prefix-test" / f"summary-k{block}.json").read_text())
    lines += ["| 条件 | 请求平均连续接受长度 | 相对 D_format 增量 | 增量请求 bootstrap 95% 区间 | 静态 R 弱匹配位置的平均增量 |", "|---|---:|---:|---|---:|"]
    for name, value in prefix.items():
        low, high = value["delta_bootstrap95_request"]
        weak = value["weak_r_mean_delta"]
        lines.append(f"| {name} | {value['mean_accepted_length_per_request']:.3f} | {value['mean_delta_vs_format_per_request']:+.3f} | [{low:+.3f}, {high:+.3f}] | {weak:+.3f} |")
    lines += ["", "无关历史直接按 H_dense 的 payload token 数匹配注入，不经过会将其过滤为空的语义阈值；H_empty 必须与 D_format 提议完全一致。H_selected 的记录 ID 在 Target 参考生成前确定，强制取用合法旧记录，不使用当前测试答案。",
              "静态 R 弱匹配定义为后缀匹配不足 3 token；此分层不覆盖动态 prompt lookup。总体接受长度先对各请求内的位置取均值，再对请求等权平均；弱匹配列按该子集位置平均，两者权重不同。离线命中不等于在线加速。", "",
              "固定间隔采样是预先规定的主诊断；为解释其与在线接受率的差异，另对相同冻结请求做全前缀探索性检查。它不改变配置、不重新选择测试请求，也不作为新的性能测试。", ""]
    dense_path = base / "prefix-dense-check" / f"summary-k{block}.json"
    if dense_path.exists():
        dense = json.loads(dense_path.read_text())
        lines += ["### 追加全前缀诊断", "", "| 条件 | 平均接受长度 | 相对 D_format 增量 | 静态 R 弱匹配位置平均增量 |", "|---|---:|---:|---:|"]
        for name, value in dense.items():
            lines.append(f"| {name} | {value['mean_accepted_length_per_request']:.3f} | {value['mean_delta_vs_format_per_request']:+.3f} | {value['weak_r_mean_delta']:+.3f} |")
        lines += ["", f"覆盖 {next(iter(dense.values()))['positions']} 个位置；新生成 AR 轨迹与计时实验的 AR token 逐一核对一致。", ""]
    lines += [
              "## 本轮发现并修复的问题", "",
              "1. Transformers 4.52.2 会回填新版检查点的默认生成参数；Qwen3 默认 do_sample=True。仅构造 GenerationConfig(do_sample=False) 不足以强制 greedy。现已在 Target 和 S 入口显式关闭 use_model_defaults，并对 Target 显式设置 do_sample=False。回归测试禁止任何 multinomial 调用。旧错误运行保存在 results/qwen3-inherited-defaults，不计有效结果。",
              "2. 上游 prompt lookup 在末尾可超出输出预算；旧模型对照曾输出 54 token 而 AR 为 48。适配器现在保留 Target 的纠正/bonus token 空间，并有回归测试。",
              "3. R 已与 H 的语义 top-k 解耦；使用 pydivsufsort 0.0.19 的后缀数组与搜索，固定宽度 uint32 编码、token 对齐及记录边界检查，避免 REST 原版 u16 文件格式截断。H 混合检索复用 rank-bm25 0.2.2 与已有 MiniLM 的排序融合。", "",
              "## 数据、来源与限制", "",
              "公开任务来自固定 Spec-Bench 提交，选择规则与原始 SHA-256 见 data/research/specbench-source.json；历史任务是明确标注的合成项目，开发/测试项目 ID 分开。它们不能代表真实长期用户分布。",
              "权重从 Qwen 官方 ModelScope 镜像获取，并逐文件核对固定 HF revision 的 SHA-256；provenance 中保留源码清单、输入哈希、tokenizer fingerprint、实际配置和权重验证记录。",
              "修复采样回填后 BF16/eager 仍有严格 token 分歧；失败现场单列保留，不与 FP32 合并计算。已测 FP32 一致性不保证所有输入/内核的位级一致。",
              "本轮仍是独立小模型与线性候选原型；成熟 EAGLE B6/B7、完整 REST 树验证、真实长期历史、CPU 峰值内存与服务并发尚未完成。RHD 的 AR 回退或复制收益不能替代 H 的独立贡献。", ""]
    (root / "docs/EXPERIMENTS_V2.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote docs/EXPERIMENTS_V2.md")
