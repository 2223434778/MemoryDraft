"""Build the next-stage report from paired, completed runs, including failures."""
import gzip
import json
import random
import statistics
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
             "结论：D/R 在合成历史任务上跨过 AR 基线，但 D 在公开任务上仅打平；H 尚未证明稳定提高候选质量或带来相对 D/RD 的净收益。R 与 H 已分离检索，不将复制收益归于 H。", "",
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
        lines += ["| 方法 | TTFT 中位数（毫秒） | PyTorch 峰值分配（GiB） |", "|---|---:|---:|"]
        for method in summary:
            group = [r for r in rows if r["method"] == method]
            ttft = statistics.median(r["ttft_seconds"] for r in group if r["ttft_seconds"] is not None) * 1000
            peak = max(max(r["peak_gpu_bytes"].values()) for r in group) / 2**30
            lines.append(f"| {method} | {ttft:.1f} | {peak:.2f} |")
        lines += ["", "显存列是 PyTorch 分配峰值，包含已加载的 T/S，不是驱动报告的全部显存。", ""]
        if stage == "public-test":
            lines += ["公开任务没有对应账户历史，因此该表 R 主要检查当前上下文动态复用的适用边界；它不是带通用静态语料库的官方 REST 复现。", ""]
            lines += ["| 任务 | D_native 加速 | D 候选接受比例 | PLD 加速 | R 加速 |", "|---|---:|---:|---:|---:|"]
            for scenario in sorted({r["scenario"] for r in rows}):
                subset = [r for r in rows if r["scenario"] == scenario]
                ar_time = sum(r["latency_seconds"] for r in subset if r["method"] == "AR")
                speeds = {m: ar_time / sum(r["latency_seconds"] for r in subset if r["method"] == m)
                          for m in ("D_native", "PLD", "R")}
                draft = [r["route_stats"].get("D", {}) for r in subset if r["method"] == "D_native"]
                acceptance = sum(r.get("accepted", 0) for r in draft) / sum(r.get("proposed", 0) for r in draft)
                lines.append(f"| {scenario} | {speeds['D_native']:.3f}× | {acceptance:.2%} | {speeds['PLD']:.3f}× | {speeds['R']:.3f}× |")
            lines += ["", "D 的低接受任务仍需付串行小模型起草成本；该相关性与总体打平一致，但这里没有将 host 计时拆成精确 GPU kernel 归因。", ""]
        if stage == "memory-test":
            lines += ["| 方法 | 接受/提议 token | 接受比例 | 在线检索累计秒数 |", "|---|---:|---:|---:|"]
            for method, route in [("D_native", "D"), ("D", "D"), ("H", "H")]:
                stats = summary[method]["routes"][route]
                retrieval = sum(r["stage_seconds"]["l1_host"] for r in rows if r["method"] == method)
                lines.append(f"| {method} | {stats['accepted']}/{stats['proposed']} | {stats['accepted']/stats['proposed']:.2%} | {retrieval:.3f} |")
            delta_h = summary["H"]["total_seconds"] - summary["D"]["total_seconds"]
            delta_rhd = summary["RHD"]["total_seconds"] - summary["RD"]["total_seconds"]
            rhd_l1 = sum(r["stage_seconds"]["l1_host"] for r in rows if r["method"] == "RHD")
            lines += ["", f"H 相对同指令 D 多用 {delta_h:.3f}s；RHD 相对 RD 多用 {delta_rhd:.3f}s，而 RHD 专用检索累计 {rhd_l1:.3f}s。检索发生在 GPU 生成前，已包含在请求耗时中。这指向检索成本，但不能直接相减当作缓存后的实测加速；路由及实际起草位置也发生了变化。", ""]
            lines += ["| 场景 | D_native 加速 | R 加速 | H 加速 | RHD 加速 |", "|---|---:|---:|---:|---:|"]
            for scenario in sorted({r["scenario"] for r in rows}):
                subset = [r for r in rows if r["scenario"] == scenario]
                ar_time = sum(r["latency_seconds"] for r in subset if r["method"] == "AR")
                ratios = [ar_time / sum(r["latency_seconds"] for r in subset if r["method"] == method)
                          for method in ("D_native", "R", "H", "RHD")]
                lines.append("| " + scenario + " | " + " | ".join(f"{r:.3f}×" for r in ratios) + " |")
            lines += ["", "R 包含静态历史索引与当前已确认上下文的动态 prompt lookup。其总加速不能单独归因于跨会话记忆；PLD 是输入复用控制。D 是带空 reference 的同指令对照，D_native 是实际原生性能基线；RD/RHD 的 D 路径使用原生上下文。", ""]
            lines += ["| 路线 | 静态历史来源接受 token | 动态当前上下文来源接受 token |", "|---|---:|---:|"]
            for method in ("R", "RD", "RHD"):
                events = [e for r in rows if r["method"] == method for e in r["rounds"] if e["route"] == "R"]
                static = sum(e["accepted"] for e in events if e.get("source_id") not in {None, "__confirmed_context__"})
                dynamic = sum(e["accepted"] for e in events if e.get("source_id") == "__confirmed_context__")
                lines.append(f"| {method} | {static} | {dynamic} |")
            lines += ["", "此表不把 Target 的纠正/bonus token 计为检索候选，也不证明静态来源的 token 无法从当前输入找到。", ""]
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
        lines += ["### 追加全前缀诊断", "", "| 条件 | 平均接受长度 | 相对 D_format 增量 | 增量 bootstrap 95% 区间 | 静态 R 弱匹配位置平均增量 |", "|---|---:|---:|---|---:|"]
        for name, value in dense.items():
            low, high = value["delta_bootstrap95_request"]
            lines.append(f"| {name} | {value['mean_accepted_length_per_request']:.3f} | {value['mean_delta_vs_format_per_request']:+.3f} | [{low:+.3f}, {high:+.3f}] | {value['weak_r_mean_delta']:+.3f} |")
        lines += ["", f"覆盖 {next(iter(dense.values()))['positions']} 个位置；新生成 AR 轨迹与计时实验的 AR token 逐一核对一致。更多位置不等于更多独立项目，仍是同一批合成任务。", ""]
        lines += ["| 条件 | 首位命中 P(L≥1) | 两位连续命中 P(L≥2) | 比 D_format 更差的位置比例 |", "|---|---:|---:|---:|"]
        for name, value in dense.items():
            survival = value["survival_by_position"]
            lines.append(f"| {name} | {survival['1']:.2%} | {survival['2']:.2%} | {value['worse_position_fraction']:.2%} |")
        lines += ["", "命中列按具备对应有效预算的位置计数，第二位列排除只剩一个参考 token 的末端位置；与上表请求等权的均值不使用相同权重。", ""]
        lines += ["全前缀结果比间隔采样的增量小得多，不能选用 +0.375 的稀疏结果代替 +0.045 的全前缀发现。H 的主要正增益出现在两个代码修改请求；复制请求没有增益，新任务有负增益。合法人工预选片段与自动检索相近，故本组证据不支持把主要问题全部归咎于检索召回。R 弱匹配位置增益接近零，尚未证明 H 的互补贡献。",
                  "在线 H/D 会在不同位置发起下一轮，候选块的起点由前面的接受量决定。因此同一 AR 前缀集合上的小幅质量增益，可以与在线接受比例下降并存；二者不能相互替代。", ""]
    cache_path = base / "embedding-cache-check" / f"summary-k{block}.json"
    if cache_path.exists():
        cache_provenance = json.loads((cache_path.parent / "provenance.json").read_text())
        if cache_provenance["status"] != "complete":
            raise ValueError("Cache ablation incomplete")
        rows = load_rows("embedding-cache-check", block)
        cache_summary = json.loads(cache_path.read_text())
        lines += ["## 追加：历史向量缓存消融", "",
                  "这是看过主测试结果后的工程诊断，沿用同六个项目，不是新的独立保留集。H/RHD 使用独立、初始为空的 4096 片段 LRU，复用相同 CPU encoder 权重；每次查询仍编码，缓存仅复用逐段文本向量。没有预先离线建库，第一次编码的开销计入在线延迟。排序候选仍仅来自当前合法用户/时间快照，缓存不参与 BM25 文档统计。",
                  "", "| 方法 | 总时间（秒） | 相对本次 AR | 请求 bootstrap 95% 区间 | 在线检索累计秒数 |", "|---|---:|---:|---|---:|"]
        for method, value in cache_summary.items():
            low, high = speed_interval(rows, method)
            retrieval = sum(r["stage_seconds"]["l1_host"] for r in rows if r["method"] == method)
            lines.append(f"| {method} | {value['total_seconds']:.3f} | {value['speedup_vs_ar']:.3f}× | [{low:.3f}, {high:.3f}] | {retrieval:.3f} |")
        lines += ["", "| 对照 | 首次访问耗时比（未缓存/缓存） | 后续两次访问耗时比 | 全部访问耗时比 |", "|---|---:|---:|---:|"]
        for method in ("H", "RHD"):
            ratios = []
            for repeats in ({0}, {1, 2}, {0, 1, 2}):
                times = {m: sum(r["latency_seconds"] for r in rows if r["method"] == m and r["repeat"] in repeats)
                         for m in (method, method + "_cached")}
                ratios.append(times[method] / times[method + "_cached"])
            lines.append(f"| {method} | " + " | ".join(f"{r:.3f}×" for r in ratios) + " |")
        lines += ["", "大于 1 表示缓存版更快；相同文本可在合法快照之间命中已有的纯文本向量，首次访问不保证每个片段都未命中。未缓存和缓存方法在同一个 replay 中交错运行，不能拿跨批次 AR 时间直接相减。", ""]
        for name, stats in cache_provenance["cache_stats"].items():
            lines.append(f"- {name}: 命中 {stats['hits']}、新编码 {stats['misses']}、结束时 {stats['chunks']} 个向量，逻辑向量 payload 为 {stats['vector_bytes']} 字节（不是内存峰值；不含 NumPy 底层批次、Python 键/对象和 encoder 权重）。")
        lines += ["", f"此消融另有 {len(rows)} 次计时方法运行，逐 token / Target prompt hash / snapshot 配对检查通过。缓存优化不直接证明 H 提议质量改善；仍须与同权重 D 及 RD 比较。", ""]
    diagnostic = json.loads((base / "bf16-diagnostic.json").read_text())
    lines += ["## BF16 首个分歧的定点诊断", "",
              f"在 `{diagnostic['request_id']}` 的 index={diagnostic['position']}，实际 Target forward 的已确认输入前缀相同，AR query 长度为 {diagnostic['AR']['state']['query']}，块验证为 {diagnostic['D_adapter']['state']['query']}；AR 前两名 logits 差 {diagnostic['AR']['top2_gap']:.2f}，块验证差 {diagnostic['D_adapter']['top2_gap']:.2f}，整向量最大绝对差 {diagnostic['max_abs_logit_difference']:.2f}。",
              "AR 的 token 13814/20757 同为 33.75，块验证中 token 20757 为 34.00、13814 为 33.75，argmax 翻转；重新生成确实复现原分歧。这支持 BF16 在不同计算形状下的数值差异解释，不等于证明所有潜在缓存错误均不存在。诊断不计入性能；没有放宽匹配或在接受时偏向 AR token。", ""]
    lines += [
              "## 本轮发现并修复的问题", "",
              "1. Transformers 4.52.2 会回填新版检查点的默认生成参数；Qwen3 默认 do_sample=True。仅构造 GenerationConfig(do_sample=False) 不足以强制 greedy。现已在 Target 和 S 入口显式关闭 use_model_defaults，并对 Target 显式设置 do_sample=False。回归测试禁止任何 multinomial 调用。旧错误运行保存在 results/qwen3-inherited-defaults，不计有效结果。",
              "2. 上游 prompt lookup 在末尾可超出输出预算；旧模型对照曾输出 54 token 而 AR 为 48。适配器现在保留 Target 的纠正/bonus token 空间，并有回归测试。",
              "3. R 已与 H 的语义 top-k 解耦；使用 pydivsufsort 0.0.19 的后缀数组与搜索，固定宽度 uint32 编码、token 对齐及记录边界检查，避免 REST 原版 u16 文件格式截断。H 混合检索复用 rank-bm25 0.2.2 与已有 MiniLM 的排序融合。", "",
              "## 数据、来源与限制", "",
              "公开任务来自固定 Spec-Bench 提交，选择规则与原始 SHA-256 见 data/research/specbench-source.json；历史任务是明确标注的合成项目，开发/测试项目 ID 分开。它们不能代表真实长期用户分布。",
              "合成项目共享任务模板，所列请求 bootstrap 区间是探索性统计，不能解释为对真实用户分布的覆盖保证。",
              "权重从 Qwen 官方 ModelScope 镜像获取，并逐文件核对固定 HF revision 的 SHA-256；provenance 中保留源码清单、输入哈希、tokenizer fingerprint、实际配置和权重验证记录。",
              "修复采样回填后 BF16/eager 仍有严格 token 分歧；失败现场单列保留，不与 FP32 合并计算。已测 FP32 一致性不保证所有输入/内核的位级一致。",
              "本轮仍是独立小模型与线性候选原型；成熟 EAGLE B6/B7、完整 REST 树验证、真实长期历史、CPU 峰值内存与服务并发尚未完成。RHD 的 AR 回退或复制收益不能替代 H 的独立贡献。", ""]
    lines += ["## 文献对应与下一步判断", "",
              "[标准投机解码](https://proceedings.mlr.press/v202/leviathan23a.html)给出分布保持的验证框架，并不保证任意模型比和任务都加速。本轮原生 D 的任务差异要求优先补上匹配的成熟 Draft B6/B7，再谈普遍性能目标。",
              "[REST](https://aclanthology.org/2024.naacl-long.88/) 与 [SAM-Decoding](https://aclanthology.org/2025.acl-long.595/)支持从 token 接续检索直接构造候选；本轮沿此方向分离 R 的完整合法历史索引，未冒充其官方树验证复现。无历史时 R 退化到动态输入复用，且比 PLD 慢；当前 R 逐轮 CPU 转移、后缀查询和路由是应检查的开销，而不是已被计时隔离证明的唯一原因。",
              "[RAPID](https://proceedings.mlr.press/v267/chen25s.html)提示可以用检索内容帮助 Draft，但它的上下文压缩及 Target 条件与本原型不同，不能直接转移论文收益。这里 H 只是追加历史，人工相关历史也只有微弱总体增量；继续更换 embedding 未必解决接续对齐。更有依据的下一项机制对照是：保持固定 Target 与 S，使用请求发生前生成、与固定 Target 表达相符的旧回答，对比现有人工历史；仍禁止把当前答案写入过去。随后才考虑有限的 student/teacher 对齐。",
              "H 的实际收益必须同时满足候选更准和额外成本可覆盖。历史向量缓存只处理其中的系统成本，不替代这个机制检验；所有后续选项均须保留当前负结果及独立项目测试。", ""]
    (root / "docs/EXPERIMENTS_V2.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote docs/EXPERIMENTS_V2.md")
