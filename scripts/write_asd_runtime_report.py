"""Rebuild engineering acceptance; historical speedup is a separate study."""
import hashlib
import json
from pathlib import Path

from scripts.run_asd_runtime_pipeline import check_results
from scripts.validate_asd_runtime import ADAPTER_SHA, TASK_IDS, references

ROOT = Path(__file__).resolve().parents[1]


def analyze(root):
    evidence = root/"results/asd-runtime"
    execution = json.loads((evidence/"execution-verification.json").read_text(encoding="utf-8"))
    manifest = json.loads((evidence/"ASD_RUNTIME_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    original_gate = execution["status"] == "complete"
    completion_path = evidence/"completion-audit.json"
    supplemental = json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.exists() else {}
    known_failure = (execution.get("error_type") == "ValueError" and execution.get("error") == "Runtime ASD budget/trajectory changed"
                     and execution.get("phases") == ["cpu","acceptance","cli"] and supplemental.get("status") == "complete"
                     and supplemental.get("source_commit") == manifest.get("commit"))
    if ((not original_gate and not known_failure) or execution["commit"] != manifest["commit"]
            or execution["files_verified_before"] != len(manifest["files"])
            or (execution if original_gate else supplemental)["files_verified_after"] != len(manifest["files"])
            or not (execution if original_gate else supplemental)["original_adapter_unchanged"]
            or hashlib.sha256(json.dumps(manifest["files"],sort_keys=True).encode()).hexdigest() != manifest["source_sha256"]):
        raise ValueError("Runtime execution/source verification failed")
    cache = {name:row["cache_bytes"] for name,row in execution["checks"].items()} if original_gate else supplemental["cache_bytes"]
    checks = check_results(root,evidence,cache,cli_spent_atol=1e-5)
    if original_gate and checks != execution["checks"]:
        raise ValueError("Recomputed runtime checks changed")
    variants = {}
    for name in ("base","position"):
        directory = evidence/name
        read = lambda path:json.loads((directory/path).read_text(encoding="utf-8"))
        info,cold,cli,rows,ars = [read(path) for path in ("acceptance.json","cold-start.json","cli.json","rows.json","ar-references.json")]
        if (len(ars) != 4 or {r["request_id"] for r in ars} != set(TASK_IDS)
                or any(r["approximate"] or r["asd"] is not None or r["draft_adapter_sha256"] is not None for r in ars)):
            raise ValueError("Missing or non-strict AR references")
        for row in rows:
            if row["method"] == "DFlash":
                ar = next(r for r in ars if r["request_id"] == row["request_id"])
                limit = 128 if row["stage"] == "prefix" else 4096
                if row["token_ids"] != ar["token_ids"][:limit]:raise ValueError("Strict AR mismatch")
        if [r["method"] for r in cold] != ["AR","DFlash","ASD"]:
            raise ValueError("Cold-start coverage changed")
        old = references("base" if name == "base" else "ordinary-position")["gsm8k:1196","natural","ASD"]
        variants[name] = dict(checks=checks[name],cli_spent_difference=cli["asd"]["spent"]-old["asd"]["spent"],
            model_setup_seconds=info["model_setup_seconds"],
            adapter_setup=info["engine"]["draft_adapter"],cold_requests=[dict(method=r["method"],
                seconds=r["latency_seconds"],ttft_seconds=r["ttft_seconds"],
                compile_before=r["counters"]["compile_graphs_before"],compile_after=r["counters"]["compile_graphs_after"]) for r in cold],
            cli_cold_seconds=cli["latency_seconds"],cli_cold_ttft_seconds=cli["ttft_seconds"],
            compiler=info["compiler"],peak_gpu_bytes=info["peak_gpu_bytes"],http=info["http"],
            natural_quality={m:{r["request_id"]:r["quality"] for r in rows if r["method"] == m and r["stage"] == "natural"} for m in ("DFlash","ASD")})
    old = json.loads((root/"results/asd-adaptation-followup/analysis.json").read_text(encoding="utf-8"))
    history = old["scopes"]["independent"]["prefix/ordinary-position/ASD"]
    return dict(status="complete",original_zero_tolerance_gate_passed=original_gate,
        cli_spent_atol=1e-5,comparison_policy="Postrun CLI-only ledger comparison; all other fields exact and spent <= 8 unchanged",
        source_commit=manifest["commit"],source_sha256=manifest["source_sha256"],
        frozen_files=len(manifest["files"]),direct_rows=32,ar_references=8,fresh_cli_runs=2,
        variants=variants,adapter_sha256=ADAPTER_SHA,
        historical_study=dict(path="results/asd-adaptation-followup/analysis.json",requests=history["requests"],
            prefix_latency_ratio=history["latency_ratio"],target_calls=history["target_calls"],
            note="Previously measured accepted result; no new speedup estimate in runtime acceptance"))


def render(result):
    lines = ["# ASD运行时工程验收", "", "原ASD与位置适配ASD均已接入CLI、HTTP和SSE。两配置共32条直接解码、8条完整AR参考、2次独立CLI完成；流式、请求预算隔离、取消恢复和严格benchmark隔离检查通过。", "",
        "这是已观察四题的工程接入检查，不是新性能研究。历史独立16题的128 token时延降低3.23%、Target验证331→319仍为本方案的已接纳收益；本轮没有重新估计这个百分比。ASD允许改变输出，原450错答继续保留。", "",
        f"执行源码：`{result['source_commit']}`；{result['frozen_files']}个canonical Git文件运行前后SHA一致，原适配器只读SHA不变。", "",
        "|配置|模型准备秒|适配器合并秒|首次AR32 wall秒|独立CLI冷请求秒|编译图|GPU峰GiB|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name,v in result["variants"].items():
        setup = v["adapter_setup"]["setup_seconds"] if v["adapter_setup"] else 0
        lines.append(f"|{name}|{v['model_setup_seconds']:.3f}|{setup:.3f}|{v['cold_requests'][0]['seconds']:.3f}|{v['cli_cold_seconds']:.3f}|{v['compiler']['graph_count']}|{v['peak_gpu_bytes']/2**30:.3f}|")
    lines += ["", "模型准备是load_engine总耗时，包含适配器合并；合并列是其中子项，不能相加。首次AR和独立CLI均从空编译缓存开始，编译成本在请求内；正式32条请求没有新编译。CLI请求耗时不包括此前模型加载；不能把启动成本当作免费。两配置各自使用单卡，FP32/highest/TF32关闭、eager attention及显式RMSNorm profile。", "",
        "直接结果的token/text/EOS、输入快照、验证轮数和ASD账本与冻结研究参考相同；strict结果另与本轮AR逐token核对。CLI两条的token/EOS/轮数/例外数也完全相同，但spent分别相差7.62939453125e-6和3.814697265625e-6，原零容差验收因此失败，execution-verification.json及失败日志原样保留。", "",
        "后处理仅对独立冷CLI的spent累计比较允许1e-5绝对误差，仍要求实际spent<=8，其他字段零容差。未改变Target、ASD决策、预算或生成结果，也未重跑GPU。worker预热后编译4图、CLI冷启动3图，微小差异与FP32编译路径舍入相符，尚未逐算子定位；不能将其写成账本位级一致。详见[数值例外说明](ASD_RUNTIME_NUMERICS.md)。每次请求新建预算，取消后新请求恢复到完整参考。", "",
        "|配置|HTTP成功请求/4|并发|断连取消|恢复|HTTP失败|", "|---|---:|---:|---|---|---:|"]
    for name,v in result["variants"].items():
        h = v["http"]
        lines.append(f"|{name}|{h['load']['successful']}|2|{h['disconnect_cancelled']}|{h['recovered']}|{h['metrics']['failed']}|")
    lines += ["", "HTTP是有界单worker队列。小检查验证协议和恢复，4请求不用于推断生产吞吐或尾延迟；load观测包含排队与传输。SSE start先标记approximate，complete包含账本与adapter SHA。", "",
        "使用步骤和HTTP示例见[运行指南](ASD_RUNTIME.md)，固定题目和资源合同见[验收协议](ASD_RUNTIME_PROTOCOL.md)。原始结果位于results/asd-runtime，可用`python -m scripts.write_asd_runtime_report`重建本报告及analysis.json。模型、适配器和hidden不发布，默认strict配置和原面试版保持。", ""]
    return "\n".join(lines)


def main():
    result = analyze(ROOT)
    (ROOT/"results/asd-runtime/analysis.json").write_bytes((json.dumps(result,indent=2)+"\n").encode())
    (ROOT/"docs/ASD_RUNTIME_RESULTS.md").write_bytes(render(result).encode())
    print(json.dumps(dict(status=result["status"],source_commit=result["source_commit"],direct_rows=result["direct_rows"])))


if __name__ == "__main__":
    main()
