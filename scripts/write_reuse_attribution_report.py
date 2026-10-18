"""Rebuild the post-hoc attribution report from bound public CPU analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_report(source, original, output):
    raw = (source / "offline-final.json").read_bytes()
    result = json.loads(raw)
    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    if decision["analysis_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Attribution decision is not bound to this analysis")
    for name, expected in result["input_files_sha256"].items():
        if hashlib.sha256((original / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Original evidence changed: " + name)
    for name, expected in result["analysis_files_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Analysis source changed; preserve evidence and recompute explicitly: " + name)
    summary = result["summary"]
    count, total = summary["requests"], summary["output_tokens"]
    lines = ["# 合成配置正结果的归因检查", "",
        "2026-10-18。用户授权后的CPU事后诊断；原八个请求及输出早已被查看，不属于新的独立确认。",
        "", "**结论：原19.4%的在线延迟降低仍成立；本轮发现，当前请求中已经给出的事实经过确定性展开，即可提供几乎全部历史接续机会，并逐字重建8/8 Target输出。因此这些合成正结果不足以证明长期历史是收益的必要来源。**",
        "", "## 原始性能证据", "",
        "原4项目、每项目2轮、3重复、4方法，共96次计时，经原audit重新核对token、快照、配对、KV及summary。原文件哈希在分析前后相同，并绑定在新分析JSON中。",
        "", "| 方法 | 原实测总秒数 | R轮数 | D轮数 |", "|---|---:|---:|---:|"]
    for method in ("DFlash", "Adapted", "Global", "Context"):
        counts = summary["original_route_counts"].get(method, {})
        rounds = (str(counts.get("R", 0)), str(counts.get("D", 0))) if counts else ("未插桩", "未插桩")
        lines.append(f"| {method} | {summary['original_times'][method]:.6f} | {rounds[0]} | {rounds[1]} |")
    lines += ["", "Global过去回答已展开为JSON，替代部分模型起草；接受5418个R token，D轮数498降至123，符合减少模型计算的机制解释。总耗时仍以原同步端到端计时为准，不能把轮数或覆盖直接折算成各因素贡献的秒数。",
        "", "## 逐层替代资源", "",
        "所有候选函数仅接收当前request字符串；不读取评分标签、项目生成器隐藏结构、过去回答或当前答案。解析器只支持原benchmark明确语法，不声称通用自然语言理解。虚构模板project=example、对象名item_00等；Expanded和Compiled才使用当前真实标识。",
        "", "| 资源条件（均含共同prompt） | 内容 | oracle覆盖token | 历史在该条件之外 |",
        "|---|---|---:|---:|"]
    descriptions = dict(Context="当前原始文本", Structure="schema、数组长度及中性值", Static="schema默认值、虚构标识",
                        Global="合法过去回答", StaticGlobal="Static加合法过去回答",
                        Expanded="当前标识和默认事实展开，不执行本轮修改", Compiled="当前事实展开，并执行本轮修改")
    for method, item in summary["coverage"].items():
        extra = summary["history_outside"].get(method)
        extra_text = f"{extra}/{total}（{extra/total:.2%}）" if extra is not None else "—"
        lines.append(f"| {method} | {descriptions[method]} | {item['covered']}/{total}（{item['covered']/total:.2%}） | {extra_text} |")
    observed = summary["observed_r_tokens"]
    lines += ["", "该oracle使用6-token锚点、最多32-token接续、至少8连续正确才计位置，各候选/特殊token边界独立。它利用已知答案枚举最好接续，不模拟实际选源、冷却及KV成本，不是接受率、实际速度或延迟上界。分母包含首次尚无锚点的token和EOS，所以完整正确Compiled的覆盖仍小于100%。",
        "", f"在原Global实际接受的{observed['accepted']}个R token位置上，Static的oracle也能覆盖{observed['Static']}个（{observed['Static']/observed['accepted']:.2%}），Expanded覆盖{observed['Expanded']}个（{observed['Expanded']/observed['accepted']:.2%}），Compiled覆盖{observed['Compiled']}个。这里累计包含三次重复，不是独立样本数；也不是19.4%时间收益的因果百分比分摊。",
        "", "仅中性结构模板覆盖较低，加入明确默认值后大幅提高；补齐当前对象名称/项目标识后，历史额外覆盖只剩4个位置。由此更合适的机制解释是重复默认事实与标识已经排成目标格式，而不只是JSON标点/缩进，也未显示必须使用长期历史才可获得的信息。Expanded是首轮离线结果后增加的探索性对照，时间顺序见协议。",
        "", "## 合同正确性与资源公平性", "",
        f"Compiled与冻结Target逐字一致{summary['compiled_text_exact']}/{count}、去特殊token后逐token一致{summary['compiled_tokens_exact']}/{count}、JSON语义一致{summary['compiled_semantic_exact']}/{count}。Expanded故意不应用修改，语义正确{summary['expanded_semantic_exact']}/{count}；它只能作为需严格验证的候选，不能直接交付。首轮Expanded与真实初始历史逐字相同4/4，全部两轮合计{summary['expanded_equals_history']}/{count}。",
        "", "共同额外资源上限1536 token、各来源完整独立、没有裁剪或padding。Static实际长度略小，StaticGlobal约为单源的两倍；它们并非逐token配平。Expanded、Compiled与Global额外token数完全相同的请求分别为"
        f"{summary['equal_history_token_count_requests']['Expanded']}/{count}、{summary['equal_history_token_count_requests']['Compiled']}/{count}，候选全文及逐请求长度见JSON；输入prompt始终相同。",
        "", "| 项目 | Static之外历史token | Expanded之外历史token | Compiled逐字正确 |", "|---|---:|---:|---:|"]
    for identity in sorted({r["project_id"] for r in result["rows"]}):
        group = [r for r in result["rows"] if r["project_id"] == identity]
        lines.append(f"| {identity} | {sum(r['coverage']['Static']['history_outside'] for r in group)}"
                     f" | {sum(r['coverage']['Expanded']['history_outside'] for r in group)}"
                     f" | {sum(r['compiled_text_exact'] for r in group)}/{len(group)} |")
    lines += ["", "## 决定与边界", "",
        "按本次协议的离线停止条件，不新增GPU生成或Static/Expanded/Compiled在线计时。此处没有测出替代资源也能达到19.4%的速度改善；索引、查找、选择和缓存成本仍可能不同。证据回答的是原合成任务的历史必要性和候选来源替代解释。若要量化各资源实际速度，仍需新的同批计时，不能借用原Global耗时。",
        "", "直接程序生成的8/8正确只针对这些已经明确指定所有事实、字段和格式的合成请求；它不是自然语言工作负载泛化，不影响旧解码器严格正确和原加速事实。没有使用原答案构造候选，也没有将本轮输出写回历史。",
        "", "建议保留R+DFlash实现及受控机制正结果，结束以这批合成配置证明长期记忆独特价值的主张。当前没有依据据此重启H、增加路由或扩大同类合成实验。",
        "", "原推理源码SHA：`" + result["original_source_sha256"] + "`。新分析代码SHA、原证据逐文件SHA和本轮决定保存在`results/reuse-attribution/`；不替换旧provenance。",
        "", "复算（只需本机已存在tokenizer，无Target/Draft权重加载）：", "", "```powershell",
        ".venv\\Scripts\\python.exe -m scripts.diagnose_reuse_confirmation --output results/raw/research-stage2/reuse-attribution-recomputed/offline-final.json",
        ".venv\\Scripts\\python.exe -m scripts.write_reuse_attribution_report", "```", ""]
    output.write_bytes("\n".join(lines).encode("utf-8"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/reuse-attribution")
    parser.add_argument("--original", type=Path, default=ROOT / "results/reuse-confirmation")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/REUSE_ATTRIBUTION_RESULTS.md")
    args = parser.parse_args()
    write_report(args.source, args.original, args.output)
