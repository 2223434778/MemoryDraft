"""Rebuild the six-arm second-stage comparison and allocation audit."""
import argparse
import json
from pathlib import Path

from memorydraft.evaluation.asd_joint_tree import ARMS, STUDY, aggregate
from memorydraft.evaluation.asd_tree import REPEATS, records
from scripts.diagnose_transport_online import write
from scripts.write_asd_tree_report import ROOT, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/f"results/{STUDY}")
    args = parser.parse_args()
    rows,refs = read_rows(args.input,"rows"),read_rows(args.input,"references")
    expected = {r["id"] for r in records(ROOT)}
    if ({r["request_id"] for r in refs} != expected
            or len(rows) != len(expected)*len(ARMS)*sum(REPEATS.values())):
        raise ValueError("Frozen 48-request coverage incomplete")
    resources = []
    for shard in (0,1):
        info = json.loads((args.input/f"part-{shard}/provenance.json").read_text())
        if (info["status"] != "complete" or not info["original_unchanged"] or info["count"] != 720
                or info["legacy_checked_rows"] != 480):
            raise ValueError("Worker incomplete or first-stage evidence changed")
        if info["compiler"]["graph_count"] != info["compiler_after_warmup"]["graph_count"]:
            raise ValueError("Formal generation compiled new graphs")
        resources.append(info)
    result = aggregate(rows,refs)
    result["resources"] = resources
    write(args.input/"analysis.json",result)
    lines = ["# 第二阶段：剩余regret预算参与候选树分配", "",
        "48道已观察开发题；六方法自然EOS各3次、128/EOS各2次，1440计时+48同题AR。",
        "沿用各卡第一阶段成本表，原四组960次轨迹逐项复现；新增在线成本计入请求wall。",
        "SpineTree-ASD固定保留原Draft主干；JointTree-ASD按实际剩余regret缩短强制主干，其余节点best-first。",
        "所有组原Target/FP32/RMSNorm，原ASD B8/g.25/m2；仅更改构树，不更改接受条件。", "",
        "|任务|方法|自然相对ASD|自然相对第一阶段组合|正确/8|截断|等128相对第一阶段组合（N）|验证轮|节点|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def percent(value):return "—" if value is None else f"{(value-1)*100:+.2f}%"
    for benchmark,family in result["families"].items():
        for arm in ARMS:
            natural,prefix = family[arm]["natural"],family[arm]["prefix"]
            lines.append(f"|{benchmark}|{arm}|{percent(natural['ASD']['ratio'])}"
                f"|{percent(natural['BudgetTree-ASD']['ratio'])}|{natural['correct']}/8|{natural['truncated']}"
                f"|{percent(prefix['BudgetTree-ASD']['ratio'])}（{prefix['BudgetTree-ASD']['n']}）"
                f"|{natural['calls']}|{natural['nodes']}|")
    lines += ["", "百分比为逐请求时延比的题等权均值；负数更快。重复不当作独立题。",
        "自然时延同时受答案长度影响；等128仅保留双方完整输出128个token的请求，其余短EOS照常保留。", "",
        "## 动态分配相对固定主干消融", "",
        "|任务|自然变化|等128变化（N）|正确数变化|丢失正确题|新增正确题|",
        "|---|---:|---:|---:|---|---|"]
    for benchmark,family in result["families"].items():
        natural,prefix = family["JointTree-ASD"]["natural"],family["JointTree-ASD"]["prefix"]
        paired = natural["SpineTree-ASD"]
        delta = natural["correct"]-family["SpineTree-ASD"]["natural"]["correct"]
        lines.append(f"|{benchmark}|{percent(paired['ratio'])}|{percent(prefix['SpineTree-ASD']['ratio'])}"
            f"（{prefix['SpineTree-ASD']['n']}）|{delta:+d}|{', '.join(paired['lost']) or '—'}"
            f"|{', '.join(paired['gained']) or '—'}|")
    audit = result["allocation_audit"]
    lines += ["",f"逐轮预算快照/主干长度/实际扣费已核对：{audit['checked_blocks']}轮，"
        f"其中动态主干短于窗口{audit['remaining_shortened_blocks']}轮（含重复）。",
        "该分配是资源启发式，既不是错误概率预测，也不保证全局最优或质量无损。", ""]
    (args.input/"report.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(timed=len(rows),ar=len(refs),requests=len(expected),
                         legacy_checked_rows=960,all_execution_checks=True)))


if __name__ == "__main__":main()
