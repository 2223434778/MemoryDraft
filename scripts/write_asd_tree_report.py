"""Rebuild request-paired ASD/tree statistics from complete raw evidence."""
import argparse
import gzip
import json
from pathlib import Path

from memorydraft.evaluation.asd_tree import ARMS, REPEATS, STUDY, aggregate, records
from scripts.diagnose_transport_online import write

ROOT = Path(__file__).resolve().parents[1]


def read_rows(directory, name):
    rows = []
    for shard in (0,1):
        path = directory/f"part-{shard}/{name}.jsonl"
        raw = path.read_bytes() if path.is_file() else gzip.decompress(path.with_suffix(".jsonl.gz").read_bytes())
        rows.extend(json.loads(line) for line in raw.decode().splitlines())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/f"results/{STUDY}")
    args = parser.parse_args()
    rows,refs = read_rows(args.input,"rows"),read_rows(args.input,"references")
    expected = {r["id"] for r in records(ROOT)}
    if {r["request_id"] for r in refs} != expected or len(rows) != len(expected)*len(ARMS)*sum(REPEATS.values()):
        raise ValueError("Frozen 48-request coverage incomplete")
    resources = []
    for shard in (0,1):
        info = json.loads((args.input/f"part-{shard}/provenance.json").read_text())
        if info["status"] != "complete" or not info["original_unchanged"] or info["count"] != 480:
            raise ValueError("Worker incomplete or previous evidence changed")
        if info["compiler"]["graph_count"] != info["compiler_after_warmup"]["graph_count"]:
            raise ValueError("Formal generation compiled new graphs")
        resources.append(info)
    result = aggregate(rows,refs)
    result["resources"] = resources
    write(args.input/"analysis.json",result)
    lines = ["# 成本受控候选树 + ASD：六类开发题比较", "",
        "48道已观察开发题；四方法自然EOS各3次、128/EOS各2次，960计时+48同题AR。",
        "同FP32/eager/RMSNorm、原Draft、B8/g.25/m2；树为BASTION式独立实现，非官方完整复现。",
        "成本表/模型加载/冷编译单列；全部在线构树、选择、验证和KV整理计入请求wall。", "",
        "|任务|方法|自然相对DFlash|自然相对ASD|正确/8|截断|等128相对ASD（N）|自然验证轮|自然节点|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def percent(value):return "—" if value is None else f"{(value-1)*100:+.2f}%"
    for benchmark,family in result["families"].items():
        for arm in ARMS:
            value = family[arm]
            natural,prefix = value["natural"],value["prefix"]
            lines.append(f"|{benchmark}|{arm}|{percent(natural['DFlash']['ratio'])}|{percent(natural['ASD']['ratio'])}"
                f"|{natural['correct']}/8|{natural['truncated']}|{percent(prefix['ASD']['ratio'])}（{prefix['ASD']['n']}）"
                f"|{natural['calls']}|{natural['nodes']}|")
    lines += ["", "百分比为逐请求时延比的题等权均值；负数更快。重复不当作独立题。",
              "数学同时保留保守主评分与有限等价secondary；正确仅代表既定评分器。", "",
              "## 同批组合相对成本受控严格树", "",
              "|任务|自然时延变化|等128变化（N）|正确数变化|同树额外推进合计|实际regret合计|",
              "|---|---:|---:|---:|---:|---:|"]
    for benchmark,family in result["families"].items():
        value = family["BudgetTree-ASD"]
        natural,prefix = value["natural"],value["prefix"]
        lines.append(f"|{benchmark}|{percent(natural['BudgetTree']['ratio'])}|{percent(prefix['BudgetTree']['ratio'])}"
            f"（{prefix['BudgetTree']['n']}）|{natural['correct']-family['BudgetTree']['natural']['correct']:+d}"
            f"|{natural['same_tree_extra']}|{natural['spent']:.6f}|")
    lines += ["", "同树额外推进是在实际ASD轨迹上，与该次已验证树的严格前缀比较；不能替代端到端减轮。", ""]
    (args.input/"report.md").write_bytes("\n".join(lines).encode())
    print(json.dumps(dict(timed=len(rows),ar=len(refs),requests=len(expected),all_execution_checks=True)))


if __name__ == "__main__":main()
