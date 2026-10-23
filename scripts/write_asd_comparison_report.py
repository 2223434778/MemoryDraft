"""Validate full task coverage, freeze winners, and report held-out results."""
import argparse
import json
from pathlib import Path

from memorydraft.evaluation.asd_comparison import STUDY, aggregate, records, select_candidates
from memorydraft.evaluation.runtime_migration import read_rows
from memorydraft.evaluation.adaptation import digest
from scripts.diagnose_transport_online import write
from scripts.write_asd_allocation_report import audit_resources

ROOT = Path(__file__).resolve().parents[1]


def phase_summary(base, phase, selection=None):
    rows = [r for i in (0,1) for r in read_rows(base/f"{phase}-{i}/rows.jsonl")]
    refs = [r for i in (0,1) for r in read_rows(base/f"{phase}-{i}/references.jsonl")]
    expected = {r["id"]:r["benchmark"] for r in records(ROOT,phase)}
    if {r["request_id"]:r["benchmark"] for r in rows} != expected:
        raise ValueError("Changed comparison task coverage")
    if len(refs) != len(expected) or {r["request_id"] for r in refs} != set(expected):
        raise ValueError("Missing AR quality references")
    indexed = {r["request_id"]:r for r in refs}
    for row in rows:
        reference = indexed[row["request_id"]]
        if (row["prompt_sha256"],row["snapshot_id"]) != (reference["prompt_sha256"],reference["snapshot_id"]):
            raise ValueError("Actual Target input differs from AR")
        if row["arm"].endswith(("/DFlash","/tree32")):
            limit = 4096 if row["stage"] == "natural" else 128
            if row["token_ids"] != reference["token_ids"][:limit] or (row["stage"] == "natural" and
                    (row["text"],row["finish_reason"]) != (reference["text"],reference["finish_reason"])):
                raise ValueError("Actual strict output differs from AR")
    infos = [json.loads((base/f"{phase}-{i}/provenance.json").read_text()) for i in (0,1)]
    if any(p["status"] != "complete" or not p["original_unchanged"] or not p["base_restored"] for p in infos):
        raise ValueError("Incomplete comparison workers")
    if selection is not None and any(p["selection"] != selection or
            p["selection_sha256"] != digest((base/"selection.json").read_bytes()) for p in infos):
        raise ValueError("Confirmation selection changed")
    result = aggregate(rows,phase,selection)
    result["resources"] = audit_resources(infos,rows)
    if result["resources"]["formal_compile_events"]:
        raise ValueError("Formal comparison compiled new graphs")
    result["ar_quality"] = {b:dict(n=sum(r["benchmark"]==b for r in refs),
        correct=sum(r["quality"]["correct"] for r in refs if r["benchmark"]==b),
        secondary_correct=sum(r["quality"].get("secondary_correct",r["quality"]["correct"]) for r in refs if r["benchmark"]==b))
        for b in result["families"]}
    return result


def render(result):
    lines = ["# ASD候选统一比较", "", "负百分比表示时延降低；先合并同题重复，再对请求等权平均比值。",
        "128前缀只比较双方均实际达到128的题，短EOS请求仍留在自然生成主表。",
        "MATH同时保留保守主评分和有限标量等价评分，选择使用后者；其他任务两者相同。", ""]
    for phase in ("screen","confirm"):
        if phase not in result:
            continue
        summary = result[phase]
        lines += [f"## {phase}: {summary['requests']}题 / {summary['rows']}次计时", "",
            "|任务|候选|自然相对ASD|95%区间|自然相对DFlash|共同128相对ASD / N|正确主/扩展|ASD正确→错/错→对|截断|",
            "|---|---|---:|---|---:|---|---|---|---:|"]
        for b,arms in summary["families"].items():
            for arm,item in arms.items():
                n,p = item["natural"],item["prefix"]["base/ASD"]
                a,d = n["base/ASD"],n["base/DFlash"]
                prefix = f"{100*(p['ratio']-1):+.2f}% / {p['n']}" if p["ratio"] is not None else f"无 / {p['n']}"
                ci = f"[{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]"
                lines.append(f"|{b}|{arm}|{100*(a['ratio']-1):+.2f}%|{ci}|{100*(d['ratio']-1):+.2f}%|{prefix}|{n['correct']}/{n['secondary_correct']}（共{n['requests']}）|{a['lost']}/{a['gained']}|{n['truncations']}|")
        lines += ["", "全部错答、截断及退化保留；自然时延包含输出路径变化，等token结果与调用密度另列于analysis.json。", ""]
    if "selection" in result:
        lines += ["## 初筛后冻结的确认候选", "", "```json",json.dumps(result["selection"],ensure_ascii=False,indent=2),"```",""]
    lines += ["执行使用Qwen3-8B/DFlash、FP32/eager attention、RMSNorm编译、原生16块；",
        "普通ASD/Run/Utility均用CPU摘要接受器，置信度策略使用原张量实现。全部在线判定成本计入。",
        "各适配器为旧冻结权重，离线dense合并/基准切换成本和冷编译单列，不声称线上切模型免费。",
        "树为严格DFlash候选，不冒充ASD树；未使用缓存或修改Target输入。本轮没有新训练seed。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/f"results/{STUDY}")
    parser.add_argument("--screen-only",action="store_true")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    screen = phase_summary(args.input,"screen")
    selection = select_candidates(screen)
    path = args.input/"selection.json"
    if path.exists() and json.loads(path.read_text()) != selection:
        raise ValueError("Previously frozen selection changed")
    write(path,selection)
    result = dict(screen=screen,selection=selection)
    if not args.screen_only:
        result["confirm"] = phase_summary(args.input,"confirm",selection)
    out = args.output or args.input
    out.mkdir(parents=True,exist_ok=True)
    write(out/"analysis.json",result)
    (out/"report.md").write_text(render(result),encoding="utf-8",newline="\n")
    print(json.dumps(dict(phases=list(result),selection=selection),ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
