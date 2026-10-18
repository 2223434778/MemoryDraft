"""Audit Target-only release triage and its offline, answer-aware reuse upper bound."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from memorydraft.evaluation.release_triage import (AREAS, answer_event, load_dataset, opportunity,
    opportunity_gate, quality, quality_gate, request_text, sha)
from memorydraft.memory.store import Snapshot
from scripts.run_reuse_pilot import digest, save


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/research/release-triage"


def references(source):
    path = source / "references.jsonl"
    raw = path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix(".jsonl.gz").read_bytes())
    return [json.loads(line) for line in raw.decode().splitlines() if line], sha(raw)


def audit(projects, rows, stops):
    expected = [(p["id"], t) for p in projects for t in range(3)]
    if [(r["project_id"], r["turn"]) for r in rows] != expected:
        raise ValueError("Missing, duplicate, reordered or unexpected reference")
    labels = {(r["project_id"], r["turn"]): r for r in rows}
    for project in projects:
        for turn in range(3):
            row = labels[project["id"], turn]
            events = sorted([answer_event(project, prior, labels[project["id"], prior]["text"])
                             for prior in range(turn)], key=lambda event: event.record_id)
            snapshot = Snapshot(project["id"], 10 + turn * 10, tuple(events))
            records = [dict(record_id=r.record_id, order=r.order, text_sha256=sha(r.text.encode())) for r in events]
            if (row["order"] != snapshot.before_order or row["snapshot_id"] != snapshot.snapshot_id
                    or row["history_records"] != records or row["method"] != "AR"
                    or row["request_sha256"] != sha(request_text(project, turn).encode())
                    or row["input_tokens"] != len(row["prompt_token_ids"])
                    or row["prompt_sha256"] != digest(row["prompt_token_ids"])
                    or row["budget"] != 768 or len(row["token_ids"]) > 768
                    or row["budget_hit"] != (len(row["token_ids"]) >= 768)
                    or row["eos"] != bool(row["token_ids"] and row["token_ids"][-1] in stops)):
                raise ValueError("Reference violated the frozen input or legal history contract")


def run(source, output, *, report_only=False, tokenizer_path=None):
    projects, gold, _ = load_dataset(DATA)
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    if provenance["status"] != "complete" or provenance["rows"] != 18 or provenance["data_sha256"] != sha((DATA / "manifest.json").read_bytes()):
        raise ValueError("Complete the unchanged Target-only pass first")
    if json.loads((source / "projects.json").read_text(encoding="utf-8")) != projects:
        raise ValueError("Unexpected source projects")
    rows, reference_hash = references(source)
    audit(projects, rows, provenance["eos_token_ids"])
    lookup = {(r["project_id"], r["turn"]): r for r in rows}
    scored, opportunities = [], []
    if report_only:
        cached = json.loads((source / "opportunities.json").read_text(encoding="utf-8"))
        if (cached["references_sha256"] != reference_hash or cached["data_sha256"] != provenance["data_sha256"]
                or cached["runtime_source_sha256"] != provenance["source"]["source_sha256"]):
            raise ValueError("Opportunity analysis does not bind these outputs")
        opportunities = cached["rows"]
        if [(r["project_id"], r["turn"]) for r in opportunities] != [(r["project_id"], r["turn"]) for r in rows]:
            raise ValueError("Opportunity rows incomplete or out of order")
    else:
        from transformers import AutoTokenizer

        path = tokenizer_path or provenance["target"]["model"]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        for project in projects:
            for turn in range(3):
                row = lookup[project["id"], turn]
                text = request_text(project, turn)
                past = [lookup[project["id"], t]["text"] for t in range(turn)]
                opportunities.append(dict(project_id=project["id"], turn=turn,
                    opportunity=opportunity(row["token_ids"], text, past, tokenizer)))
        save(source / "opportunities.json", dict(references_sha256=reference_hash,
             data_sha256=provenance["data_sha256"], runtime_source_sha256=provenance["source"]["source_sha256"],
             oracle=True, measured_speedup=False, rows=opportunities))
    for project in projects:
        for turn, release in enumerate(project["releases"]):
            row = lookup[project["id"], turn]
            scored.append(dict(project_id=project["id"], turn=turn, budget_hit=row["budget_hit"], eos=row["eos"],
                               quality=quality(row["text"], release, gold[project["id"] + ":" + release["tag"]])))
    qgate, ogate = quality_gate(scored), opportunity_gate(opportunities)
    decision = dict(quality=qgate, opportunity=ogate, timing_allowed=qgate["passed"] and ogate["passed"],
                    action="allow_predeclared_timing" if qgate["passed"] and ogate["passed"] else "stop_candidate",
                    references_sha256=reference_hash, data_sha256=provenance["data_sha256"],
                    runtime_source_sha256=provenance["source"]["source_sha256"], measured_speedup=False)
    save(source / "quality.json", dict(references_sha256=reference_hash, rows=scored))
    save(source / "gate-decision.json", decision)
    lines = ["# 真实发布说明影响初筛：小成本检验", "", "2026-10-18。运行前协议：[RELEASE_TRIAGE_PROTOCOL](RELEASE_TRIAGE_PROTOCOL.md)。",
        "", "6个公开项目、各3个指定版本，共18次Target AR推理。数据是真实发布说明，任务及输出合同由实验构造；没有自然用户需求/服务日志，不声称独立泛化确认。",
        "标签在推理前由开发agent按来源逐例标注并冻结；没有独立领域专家复核，不使用Target自评。",
        "", "| 质量检查 | 结果 | 门槛 |", "|---|---:|---:|",
        f"| 严格输出合同及版本全部合格 | {qgate['schema_and_identity']}/18 | 18/18 |",
        f"| 五类判断与证据全部正确的请求 | {qgate['correct_requests']}/18 | 至少17/18 |",
        f"| 严格类别得分（整份合同失败时五类均计错） | {qgate['correct_categories']}/90 | 至少86/90 |",
        f"| 关键正类未通过严格评分（含整体合同失败） | {len(qgate['critical_misses'])} | 0 |",
        f"| 截断或未结束输出 | {qgate['incomplete_outputs']} | 0 |", "",
        "| 项目 | 正确请求 | 有历史请求的独有覆盖 |", "|---|---:|---:|"]
    for project in projects:
        count = sum(r["quality"]["correct"] for r in scored if r["project_id"] == project["id"])
        lines.append(f"| {project['repository']} | {count}/3 | {ogate['per_project'][project['id']]:.1%} |")
    lines += ["", "独有覆盖是Global可复用位置并集减Context并集，仅统计12个有历史请求：",
        f"{ogate['history_only_covered']}/{ogate['output_tokens']} token（{ogate['history_only_fraction']:.1%}）；门槛10%。",
        f"达到项目级5%独有覆盖的项目为{ogate['qualifying_projects']}/6，门槛至少4/6。",
        "此为枚举来源位置、利用当前答案选择最长正确接续的离线oracle上限；忽略真实候选选择、路由和全部运行成本。不能称为实测接受率、可实现净收益或加速比。",
        "所有18例均保留，包括错误答案；正确性是分类初筛合同，不是完整迁移计划、功能测试或安全结论。", "", "失败详情："]
    for row in scored:
        if not row["quality"]["correct"]:
            wrong = [area for area in AREAS if not row["quality"]["areas"][area]]
            lines.append(f"- {row['project_id']} / turn={row['turn']}：错误类别 {wrong}；schema={row['quality']['schema']}，identity={row['quality']['identity']}，error={row['quality']['error']}。")
    lines += ["", f"质量门槛：{qgate['passed']}；机会门槛：{ogate['passed']}。决定：`{decision['action']}`。",
        "尚无本场景DFlash/Global/Context/Arctic端到端计时结果，不将AR参考成本或oracle覆盖换算成速度。" if decision["timing_allowed"] else
        "未通过预定前置门槛，未启动最多48次加速方法计时；未扫提示、修改标签、放宽验证或剔除失败请求。这个候选检查按协议结束。",
        "", f"Target请求累计墙钟（含本次快照/写回等）：{sum(r['latency_seconds'] for r in rows):.3f}s；模型加载{provenance['load_seconds']:.3f}s，单列。",
        "AR请求时间不是加速对照结论，也不包含模型安装、全部权重审计和离线评分。",
        "FP32/eager、greedy、非思考、Qwen3-8B、Transformers4.57.3、A6000 GPU0、4线程。",
        "同项目过去回答保留错误，并按完成顺序写回；每轮Target只见本次说明和共同定义，候选历史不补充缺失的当前事实。",
        "", f"推理源码SHA-256：`{provenance['source']['source_sha256']}`。",
        f"冻结数据manifest SHA-256：`{provenance['data_sha256']}`。",
        f"原始参考JSONL SHA-256：`{reference_hash}`。", "",
        "服务器首次评分：`PYTHONPATH=. HF_HUB_OFFLINE=1 .venv-dflash/bin/python -m scripts.write_release_triage_report`。",
        "无模型离线复算：`python -m scripts.write_release_triage_report --report-only`。", ""]
    diagnosis_path = source / "posthoc-diagnosis.json"
    if diagnosis_path.exists():
        diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
        if (diagnosis["references_sha256"] != reference_hash or diagnosis["data_sha256"] != provenance["data_sha256"]
                or diagnosis["runtime_source_sha256"] != provenance["source"]["source_sha256"]):
            raise ValueError("Post-run diagnosis does not bind these outputs")
        detail = diagnosis["summary"]
        lines += ["事后诊断（不更改上述门槛/得分/停止决定）：",
            f"实际{detail['base_json_identity_and_area_order']}/18均可解析JSON且版本/类别顺序正确；"
            f"只看status为{detail['status_correct_categories']}/90类、{detail['status_correct_requests']}/18请求正确。",
            f"{detail['evidence_count_violation_requests']}请求超过1–2条证据上限，"
            f"{detail['unsupported_evidence_requests']}请求引用不支持该类的行；两者可重叠。"
            f"关键正类实际status漏判{detail['critical_status_false_negatives']}，关键类误报{detail['critical_status_false_positives']}。",
            "严格合同失败会令整份评分归零；不能把30/90称为纯分类准确率，或把严格评分的7项关键失败全部称为真实漏判。",
            f"另用两个不含来源事实/标签/当前答案的静态schema示例（共{detail['static_example_tokens']} token）作探索性oracle对照："
            f"历史在prompt+静态示例之外独有覆盖为{detail['history_beyond_schema']}/{detail['warm_output_tokens']}"
            f"（{detail['history_beyond_schema_fraction']:.1%}）。",
            "静态对照在看过输出后增加，格式选择有事后性，资源预算未逐请求严格配平，没有在线计时；"
            "它解释结构重复，不证明静态方案的实际速度或新门槛通过。详见 [诊断](RELEASE_TRIAGE_DIAGNOSIS.md)。", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes("\n".join(lines).encode())
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/release-triage-feasibility")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/RELEASE_TRIAGE_RESULTS.md")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output, report_only=args.report_only, tokenizer_path=args.tokenizer), ensure_ascii=False, indent=2))
