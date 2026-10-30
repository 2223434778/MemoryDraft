"""Publish fixed-width/feedback results with unchanged-gate and history checks."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

from scripts.report_branch_drift_optimization import compressed, percent, write
from scripts.run_gate_width import ARMS, summarize_results

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/raw/gate-width')
    parser.add_argument('--output', default='results/gate-width')
    parser.add_argument('--report', default='docs/GATE_WIDTH_RESULTS.md')
    args = parser.parse_args()
    source, out = ROOT/args.input, ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    def raw(name):
        return (source/name).read_bytes() if (source/name).exists() else gzip.decompress((source/(name+'.gz')).read_bytes())
    rows_raw, refs_raw = raw('rows.jsonl'), raw('references.json')
    rows = [json.loads(s) for s in rows_raw.decode('utf-8').splitlines()]
    refs = json.loads(refs_raw)
    analysis = summarize_results(rows, refs)
    if analysis != json.loads((source/'analysis.json').read_bytes()):
        raise ValueError('Server analysis differs from independent rebuild')
    if json.loads((source/'provenance.json').read_bytes())['status'] != 'completed':
        raise ValueError('Incomplete study')
    fixed = {'GateCap-ASD': 1., 'WidthHalf-ASD': .5, 'WidthZero-ASD': 0.}
    for r in rows:
        if r['arm'] in fixed and any(b['tree_budget']['gate_penalty'] != fixed[r['arm']] for b in r['rounds']):
            raise ValueError('Incorrect fixed width coefficient')
        if r['arm'] in ARMS[2:] and any(b.get('relaxed_tokens', 0) > 2
                or b.get('regret', 0) > math.log(4)*b.get('relaxed_tokens', 0)+1e-7 for b in r['rounds']):
            raise ValueError('Exception or aggregate raw regret cap exceeded')
    (out/'rows.jsonl.gz').write_bytes(compressed(rows_raw))
    (out/'references.json.gz').write_bytes(compressed(refs_raw))
    for name in ('tasks.json', 'provenance.json', 'cost-table.json'):
        if (source/name).resolve() != (out/name).resolve():
            shutil.copyfile(source/name, out/name)
    write(out/'analysis.json', analysis)
    ref_index = {r['request_id']: r for r in refs}
    strict = [r for r in rows if r['arm'] == 'BudgetTree' and r['stage'] == 'natural']
    audit = dict(timings=len(rows), matched_inputs=len(refs), matrix_inputs_snapshots_budget_valid=True,
        gate_parameters_unchanged=True, feedback_past_four_rounds_replayed=True,
        server_analysis_rebuilt_equal=True,
        strict_ar_differences=[r['request_id'] for r in strict if any(r[k] != ref_index[r['request_id']][k]
                               for k in ('token_ids', 'text', 'finish_reason'))])
    previous = ROOT/'results/branch-drift-optimization/rows.jsonl.gz'
    if previous.exists():
        old_rows = [json.loads(s) for s in gzip.decompress(previous.read_bytes()).decode().splitlines()]
        old = {(r['request_id'], r['stage'], r['arm']): r for r in old_rows}
        controls = [r for r in rows if r['arm'] in ('ASD', 'BudgetTree', 'GateCap-ASD')]
        fields = ('token_ids', 'text', 'finish_reason', 'prompt_sha256', 'snapshot_id')
        audit['previous_baseline_identity'] = dict(pairs=len(controls), differences=[
            dict(request_id=r['request_id'], stage=r['stage'], arm=r['arm']) for r in controls
            if any(r[k] != old[r['request_id'], r['stage'], r['arm']][k] for k in fields)],
            timing_policy='Only compare identities; old latency is not a same-session baseline')
    write(out/'audit.json', audit)
    indexed = {(r['request_id'], r['stage'], r['arm']): r for r in rows}
    controls = {}
    for benchmark in analysis['families']:
        controls[benchmark] = {}
        for arm in ARMS[3:]:
            candidates = [r for r in rows if r['benchmark'] == benchmark and r['arm'] == arm and r['stage'] == 'natural']
            pairs = [(r, indexed[r['request_id'], 'natural', 'GateCap-ASD']) for r in candidates]
            pairs = [(a, b) for a, b in pairs if a['token_ids'] == b['token_ids'] and a['finish_reason'] == b['finish_reason']]
            controls[benchmark][arm] = dict(n=len(pairs),
                ratio=statistics.mean(a['latency_seconds']/b['latency_seconds'] for a, b in pairs) if pairs else None,
                variant_calls=sum(len(a['rounds']) for a, b in pairs), baseline_calls=sum(len(b['rounds']) for a, b in pairs))
    write(out/'same-output.json', controls)
    lines = ['# 保真门控固定的树宽度实测', '',
        f"{len(refs)}匹配输入、{len(rows)}新计时，单次配对开发检查；负数更快。全部错误和截断保留。", '',
        '|任务|方法|自然 vs ASD|自然 vs 惩罚1|自然 vs 严格树|AR相同|正确/可评分|截断|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for benchmark, family in analysis['families'].items():
        for arm in ARMS:
            v = family[arm]['natural']
            quality = f"{v['correct']}/{v['scored']}" if v['scored'] else '—'
            lines.append(f"|{benchmark}|{arm}|{percent(v['ASD']['ratio'])}|{percent(v['GateCap-ASD']['ratio'])}|{percent(v['BudgetTree']['ratio'])}|{v['identical_ar']}/{v['n']}|{quality}|{v['truncated']}|")
    lines += ['', '## 共同128/EOS', '',
        '|任务|方法|vs ASD|有效数|vs 惩罚1|有效数|', '|---|---|---:|---:|---:|---:|']
    for benchmark, family in analysis['families'].items():
        for arm in ARMS[2:]:
            v = family[arm]['prefix']
            lines.append(f"|{benchmark}|{arm}|{percent(v['ASD']['ratio'])}|{v['ASD']['n']}|{percent(v['GateCap-ASD']['ratio'])}|{v['GateCap-ASD']['n']}|")
    totals = {}
    lines += ['', '## 自然请求计算与宽度', '',
        '不同方法可能处于不同生成轨迹。节点/深度和轮数为请求汇总，不是同状态因果分项；所有反馈开销计入wall。', '',
        '|任务|方法|总秒|输出token|验证轮|平均节点|平均最大深度|系数0/.5/1轮数|',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for benchmark, family in analysis['families'].items():
        totals[benchmark] = {}
        for arm in ARMS:
            selected = [r for r in rows if r['benchmark'] == benchmark and r['arm'] == arm and r['stage'] == 'natural']
            blocks = [b for r in selected for b in r['rounds']]
            trees = [b['tree_budget'] for b in blocks if b.get('tree_budget')]
            v = family[arm]['natural']
            counts = [sum(t.get('gate_penalty') == x for t in trees) for x in (0., .5, 1.)]
            extra = dict(mean_nodes=statistics.mean(t['nodes'] for t in trees) if trees else None,
                mean_depth=statistics.mean(t['max_depth'] for t in trees) if trees else None,
                penalty_counts=counts, recoverable_misses=sum(t.get('width_recoverable_miss', False) for t in trees),
                feedback_seconds=sum((r['asd'] or {}).get('width_feedback', {}).get('feedback_seconds', 0) for r in selected))
            totals[benchmark][arm] = dict(seconds=v['seconds'], tokens=v['tokens'], calls=v['calls'], **extra)
            nodes = '—' if extra['mean_nodes'] is None else f"{extra['mean_nodes']:.2f}"
            depth = '—' if extra['mean_depth'] is None else f"{extra['mean_depth']:.2f}"
            lines.append(f"|{benchmark}|{arm}|{v['seconds']:.3f}|{v['tokens']}|{v['calls']}|{nodes}|{depth}|{'/'.join(map(str, counts))}|")
    write(out/'totals.json', totals)
    lines += ['', '## 诊断题', '', '|输入|方法|秒|token|正确|AR相同|停止|', '|---|---|---:|---:|---|---|---|']
    for r in rows:
        if r['stage'] == 'natural' and r['request_id'] in ('gsm8k:750', 'math500:372', 'math500:101', 'humaneval:76'):
            q = r['quality'].get('secondary_correct', r['quality']['correct'])
            lines.append(f"|{r['request_id']}|{r['arm']}|{r['latency_seconds']:.3f}|{len(r['token_ids'])}|{q}|{r['token_ids']==ref_index[r['request_id']]['token_ids']}|{r['finish_reason']}|")
    lines += ['', '规则见[协议](GATE_WIDTH_PROTOCOL.md)，[原始公开摘要](../results/gate-width/)包含输入身份、预算、反馈历史重放和源码清单。',
        '开放任务本轮无新增偏好评分。AR一致率不等同正确率或全序列分布距离。',
        '`python -m scripts.report_gate_width --input results/gate-width`可从压缩输入重建。', '']
    (ROOT/args.report).write_bytes('\n'.join(lines).encode('utf-8'))
    write(out/'publication.json', dict(reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        payloads={p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'publication.json'
                  and '__pycache__' not in p.parts and p.suffix != '.pyc'},
        report=dict(path=args.report, sha256=hashlib.sha256((ROOT/args.report).read_bytes()).hexdigest())))
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
