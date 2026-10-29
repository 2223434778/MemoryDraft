"""Rebuild paired results and publish small public study evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from scripts.run_branch_drift_optimization import ARMS, summarize_results

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))


def compressed(raw):
    payload = bytearray(gzip.compress(raw, mtime=0))
    payload[9] = 255  # Canonical OS marker for Windows/Linux publication rebuilds.
    return bytes(payload)


def percent(ratio):
    return '—' if ratio is None else f'{(ratio-1)*100:+.2f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/raw/branch-drift-optimization')
    parser.add_argument('--output', default='results/branch-drift-optimization')
    parser.add_argument('--report', default='docs/BRANCH_DRIFT_OPTIMIZATION_RESULTS.md')
    args = parser.parse_args()
    source = ROOT/args.input
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    rows_path = source/'rows.jsonl'
    rows_raw = rows_path.read_bytes() if rows_path.exists() else gzip.decompress((source/'rows.jsonl.gz').read_bytes())
    rows = [json.loads(line) for line in rows_raw.decode('utf-8').splitlines()]
    refs_path = source/'references.json'
    refs_raw = refs_path.read_bytes() if refs_path.exists() else gzip.decompress((source/'references.json.gz').read_bytes())
    refs = json.loads(refs_raw)
    analysis = summarize_results(rows, refs)
    if analysis != json.loads((source/'analysis.json').read_text(encoding='utf-8')):
        raise ValueError('Independent analysis rebuild differs from server')
    info = json.loads((source/'provenance.json').read_text(encoding='utf-8'))
    if info['status'] != 'completed':
        raise ValueError('Study did not complete')
    for row in rows:
        if any(r.get('tree_nodes', 0) > 32 or r.get('relaxed_tokens', 0) > 2 for r in row['rounds']):
            raise ValueError('Exceeded fixed node/exception cap')
        if row['arm'] in ('TokenCap-ASD', 'GateCap-ASD'):
            bound = row['asd']['branch_drift']['max_token_regret']
            if any(r.get('regret', 0) > bound*r.get('relaxed_tokens', 0)+1e-7 for r in row['rounds']):
                raise ValueError('Exceeded per-token aggregate regret bound')
    (out/'rows.jsonl.gz').write_bytes(compressed(rows_raw))
    (out/'references.json.gz').write_bytes(compressed(refs_raw))
    for name in ('tasks.json', 'provenance.json', 'cost-table.json'):
        if (source/name).resolve() != (out/name).resolve():
            shutil.copyfile(source/name, out/name)
    write(out/'analysis.json', analysis)
    audit = dict(timed_generations=len(rows), matched_inputs=len(refs),
        eager_lazy_decision_pairs=sum(r['arm'] == 'LazyDrift-ASD' for r in rows),
        eager_lazy_tokens_paths_stop_and_budget_equal=True, input_snapshot_and_matrix_valid=True,
        server_analysis_rebuilt_equal=True, max_nodes=32, cap=2,
        raw_regret_check='sum per block <= count*ln4; per-edge rule tested separately')
    write(out/'audit.json', audit)
    lines = ['# 按需JS与门控节点分配：七任务开发检查', '',
        '固定开发诊断子集，单次配对计时。负数更快；自然回答与共同128前缀分别报告。',
        'AR参考用于逐token保真和既定质量检查；开放任务没有新增偏好评分。', '',
        f"完成{len(rows)}计时、{len(refs)}匹配输入；Lazy/eager的{audit['eager_lazy_decision_pairs']}对token、停止、逐轮路径和预算完全一致。", '',
        '|任务|方法|自然 vs ASD|自然 vs Lazy|自然 vs 严格树|AR相同|正确/可评分|截断|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for benchmark, family in analysis['families'].items():
        for arm in ARMS:
            v = family[arm]['natural']
            quality = f"{v['correct']}/{v['scored']}" if v['scored'] else '—'
            lines.append(f"|{benchmark}|{arm}|{percent(v['ASD']['ratio'])}|{percent(v['LazyDrift-ASD']['ratio'])}|{percent(v['BudgetTree']['ratio'])}|{v['identical_ar']}/{v['n']}|{quality}|{v['truncated']}|")
    lines += ['', '## 相同生成量', '',
        '|任务|方法|共同128 vs ASD|有效数|共同128 vs Lazy|有效数|', '|---|---|---:|---:|---:|---:|']
    for benchmark, family in analysis['families'].items():
        for arm in ('LazyDrift-ASD', 'GateTree-ASD', 'TokenCap-ASD', 'GateCap-ASD'):
            v = family[arm]['prefix']
            lines.append(f"|{benchmark}|{arm}|{percent(v['ASD']['ratio'])}|{v['ASD']['n']}|{percent(v['LazyDrift-ASD']['ratio'])}|{v['LazyDrift-ASD']['n']}|")
    lines += ['', '## 检查量与在线成本', '',
        '下表自然回答汇总。gate和JS为CPU取值边界的wall摘要，JS包含在gate中，二者不能相加。构树入口wall可能包含尚未完成的Draft GPU工作；不能全归因于CPU构树，未插入额外同步单独剖析。未记录项用—，全部在线工作均在总请求wall内。', '',
        '|方法|AR相同|近似token|JS边|JS批|跳过边|gate秒|JS秒|平均节点|平均树深|构树入口wall秒|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    totals = {}
    for arm in ARMS:
        fs = [f[arm]['natural'] for f in analysis['families'].values()]
        rounds = [b for r in rows if r['arm'] == arm and r['stage'] == 'natural' for b in r['rounds']]
        counters = {key: sum(f['drift'].get(key, 0) for f in fs) for key in
                    ('compared_edges', 'js_batches', 'skipped_comparisons', 'gate_seconds', 'js_seconds')}
        tree = [b['tree_budget'] for b in rounds if b.get('tree_budget')]
        total = dict(identical_ar=sum(f['identical_ar'] for f in fs),
            relaxed=sum(f['relaxed'] for f in fs), tokens=sum(f['tokens'] for f in fs),
            seconds=sum(f['seconds'] for f in fs), calls=sum(f['calls'] for f in fs), **counters,
            mean_nodes=statistics.mean(b['nodes'] for b in tree) if tree else 0,
            mean_depth=statistics.mean(b['max_depth'] for b in tree) if tree else 0,
            allocation_seconds=sum(b['allocation_seconds'] for b in tree)
                if tree and all('allocation_seconds' in b for b in tree) else None)
        totals[arm] = total
        allocation = '—' if total['allocation_seconds'] is None else f"{total['allocation_seconds']:.3f}"
        lines.append(f"|{arm}|{total['identical_ar']}|{total['relaxed']}|{int(total['compared_edges'])}|{int(total['js_batches'])}|{int(total['skipped_comparisons'])}|{total['gate_seconds']:.3f}|{total['js_seconds']:.3f}|{total['mean_nodes']:.2f}|{total['mean_depth']:.2f}|{allocation}|")
    write(out/'totals.json', totals)
    lines += ['', '## 诊断题与新增失败', '', '|输入|方法|秒|token|正确|AR相同|停止|', '|---|---|---:|---:|---|---|---|']
    for row in rows:
        if row['stage'] == 'natural' and row['request_id'] in ('math500:372', 'math500:101', 'humaneval:76'):
            ref = next(r for r in refs if r['request_id'] == row['request_id'])
            quality = row['quality'].get('secondary_correct', row['quality']['correct'])
            lines.append(f"|{row['request_id']}|{row['arm']}|{row['latency_seconds']:.3f}|{len(row['token_ids'])}|{quality}|{row['token_ids']==ref['token_ids']}|{row['finish_reason']}|")
    lines += ['', '## 复现与证据', '',
        '规则、固定参数与任务选择见[协议](BRANCH_DRIFT_OPTIMIZATION_PROTOCOL.md)。',
        '源码身份、实际命令与恢复结果见[公开证据](../results/branch-drift-optimization/)。',
        '`python -m scripts.report_branch_drift_optimization --input results/branch-drift-optimization`可从压缩发布数据重建。', '']
    (ROOT/args.report).write_bytes('\n'.join(lines).encode('utf-8'))
    write(out/'publication.json', dict(
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        payloads={p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'publication.json'
                  and '__pycache__' not in p.parts and p.suffix != '.pyc'},
        report=dict(path=args.report, sha256=hashlib.sha256((ROOT/args.report).read_bytes()).hexdigest())))
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
