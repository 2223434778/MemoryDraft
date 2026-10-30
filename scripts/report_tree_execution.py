"""Publish the direct execution comparison without hiding output differences."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from scripts.report_branch_drift_optimization import compressed, percent, write
from scripts.run_tree_execution import ARMS, summarize_results

ROOT = Path(__file__).resolve().parents[1]
ROUND_FIELDS = ('generated_before', 'accepted', 'advance', 'tree_nodes', 'tree_path',
                'strict_accepted', 'relaxed_tokens', 'regret', 'pending_before', 'pending_after', 'target_cache_after')
META_FIELDS = ('gate_penalty', 'min_extra_tokens', 'min_gain_candidate_extra', 'min_gain_strict_fallback',
               'width_recoverable_miss', 'width_next_penalty')


def trajectory(row):
    return [({k:b.get(k) for k in ROUND_FIELDS}, {k:b['tree_budget'].get(k) for k in META_FIELDS})
            for b in row['rounds']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/raw/tree-execution')
    args = parser.parse_args()
    source, out = ROOT/args.input, ROOT/'results/tree-execution'
    out.mkdir(parents=True, exist_ok=True)
    def raw(name):
        return (source/name).read_bytes() if (source/name).exists() else gzip.decompress((source/(name+'.gz')).read_bytes())
    row_bytes, ref_bytes = raw('rows.jsonl'), raw('references.json')
    rows = [json.loads(s) for s in row_bytes.decode('utf-8').splitlines()]
    refs = json.loads(ref_bytes)
    analysis = summarize_results(rows, refs)
    if analysis != json.loads((source/'analysis.json').read_bytes()):
        raise ValueError('Server analysis mismatch')
    if len(rows) != 198 or len(refs) != 66 or json.loads((source/'provenance.json').read_bytes())['status'] != 'completed':
        raise ValueError('Incomplete direct comparison')
    (out/'rows.jsonl.gz').write_bytes(compressed(row_bytes))
    (out/'references.json.gz').write_bytes(compressed(ref_bytes))
    for name in ('tasks.json', 'provenance.json', 'cost-table.json', 'execution.json',
                 'source-manifest.json', 'server-tests.xml', 'local-tests.xml', 'transfer.json'):
        if (source/name).exists() and (source/name).resolve() != (out/name).resolve():
            shutil.copyfile(source/name, out/name)
    write(out/'analysis.json', analysis)
    indexed = {(r['request_id'], r['arm']):r for r in rows}
    audit = dict(output_differences=analysis['output_differences'], trajectory_differences=[], ledger_differences=[])
    for r in rows:
        if r['arm'] == ARMS[0]:
            continue
        baseline = indexed[r['request_id'], ARMS[0]]
        for field, difference in (('trajectory_differences', trajectory(r) != trajectory(baseline)),
                                  ('ledger_differences', r['asd']['spent'] != baseline['asd']['spent'])):
            if difference:
                audit[field].append(dict(id=r['request_id'], arm=r['arm']))
    previous_path = ROOT/'results/gate-min-gain/rows.jsonl.gz'
    if previous_path.exists():
        previous = {r['request_id']:r for r in (json.loads(s) for s in gzip.decompress(previous_path.read_bytes()).decode().splitlines())
                    if r['arm'] == ARMS[0]}
        audit['previous_baseline_differences'] = [r['request_id'] for r in rows if r['arm'] == ARMS[0]
            and any(r[k] != previous[r['request_id']][k] for k in ('token_ids','text','finish_reason','prompt_sha256','snapshot_id'))]
    write(out/'audit.json', audit)
    totals = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        totals[arm] = dict(seconds=sum(r['latency_seconds'] for r in selected),
            identical_ar=sum(analysis['families'][b][arm]['natural']['identical_ar'] for b in analysis['families']),
            tokens=sum(len(r['token_ids']) for r in selected), calls=sum(len(r['rounds']) for r in selected),
            exceptions=sum(r['asd']['relaxed_tokens'] for r in selected),
            peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in selected),
            kv_allocations=sum(c['allocations'] for r in selected for c in r.get('cache_statistics', {}).values()),
            kv_updates=sum(c['updates'] for r in selected for c in r.get('cache_statistics', {}).values()),
            history_copy_bytes=sum(c.get('history_copy_bytes', 0) for r in selected for c in r.get('cache_statistics', {}).values()))
    write(out/'totals.json', totals)
    lines = ['# 树验证执行优化：直接对比结果', '',
        '同一高保真接受策略、七任务66匹配输入、三组198次自然4096/EOS计时。各一次，负数表示更快。', '',
        '|任务|方法|配对耗时 vs 原版|配对耗时 vs 仅去重|AR完全一致|正确/可评分|截断|',
        '|---|---|---:|---:|---:|---:|---:|']
    for benchmark, family in analysis['families'].items():
        for arm in ARMS:
            v = family[arm]['natural']
            quality = f"{v['correct']}/{v['scored']}" if v['scored'] else '—'
            lines.append(f"|{benchmark}|{arm}|{percent(v[ARMS[0]]['ratio'])}|{percent(v[ARMS[1]]['ratio'])}|{v['identical_ar']}/{v['n']}|{quality}|{v['truncated']}|")
    lines += ['', '## 总量及一致性', '', '|方法|总秒|总wall vs 原版|AR完全一致|输出token|验证轮|近似token|峰allocated GiB|',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, v in totals.items():
        lines.append(f"|{arm}|{v['seconds']:.3f}|{percent(v['seconds']/totals[ARMS[0]]['seconds'])}|{v['identical_ar']}/66|{v['tokens']}|{v['calls']}|{v['exceptions']}|{v['peak_gpu_bytes']/2**30:.2f}|")
    lines += ['', f"132个优化/原版配对：输出差异{len(audit['output_differences'])}，逐轮路径/接受/反馈/原始regret轨迹差异{len(audit['trajectory_differences'])}，累计账本差异{len(audit['ledger_differences'])}。",
        '总wall比与逐题配对均值为不同统计，不相除推导其他对照。分配、写入、扩容、选择和释放开销计在线；模型加载/预热单列。',
        f"缓存组KV更新{totals[ARMS[2]]['kv_updates']}次、分配{totals[ARMS[2]]['kv_allocations']}次；扩容时历史复制{totals[ARMS[2]]['history_copy_bytes']} bytes。这不是与旧动态缓存实测流量的直接对照，不据此推断带宽瓶颈占比。",
        '准确度以输出保持为本轮要求；出现输出差异的选项不作为保持输出的优化接纳。差异详情保留在audit.json。']
    execution = json.loads((out/'execution.json').read_bytes())
    manifest = json.loads((out/'source-manifest.json').read_bytes())
    runtime = json.loads((out/'provenance.json').read_bytes())
    lines += ['', '## 执行记录', '',
        f"实际GPU源码commit `{manifest['commit']}`，{execution['source_files_verified_before']}文件运行前后SHA核验。实际ZIP SHA `{execution['archive_sha256']}`。",
        f"生成{runtime['started_utc']}至{runtime['finished_utc']}，加载{runtime['load_seconds']:.2f}秒，峰allocated {runtime['peak_gpu_bytes']} bytes。",
        '目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/tree-execution-v1`，GPU1，原Qwen3-8B/DFlash、FP32/eager/RMSNorm、成本表。',
        '命令：`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_tree_execution`。',
        f"服务恢复退出码：{execution['restore_exit_codes']}。本机76 passed/2 skipped/14.71s；服务器78 passed/8.72s。",
        '旧目录与线上默认保持；报告为生成后的离线后处理，不冒充冻结GPU源码。', '',
        '开发输入非新保留集；MMLU仍为字母格式合同，开放聊天未新增偏好评分。',
        '协议：[TREE_EXECUTION_PROTOCOL.md](TREE_EXECUTION_PROTOCOL.md)。[公开数据](../results/tree-execution/)。',
        '复现：`python -m scripts.report_tree_execution --input results/tree-execution`。', '']
    report = ROOT/'docs/TREE_EXECUTION_RESULTS.md'
    report.write_bytes('\n'.join(lines).encode('utf-8'))
    write(out/'publication.json', dict(reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        payloads={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
                  if p.is_file() and p.name != 'publication.json'}))
    print(json.dumps(dict(totals=totals, audit=audit), indent=2))


if __name__ == '__main__':
    main()
