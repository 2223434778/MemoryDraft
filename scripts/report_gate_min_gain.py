"""Summarize the direct three-arm natural-answer minimum-gain run."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from scripts.report_branch_drift_optimization import compressed, percent, write
from scripts.run_gate_min_gain import ARMS, summarize_results

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='results/raw/gate-min-gain')
    args = parser.parse_args()
    source, out = ROOT/args.input, ROOT/'results/gate-min-gain'
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
    totals = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        totals[arm] = dict(seconds=sum(r['latency_seconds'] for r in selected),
            identical_ar=sum(analysis['families'][b][arm]['natural']['identical_ar'] for b in analysis['families']),
            tokens=sum(len(r['token_ids']) for r in selected), calls=sum(len(r['rounds']) for r in selected),
            exceptions=sum((r['asd'] or {}).get('relaxed_tokens', 0) for r in selected),
            fallbacks=sum((r['asd'] or {}).get('min_gain', {}).get('strict_fallbacks', 0) for r in selected),
            foregone_tokens=sum((r['asd'] or {}).get('min_gain', {}).get('foregone_tokens', 0) for r in selected))
    write(out/'totals.json', totals)
    old = ROOT/'results/gate-width/rows.jsonl.gz'
    baseline_check = None
    if old.exists():
        previous = {(r['request_id'], r['stage'], r['arm']): r for r in
                    (json.loads(s) for s in gzip.decompress(old.read_bytes()).decode().splitlines())}
        controls = [r for r in rows if r['arm'] != ARMS[2]]
        baseline_check = dict(pairs=len(controls), differences=[dict(id=r['request_id'], arm=r['arm']) for r in controls
            if any(r[k] != previous[r['request_id'], r['stage'], r['arm']][k]
                   for k in ('token_ids', 'text', 'finish_reason', 'prompt_sha256', 'snapshot_id'))])
        write(out/'baseline-identity.json', baseline_check)
    lines = ['# 最小增量接受：直接对比结果', '',
        '七任务66匹配输入，三组198次自然4096/EOS计时。沿用既有开发输入，各一次，负数表示更快。',
        '只增加“近似路径至少领先同树严格路径2个token”的提交条件；预算、KV、反馈均以实际提交路径更新。', '',
        '|任务|方法|配对耗时 vs ASD|配对耗时 vs 原反馈|AR完全一致|正确/可评分|截断|',
        '|---|---|---:|---:|---:|---:|---:|']
    for benchmark, family in analysis['families'].items():
        for arm in ARMS:
            v = family[arm]['natural']
            quality = f"{v['correct']}/{v['scored']}" if v['scored'] else '—'
            lines.append(f"|{benchmark}|{arm}|{percent(v['ASD']['ratio'])}|{percent(v[ARMS[1]]['ratio'])}|{v['identical_ar']}/{v['n']}|{quality}|{v['truncated']}|")
    lines += ['', '## 混合请求总量', '', '|方法|总秒|AR完全一致|输出token|验证轮|近似token|回退轮|本轮放弃推进token|',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, v in totals.items():
        lines.append(f"|{arm}|{v['seconds']:.3f}|{v['identical_ar']}/66|{v['tokens']}|{v['calls']}|{v['exceptions']}|{v['fallbacks']}|{v['foregone_tokens']}|")
    new, previous = totals[ARMS[2]], totals[ARMS[1]]
    lines += ['', f"新方案总wall相对原反馈{percent(new['seconds']/previous['seconds'])}，相对ASD{percent(new['seconds']/totals['ASD']['seconds'])}。总时间比区别于逐题配对均值，包含生成路径/长度变化。",
        f"预定速度代价≤3%：{new['seconds']/previous['seconds'] <= 1.03}；AR完全一致数增加：{new['identical_ar'] > previous['identical_ar']}。", '',
        '## 正确集合变化', '', '|任务|新方案较原反馈新增正确|新方案较原反馈失去正确|', '|---|---|---|']
    for benchmark, family in analysis['families'].items():
        v = family[ARMS[2]]['natural'][ARMS[1]]
        lines.append(f"|{benchmark}|{', '.join(v['gained']) or '—'}|{', '.join(v['lost']) or '—'}|")
    indexed = {r['request_id']: r for r in refs}
    lines += ['', '## 既有关键题', '', '|输入|方法|秒|token|正确|AR完全一致|停止|', '|---|---|---:|---:|---|---|---|']
    for r in rows:
        if r['request_id'] in ('math500:101', 'math500:372', 'gsm8k:750'):
            q = r['quality'].get('secondary_correct', r['quality']['correct'])
            lines.append(f"|{r['request_id']}|{r['arm']}|{r['latency_seconds']:.3f}|{len(r['token_ids'])}|{q}|{r['token_ids']==indexed[r['request_id']]['token_ids']}|{r['finish_reason']}|")
    if baseline_check:
        lines += ['', f"与上一轮比较的{baseline_check['pairs']}个原基线输出/输入身份，差异{len(baseline_check['differences'])}；旧时延未参与本轮配对。"]
    execution = json.loads((out/'execution.json').read_bytes())
    manifest = json.loads((out/'source-manifest.json').read_bytes())
    runtime = json.loads((out/'provenance.json').read_bytes())
    lines += ['', '## 执行记录', '',
        f"实际GPU源码commit `{manifest['commit']}`，{execution['source_files_verified_before']}文件运行前后SHA核验。实际ZIP SHA `{execution['archive_sha256']}`。",
        f"生成时间{runtime['started_utc']}至{runtime['finished_utc']}，加载{runtime['load_seconds']:.2f}秒单列，峰allocated {runtime['peak_gpu_bytes']} bytes。",
        '目录`/media/a6000/D/ZhaoBinHe/MemoryDraft/gate-min-gain-v1`，GPU1，原Qwen3-8B/DFlash、FP32/eager/RMSNorm、成本表。',
        '命令：`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=. ../.venv-dflash/bin/python -u -m scripts.run_gate_min_gain`。',
        f"服务恢复退出码：{execution['restore_exit_codes']}。本机63 passed/14.41s；服务器63 passed/5.20s。",
        '原反馈版与新方案均保留，未变更线上默认。报告脚本为生成后的离线后处理，不冒充冻结GPU源码。']
    lines += ['', 'MMLU沿用字母格式合同；开放聊天未新增偏好评分。AR一致不等同正确率，单次开发计时不作普遍保证。',
        '协议：[GATE_MIN_GAIN_PROTOCOL.md](GATE_MIN_GAIN_PROTOCOL.md)。[公开数据](../results/gate-min-gain/)。',
        '复现：`python -m scripts.report_gate_min_gain --input results/gate-min-gain`。', '']
    report = ROOT/'docs/GATE_MIN_GAIN_RESULTS.md'
    report.write_bytes('\n'.join(lines).encode('utf-8'))
    write(out/'publication.json', dict(reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        payloads={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
                  if p.is_file() and p.name != 'publication.json'}))
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
