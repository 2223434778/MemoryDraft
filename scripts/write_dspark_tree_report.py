"""Validate and summarize all eight frozen methods, retaining negative cases."""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_comparison import bootstrap
from memorydraft.evaluation.asd_paper import BENCHMARKS, expanded_ids
from memorydraft.evaluation.asd_paper_extension import load_records
from memorydraft.evaluation.asd_paper_metrics import acceptance
from scripts.run_dspark_tree import ARMS, REPEATS


def read(directory,name):
    rows=[]
    for shard in (0,1):
        path=directory/f'part-{shard}/{name}.jsonl'
        raw=path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix('.jsonl.gz').read_bytes())
        rows.extend(json.loads(line) for line in raw.decode().splitlines())
    return rows


def summarize(rows,references,expected):
    refs={r['request_id']:r for r in references}
    if set(refs)!=set(expected) or len(refs)!=len(references):raise ValueError('Reference coverage differs')
    groups=defaultdict(list)
    for row in rows:groups[row['request_id'],row['stage'],row['arm']].append(row)
    if set(groups)!={(identity,stage,arm) for identity in expected for stage in REPEATS for arm in ARMS}:
        raise ValueError('Incomplete or extra method coverage')
    samples=[]
    for (identity,stage,arm),entries in sorted(groups.items()):
        if sorted(r['repeat'] for r in entries)!=list(range(REPEATS[stage])):raise ValueError('Repeat coverage differs')
        if any((r['prompt_sha256'],r['snapshot_id'])!=(refs[identity]['prompt_sha256'],refs[identity]['snapshot_id']) for r in entries):
            raise ValueError('Mismatched input')
        entries.sort(key=lambda r:r['repeat']);row=entries[0]
        if any((r['token_ids'],r['rounds'],r['asd'],r['finish_reason'])!=(row['token_ids'],row['rounds'],row['asd'],row['finish_reason']) for r in entries):
            raise ValueError('Unstable repeat')
        if any(not math.isfinite(r['latency_seconds']) or r['latency_seconds']<=0 for r in entries):raise ValueError('Invalid latency')
        if stage=='fixed' and any(len(r['token_ids'])!=256 for r in entries):raise ValueError('Wrong fixed work')
        samples.append(dict(request_id=identity,benchmark=row['benchmark'],stage=stage,arm=arm,
            seconds=statistics.mean(r['latency_seconds'] for r in entries),tokens=len(row['token_ids']),
            quality=row['quality'],finish_reason=row['finish_reason'],acceptance=acceptance(row) if row['rounds'] else None,
            peak_bytes=max(r['peak_gpu_bytes'] for r in entries)))
    pairs=[('DSpark-Tree-ASD','DFlash-Tree-ASD'),('DSpark','DFlash'),('DSpark-ASD','DFlash-ASD'),
           ('DSpark-Tree','DSpark'),('DSpark-Tree-ASD','DSpark-ASD'),('DSpark-Tree-ASD','DSpark-Tree'),
           ('DFlash-Tree-ASD','DFlash-ASD')]
    indexed={(r['request_id'],r['stage'],r['arm']):r for r in samples};comparison=[]
    for benchmark in BENCHMARKS:
        ids=[i for i in expected if refs[i]['benchmark']==benchmark]
        for stage in REPEATS:
            for candidate,baseline in pairs:
                xs=[indexed[i,stage,candidate] for i in ids];ys=[indexed[i,stage,baseline] for i in ids]
                if stage=='prefix':
                    valid=[i for i,(x,y) in enumerate(zip(xs,ys)) if x['tokens']==y['tokens']==128]
                    xs=[xs[i] for i in valid];ys=[ys[i] for i in valid]
                ratios=[x['seconds']/y['seconds'] for x,y in zip(xs,ys)]
                quality=lambda rs:sum(bool(r['quality'].get('secondary_correct',r['quality']['correct'])) for r in rs if r['quality'] and r['quality']['correct'] is not None)
                scored=sum(r['quality'] is not None and r['quality']['correct'] is not None for r in xs)
                metric=dict(benchmark=benchmark,stage=stage,candidate=candidate,baseline=baseline,n=len(xs),
                    mean_time_ratio=statistics.mean(ratios) if ratios else None,ci=bootstrap(ratios) if ratios else None,
                    candidate_tps=sum(x['tokens'] for x in xs)/sum(x['seconds'] for x in xs) if xs else None,
                    baseline_tps=sum(y['tokens'] for y in ys)/sum(y['seconds'] for y in ys) if ys else None,
                    candidate_tokens=sum(x['tokens'] for x in xs),baseline_tokens=sum(y['tokens'] for y in ys),
                    candidate_correct=quality(xs) if scored else None,baseline_correct=quality(ys) if scored else None,scored=scored,
                    candidate_truncated=sum(x['finish_reason']=='length' for x in xs),baseline_truncated=sum(y['finish_reason']=='length' for y in ys))
                comparison.append(metric)
    return dict(requests=len(refs),timed_rows=len(rows),samples=samples,comparisons=comparison)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    records=load_records(Path.cwd());selected=[r for b in BENCHMARKS for r in [x for x in records if x['benchmark']==b][:4]]
    result=summarize(read(args.input,'rows'),read(args.input,'references'),expanded_ids(selected))
    result['provenance']=[json.loads((args.input/f'part-{i}/provenance.json').read_bytes()) for i in (0,1)]
    if any(p['status']!='complete' for p in result['provenance']):raise ValueError('Incomplete workers')
    (args.input/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# DSpark + 成本树 + ASD：八组配对回放','',
        f"完成 {result['requests']} 个匹配输入、{result['timed_rows']} 次主计时。每族4个已观察源任务，为开发回放比较。",
        'DSpark原生7候选；旧DFlash原生15候选。相同Target/FP32/eager/RMSNorm、B8/g.25/M2；置信度裁剪关闭。',
        '自然时延为逐请求配对比均值，固定TPS为总token/总耗时；分别报告，开放质量未另用裁判评估。','',
        '|任务|自然时延变化：DSpark组合 vs DFlash组合|95%区间（时延比）|原评分正确数：DFlash→DSpark|截断数：DFlash→DSpark|',
        '|---|---:|---|---|---|']
    for r in result['comparisons']:
        if r['stage']=='natural' and r['candidate']=='DSpark-Tree-ASD' and r['baseline']=='DFlash-Tree-ASD':
            q=f"{r['baseline_correct']}→{r['candidate_correct']}/{r['scored']}" if r['scored'] else '开放任务未独立评分'
            lines.append(f"|{r['benchmark']}|{(r['mean_time_ratio']-1)*100:+.2f}%|{r['ci']}|{q}|{r['baseline_truncated']}→{r['candidate_truncated']}|")
    lines+=['','## 每种Draft内部，成本树的增量','',
        '|任务|Draft|树+ASD vs 线性ASD自然时延变化|固定TPS变化|共同128时延变化（N）|',
        '|---|---|---:|---:|---|']
    lookup={(r['benchmark'],r['stage'],r['candidate'],r['baseline']):r for r in result['comparisons']}
    for b in BENCHMARKS:
        for draft in ('DFlash','DSpark'):
            natural=lookup[b,'natural',draft+'-Tree-ASD',draft+'-ASD']
            fixed=lookup[b,'fixed',draft+'-Tree-ASD',draft+'-ASD'];prefix=lookup[b,'prefix',draft+'-Tree-ASD',draft+'-ASD']
            p=f"{(prefix['mean_time_ratio']-1)*100:+.2f}%（{prefix['n']}）" if prefix['n'] else '无共同128'
            lines.append(f"|{b}|{draft}|{(natural['mean_time_ratio']-1)*100:+.2f}%|{(fixed['candidate_tps']/fixed['baseline_tps']-1)*100:+.2f}%|{p}|")
    lines+=['','主原始行、8组自然质量、接受长度/轮次、固定工作量及严格输出差异全部保留于analysis/provenance。',
        'MT为统一AR历史的32请求矩阵中的8个匹配轮次，不能当作各方法自身历史的对话级结论。',
        '本轮模型/原生块长均变化，不能将底座更换收益单独归于Markov头。旧DFlash分支、权重和结果未覆盖。','']
    (args.input/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(requests=result['requests'],timed_rows=result['timed_rows'],comparisons=len(result['comparisons']))))


if __name__=='__main__':main()
