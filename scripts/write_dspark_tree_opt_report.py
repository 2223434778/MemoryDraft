"""Paired timings plus exact tree/output/ledger checks for DSpark optimization."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from memorydraft.backends.asd import ASDConfig
from memorydraft.evaluation.asd_paper import expanded_ids
from memorydraft.evaluation.asd_paper_extension import validate_policy
from scripts.run_dspark_tree_opt import ARMS, COUNTS, REPEATS, selected_records, signature
from scripts.write_dspark_tree_report import read


def summarize(rows,references,expected):
    refs={r['request_id']:r for r in references}
    if set(refs)!=set(expected) or len(refs)!=len(references):raise ValueError('Reference coverage differs')
    groups=defaultdict(list)
    for row in rows:groups[row['request_id'],row['stage'],row['arm']].append(row)
    if set(groups)!={(i,s,a) for i in expected for s in REPEATS for a in ARMS}:raise ValueError('Incomplete method matrix')
    samples=[];exact=0
    for (identity,stage,arm),entries in sorted(groups.items()):
        if sorted(r['repeat'] for r in entries)!=list(range(REPEATS[stage])):raise ValueError('Wrong repeat coverage')
        entries.sort(key=lambda r:r['repeat']);first=entries[0]
        for row in entries:
            if (row['prompt_sha256'],row['snapshot_id'])!=(refs[identity]['prompt_sha256'],refs[identity]['snapshot_id']):
                raise ValueError('Input differs')
            if signature(row)!=signature(first):raise ValueError('Repeat output differs')
            if not math.isfinite(row['latency_seconds']) or row['latency_seconds']<=0:raise ValueError('Invalid latency')
            if stage=='fixed' and len(row['token_ids'])!=256:raise ValueError('Fixed work differs')
            validate_policy(row,ASDConfig(budget=8))
            if row['counters']['compile_graphs_before']!=row['counters']['compile_graphs_after']:
                raise ValueError('Formal compilation occurred')
        if arm=='DSpark-Tree-ASD-Fast':
            baseline=groups[identity,stage,'DSpark-Tree-ASD'][0]
            if signature(first)!=signature(baseline) or first['quality']!=baseline['quality']:
                raise ValueError('Optimized tree/output/quality/ledger differs')
            exact+=len(entries)
        samples.append(dict(request_id=identity,benchmark=first['benchmark'],stage=stage,arm=arm,
            seconds=statistics.mean(r['latency_seconds'] for r in entries),tokens=len(first['token_ids']),
            quality=first['quality'],finish_reason=first['finish_reason'],rounds=len(first['rounds']),
            built_nodes=sum(r.get('tree_budget',{}).get('built_nodes',0) for r in first['rounds']),
            conditional_distributions=sum(r.get('tree_budget',{}).get('conditional_distributions',0) for r in first['rounds']),
            cache=first.get('markov_cache'),peak_bytes=max(r['peak_gpu_bytes'] for r in entries)))
    indexed={(r['request_id'],r['stage'],r['arm']):r for r in samples};comparison=[]
    for benchmark in sorted({r['benchmark'] for r in references}):
        ids=[i for i in expected if refs[i]['benchmark']==benchmark]
        for stage in REPEATS:
            for candidate,baseline in (('DSpark-Tree-ASD-Fast','DSpark-Tree-ASD'),
                    ('DSpark-Tree-ASD-Fast','DFlash-Tree-ASD'),('DSpark-Tree-ASD-Fast','DSpark-ASD')):
                xs=[indexed[i,stage,candidate] for i in ids];ys=[indexed[i,stage,baseline] for i in ids]
                if stage=='prefix':
                    valid=[n for n,(x,y) in enumerate(zip(xs,ys)) if x['tokens']==y['tokens']==128]
                    xs=[xs[n] for n in valid];ys=[ys[n] for n in valid]
                comparison.append(dict(benchmark=benchmark,stage=stage,candidate=candidate,baseline=baseline,n=len(xs),
                    mean_time_ratio=statistics.mean(x['seconds']/y['seconds'] for x,y in zip(xs,ys)) if xs else None,
                    total_time_ratio=sum(x['seconds'] for x in xs)/sum(y['seconds'] for y in ys) if xs else None,
                    candidate_tps=sum(x['tokens'] for x in xs)/sum(x['seconds'] for x in xs) if xs else None,
                    baseline_tps=sum(y['tokens'] for y in ys)/sum(y['seconds'] for y in ys) if xs else None,
                    candidate_tokens=sum(x['tokens'] for x in xs),baseline_tokens=sum(y['tokens'] for y in ys),
                    candidate_truncated=sum(x['finish_reason']=='length' for x in xs),
                    baseline_truncated=sum(y['finish_reason']=='length' for y in ys)))
    return dict(requests=len(refs),timed_rows=len(rows),exact_fast_reference_generations=exact,
                samples=samples,comparisons=comparison)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    result=summarize(read(args.input,'rows'),read(args.input,'references'),expanded_ids(selected_records(Path.cwd())))
    result['provenance']=[json.loads((args.input/f'part-{i}/provenance.json').read_bytes()) for i in (0,1)]
    if any(p['status']!='complete' for p in result['provenance']):raise ValueError('Incomplete workers')
    replay=read(args.input,'mechanism');result['mechanism']=[]
    for benchmark in sorted({r['benchmark'] for r in replay}):
        for variant in ('reference','cache','bound','both'):
            chosen=[r for r in replay if r['benchmark']==benchmark and r['variant']==variant and not r['warmup']]
            if len(chosen)!=6 or any(not r['exact'] for r in chosen):raise ValueError('Incomplete mechanism replay')
            result['mechanism'].append(dict(benchmark=benchmark,variant=variant,
                mean_seconds=statistics.mean(r['seconds'] for r in chosen),states=sum(r['states'] for r in chosen),
                built_nodes=sum(t['built_nodes'] for r in chosen for t in r['trees']),
                conditionals=sum(t['conditional_distributions'] for r in chosen for t in r['trees']),
                cache_hits=sum(r['cache']['hits'] if r['cache'] else 0 for r in chosen),
                cache_misses=sum(r['cache']['misses'] if r['cache'] else 0 for r in chosen)))
    (args.input/'analysis.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    lines=['# DSpark 成本树等价优化：配对复测','',
        f"{result['requests']} 匹配输入、{result['timed_rows']} 次主计时；优化/原版 {result['exact_fast_reference_generations']} 次生成的完整选树摘要、token、文本、停止及ASD账本一致。",
        f'源任务数 {COUNTS}。已观察开发回放，Alpaca中12题首次用于DSpark；不能视为全新保留集。',
        '每请求冷LRU64行缓存、保守上界停搜；同FP32 GEMV形状。旧每卡树成本表冻结，原生7/15候选，B8/g.25/M2与Target输入不变。',
        '所有方法本轮实测；旧AR仅复用输入与MT历史，其时延不进入对比。自然时延用逐请求比均值，固定TPS用总token/总秒。','',
        '|任务|对照|自然时延变化|固定256 TPS变化|共同128时延变化（N）|',
        '|---|---|---:|---:|---|']
    lookup={(r['benchmark'],r['stage'],r['baseline']):r for r in result['comparisons']}
    for benchmark in sorted(COUNTS):
        for baseline in ('DSpark-Tree-ASD','DFlash-Tree-ASD','DSpark-ASD'):
            n=lookup[benchmark,'natural',baseline];f=lookup[benchmark,'fixed',baseline];p=lookup[benchmark,'prefix',baseline]
            prefix=f"{100*(p['mean_time_ratio']-1):+.2f}%（{p['n']}）" if p['n'] else '无共同128'
            lines.append(f"|{benchmark}|{baseline}|{100*(n['mean_time_ratio']-1):+.2f}%|{100*(f['candidate_tps']/f['baseline_tps']-1):+.2f}%|{prefix}|")
    lines+=['','## 真实GPU状态消融（只计构树，不代替端到端）','',
        '七族各2源任务，每请求取前≤4个真实状态；每变体/重复冷缓存，首遍不计，余3遍交替顺序。无logits/特征导出。','',
        '|任务|仅缓存|仅上界|二者|','|---|---:|---:|---:|']
    for benchmark in sorted({r['benchmark'] for r in result['mechanism']}):
        values={r['variant']:r['mean_seconds'] for r in result['mechanism'] if r['benchmark']==benchmark}
        changes=[f"{100*(values[v]/values['reference']-1):+.2f}%" for v in ('cache','bound','both')]
        lines.append('|'+benchmark+'|'+'|'.join(changes)+'|')
    lines+=['','缓存、构树停搜、总内存与逐题质量保留在analysis及原始行。工程优化不改变原DSpark树质量；不同Draft间仍可能有ASD质量/长度取舍。',
        'HumanEval仅4源任务为边界检查；MT为固定共同历史的8匹配单轮，不能等同自身历史对话。此次没有长块DSpark重训或ASD阈值修改。','']
    (args.input/'report.md').write_bytes(('\n'.join(lines)).encode())
    print(json.dumps({k:result[k] for k in ('requests','timed_rows','exact_fast_reference_generations')}))


if __name__=='__main__':main()
