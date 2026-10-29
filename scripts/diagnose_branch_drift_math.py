"""Attribute existing MATH-500 latency; never regenerate or filter benchmark rows."""
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_paper import quality_ok
from memorydraft.evaluation.asd_paper_metrics import acceptance

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('DFlash','ASD','BudgetTree','BudgetTree-ASD','CoverageTree-ASD','DriftTree-ASD')
FOCAL = 'math500:372'


def factors(candidate, baseline):
    """An exact accounting identity, not independent causal contributions."""
    for row in (candidate,baseline):
        if any(not math.isfinite(row[k]) or row[k]<=0 for k in ('seconds','tokens','rounds')):
            raise ValueError('Positive finite time, output count and rounds required')
    length = candidate['tokens']/baseline['tokens']
    density = (candidate['rounds']/candidate['tokens'])/(baseline['rounds']/baseline['tokens'])
    cycle = (candidate['seconds']/candidate['rounds'])/(baseline['seconds']/baseline['rounds'])
    ratio = candidate['seconds']/baseline['seconds']
    if not math.isclose(length*density*cycle,ratio,rel_tol=1e-12):
        raise ValueError('Invalid latency factorization')
    return dict(latency_ratio=ratio,length_ratio=length,rounds_per_token_ratio=density,
        amortized_wall_per_round_ratio=cycle,product=length*density*cycle)


def periodic_tail(tokens,max_period=512,min_repeats=3):
    """Find the longest exactly periodic suffix, including a partial final cycle."""
    if max_period<1 or min_repeats<2:
        raise ValueError('Invalid repeat bounds')
    n=len(tokens);best=None
    for period in range(1,min(max_period,n//min_repeats)+1):
        i=n-1
        while i>=period and tokens[i]==tokens[i-period]:
            i-=1
        matched=n-1-i
        covered=matched+period
        if matched>=(min_repeats-1)*period and (best is None or covered>best['tokens']):
            best=dict(period=period,tokens=covered,start_1based=n-covered+1,
                      complete_cycles=covered//period,trailing_partial=covered%period)
    return best


def exceptions(row):
    windows=[]
    for index,block in enumerate(row['rounds'],1):
        count=block.get('relaxed_tokens',0)
        if not count:
            continue
        if not 0<count<=block['accepted']:
            raise ValueError('Exception outside accepted proposal prefix')
        windows.append(dict(round=index,possible_position_1based=[block['generated_before']+1,
            block['generated_before']+block['accepted']],count=count,regret=block['regret'],
            candidate_to_argmax_ratio=math.exp(-block['regret']) if count==1 else None))
    return windows


def summarize_row(row):
    measured=acceptance(row)
    return dict(seconds=row['latency_seconds'],tokens=len(row['token_ids']),rounds=len(row['rounds']),
        tps=len(row['token_ids'])/row['latency_seconds'],finish_reason=row['finish_reason'],
        correct=quality_ok(row['quality']) if row['quality'] else None,
        relaxed=(row['asd'] or {}).get('relaxed_tokens',0),spent=(row['asd'] or {}).get('spent',0),
        emitted_draft_per_round=measured['emitted_draft']/measured['rounds'],
        amortized_wall_per_round_ms=1000*row['latency_seconds']/measured['rounds'])


def main():
    source=ROOT/'results/branch-drift';out=ROOT/'results/branch-drift-math-diagnosis'
    published=json.loads((source/'publication.json').read_bytes())['files']
    all_fingerprints={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in published}
    if any(digest!=published[n]['sha256'] for n,digest in all_fingerprints.items()):
        raise ValueError('Original publication payload changed')
    fingerprints={}
    for name in ('rows.jsonl.gz','tasks.json','analysis.json'):
        digest=hashlib.sha256((source/name).read_bytes()).hexdigest()
        if digest!=published[name]['sha256']:
            raise ValueError('Published source changed: '+name)
        fingerprints[name]=digest
    rows=[json.loads(line) for line in gzip.decompress((source/'rows.jsonl.gz').read_bytes()).decode().splitlines()]
    tasks={r['id']:r for r in json.loads((source/'tasks.json').read_bytes()) if r['benchmark']=='math500'}
    groups=defaultdict(dict)
    for row in rows:
        if row['benchmark']=='math500' and row['stage']=='natural':
            if row['arm'] in groups[row['request_id']]:
                raise ValueError('Duplicate natural measurement')
            groups[row['request_id']][row['arm']]=row
    if len(groups)!=12 or set(groups)!=set(tasks) or any(set(v)!=set(ARMS) for v in groups.values()):
        raise ValueError('Incomplete frozen math comparison')
    totals={}
    for arm in ARMS:
        selected=[v[arm] for v in groups.values()]
        totals[arm]=dict(seconds=sum(r['latency_seconds'] for r in selected),
            tokens=sum(len(r['token_ids']) for r in selected),rounds=sum(len(r['rounds']) for r in selected))
        totals[arm]['tps']=totals[arm]['tokens']/totals[arm]['seconds']
    delta=totals['DriftTree-ASD']['seconds']-totals['DFlash']['seconds'];cases=[]
    for identity,arms in groups.items():
        values={a:summarize_row(r) for a,r in arms.items()}
        extra=values['DriftTree-ASD']['seconds']-values['DFlash']['seconds']
        cases.append(dict(request_id=identity,arms=values,extra_seconds=extra,
            share_of_net_extra_seconds=extra/delta,factors=factors(values['DriftTree-ASD'],values['DFlash'])))
    cases.sort(key=lambda v:v['extra_seconds'],reverse=True)
    ratios=[c['factors']['latency_ratio'] for c in cases]
    official=json.loads((source/'analysis.json').read_bytes())['families']['math500']['DriftTree-ASD']['natural']['DFlash']['ratio']
    if not math.isclose(statistics.mean(ratios),official,rel_tol=1e-12):
        raise ValueError('Official paired mean changed')
    row=groups[FOCAL]['DriftTree-ASD'];baseline=groups[FOCAL]['DFlash'];windows=exceptions(row)
    tail=periodic_tail(row['token_ids'])
    first=next((i+1 for i,(a,b) in enumerate(zip(row['token_ids'],baseline['token_ids'])) if a!=b),None)
    if first is None or not windows[0]['possible_position_1based'][0]<=first<=windows[0]['possible_position_1based'][1]:
        raise ValueError('First divergence is not inside the first logged exception window')
    focal=dict(request_id=FOCAL,problem=tasks[FOCAL]['problem'],expected_answer=tasks[FOCAL]['answer'],
        periodic_tail=tail,exception_windows=windows,first_difference_from_ar_1based=first,
        selected_js_sum=row['asd']['branch_drift']['selected_js_sum'],
        target_greedy_tokens_after_last_exception_lower_bound=len(row['token_ids'])-windows[-1]['possible_position_1based'][1],
        tail_excerpt=row['text'][-900:],baseline_tail=baseline['text'][-300:],
        limitation='Per-edge alternative token/logits are not stored; no individual-exception causal replay performed')
    same=[]
    for identity,arms in groups.items():
        candidate=arms['DriftTree-ASD']
        if not candidate['asd']['relaxed_tokens']:
            if candidate['token_ids']!=arms['DFlash']['token_ids']:
                raise ValueError('Unexpected zero-exception identity difference')
            same.append(identity)
    controls={baseline_arm:dict(n=len(same),mean_latency_ratio=statistics.mean(
        groups[i]['DriftTree-ASD']['latency_seconds']/groups[i][baseline_arm]['latency_seconds'] for i in same))
        for baseline_arm in ('DFlash','BudgetTree')}
    diagnostic=dict(source_files_sha256=fingerprints,source_payloads_checked=len(all_fingerprints),
        source_publication_sha256=hashlib.sha256((source/'publication.json').read_bytes()).hexdigest(),
        requests=12,totals=totals,cases=cases,
        paired_mean_ratio=official,net_extra_seconds=delta,total_factors=factors(totals['DriftTree-ASD'],totals['DFlash']),
        without_focal=dict(n=11,paired_mean_ratio=statistics.mean(c['factors']['latency_ratio'] for c in cases if c['request_id']!=FOCAL),
            interpretation='Attribution-only diagnostic; all 12 requests remain in the official result'),
        zero_exception_same_output=dict(ids=same,comparisons=controls),focal=focal,
        profiling_limit='Wall/round includes prefill, CPU and all online work; no separate JS/Target timings recorded',
        inference_changes=False,new_gpu_generations=0)
    out.mkdir(parents=True,exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(diagnostic,ensure_ascii=True,indent=2)+'\n',encoding='utf8')
    t=totals;f=diagnostic['total_factors'];focus=next(c for c in cases if c['request_id']==FOCAL)
    lines=['# MATH-500为什么耗时增加','',
        '**主要拖累是一道进入循环并达到长度上限的题目；其余题目还存在较小的计算与路径差异。** 这次只分析已发布日志，不新增GPU生成、不改变评分和原始结果。','',
        f"原12题逐题配对均值仍为+{(official-1)*100:.2f}%；总wall {t['DFlash']['seconds']:.2f}→{t['DriftTree-ASD']['seconds']:.2f}秒，净增{delta:.2f}秒。两种统计口径不同。",'',
        '## 1. 主要拖累：math500:372','',
        f"DFlash为824 token/7.81秒，正常EOS且答案90°正确；新版为4096 token/{row['latency_seconds']:.2f}秒，达到上限且未答对。该题新增{focus['extra_seconds']:.2f}秒，占12题净新增wall的{focus['share_of_net_extra_seconds']*100:.2f}%。",
        f"最终token序列后{tail['tokens']}个token存在精确的{tail['period']}-token周期，包含{tail['complete_cycles']}个完整周期；从第{tail['start_1based']}个token起，不断重复同一组等式。它反复得到x=140°−y，没有利用已知角收敛到90°。",'',
        '这是已经提交的最终输出发生推导循环。日志记录正常推进到4096上限，没有显示推理引擎停在同一次调用中。','',
        '## 2. 为什么JS没有挡住','',
        f"该题仅5次近似例外，所选JS累计{focal['selected_js_sum']:.6f}，但仍走入循环路径。第一轮例外regret={windows[0]['regret']:.4f}，候选/Target首选概率比约{windows[0]['candidate_to_argmax_ratio']*100:.2f}%；原ASD按剩余长度归一化的门槛允许了它。",'',
        '一步后续分布相似，并不约束当前选择对完整推导的影响。每个后续token只要在已改变的上下文下仍为Target贪心，就不再额外扣regret；预算不能自动纠正这条上下文。',
        f"最后一次例外位于第{windows[-1]['possible_position_1based'][0]}–{windows[-1]['possible_position_1based'][1]}个token之间，之后至少{focal['target_greedy_tokens_after_last_exception_lower_bound']}个token没有再接受例外，仍持续循环。停止继续花预算也不会恢复原AR路径。",'',
        '逐次区间如下；没有保存逐边完整logits，不能仅凭日志断言哪一次例外单独引起了全部后果。','',
        '|验证轮|例外可能位置（1起）|regret|候选/首选概率比|','|---:|---|---:|---:|']
    for w in windows:
        lines.append(f"|{w['round']}|{w['possible_position_1based'][0]}–{w['possible_position_1based'][1]}|{w['regret']:.4f}|{w['candidate_to_argmax_ratio']*100:.2f}%|")
    lines+=['','## 3. 全部12题的耗时分布','',
        '|题号|DFlash token/秒|新版token/秒|新增秒|验证轮DFlash→新版|正确DFlash→新版|新版停止|',
        '|---|---:|---:|---:|---|---|---|']
    for c in cases:
        a,b=c['arms']['DFlash'],c['arms']['DriftTree-ASD']
        lines.append(f"|{c['request_id']}|{a['tokens']} / {a['seconds']:.2f}|{b['tokens']} / {b['seconds']:.2f}|{c['extra_seconds']:+.2f}|{a['rounds']}→{b['rounds']}|{a['correct']}→{b['correct']}|{b['finish_reason']}|")
    lines+=['',
        f"仅作归因诊断，暂不计372时，其余11题的配对平均变化为+{(diagnostic['without_focal']['paired_mean_ratio']-1)*100:.2f}%。正式结果仍保留全部12题的+31.19%，不据此删除坏题重报成绩。",'',
        '101题新增5.81秒，但把原错误答案10/169改为正确850；445题两组都正确，新版多2.83秒。这些不应与372的无效循环混为一类。','',
        '## 4. 计算本身也有成本，但不是所有token慢31%','',
        f"总输出{t['DFlash']['tokens']}→{t['DriftTree-ASD']['tokens']}，增加{(f['length_ratio']-1)*100:.2f}%；汇总吞吐{t['DFlash']['tps']:.2f}→{t['DriftTree-ASD']['tps']:.2f} token/s，变化{(t['DriftTree-ASD']['tps']/t['DFlash']['tps']-1)*100:.2f}%。因此总体主要是多写了token。",
        f"验证轮{t['DFlash']['rounds']}→{t['DriftTree-ASD']['rounds']}；每输出token轮数下降{(1-f['rounds_per_token_ratio'])*100:.2f}%，每轮请求wall摊销增加{(f['amortized_wall_per_round_ratio']-1)*100:.2f}%，两者大体抵消。",'',
        '上述为恒等分解：时间比=长度比×每token轮数比×每轮wall摊销比。wall/轮包含prefill和CPU等工作，不能把全部差值归为JS，也不能当成单次Target前向测量。',
        f"零例外的{', '.join(same)}三题与DFlash输出完全相同；新版较DFlash配对均值{(controls['DFlash']['mean_latency_ratio']-1)*100:+.2f}%，较严格树{(controls['BudgetTree']['mean_latency_ratio']-1)*100:+.2f}%。这支持存在构树/检查/额外轮次的小幅组合成本，未支持普遍31%的硬件卡顿。",'',
        '原表中新版相对原组合的共同128前缀慢5.65%（12题），相对严格树慢2.13%。这部分等生成量成本仍需承认；本轮没有DFlash等128前缀直接对照。','',
        '## 5. 改进重点','',
        '优先约束当前近似token的分歧强度，例如增加独立于剩余后缀长度的候选/首选概率比下限（等价于regret上限）；它能过滤这里已观察到的高regret例外，但尚未做反事实验证，不能保证该题必然恢复。',
        '再考虑重复周期检测与有计时的纠错/重生成策略。直接把4096上限改短只会更早截断错误答案，不是保持质量的加速。单纯优化JS算子难以解决这道题多写3272token的问题。','',
        '诊断证据见[analysis.json](../results/branch-drift-math-diagnosis/analysis.json)。复核命令：`python -m scripts.diagnose_branch_drift_math`。原始7任务成绩、输出及参数保持。','']
    (ROOT/'docs/BRANCH_DRIFT_MATH_DIAGNOSIS.md').write_text('\n'.join(lines),encoding='utf8')
    if any(hashlib.sha256((source/n).read_bytes()).hexdigest()!=digest for n,digest in all_fingerprints.items()):
        raise ValueError('Original inputs changed during diagnosis')
    print(json.dumps(dict(requests=12,net_extra_seconds=delta,focal_share=focus['share_of_net_extra_seconds'],
        without_focal_ratio=diagnostic['without_focal']['paired_mean_ratio'],periodic_tail=tail,
        new_gpu_generations=0,original_inputs_unchanged=True,original_payloads_checked=len(all_fingerprints))))


if __name__=='__main__':
    main()
