"""Rebuild AR reference speedups, explicit tau definitions and reward diagnostics."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_paper import ARMS,BENCHMARKS
from memorydraft.evaluation.asd_paper_metrics import STUDY,aggregate
from scripts.run_asd_reward import pair_score

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    if path.is_file():raw=path.read_bytes()
    else:raw=gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes())
    return [json.loads(line) for line in raw.decode().splitlines()]


def rewards(out):
    path=out/'reward/provenance.json'
    if not path.exists():return dict(status='not_available')
    info=json.loads(path.read_bytes())
    if info['status']!='complete':return dict(status=info['status'],provenance=info)
    rows=read(out/'reward/rows.jsonl')
    if len(rows)!=120 or len({(r['request_id'],r['arm']) for r in rows})!=120:raise ValueError('Incomplete reward matrix')
    groups=defaultdict(dict)
    for r in rows:groups[r['request_id']][r['arm']]=r
    if any(set(g)!=set(ARMS) for g in groups.values()):raise ValueError('Incomplete reward arms')
    summaries={};samples=[]
    for benchmark in ('mtbench','alpaca'):
        summaries[benchmark]={}
        selected=[g for g in groups.values() if g['ASD']['benchmark']==benchmark]
        for arm in ARMS:
            scores=[];deltas=[]
            for group in selected:
                a,b=group[arm],group['ASD']
                score=pair_score(a['score'],b['score'],identical=a['messages_sha256']==b['messages_sha256'])
                sample=dict(request_id=a['request_id'],benchmark=benchmark,arm=arm,preference=score,
                    reward_delta=None if score is None else a['score']-b['score'],
                    identical=a['messages_sha256']==b['messages_sha256'])
                samples.append(sample)
                if score is not None:scores.append(score);deltas.append(sample['reward_delta'])
            summaries[benchmark][arm]=dict(requests=len(selected),scored=len(scores),
                wins=scores.count(1.),ties=scores.count(.5),losses=scores.count(0.),
                preference=statistics.mean(scores) if scores else None,mean_reward_delta=statistics.mean(deltas) if deltas else None)
    return dict(status='complete',provenance=info,families=summaries,samples=samples,
                interpretation='Independent trained pointwise reward proxy; not accuracy, official MT-Bench, or AlpacaEval.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=ROOT/f'results/{STUDY}')
    args=parser.parse_args();out=args.input
    rows=[r for shard in (0,1) for r in read(out/f'part-{shard}/rows.jsonl')]
    result=aggregate(ROOT,rows);result['reward']=rewards(out)
    for shard in (0,1):
        info=json.loads((out/f'part-{shard}/provenance.json').read_bytes())
        if info['status']!='complete' or not info['original_unchanged']:raise ValueError('Worker incomplete or changed old evidence')
    (out/'analysis.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    tau={(r['benchmark'],r['arm']):r for r in result['acceptance'] if r['stage']=='fixed'}
    old=json.loads((ROOT/'results/asd-paper/analysis.json').read_bytes())
    lines=['# 七任务补充：Target-only基线、接受长度与独立奖励评分','',
        '249原请求、747新计时，256固定token且忽略EOS。前后严格DFlash与AR同窗；原五组时间保留。',
        '下表原TPS来自上一轮，AR倍率为跨会话比；锚点校准列是按新旧DFlash逐题漂移折算的估计，不冒称重新配对测量候选。',
        '接受长度不含Target bonus；selected是原选择前缀，emitted裁去末轮未输出部分。自然质量与固定吞吐分开。','',
        '|任务|方法|原固定TPS|跨会话AR倍率|锚点校准估计倍率|tau selected / emitted|自然质量变化vs ASD|',
        '|---|---|---:|---:|---:|---:|---:|']
    for b in BENCHMARKS:
        family=result['families'][b]
        lines.append(f"|{b}|AR（新测）|{family['ar_tps']:.3f}|1.000|1.000|—|—|")
        for arm in ARMS:
            v=family['arms'][arm];t=tau[b,arm];q=old['families'][b][arm]['natural'];base=old['families'][b]['ASD']['natural']
            delta=f"{100*(q['correct']-base['correct'])/q['scored']:+.2f} pp" if q['scored'] else '另见奖励proxy'
            if b=='mmlu':delta+='（输出契约）'
            lines.append(f"|{b}|{arm}|{v['old_fixed_tps']:.3f}|{v['split_session_speedup_vs_ar']:.3f}|{v['anchor_adjusted_estimated_speedup_vs_ar']:.3f}|{t['tau_selected']:.3f} / {t['tau_emitted']:.3f}|{delta}|")
    lines+=['','|任务|新严格DFlash/AR直接倍率|前后漂移>3%请求|跨会话漂移>3%请求|','|---|---:|---:|---:|']
    for b,f in result['families'].items():
        lines.append(f"|{b}|{f['measured_current_dflash_speedup']:.3f}|{f['within_drift_over_3pct']}/{f['requests']}|{f['cross_drift_over_3pct']}/{f['requests']}|")
    reward=result['reward'];lines+=['','## 独立奖励模型','',f"执行状态：{reward['status']}。"]
    if reward['status']=='complete':
        lines+=['','|任务|方法vs ASD|胜/平/负|偏好（平局半分）|覆盖|','|---|---|---:|---:|---:|']
        for b,arms in reward['families'].items():
            for arm,v in arms.items():
                lines.append(f"|{b}|{arm}|{v['wins']}/{v['ties']}/{v['losses']}|{100*v['preference']:.2f}%|{v['scored']}/{v['requests']}|")
    lines+=['','奖励模型是Skywork独立偏好训练权重，逐对话单独评分，不呈现A/B顺序；仍与Target共享Qwen3基础模型，可能有长度、风格和训练数据偏差。',
        '该proxy不能替代官方MT-Bench/AlpacaEval或人工质量保证；旧第一token裁判失败证据保留，不据新评分调整生成算法。','']
    (out/'report.md').write_bytes('\n'.join(lines).encode())
    print(json.dumps(dict(timed=len(rows),requests=result['requests'],reward=reward['status'])))


if __name__=='__main__':main()
