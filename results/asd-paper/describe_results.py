"""Summarize measurement boundaries without changing frozen primary statistics."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics

from memorydraft.evaluation.asd_comparison import bootstrap
from memorydraft.evaluation.asd_paper import ARMS,BENCHMARKS
from scripts.write_asd_paper_report import read_rows

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/asd-paper'


def main():
    analysis=json.loads((OUT/'analysis.json').read_bytes())
    rows=read_rows(OUT,'rows');refs=read_rows(OUT,'references')
    rollout=read_rows(OUT,'rollout');warmup=read_rows(OUT,'warmup')
    eos_ids={r['token_ids'][-1] for r in rows if r['stage']=='natural' and r['finish_reason']=='eos'}
    forced=[]
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            selected=[r for r in rows if r['benchmark']==benchmark and r['arm']==arm and r['stage']=='fixed']
            ends=[next((i+1 for i,t in enumerate(r['token_ids']) if t in eos_ids),256) for r in selected]
            forced.append(dict(benchmark=benchmark,arm=arm,requests=len(selected),
                eos_before_256=sum(x<256 for x in ends),
                forced_after_first_eos_tokens=sum(256-x for x in ends),
                forced_fraction=sum(256-x for x in ends)/(256*len(selected))))
    # Cluster both turns together; these are descriptive conversation ratios, not the
    # primary request-equal mean. The own-history rollout has different inputs.
    conversations=defaultdict(dict)
    for sample in analysis['samples']:
        if sample['benchmark']=='mtbench':conversations[sample['request_id'].rsplit(':turn',1)[0]][sample['request_id']]=sample
    conversation_stats=[]
    for arm in ARMS:
        ratios=[]
        for turns in conversations.values():
            assert len(turns)==2
            ratios.append(sum(s['arms']['natural/'+arm]['seconds'] for s in turns.values())/
                          sum(s['arms']['natural/ASD']['seconds'] for s in turns.values()))
        own=[r for r in rollout if r['arm']==arm]
        own_asd={r['conversation_id']:r for r in rollout if r['arm']=='ASD'}
        own_ratios=[(r['first_latency_seconds']+r['latency_seconds'])/
            (own_asd[r['conversation_id']]['first_latency_seconds']+own_asd[r['conversation_id']]['latency_seconds']) for r in own]
        conversation_stats.append(dict(arm=arm,conversations=8,matched_history_mean_ratio=statistics.mean(ratios),
            matched_history_conversation_ci95=bootstrap(ratios),own_history_mean_ratio=statistics.mean(own_ratios),
            own_history_total_seconds=analysis['realized_dialogue'][arm]['seconds'],
            own_history_total_ratio=analysis['realized_dialogue'][arm]['seconds']/analysis['realized_dialogue']['ASD']['seconds']))
    paired=[s for s in analysis['samples'] if s['benchmark']=='gsm8k' and
            all(s['arms']['prefix/'+arm]['tokens']==128 for arm in ARMS)]
    common_gsm={arm:{k:sum(s['arms']['prefix/'+arm][k] for s in paired)
                    for k in ('tokens','calls','nodes','seconds')} for arm in ARMS}
    raw=gzip.decompress((OUT/'local-judge.jsonl.gz').read_bytes()).decode().splitlines()
    judge=[json.loads(x) for x in raw]
    drift=[s['arms']['natural/'+arm]['repeat_seconds'][1]/s['arms']['natural/'+arm]['repeat_seconds'][0]
           for s in analysis['samples'] for arm in ARMS]
    resources=dict(model_load_seconds=[r['load_seconds'] for r in analysis['resources']],
        worker_peak_gib=[r['peak_gpu_bytes']/2**30 for r in analysis['resources']],
        maximum_row_peak_gib=max(r['peak_gpu_bytes'] for r in [*rows,*refs,*rollout])/2**30,
        cold_backend_seconds=[r['compiler']['backend_seconds'] for r in analysis['resources']],
        warmup_and_zero_generations=len(warmup),
        warmup_and_zero_wall_seconds=sum(r['latency_seconds'] for r in warmup),
        compiler_cache_bytes=[r['cache_bytes'] for r in analysis['resources']],
        judge_load_seconds=analysis['local_judge_provenance']['load_seconds'],
        judge_forwards_seconds=sum(r.get('seconds',0) for r in judge),
        judge_peak_gib=analysis['local_judge_provenance']['peak_gpu_bytes']/2**30,
        cost_table_read_seconds=None,
        missing_measurement='The frozen worker did not separately time one-time cost-table reads.')
    result=dict(note='Supplementary descriptive diagnostics; all primary rows and scores remain unchanged.',
        fixed_eos_ids=sorted(eos_ids),fixed_post_eos=forced,mtbench_conversation_statistics=conversation_stats,
        common_gsm_prefix_requests=len(paired),common_gsm_prefix=common_gsm,resources=resources,
        natural_repeat_drift=dict(pairs=len(drift),minimum=min(drift),maximum=max(drift),
                                 outside_three_percent=sum(not .97<=x<=1.03 for x in drift)))
    (OUT/'descriptive.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(dict(resources=resources,common_gsm_prefix_requests=len(paired),
        repeat_drift=result['natural_repeat_drift'],mtbench_conversations=conversation_stats,
        fixed_post_eos=[v for v in forced if v['arm']=='BudgetTree-ASD'])))


if __name__=='__main__':main()
