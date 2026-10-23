"""Reproduce the report and descriptive route interactions from public evidence."""
import json
from pathlib import Path
import statistics

from scripts.write_asd_comparison_report import phase_summary,render
from memorydraft.evaluation.asd_comparison import select_candidates


def main():
    base=Path(__file__).resolve().parent
    original=json.loads((base/'analysis.json').read_bytes())
    screen=phase_summary(base,'screen')
    selection=select_candidates(screen)
    confirm=phase_summary(base,'confirm',selection)
    rebuilt=dict(screen=screen,selection=selection,confirm=confirm)
    if rebuilt != original or render(rebuilt).encode() != (base/'report.md').read_bytes():
        raise ValueError('Published statistics or report changed')
    print(json.dumps(dict(rebuilt_exact=True,screen_rows=screen['rows'],confirm_rows=confirm['rows'])))
    for benchmark,arms in confirm['families'].items():
        print(benchmark,{a:dict(natural=v['natural']['base/ASD']['ratio'],prefix=v['prefix']['base/ASD']['ratio'],
            prefix_n=v['prefix']['base/ASD']['n'],correct=v['natural']['secondary_correct']) for a,v in arms.items()})
    # A paired four-arm description on screening data, not a causal decomposition.
    interactions={}
    for benchmark in screen['families']:
        cases=[s for s in screen['samples'] if s['benchmark']==benchmark]
        interactions[benchmark]={}
        for variant in ('asd-prefix','ordinary-position'):
            for policy in ('ASD-Run','ASD-Utility'):
                for stage in ('natural','prefix'):
                    keys=[f'{stage}/base/ASD',f'{stage}/base/{policy}',f'{stage}/{variant}/ASD',f'{stage}/{variant}/{policy}']
                    pairs=[[s['arms'][k] for k in keys] for s in cases]
                    if stage=='prefix':pairs=[p for p in pairs if all(v['tokens']==128 for v in p)]
                    if not pairs:continue
                    interactions[benchmark][f'{stage}/{variant}/{policy}']=dict(n=len(pairs),
                        adapter_increment=statistics.mean(p[3]['seconds']/p[1]['seconds'] for p in pairs),
                        policy_increment=statistics.mean(p[3]['seconds']/p[2]['seconds'] for p in pairs),
                        multiplicative_interaction=statistics.mean((p[3]['seconds']/p[1]['seconds'])/(p[2]['seconds']/p[0]['seconds']) for p in pairs))
    print('SCREENING_INTERACTIONS',json.dumps(interactions,sort_keys=True))


if __name__=='__main__':main()
