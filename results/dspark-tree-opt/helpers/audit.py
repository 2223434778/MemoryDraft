"""Independent local reconstruction and accounting from the sealed GPU archive."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from unittest.mock import patch

from scripts.run_dspark_tree_opt import ARMS, ROOT, signature
from scripts.write_dspark_tree_report import read
from scripts import write_dspark_tree_opt_report as report


def main():
    root=ROOT/'results/dspark-tree-opt'
    captures={}
    def capture(path,value):captures[path.name]=value;return len(value)
    with patch('sys.argv',['report','--input',str(root)]),patch.object(Path,'write_bytes',capture):report.main()
    rebuilt={}
    for name,value in captures.items():
        assert value==(root/name).read_bytes(),name
        rebuilt[name]=hashlib.sha256(value).hexdigest()
    rows=read(root,'rows');refs=read(root,'references');replay=read(root,'mechanism')
    assert len(rows)==448 and len(refs)==28 and len(replay)==224
    assert len({(r['request_id'],r['variant'],r['repeat']) for r in replay})==224
    assert all(r['exact'] for r in replay)
    states=sum(r['states'] for r in replay if r['variant']=='reference' and r['repeat']==0)
    assert states==49
    indexed={(r['request_id'],r['stage'],r['arm'],r['repeat']):r for r in rows}
    assert len(indexed)==448
    fast=[r for r in rows if r['arm']=='DSpark-Tree-ASD-Fast']
    for r in fast:
        other=indexed[r['request_id'],r['stage'],'DSpark-Tree-ASD',r['repeat']]
        assert signature(r)==signature(other) and r['quality']==other['quality']
        cache=r['markov_cache']
        assert cache['peak_entries']<=cache['max_entries']==64
        assert cache['hits']+cache['misses']==sum(b['tree_budget']['conditional_distributions'] for b in r['rounds'])
    assert all(r['counters']['compile_graphs_before']==r['counters']['compile_graphs_after'] for r in rows)
    assert all(r['reference_reused'] for r in refs)
    mechanism=[]
    for benchmark in ('alpaca','humaneval','mtbench'):
        selected=[r for r in rows if r['benchmark']==benchmark and r['stage']=='natural' and r['repeat']==0]
        f=[r for r in selected if r['arm'].endswith('-Fast')]
        old=[r for r in selected if r['arm']=='DSpark-Tree-ASD']
        rounds=sum(len(r['rounds']) for r in f)
        hits=sum(r['markov_cache']['hits'] for r in f);misses=sum(r['markov_cache']['misses'] for r in f)
        item=dict(benchmark=benchmark,requests=len(f),rounds=rounds,cache_hits=hits,cache_misses=misses,
            cache_hit_fraction=hits/(hits+misses),
            old_nodes_per_round=sum(b['tree_budget']['built_nodes'] for r in old for b in r['rounds'])/rounds,
            fast_nodes_per_round=sum(b['tree_budget']['built_nodes'] for r in f for b in r['rounds'])/rounds,
            old_conditionals_per_round=sum(b['tree_budget']['conditional_distributions'] for r in old for b in r['rounds'])/rounds,
            fast_conditionals_per_round=sum(b['tree_budget']['conditional_distributions'] for r in f for b in r['rounds'])/rounds)
        if benchmark=='humaneval':
            item['correct']={arm:sum(bool(r['quality']['correct']) for r in selected if r['arm']==arm) for arm in ARMS}
        if benchmark=='alpaca':
            def seconds(identity,arm):
                return statistics.mean(indexed[identity,'natural',arm,k]['latency_seconds'] for k in (0,1))
            ids=[r['request_id'] for r in f]
            item['original_dspark_vs_dflash_time_ratio']=statistics.mean(seconds(i,'DSpark-Tree-ASD')/seconds(i,'DFlash-Tree-ASD') for i in ids)
        mechanism.append(item)
    result=dict(report_rebuilt_bytes=rebuilt,rows=len(rows),requests=len(refs),exact_fast_reference_generations=len(fast),
        gpu_replay_rows=len(replay),unique_gpu_states=states,formal_new_graphs=0,
        old_reference_origins=dict(Counter(r['reference_origin'] for r in refs)),
        reconstructed_reference_messages=sum(r['messages_reconstructed'] for r in refs),
        maximum_cache_bytes=max(r['markov_cache']['peak_bytes'] for r in fast),
        maximum_peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in rows),natural_mechanism=mechanism)
    (root/'audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
