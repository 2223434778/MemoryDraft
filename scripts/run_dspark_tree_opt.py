"""Cold-request, paired DSpark tree optimization with frozen tree prices."""
import argparse
from dataclasses import asdict, replace
from functools import partial
import gzip
import hashlib
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import torch

from memorydraft.backends.asd import ASDConfig
from memorydraft.backends.budget_tree import BudgetTreeConfig, TreeCosts
from memorydraft.backends.dspark import load_dspark
from memorydraft.backends import dspark_reuse
from memorydraft.backends.dspark_tree import build_dspark_tree
from memorydraft.backends.dspark_tree_fast import MarkovRowCache, build_dspark_tree_fast
from memorydraft.backends.regional_compile import runtime_rmsnorm
from memorydraft.evaluation.asd_paper import BENCHMARKS, conversation_record, request_for, score
from memorydraft.evaluation.asd_paper_extension import load_records, validate_policy
from memorydraft.runtime.config import RuntimeConfig, load_engine
from scripts.diagnose_transport_online import write
from scripts.run_dspark_tree import DraftPair, execute as original_execute
from scripts.run_runtime_migration import append

ROOT=Path(__file__).resolve().parents[1]
COUNTS={'humaneval':4,'mtbench':4,'alpaca':16}
ARMS=('DFlash-Tree-ASD','DSpark-ASD','DSpark-Tree-ASD','DSpark-Tree-ASD-Fast')
REPEATS={'natural':2,'prefix':1,'fixed':1}
COMPUTE_KEYS={'built_nodes','conditional_distributions','search_bound_stopped'}


def selected_records(root):
    records=load_records(root)
    return [r for b in BENCHMARKS if b in COUNTS for r in [x for x in records if x['benchmark']==b][:COUNTS[b]]]


def read_reference_files(root):
    references={}
    # Prefer the old DSpark run; extension supplies the twelve additional Alpaca tasks.
    for study in ('asd-paper-extension','dspark-tree'):
        for shard in (0,1):
            path=root/f'results/{study}/part-{shard}/references.jsonl.gz'
            for line in gzip.decompress(path.read_bytes()).splitlines():
                row=json.loads(line)
                references[row['request_id']]=dict(row,reference_origin=path.relative_to(root).as_posix(),
                                                  reference_reused=True)
    return references


def frozen_trees(root,shard):
    directory=root/f'results/dspark-tree/part-{shard}'
    dflash=json.loads((directory/'calibration.json').read_bytes())['costs']
    dspark=json.loads((directory/'dspark-calibration.json').read_bytes())['config']
    def costs(value):
        return TreeCosts(tuple(value['contexts']),tuple(value['nodes']),tuple(tuple(x) for x in value['seconds']),
                         value['draft_seconds'],value['auxiliary_seconds'])
    return {'DFlash':BudgetTreeConfig(costs(dflash)),
            'DSpark':BudgetTreeConfig(costs(dspark['costs']),**{k:v for k,v in dspark.items() if k!='costs'})}


def tree_semantic(tree):
    return dict(tokens=tree.tokens.tolist(),parents=tree.parents,depths=tree.depths,
                budget={k:v for k,v in tree.budget_metadata.items() if k not in COMPUTE_KEYS})


def signature(row):
    rounds=[dict(r,tree_budget={k:v for k,v in r['tree_budget'].items() if k not in COMPUTE_KEYS})
            if 'tree_budget' in r else r for r in row['rounds']]
    return row['token_ids'],row['text'],row['finish_reason'],row['asd'],rounds,row.get('selected_tree_sha256')


def execute(engine,draft,record,arm,limit,trees,*,fixed=False):
    engine.synchronize();started=time.perf_counter()
    fast=arm=='DSpark-Tree-ASD-Fast'
    cache=MarkovRowCache(draft) if fast else None
    digest=hashlib.sha256()
    def builder(*args,**kwargs):
        tree=(build_dspark_tree_fast(*args,cache=cache,**kwargs) if fast else build_dspark_tree(*args,**kwargs))
        digest.update(json.dumps(tree_semantic(tree),sort_keys=True,separators=(',',':')).encode())
        return tree
    with patch.object(dspark_reuse,'build_dspark_tree',builder):
        row=original_execute(engine,draft,record,'DSpark-Tree-ASD' if fast else arm,limit,trees,fixed=fixed)
    row.update(method=arm,arm=arm,markov_cache=cache.summary() if cache else None,
               selected_tree_sha256=digest.hexdigest() if arm.startswith('DSpark-Tree') else None)
    row['latency_seconds']=time.perf_counter()-started
    return row


@torch.inference_mode()
def mechanism(engine,draft,trees,out,shard):
    """All seven families, first two sources each; ephemeral real GPU states only."""
    records=load_records(ROOT)
    for benchmark in BENCHMARKS:
        original=[r for r in records if r['benchmark']==benchmark][shard]
        record=conversation_record(original,1,None) if benchmark=='mtbench' else original
        ids=engine.tokenizer.apply_chat_template(list(request_for(record,128).messages),add_generation_prompt=True,
            return_tensors='pt',**engine.config.template_options).to(engine.target.device)
        captured=[]
        def capture(anchor,base):
            if len(captured)<4:captured.append((anchor.clone(),base.clone()))
        generated=dspark_reuse.generate_dspark(engine.target,draft,ids,max_new_tokens=128,
                                               stop_token_ids=engine.stops,on_draft=capture)
        contexts=[ids.shape[1]+r['generated_before']-1 for r in generated.rounds[:len(captured)]]
        reference=[tree_semantic(build_dspark_tree(a,b,draft,trees['DSpark'],c,stop_token_ids=engine.stops))
                   for (a,b),c in zip(captured,contexts)]
        variants=['reference','cache','bound','both']
        for repeat in range(4):
            order=variants if repeat%2==0 else list(reversed(variants))
            for variant in order:
                engine.synchronize();started=time.perf_counter()
                cache=MarkovRowCache(draft) if variant in ('cache','both') else None
                built=[]
                for (a,b),context in zip(captured,contexts):
                    tree=(build_dspark_tree(a,b,draft,trees['DSpark'],context,stop_token_ids=engine.stops)
                          if variant=='reference' else build_dspark_tree_fast(a,b,draft,trees['DSpark'],context,
                              stop_token_ids=engine.stops,cache=cache,early_stop=variant in ('bound','both')))
                    built.append(tree)
                engine.synchronize();seconds=time.perf_counter()-started
                if [tree_semantic(t) for t in built]!=reference:raise ValueError('GPU replay tree changed: '+variant)
                append(out/'mechanism.jsonl',dict(request_id=record['id'],benchmark=benchmark,variant=variant,
                    repeat=repeat,warmup=repeat==0,states=len(captured),seconds=seconds,
                    cache=cache.summary() if cache else None,trees=[t.budget_metadata for t in built],exact=True))
        del captured,generated,built


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,choices=(0,1),required=True)
    args=parser.parse_args();out=ROOT/f'results/raw/dspark-tree-opt/part-{args.shard}'
    out.mkdir(parents=True,exist_ok=False)
    selected=[r for b in COUNTS for i,r in enumerate([x for x in selected_records(ROOT) if x['benchmark']==b]) if i%2==args.shard]
    refs=read_reference_files(ROOT);trees=frozen_trees(ROOT,args.shard)
    info=dict(status='started',shard=args.shard,task_ids=[r['id'] for r in selected],arms=ARMS,repeats=REPEATS,
        tree_configs={k:asdict(v) for k,v in trees.items()},prices='frozen prior per-card DSpark study tables',
        cache_entries=64,cache_scope='cold each request',native_proposals={'DFlash':15,'DSpark':7},
        split='observed extension development replay; 12 additional DSpark Alpaca sources',
        reference_reuse='inputs/history only; old AR timings excluded from all comparisons')
    engine=None;write(out/'provenance.json',info)
    try:
        models=ROOT.parent/'models';started=time.perf_counter()
        config=RuntimeConfig(backend='dflash',target=str(models/'qwen3-8b'),draft=str(models/'dflash-qwen3-8b'),
            target_revision='b968826d9c46dd6066d109eabc6255188de91218',max_new_tokens=4096)
        engine=load_engine(config);draft=load_dspark(models/'dspark-qwen3-8b-block7',engine.target,engine.draft)
        from scripts.check_dspark_backbone import check
        write(out/'upstream-backbone-check.json',check(engine,draft,ROOT/'results/dspark-tree/reference-source'))
        regional=runtime_rmsnorm(engine.target,DraftPair({'dflash':engine.draft,'dspark':draft}))
        engine._regional=regional;engine._regional_context=regional.activate('rmsnorm-inductor')
        engine._regional_context.__enter__();engine.execution_profile='rmsnorm-inductor'
        engine.config=replace(config,execution_profile='rmsnorm-inductor')
        info.update(load_seconds=time.perf_counter()-started,runtime=engine.describe())
        warm=dict(benchmark='gsm8k',id='warmup',question='What is 3 plus 4?',answer='#### 7')
        append(out/'warmup.jsonl',asdict(engine.generate(request_for(warm,32),'AR')))
        for arm in ARMS:append(out/'warmup.jsonl',execute(engine,draft,warm,arm,64,trees))
        mechanism(engine,draft,trees,out,args.shard)
        info['compiler_after_warmup']=regional.summary();graphs=len(regional.backend.events)
        count=0
        for original in selected:
            first_answer=None
            for turn in ((1,2) if original['benchmark']=='mtbench' else (0,)):
                record=conversation_record(original,turn,first_answer) if turn else original
                ref=refs[record['id']]
                messages=list(request_for(record,4096).messages)
                if 'messages' in ref and messages!=ref['messages']:raise ValueError('Reused reference prompt changed')
                append(out/'references.jsonl',dict(ref,messages=messages,messages_reconstructed='messages' not in ref))
                if turn==1:first_answer=ref['text']
                order=list(ARMS);random.Random('dspark-opt-v1/'+record['id']).shuffle(order)
                signatures={}
                for stage in REPEATS:
                    for repeat in range(REPEATS[stage]):
                        for arm in (order if repeat==0 else list(reversed(order))):
                            row=execute(engine,draft,record,arm,4096 if stage=='natural' else 128 if stage=='prefix' else 256,
                                        trees,fixed=stage=='fixed')
                            row.update(stage=stage,repeat=repeat,quality=score(record,row['text']) if stage=='natural' else None,
                                       stop_policy='ignore-eos' if stage=='fixed' else 'eos')
                            if (row['prompt_sha256'],row['snapshot_id'])!=(ref['prompt_sha256'],ref['snapshot_id']):
                                raise ValueError('Target input/snapshot changed')
                            validate_policy(row,ASDConfig(budget=8))
                            if stage=='fixed' and len(row['token_ids'])!=256:raise ValueError('Incomplete fixed work')
                            key=stage,arm;value=signature(row)
                            if key in signatures and signatures[key]!=value:raise ValueError('Unstable repeat: '+arm)
                            signatures[key]=value
                            other='DSpark-Tree-ASD' if arm.endswith('-Fast') else 'DSpark-Tree-ASD-Fast'
                            if arm.startswith('DSpark-Tree') and (stage,other) in signatures and signatures[stage,other]!=value:
                                raise ValueError('Fast/reference tree, output or budget changed')
                            if row['counters']['compile_graphs_after']!=graphs:raise ValueError('Formal compilation occurred')
                            append(out/'rows.jsonl',row)
                count+=1;info['completed_requests']=count;write(out/'provenance.json',info)
                print(json.dumps(dict(completed=count,request_id=record['id'],benchmark=record['benchmark'])),flush=True)
        info.update(status='complete',compiler_end=regional.summary())
    except Exception as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        write(out/'provenance.json',info)
        if engine is not None:engine.close()


if __name__=='__main__':main()
