"""Official DDTree/BASTION algorithms on the frozen shared execution stack."""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from unittest.mock import patch

import torch

from memorydraft.evaluation.asd_paper import conversation_record, request_for, score
from memorydraft.evaluation.branch_drift import references, summarize
from memorydraft.runtime.benchmark import provenance
from memorydraft.runtime.config import RuntimeConfig, load_engine
from memorydraft.memory.store import Snapshot
from scripts import run_asd_tree as common
from scripts.diagnose_transport_online import write
from scripts.run_branch_drift_optimization import select_records
from scripts.run_runtime_migration import append
from scripts.run_tree_execution import execute as current_execute

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('AR', 'DFlash', 'ASD', 'BudgetTree', 'BudgetTree-ASD', 'BufferedKV-ASD',
        'DDTree-32', 'DDTree-128', 'BASTION')
UPSTREAMS = {
    'bastion': ('1989b80b304f01710946d7fb68af330c03d97461', 'https://github.com/kaist-ai-osi-lab/BASTION'),
    'ddtree': ('c96427a185677bf4133ed865dd1626a5041aef9b', 'https://github.com/liranringel/ddtree'),
}


def load_upstreams(root):
    manifest = json.loads((root/'UPSTREAM_TREES.json').read_bytes())
    for name, (commit, url) in UPSTREAMS.items():
        item = manifest[name]
        if (item['commit'], item['url'], item['license']) != (commit, url, 'MIT'):
            raise ValueError('Changed upstream identity: '+name)
        for path, digest in item['files'].items():
            if hashlib.sha256((root/'third_party'/name/path).read_bytes()).hexdigest() != digest:
                raise ValueError('Changed upstream file: '+name+'/'+path)
    # DDTree's released modules use top-level imports; preserve their own globals.
    sys.path[:0] = [str(root/'.deps'), str(root/'third_party/ddtree'), str(root/'third_party/bastion')]
    os.environ['PATH'] = str(root/'.deps/bin')+os.pathsep+os.environ['PATH']
    dd = importlib.import_module('ddtree')
    bastion = importlib.import_module('bastion.tree_draft')
    calibration = importlib.import_module('bastion.cost_model')
    benchmark = importlib.import_module('bastion.benchmark')
    # The official CLI defaults to the optional compiled compactor.
    dd.maybe_enable_cpp_compact(True)
    return dd, bastion, calibration, benchmark, manifest


def result_tokens(output, prompt, limit, stops):
    ids = output[0].tolist()
    if ids[:len(prompt)] != prompt:
        raise ValueError('Upstream changed the input prefix')
    tokens = ids[len(prompt):]
    if len(tokens) > limit:
        raise ValueError('Upstream exceeded output limit')
    first_stop = next((i for i, token in enumerate(tokens) if token in stops), None)
    if first_stop is not None and first_stop != len(tokens)-1:
        raise ValueError('Upstream returned tokens after EOS')
    return tokens


@torch.inference_mode()
def collect_bastion(calibration, **kwargs):
    # The upstream outer context-prefill loop is not decorated, unlike its probes.
    return calibration.collect_calibration_data(**kwargs)


def external_execute(engine, record, arm, limit, dd, bastion, profile, *, request_factory=request_for):
    request = request_factory(record, limit)
    engine.synchronize()
    torch.cuda.reset_peak_memory_stats(engine.target.device)
    started = time.perf_counter()
    ids = engine.tokenizer.apply_chat_template(list(request.messages), add_generation_prompt=True,
        return_tensors='pt', **engine.config.template_options).to(engine.target.device)
    prompt = ids[0].tolist()
    if len(prompt) > engine.config.max_input_tokens or len(prompt)+limit > engine.target.config.max_position_embeddings:
        raise ValueError('Upstream request exceeds model context')
    nodes = []
    if arm.startswith('DDTree-'):
        build = dd.build_ddtree_tree
        def measured(*args, **kwargs):
            value = build(*args, **kwargs)
            nodes.append(len(value[0])+1)
            return value
        with patch.object(dd, 'build_ddtree_tree', measured):
            output = dd.ddtree_generate(engine.draft, engine.target, ids, engine.draft.mask_token_id,
                limit, engine.draft.block_size, list(engine.stops), temperature=0.,
                tree_budget=int(arm.split('-')[1])-1, save_tree_traces=False)
    elif arm == 'BASTION':
        build = bastion.build_adaptive_best_tree_from_draft_logits
        def measured(*args, **kwargs):
            value = build(*args, **kwargs)
            nodes.append(value.token_ids.numel())
            return value
        with patch.object(bastion, 'build_adaptive_best_tree_from_draft_logits', measured):
            output = bastion.bastion_generate(engine.draft, engine.target, ids, limit,
                list(engine.stops), 0., return_stats=True, cost_model_name='qwen3-8b', cost_gpu_type='a6000',
                **profile)
    else:
        raise ValueError('Unknown external tree')
    tokens = result_tokens(output.output_ids, prompt, limit, engine.stops)
    text = engine.tokenizer.decode(tokens, skip_special_tokens=True)
    engine.synchronize()
    elapsed = time.perf_counter()-started
    lengths = output.acceptance_lengths
    if len(lengths) != len(nodes):
        raise ValueError('Missing upstream round measurements')
    return dict(request_id=request.id, method=arm, arm=arm, benchmark=record['benchmark'],
        token_ids=tokens, text=text, prompt_tokens=len(prompt),
        prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        snapshot_id=Snapshot(request.user_id, request.order, ()).snapshot_id,
        latency_seconds=elapsed, ttft_seconds=None, upstream_prefill_seconds=output.time_to_first_token,
        finish_reason='eos' if tokens and tokens[-1] in engine.stops else 'length',
        peak_gpu_bytes=torch.cuda.max_memory_allocated(engine.target.device), max_new_tokens=limit,
        approximate=False, asd=None, rounds=[dict(advance=int(a), tree_nodes=int(n)) for a,n in zip(lengths,nodes)],
        upstream_stage_seconds=getattr(output, 'stage_times', None),
        upstream_acceptance_lengths=lengths)


def analyze(rows):
    refs = [r for r in rows if r['arm'] == 'AR']
    result = summarize(rows, refs, arms=ARMS, prefix_arms=())
    result['reference_policy'] = 'Same-session AR; identical frozen prompts and strict-AR turn-1 history'
    by_id = {r['request_id']:r for r in refs}
    differences = []
    for r in rows:
        if r['arm'] in ('DFlash', 'BudgetTree', 'DDTree-32', 'DDTree-128', 'BASTION'):
            ref = by_id[r['request_id']]
            if any(r[k] != ref[k] for k in ('token_ids','text','finish_reason')):
                differences.append(dict(id=r['request_id'],arm=r['arm']))
    result['strict_identity_differences'] = differences
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/raw/external-trees')
    args = parser.parse_args()
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=False)
    dd, bastion, calibration, benchmark, manifest = load_upstreams(ROOT)
    write(out/'upstreams.json', manifest)
    pool = select_records(ROOT)
    refs = references(ROOT, pool)
    indexed = {r['request_id']:r for r in refs}
    write(out/'tasks.json', pool)
    write(out/'frozen-references.json', refs)
    info = dict(status='started', arms=ARMS, selected_tasks=len(pool), matched_inputs=len(refs),
        max_new_tokens=4096, temperature=0., dtype='float32', attention='eager',
        selection='Existing 62 development tasks / 66 matched inputs; not a new holdout',
        upstream_scope='Official generation/build/verify/cache algorithms with shared preloaded Target/DFlash; not paper environment reproduction',
        timing='Synchronized whole request including tokenization/prefill/first draft/CPU/output transfer/decode; calibration/load/warmup separate',
        ddtree_total_nodes=[32,128], bastion_nodes='Official default adaptive 32..8192',
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write(out/'provenance.json', info)
    engine, rows = None, []
    try:
        started = time.perf_counter()
        engine = load_engine(replace(RuntimeConfig.load(ROOT/'configs/runtime-dflash-rmsnorm.json'),
            db=None, max_new_tokens=4096))
        info['load_seconds'] = time.perf_counter()-started
        warm = dict(id='warmup',benchmark='gsm8k',question='What is 3 plus 4?',answer='#### 7')
        # Both cost models are recalibrated on synthetic inputs on this GPU.
        started = time.perf_counter()
        tree = common.calibrate(engine,warm,out)
        collect_bastion(calibration,target=engine.target,tokenizer=engine.tokenizer,
            device=engine.target.device,model_name_or_path='Qwen/Qwen3-8B',gpu_type='a6000',
            output_path=out/'bastion-probe.jsonl',attn_implementation='eager')
        params = calibration.fit_linear_calibration(out/'bastion-probe.jsonl',sigma_scale=30.)
        write(out/'bastion-calibration.json',dict(fit_parameters=params,fit_version='sigma_sequence_v1',
            sigma_scale=30.,attn_impl='eager',dtype='float32'))
        os.environ['CALIBRATION_JSON_PATH'] = str(out/'bastion-calibration.json')
        warm_ids = engine.tokenizer.apply_chat_template(list(request_for(warm,64).messages),
            add_generation_prompt=True,return_tensors='pt',**engine.config.template_options).to(engine.target.device)
        dl,nv = benchmark.profile_tree_adaptive_constants(draft_model=engine.draft,target=engine.target,
            input_ids=warm_ids,block_size=16,temperature=0.,model_name='qwen3-8b',gpu_type='a6000',num_trials=3)
        profile = dict(draft_latency_s=dl,non_verify_latency_s=nv)
        write(out/'bastion-profile.json',profile)
        info['calibration_seconds'] = time.perf_counter()-started
        def execute(record,arm,limit):
            if arm == 'AR':
                r = asdict(engine.generate(request_for(record,limit),'AR'))
                return dict(r,arm=arm,benchmark=record['benchmark'],rounds=[],max_new_tokens=limit)
            if arm in ('DDTree-32','DDTree-128','BASTION'):
                return external_execute(engine,record,arm,limit,dd,bastion,profile)
            if arm == 'BufferedKV-ASD':
                return current_execute(engine,record,arm,limit,tree)
            return common.execute(engine,record,arm,limit,tree,request_factory=request_for)
        for arm in ARMS:
            append(out/'warmup.jsonl',execute(warm,arm,64))
        info.update(runtime=provenance(engine),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            ddtree_cpp_compaction=dd._CPP_COMPACT_ENABLED and bool(dd.load_cpp_compact_module()))
        write(out/'provenance.json',info)
        for index,task in enumerate(pool):
            for turn in ((1,2) if task['benchmark']=='mtbench' else (1,)):
                first = indexed.get(task['id']+':turn1')
                record = conversation_record(task,turn,first['text'] if turn==2 else None) if task['benchmark']=='mtbench' else task
                order = list(ARMS)
                random.Random('external-trees:'+record['id']).shuffle(order)
                cache = {}
                for arm in order:
                    r = execute(record,arm,4096)
                    ref = indexed[record['id']]
                    if (r['prompt_sha256'],r['snapshot_id']) != (ref['prompt_sha256'],ref['snapshot_id']):
                        raise ValueError('Changed frozen input: '+record['id'])
                    if r['text'] not in cache:
                        cache[r['text']] = score(record,r['text'])
                    r.update(stage='natural',repeat=0,quality=cache[r['text']],stop_policy='natural-eos',
                        context_policy='shared frozen strict-AR first-turn history' if turn==2 else 'original request')
                    rows.append(r)
                    append(out/'rows.jsonl',r)
                    info.update(timed_generations=len(rows),last_task=record['id'],last_arm=arm)
                    write(out/'provenance.json',info)
            print(json.dumps(dict(completed=index+1,total=len(pool),rows=len(rows),benchmark=task['benchmark'])),flush=True)
        write(out/'analysis.json',analyze(rows))
        info.update(status='completed',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    except BaseException as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error))
        raise
    finally:
        if engine is not None:
            info['compiler'] = engine._regional.summary()
            engine.close()
        write(out/'provenance.json',info)


if __name__ == '__main__':
    main()
