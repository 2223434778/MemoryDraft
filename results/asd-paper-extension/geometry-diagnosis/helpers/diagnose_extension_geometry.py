"""Post-run actual-forward checks. No vector/KV payloads or speed claims."""
import argparse
from dataclasses import asdict,replace
import hashlib,json,os
from pathlib import Path

import torch

from memorydraft.evaluation.asd_joint_tree import frozen_config
from memorydraft.evaluation.asd_paper import conversation_record,request_for
from memorydraft.evaluation.asd_paper_extension import load_records,signature
from memorydraft.runtime.config import RuntimeConfig,load_engine
from scripts.run_asd_joint_tree import execute


class Capture:
    def __init__(self,model,prompt,index):
        self.model=model;self.native=model.forward;self.prompt=prompt;self.index=index;self.rows=[]

    def __enter__(self):
        def forward(*args,**kwargs):
            ids=args[0] if args else kwargs['input_ids'];cache=kwargs.get('past_key_values')
            before=cache.get_seq_length() if cache is not None else 0
            positions=kwargs.get('position_ids')
            if positions is None:positions=kwargs.get('cache_position')
            if positions is None:positions=torch.arange(before,before+ids.shape[1],device=ids.device)
            positions=positions.reshape(-1).tolist();output=self.native(*args,**kwargs)
            offset=ids.shape[1]-output.logits.shape[1]
            wanted=[i for i,p in enumerate(positions) if p+1-self.prompt==self.index and i>=offset]
            if wanted:self.rows.append(dict(cache_before=before,
                vectors={i:output.logits[0,i-offset].detach().cpu().clone() for i in wanted}))
            return output
        self.model.forward=forward;return self

    def __exit__(self,*args):self.model.forward=self.native


def vector_for(capture,row,index,prompt):
    if index==0:before=0;node=prompt-1
    else:
        block=next(b for b in row['rounds'] if b['generated_before']<=index<=b['generated_before']+b['accepted'])
        depth=index-block['generated_before'];node=block['tree_path'][depth] if 'tree_path' in block else depth
        before=block['generated_before']+prompt-1
    return next(c['vectors'][node] for c in capture.rows if c['cache_before']==before and node in c['vectors'])


def summary(vector,left,right):
    values,ids=vector.topk(2)
    return dict(argmax_id=int(ids[0]),second_id=int(ids[1]),top_two_gap=float(values[0]-values[1]),
        relative_left_over_right=float(vector[left]-vector[right]))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,choices=(0,1),required=True)
    args=parser.parse_args();root=Path.cwd();out=root/f'artifacts/geometry-diagnosis-{args.shard}';out.mkdir(exist_ok=False)
    manifest=json.loads((root/'ASD_PAPER_EXTENSION_CONTINUATION_SOURCE_MANIFEST.json').read_bytes())
    def verify():
        for n,h in manifest['files'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
        return len(manifest['files'])
    def save(name,value):(out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    directory=root/f'results/raw/asd-paper-extension/part-{args.shard}'
    def read(name):
        p=directory/(name+'.jsonl');return [json.loads(s) for s in p.read_bytes().splitlines()] if p.exists() else []
    geometry=read('geometry-differences');rows=read('rows');refs={r['request_id']:r for r in read('references')}
    rollout=read('rollout');pool={r['id']:r for r in load_records(root)};jobs=[];seen=set()
    for g in geometry:
        if g['comparison']=='different-own-history-inputs':continue
        key=(g['request_id'],g['arm'],g['stage'],g['comparison'])
        if key in seen:continue
        seen.add(key);jobs.append(g)
    info=dict(status='started',shard=args.shard,source_commit=manifest['commit'],source_sha256=manifest['source_sha256'],
        files_verified_before=verify(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],planned_comparisons=len(jobs),comparisons=[],
        scope='Actual-pass argmax and first-divergence diagnostics; no acceptance changes or speed evidence')
    save('provenance',info);engine=None
    try:
        if jobs:
            engine=load_engine(replace(RuntimeConfig.load(root/'configs/runtime-dflash-rmsnorm.json'),db=None,max_new_tokens=4096))
            tree,_=frozen_config(root,args.shard)
            warm=dict(id='warmup',benchmark='gsm8k',question='What is 3 plus 4?',answer='#### 7')
            engine.generate(request_for(warm,32),'AR')
            for arm in ('DFlash','ASD','BudgetTree','BudgetTree-ASD'):execute(engine,warm,arm,64,tree,request_factory=request_for)
            execute(engine,warm,'BudgetTree-ASD',256,tree,request_factory=request_for,fixed=True)
            execute(engine,warm,'BudgetTree-ASD',64,tree,request_factory=request_for,zero=True)
            execute(engine,warm,'BudgetTree',64,tree,request_factory=request_for)
        for number,g in enumerate(jobs):
            identity=g['request_id'];stage=g['stage'];arm=g['arm'];index=g['index'];kind=g['comparison']
            own=stage=='rollout';source=rollout if own else rows
            def chosen(method):
                return next(r for r in source if r['request_id']==identity and r['arm']==method and r['stage']==stage
                    and r['repeat']==0 and (stage!='fixed' or r.get('role')=='before'))
            original=chosen(arm);other_arm=g.get('right_arm','AR')
            other=refs[identity] if other_arm=='AR' else chosen(other_arm)
            task_id=identity.rsplit(':turn',1)[0];task=pool[task_id]
            if task['benchmark']=='mtbench':
                turn=int(identity.rsplit(':turn',1)[1]);first=original['first_text'] if own else refs[task_id+':turn1']['text']
                task=conversation_record(task,turn,first if turn==2 else None)
            assert original['token_ids'][:index]==other['token_ids'][:index]
            assert original['prompt_sha256']==other['prompt_sha256']
            assert index<len(original['token_ids']) and index<len(other['token_ids'])
            prompt=original['prompt_tokens'];limit=256 if stage=='fixed' else 128 if stage=='prefix' else 4096
            with Capture(engine.target,prompt,index) as left_capture:
                left=execute(engine,task,arm,limit,tree,request_factory=request_for,fixed=stage=='fixed')
            save(f'{number:03d}-left',left)
            assert signature(left)==signature(original),('Unstable diagnostic replay',identity,arm,stage)
            left_vector=vector_for(left_capture,left,index,prompt)
            with Capture(engine.target,prompt,index) as right_capture:
                right=asdict(engine.generate(request_for(task,index+1),'AR')) if other_arm=='AR' else execute(
                    engine,task,other_arm,limit,tree,request_factory=request_for,fixed=stage=='fixed')
            save(f'{number:03d}-right',right)
            assert right['token_ids']==other['token_ids'][:index+1] if other_arm=='AR' else signature(right)==signature(other)
            right_vector=next(iter(right_capture.rows[-1]['vectors'].values())) if other_arm=='AR' else vector_for(right_capture,right,index,prompt)
            ids=engine.tokenizer.apply_chat_template(list(request_for(task,limit).messages),add_generation_prompt=True,
                return_tensors='pt',**engine.config.template_options).to(engine.target.device)
            common=torch.cat((ids,ids.new_tensor([original['token_ids'][:index]])),dim=1)
            full=engine.target(common,attention_mask=torch.ones_like(common),use_cache=False,logits_to_keep=1).logits[0,-1].detach().cpu()
            a=original['token_ids'][index];b=other['token_ids'][index]
            assert int(left_vector.argmax())==a and int(right_vector.argmax())==b
            values=dict(request_id=identity,arm=arm,right_arm=other_arm,stage=stage,comparison=kind,index=index,observation=g,
                left_replay_matches_original=True,right_replay_matches_original_prefix=True,same_confirmed_prefix=True,
                left_pass=summary(left_vector,a,b),right_pass=summary(right_vector,a,b),full_prefix_pass=summary(full,a,b),
                max_abs_difference=float((left_vector.double()-right_vector.double()).abs().max()),
                left_vs_full_max_abs=float((left_vector.double()-full.double()).abs().max()),
                left_output_ids_sha256=hashlib.sha256(json.dumps(original['token_ids'],separators=(',',':')).encode()).hexdigest())
            save(f'{number:03d}-numeric',values);info['comparisons'].append(values);save('provenance',info)
            print(json.dumps(values),flush=True)
        info.update(status='complete',files_verified_after=verify())
    except Exception as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        save('provenance',info)
        if engine is not None:engine.close()


if __name__=='__main__':main()
