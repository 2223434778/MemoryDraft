"""One-case correctness replay; instrumented timings are not speed evidence."""
from dataclasses import asdict,replace
import hashlib,json,os
from pathlib import Path
import time

import torch

from memorydraft.evaluation.asd_joint_tree import frozen_config
from memorydraft.evaluation.asd_paper import request_for
from memorydraft.evaluation.asd_paper_extension import load_records
from memorydraft.runtime.config import RuntimeConfig,load_engine
from scripts.run_asd_joint_tree import execute

ROOT=Path.cwd();OUT=ROOT/'artifacts/strict-diagnosis-1';OUT.mkdir(exist_ok=False)


def save(name,value):
    (OUT/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')


def difference(a,b):
    left=a['token_ids'];right=b['token_ids']
    index=next((i for i,(x,y) in enumerate(zip(left,right)) if x!=y),min(len(left),len(right)))
    return dict(equal=left==right,index=index,left_tokens=len(left),right_tokens=len(right),
        left_id=left[index] if index<len(left) else None,right_id=right[index] if index<len(right) else None)


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
            positions=positions.reshape(-1).tolist()
            output=self.native(*args,**kwargs)
            offset=ids.shape[1]-output.logits.shape[1]
            wanted=[i for i,p in enumerate(positions) if p+1-self.prompt==self.index and i>=offset]
            if wanted:
                self.rows.append(dict(cache_before=before,vectors={i:output.logits[0,i-offset].detach().cpu().clone() for i in wanted}))
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


def summary(vector,first,second):
    values,ids=vector.topk(2)
    return dict(argmax_id=int(ids[0]),second_id=int(ids[1]),top_two_gap=float(values[0]-values[1]),
        relative_first_over_second=float(vector[first]-vector[second]))


@torch.inference_mode()
def main():
    info=dict(status='started',case='math500:323',purpose='Correctness only; no performance conclusion',
        source_commit='6d89480c52e0d89714778668c25ef7027a9f0ac8',physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save('provenance',info);engine=None
    try:
        task=next(r for r in load_records(ROOT) if r['id']==info['case'])
        old_refs=[json.loads(s) for s in (ROOT/'results/raw/asd-paper-extension/part-1/references.jsonl').read_text().splitlines()]
        old_ref=next(r for r in old_refs if r['request_id']==task['id'])
        old_rows=[json.loads(s) for s in (ROOT/'results/raw/asd-paper-extension/part-1/rows.jsonl').read_text().splitlines()]
        old_partial={r['arm']:r for r in old_rows if r['request_id']==task['id'] and r['stage']=='natural'}
        engine=load_engine(replace(RuntimeConfig.load(ROOT/'configs/runtime-dflash-rmsnorm.json'),db=None,max_new_tokens=4096))
        tree,_=frozen_config(ROOT,1)
        warm=dict(id='warmup',benchmark='gsm8k',question='What is 3 plus 4?',answer='#### 7')
        engine.generate(request_for(warm,32),'AR')
        for arm in ('DFlash','ASD','BudgetTree','BudgetTree-ASD'):execute(engine,warm,arm,64,tree,request_factory=request_for)
        execute(engine,warm,'BudgetTree-ASD',256,tree,request_factory=request_for,fixed=True)
        execute(engine,warm,'BudgetTree-ASD',64,tree,request_factory=request_for,zero=True)
        execute(engine,warm,'BudgetTree',64,tree,request_factory=request_for)
        info['runtime']=engine.describe();info['generation_config']=engine.target.generation_config.to_dict()
        reference=asdict(engine.generate(request_for(task,4096),'AR'));save('ar-replay',reference)
        info['ar_matches_original']=difference(reference,old_ref);info['replays']={};replays={}
        assert reference['prompt_sha256']==old_ref['prompt_sha256']
        for arm in ('BudgetTree-ASD','ASD','BudgetTree','DFlash'):
            row=execute(engine,task,arm,4096,tree,request_factory=request_for);save(arm,row);replays[arm]=row
            info['replays'][arm]=difference(row,reference)
            if arm in old_partial:info['replays'][arm]['matches_original_partial']=difference(row,old_partial[arm])
            save('provenance',info);print(arm,json.dumps(info['replays'][arm]),flush=True)
        failed=next((a for a in ('BudgetTree','DFlash') if not info['replays'][a]['equal']),None)
        if failed:
            mismatch=info['replays'][failed];index=mismatch['index']
            if mismatch['left_id'] is not None and mismatch['right_id'] is not None:
                prompt=reference['prompt_tokens'];ids=engine.tokenizer.apply_chat_template(list(request_for(task,4096).messages),
                    add_generation_prompt=True,return_tensors='pt',**engine.config.template_options).to(engine.target.device)
                with Capture(engine.target,prompt,index) as captured:
                    repeated=execute(engine,task,failed,4096,tree,request_factory=request_for)
                save('strict-instrumented',repeated);strict_vector=vector_for(captured,repeated,index,prompt)
                with Capture(engine.target,prompt,index) as ar_capture:
                    shortened=asdict(engine.generate(request_for(task,index+1),'AR'))
                save('ar-instrumented-prefix',shortened)
                ar_vector=next(iter(ar_capture.rows[-1]['vectors'].values()))
                common=torch.cat((ids,ids.new_tensor([reference['token_ids'][:index]])),dim=1)
                full=engine.target(common,attention_mask=torch.ones_like(common),use_cache=False,logits_to_keep=1).logits[0,-1].detach().cpu()
                first=reference['token_ids'][index];second=repeated['token_ids'][index]
                numeric=dict(index=index,strict_repeat_matches=difference(repeated,replays[failed]),
                    shortened_ar_matches_full=shortened['token_ids']==reference['token_ids'][:index+1],
                    strict_pass=summary(strict_vector,first,second),ar_pass=summary(ar_vector,first,second),
                    full_prefix_pass=summary(full,first,second),
                    strict_vs_ar_max_abs=float((strict_vector.double()-ar_vector.double()).abs().max()),
                    strict_vs_full_max_abs=float((strict_vector.double()-full.double()).abs().max()),
                    context=engine.tokenizer.decode(reference['token_ids'][max(0,index-6):index+6]),
                    rule='Every strict emitted token must equal its actual-pass argmax. No epsilon acceptance or prompt changes.')
                assert int(strict_vector.argmax())==second
                assert int(ar_vector.argmax())==shortened['token_ids'][-1]
                info['numeric']=numeric;save('numeric',numeric);print('NUMERIC',json.dumps(numeric),flush=True)
        info['status']='complete'
    except Exception as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        save('provenance',info)
        if engine is not None:engine.close()


if __name__=='__main__':main()
