"""Independent trained pointwise reward audit, separate from generation timing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer

from memorydraft.evaluation.asd_paper_metrics import STUDY,message_sha,open_units
from scripts.diagnose_transport_online import write
from scripts.download_asd_reward import verify
from scripts.run_runtime_migration import append

ROOT=Path(__file__).resolve().parents[1]
CONTROLS=(
    ('arithmetic','What is 7 times 8? Reply with only the number.','56','59'),
    ('code','Write a Python function double(x) that returns twice x.','def double(x):\n    return 2*x','def double(x):\n    return x+2'),
    ('json','Return only a JSON object with key label and string value blue.','{"label":"blue"}','The label that you requested is blue.'),
    ('capital','What is the capital of France? Reply with only the city.','Paris','Berlin'),
    ('reverse','Reverse [1,2,3]. Return only the resulting list.','[3,2,1]','[1,2,3]'),
    ('count','Name exactly three additive primary colors.','Red, green, blue.','Red and blue.'),
    ('maximum','What is the largest number in [3,8,2]? Reply with only the number.','8','3'),
    ('translation','Translate the French greeting Bonjour into English.','Hello.','Goodbye.'),
)


def control_messages(question,answer):
    return [dict(role='user',content=question),dict(role='assistant',content=answer)]


def pair_score(candidate,baseline,*,identical=False):
    if candidate is None or baseline is None:return None
    if identical or abs(candidate-baseline)<=1e-4:return .5
    return float(candidate>baseline)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-directory',required=True)
    parser.add_argument('--study',choices=(STUDY,'asd-paper-extension'),default=STUDY)
    parser.add_argument('--input',type=Path)
    args=parser.parse_args();out=ROOT/f'results/raw/{args.study}/reward';out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'configs/asd-reward-model.json').read_bytes());directory=Path(args.model_directory)
    info=dict(status='started',model=manifest['repo'],revision=manifest['revision'],license=manifest['license'],
        max_input_tokens=manifest['max_input_tokens'],dtype='bfloat16',attention='sdpa',device='cuda:0',
        study=args.study,
        interpretation='trained reward-model preference proxy; not official MT-Bench or AlpacaEval, not calibrated correctness')
    write(out/'provenance.json',info)
    try:
        start=time.perf_counter()
        if not all(verify(directory/f['filename'],f) for f in manifest['files']):raise ValueError('Reward checkpoint hash mismatch')
        info['verification_seconds']=time.perf_counter()-start
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        start=time.perf_counter()
        tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
        model=AutoModelForSequenceClassification.from_pretrained(directory,local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa',num_labels=1).to('cuda:0').eval()
        info.update(load_seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(0),
                    torch=torch.__version__,model_class=type(model).__name__)
        def score(messages):
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,return_tensors='pt')
            count=int(ids.shape[1])
            if count>manifest['max_input_tokens']:return dict(score=None,status='too_long',prompt_tokens=count,seconds=0.)
            ids=ids.to('cuda:0');torch.cuda.synchronize();start=time.perf_counter()
            logits=model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits
            value=float(logits.squeeze().float());torch.cuda.synchronize()
            if not math.isfinite(value):raise ValueError('Non-finite reward')
            return dict(score=value,status='scored',prompt_tokens=count,seconds=time.perf_counter()-start)
        controls=[]
        for identity,question,chosen,rejected in CONTROLS:
            good=score(control_messages(question,chosen));bad=score(control_messages(question,rejected))
            row=dict(control_id=identity,question=question,chosen=chosen,rejected=rejected,
                good=good,bad=bad,passed=good['score']>bad['score'])
            controls.append(row);append(out/'controls.jsonl',row)
        repeated=[score(control_messages(CONTROLS[0][1],answer)) for answer in reversed(CONTROLS[0][2:])]
        repeat_error=max(abs(repeated[0]['score']-controls[0]['bad']['score']),abs(repeated[1]['score']-controls[0]['good']['score']))
        info.update(controls_passed=sum(r['passed'] for r in controls),controls_total=len(controls),repeat_max_abs_error=repeat_error)
        if info['controls_passed']<7 or repeat_error>1e-5:
            info.update(status='control_failed',rows=0);return
        if args.study==STUDY:
            if args.input is not None:raise ValueError('Original reward audit uses its frozen published inputs')
            units=open_units(ROOT)
        else:
            from memorydraft.evaluation.asd_paper_extension import open_units as extension_units
            units=extension_units(ROOT,args.input or ROOT/f'results/{args.study}')
        jobs=[dict(request_id=u['request_id'],benchmark=u['benchmark'],arm=arm,messages=messages)
              for u in units for arm,messages in u['answers'].items()]
        expected_rows=len(jobs);info['expected_rows']=expected_rows
        random.Random('MemoryDraft-pointwise-reward-v1').shuffle(jobs)
        cache={};count=0
        for job in jobs:
            messages=job.pop('messages');digest=message_sha(messages);cached=digest in cache
            if not cached:cache[digest]=score(messages)
            result=cache[digest]
            append(out/'rows.jsonl',dict(job,messages_sha256=digest,reused_identical=cached,**result));count+=1
        if count!=expected_rows:raise ValueError('Incomplete pointwise reward matrix')
        peak=torch.cuda.max_memory_allocated(0)
        if peak>22*2**30:raise ValueError('Local RTX3090 reward memory bound exceeded')
        info.update(status='complete',rows=count,unique_conversations=len(cache),peak_gpu_bytes=peak,
            timed_forward_seconds=sum(r['seconds'] for r in cache.values()))
    except Exception as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        write(out/'provenance.json',info)


if __name__=='__main__':main()
