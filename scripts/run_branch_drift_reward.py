"""Independent reward proxy for matched open-ended branch-drift generations."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from memorydraft.evaluation.asd_paper import request_for
from memorydraft.evaluation.branch_drift import ARMS
from scripts.download_asd_reward import verify
from scripts.run_asd_reward import CONTROLS, control_messages, pair_score
from scripts.diagnose_transport_online import write
from scripts.run_runtime_migration import append

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-directory',type=Path,required=True)
    parser.add_argument('--input',type=Path,default=Path('results/raw/branch-drift'))
    args = parser.parse_args()
    root = ROOT/args.input
    out = root/'reward'
    out.mkdir(exist_ok=False)
    manifest = json.loads((ROOT/'configs/asd-reward-model.json').read_bytes())
    info = dict(status='started',model=manifest['repo'],revision=manifest['revision'],
        dtype='bfloat16',attention='sdpa',device='cuda:0',max_input_tokens=manifest['max_input_tokens'],
        scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_manifest_sha256=hashlib.sha256((ROOT/'configs/asd-reward-model.json').read_bytes()).hexdigest(),
        interpretation='Independent trained reward proxy, not official MT-Bench/AlpacaEval or correctness')
    write(out/'provenance.json',info)
    try:
        if not all(verify(args.model_directory/f['filename'],f) for f in manifest['files']):
            raise ValueError('Reward checkpoint hash mismatch')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model_directory,local_files_only=True,trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_directory,local_files_only=True,
            trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation='sdpa',num_labels=1).to('cuda:0').eval()
        info['load_seconds'] = time.perf_counter()-started
        def score(messages):
            ids = tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,return_tensors='pt')
            if ids.shape[1] > manifest['max_input_tokens']:
                return None
            ids = ids.to('cuda:0')
            value = float(model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits.squeeze().float())
            if not math.isfinite(value):
                raise ValueError('Non-finite reward')
            return value
        controls = []
        for identity, question, good, bad in CONTROLS:
            a,b = score(control_messages(question,good)),score(control_messages(question,bad))
            controls.append(dict(id=identity,good=a,bad=b,passed=a is not None and b is not None and a>b))
        write(out/'controls.json',controls)
        info['controls_passed'] = sum(r['passed'] for r in controls)
        if info['controls_passed'] < 7:
            info['status'] = 'control_failed'
            return
        tasks = json.loads((root/'tasks.json').read_bytes())
        selected = {r['id']:r for r in tasks if r['benchmark'] in ('alpaca','mtbench')}
        refs = {r['request_id']:r for r in json.loads((root/'references.json').read_bytes())}
        rows = [json.loads(line) for line in (root/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
        cells = defaultdict(dict)
        cache = {}
        for row in rows:
            if row['stage'] != 'natural' or row['benchmark'] not in ('alpaca','mtbench'):
                continue
            identity = row['request_id']
            task_id = identity.rsplit(':turn',1)[0] if row['benchmark']=='mtbench' else identity
            task = selected[task_id]
            if row['benchmark']=='mtbench':
                from memorydraft.evaluation.asd_paper import conversation_record
                turn = int(identity[-1])
                task = conversation_record(task,turn,refs[task_id+':turn1']['text'] if turn==2 else None)
            messages = [*request_for(task,4096).messages,dict(role='assistant',content=row['text'])]
            key = json.dumps(messages,ensure_ascii=False,sort_keys=True)
            if key not in cache:
                cache[key] = score(messages)
            result = dict(request_id=identity,benchmark=row['benchmark'],arm=row['arm'],score=cache[key])
            cells[identity][row['arm']] = (row['text'],cache[key])
            append(out/'rows.jsonl',result)
        summary = []
        for benchmark in ('mtbench','alpaca'):
            for arm in ('CoverageTree-ASD','DriftTree-ASD'):
                for baseline in ('ASD','BudgetTree','BudgetTree-ASD','CoverageTree-ASD'):
                    if arm==baseline:
                        continue
                    values = []
                    for identity,answers in cells.items():
                        if refs[identity]['benchmark'] != benchmark:
                            continue
                        if set(answers) != set(ARMS):
                            raise ValueError('Incomplete open reward matrix')
                        a,b = answers[arm],answers[baseline]
                        value = pair_score(a[1],b[1],identical=a[0]==b[0])
                        if value is not None:
                            values.append(value)
                    summary.append(dict(benchmark=benchmark,arm=arm,baseline=baseline,n=len(values),
                        wins=values.count(1.),ties=values.count(.5),losses=values.count(0.),
                        preference_rate=sum(values)/len(values) if values else None))
        write(out/'analysis.json',summary)
        scores=[value[1] for answers in cells.values() for value in answers.values()]
        info.update(status='completed',rows=len(scores),scored_rows=sum(v is not None for v in scores),
            unscored_rows=sum(v is None for v in scores),unique_inputs=len(cache),
            gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__))
    except BaseException as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error))
        raise
    finally:
        write(out/'provenance.json',info)


if __name__ == '__main__':
    main()
