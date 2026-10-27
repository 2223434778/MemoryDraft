"""Synthetic raw-token check; run against an existing authenticated deployment."""

import argparse
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import re
import secrets
import shlex
import time

import httpx
from transformers import AutoTokenizer

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--chat-root',type=Path,required=True)
parser.add_argument('--url',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.chat_root
env={k:shlex.split(v)[0] for line in (root/'private/inference.env').read_text().splitlines() if line and not line.startswith('#') for k,v in [line.split('=',1)]}
cfg=json.loads((root/'private/runtime.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(cfg['target'],local_files_only=True,trust_remote_code=False)
request={'id':'draft-repetition-'+secrets.token_hex(5),'user_id':'synthetic-draft-repetition','order':time.time_ns(),
         'messages':[{'role':'user','content':'请用中文简洁介绍大语言模型能提供哪些帮助，以及它有哪些限制。'}],
         'max_new_tokens':128,'timeout_seconds':600}
output=args.output
report={'purpose':'synthetic raw candidate/display diagnosis, not a performance benchmark','real_gpu':True,
        'prompt':request['messages'][0]['content'],'output_limit':128,'source_release':(root/'current').resolve().name,
        'draft_block_slots':16,'confirmed_anchor_slots':1,'max_candidate_slots':15}
def write():
    tmp=output.with_suffix('.next.json');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(output)
write()
with httpx.Client(base_url=args.url,headers={'Authorization':'Bearer '+env['MEMORYDRAFT_API_KEY']},
                  trust_env=False,timeout=httpx.Timeout(connect=5,read=650,write=15,pool=5)) as client:
    response=client.post('/v1/generate',json={**request,'method':'AR'})
    response.raise_for_status();baseline=response.json()
    report['ar_output_tokens']=len(baseline['token_ids']);write()
    rows={};confirmed=[];final=None;inline_matches=0
    with client.stream('POST','/v1/generate',json={**request,'id':request['id']+'-draft','method':'DFlash',
                       'stream':True,'text_stream':True,'visualize':True}) as stream:
        stream.raise_for_status();name=None
        for line in stream.iter_lines():
            if line.startswith('event: '):name=line[7:]
            elif line.startswith('data: '):
                value=json.loads(line[6:])
                if name=='error':raise RuntimeError({'inference_error':value})
                if name=='tokens':confirmed.extend(value['token_ids'])
                elif name=='speculation':
                    number=value['round']
                    if value['phase']=='draft':
                        ids=value['proposal_ids']
                        assert [t['id'] for t in value['proposal']]==ids
                        before=tokenizer.decode(confirmed,skip_special_tokens=True).rstrip('\ufffd')
                        future=tokenizer.decode(confirmed+ids,skip_special_tokens=True).rstrip('\ufffd')
                        expected=future[len(os.path.commonprefix([before,future])):][:80]
                        assert value['inline']['candidate']==expected
                        inline_matches+=1
                        pairs=[{'index':i,'token_id':ids[i],'text':tokenizer.decode([ids[i]],skip_special_tokens=True)}
                               for i in range(len(ids)-1) if ids[i]==ids[i+1]]
                        rows[number]={'round':number,'proposal_ids':ids,'candidate':expected,
                                      'labels':[t['text'] for t in value['proposal']],'adjacent_pairs':pairs}
                    else:
                        row=rows[number];accepted=value['accepted']
                        assert value['committed_ids'][:accepted]==row['proposal_ids'][:accepted]
                        row['accepted']=accepted
                        row['fully_accepted_pairs']=sum(pair['index']+1<accepted for pair in row['adjacent_pairs'])
                elif name=='complete':final=value
    assert final is not None and final['token_ids']==confirmed
    values=list(rows.values());repeated=[row for row in values if row['adjacent_pairs']]
    word_repeated=[row for row in repeated if any(re.search(r'[\u4e00-\u9fffA-Za-z]',pair['text']) for pair in row['adjacent_pairs'])]
    examples=word_repeated[:5] if word_repeated else repeated[:5]
    positions=[{'position':i+1,'proposed':sum(len(row['proposal_ids'])>i for row in values),
                'accepted':sum(row['accepted']>i for row in values)} for i in range(15)]
    report.update({'passed':True,'draft_rounds':len(values),'contextual_inline_matches_raw_token_decode':inline_matches,
        'blocks_with_adjacent_repeated_ids':len(repeated),'blocks_with_repeated_word_ids':len(word_repeated),
        'adjacent_repeat_pairs':sum(len(row['adjacent_pairs']) for row in values),
        'fully_accepted_adjacent_repeat_pairs':sum(row['fully_accepted_pairs'] for row in values),
        'candidate_tokens_proposed':sum(len(row['proposal_ids']) for row in values),
        'candidate_tokens_accepted':sum(row['accepted'] for row in values),
        'dflash_output_tokens':len(final['token_ids']),'identical_ar_and_dflash_tokens':baseline['token_ids']==final['token_ids'],
        'ar_token_sha256':hashlib.sha256(json.dumps(baseline['token_ids']).encode()).hexdigest(),
        'dflash_token_sha256':hashlib.sha256(json.dumps(final['token_ids']).encode()).hexdigest(),
        'examples':examples,'position_counts':positions,'final_text_excerpt':final['text'][:240]})
    write()
print(json.dumps({key:report[key] for key in ['passed','draft_rounds','blocks_with_adjacent_repeated_ids',
      'blocks_with_repeated_word_ids','identical_ar_and_dflash_tokens','contextual_inline_matches_raw_token_decode']}))
