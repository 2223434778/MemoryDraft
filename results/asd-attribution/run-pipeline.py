from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gzip
import hashlib
import json
import os
import subprocess
import zipfile

root=Path.cwd()
exe=str(root.parent/'.venv-dflash/bin/python')
env=dict(os.environ,OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONPATH='.')
subprocess.run([exe,'-m','pytest','-q','tests/test_asd.py','tests/test_asd_budget.py',
                'tests/test_asd_completion.py','tests/test_asd_attribution.py',
                '--basetemp','artifacts/test-tmp','--junitxml=artifacts/tests-server.xml'],env=env,check=True)

def worker(gpu):
    selections=[('math-'+('a' if gpu==0 else 'b'),'math500',range(8+gpu*8,16+gpu*8)),
                ('gsm-'+('a' if gpu==0 else 'b'),'gsm8k',range(24+gpu*4,28+gpu*4))]
    for part,benchmark,indices in selections:
        command=[exe,'-u','-m','scripts.run_asd_budget_probe','--config','configs/runtime-ot-benefit.json',
                 '--data','data/research/asd-attribution/records.json','--policies','configs/asd-attribution-policies.json',
                 '--benchmark',benchmark,'--task-ids',*[f'{benchmark}:{i}' for i in indices],
                 '--max-new-tokens','2048','--prefix-tokens','128','--protocol','docs/ASD_ATTRIBUTION_PROTOCOL.md',
                 '--output','results/raw/asd-attribution/'+part]
        print('START',gpu,part,json.dumps(command),flush=True)
        with (root/(part+'.log')).open('w') as log:
            result=subprocess.run(command,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
        print('FINISH',gpu,part,result.returncode,flush=True)
        if result.returncode:
            raise RuntimeError(part+' failed '+str(result.returncode))

with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(worker,[0,1]))
manifest=json.loads((root/'ASD_ATTRIBUTION_SOURCE_MANIFEST.json').read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in manifest['files'].items())
path=root/'execution-verification.json'
v=json.loads(path.read_text())
v.update(files_verified_after=len(manifest['files']),status='complete')
path.write_text(json.dumps(v,indent=2))
with zipfile.ZipFile(root.parent/'asd-attribution-results.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted((root/'results/raw/asd-attribution').rglob('*')):
        if not p.is_file():continue
        name=p.relative_to(root/'results/raw/asd-attribution').as_posix()
        if name.endswith('.jsonl'):z.writestr(name+'.gz',gzip.compress(p.read_bytes(),mtime=0))
        else:z.write(p,name)
    for name in ('ASD_ATTRIBUTION_SOURCE_MANIFEST.json','execution-verification.json','run-pipeline.py',
                 'math-a.log','math-b.log','gsm-a.log','gsm-b.log','artifacts/tests-server.xml'):
        z.write(root/name,name.replace('artifacts/',''))
print('PIPELINE COMPLETE',flush=True)
