from pathlib import Path
import subprocess,os,sys,json,hashlib,zipfile,gzip
root=Path.cwd(); exe=str(root.parent/'.venv-dflash/bin/python')
env=dict(os.environ,OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONPATH='.')
subprocess.run([exe,'-m','pytest','-q','tests/test_asd.py','tests/test_asd_budget.py','--basetemp','artifacts/test-tmp','--junitxml=artifacts/tests-server.xml'],env=env,check=True)
processes=[]; logs=[]
for gpu,bench in enumerate(('gsm8k','math500')):
 log=open(root/(bench+'.log'),'w'); logs.append(log)
 processes.append(subprocess.Popen([exe,'-u','-m','scripts.run_asd_budget_probe','--config','configs/runtime-ot-benefit.json','--benchmark',bench,'--output','results/raw/asd-budget/'+bench],env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT))
codes=[p.wait() for p in processes]
for log in logs: log.close()
if any(codes): raise RuntimeError('Probe exit codes '+str(codes))
manifest=json.loads((root/'ASD_BUDGET_SOURCE_MANIFEST.json').read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in manifest['files'].items())
verification=json.loads((root/'execution-verification.json').read_text()); verification['files_verified_after']=len(manifest['files']); verification['status']='complete'
(root/'execution-verification.json').write_text(json.dumps(verification,indent=2))
with zipfile.ZipFile(root.parent/'asd-budget-results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted((root/'results/raw/asd-budget').rglob('*')):
  if not p.is_file(): continue
  name=p.relative_to(root/'results/raw/asd-budget').as_posix()
  if name.endswith('.jsonl'): z.writestr(name+'.gz',gzip.compress(p.read_bytes(),mtime=0))
  else: z.write(p,name)
 for name in ('ASD_BUDGET_SOURCE_MANIFEST.json','execution-verification.json','run-pipeline.py','gsm8k.log','math500.log','artifacts/tests-server.xml'):
  z.write(root/name,name.replace('artifacts/',''))
print('PIPELINE COMPLETE',flush=True)
