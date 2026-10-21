from pathlib import Path
import ast, hashlib, json, subprocess, zipfile
from scripts.diagnose_transport_online import write
root=Path.cwd();base=root/'results/asd-block-suite';freeze='b107e53213137268415d44bab0e954ae704fd727'
sha=lambda b:hashlib.sha256(b).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=root)
manifest=json.loads((base/'ASD_BLOCK_SUITE_SOURCE_MANIFEST.json').read_text())
info=json.loads((base/'execution-verification.json').read_text())
assert manifest['commit']==info['commit']==freeze and info['status']=='complete'
assert info['files_verified_before']==info['files_verified_after']==len(manifest['files'])==1216
assert sha(json.dumps(manifest['files'],sort_keys=True).encode())==manifest['source_sha256']
functions=[]
for path,names in {
 'scripts/run_asd_block_suite.py':['RecordedVerifier','records_and_positions','record_mechanism','run_question','main'],
 'scripts/run_asd_block_suite_pipeline.py':['main'],
 'scripts/write_asd_block_suite_report.py':['audit_events','audit','aggregate','estimate','compare','summarize'],
 'scripts/write_asd_suite_report.py':['paired'],
 'scripts/run_asd_block_probe.py':['experimental_block','run','prefix_check','check_native'],
}.items():
 old=git('show',freeze+':'+path).decode();new=(root/path).read_text(encoding='utf-8')
 for name in names:
  def segment(text):
   node=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
   return ast.get_source_segment(text,node).encode()
  assert segment(old)==segment(new),(path,name)
  functions.append(dict(path=path,function=name,sha256=sha(segment(new))))
for path in ('memorydraft/backends/dflash_reuse.py','memorydraft/backends/asd.py','memorydraft/runtime/engine.py','memorydraft/runtime/config.py','memorydraft/evaluation/asd_suite.py','scripts/run_asd_suite.py',
             'scripts/run_asd_block_suite.py','scripts/run_asd_block_suite_pipeline.py','scripts/write_asd_block_suite_report.py','docs/ASD_BLOCK_SUITE_PROTOCOL.md'):
 assert git('show',freeze+':'+path).decode()==(root/path).read_text(encoding='utf-8'),path
data_files={p:h for p,h in manifest['files'].items() if p.startswith('data/research/asd-suite/')}
for path,digest in data_files.items():assert sha((root/path).read_bytes())==digest,path
protected={}
for i in (0,1):
 p=json.loads((base/f'part-{i}/provenance.json').read_text())
 assert p['status']=='complete' and p['counts']==dict(main=1560,mechanism=72) and p['original_unchanged'] and p['split']=='development'
 for k,v in p['original_sha256'].items():
  assert k not in protected or protected[k]==v
  protected[k]=v
legacy=[]
for name,digest in protected.items():
 raw=(root/name).read_bytes()
 if sha(raw)!=digest:
  assert name in {'results/research-audit/model_preflight.json','results/research-audit/pilot_summary.json'} and sha(raw.replace(b'\r\n',b'\n'))==digest,name
  legacy.append(name)
publications=[]
for path,commit in (('results/asd-suite-stage1/publication.json','dcfc01de31b4e619d89872e8cb7487648aeb6ce5'),('results/asd-block-preflight/publication.json','2a4ab50f72098c9cba74450d951f276fbd8c25aa')):
 data=json.loads((root/path).read_text())
 for name,digest in data['files'].items():assert sha(git('show',commit+':'+name))==digest,name
 publications.append(dict(path=path,commit=commit,files_verified=len(data['files'])))
refs={ref:git('rev-parse',ref).decode().strip() for ref in ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}')}
assert set(refs.values())=={'27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'}
archive=root/'artifacts/asd-block-suite-results.zip'
with zipfile.ZipFile(archive) as z:
 for n in z.namelist():assert (base/n).read_bytes()==z.read(n),n
write(base/'postrun-audit.json',dict(status='complete',source_commit=freeze,source_sha256=manifest['source_sha256'],execution_files=1216,
 frozen_functions=functions,frozen_dataset_files_verified=len(data_files),protected_results_verified=len(protected),legacy_crlf_normalized=legacy,
 prior_publications=publications,infra_refs=refs,result_zip=dict(bytes=archive.stat().st_size,sha256=sha(archive.read_bytes())),
 note='Execution and predeclared statistics/policy/quality functions unchanged. Human-readable interpretation is postrun. Live roadmap is checked at each prior publication commit.'))
print(json.dumps(dict(protected_results=len(protected),frozen_functions=len(functions),prior_publications=publications)))
