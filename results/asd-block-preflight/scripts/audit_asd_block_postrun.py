from pathlib import Path
import ast,hashlib,json,subprocess,zipfile
from scripts.diagnose_transport_online import write
root=Path.cwd();base=root/'results/asd-block-preflight';freeze='852a08e0112252799021a4e50319c3f140437deb'
sha=lambda b:hashlib.sha256(b).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=root)
manifest=json.loads((base/'ASD_BLOCK_SOURCE_MANIFEST.json').read_text())
info=json.loads((base/'execution-verification.json').read_text())
assert manifest['commit']==info['commit']==freeze and info['status']=='complete'
assert info['files_verified_before']==info['files_verified_after']==len(manifest['files'])==1184
assert sha(json.dumps(manifest['files'],sort_keys=True).encode())==manifest['source_sha256']
functions=[]
for name in ('experimental_block','selected_records','run','check_native','prefix_check','main'):
 path='scripts/run_asd_block_probe.py';old=git('show',freeze+':'+path).decode();new=(root/path).read_text()
 def segment(text):
  node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
  return ast.get_source_segment(text,node).encode()
 assert segment(old)==segment(new),name
 functions.append(dict(path=path,function=name,sha256=sha(segment(new))))
for path in ('memorydraft/backends/dflash_reuse.py','memorydraft/backends/asd.py','memorydraft/runtime/engine.py','memorydraft/runtime/config.py','memorydraft/evaluation/asd_suite.py','scripts/run_asd_suite.py'):
 assert git('show',freeze+':'+path).decode()==(root/path).read_text(encoding='utf-8'),path
protected={}
for i in (0,1):
 p=json.loads((base/f'part-{i}/provenance.json').read_text())
 assert p['status']=='complete' and p['rows']==114 and p['original_unchanged']
 for k,v in p['original_sha256'].items():
  assert k not in protected or protected[k]==v
  protected[k]=v
legacy=[]
for name,digest in protected.items():
 raw=(root/name).read_bytes()
 if sha(raw)!=digest:
  assert name in {'results/research-audit/model_preflight.json','results/research-audit/pilot_summary.json'} and sha(raw.replace(b'\r\n',b'\n'))==digest,name
  legacy.append(name)
publication=json.loads((root/'results/asd-suite-stage1/publication.json').read_text())
previous_commit='dcfc01de31b4e619d89872e8cb7487648aeb6ce5'
for path,digest in publication['files'].items():assert sha(git('show',previous_commit+':'+path))==digest,path
refs={ref:git('rev-parse',ref).decode().strip() for ref in ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}')}
assert set(refs.values())=={'27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'}
archive=root/'artifacts/asd-block-preflight-results.zip'
with zipfile.ZipFile(archive) as z:
 for n in z.namelist():assert (base/n).read_bytes()==z.read(n),n
write(base/'postrun-audit.json',dict(status='complete',source_commit=freeze,source_sha256=manifest['source_sha256'],execution_files=1184,
 frozen_runner_functions=functions,protected_results_verified=len(protected),legacy_crlf_normalized=legacy,
 previous_publication=dict(commit=previous_commit,files_verified=len(publication['files']),note='Roadmap is a live status file updated for the next stage; prior SHA is checked at its publication commit.'),
 infra_refs=refs,result_zip=dict(bytes=archive.stat().st_size,sha256=sha(archive.read_bytes())),
 note='Reporter and presentation audits added after execution; runner, prompts, scoring, production runtime and policies unchanged.'))
print(json.dumps(dict(protected_results=len(protected),frozen_functions=len(functions),prior_publication=len(publication['files']))))
