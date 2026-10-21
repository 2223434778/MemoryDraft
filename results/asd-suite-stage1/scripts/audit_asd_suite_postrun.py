"""Task-local audit of frozen stage-1 source and protected evidence."""
from pathlib import Path
import ast, hashlib, json, subprocess, zipfile
from scripts.diagnose_transport_online import write
ROOT=Path.cwd(); base=ROOT/'results/asd-suite-stage1'
freeze='be1251b24f7a53f38fd2e817d68d62d4ab8e30d6'
sha=lambda b:hashlib.sha256(b).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
manifest=json.loads((base/'ASD_SUITE_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
assert manifest['commit']==freeze
source_map_sha=sha(json.dumps(manifest['files'],sort_keys=True).encode())
assert source_map_sha==manifest['source_sha256']
verification=json.loads((base/'execution-verification.json').read_text())
assert verification['commit']==freeze and verification['status']=='complete'
assert verification['files_verified_before']==verification['files_verified_after']==len(manifest['files'])
functions=[]
for path,names in {
 'scripts/write_asd_suite_report.py':['paired','aggregate'],
 'scripts/run_asd_suite.py':['execute','prefix_agreement','main'],
 'scripts/run_asd_suite_pipeline.py':['main'],
 'memorydraft/evaluation/asd_suite.py':['request_for','score','code_score','qa_score'],
 'memorydraft/backends/asd.py':['ASDVerifier'],
 'memorydraft/backends/dflash_reuse.py':['generate_reuse'],
}.items():
 old=git('show',freeze+':'+path).decode(); new=(ROOT/path).read_text(encoding='utf-8')
 for name in names:
  def segment(source):
   node=next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
   return ast.get_source_segment(source,node).encode()
  assert segment(old)==segment(new),(path,name)
  functions.append(dict(path=path,function=name,sha256=sha(segment(new))))
old_evidence={}
for i in (0,1):
 p=json.loads((base/f'part-{i}/provenance.json').read_text())
 assert p['status']=='complete' and p['original_unchanged']
 for k,v in p['original_sha256'].items():
  assert k not in old_evidence or old_evidence[k]==v
  old_evidence[k]=v
legacy=[]
allowed={'results/research-audit/model_preflight.json','results/research-audit/pilot_summary.json'}
for name,digest in old_evidence.items():
 raw=(ROOT/name).read_bytes()
 if sha(raw)!=digest:
  assert name in allowed and sha(raw.replace(b'\r\n',b'\n'))==digest,name
  legacy.append(name)
refs={r:git('rev-parse',r).decode().strip() for r in ('main','codex/infra-interview-v0.3','infra-v0.3-interview^{}')}
assert set(refs.values())=={'27e36199f8a65acdb9a9c42ccb744ddfa9a9402b'}
archive=ROOT/'artifacts/asd-suite-stage1-results.zip'
with zipfile.ZipFile(archive) as z:
 for n in z.namelist():assert sha((base/n).read_bytes())==sha(z.read(n)),n
write(base/'postrun-audit.json',dict(status='complete',source_commit=freeze,execution_files=len(manifest['files']),source_sha256=source_map_sha,
 frozen_functions=functions,old_evidence_verified=len(old_evidence),legacy_crlf_normalized=legacy,infra_refs=refs,
 result_zip=dict(bytes=archive.stat().st_size,sha256=sha(archive.read_bytes())),
 report_changes='Offline diagnostics expanded after launch: censored/common-EOS subsets, restricted-code labels, cost-per-forward factor, scalar compatibility and matrix tamper checks. Paired estimator, aggregation, prompts, scoring and execution unchanged.'))
print(json.dumps(dict(old_evidence=len(old_evidence),functions=len(functions),legacy=legacy)))
