from pathlib import Path
import datetime,hashlib,importlib.metadata,json,os,subprocess,sys,time

workspace=Path(__file__).resolve().parents[1];root=workspace/'artifacts/asd-paper-extension-local'
source=root/'ASD_PAPER_EXTENSION_SOURCE_MANIFEST.json';manifest=json.loads(source.read_bytes())
model=workspace/'artifacts/models/skywork-reward-v2-qwen3-8b'
output=root/'artifacts/reward-execution.json';output.parent.mkdir(exist_ok=True)
assert not output.exists()
info=dict(status='started',source_commit=manifest['commit'],source_sha256=manifest['source_sha256'],
    generation_source_commits=['6d89480c52e0d89714778668c25ef7027a9f0ac8','1d39c56e5a81c896077258433b476af1a934ce6f'],
    reused_previously_verified_checkpoint=True,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    python=sys.version,transformers=importlib.metadata.version('transformers'),
    supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def save():output.write_bytes((json.dumps(info,indent=2)+'\n').encode())
def verify():
    for name,digest in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    return len(manifest['files'])

info['source_files_verified_before']=verify();save()
env=dict(os.environ,HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',OMP_NUM_THREADS='4',PYTHONPATH='.',
    CUDA_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='false')
for name in ('GIT_DIR','GIT_WORK_TREE','GIT_COMMON_DIR'):env.pop(name,None)
started=time.perf_counter()
with (workspace/'artifacts/paper-extension-reward.log').open('x') as log:
    process=subprocess.Popen([sys.executable,'-u','-m','scripts.run_asd_reward','--study','asd-paper-extension',
        '--input','results/asd-paper-extension','--model-directory',str(model)],cwd=root,env=env,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    info['pid']=process.pid;save()
    try:code=process.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        if process.poll() is None:subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT,check=False)
        info['status']='timed_out';save();raise
reward=json.loads((root/'results/raw/asd-paper-extension/reward/provenance.json').read_bytes())
success=code==0 and reward['status']=='complete' and reward.get('rows')==320
info.update(status='complete' if success else 'failed',exit_code=code,reward_status=reward['status'],
    wall_seconds=time.perf_counter()-started,
    end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files_verified_after=verify())
save();print(json.dumps(info),flush=True)
if not success:raise SystemExit(code or 1)
