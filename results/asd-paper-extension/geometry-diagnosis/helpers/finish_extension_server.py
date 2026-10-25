"""Wait for the closed formal archive, then audit geometry and add metadata."""
from concurrent.futures import ThreadPoolExecutor
import datetime,hashlib,json,os,subprocess,time,zipfile
from pathlib import Path

from scripts.run_rmsnorm_runtime_pipeline import run_owned

ROOT=Path.cwd();OUT=ROOT/'results/raw/asd-paper-extension'
TARGET=ROOT.parent/'asd-paper-extension-continuation-final-results.zip'
MANIFEST=ROOT/'ASD_PAPER_EXTENSION_CONTINUATION_SOURCE_MANIFEST.json'
info=dict(status='waiting_for_formal_pipeline',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Post-run diagnosis and archive only; never overlaps timed GPU work')


def save():
    (ROOT/'POSTPROCESS.json').write_text(json.dumps(info,indent=2)+'\n')


def verify():
    manifest=json.loads(MANIFEST.read_bytes())
    for name,digest in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    return len(manifest['files'])


def main():
    save();start=time.monotonic();original=ROOT.parent/(ROOT.name+'-results.zip')
    try:
        while True:
            p=OUT/'execution-verification.json'
            if p.exists():
                try:formal=json.loads(p.read_bytes())
                except json.JSONDecodeError:formal={}
                if formal.get('status')=='failed':raise RuntimeError('Formal pipeline failed; do not start GPU diagnostics')
                if formal.get('status')=='complete' and original.exists() and 'ASD BUDGET TREE COMPLETE' in (ROOT/'PIPELINE.log').read_text():break
            if time.monotonic()-start>24000:raise TimeoutError('Formal pipeline did not finish within postprocess wait budget')
            time.sleep(10)
        with zipfile.ZipFile(original) as archive:assert archive.testzip() is None
        info.update(status='diagnosing',formal_closed_archive_bytes=original.stat().st_size,
            formal_closed_archive_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
            files_verified_before=verify(),formal_finished_observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        idle=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
        assert len(idle.splitlines())==2 and all(int(line.split(',')[1].strip().split()[0])<100 for line in idle.splitlines()),idle
        info['gpu_before_diagnosis']=idle.splitlines();save()
        exe=str(ROOT.parent/'.venv-dflash/bin/python')
        def diagnose(shard):
            env=dict(os.environ,HF_HUB_OFFLINE='1',PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=str(shard),
                TORCHINDUCTOR_CACHE_DIR=str(ROOT/f'artifacts/geometry-inductor-{shard}'),
                TRITON_CACHE_DIR=str(ROOT/f'artifacts/geometry-triton-{shard}'))
            run_owned([exe,'-u','artifacts/diagnose_extension_geometry.py','--shard',str(shard)],env,
                ROOT/f'artifacts/geometry-diagnosis-{shard}.log',timeout=1800)
            result=json.loads((ROOT/f'artifacts/geometry-diagnosis-{shard}/provenance.json').read_bytes())
            assert result['status']=='complete' and result['files_verified_before']==result['files_verified_after']==2341
            return dict(shard=shard,comparisons=len(result['comparisons']),status=result['status'])
        with ThreadPoolExecutor(max_workers=2) as pool:info['diagnostics']=list(pool.map(diagnose,(0,1)))
        info.update(status='complete',files_verified_after=verify(),ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        with zipfile.ZipFile(original) as source,zipfile.ZipFile(TARGET,'x',zipfile.ZIP_DEFLATED) as dest:
            for name in source.namelist():dest.writestr(name,source.read(name))
            for name in ('LAUNCH.json','DEPLOYMENT.json','PIPELINE.log','HARDWARE_SNAPSHOTS.jsonl',
                         'ORIGINAL_FAILURE_CACHE_OBSERVATION.json','POSTPROCESS.json','POSTPROCESS_LAUNCH.json','POSTPROCESS.log'):
                dest.write(ROOT/name,'auxiliary/'+name)
            for shard in (0,1):
                for path in sorted((ROOT/f'artifacts/geometry-diagnosis-{shard}').glob('*.json')):
                    dest.write(path,f'geometry-diagnosis/part-{shard}/'+path.name)
                dest.write(ROOT/f'artifacts/geometry-diagnosis-{shard}.log',f'geometry-diagnosis/part-{shard}/run.log')
            for name in ('diagnose_extension_geometry.py','finish_extension_server.py'):
                dest.write(ROOT/'artifacts'/name,'geometry-diagnosis/helpers/'+name)
        receipt=dict(bytes=TARGET.stat().st_size,sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),
            original_bytes=original.stat().st_size,original_sha256=info['formal_closed_archive_sha256'],
            filename=str(TARGET),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (ROOT/'FINAL_ARCHIVE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    except Exception as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));save();raise


if __name__=='__main__':main()
