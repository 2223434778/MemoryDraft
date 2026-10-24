"""Bounded AR-only supplemental campaign; original evidence stays read-only."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import zipfile

from memorydraft.evaluation.asd_paper_metrics import STUDY
from scripts.run_rmsnorm_runtime_pipeline import run_owned


def main():
    root=Path.cwd();manifest=json.loads((root/'ASD_PAPER_METRICS_SOURCE_MANIFEST.json').read_bytes())
    out=root/f'results/raw/{STUDY}';out.mkdir(parents=True,exist_ok=False)
    def verify():
        for name,digest in manifest['files'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed: '+name)
        return len(manifest['files'])
    info=dict(status='started',commit=manifest['commit'],files_verified_before=verify())
    exe=str(root.parent/'.venv-dflash/bin/python')
    env=dict(os.environ,HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONPATH='.',OMP_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2')
    try:
        run_owned([exe,'-m','pytest','-q','tests/test_asd_paper_metrics.py','tests/test_asd_paper.py',
            '--basetemp','artifacts/test-tmp','--junitxml='+str(out/'tests-server.xml')],env,out/'tests.log',timeout=300)
        def worker(shard):
            cache=dict(TORCHINDUCTOR_CACHE_DIR=str(root/f'artifacts/inductor-{shard}'),TRITON_CACHE_DIR=str(root/f'artifacts/triton-{shard}'))
            for directory in cache.values():Path(directory).mkdir(parents=True,exist_ok=False)
            run_owned([exe,'-u','-m','scripts.run_asd_paper_metrics','--shard',str(shard)],
                      dict(env,**cache,CUDA_VISIBLE_DEVICES=str(shard)),out/f'part-{shard}.log',timeout=7200)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,(0,1)))
        info.update(status='complete',files_verified_after=verify())
    except Exception as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        (out/'execution-verification.json').write_text(json.dumps(info,indent=2)+'\n')
        with zipfile.ZipFile(root.parent/(root.name+'-results.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob('*')):
                if not path.is_file():continue
                relative=path.relative_to(out).as_posix()
                if path.suffix=='.jsonl':archive.writestr(relative+'.gz',gzip.compress(path.read_bytes(),mtime=0))
                elif path.suffix in ('.json','.xml','.log'):archive.write(path,relative)
                else:raise ValueError('Unexpected result payload')
            archive.write(root/'ASD_PAPER_METRICS_SOURCE_MANIFEST.json','ASD_PAPER_METRICS_SOURCE_MANIFEST.json')


if __name__=='__main__':main()
