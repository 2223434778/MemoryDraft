"""Frozen source, owned GPU acceptance, and a bounded public-evidence archive."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import zipfile

from scripts.run_rmsnorm_runtime_pipeline import run_owned


def main():
    root=Path.cwd();name='INFRA_TREE_SOURCE_MANIFEST.json'
    manifest=json.loads((root/name).read_bytes());out=root/'results/raw/infra-tree-v04'
    out.mkdir(parents=True,exist_ok=False)
    def verify():
        for name,digest in manifest['files'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed: '+name)
        return len(manifest['files'])
    info=dict(status='started',commit=manifest['commit'],files_verified_before=verify())
    exe=str(root.parent/'.venv-dflash/bin/python')
    env=dict(os.environ,HF_HUB_OFFLINE='1',PYTHONPATH='.',OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0',
             TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_CACHE_DIR=str(root/'artifacts/inductor'),
             TRITON_CACHE_DIR=str(root/'artifacts/triton'))
    (root/'artifacts').mkdir(exist_ok=True)
    try:
        run_owned([exe,'-m','pytest','-q','tests/test_runtime_cost_tree.py','tests/test_runtime_asd.py',
            'tests/test_runtime.py','tests/test_serving.py','tests/test_budget_tree.py','tests/test_draft_tree.py',
            '--basetemp','artifacts/server-tests','--junitxml='+str(out/'tests-server.xml')],env,out/'tests.log',timeout=300)
        run_owned([exe,'-u','-m','scripts.validate_infra_tree'],env,out/'validation.log',timeout=1200)
        info.update(status='complete',files_verified_after=verify())
    except Exception as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        (out/'execution-verification.json').write_text(json.dumps(info,indent=2)+'\n')
        with zipfile.ZipFile(root.parent/(root.name+'-results.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob('*')):
                if path.is_file():
                    if path.suffix not in {'.json','.jsonl','.log','.xml'}:raise ValueError('Unexpected evidence type')
                    name=path.relative_to(out).as_posix()
                    if path.suffix=='.jsonl':archive.writestr(name+'.gz',gzip.compress(path.read_bytes(),mtime=0))
                    else:archive.write(path,name)
            archive.write(root/'INFRA_TREE_SOURCE_MANIFEST.json','INFRA_TREE_SOURCE_MANIFEST.json')
    print('INFRA TREE ACCEPTANCE COMPLETE',flush=True)


if __name__=='__main__':main()
