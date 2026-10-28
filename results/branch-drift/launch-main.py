import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

ROOT=Path('/media/a6000/D/ZhaoBinHe/MemoryDraft/branch-drift-v3')
PYTHON=ROOT.parent/'.venv-dflash/bin/python'
UNITS=('memorydraft-chat-large-adapter.service','memorydraft-chat-large-model.service')


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def verify():
    manifest=json.loads((ROOT/'BRANCH_DRIFT_SOURCE_MANIFEST.json').read_bytes())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source mismatch: '+name)
    return len(manifest['files'])


def owned():
    info={'status':'starting','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    states={}
    try:
        info['source_files_verified_before']=verify()
        with urllib.request.urlopen('http://127.0.0.1:8012/slots',timeout=10) as response:
            if any(slot['is_processing'] for slot in json.load(response)):
                raise RuntimeError('27B is processing a user generation; no services stopped')
        states={unit:subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0 for unit in UNITS}
        info['initial_active_services']=states
        subprocess.run(['systemctl','--user','stop',*UNITS],check=True)
        write(ROOT/'execution.json',info)
        commands=[
            [str(PYTHON),'-m','pytest','-q','tests/test_branch_drift.py','tests/test_branch_drift_evaluation.py',
             'tests/test_budget_tree.py','tests/test_joint_tree.py','tests/test_dflash_reuse.py',
             '--basetemp','artifacts/pytest-server','--junitxml=server-tests.xml','--tb=short'],
            [str(PYTHON),'-u','-m','scripts.run_branch_drift'],
        ]
        for command in commands:
            info['command']=command
            write(ROOT/'execution.json',info)
            result=subprocess.run(command,cwd=ROOT)
            if result.returncode:
                raise RuntimeError('Command failed with exit '+str(result.returncode))
        info['status']='completed'
    except BaseException as error:
        info.update(status='failed',error=type(error).__name__+': '+str(error))
        raise
    finally:
        restored={}
        for unit in reversed(UNITS):
            if states.get(unit):
                restored[unit]=subprocess.run(['systemctl','--user','start',unit]).returncode
        info['restore_exit_codes']=restored
        info['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        info['source_files_verified_after']=verify()
        write(ROOT/'execution.json',info)


if __name__=='__main__':
    import sys
    if '--owned' in sys.argv:
        owned()
    else:
        archive=ROOT/'source.zip'
        if hashlib.sha256(archive.read_bytes()).hexdigest()!='6f8421f045a70778563ba8de0fc9de8da7959ab6e74d1c8c1c0d39c5508f9f05':
            raise ValueError('Source ZIP checksum')
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not (ROOT/name).resolve().is_relative_to(ROOT):
                    raise ValueError('Invalid ZIP path')
            z.extractall(ROOT)
        (ROOT/'models').symlink_to(ROOT.parent/'models', target_is_directory=True)
        failed=ROOT/'results/raw/branch-drift'
        if failed.exists():
            if (failed/'rows.jsonl').exists() or json.loads((failed/'provenance.json').read_bytes())['status']!='failed':
                raise ValueError('Refuse to replace generated benchmark data')
            failed.rename(ROOT/'results/raw/branch-drift-startup-failed')
        if (ROOT/'execution.json').exists():
            (ROOT/'execution.json').rename(ROOT/'execution-startup-failed.json')
        print(json.dumps({'verified_files':verify()}))
        subprocess.run(['systemd-run','--user','--unit=memorydraft-branch-drift-v3','--property=Type=exec',
            '--working-directory='+str(ROOT),'--setenv=CUDA_VISIBLE_DEVICES=1','--setenv=OMP_NUM_THREADS=4',
            '--setenv=HF_HUB_OFFLINE=1','--setenv=PYTHONPATH=.',
            str(PYTHON),str(ROOT/'launch-main.py'),'--owned'],check=True)
