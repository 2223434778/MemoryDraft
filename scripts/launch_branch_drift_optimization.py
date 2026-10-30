"""Isolated A6000 study owner; restore initially active 27B services in finally."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import zipfile

UNITS = ('memorydraft-chat-large-adapter.service', 'memorydraft-chat-large-model.service')


def verify(root):
    manifest = json.loads((root/'BRANCH_DRIFT_SOURCE_MANIFEST.json').read_bytes())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Source mismatch: '+name)
    return len(manifest['files'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive-sha', required=True)
    parser.add_argument('--owned', action='store_true')
    parser.add_argument('--study', choices=('branch-drift-optimization', 'gate-width', 'gate-min-gain'), default='branch-drift-optimization')
    args = parser.parse_args()
    root = args.root.resolve()
    python = root.parent/'.venv-dflash/bin/python'
    if not args.owned:
        archive = root/'source.zip'
        if hashlib.sha256(archive.read_bytes()).hexdigest() != args.archive_sha:
            raise ValueError('Archive checksum mismatch')
        if (root/'BRANCH_DRIFT_SOURCE_MANIFEST.json').exists():
            raise ValueError('Refuse to overwrite an extracted study')
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                if not (root/name).resolve().is_relative_to(root):
                    raise ValueError('Invalid archive path')
            handle.extractall(root)
        (root/'models').symlink_to(root.parent/'models', target_is_directory=True)
        print(json.dumps(dict(verified_files=verify(root))), flush=True)
        unit = 'memorydraft-'+root.name
        subprocess.run(['systemd-run', '--user', '--unit='+unit,
            '--property=Type=exec', '--working-directory='+str(root),
            '--setenv=CUDA_VISIBLE_DEVICES=1', '--setenv=OMP_NUM_THREADS=4',
            '--setenv=HF_HUB_OFFLINE=1', '--setenv=PYTHONPATH=.', str(python),
            str(root/'scripts/launch_branch_drift_optimization.py'), '--root', str(root),
            '--archive-sha', args.archive_sha, '--study', args.study, '--owned'], check=True)
        return
    info = dict(status='starting', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                archive_sha256=args.archive_sha, study=args.study)
    states = {}
    def save():
        (root/'execution.json').write_text(json.dumps(info, indent=2)+'\n')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        info['source_files_verified_before'] = verify(root)
        with urllib.request.urlopen('http://127.0.0.1:8012/slots', timeout=10) as response:
            if any(slot['is_processing'] for slot in json.load(response)):
                raise RuntimeError('27B busy; no services stopped')
        states = {u: subprocess.run(['systemctl', '--user', 'is-active', '--quiet', u]).returncode == 0
                  for u in UNITS}
        info['initial_active_services'] = states
        subprocess.run(['systemctl', '--user', 'stop', *UNITS], check=True)
        (root/'artifacts').mkdir(exist_ok=True)
        commands = [[str(python), '-m', 'pytest', '-q', 'tests/test_branch_drift.py',
            'tests/test_branch_drift_optimization.py', 'tests/test_branch_drift_evaluation.py',
            'tests/test_budget_tree.py', 'tests/test_joint_tree.py', 'tests/test_dflash_reuse.py',
            '--basetemp', 'artifacts/pytest-server-opt', '--junitxml=server-tests.xml', '--tb=short'],
            [str(python), '-u', '-m', 'scripts.run_'+args.study.replace('-', '_')]]
        if args.study == 'gate-width':
            commands[0].insert(4, 'tests/test_gate_width.py')
        if args.study == 'gate-min-gain':
            commands[0] = [str(python), '-m', 'pytest', '-q', 'tests/test_gate_min_gain.py',
                'tests/test_gate_width.py', 'tests/test_budget_tree.py', 'tests/test_branch_drift.py',
                '--basetemp', 'artifacts/pytest-server-min-gain', '--junitxml=server-tests.xml', '--tb=short']
        for command in commands:
            info['command'] = command
            save()
            subprocess.run(command, cwd=root, check=True)
        info['status'] = 'completed'
    except BaseException as error:
        info.update(status='failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        info['restore_exit_codes'] = {u: subprocess.run(['systemctl', '--user', 'start', u]).returncode
                                      for u in reversed(UNITS) if states.get(u)}
        info['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        info['source_files_verified_after'] = verify(root)
        save()


if __name__ == '__main__':
    main()
