"""Run GPU diagnostics only after the primary timed suite has released the GPU."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    base = root / "results/qwen3-pilot"
    deadline = time.monotonic() + 7200
    while not (base / "complete.txt").exists():
        status = base / "public-test/provenance.json"
        if status.exists():
            try:
                if json.loads(status.read_text()).get("status") == "failed":
                    raise RuntimeError("Primary suite failed; do not overlap diagnostics")
            except json.JSONDecodeError:
                pass
        if time.monotonic() > deadline:
            raise RuntimeError("Primary suite did not complete in time")
        time.sleep(10)
    env = dict(os.environ, OMP_NUM_THREADS="4", PYTHONPATH=str(root), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    for script in ("run_dense_prefix_check.py", "diagnose_saved_mismatch.py"):
        print("starting diagnostic", script, flush=True)
        subprocess.run([sys.executable, "-u", "scripts/" + script], cwd=root, env=env, check=True)
    (base / "post-checks-complete.txt").write_text("Dense prefix and saved mismatch diagnostics completed.\n")
