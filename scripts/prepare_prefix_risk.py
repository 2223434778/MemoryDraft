"""Freeze old-only calibration and new authored projects before any new Target run."""
from __future__ import annotations

import json
from pathlib import Path

from memorydraft.evaluation.runtime_migration import file_hash, read_rows
from memorydraft.retrieval.acceptance import AcceptancePredictor

ROOT = Path(__file__).resolve().parents[1]


def cases():
    # Inputs/expected edits are authored without observing Target outputs.
    specifications = [
        ("archive-pruner", "edited-continuation",
         "SELECT object_id, storage_path, created_at\nFROM archive_objects\nWHERE retention_days > 45\n  AND deletion_marked = false\n  AND legal_hold = false\nORDER BY created_at ASC, object_id ASC\nLIMIT 240;\n\n-- Cleanup runs after the daily inventory transaction.\n-- Preserve objects with an active legal hold.\n-- Record the selected object identifiers before deletion.\n-- Retry transient storage failures in the next scheduled run.\n",
         [("retention_days > 45", "retention_days > 60"),
          ("  AND legal_hold = false\n", "  AND legal_hold = false\n  AND replication_complete = true\n")]),
        ("vector-ingest", "edited-continuation",
         '[ingest]\nservice = "vector-ingest"\nbatch_size = 160\nflush_interval_ms = 800\nordered_delivery = true\n\n[queue]\nname = "document-events"\nvisibility_timeout_seconds = 90\nmax_delivery_attempts = 4\n\n[storage]\nnamespace = "search-documents"\ncompression = "zstd"\nwrite_checksum = true\n\n[metrics]\nenabled = true\nlabels = ["service", "region", "result"]\nreport_interval_seconds = 30\n',
         [("batch_size = 160", "batch_size = 224"),
          ('ordered_delivery = true\n', 'ordered_delivery = true\nregion = "ap-southeast"\n')]),
        ("blob-recovery", "edited-continuation",
         "Blob recovery procedure\n\nStep 1: Pause the recovery worker before inspecting the manifest.\nStep 2: Set the retry interval to 18 seconds.\nStep 3: Read the newest committed manifest from the control bucket.\nStep 4: Compare each object checksum with its manifest entry.\nStep 5: Restore missing objects into the staging namespace.\nStep 6: Verify the restored object count before enabling reads.\nStep 7: Resume the recovery worker and observe the error counter.\nStep 8: Record the manifest identifier in the incident log.\n",
         [("18 seconds", "27 seconds"),
          ("Step 2: Set the retry interval to 18 seconds.\n", "Step 2: Set the retry interval to 18 seconds and enable checksum logging.\n")]),
        ("quota-ledger", "edited-continuation",
         '{\n  "service": "quota-ledger",\n  "revision": 3,\n  "limits": {\n    "burst_requests": 360,\n    "sustained_requests": 120,\n    "window_seconds": 60\n  },\n  "storage": {\n    "table": "tenant_quota_events",\n    "partition_key": "tenant_id",\n    "retention_days": 14\n  },\n  "audit": {\n    "enabled": true,\n    "include_request_id": true,\n    "fields": ["tenant_id", "operation", "result", "duration_ms"]\n  }\n}\n',
         [('"burst_requests": 360', '"burst_requests": 480'),
          ('  "revision": 3,\n', '  "revision": 3,\n  "region": "west",\n')]),
        ("scheduler-copy", "control",
         '[scheduler]\nname = "snapshot-scheduler"\ninterval_seconds = 45\nmax_parallel_jobs = 3\n\n[lease]\nrenewal_seconds = 10\nexpiry_seconds = 35\n\n[retry]\nattempts = 4\nbackoff_seconds = 8\n\n[logging]\nlevel = "info"\ninclude_job_id = true\ninclude_lease_id = true\n',
         [("interval_seconds = 45", "interval_seconds = 45"), ("attempts = 4", "attempts = 4")]),
        ("status-rewrite", "control",
         "The export queue is paused. Three workers are waiting for a storage lease. The last completed batch belongs to the northern region. The current incident is assigned to the daytime operations team.\n",
         [("The export queue is paused. Three workers are waiting for a storage lease. The last completed batch belongs to the northern region. The current incident is assigned to the daytime operations team.",
           "The import service is healthy. All partitions have completed validation. The next maintenance window begins at midnight. No operator action is required."),
          ("The export queue is paused. Three workers are waiting for a storage lease. The last completed batch belongs to the northern region. The current incident is assigned to the daytime operations team.",
           "The indexing job has finished. Every document passed the consistency check. The search endpoint is ready for traffic. The release manager has approved deployment.")]),
    ]
    projects = []
    for name, group, source, edits in specifications:
        user = "prefix-risk-" + name
        history = [dict(user_id=user, record_id="source-v1", order=2, text=source, project_id=name)]
        requests = []
        for i, (old, new) in enumerate(edits):
            assert source.count(old) == 1
            instruction = ("Return the supplied artifact unchanged." if old == new else
                           "Replace exactly this text:\n" + old + "\nwith:\n" + new + "\nChange nothing else.")
            request = dict(id=f"{name}:{i}", user_id=user, order=10 + i, max_new_tokens=512,
                messages=[dict(role="system", content="Return only the complete resulting artifact. Do not add Markdown fences or explanations. Preserve all unchanged text and whitespace."),
                          dict(role="user", content=instruction + "\n\nSOURCE BEGIN\n" + source + "SOURCE END\n")])
            requests.append(dict(request=request, edit=dict(old=old, new=new), expected_text=source.replace(old, new, 1)))
        projects.append(dict(id=name, group=group, source=source, history=history, requests=requests))
    return dict(schema=1, origin="Six new authored controlled projects; not production traffic or a history-necessity test", projects=projects)


def main():
    training = ROOT / "results/acceptance/collection/states.jsonl.gz"
    states = read_rows(training.with_suffix(""))
    fitted = {name: AcceptancePredictor.fit(states, excluded_project="__unseen_prefix_risk_projects__", confidence=enabled)
              for name, enabled in (("Risk-Splice", True), ("Risk-NoConfidence", False))}
    data = dict(schema=1, training_sha256=file_hash(training), training_states=len(states),
                training_projects=sorted({s["project_id"] for s in states}),
                models={name: dict(weights=m.prefix_weights, confidence=m.confidence) for name, m in fitted.items()},
                fit="Unchanged acceptance.py regularized project-balanced logistic; tail head discarded")
    directory = ROOT / "data/research/prefix-risk"
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in (("cases.json", cases()), ("calibration.json", data)):
        (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: data[k] for k in ("training_sha256", "training_states", "training_projects")}))


if __name__ == "__main__":
    main()
