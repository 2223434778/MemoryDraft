"""Export new, authored controlled tasks before any Target output is inspected."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from memorydraft.memory.store import MemoryEvent
from memorydraft.runtime.contracts import GenerationRequest


SYSTEM = ("Edit the supplied artifact exactly as requested. Return only the complete resulting artifact, "
          "without Markdown fences, introductions or explanations. Preserve all other text and whitespace.")


def authored_projects():
    projects = []

    def add(name, group, kind, source, edits, *, distractors=(), history=True):
        user = "otv2-" + name
        events = [asdict(MemoryEvent(user, f"distractor-{i}", 1, text, project_id=name))
                  for i, text in enumerate(distractors)]
        if history:
            events.append(asdict(MemoryEvent(user, "source-v1", 2, source, project_id=name)))
        requests = []
        for turn, (instruction, old, new) in enumerate(edits):
            if old not in source:
                raise ValueError("Authored edit must refer to the current input")
            expected = source.replace(old, new, 1)
            prompt = instruction + "\n\nSOURCE BEGIN\n" + source + "SOURCE END\n"
            request = GenerationRequest(f"{name}:{turn}", user, 10 + turn,
                                        ({"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": prompt}), 512)
            requests.append({"request": asdict(request), "edit": {"old": old, "new": new},
                             "expected_text": expected})
        projects.append(dict(id=name, group=group, kind=kind, source=source, history=events, requests=requests))

    source = '''{
  "service": "parcel-dispatch",
  "queue_capacity": 48,
  "consumer": {
    "batch_size": 12,
    "visibility_timeout_seconds": 90,
    "retry": {
      "attempts": 4,
      "initial_delay_ms": 250,
      "maximum_delay_ms": 8000,
      "jitter": true
    }
  },
  "dead_letter": {
    "enabled": true,
    "retention_days": 14,
    "alert_after_messages": 30
  },
  "telemetry": {
    "sample_rate": 0.1,
    "include_payload": false
  }
}
'''
    add("parcel-queue", "local-edit", "json", source,
        [("Change only queue_capacity from 48 to 72.", '"queue_capacity": 48', '"queue_capacity": 72'),
         ("Add the root field delivery_mode with value durable immediately after service.",
          '"service": "parcel-dispatch",', '"service": "parcel-dispatch",\n  "delivery_mode": "durable",')])
    source = '''from pathlib import Path
import hashlib


def store_artifact(payload, directory, filename="bundle.bin"):
    """Store one artifact and return its content digest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    temporary = destination.with_suffix(destination.suffix + ".pending")
    temporary.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    temporary.replace(destination)
    manifest = destination.with_suffix(destination.suffix + ".sha256")
    manifest.write_text(digest + "  " + filename + "\\n", encoding="utf-8")
    return {"path": str(destination), "sha256": digest, "bytes": len(payload)}
'''
    add("artifact-store", "local-edit", "python", source,
        [("Change only the default filename from bundle.bin to archive.dat.", 'filename="bundle.bin"', 'filename="archive.dat"'),
         ("Change only the temporary suffix from .pending to .staging.", 'suffix + ".pending"', 'suffix + ".staging"')])
    source = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: catalog-reader
spec:
  replicas: 3
  selector:
    matchLabels:
      app: catalog-reader
  template:
    metadata:
      labels:
        app: catalog-reader
    spec:
      containers:
        - name: reader
          image: registry.example/catalog:1.7
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            periodSeconds: 5
          resources:
            requests:
              cpu: 250m
              memory: 256Mi
'''
    add("catalog-deploy", "local-edit", "yaml", source,
        [("Change only replicas from 3 to 5.", 'replicas: 3', 'replicas: 5'),
         ("Change only the container image tag from 1.7 to 1.8.", 'catalog:1.7', 'catalog:1.8')])
    source = '''# Payment queue recovery

Owner: payments-platform
Severity: SEV-2

## Establish scope
1. Record the first alert timestamp and affected region.
2. Compare queue depth with the previous fifteen minutes.
3. Confirm that the producer error rate remains below one percent.

## Restore consumers
1. Pause the scheduled backfill before changing replica counts.
2. Restart one consumer and wait for its readiness check.
3. Increase capacity only after the oldest message age decreases.

## Validate recovery
1. Verify that the dead-letter queue is not growing.
2. Keep the incident open for two complete sampling intervals.
3. Record the final backlog size and link the dashboard snapshot.
'''
    add("payment-runbook", "local-edit", "markdown", source,
        [("Change only the severity from SEV-2 to SEV-1.", 'Severity: SEV-2', 'Severity: SEV-1'),
         ("Insert 'Region: eu-west' on a new line immediately after the Owner line.",
          'Owner: payments-platform\n', 'Owner: payments-platform\nRegion: eu-west\n')])

    source = '''{
  "pools": [
    {"name": "ingest", "workers": 4, "policy": {"attempts": 3, "delay_ms": 100, "overflow": "park"}},
    {"name": "transform", "workers": 6, "policy": {"attempts": 3, "delay_ms": 100, "overflow": "reject"}},
    {"name": "export", "workers": 2, "policy": {"attempts": 3, "delay_ms": 100, "overflow": "spill"}}
  ],
  "shutdown": {"drain": true, "timeout_seconds": 45},
  "accounting": {"labels": ["tenant", "pool"], "interval_seconds": 20}
}
'''
    add("worker-pools", "ambiguity", "json", source,
        [("Change only the transform pool workers from 6 to 9.", '"transform", "workers": 6', '"transform", "workers": 9'),
         ("Change only the ingest pool delay_ms from 100 to 180.",
          '"ingest", "workers": 4, "policy": {"attempts": 3, "delay_ms": 100',
          '"ingest", "workers": 4, "policy": {"attempts": 3, "delay_ms": 180')],
        distractors=(source.replace('"overflow": "reject"', '"overflow": "park"').replace('"workers": 6', '"workers": 5'),))
    source = '''def read_primary(client, key, timeout=10):
    response = client.get("/primary/" + key, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return {"source": "primary", "value": data["value"], "revision": data["revision"]}


def read_replica(client, key, timeout=10):
    response = client.get("/replica/" + key, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return {"source": "replica", "value": data["value"], "lag_ms": data["lag_ms"]}


def read_archive(client, key, timeout=10):
    response = client.get("/archive/" + key, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return {"source": "archive", "value": data["value"], "checksum": data["checksum"]}
'''
    add("tiered-reader", "ambiguity", "python", source,
        [("Change only read_replica's default timeout from 10 to 18.", 'read_replica(client, key, timeout=10)', 'read_replica(client, key, timeout=18)'),
         ("Change only read_archive's default timeout from 10 to 25.", 'read_archive(client, key, timeout=10)', 'read_archive(client, key, timeout=25)')])
    source = '''routes:
  - name: bronze
    match: /bronze
    backend: bronze-v2
    timeout_ms: 900
    retry:
      attempts: 2
      backoff_ms: 80
    headers:
      x-tier: bronze
  - name: silver
    match: /silver
    backend: silver-v3
    timeout_ms: 900
    retry:
      attempts: 2
      backoff_ms: 80
    headers:
      x-tier: silver
  - name: gold
    match: /gold
    backend: gold-v4
    timeout_ms: 900
    retry:
      attempts: 2
      backoff_ms: 80
    headers:
      x-tier: gold
'''
    add("gateway-routes", "ambiguity", "yaml", source,
        [("Change only the silver route timeout_ms from 900 to 1200.", 'backend: silver-v3\n    timeout_ms: 900', 'backend: silver-v3\n    timeout_ms: 1200'),
         ("Change only the gold route backend from gold-v4 to gold-v5.", 'backend: gold-v4', 'backend: gold-v5')],
        distractors=(source.replace('silver-v3', 'silver-v1').replace('gold-v4', 'gold-v2'),))
    source = '''# Cache operations

## Read cache
Owner: serving
TTL: 60 seconds
Before changing capacity, record the current hit rate and memory usage.
After changing capacity, inspect evictions and confirm that latency recovers.
Escalation: serving-oncall

## Write cache
Owner: ingestion
TTL: 60 seconds
Before changing capacity, record the current hit rate and memory usage.
After changing capacity, inspect retries and confirm that queue depth recovers.
Escalation: ingestion-oncall

## Metadata cache
Owner: catalog
TTL: 60 seconds
Before changing capacity, record the current hit rate and memory usage.
After changing capacity, inspect refresh errors and confirm that freshness recovers.
Escalation: catalog-oncall
'''
    add("cache-operations", "ambiguity", "markdown", source,
        [("Change only the Write cache TTL from 60 to 90 seconds.", 'Owner: ingestion\nTTL: 60 seconds', 'Owner: ingestion\nTTL: 90 seconds'),
         ("Change only the Metadata cache owner from catalog to metadata-platform.", 'Owner: catalog', 'Owner: metadata-platform')])

    source = '{"region":"north","limits":{"read":80,"write":25},"retry":{"attempts":3,"delay_ms":150},"audit":{"enabled":true,"fields":["actor","action","resource","timestamp"]}}\n'
    pretty = json.dumps(json.loads(source), indent=2) + "\n"
    # Whole-source replacements define a deterministic formatting contract only.
    add("format-conversion", "control", "json-format", source,
        [("Reformat this JSON with exactly two spaces of indentation, preserving key order and values, and a final newline.", source, pretty),
         ("Reformat this JSON with exactly four spaces of indentation, preserving key order and values, and a final newline.", source, json.dumps(json.loads(source), indent=4) + "\n")])
    source = '''[storage]
engine = "local"
directory = "/var/lib/ledger"
sync_on_commit = true
checkpoint_records = 500

[rotation]
max_segment_mb = 64
retain_segments = 8
compress_closed_segments = true

[verification]
checksum = "sha256"
verify_on_open = true
repair_partial_tail = false
'''
    add("empty-history", "control", "toml", source,
        [("Change only checkpoint_records from 500 to 750.", 'checkpoint_records = 500', 'checkpoint_records = 750'),
         ("Change only retain_segments from 8 to 12.", 'retain_segments = 8', 'retain_segments = 12')], history=False)
    source = '''# Search index maintenance

Start by checking the replication delay for every index shard.
Freeze scheduled rebuilds before taking a consistent snapshot.
Copy the snapshot manifest to the verification directory.
Validate the document count against the last completed checkpoint.
Resume rebuilds only after every shard reports a healthy status.
Retain the previous snapshot until the next successful verification.
'''
    unrelated = 'Recipe: steep green tea for two minutes, then remove the leaves.\n' * 3
    add("unrelated-history", "control", "markdown", source,
        [("Change only 'verification directory' to 'audit directory'.", 'verification directory', 'audit directory'),
         ("Change only 'next successful verification' to 'next weekly audit'.", 'next successful verification', 'next weekly audit')],
        history=False, distractors=(unrelated,))
    source = 'status=ready\nworkers=4\nregion=east\n'
    add("short-settings", "control", "text", source,
        [("Change only workers from 4 to 6.", 'workers=4', 'workers=6'),
         ("Change only region from east to west.", 'region=east', 'region=west')])
    return {"schema": 1, "origin": "new authored controlled fixtures; not natural production traffic",
            "history_policy": "Both requests independently edit the supplied v1. No current output is written to history.",
            "projects": projects}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(authored_projects(), ensure_ascii=False, indent=2) + "\n")
    print("Wrote 12 authored projects / 24 requests; no model was loaded.")


if __name__ == "__main__":
    main()
