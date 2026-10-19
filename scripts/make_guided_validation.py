"""Author eight new controlled projects before inspecting any Target answers."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from memorydraft.memory.store import MemoryEvent
from memorydraft.runtime.contracts import GenerationRequest
from scripts.make_transport_validation import SYSTEM


def authored_projects():
    projects = []

    def add(name, group, source, edits, history=None):
        user = "ot-guided-" + name
        events = [] if history is False else [asdict(MemoryEvent(user, "source-v1", 2,
                    source if history is None else history, project_id=name))]
        requests = []
        for turn, (instruction, old, new) in enumerate(edits):
            if source.count(old) != 1:
                raise ValueError("Each edit must identify one supplied source span")
            request = GenerationRequest(f"{name}:{turn}", user, 10 + turn,
                ({"role": "system", "content": SYSTEM}, {"role": "user", "content":
                 instruction + "\n\nSOURCE BEGIN\n" + source + "SOURCE END\n"}), 512)
            requests.append(dict(request=asdict(request), edit=dict(old=old, new=new),
                                 expected_text=source.replace(old, new, 1)))
        projects.append(dict(id=name, group=group, source=source, history=events, requests=requests))

    source = '''[proxy]
name = "edge-router"
mode = "balanced"
connect_timeout_ms = 120
upstream_group = "public-api"

[health]
path = "/ready"
interval_seconds = 15
healthy_threshold = 3
unhealthy_threshold = 2

[limits]
max_connections = 2048
max_pending_requests = 256
idle_timeout_seconds = 90

[access_log]
enabled = true
format = "request_id method path status duration_ms"
include_headers = false
'''
    add("edge-router", "edited-continuation", source, [
        ("Change only connect_timeout_ms from 120 to 175.", "connect_timeout_ms = 120", "connect_timeout_ms = 175"),
        ('Insert zone = "eu-central" immediately after the mode line.', 'mode = "balanced"\n', 'mode = "balanced"\nzone = "eu-central"\n')])

    source = '''WITH parameters AS (
    SELECT 30 AS freshness_days,
           10 AS minimum_available
), eligible_items AS (
    SELECT item_id, warehouse_id, available_units, updated_at
    FROM inventory_snapshots
    WHERE archived = false
      AND available_units > 0
), ranked_items AS (
    SELECT item_id, warehouse_id, available_units,
           ROW_NUMBER() OVER (
               PARTITION BY item_id ORDER BY updated_at DESC
           ) AS version_rank
    FROM eligible_items
)
SELECT item_id, warehouse_id, available_units
FROM ranked_items
WHERE version_rank = 1
ORDER BY item_id, warehouse_id;
'''
    add("inventory-query", "edited-continuation", source, [
        ("Change only freshness_days from 30 to 45.", "30 AS freshness_days", "45 AS freshness_days"),
        ("Insert the comment '-- Nightly inventory view' immediately before WITH parameters.", "WITH parameters AS (", "-- Nightly inventory view\nWITH parameters AS (")])

    source = '''def image_worker_options():
    return {
        "name": "thumbnail-generator",
        "workers": 6,
        "input": {
            "queue": "images.pending",
            "prefetch": 24,
            "acknowledge_after_write": True,
        },
        "transform": {
            "width": 320,
            "height": 240,
            "preserve_aspect_ratio": True,
            "format": "webp",
        },
        "output": {
            "bucket": "image-previews",
            "prefix": "generated/",
            "cache_control": "public,max-age=3600",
        },
    }
'''
    add("image-worker", "edited-continuation", source, [
        ("Change only workers from 6 to 9.", '"workers": 6', '"workers": 9'),
        ('Insert the dictionary entry "priority": "interactive", immediately after name.', '"name": "thumbnail-generator",\n', '"name": "thumbnail-generator",\n        "priority": "interactive",\n')])

    source = '''{
  "title": "ShipmentRecorded",
  "version": 2,
  "type": "object",
  "required": ["shipment_id", "occurred_at", "items"],
  "properties": {
    "shipment_id": {"type": "string", "minLength": 8},
    "occurred_at": {"type": "string", "format": "date-time"},
    "items": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["sku", "quantity"],
        "properties": {
          "sku": {"type": "string"},
          "quantity": {"type": "integer", "minimum": 1}
        }
      }
    }
  },
  "additionalProperties": false
}
'''
    add("shipment-contract", "edited-continuation", source, [
        ("Change only the root version from 2 to 3.", '"version": 2', '"version": 3'),
        ('Insert the root field "namespace": "logistics", immediately after title.', '"title": "ShipmentRecorded",\n', '"title": "ShipmentRecorded",\n  "namespace": "logistics",\n')])

    source = "host=resolver\nport=53\nprotocol=udp\n"
    add("resolver-short", "control", source, [
        ("Change only port from 53 to 54.", "port=53", "port=54"),
        ("Change only protocol from udp to tcp.", "protocol=udp", "protocol=tcp")])
    source = '''# Recovery checklist

Confirm that the replica is reachable before changing the routing table.
Record the current checkpoint identifier in the incident notebook.
Pause the scheduled compaction job and wait for active writes to finish.
Mount the snapshot in a separate read-only verification directory.
Compare object counts and checksums with the saved manifest.
Restore traffic gradually while monitoring the replication delay.
Keep the previous snapshot until the next completed checkpoint.
'''
    add("recovery-empty", "control", source, [
        ("Change only 'incident notebook' to 'operations notebook'.", "incident notebook", "operations notebook"),
        ("Change only 'replication delay' to 'replication lag'.", "replication delay", "replication lag")], history=False)
    source = '''[backup]
destination = "/srv/archives"
compress = true
encryption = "aes256"
parallel_streams = 3

[schedule]
timezone = "UTC"
hour = 2
minute = 15

[retention]
daily = 7
weekly = 4
monthly = 6
'''
    add("backup-unrelated", "control", source, [
        ("Change only parallel_streams from 3 to 5.", "parallel_streams = 3", "parallel_streams = 5"),
        ("Change only monthly retention from 6 to 9.", "monthly = 6", "monthly = 9")],
        history="Garden note: water the seedlings at sunrise and check the soil before sunset.\n" * 4)
    source = "SET statement_timeout = 800;\nSET lock_timeout = 200;\n"
    add("database-tail", "control", source, [
        ("Change only statement_timeout from 800 to 950.", "statement_timeout = 800", "statement_timeout = 950"),
        ("Change only lock_timeout from 200 to 350.", "lock_timeout = 200", "lock_timeout = 350")])
    return dict(schema=1, origin="Eight new authored controlled projects; not production traffic", projects=projects)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(authored_projects(), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
