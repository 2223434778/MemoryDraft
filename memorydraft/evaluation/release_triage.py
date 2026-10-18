"""Frozen release-impact classification and post-hoc exact reuse opportunity."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path


AREAS = ("python_runtime", "dependency", "api_migration", "security_fix", "other_behavior")
CRITICAL = AREAS[0], AREAS[1], AREAS[3]
SYSTEM = "You classify the impact of software release notes. Follow the supplied definitions and return JSON only."
DEFINITIONS = """Classify only this release, not earlier releases or your prior knowledge.
python_runtime: explicitly added/dropped Python or PyPy version support or a changed Python version requirement. A feature available on one Python version, or a bug on a special Python build, is not a support-policy change.
dependency: explicitly added/removed dependencies, changed minimum dependency versions, or changed build backend. Delaying imports and merely changing packaging metadata do not qualify.
api_migration: removed, deprecated, renamed or moved public APIs, or changes to an existing call contract that require callers to adapt. Pure type-annotation fixes/widening and new optional capabilities do not qualify. Only actual deprecations/removals qualify, not use of the word deprecated when describing a newly supported feature.
security_fix: an explicitly identified security vulnerability fix, including CVE/GHSA. General TLS correctness fixes, FIPS compatibility, authentication correctness, and new security-related capabilities do not qualify without an identified security vulnerability.
other_behavior: other observable functional additions or fixes affecting users. Exclude changes wholly covered by the preceding four areas, documentation, internal refactors and pure type-annotation changes. New optional functionality belongs here, even if it introduces an API.\n"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["files"]) != {"projects.json", "labels.json", *(
            "licenses/" + name + ".txt" for name in ("psf--requests", "encode--httpx", "pallets--click",
                                                    "pallets--flask", "Kludex--starlette", "Textualize--rich"))}:
        raise ValueError("Unexpected frozen files")
    for name, expected in manifest["files"].items():
        if sha((directory / name).read_bytes()) != expected:
            raise ValueError("Frozen data changed: " + name)
    projects = json.loads((directory / "projects.json").read_text(encoding="utf-8"))["projects"]
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))["labels"]
    identities = {p["id"] + ":" + r["tag"] for p in projects for r in p["releases"]}
    if len(projects) != 6 or any(len(p["releases"]) != 3 for p in projects) or len(identities) != 18 or set(labels) != identities:
        raise ValueError("Expected exactly the predeclared 18 releases and labels")
    for project in projects:
        if sha((directory / project["license"]["path"]).read_bytes()) != project["license"]["sha256"]:
            raise ValueError("License mismatch")
        for release in project["releases"]:
            if sha(release["notes"].encode()) != release["notes_sha256"]:
                raise ValueError("Source excerpt mismatch")
            gold = labels[project["id"] + ":" + release["tag"]]
            if not set(gold) <= set(AREAS):
                raise ValueError("Unknown label area")
            for item in gold.values():
                valid = {f"L{i:03d}" for i, line in enumerate(release["notes"].splitlines(), 1) if line.strip()}
                if not item["rationale"] or not item["lines"] or not set(item["lines"]) <= valid:
                    raise ValueError("Invalid pre-run evidence label")
    return projects, labels, manifest


def request_text(project, turn):
    release = project["releases"][turn]
    numbered = "\n".join(f"[L{i:03d}] {line}" for i, line in enumerate(release["notes"].splitlines(), 1))
    return ("Create an upgrade-impact triage checklist for this project's release. This is not a complete migration plan.\n"
            + DEFINITIONS + "\nReturn exactly one JSON object with keys version and checks. version must equal the requested version.\n"
            "checks must contain exactly five objects in this order: " + ", ".join(AREAS) + ".\n"
            "Each object has exactly the keys area, status, evidence. status is review if this release contains any qualifying change, "
            "otherwise not_mentioned. For review, evidence is an array of one or two supporting line IDs from the current notes; "
            "a line anywhere within a supporting change's bullet is allowed. For not_mentioned, evidence is []. "
            "Do not add explanations, markdown, unknown facts, or recommendations. not_mentioned never means safe to upgrade.\n"
            f"Repository: {project['repository']}\nRequested version: {release['version']}\nRelease notes:\n{numbered}\n")


def messages(text):
    return [dict(role="system", content=SYSTEM), dict(role="user", content=text)]


def answer_event(project, turn, text):
    from memorydraft.memory.store import MemoryEvent

    return MemoryEvent(project["id"], "release:" + project["releases"][turn]["tag"], 11 + 10 * turn,
                       text, project_id=project["id"])


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def quality(text, release, gold):
    result = dict(schema=False, identity=False, correct=False, areas={area: False for area in AREAS},
                  critical_misses=[area for area in CRITICAL if area in gold], error=None)
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
        if not isinstance(value, dict) or set(value) != {"version", "checks"} or not isinstance(value["version"], str):
            raise ValueError("Expected exactly version and checks")
        checks = value["checks"]
        if not isinstance(checks, list) or len(checks) != len(AREAS):
            raise ValueError("Expected five checks")
        valid_lines = {f"L{i:03d}" for i, line in enumerate(release["notes"].splitlines(), 1) if line.strip()}
        for area, check in zip(AREAS, checks):
            if (not isinstance(check, dict) or set(check) != {"area", "status", "evidence"}
                    or check["area"] != area or check["status"] not in {"review", "not_mentioned"}
                    or not isinstance(check["evidence"], list)
                    or any(not isinstance(x, str) for x in check["evidence"])
                    or len(set(check["evidence"])) != len(check["evidence"])
                    or not set(check["evidence"]) <= valid_lines
                    or (check["status"] == "review" and not 1 <= len(check["evidence"]) <= 2)
                    or (check["status"] == "not_mentioned" and check["evidence"])):
                raise ValueError("Invalid area/status/evidence contract")
        result["schema"] = True
        result["identity"] = value["version"] == release["version"]
        for area, check in zip(AREAS, checks):
            expected = "review" if area in gold else "not_mentioned"
            result["areas"][area] = (check["status"] == expected and
                (area not in gold or set(check["evidence"]) <= set(gold[area]["lines"])))
        # A false positive is a classification error; missing a critical positive is also a stop condition.
        result["critical_misses"] = [area for area in CRITICAL if area in gold and not result["areas"][area]]
        result["correct"] = result["identity"] and all(result["areas"].values())
    except (ValueError, TypeError, KeyError) as error:
        result["error"] = str(error)
    return result


def quality_gate(rows):
    correct = sum(row["quality"]["correct"] for row in rows)
    categories = sum(sum(row["quality"]["areas"].values()) for row in rows)
    schema = sum(row["quality"]["schema"] and row["quality"]["identity"] for row in rows)
    misses = [(row["project_id"], row["turn"], area) for row in rows for area in row["quality"]["critical_misses"]]
    truncations = sum(row["budget_hit"] or not row["eos"] for row in rows)
    return dict(passed=len(rows) == 18 and schema == 18 and correct >= 17 and categories >= 86 and not misses and not truncations,
                rows=len(rows), schema_and_identity=schema, correct_requests=correct, correct_categories=categories,
                critical_misses=misses, incomplete_outputs=truncations)


def continuation_map(texts, tokenizer, anchor=6, limit=32):
    candidates = defaultdict(set)
    special = set(tokenizer.all_special_ids)
    for text in texts:
        segment = []
        for token in [*tokenizer.encode(text, add_special_tokens=False), None]:
            if token is None or token in special:
                for position in range(anchor, len(segment)):
                    candidates[tuple(segment[position-anchor:position])].add(tuple(segment[position:position+limit]))
                segment = []
            else:
                segment.append(token)
    return candidates


def oracle_coverage(tokens, candidates, anchor=6, minimum=8):
    covered = set()
    lengths = []
    for position in range(anchor, len(tokens)):
        best = 0
        for candidate in candidates.get(tuple(tokens[position-anchor:position]), ()):
            accepted = 0
            for actual, proposed in zip(tokens[position:], candidate):
                if actual != proposed:
                    break
                accepted += 1
            best = max(best, accepted)
        lengths.append(best)
        if best >= minimum:
            covered.update(range(position, position + best))
    return covered, lengths


def opportunity(tokens, prompt, past_answers, tokenizer):
    context, context_lengths = oracle_coverage(tokens, continuation_map([prompt], tokenizer))
    history, history_lengths = oracle_coverage(tokens, continuation_map(past_answers, tokenizer))
    extra = history - context
    return dict(tokens=len(tokens), context_covered=len(context), history_covered=len(history),
                global_covered=len(context | history), history_only_covered=len(extra),
                history_only_fraction=len(extra) / max(1, len(tokens)),
                context_max=max(context_lengths, default=0), history_max=max(history_lengths, default=0),
                oracle=True, measured_speedup=False)


def opportunity_gate(rows):
    warm = [row for row in rows if row["turn"] > 0]
    total = sum(row["opportunity"]["tokens"] for row in warm)
    extra = sum(row["opportunity"]["history_only_covered"] for row in warm)
    project_ids = {row["project_id"] for row in warm}
    per_project = {identity: sum(r["opportunity"]["history_only_covered"] for r in warm if r["project_id"] == identity)
                   / max(1, sum(r["opportunity"]["tokens"] for r in warm if r["project_id"] == identity))
                   for identity in sorted(project_ids)}
    qualifying = sum(value >= .05 for value in per_project.values())
    return dict(passed=len(warm) == 12 and len(project_ids) == 6 and extra >= .1 * total and qualifying >= 4,
                warm_requests=len(warm), output_tokens=total, history_only_covered=extra,
                history_only_fraction=extra / max(1, total), per_project=per_project, qualifying_projects=qualifying,
                oracle=True, measured_speedup=False)
