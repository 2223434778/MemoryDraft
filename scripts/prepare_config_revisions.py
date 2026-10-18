"""Explicit, cached acquisition of public CI revisions; never run during inference."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2025-09-01T00:00:00Z"
REPOSITORIES = {
    "development": ["astral-sh/ruff", "fastapi/fastapi", "encode/httpx", "pallets/flask",
                    "psf/requests", "Textualize/rich"],
    "confirmation": ["pydantic/pydantic", "python/mypy", "pytest-dev/pytest", "psf/black",
                     "python-poetry/poetry", "pre-commit/pre-commit", "pallets/click",
                     "encode/starlette", "numpy/numpy", "pandas-dev/pandas", "scipy/scipy",
                     "scikit-learn/scikit-learn"],
}
LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC"}
REVIEWED_LICENSES = {
    "python/mypy": ("MIT", "Repository LICENSE explicitly licenses mypy/mypyc as MIT; PSF exceptions name three unrelated mypyc runtime files. Full combined notice retained."),
    "numpy/numpy": ("BSD-3-Clause", "Read LICENSE.txt at a1098a8279fb960f53c8f3cfdaa3b0fba5203a6b: copyright NumPy Developers and the three BSD redistribution/endorsement conditions. Full notice retained."),
}
API_ALIASES = {"encode/starlette": "Kludex/starlette"}  # Verified unchanged repository ID 138597372.


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PublicCacheMiss(Exception):
    pass


class PublicClient:
    def __init__(self, cache: Path, offline_api=False, offline=False):
        self.cache = cache
        self.offline_api = offline_api
        self.offline = offline
        cache.mkdir(parents=True, exist_ok=True)

    def get(self, url: str) -> bytes:
        path = self.cache / sha(url.encode())
        if not path.exists():
            if self.offline or (self.offline_api and url.startswith("https://api.github.com/")):
                print("PUBLIC_CACHE_MISS:" + json.dumps(dict(url=url, path=path.as_posix())), flush=True)
                raise PublicCacheMiss()
            request = Request(url, headers={"User-Agent": "MemoryDraft-public-research",
                                            "Accept": "application/vnd.github+json"})
            with urlopen(request, timeout=20) as response:
                path.write_bytes(response.read())
        data = path.read_bytes()
        if data.startswith(b'{"memorydraft_public_response":'):
            return json.loads(data)["content"].encode("utf-8")
        return data

    def api(self, repository: str, endpoint: str, **query):
        repository = API_ALIASES.get(repository, repository)
        url = f"https://api.github.com/repos/{repository}/{endpoint}".rstrip("/")
        if query:
            url += "?" + urlencode(query)
        return json.loads(self.get(url))

    def file(self, repository: str, revision: str, path: str) -> bytes:
        repository = API_ALIASES.get(repository, repository)
        return self.get(f"https://raw.githubusercontent.com/{repository}/{revision}/{quote(path)}")


def eligible(text: str) -> bool:
    return 300 <= len(text.encode()) <= 6000 and 12 <= len(text.splitlines()) <= 180


def collect(client: PublicClient, repository: str, split: str, output: Path):
    candidates = client.api(repository, "commits", path=".github/workflows", until=CUTOFF, per_page=12)
    rejected = []
    checked = set()
    for candidate in candidates:
        detail = client.api(repository, "commits/" + candidate["sha"])
        for file in sorted(detail["files"], key=lambda value: value["filename"]):
            name = file["filename"]
            if (not name.startswith(".github/workflows/") or not name.endswith((".yml", ".yaml"))
                    or file["status"] != "modified" or name in checked):
                continue
            checked.add(name)
            history = client.api(repository, "commits", path=name, until=CUTOFF, per_page=3)
            if len(history) != 3:
                rejected.append(dict(path=name, reason="fewer than three file revisions"))
                continue
            revisions = []
            try:
                for commit in reversed(history):
                    data = client.file(repository, commit["sha"], name)
                    text = data.decode("utf-8")
                    revisions.append(dict(commit=commit["sha"], date=commit["commit"]["committer"]["date"],
                                          url=f"https://github.com/{repository}/blob/{commit['sha']}/{name}",
                                          title=commit["commit"]["message"].splitlines()[0],
                                          sha256=sha(data), text=text))
            except UnicodeDecodeError:
                rejected.append(dict(path=name, reason="non-UTF8"))
                continue
            if not all(eligible(value["text"]) for value in revisions):
                rejected.append(dict(path=name, reason="outside fixed byte/line bounds"))
                continue
            if any(left["text"] == right["text"] for left, right in zip(revisions, revisions[1:])):
                rejected.append(dict(path=name, reason="no content change"))
                continue
            latest = revisions[-1]["commit"]
            metadata = client.api(repository, "")
            reported = metadata["license"]["spdx_id"]
            spdx, review = REVIEWED_LICENSES.get(repository, (reported, "GitHub SPDX metadata and retained source notice"))
            if spdx not in LICENSES:
                raise ValueError(f"Review license before redistributing {repository}: {spdx}, revision={latest}, file={name}")
            root_files = client.api(repository, "contents", ref=latest)
            license_files = sorted((item for item in root_files if item["type"] == "file"
                                    and item["name"].lower() in {"license", "license.txt", "license.md", "copying"}),
                                   key=lambda item: item["name"])
            if not license_files:
                raise ValueError("Review nonstandard license path in " + repository)
            license_data = client.file(repository, latest, license_files[0]["name"])
            license_name = repository.replace("/", "--") + ".txt"
            (output / "licenses").mkdir(exist_ok=True)
            (output / "licenses" / license_name).write_bytes(license_data)
            currents = []
            for before, after in zip(revisions, revisions[1:]):
                patch = "".join(difflib.unified_diff(before["text"].splitlines(keepends=True),
                                                   after["text"].splitlines(keepends=True),
                                                   fromfile="a/" + name, tofile="b/" + name))
                currents.append(dict(before_commit=before["commit"], after_commit=after["commit"],
                                     title=after["title"], before=before["text"], expected=after["text"], patch=patch))
            return dict(id=repository.replace("/", "--"), repository=repository,
                        canonical_repository=API_ALIASES.get(repository, repository), split=split, path=name,
                        revisions=revisions, turns=currents, license=dict(spdx=spdx, reported_spdx=reported, review=review, revision=latest,
                        path="licenses/" + license_name, sha256=sha(license_data),
                        source_url=license_files[0]["html_url"]), rejected_candidates=rejected)
    raise ValueError(f"No eligible consecutive workflow revisions in {repository}; preserve audit and review")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/config-revisions")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/raw/config-revisions")
    parser.add_argument("--offline-api", action="store_true", help="Use public API responses populated by the connected GitHub reader")
    parser.add_argument("--offline", action="store_true", help="Require URL-addressed cached responses for all resources")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "projects.json").exists():
        raise RuntimeError("Frozen projects already exist; do not silently recollect")
    client = PublicClient(args.cache, args.offline_api, args.offline)
    projects = []
    # A partial collection is resumable solely through URL-addressed public response caches.
    for split, repositories in REPOSITORIES.items():
        for repository in repositories:
            project = collect(client, repository, split, args.output)
            projects.append(project)
            print(split, repository, project["path"], [len(r["text"]) for r in project["revisions"]], flush=True)
            (args.output / "collection-partial.json").write_text(json.dumps(projects, indent=2) + "\n", encoding="utf-8")
    payload = dict(cutoff=CUTOFF, selection="Latest eligible modified workflow, lexicographic path; latest three file revisions",
                   byte_bounds=[300, 6000], line_bounds=[12, 180], repositories=REPOSITORIES, projects=projects)
    (args.output / "projects.json").write_bytes((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode())
    files = [args.output / "projects.json", *sorted((args.output / "licenses").glob("*.txt"))]
    manifest = {path.relative_to(args.output).as_posix(): sha(path.read_bytes()) for path in files}
    (args.output / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    (args.output / "collection-partial.json").unlink()
    print("Frozen projects SHA256", sha((args.output / "projects.json").read_bytes()), flush=True)


if __name__ == "__main__":
    try:
        main()
    except PublicCacheMiss:
        sys.exit(2)
