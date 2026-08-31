#!/usr/bin/env python3
"""Rebuild RC10 metadata and the SHA-256 manifest from the release tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXCLUDED_DIRS = {".git", "replay_output", "tmp", "test_scratch", "__pycache__"}
EXCLUDED_FILES = {"MANIFEST.sha256", "artifact_metadata.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if relative.as_posix() in EXCLUDED_FILES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    if (root / "paper").exists():
        raise SystemExit("Refusing to build a code release containing paper/")
    leaked_logs = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.log"))
    if leaked_logs:
        raise SystemExit(f"Refusing to build a release containing log files: {leaked_logs[:5]}")

    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    files = release_files(root)
    records = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest = sha256(path)
        records.append(
            {
                "path": relative,
                "source_path": relative,
                "source_sha256": digest,
                "artifact_sha256": digest,
                "bytes": path.stat().st_size,
                "sanitized": False,
            }
        )

    metadata = {
        "schema_version": 2,
        "artifact_version": version,
        "profile": "public-code-core",
        "source_revision": "artifact-v2026.08.31-rc9",
        "authors": "Anonymous Author(s)",
        "build_policy": "code_release_without_paper_logs_or_local_paths",
        "repository": "https://github.com/zxu700708-hub/zxu700708-hub-bkan-photodetector-compact-modeling",
        "commercial_dependencies": ["commercial TCAD solver", "Cadence Spectre 18.1"],
        "records": records,
        "payload_file_count": len(records),
        "payload_bytes_excluding_metadata_and_manifest": sum(item["bytes"] for item in records),
        "manifest_file_count": len(records) + 1,
    }
    metadata_path = root / "artifact_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest_files = release_files(root) + [metadata_path]
    manifest_files.sort(key=lambda item: item.relative_to(root).as_posix())
    entries = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in manifest_files
    ]
    (root / "MANIFEST.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed",
                "version": version,
                "manifest_files": len(entries),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
