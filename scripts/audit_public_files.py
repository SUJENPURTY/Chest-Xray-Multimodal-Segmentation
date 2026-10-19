"""Inventory the Git candidate set without staging, committing, or root git init.

Creates private reports and isolated empty Git metadata under outputs/. This is
a publication-risk screen, not an exhaustive secrets/medical de-identification
audit. Suspected secret values are never included in reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*arguments):
    return subprocess.check_output(["git", *map(str, arguments)], cwd=ROOT)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New directory below outputs/")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = (args.output_dir or ROOT / "outputs/public_file_audit" / stamp).resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()) or output.exists():
        parser.error("Choose a new directory below outputs/; existing reports are never overwritten.")
    output.mkdir(parents=True)
    metadata = output / "candidate.git"
    git("init", "--bare", "--quiet", "--template=", metadata)
    excludes = output / "empty_global_excludes"
    excludes.write_text("", encoding="utf-8")
    raw = git("--git-dir=" + str(metadata), "--work-tree=" + str(ROOT),
              "-c", "core.excludesFile=" + str(excludes),
              "ls-files", "--others", "--exclude-standard", "-z")
    candidates = sorted(path.decode("utf-8") for path in raw.split(b"\0") if path)
    patterns = {
        "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        "github_fine_grained_token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
        "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{25,}\b"),
        "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    }
    home_path = re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/(?:home|Users)/)[^\\/\s\"']+")
    machine_path = re.compile(r"(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/]|/(?:home|Users|kaggle|content)/")
    flags, manifest = [], []
    for relative in candidates:
        path = ROOT / relative
        size = path.stat().st_size
        manifest.append({"path": relative, "bytes": size, "sha256": sha256(path)})
        if size >= 100 * 1024**2:
            flags.append({"path": relative, "kind": "oversized_at_least_100_MiB"})
        if path.suffix.lower() in {".ckpt", ".pth", ".pt", ".safetensors", ".bin", ".pem", ".key"}:
            flags.append({"path": relative, "kind": "restricted_or_weight_file"})
        try:
            content = path.read_text(encoding="utf-8") if size <= 20 * 1024**2 else None
        except (UnicodeError, OSError):
            content = None
        if content is None:
            flags.append({"path": relative, "kind": "binary_or_large_content_not_text_scanned"})
            continue
        for kind, pattern in patterns.items():
            count = len(pattern.findall(content))
            if count:
                flags.append({"path": relative, "kind": kind, "count": count})
        for kind, pattern in (("personal_home_path_review", home_path),
                              ("machine_path_portability_review", machine_path)):
            lines = [index for index, line in enumerate(content.splitlines(), 1) if pattern.search(line)]
            if lines:
                flags.append({"path": relative, "kind": kind, "lines": lines})
        if path.suffix == ".ipynb":
            notebook = json.loads(content)
            images = sum(any(mime.startswith("image/") for mime in item.get("data", {}))
                         for cell in notebook["cells"] for item in cell.get("outputs", []))
            attachments = sum(bool(cell.get("attachments")) for cell in notebook["cells"])
            if images or attachments:
                flags.append({"path": relative, "kind": "embedded_images_rights_and_privacy_review",
                              "image_outputs": images, "cells_with_attachments": attachments})
    root_git_exists = (ROOT / ".git").exists()
    tracked, tracked_ignored = [], []
    if root_git_exists:
        tracked = [p.decode("utf-8") for p in git("ls-files", "--cached", "-z").split(b"\0") if p]
        tracked_ignored = [p.decode("utf-8") for p in git(
            "ls-files", "--cached", "--ignored", "--exclude-standard", "-z").split(b"\0") if p]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Candidate files using current repository ignore rules and empty global excludes; no staging",
        "candidate_count": len(manifest), "candidate_bytes": sum(row["bytes"] for row in manifest),
        "root_git_exists": root_git_exists, "actual_tracked_files": tracked,
        "actual_tracked_but_ignored": tracked_ignored, "flags": flags,
        "limitations": ["Pattern-based text scan only; secret absence is not certified",
                        "Embedded image pixels and binary metadata need human review",
                        "No root Git history is scanned", "Manifest describes this invocation only"],
    }
    (output / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "candidate_files.txt").write_text("\n".join(candidates) + "\n", encoding="utf-8")
    (output / "publication_review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("candidate_count", "candidate_bytes", "root_git_exists")}))
    print(f"Review flags: {len(flags)}; private reports: {output}")


if __name__ == "__main__":
    main()
