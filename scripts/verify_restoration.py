"""Verify retained restoration files without loading models or downloading data.

Checksums identify the observed local research bundle, not redistribution rights.
Dataset pixels and patient separation require separate authorized verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_snapshot(model_id):
    caches = [Path(os.environ[key]).expanduser() for key in
              ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE") if os.environ.get(key)]
    if os.environ.get("HF_HOME"):
        caches.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    caches.append(Path.home() / ".cache/huggingface/hub")
    for cache in caches:
        snapshots = cache / ("models--" + model_id.replace("/", "--")) / "snapshots"
        if snapshots.is_dir():
            for folder in sorted(snapshots.iterdir(), reverse=True):
                if (folder / "config.json").is_file() and any(
                        (folder / name).is_file() for name in ("pytorch_model.bin", "model.safetensors")):
                    return folder
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=("phase2", "original", "all"), default="phase2")
    parser.add_argument("--convnext-dir", type=Path, help="Complete trusted ConvNeXt snapshot")
    parser.add_argument("--cxrbert-dir", type=Path, help="Complete trusted CXR-BERT resource folder")
    args = parser.parse_args()
    bundled = ROOT / "Project/LanGuideMedSeg-MICCAI2023/lib/BiomedVLP-CXR-BERT-specialized-official"
    bases = {
        "repository": ROOT,
        "convnext": args.convnext_dir or cached_snapshot("facebook/convnext-tiny-224"),
        "cxrbert": args.cxrbert_dir or (bundled if bundled.is_dir() else
                                      cached_snapshot("microsoft/BiomedVLP-CXR-BERT-specialized")),
    }
    manifest = json.loads((ROOT / "docs/resource_checksums.json").read_text(encoding="utf-8"))
    failures, checked = [], 0
    for record in manifest["files"]:
        if args.workflow != "all" and record["workflow"] not in (args.workflow, "pretrained"):
            continue
        checked += 1
        label = record["base"] + "/" + record["path"]
        base = bases[record["base"]]
        path = base / record["path"] if base else None
        if path is None or not path.is_file():
            failures.append({"resource": label, "reason": "missing"})
        elif path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            failures.append({"resource": label, "reason": "size_or_sha256_mismatch"})
    print(json.dumps({"workflow": args.workflow, "files_checked": checked,
                      "status": "failed" if failures else "passed", "failures": failures,
                      "dataset_verified": False, "model_loaded": False}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
