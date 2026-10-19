"""Resolve a complete local pretrained model or its official Hugging Face ID."""
import os
from pathlib import Path

from src.config import CXR_BERT_ID


def resolve_model(spec, official_id):
    def complete(folder):
        weights = any((folder / name).is_file() for name in ("pytorch_model.bin", "model.safetensors"))
        if official_id == CXR_BERT_ID:
            return weights and all((folder / name).is_file() for name in (
                "config.json", "tokenizer_config.json", "vocab.txt",
                "modeling_cxrbert.py", "configuration_cxrbert.py"))
        return weights and (folder / "config.json").is_file()

    path = Path(spec).expanduser()
    if path.is_dir():
        if not complete(path):
            raise FileNotFoundError(f"Incomplete pretrained model directory: {path}")
        return str(path.resolve())
    if spec != official_id:
        raise ValueError(f"Use a complete local directory or official model ID {official_id}")
    caches = [Path(os.environ[name]).expanduser() for name in
              ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE") if os.environ.get(name)]
    if os.environ.get("HF_HOME"):
        caches.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    caches.append(Path.home() / ".cache" / "huggingface" / "hub")
    key = "models--" + official_id.replace("/", "--")
    for cache in caches:
        snapshots = cache / key / "snapshots"
        if snapshots.is_dir():
            for snapshot in sorted(snapshots.iterdir(), reverse=True):
                if complete(snapshot):
                    return str(snapshot)
    return official_id
