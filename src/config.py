"""Shared, environment-independent configuration."""
from dataclasses import dataclass, replace
from pathlib import Path
import os
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "Project" / "LanGuideMedSeg-MICCAI2023"
DEFAULT_DATA = (LEGACY / "Data" / "QaTa-COV19-v2") if (LEGACY / "Data" / "QaTa-COV19-v2").is_dir() else ROOT / "data" / "QaTa-COV19-v2"
DEFAULT_OUTPUT = Path("/kaggle/working/xray-output") if Path("/kaggle/working").is_dir() else ROOT / "outputs" / "run"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT / "checkpoints" / "best_dice.ckpt"
CONVNEXT_ID = "facebook/convnext-tiny-224"
CXR_BERT_ID = "microsoft/BiomedVLP-CXR-BERT-specialized"


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    image_size: int = 224
    batch_size: int = 1  # The original run needed batch 1 on limited VRAM.
    epochs: int = 20
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    num_workers: int = 0
    mixed_precision: bool = True
    threshold: float = 0.5
    data_root: Path = DEFAULT_DATA
    output_dir: Path = DEFAULT_OUTPUT
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    train_csv: Path = DEFAULT_OUTPUT / "prepared_annotations" / "train.csv"
    test_csv: Path = DEFAULT_OUTPUT / "prepared_annotations" / "test.csv"
    vision_model: str = CONVNEXT_ID
    text_model: str = CXR_BERT_ID
    max_text_length: int = 24


def from_environment() -> Config:
    cfg = Config()
    if os.environ.get("OUTPUT_DIR"):
        cfg = with_output_dir(cfg, Path(os.environ["OUTPUT_DIR"]).expanduser())
    overrides = {}
    for env, field in (("DATA_ROOT", "data_root"),
                       ("MODEL_CHECKPOINT", "checkpoint_path"),
                       ("TRAIN_CSV", "train_csv"), ("TEST_CSV", "test_csv")):
        if os.environ.get(env):
            overrides[field] = Path(os.environ[env]).expanduser()
    return replace(cfg, **overrides)


def with_output_dir(cfg: Config, output_dir: Path) -> Config:
    """Move generated paths together, preserving explicit path overrides."""
    output_dir = Path(output_dir).expanduser()
    changes = {"output_dir": output_dir}
    if not os.environ.get("MODEL_CHECKPOINT"):
        changes["checkpoint_path"] = output_dir / "checkpoints" / "best_dice.ckpt"
    if not os.environ.get("TRAIN_CSV"):
        changes["train_csv"] = output_dir / "prepared_annotations" / "train.csv"
    if not os.environ.get("TEST_CSV"):
        changes["test_csv"] = output_dir / "prepared_annotations" / "test.csv"
    return replace(cfg, **changes)


def provision_checkpoint(cfg: Config) -> Config:
    """Download an explicitly configured trusted checkpoint once, with SHA256."""
    if cfg.checkpoint_path.is_file():
        return cfg
    url = os.environ.get("MODEL_CHECKPOINT_URL")
    if not url:
        return cfg
    digest = os.environ.get("MODEL_CHECKPOINT_SHA256", "").lower()
    if not url.startswith("https://") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("MODEL_CHECKPOINT_URL requires HTTPS and MODEL_CHECKPOINT_SHA256")
    path = cfg.checkpoint_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        sha = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                sha.update(chunk)
        if sha.hexdigest() != digest:
            raise ValueError("Downloaded checkpoint SHA256 mismatch")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return cfg
