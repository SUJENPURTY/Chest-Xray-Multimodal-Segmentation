"""Evaluate the selected checkpoint on the held-out QaTa test split."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import from_environment, with_output_dir
from src.training.trainer import evaluate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--output-dir", type=Path)

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = from_environment()
    if args.output_dir is not None:
        cfg = with_output_dir(cfg, args.output_dir)
    overrides = {"data_root": args.data_root, "checkpoint_path": args.checkpoint,
                 }
    cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    metrics, per_image_results = evaluate(cfg)
    print(metrics)
