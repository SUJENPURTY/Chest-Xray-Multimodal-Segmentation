"""Train the canonical LanGuide pipeline. Run python scripts/train.py --help."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import from_environment, with_output_dir
from src.training.trainer import train

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path)
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--num-workers", type=int)
parser.add_argument("--resume", type=Path)
parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = from_environment()
    if args.output_dir is not None:
        cfg = with_output_dir(cfg, args.output_dir)
    changes = {key: value for key, value in vars(args).items()
               if key not in ("resume", "no_amp", "output_dir") and value is not None}
    if args.no_amp:
        changes["mixed_precision"] = False
    cfg = replace(cfg, **changes)
    train(cfg, args.resume)
