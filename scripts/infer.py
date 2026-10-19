"""Run one image and prompt through the same predictor as Streamlit."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import from_environment
from src.inference.predictor import LanGuidePredictor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--image", required=True, type=Path)
parser.add_argument("--prompt", required=True)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--output-dir", required=True, type=Path)

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = from_environment()
    if args.checkpoint:
        cfg = replace(cfg, checkpoint_path=args.checkpoint)
    result = LanGuidePredictor(cfg).predict(args.image, args.prompt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result["binary_mask"].save(args.output_dir / "mask.png")
    result["overlay"].save(args.output_dir / "overlay.png")
    result["probability_image"].save(args.output_dir / "probability.png")
    print(f"Saved prediction to {args.output_dir}")
