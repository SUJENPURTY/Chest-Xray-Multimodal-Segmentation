"""CPU-only layout repair using already saved initial images; no inference."""
import csv
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / 'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

with (HERE / 'overfit/initial_per_image.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
folder = HERE / 'overfit/initial'
fig, axes = plt.subplots(8, 4, figsize=(12, 24), squeeze=False)
for i, row in enumerate(rows):
    titles = ['X-ray', 'Ground truth', f"Initial Dice {float(row['dice']):.3f}", 'GT green / prediction red']
    for j, kind in enumerate(['xray', 'ground_truth', 'prediction', 'overlay']):
        axes[i, j].imshow(Image.open(folder / f'{i:02d}_{kind}.png'), cmap='gray' if j < 3 else None)
        axes[i, j].set_title(titles[j])
        axes[i, j].axis('off')
fig.suptitle('Eight training images: INITIAL epoch-18 predictions at threshold 0.5', y=0.998)
fig.tight_layout(rect=(0, 0, 1, 0.984))
fig.savefig(folder / 'comparison.png', dpi=120)
plt.close(fig)
print('Initial figure reformatted from existing PNGs; no model execution')
