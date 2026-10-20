# Checkpoints and offline model resources

All checkpoints and pretrained weights stay local and Git-ignored. There is no
approved public checkpoint URL or runtime-bundle release yet. A Git-only clone
cannot reproduce the Phase-2 result without restoration from the retained local
research artifacts. Do not remove those artifacts after publishing source code.

## Two checkpoint formats

| Workflow | Required format | Typical location |
| --- | --- | --- |
| Original training/evaluation, CLI, Streamlit | `CLOUD_LANGUIDE_BINARY_FIXED_V1`, `model.`-prefixed keys | `outputs/run/checkpoints/best_dice.ckpt` |
| Notebook 07 / Phase2Inference / Flask `run.py` | `PHASE2_FROZEN_BN_PILOT_V1`, unprefixed model keys | Selected checkpoint below |

The loaders are intentionally separate. A Phase-2 checkpoint is not a drop-in
replacement for the original application's checkpoint. The `legacy_*` architecture
files are active dependencies, despite their names.

## Minimal Phase-2 restoration bundle

Restore these paths relative to the repository root without rewriting contents:

```text
experiments/phase2_pilot/config.json
experiments/phase2_pilot/data_manifest.json
experiments/phase2_pilot/checkpoints/best.json
experiments/phase2_pilot/checkpoints/e02_p05722_u01432_68dcea20.ckpt
experiments/phase2_pilot/validation/epoch_2/summary.json
experiments/phase2_pilot/validation/epoch_2/per_image.csv
outputs/run/prepared_annotations/train.csv
outputs/run/phase2_live_inference/validation_summary.json
outputs/run/phase2_live_inference/validation_per_image.csv
```

Also restore the compatible dataset and complete pretrained resources described
below. The last two files permit verification without another full validation
forward pass; their checkpoint, implementation, CSV, and every validation input
hash are checked. They are not substitutes for the dataset or checkpoint.

For **Flask uploaded-file inference**, the compatible dataset, `data_manifest.json`
and the last two cached evaluation files are not read. Flask still requires the
selected checkpoint, `best.json`, pilot configuration, historical epoch-2 summary
and per-image CSV, preserved training CSV, and complete pretrained folders. The
larger list above supports notebook-07 validation as well. Dataset restoration is
required for sample/reference validators and scientific evaluation, not for an
independent authorized upload.

The selected epoch-2 checkpoint is **979,760,366 bytes**, SHA256:

```text
09487406a6783da76c97746bf24e5ba4134317dfd9d7654ef84863cf7d71c7cc
```

The historical epoch-2 per-image CSV SHA256 is:

```text
ec057754bbc56bd4f470094ed4453531d289e5c4b61fbf23777b65988e07d5f5
```

The current live-inference validation CSV SHA256 is:

```text
881f0eeaea6389e766090d6ad8f2ddcd294d9820f585e4880efa7b494d04fe11
```

PowerShell verification example:

```powershell
Get-FileHash experiments/phase2_pilot/checkpoints/e02_p05722_u01432_68dcea20.ckpt -Algorithm SHA256
```

Checkpoint deserialization uses trusted local PyTorch pickle files. Verify the
source and checksum before loading; a checksum alone does not establish trust.
The cleanup has not modified or converted weights.

## Verified restoration manifest

The checksum manifest is retained privately and is not included in the public repository. It records freshly computed
SHA-256 values and sizes for 19 retained files on 2026-10-19: the Phase-2 bundle,
original epoch-18 checkpoint, and complete observed ConvNeXt/CXR-BERT resource
files. It contains fingerprints only; no weights, annotations or patient images.
`repository` paths are relative to the project root. `convnext` and `cxrbert` paths
are relative to their respective complete resource folders.

Verify restored bytes without loading pickle files, importing Torch, or making
network requests:

```powershell
python scripts/verify_restoration.py --workflow phase2 --manifest path/to/your/private/resource_checksums.json
python scripts/verify_restoration.py --workflow original --manifest path/to/your/private/resource_checksums.json
python scripts/verify_restoration.py --workflow all --manifest path/to/your/private/resource_checksums.json
```

The default checks the full Phase-2 notebook bundle and pretrained resources;
`all` also checks the original checkpoint. To check explicit resource locations:

```powershell
python scripts/verify_restoration.py --workflow all --convnext-dir path/to/convnext-snapshot --cxrbert-dir path/to/cxrbert-resources
```

Explicit paths select folders for this verifier only. Runtime selection still
follows the resolver below; use its supported cache structure or bundled text
folder. The verifier returns a nonzero exit status for missing or mismatched
files. It does not establish dataset integrity, patient separation, license
permission, or trust in custom code, and it does not rewrite provenance records.

The original `outputs/run/checkpoints/best_dice.ckpt` is **979,158,414 bytes**,
freshly verified SHA-256:

```text
748c4e318b88483e3e83b2d4c9b877960d322e3548d7f6cc8435aaa5bd88f72b
```

Retain its original format for CLI and Streamlit use. `MODEL_CHECKPOINT` can point
those workflows to a trusted local copy. The optional existing HTTPS provisioning
mechanism requires both `MODEL_CHECKPOINT_URL` and `MODEL_CHECKPOINT_SHA256`; no URL
is supplied or authorized here. Phase-2 Flask does not use that mechanism.

## Pretrained resources and offline resolution

The model uses `facebook/convnext-tiny-224` and
`microsoft/BiomedVLP-CXR-BERT-specialized`. The resolver searches complete snapshots
in `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME/hub`, and the default home-directory
Hugging Face cache. It also prefers this existing local CXR-BERT folder when present:

```text
Project/LanGuideMedSeg-MICCAI2023/lib/BiomedVLP-CXR-BERT-specialized-official/
```

A complete vision folder has `config.json` and `pytorch_model.bin` or
`model.safetensors`. The CXR-BERT folder additionally requires
`tokenizer_config.json`, `vocab.txt`, `modeling_cxrbert.py`, and
`configuration_cxrbert.py`, plus any referenced tokenizer/special-token resources
from that same release. Preserve the complete trusted snapshot, not a mixture of
files from different versions. The loader enables custom model code; inspect and
trust that code before execution.

The observed ConvNeXt cache revision was
`6166b7613034066690a621d8bf25ffdf181a34f0`. The local CXR-BERT folder is not identified
by a verified upstream revision in the current records. Its configuration, custom
code, and tokenizer must be retained with the local bundle; a public immutable
resource release remains to be arranged and verified.

Notebook 07 sets Hugging Face offline mode. It constructs backbone configurations
and then strictly restores the selected complete model state. The current resolver
still requires complete resource folders even though redundant backbone weights
are not loaded into another model. The root `pretrained/convnext-tiny-224/` folder
is not automatically selected by notebook 07; do not assume that keeping it alone
makes an offline clone runnable. Restore a supported cache snapshot or the existing
local text folder before verification. No download was performed during cleanup.

## Full training lineage

The bundle above is sufficient for the current inference contract, not for every
historical training/resume audit. Full experiment reproduction also needs original
epoch-18 weights, earlier Phase-1 records, immutable split/order manifests, optimizer
and RNG state where resuming, and the associated experiment records. These remain
locally preserved and ignored. Review their distribution terms and personal paths
before selecting any for a separate release.
