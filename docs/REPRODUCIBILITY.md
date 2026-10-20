# Reproducibility and verification

## Scope

The repository contains preserved research records and working local inference.
Source availability is not complete artifact availability: medical data,
annotations, checkpoints, and several required runtime provenance files are
intentionally excluded from Git. Follow [DATASET.md](DATASET.md) and
[CHECKPOINTS.md](CHECKPOINTS.md) before attempting execution. A fresh-machine,
Git-only reproduction has not been demonstrated.

Earlier verification sections below describe their historical runs. The latest inspection found notebook 07
already differed from its preservation hash in kernel metadata only. A concurrent
save subsequently inserted one empty Markdown cell; all existing cells and outputs
remain identical. No original notebook was written by this task, and all observed
versions are preserved.

Datasets, annotations, checkpoints, and pretrained weights are external and are
not included. Dependency ranges do not lock every transitive package, and a fresh
machine installation has not been verified. A Git-only clone therefore cannot
reproduce the training or inference workflows.

The new Flask frontend reuses the preserved Phase-2 engine. Its setup, API, real
reference validators, browser checks and CPU verification commands are documented
in [FLASK_APPLICATION.md](FLASK_APPLICATION.md). The original Streamlit frontend is
retained; no original model module or numbered notebook was rewritten for migration.

## Observed environment

The local environment reported Python 3.10.20 and these package versions:

| Package | Version |
| --- | --- |
| torch | 2.6.0+cu124 |
| numpy | 1.26.4 |
| Pillow | 11.3.0 |
| monai | 1.4.0 |
| transformers | 4.38.2 |
| pandas | 2.3.3 |
| matplotlib | 3.10.9 |
| nbformat | 5.11.1 |
| nbclient | 0.11.0 |
| nbconvert | 7.17.1 |
| jupyter-client | 8.10.0 |
| ipykernel | 6.31.0 |
| ipywidgets | 8.1.9 |
| psutil | 7.2.2 |
| pytest | 9.1.1 |

These are observed versions, not a resolved cross-platform lockfile. The CUDA
recipe pins the observed torch build; the portable requirement files retain
compatibility ranges. No dependencies were installed or upgraded for cleanup.
Verification uses the existing Windows environment with a GTX 1650 (4 GiB).
Other devices and dependency resolutions need their own verification.

`constraints-observed.txt` now captures the installed versions of direct workflow
dependencies, including Flask, Waitress and the retained Streamlit installation.
Use `pip install -c constraints-observed.txt -r <selected-requirements-file>`.
The PyTorch constraint allows the chosen platform build of version 2.6.0; the
CUDA recipe retains the explicit `+cu124` pin. No package was installed, removed,
or upgraded during this Phase-1 review. Transitive locking and a fresh-environment
installation remain outstanding.

## Cleanup verification recorded on 2026-10-19

- Selected suite: **19 passed** (`test_pipeline.py` and `test_phase2_inference.py`).
  The entire test suite was not asserted to pass.
- Notebook 07: all 14 original code cells executed in order in an isolated copy,
  followed by one additional widget-verification cell; all 15 passed on CUDA.
- The cached 1,423-image validation evidence passed checkpoint, code, CSV, and
  every validation image/mask hash check. Counts differed from the historical
  epoch-2 table for **zero images**. Full validation was not recomputed.
- Fresh single-image, deterministic repeat, external-file, and representative
  sample predictions passed. Two real widget callbacks each produced one PNG;
  widget-cell re-execution removed the old callback, and blank input cleared it.
- The generated widget image, validation distributions, and another representative
  prediction were visually inspected. No live browser/frontend session was tested.
- The preservation comparison found all 2,999 original records present, with
  105 protected content hashes matching, including all seven notebooks and the
  inventoried final Phase-2 artifacts. Selected checkpoint SHA256 matched; other
  checkpoint inventory sizes/timestamps were unchanged.

Local evidence is retained under
`outputs/github_cleanup/20261019_pre_github/`: `pytest-results.xml`,
`preservation_final.json`, and `verification/{execution_report,integrity_checks,widget_verification}.json`.
These reports and the partial backup are deliberately excluded from Git.

## Recommended verification

Activate the root `.venv`, then run:

```powershell
python -m pytest tests/test_pipeline.py tests/test_phase2_inference.py -q
python scripts/verify_phase2_notebook.py
python scripts/audit_public_files.py
```

The first command checks metrics, transforms, model-output conventions, loaders,
and no-gradient inference with small fixtures; it does not need a real dataset or
pretrained downloads. It is a selected test subset. The preserved
`tests/test_memory_optimizations.py` has a stale encoder checkpointing expectation
and is not represented as passing. Repairing it needs a separate reviewed change;
altering the working model just to satisfy that test is not part of cleanup.

The notebook verifier validates notebook format, starts a fresh kernel, executes
all original code cells in order in a copy, and appends a verification-only widget
check. It redirects `OUTPUT` in the copy and checks the original notebook hash.
It performs fresh sample predictions, checks deterministic repeat inference,
retains one model, confirms no gradients or open figures, verifies all saved
validation input hashes and aggregates, and checks that two actual button clicks
each produce one result image. It also reruns the widget cell and checks that the
old callback is disconnected. It forces cached validation reuse and fails on
missing/mismatched evidence rather than silently recomputing 1,423 model forwards.
All reports, notebook copies, plots, and logs go under a new ignored `outputs/`
directory. Review `execution_report.json`, `integrity_checks.json`, and
`widget_verification.json` there. Interactive frontend rendering is a separate
check from kernel-side widget state and PNG inspection.

The public-file audit creates an isolated empty Git metadata directory under
ignored `outputs/`, uses the workspace's `.gitignore` with no global excludes to
enumerate candidates, and scans their names and text for publication risks. It
does not initialize the workspace root, stage, commit, or push. Its manifest is
the exact candidate set for that audit invocation, not proof of future tracked
files. If a real root repository exists, tracked files are audited separately:
`.gitignore` cannot untrack an already tracked file.

## Scientific definitions

- Split seed 2026; image size 224; text length 24; Phase-2 FP32, no AMP, threshold 0.5.
- Validation: 1,423 images from the training manifest; training: 5,722 images.
- Macro scores average per-image ratios; global scores pool confusion counts.
- `Dice = 2TP/(2TP+FP+FN)`; `IoU = TP/(TP+FP+FN)`.
- Empty prediction and target give Dice/IoU 1; undefined precision/recall are 0.
- Confusion counts are on the model grid. Display masks use resized probabilities.
- The selected epoch used this validation set; no Phase-2 independent test score
  or statistical confidence interval is claimed.
- Filename grouping does not prove patient separation when no patient ID exists.
- Text annotations encode finding and location information; generalization to
  unconstrained prompts or clinical deployment has not been established.

The Phase-2 values documented in the README are aggregate validation results
from the checkpoint-selection partition, not an independent test evaluation or
clinical performance claim. The detailed aggregate file and per-image records
are not included in the public repository.

## Historical tools and notebook cautions

| Tool or notebook | Preserved limitation |
| --- | --- |
| Notebook 01 | Contains compatibility-dataset recreation using `shutil.rmtree`; inspect first |
| Notebook 04 | Performs optimization and writes diagnostics; not a verification-only notebook |
| Notebooks 01-06 | Historical environment paths and outputs; not rerun during cleanup |
| `scripts/execute_notebooks.py` | Bulk execution can train and recreate data; older kernel assumptions |
| `scripts/execute_phase2_notebook.py` | Writes the original notebook and normal output folder; use the new isolated verifier for preservation |
| `scripts/verify_phase2_artifacts.py` | Depends on local historical inspection artifacts, earlier execution/widget assumptions, and rewrites the historical repair document |
| `scripts/audit_presentation.py`, `validate_presentation.py` | Six-notebook/epoch-18 presentation assumptions and local archived dependencies; not a complete Phase-2 acceptance check |
| `experiments/phase3_finetuning/` | Prepared programs/configuration; no completed Phase-3 result claimed |

Do not run migration, repair, presentation-generation, or archived execution tools
as a setup step. Do not import experiment launchers to discover dependencies:
some execute work at module scope. No training or checkpoint conversion was
performed during this cleanup.
