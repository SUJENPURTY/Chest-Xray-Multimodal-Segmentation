# GitHub cleanup review and retained decisions

Safe changes preserve all original notebooks, source model logic, datasets,
weights, experiment records, and final Phase-2 outputs. No paths were renamed or
moved. Documentation and Git exclusions provide the organization; physical
reorganization would risk existing experiment and restoration paths.

The subsequent Flask migration adds `webapp/`, `run.py`, web validation scripts,
transport tests and an empty-interface screenshot. See
[FLASK_APPLICATION.md](FLASK_APPLICATION.md). These source files are KEEP; generated
uploads, model caches, browser profiles, prediction screenshots and private
validation artifacts stay under ignored `outputs/`. The original `app.py` remains
KEEP pending explicit approval of any later Streamlit removal.

## Preservation inventory

Before editing existing files, the local cleanup saved
`outputs/github_cleanup/20261019_pre_github/inventory.json` and verified small-file
backup copies under its `files/` directory. It inventories **2,999** research/source
files and contains **2,427** SHA256-verified copies, including all seven notebooks,
original documentation, code, configurations, and smaller research artifacts.
Larger files have size/mtime records; the selected Phase-2 checkpoint has a fresh
SHA256 matching its saved pointer. The coverage and exclusions are recorded in
the inventory. Dataset pixels, environments, and large weights were not duplicated.
This is a local preservation inventory and partial backup, not an independent
disaster-recovery backup of every checkpoint and dataset. Keep it private.

`external_resources_inventory.json` supplements that baseline with 17 upstream
library and external Hugging Face snapshot resource records. Small resource
files were copied and hashed; larger pretrained weights have metadata records.

An external backup of ignored datasets and weights remains advisable before any
later destructive cleanup. Do not mistake exclusion from Git for permission to
delete a local artifact.

## Classification

`KEEP` means retain in the proposed source repository. `GITIGNORE` means preserve
locally and exclude from Git. `REMOVE` identifies a regenerable deletion candidate
only; **nothing was deleted**. `REVIEW` requires a decision before removal or public
distribution. Entries with publication review retain their local originals.

| Files or folders | Classification | Reason / dependency |
| --- | --- | --- |
| `src/` Python source and package files | KEEP | Active shared data/model/training/inference implementation |
| `src/models/legacy_model.py`, `legacy_layers.py` | KEEP | Required architecture and parameter names; not obsolete |
| All seven numbered `notebooks/*.ipynb` | KEEP + REVIEW publication | Preserve experiments, Markdown, metrics, and images; embedded medical images and paths need review |
| `scripts/train.py`, `evaluate.py`, `infer.py` | KEEP | Original pipeline entry points |
| `scripts/create_languide_annotations.py` | KEEP | Workbook-to-manifest preparation and validation |
| `scripts/validate_streamlit_demo.py`, `validate_streamlit_ui.py` | KEEP | Original application's verification utilities |
| `scripts/verify_phase2_notebook.py`, `audit_public_files.py` | KEEP | Safe isolated execution and public payload inventory |
| `scripts/execute_notebooks.py` | REVIEW, retained | Bulk execution may retrain/recreate data; not proven unused |
| `scripts/execute_phase2_notebook.py`, `verify_phase2_artifacts.py` | REVIEW, retained | Historical preservation/widget contracts differ; retain until replacement is reviewed |
| `scripts/audit_presentation.py`, `validate_presentation.py` | REVIEW, retained | Historical six-notebook presentation pipeline depends on archived originals |
| `tests/test_pipeline.py`, `test_phase2_inference.py` | KEEP | Selected validation suite, no real model downloads |
| `tests/test_memory_optimizations.py` | REVIEW, retained | Encoder checkpointing assertion is stale; no deletion or model changes authorized |
| Experiment Python programs, `config.json`, protocols and README files | KEEP | Diagnostic/training lineage and reproducible settings; not rerun by cleanup |
| `experiments/**/executed_*.py` and source snapshots | GITIGNORE + REVIEW | Byte-identical copies can be evidence of executed code; retain locally |
| Experiment checkpoints, histories, order/split manifests, per-image tables, arrays, monitoring, logs, and reports | GITIGNORE | Preserve experiment record; some required by downstream experiments and Phase-2 restoration |
| `experiments/phase1b_memory_fix/checkpoints/periodic_step_0320.ckpt.tmp` | REVIEW, GITIGNORE | Interrupted-write evidence, not a valid selected checkpoint; forensic report refers to it |
| `experiments/phase3_finetuning/` code/config | KEEP | Prepared future work; not a completed result |
| `outputs/run/phase2_live_inference/` | GITIGNORE | Final scientific outputs, cache, inspection and execution evidence; preserve unchanged |
| Other `outputs/run/` subfolders | GITIGNORE | Original checkpoints, annotations, metrics, figures, predictions and diagnostics |
| `outputs/github_cleanup/`, future `outputs/verification/` | GITIGNORE | Private backups, candidate manifests, kernel state, logs and copied notebook outputs |
| `results/phase2_validation_summary.json` | KEEP | Derived aggregate without per-patient records or medical images |
| `results/.gitkeep`, `outputs/.gitkeep` if present | KEEP | Lightweight layout placeholders |
| New dataset/checkpoint/reproducibility/review docs | KEEP | Restore instructions, limitations and publication decisions |
| Seven historical docs/audit files listed explicitly in `.gitignore` | GITIGNORE + REVIEW | Preserve prior evidence; local paths and superseded presentation assumptions |
| `README.md`, dependency files, `pytest.ini`, `LICENSE` | KEEP | Setup, tests, license and attribution; GPL license preserved |
| `app.py` | KEEP | Existing original-checkpoint Streamlit application |
| `.venv/`, `.venv-demo/`, `venv/` | GITIGNORE | Working local environments; not safe to remove during preservation |
| `Project/` | GITIGNORE | Local upstream checkout, dataset and required text model resources |
| `_qata_compatible/`, `data/`, `dataset/`, `datasets/` | GITIGNORE | Local medical datasets; do not delete |
| `text_annotation/` | GITIGNORE | Original annotation workbooks and identifiers |
| `pretrained/`, checkpoints anywhere | GITIGNORE | Large local weights; preserve restoration sources |
| `archive/` | GITIGNORE + REVIEW | Historical notebooks, migration/repair/generation tools; some audit tools depend on it |
| `.cache/` | GITIGNORE | Pip downloads, edit backups, notebook state and previews; mixed contents need review before deletion |
| `.cache/pip/` | REMOVE candidate, GITIGNORE | Re-downloadable package cache; no runtime import depends on cached wheel bodies; offline availability may matter |
| `**/__pycache__/`, `.pytest_cache/` | REMOVE candidate, GITIGNORE | Regenerable bytecode/test cache; source and tests remain |
| `**/.ipynb_checkpoints/` | REVIEW, GITIGNORE | Recovery copies may contain unique unsaved work; compare before deleting |
| Jupyter connection JSON, IPython history databases | REVIEW, GITIGNORE | Local connection secrets/history; not research inputs; inspect before later disposal |
| `.vscode/`, `x-ray.code-workspace` | GITIGNORE | Machine-specific editor configuration |
| `presentation/` | REVIEW, GITIGNORE if restored | Root folder absent in the current inventory; any restored slides/assets need metadata and medical-image rights review |
| `reference_papers/` | REVIEW, GITIGNORE | Third-party PDF redistribution not verified; README retains a paper link |

All unlisted generated files under the classified trees inherit that tree's
decision. No file is considered unused solely because it lacks a direct import.

## Duplicate and obsolete candidates: no deletion performed

| Candidate | Evidence / proposed decision | What may depend on it |
| --- | --- | --- |
| Phase-2 `executed_run_pilot_*.py` copies versus `run_pilot.py` | Prior audit found identical copies; retain ignored as executed-code evidence | Historical run and preservation reports |
| Phase-1C B/C `executed_run_variant.py` versus `run_variant.py` | Identical snapshot bodies in prior audit; retain until evidence policy is agreed | Experiment lineage/reports |
| Final and Reviewed presentation PPTX/PDF copies described in historical audits | Historical audit reported identical final/reviewed pairs; current root `presentation/` is absent, so existence and equality are not reverified | Delivery records and presentation review |
| Presentation build/review workspace assets described in historical audits | Not currently inventoried at the root; review restored files for unique content before proposing removal | Slide rebuild/review utilities |
| Archived migration, repair, Kaggle and notebook-generation utilities | Historical, sometimes destructive; no blanket deletion proposed | `audit_presentation.py` uses archived notebook originals |
| Phase-2 inspection `rebuild_notebook.py`, `polish_notebook.py`, exported text snapshots | One-time repair artifacts, retained ignored | Repair provenance; not normal inference |
| Standard bytecode/test caches and pip downloads | Regenerable; first candidates for an explicitly approved local deletion | No scientific pipeline dependence; pip cache can support offline reinstall |

No exact duplicate nonempty code cells were found across the seven final notebooks
in the preceding audit. Similar functions across experiments often preserve an
isolated experiment implementation; consolidation could invalidate historical
comparisons and is not part of this cleanup.

## Size and privacy findings

The preceding full recursive metadata audit measured approximately 58.7 GiB,
including about 44 GiB of experiments, 5 GiB of environment files, 4.9 GiB under
`Project/`, 2.7 GiB of cache, and 1.8 GiB of outputs. These are audit-time approximate
sizes, not new measurements after backups. They remain local and excluded.

A connection JSON in `.cache/tmp/` contains Jupyter session key/port fields, and
IPython history is present. Values are intentionally not reproduced here. The
earlier targeted scan found no common token/private-key patterns in its text
scope; that does not certify all binary assets, images, archives or history.
The exact candidate scan records path/secret/large-file/embedded-image flags
without printing suspected secret values. OCR, exhaustive binary forensics, and
Git history scanning are outside that check. No root `.git` existed at the start,
so there was no local tracked-file or commit-history baseline to inspect.

The candidate scan identified home-directory strings in notebooks **02, 03, 05,
and 06**, and **34 embedded image outputs across the seven notebooks**. These are
review flags, not a claim that every image contains private medical information.
All seven originals remained byte-identical during that preceding audit. The
current Phase-1 review found a pre-existing notebook-07 kernel-metadata change;
its existing cells and outputs are identical, and a later concurrent save added
one empty Markdown cell. All observed versions remain preserved. See
[PUBLICATION_PHASE1.md](PUBLICATION_PHASE1.md). Publication should wait for an approved
decision on sanitized public copies and image permissions.

## Proposed public layout

```text
.gitignore
LICENSE
README.md
requirements.txt
requirements-notebook.txt
requirements-local-cuda.txt
pytest.ini
app.py
src/
scripts/
notebooks/01_...ipynb through 07_...ipynb  [publication review pending]
tests/
docs/{DATASET,CHECKPOINTS,REPRODUCIBILITY,GITHUB_CLEANUP_REVIEW}.md
experiments/<phase>/{programs.py,config.json,PROTOCOL.md,README.md}
results/phase2_validation_summary.json
```

This is the current proposed payload, not permission to distribute medical
images. A later review may produce public notebook copies with sanitized outputs
while preserving all originals locally. That change has not been made here.

## Ordered next steps requiring approval

1. Review the exact candidate manifest and scan flags under the private cleanup
   folder; approve how to remove personal paths and restricted embedded images
   from public notebook copies without changing originals.
2. Confirm dataset/model/paper/slide redistribution terms and choose whether any
   images, reference materials, or presentation should be distributed at all.
3. Choose a trusted external release/backup location for permitted checkpoint and
   provenance bundles; verify restoration on a fresh clone before claiming full
   reproducibility. No distribution location has been invented.
4. Review a separate repair of the stale memory test and historical verification
   tools. Keep the working inference model unchanged unless an independent fix
   is needed and approved.
5. Only after an independent backup and an explicit deletion list is approved,
   remove selected caches or proven duplicate delivery artifacts. Do not remove
   datasets, selected/earlier checkpoints, experiment records, or archived unique
   notebooks as part of a generic cache cleanup.
6. Re-run the candidate audit immediately before any future staging. No files
   were staged, committed, pushed, or publicly published by this cleanup.
