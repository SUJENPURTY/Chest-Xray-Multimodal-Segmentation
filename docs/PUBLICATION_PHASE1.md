# Phase 1 publication preparation — 2026-10-19

This phase prepares local source and documentation for review. No root Git
repository was present. No commit, push, public upload, deletion, model change,
training, or checkpoint conversion was performed. Flask's visual design remains
unchanged. Research data, results, code, notebooks and preservation backups remain
on disk; exclusion from Git is not permission to delete them.

## Preservation and provenance

The original root `app.py` is identified by the pre-Git preservation inventory,
not by its filename alone. Its 6,240-byte content is SHA-256:

```text
6a1fbf4975946a0c81c6f1f89b88f2ee9e33f69503691ffe3f0c4eecc51576a1
```

The current root file and both retained copies match:

- `outputs/github_cleanup/20261019_pre_github/files/app.py`
- `outputs/flask_migration/20261019/backup/app.py`

The earlier Phase-1 baseline records that the root file was absent before that
preparation. It is already restored at the start of this review and matches the
verified original, so it was not rewritten again. The 4,925-byte archived legacy
app has SHA-256 `79a4bdac6d50240101c6637bd3276667530acbd241c60fe4f89a1a858644a1af`.
It is a different application and was not used as a substitute.

All seven current notebooks were copied privately at review start. Notebooks
01–06 match both preceding preservation inventories byte-for-byte. Notebook 07
already has a discrepancy:

| Version | SHA-256 |
| --- | --- |
| Earlier verified original | `f338aba70d4b63e5842e5f5bcb564cca257833dff7150a3f653761c69366caf2` |
| Current local notebook | `8f2de1eb61e8f41ad555fb68d90922d929f163a6697c21b744b550f0fbfbf631` |

At initial inspection, JSON comparison found only the top-level
`metadata.kernelspec` differed; every cell, source, execution count and output
was identical. During this review, another save inserted one empty Markdown cell
without changing or reordering any existing cell or output. The observed file
then had 34 cells and SHA-256
`2659035c98ee132c6d9e9a4f53b8f43a26e335355236c9fd3410ce692d5f2a16`.
No operation in this task writes to the original notebook; the origin of that
save is not established. This observed version was copied privately as
`observed_external_notebook07.ipynb`, with comparison details in
`notebook07_concurrent_change.json`. All versions remain preserved. Restoration
or removal of that cell is deferred for an explicit decision.

The review evidence is private under
`outputs/publication_phase1/20261019T103718_review/`. Its baseline inventories
172 small-file copies and metadata for 52 previously inventoried weight files.
It supplements the older preservation inventories; it is not a full independent
backup of large datasets and weights. A broad new metadata traversal was stopped
after the small-file copies completed to avoid unnecessary dataset traversal;
the 52 weight paths come from the preceding inventory. Selected checkpoint and
pretrained byte checks are separate fresh hashes in the restoration manifest.

## Notebook publication review

The seven verified pre-Git notebook copies were reviewed for image outputs,
attachments, personal home paths, email/secret patterns and inline image links.
There are 34 embedded image representations. Rights to redistribute medical
images are not established. Personal home paths occur in historical outputs;
source cells also retain fixed project paths and scientific sample references.
Pattern scans do not establish absence of private information or grant rights.

Seven **private, unexecuted publication drafts** are prepared under
`outputs/publication_phase1/20261019T103718_review/publication_notebooks/`.
Every cell source is unchanged. All outputs, execution counts, attachments, and
notebook/cell metadata are omitted. Original image outputs and numerical research
results remain intact in the originals and backups. No medical images are carried
into these drafts. `notebook_publication_review.json` records original/current/
draft hashes and review flags without copying suspected secret values.

The drafts remain ignored and are not approved public artifacts. Human review of
source literals, external links, dataset/model terms and upstream attribution is
still required. Fixed project paths in notebooks 01–06 require a separately
reviewed portability change in publication copies before claiming they execute
on arbitrary clones. No original notebook was rerun or edited by this task.

## File-by-file preparation summary

Some transport fixes and dependency separation were already present when this
review began. They were retained and tested, not attributed to a model rewrite.

| File | Result |
| --- | --- |
| `app.py` | Verified exact original against inventory and two copies; unchanged in this review |
| `webapp/static/js/app.js` | Existing replacement fix retained: invalidate selection/results/downloads before validation; reject stale decodes |
| `webapp/__init__.py` | Existing image/request split retained: image bytes plus 64 KiB multipart budget |
| `webapp/routes.py` | Existing separate image-byte enforcement and template limit data retained |
| `webapp/templates/index.html` | Existing dynamic byte limit and MiB label retained; no layout redesign |
| `webapp/static/css/style.css` | Unchanged |
| `run.py` | Unchanged; Waitress uses Flask's request cap |
| `requirements-core.txt` | Existing shared scientific runtime group retained |
| `requirements.txt` | Existing Flask/Waitress runtime group retained |
| `requirements-research.txt` | Existing training/export/plotting group retained |
| `requirements-notebook.txt` | Existing research/Jupyter/widgets group retained |
| `requirements-test.txt` | Existing Flask/shared unit-test group retained |
| `requirements-web-test.txt` | Existing HTTP/browser verification group retained |
| `requirements-legacy.txt` | Add Matplotlib, imported directly by the retained root app |
| `requirements-local-cuda.txt` | Existing CUDA 12.4/all-workflows recipe retained |
| `constraints-observed.txt` | New observed direct-package versions; platform choice and transitive-lock limits documented |
| `tests/test_flask_app.py` | Add exact image-byte boundary with maximum Unicode prompt; retain multipart/error/cleanup contracts |
| `scripts/validate_flask_browser.py` | Retain real-prediction/replacement/size/race/download regressions; add clear browser-startup diagnostics instead of a misleading `StopIteration` |
| `scripts/verify_restoration.py` | New standard-library, read-only checksum verifier; no deserialization/download |
| `docs/resource_checksums.json` | New 19-file freshly hashed restoration manifest; fingerprints only |
| `.gitignore` | Protect nested private backups/session files, medical volumes, binary tables/weights, spreadsheets and recovery archives |
| `README.md` | Correct startup, dependency groups, limits, restoration, notebook provenance/publication status and verification |
| `docs/CHECKPOINTS.md` | Distinguish Flask-upload and notebook bundles; document checksums, original checkpoint and verifier |
| `docs/FLASK_APPLICATION.md` | Document exact image/multipart limits and replacement behavior |
| `docs/REPRODUCIBILITY.md` | Distinguish historical runs from this review and explain observed constraints |
| `docs/GITHUB_CLEANUP_REVIEW.md` | Correct blanket notebook hash claim with current metadata discrepancy |
| `docs/PUBLICATION_PHASE1.md` | This report, blockers and proposed cleanup |
| `notebooks/*.ipynb`, all `src/` | No writes by this task; six notebooks and all source hashes match, with the concurrent notebook-07 change recorded above |
| Private review folder | Verified small-file copies, notebook drafts, audits and test evidence; excluded from Git |

Streamlit usage is established in `app.py`, `scripts/validate_streamlit_demo.py`,
`scripts/validate_streamlit_ui.py`, and `scripts/validate_presentation.py`.
`src/inference/streamlit_inference.py` remains an active original-checkpoint
helper. Archived Streamlit material is also retained. No related file, installed
package or workflow was removed. The optional requirements group is necessary
for those entry points. Legacy model/layer modules are used by the Flask model
architecture and remain required regardless of the frontend.

## Tests and acceptance evidence

- Full unit suite: **36 passed, 1 failed**. The sole failure is
  `test_encoder_checkpointing_preserves_output_and_gradients`: it expects one
  `torch.utils.checkpoint.checkpoint` call, while the encoder makes none. Setting
  an unused attribute in a test does not enable a supported model feature. The
  scientific model and historical test were left unchanged.
- The first attempts using pytest's default Windows temporary directory hit
  permissions errors. Rerunning with a new, previously nonexistent `--basetemp`
  inside ignored `outputs/` eliminated those fixture errors. Existing temp
  directories were not cleared to make the run pass.
- Image/request boundary test: exact configured image bytes plus a 4,000-character
  Unicode prompt accepted; one byte above the image allowance rejected. Default
  request budget remains image allowance plus 64 KiB.
- Restoration verifier: **19/19 passed**, including both trained checkpoints,
  provenance records and pretrained resources. A second check after CPU/CUDA
  inference also passed **19/19**, confirming the same recorded bytes. No model
  was loaded by these checksum checks.
- Ignore-rule probes: **32/32 passed**, covering secrets, nested patient/private
  trees, notebooks, checkpoints, environments, archives and intended source files.
- Real CUDA inference: passed; native mask and overlay match the preserved
  reference with **zero differing pixels**. Original-image alignment, binary mask,
  temporary-upload cleanup and absence of invented clinical metrics passed.
- Real Chrome browser: passed on retry at widths **1440, 1024, 768 and 390** for
  empty and prediction states, with no horizontal overflow or JavaScript errors.
  Remove/drop upload, prompt selection, comparison, opacity, fullscreen, PNG
  downloads, invalid replacement, oversize, stale decode and reset passed.
  Downloaded mask/overlay pixels match the retained reference exactly.
- The first Chrome start failed because the execution sandbox denied local
  IPC/crash-handler access. The same validator succeeded outside that restriction
  with a dedicated profile and all artifacts confined to ignored `outputs/`.
- Dependency closure: all **8 requirements groups** are satisfied by installed
  direct-package versions. No package installation or upgrade was performed.
- Preservation check: no baseline file missing; all `src/`, original `app.py`,
  Flask CSS and original final-inference artifacts match; **52/52 weight metadata
  records** are unchanged. Notebook 07's concurrent empty-cell insertion is
  explicitly recorded rather than reported as byte-identical.
- Real CPU inference: passed; maximum absolute probability-map difference from
  the retained CUDA reference was **2.562999725341797e-6**, within the existing
  FP32 tolerances (`atol=5e-5`, `rtol=5e-4`). Native mask and overlay both match
  with **zero differing pixels**. Alignment, binary output, upload cleanup and
  absence of invented metrics passed.
- All seven private notebook drafts pass output/attachment/metadata omission
  checks, with every original cell source retained. They remain unpublished.
- Desktop/mobile empty-state screenshots were visually inspected. Medical
  prediction screenshots and downloads stay in the ignored private review folder.
- The temporary CUDA test server on port 5001 was stopped after verification.
  The normal startup remains `python run.py` on port 5000.

Private evidence includes `cuda_validation/report.json`,
`cpu_validation/report.json`, `browser_validation_retry/report.json`,
`ignore_verification.json`, `dependency_verification.json`,
`preservation_verification.json`, `notebook_publication_review.json`, and
`notebook07_concurrent_change.json`. Full-suite JUnit evidence is at
`outputs/publication_phase1/full_suite_workspace_temp.xml`. The final candidate
audit and post-inference restoration check are retained alongside those reports.

These checks do not retrain, recompute full validation, establish patient-level
split independence, or demonstrate clinical utility. Recorded Phase-2 metrics are
validation scores used in checkpoint selection. The original epoch-18 test score
is from a different checkpoint/population. Preserve that distinction.

## Remaining publication gates

1. Approve reviewed publication notebook copies and any source-only portability
   edits; original notebooks and both notebook-07 versions remain protected.
2. Establish dataset, medical-image, annotation and pretrained-resource rights.
   No checkpoint URL, weight release or medical-image distribution is authorized
   or invented by these instructions.
3. Arrange a trusted permitted restoration bundle, including private manifests
   needed by current provenance checks. A Git-only clone is intentionally
   insufficient for full scientific reproduction.
4. Verify installation and restoration in a fresh environment. Observed constraints
   do not resolve all transitive packages or cover every platform.
5. Review a replacement contract for the stale checkpointing test. Change the
   test only after deciding the intended supported behavior; changing scientific
   inference to obtain a green test is outside this phase.
6. Review the exact candidate manifest immediately before staging. Ignore rules
   cannot identify every medical image under arbitrary filenames or remove files
   from existing Git history. No root Git history was available to inspect.

## Proposed final cleanup — approval required before execution

No deletion is needed to publish a reviewed source-only repository. If local
space cleanup is desired later, present an exact file manifest with sizes and
backup coverage for approval first:

| Candidate | Proposed action and condition |
| --- | --- |
| `src/**/__pycache__/`, `scripts/__pycache__/`, `tests/__pycache__/`, `webapp/**/__pycache__/` | Optional deletion of regenerable bytecode after exact manifest review |
| Root `.pytest_cache/` | Optional deletion of test cache after exact manifest review |
| `.cache/pip/` | Optional package-cache deletion after checking offline reinstall needs and distinguishing it from other `.cache/` contents |
| Verified duplicate executed-code/delivery copies | Keep as provenance unless an explicit evidence-retention decision and exact duplicate hashes authorize removal |
| `.ipynb_checkpoints/`, archives, interrupted checkpoint writes | Retain; may contain unique recovery or forensic evidence |
| Datasets, all weights, research source/results, original notebooks, preservation backups | Retain; excluded from blanket cleanup |
| Streamlit files/dependency group | Retain while original-app and validation workflows exist |

Physical reorganization, notebook restoration, model behavior changes, destructive
cleanup and publication are deferred. This phase supplies reviewable evidence
without performing those operations.
