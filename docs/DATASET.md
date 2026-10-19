# Dataset and annotation restoration

Medical images, masks, patient identifiers, annotation workbooks, and per-image
result tables are local research resources. They are excluded from the proposed
public repository. Obtain QaTa-COV19-v2 and the matching text annotations through
an authorized source and retain their applicable terms. This repository does not
supply or claim redistribution permission for them.

## Layout

The original pipeline accepts `DATA_ROOT` or `--data-root`:

```text
data/QaTa-COV19-v2/
  Train/Images/<image filename>
  Train/GTs/mask_<image filename>
  Test/Images/<image filename>
  Test/GTs/mask_<image filename>
```

The current Phase-2 default is `_qata_compatible/` with the same layout. Set
`PHASE2_DATA_ROOT` to another compatible root rather than moving or recreating
the current dataset. Phase-2 validation uses the `Train/` population only.

Restore these original workbooks under `text_annotation/` when preparing new
manifests:

- `Train_Val_text_for_Covid19.xlsx`
- `Test_text_for_Covid19.xlsx`
- `Train_ID_for_Covid19.xlsx`

`python scripts/create_languide_annotations.py` validates the workbooks, image
pairs, and split membership, then writes `outputs/run/prepared_annotations/` by
default. `ANNOTATIONS_DIR` and `ANNOTATION_OUTPUT_DIR` select alternative locations.
Generated CSVs require the exact columns `Image,Description`.

## Exact Phase-2 population

Restore `outputs/run/prepared_annotations/train.csv` byte for byte. Expected SHA256:

```text
3b22a1b434dbfc0f0da03c111282686515395ae49fae7812c3a168915826e29c
```

The seed-2026 grouped split contains 5,722 training and 1,423 validation images.
Grouping recognizes `sub-...` identifiers; filenames without that identifier are
treated as individual image groups. This does not establish patient independence
where the naming convention lacks patient information. Preserve the original
`experiments/phase2_pilot/data_manifest.json` to verify names, prompts, and order.
The notebook's cached evaluation also verifies the image and mask hashes for every
validation record. No augmented or randomly regenerated split should replace it.

Changing CSV encoding, line endings, ordering, or prompts changes its hash.
Phase-2 intentionally fails on provenance mismatches. Do not edit hashes in saved
records to bypass this check. A new population requires a separately documented
experiment and evaluation.

## Preservation and publication

No dataset file was deleted, moved, or rewritten during GitHub cleanup. The
preservation inventory references the existing dataset metadata inventory; it
does not claim a new full-dataset content hash or make a second copy of all pixels.
All original notebook images remain locally intact. Confirm authorization,
de-identification, and embedded-image rights before publishing those notebooks.
