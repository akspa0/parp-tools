# Universal Image-to-Terrain Architecture

**Date:** 2026-07-19  
**Spec owner:** `specs/114-direct-terrain-reconstruction/`  
**Status:** universal contracts, curriculum, student, trainer, and inference implemented; real corpus/training pending

## Product boundary

The product accepts any decodable RGB, RGBA, or grayscale raster at arbitrary practical dimensions
and aspect ratio. It outputs one normalized view-axis-relief field, a deterministic terrain mesh,
and UVs covering the complete source raster.

WoW authored/synthetic minimaps are one exact paired top-down family. They do not limit deployment
to WoW, minimaps, a client build, or a known map. For top-down images, view-axis relief is relative
terrain height. For perspective photos or artwork, the output is a truthful bas-relief terrain
interpretation, not a claim of unique metric scene reconstruction.

## Why the first run is not the model

The user-run `direct_cnn_v112-authored-v1` checkpoint trained on 1,384 WoW authored rows and failed
even its narrow gate:

| Metric | Result |
|---|---:|
| validation MAE | 0.1493349 |
| tile-mean baseline | 0.1387470 |
| delta versus baseline | +0.0105879 |
| gradient MAE | 0.0058671 |
| border MAE | 0.1607286 |

The missing AMP/EMA/scheduler/guidance stack is real, but it is not the fundamental correction.
Optimizer changes cannot manufacture image-domain coverage absent from the corpus. The checkpoint
and backfilled validation sheets remain immutable negative evidence; do not rerun it as the next
universal candidate.

## Geometry stage

```text
any raster
  -> aspect-preserving normalize/tile/pad
  -> general visual student + one continuous relief decoder
  -> normalized view-axis relief
  -> deterministic stitch/grid mesh/normals/UV export
```

The first student candidate uses pinned
[`facebook/dinov2-small`](https://huggingface.co/facebook/dinov2-small) general visual features
(22.1M parameters, Apache-2.0) with one newly trained relief decoder. The smaller
[`nvidia/mit-b0`](https://huggingface.co/nvidia/mit-b0) encoder remains an ablation. Neither model
card claims terrain reconstruction; promotion belongs to Spec 114's gates.

The student architecture is landed: a pinned DINOv2-small patch encoder feeds one compact
progressive continuous-relief decoder, while a trainable full-resolution RGB detail path is fused
before the one relief head so 16x16 patch tokens do not erase local terrain evidence. The default
freezes the encoder and trains the decoder/detail path; unfreezing is an explicit later ablation.
The model accepts one RGB 224x224 tile and emits one bounded 224x224 relief tile; arbitrary image
sizes remain the already-proven tiling/stitching contract. Seven CPU model tests pass without
downloading Hub weights.

Broad images without exact relief may receive offline pseudo-labels from pinned
[`Intel/dpt-hybrid-midas`](https://huggingface.co/Intel/dpt-hybrid-midas), an Apache-2.0 relative
monocular-depth model trained on roughly 1.4M mixed images. It is a non-DepthAnything teacher, not a
deployment dependency. Exact v50 `height_257` remains authoritative for top-down WoW rows.

The teacher builder is landed and pins revision
`17fb43d4437eb62c260a593400db13c22b04511a` plus safetensors SHA-256
`9599793d3ce64d7ebc85657360831596c1df9abc61f6820fe623fe7efb2e29c5`. It is dry-run by default,
requires license/BYOD authority, refuses DepthAnything identities, verifies the downloaded weight
hash, and writes variable-aspect `teacher_pseudo` rows to one Zarr store only after user confirmation.

## Universal curriculum

The first promotable curriculum must contain at least five visual/source families, including exact
v50 top-down terrain plus broad non-WoW imagery. Every crop, render, style variant, and teacher label
shares the underlying `source_group_id`. At least one whole family is held out. Random row or
within-map validation alone cannot promote a universal checkpoint.

Each row records:

- original image identity, mode, dimensions, license/BYOD authority, and visual family;
- complete spatial/style transform lineage;
- exact-numeric or teacher-pseudo target authority;
- teacher ID/revision/hash/orientation when applicable;
- immutable split/source-group identity.

The curriculum index builder is landed. It requires v50 plus at least four distinct external
families, at least one complete compatibility holdout, nonzero exact and teacher-pseudo authority,
and zero group/family leakage. It rejects identical source content relabeled under multiple family
names. Authored v50 rows are usable immediately; synthetic selection fails until the source store
records `NoonWhiteGlobal`. Teacher source images and the generated relief arrays are individually
hashed; curriculum build and training refuse either kind of drift. The Parquet writer explicitly
preserves the union of exact-row and teacher-row lineage fields. The builder is dry-run by default
and writes only after user confirmation.

The landed `universal_relief_contract.py` now proves the pre/post-model boundary independently of a
checkpoint: common raster modes, aspect-preserving overlap tiling, exact-coverage relief stitching,
blank stability, finite deterministic mesh construction, complete UVs, and OBJ/MTL export. Focused
CPU proof is 9/9 tests. This does not yet prove model quality.

## Training and evaluation

Deployment input is the raster only. Exact normals, liquid masks, and heights may guide/mask their
own training losses where available; they never become input channels. The bounded recipe includes
AMP, EMA deploy weights, warmup/cosine decay, gradient clipping, multiscale relief, gradient and
normal guidance, detached hard-error weighting, peak-VRAM/history evidence, and paired spatial plus
photometric/style augmentation.

The landed trainer uses family-balanced sampling, lower authority weight for pseudo labels,
multiscale L1 plus gradient/exact-normal/liquid-aware/hard-error guidance, AdamW, AMP, OneCycle
warmup/cosine, gradient clipping, and EMA deploy weights. It writes immutable checkpoint identity,
per-family/per-row metrics, fixed-scale named validation sheets, global worst cases, history, and
peak VRAM. Model selection uses all validation families; promotion uses only the completely unseen
compatibility family. The inference CLI accepts any supported raster/aspect, stitches normalized
relief, preserves aspect in a bounded mesh grid, and writes a source-textured OBJ/MTL plus 16-bit
relief and a visual proof sheet. Both CLIs are no-write dry runs without explicit confirmation.

Focused contract/model/trainer/inference proof is 48 tests; the broader v50 suite is 224 passed / 4
skipped. Ruff, `py_compile`, and CLI help pass.
No Hub download, broad label build, real curriculum build, CUDA training, or real checkpoint
inference has been performed by the assistant.

Promotion requires:

1. every raster in the >=100-image compatibility suite emits finite relief and mesh/UV artifacts;
2. whole-family paired holdouts beat constant-relief and direct-luminance baselines by >=5% in MAE
   and gradient MAE;
3. no source-group or held-out-family leakage;
4. zero teacher, WDL, map/client identity, or ground-truth signal at deployment;
5. adjacent-border continuity passes;
6. the user accepts >=80% of a >=30-image arbitrary-domain review sheet.

## Texturing boundary

The source image can be UV-projected onto the mesh immediately. Editable terrain texturing is a
later modular chain: land-feature classification, ordered canonical texture-family selection, then
one alpha/blend-stack model. These stages have separate weights, targets, summaries, and gates. They
do not share geometry weights, and they do not delay basic image-textured mesh export.

## Execution ownership

The assistant prepares/tests code and exact commands. The user alone launches Hub downloads,
teacher labeling, corpus builds, CUDA training, and other heavy runs.
