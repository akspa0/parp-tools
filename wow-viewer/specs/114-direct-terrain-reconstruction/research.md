# Phase 0 Research: Direct Minimap-to-Terrain Reconstruction

**Date**: 2026-07-19

## Evidence inventory

The Hugging Face CLI was used from the existing uv environment:

```powershell
uv run hf models list --search "segformer semantic segmentation" --sort downloads --limit 8 --format json
uv run hf models list --search "mask2former" --sort downloads --limit 8 --format json
uv run hf models list --search "dinov2 semantic segmentation" --sort downloads --limit 8 --format json
```

The Hub has official, Transformers-native checkpoints for NVIDIA SegFormer/MiT and Meta
Mask2Former/DINOv2. None is a ready-made WoW minimap-to-height or minimap-to-MCAL model. They are
candidate encoders/architectures; the project still needs task-specific heads, targets, split
discipline, and BYOD training.

Primary references and official model surfaces:

- [SegFormer paper](https://arxiv.org/abs/2105.15203) and
  [Hugging Face SegFormer docs](https://huggingface.co/docs/transformers/model_doc/segformer)
- [NVIDIA MiT-B0 encoder](https://huggingface.co/nvidia/mit-b0) and
  [SegFormer-B0 ADE checkpoint](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [Mask2Former paper](https://arxiv.org/abs/2112.01527) and
  [Swin-tiny semantic checkpoint](https://huggingface.co/facebook/mask2former-swin-tiny-ade-semantic)
- [DINOv2 paper](https://arxiv.org/abs/2304.07193) and
  [DINOv2-small checkpoint](https://huggingface.co/facebook/dinov2-small)
- [U-Net paper](https://arxiv.org/abs/1505.04597)
- [Single-aerial-image height ordinal regression](https://arxiv.org/abs/2006.02801)
- [Pix2pix conditional image translation](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf)

## Decision 1 — direct relative height replaces a mandatory WDL prior

**Decision**: the first model consumes minimap RGB (and, after US2, a generated object-cleanup
signal) and directly predicts the existing `v112.1` relative-height target. There is no WDL input,
WDL auxiliary head, or WDL teacher forcing. In constitution terms, the output is one residual signal
relative to the target's fixed zero/mean baseline.

**Rationale**: the v50 corpus now pairs exact `height_257` with corrected synthetic terrain RGB and
authored RGB. A WDL lattice is a lossy coarse projection of the target we already own; making it a
required intermediate can only constrain or leak the answer. Direct single-image height estimation
is an established supervised dense-prediction formulation. The cited aerial-height work uses an
encoder/decoder and shows multi-scale context is important; unlike unconstrained aerial imagery,
our source/target projection is fixed and exact.

**Baseline/candidate**:

1. Keep Spec 112's lean from-scratch relative-height CNN as the mandatory baseline.
2. Candidate A: a MiT-B0/SegFormer-style hierarchical encoder with a one-channel continuous
   regression decoder at 257×257. SegFormer-B0 is attractive because the paper reports a compact
   multiscale design (3.7M parameters for B0) and the official HF implementation accepts variable
   input sizes without learned positional interpolation.
3. Candidate B: a lean U-Net/ConvNeXt-style encoder-decoder with the same one-channel contract.
4. Optional ablation: ordinal height bins plus continuous residual refinement only if direct
   continuous regression demonstrably stalls. It is not the initial design because `v112.1` already
   provides a bounded, offset-invariant target.

**Rejected**:

- WDL-first or WDL-plus-height multi-head models: redundant intermediate and constitution violation.
- DepthAnything/DA-V2 and generic monocular-depth checkpoints: explicitly out of this terrain lane,
  perspective/depth priors mismatch the top-down orthographic target, and direct exact labels exist.
- A conditional GAN for height: a plausible-looking hallucinated surface is worse than a measurable
  numeric error. Pix2pix is relevant as image-translation history, not the geometry loss owner.

## Decision 2 — train on paired views, but preserve deployment truth

**Decision**: every source tile may contribute two input rows with the same relative-height target:
corrected synthetic RGB and authored RGB. Both rows share `source_group_id` and split. The curriculum
records input origin, object-mask availability, and upstream checkpoint identities. Deployment
metrics are reported on authored RGB; synthetic metrics diagnose the clean-domain ceiling.

**Rationale**: synthetic RGB is clean and exactly registered to numeric terrain, while authored RGB
is the actual deployment domain and may contain objects/icons/water/material differences. Treating
them as separate views of one target uses both without claiming pixel equality. This extends the
dual-view discipline already proven in Spec 112.

**Rejected**: using high-resolution synthetic RGB as numeric height truth. It is a rendered
observation and may supervise SR/detail appearance, while `height_257`, `normal_xyz`, `alpha_256`,
and material IDs remain the actual numeric truth.

### T017/T018 record: authored-only bootstrap run completed and failed SC-001

The frozen dual-source curriculum already contains 1,629 authored rows: 1,384 train and 245
validation, spanning Kalimdor (951) and Azeroth (678). These images and their exact `height_257`
targets were not invalidated by the synthetic compositor lighting fix. Therefore the first bounded
run used only `minimap_source=authored` with the existing 1,561,537-parameter Spec 112 CNN.

**Outcome (immutable negative evidence, 2026-07-19)**: the user-owned
`direct_cnn_v112-authored-v1` run completed all 100 epochs. Best epoch 92 reached validation MAE
0.1492665126; the in-run per-tile constant baseline was 0.1387469612. The checkpoint is therefore
7.59% worse than predicting each tile's mean and fails SC-001. The separate evaluator confirms MAE
0.1493349, gradient MAE 0.0058671, and border MAE 0.1607286 over the 245 held-out rows, with
per-row, quantile, and worst-case artifacts backfilled. **Do not rerun the same recipe.**

**Audit finding**: the bootstrap used AdamW at a constant learning rate and Smooth-L1 plus one
gradient term. It omitted the AMP, EMA, warmup/cosine, gradient clipping, multiscale loss,
normal-guidance, hard-error, validation-preview, and VRAM/history patterns already proven by the
repo's terrain trainers. It also wrote no prediction sheet, per-row errors, or border metrics at
training time; those were backfilled afterward and are now required observability for every future
run.

**Implication**: the failure does not invalidate the direct dual-view route; it invalidates this
specific narrow recipe. The next geometry candidates remain the original plan's T014
`mit_b0_regression` and Candidate B U-Net-style decoder on the corrected dual-view curriculum,
trained with the proven bounded optimization stack, compared against the frozen `direct_cnn_v112`
metrics on the same split. Any `synthetic` or `all` run stays fail-closed unless the curriculum
records `synthetic_lighting_contract=NoonWhiteGlobal`.

**Reverted detour (2026-07-19)**: an unauthorized "universal arbitrary-raster" reset briefly
replaced this spec with a DINOv2 student, a DPT-Hybrid/MiDaS pseudo-label teacher, and broad
third-party image folders. That route was reverted: the deployment contract is the authored WoW
minimap, the dataset is the project-owned v50 Zarr store, and no third-party image corpus or
teacher model is part of this spec. Pretrained encoder weights remain only an optional
license-recorded, hash-pinned ablation per FR-013, compared against the from-scratch baseline on
the same split.

## Decision 3 — trusted object visibility is a prerequisite, not an RGB difference

**Decision**: add a dedicated object-label proof before training object cleanup. Labels must be
rendered from verified object placement/geometry into the same top-down tile projection, preserve
per-row/build hashes, and distinguish unavailable from empty. Raw `authored - synthetic` difference
is forbidden as mask truth.

**Rationale**: the two RGB domains differ in lighting, water, material selection, and objects. An
image difference would label all domain changes as objects—the same mistake that made NCC an invalid
pair owner in Spec 113. Current v50.1 explicitly dropped the old interpolated object masks, so the
new spec must earn a trustworthy signal instead of relabeling a dead one.

**Architecture choice**:

- Primary: a separate SegFormer-B0 semantic model for `object / terrain / unknown` visibility. It is
  compact, multiscale, and directly supported in Transformers/Hugging Face.
- Escalation only: Mask2Former with a Swin-tiny backbone if the downstream contract genuinely needs
  separate object instances or overlapping masks. HF exposes both semantic and instance-capable
  Mask2Former checkpoints, but the architecture is substantially heavier and its Hub checkpoint
  license is not the clean default for a first baseline.
- DINOv2-small (22.1M parameters, Apache-2.0 checkpoint on HF) is an optional frozen/fine-tuned
  encoder ablation for domain robustness, not a default dependency and not a segmentation head by
  itself.

**Downstream rule**: geometry training must see generated mask/cleaning outputs (including mistakes),
not only ground-truth masks. The first clean baseline remains raw RGB so the mask's value is measured.

## Decision 4 — a deterministic terrain-feature library precedes a classifier

**Decision**: derive versioned feature labels from exact numeric evidence before choosing a model.
The first library uses geometry and material facts that can be reproduced:

- flat/rolling/steep/cliff/basin/ridge categories from relative height, slope, and curvature;
- coast/river/lake/wetland/magma adjacency from liquid mask/type and geometry;
- material-context families from `mcly_tileset_ids`, ordered `mcly_texture_ids`, layer presence, and
  alpha statistics;
- full-map pattern/family identifiers from the existing Spec 076/103 evidence ledger;
- explicit `unknown`, `mixed`, and `unavailable` states.

Then train a separate SegFormer-B0 semantic classifier from authored minimap (plus generated object
mask if proven) to the library labels.

**Rationale**: a model cannot invent a stable library definition. Deterministic labels make the
classifier inspectable and allow family-safe partitions. SegFormer is a better first fit than
Mask2Former because this is per-pixel/region semantic classification, not instance discovery.

**Rejected**: clustering only the RGB pixels. That would learn map palettes and lighting rather
than reusable terrain semantics. Learned embeddings may assist family discovery but cannot replace
numeric/provenance rules.

## Decision 5 — texture identity and alpha blending are separate models

**Decision**:

1. **Texture-family selector**: predicts an ordered canonical family tuple plus confidence from
   authored RGB and generated land-feature context. Start with a compact classifier/dense semantic
   model; raw client texture IDs are mapped into a versioned library and never treated as universal
   classes.
2. **Alpha-stack reconstructor**: predicts one bounded ordered four-layer blend stack conditioned
   on RGB plus generated family selections. Start with a lean U-Net/feature-pyramid regressor because
   alpha is a spatially precise image-to-field problem. Its one output signal is the complete
   ordered alpha stack; layer IDs remain the selector's output.

**Rationale**: identity and mixture shape have different error modes. A texture library can improve
without retraining alpha geometry, and alpha can improve without changing which families exist.
The existing `alpha_256`, `mcly_layer_mask`, `mcly_texture_ids`, and `mcly_tileset_ids` provide
direct supervision. Recomposition through `TerrainMinimapCompositor` supplies a meaningful visual
diagnostic without replacing numeric alpha metrics.

**Guardrail**: no MCAL reader rewrite and no `AlphaWdtWriter` change. The models consume the frozen
decoded contract and produce new prediction artifacts only.

## Decision 6 — Spec 113 owns visual detail enhancement

**Decision**: do not create a second detail/SR model here. Spec 113's RealPLKSR ×4 remains the owner
of authored-minimap visual upscaling and high-resolution synthetic-detail supervision. Spec 114 may
consume a validated SR result for operator visualization or future high-resolution feature
experiments, but geometry/alpha truth remains numeric.

**Rationale**: one owner avoids competing trainers and checkpoints. RealPLKSR is already selected for
ComfyUI-native delivery, while terrain geometry and alpha have different objectives.

## Decision 7 — pretrained weights are optional ablations

**Decision**: every stage first proves a small from-scratch baseline. A Hub checkpoint may be tested
only when its license/source/revision/hash are recorded and it uses the same frozen split and output
contract. Pretraining promotion requires a material improvement, not assumed transfer.

**Rationale**: ImageNet/ADE/COCO features may help edges and regions, but WoW minimaps are a stylized
orthographic domain. SegFormer and DINOv2 provide strong, accessible starting points; neither model
card claims terrain reconstruction. Architecture reuse is safer than assuming checkpoint semantics.

## Decision 8 — one stage at a time

Implementation order:

1. corrected synthetic RGB visual gate (Spec 113 dependency);
2. direct geometry baseline and MiT-B0 bakeoff;
3. trusted object-label renderer/audit;
4. object-mask model and geometry ablation with generated masks;
5. deterministic land-feature library and classifier;
6. texture-family library/selector;
7. alpha-stack reconstructor and recomposition proof.

No later model is implemented merely because its architecture is known. Each phase ends with its own
real-data and user visual gate.
