# Phase 0 Research: Universal Image-to-Terrain Reconstruction

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

### Tonight bootstrap: authored-only direct geometry

The frozen dual-source curriculum already contains 1,629 authored rows: 1,384 train and 245
validation, spanning Kalimdor (951) and Azeroth (678). These images and their exact `height_257`
targets were not invalidated by the synthetic compositor lighting fix. Therefore the first bounded
run uses only `minimap_source=authored` with the existing 1,561,537-parameter Spec 112 CNN.

This run is evidence, not a substitute for the corrected dual-view bakeoff. It answers the most
immediate question—whether real minimap pixels can learn the offset-invariant terrain field—without
polluting training with stale synthetic lighting. Any `synthetic` or `all` run is now fail-closed
unless the curriculum records `synthetic_lighting_contract=NoonWhiteGlobal`.

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

## Decision 7 — general visual initialization is part of the universal geometry candidate

**Decision**: preserve the failed from-scratch CNN as reproducible negative evidence, but do not make
another from-scratch WoW-only run the universal candidate. The first universal student uses a pinned
general visual initialization and one newly trained continuous relief decoder. The current primary
candidate is `facebook/dinov2-small` (22.1M parameters, Apache-2.0) because it is explicitly a
self-supervised general image feature extractor. `nvidia/mit-b0` remains the smaller ImageNet
baseline. Exact Hub revision, content hash, license, preprocessing, and frozen/fine-tuned state are
part of every run identity.

**Rationale**: the deployment contract now includes images outside the WoW minimap distribution.
The v50 corpus cannot teach broad image semantics by optimizer changes alone. DINOv2-small supplies
general image features while remaining a tractable student backbone; the decoder still predicts one
signal and does not share weights with later models. Neither Hub model card claims terrain output,
so promotion still depends on the repo's paired-relief and arbitrary-image gates.

## Decision 8 — one stage at a time

Implementation order:

1. universal input/relief/mesh contract and whole-domain evaluation suite;
2. universal curriculum plus general visual student and relief-teacher bakeoff;
3. trusted object-label renderer/audit;
4. object-mask model and geometry ablation with generated masks;
5. deterministic land-feature library and classifier;
6. texture-family library/selector;
7. alpha-stack reconstructor and recomposition proof.

No later model is implemented merely because its architecture is known. Each phase ends with its own
real-data and user visual gate.

## Decision 9 — failed authored CNN run requires observability, but not a narrow retry

**Evidence**: the user-owned `direct_cnn_v112-authored-v1` run completed all 100 epochs. Best epoch
92 reached validation MAE 0.1492665126; the in-run per-tile constant baseline was 0.1387469612.
The checkpoint is therefore 7.59% worse and fails SC-001. It learned nontrivial structure, but the
run wrote no prediction sheet, per-row errors, gradient/border metrics, or worst-case review.

**Audit finding**: the bootstrap used AdamW at a constant learning rate and Smooth-L1 plus one
gradient term. It omitted the AMP, EMA, warmup/cosine, gradient clipping, multiscale loss,
normal-guidance, hard-error, validation-preview, and VRAM/history patterns already proven by the
repo's terrain trainers. Repeating it unchanged would not answer why it lost.

**Decision**:

1. Backfill the immutable checkpoint through a separate evaluator and require future runs to emit
   fixed-sample best-epoch previews plus final all-validation metrics and sheets.
2. Keep a source raster as the only deployment input and one normalized-relief output.
3. Use clean numeric signals only as training-time supervision/masking: `normal_xyz` with
   `normal_mask`/`mcnr_mask_257`, `liquid_mask`, and height-derived multiscale/gradient structure.
4. Do not repeat the narrow authored-only trainer. Universal training uses paired spatial transforms
   on raster and relief plus broad photometric/style changes; baked-light direction is deliberately
   varied because fixed WoW lighting is no longer a deployment assumption.
5. Port the proven bounded optimization stack into the universal student only after its curriculum,
   universal compatibility suite, and whole-domain split are fixture-proven.

## Decision 10 — distill broad view-axis relief; exact v50 height stays authoritative

**Decision**: define the universal geometry output as normalized view-axis relief. For top-down
terrain imagery this is relative terrain height. For perspective photographs or artwork it is a
bas-relief interpretation of visible structure, not an assertion of metric scene reconstruction.
Use exact `height_257` for v50 rows. For broad licensed/BYOD imagery without relief truth, allow a
pinned non-DepthAnything teacher such as `Intel/dpt-hybrid-midas` (Apache-2.0, trained on roughly
1.4M mixed monocular-depth images) to create normalized pseudo-labels in a separate user-run build.
Teacher identity and output orientation are stored per row; the teacher is never a deployment input.

**Rationale**: an arbitrary 2D raster does not identify a unique metric 3D scene, but it can define a
stable view-axis relief surface suitable for terrainification. A general monocular-depth teacher
supplies broad image structure; exact v50 height corrects it on the orthographic top-down terrain
family. This keeps one output and permits a compact deployment student.

**Alternatives rejected**:

- Optimizing the 1,384-row authored WoW CNN harder: it cannot create missing image-domain coverage.
- Treating image luminance as truth: retained only as a mandatory baseline because it produces an
  embossing, not learned terrain interpretation.
- DepthAnything-family teachers: explicitly disallowed by the standing project decision.
- Claiming exact terrain for arbitrary perspective art/photos: the inverse problem is non-unique;
  outputs are truthfully labeled image-conditioned relief.

**Pinned teacher implementation evidence**: T010-T011 freeze revision
`17fb43d4437eb62c260a593400db13c22b04511a` and `model.safetensors` SHA-256
`9599793d3ce64d7ebc85657360831596c1df9abc61f6820fe623fe7efb2e29c5`. The builder downloads only
safe weights/config after explicit user confirmation, verifies the file hash before loading, keeps
larger-is-closer/higher orientation, robustly normalizes each label, and writes variable-aspect
`teacher_pseudo` rows into one Zarr store. Seven focused tests pass without model download.
