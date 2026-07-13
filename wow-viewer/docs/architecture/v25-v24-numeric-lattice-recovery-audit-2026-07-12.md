# V25 / V24 Numeric Lattice Recovery Audit

**Status:** Recovery Phase 0 passed; M0 is the only unblocked learned stage.

## Decision

Recover the small V24-style sequential path, but retain its correct numeric WDL lattice contract. Do not blindly revert to `ef99e7152859f5fa86a79962908727bbffc912c0`: it is the last comparatively simple trainer-control-flow reference, not deployable-input proof. Its useful properties are small separate models, separate checkpoints, numeric lattice target, progress reporting, and early stopping. Its synthetic-WDL forward input must not be retained for deployment.

The unified V25 decompiler is permanently invalid as architecture and proof: it jointly trains terrain, object, clean-minimap, texture, alpha, and WDL heads, violates the residual-chain constitution, and contains a false WDL target.

## Confirmed Contract Failure

The authoritative C# WDL reader/writer represents one MARE tile as exactly:

- `outer_17`: 17 x 17 = 289 signed-height samples at `height_257[::16, ::16]`.
- `inner_16`: 16 x 16 = 256 signed-height samples at `height_257[8::16, 8::16]`.

That is 545 real numeric samples. It is not a 33 x 33 raster. A 33 x 33 array can only be a derived helper when it preserves outer/inner sample identity and an invalid-node mask. Filling the other 544 locations with interpolation invents targets and changes the task.

V24's `harvester.v24.lattice` preserves the paired grids and makes a deterministic quincunx helper. V25's `wdl_height_33_from_257`, `WdlDownsampler`, H1 target, unified trainer, inference, and validation instead use `height_257[::8, ::8]`. They must not be called WDL or used as WDL supervision/export input.

## Inputs, Labels, and Numeric Signals

| Signal | Deployment availability | Correct role now |
|---|---|---|
| Raw minimap RGB | yes | input to object segmentation, then deterministic cleaning and terrain models |
| Object mask | no | target for its own segmentation model; predicted mask feeds the cleaner |
| Clean minimap RGB | yes, after segmentation | terrain-model input |
| H0 tile datum | yes, frozen upstream prediction | numeric scalar input to W1 |
| `outer_17` + `inner_16` | no | one structured numeric WDL-lattice target/output |
| Raw MCVT vertex Z + topology | no | canonical numeric terrain target/output |
| `height_257` | no | derived display/projection only until its lossless raw-vertex mapping is proven |
| Native MCNR normals | no | numeric geometry validation first; possible later consistency loss only after alignment proof |
| Alpha, textures, placements, PM4, liquids | no | separate future lanes; not height-chain inputs or heads |

Normals are numeric surface-direction facts, not RGB-like images. They are valuable for checking predicted local slope, but cannot be forward inputs in a minimap-only model. Do not add a normal head or image encoder.

Terrain-shadow imagery is different: it is an observed raster input whose lighting pattern can correlate with terrain shape. The intended model is therefore **raster input -> numeric mesh-vertex output**, not image-to-image height reconstruction. It may use a small image encoder, but its decoder/loss must address the real ADT vertex lattice and world-coordinate ordering. PNGs, OBJ files, and mesh renders are post-inference diagnostics only.

### Real-client terrain-shadow guidance

The viewer's existing capture automation is the canonical way to create a
terrain-shadow reference from real client terrain. The capture configuration
must use a fixed camera and the renderer's fixed global-light settings, while
disabling objects, liquids, diffuse textures, alpha-mask composition, and
vertex-colour tint. The output is a flat-albedo terrain lighting/shadow capture
whose variation is driven by the actual terrain surface and the recorded
viewer-lighting contract. The current values are fixed renderer constants,
not decoded client light-table values.

This is not a new deployment input and not a model output. It has two valid
uses only:

1. An upper-bound probe: measure how strongly real terrain-shadow captures
   correlate with raw mesh vertices before spending GPU time on a learner.
2. After a vertex-Z model clears its numeric gate, a separately reported
   guidance metric/loss: deterministically render the predicted mesh under the
   identical capture contract and compare it with the real terrain-shadow
   capture.

Every capture manifest must record the staged-client build, tile identity,
camera, light parameters, disabled renderer features, and viewer revision.
The captured PNG must never enter a deployment `forward` input or replace the
numeric vertex-Z loss.

## Canonical Recovery Chain

1. **M0 object segmentation:** raw minimap -> one object-mask signal.
2. **Deterministic cleaner:** raw minimap + frozen M0 mask -> cleaned minimap.
3. **H0 datum:** retain the passed frozen scalar checkpoint only after checking it against the corrected data manifest.
4. **W1 WDL lattice residual:** cleaned minimap + frozen H0 datum -> one structured 545-sample residual (`outer_17`, `inner_16`). One decoder may emit a 545-vector sliced into the two grids; it is not two independently-trained heads.
5. **H2 mesh-vertex residual:** only after W1 freezes; cleaned minimap + frozen deterministic W1 upsample -> one raw ADT vertex-Z residual at real lattice nodes.

Each learned stage has its own trainer, optimizer, checkpoint, three-epoch gate, and report. Start in the 3-12M parameter range only where justified; do not return to a shared 30M multi-head system.

## Retained and Retired

Retain V24's paired-lattice representation, deterministic upsample, and `ef99e715` small-trainer discipline. Retain V25's captured clean-minimap/object-mask/native-normal arrays only after source/alignment audit; they are labels, not permission for joint training.

Retire DepthAnything / DA-V2 / DPT / LoRA / PatchGAN / SiLog; the `wdl_height_33` name, target, metric, and export; the unified V25 decompiler; all target-derived forward inputs; and treating normal data as imagery.

## Phase 0 Evidence Required Before Code Changes

1. Extract raw MCVT vertices/topology from real staged-client tiles and prove the exact mapping to every existing `height_257` value. Treat unmapped dense positions as invalid, not vertices.
2. Define a canonical numeric terrain-lattice type containing vertex Z, world X/Y, source chunk, and valid-node mask; it becomes the only H2 target contract.
3. Prove the C# `outer_17` / `inner_16` equations on the real vertex contract, not on fabricated raster cells.
4. Define a paired WDL lattice type and an invalid-node mask for every 33 x 33 helper; never serialize that helper as WDL.
5. Audit each `wdl_height_33` consumer and mark it invalid before changing a trainer.
6. Audit native-normal orientation, checkerboard validity, scale, and finite-difference agreement with raw vertex Z.
7. Freeze a held-out-map and era split for M0/W1/H2 with deployment inputs and target-only arrays listed per stage.
8. Define validation rendering as one-way post-inference output (shadow input, predicted mesh, ground-truth mesh, vertex-error overlay); forbid its use in training.
9. Prove a five-tile real-client terrain-shadow capture set is deterministic
   under the recorded viewer/camera/light contract before treating it as
   guidance evidence.

## Spec Kit Routing Note

At audit time `.specify/feature.json` routes to `092-heightmap-pattern-miner`, not Spec 094 or 102. Do not run `setup-plan.ps1` until routing is corrected: it recreates the selected feature's `plan.md`. This audit and amended Spec 102 artifacts are the active planning surface.

## 2026-07-12 Implementation Evidence

The C# tensor contract now serializes the following arrays for both Alpha and
split-ADT paths:

- `mcvt_vertex_z`, `mcvt_vertex_world_x`, `mcvt_vertex_world_y`: `[16,16,145]`
- `mcvt_vertex_present`: `[16,16,145]`
- `mcvt_vertex_mask_257`: `[257,257]`, with only even/even and odd/odd nodes valid
- `mcvt_triangle_indices`: `[256,3]` per-chunk native topology
- `wdl_outer_17`, `wdl_inner_16` and their presence masks
- `mcnr_normal_xyz`, `mcnr_mask_257`, plus numeric mesh-normal agreement code

The MCNR decoder now converts disk X/Z/Y bytes to normalized public X/Y/Z.
Legitimate zero-height vertices no longer disappear as missing data. Both
`RawArraySerializer` and `NpzTileSerializer` emit the canonical arrays.

Real staged-client single-tile proof:

- `0_5_3_3368`, Azeroth `(35,55)`: NPZ SHA-256 `0693FD1BEBE9513C8F0D44B00F0E749143F8ACEC0E51C3F6409077FD797BB794`
- `3_3_5_12340`, Azeroth `(35,55)`: NPZ SHA-256 `C7CA2BD415D857F73DA3AC45C102CBDA5ABD2FF5883D940AF1A9FCC7F08AD33D`

The validation renderer gained a `TerrainShade` variant. It disables terrain
textures/alpha composition, liquids, objects, WMOs, doodads, sky, and WDL, and
records camera/light/disabled-pass metadata beside the PNG. The first proof
uncovered and fixed a pre-existing camera-axis defect: the solver used ADT X as
world X, while the terrain mesh uses ADT Y as world X and ADT X as world Y.
Before that fix captures were only the clear colour.

Five staged `0_5_3_3368` Azeroth tiles were captured twice at 256x256. Every
pair was byte-identical. SHA-256 values:

- `(24,36)` `7E2202CB44549BC833B3EB6EEA4DE99F0F70E493A4FA04F95858D71C75DD3EE1`
- `(28,57)` `EF4AA4D0EFFD29946DB38504A32CE1644994C62B828C0499F4C5FEF635858C00`
- `(30,48)` `67E32E4A66D7B888FC44AEBF5EC1667E0EAF1EEDA8675B77EDFF8AB5C518055B`
- `(31,49)` `113658ACAEB62FBC929E877EE906D98DECB06194B3781AE288006CA923FA2BE3`
- `(35,55)` `611E4F820C97857FF3EFF4F7D868B9A9A7E39E312C18AEC08C489E4FFD893A16`

The numeric upper-bound probe sampled about 36.8k projected real vertices per
tile. Luminance-to-directional-`NdotL` correlation ranged from `0.8596` to
`0.9819`, confirming a strong deterministic surface-orientation signal.
Luminance-to-absolute-Z correlation was weak (`-0.4909` to `0.0803`), which is
expected: directional shade constrains local orientation, not the arbitrary
absolute height datum. This supports using shade as a separately reported
guidance metric while keeping raw vertex Z as the primary target.

### Legacy `wdl_height_33` consumer disposition

| Consumer | Disposition |
|---|---|
| `harvester/v25/dataset.py` schema/materializer/reader | invalid historical dataset contract; do not use for W1 |
| `harvester/v25/prior.py::WdlDownsampler` | invalid stride-8 raster helper; not WDL |
| `train_v25_h1_coarse.py` | invalid historical trainer; W1 replacement required after Phase 0 |
| `train_v25_decompiler.py` | unified multi-head path remains fail-closed |
| `infer_v25_decompiler.py` | historical inference only; output is not WDL export |
| `validate_v25.py` | historical validation only; metrics are non-comparable |
| `train_v25_h0_offset.py` prohibited-input list | valid guard reference only; it does not consume the array |

### Numeric baseline gate

The identity-checked numeric store copies only raw minimap, precise mask,
liquid, MCNK flags, normals, and unrepaired height signals after proving every
build/map/tile identity against its originating V18 index. Curation rejects 27
liquid-occluded tiles, 289 near-uniform/water minimaps, 330 hard RGB/liquid
disagreements, 13 placeholders, and 3 known mismatches. M0 receives 2,037
train / 308 map-validation / 770 era rows. H2 uses a stricter liquid-evidence
gate and receives 1,859 / 206 / 770. Manifest:
`output/analysis/spec102_curated_v4/split_manifest.json`; SHA-256:
`409f4f260b8877d75e0b7ac5ccebc3c71caaf91f517c56cd6de3bcb1af477833`.
W1 has no baseline because real paired WDL arrays are absent.

## M0 and W1 Decision Runs

M0 is a 3,043,041-parameter U-Net with one output: object-mask logits. Its
`forward` input is raw RGB only. The only target is canonical
`object_precise_mask_257`, projected inside the trainer by four-corner maximum
to the 256-cell loss grid (6.97% positive pixels). A fresh three-epoch run
improved monotonically and authorized a bounded continuation. Epoch 12 reached
raw validation IoU `0.2630` and calibrated validation IoU `0.2696`; calibrated
era IoU was `0.0730`. The map gate passed but the era
gate did not. Peak VRAM was `2.15 GB`. M0 is not frozen.

W1 code emits one 545-sample numeric residual vector and has no separate
outer/inner heads. Its earlier run consumed invalid M0 materialization and is
retracted. The curated manifest has zero W1-eligible rows because the store
lacks real paired WDL arrays. W1 is blocked and H2 was not started.

Validation grids now draw only from curated terrain-visible rows. Preview
images remain inspection output, never model inputs or mesh supervision.
