# Feature Specification: WDL Prior + Lattice Detailer (V24)

**Feature Branch**: `094-wdl-prior-v24`
**Created**: 2026-07-06
**Status**: Draft
**Owner**: wow-viewer

**Input**: User description (collected, in order) — "we want to explore building a v7-like model using the newer v23 model architecture, but with one major change - we want to build a WDL prior generator, and then use the prior low-res terrain as a prior in the v23 model... [redirect] the wdl prior can be pretty well figured out, since we have terrain and wdl data that match for 99% of the valid tiles... we have all the pieces, just need to find the convergence between them, and also figure out how to create WDL priors from only Minimap priors. That's the key here. Build a LR representation, then work out the details with a model that uses the lattice structure that the LR mesh provides. This in turn, makes the final secondary model, simply a lattice smoothing model, with detailing work. [clarification on v7 and merged coverage] the biggest issue with our v7 model approach was that it required the WDL prior to do any detail refinement on, buuuuut - it worked. We need the image to WDL prior first, and then, once that's out of the way, we should merge existing WDL data that is missing tile data, with our generated tile data (OR, load up a WDL that may exist for a set of minimaps, so we can figure out the correlation between WDL tiles and minimap tiles, then apply what we learn to the rest of the missing tiles, then we have a WDL prior that we can use to feed a higher quality v7-like model for the detail work. [latest clarification on the model idea] read the data as full resolution WDL data, it's meant to be roughly 1/16th resolution of the ADT's mesh. If we have a model that can take existing minimap rgb data, combine that with existing heightmap data, and WDL data, then we can build a model that converges across all 3 to build a minimap to WDL prior model. It's simple. Since we know how to build a new WDL file from heightmap/ADT data already, this gives us an advantage that we didn't have before - We know the relationship and algorithm used to build LR (Low Resolution) data from the High Resolution terrain mesh. That's a giant part of the puzzle already solved and accessible to us. Combine that with what we don't know - we have signals for cleaning the minimap rgb via precise object masks, which in turn produces a cleaned up minimap prior that should match up quite well with the existing heightmap prior from an ADT. That gives us a good input signal -- but we may want to have the v22 precise object masks per-object (not sure this data even exists, as chatGPT munged up the spec so bad, we have stuff we probably will never use in there! -- or don't work, for that matter), so we can train a model to automatically mask out known objects from minimaps. [followup] the wowdev wiki may not be 100% accurate — our existing C# WDL reader in WowViewer.Core.IO was built from decompiled clients and is the de-facto ground truth. The spec should defer to the C# reader's actual output for the WDL grid shape."

---

## Problem Statement

V7 was a height regressor that worked only when the WDL prior was present. The WDL prior was partial coverage, so v7 worked on some maps and not others. v7 is deleted; the 57% number is folklore. The v7 *idea* — low-res prior + small residual network — is what we're keeping. The blocker was always data: WDL coverage was partial.

V24 fixes the data side first, then trains a v7-style detailer on top. We have three signals that converge on the same ground truth:

1. **Real WDL data** (where it exists in the staged client) — partial coverage, per-map `.wdl` files.
2. **Heightmap data** (V18 `height_257`) — full resolution, always present on real V18 tiles.
3. **Minimap RGB** (V18) — always present, but noisy because of object roofs and compositing artifacts.

The clean input signal is: **cleaned minimap (with object roofs masked) + heightmap prior (the LR signal we can already derive) + real WDL (where available)** → a convergence model that learns the minimap→WDL prior relationship. The object-mask cleaning is the first-class concern: the minimap is not a pure terrain signal until objects are removed.

The synthetic-WDL builder is **not a research problem** — it's a deterministic transform from V18 `height_257` to the WDL lattice. The user has existing C# code in `WowViewer` (the "click on map to spawn" visualization) that already does this. V24 wraps that path; the C# side is not modified.

V24 is a two-stage, height-only, small-model lane on V18 substrate:

- **Stage 0 — Merged WDL prior coverage.** Build a per-tile WDL prior for every V18 tile. Where a real staged-client WDL exists and is consistent with V18 `height_257`, use the real WDL. Where it doesn't exist or disagrees, generate a synthetic WDL from V18 `height_257` via the existing terrain→WDL path. Audit-empty tiles get a learned-fill placeholder. The deliverable is a complete WDL prior for every tile.
- **Stage A — Minimap → WDL prior correlation.** A small model that takes a V18 minimap tile (cleaned via the V18 `object_precise_mask`, plus the synthetic-WDL-as-pseudo-input from V18 heightmap, plus alpha/normal/mcnr channels) and predicts the WDL prior. The model learns the minimap-to-WDL correlation from real coverage where it exists and from synthetic coverage everywhere else.
- **Stage B — Lattice detailer (v7-style).** A small model that consumes the upsampled WDL prior + the cleaned minimap and predicts a residual over the prior at full 257×257 resolution.

V24 is not a production cutover. It is a bounded research slice. It must be cheap, deterministic, and small. No DepthAnything, no DPT, no LoRA, no GPCT, no CAI, no RunPod 24 GB envelope. Total ≤ 3M params. Runs locally on a 6 GB consumer GPU.

**WDL grid shape — defer to the C# reader.** Per [`gillijimproject_refactor/reference_data/wowdev.wiki/WDL_v18.md`](gillijimproject_refactor/reference_data/wowdev.wiki/WDL_v18.md:131), the documented 3.3.5a MARE layout is "17*17 + 16*16 = 545 signed 16-bit integers" per tile (17×17 outer + 16×16 inner). The user has noted that the wowdev wiki may not be 100% accurate and that the existing C# WDL reader in `WowViewer.Core.IO` (built from decompiled clients across multiple eras) is the de-facto ground truth. The spec **uses the C# reader's actual output as the WDL grid shape** — whatever the C# reader returns per MARE (which may be 17×17+16×16, or 16×16, or some other layout the decompilers revealed), V24 consumes that. The spec does not invent a new WDL grid shape. If the C# reader returns something other than 17×17+16×16, the spec accommodates the C# reader's actual output. The spec also acknowledges that Alpha-era WDLs (used in `0_5_3_3368`) may have a different layout than LK-era WDLs (`3_3_5_12340`); the C# reader handles this and V24 reads whatever the C# reader returns.

WDL heights are signed int16 on the same scale as the regular height maps (per WDL_v18.md) — no scale conversion needed. Liquid level is not in the WDL (best guess is 0/sea level — minimap water from a WDL shows below-sea-level as blue; this is a known client-side artifact and is not V24's problem to fix). MAHO is a per-MARE 16-uint16 hole bitmask; V24 reads it if the C# reader exposes it and stores it as part of the WDL prior, but does not train a hole-aware Stage A in this spec.

**Minimap cleaning is a first-class input concern.** The V18 minimap contains object roofs, shadows, and compositing artifacts. V18's `object_precise_mask` is the tile-level mask we have today (a bool 257×257 indicating object pixels). V22's spec advertises per-object precise masks, but the user has flagged that this data may be broken or unused ("chatGPT munged up the spec so bad, we have stuff we probably will never use in there! -- or don't work, for that matter"). V24 does NOT depend on V22 per-object mask data. V24 uses V18's tile-level `object_precise_mask` to clean the minimap input. If/when V22 per-object mask data is verified to work, a future spec can promote it to per-object cleaning.

---

## Relationship To Existing Specs

- **Re-uses**: V18 (Spec 001) as the substrate. V18's `height_257`, `minimap_rgb`, `alpha_256`, `mcnr_mask_257`, `normal_xyz`, `object_precise_mask`, `liquid_mask_256` are the inputs to V24.
- **Re-uses**: The terrain→WDL path used by `WoWViewer`'s "click on map to spawn" visualization (C# under `WowViewer.Core.IO`). V24 wraps it; C# is not modified.
- **Re-uses**: The C# WDL reader in `WowViewer.Core.IO` (built from decompiled clients). V24 wraps it; C# is not modified. The C# reader is the source of truth for the WDL grid shape.
- **Re-uses**: V18's tile-level `object_precise_mask` as the minimap-cleaning signal. V24 does NOT depend on V22 per-object mask data.
- **Does not amend**: Spec 089 (V23). V24 is a separate lane. V23 stays as the trunk height model until V24 proves it should replace V23.
- **Does not touch**: V22 (Spec 088). V22's brokenness is out of scope here.
- **Reference only**: `gillijimproject_refactor/reference_data/wowdev.wiki/WDL_v18.md` (read as documentation, not as ground truth).
- **Pauses nothing**: Spec 068 (fractal-aware height loss) remains paused unless reopened.

---

## Out Of Scope (Explicit)

- DepthAnything / V23 / DPT / LoRA / GPCT / CAI / bitsandbytes / RunPod 24 GB envelope. None of that belongs in V24.
- Multi-task terrain models (RULE 7). V24 is two height-only stages.
- V22 / V22 audit. V18 is the substrate. The V22 audit is a separate spec.
- Editing `gillijimproject_refactor` (RULE 1).
- Editing the V23 trainer / V23 inference / V23 RunPod bundle. V24 is independent.
- Editing any C# WDL reader. The C# reader is the source of truth; V24 wraps it.
- The v7 model. It is deleted. V24 has no v7 comparison target.
- Producing `.wdl` files for the game client. V24 produces training-time prior tensors in the V24 dataset contract, not game-client WDL binaries.
- WebGL / Vulkan / Unreal integration.
- V22 per-object precise mask data. The user has flagged this as suspect. V24 uses V18's tile-level `object_precise_mask` and does not depend on V22 per-object data.
- MAHO-aware priors. MAHO is a 16-uint16 hole bitmask per MARE; V24 reads it if the C# reader exposes it and stores it as part of the WDL prior, but does not train a hole-aware Stage A. Holes are a separate lane (Spec 077 / future).
- Inventing a new WDL grid shape. The C# reader's actual output is the shape.

---

## User Scenarios & Testing

### User Story 1 - Merged WDL Prior Coverage (Real + Synthetic + Learned-Fill) (Priority: P1)

As a V24 owner, I can run a script that takes a V18 Zarr store and a staged-client root and produces, for every V18 tile, a per-tile WDL prior that uses the real staged-client WDL (read via the C# WDL reader) where it exists and is consistent with V18 `height_257`, the synthetic WDL (built from V18 `height_257` via the existing terrain→WDL path) where the real WDL is missing or disagrees, and a learned-fill placeholder on audit-empty tiles. The prior's grid shape is whatever the C# reader returns (likely 17×17+16×16 = 545-point MARE lattice on 3.3.5a; the C# reader handles era differences).

**Why this priority**: Without a complete WDL prior for every tile, neither Stage A nor Stage B has a target / input. The user explicitly said: "we should merge existing WDL data that is missing tile data, with our generated tile data... then we have a WDL prior that we can use to feed a higher quality v7-like model for the detail work."

**Acceptance Scenarios**:
1. **Given** a V18 Zarr store with real `height_257` and a staged-client root with real `.wdl` files, **When** `build_wdl_prior.py build --v18-store <path> --staged-client <path> --output <v24.zarr>` runs, **Then** every V18 tile gets: `wdl_prior` (the C# reader's output, stored as float32 with shape matching what the C# reader returned per MARE), `wdl_prior_source` (per-cell uint8: `0=real`, `1=synthetic`, `2=learned-fill`), `wdl_prior_confidence` (per-cell float32 in [0, 1]), and (if MAHO is exposed by the C# reader) `wdl_prior_holes` (per-cell bool).
2. **Given** a tile where the staged client has a real WDL MARE and the real WDL heights match V18 `height_257` at the corresponding grid points (within a configurable absolute threshold, default 1.0 world unit), **When** the builder runs, **Then** `wdl_prior` is the real WDL MARE grid, `wdl_prior_source` is `0` (real) for every cell, and `wdl_prior_confidence` is 1.0 for every cell.
3. **Given** a tile where the real WDL disagrees with V18 `height_257` (the rare ~1% case), **When** the builder runs, **Then** the disagreeing cells use the synthetic WDL (synthetic wins for training), `wdl_prior_source` is `1` (synthetic) for those cells, and `wdl_prior_confidence` is < 0.5 for the affected cells. A `wdl_prior_disagrees_with_real=True` flag is set on the tile.
4. **Given** a tile with no real WDL in the staged client, **When** the builder runs, **Then** `wdl_prior` is built from V18 `height_257` via the synthetic-WDL builder, source flags are `1` (synthetic) for every cell, confidence is 0.7 for every cell.
5. **Given** a V18 tile with audit-empty `height_257`, **When** the builder runs, **Then** `wdl_prior` is the per-tile mean height (a flat grid), source flags are `2` (learned-fill) for every cell, confidence is 0.0 for every cell. The tile is trainable only via Stage A's prediction of the prior.
6. **Given** the builder output, **When** `inspect_v24_dataset.py` runs, **Then** it reports coverage stats: `real_cell_ratio`, `synthetic_cell_ratio`, `learned_fill_cell_ratio` across the V24 store. The combined real+synthetic coverage MUST be ≥ 95% of all non-empty cells in the WDL prior grid (per what the C# reader returned).

### User Story 2 - Minimap Cleaning Input (Priority: P1)

As a V24 owner, I can run a script that takes a V18 minimap tile and V18's `object_precise_mask` and produces a "cleaned" minimap where object pixels are masked or inpainted. The cleaned minimap is a first-class input to Stage A. The cleaning uses V18's tile-level mask; it does not depend on V22 per-object mask data (which the user has flagged as suspect).

**Why this priority**: The user explicitly said the minimap is noisy and the cleaning signal is the object mask. Without cleaning, Stage A trains on a noisy input that includes object roofs.

**Acceptance Scenarios**:
1. **Given** a V18 minimap tile and V18's `object_precise_mask` (257×257 bool), **When** `clean_minimap.py --minimap <npz> --object-mask <npz> --output <cleaned.npz>` runs, **Then** the output is a 257×257×3 float32 RGB image where object pixels are replaced by the median of the surrounding non-object 8-connected neighbourhood (or, if no non-object neighbour exists, by the global mean colour).
2. **Given** a tile where `object_precise_mask` is all-False, **When** the cleaner runs, **Then** the cleaned minimap is identical to the input.
3. **Given** a tile where `object_precise_mask` is all-True, **When** the cleaner runs, **Then** the cleaned minimap is the global mean colour, and the script emits `cleaned_minimap_unavailable=True`.
4. **Given** the cleaned minimap, **When** Stage A consumes it as input, **Then** the loss gate uses `object_precise_mask` to skip object pixels (matching the input cleaning).

### User Story 3 - Stage A: Minimap → WDL Prior Correlation (Priority: P1)

As a V24 owner, I can run `train_v24_stage_a.py` to train a small model that takes `[cleaned_minimap, alpha_256, normal_xyz, mcnr_mask_257, downsampled_synthetic_wdl]` and predicts the merged WDL prior. The model is a small U-Net (≤ 1M params). The training target is `wdl_prior`. Sample weight is `wdl_prior_confidence`. Sample selection is stratified by `wdl_prior_source`: real cells weighted 1.0, synthetic cells weighted 0.7, learned-fill cells excluded from loss. Stage A's *prediction* is the source of truth for learned-fill cells.

**Why this priority**: Stage A is the prior generator. It learns the minimap-to-WDL correlation from real coverage where it exists and from synthetic coverage everywhere else. The downsampled synthetic WDL is a "cheat" input — the heightmap prior derived from V18 — that gives the model a strong starting point.

**Acceptance Scenarios**:
1. **Given** the V24 dataset (with cleaned minimap and merged WDL prior), **When** `train_v24_stage_a.py` runs, **Then** Stage A converges to a per-tile L1 on `wdl_prior` that is **lower than the V18 `block_reduce(height_257)` baseline** on real-WDL-available tiles (i.e. the model is doing real work, not just memorizing the mean).
2. **Given** a trained Stage A checkpoint, **When** `infer_v24_stage_a.py` runs on a single V18 minimap tile, **Then** it produces a WDL prior in well under 1 second on a 6 GB consumer GPU.
3. **Given** a held-out validation set, **When** Stage A is evaluated separately on real-WDL-available tiles vs synthetic-only tiles, **Then** the L1 on real-WDL tiles is lower than on synthetic-only tiles (proof the model learned the WDL correlation, not just the synthetic pattern).
4. **Given** a tile with no real WDL and no V18 height (audit-empty), **When** Stage A infers, **Then** it returns the per-tile mean prior and emits `prior_unavailable=True`.
5. **Given** a minimap tile that V18 has but no other model has, **When** Stage A runs, **Then** it produces a usable prior with no other inputs (this is the deployment case).

### User Story 4 - Stage B: Lattice Detailer (Prior + Minimap → Full 257×257 Residual) (Priority: P1)

As a V24 owner, I can run `train_v24_stage_b.py` to train a small model that takes `[upsampled Stage A prior, cleaned_minimap, alpha, normal, mcnr_mask, object_mask]` and predicts a 257×257 residual over the prior. The model is a small conv-deconv (≤ 2M params). The training target is `height_257 - bilinear_upsample(stage_a_prior, 257)`.

**Why this priority**: Stage A alone is too coarse (whatever the C# reader returns per MARE, e.g. ~545 grid points vs 257×257 = 66049). Stage B is the lattice-smoothing / detailing pass. This is the v7 two-stage idea, with a now-complete prior instead of a partial-coverage one.

**Acceptance Scenarios**:
1. **Given** the V24 dataset, **When** `train_v24_stage_b.py` runs, **Then** Stage B converges to a per-tile L1 on `height_257` that is **lower than the V18 `block_reduce(height_257) + bilinear_upsample` baseline** on the same validation set.
2. **Given** a trained Stage B checkpoint, **When** `infer_v24_stage_b.py` runs, **Then** the final `height_257` prediction is `bilinear_upsample(stage_a_prior, 257) + predicted_residual`, and the full pipeline (Stage A + Stage B) runs in well under 3 seconds per tile on a 6 GB consumer GPU.
3. **Given** a tile with `liquid_mask_256 > 0`, **When** the residual is supervised, **Then** the loss is gated to non-liquid pixels (the WDL has no liquid level).
4. **Given** a tile with `object_precise_mask > 0`, **When** the residual is supervised, **Then** the loss is gated away from object pixels (V24 predicts terrain, not objects).
5. **Given** a tile where the C# WDL reader exposed MAHO holes, **When** the residual is supervised, **Then** the loss is gated away from MAHO hole cells.

### User Story 5 - Self-Consistency + Coverage Validation (Priority: P2)

As a V24 owner, I can run `validate_v24.py` which reports coverage stats, training curves, and a self-consistency check on the V24 store + trained checkpoints. There is no v7 baseline. The validation is: (a) merged-coverage stats, (b) Stage A's convergence relative to the `block_reduce` baseline, (c) Stage A's real-vs-synthetic L1 gap, (d) Stage B's convergence relative to the upsampled-prior baseline, (e) determinism check, (f) VRAM and wall-time measurements on a 6 GB consumer GPU.

**Why this priority**: Without a v7 baseline, success is defined as "V24 is measurably better than the trivial `block_reduce` baseline on real data, with the WDL correlation learned from real coverage where it exists."

**Acceptance Scenarios**:
1. **Given** the V24 store + Stage A + Stage B checkpoints, **When** `validate_v24.py` runs, **Then** it emits a JSON report at `output/v24_validation/<run_id>/report.json` and a side-by-side preview PNG.
2. **Given** the report, **When** reviewed, **Then** Stage A's real-WDL L1 < synthetic-only L1 < `block_reduce` baseline L1.
3. **Given** the report, **When** reviewed, **Then** Stage B's final L1 < upsampled-prior L1 < `block_reduce + bilinear_upsample` baseline L1.
4. **Given** two `infer_v24_stage_b.py` runs with different seeds, **When** outputs are compared, **Then** they are bit-identical (`torch.allclose(atol=0, rtol=0`).

### Edge Cases

- A V18 tile where every required array is zero-filled (audit-empty). Stage A returns flat prior + `prior_unavailable=True`. Stage B does not run. The tile is excluded from training.
- A V18 tile where `height_257` is real but `minimap_rgb` is missing. Stage A and Stage B both skip the tile.
- A tile where the staged client has no WDL for that map at all. Synthetic WDL is the only source. Real-WDL cross-check is bypassed.
- A WDL cell that is entirely liquid in V18. The synthetic WDL uses nearest-non-liquid-neighbour; confidence is bumped. Stage A's loss is gated away from this cell.
- A tile where the real WDL MARE disagrees with V18 `height_257` on a majority of cells (>50%). The tile is flagged `wdl_prior_disagrees_with_real=True` but still trainable (synthetic wins for the disagreeing cells).
- A real WDL MARE that has MAHO holes. MAHO is read and stored as `wdl_prior_holes` (per-cell bool, derived from the 16-uint16 MAHO bitmask if the C# reader exposes it; all-False otherwise). Stage B's loss is gated away from MAHO hole cells. Stage A does not train on MAHO.
- The C# WDL reader returns a different grid shape than 17×17+16×16 (e.g. flat 16×16, or some other layout the decompilers revealed). V24 accommodates whatever the C# reader returns — the merged-WDL prior grid shape is whatever the C# reader produced.
- Alpha-era WDLs (`0_5_3_3368`) have a different layout than LK-era WDLs (`3_3_5_12340`). The C# reader handles this; V24 reads whatever the C# reader returns per build.
- A V24 inference on a 4 GB consumer GPU. Should fit (≤ 3M params + small activations). If it OOMs, the failure is reported cleanly; no silent fp32 fallback.
- A V24 training run that loses determinism. The trainer must be deterministic under `torch.manual_seed + model.eval + use_deterministic_algorithms(True)`.

---

## Requirements

### Phase 1: C# WDL Reader Wrapper (Foundation)

- **FR-001**: A new module `wow-viewer/data-harvester/src/harvester/v24/wdl_reader.py` MUST provide `read_wdl_mare(wdl_path: Path, tile_x: int, tile_y: int) -> tuple[np.ndarray, np.ndarray | None] | None` that reads the MARE height grid (and MAHO bitmask if exposed) for a specific (tile_x, tile_y) from a real staged-client `.wdl` file, using the C# WDL reader in `WowViewer.Core.IO` (subprocess or in-process bridge). The C# side MUST NOT be modified. The function returns whatever the C# reader returns (per `WDL_v18.md`, likely a (17×17, 16×16) tuple on 3.3.5a, but the actual shape is the C# reader's output). Return `None` if the WDL has no entry for that tile.
- **FR-002**: A small CLI shim MUST exist (under `wow-viewer/tools/wdl-read/` or similar) that takes a `.wdl` path, tile_x, tile_y as args and emits the MARE grid + MAHO bitmask as JSON or NPZ. The shim is the canonical Python-callable entry point to the C# WDL reader. The shim is built once and is not modified by V24.

### Phase 2: Synthetic WDL Builder

- **FR-003**: A new module `wow-viewer/data-harvester/src/harvester/v24/synth_wdl.py` MUST contain the synthetic-WDL builder. Function signature: `def build_synth_wdl(height_257: np.ndarray, liquid_mask_256: np.ndarray | None = None) -> np.ndarray` returning a single float32 array whose shape matches the C# WDL reader's per-MARE output. (Per `WDL_v18.md` this is likely (17×17, 16×16) flattened to a 545-vector or kept as a tuple; the spec accommodates whatever the C# reader returns.) The function MUST emit the same grid layout as the C# reader, with liquid cells replaced by nearest-non-liquid-neighbour.
- **FR-004**: The synthetic-WDL builder MUST be a thin wrapper around the existing terrain→WDL path used by `WoWViewer`'s "click on map to spawn" visualization. That path lives in C# under `WowViewer.Core.IO`. V24 calls it via the same Python shim pattern as FR-001 / FR-002. **The C# side MUST NOT be modified.** If the existing C# path is missing, Phase 2 is blocked and a separate spec is filed to add it.

### Phase 3: Merged WDL Prior Coverage (Stage 0)

- **FR-005**: A new module `wow-viewer/data-harvester/src/harvester/v24/merged_wdl_prior.py` MUST contain the merged-coverage builder. Function signature: `def build_merged_wdl_prior(height_257: np.ndarray, real_wdl: np.ndarray | None, real_wdl_available: bool, liquid_mask_256: np.ndarray | None = None, disagree_threshold: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]` returning `(prior, source, confidence, holes)`. The merge rules:
  - If `real_wdl_available` and the real WDL value is within `disagree_threshold` of the synthetic-WDL value at a cell, use the real WDL (`source=0`, `confidence=1.0`).
  - Otherwise use the synthetic WDL (`source=1`, `confidence=0.7`). If the synthetic disagrees with the real WDL on a real-WDL-available cell, drop `confidence` to 0.4.
  - Audit-empty V18 tiles get `source=2` (learned-fill), `confidence=0.0`, and `prior = global mean height` for every cell.
  - `holes` is `None` if the C# reader didn't expose MAHO; otherwise it's the per-cell bool mask.
- **FR-006**: A new script `wow-viewer/data-harvester/scripts/build_wdl_prior.py` MUST accept `build --v18-store <path> --staged-client <path> --output <v24.zarr>` and `infer --minimap <npz> --height <npz> --output <prior.npz>` subcommands.
- **FR-007**: The V24 Zarr store layout MUST be a superset of V18 with new top-level arrays for the merged prior: `wdl_prior` (per-tile float32, shape = C# reader's output per MARE), `wdl_prior_source` (per-tile uint8), `wdl_prior_confidence` (per-tile float32 in [0, 1]), and `wdl_prior_holes` (per-tile bool, present only if the C# reader exposed MAHO). V18 arrays MUST be present and unchanged.
- **FR-008**: A bounded real-data build MUST run on `0_5_3_3368` Azeroth and `3_3_5_12340` Northrend with `--limit 50` tiles each. The build MUST produce a V24 store with the prior arrays populated for every non-empty tile, and the combined real+synthetic coverage MUST be ≥ 95% of all non-empty cells in the WDL prior grid.

### Phase 4: Minimap Cleaning Input

- **FR-009**: A new script `wow-viewer/data-harvester/scripts/clean_minimap.py` MUST accept `--minimap <npz> --object-mask <npz> --output <cleaned.npz>` and emit a 257×257×3 float32 cleaned RGB image where object pixels are replaced by the median of the surrounding non-object 8-connected neighbourhood (or, if no non-object neighbour exists, by the global mean colour).
- **FR-010**: A new module `wow-viewer/data-harvester/src/harvester/v24/clean_minimap.py` MUST contain the cleaning function. The cleaner is pure NumPy; no model, no inference. The cleaner uses V18's tile-level `object_precise_mask` (257×257 bool). It does NOT use V22 per-object mask data.

### Phase 5: Stage A (Minimap → WDL Prior Correlation)

- **FR-011**: A new module `wow-viewer/data-harvester/src/harvester/v24/stage_a.py` MUST contain the Stage A model (≤ 1M params) and dataset adapter. The model MUST be a small U-Net. Inputs: `[cleaned_minimap (down-sampled to WDL prior grid size), alpha_256 (down-sampled), normal_xyz (down-sampled), mcnr_mask_257 (down-sampled), downsampled_synthetic_wdl]`. Output: the WDL prior (shape = C# reader's output per MARE). No DA-V2, no DPT, no LoRA, no transformer blocks.
- **FR-012**: A new script `wow-viewer/data-harvester/scripts/train_v24_stage_a.py` MUST consume the V24 store and train Stage A. Target: `wdl_prior`. Sample weight: `wdl_prior_confidence`. **Sample selection rule:** training samples are weighted by `wdl_prior_confidence` and stratified by `wdl_prior_source` — real cells (source=0) weighted 1.0, synthetic cells (source=1) weighted 0.7, learned-fill cells (source=2) excluded from loss. This is how Stage A learns the minimap-to-WDL correlation from real coverage where it exists and from synthetic coverage everywhere else.
- **FR-013**: A new script `wow-viewer/data-harvester/scripts/infer_v24_stage_a.py` MUST run Stage A on a single tile or a tile grid and emit the WDL prior. The script MUST emit `prior_unavailable=True` for audit-empty tiles.
- **FR-014**: The Stage A model MUST be deterministic under `torch.manual_seed + model.eval + use_deterministic_algorithms(True)`. Same input + same seed = bit-identical output.
- **FR-015**: The Stage A model MUST fit on a 6 GB consumer GPU at fp16 with peak VRAM < 2 GB and inference wall-time < 1 second per tile.

### Phase 6: Stage B (Lattice Detailer)

- **FR-016**: A new module `wow-viewer/data-harvester/src/harvester/v24/stage_b.py` MUST contain the Stage B model (≤ 2M params) and dataset adapter. The model is a small conv-deconv on 257×257 inputs. Input channels: `[bilinear_upsample(stage_a_prior, 257), cleaned_minimap, alpha_256, normal_xyz, mcnr_mask_257, object_precise_mask]`. The Stage A prior is upsampled from the WDL prior grid to 257×257 via bilinear interpolation. The target is `height_257 - bilinear_upsample(stage_a_prior, 257)`.
- **FR-017**: A new script `wow-viewer/data-harvester/scripts/train_v24_stage_b.py` MUST consume the V24 store and train Stage B. Loss is gated to non-liquid, non-object, non-MAHO-hole pixels. Mixed precision: fp16. Optimizer: AdamW. No LoRA, no GPCT, no CAI. Determinism is required (FR-019).
- **FR-018**: A new script `wow-viewer/data-harvester/scripts/infer_v24_stage_b.py` MUST run the full V24 inference pipeline (Stage A + Stage B) and emit final `height_257`. Final = `bilinear_upsample(stage_a_prior, 257) + predicted_residual`. The script MUST be a single entry point so downstream callers don't have to chain Stage A and Stage B themselves.
- **FR-019**: The Stage B model MUST be deterministic under `torch.manual_seed + model.eval + use_deterministic_algorithms(True)`. The full pipeline (Stage A + Stage B) MUST produce bit-identical output across two runs with different seeds.
- **FR-020**: The Stage B model MUST fit on a 6 GB consumer GPU at fp16 with peak VRAM < 4 GB and inference wall-time < 3 seconds per tile.

### Phase 7: Validation

- **FR-021**: A bounded real-data build MUST run on `0_5_3_3368` Azeroth and `3_3_5_12340` Northrend with `--limit 50` tiles each. The V24 store MUST have the prior arrays populated for every non-empty tile.
- **FR-022**: A bounded Stage A training run MUST run for at least 50 epochs on the bounded 50-tile V24 set. The validation report MUST show: Stage A's L1 on real-WDL cells < Stage A's L1 on synthetic-only cells < `block_reduce` baseline L1.
- **FR-023**: A bounded Stage B training run MUST run for at least 50 epochs on the bounded 50-tile V24 set. The validation report MUST show: Stage B's final L1 < upsampled-prior L1 < `block_reduce + bilinear_upsample` baseline L1.
- **FR-024**: A determinism check MUST run: two `infer_v24_stage_b.py` runs with different seeds, same input, same checkpoints, output is bit-identical (`torch.allclose(atol=0, rtol=0`).
- **FR-025**: A `validate_v24.py` script MUST emit a JSON report at `output/v24_validation/<run_id>/report.json` with coverage stats, training curves, real-vs-synthetic L1 gap, and a side-by-side preview PNG.

### What This Spec Does NOT Do (Negative Space)

- Does NOT introduce DepthAnything / DA-V2 / DPT / LoRA / GPCT / CAI / PagedAdamW8bit / RunPod 24 GB envelope. V24 is small and runs locally.
- Does NOT touch V22, V23, or any V23 surface.
- Does NOT edit `gillijimproject_refactor` (RULE 1).
- Does NOT touch the V18 build path (Spec 001). V18 is the substrate; we read from it, we do not modify it.
- Does NOT replace V23 in production. V24 is a research/exploration lane.
- Does NOT introduce a multi-task terrain model (RULE 7). V24 is two height-only stages.
- Does NOT ship a final "V24 is better than V23" or "V24 is better than v7" claim. There is no v7 model. Success is "V24 beats the `block_reduce` baseline by a documented margin and learns the WDL correlation from real coverage."
- Does NOT modify any C# WDL reader. C# is the source of truth for the MARE format and grid shape.
- Does NOT depend on V22 per-object mask data. V24 uses V18's tile-level `object_precise_mask`.
- Does NOT invent a new WDL grid shape. The C# reader's actual output is the shape.

---

## Success Criteria

- **SC-001**: `build_wdl_prior.py build` produces a V24 store with non-zero `wdl_prior` for every non-empty V18 tile. The combined real+synthetic coverage (`wdl_prior_source ∈ {0, 1}`) MUST be ≥ 95% of all non-empty cells in the WDL prior grid. The "99% match" user claim is bounded by `wdl_prior_confidence ≥ 0.9` on ≥ 80% of real-WDL-available cells.
- **SC-002**: `train_v24_stage_a.py` produces a Stage A model with per-tile L1 on real-WDL cells strictly less than its L1 on synthetic-only cells, AND strictly less than the V18 `block_reduce(height_257)` baseline. The model is ≤ 1M trainable params.
- **SC-003**: `train_v24_stage_b.py` produces a Stage B model with per-tile L1 on `height_257` strictly less than the upsampled-prior L1, AND strictly less than the V18 `block_reduce(height_257) + bilinear_upsample` baseline. The model is ≤ 2M trainable params.
- **SC-004**: The full V24 pipeline (Stage A + Stage B) produces bit-identical output across two runs with different seeds (`torch.allclose(atol=0, rtol=0`).
- **SC-005**: The full V24 pipeline fits on a 6 GB consumer GPU at fp16: peak VRAM < 4 GB, inference wall-time < 3 seconds per tile.
- **SC-006**: The `validate_v24.py` report at `output/v24_validation/<run_id>/report.json` shows SC-001 / SC-002 / SC-003 / SC-004 / SC-005 all pass on a 50-tile V24 validation set.
- **SC-007**: The V24 store is built directly from V18 substrate + staged-client real WDLs (read via the C# WDL reader wrapper). No V22 dependency. The V22 audit question is out of scope for this spec.
- **SC-008**: All V24 artifacts (V24 store, Stage A checkpoint, Stage B checkpoint, validation report) are committed under `wow-viewer/output/v24_validation/<run_id>/` and a summary doc lands at `wow-viewer/docs/architecture/v24-validation-2026-07-XX.md`.

---

## Architecture Sketch (For Review; Not Implementation Detail)

```
[ existing V18 Zarr store ] (Spec 001 substrate)
[ staged-client real .wdl files ] (per-map, partial coverage)
            |
            |  [FR-001/FR-002] C# WDL reader wrapper (shim calls WowViewer.Core.IO)
            |     returns the WDL grid shape the C# reader actually produces
            |     (per WDL_v18.md, likely 17x17+16x16 on 3.3.5a, but C# is ground truth)
            |
            |  build_wdl_prior.py build
            |    for each V18 tile:
            |      1. read real WDL MARE from staged client via C# reader shim
            |      2. build_synth_wdl from V18 height_257 via the C# terrain->WDL
            |         path (the "click on map to spawn" visualization)
            |      3. merge: real where it exists & agrees, synthetic where it
            |         doesn't, learned-fill placeholder where V18 is empty
            |      4. emit wdl_prior, wdl_prior_source, wdl_prior_confidence,
            |         wdl_prior_holes (if C# reader exposed MAHO)
            v
[ V24 Zarr store = V18 arrays + wdl_prior + wdl_prior_source + wdl_prior_confidence (+ wdl_prior_holes) ]
            |
            |  [FR-009/FR-010] clean_minimap.py
            |    cleans the V18 minimap using V18's object_precise_mask
            |    (NOT V22 per-object mask data — that's suspect)
            v
[ Cleaned minimap: 257x257x3 float32, object roofs removed ]
            |
            |  train_v24_stage_a.py
            |    learns: [cleaned_minimap, alpha, normal, mcnr, synthetic_wdl] -> wdl_prior
            |    weighted by wdl_prior_confidence, stratified by wdl_prior_source
            v
[ Stage A model: small U-Net, <= 1M params ]
            |  minimap_rgb (cleaned, down-sampled) -> wdl_prior
            |
            |  infer_v24_stage_a.py (or build_wdl_prior.py infer for single tiles)
            v
[ Stage A prior: same shape as C# reader's output, fills in the learned-fill cells ]
            |
            |  train_v24_stage_b.py
            v
[ Stage B model: small conv-deconv on 257x257, <= 2M params ]
            |  [upsampled prior, cleaned_minimap, alpha, normal, mcnr, object] -> residual
            |
            |  infer_v24_stage_b.py
            v
[ Final height_257 = bilinear_upsample(stage_a_prior, 257) + predicted_residual ]
```

Total trainable params: ≤ 3M. Total inference VRAM at fp16: < 4 GB. Total inference wall-time: < 3 s/tile on a 6 GB consumer GPU. No RunPod. No DA-V2. No LoRA. No GPCT. No CAI. Just two small models, a clean V18 substrate, and the existing C# WDL reader wrapped via a Python shim.

---

## Key Entities

- **V24 Zarr store**: A V18 store extended with the merged-WDL-prior arrays. Lives at `wow-viewer/output/datasets/v24/<build>.zarr/`. V18 arrays are present and unchanged; the new arrays are additive.
- **Merged WDL prior (`wdl_prior`)**: per-tile float32, shape = whatever the C# WDL reader returns per MARE. The Stage A training target and the Stage B LR input.
- **Prior source map (`wdl_prior_source`)**: per-cell uint8. `0=real`, `1=synthetic`, `2=learned-fill`. Used by Stage A to stratify training samples.
- **Prior confidence map (`wdl_prior_confidence`)**: per-cell float32 in [0, 1]. The Stage A per-cell sample weight.
- **Prior holes map (`wdl_prior_holes`)**: per-cell bool, derived from MAHO if the C# reader exposed it; absent otherwise. Used by Stage B to gate loss away from MAHO hole cells.
- **C# WDL reader wrapper (`wdl_reader.py`)**: Python module that wraps the existing C# WDL reader in `WowViewer.Core.IO` (subprocess or in-process bridge). Returns the per-MARE grid + MAHO bitmask. C# is not modified.
- **C# terrain→WDL wrapper (`synth_wdl.py`)**: Python module that wraps the existing C# terrain→WDL path used by `WoWViewer` for "click on map to spawn". Generates the WDL grid from V18 `height_257` with liquid exclusion.
- **Merged-WDL prior builder (`merged_wdl_prior.py`)**: Combines real-WDL reads with the synthetic-WDL output into the merged prior. Merge rules in FR-005.
- **Minimap cleaner (`clean_minimap.py`)**: Pure NumPy function that masks object pixels via V18's tile-level `object_precise_mask`. No model, no inference.
- **Stage A model**: Small U-Net (≤ 1M params). Maps `[cleaned_minimap, alpha, normal, mcnr, synthetic_wdl]` to the WDL prior.
- **Stage A prior**: per-tile float32, same shape as the C# reader's output. The Stage B input (after bilinear upsample to 257×257).
- **Stage B model**: Small conv-deconv (≤ 2M params) on 257×257 inputs. Maps `[upsampled Stage A prior, cleaned_minimap, alpha, normal, mcnr, object_mask]` to a full-resolution residual over the prior.
- **V24 final height**: 257×257 float32 per-tile `bilinear_upsample(stage_a_prior, 257) + predicted_residual`. The V24 inference output.

---

## Risks

- **Risk 1 (high):** The C# WDL reader in `WowViewer.Core.IO` may not be exposed in a way that's callable from Python. If it isn't, FR-001 blocks Phase 1 and a separate spec is filed to add a Python-callable shim. The spec does not re-implement the WDL reader in Python.
- **Risk 2 (high):** The terrain→WDL path used by `WoWViewer`'s "click on map to spawn" visualization may not exist, or may not be callable from Python. If it doesn't, FR-004 blocks Phase 2 and a separate spec is filed to add it. The spec does not invent a new terrain→WDL algorithm.
- **Risk 3 (medium):** The wowdev wiki WDL layout (17×17+16×16 = 545 per MARE) may not match what the C# reader actually returns (the user has flagged this). The spec accommodates the C# reader's actual output. If the C# reader returns a different shape, all the "17×17" and "16×16" mentions in this spec become "whatever the C# reader returns."
- **Risk 4 (medium):** The synthetic WDL builder may not match real WDLs in the 99% case. If it doesn't, the merged-coverage grid falls back to synthetic more often than expected, and Stage A's training target is noisier on the real-WDL cells. Mitigation: SC-001's `wdl_prior_confidence ≥ 0.9` on ≥ 80% of real-WDL-available cells is the bound. The merged-coverage design means a noisy synthetic is bounded — real-WDL cells still train with confidence 1.0, and synthetic-only cells train with confidence 0.7.
- **Risk 5 (medium):** V18's tile-level `object_precise_mask` is coarser than per-object masking. Some object pixels will remain in the cleaned minimap. Mitigation: the cleaner is a simple first step. Per-object masking is a future lane that requires V22 per-object mask data (which the user has flagged as suspect — so this future lane is also blocked on V22 data verification).
- **Risk 6 (medium):** The `block_reduce(height_257)` baseline isn't a perfect WDL-grid-shaped baseline. The comparison in SC-002 / SC-003 is approximate, not exact. The spec acknowledges this: the baseline is the trivial "no learning" answer, and any model that beats it is doing real work.
- **Risk 7 (low):** MAHO is optional in WDL files. The C# reader may or may not expose it. The spec handles MAHO-missing tiles by storing all-False in `wdl_prior_holes` (or omitting the array). Future lane.
- **Risk 8 (low):** WDL signed int16 vs V18 float32 height. WDL heights are on the same scale as the regular height maps (per WDL_v18.md), so no scale conversion is needed. The Python shim MUST convert int16 → float32 at the read boundary.
- **Risk 9 (low):** Alpha-era WDLs (`0_5_3_3368`) may have a different layout than LK-era WDLs (`3_3_5_12340`). The C# reader handles this; V24 reads whatever the C# reader returns per build.

---

## Assumptions

- The terrain→WDL path used by `WoWViewer`'s "click on map to spawn" visualization exists and is callable from Python (via a subprocess or in-process bridge). If it doesn't, FR-004 is the first thing fixed and the spec blocks.
- The C# WDL reader in `WowViewer.Core.IO` exists and can be invoked from Python to return the MARE grid + MAHO bitmask for a specific (tile_x, tile_y). If it can't, FR-001 is the first thing fixed and the spec blocks.
- The wowdev wiki MARE layout (17×17+16×16) is a working starting point for 3.3.5a but the C# reader is the source of truth. V24 accommodates whatever the C# reader actually returns.
- V18's `height_257` is the canonical terrain ground truth. We do not invent a new ground truth.
- Real WDLs in the staged client are partial coverage. The 99% match claim is the user's and is bounded by SC-001.
- The v7 model is deleted. There is no v7 to compare against. Success is "V24 beats the trivial `block_reduce` baseline on real data, with the WDL correlation learned from real coverage where it exists."
- V22 per-object mask data is suspect (per the user). V24 uses V18's tile-level `object_precise_mask` and does not depend on V22 per-object data.
- V22's brokenness is not relevant to V24. V18 is the substrate. The V22 audit question is a separate spec, not part of this one.
- V24 runs locally. No RunPod, no Pod packaging, no Spec 079 bundle.

---

## Open Questions (For User Review Before Plan)

1. **C# Python shim form:** is there an existing in-process bridge from Python to C# in the wow-viewer toolchain (e.g. pythonnet, a CLI shim, or an HTTP service)? If not, FR-001 / FR-002 / FR-004 all need a small Python shim. Recommend: build a tiny CLI shim that calls the existing C# WDL reader and the existing terrain→WDL path, and V24 calls the shim via subprocess.
2. **Minimap cleaning quality:** the spec's NumPy-based cleaner is a placeholder. If the user wants a model-based cleaner (e.g. a small U-Net that learns object masks from V18's `object_precise_mask`), that's a different spec. Recommend: NumPy cleaner for this spec; model-based cleaner is a future lane.
3. **WDL grid shape deferral:** the spec uses the C# reader's actual output as the WDL grid shape. If the C# reader returns 17×17+16×16, the spec works as written. If it returns something else (e.g. flat 16×16), the spec accommodates. The user should confirm whether the C# reader has been audited for which builds return which shape.

---

## Implementation Amendments (2026-07-06, verified against code and data)

These facts were verified by direct inspection of the C# source and the on-disk stores before implementation. Where they contradict the body of the spec above, **this section wins**.

### A1: C# WDL reader — audited shape (resolves Open Question 3 / Risk 3)

- Reader: `WowViewer.Core.IO.Maps.WdlSummaryReader.Read(path|stream)` → `WdlSummary` with per-tile `WdlTileSummary`.
- Per-MARE output: `OuterHeights` = 17×17 = 289 `short` + `InnerHeights` = 16×16 = 256 `short` (row-major). 64×64 tile grid per WDL, MAOF offsets, tolerant of both FourCC byte orders and headerless MARE payloads (era differences handled).
- **MAHO is NOT read** by the C# reader. Per Risk 7, `wdl_prior_holes` is therefore **omitted from the V24 store**. Stage B's hole gating uses V18's real `holes_16` (16×16 bool from ADT MCNK) instead — a strictly better signal.
- Terrain→WDL path: `WowViewer.Core.IO.Maps.WdlWriter.ExtractTileHeightsFromAlpha(float[,] height257, tileX, tileY)` — outer[r,c] = round(height257[16r, 16c]), inner[r,c] = round(height257[16r+8, 16c+8]), clamped to int16. This is the exact LR-from-HR algorithm; synthetic WDL error vs real is bounded by int16 rounding (≤ 0.5) wherever the client's WDL was generated the same way.

### A2: WDLs live inside MPQs, not as loose files

Staged clients do not contain loose `.wdl` files. On `3_3_5_12340` the WDL is `World\Maps\<map>\<map>.wdl` inside the big MPQs; on `0_5_3_3368` it is a loose per-map `<map>.wdl.mpq` mini-MPQ. The C# shim therefore takes `--client-root <staged-root> --map <name>` and resolves via `WowViewer.Core.IO.Files.NativeMpqService` (same pattern as `WowViewer.Tool.Harvest`), with a `--wdl <loose-path>` escape hatch for tests. FR-001/FR-002's "`.wdl` path" signature is amended accordingly.

### A3: Shim is batch-first (performance amendment to FR-001..FR-004)

Per-tile subprocess calls (4096/map) are not viable. The shim modes are:
- `read --client-root <root> --map <name> [--tile-x N --tile-y N] --output <npz>` — emits ALL present MARE tiles (`tile_xy` (N,2) int32, `outer` (N,17,17) float32, `inner` (N,16,16) float32) or a single tile when `--tile-x/--tile-y` given.
- `synth --height <npz> [--liquid <npz>] --output <npz>` — input `height_257` (257,257) or stacked (N,257,257) float32; emits `outer`/`inner` with the same stacking. Liquid-covered sample points are replaced by nearest-non-liquid-neighbour **in the shim** (new shim-local logic; core C# untouched).
- NPZ I/O in C# reuses the NPY conventions already proven in `NpzTileSerializer` (shim-local copy; core untouched).

### A4: V18 store actual schema (corrects input-channel claims throughout)

Verified on `output/datasets/v18/3_3_5_12340.zarr` (5134 tiles) and `0_5_3_3368.zarr` (1629 tiles):
- `minimap_rgb` is **(N, 256, 256, 3) uint8** — not 257×257. Minimap cleaning and all minimap model inputs operate at 256×256.
- `object_precise_mask` is **(N, 257, 257) float32** in [0,1] — not bool. Threshold at > 0.5 for masking; downsample to 256×256 (2×2 max) for minimap cleaning.
- The liquid array is **`liquid_mask` (N, 256, 256) float32** — the spec's `liquid_mask_256` name maps to this.
- `alpha_256` is (N, 256, 256, 4) float32; `normal_xyz` (N, 257, 257, 3) float32; `mcnr_mask_257` (N, 257, 257) bool; `holes_16` (N, 16, 16) bool.
- Tile keying via `index.parquet` at the store root: `tile_id`, `build`, `map`, `tile_x`, `tile_y`, `height_mean`, `height_std`, plus per-signal `has_*` flags.
- The `0_5_3_3368` V18 store additionally has **`no_object_minimap` (N, 256, 256, 3) uint8** — a viewer-rendered object-free minimap. Where present, the minimap cleaner prefers it outright over mask-median fill (US2 amendment). It is absent on `3_3_5_12340`.

### A5: V24 store schema (amends FR-007 / data-model)

Because the C# reader returns two grids, the merged prior is stored as paired arrays (no invented single shape):
- `wdl_prior_outer` (N, 17, 17) float32 and `wdl_prior_inner` (N, 16, 16) float32.
- `wdl_prior_source_outer` / `wdl_prior_source_inner` (uint8, 0=real / 1=synthetic / 2=learned-fill) and `wdl_prior_confidence_outer` / `wdl_prior_confidence_inner` (float32 [0,1]), same shapes.
- No `wdl_prior_holes` (per A1). V18 arrays are not copied; the V24 store carries the prior arrays plus a copied `index.parquet` and `.zattrs` provenance (`v18_store_path`, `staged_client_path`, coverage ratios). Consumers read V18 arrays from the V18 store by `tile_id` — this keeps the V24 store small and V18 canonical.

### A6: Prior ↔ 257×257 lattice mapping (Stage B upsample, exact)

Outer and inner grids interleave as a quincunx on a 33×33 half-step lattice: position (2i, 2j) = outer[i,j], (2i+1, 2j+1) = inner[i,j]; the remaining (even,odd)/(odd,even) positions are filled by the mean of their valid 4-neighbours. Since (33−1)×8+1 = 257, `bilinear_upsample(quincunx_33, 257, align_corners=True)` is exact at every WDL lattice point. This is the canonical `upsample(prior, 257)` used by Stage B and the baselines.

### A7: Stage A output heads

Stage A's trunk is a small U-Net at 64×64; two interpolation heads emit the (17,17) outer and (16,16) inner grids. The `block_reduce` baseline is evaluated at the same lattice points via the A1 sampling rule (point-sample at (16r,16c) / (16r+8,16c+8)), which makes the SC-002 comparison exact rather than approximate (retires Risk 6).

### A8: V22 dataset audit lane (user redirect, added scope)

The user has pulled the V22 audit into this work (amending "Out Of Scope"). The audit is **C#-grounded**: the existing, working C# harvester (`WowViewer.Tool.Harvest extract-unified`) re-extracts reference signals for a sampled tile set directly from the staged client, and a Python comparator (`data-harvester/scripts/audit_v22_dataset.py`) checks the V22 store (and the V18 substrate it carries) signal-by-signal against the C# reference: presence, shape, dtype, zero-fill rate, value-range and per-signal mismatch stats, plus `index.parquet` `has_*` flag truthfulness. Deliverables: audit script + JSON report + `docs/architecture/v22-dataset-audit-2026-07-06.md`. Python does the comparison only; all game-data reads stay in C#.

## End of Spec

This spec is bounded, cheap, and re-uses everything we already have. The hard preconditions (no DA-V2, no v7 comparison, C# wraps the WDL reader and the terrain→WDL path, ≤ 3M total params, runs locally on a 6 GB GPU) are explicit. The success criteria are measurable against the trivial `block_reduce(height_257)` baseline. The Implementation Amendments above record the verified ground truth this implementation is built on.
