# Multi-Model Terrain Reconstruction Architecture

**Date:** 2026-05-16
**Status:** Draft — spec for discussion, not yet implemented
**Depends on:** V16 terrain model (trained), object segmentation model (not yet built)

---

## 0. Problem Statement

The terrain model (V16) predicts height, normals, alpha, liquid, and holes from a
minimap image. During training, we downweight pixels under objects using the
`object_mask_257` signal — so the model doesn't waste capacity predicting terrain
under buildings and trees.

**At inference time, when we only have a minimap and no ADT data, we have no
object mask.** The model must either:

1. Learn to implicitly ignore object pixels (fragile, depends on training
   signal quality), or
2. Be provided with a predicted object mask from a separate model.

But the goal is bigger than just "where are objects?". If we can **identify which
asset** each object pixel corresponds to — not just "something is here" but
"this is HumanTower.wmo at position (X,Y,Z) with rotation (R)" — then we can
reconstruct full MDDF/MODF placement data from a minimap alone. This would
unlock:

- Terrain predictions uncorrupted by object pixels
- Placement data for tiles where no ADT/PM4 data exists
- Cross-referencing with PM4 CK24 layers to map those layers to real assets
- Underground object detection by elimination (visible vs. PM4-only)

## 1. Current Signal Inventory

### What We Harvest Per Tile (NPZ keys)

| Signal | Shape | Source | Per-Instance? |
|--------|-------|--------|---------------|
| `minimap_rgb_256` | 256×256×3 | Game BLP (terrain+objects+shadows+liquid baked in) | No |
| `height_257` | 257×257 | MCVT | No |
| `mcnr_normal_xyz` | 257×257×3 | MCNR | No |
| `mcal_alpha_pack_256` | 256×256×4 | MCAL + MCLY | No |
| `hole_mask_16` | 16×16 | MCNK flags | No |
| `unified_liquid_mask` | varies | MH2O/MCLQ/WL* | No |
| `unified_liquid_height` | varies | MH2O/MCLQ/WL* | No |
| `object_mask_257` | 257×257 | MDDF/MODF projected footprint | **No — collapsed binary** |
| `object_precise_mask_257` | 257×257 | MDDF/MODF with soft edges | No — collapsed binary |
| `shadow_residual_mask_256` | 256×256 | MCSH - object-shadow subtraction | No |
| `placement_mddf_data` | N×9 | MDDF structured data | **Yes — per-placement rows** |
| `placement_modf_data` | N×14 | MODF structured data | **Yes — per-placement rows** |
| `mcly_texture_ids` | 16×16×4 | MCLY | No |
| `pm4_path_mask` | 257×257 | PM4 MSLK/MSPI | No — collapsed binary |
| `pm4_building_footprint_mask` | 257×257 | PM4 MSUR/MSVT | No — collapsed binary |
| `pm4_mprl_mask` | 257×257 | PM4 MPRL | No — collapsed binary |
| `metadata.json` | — | Per-tile metadata | **Yes — name tables** |

### What The Metadata Contains

Each NPZ shard's `metadata.json` carries:

- `placement_mddf_names`: List of M2 model paths, indexed by `nameId`
- `placement_modf_names`: List of WMO model paths, indexed by `nameId`
- `placement_mddf_name_table`: Sparse array mapping nameId → path
- `placement_modf_name_table`: Sparse array mapping nameId → path

**This is the asset-path attribution we need.** The structured placement data
(`placement_mddf_data` / `placement_modf_data`) already links each placement
to a nameId, which maps to an asset path. The pipeline produces this data but
the V16 Zarr builder does not yet carry it into the training dataset.

### What The Minimap Actually Contains

`minimap_rgb_256` is a direct decode of the game's pre-rendered BLP minimap
texture. It contains terrain, shadows, liquids, MCCV tinting, WMO buildings,
and M2 trees/doodads all composited together by the WoW client. Objects are
baked in and cannot be separated by channel alone.

The C# harvest tool also has a `GenerateSyntheticMinimap()` path that composites
real BLP tileset textures using MCAL alpha weights, producing a terrain-only
estimate. The residual `minimap_rgb - synthetic` approximates objects+shadows+
liquid+tinting.

### What PM4 Data Contains

PM4 files contain:

- **CK24 entries**: Object layers, where each entry is (typically) the same
  object stamped across multiple ADT/PM4 tiles. This is object-layer data,
  not per-pixel masks.
- **MSLK/MSPI/MSPV**: Navigable path lines (rasterized as `pm4_path_mask`)
- **MSUR/MSVI/MSVT**: Building surface triangles (rasterized as
  `pm4_building_footprint_mask`)
- **MPRL**: Portal/zone-transition markers (rasterized as `pm4_mprl_mask`)

The PM4 masks are **binary composites** — all object footprints collapsed into
one mask with no per-instance or per-asset differentiation. The CK24 layer
structure is analyzed by `Pm4Ck24ForensicsAnalyzer` and
`Pm4ResearchHierarchyAnalyzer` in the inspect tool but is **not** carried into
the NPZ harvest pipeline.

### Key Gap

**We have per-placement structured data with asset paths (MDDF/MODF) but no
per-instance rasterized segmentation.** The `object_mask_257` is a collapsed
binary mask that tells us "something is here" but not "what" or "which object
this pixel belongs to."

We need a per-instance segmentation that maps each object pixel back to a
specific `placement_mddf_data[i]` or `placement_modf_data[j]` row — and from
there to an asset path.

---

## 2. Multi-Model Architecture

The full system consists of **four independent models**, each trained on its
own residual signal, chained at inference time but not sharing weights:

```
                          ┌─────────────────────┐
                          │   Minimap RGB 256    │
                          └─────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
            │  Model A:    │ │  Model B:   │ │  Model C:    │
            │  Object Seg  │ │  Liquid Seg │ │  Shadow Seg  │
            │  minimap→mask│ │  minimap→mask│ │  minimap→mask│
            └──────┬───────┘ └──────┬──────┘ └──────┬───────┘
                   │                │                │
            predicted obj mask     predicted        predicted
            + instance IDs         liquid mask      shadow mask
                   │                                │
                   ▼                                ▼
            ┌──────────────┐               ┌──────────────┐
            │  Model D:    │               │  Model E:    │
            │  Asset Attr  │               │  Shadow      │
            │  instance→  │               │  Explanation │
            │  asset path  │               │  (optional)  │
            └──────┬───────┘               └──────────────┘
                   │
         predicted asset paths + positions
                   │
                   ▼
    ┌──────────────────────────────────────────────┐
    │  Clean Minimap = Raw - Predicted Object Region │
    │  + predicted object mask weighting             │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Model F:      │
                  │   Terrain V16   │
                  │   clean minimap │
                  │   → height, etc │
                  └─────────────────┘
```

### Model A — Object Segmentation

| Property | Value |
|----------|-------|
| Input | minimap RGB 256×256×3 |
| Outputs | object mask 257×257 (binary), instance IDs 257×257 (int32) |
| Ground truth | `object_mask_257` (binary) for mask; `placement_mddf_data` + `placement_modf_data` projected individually for instance IDs |
| Loss | L1 mask + discriminative instance loss |
| Purpose | Identify which pixels belong to objects, and which object each pixel belongs to |

**Critical design point:** The instance segmentation must project each placement
from `placement_mddf_data[i]` and `placement_modf_data[j]` individually onto the
tile to create a per-instance ground truth mask. M2 placements paint as circles
at position with known radius; WMO placements paint as rectangles from their
bounding box extents. This gives us per-instance label maps with `0=terrain`,
`1=WMO_0`, `2=WMO_1`, `3=M2_0`, etc.

### Model B — Liquid Segmentation (V16 head, extended)

| Property | Value |
|----------|-------|
| Input | minimap RGB 256×256×3 |
| Output | liquid mask 256×256 |
| Ground truth | `unified_liquid_mask` |
| Loss | L1, weighted by `has_liquid` |
| Purpose | Identify water/lava pixels for terrain reconstruction |

This is the existing liquid head in V16. It may already converge well enough to
serve as the standalone liquid segmentation model. If so, Model B is just the
V16 liquid head extracted as an independent network.

### Model C — Shadow Segmentation (optional, deferred)

Identify shadow pixels from the minimap. Ground truth is `shadow_residual_mask_256`.
Deferred until Models A and F converge.

### Model D — Asset Attribution

| Property | Value |
|----------|-------|
| Input | minimap RGB 256×256×3 + object instance mask 257×257 |
| Output | Per-instance asset classification (WMO/M2 type + top-K asset path IDs) |
| Ground truth | `placement_mddf_data` nameId column + `metadata.json` name tables |
| Loss | Cross-entropy over known asset vocabulary + metric learning for unseen assets |
| Purpose | Map each detected object instance to an asset path |

This is the hardest model. The asset vocabulary is large (thousands of unique
WMO and M2 paths). Two approaches:

1. **Closed vocabulary classification** — enumerate all asset paths seen in
   training, treat as a classification problem. Works for common buildings.
   Fails for rare or unseen assets.

2. **Embedding/metric learning** — learn a visual embedding for each instance
   crop, then match against an asset embedding database. Can generalize to
   unseen assets if they share visual features with known ones.

Hybrid approach: classify into top-10 candidates, then refine with embedding
distance. For development map tiles (which reuse a small set of WMO/M2
assets), closed vocabulary may suffice.

### Model E — Shadow Explanation (optional, deferred)

Segment shadows from minimap and explain them as "shadow of building at
position X" vs. "hill shadow". Requires 3D data for ground truth. Deferred.

### Model F — Terrain Reconstruction (existing V16)

| Property | Value |
|----------|-------|
| Input | clean minimap 256×256×3 (objects subtracted or masked) |
| Outputs | height 257×257, normals 257×257×3, alpha 256×256×4, holes 16×16 |
| Ground truth | All terrain signals from ADT |
| Loss | L1 height + 2× cosine normals + L1 alpha + L1 holes |
| Purpose | Reconstruct terrain geometry from the clean minimap |

This is the existing V16 model, potentially modified to use a predicted object
mask for downweighting instead of the ground-truth mask.

---

## 3. Training Data Gaps

### Gap 1: Per-Instance Object Segmentation Ground Truth

**Current:** `object_mask_257` is a collapsed binary mask.

**Needed:** Per-instance label map where each pixel is labeled with the index of
the placement that covers it (0 = terrain, 1 = WMO placement 0, 2 = WMO
placement 1, 3 = M2 placement 0, etc.).

**How to build it:** Project each placement from `placement_mddf_data[i]` and
`placement_modf_data[j]` individually onto the tile. M2 placements are circles
at world position with radius 2-3 pixels. WMO placements are axis-aligned
rectangles projected from their bounds (columns 8-13 of `placement_modf_data`).
Overlapping instances are resolved by painter's order (later placements painted
on top of earlier ones, matching the game's rendering order).

This is a **harvest-side change** — we need a new `object_instance_mask_257`
NPZ key that carries per-instance labels. This can be built entirely from data
we already harvest (`placement_mddf_data` + `placement_modf_data` + their name
tables + world-to-tile coordinate mapping).

### Gap 2: Asset Path Vocabulary

**Current:** `metadata.json` in each NPZ shard contains `placement_mddf_names`
and `placement_modf_names` — the full list of asset paths used by that tile.

**Needed:** A global vocabulary of unique asset paths across the entire dataset,
plus per-tile mappings from instance mask IDs to asset path indices.

**How to build it:** Scan all harvested tiles, collect unique asset paths, build
a sorted vocabulary file. Each tile's instance mask then maps its per-pixel
labels to indices in this vocabulary. This is a dataset curation step, not a
harvest change.

### Gap 3: PM4-to-Object Mapping

**Current:** PM4 data is harvested as collapsed binary masks
(`pm4_path_mask`, `pm4_building_footprint_mask`, `pm4_mprl_mask`). The CK24
layer structure is analyzed by the inspect tool but not carried into the
dataset.

**Needed:** A mapping from PM4 CK24 entries to real MDDF/MODF placements. This
would let us identify which real objects correspond to which PM4 features.

**How to build it:** This is currently unsolved. PM4 CK24 entries reference
object IDs that may correspond to MDDF/MODF nameId values, but the exact
mapping varies by build and requires reverse engineering the PM4 format. The
inspect tool can dump CK24 forensics but we don't have an automatic mapping yet.

**Progressive approach:** Start without PM4 data. Train Model A and D on tiles
where we have ADT data. Later, use Model D's output to identify objects in tiles
where we only have PM4 data, and cross-reference to build the CK24 mapping.

### Gap 4: Clean Minimap for Terrain Training

**Current:** V16 trains on raw `minimap_rgb_256` and uses `object_mask_257` as a
downweighting mask. The model sees object pixels but is told not to predict
terrain there.

**Needed for inference:** A "clean" minimap where object pixels have been
replaced by terrain inpainting, so the terrain model never sees object artifacts.

**How to build it:**
- At training time: use the ground-truth `object_mask_257` to mask out object
  pixels and inpaint them with surrounding terrain (median blur, neighbor
  sampling, or the synthetic compositor residual).
- At inference time: use Model A's predicted object mask to do the same
  inpainting before feeding the minimap to Model F.

The C# harvest tool already has the infrastructure for this
(`GenerateSyntheticMinimap()` and `MinimapBakeService`).

---

## 4. Data Pipeline Changes

### Phase 1 — Extend V16 Zarr with Placement Data (harvest-side)

The V16 Zarr builder (`build_v16_dataset.py`) currently drops structured
placement data. We need to add:

| New Zarr Array | Shape | dtype | Source |
|----------------|-------|-------|--------|
| `object_instance_mask_257` | (N, 257, 257) | int32 | Per-pixel instance label (0=terrain, 1+=placement index) |
| `placement_mddf_data` | (N, max_mddf, 9) | float32 | Per-placement MDDF rows (padded) |
| `placement_modf_data` | (N, max_modf, 14) | float32 | Per-placement MODF rows (padded) |
| `placement_mddf_names` | (N, max_mddf) | str | Per-placement M2 asset path |
| `placement_modf_names` | (N, max_modf) | str | Per-placement WMO asset path |

The Parquet index should also gain:

| New Column | dtype | Content |
|------------|-------|---------|
| `has_placements` | bool | Tile has nonzero placement data |
| `n_mddf` | int32 | Number of MDDF placements |
| `n_modf` | int32 | Number of MODF placements |

Variable-length placement data (tiles range from 0 to ~200+ placements) can be
stored as padded arrays with a maximum capacity, or as ragged arrays if Zarr
supports them. Alternative: store placement data as separate per-tile entries
in a second Parquet table indexed by `(build, map, tile_x, tile_y)`.

**This is the highest-priority data pipeline change** because it unlocks all
downstream models (A and D).

### Phase 2 — Asset Vocabulary

Build a global asset vocabulary from all harvested tiles:

```
asset_vocab.json = {
    "m2_paths": ["World\\Kalimdor\\...\\TreeModel.m2", ...],
    "wmo_paths": ["World\\Kalimdor\\...\\HumanTower.wmo", ...],
    "m2_count": 1234,
    "wmo_count": 567
}
```

Each tile's instance mask maps pixel labels to indices in this vocabulary,
enabling the classification model to operate on a fixed output space.

### Phase 3 — Clean Minimap Generation

For terrain model training, generate a "terrain-only" minimap variant by
masking out object pixels and inpainting with terrain:

1. Use `object_mask_257` (binary) to identify object pixels
2. Inpaint those pixels using the synthetic compositor output or median
   blur of surrounding terrain
3. Store as `clean_minimap_rgb_256` in the Zarr

Existing C# code in `GenerateSyntheticMinimap()` + `MinimapBakeService` can
produce the terrain-only baseline. The inpainting step can be a simple
median filter or a learned inpainter trained on tiles where we have both
raw and synthetic minimaps.

---

## 5. Model Training Order (Respecting Rule 7 and Rule 8)

Models train independently. Each model's output becomes input to downstream
models at inference time, but not during training. Each model trains on ground
truth directly.

### Phase N: V16 Terrain (current, in progress)

Train V16 terrain model using ground-truth `object_mask_257` for downweighting.
This is already wired. No blockers.

**Deliverable:** Converged V16 terrain checkpoint.

### Phase N+1: Object Segmentation (Model A)

Train object segmentation on tiles with placement data. Ground truth is the
per-instance mask generated from `placement_mddf_data` + `placement_modf_data`
projection. Uses the same minimap RGB input as V16.

**Deliverable:** Per-pixel object mask + instance ID predictions.

**Blocker:** Needs `object_instance_mask_257` in the Zarr builder (Gap 1).

### Phase N+2: Asset Attribution (Model D)

Train asset attribution on tiles with placement data + name tables. Ground
truth is the nameId-to-asset-path linkage from `metadata.json`. Takes minimap
RGB + predicted instance mask as input.

**Deliverable:** Asset path predictions for each detected object instance.

**Blocker:** Needs asset vocabulary (Gap 2) and Model A predictions for
inference (not for training — during training, ground-truth instance masks
are available).

### Phase N+3: Clean Minimap Terrain (Model F upgrade)

Retrain or fine-tune V16 terrain model using clean minimaps (objects removed
and inpainted) instead of raw minimaps with object downweighting. At inference,
use Model A's predicted mask to clean the input before feeding to terrain.

**Deliverable:** Improved terrain predictions on tiles with heavy object
coverage.

**Blocker:** Needs clean minimap generation pipeline (Gap 4) and Model A
predictions for inference.

### Phase N+4: PM4 Cross-Reference (analysis, not a model)

Use Model D's asset path predictions on tiles where only PM4 data exists (no
ADT). Cross-reference predicted assets with PM4 CK24 entries to build a
mapping from CK24 layers to real WMO/M2 asset paths.

**Deliverable:** CK24-to-asset mapping table.

**Blocker:** Needs trained Model D and PM4-analyzed tiles.

---

## 6. Inference Pipeline (Full Chain)

For a tile where we only have a minimap (no ADT data):

```
1. minimap_rgb → Model A → object_mask + instance_ids
2. minimap_rgb → Model B → liquid_mask
3. instance crops → Model D → asset paths + placement positions
4. clean_minimap = inpaint(minimap_rgb, object_mask)
5. clean_minimap → Model F → height, normals, alpha, holes
6. Combine: height + normals + alpha + holes + liquid + placements
```

For a tile where we have ADT data (training or validation):

```
Ground truth is available for all signals.
Models train independently on ground truth.
At inference, the ground truth mask substitutes for Model A's prediction.
```

---

## 7. What We Have vs. What We Need

| Signal | Harvested? | In V16 Zarr? | Per-Instance? | Notes |
|--------|-----------|---------------|---------------|-------|
| minimap_rgb_256 | Yes | Yes | No | Raw game BLP, objects baked in |
| height_257 | Yes | Yes | No | Per-vertex height |
| mcnr_normal_xyz | Yes | Yes | No | Per-vertex normals |
| mcal_alpha_pack_256 | Yes | Yes | No | Per-pixel alpha blend |
| hole_mask_16 | Yes | Yes | No | Per-chunk hole flags |
| unified_liquid_mask | Yes | Yes | No | Per-pixel water presence |
| unified_liquid_height | Yes | Yes | No | Per-pixel water surface |
| object_mask_257 | Yes | Yes | **No — binary only** | Collapsed binary footprint |
| object_instance_mask_257 | **No** | **No** | **Yes (needed)** | Must generate from placement data |
| placement_mddf_data | Yes | **No** | Yes (per-placement) | Structured M2 placement rows |
| placement_modf_data | Yes | **No** | Yes (per-placement) | Structured WMO placement rows |
| placement_mddf_names | Yes (metadata) | **No** | Yes (per-placement) | M2 asset path strings |
| placement_modf_names | Yes (metadata) | **No** | Yes (per-placement) | WMO asset path strings |
| shadow_residual_mask_256 | Yes | Yes | No | Per-pixel shadow occupancy |
| pm4_path_mask | Yes | **No** | No | Collapsed binary from MSPI/MSLK |
| pm4_building_footprint_mask | Yes | **No** | No | Collapsed binary from MSUR/MSVT |
| pm4_mprl_mask | Yes | **No** | No | Collapsed binary from MPRL |
| clean_minimap_rgb_256 | **No** | **No** | No | Needs object inpainting |
| asset_vocab.json | **No** | **No** | — | Global vocabulary from all tiles |

---

## 8. Critical Design Decision: PM4's Role

PM4 data is **not** training data for the segmentation models. It is
**cross-reference data** for the inference pipeline:

1. PM4 CK24 entries describe object layers that span multiple tiles
2. PM4 building footprints and path masks give us spatial layout where ADT
   data is missing (e.g., development maps that only have PM4)
3. After Model D can identify assets from minimap pixels, we can compare
   Model D's predictions against PM4 spatial data to build the CK24 mapping
4. Once we have the CK24 mapping, we can use PM4 as an additional ground
   truth source for tiles without ADT data

The PM4 masks currently in the NPZ (`pm4_path_mask`,
`pm4_building_footprint_mask`, `pm4_mprl_mask`) are **collapsed binary masks
with no per-instance differentiation**. They are useful as auxiliary signals
for the terrain model (e.g., "this area has underground geometry") but not
for object segmentation training.

---

## 9. Chain of Evidence

Every output signal must trace back to a ground-truth source. No circular
reasoning. No model output used as ground truth for another model.

| Output | Ground Truth Source | Chain |
|--------|-------------------|-------|
| Object mask (Model A) | `object_mask_257` from ADT MDDF/MODF projection | Direct game data |
| Instance IDs (Model A) | `placement_mddf_data` + `placement_modf_data` individually projected | Direct game data |
| Asset paths (Model D) | `placement_*_names` from `metadata.json` | Direct game data |
| Liquid mask (Model B) | `unified_liquid_mask` from MH2O/MCLQ/WL* | Direct game data |
| Height (Model F) | `height_257` from MCVT | Direct game data |
| Normals (Model F) | `mcnr_normal_xyz` from MCNR | Direct game data |
| Alpha (Model F) | `mcal_alpha_pack_256` from MCAL/MCLY | Direct game data |
| Holes (Model F) | `hole_mask_16` from MCNK flags | Direct game data |
| Clean minimap | `minimap_rgb_256` - `object_mask_257` inpainted | Synthetic, but derived from game data |
| PM4 CK24 mapping | Model D predictions cross-referenced with PM4 spatial data | **Inferred** — not ground truth, requires validation |

The PM4 CK24 mapping is the only output that is **inferred** rather than
ground-truth-derived. It must be validated independently (e.g., by checking
that predicted asset positions match PM4 spatial features across multiple
tiles).

---

## 10. Implementation Priority

1. **Train V16 terrain model** — no blockers, already wired
2. **Add `object_instance_mask_257` to Zarr builder** — project placements
   individually, label each pixel with a placement index
3. **Add placement data + name tables to Zarr** — carry
   `placement_mddf_data`, `placement_modf_data`, and name strings into
   the consolidated dataset
4. **Build global asset vocabulary** — scan all tiles, collect unique paths
5. **Train Model A (object segmentation)** — minimap → instance mask
6. **Train Model D (asset attribution)** — instance crop → asset path
7. **Build clean minimap pipeline** — inpaint objects from terrain
8. **Retrain terrain Model F on clean minimaps**
9. **Cross-reference PM4 CK24 with Model D predictions**

Steps 2-4 are data pipeline work (no model training needed). Steps 5-6 are
new models. Step 7 is a preprocessing change. Step 8 is a retraining run.
Step 9 is analysis work that produces the CK24 mapping table.