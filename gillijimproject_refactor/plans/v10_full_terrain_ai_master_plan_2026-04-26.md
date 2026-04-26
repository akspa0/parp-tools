# V10 Full Terrain AI Master Plan

## Vision

Build a Mixture-of-Experts terrain reconstruction system that can take **any subset of available signals** — from a single minimap RGB image up to a full ADT with MCAL, MCLY, MH2O, MCCV, objects, PM4, and brush data — and produce complete, patchable terrain.

The system is trained on **pattern-mined dictionaries** extracted from real client data: repeating terrain brushes, texture layer combinations, alpha blend signatures, and object placement templates. It does not guess terrain from nothing; it learns the vocabulary of WoW terrain and recomposes it.

This is a five-phase plan. Phase 1 is the foundation: wiring every extractable signal into `wow-viewer` shared libraries.

This remains a Bring Your Own Data workflow.

---

## Phase 1: wow-viewer Library Signal Extraction

**Goal:** Every signal that exists in an ADT, WDT, PM4, or client DBC must be extractable through `wow-viewer` shared libraries and emitted as a normalized training tensor.

### 1.1 Core.IO ADT Deep Reader Expansion

Extend `WowViewer.Core.IO` ADT reading beyond top-level chunk enumeration to deep payload parsing for all terrain-relevant sub-chunks:

| Chunk | What to Extract | Output Tensor | Resolution | Library Home |
|---|---|---|---|---|
| `MCVT` | Per-vertex heights | `height_257` | 257×257 | `WowViewer.Core.IO` |
| `MCLY` | Layer flags, texture IDs, blend mode | `mclay_layer_mask` (4-class one-hot) + `mclay_texture_ids` | 16×16 per chunk, 16 chunks | `WowViewer.Core.IO` |
| `MCAL` | Alpha layer blend weights (up to 4 layers, packed) | `mcal_alpha_pack` (4 channels, 0-1) | 64×64 per chunk, 16 chunks = 256×256 | `WowViewer.Core.IO` |
| `MCCV` | Per-vertex ambient color (RGBA) | `mccv_rgb` | 257×257 | `WowViewer.Core.IO` |
| `MCNR` | Per-vertex normals | `mcnr_normal_xyz` | 257×257 | `WowViewer.Core.IO` |
| `MH2O` | Liquid height, depth, flags, type (WotLK+) | `mh2o_surface_height` + `mh2o_depth` + `mh2o_type_mask` | 257×257 | `WowViewer.Core.IO` |
| `MCLQ` | Inline liquid data (Alpha 0.5.3–3.0.1) | `mclq_surface_height` + `mclq_type_mask` | 257×257 | `WowViewer.Core.IO` |
| `MTXF` | Texture animation flags, transform | `mtxf_animated_mask` + `mtxf_transform_id` | 16×16 | `WowViewer.Core.IO` |
| `MCRF` | Doodad/WMO references per chunk | `mcrf_object_presence` | 16×16 | `WowViewer.Core.IO` |
| `WLW/WLM/WLQ/WLL` | Loose liquid project files (editor-era liquid) | `wl_liquid_mask_257` + `wl_liquid_height_257` | 257×257 | `WowViewer.Core.IO` |

**Removed from Wave 1:** `MFBO` (flight bounds are engine-only player constraints, not terrain shape signals) and `MCSE` (sound emitters are not visual data and do not appear in minimaps).

**Added to Wave 1:**
- **`MCLQ`** — Inline liquid chunks for pre-WotLK terrain (0.5.3 through 3.0.1). These carry liquid surface heights where `MH2O` does not exist.
- **`WLW/WLM/WLQ/WLL`** — Loose liquid project files found in `World\Maps\<map>` directories. These preserve liquid data for deleted or missing tiles and are a critical hint for terrain reconstruction in sparse maps like the development map. The existing `MdxViewer` `WlLiquidLoader` has proven parsing code that must be ported into `WowViewer.Core.IO`.

**Important:** All raw chunk data must be preserved with FourCCs forward in memory. The reader should emit both the raw chunk bytes and the parsed tensor so downstream tools can choose.

### 1.2 Object Silhouette and Mask Renderer

Add a new renderer or rasterizer in `WowViewer.Core` or `WowViewer.App` that can produce overhead 2D masks from 3D object placements:

**Inputs:**
- WMO group placement list (from ADT `MDDF`/`MODF` or WDT)
- M2 model placement list
- Optional: DBC-driven model classification (tree, building, rock, etc.)

**Outputs:**
- `object_mask_257` — binary mask of any object footprint
- `object_precise_mask_257` — anti-aliased silhouette using actual model bounds
- `object_class_mask_257` — N-channel one-hot per object class (tree, structure, doodad, liquid-edge-object)
- `object_height_override_257` — where an object explicitly overrides terrain Z (bridges, platforms)

**Implementation path:**
- Reuse `MdxViewer` `ModelRenderer` and `WmoRenderer` bounding-box or low-LOD mesh extraction
- Render from orthographic top-down camera into a 257×257 float framebuffer
- Store silhouette alpha as mask intensity
- Classify objects using `GameObjectDisplayInfo`-style DBC mapping when available

### 1.3 PM4 Signal Integrator

Leverage `WowViewer.Core.PM4` to produce terrain-supervision masks from PM4 pathfinding data:

- `pm4_path_mask_257` — binary mask of navigable path regions
- `pm4_building_footprint_mask_257` — MSLK-derived structure footprints
- `pm4_road_width_mask_257` — MPRR-derived road corridor width
- `pm4_elevation_hint_257` — sparse height hints from PM4 vertices where they overlap ADT tiles

**Boundary:** PM4 data is sparse and development-map-centric. The integrator should gracefully return empty/zero masks when no PM4 data exists for a tile.

### 1.4 ~~Brush and Sculpt Detection~~ → Removed

The user explicitly rejected generated brush-detection heuristics. Instead, Wave 2 will mine **real MCAL alpha masks** to discover the actual WoWEdit brush stroke patterns used by artists. This is pattern-finding on original sampled data, not derived data.

### 1.5 Minimap Variant Renderer

Extend the existing `WorldMinimapRenderer` to emit the full suite of control-signal minimaps:

1. `minimap_terrain_only_rgb_256` — no objects, no liquid, no MCCV tint
2. `minimap_no_liquid_rgb_256` — objects present, liquid removed
3. `minimap_no_object_rgb_256` — liquid present, objects removed
4. `minimap_no_mccv_rgb_256` — vertex colors neutralized
5. `minimap_wireframe_rgb_256` — mesh edges only
6. `minimap_height_colormap_256` — false-color elevation
7. `minimap_normal_rgb_256` — surface normals as color
8. `minimap_layer_index_rgb_256` — MCLY layer 0-3 as distinct colors
9. `minimap_liquid_mask_256` — binary water mask
10. `minimap_shadow_ao_rgb_256` — MCCV-only render (ambient occlusion visualization)

All variants should be renderable from a single ADT tile in one pass or a small multi-pass batch.

### 1.6 WDL and MCVT Coarse Prior Harmonizer

Create a unified coarse height extractor that produces a single `coarse_height_17` regardless of source:

- If WDL exists for the tile: sample WDL at tile center
- If WDL is missing but MCVT exists: downsample MCVT 257→17 via area averaging
- If neither exists: emit `null` / sentinel value (Stage 1 will fill this)

This removes the v9 "WDL-or-nothing" dependency and gives Stage 1 ground truth on every tile that has any height data.

### 1.7 Data Normalization and Tensor Emitter

Add a `TerrainTileTensorPack` class in `WowViewer.Core` that assembles all of the above into a single normalized payload:

```csharp
public class TerrainTileTensorPack
{
    public string TileName;                          // e.g. "development_16_32"
    public string MapName;
    public string BuildKey;
    public float[,] Height257;                       // 257×257
    public float[,] Height65;                        // 65×65
    public float[,] Height17;                        // 17×17
    public float[,,] MinimapVariants;                // [variant_count, 256, 256, 3]
    public float[,,] McalAlphaPack;                  // [256, 256, 4]
    public int[,,] MclyLayerMask;                    // [16, 16, 4] per-chunk, upsampled to [256, 256, 4]
    public float[,,] MccvRgb;                        // [257, 257, 3]
    public float[,,] McnrNormal;                     // [257, 257, 3]
    public float[,] Mh2oSurfaceHeight;               // 257×257
    public float[,] Mh2oDepth;                       // 257×257
    public float[,] ObjectMask;                      // 257×257
    public float[,] ObjectPreciseMask;               // 257×257
    public float[,,] ObjectClassMask;                // [257, 257, N_classes]
    public float[,] Pm4PathMask;                     // 257×257
    public float[,] Pm4BuildingFootprintMask;        // 257×257
    public float[,] BrushImprintMask;                // 257×257
    public float[,] HoleMask;                        // 16×16, upsampled
    public float[,] MclqSurfaceHeight;               // 257×257
    public float[,] MclqTypeMask;                    // 257×257
    public float[,] WlLiquidMask;                    // 257×257
    public float[,] WlLiquidHeight;                  // 257×257
    // ... metadata scalars
}
```

This class should serialize to `.npz` (via `Python.NET` bridge or direct `NumSharp` if available) or to a structured binary format that Python can mmap efficiently.

---

## Phase 2: Pattern Mining and Dictionary Learning

**Goal:** Discover the repeating vocabulary of WoW terrain from the extracted tensors, with emphasis on the **artist's digital brush toolkit** embedded in MCAL alpha masks.

### 2.1 MCLY Combination Dictionary

Mine all unique 4-tuples of `(layer0_texture_id, layer1_texture_id, layer2_texture_id, layer3_texture_id)` across the corpus.

- Output: `mclay_dictionary.json` — mapping from combination hash to frequency, example tiles, and inferred biome tag
- Insight: Most tiles reuse the same 50-100 texture combinations. These are the "biome palettes."

### 2.2 MCAL Brush Stroke Pattern Dictionary (Primary Focus)

The ADT is a virtual canvas. MCAL alpha layers are the paint. This phase mines real MCAL data to discover the **actual digital brush shapes** used by Blizzard artists in WoWEdit.

**Approach:**
1. Extract all non-uniform MCAL alpha patches (64×64 per chunk, per layer) from the corpus
2. Filter out near-uniform patches (solid 0 or solid 1) — these are bucket fills, not brush strokes
3. Cluster remaining patches by shape signature using a mixture of:
   - **Circular brush detector**: radial symmetry score + diameter estimation
   - **Square brush detector**: axis-aligned edge detection + corner sharpness
   - **Soft-edge brush detector**: Gaussian blur kernel fitting on alpha gradients
   - **Hard-edge brush detector**: step-function edge profile
4. For each discovered brush type, store:
   - Centroid alpha pattern (the "brush stamp")
   - Estimated brush size in world units
   - Estimated hardness (soft vs hard edge)
   - Frequency of occurrence across corpus
   - Example tile coordinates where it appears

- Output: `mcal_brush_dictionary.npz` — N discovered brush stamps (e.g. N=64-128)
- This is **not generated data** — it is direct pattern extraction from real artist brushwork
- At generation time, the model can compose terrain by "painting" with these discovered brushes

### 2.3 MCAL Alpha Composition Vocabulary

Beyond individual brush strokes, mine **multi-brush compositions** — small 64×64 patches that show how multiple brushes were combined:

- "road brush + grass feathered edge = path through meadow"
- "cliff hard-edge + dirt soft-edge = eroded slope"
- "snow cap circular + rock base = mountain peak"

- Output: `mcal_composition_dictionary.npz` — pairs/triples of brush indices + their spatial arrangement
- This becomes a **style grammar** for the texture generation model

### 2.4 Height Profile Clustering

Cluster 257×257 height fields to find repeating terrain archetypes:

- Flat plain, rolling hills, cliff face, valley floor, mountain ridge, beach/shoreline, plateau, crater
- Output: `height_profile_dictionary.npz` — centroid height fields + cluster labels
- Use as terrain prefab templates

### 2.5 Minimap-to-Biome Classifier

Train a lightweight classifier mapping minimap RGB → MCLY combination hash.

- Output: `minimap_to_mclay_classifier.pt`
- Provides the texture generation model with a predicted palette before painting begins

---

## Phase 3: Synthesized Control Data Generation

**Goal:** Create unlimited high-quality training pairs by rendering synthetic minimaps from real terrain, and by augmenting real data with controlled perturbations.

### 3.1 Synthetic Minimap Renderer

A standalone tool (C# in `wow-viewer` or Python) that takes a `TerrainTileTensorPack` and produces a configurable minimap:

**Render modes:**
- Standard top-down orthographic
- Oblique angle (45°) for slope-sensitive training
- Multiple lighting directions (N, S, E, W) to make the model lighting-invariant
- Seasonal tint variants (summer, winter, autumn) using MCLY palette substitution
- Low-resolution variants (128×128, 64×64) for multi-scale training

**Why this matters:**
- A single real tile can generate 20+ synthetic minimaps
- The model sees the same height with different colors, lighting, and angles
- This forces the model to learn **invariant terrain features** rather than memorizing specific color textures

### 3.2 Object Silhouette Atlas

Render every unique M2 and WMO model from the client MPQs into a top-down silhouette atlas:

- `object_silhouette_atlas/` — one PNG per model, 256×256, alpha = footprint
- Classify by bounding box aspect ratio: tree (tall, narrow), rock (irregular), building (boxy), bridge (long, thin)
- At inference time, if an object is detected in a minimap, its silhouette can be looked up and subtracted

### 3.3 Controlled Ablation Dataset

Generate training pairs where specific signals are systematically removed:

| Variant | Removed Signal | Purpose |
|---|---|---|
| `no_mcal` | MCAL alpha | Forces model to infer blend from MCLY + height only |
| `no_mccv` | Vertex colors | Tests lighting independence |
| `no_mh2o` | Liquid data | Tests shoreline inference from height alone |
| `no_objects` | All object masks | Tests pure terrain recovery |
| `low_res_minimap` | 64×64 minimap | Tests robustness to input quality |
| `noise_minimap` | Gaussian noise on RGB | Tests robustness to compression artifacts |

### 3.4 Cross-Build Style Transfer

Use the same ADT tile geometry but swap texture assets from different client builds:

- Take a 3.3.5 ADT height + MCLY layout
- Substitute 0.5.3 texture IDs where equivalents exist
- Render a "retro minimap" of the same terrain
- Train the model to recover the same height from both modern and retro color schemes

---

## Phase 4: Ensemble of Specialized Small Models

**Goal:** Instead of one giant MoE, train **multiple small, focused models** that each master one reconstruction task. They share a common feature backbone where efficient, but each has its own output head and loss. At inference time, they run in sequence or in parallel depending on available inputs.

### 4.1 Model Family

| Model | Inputs | Outputs | Size | Purpose |
|---|---|---|---|---|
| `minimap2height` | minimap RGB 256×256 | `height_17` + `height_65` + `height_257` | ~5M params | Coarse-to-fine terrain shape from vision alone |
| `minimap2mclay` | minimap RGB 256×256 | `mclay_layer_mask_256` (4-class) | ~2M params | Predict which texture palette belongs to the tile |
| `height2normal` | `height_257` + `mccv_rgb` | `refined_normal_257` | ~1M params | Derive surface normals from height + AO |
| `mclay2mcal` | `mclay_layer_mask_256` + `height_257` + `minimap_rgb` | `mcal_alpha_pack_256` (4 channels) | ~8M params | **Paint** alpha masks using discovered brush dictionary |
| `liquid_solver` | `mh2o`/`mclq`/`wl` liquid masks + `height_257` | `liquid_surface_height_257` + `shoreline_mask_257` | ~2M params | Constrain water/land boundaries |
| `object_terrain` | object masks + `height_257` + `minimap_rgb` | `height_257` (object-cleared) | ~3M params | Flatten/build terrain around structures |
| `pm4_path` | PM4 path mask + `height_257` | `height_257` (path-integrated) | ~2M params | Embed road/path corridors into terrain |

### 4.2 Shared Backbone

The first three models (`minimap2height`, `minimap2mclay`, `height2normal`) share a lightweight vision encoder backbone (ResNet-18 style) to avoid redundant computation. The other models operate on the outputs of these foundation models plus their own specialized inputs.

### 4.3 The `mclay2mcal` Model — The Brush Painter

This is the crown jewel. It does not regress raw alpha values. Instead, it uses the **MCAL brush dictionary** from Phase 2 as a codebook:

1. Encode the tile context (MCLY palette + height + minimap) into a latent vector
2. Predict a **sparse set of brush strokes**: `(brush_index, x, y, size, rotation, opacity)` for each of up to K strokes
3. Render each stroke by looking up the brush stamp from `mcal_brush_dictionary` and alpha-compositing it onto the canvas
4. Final output is the composed 256×256 alpha pack

**Training loss:**
- L1 between rendered alphas and real MCAL
- Perceptual loss on gradient magnitude (edges must match)
- Sparsity loss on stroke count (prefer fewer, larger brushes — like real artists)

At inference time, this model literally **paints** the alpha layers using the discovered artist brush toolkit.

### 4.4 Inference Pipeline

```
minimap_rgb ──→ minimap2height ──→ height_257
           └──→ minimap2mclay ──→ mclay_palette
                                  ↓
height_257 ───→ height2normal ──→ normal_257
mclay_palette ─→ mclay2mcal ────→ mcal_alpha_pack
                                  ↓
height_257 + mcal ──→ [liquid_solver, object_terrain, pm4_path] ──→ refined_height_257
```

Each arrow is one small model. The pipeline is modular: if a model is missing, the downstream models use zeros/defaults and still produce output.

### 4.5 Training Strategy

1. Train `minimap2height` and `minimap2mclay` first (they share the backbone)
2. Freeze backbone, train `height2normal` and `mclay2mcal`
3. Train `liquid_solver`, `object_terrain`, `pm4_path` independently (they depend on Stage 1 outputs)
4. Optional end-to-end fine-tuning: unfreeze all, train jointly with a combined ADT-reconstruction loss

---

## Phase 5: Training Pipeline and Local Execution

### 5.1 Dataset Builder

`build_v10_moe_dataset.py`:
1. Scan all client roots via `wow-viewer` shared I/O
2. Emit `TerrainTileTensorPack` per tile as `.npz`
3. Run Phase 2 pattern mining, append dictionary lookups to each pack
4. Run Phase 3 synthesis, append synthetic variants
5. Produce a manifest with per-tile `available_signals` bitmask

### 5.2 Training Loop

`train_v10_moe.py`:
1. Load manifest, group tiles by `available_signals` pattern
2. Sample batches that mix different signal-availability patterns
3. Forward through gating network + active experts
4. Loss = ground_truth_L1 + distillation_L1(from full_expert) + gradient_loss + layer_dominance_CE
5. Log per-expert utilization and per-signal-pattern accuracy

### 5.3 Local Hardware Plan

| Stage | VRAM | Local Time | Notes |
|---|---|---|---|
| Phase 1 extraction | N/A (CPU/disk) | Hours | One-time per client root |
| Phase 2 mining | 8 GB | 1-2 hours | K-means + clustering on extracted tensors |
| Phase 3 synthesis | N/A (render) | Hours | Parallelizable across tiles |
| Phase 4-5 training | 20-24 GB | Days | RTX 4090 with gradient accumulation, batch size 2-4 effective 8-16 |

### 5.4 Evaluation Metrics

- `global_mae` — mean absolute error on height_257
- `edge_mae` — MAE on gradient magnitude > threshold (cliff/road accuracy)
- `liquid_boundary_mae` — MAE within 5 pixels of MH2O boundary
- `object_clearance_mae` — MAE on pixels where object mask = 1 (does terrain respect objects?)
- `pm4_path_mae` — MAE on PM4 path corridors
- `mclay_accuracy` — classification accuracy of predicted MCLY palette
- `mcal_l1` — L1 between predicted and real MCAL alpha pack
- `mcal_edge_ssim` — SSIM on MCAL gradient magnitude

---

## Implementation Order (Condensed)

### Wave 1: Foundation (wow-viewer libraries)
1. Extend `WowViewer.Core.IO` ADT reader for MCAL, MCLY deep parse
2. Add MCCV, MCNR, MH2O, MCLQ, MTXF extractors; port WLW/WLM/WLQ/WLL reader from MdxViewer
3. Add `TerrainTileTensorPack` serializer
4. Add overhead object silhouette renderer
5. Add PM4 mask integrator

### Wave 2: Pattern Mining
7. Run MCLY combination dictionary across all client roots
8. Run MCAL brush stroke pattern mining (circular, square, soft-edge, hard-edge detectors)
9. Run MCAL composition vocabulary mining (multi-brush arrangements)
10. Run height profile clustering
11. Train `minimap_to_mclay_classifier`

### Wave 3: Synthesis
11. Build synthetic minimap renderer (multi-angle, multi-lighting)
12. Build object silhouette atlas from client MPQs
13. Generate controlled ablation dataset

### Wave 4: Model
14. Implement shared vision encoder backbone
15. Implement `minimap2height` model
16. Implement `minimap2mclay` model
17. Implement `mclay2mcal` brush painter model with discovered brush dictionary
18. Implement `height2normal` model
19. Implement `liquid_solver`, `object_terrain`, `pm4_path` constraint models
20. Optional: joint fine-tuning pass

### Wave 5: Integration
21. End-to-end inference: minimap → ensemble → height + MCLY + MCAL + normals
22. ADT patch writer: inject all predicted layers into ADT template
23. Viewer integration: load predicted ADT, render side-by-side with real
24. Evaluation suite: all metrics on development-map holdout

---

## What V10 Should Claim

**Can claim:**
- Reconstructs plausible terrain height from minimap-only inputs
- Predicts appropriate texture layers (MCLY palette) from minimap color cues
- **Paints** alpha blend masks (MCAL) using discovered artist brush patterns from real data
- Produces complete, patchable ADT output with height + texture + alpha layers
- Respects objects, liquid boundaries, and PM4 corridors through constraint models
- Learns and reuses repeating terrain patterns from the artist's original brush toolkit

**Cannot yet claim:**
- Exact object placement recovery (we clear terrain around objects, don't place them)
- Perfect runtime world parity with native client rendering
- Generalization to non-WoW terrain art styles
- Recovery of data that was never present in any client file

---

## Immediate Next Step

The user's directive is clear: **wire every bit of this up in wow-viewer's library first.**

The first implementation slice should be Wave 1, Item 1: extending `WowViewer.Core.IO` to deeply parse MCAL and MCLY chunks and emit them as normalized tensors. This unlocks everything downstream.
