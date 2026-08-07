# Feature Specification: Unbaked Minimap Decomposition

**Feature Branch**: `133-unbaked-minimap-decomposition`

**Created**: 2026-08-05

**Status**: Draft

**Input**: User description: "The current minimap_rgb = albedo × lighting(lambert + ambient + cast_shadows), which blends texture, shadow, and normals into one RGB signal. Since we know how to bake it (the C# compositor produces the minimap from ground-truth height, normals, texture, and lighting), we can produce the decomposed signals directly from the synthesizer — no need for real authored minimaps. The compositor should emit terrain_shadow_256 (textureless lighting component) as a separate signal alongside the existing normal_xyz and height_257, so the model can learn each signal independently."

## Context

### The problem

The current minimap synthesis pipeline in [`TerrainMinimapCompositor.Compose()`](../../src/core/WowViewer.Core.IO/Maps/TerrainMinimapCompositor.cs) produces a single RGB image per tile:

```
for each pixel (x, y):
    albedo = BlendLayers(texture, alpha, ...)     // the terrain texture color
    lighting = Evaluate(lambert, ambient, shadow)  // the lighting term
    pixel = ApplyAlbedo(albedo, lighting)          // albedo × lighting
```

The output `minimap_rgb` therefore bakes **texture × shadow × normal** into one RGB image. A model trained on `minimap_rgb → height_257` must simultaneously disentangle:
1. What is texture color (albedo)
2. What is terrain shadow (lighting direction + cast shadows)
3. What is surface orientation (normals)
4. What is elevation (height)

This is why no direct minimap→height model beats the tile-mean baseline — the blended signal is too entangled.

### The fix

Since the compositor already has access to all the ground-truth signals (height, normals, texture, lighting), it can produce the decomposed signals directly:

- **`terrain_shadow_256`** — the textureless lighting component: just the Lambert N·L term + ambient + cast shadows, rendered over neutral white albedo. This is what the terrain shadow *looks like* independent of what the terrain is *made of*.
- **`minimap_rgb`** — keep the existing blended output for backward compat and comparison.
- **`normal_xyz`** — already exists as a separate signal (MCNR normals).
- **`height_257`** — already exists as a separate signal (MCVT heightmap).

The model can then be trained on any combination of these decomposed signals as inputs and targets, learning each relationship independently.

### Why this works

The compositor already has the decomposition path — `CreateWhiteTopEdge` and `CreateShadedTerrain` produce achromatic lighting renders. The missing piece is emitting the textureless lighting term as a separate dataset signal alongside the existing full minimap. Since we control the synthesis, we can produce the decomposition for every tile at zero additional data cost — no authored minimaps needed.

## User Scenarios & Testing

### User Story 1 - Emit terrain_shadow_256 as a separate harvest signal (Priority: P1)

A dataset operator can run the harvest pipeline and get a `terrain_shadow_256` signal (256x256 float32) alongside the existing `minimap_rgb` for every tile. The shadow signal contains only the lighting component (Lambert N·L + ambient + cast shadows) with neutral white albedo.

**Why this priority**: Without the decomposed signal, every model must learn to unbake the minimap as a hidden first step. Emitting the shadow signal directly removes this burden.

**Independent Test**: Harvest a tile, load both `minimap_rgb` and `terrain_shadow_256`, and confirm:
- `terrain_shadow_256` is a single-channel float32 array in [0, 1]
- `terrain_shadow_256` varies with surface orientation (normals) and cast shadows, not with texture color
- A flat white-textured tile's `terrain_shadow_256` matches its `minimap_rgb` grayscale conversion

**Acceptance Scenarios**:

1. **Given** a tile with known height and normals, **When** harvested with the decomposition enabled, **Then** `terrain_shadow_256` contains the textureless lighting term.
2. **Given** a tile with uniform albedo (e.g., snow), **When** compared, **Then** `terrain_shadow_256` is proportional to `minimap_rgb` grayscale within 5%.
3. **Given** a tile with varied albedo (e.g., grass + dirt), **When** compared, **Then** `terrain_shadow_256` is NOT proportional to `minimap_rgb` grayscale (because the texture color varies independently of the lighting).
4. **Given** the same tile harvested twice, **When** compared, **Then** both `terrain_shadow_256` arrays are bit-identical (deterministic).

---

### User Story 2 - Build a decomposed-signal training curriculum (Priority: P2)

A researcher can build a training curriculum that carries `terrain_shadow_256`, `minimap_rgb`, `normal_xyz`, and `height_257` as separate, row-aligned signals — enabling models that train on any combination of decomposed inputs.

**Why this priority**: The decomposed signals are useless for training until they're in a curriculum store the trainer can read.

**Independent Test**: Build the curriculum, verify every row has all four signals, and confirm the signals are pixel-aligned (same tile, same coordinates).

**Acceptance Scenarios**:

1. **Given** a rebuilt v50 store with `terrain_shadow_256`, **When** the curriculum builder runs, **Then** the curriculum carries all four signals.
2. **Given** a curriculum row, **When** `terrain_shadow_256` and `normal_xyz` are compared, **Then** the shadow gradient direction matches the normal-derived Lambert term.
3. **Given** a curriculum row, **When** `terrain_shadow_256` and `height_257` are compared, **Then** cast shadows in the shadow signal align with height ridges in the heightmap.

---

### User Story 3 - Train a model on decomposed signals (Priority: P3)

A researcher can train a model that takes `terrain_shadow_256` as input and predicts `height_257` as output, with the shadow→height relationship learned independently of texture variation.

**Why this priority**: This is the end goal — a model that learns the physical relationship between terrain shadow and terrain height, without the confounding texture signal.

**Independent Test**: Train on decomposed signals, evaluate on held-out tiles, and measure whether the shadow→height model beats the tile-mean baseline.

**Acceptance Scenarios**:

1. **Given** a trained shadow→height model, **When** evaluated on held-out tiles, **Then** it beats the tile-mean baseline by at least 5% relative.
2. **Given** a trained shadow→height model, **When** fed a tile with re-textured albedo (different texture, same shadow), **Then** the height prediction is unchanged (the model learned shadow, not texture).
3. **Given** a trained shadow→height model, **When** fed a tile with no shadow (flat lighting), **Then** the model reports low confidence rather than guessing.

### Edge Cases

- Tiles with no cast shadows (flat terrain, no ridges to occlude)
- Tiles with no texture (untextured terrain, neutral white base)
- Tiles where the lighting term is zero (fully shadowed)
- Cross-client consistency: the decomposition must work for 0.5.3 through 3.3.5

## Requirements

### Functional Requirements

- **FR-001**: The C# compositor MUST emit a `terrain_shadow_256` signal (256x256 float32, single channel, values in [0, 1]) containing the textureless lighting term (Lambert N·L + ambient + cast shadows) for every tile it synthesizes.
- **FR-002**: The `terrain_shadow_256` signal MUST be deterministic — the same tile with the same lighting profile must produce the same shadow signal.
- **FR-003**: The harvest pipeline MUST write `terrain_shadow_256` to the NPZ output alongside the existing `minimap_rgb`.
- **FR-004**: The V50 store builder MUST include `terrain_shadow_256` as a declared signal in the store manifest.
- **FR-005**: The curriculum builder MUST support selecting decomposed signals as training inputs and targets.
- **FR-006**: The model trainer MUST support `terrain_shadow_256` as an input channel (replacing or supplementing `minimap_rgb`).

### Non-Functional Requirements

- **NFR-001**: Adding `terrain_shadow_256` must not increase per-tile harvest time by more than 10% (the compositor already computes the lighting term; this is just writing it to a separate array).
- **NFR-002**: The decomposition must work for all client builds from 0.5.3 through 3.3.5.
- **NFR-003**: The model must be trainable on a single GPU in under 24 hours.

## Success Criteria

1. **Decomposition**: Every tile in the development corpus has a `terrain_shadow_256` signal that is demonstrably textureless (varies with normals and cast shadows, not with albedo).
2. **Curriculum**: A training curriculum exists with `terrain_shadow_256`, `minimap_rgb`, `normal_xyz`, and `height_257` as row-aligned signals.
3. **Model**: A shadow→height model beats the tile-mean baseline by at least 5% relative on held-out tiles.
4. **Reproducibility**: All results are reproducible from the same input data without manual intervention.

## Key Entities

### TerrainShadowSignal
- `tile_key`: string (map_tileX_tileY)
- `shadow_256`: float32 array (256x256) — the textureless lighting term
- `lighting_profile`: string — the lighting profile used (e.g., "CreateShadedTerrain(0.5)")
- `has_cast_shadows`: bool — whether cast shadows were applied
- `ambient`: float — the ambient term used
- `light_direction`: (float, float, float) — the light direction used

### DecomposedCurriculum
- `store_path`: string — path to the Zarr store
- `signals`: list of signal names (terrain_shadow_256, minimap_rgb, normal_xyz, height_257)
- `row_count`: int
- `split`: dict — train/val split metadata

## Assumptions

1. The existing compositor's lighting math (Lambert N·L + ambient + cast shadows) is the correct decomposition — no new lighting model is needed.
2. The textureless lighting term is a single-channel grayscale signal (the lighting is achromatic in the production profile).
3. The existing harvest pipeline (C# → NPZ → V50 Zarr store) is the right place to add the new signal.
4. The model architecture (U-Net-lite / MiT-B0) does not need to change — only the input channel count and signal selection.
5. The decomposition is a C# change only — no new Python code is needed for the signal emission (the Python side reads whatever the C# side writes).