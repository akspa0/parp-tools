# Data Model: V22 Enrichment From V18

This document describes the three schemas that Spec 088 touches. The V22 Zarr output layout is defined canonically in [`docs/architecture/v22-dataset-signals-2026-06-30.md`](../../docs/architecture/v22-dataset-signals-2026-06-30.md) — this document is the data-flow view, not the schema reference.

## 1. V18 Zarr Store (Input)

**Path**: `output/datasets/v18/<build>.zarr/`

**What we read**:
- All 20 base V18 root arrays (per V18 build). They become the V22 root arrays verbatim.
- `mcnr_mask_257` if present (else derived as checkerboard).
- `placements.parquet` (columnar, sidecar). The V22 builder promotes this to native V22 flat arrays.
- `index.parquet` (per-tile metadata). The V22 builder copies this as an audit sidecar.

**What we do NOT touch**: the V18 build path, the V18 trainers, the V18 consumers (Spec 047, 074, 075, 076, 077, etc.).

### 1.1 V18 `placements.parquet` schema

```text
instance_type:    string  ("mddf" or "modf")
instance_idx:     int
nameId:           int     (legacy, kept for parity with V18 trainers)
uniqueId:         int
posX, posY, posZ: float
rotX, rotY, rotZ: float
scale:            float
asset_path:       string  (canonical asset path, resolved from placement_mddf_names[mddf_data[i, nameId]] at V18 build time)
```

For MODF rows, additional columns: `bbMinX, bbMinY, bbMinZ, bbMaxX, bbMaxY, bbMaxZ` (14 columns total). The V22 builder expands MODF to 17 columns by zero-filling `flags`, `doodadSet`, `nameSet` (matching the C# reference at `RawArraySerializer.ConvertModfPlacementDataToV22`).

## 2. Enrichment Stream (Intermediate)

**Path**: `output/tmp/v22_enrich/<build>.bin` (debug seam, not part of the V22 contract)

**Format**: length-prefixed binary, written by `WowViewer.Tool.V22Enrich`, read by `build_v22_dataset.py`.

### 2.1 Stream Layout

```text
[Header]
  'V22E'  magic (4 bytes ASCII)
  uint32  version (little-endian, currently 1)

[One or more ENTRY records]
  'ENTRY' magic (4 bytes ASCII)
  uint32  path_len
  bytes   path_utf8          (canonical asset path; no GetHashCode)
  uint8   kind               (0=unknown, 1=M2, 2=WMO, 3=BLP)
  uint8   load_error         (0=success, 1=decode failed)
  uint32  array_count
  for each array:
    uint32  name_len
    bytes   name_utf8
    uint32  ndim
    uint32 × ndim  shape
    bytes   dtype (8 bytes ASCII, null-padded)
    int64   data_len
    bytes   data

[Outer terminator]
  'ENDS'  magic (4 bytes ASCII)
```

### 2.2 Per-kind Fields

The `array_count` is the number of FR-008 (M2) / FR-009 (WMO) / tileset (BLP) arrays for that entry. The Python reader reconstructs the field list from the `name_utf8` of each array.

**M2 entry arrays** (FR-008): `vertices`, `normals`, `texcoords_0`, `texcoords_1`, `bone_indices`, `bone_weights`, `triangles` (from skin), `render_flags`, `blend_modes`, `texture_lookup`, `texture_paths`, `texture_replaceable_ids`, `texture_flags`, `transparency_lookup`, `bone_lookup`, `bounds`.

**WMO entry arrays** (FR-009): `vertices`, `triangles`, `normals`, `group_counts`, `group_indices`, `materials`, `material_texture_paths`, `bounds`, `portal_vertices`, `portal_indices`, `doodad_set_paths`, `flags`, `version`.

**BLP entry arrays**: `texture_rgb` (H, W, 3 uint8), `texture_shape` (2,) int32.

### 2.3 Dedup Contract

The same canonical path emitted twice in the same enrichment run produces two `ENTRY` records. The Python reader dedups by canonical path on first-seen-wins. The C# writer does NOT dedup at write time — the caller's responsibility.

Across runs of the same V18 store + the same client, the stream is byte-deterministic (assuming the C# writer's enumeration order is stable). The Python reader accumulates into a `dict[canonical_path, entry]` keyed by string equality.

## 3. V22 Zarr Store (Output)

**Path**: `output/datasets/v22/<build>.zarr/`

### 3.1 Root Arrays (V18-derived + V22-patched)

The 20 V18 base arrays become the V22 root arrays. The 5 V22-patched signals are derived in pure Python from the V18 inputs:

| Array | Source | Notes |
|-------|--------|-------|
| `height_257` | V18 `height_257` | direct copy |
| `normal_xyz` | V18 `normal_xyz` | direct copy |
| `normal_mask` | V18 `normal_mask` | direct copy |
| `alpha_256` | V18 `alpha_256` | direct copy |
| `holes_16` | V18 `holes_16` | direct copy |
| `liquid_mask` | V18 `liquid_mask` | direct copy |
| `liquid_height` | V18 `liquid_height` | direct copy |
| `object_mask` | V18 `object_mask` | direct copy |
| `object_precise_mask` | V18 `object_precise_mask` | direct copy |
| `object_instance_mask` | V18 `object_instance_mask` | direct copy |
| `mcnk_flags_16` | V18 `mcnk_flags_16` | direct copy |
| `mddf_mask` | V18 `mddf_mask` | direct copy |
| `modf_mask` | V18 `modf_mask` | direct copy |
| `object_filtered_mask` | V18 `object_filtered_mask` | direct copy |
| `model_focus_mask` | V18 `object_filtered_mask` | alias |
| `model_above_terrain_mask` | derived | placements vs heightmap (FR-020) |
| `object_roof_mask` | V18 `object_roof_mask` | direct copy |
| `object_roof_confidence` | V18 `object_roof_confidence` | direct copy |
| `minimap_rgb` | V18 `minimap_rgb` | direct copy |
| `shadow_mask` | V18 `shadow_mask` | direct copy |
| `mcly_texture_ids` | V18 `mcly_texture_ids` | direct copy |
| `mcly_layer_mask` | V18 `mcly_layer_mask` | direct copy |
| `mcnr_mask_257` | V18 `mcnr_mask_257` or derived | fallback to checkerboard |
| `liquid_type_256` | derived | match `RawArraySerializer.BuildLiquidType256` |
| `ground_intent_height_257` | derived | inpaint `height_257` over `object_precise_mask` |
| `mddf_count`, `modf_count`, `mcly_tileset_ids` | per-tile scalars | derived |
| `mddf_placement_offset`, `modf_placement_offset` | flat int64 | prefix-sum |
| `mddf_placement_data` (total, 9) float32 | from V18 placements.parquet | promoted |
| `modf_placement_data` (total, 17) float32 | from V18 placements.parquet | 14→17 expand |
| `mddf_unique_ids`, `modf_unique_ids` (flat int32) | from V18 placements.parquet | promoted |
| `mddf_model_ids`, `modf_model_ids` (flat int32) | resolved against `models/model_paths` | remapped |

Missing source arrays produce zero-filled V22 root arrays. No `has_*` branches.

### 3.2 `models/` Group

```text
models/
├── model_paths     # string (num_models)
├── model_kind      # uint8  (num_models) 0=unknown, 1=M2, 2=WMO
├── load_error      # uint8  (num_models)
├── m2/<id>/        # one group per M2 model
│   ├── vertices
│   ├── normals
│   ├── texcoords_0
│   ├── texcoords_1
│   ├── bone_indices
│   ├── bone_weights
│   ├── triangles
│   ├── render_flags
│   ├── blend_modes
│   ├── texture_lookup
│   ├── texture_paths
│   ├── texture_replaceable_ids
│   ├── texture_flags
│   ├── transparency_lookup
│   ├── bone_lookup
│   └── bounds
└── wmo/<id>/       # one group per WMO model
    ├── vertices
    ├── triangles
    ├── normals
    ├── group_counts
    ├── group_indices
    ├── materials
    ├── material_texture_paths
    ├── bounds
    ├── portal_vertices
    ├── portal_indices
    ├── doodad_set_paths
    ├── flags
    └── version
```

The `<id>` is a string key derived from the canonical asset path (e.g. `World/M2/Peasant.m2` becomes `World/M2/Peasant_m2`). The same canonical path always produces the same `<id>`.

### 3.3 `tilesets/` Group

```text
tilesets/
├── tileset_paths    # string (num_tilesets)
├── load_error       # uint8  (num_tilesets)
├── texture_shape    # int32  (num_tilesets, 2)
└── texture_rgb/<id>/  # uint8 (H, W, 3)
```

### 3.4 Audit Sidecars (outside the `.zarr`)

```text
output/datasets/v22/<build>.zarr/
├── finalization.json           # build status, missing components
├── index.parquet               # copy of V18 index.parquet (audit)
├── placements.parquet          # copy of V18 placements.parquet (audit)
└── asset_inventory.parquet     # unique M2/WMO/BLP path counts per kind
```

## 4. Connection Summary

```text
V18 Zarr store (.zarr)            Enrichment stream (.bin)         V22 Zarr store (.zarr)
  + 20 base arrays          ──┐                                    + 20 base arrays
  + mcnr_mask_257             │                                    + 5 V22 patched signals
  + placements.parquet        │  WowViewer.Tool.V22Enrich         + native V22 placement arrays
                              ├─► (decodes each unique  ──►       + models/ group
                              │   M2/WMO/BLP once)                + tilesets/ group
                              │                                   + mcly_tileset_ids
                              │                                   + remapped mddf_model_ids
                              │                                   + remapped modf_model_ids
                              │                                   + audit sidecars
```

The two tools are independent. The Python `build_v22_dataset.py build` command does not shell out to the C# enrich tool — that is the `enrich` subcommand's job. This split lets operators debug the enrichment step in isolation (e.g. inspect the stream without parsing the Zarr).
