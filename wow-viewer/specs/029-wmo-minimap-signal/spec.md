# Feature Specification: WMO Minimap BLP Harvest and Asset Signal

**Feature Branch**: `029-wmo-minimap-signal`

**Created**: 2026-05-30

**Status**: Draft

**Input**: Ghidra RE of wowclient.exe build 3368 reveals the exact WMO minimap naming convention and query pipeline. Per-WMO-group minimap BLPs exist in the game client under `Textures\Minimap\<WMOName>_<groupIdx>_<quadY>_<quadX>.blp`, resolved through an MD5 name hash table. These are pre-authored top-down images for interior WMO groups and serve as ground truth for identifying WMO assets by their visual footprint. The user wants to extract these BLPs, track asset/group/WMO provenance, and store them alongside the existing object roof datastore for downstream model consumption.

## Problem Statement

The current object roof capture approach relies on GPU rendering via MdxViewer, which has been repeatedly broken by regressions. An alternative ground-truth source exists: the game client ships pre-rendered minimap BLPs for many WMO groups (especially interior/dungeon groups). These BLPs are:

1. **Authored by Blizzard** as top-down views of interior WMO groups — pure MPQ reads, no GPU required
2. **Named with a known pattern** (confirmed via Ghidra): `<WMOName>_<groupIndex>_<quadY>_<quadX>.blp`
3. **Resolved through the MD5 name hash table** (the `MINIMAPMD5NAME` struct loaded at minimap init)
4. **Queried by the client** via a portal-walk traversal starting from the player's current WMO group

These minimap BLPs are a valuable ground truth because:
- They identify which WMO asset and group produced a given minimap tile
- They provide a top-down view of interior geometry without any renderer dependency
- They can be used as exemplars for object-family identification
- They can be correlated with terrain minimap tiles to build WMO footprint masks

## Ghidra RE Findings (Build 3368)

### WMO Minimap Naming Convention

**Two naming patterns exist in the client:**

1. **Outdoor/terrain minimaps**: `%s\map%d_%d.blp` → `Textures\Minimap\<continent>\map<X>_<Y>.blp`
   - Continent name from DB lookup on continent ID
   - Resolved through `BuildPathName()`

2. **WMO interior minimaps**: `%s_%03d_%02d_%02d.blp` → `<WMOName>_<groupIndex>_<quadY>_<quadX>.blp`
   - Resolved through `SetupQuad()`
   - Final path: `Textures\Minimap\<resolved_name>.blp`
   - The format string parameters are: WMO stem name, group index (3-digit zero-padded), quad Y (2-digit), quad X (2-digit)

### WMO Minimap Query Pipeline

1. `MinimapUpdate()` detects player inside a WMO via `CWorld::QueryMapObjIDs()`
2. Dispatches to `CWorld::QueryMapObjMinimap()` → `CMapEntity::QueryMapObjMinimap()`
3. `CMapObj::QueryMapObjMinimapGroup()` performs a **portal-walk traversal**:
   - Starts from the group the player is in
   - Walks portal references to adjacent groups
   - Checks portal vertices are inside the query AABB
   - Recursively visits connected groups
4. `CMapObjGroup::QueryMinimap()` generates quad descriptors:
   - Skips groups with `flags & 0x88` (no-render / no-collide)
   - Calculates quad grid from group AABB size / tile size constant
   - Quad dimensions are power-of-2 clamped between 32 and 256
   - Emits `MinimapQuad { groupIndex, quadX, quadY, aaBox }`
5. `SetupQuad()` resolves each quad to a BLP file:
   - Formats filename as `<WMOName>_<groupIndex>_<quadY>_<quadX>.blp`
   - Looks up in `MINIMAPMD5NAME` hash table (loaded by `LoadMD5Names()`)
   - If found, rewrites path as `Textures\Minimap\<md5_resolved_name>`
   - Loads the BLP via `TextureCreate()`

### Key Insight: Quad Grid

The client tiles each WMO group's AABB into a quad grid. The quad size is derived from the group's world-space extents divided by a constant (`_DAT_00e6e4dc`), with power-of-2 clamping. This means a single WMO group can produce multiple minimap BLPs if it's large enough. The `(quadY, quadX)` coordinates identify which sub-tile of the group the BLP covers.

## Scope

### In Scope

- Discovering all WMO minimap BLPs in a staged client's MPQ archives by enumerating `Textures\Minimap\` entries matching the WMO naming pattern
- Decoding each BLP to RGB using existing BLP reader
- Parsing the WMO name, group index, quad Y, quad X from each BLP filename
- Resolving the WMO's full asset path from its stem name (by finding the matching `.wmo` root in the file list)
- Writing per-BLP metadata: asset path, group index, quad coordinates, group BLP path, group flags, image dimensions
- Storing results in a Zarr-compatible structure (either extending the object roof datastore or a new dedicated store)
- Per-WMO-group aggregation: combining multiple quads into a single group-level composite when multiple quads exist
- Cross-referencing with existing placement metadata (MODF entries) to tie minimap BLPs back to specific WMO instances on the map

### Out of Scope

- GPU rendering of any kind (this is a pure MPQ/archive read operation)
- M2 doodad captures (still require GPU or separate pipeline)
- WMO rendering improvements (separate concern, tracked elsewhere)
- DBC-chain resolution through WMOAreaTable/AreaTable/Map.dbc (the Ghidra evidence shows the client resolves minimap BLPs directly by filename pattern, not through the DBC chain — the DBC chain is used for outdoor minimap naming, not WMO interior minimaps)

## User Scenarios & Testing

### User Story 1 — WMO minimap BLPs are harvested from client archives (Priority: P1)

A data operator can run a tool that discovers and extracts all WMO minimap BLPs from a staged game client, producing per-group RGB images with full asset provenance, without any GPU rendering.

**Why this priority**: This is the foundational data extraction step. Without it, no downstream model can consume WMO minimap ground truth.

**Independent Test**: Run the harvester on build `3_3_5_12340`. Verify that at least one WMO (e.g., Deadmines, Stockades, or any dungeon WMO) has minimap BLPs found and decoded successfully.

**Acceptance Scenarios**:

1. **Given** a staged game client with MPQ archives, **When** the harvester runs, **Then** it enumerates all entries under `Textures\Minimap\` matching the WMO naming pattern `<stem>_<groupIdx>_<quadY>_<quadX>.blp`.
2. **Given** a matching BLP entry, **When** it is decoded, **Then** it produces a non-zero RGB image with known dimensions.
3. **Given** a WMO stem name extracted from a minimap BLP filename, **When** the file list is searched, **Then** the full `.wmo` root asset path is resolved.
4. **Given** a completed harvest run, **When** the metadata is inspected, **Then** each record carries `asset_path`, `wmo_stem`, `group_index`, `quad_y`, `quad_x`, `blp_path`, `image_shape`, `source`.
5. **Given** a WMO with multiple group minimap BLPs, **When** the metadata is aggregated, **Then** a per-group composite record exists that lists all quad BLPs for that group.

---

### User Story 2 — WMO minimap metadata is stored in a queryable Zarr-compatible structure (Priority: P1)

A researcher can load the WMO minimap metadata as a Parquet table and the images as Zarr arrays, enabling cross-referencing with placement data and the existing object roof library.

**Why this priority**: Without structured storage, the extracted data cannot be consumed by downstream training or curation pipelines.

**Independent Test**: Load the metadata Parquet and verify it can be joined with `placements.parquet` from the V16 dataset on `asset_path`.

**Acceptance Scenarios**:

1. **Given** a completed harvest, **When** the metadata Parquet is loaded, **Then** it has columns: `asset_path`, `wmo_stem`, `group_index`, `quad_y`, `quad_x`, `blp_path`, `image_width`, `image_height`, `source`, `build`.
2. **Given** the metadata Parquet, **When** joined with `placements.parquet` on `asset_path`, **Then** rows match for WMOs that appear in both datasets.
3. **Given** the image Zarr arrays, **When** indexed by row, **Then** each row's image matches the corresponding metadata record.
4. **Given** an existing object roof library, **When** the WMO minimap store is co-located, **Then** both stores can be loaded together without conflict.

---

### User Story 3 — WMO minimap BLPs serve as ground truth for asset identification (Priority: P2)

A curation pipeline can use WMO minimap BLPs as canonical top-down views to identify which WMO asset a given terrain minimap tile region belongs to, by correlating WMO footprint positions with terrain tile coordinates.

**Why this priority**: This is the downstream value — the data must be extracted first (P1) before it can be used for asset identification.

**Independent Test**: For a known WMO placement (e.g., Deadmines entrance on a specific tile), verify that the WMO minimap BLP's spatial footprint overlaps with the corresponding terrain minimap region.

**Acceptance Scenarios**:

1. **Given** a WMO placement from `placements.parquet` with position and rotation, **When** the WMO's group AABB is projected onto the terrain coordinate grid, **Then** the overlapping terrain minimap region can be identified.
2. **Given** an overlapping region, **When** the WMO minimap BLP is overlaid on the terrain minimap, **Then** the WMO footprint is visually consistent with the terrain minimap content.
3. **Given** a WMO minimap BLP, **When** its pixel content is analyzed, **Then** it can serve as an exemplar for that WMO's visual family in the object roof library.

---

### Edge Cases

- Some WMO minimap BLPs may have different sizes (not always 128×128) — the quad grid sizing in the client produces power-of-2 dimensions between 32 and 256.
- Some WMO groups have `flags & 0x88` set (no-render / no-collide) and will not have minimap BLPs.
- Some WMO stem names may not resolve to a `.wmo` root file in the archive (orphaned minimap textures).
- The MD5 name hash table (used by the client for resolution) is not directly available as a file — we must discover BLPs by enumerating the MPQ file list instead.
- Build-to-build differences: the naming pattern may vary slightly across builds (3-digit vs 2-digit group index padding, quad coordinate ordering). The harvester must be flexible.
- Some minimap BLPs may be shared across WMO groups (same texture used for multiple groups).

## Requirements

### Functional Requirements

- **FR-001**: The harvester MUST enumerate all files in the staged client's MPQ archives matching the pattern `Textures\Minimap\*_*_??_??.blp` (or equivalent case-insensitive glob).
- **FR-002**: The harvester MUST parse each matching filename to extract `wmo_stem`, `group_index`, `quad_y`, `quad_x` from the pattern `<wmo_stem>_<group_index>_<quad_y>_<quad_x>.blp`.
- **FR-003**: The harvester MUST resolve each `wmo_stem` to a full asset path by searching the MPQ file list for a `.wmo` root file whose basename (without extension) matches the stem.
- **FR-004**: The harvester MUST decode each BLP to an RGB array using the existing BLP reader (`SereniaBLPLib` or equivalent in `WowViewer.Core.IO`).
- **FR-005**: The harvester MUST write per-BLP metadata to a Parquet file with columns: `asset_path`, `wmo_stem`, `group_index`, `quad_y`, `quad_x`, `blp_path`, `image_width`, `image_height`, `source`, `build`.
- **FR-006**: The harvester MUST write decoded images to a Zarr array `wmo_minimap_rgb` of shape `(N, H, W, 3)` dtype `uint8`, where N is the number of discovered BLPs and H/W are the maximum dimensions (smaller images padded).
- **FR-007**: The harvester MUST produce a per-group aggregation record that lists all quad BLPs for each `(asset_path, group_index)` pair, enabling reconstruction of the full group minimap from its quad tiles.
- **FR-008**: The harvester MUST handle missing BLPs gracefully (skip entries that fail to decode, log errors, continue).
- **FR-009**: The harvester MUST handle filename pattern variations across builds (e.g., 3-digit group index padding in some builds, 2-digit in others).
- **FR-010**: All code MUST live under `wow-viewer/`.
- **FR-011**: The output store SHOULD be co-located with or compatible with the existing object roof library at `output/datasets/object_roof_library/`.
- **FR-012**: The metadata Parquet MUST be joinable with `placements.parquet` from V16 datasets on the `asset_path` column.

### Key Entities

- **WMO Minimap BLP**: A pre-authored top-down BLP texture for a WMO group's quad tile, stored at `Textures\Minimap\<WMOName>_<groupIdx>_<quadY>_<quadX>.blp` in MPQ archives.
- **MinimapQuad**: A descriptor `{ groupIndex, quadX, quadY, aaBox }` representing one sub-tile of a WMO group's minimap coverage.
- **WMO Group Composite**: The full minimap coverage for a WMO group, composed from its constituent quad BLPs.
- **MINIMAPMD5NAME**: The client's internal hash table struct that maps minimap BLP filenames to resolved MPQ paths. Not directly available as a file; we discover BLPs by MPQ enumeration instead.
- **WMO Minimap Zarr Store**: A Zarr-backed datastore containing per-BLP RGB images and metadata Parquet, co-located with or extending the object roof library.

## Success Criteria

- **SC-001**: Running the harvester on build `3_3_5_12340` discovers at least 100 WMO minimap BLPs.
- **SC-002**: At least 5 distinct WMO stem names have minimap BLPs found and decoded with non-zero RGB content.
- **SC-003**: The metadata Parquet can be joined with `placements.parquet` from the V16 dataset on `asset_path` and produces non-empty join results.
- **SC-004**: The entire harvest for one build completes in under 10 minutes.
- **SC-005**: Per-group aggregation records exist and correctly list all quad BLPs for each group that has multiple quads.

## Assumptions

- The existing MPQ reader in `WowViewer.Core.IO` can enumerate file entries in the `Textures\Minimap\` directory of staged client archives.
- The existing BLP reader can decode minimap BLPs (DXT compressed or uncompressed paletted).
- The WMO minimap BLP naming pattern `<stem>_<groupIdx>_<quadY>_<quadX>.blp` is consistent across the target builds (0.5.3, 3.0.1, 3.3.5). Ghidra confirmed this pattern for build 3368 (0.5.3).
- Not every WMO will have minimap BLPs — only interior/dungeon groups that the client's minimap system was configured to display.
- The MPQ file list can be filtered by prefix `Textures\Minimap\` to efficiently discover candidate BLPs.
- Some BLP filenames may not parse cleanly into the expected pattern — these should be logged and skipped rather than causing failures.

## Relationship to Other Specs

- **Replaces**: The original DBC-chain resolution approach described in the initial draft of spec 029. Ghidra RE shows the client resolves WMO minimap BLPs directly by filename pattern through the MD5 hash table, not through the WMOAreaTable/AreaTable/Map.dbc chain. The DBC chain is used for outdoor terrain minimap naming (`BuildPathName`), not WMO interior minimaps.
- **Extends**: `025-object-roof-mask-library-and-minimap-sieve` — WMO minimap BLPs serve as additional ground-truth exemplars for the object roof library, complementing GPU-rendered roof captures.
- **Complements**: Spec 025's T002 (object-capture seam) — WMO minimap BLPs provide an alternative, renderer-independent ground truth that does not require per-asset GPU capture.
- **Informs**: `023-v17-1-global-minimap-signal-reconstruction` — knowing which WMO minimap BLPs exist enables correlation of terrain minimap tiles with specific WMO assets for signal decomposition.
