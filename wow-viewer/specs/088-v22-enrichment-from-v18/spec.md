# Feature Specification: V22 Enrichment From V18

**Feature Branch**: `088-v22-enrichment-from-v18`
**Date**: 2026-06-30
**Status**: Draft
**Supersedes**: `086-v22-consolidated-dataset` (per-tile stream, never produced a populated store), `087-v22-asset-library-payloads` (per-tile stream with non-deterministic `Path.GetHashCode()` keys — see Risks)

**Input**: Specs 086 and 087 designed V22 as a parallel C# harvester stream that emits per-tile model payloads and a Python writer that dedupes them. The C# side never produced the three-message-class stream (tile / model-library / tileset-library) that the spec required. The Python writer existed but its `add_model()` / `add_tileset()` methods were never reached by a real producer. Result: zero populated `models/` or `tilesets/` groups in any V22 store, no successful end-to-end real-data build.

This spec takes a different shape. **V18 is the substrate. V22 is a V18-derived enrichment.**

V18 Zarr stores already ship 20 base signal arrays and a `placements.parquet` sidecar. They are produced by a working C# harvester and a working Python builder. We do not touch the V18 build path. Instead we add:

1. A new C# tool `WowViewer.Tool.V22Enrich` that takes a finished V18 store, walks its placement asset paths, opens the staged game client, decodes every unique M2 / WMO / BLP exactly once, and writes a stable-keyed binary enrichment stream. The stream is keyed by canonical asset path (via `M2ModelIdentity.NormalizePath`), not by `Path.GetHashCode()`.
2. A rewritten Python `scripts/build_v22_dataset.py` that reads a V18 store, derives the V22 patched signals in pure Python, promotes V18 placements to native V22 flat arrays, consumes the enrichment stream, and writes a V22 Zarr store. No C# Zarr implementation. No Python client reparse. The C# side only emits the enrichment stream; the Python side owns the Zarr store.

**Client Scope**: `0_5_3_3368`, `3_3_5_12340`, `4_0_0_11927`. Same scope as Spec 086. Cata `4_0_0_11927` stays in scope because the development map references Cata-only assets. Expansion beyond these three builds requires reopening this spec.

**Repo Boundary**: All work is in `wow-viewer/`. The new C# tool lives at `wow-viewer/tools/enrich/WowViewer.Tool.V22Enrich/`. The Python builder stays at `wow-viewer/data-harvester/scripts/build_v22_dataset.py`. The Python Zarr I/O stays at `wow-viewer/data-harvester/src/harvester/v22_zarr_io.py`.

---

## User Stories

### User Story 1 — V18 Substrate Survives (Priority: P1)

As a dataset consumer, my existing V18 training scripts (`train_v18.py`, `train_v18_focus.py`, Spec 047, Spec 077) keep working without changes. The V18 Zarr layout and `placements.parquet` sidecar remain the canonical training surface.

**Why this priority**: V18 is the live training substrate. Touching it would invalidate every spec that consumes it (047, 074, 075, 076, 077, etc.) and every checkpoint trained against it.

**Independent Test**: After V22 enrich tooling lands, run `train_v18.py normal --builds 3_3_5_12340 --curation-manifest ...` against the existing V18 store and confirm the same training behavior. V18 build output is bit-for-bit identical to pre-spec-088 output.

**Acceptance Scenarios**:
1. **Given** an existing V18 Zarr store built before this spec, **When** the V18 build is rerun, **Then** the resulting store is byte-identical to the pre-spec store.
2. **Given** the V18 builder, **When** the V22 enrich tool is added, **Then** `WowViewer.Tool.Harvest`, `build_v18_dataset.py`, and the V18 trainers are not modified.

### User Story 2 — V22 Build From V18 + Enrichment (Priority: P1)

As a dataset builder, I can produce a V22 Zarr store by running two commands: `WowViewer.Tool.V22Enrich` against a V18 store, then `build_v22_dataset.py build --v18-store ... --enrichment ... --output ...`.

**Why this priority**: This is the entire V22 contract. Without this, the spec is unfulfilled.

**Independent Test**: A bounded V22 build on a single V18 tile produces a Zarr store with `tile_count = 1`, all 20+5 root arrays present, and `models/` / `tilesets/` groups populated for every unique M2/WMO/BLP the tile references.

**Acceptance Scenarios**:
1. **Given** a V18 store for `3_3_5_12340` Azeroth with 1 tile, **When** the V22 enrich tool runs, **Then** the enrichment stream contains one M2/WMO/BLP entry per unique asset path referenced by the tile's placements.
2. **Given** the same V18 store + enrichment stream, **When** the rewritten `build_v22_dataset.py build` runs, **Then** the V22 Zarr store has `tile_count = 1`, all root arrays non-empty, `models/model_paths.shape[0] > 0`, and `tilesets/tileset_paths.shape[0] > 0`.
3. **Given** a V18 tile with zero placements, **When** V22 is built, **Then** the V22 store has the tile, empty native placement arrays with correct second dimension (`(0, 9)` and `(0, 17)`), and empty `models/` / `tilesets/` groups (no crash).

### User Story 3 — Stable Per-Build Library Keys (Priority: P1)

As a downstream model consumer, I can resolve `mddf_model_ids[i]` and `mcly_tileset_ids[i]` against the `models/model_paths` and `tilesets/tileset_paths` string arrays without rereading MPQ files or recomputing hashes.

**Why this priority**: This is the failure mode of Spec 087. `Path.GetHashCode()` is randomized per process in .NET 6+; the same model gets different keys every harvest run, so dedup across runs is impossible. We avoid this by keying the enrichment stream and the Zarr store by canonical path.

**Independent Test**: Run V22 enrich twice on the same V18 store, then run V22 build twice. `models/model_paths[0]` is byte-identical between the two runs.

**Acceptance Scenarios**:
1. **Given** a V18 store with 100 tiles referencing 50 unique M2 paths, **When** the V22 store is built, **Then** `models/model_paths.shape[0] = 50` and every `mddf_model_ids` value is a valid index into `models/model_paths`.
2. **Given** the same V18 store, **When** V22 is built twice in a row, **Then** both V22 stores have the same `models/model_paths` ordering and the same `mddf_model_ids` per tile.
3. **Given** a placement referencing a corrupt M2 file, **When** the enrich tool runs, **Then** the resulting `models/<id>` entry has `load_error = 1`, zero-length geometry arrays, and the build does not crash.

### User Story 4 — Pure-Python V22 Patched Signals (Priority: P2)

As a dataset builder, the V22 patched signals (`mcnr_mask_257`, `liquid_type_256`, `ground_intent_height_257`, `model_focus_mask`, `model_above_terrain_mask`) are derived in pure Python from the V18 store. No C# re-run needed.

**Why this priority**: These signals are required by the V22 contract, but they are deterministic transforms of V18 arrays. Doing them in Python keeps the enrich tool focused on the C#-only work (M2/WMO/BLP decode) and the build step self-contained.

**Independent Test**: The V22 store's `ground_intent_height_257` for tile N matches the value computed by a standalone Python helper that takes V18 `height_257` and `object_precise_mask` for tile N.

**Acceptance Scenarios**:
1. **Given** a V18 store, **When** V22 build runs, **Then** every V22 root array exists in the output store with the documented shape and dtype, even if the V18 source array is missing (zero-filled).
2. **Given** a tile where `mcnr_mask_257` is present in V18, **When** V22 build runs, **Then** V22's `mcnr_mask_257` is byte-identical to V18's.
3. **Given** a tile where V18's `object_precise_mask` has object pixels, **When** V22 build runs, **Then** `ground_intent_height_257` has those pixels inpainted from neighbouring `height_257` values (matches the C# reference algorithm in `RawArraySerializer.BuildGroundIntentHeight257`).

### User Story 5 — V18 Placements Promoted To Native V22 Arrays (Priority: P2)

As a downstream consumer, I can read placement data directly from V22 native arrays without opening `placements.parquet`. The V22 arrays are `(total, 9)` for MDDF and `(total, 17)` for MODF with per-tile offsets.

**Why this priority**: The spec 086 V22 contract requires native placement arrays. V18 stores them in `placements.parquet`; V22 must promote them.

**Independent Test**: `V22Dataset[i]['mddf_placement_data'].shape == (tile_mddf_count, 9)` and the rows match `placements.parquet[tile_id == i]` row-for-row.

**Acceptance Scenarios**:
1. **Given** a V18 store with `placements.parquet`, **When** V22 build runs, **Then** `mddf_placement_data.shape[0] == sum(per-tile mddf_count)`, `modf_placement_data.shape[0] == sum(per-tile modf_count)`, and per-tile offsets are correct.
2. **Given** a placement row, **When** V22 build runs, **Then** `placement_mddf_asset_paths[row]` and `placement_modf_asset_paths[row]` resolve to canonical paths (no `nameId`-only references).

### User Story 6 — V22 Bounded Real-Data Proof (Priority: P2)

As a dataset builder, V22 produces a non-empty store on real staged client data. Bounded proof on `3_3_5_12340` Azeroth, then `0_5_3_3368`, then `4_0_0_11927`.

**Why this priority**: Specs 086 and 087 never got past this. Phase 1+2 work is moot without a real store.

**Independent Test**: `inspect_v22_dataset.py summary --store <bounded.zarr>` reports `tile_count > 0`, `model_count > 0`, `tileset_count > 0`.

**Acceptance Scenarios**:
1. **Given** staged `3_3_5_12340` Azeroth V18 store, **When** V22 enrich + build runs, **Then** the V22 store passes a coverage gate: every `mddf_placement_data` row references a real `models/model_paths` entry, and every `mcly_tileset_ids` value >= 0 references a real `tilesets/tileset_paths` entry.
2. **Given** the V22 store, **When** `inspect_v22_dataset.py tile` runs on tile 0, **Then** per-array shape, dtype, and `nonzero_count` look sane (not all zeros except `height_257` and `minimap_rgb`).

---

## Requirements

### Foundational — V18 Substrate Untouched

- **FR-001**: The V18 build path (`WowViewer.Tool.Harvest harvest-stream`, `build_v18_dataset.py build`) MUST NOT be modified by this spec.
- **FR-002**: The V18 Zarr store MUST remain a complete training substrate. V22 enrichment is a downstream consumer of V18, not a replacement.
- **FR-003**: Existing V18 trainers (`train_v18.py`, `train_v18_focus.py`) MUST continue to read V18 stores without changes.
- **FR-004**: `WowViewer.Core.IO.Maps.RawArraySerializer` V22 profile (the per-tile model payload emission in `WriteV22Arrays`) MUST be reverted or marked deprecated. The V22 stream profile in its current form is unsafe (non-deterministic `Path.GetHashCode()` keys, per-tile duplication).

### New C# Tool: `WowViewer.Tool.V22Enrich`

- **FR-005**: A new CLI tool `WowViewer.Tool.V22Enrich` MUST be created at `wow-viewer/tools/enrich/WowViewer.Tool.V22Enrich/`.
- **FR-006**: The tool MUST accept `--v18-store <path>`, `--client-root <path>`, `--output <stream-path>`, and `--build-key <name>`.
- **FR-007**: The tool MUST read `placements.parquet` from the V18 store, collect the set of unique canonical M2 paths, WMO paths, and BLP paths across all tiles.
- **FR-008**: For each unique M2 path, the tool MUST open the file from the staged client (via the same MPQ / filesystem resolver used by the harvester), call `WowViewer.Core.IO.M2.M2GeometryReader.Read` and `WowViewer.Core.IO.M2.M2SkinReader.Read`, and flatten the result into V22 model payload arrays: `vertices` (N, 3), `normals` (N, 3), `texcoords_0` (N, 2), `texcoords_1` (N, 2), `bone_indices` (N, 4) uint8, `bone_weights` (N, 4), `triangles` (M, 3) int32 (from companion `.skin`), `render_flags` (R,) uint32, `blend_modes` (R,) uint8, `texture_lookup` (T,) uint16, `texture_paths` (P,) string, `texture_replaceable_ids` (P,) uint32, `texture_flags` (P,) uint32, `transparency_lookup` (R,) uint16, `bone_lookup` (B,) uint16, `bounds` (2, 3) float32.
- **FR-009**: For each unique WMO path, the tool MUST call `WowViewer.Core.IO.Wmo.WmoRenderDocumentReader.Read` and flatten to: `vertices` (N, 3), `triangles` (M, 3), `normals` (N, 3), `group_counts` (G,) int32, `group_indices` (G,) int32, `materials` (K, 8) int32, `material_texture_paths` (P,) string, `bounds` (2, 3) float32, `portal_vertices` (PV, 3), `portal_indices` (PI, 3), `doodad_set_paths` (DS,) string, `flags` scalar uint32, `version` scalar uint32.
- **FR-010**: For each unique BLP path, the tool MUST call `WowViewer.Core.IO.Blp.AlphaBlpCompatibilityService` to decode the BLP to `Image<Rgba32>` and serialize to a contiguous RGB uint8 byte array. Shape and dimensions recorded alongside.
- **FR-011**: Models and BLPs that fail to decode MUST be emitted with `load_error = 1` and zero-length arrays. The tool MUST NOT crash on a corrupt or missing asset.
- **FR-012**: The enrichment stream MUST use a stable binary format keyed by canonical asset path. The key MUST be the original canonical path string (e.g. `World/M2/Peasant.m2`), not `Path.GetHashCode()`. The stream must be a sequence of `ENTRY` records: `[ENTRY magic][path_len][path_utf8][kind][load_error][array_count][for each array: name_len, name_utf8, ndim, shape, dtype, data_len, data_bytes][ENDS]`. An outer `ENDS` terminates the stream.
- **FR-013**: The same canonical path emitted twice in the same enrichment run MUST appear only once. Dedup is by canonical path, not by hash.
- **FR-014**: The tool MUST resolve staged build directories like `output/tmp/wowarchive-clients/3_3_5_12340` to the nested `World of Warcraft` game root when that nested directory exists (matching the existing `WowViewer.Tool.Harvest` resolution).
- **FR-015**: The tool MUST use `WowViewer.Core.IO.Maps.M2ModelIdentity.NormalizePath` (or equivalent) for canonical path normalization. Same input path always produces the same canonical path.
- **FR-016**: Asset reads from MPQ MUST use the same `assetReader` callback pattern already used by `AdtTensorPackBuilder` (a `Func<string, byte[]?>` that resolves a path to raw bytes), so the same path-resolution seam handles staged-client MPQ, filesystem, and archive-backed inputs.
- **FR-017**: The tool MUST set a real non-zero process exit code on failure and emit a human-readable error to stderr.

### Python Builder: `build_v22_dataset.py` (Rewritten)

- **FR-018**: The Python builder MUST accept `build --v18-store <path> --enrichment <stream-path> --output <zarr-path>` and `enrich --v18-store <path> --client-root <path> --enrichment-output <path>` subcommands. The `enrich` subcommand invokes `WowViewer.Tool.V22Enrich` and is a convenience wrapper; the `build` subcommand consumes a pre-built enrichment stream.
- **FR-019**: The builder MUST read the V18 store as the substrate. The 20 V18 base arrays become the V22 root arrays directly. Missing source arrays produce zero-filled V22 root arrays (no `has_*` branching).
- **FR-020**: The builder MUST derive the V22 patched signals in pure Python from the V18 store:
  - `mcnr_mask_257` (bool 257×257) — V18 already has this as `mcnr_mask_257`; copy directly. If missing, derive as `(x % 2 == y % 2)` checkerboard.
  - `liquid_type_256` (uint8 256×256) — derive from V18's `liquid_basic_type_257` (or `unified_liquid_mask` + `mh2o_type_mask`), 257→256 crop, 0xFF → 0, others + 1. Reference: `RawArraySerializer.BuildLiquidType256` at `wow-viewer/src/core/WowViewer.Core.IO/Maps/RawArraySerializer.cs:382-400`.
  - `ground_intent_height_257` (float32 257×257) — inpaint `height_257` over `object_precise_mask` using a 4-neighbour Laplace fill (max H+W iterations, fall through if no progress). Reference: `RawArraySerializer.BuildGroundIntentHeight257` at lines 418-495.
  - `model_focus_mask` (float32 257×257) — alias of `object_filtered_mask` (copy).
  - `model_above_terrain_mask` (float32 257×257) — for each MDDF/MODF placement in the tile, project `(posX, posY, posZ)` to a tile pixel using the four candidate projections in `RawArraySerializer.BuildModelAboveTerrainMask` at lines 555-629, set the pixel to 1.0 if `posZ >= height[py, px] - 1.0`.
- **FR-021**: The builder MUST read `placements.parquet` from the V18 store and promote it to native V22 placement arrays: `mddf_placement_data` (total, 9) float32 with `mddf_count` (N,) int32 and `mddf_placement_offset` (N,) int64; same for MODF at 17 columns (V18's 14-col MODF rows are expanded to 17 with zero-fill for `flags`/`doodadSet`/`nameSet` when missing — match `RawArraySerializer.ConvertModfPlacementDataToV22` at lines 506-538). Per-row `placement_mddf_asset_paths` and `placement_modf_asset_paths` are read from V18's `placement_mddf_data` / `placement_modf_data` rows (V18 already records resolved `asset_path` per row at `build_v18_dataset.py:1125-1148`).
- **FR-022**: The builder MUST consume the enrichment stream produced by `WowViewer.Tool.V22Enrich` and write a `models/` group with one entry per unique asset path. Each entry contains the FR-008 (M2) or FR-009 (WMO) arrays. `models/model_paths` is a string array; `models/model_kind` is uint8 (0=unknown, 1=M2, 2=WMO); `models/load_error` is uint8 per entry.
- **FR-023**: The builder MUST consume the enrichment stream's BLP entries and write a `tilesets/` group with one entry per unique BLP path. Each entry contains `texture_rgb` (H, W, 3) uint8 and `texture_shape` (2,) int32. `tilesets/tileset_paths` is a string array; `tilesets/load_error` is uint8 per entry.
- **FR-024**: The builder MUST write `mcly_tileset_ids` (N, 16, 16, 4) int32 by remapping each tile's `mcly_texture_ids` (which is tile-local MTEX index) through the build-wide `tilesets/tileset_paths` index. Each tile's `mtex_texture_paths` (read from V18's per-tile `metadata.mtex_texture_paths` or the row's `mtex_texture_names`) is used to build the per-tile index. Unused layers are `-1`.
- **FR-025**: The builder MUST write `mddf_model_ids` (total_mddf,) int32 and `modf_model_ids` (total_modf,) int32 by resolving each placement's `asset_path` to the index in `models/model_paths`. Placements with no resolvable path get `-1`.
- **FR-026**: The builder MUST write `mddf_unique_ids` (total_mddf,) int32 and `modf_unique_ids` (total_modf,) int32 by reading V18's `placement_mddf_data` / `placement_modf_data` column 1 (uniqueId).
- **FR-027**: The builder MUST keep `index.parquet`, `placements.parquet` (audit copy), and `asset_inventory.parquet` as audit-only sidecars outside the `.zarr` store, in the style of V18.
- **FR-028**: The builder MUST emit `finalization.json` recording array existence, model count, tileset count, and the source V18 store path. The store is only considered "finalized" when `finalization.json` reports zero missing components.
- **FR-029**: The builder MUST set a real non-zero process exit code on any failure (empty V18 store, missing enrichment stream, parse error).

### `V22ZarrWriter` Python Module

- **FR-030**: `wow-viewer/data-harvester/src/harvester/v22_zarr_io.py` MUST be refactored to read the new enrichment stream and V18 store, not the broken per-tile V22 stream.
- **FR-031**: `V22Dataset` MUST return the same fixed-key contract as before (FR-006 in the old spec). All 20+5 root arrays, plus `mddf_placement_data`, `modf_placement_data`, counts, offsets, unique_ids, model_ids, and `mcly_tileset_ids`. Empty tiles return zero-length placement arrays with the correct second dimension.
- **FR-032**: `V22ZarrWriter.add_tile` MUST be replaced by a new API: `V22ZarrWriter.add_from_v18(v18_path: Path, enrichment_path: Path)`. The new method reads the V18 store, derives the patched signals, promotes placements, and consumes the enrichment stream.

### `inspect_v22_dataset.py`

- **FR-033**: `wow-viewer/data-harvester/scripts/inspect_v22_dataset.py` MUST continue to provide `summary` and `tile` subcommands, reading the new V22 store layout. `summary` reports `tile_count`, `builds`, root array layout, `model_count`, `tileset_count`. `tile` reports per-tile metadata and per-array shape / dtype / nonzero count / min / max / mean.

### Testing

- **FR-034**: A pytest test MUST round-trip a synthetic V22 store with a mock enrichment stream (a few M2 / WMO / BLP entries) and assert the V22 store has the expected `models/` and `tilesets/` contents.
- **FR-035**: A pytest test MUST derive the V22 patched signals from a synthetic V18 store and assert they match the documented fill values and shapes.
- **FR-036**: An xUnit test MUST parse the enrichment stream format (round-trip a few entries) and assert stable-path keying.
- **FR-037**: A bounded real-data proof MUST run end-to-end on staged `3_3_5_12340` Azeroth with `--limit 1` (one tile) and produce a non-empty V22 store. The proof records the exact staged client path and output store path.

### Specs To Archive

- **FR-038**: Spec `086-v22-consolidated-dataset` MUST be moved to `wow-viewer/specs/archived/` with rationale "design was per-tile stream with broken three-message-class producer; no real store was ever produced; superseded by 088 which uses V18 as substrate and a separate enrich tool."
- **FR-039**: Spec `087-v22-asset-library-payloads` MUST be moved to `wow-viewer/specs/archived/` with rationale "inherits 086's per-tile design; uses non-deterministic `Path.GetHashCode()` as the model key, breaking cross-run dedup; superseded by 088 which uses canonical path keys via the V18 enrich tool."

---

## Success Criteria

- **SC-001**: A bounded V22 build on one `3_3_5_12340` Azeroth tile produces a Zarr store with `tile_count = 1`, all root arrays present, `models/model_paths.shape[0] > 0`, `tilesets/tileset_paths.shape[0] > 0`, and per-row placement ↔ model id round-trip verified.
- **SC-002**: V22 stores built twice from the same V18 store + same enrichment stream have byte-identical `models/model_paths` and `mcly_tileset_ids` per tile.
- **SC-003**: V18 trainers (`train_v18.py`, `train_v18_focus.py`) keep working on V18 stores produced alongside V22 stores, with no spec-088 changes to the V18 build or trainer paths.
- **SC-004**: The `WowViewer.Tool.V22Enrich` tool reads a real V18 store and produces a non-empty enrichment stream for `3_3_5_12340` Azeroth with at least one M2 entry (the development map's only placed M2 / WMO will do).
- **SC-005**: A corrupt or missing M2 in the V18 store's referenced assets produces a `models/<path>` entry with `load_error = 1` and zero-length geometry arrays. The build does not crash.
- **SC-006**: V22 root array existence and shape / dtype checks pass for every documented array, including zero-fill behavior for V18 tiles missing `mcnr_mask_257` or other derived inputs.
- **SC-007**: The Python builder's `models/model_paths` count for a real `3_3_5_12340` tile matches the count of unique M2 paths in the tile's placements (no duplicates, no missing).
- **SC-008**: Specs 086 and 087 are moved to `archived/` and `ARCHIVED.md` is updated.

---

## Assumptions

- V18 Zarr stores always carry `placements.parquet` (or equivalent columnar placement data) with `asset_path` resolved per row. Confirmed: `build_v18_dataset.py:1122-1148` writes per-row `asset_path` from `placement_mddf_names[mddf_data[i, nameId]]`.
- V18's `mcnr_mask_257` is present (the V18 build emits it; if it isn't, derive from checkerboard).
- `WowViewer.Core.IO.M2.M2GeometryReader`, `WowViewer.Core.IO.M2.M2SkinReader`, `WowViewer.Core.IO.Wmo.WmoRenderDocumentReader` all produce the FR-008 / FR-009 fields without modification. Confirmed by reading the reader headers; all required fields are already exposed via `M2GeometryDocument`, `M2SkinDocument`, and `WmoRenderDocument`.
- `AlphaBlpCompatibilityService` decodes BLP to `Image<Rgba32>` via `SereniaBLPLib`. A small helper `BlpRgbReader` that takes `(byte[], path) -> (int H, int W, ndarray uint8 RGB)` can be added in this spec.
- `M2ModelIdentity.NormalizePath` is the canonical path key. Same path always normalizes the same way.
- Per-build model and tileset libraries fit in memory: < 10K unique M2 + WMO, < 1K unique BLP. Confirmed for Azeroth scope (Spec 047 reports ~764 MDDF + 7 MODF unique models).
- Stage boundaries (Spec 047 + Spec 086) constrain the dataset to `0_5_3_3368`, `3_3_5_12340`, `4_0_0_11927`. This spec inherits that scope.
- The C# harvester's existing asset reader pattern (`Func<string, byte[]?>`) already handles staged-client MPQ + filesystem fallback. The enrich tool reuses this seam.

---

## Risks

- **Risk 1 (mitigated):** The old V22 stream profile in `RawArraySerializer.WriteV22Arrays` still emits per-tile model payloads with `Path.GetHashCode()` keys. We will revert that block as part of this spec (FR-004). The reversion does not affect the V18 builder (which uses the V16 profile or Full profile, not V22).
- **Risk 2:** V18 may have placements referencing assets that don't exist in the staged client (development-map asset gaps). The enrich tool must report missing assets and emit `load_error = 1` entries, not crash. (FR-011, FR-017 cover this.)
- **Risk 3:** WMO and BLP asset paths in the staged client may use backslashes or case differences vs V18's `asset_path`. The enrich tool must normalize paths consistently (FR-015).
- **Risk 4:** `RawArraySerializer.WriteV22Arrays` already produces partial V22 data (per-tile model payloads). If the V18 builder ever gets called with `--stream-profile v22`, the per-tile model payloads would still be emitted and could confuse a downstream consumer. After FR-004, the V22 profile should either be reverted to V16-equivalent or be removed entirely. Recommend removal to prevent accidental re-introduction.
- **Risk 5:** The V22 reader (`V22Dataset`) currently has a fixed-key contract that returns zeros for missing per-tile signals. The rewritten writer must keep that contract (FR-031). The existing test suite in `test_v22_zarr_io.py` is the safety net.
- **Risk 6:** Python-side BLP→RGB decode is not feasible (the Python data-harvester has no BLP reader). The architecture avoids this: the C# enrich tool does the decode, the Python side only receives already-decoded RGB bytes. Confirmed by design.
- **Risk 7:** The `placements.parquet` schema in V18 stores may not carry every V22 placement column. Confirmed via `build_v18_dataset.py:505-511`: V18 writes 9-col MDDF (matches V22) and 14-col MODF (V22 needs 17). FR-021 mandates the 14→17 expand with zero-fill, matching `ConvertModfPlacementDataToV22` in the C# side.
- **Risk 8:** `placement_mddf_asset_paths` / `placement_modf_asset_paths` in V22 must be canonical paths, not `nameId`s. V18's `placements.parquet` rows already carry `asset_path` (resolved at V18 build time from `placement_mddf_names[mddf_data[i, nameId]]` per `build_v18_dataset.py:1125-1148`). FR-021 mandates the builder read that column.

---

## Key Entities

- **V18 Zarr store**: Substrate. Located at `output/datasets/v18/<build>.zarr/`. Carries 20 base signal arrays + `mcnr_mask_257` (already in V18) + `placements.parquet` sidecar. Read-only for this spec.
- **Enrichment stream**: New artifact. Binary file produced by `WowViewer.Tool.V22Enrich`. Carries one `ENTRY` record per unique M2 / WMO / BLP path. Stable path keys. Format defined in FR-012.
- **V22 Zarr store**: Output. Located at `output/datasets/v22/<build>.zarr/`. Carries V18-derived root arrays + V22 patched signals + V22 native placement arrays + `models/` group + `tilesets/` group + `mcly_tileset_ids`.
- **Canonical asset path**: A normalized game-client-relative path (e.g. `World/M2/Peasant.m2`). Produced by `M2ModelIdentity.NormalizePath`. Stable across processes and runs.
- **`WowViewer.Tool.V22Enrich`**: New C# CLI tool. Thin wrapper around the existing C# readers. No new format parsing. Output: enrichment stream.
- **`build_v22_dataset.py`**: Rewritten Python builder. Reads V18 store, reads enrichment stream, writes V22 Zarr store. No game client reparse, no C# Zarr implementation.

---

## Out Of Scope

- Editing or augmenting the V18 builder or trainers. (Spec 047, 074, 075, 076, 077 all consume V18; this spec does not touch those surfaces.)
- A new C# V22 stream profile. The V22 stream profile is being deprecated (FR-004); V22 is now built from V18 + an enrichment stream, not from a parallel C# stream.
- Multi-build aggregation. Each V22 store is for a single build session. Cross-build training already works via Spec 047.
- Vulkan / WebGL / Unreal integration. Per the constitution, those are out of scope.
- Renderer-truth arrays (`object_visibility_mask`, `no_object_minimap`). Spec 086 FR-028 required these; this spec deprecates them. If a future spec needs them, the V18 store already has the source data (per `EXPERIMENTAL_RENDERER_TRUTH_SIGNAL_KEYS` in `build_v18_dataset.py:99-102`); promoting them is a small follow-up.
- Replacing the existing `data-harvester/tests/test_v22_zarr_io.py`. The test is the round-trip safety net; the new writer keeps the same fixed-key contract.
