# Implementation Plan: V22 Enrichment From V18

**Branch**: `088-v22-enrichment-from-v18` | **Date**: 2026-06-30 | **Spec**: [`spec.md`](spec.md)

## Summary

Spec 086 and 087 designed V22 as a parallel C# harvester stream that emits per-tile model payloads and a Python writer that dedupes them. The C# side never produced the three-message-class stream (tile / model-library / tileset-library) the spec required; the Python writer's `add_model` / `add_tileset` were never reached by a real producer. Result: zero populated `models/` or `tilesets/` groups in any V22 store, no end-to-end real-data build ever succeeded.

This spec takes a different shape. **V18 is the substrate, untouched. V22 is a V18-derived enrichment.** A new C# tool `WowViewer.Tool.V22Enrich` reads a finished V18 store, decodes every unique M2 / WMO / BLP exactly once, and writes a stable-keyed binary enrichment stream. A rewritten Python `build_v22_dataset.py` reads the V18 store + the enrichment stream, derives the V22 patched signals in pure Python, promotes V18 placements to native V22 arrays, and writes a V22 Zarr store. No C# Zarr implementation. No Python client reparse. Stable path keys (not `Path.GetHashCode()`).

## Technical Context

**Language/Version**:
- C# .NET 10 (new enrich tool + tiny `BlpRgbReader` helper)
- Python 3.11+ via `uv` (rewritten `build_v22_dataset.py`, refactored `v22_zarr_io.py`)

**Primary Dependencies** (all existing, no new packages):
- `WowViewer.Core.IO.M2.M2GeometryReader` — already produces `M2GeometryDocument` (vertices, normals, texcoords, bone_indices, bone_weights, render_flags, blend_modes, texture_lookup, texture_paths, texture_replaceable_ids, texture_flags, transparency_lookup, bone_lookup, bounds). Used in `AdtTensorPackBuilder.BuildObjectMasks` for spec 001 mask rasterization.
- `WowViewer.Core.IO.M2.M2SkinReader` — already produces `M2SkinDocument` with `triangleIndices`. Skin path is built via `M2ModelIdentity.BuildSkinPath(0)`.
- `WowViewer.Core.IO.Wmo.WmoRenderDocumentReader` — already produces `WmoRenderDocument` (vertices, triangles, normals, group_counts, group_indices, materials, material_texture_paths, bounds, portal_vertices, portal_indices, doodad_set_paths, flags, version).
- `WowViewer.Core.IO.Blp.AlphaBlpCompatibilityService` — already decodes BLP via `SereniaBLPLib` to `Image<Rgba32>`. Wrapped by a new ~30-line `BlpRgbReader` that returns `(H, W, ndarray<uint8 RGB>)`.
- `WowViewer.Core.M2.M2ModelIdentity.NormalizePath` — already provides canonical path normalization.
- `WowViewer.Core.IO.Maps.AdtPlacementReader` — already resolves `nameId` → canonical asset path via `ResolveNameViaXid`.
- Python: `zarr`, `pyarrow`, `pyarrow.parquet`, `numpy`. All already in `wow-viewer/data-harvester/pyproject.toml`.

**Storage**:
- Input: existing V18 Zarr at `output/datasets/v18/<build>.zarr/`.
- Intermediate: enrichment stream at `output/tmp/v22_enrich/<build>.bin` (debug seam only, not part of the V22 contract).
- Output: V22 Zarr at `output/datasets/v22/<build>.zarr/`.
- Audit sidecars outside the `.zarr`: `index.parquet`, `placements.parquet` (audit copy), `asset_inventory.parquet`, `finalization.json`, `_resume_state.json`.

**Testing**:
- xUnit for the new C# enrich tool and the enrichment stream format library.
- pytest for the rewritten Python builder, the patched-signal derivation helpers, and the round-trip with the V22 reader.
- Bounded real-data proof: one tile from staged `3_3_5_12340` Azeroth with `--limit 1`, prove a non-empty V22 store with populated `models/` and `tilesets/`.

**Target Platform**: Staged clients at `output/tmp/wowarchive-clients/{0_5_3_3368, 3_3_5_12340, 4_0_0_11927}`. No `H:\CLIENTS`. No new staged clients.

**Performance Goals**:
- V22 enrich + build of one Azeroth tile: < 30s wall-clock.
- V22 enrich + build of full Azeroth map (622 tiles for 0.5.3): < 10 min wall-clock on a single thread.
- Memory peak: < 1 GB for the in-process model + tileset library dictionary.

**Constraints**:
- V18 build path (harvester, `build_v18_dataset.py`, V18 trainers) MUST NOT be modified. This is FR-001 of the spec.
- Stable path keys: never use `Path.GetHashCode()`. Always use `M2ModelIdentity.NormalizePath`.
- Real-data validation gate before any consumer migration.

**Scale/Scope**:
- 3 staged builds: `0_5_3_3368`, `3_3_5_12340`, `4_0_0_11927`.
- ~764 MDDF + 7 MODF unique models per Azeroth-sized map (Spec 047 numbers).
- ~200-400 unique terrain textures per build.
- Up to 622 tiles per map (full Azeroth 0.5.3).

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| Repo Independence | PASS | All work in `wow-viewer/`. New tool at `tools/enrich/WowViewer.Tool.V22Enrich/`. Python builder stays in `data-harvester/`. |
| Library-First | PASS | New C# logic lives in `WowViewer.Core.IO.Maps.EnrichmentStream` and a new `WowViewer.Core.IO.Blp.BlpRgbReader`. The new tool is a thin CLI wrapper. |
| Real-Data Validation | REQUIRED | Bounded real-data proof on staged `3_3_5_12340` is Phase 7 of the plan. SC-001 of the spec is the gate. |
| Residual Model Chain | N/A | This spec is a data-pipeline spec. Model concerns belong to Spec 047 / 077 / etc. |
| Streaming-First | PASS | The enrichment stream is a length-prefixed binary format (FR-012) on stdout. The Python Zarr writer reads it. No NPZ intermediates. |
| No `H:\CLIENTS` | PASS | Staged clients only. |
| Read-Only Reference Codebase | PASS | `gillijimproject_refactor` not touched. |
| Format Reader/Writer Ownership | PASS | `M2GeometryReader`, `M2SkinReader`, `WmoRenderDocumentReader`, `AlphaBlpCompatibilityService` already exist and produce the required fields. No rewrite. |
| Terrain Alpha Risk Area | N/A | No terrain decode changes. |
| `AlphaWdtWriter` Frozen | N/A | Not touched. |
| One Phase at a Time | PASS | Eight phases, sequential dependencies. Each phase ends with a validation gate. |
| Spec Docs Source of Truth | PASS | Spec 088 supersedes 086/087. `docs/architecture/v22-dataset-signals-2026-06-30.md` is the V22 contract doc; this spec's `spec.md` references it. |
| Training Script Changes | N/A | No training scripts in this spec. |
| Memory Bank Discipline | REQUIRED | Phase 8 updates `activeContext.md` and `progress.md`. |
| Bite-Sized Plans | PASS | Each phase has ≤ 10 tasks. Tasks are independently validatable. |

## Project Structure

### Documentation

```text
specs/088-v22-enrichment-from-v18/
├── spec.md              # this spec
├── plan.md              # this file
├── tasks.md             # task breakdown
├── data-model.md        # data-model.md for V22 store layout
└── quickstart.md        # operator quickstart
```

`docs/architecture/v22-dataset-signals-2026-06-30.md` remains the canonical V22 contract doc. The spec's FR-022 through FR-025 reference it.

### Source Code (paths that change)

```text
wow-viewer/src/core/WowViewer.Core.IO/
├── Blp/
│   └── BlpRgbReader.cs                       # NEW: BLP → (H, W, ndarray uint8 RGB)
├── Maps/
│   ├── EnrichmentStreamFormat.cs             # NEW: ENTRY/ENDS binary writer/reader
│   ├── V22EnrichmentAssetReader.cs           # NEW: high-level asset reader (MPQ + filesystem)
│   └── RawArraySerializer.cs                 # MODIFY: revert V22 per-tile model payload block
└── Models/
    └── V22EnrichmentRecord.cs                # NEW: in-memory record of one decoded asset

wow-viewer/tools/enrich/
└── WowViewer.Tool.V22Enrich/
    ├── Program.cs                            # NEW: CLI entry point
    ├── WowViewer.Tool.V22Enrich.csproj       # NEW
    └── V18StorePlacementsReader.cs           # NEW: read V18 placements.parquet for asset paths

wow-viewer/data-harvester/
├── src/harvester/
│   ├── v22_zarr_io.py                        # MODIFY: add add_from_v18() API
│   └── v22_patched_signals.py                # NEW: pure-Python derivations (ground_intent_height_257, liquid_type_256, model_above_terrain_mask)
├── scripts/
│   └── build_v22_dataset.py                  # MODIFY: rewrite as V18 + enrichment → V22 builder
└── tests/
    ├── test_v22_zarr_io.py                   # MODIFY: keep existing fixed-key contract tests
    ├── test_v22_enrichment_stream.py         # NEW: round-trip synthetic records
    └── test_v22_patched_signals.py           # NEW: per-signal derivation tests

wow-viewer/tests/WowViewer.Core.Tests/
├── EnrichmentStreamFormatTests.cs            # NEW: stream format round-trip
└── BlpRgbReaderTests.cs                      # NEW: BLP decode round-trip on real BLP
```

### Source Code (paths unchanged but referenced)

- `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/` — V18 harvester, untouched.
- `wow-viewer/data-harvester/scripts/build_v18_dataset.py` — V18 builder, untouched.
- `wow-viewer/data-harvester/src/harvester/raw_reader.py` — existing ARRY/NPZ reader, reused.
- `wow-viewer/data-harvester/src/harvester/v16_1_dataset.py` — V18 reference for placement schema, reused.

## Implementation Phases

### Phase 0 — Spec Hygiene (archive 086 and 087)

**Goal**: Mark the broken V22 specs as superseded; update the archived spec registry; ensure no active spec still references 086/087 as the source of truth.

**Approach**:
- Add `SUPERSEDED.md` to `specs/086-v22-consolidated-dataset/` and `specs/087-v22-asset-library-payloads/` pointing to 088.
- Update `specs/archived/ARCHIVED.md` with rationale entries for 086 and 087.
- Grep the active spec list and the activeContext for references to 086/087 and replace with 088.

**Validation**:
- `grep -r "086-v22-consolidated-dataset" wow-viewer/specs/` returns no matches except in SUPERSEDED.md and ARCHIVED.md.
- `grep -r "087-v22-asset-library-payloads" wow-viewer/specs/` returns no matches except in SUPERSEDED.md and ARCHIVED.md.
- `ARCHIVED.md` lists 086 and 087 with rationale.

### Phase 1 — Revert Broken V22 Stream Profile

**Goal**: Remove the broken per-tile model payload emission in `RawArraySerializer.WriteV22Arrays` so the C# harvester cannot accidentally produce a non-deterministic-keyed V22 stream. This is FR-004 of the spec.

**Approach**:
- In `wow-viewer/src/core/WowViewer.Core.IO/Maps/RawArraySerializer.cs`, remove the `// ── Per-tile model payloads ──` block (lines 211-233) and the `if (pack.MclyTexturePixels is ... pixels)` block (lines 235-239) from `WriteV22Arrays`. The V22 profile becomes a thin wrapper over V16 (same tile signals, no per-tile model payloads, no per-tile `tileset_texture_rgb_*`).
- Update `BuildMetadataJson` to drop `tile_model_paths`, `tile_model_kinds`, `tile_model_texture_paths` (now empty since payloads are gone).
- Existing `RawArraySerializerTests.Serialize_V22_WritesFinalDatasetKeysAndDerivedArrays` still passes because the per-tile model payload assertions were never added (the existing test pins the V22 array NAMES, not the model payloads).

**Validation**:
- `dotnet build wow-viewer/WowViewer.slnx -c Debug` passes.
- `dotnet test wow-viewer/tests/WowViewer.Core.Tests --filter "FullyQualifiedName~RawArraySerializer"` passes.
- The diff for `RawArraySerializer.cs` shows the per-tile model payload block removed.
- The V22 profile no longer references `Path.GetHashCode()` anywhere.

### Phase 2 — BLP → RGB Helper

**Goal**: A small `BlpRgbReader` in `WowViewer.Core.IO.Blp` that wraps `AlphaBlpCompatibilityService` and returns `(H, W, ndarray<uint8 RGB>)` plus shape metadata. This unblocks the C# enrich tool from a separate BLP decode path.

**Approach**:
- New file `wow-viewer/src/core/WowViewer.Core.IO/Blp/BlpRgbReader.cs`.
- Public static method `BlpRgbReader.ReadRgb(byte[] source, string virtualPath) -> (int Width, int Height, byte[] Rgb)`. Internally calls `AlphaBlpCompatibilityService.NormalizeForAlphaClient` (for Alpha resize logic), then `BlpFile.GetImage(0)` via `SereniaBLPLib` directly to extract RGB pixels.
- Failure path returns `(-1, -1, null)` with a `load_error = 1` flag. Never throws.

**Validation**:
- xUnit test on a real staged BLP (e.g. `Tileset\\Generic\\Black.blp` from 3.3.5): assert width, height, and that the returned RGB has shape `(H*W*3)` with at least one nonzero pixel.
- xUnit test on a corrupt BLP (e.g. an empty byte array): assert graceful failure with `load_error = 1`, no throw.

### Phase 3 — Enrichment Stream Format Library

**Goal**: A length-prefixed binary stream format with stable path keys. This is the seam between the C# enrich tool and the Python builder. FR-012 of the spec.

**Approach**:
- New file `wow-viewer/src/core/WowViewer.Core.IO/Maps/EnrichmentStreamFormat.cs`.
- Stream layout (per FR-012):
  - Header: `V22E` (4 bytes) + version uint32 (little-endian, currently 1).
  - One `ENTRY` per asset: `[ENTRY magic 5][path_len uint32][path_utf8][kind uint8 (0=unknown, 1=M2, 2=WMO, 3=BLP)][load_error uint8][array_count uint32][for each array: name_len uint32, name_utf8, ndim uint32, shape uint32 × ndim, dtype 8 bytes ASCII null-padded, data_len int64, data_bytes]`.
  - Outer `ENDS` (4 bytes) terminates the stream.
- Public API: `EnrichmentStreamWriter` (writes one entry at a time), `EnrichmentStreamReader` (iterates entries). Both `IDisposable`. Stream is forward-only; no random access.
- All keys are the original canonical path string — no `Path.GetHashCode()`. Two writers with the same canonical path produce streams whose `ENTRY` records can be matched by string equality.

**Validation**:
- xUnit test: write 3 entries (M2, WMO, BLP), read back, assert paths, kinds, load_errors, and a sample array round-trip.
- xUnit test: empty stream (no entries) writes only header + `ENDS`. Reader returns zero entries.
- xUnit test: stream with 100 entries of the same path produces 100 distinct `ENTRY` records (the writer does NOT dedup; dedup is the caller's job).

### Phase 4 — C# V18 Placements Reader

**Goal**: A small helper that reads `placements.parquet` from a V18 Zarr store, returns unique canonical asset paths grouped by kind (M2 / WMO / BLP). This is the front-end of the enrich tool. FR-007 of the spec.

**Approach**:
- New file `wow-viewer/tools/enrich/WowViewer.Tool.V22Enrich/V18StorePlacementsReader.cs`.
- Reads `placements.parquet` from the V18 store using `pyarrow.parquet.read_table`. (Or hand-rolled Parquet read if the C# tool does not want a Python dependency. Decision: use `Parquet.Net` NuGet, already pulled in elsewhere in `wow-viewer/`. If not, use the existing `pyarrow` via a small `python3` subprocess. Decision deferred to Phase 4 task T0401.)
- Returns `EnrichmentAssetInventory` with three sorted sets: `UniqueM2Paths`, `UniqueWmoPaths`, `UniqueBlpPaths`.

**Validation**:
- xUnit test on a synthetic V18-style `placements.parquet` (no real V18 store needed): write a small parquet with 3 unique M2 paths, 2 unique WMO paths, 4 unique BLP paths, assert the inventory has exactly those counts.

### Phase 5 — C# `WowViewer.Tool.V22Enrich` Tool

**Goal**: The new C# CLI tool. Reads V18 store, walks unique asset paths, decodes each once, writes the enrichment stream. FR-005 through FR-017 of the spec.

**Approach**:
- New csproj `wow-viewer/tools/enrich/WowViewer.Tool.V22Enrich/WowViewer.Tool.V22Enrich.csproj`. Reference `WowViewer.Core.IO`.
- `Program.cs`: arg parse `--v18-store <path> --client-root <path> --output <stream-path> --build-key <name> --limit <N>`.
- Resolve client root via existing `NativeMpqService`-style path resolution (FR-014).
- Use `V18StorePlacementsReader` to get unique asset paths.
- For each M2 path: open via `Func<string, byte[]?> assetReader` (the same seam used by `AdtTensorPackBuilder`), call `M2GeometryReader.Read` + `M2SkinReader.Read`, flatten to arrays per FR-008, write an `ENTRY` of kind M2.
- For each WMO path: open via `assetReader`, call `WmoRenderDocumentReader.Read`, flatten per FR-009, write an `ENTRY` of kind WMO.
- For each BLP path: open via `assetReader`, call `BlpRgbReader.ReadRgb`, flatten to `(H, W, RGB uint8)`, write an `ENTRY` of kind BLP.
- On any failure: emit `load_error = 1` with zero-length arrays. Never throw past this point.
- Emit a final `ENDS`.
- Process exit code: 0 on success, 1 on unresolvable input, 2 on partial failure (any asset unreadable).

**Validation**:
- xUnit test of the main flow with a fake `assetReader` that returns canned M2/WMO/BLP bytes: write a stream, read it back, assert all entries.
- Real-data test (manual, gated on staged client): run the tool against `output/datasets/v18/3_3_5_12340.zarr` with `--limit 1` and assert the stream is non-empty and contains at least one M2 entry from the development map's MDDF placements.

### Phase 6 — Python `v22_patched_signals` Module

**Goal**: Pure-Python derivations of the V22 patched signals from the V18 store. This is FR-020 of the spec.

**Approach**:
- New file `wow-viewer/data-harvester/src/harvester/v22_patched_signals.py`.
- Functions, each takes a `V18Dataset` reader and tile index, returns a numpy array:
  - `derive_mcnr_mask_257(tile) -> (257, 257) bool` — copy from `tile["mcnr_mask_257"]` if present, else `(x % 2 == y % 2)` checkerboard.
  - `derive_liquid_type_256(tile) -> (256, 256) uint8` — match `RawArraySerializer.BuildLiquidType256` reference: take `liquid_basic_type_257` (or fallback), 257→256 crop, 0xFF → 0, others + 1.
  - `derive_ground_intent_height_257(tile) -> (257, 257) float32` — match `RawArraySerializer.BuildGroundIntentHeight257` reference: 4-neighbour Laplace inpainting over `object_precise_mask`.
  - `derive_model_focus_mask(tile) -> (257, 257) float32` — alias of `tile["object_filtered_mask"]`.
  - `derive_model_above_terrain_mask(tile) -> (257, 257) float32` — match `RawArraySerializer.BuildModelAboveTerrainMask` reference: project each MDDF/MODF placement's `(posX, posY, posZ)` to a tile pixel, set pixel to 1.0 if `posZ >= height[py, px] - 1.0`.
- All functions are deterministic, allocation-light, and never call into game clients.

**Validation**:
- pytest: synthetic tile with a known heightmap and known placements produces the expected `model_above_terrain_mask`.
- pytest: synthetic tile with object pixels produces the expected inpainted `ground_intent_height_257`.
- pytest: reference parity check — given the same inputs, Python output matches the C# reference output recorded in the C# test fixtures.

### Phase 7 — Python `v22_zarr_io` Refactor

**Goal**: Replace `V22ZarrWriter.add_tile(record)` with `V22ZarrWriter.add_from_v18(v18_path, enrichment_path)`. Keep the existing `V22Dataset` reader contract unchanged. FR-030 through FR-032 of the spec.

**Approach**:
- Modify `wow-viewer/data-harvester/src/harvester/v22_zarr_io.py`:
  - Remove `add_tile`, `V22TileRecord`, `_ingest_model_payloads`, `_ingest_tileset_payloads`. They were the broken per-tile accumulation path.
  - Add `add_from_v18(v18_path: Path, enrichment_path: Path) -> None`. Internally:
    - Open V18 store as a `V18Dataset` (or use the existing V18 reader; the choice is in `v18_io.py` or a small helper).
    - For each V18 tile index, build a per-tile V22 dict: copy V18 root arrays verbatim + apply `v22_patched_signals` derivations to fill the V22-only arrays.
    - Read `placements.parquet` from V18, promote to native V22 placement arrays (FR-021).
    - Open the enrichment stream, group `ENTRY` records by kind, accumulate into `self._models` and `self._tilesets` keyed by canonical path.
    - Compute `mcly_tileset_ids` (FR-024) and remapped `mddf_model_ids` / `modf_model_ids` (FR-025) per tile.
  - Keep `V22Dataset.__getitem__` exactly as it is. The fixed-key contract is unchanged. The existing `test_v22_zarr_io.py` tests should pass without modification (other than removing the `add_tile` API).
- `finalize()` writes the same Zarr layout (root arrays + `models/` group + `tilesets/` group + audit metadata) but now from a single `add_from_v18` call instead of many `add_tile` calls.

**Validation**:
- pytest: synthetic V18 store (small in-memory dataset) + synthetic enrichment stream → `add_from_v18` → `finalize` → `V22Dataset` round-trip. Assert the documented fixed-key contract.
- pytest: a 2-tile V18 store with shared M2 paths produces 1 model entry in `models/model_paths`, not 2. Confirms dedup.
- pytest: a 2-tile V18 store with shared BLP paths produces 1 tileset entry in `tilesets/tileset_paths`, not 2. Confirms dedup.
- pytest: empty V18 store + empty enrichment stream → empty V22 store with `tile_count = 0`, no crash.

### Phase 8 — Python `build_v22_dataset.py` Rewrite

**Goal**: New CLI with two subcommands: `enrich` (run the C# tool as a subprocess) and `build` (consume V18 + enrichment → V22 Zarr). FR-018, FR-029.

**Approach**:
- Rewrite `wow-viewer/data-harvester/scripts/build_v22_dataset.py`:
  - Subcommand `enrich --v18-store <path> --client-root <path> --enrichment-output <path> --build-key <name> --limit <N>`: spawns the C# tool as a subprocess, waits for completion, checks exit code. Wrapper only — all the real work is in the C# tool.
  - Subcommand `build --v18-store <path> --enrichment <stream-path> --output <zarr-path>`: instantiates `V22ZarrWriter`, calls `add_from_v18`, calls `finalize`, prints the output path.
  - Subcommand `stats --store <zarr-path>`: keep existing `_stats` logic.
  - All exit codes are real (FR-029): 0 success, 1 missing input, 2 partial failure.
- `scripts/inspect_v22_dataset.py` is unchanged (FR-033) — it reads the same Zarr layout.

**Validation**:
- `python build_v22_dataset.py --help` shows the new subcommand structure.
- `python build_v22_dataset.py build --v18-store <synthetic.zarr> --enrichment <synthetic.bin> --output <out.zarr>` returns 0.
- `python build_v22_dataset.py build --v18-store <missing> ...` returns 1.

### Phase 9 — Tests and Bounded Real-Data Proof

**Goal**: SC-001 of the spec. End-to-end on staged `3_3_5_12340` Azeroth with `--limit 1`. This is the gate that 086/087 never crossed.

**Approach**:
- A bounded proof script at `wow-viewer/specs/088-v22-enrichment-from-v18/scripts/proof_v22_bounded_335.py`. Steps:
  1. Build a V18 store for `3_3_5_12340` Azeroth with `--limit 1` if it does not already exist. (Use existing `build_v18_dataset.py`.)
  2. Run `WowViewer.Tool.V22Enrich` against the V18 store.
  3. Run the rewritten `build_v22_dataset.py build`.
  4. Run `inspect_v22_dataset.py summary` and `inspect_v22_dataset.py tile 0`.
  5. Assert: `tile_count == 1`, `model_count > 0`, `tileset_count > 0`, every documented root array exists with correct shape and dtype.
  6. Save the proof output to `output/proofs/v22_bounded_335_<timestamp>.json`.
- A bounded proof for `0_5_3_3368` Azeroth and `4_0_0_11927` development map. Same script, parameterized by build.
- pytest test that runs the bounded proof synthetically (using a fake V18 store + fake enrichment stream) and asserts the same end-to-end contract.

**Validation**:
- Bounded proof runs end-to-end on `3_3_5_12340` Azeroth. The proof output is committed alongside the spec.
- All bounded proofs (3 builds) produce non-empty V22 stores.
- `inspect_v22_dataset.py tile 0` per-tile JSON shows per-array nonzero counts that are not all zero for `height_257`, `minimap_rgb`, `mcly_texture_ids`, `object_precise_mask` (where applicable), `mcly_tileset_ids`.

### Phase 10 — Documentation and Memory Bank

**Goal**: Update the memory bank, the docs, and the archived spec registry to reflect the new state.

**Approach**:
- `wow-viewer/specs/archived/ARCHIVED.md`: add rationale entries for 086 and 087.
- `wow-viewer/memory-bank/activeContext.md`: replace "Spec 086 V22 schema freeze complete" entry with a one-paragraph summary of Spec 088 status.
- `wow-viewer/memory-bank/progress.md`: add a dated entry for the spec landing.
- `wow-viewer/data-harvester/README.md`: update the V22 section to reflect the new two-tool flow (`V22Enrich` + `build_v22_dataset.py`).
- `wow-viewer/docs/architecture/v22-dataset-signals-2026-06-30.md`: add a brief note that the contract is now satisfied by Spec 088's two-tool flow, not by Spec 086's per-tile stream.

**Validation**:
- `ARCHIVED.md` lists 086 and 087.
- `activeContext.md` mentions 088.
- `data-harvester/README.md` V22 section references the new commands.

## Validation and Diagnostics Matrix

| Phase | Gate |
|-------|------|
| 0 | `ARCHIVED.md` updated, `SUPERSEDED.md` files in 086/087 directories. |
| 1 | `dotnet build` and `dotnet test` on `WowViewer.Core.Tests` pass. `RawArraySerializer.cs` V22 profile no longer references `Path.GetHashCode()`. |
| 2 | `BlpRgbReader` xUnit tests pass on real BLP and corrupt BLP. |
| 3 | `EnrichmentStreamFormat` xUnit tests pass (write/read round-trip on M2, WMO, BLP). |
| 4 | `V18StorePlacementsReader` xUnit test passes on synthetic parquet. |
| 5 | `WowViewer.Tool.V22Enrich` xUnit tests pass with fake `assetReader`. Real-data gate: tool runs on `3_3_5_12340` Azeroth with `--limit 1` and produces a non-empty stream. |
| 6 | `v22_patched_signals` pytest tests pass; reference parity with C# reference values. |
| 7 | `v22_zarr_io` round-trip tests pass; existing `test_v22_zarr_io.py` modified minimally. |
| 8 | `build_v22_dataset.py --help` shows the new subcommands; `--v18-store <missing>` returns exit 1. |
| 9 | Bounded real-data proof on `3_3_5_12340` Azeroth: non-empty V22 store, populated `models/` and `tilesets/`. Repeat for `0_5_3_3368` and `4_0_0_11927`. |
| 10 | `ARCHIVED.md`, `activeContext.md`, `progress.md`, `data-harvester/README.md` updated. |

## Complexity Tracking

No constitution violations. The spec uses existing libraries (M2GeometryReader, M2SkinReader, WmoRenderDocumentReader, AlphaBlpCompatibilityService) and a small new helper (BlpRgbReader, ~30 lines). The enrich tool is a thin CLI wrapper. The Python builder is a rewrite of an existing script, not a new architecture. The enrichment stream format is a small addition to the existing binary-stream family.
