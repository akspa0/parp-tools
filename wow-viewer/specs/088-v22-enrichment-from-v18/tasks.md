# Tasks: V22 Enrichment From V18

**Input**: `specs/088-v22-enrichment-from-v18/spec.md`, `specs/088-v22-enrichment-from-v18/plan.md`

---

## Phase 0: Spec Hygiene (archive 086 and 087)

**Purpose**: Mark the broken V22 specs as superseded so the active spec list is unambiguous.

- [ ] T0001 [US1] Write `wow-viewer/specs/086-v22-consolidated-dataset/SUPERSEDED.md` pointing to 088. Include reason (C# three-message-class producer never written; per-tile model payloads emitted but never deduped) and the canonical replacement (088).
- [ ] T0002 [US1] Write `wow-viewer/specs/087-v22-asset-library-payloads/SUPERSEDED.md` pointing to 088. Include reason (non-deterministic `Path.GetHashCode()` keys; per-tile duplication) and the canonical replacement (088).
- [ ] T0003 [US1] Append 086 and 087 entries to `wow-viewer/specs/archived/ARCHIVED.md` with rationale matching the SUPERSEDED.md files. Add a "Supersession note" section explaining why the directories are not physically moved.
- [ ] T0004 [US1] Grep the active specs directory (`wow-viewer/specs/`) and `wow-viewer/memory-bank/` for references to `086-v22-consolidated-dataset` or `087-v22-asset-library-payloads` (other than in the SUPERSEDED.md files). For any hits, replace with a 088 reference.

**Checkpoint**: Phase 0 complete — active spec list no longer references 086 or 087 as canonical.

---

## Phase 1: Revert Broken V22 Stream Profile (FR-004)

**Purpose**: Remove the broken per-tile model payload emission in `RawArraySerializer.WriteV22Arrays` so the C# harvester cannot accidentally produce a non-deterministic-keyed V22 stream.

- [ ] T0101 [US1] In `wow-viewer/src/core/WowViewer.Core.IO/Maps/RawArraySerializer.cs`, remove the `// ── Per-tile model payloads ──` block (lines 211-233) from `WriteV22Arrays`. Also remove the `if (pack.MclyTexturePixels is ... pixels)` block that emits `tileset_texture_rgb_<index>` (lines 235-239).
- [ ] T0102 [US1] In `RawArraySerializer.BuildMetadataJson`, drop the `tile_model_paths`, `tile_model_kinds`, and `tile_model_texture_paths` fields from the metadata JSON. They are no longer populated.
- [ ] T0103 [US1] `dotnet build wow-viewer/WowViewer.slnx -c Debug` — confirm clean build.
- [ ] T0104 [US1] `dotnet test wow-viewer/tests/WowViewer.Core.Tests --filter "FullyQualifiedName~RawArraySerializer" -c Debug` — confirm `Serialize_V22_WritesFinalDatasetKeysAndDerivedArrays` still passes (it pins array names, not model payloads).

**Checkpoint**: Phase 1 complete — V22 stream profile no longer references `Path.GetHashCode()` or emits per-tile model payloads. Build is clean.

---

## Phase 2: BLP → RGB Helper (FR needed for Phase 5)

**Purpose**: A small `BlpRgbReader` in `WowViewer.Core.IO.Blp` that wraps `AlphaBlpCompatibilityService` and returns `(H, W, ndarray uint8 RGB)`. Unblocks the C# enrich tool from a separate BLP decode path.

- [ ] T0201 [US2] Create `wow-viewer/src/core/WowViewer.Core.IO/Blp/BlpRgbReader.cs` with public static method `BlpRgbReader.ReadRgb(byte[] source, string virtualPath) -> (int Width, int Height, byte[] Rgb)`. Internally use `AlphaBlpCompatibilityService.NormalizeForAlphaClient` for the Alpha resize logic, then `BlpFile.GetImage(0)` via SereniaBLPLib to extract RGBA pixels and slice to RGB.
- [ ] T0202 [US2] Add failure path: corrupt or empty bytes returns `(-1, -1, null)` with no throw.
- [ ] T0203 [US2] Create `wow-viewer/tests/WowViewer.Core.Tests/BlpRgbReaderTests.cs`. Test 1: real staged BLP (e.g. `Tileset\\Generic\\Black.blp` from 3.3.5) decodes with non-zero width, height, and at least one nonzero RGB byte. Test 2: empty byte array returns failure tuple without throwing.
- [ ] T0204 [US2] `dotnet test wow-viewer/tests/WowViewer.Core.Tests --filter "FullyQualifiedName~BlpRgbReader" -c Debug` — confirm tests pass.

**Checkpoint**: Phase 2 complete — `BlpRgbReader` exists, is tested, and the C# enrich tool can use it without a separate BLP path.

---

## Phase 3: Enrichment Stream Format Library (FR-012)

**Purpose**: A length-prefixed binary stream format with stable path keys. The seam between the C# enrich tool and the Python builder.

- [ ] T0301 [US3] Create `wow-viewer/src/core/WowViewer.Core.IO/Maps/EnrichmentStreamFormat.cs` with public classes `EnrichmentStreamWriter` and `EnrichmentStreamReader`. Stream layout per FR-012: header `V22E` + version uint32, one `ENTRY` per asset, outer `ENDS`. All keys are the original canonical path string.
- [ ] T0302 [US3] `EnrichmentStreamWriter` API: `WriteHeader()`, `WriteEntry(EnrichmentEntry entry)`, `WriteEnds()`. Each entry carries `(path, kind, load_error, IReadOnlyList<EnrichmentArray>)`. `EnrichmentArray` carries `(name, ndim, shape, dtype, data_bytes)`.
- [ ] T0303 [US3] `EnrichmentStreamReader` API: `ReadHeader()`, `IEnumerable<EnrichmentEntry> ReadEntries()`, `ReadEnds()`. Forward-only iteration; no random access.
- [ ] T0304 [US3] Create `wow-viewer/src/core/WowViewer.Core/Maps/EnrichmentEntry.cs` and `wow-viewer/src/core/WowViewer.Core/Maps/EnrichmentArray.cs` (data classes). `EnrichmentEntry.Kind` is an enum: `Unknown = 0, M2 = 1, WMO = 2, BLP = 3`.
- [ ] T0305 [US3] Create `wow-viewer/tests/WowViewer.Core.Tests/EnrichmentStreamFormatTests.cs`. Test 1: write 3 entries (M2, WMO, BLP), read back, assert paths, kinds, load_errors, and a sample array round-trip with shape and dtype. Test 2: empty stream (no entries) writes only header + ENDS. Test 3: stream with 100 entries of the same path produces 100 distinct ENTRY records (writer does not dedup; caller does).
- [ ] T0306 [US3] `dotnet test wow-viewer/tests/WowViewer.Core.Tests --filter "FullyQualifiedName~EnrichmentStreamFormat" -c Debug` — confirm tests pass.

**Checkpoint**: Phase 3 complete — the enrichment stream format is locked. Stable path keys, no `GetHashCode()`.

---

## Phase 4: C# V18 Placements Reader (FR-007)

**Purpose**: A small helper that reads `placements.parquet` from a V18 Zarr store and returns unique canonical asset paths grouped by kind.

- [ ] T0401 [US2] In `wow-viewer/tools/enrich/WowViewer.Tool.V22Enrich/`, create `V18StorePlacementsReader.cs`. Public method `V18StorePlacementsReader.ReadPlacementsParquet(Path v18StorePath) -> EnrichmentAssetInventory`. Inventory has `UniqueM2Paths`, `UniqueWmoPaths`, `UniqueBlpPaths` (all `IReadOnlyList<string>`).
- [ ] T0402 [US2] Implementation choice: use `Parquet.Net` (NuGet) to read `placements.parquet` without a Python dependency. Confirm `Parquet.Net` is already a dependency in `wow-viewer/` by inspecting any other csproj; if not, add it.
- [ ] T0403 [US2] The reader groups paths by extension: `.m2` → M2, `.wmo` → WMO, `.blp` → BLP. Empty extension or unknown extension → ignore.
- [ ] T0404 [US2] Create `wow-viewer/tests/WowViewer.Core.Tests/V18StorePlacementsReaderTests.cs`. Write a synthetic `placements.parquet` with 3 unique M2, 2 unique WMO, 4 unique BLP paths (use `Parquet.Net` writer). Call the reader. Assert the inventory has exactly those counts and the same paths.
- [ ] T0405 [US2] `dotnet test wow-viewer/tests/WowViewer.Core.Tests --filter "FullyQualifiedName~V18StorePlacementsReader" -c Debug` — confirm tests pass.

**Checkpoint**: Phase 4 complete — the V18 placements reader is unit-tested on synthetic data. Real-data gate is Phase 9.

---

## Phase 5: C# `WowViewer.Tool.V22Enrich` Tool (FR-005 through FR-017)

**Purpose**: The new C# CLI tool. Reads V18 store, walks unique asset paths, decodes each once, writes the enrichment stream.

- [ ] T0501 [US2] Create `wow-viewer/tools/enrich/WowViewer.Tool.V22Enrich/WowViewer.Tool.V22Enrich.csproj` with references to `WowViewer.Core.IO` and `Parquet.Net`. Match the existing csproj style of `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj`.
- [ ] T0502 [US2] Create `wow-viewer/tools/enrich/WowViewer.Tool.V22Enrich/Program.cs` with arg parse: `--v18-store <path>`, `--client-root <path>`, `--output <stream-path>`, `--build-key <name>`, `--limit <N>` (optional).
- [ ] T0503 [US2] In `Program.cs`, resolve client root via the same path-resolution pattern as `WowViewer.Tool.Harvest` (FR-014): if `--client-root` is `output/tmp/wowarchive-clients/<build>` and a nested `World of Warcraft` exists, use the nested path.
- [ ] T0504 [US2] In `Program.cs`, instantiate `V18StorePlacementsReader` and read the V18 store's `placements.parquet`. Get unique M2 / WMO / BLP paths.
- [ ] T0505 [US2] In `Program.cs`, build an `assetReader` lambda: `Func<string, byte[]?>`. Use the existing `MpqFileSystem`-style resolution pattern from `WowViewer.Core.Archive` (same seam `AdtTensorPackBuilder` uses). Reference the path-resolution helper that already exists in `WowViewer.Core.Archive` or write a small one in this tool.
- [ ] T0506 [US2] In `Program.cs`, for each unique M2 path: call `assetReader(path)` → bytes; if null, emit a `load_error=1` BLP-style entry with empty payload. Otherwise: load via `M2GeometryReader.Read(stream, path)` and `M2SkinReader.Read(stream, BuildSkinPath(path, 0))`. Build the V22 model payload array list per FR-008. Emit one `EnrichmentEntry` of kind M2.
- [ ] T0507 [US2] In `Program.cs`, for each unique WMO path: same pattern via `WmoRenderDocumentReader.Read`. Build the V22 model payload array list per FR-009. Emit one `EnrichmentEntry` of kind WMO.
- [ ] T0508 [US2] In `Program.cs`, for each unique BLP path: call `assetReader(path)` → bytes; if null, emit `load_error=1`. Otherwise: call `BlpRgbReader.ReadRgb(bytes, path)`. On success, emit one `EnrichmentEntry` of kind BLP with `texture_rgb` and `texture_shape` arrays. On failure, emit `load_error=1`.
- [ ] T0509 [US2] In `Program.cs`, write the enrichment stream to `--output`. Use `EnrichmentStreamWriter` from Phase 3. Write header, all entries, ENDS.
- [ ] T0510 [US2] In `Program.cs`, on any unhandled exception, set `Environment.ExitCode = 1` and print the error to stderr. On success, exit 0. If at least one asset failed to decode (and emitted `load_error=1`), exit 0 (partial failure is expected and recoverable).

**Checkpoint**: Phase 5 complete — the C# enrich tool is buildable and unit-testable. Real-data gate is Phase 9.

---

## Phase 6: Python `v22_patched_signals` Module (FR-020)

**Purpose**: Pure-Python derivations of the V22 patched signals from the V18 store. No C#, no game client reparse.

- [ ] T0601 [US4] Create `wow-viewer/data-harvester/src/harvester/v22_patched_signals.py` with public functions:
  - `derive_mcnr_mask_257(tile: dict) -> np.ndarray` — copy from `tile["mcnr_mask_257"]` if present, else `(x % 2 == y % 2)` checkerboard.
  - `derive_liquid_type_256(tile: dict) -> np.ndarray` — match `RawArraySerializer.BuildLiquidType256`.
  - `derive_ground_intent_height_257(tile: dict) -> np.ndarray` — match `RawArraySerializer.BuildGroundIntentHeight257` (4-neighbour Laplace fill).
  - `derive_model_focus_mask(tile: dict) -> np.ndarray` — alias of `tile["object_filtered_mask"]`.
  - `derive_model_above_terrain_mask(tile: dict, placements: dict) -> np.ndarray` — match `RawArraySerializer.BuildModelAboveTerrainMask`.
- [ ] T0602 [US4] Each function takes a tile dict (per-tile numpy arrays keyed by V18 name) and a placements dict (per-tile MDDF/MODF rows). Each is deterministic, allocation-light, no I/O.
- [ ] T0603 [US4] Create `wow-viewer/data-harvester/tests/test_v22_patched_signals.py` with one pytest per function. Tests use synthetic tiles; reference values are derived from the C# algorithms in `RawArraySerializer.cs` (lines 382-400, 418-495, 555-629).
- [ ] T0604 [US4] Run `uv run pytest wow-viewer/data-harvester/tests/test_v22_patched_signals.py -q` from `wow-viewer/data-harvester/` — confirm all tests pass.

**Checkpoint**: Phase 6 complete — V22 patched signals derivable in pure Python with reference parity.

---

## Phase 7: Python `v22_zarr_io` Refactor (FR-030 through FR-032)

**Purpose**: Replace `V22ZarrWriter.add_tile(record)` with `V22ZarrWriter.add_from_v18(v18_path, enrichment_path)`. Keep the existing `V22Dataset` reader contract unchanged.

- [ ] T0701 [US2] In `wow-viewer/data-harvester/src/harvester/v22_zarr_io.py`, remove `add_tile`, `V22TileRecord`, `_ingest_model_payloads`, `_ingest_tileset_payloads`, `empty_tile`. These were the broken per-tile accumulation path.
- [ ] T0702 [US2] In `v22_zarr_io.py`, add `add_from_v18(v18_path: Path, enrichment_path: Path) -> None` on `V22ZarrWriter`. Internally:
  - Open V18 store via a small `V18Dataset` reader (or use `zarr` directly to read the 20 root arrays).
  - For each V18 tile index, build a per-tile V22 dict: copy V18 root arrays verbatim + apply `v22_patched_signals` derivations to fill the V22-only arrays.
  - Read V18's `placements.parquet`, promote to native V22 placement arrays (FR-021).
  - Open the enrichment stream, group entries by kind, accumulate into `self._models` and `self._tilesets` keyed by canonical path.
  - Compute `mcly_tileset_ids` (FR-024) and remapped `mddf_model_ids` / `modf_model_ids` (FR-025) per tile.
- [ ] T0703 [US2] Update `V22ZarrWriter.finalize()` to write the same Zarr layout (root arrays + `models/` group + `tilesets/` group + audit metadata). No changes to the public surface that `V22Dataset` reads.
- [ ] T0704 [US2] Confirm `V22Dataset.__getitem__` is unchanged. The fixed-key contract is preserved exactly.
- [ ] T0705 [US2] Update `wow-viewer/data-harvester/tests/test_v22_zarr_io.py`: replace per-tile `V22TileRecord` tests with `add_from_v18` tests using synthetic V18 stores (in-memory) and synthetic enrichment streams. Keep the existing fixed-key contract tests.
- [ ] T0706 [US2] Add a pytest that confirms dedup: 2-tile synthetic V18 store with shared M2 paths produces 1 model entry, not 2. Same for BLP.
- [ ] T0707 [US2] Add a pytest that confirms empty-input safety: empty V18 store + empty enrichment stream → empty V22 store with `tile_count = 0`, no crash.
- [ ] T0708 [US2] Run `uv run pytest wow-viewer/data-harvester/tests/test_v22_zarr_io.py -q` — confirm all tests pass.

**Checkpoint**: Phase 7 complete — `V22ZarrWriter` rewritten; `V22Dataset` contract preserved; dedup and empty-input safety verified.

---

## Phase 8: Python `build_v22_dataset.py` Rewrite (FR-018, FR-029)

**Purpose**: New CLI with two subcommands: `enrich` (run the C# tool as a subprocess) and `build` (consume V18 + enrichment → V22 Zarr). Plus the existing `stats`.

- [ ] T0801 [US2] In `wow-viewer/data-harvester/scripts/build_v22_dataset.py`, replace the existing `harvest-build`, `build`, and `stats` subcommand logic. New subcommand structure:
  - `enrich --v18-store <path> --client-root <path> --enrichment-output <path> --build-key <name> --limit <N>`: spawns `WowViewer.Tool.V22Enrich` as a subprocess. Checks exit code. Emits a friendly error if the C# tool is missing.
  - `build --v18-store <path> --enrichment <stream-path> --output <zarr-path>`: instantiates `V22ZarrWriter`, calls `add_from_v18`, calls `finalize`, prints the output path.
  - `stats --store <zarr-path>`: keep existing `_stats` logic.
- [ ] T0802 [US2] In `build_v22_dataset.py`, set real exit codes (FR-029): 0 on success, 1 on missing input, 2 on partial failure.
- [ ] T0803 [US2] In `build_v22_dataset.py`, the `enrich` subcommand resolves paths against the workspace root (`WOW_VIEWER_ROOT / ".."`), `WOW_VIEWER_ROOT`, and `DATA_HARVESTER_ROOT` (matching the existing path resolution pattern).
- [ ] T0804 [US2] `python build_v22_dataset.py --help` shows the new subcommand structure.
- [ ] T0805 [US2] `python build_v22_dataset.py build --v18-store <missing> ...` returns exit 1 with a clear error.

**Checkpoint**: Phase 8 complete — the Python CLI is rewritten and exits with real codes on failure.

---

## Phase 9: Bounded Real-Data Proof (SC-001)

**Purpose**: End-to-end on staged `3_3_5_12340` Azeroth with `--limit 1`. This is the gate that 086 and 087 never crossed.

- [ ] T0901 [US6] Create `wow-viewer/specs/088-v22-enrichment-from-v18/scripts/proof_v22_bounded.py`. Steps:
  1. Verify staged `output/tmp/wowarchive-clients/3_3_5_12340` exists. Bail with a clear error if not.
  2. Build a V18 store for `3_3_5_12340` Azeroth with `--limit 1` if it does not already exist (use existing `build_v18_dataset.py`).
  3. Run `WowViewer.Tool.V22Enrich` against the V18 store.
  4. Run the rewritten `build_v22_dataset.py build`.
  5. Run `inspect_v22_dataset.py summary` and `inspect_v22_dataset.py tile 0`.
  6. Assert: `tile_count == 1`, `model_count > 0`, `tileset_count > 0`, every documented root array exists with correct shape and dtype.
  7. Save the proof output to `output/proofs/v22_bounded_335_<timestamp>.json`.
- [ ] T0902 [US6] Run the proof on `3_3_5_12340` Azeroth. Save the proof output JSON.
- [ ] T0903 [US6] Run the proof on `0_5_3_3368` Azeroth. Save the proof output JSON.
- [ ] T0904 [US6] Run the proof on `4_0_0_11927` development map. Save the proof output JSON.
- [ ] T0905 [US6] For each proof, manually inspect `inspect_v22_dataset.py tile 0` per-array nonzero counts. Confirm `height_257`, `minimap_rgb`, `mcly_texture_ids` are non-zero; `object_precise_mask` and `mcly_tileset_ids` non-zero where applicable.

**Checkpoint**: Phase 9 complete — three bounded real-data proofs pass. SC-001, SC-005, SC-006, SC-007 of the spec are satisfied.

---

## Phase 10: Documentation and Memory Bank

**Purpose**: Update the memory bank, the docs, and the archived spec registry to reflect the new state.

- [ ] T1001 [US1] In `wow-viewer/memory-bank/activeContext.md`, replace the "Spec 086 V22 schema freeze complete" entry with a one-paragraph summary: "V22 was a stuck design (per-tile stream, never produced a populated store). Spec 088 replaces it with V18-substrate + a separate C# enrich tool. Phase N+1 is bounded real-data proof on `3_3_5_12340` Azeroth."
- [ ] T1002 [US1] In `wow-viewer/memory-bank/progress.md`, add a dated entry for the spec landing (2026-06-30).
- [ ] T1003 [US1] In `wow-viewer/data-harvester/README.md`, update the V22 section to reflect the new two-tool flow (`V22Enrich` + `build_v22_dataset.py`). Replace the "Spec 086 V22 Stream → Zarr Quickstart" section with "V22 Enrichment From V18 Quickstart".
- [ ] T1004 [US1] In `wow-viewer/docs/architecture/v22-dataset-signals-2026-06-30.md`, add a brief note: "Implementation lives in Spec 088 (V18 substrate + separate C# enrich tool). Spec 086's per-tile stream design was abandoned."

**Checkpoint**: Phase 10 complete — memory bank, data-harvester README, and architecture doc reflect the new state.

---

## Validation Summary

- [ ] V01 SC-001: bounded V22 build on one `3_3_5_12340` Azeroth tile produces a Zarr store with `tile_count = 1`, `model_count > 0`, `tileset_count > 0`.
- [ ] V02 SC-002: V22 stores built twice from the same V18 store + same enrichment stream have byte-identical `models/model_paths` and `mcly_tileset_ids` per tile.
- [ ] V03 SC-003: V18 trainers (`train_v18.py`, `train_v18_focus.py`) keep working on V18 stores without changes.
- [ ] V04 SC-004: `WowViewer.Tool.V22Enrich` reads a real V18 store and produces a non-empty enrichment stream for `3_3_5_12340` Azeroth.
- [ ] V05 SC-005: a corrupt or missing M2 produces a `models/<path>` entry with `load_error = 1` and zero-length geometry arrays; the build does not crash.
- [ ] V06 SC-006: V22 root array existence and shape / dtype checks pass for every documented array, including zero-fill behavior.
- [ ] V07 SC-007: Python builder's `models/model_paths` count for a real `3_3_5_12340` tile matches the count of unique M2 paths in the tile's placements.
- [ ] V08 SC-008: Specs 086 and 087 are marked superseded; `ARCHIVED.md` is updated.

---

## Dependencies & Execution Order

```
Phase 0  ──► (no deps; spec hygiene, can start immediately)
Phase 1  ──► (no deps; standalone revert of broken stream profile)
Phase 2  ──► (no deps; new BlpRgbReader)
Phase 3  ──► (no deps; new stream format library)
Phase 4  ──► (no deps; new V18 placements reader)
Phase 5  ──► Phase 2, Phase 3, Phase 4 (uses BlpRgbReader, EnrichmentStreamWriter, V18StorePlacementsReader)
Phase 6  ──► (no deps; pure-Python derivations)
Phase 7  ──► Phase 6 (uses v22_patched_signals)
Phase 8  ──► Phase 7 (uses V22ZarrWriter.add_from_v18)
Phase 9  ──► Phase 5, Phase 8 (end-to-end on real data)
Phase 10 ──► Phase 9 (docs after proof)
```

## Parallel Opportunities

- Phases 1, 2, 3, 4 are independent and can run in parallel.
- Phase 6 is independent of Phases 1-5 and can run in parallel with them.
- Phase 7 depends on Phase 6; Phase 8 depends on Phase 7; Phase 9 depends on Phase 5 + 8; Phase 10 depends on Phase 9.

## Total Task Count

10 phases, 41 tasks. Each task is independently completable in 1-3 tool calls per the constitution. No phase exceeds 10 tasks. Tasks are bite-sized enough that any LLM can implement a single task in a focused pass.
