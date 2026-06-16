# Tasks: 064 — Blank Map Generation + Relational ADT Understanding

**Plan**: [plan.md](plan.md) | **Spec**: [spec.md](spec.md)

## Phase 1: Blank LK Map Generation

### P1-A: BlankAdtFactory

- [ ] T001: Create `WowViewer.Core/Maps/BlankAdtFactory.cs` — static class with `CreateBlankLkAdtData(string mapName, int tileX, int tileY)` that produces a valid `LkAdtData` with 256 flat MCNK chunks, zero placements, zero textures, zero liquid
- [ ] T002: Investigate whether the viewer requires at least one texture layer per MCNK — add `--single-grass` flag if needed for minimal rendering
- [ ] T003: Test blank ADT generation: write `LkAdtWriter.Build(factory.CreateBlankLkAdtData("blanktest", 32, 43))` to file, verify with `map inspect` that all mandatory chunks exist with valid offsets
- [ ] T004: Fix any MCIN offset computation issues — offsets must be computed after all chunk sizes are finalized

### P1-B: CLI Command + WDT/WDL Generation

- [ ] T005: Add `map generate-blank-map` CLI command to inspect tool — `--map-name <name> --tile-x <n> --tile-y <n> --output-dir <dir>` — generates WDT + ADT + WDL
- [ ] T006: Add `BlankWdtFactory` or equivalent to construct WDT data with MAIN mask flagging the specified tile(s)
- [ ] T007: Generate WDL with flat heights for the specified tile(s)
- [ ] T008: Validate generated WDT + WDL with `map inspect`

### P1-C: Viewer Validation

- [ ] T009: Load blank ADT in viewer and confirm no assertion errors, no null references, no missing chunk warnings
- [ ] T010: Confirm flat terrain renders (visible grid) — if MCNK renders as invisible, investigate and fix normal/height defaults
- [ ] T011: Load full WDT+ADT+WDL set in viewer as a named map and confirm tile is visible in the world

### P1-D: Alpha Blank WDT (Optional, requires Rule 10 reopening)

- [ ] T012: [OPTIONAL] Generate blank Alpha WDT using existing `AlphaWdtWriter.Build()` with blank `AlphaTileData` — skip unless user explicitly reopens AlphaWdtWriter

### Phase 1 Gate

Phase 1 is **NOT done** until T009 through T011 pass. A blank ADT that doesn't load in the viewer is a failure.

## Phase 2: ADT Relational Schema + Round-Trip Proof

- [ ] T013: Write `wow-viewer/docs/architecture/adt-relational-schema.md` — document every LK ADT chunk as a database table with columns, types, offsets, primary keys, and foreign keys
- [ ] T014: Document Alpha WDT schema in the same document or a companion doc
- [ ] T015: Implement lossless round-trip test: read real LK ADT → `LkAdtData` → write back → compare bytes — fix reader/writer until byte-identical
- [ ] T016: Add round-trip test to `WowViewer.Core.Tests` with a known real ADT file

## Phase 3: Zarr ADT Datastore (Stretch Goal)

- [ ] T017: Define Zarr group schema for ADT tables (metadata, textures, models, placements, chunks)
- [ ] T018: Implement ADT → Zarr reader in `WowViewer.Core.IO/Maps/AdtZarrReader.cs`
- [ ] T019: Implement Zarr → ADT writer in `WowViewer.Core.IO/Maps/AdtZarrWriter.cs`
- [ ] T020: Test: modify a texture name in Zarr store, write back, confirm only MTEX/MMID/MHDR offsets changed