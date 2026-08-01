# Tasks: 064 — Blank Map Generation + Relational ADT Understanding

**Plan**: [plan.md](plan.md) | **Spec**: [spec.md](spec.md)

## Phase 1: Blank LK Map Generation

### P1-A: BlankAdtFactory

- [x] T001: Create `WowViewer.Core/Maps/BlankAdtFactory.cs` — static class with `CreateBlank(mapName, tileX, tileY, l0Texture)` that produces valid `LkAdtData` with 256 MCNK chunks, 1 L0 texture layer, zero placements
- [x] T002: Every MCNK has at least 1 L0 texture layer — default texture is `tileset\ocean\westfallseafloor.blp`, configurable via `--texture` CLI param
- [x] T003: Blank ADT generation: `LkAdtWriter.Build(factory.CreateBlank(...))` produces valid file; `map inspect` confirms all mandatory chunks with correct offsets
- [x] T004: Fix MHDR ofsMCIN bug — offsets must point to chunk header (FourCC+size), not chunk data. Fixed in `LkAdtWriter.cs`

### P1-B: CLI Command + WDT/WDL Generation

- [x] T005: Add `map generate-blank` CLI command — `--tile-x <n> --tile-y <n> [--map-name <name>] [--format lk|alpha] [--texture <path>] [--output-dir <dir>]`
- [x] T006: WDT generation via `LkWdtWriter` with MAIN mask flagging specified tile(s)
- [x] T007: WDL generation via `WdlWriter` with flat heights for specified tile(s)
- [x] T008: Output goes to `World/Maps/<mapname>/` subdirectory so viewer can discover files from client root

### P1-C: Viewer Validation

- [x] T009: Blank LK ADT loads in WowViewer.App without assertion errors or missing chunk warnings
- [x] T010: MCIN offset fix validated — viewer reads MCNK signature correctly
- [x] T011: Confirm flat terrain renders (visible grid) in WowViewer.App — validated 2026-06-16: testing map loads with 335 client + loose overlay

### P1-D: Alpha Blank WDT

- [x] T012: Alpha blank WDT generation works via `AlphaWdtWriter.Build()` with `CreateBlankAlphaTile()` — `--format alpha` CLI flag produces inline MCNK WDT

### Phase 1 Gate

Phase 1 is **done** — all tasks T001-T012 validated.

## Phase 1.5: PM4 Match Patching onto Blank Tiles

- [x] T013: Define PM4-to-blank-ADT patching workflow: generate blank tiles per PM4 match report, then inject MDDF/MODF placements from matched PM4 objects
- [x] T014: Implement placement injection: read PM4 match report (markdown), resolve M2/WMO paths to MTEX/MMDX/MWMO entries, write into `LkAdtData` placements
- [x] T014a: Fix MHDR offset double-subtraction bug — MTEX/MMDX/MMID/MWMO/MWID/MDDF/MODF offsets were 8 bytes too small, causing viewer to read garbage. Fixed in `LkAdtWriter.cs` PatchMhdr call.
- [x] T014b: UniqueID handling — `UniqueIdSource` enum: `PreserveFromCatalog` (default, keeps real IDs from source) or `SyntheticSequential` (allocates from 12M+). CLI: `--unique-id-mode preserve|synthetic`.
- [x] T014c: WDT merge — `--source-wdt <path>` flag copies the real WDT and patches in our tile flag, preserving all existing tiles instead of nuking them with a 1-tile WDT.
- [ ] T015: Validate patched ADT loads in viewer with placed objects visible — needs testing against PM4-recovered tiles (not tiles we already have ADTs for)

## Phase 2: ADT Relational Schema + Round-Trip Proof

- [ ] T016: Write `wow-viewer/docs/architecture/adt-relational-schema.md` — document every LK ADT chunk as database table with columns, types, offsets, PKs, FKs
- [ ] T017: Document Alpha WDT schema in same or companion doc
- [ ] T018: Implement lossless round-trip test: read real LK ADT → `LkAdtData` → write back → compare bytes
- [ ] T019: Add round-trip test to `WowViewer.Core.Tests`

## Phase 3: Zarr ADT Datastore (Stretch Goal)

- [ ] T020: Define Zarr group schema for ADT tables
- [ ] T021: Implement ADT → Zarr reader
- [ ] T022: Implement Zarr → ADT writer
- [ ] T023: Test: modify texture name in Zarr store, write back, confirm only MTEX/MMID/MHDR offsets changed