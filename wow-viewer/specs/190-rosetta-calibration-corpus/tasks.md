# Tasks: Rosetta Calibration Corpus for PM4 Object Identification

**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)

## Phase 1: Generator Enhancements, MCAL Painting, Pedestals & Full-Map Alpha WDT

- [x] **T001**: Era auto-detection from client archives (`.mdx`/`.mdl` vs `.m2`) in `Program.cs`.
- [x] **T002**: Retain file extensions on painted labels with middle-elision on long names.
- [x] **T003**: Implement `RosettaAlphaPainter` for 1024×1024 MCAL text rasterization and 4-bit chunk slicing.
- [x] **T004**: Integrate `RosettaAlphaPainter` into `RosettaTilesetGenerator` to emit Layer 1 MCLY/MCAL textures.
- [x] **T005**: Implement museum pedestal mesh generation in `MCVT` heightmaps with beveled ramps under object cells.
- [x] **T006**: Remove artificial 512-tile map limit for Alpha WDT; default to full 4096-tile capacity.
- [x] **T007**: Add CLI options for `--ink-texture` and `--pedestal-height` in `wowviewer-inspect rosetta-generate`.
- [x] **T008**: Author comprehensive unit tests in `RosettaTilesetGeneratorTests.cs` covering MCAL text, pedestal heights, and full-map Alpha WDT round-trips.
- [x] **T008b**: Align Alpha WDT tile dictionary coordinate transforms in `Program.cs` with `LkWdtWriter` and `RosettaMinimapPainter`.
- [x] **T009**: Add minimap object-band marker regression without changing renderer, terrain-loading, protected writer code, or generated map placement bytes.

## Phase 1.5: Multi-Version Zarr Datastore, Cross-Era Extension Shifting & Viewer Integration

- [x] **T019**: Implement cross-era model extension shifting (`.mdx` $\leftrightarrow$ `.mdl` $\leftrightarrow$ `.m2`) in `WorldAssetManager`, `WmoRenderer`, and `ViewerApp`.
- [x] **T020**: Add `RosettaBuildMetadata`, `RosettaBuildDiff`, and `ComputeBuildDiff` to `RosettaObjectLibrary` and `RosettaDatastoreWriter`.
- [x] **T021**: Add `rosetta-datastore-diff` command in `WowViewer.Tool.Inspect` (`Program.cs`).
- [x] **T022**: Add "Load from Rosetta Datastore..." menu item and modal in `ViewerApp` with Data Version, Map Name, Base Game Version dropdowns, and live cross-build diff statistics.
- [x] **T023**: Update `docs/WoWViewer/USERGUIDE.md` and `README.md` to document Phased Maps and Rosetta Datastore loading / cross-build inspection.
- [x] **T024**: Unit tests in `RosettaDatastoreTests.cs` and `RosettaTilesetGeneratorTests.cs`.

## Phase 2: Reference Library Builder (US2)

- [x] **T010**: Implement synthetic tile reader and placement loader from generated Rosetta maps.
- [x] **T011**: Extract PM4/geometry signatures across all placed Rosetta objects.
- [x] **T012**: Build and serialize `ReferenceLibrary` with versioning and manifest metadata.
- [x] **T013**: Implement library self-test suite verifying $\ge 99\%$ top-1 identification accuracy.

## Phase 2.5: In-Game Alpha 0.5.3 Native Client Ergonomics & DBC Generation (Completed)

- [x] **T025**: Implement `RosettaDbcGenerator` building authentic `Map.dbc` and `AreaTable.dbc` binary files for Alpha 0.5.3.
- [x] **T026**: Implement automatic low-resolution terrain `.wdl` generation via `WdlWriter`.
- [x] **T027**: Implement `minimap.trs` / `md5translate.trs` translation generator in `RosettaMinimapPainter`.
- [x] **T028**: Implement model (`.mdx`) and world model (`.wmo`) map splitting with an 800-tile map budget.
- [x] **T029**: Implement bounding-box centering offset and $+20\text{Z}$ elevation ($Z = \text{groundZ} - \text{BoundsMin.Z} + 20\text{m}$).
- [x] **T030**: Author comprehensive unit tests covering DBCs, TRS, WDL, map splitting, and $+20\text{Z}$ elevation (56/56 passing green).

## Phase 3: Deterministic PM4 Lookup Engine (US3)

- [ ] **T014**: Implement candidate lookup against `ReferenceLibrary` for real PM4 geometry segments.
- [ ] **T015**: Surface structured results (`Identified`, `Ambiguous`, `NoReference`) with comparison evidence.
- [ ] **T016**: Connect lookup engine to Spec 176 Reconciliation UI in the viewer.

## Phase 4: Companion ADT Synthesizer (US4)

- [ ] **T017**: Implement scan for orphan PM4 tiles lacking companion `_obj0.adt`.
- [ ] **T018**: Synthesize minimal compliant companion ADT files with provenance sidecars.
