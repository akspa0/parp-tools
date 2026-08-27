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

## Phase 2: Reference Library Builder (US2)

- [ ] **T009**: Implement synthetic tile reader and placement loader from generated Rosetta maps.
- [ ] **T010**: Extract PM4/geometry signatures across all placed Rosetta objects.
- [ ] **T011**: Build and serialize `ReferenceLibrary` with versioning and manifest metadata.
- [ ] **T012**: Implement library self-test suite verifying $\ge 99\%$ top-1 identification accuracy.

## Phase 3: Deterministic PM4 Lookup Engine (US3)

- [ ] **T013**: Implement candidate lookup against `ReferenceLibrary` for real PM4 geometry segments.
- [ ] **T014**: Surface structured results (`Identified`, `Ambiguous`, `NoReference`) with comparison evidence.
- [ ] **T015**: Connect lookup engine to Spec 176 Reconciliation UI in the viewer.

## Phase 4: Companion ADT Synthesizer (US4)

- [ ] **T016**: Implement scan for orphan PM4 tiles lacking companion `_obj0.adt`.
- [ ] **T017**: Synthesize minimal compliant companion ADT files with provenance sidecars.
