# Tasks: Legacy M2 rendering — 1.0.0 M2 slice

**Input**: `spec.md`, `plan.md`, `research.md`, `research-1.0.0-ghidra-trace.md`,
`data-model.md`, and `contracts/m2-format-profile.md`.

**Scope**: only the 1.0.0 `MD20`/`0x100` classic-layout M2 route. A shared numeric version with
1.12.1 does not authorize sharing the layout reader.

## Phase 1: Setup and contract

- [ ] T001 Record the 1.x M2-not-MDX rule and the no-cross-layout-fallback requirement in `wow-viewer/specs/104-legacy-m2-rendering/spec.md` and `research.md`.
- [ ] T002 [P] Add the 1.0.0 staged-model validation procedure and evidence fields in `wow-viewer/specs/104-legacy-m2-rendering/quickstart.md` and `contracts/m2-format-profile.md`.

## Phase 2: Foundational classification proof

- [ ] T003 Add a synthetic classic-layout `0x100` header fixture and an era-100 classification assertion in `wow-viewer/tests/WowViewer.Core.Tests/M2Era1121ModelReaderTests.cs`.
- [ ] T004 Add a distinct 1.12.1-shaped `0x100` fixture and assert it remains on the era-1121 route in `wow-viewer/tests/WowViewer.Core.Tests/M2Era1121ModelReaderTests.cs`.
- [ ] T005 Add malformed/truncated classic-header tests proving `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/M2ModelReaderDispatcher.cs` never reclassifies or crashes.

## Phase 3: User Story 1 — load a 1.0.0 M2 as M2 (Priority: P1)

**Goal**: A 1.0.0 asset reaches the dedicated M2 reader and the user is never told to use MDX/MDL.

**Independent test**: A valid classic-layout header selects era-100; a parser failure reports the
era-100 M2 error without generic-parser fallback.

- [ ] T006 [US1] Validate and complete classic-layout header, division, vertex, section, batch, and texture bounds checks in `wow-viewer/src/core/WowViewer.Core.IO/M2Era100/M2Era100ModelReader.cs`.
- [ ] T007 [US1] Preserve the layout discriminator and route classic `0x100` documents only to `M2Era100ModelReader` in `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/M2ModelReaderDispatcher.cs`.
- [ ] T008 [US1] Carry resolved era-100 geometry through `wow-viewer/src/core/WowViewer.Core/M2/M2ModelDocument.cs` and `wow-viewer/src/core/WowViewer.Core/M2/M2Era100Geometry.cs` without widening consumer contracts.
- [ ] T009 [US1] Build era-100 render vertices, sections, batches, and texture references directly into `M2StaticRenderModel` in `wow-viewer/src/viewer/WoWViewer/Rendering/WowViewerM2RuntimeBridge.cs`.
- [ ] T010 [US1] Make era-100 parse failure terminal and descriptive, with no generic M2/MDX fallback, in `wow-viewer/src/viewer/WoWViewer/Rendering/WarcraftNetM2Adapter.cs`.
- [ ] T011 [US1] Replace the incorrect MDX/MDL standalone-load advice with M2-specific wording in `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs`.
- [ ] T012 [US1] Route era-100 standalone loads only through `LoadM2RuntimeModel` in `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs`; do not construct `MdxFile` or `MdxRenderer` on this path.
- [ ] T013 [US1] Run focused unit tests and `dotnet build wow-viewer/WowViewer.slnx -c Debug`; record code-proof output in `wow-viewer/specs/104-legacy-m2-rendering/contracts/m2-format-profile.md`.

## Phase 4: User Story 2 — real 1.0.0 render signoff (Priority: P2)

**Goal**: A staged 1.0.0 M2 visibly renders its mesh and materials through the new route.

**Independent test**: The user loads a named `.m2` from the trusted staged client and compares it
with a reference render.

- [ ] T014 [US2] Select a representative model from `I:/parp/parp-tools/output/tmp/wowarchive-clients/` and record its client root, virtual path, and MD20 version in `wow-viewer/specs/104-legacy-m2-rendering/contracts/m2-format-profile.md`.
- [ ] T015 [US2] Have the user load that `.m2` in `WowViewer`; verify the info panel says `M2Renderer`, then record detected era, visible mesh/material result, and any error in `wow-viewer/specs/104-legacy-m2-rendering/contracts/m2-format-profile.md`.
- [ ] T016 [US2] Compare the same model against a reference render and record the outcome in `wow-viewer/specs/104-legacy-m2-rendering/contracts/m2-format-profile.md`.
- [ ] T017 [US2] User-check one WotLK+ model through the external-skin route and record the no-regression result in `wow-viewer/specs/104-legacy-m2-rendering/contracts/m2-format-profile.md`.

## Phase 5: Deferred independent stories

- [ ] T018 [US3] After T017 passes, define a separate 1.12.1 `0x100` layout slice in `wow-viewer/specs/104-legacy-m2-rendering/plan.md` before changing its reader.
- [ ] T019 [US3] After the 1.12.1 slice is signed off, create one bounded TBC version-boundary slice in `wow-viewer/specs/104-legacy-m2-rendering/plan.md`.

## Dependencies and MVP

- T001–T005 block T006–T013.
- T014–T017 require T013 and user-run viewer access.
- T018 and T019 are explicitly out of scope until the 1.0.0 phase gate passes.
- MVP: T001–T013. It proves correct M2 ownership and safe failure but does not claim visual signoff.
