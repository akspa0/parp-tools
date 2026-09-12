# Tasks: Legacy MDX & M2 Rendering Correctness (1.0.0 Through 3.0.1) & Fuckported-Asset Compatibility

**Feature Branch**: `235-legacy-mdx-m2-rendering`
**Owner Spec**: `specs/235-legacy-mdx-m2-rendering/spec.md`
**Plan**: `specs/235-legacy-mdx-m2-rendering/plan.md`

## Phase 0 — Discovery & Real-Client Layout Validation (Completed 2026-09-11)

- [x] T001 [P] [US1] Reconcile `FormatProfileRegistry` vs `M2ModelReaderDispatcher` with code evidence in `research.md`.
- [x] T002 [P] [US1] Resolve test asset naming (`xyz.m2`) and listfile caching for 1.0.0.3980.
- [x] T003 [P] [US1] Binary dump and comparison of M2 headers across 1.0.0, 2.0.0, 2.4.3, 3.0.1, and 3.3.0. Proven unified layout across 0x100–0x107 (0x34 bones, 0x3C keybones, 0x44 vertices, 0x4C embedded views).
- [x] T004 [P] [US1] Prove 108-byte legacy bone layout (0x00 keyBoneId, 0x04 flags, 0x08 parent, 0x0A submesh, 0x0C translation 28B, 0x28 rotation 28B, 0x44 scaling 28B, 0x60 pivot 12B).

## Phase 1 — Core Legacy Reader & Dispatcher Unification (`WowViewer.Core.IO`)

- [x] T010 [P] [US1] Update `M2ModelReaderDispatcher.cs`: drop the `0x102`–`0x107` `NotSupportedException` wall and route all MD20 versions `<= 0x107` (versions <= 263) where `M2Era100ModelReader.ValidateLayout` succeeds to `M2Era100ModelReader`.
- [x] T011 [P] [US1] Generalize `M2Era100ModelReader.cs`:
  - Allow versions `0x100` through `0x107` (do not throw on `version != 0x100`).
  - Implement 108-byte and 112-byte `ReadBones` matching the verified layout and populate `M2ModelDocument.Bones`.
  - Populate `M2ModelDocument.Skins` from the embedded `M2Division` records.
- [x] T012 [P] [US1] Update `M2SkinProfileRuntime.cs` (`WowViewer.Core.Runtime`):
  - When `model.EmbeddedSkinProfileCount > 0` and embedded skin data is available in `M2ModelDocument`, initialize `ActiveSkinProfile` directly from embedded data without attempting external `.skin` file lookup.
- [x] T013 [P] [US1] Add unit tests in `tests/WowViewer.Core.Tests` for legacy M2 parsing (header validation, bone parsing, embedded division parsing across `0x100` and `0x107` fixtures).
- [x] T014 [P] [US1] Phase 1 Verification Gate: `dotnet test` passes with zero regressions in existing test suite.

## Phase 2 — World Placement Routing & Bounding-Box Fallback (`WoWViewer`)

- [x] T020 [P] [US2] Update `WorldAssetManager.cs`:
  - Add explicit handling for `M2Era1121EraTag.Md20_1X_V100_Era100` in the model placement loader so legacy models are parsed and rendered via the legacy embedded path rather than falling through to `ConvertM2ToMdx`.
- [x] T021 [P] [US2] Implement FR-005 Bounding-Box Fallback in `WorldAssetManager.cs`:
  - When a model load fails or returns no renderable geometry, synthesize a bounding-box wireframe / box mesh from the model bounds rather than returning `null`.
- [x] T022 [P] [US2] Phase 2 Verification Gate: Compilation and clean build of `WowViewer.slnx`.

## Phase 3 — Tooling & Multi-Era Client Inspection

- [x] T030 [P] [US1] Update `RunM2Inspect` in `tools/inspect/WowViewer.Tool.Inspect/Program.cs`:
  - When `detectedEra == Md20_1X_V100_Era100`, construct `M2GeometryDocument` from the legacy model's inline geometry rather than running the WotLK-oriented `M2GeometryReader`.
- [x] T031 [P] [US1] Execute `m2 inspect` across staged clients:
  - 1.0.0.3980: `World\ArtTest\Boxtest\xyz.m2`
  - 2.0.0.5610: `CHARACTER\BloodElf\Male\BloodElfMale.m2`
  - 2.4.3.8606: `CHARACTER\BloodElf\Male\BloodElfMale.m2`
  - 3.0.1.8303: `CHARACTER\BloodElf\Male\BloodElfMale.m2`
  - 3.3.0.10958: `CHARACTER\BloodElf\Male\BloodElfMale.m2`
- [x] T032 [P] [US1] Phase 3 Verification Gate: Verify each inspect command succeeds with `available=true` for geometry and `bones > 0`.

## Phase 4 — Documentation & Governance

- [x] T040 [P] Author receipt in `specs/235-legacy-mdx-m2-rendering/evidence/phase1-reader-unification.md` documenting changes, commands, and outputs.
- [x] T041 [P] Update `activeContext.md` and `progress.md`.
