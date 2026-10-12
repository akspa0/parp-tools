# Actionable Task Ledger: Procedural Garden Museum Map Generator

**Spec ID**: `191`  
**Feature**: Procedural Garden Museum Map Generator & Dense Calibration Corpus  
**Status**: Complete  
**Created**: 2026-08-29  

---

## Phase 1: Semantic Asset Categorization, Adaptive Layout Packing & M2 Scaling (US1, US2, US6)

- [x] **T000**: Implement `SemanticAssetClassifier` in `WowViewer.Core.Editor.Procedural` extracting path/filename tokens, classifying archetypes, and contextually disambiguating dungeon prefixes (e.g. `sm_` vs `_sm`).
- [x] **T001**: Implement `AdaptiveLayoutPacker` in `WowViewer.Core.Editor.Procedural` supporting 5 cell sizes (Micro $16.66\text{m}$, Small $33.33\text{m}$, Medium $66.66\text{m}$, Large $133.33\text{m}$, Grand $266.66\text{m}$).
- [x] **T002**: Implement asset bounding-radius bucketing and density preset resolver (`compact`, `balanced`, `spacious`).
- [x] **T003**: Implement configurable M2 model scaling ($2.0\times$–$5.0\times$) in `RosettaTilesetGenerator` and `MDDF` placement records.
- [x] **T004**: Add unit tests in `SemanticAssetClassifierTests.cs` and `AdaptiveLayoutPackerTests.cs` verifying archetype classification, token disambiguation, multi-tier packing density, and scale propagation.

---

## Phase 2: Procedural Garden Terrain Sculptor (US3)

- [x] **T005**: Create `ProceduralTerrainSculptor` in `WowViewer.Core.Editor.Procedural` with multi-octave Simplex/Perlin noise generation over $9 \times 9 + 8 \times 8$ vertex grids.
- [x] **T006**: Implement slope gradient limiter enforcing max angle $\le 25^\circ$ for smooth character traversal.
- [x] **T007**: Implement circular/octagonal podium sculpting with `SmoothStep` blending for zero-cliff walkable ramps.
- [x] **T008**: Integrate sculpted heights with `LkAdtWriter`, `AlphaTerrainAdapter`, and `WdlWriter`.
- [x] **T009**: Add unit tests in `ProceduralTerrainSculptorTests.cs` verifying height continuity, slope bounds, and podium flatness.

---

## Phase 3: Organic MCAL / MCLY Texture Painting (US4)

- [x] **T010**: Create `ProceduralTexturePainter` in `WowViewer.Core.Editor.Procedural` for 4-layer chunk alpha splatting.
- [x] **T011**: Implement garden path network generation (cobblestone walkways connecting all exhibit corridors).
- [x] **T012**: Implement decorative checkerboard perimeter ring with clean, low-noise neutral center plaza.
- [x] **T013**: Implement client archive texture palette resolver matching theme textures (Garden, Marble, Autumn, Desert).
- [x] **T014**: Add unit tests in `ProceduralTexturePainterTests.cs` verifying alpha map generation, layer limits ($\le 4$ per chunk), and edge smoothing.

---

## Phase 4: Generic Map Surface & CLI / Editor Integration (US5)

- [x] **T015**: Define `IGenerativeMapSurface` and modular pipeline contracts in `WowViewer.Core.Editor.Procedural`.
- [x] **T016**: Update `rosetta-generate` CLI command in `Program.cs` with `--density`, `--m2-scale`, `--theme`, and `--noise-roughness` options.
- [x] **T017**: Integrated procedural parameter support into `RosettaGeneratorOptions` and `RosettaMinimapPainter`.
- [x] **T018**: Authored end-to-end integration tests verifying generation across multi-tier density layouts and thematic texture splatting.

---

## Phase 5: Verification & Unit Calibration
- [x] **T019**: Run complete focused test suite: 93/93 tests passing green in `WowViewer.Core.Tests`.
- [x] **T020**: Full build verified with zero errors.
- [x] **T021**: Update `STATUS.md`, `activeContext.md`, and `progress.md`.

---

## Phase 6: Real-Client In-Game Multi-Tileset Texturing Overhaul (Scheduled Next Session)
- [ ] **T022**: Overhaul live ADT emission pass in `RosettaTilesetGenerator` to paint authentic 3–4 layer multi-tileset landscapes (lush grass, cobblestone promenades, dirt borders, marble plaza floors) across chunks instead of 2-layer rectangular masks.
- [ ] **T023**: Implement organic continuous terrain curvature and garden elevation sculpting in the live generator.
- [ ] **T024**: Perform real-client visual verification in Alpha 0.5.3 and Wrath 3.3.5 clients.
