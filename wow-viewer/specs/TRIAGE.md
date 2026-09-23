# Backlog Triage — Want / Drop / Later

**Opened**: 2026-09-23 after the [spec reconciliation](archived/reconciliation-2026-09-23/README.md).
**Owner**: operator. Every item below is spec-stated residue from an archived spec, or (F-13, R-38) an
operator-reported open item recorded in the memory bank. Nothing here is
scheduled until it is marked **Want**; no implementation approach is chosen until triage is done.

**How to fill in**: put `Want`, `Drop` or `Later` in the *Decision* column. Recommendations are an
agent's suggestion only. Sizes are rough (S ≈ a day, M ≈ days, L ≈ a week+, XL ≈ multi-week).

## 1. "We thought we had it" — measured gaps between belief and code

| Belief | Reality (2026-09-23) | Item |
|---|---|---|
| Composed/merged maps can be saved | No `MapSaveService`; `EditorSession.SaveAll()` writes no file | E-01 |
| Editor edits are undoable | `EditorApplierAdapter` reverses 2 of 5+ ops; rotate/scale/delete/transposition no-op on undo | E-10 |
| Zone music plays (236 notes) | Disabled by policy; `ZoneMusic` still read as a SoundEntries id | W-01 |
| Fog bounds streaming | Fog is render-only; no `fogEnd` admission radius | R-21 |
| Native M2 route (no MDX conversion) | `WorldAssetManager` still falls back to M2→MDX + `MdxRenderer` | F-25 |
| Scene graph drives rendering | Graph selector ships default-off (A/B showed it slower) | R-23 |
| Modern data is interactive | Scene-light gate disables WMO instancing: ~5.5 FPS, 16,431 draws | R-10 |
| Export lands in the workspace | Resolves under `bin/Debug` | E-03 |
| v22 DAT terrain blends | Layer 0 only (all-layers 4096-byte guard + unknown `AMAP` codec) | F-01 |
| Video capture has audio | Play+Video muxing unbuilt | W-03 |
| Chunk clipboard is a plugin | 124 god-object refs remain; migration never started | E-12 |
| `ViewerApp_*` split reduced context | `WorldScene.cs` 17,153 lines, `ViewerApp.cs` 16,746 lines | U-01 |
| CASC remote builds | Local install (+CDN fill) only; CDN-only streaming absent | F-07 |
| Conformance survey exists | Survey library absent (only the `wmo-survey`/`map-survey` verbs) | F-05 |
| `data-paths.md` env-var overrides | None of the 7 documented `WOWVIEWER_*` variables is read by code | memory bank (corrected) |
| Spatial 3D UI shell | Scaffolding with `Enabled=false`, no consumer | U-90 |

## 2. Quick-win shortlist (small, cause known, visible value)

R-10 · R-38 (capture first) · F-01 (guard relax half) · E-03 · U-18 · U-21 · R-30 · F-27 · W-01 · E-10 · U-15 · R-31 (investigate).

## 3. Decisions

### Epic 248 — [Formats, Readers, Writers & Conversion](248-epic-formats-and-conversion/spec.md)

| ID | Item | Size | Rec. | Decision |
|---|---|---|---|---|
| F-01 | v22 AMAP codec + relax partial-alpha guard | S (guard) / L (codec) | Quick win (guard) · Core (codec) | |
| F-02 | Real-rendered DAT tile capture + overview | M | Later | |
| F-03 | Modern → legacy map conversion (LK + Alpha, batch, low-touch) | XL | Core (operator-flagged) | |
| F-04 | Modern chunk completeness survey | M | Core (feeds F-03) | |
| F-05 | Conformance survey library (one component) | M | Core | |
| F-06 | WDT MAI2 liquid flow | M | Later | |
| F-07 | CASC CDN-only streaming + builds by id | L | Later | |
| F-08 | Single file-reference resolver | M | Core | |
| F-09 | Tier A/B (6.x–8.3) coverage | L | Later | |
| F-10 | WMO shader materials + split-group portals | L | Core | |
| F-11 | High-res holes + MTXP blending | M | Later | |
| F-12 | M2 chunk audit, BLP/WDT companions, wiki write-up | M | Later | |
| F-13 | Synthesized minimap for DAT folders | M | Later | |
| F-20 | Single M2↔MDX converter | M | Core | |
| F-21 | "Fuckported"-asset parity | L | Later | |
| F-22 | MDX light-emitter effects | M | Later | |
| F-23 | Profile unification + delete inert profile | S–M | Core | |
| F-24 | Animation addressing model | M | Drop? | |
| F-25 | Remove M2→MDX render fallback | M | Core | |
| F-26 | Published capability tables | S | Later | |
| F-27 | Liquid shoreline-culling fix | S–M | Quick win | |
| F-28 | Converter harness (object validator, corpus gates, oracle, client) | L | Core | |
| F-29 | MoP native semantics + split writer | L | Later | |
| F-30 | MCAL decode correctness | M | Later | |
| F-31 | Asset reference comparison/chronology/repair | L | Drop? | |
| F-90 | Cross-era rig comparison | M | Drop? | |
| F-91 | Benilla oracle methodology | L | Drop? | |
| F-92 | DAT as interchange format | — | Drop? (operator-deferred) | |

### Epic 249 — [Renderer Performance, Lighting & Correctness](249-epic-renderer-performance-and-correctness/spec.md)

| ID | Item | Size | Rec. | Decision |
|---|---|---|---|---|
| R-01 | Modern path-driven renderer benchmark | M | Core (proves R-10) | |
| R-02 | MD21 camera tracks as paths | M | Later | |
| R-03 | Capture receipt model + authoring handoff | M | Core | |
| R-04 | GPU timer-query attribution | M | Later | |
| R-05 | M2/MDX metric close-out (operator flight) | S | Quick win | |
| R-10 | Per-placement WMO instancing under lights | S | Quick win | |
| R-11 | Lit placements off GPU instancing | M | Core | |
| R-12 | Unified batching planner | L | Core | |
| R-13 | Doodad batch planning + diagnostics | M | Core (with R-12) | |
| R-14 | Doodad instancing overhaul | L | Core (with R-12) | |
| R-15 | WMO group admission in dense interiors | L | Core | |
| R-16 | Skin-profile LOD | M | Later | |
| R-20 | Off-thread asset decode | L | Core | |
| R-21 | Fog-bounded residency + lease attribution | L | Core | |
| R-22 | Camera as world actor | L | Later | |
| R-23 | Ordered pass lists, shared spatial queries, modern submission | L | Later | |
| R-30 | Wireframe slope bias + material-gated specular | S | Quick win | |
| R-31 | `WmoRenderer.DrawBatch` access violation | M | Quick win (investigate) | |
| R-32 | Per-era terrain lighting | M | Core | |
| R-33 | WMO/MDX lighting-selection wiring | M | Core | |
| R-34 | Day/night transform calibration | M | Later | |
| R-35 | Skybox rendering | L | Later | |
| R-36 | Shader permutation system | L | Later | |
| R-37 | Diagnostic render profiles | S | Drop? | |
| R-38 | Liquid grid-line / omission defect (operator-reported) | S–M | Quick win (capture first) | |
| R-90 | Shared renderer library promotion | XL | Drop? | |
| R-91 | 4.x renderer evidence epic | XL | Drop? | |

### Epic 250 — [Map Reconstruction, Composition & Editor Platform](250-epic-reconstruction-and-editor-platform/spec.md)

| ID | Item | Size | Rec. | Decision |
|---|---|---|---|---|
| E-01 | Map save pipeline (LK + Alpha, Archaeology + Editor) | L | Core (operator pain point) | |
| E-02 | Cartography project persistence | M | Core | |
| E-03 | Export path under workspace | S | Quick win | |
| E-04 | New Map creator acceptance | S | Core | |
| E-10 | Complete undo applier for all ops | M | Quick win (correctness) | |
| E-11 | One undo/save path + exit warning | M | Core | |
| E-12 | Chunk clipboard plugin migration | M | Later | |
| E-13 | Add-placement in viewport | M | Later | |
| E-14 | Edit journal / crash recovery | L | Later | |
| E-15 | DBC browser + editing | L | Later | |
| E-16 | Asset integrity census + repair | L | Later | |
| E-17 | ADT tile creation | M | Later | |
| E-18 | PM4-guided object transfer completion | L | Later | |
| E-20 | uniqueId placement reconciliation | M | Core | |
| E-21 | Ortho selection canvas; Chunk Manipulator retirement | L | Later | |
| E-22 | 45°/free-angle rotation | M | Drop? | |
| E-23 | Composition regression suite + doc cleanup | M | Core | |
| E-24 | Layer-stack panel; DBC child-map suggestions | M | Later | |
| E-25 | Operator-reported cartography defects (T067–T071) | M | Core | |
| E-26 | Composed minimap synthesis; stripped-map textures; MDX light regression | M | Later | |
| E-27 | UniqueId colouring; chunk off-by-one re-audit | S | Core | |
| E-28 | Transplant provenance record; pop-out picker | M | Drop? | |
| E-30 | Client-constrained generator assets | M | Core | |
| E-31 | Generator texturing/curvature + UI panel | M | Later | |
| E-32 | Rosetta-indexed placement | M | Later | |
| E-33 | WMO doodad editing + WMO writing | XL | Later | |

### Epic 251 — [Viewer UX, Shell & Code Health](251-epic-viewer-ux-and-code-health/spec.md)

| ID | Item | Size | Rec. | Decision |
|---|---|---|---|---|
| U-01 | God-class extraction (selection service first) | L | Core | |
| U-02 | Governance Gate 1 + ledger sync | S | Core | |
| U-03 | MCP tooling/automation surface | L | Later | |
| U-10 | Sidebar standard (SharedUiWidgets) | M | Core | |
| U-11 | Dedupe weak-signal / minimaps / Inspector | M | Core | |
| U-12 | Approachability pass | M | Core | |
| U-13 | Unified Inspector | M | Core | |
| U-14 | Retire floating windows | M | Later | |
| U-15 | Fog Defaults single source | S | Quick win | |
| U-16 | Editor/Archaeology merge + re-split | M | Later | |
| U-17 | Converters page + removal pass | M | Later | |
| U-18 | Doodad-set combo regression | S | Quick win | |
| U-19 | Single ownership of renderer/lighting controls | S | Core | |
| U-20 | Tools menu inventory + diagnostics | S | Later | |
| U-21 | Toolbar width scoping | S | Quick win | |
| U-30 | WoW shell + keybind profiles | L | Later | |
| U-31 | Overhead ortho view | M | Later | |
| U-32 | Selection-silhouette outlines | M | Later | |
| U-33 | Simple viewer surface | M | Drop? | |
| U-34 | Workspace modes + MK Dataset purge | M | Later | |
| U-90 | 3D spatial UI shell | XL | Drop? | |

### Epic 252 — [World Simulation, Audio & Interaction](252-epic-world-simulation-and-audio/spec.md)

| ID | Item | Size | Rec. | Decision |
|---|---|---|---|---|
| W-01 | ZoneMusic indirection → enable zone music | S–M | Quick win | |
| W-02 | Camera-path audio binding | M | Later | |
| W-03 | Play+Video audio muxing | M | Later | |
| W-04 | Audio diagnostics + bus controls | M | Later | |
| W-05 | Audio lifecycle correctness | M | Core | |
| W-06 | Event seam, optional backends | L | Drop? | |
| W-10 | WMO interior area context | M | Later | |
| W-11 | Player-head camera rig | M | Later | |
| W-12 | Cross-era context/lighting gate | S | Later | |
| W-13 | Game mode (head camera + bounded physics) | L | Later | |
| W-14 | Triangle-precise picking | M | Later | |
| W-15 | Alpha demo restoration | L | Drop? (blocked on data) | |
| W-16 | Finish WTF sweep | S | Later | |
| W-20 | 5.0.1 physics solver integration | XL | Later | |
| W-21 | 5.0.1 weather | L | Later | |
| W-22 | Model cursor as light source | M | Later | |
| W-23 | Creature staging | L | Later | |
| W-24 | Server data transformer | XL | Drop? | |
| W-25 | Museum world simulation | XL | Drop? | |

### Epic 253 — [PM4/PD4 Navmesh](253-epic-pm4-navmesh-research/spec.md)

| ID | Item | Size | Rec. | Decision |
|---|---|---|---|---|
| P-01 | Remaining decode | L | Later | |
| P-02 | Field map close-out | M | Later | |
| P-03 | Terminology restoration (renames) | S–M | Quick win | |
| P-04 | Live grouping readout | M | Later | |
| P-05 | Top-1 disambiguation + WMO enumeration | M | Later | |
| P-06 | Rotation recovery | M | Later (needs data) | |
| P-07 | Negative-BSP matching | L | Drop? | |
| P-08 | PM4 Zarr dataset | L | Drop? | |
| P-09 | Generation from source geometry | XL | Later | |
| P-10 | Region browser; remove Correlation tab | M | Later | |
| P-11 | Confirmed-match library in viewer | M | Later | |

### Epic 254 — [Datasets, Client Datastore & Terrain ML](254-epic-datasets-and-terrain-ml/spec.md)

| ID | Item | Size | Rec. | Decision |
|---|---|---|---|---|
| D-01 | MPQ patch-chain resolver | M | Later | |
| D-02 | Multi-build datastore | XL | Later | |
| D-03 | Incremental processing | L | Later | |
| D-04 | Adaptive encoding | M | Later | |
| D-05 | Viewer loads Zarr datastore | L | Later | |
| D-06 | Zarr-first residency | L | Drop? | |
| D-10 | Real WDL prior + residual detailer model | L | Later | |
| D-11 | Clean-signal regression fix + promotion | M | Later | |
| D-12 | Later-client v60 adapters | M | Later | |
| D-13 | Texture-family + alpha-stack reconstruction | L | Later | |
| D-14 | Research-lead ledger | S | Drop? | |
| D-20 | Weak-tile explorer UI | M | Later | |
| D-21 | Brush/motif archaeology (132+140) | XL | Drop? | |
| D-30 | Legacy Python detangle + RunPod tooling | L | Later | |
| D-90 | Minimap super-res etc. | L | Drop? | |
| D-91 | Minimap texture tier + seams | L | Drop? | |

## 4. After triage

For each epic: record decisions as a dated amendment in its `spec.md`, then run `speckit-tasks` for the
**Want** items only, choosing the implementation approach in that epic's `plan.md`.
