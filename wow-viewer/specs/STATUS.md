# Spec Status Router

**Reconciled 2026-09-23** (branch `v0.6.0-dev`). Every earlier spec was audited against the code and
archived; open work now lives in **7 epics** (248–254). Receipt and per-spec ledger:
[archived/reconciliation-2026-09-23/README.md](archived/reconciliation-2026-09-23/README.md).
Previous router: [archived/status-history/2026-09-23-status-pre-reconciliation.md](archived/status-history/2026-09-23-status-pre-reconciliation.md).

## Current step — operator triage

**Operator P1 (2026-09-23, both Want, approaches approved):** R-10 modern-data lighting performance
(Epic 249) — code landed, operator captures owed; U-01 god-class decomposition (Epic 251) — E1 landed
2026-09-25, E1 smoke owed. R-10's after-capture lands before U-01 step E2. U-01 E3 and the operator-directed ViewerApp campaign
(2026-09-25) landed; smoke owed.


Nothing else is scheduled. The operator marks each backlog item **Want / Drop / Later** in
[TRIAGE.md](TRIAGE.md); only then does each epic get phased `tasks.md` and a chosen implementation
approach. TRIAGE.md opens with the measured "we thought we had it" gaps and a quick-win shortlist.

## Active epics

| Epic | Scope | Backlog | Status |
|---|---|---|---|
| [248 Formats, Readers, Writers & Conversion](248-epic-formats-and-conversion/spec.md) | Legacy + modern (CASC/FDID/DAT) readers and writers; modern→legacy conversion; converter harness | F-01…F-31, F-9x | Triage pending |
| [249 Renderer Performance, Lighting & Correctness](249-epic-renderer-performance-and-correctness/spec.md) | Benchmarks and receipts, batching/instancing, WMO admission, streaming, lighting, correctness | R-01…R-38, R-9x | **R-10 P1**: code landed 2026-09-23; per-placement WMO gate reverted 2026-09-27 (operator saw <1 FPS); FPS check + captures owed. **R-39** CASC minimap change reverted 2026-09-27 (made things worse); measure first; rest pending |
| [250 Map Reconstruction, Composition & Editor Platform](250-epic-reconstruction-and-editor-platform/spec.md) | Map save, undo/session, cartography composition, generator, editor plugins | E-01…E-33 | Triage pending |
| [251 Viewer UX, Shell & Code Health](251-epic-viewer-ux-and-code-health/spec.md) | UI consolidation, god-class extraction, governance, automation surface | U-01…U-34, U-90 | **U-01 P1**: E1 + E3 + ViewerApp campaign landed 2026-09-25, E4 2026-09-26 (`WorldScene.cs` 17,175 → 8,326; `ViewerApp` class 42,973 → 2,512 lines); smoke owed; rest pending |
| [252 World Simulation, Audio & Interaction](252-epic-world-simulation-and-audio/spec.md) | Audio, area context, camera, picking, 5.0.1 physics/weather, server data | W-01…W-25 | Triage pending |
| [253 PM4/PD4 Navmesh](253-epic-pm4-navmesh-research/spec.md) | Decode, field naming, matching, placement restore, generation | P-01…P-11 | Triage pending |
| [254 Datasets, Client Datastore & Terrain ML](254-epic-datasets-and-terrain-ml/spec.md) | Multi-build datastore, v50/v60 models, archaeology tooling | D-01…D-30, D-9x | Triage pending |

## Active specs

| Spec | Scope | Status |
|---|---|---|
| [255 WorldScene Decomposition](255-worldscene-decomposition/spec.md) | Split `WorldScene.cs` (8,275 lines at start) into owned scene services, continuing Epic 251 U-01 | Plan approved 2026-09-27; **W0–W7 landed 2026-09-27** (8,275 → 4,922 lines; smoke T018 owed); W8/W9/E2 gated on R-10 |
| [256 Modern-Data Streaming & M2 Submission](256-modern-data-streaming-performance/spec.md) | Name the hidden frame time; off-thread model loading; native M2 instancing (measured baseline: 4 FPS, 98 ms/frame loading, 0 of 5,169 M2s instanced) | Approved 2026-09-27; **P0, P1, minimap budget, P2a/b/c, D, P3, A, and file splits (T060/T092) landed**; operator captures owed (T015, T022, T042, T073, T081, T102) |
| [257 Unified Video Recording & Automation](257-unified-recording-system/spec.md) | Centralized recording coordinator, taxi node stop-crash fix, automated CLI video recording, camera path and feature tour integration | Implemented (2026-10-02); receipt in `evidence/receipt-spec257.md`; operator smoke owed |
| [258 Taxi Route Playlists & Dynamic Marketing Showreel Tour](258-taxi-playlist-showreel-tour/spec.md) | Multi-route taxi flight chaining/playlists, continuous multi-segment recording, dynamic showreel telemetry HUD, zone banners, and landmark discovery callouts | Implemented (2026-10-02); receipt in `evidence/receipt-spec258.md`; operator smoke owed |
| [259 Ground Effects & WMO Detail Doodad Engine](259-ground-effects-and-wmo-detail-doodads/spec.md) | Dynamic grass/foliage on terrain and WMO surfaces (`MDDL`), client slope ($Z \ge 0.4$) & MCCV rules, 1.60 CASC support, and instanced GPU rendering | Original work implemented (2026-10-02); operator-requested tint propagation was reverted 2026-10-14 after a viewer-load regression report; AC-009/T014 remain open; real-client proof and oversized `MdxRenderer` split follow-up owed |
| [260 Live Engine Diagnostics, Pipeline Telemetry & Hitch Overlay](260-engine-diagnostics-telemetry-overlay/spec.md) | Real-time pipeline counters (M2 instancing, flora, WMO, ADT), memory & draw calls, rolling hitch detection, bottleneck stage attribution, and diagnostic HUD | Implemented (2026-10-02); receipt in `evidence/receipt-spec260.md`; operator smoke owed |
| [261 WoW Forever Terrain Hole Fidelity](261-wow-forever-terrain-hole-fidelity/spec.md) | WoW Forever (11.2.7 / 1.60.1) 64-bit high-res hole decoding, ADT v18 format conformance, Little-Endian bit-mapping, and cross-subsystem hole consistency | Implemented (2026-10-04); receipt in `evidence/receipt-spec261.md`; operator smoke owed |
| [262 Minimap Residual Model & 3D Fractal Brush Reconstruction](262-minimap-shadow-sieve-terrain-reconstruction/spec.md) | Automated lighting calibration, SAM 3.1 minimap object sieve, Rosetta overhead vision catalog, 3D fractal editor brush discovery ($\Delta Z$ + $\alpha$), and $\ge 75\%$ terrain mesh reconstruction | Complete (2026-10-07); operator smoke passed on RTX 4070 Ti SUPER (SAM 3.1 + AC-007 97.37% parity); receipt in `evidence/receipt-spec262.md` |
| [263 1.60 MCCV Terrain Shadow Ground-Truth Validation & Bidirectional Synthesis](263-mccv-terrain-shadow-validation-and-synthesis/spec.md) | 1.60 MCCV terrain vertex shadow extraction (145 vertices BGRA), cross-correlation against 1.12 bare minimap residual shadows, geographic sampling stratification (Westfall/STV/Elwynn), and bidirectional ADT synthesis | Complete (2026-10-07); receipt in `evidence/receipt-spec263.md`; NCC 0.9242 / 100% ridge coincidence |
| [264 Authentic Development Shadow-to-Height & Rosetta Object Placement](264-development-ground-truth-shadow-height-and-object-placement/spec.md) | Calibration of shadow residual signals to real world-space Z elevations (yards) using non-museum assets (`test_data/original_development`), Rosetta object placements (`_obj0.adt`), leveled building foundation plateau carving, and side-by-side 3D GLB/OBJ validation | In Progress; Poisson SFS hits dead end (land $r=0.33$, MAE 31.7 yds, max error 179 yds); deep elevation network mapping minimap RGB to absolute yards required |
| [265 Two-Stage Residual Elevation Refinement & Scale Calibration](265-residual-elevation-refinement-and-scale-calibration/spec.md) | Two-stage cascaded residual network predicting $\Delta Z = Z_{\text{gt}} - Z_{\text{initial}}$ to uncompress squashed mountain spans ($>200$ yds), learn explicit error attribution, and calibrate boundary seams across authentic development tiles | Active (2026-10-08); Spec Kit authored; Phase 1 underway |
| [266 WDL Trestle Elevation Reconstruction & Lineage Synthesis](266-wdl-trestle-elevation-reconstruction/spec.md) | Modern lean successor to Feb 2026 v7 WDL trestle model (4.7M params vs 100M+); synthesizes missing development WDL lattices from Northrend 3.0.1.8303 prototype matches, restoring authentic $>250$ yd mountain relief (measured 424.8 yds) and upright CCW meshes | Complete (2026-10-08); validation MAE 5.20 yds (target $\le 25$), $r=0.8754$, 100% upward normals (+Z) verified; receipt in `tasks.md` |
| [267 Generative 3D Foundation Refinement & ComfyUI Nodes](267-trellis-pixel3d-comfyui-adt-refinement/spec.md) | Downstream 3D foundation model refinement (Microsoft Trellis.2 / Pixel3D); hard-Z metric WDL elevation anchor; semantic ADT aspect dissection; parp-tools ComfyUI custom node package (`http://127.0.0.1:8199`) | Proposed / Specified (2026-10-08); Spec Kit authored; Phase 1 discovery queued |
| [269 v60 Classic Terrain Zarr Ingestion & Felwood Clean Elevation Training](269-v60-classic-terrain-ingestion-and-elevation-dataset/spec.md) | Ingest Wrath 3.0.1.8303 terrain (37 maps, 4,448 tiles) via C# harvest engine into `output/datasets/v60/`, train `TrestleElevationUNet` on authentic Felwood canyons/indentations (120 epochs, MAE 4.62 yds, $r=0.8842$), and carve $-96.13$ yd canyon into unauthored quilt tile `(17, 14)` | Complete (2026-10-10); receipt in `evidence/receipt-spec269.md` |
| [270 PM4 Object Association Ledger & Unauthored Placement Pipeline](270-pm4-object-association-ledger-and-placement/spec.md) | Harvest PM4 $\leftrightarrow$ ADT ground-truth object associations across 399 intact development tiles into a persistent Rosetta Ledger (`pm4_object_association_ledger.json`), fingerprint multi-feature geometry, and synthesize authentic `_obj0.adt` placements on unauthored tiles | Complete (2026-10-11, shipped in v0.6.0); receipt in `evidence/receipt-spec270.md`; 616/616 tiles round-trip; **open gap**: objects on tiles adjacent to PM4-covered tiles that exist in museum data are still missing |
| [271 v0.6.0 Release & Integrated Help / Image Guide](271-v060-release-and-integrated-help/spec.md) | Ship v0.6.0 (docs rewrite, BYO-data, support matrix, Archaeology phase maps) and build in-app help + image guide on `v0.6.1-dev` | Phase A v0.6.0 shipped; v0.6.0.1 reissued 2026-10-14 with tint code reverted and eight split/deduplicated archives; operator map-load verification owed; AC-C3 lighting deferred; Phase B queued for `v0.6.1-dev` |
| [272 WC3 File Browser and Tileset Preview](272-wc2-wc3-viewer-support/spec.md) | WC3 loose/MPQ data, BLP0 external-mip/image/build-text preview, isolated existing-renderer MDX preview and selected-file unpacking; shared readers unchanged | **Beta follow-up implemented 2026-10-15 UTC**: real Crown1 decodes (128×32 / mip 3 16×4), adjacent v800 MDX parses (6 geosets / 90 vertices / 3 sequences); viewer build 0 errors, 86 focused tests pass. Actual model/image appearance and world isolation still operator-owned. [Receipt](272-wc2-wc3-viewer-support/evidence/receipt-beta-blp0-and-mdx-2026-10-15.md) |
| [273 Client Ground-Truth Renderer Performance](273-client-ground-truth-renderer-performance/spec.md) | Ground-truth 3.3.5a decompilation renderer refinements: decoupled shell instancing (R-10), spatial M2 doodad submission, and terrain streaming optimization committed; WMO portal occlusion quarantined with fail-safe frustum admission active pending dedicated fresh-chat investigation | Partially Shipped (2026-10-18); lights/streaming committed (`0ab72a43`); WMO culling quarantined |
| [274 Minimap Hillshade Correlation & Domain Translation](274-minimap-hillshade-correlation-and-domain-translation/spec.md) | Batch-export paired clean dataset across Zarr v3 map revisions (real minimap, synthetic DXT1 hillshade, WDL lattice, ADT heightmap), train lightweight domain translation network for object stripping & texture normalization, and correlate clean hillshade directly with 3D terrain implication data | Implemented & Committed (2026-10-18, `e667ec6a`); clean dataset pipeline & domain translation model complete; 228/228 tests pass |
| [275 Improved Asset Streaming for MPQ and CASC](275-asset-streaming-mpq-casc/spec.md) | High-performance off-thread model decoding, MPQ/CASC lookahead dependency prefetching, in-memory WMO parsing, dynamic frustum priority queue, and direct DXT WMO material streaming | Specified (2026-10-18); Spec Kit authored; Phase 1 underway |
| [276 TrinityCore Multi-Version Population Pipeline & Ground Effect Fidelity](276-trinitycore-population-pipeline-and-ground-effect-fidelity/spec.md) | TrinityCore multi-version SQL extraction pipeline, full multi-map streaming (all 113 maps in 3.3.5a, all 31 maps in 0.5.3), automatic client-version detection & map loading, Map 451 developer test spawns + alpha-core corner dev transposition, NPC idle animation selection, authentic entity nameplates / inspector display, and ground-truth 4.3.4 detail doodad alpha blending | Complete & Verified (2026-10-18); extracted 3.3.5a (113 maps, 246k spawns) & 0.5.3 (31 maps, 92k spawns) datasets; auto-streaming on map load; entity names & floating nameplates wired; unit tests pass; solution builds with 0 errors; receipt in `tasks.md` |
| [277 UniqueId Stratigraphy & Cross-Map Chronological Fork Analysis](277-uniqueid-stratigraphy-and-cross-map-chronology/spec.md) | UniqueId work-layer stratigraphy, quantiles ($p_{10}..p_{90}$), cross-map spatial twin matching, branching boundary ($UID_{div}$) detection, CLI divergence report (`map stratigraphy`), and interactive in-viewer Archaeology Time Machine | Complete & Verified (2026-10-19); Core models, stratigraphy and divergence engines implemented; CLI profiler and Archaeology workbench slider verified; 5/5 unit tests pass; 0.5.3 loading performance fix resolved; receipt in `tasks.md` |


## Operator verification owed on shipped code

Each epic's `tasks.md` carries a **verification sweep** (V-tasks): proof-only checks on code that
already shipped, including everything v0.6.0-alpha2 shipped unverified (M2 texture-wrap fix,
`LkAdtWriter` chunk fixes, DAT→LK export client load, DAT Cartography layer on screen, new export
buttons).

## Release line

`eng/Version.props` = `0.6.1` / `0.6.1-dev` on active `v0.6.1-dev`. Latest stable: reissued `v0.6.0.1` (2026-10-14; eight split/deduplicated ZIP assets); operator map-load verification remains owed; base stable: `v0.6.0` on `main` (2026-10-11).
The v0.6 theme — modern data access + experimental terrain formats — maps to items F-01…F-12 (Epic 248)
and R-01, R-02, R-10 (Epic 249). Release scope for v0.6.1 is set by Spec 271 Phase B and operator triage.

## Status rules

- An epic backlog item is **not** scheduled until marked Want in TRIAGE.md.
- `[x]` requires a §9.2 receipt (files, commands + exit status, criterion→evidence with real output).
  Compilation and unit tests never prove runtime, visual, FPS, audio or client behaviour.
- Archived specs are history, not authority. An archived design document is authority only for the
  epic item whose `plan.md` adopts it by link.
- New work: add an item to the owning epic (operator-approved wording, §9.1) or, for a genuinely new
  area, a new spec number ≥ 255 registered here. Follow AGENTS.md §9–11.
- Monthly `speckit-cleanup` audits the epics' checked tasks; next due 2026-10-01.
