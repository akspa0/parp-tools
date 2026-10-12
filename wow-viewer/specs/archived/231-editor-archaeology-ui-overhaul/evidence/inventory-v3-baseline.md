# Inventory v3 Baseline — Spec 231 Phase 0 (T001)

**Date**: 2026-09-07 · Captured via the viewer's built-in startup capture automation
(`--capture-shot current --capture-with-ui --exit-after-capture`), one launch per workbench page.

**Run configuration (fingerprint)**

| Item | Value |
|---|---|
| Build | wow-viewer Debug, commit state 2026-09-07 (Spec 231 Phase 1 landed: 4-page Editor IA live) |
| Client root | `H:\CLIENTS\Vanilla\0.x\0_5_3_3368\World of Warcraft` |
| Build version | `0.5.3.3368` |
| World | `World\Maps\Shadowfang\Shadowfang.wdt` (25 tiles, WDL preview) |
| Window / capture | 1600×900, with-UI framebuffer |
| Operator settings | backed up to `viewer_settings.json.pre-231-baseline.bak` and restored after the run |

## Baseline matrix (post-Phase-1 IA)

Each folder under [`screenshots/`](screenshots/) holds the raw capture
(`Shadowfang/0.5.3.3368/<timestamp>_current_<timestamp>_with_ui.png`).

| # | Page | Screenshot folder | Verified content (visual check) |
|---|---|---|---|
| E0 | Editor → Placement & Objects | [editor-0-placement-objects/](editor-0-placement-objects/) | Page combo "Placement & Objects"; sections Tasks & Workspace (Editor Tasks & Inspector, task buttons), Chunk Clipboard, Terrain Import/Export; 3D Object Library + Population below the fold |
| E1 | Editor → Terrain Tools | [editor-1-terrain-tools/](editor-1-terrain-tools/) | Page combo "Terrain Tools"; Terrain Lab section (tile targeting, selection map, focused ADT chunk grid) |
| E2 | Editor → Data I/O | [editor-2-data-io/](editor-2-data-io/) | Page combo "Data I/O"; Imports & Exports dashboard: Synthesized Terrain Minimap, 3D Geometry & GLB Export, Terrain Layer Export/Import |
| E3 | Editor → Converters | [editor-3-converters/](editor-3-converters/) | Page combo "Converters"; converter sub-tab content |
| A0 | Archaeology → Weak Signal & Stratigraphy | [archaeology-0-weak-signal-stratigraphy/](archaeology-0-weak-signal-stratigraphy/) | Archaeology top tab, weak-signal/stratigraphy content |
| A1 | Archaeology → UniqueId Timeline | [archaeology-1-uniqueid-timeline/](archaeology-1-uniqueid-timeline/) | Timeline content |
| A2 | Archaeology → Layers & Provenance | [archaeology-2-layers-provenance/](archaeology-2-layers-provenance/) | Layers & provenance content |
| A3 | Archaeology → Playback & Capture | [archaeology-3-playback-capture/](archaeology-3-playback-capture/) | Playback & capture content |
| A4 | Archaeology → PM4 Analysis | [archaeology-4-pm4-analysis/](archaeology-4-pm4-analysis/) | PM4 Analysis: overlay page, PM4/PD4 loader, surface-class toggles, CK24 splitting |
| A5 | Archaeology → Cartography | [archaeology-5-cartography/](archaeology-5-cartography/) | Cartography content |

## Observations feeding later phases (recorded, not acted on)

1. E0 already stacks Tasks & Workspace / 3D Object Library / Population in one page — Phase 3
   (D2/D4/D5a) will consolidate the transform/correlation panels into it and prune duplication.
2. E2 dashboard still exposes only the terrain-layer/GLB/minimap groups; the five scattered PM4
   JSON/OBJ export button sites (D1) and correlation JSON / object-match / PM4 JSON dump sync
   exports (D6) are the Phase 2 consolidation targets.
3. The Archaeology tab still hosts "Tasks & Workspace"-style editor entries in its navigation
   (D5 removal is Phase 4).
4. Capture status text ("Queued capture …") leaks into the Data I/O dashboard status row —
   shared status label; cosmetic, note for Spec 227 surface polish, not a Spec 231 criterion.

## Scope note

This baseline is the pre-consolidation reference for dedupe receipts D1–D6 (SC-1 grep proofs use
the sources as of this commit). Runtime/interactive acceptance beyond these frames remains
operator-owned per AGENTS.md.
