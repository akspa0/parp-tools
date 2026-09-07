# Active Context — wow-viewer

Last updated: 2026-09-07

## Fresh-chat route

1. [Spec status](../specs/STATUS.md) — select exactly one active owner.
2. This compact handoff.
3. That owner's `spec.md`, `plan.md`, and `tasks.md`, and linked receipt only.

The [documentation router](../docs/README.md), [spec routing registry](../specs/registry.md), and
[archives](archive/README.md) are on-demand context, never default reading.

## Current lane — Spec 231 Editor/Archaeology UI Overhaul (Phase 0–4 complete, T041 open)

- **Spec + plan + tasks** at [231-editor-archaeology-ui-overhaul/](../specs/231-editor-archaeology-ui-overhaul/spec.md):
  4-page Editor IA (Placement & Objects / Terrain Tools / Data I/O / Converters), Archaeology
  de-hosting, dedupe moves D1–D6, Spec 228 extraction pattern, phased gates P0–P5.
- **Landed 2026-09-07**: Phase 0 complete — T002 (operator scope-freeze ack), T003 (supersession
  note in 227 tasks.md), T001 (inventory-v3 baseline: 10-page screenshot matrix captured via the
  viewer's built-in `--capture-shot current --capture-with-ui --exit-after-capture` automation on
  `H:\CLIENTS\Vanilla\0.x\0_5_3_3368` / Shadowfang WDT; operator settings backed up + restored).
  Phase 1 complete — `Workbench/Pages/` page classes, 4-label navigator, delegation switch,
  settings-load remap; build 0 errors.
  Receipts: [evidence/](../specs/231-editor-archaeology-ui-overhaul/evidence/inventory-v3-baseline.md).
- **Phase 2 landed 2026-09-07** (86394fb6): D1 single export draw site + "Open Data I/O" links;
  D6 backgrounded JSON dumps. Receipt: [evidence/t020-t023-phase2-receipt.md](../specs/231-editor-archaeology-ui-overhaul/evidence/t020-t023-phase2-receipt.md).
- **Phases 3+4 landed 2026-09-07** (operator pushed on visible mess): D2 — workbench correlation
  subset deleted (149 lines), both routes draw the canonical correlation page; D4 — doodad-set
  combos down to toolbar quick combo + Selected WMO Controls; D3 — both dead duplicate
  Clipboard+Save hosts deleted (floating Terrain Workbench was unreachable); D5 — dead Archaeology
  editor dispatcher deleted (0 refs). Placement & Objects page de-cluttered (Tasks open; PM4
  Placement Tools / Library / Population collapsed). **Operator-directed**: PM4 Glossary rewritten
  to the 2026-08-24 measured truths (0x18 = MSLK window, CK24 = placement-Z float, MSHD spans).
  Sidebars 6,173→6,013; Pm4Utilities 4,451→4,368. Receipt:
  [evidence/t030-t043-phase3-4-receipt.md](../specs/231-editor-archaeology-ui-overhaul/evidence/t030-t043-phase3-4-receipt.md).
- **Open**: T041 (Spec 221 harness surfaces on Converters page); Tasks-panel interior still embeds
  Chunk Clipboard + Terrain Import/Export blocks → relocate to Terrain Tools / Data I/O (next
  visible de-clutter); Phase 5 (operator navigation smoke, inventory v3 final).
- **Commit**: Phase 3+4 = the commit after 86394fb6.
  Capture automation recipe: patch `viewer_settings.json` (ActiveTopTab 5=Editor /
  6=Archaeology, ActiveBottomTab=page, WorkbenchNavigationVersion=4) then launch the exe with
  `--game-path --build --world 'World\Maps\Shadowfang\Shadowfang.wdt' --capture-shot current
  --capture-with-ui --capture-output <dir> --capture-after-frames 120 --exit-after-capture`.

## Landed this session (awaiting operator visual receipts)

- Version plumbing repaired (eng/Version.props import chain + de-hardcoded both viewer csprojs);
  both targets stamp `0.5.3-rc1`; release notes rewritten as a real changelog
  ([docs/releases/v0.5.3-rc1.md](../docs/releases/v0.5.3-rc1.md)).
- Wireframe root cause fixed (textured lines were invisible / alpha-cutout orange fragments):
  flat-color passes in TerrainRenderer (semi-transparent white), M2Renderer, ModelRenderer (MDX);
  selection now draws red wireframe instead of bounding boxes (box fallback only when the model
  is not streamed).
- Editor toolbar: `Anim` toggle (world doodad animations default ON now), hovered-WMO doodad-set
  combo; left sidebar "Layers, Grids & Overlays" collapsed by default; Imports & Exports groups
  collapsed by default.
- PM4 OBJ export freeze fixed: export moved off the render thread with status + re-entrancy
  guard ([ExportPm4ObjectsObjSet](../src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs)).
- All of the above are build-verified (0 errors); interactive/visual acceptance remains
  operator-owned.

## Separate operator gates

- Spec 223 retest (fog, WMO-only global WMO, playback/capture, ffmpeg, spatial UI) with a real
  client.
- v0.5.3-rc1: tag push + GitHub Actions release are operator-owned; visual pass on wireframe,
  selection, and toolbar changes.
- Spec 226 wireframe items: root causes fixed; captures still owed.

## Non-negotiable constraints

- Preserve MPQ/ADT/WMO/M2/MDX readers and `AlphaWdtWriter`; no feature scope rides on a refactor.
- Runtime visual, input, FPS, audio, video, and client-data proof are operator-owned.
- Preserve unrelated dirty work; stage named files only.
- Spec 231 implementation must follow Spec 228 (owned page classes, no new ViewerApp partials)
  and AGENTS.md §9 receipts.

## Handoff

**Current target:** Spec 231 Phase 0 (T001–T003) in a fresh session.

**Then:** phases P1–P5 per [231 tasks.md](../specs/231-editor-archaeology-ui-overhaul/tasks.md).
**Do not claim:** UI acceptance, real-client acceptance, or Spec 224 audit completion.
