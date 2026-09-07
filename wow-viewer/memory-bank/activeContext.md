# Active Context — wow-viewer

Last updated: 2026-09-07

## Fresh-chat route

1. [Spec status](../specs/STATUS.md) — select exactly one active owner.
2. This compact handoff.
3. That owner's `spec.md`, `plan.md`, and `tasks.md`, and linked receipt only.

The [documentation router](../docs/README.md), [spec routing registry](../specs/registry.md), and
[archives](archive/README.md) are on-demand context, never default reading.

## Current lane — Spec 231 Editor/Archaeology UI Overhaul (implementation fresh session)

- **Spec + plan + tasks are complete** at [231-editor-archaeology-ui-overhaul/](../specs/231-editor-archaeology-ui-overhaul/spec.md):
  4-page Editor IA (Placement & Objects / Terrain Tools / Data I/O / Converters), Archaeology
  de-hosting, dedupe moves D1–D6, Spec 228 extraction pattern, phased gates P0–P5.
- **Implementation directive**: operator runs it in a fresh chat from `tasks.md` T001 onward.
  Start at Phase 0 (inventory v3 baseline; folds the Spec 227 Editor-tab item via dated
  supersession note).

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
