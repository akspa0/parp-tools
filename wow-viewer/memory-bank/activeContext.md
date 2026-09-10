# Active Context — wow-viewer

Last updated: 2026-09-10

## Fresh-chat route

1. [Spec status](../specs/STATUS.md) — select exactly one active owner.
2. This compact handoff.
3. That owner's `spec.md`, `plan.md`, and `tasks.md`, and linked receipt only.

The [documentation router](../docs/README.md), [spec routing registry](../specs/registry.md), and
[archives](archive/README.md) are on-demand context, never default reading.

## New lane — Spec 234 Map Save & New Map Creator (spec authored 2026-09-10)

- **Spec** at [234-map-save-new-map/spec.md](../specs/234-map-save-new-map/spec.md) (operator
  directive 2026-09-09): save merged/composed maps to **Alpha 0.5.3 WDT** and **LK v18 ADT** from
  BOTH the Archaeology cartography surface and the Editor's Data I/O page (one shared save
  pipeline, no forks), plus a **New Map creator** in the Editor tab composing the Spec 192
  generator. **Multi-map support is explicitly out of scope** (operator deferred it as a bigger
  feature; the spec's save pipeline operates on named map identities so it is not foreclosed).
  Supersedes Spec 230 US2/US3 (dated amendment added to 230; 230 retains US1 Rosetta placement).
  Registered in [STATUS.md](../specs/STATUS.md) row 9 and Epic 2. Next: speckit-plan.

## Prior lane — Spec 232 Cartography: placement coordinates, Z transform, minimap, tile-rigid cells, WDL edge-snap

- **T056–T059 + T064 + T066 landed 2026-09-09** (operator directives): donor-tile picker now reads
  ADT-name order (xx = column, yy = row — placements previously landed on the diagonal mirror;
  T056); per-layer `ZOffset`/`ZScale` world-Z transform on terrain, liquids, and placements with
  layer-card inputs + project persistence (T057); minimap surfaces got per-surface pointer states
  so the 3-click teleport works (T058); placed-only layers are draggable on the minimap, moving
  each placement's target (T059); **cell fine-tune is LAYER-RIGID** — each target tile collects
  its own donor tile's retained content plus the 3×3-neighborhood spill, so nothing drops between
  tiles (T066); **magnetic WDL edge-snap** — per-layer `EdgeBlendWdl` strength blends
  footprint-boundary tile edges toward the base map's WDL macro lattice (PhaseEdgeBlender in
  Core.Runtime), applied in both adapters after the Z transform, with host wiring from the shared
  stratigraphy WDL parse (T064). Maps tests 181/181; solution build 0 errors. Receipts:
  [t056-t059](../specs/232-cartography-composition-project/evidence/t056-t059-placement-z-minimap-receipt.md),
  [t066](../specs/232-cartography-composition-project/evidence/t066-tile-rigid-cell-shift-receipt.md),
  [t064](../specs/232-cartography-composition-project/evidence/t064-wdl-edge-snap-receipt.md).
  **All need operator interactive witnesses.** Remaining operator directive: T065 (chunk-level
  MCAL/MCLY off-by-one + height smoothness re-audit after T056).

## Prior lane — Spec 233 Renderer Marketing Capture Automation: P1 operator tour gate

- **Spec + plan + tasks** at [233-marketing-capture-automation/](../specs/233-marketing-capture-automation/spec.md): a built-in, versioned camera-path overview recipe, clean-scene timed callouts, direct renderer-frame capture, future receipt and safe authoring handoff contracts.
- **P1 source implementation 2026-09-08**: **Feature Tour + Video** now starts from the existing Camera Path pane, carries the validated tour through the existing warmup gate and ffmpeg framebuffer route, hides ordinary chrome during recording, draws only timed callouts at the full-frame capture tap, and restores the prior chrome state. The Core Runtime model rejects invalid recipes and output traversal, while the handoff model emits only relative artifact references. Focused marketing tests: 12/12; Debug solution build: 0 errors. **Not runtime/visual proof**: T015 needs the operator's `FlybyUndead` warm-and-record witness. Receipts: [design](../specs/233-marketing-capture-automation/evidence/t001-design-receipt.md), [foundation](../specs/233-marketing-capture-automation/evidence/t003-t008-foundation-receipt.md), [P1 source](../specs/233-marketing-capture-automation/evidence/t009-t014-us1-source-receipt.md).
- **Open next**: Phase 4 receipt serialization and actual named-control registry; a Comfy/MCP transport remains blocked until a callable contract is selected. Never claim a generated marketing video, benchmark result, Comfy authoring action, or README showcase asset before operator evidence.

## Prior / parallel lane — Spec 231 UI Overhaul: Phase 6 tools-curation pass (operator amendment)

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
- **Phase 7 landed + fix round 2026-09-07** (12cb1b82 + 4bd50b4d): layer rotation/mirror wired
  end-to-end — panel controls, ResolveTileSource-driven donor lookup in BOTH adapters,
  TileContentTransform content rotation, placement pose rotation, composed footprints/minimap.
  Fix round: TileExists streaming admission routed through the policy (rotated layers previously
  rendered nothing), 64x64 grid confinement enforced, rotation origin auto-centers on first
  transform, minimap double-offset removed. **T074 visual gate PASSED** (operator screenshot:
  DeadminesInstance rotated 90° CW on Azeroth). 157/157 Maps tests.
  Receipt: [evidence/t070-t074-phase7-layer-rotation-receipt.md](../specs/231-editor-archaeology-ui-overhaul/evidence/t070-t074-phase7-layer-rotation-receipt.md).
- **Spec 232 T015e 2026-09-09 (MCAL alpha repair — fixes the operator's "MCLY layers on overlapped
  maps broke" report)**: the T015c full-tile route sliced per-chunk MCAL alpha from the reader's
  256×256 *downsampled* pack at a 64-px-per-chunk stride — chunks past (3,3) decoded silent zero
  alpha, so transformed overlapped tiles rendered flat single-texture patches. The reader now also
  carries the full-resolution 1024×1024 pack (`AlphaTileData.McalAlphaPackFull`), `ToTileLoadResult`
  slices from it (256-pack upsample fallback), and `RotateQuarterTurn` moves it with the same index
  map. Focused Maps: 174/174; solution build 0 errors. **Visual MCLY witness still operator-owned**
  (folds into T015d). Receipt: [t015e MCAL alpha repair](../specs/232-cartography-composition-project/evidence/t015e-mcal-alpha-repair-receipt.md).
- **Spec 232 T015a–c 2026-09-08**: the Alpha adapter now reads the full donor lattice, applies
  `RotateQuarterTurn`, then slices/re-homes at the target in both ordinary and cell-shifted
  transformed paths; it never falls back to the known seam-producing per-MCNK rotation.
  33/33 focused tests and a solution build (0 errors) pass. **Still not visually accepted**:
  T015d needs the operator's rotated DeadminesInstance seam screenshot and cell-nudge witness.
  Receipts: [core lattice](../specs/232-cartography-composition-project/evidence/t015a-t015b-core-lattice-receipt.md),
  [adapter route](../specs/232-cartography-composition-project/evidence/t015c-alpha-adapter-full-tile-receipt.md).
- **Spec 232 T050 2026-09-08**: picker-created layers now compose only their explicit
  donor-to-target placement by default. The persisted `UsePlacedTilesOnly` mode prevents offset
  fallthrough at all unplaced targets; the Layers-panel toggle restores offset mode, and minimap
  footprints show only placed targets. Focused maps: 37/37; full solution build: 0 errors.
  Receipt: [t050 placed tiles only](../specs/232-cartography-composition-project/evidence/t050-placed-tiles-only-receipt.md).
- **Spec 232 T051 implementation 2026-09-08**: explicit placement locks are persisted and
  enforced in both terrain adapters and minimap composition; a per-placement panel toggle and
  `L` minimap badge are built. The focused contract suite is 39/39 and the solution build has 0
  errors. **Still unchecked** pending an operator capture proving the badge and that a later layer
  cannot override the owner. Receipt:
  [t051 per-tile locks](../specs/232-cartography-composition-project/evidence/t051-per-tile-lock-implementation-receipt.md).
- **Spec 232 T053 implementation 2026-09-08**: the WL click-inspector failure was not caused by
  minimap footprint drag/hit-testing. A terrain-occlusion guard was clearing the WL
  source-data hover after a composed layer supplied terrain in front of its bounds, so the viewport
  click path had no WL candidate to inspect. WL is now exempt from that physical-object-only
  occlusion guard; the full solution build has 0 errors. **Still unchecked** pending an operator
  with/without-layer-stack WL inspector witness. Receipt:
  [t053 WL fall-through](../specs/232-cartography-composition-project/evidence/t053-wl-inspector-fallthrough-receipt.md).
- **Spec 232 T054 implementation 2026-09-08**: a no-page entry to Archaeology now defaults to
  Cartography page 5, whose initial tab is Map Layers. Explicit Range/UniqueId routes and the
  per-tab remembered choice still override that default. The solution build has 0 errors.
  **Still unchecked** pending a default-entry/remembered-page UI witness. Receipt:
  [t054 Map Layers default](../specs/232-cartography-composition-project/evidence/t054-archaeology-map-layers-default-receipt.md).
- **Phase 6 batch 1 landed 2026-09-07** (fbf600a1): operator amendment recorded verbatim in the
  spec — Data I/O + Quick are the UI model (SharedUiWidgets, no one-offs); non-working/never-wired
  tools get removed; ViewerApp size reduction. Wiring audit removed **27 dead draw methods**
  (entire legacy Model/Experimental/Scene/Terrain/Selection sub-tab family, old minimap, dead
  sidebar summaries) — Sidebars 6,013→5,648, Pm4Utilities 4,368→4,178, ViewerApp 16,750→16,556.
  Receipt: [evidence/p6-wiring-audit.md](../specs/231-editor-archaeology-ui-overhaul/evidence/p6-wiring-audit.md).
- **Open**: Spec 232 (fresh session); Phase 6 batch 2 (T061–T064: weak-signal amplifier removal,
  terrain task blocks, Quick-panel convergence); 231 T041 (Spec 221 harness surfaces); Phase 5
  (operator navigation smoke, inventory v3 final).
- **Commits**: ee223e78 (P0+P1) → 86394fb6 (P2) → 18a49732 (P3+P4) → fbf600a1 (P6 batch 1) →
  c4abc15e (doodad-set combo fix) → 12cb1b82 (P7 wiring) → 4bd50b4d (P7 fix round).
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
- Video-capture release hardening (2026-09-08): Capture Automation now resolves a licensed
  `ffmpeg.exe` beside the viewer before configured/PATH fallback and probes `libx264`; publish
  copies an operator-supplied binary from `src/viewer/WoWViewer/Capture/ffmpeg/win-x64/` when
  present. The binary and its licence/notices are operator-owned. Do not ship until the published
  viewer passes **Verify ffmpeg**, direct with-UI/no-UI recording + playback, and a camera-path
  **Play + Video** witness. Receipt:
  [t609-video-capture-release-hardening-2026-09-08.md](../specs/223-ui-consolidation-audit/evidence/t609-video-capture-release-hardening-2026-09-08.md).
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

**Current target:** Spec 233 T015 operator tour witness — import/load `FlybyUndead.mdx` or `.m2`, use **Warm Path**, then **Feature Tour + Video** and record the video/UI-timing observation. Capture/receipt authoring is not yet implemented, so no benchmark or Comfy handoff is claimed. Then proceed with Spec 233 Phase 4 receipt work.

**Parallel target:** Spec 232 operator witnesses — (1) show an `L` minimap badge and prove a later
layer cannot override the earlier locked placement (T051); (2) click the same WL body with and
without a placed layer and capture its inspector identity/source path (T053); (3) enter
Archaeology with no remembered page and confirm Cartography > Layers, then preserve an explicit
UniqueId page selection across a return (T054). Then continue with T055, UniqueId-era color-coding.

**Separate operator gate:** T015d — obtain the seam-free rotated DeadminesInstance screenshot
(coastline/roadway continuous) and demonstrate a cell nudge still moves the rotated layer.
**Do not claim:** T015 visual acceptance, real-client acceptance, or Spec 224 audit completion.
