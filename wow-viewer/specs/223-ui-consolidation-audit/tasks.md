# Tasks: Viewer UI Consolidation Audit (Spec 223)

**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md) · **Inventory**: [surface-inventory.md](surface-inventory.md)

## Phase 1 — Unified Object Inspector (US2a, US3)

- [x] 223-T101: Inspector host + HUD-ready content model. New right-sidebar destination
      "Inspector" (replaces legacy `##InspectorTabs`); content model (sections of rows, actions,
      sub-blocks) in `WowViewer.Core.Runtime.World.Inspection` so it can render as the 212 HUD
      object later. Consumes the existing selection pipeline. Verify: build green; selecting each
      object type routes to the right section.
- [x] 223-T102: ADT section — absorb MCNK Explorer, ADT Chunk Investigation,
      TerrainChunkHoverOverlay, Terrain Lab MCNK page. Capability checklist from inventory §5
      before removal.
- [x] 223-T103: WMO section — absorb WMO details; **doodad-set switching** (MODS list, active
      set, re-render without map reload, selection/camera preserved). Core-side set-resolution
      model + focused tests.
- [x] 223-T104: MDX/M2 section — absorb Model Info + Inspect Animations/Context.
- [x] 223-T105: PM4 section — absorb Selected PM4 / Match details / Scene measurements.
- [x] 223-T106: WL\* liquid section — absorb WL Liquid Investigation.
- [x] 223-T107: Hover overlays become compact summaries deep-linking into the Inspector.
- [x] **Gate A**: capability walkthrough — every absorbed surface's features present; old
      surfaces removed in the same change (FR-2/FR-3). Operator verifies.

## Phase 2 — Duplicate retirement (US2)

- [x] 223-T201: Retire floating windows absorbed by the Inspector; Log/Perf/RenderQuality
      floaters → Utilities pages (single implementation each).
- [x] 223-T202: Merge ×2 implementations (World Overview, World Maps, Chunk Clipboard) into one
      shared draw method used by both tab UI and legacy UI.
- [ ] 223-T203: Retire Terrain Workbench floating window after the 222-T111 parity checklist.

## Phase 3 — Editor + Archaeology merge (US4 step 1)

- [x] 223-T301: Editor content (converters, ML dataset, imports, editor task nav) into
      Archaeology as an "Editor" page group; top-bar Editor mode routes to Archaeology.
- [x] 223-T302: Cartography under Archaeology (222-T108/T109/T109a/T110): layer panel in right
      sidebar, synthesized minimap tab, save-merged-output; delete left-sidebar panel.
- [x] **Gate B**: feature-parity walkthrough against the inventory — nothing lost (SC-4).

## Phase 4 — True-editor split (US4 step 2)

- [x] 223-T401: Dedicated Editor profile with authoring/writing features; Archaeology keeps
      analysis/inspection; old separate tabs removed.

## Phase 5 — Quick per profile (US5)

- [x] 223-T501: Shared Quick mirror component; per-profile top controls from the inventory;
      **Fog End first in every profile**.
- [x] 223-T502: Settings>Fog Defaults is the single implementation; Quick mirrors it (FR-6).

## Phase 6 — Transport Fixes, MCNK Deep-Link, Shared UI Library & Documentation (US6)

- [x] 223-T601: Fix Taxi Riding and Camera Path Riding:
      - Unconditionally simulate active taxi ride route poses in `WorldScene.UpdateTaxiActorInstances()` during rides regardless of UI selection filter.
      - Relax strict map/build string matching and pending preloads in `UpdateCameraPathPlayback()`.
- [x] 223-T602: Fix Terrain/ADT Inspector blanking bug (`_renderer != null` trap) and add terrain chunk click-to-pin (`_selectedTerrainChunk`).
- [x] 223-T603: Absorb 'Scene' tab (`Placements` and `LOD & Budget`) into the Inspector tab; remove top-level 'Scene' tab.
- [x] 223-T604: Disperse 'Utilities' tab into Quick tab, Inspector, View menu, and Tools menu; remove top-level 'Utilities' tab.
- [x] 223-T605: Deep-link and integrate full MCNK Flag Overlay & MCNK Chunk Flags directly into the Inspector's ADT section; eliminate duplicate scattered MCNK explorer panels.
- [x] 223-T606: Build central shared UI library (`WoWViewer.UI.SharedUiWidgets`) standardizing section headers, `[?]` help pop-up ready widgets, action groups, and compact status readouts.
- [x] 223-T607: Universal Quick Tab & Profile State Preservation — retain active `Quick` tab and all viewer settings when switching top-level profiles (`Viewer`, `Editor`, `Archaeology`).
- [x] 223-T608: Update end-user documentation in `docs/WoWViewer/USERGUIDE.md` detailing the reorganized 4-tab workbench, 3D Object Library, Imports/Exports dashboard, MCNK flags in Inspector, and transport features.

## Phase 6 acceptance remediation — reported failures

- [ ] 223-T609: Validate the repaired fog, WMO-only, and Playback & Capture paths in the running
      viewer. Source changes centralize every fog editor on the `WorldScene` override, route WMO-only
      global MODF instances through the external admission collector, and expose Capture Automation /
      Camera Path plus **Apply playback to next capture** inside Archaeology > Playback & Capture.
      Verify: a live fog drag holds under lighting recomposition; a known WMO-only map draws its global
      WMO; the displayed capture controls start, stop, and produce a valid ffmpeg output.
- [ ] 223-T610: Record the operator's UI acceptance result separately from source/build evidence.
      The current decorative camera rig is not interactive spatial UI; do not close this task or
      represent the workbench as a 3D HUD until Spec 212's interactive phases and gates pass.

## Standing verification

- [x] **Phase 6 source gate**: solution Debug build, focused transport/inspection checks, affected
      solution suites and diff review. The acceptance-remediation source check also passed an isolated
      viewer Debug build and 29 focused WDT/WMO visibility tests. Record exact outcomes in
      `evidence/phase6-validation.md`; this is not runtime proof.
- [ ] **Phase 6 operator gate**: taxi ride continues with route filters/visibility changed; camera
      path starts while streaming and capture still waits for readiness; actual map changes stop
      transport; terrain click pins the intended chunk and source changes clear it; Context,
      Placements and LOD work; MCNK flags/diagonal highlights render; four tabs and Quick settings
      survive profile switches at normal and compact widths; every moved utility remains reachable.
      Additionally, Fog Start/Fog End must not rubberband, a known WMO-only map must draw its global
      WMO, and Archaeology > Playback & Capture must expose and complete video/path/apply workflows.

- Build per phase: `dotnet build wow-viewer/WowViewer.slnx -c Debug`
- Focused tests for any Core inspection model added in T101/T103.
- Operator gates A and B; SC-1..SC-6 from the spec.
