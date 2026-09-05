# Tasks: Viewer UI Consolidation Audit (Spec 223)

**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md) · **Inventory**: [surface-inventory.md](surface-inventory.md)

## Phase 1 — Unified Object Inspector (US2a, US3)

- [ ] 223-T101: Inspector host + HUD-ready content model. New right-sidebar destination
      "Inspector" (replaces legacy `##InspectorTabs`); content model (sections of rows, actions,
      sub-blocks) in `WowViewer.Core.Runtime.World.Inspection` so it can render as the 212 HUD
      object later. Consumes the existing selection pipeline. Verify: build green; selecting each
      object type routes to the right section.
- [ ] 223-T102: ADT section — absorb MCNK Explorer, ADT Chunk Investigation,
      TerrainChunkHoverOverlay, Terrain Lab MCNK page. Capability checklist from inventory §5
      before removal.
- [ ] 223-T103: WMO section — absorb WMO details; **doodad-set switching** (MODS list, active
      set, re-render without map reload, selection/camera preserved). Core-side set-resolution
      model + focused tests.
- [ ] 223-T104: MDX/M2 section — absorb Model Info + Inspect Animations/Context.
- [ ] 223-T105: PM4 section — absorb Selected PM4 / Match details / Scene measurements.
- [ ] 223-T106: WL\* liquid section — absorb WL Liquid Investigation.
- [ ] 223-T107: Hover overlays become compact summaries deep-linking into the Inspector.
- [ ] **Gate A**: capability walkthrough — every absorbed surface's features present; old
      surfaces removed in the same change (FR-2/FR-3). Operator verifies.

## Phase 2 — Duplicate retirement (US2)

- [ ] 223-T201: Retire floating windows absorbed by the Inspector; Log/Perf/RenderQuality
      floaters → Utilities pages (single implementation each).
- [ ] 223-T202: Merge ×2 implementations (World Overview, World Maps, Chunk Clipboard) into one
      shared draw method used by both tab UI and legacy UI.
- [ ] 223-T203: Retire Terrain Workbench floating window after the 222-T111 parity checklist.

## Phase 3 — Editor + Archaeology merge (US4 step 1)

- [ ] 223-T301: Editor content (converters, ML dataset, imports, editor task nav) into
      Archaeology as an "Editor" page group; top-bar Editor mode routes to Archaeology.
- [ ] 223-T302: Cartography under Archaeology (222-T108/T109/T109a/T110): layer panel in right
      sidebar, synthesized minimap tab, save-merged-output; delete left-sidebar panel.
- [ ] **Gate B**: feature-parity walkthrough against the inventory — nothing lost (SC-4).

## Phase 4 — True-editor split (US4 step 2)

- [ ] 223-T401: Dedicated Editor profile with authoring/writing features; Archaeology keeps
      analysis/inspection; old separate tabs removed.

## Phase 5 — Quick per profile (US5)

- [ ] 223-T501: Shared Quick mirror component; per-profile top controls from the inventory;
      **Fog End first in every profile**.
- [ ] 223-T502: Settings>Fog Defaults is the single implementation; Quick mirrors it (FR-6).

## Standing verification

- Build per phase: `dotnet build wow-viewer/WowViewer.slnx -c Debug`
- Focused tests for any Core inspection model added in T101/T103.
- Operator gates A and B; SC-1..SC-6 from the spec.