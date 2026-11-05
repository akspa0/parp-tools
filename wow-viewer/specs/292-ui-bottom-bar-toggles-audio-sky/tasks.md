# Tasks — Spec 292 (no task checked without a §9.2 receipt)

Receipt: [`evidence/final-receipt.md`](evidence/final-receipt.md)

## Phase 0 — Inventory
- [x] T001 Inventory bottom-bar, Pages, Spawns, Options, Profile controls → [`evidence/inventory.md`](evidence/inventory.md)
- [x] T002 Inventory all audio settings and their current locations → [`evidence/inventory.md`](evidence/inventory.md)
- [x] T003 Identify empty sub-tabs and scroll-reset root cause → [`evidence/inventory.md`](evidence/inventory.md)
- [x] Gate A: operator reviews inventory / proceeds with execution

## Phase 1 — Bottom bar
- [x] T010 `BottomBarService` with 9 toggles + 4 mode switches (receipt in `evidence/final-receipt.md`)
- [x] T011 Remove floating bottom `destination` window and Profile buttons (receipt in `evidence/final-receipt.md`)

## Phase 2 — Dock re-homing
- [x] T020 Viewer/Options → left; Editor/Archaeology/Spawns → right (receipt in `evidence/final-receipt.md`)
- [x] T021 Remove Pages (or re-home content) (receipt in `evidence/final-receipt.md`)

## Phase 3 — Audio
- [x] T030 `AudioPanelService` left-dock panel; remove duplicates (receipt in `evidence/final-receipt.md`)

## Phase 4 — Sky
- [x] T040 `SkyControlsService` (on/off, source, time-of-day+lock, colors, stars/moon/clouds) (receipt in `evidence/final-receipt.md`)

## Phase 5 — Panel defects
- [x] T050 Fix scroll reset in shared section helper (receipt in `evidence/final-receipt.md`)
- [x] T051 Remove/fill empty sub-tabs (receipt in `evidence/final-receipt.md`)

## Phase 6 — Verification
- [x] T060 Update `ui-controls.json` + `UI-REFERENCE.md`; `Verify-UiReference.ps1` (receipt in `evidence/final-receipt.md`)
- [x] T061 Build + tests; operator screenshot proof (AC-1/2/5/6/7) (receipt in `evidence/final-receipt.md`)

## Phase 7 — Operator-directed bottom-toolbar reversion (2026-11-05)
- [x] T070 Correct the toolbar to per-liquid-family toggles (Water/Ocean/Magma/Slime), Doodads/WMOs/Terrain/PM4/Grid/Mute, resident-terrain L0–Ln, one A1–An selector per non-base layer, both opacity sliders, and no Sky/Fog/Texture Masks. Receipt: [`evidence/bottom-toolbar-reversion-2026-11-05.md`](evidence/bottom-toolbar-reversion-2026-11-05.md).
- [x] T071 Persist the selected alpha overlay/channel and the four liquid-family toggles; restore them when scene renderers are available. Receipt: [`evidence/bottom-toolbar-reversion-2026-11-05.md`](evidence/bottom-toolbar-reversion-2026-11-05.md).
- [x] T072 Update `ui-controls.json` and `UI-REFERENCE.md`; pass `Verify-UiReference.ps1`, focused footer tests, and solution build. Operator real-client visual/restart confirmation remains owed. Receipt: [`evidence/bottom-toolbar-reversion-2026-11-05.md`](evidence/bottom-toolbar-reversion-2026-11-05.md).
- [x] T073 Remove the duplicate status-bar Mute control and default audio to muted. Receipt: [`evidence/bottom-toolbar-reversion-2026-11-05.md`](evidence/bottom-toolbar-reversion-2026-11-05.md).
