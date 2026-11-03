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
