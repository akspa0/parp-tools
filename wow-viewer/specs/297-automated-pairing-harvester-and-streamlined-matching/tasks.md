# Tasks 297 — Automated Known-Good Pairing Harvester & Streamlined Matching Workbench

Receipt: `evidence/receipt-spec297.md`

## Phase 1 — Core Harvester Engine (FR-001, FR-005)

- [ ] T001 [Core] Add `HarvestPairingsFromPlacedObjects` to `Pm4DoodadDuplicationService` with spatial proximity, bounding box overlap, and footprint correlation.
- [ ] T002 [Core] Support deduplication and updating of `_taughtPairings` with disk persistence (`<map>.pairings.json` and `<map>.manifest.json`).

## Phase 2 — Streamlined UI Cockpit (FR-002, FR-003, FR-004)

- [ ] T003 [UI] Redesign `Pm4WorkbenchSidebarService` to present a unified Cockpit layout: Status Summary, Quick Action Bar, Active Selection Card, and Clean Learned Rules List.
- [ ] T004 [UI] Connect `[ Harvest Pairings from Placed Models ]` action button with live progress and status readout.
- [ ] T005 [UI] Connect `[ Apply Learned Rules to All Tiles ]` action button with live count of populated hulls.
- [ ] T006 [UI] Move low-level binary chunk forensics (`MSHD`, `MSLK`, raw bytes, regional peer lists) into an expandable `Advanced Forensics & Raw Chunks` drawer, collapsed by default.

## Phase 3 — Verification & Receipts (SC-001, SC-002, SC-003, SC-004)

- [ ] T007 [Tests] Add unit tests for automated pairing harvesting in `Pm4DoodadDuplicationTests.cs`.
- [ ] T008 [Verification] Run full solution build (`dotnet build WowViewer.slnx -c Debug`) and verify 0 errors.
- [ ] T009 [Verification] Run unit tests (`dotnet test`) and verify 100% pass.
- [ ] T010 [Verification] Run `Verify-UiReference.ps1` and verify 0 errors, 0 warnings.
- [ ] T011 [Receipt] Record verification commands, exit status, and criterion-to-evidence matrix in `evidence/receipt-spec297.md`.
