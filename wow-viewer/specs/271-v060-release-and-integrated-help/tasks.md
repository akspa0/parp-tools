# Spec 271 — Tasks

Receipts live in `evidence/`. A box is checked only with a §9.2 receipt.

## Phase A — v0.6.0 release

- [x] **A1** Version bump to `0.6.0` in `eng/Version.props`. (receipt: `evidence/receipt-phase-a.md`; title-bar witness is operator-owned)
- [x] **A2** Release notes, CHANGELOG, READMEs, USERGUIDE rewritten (BYO data, support matrix, Archaeology phase maps, restored development map, experimental-ML warning).
- [ ] **A3** `main` merged, `v0.6.0` tagged and pushed; release workflow green.
- [ ] **A4** `v0.6.1-dev` created and pushed; memory bank + STATUS updated.

## Phase B — Integrated help & image guide (v0.6.1-dev)

- [ ] **B1** `HelpGuideService` + `guide.json` manifest + Help menu / per-panel entry points.
- [ ] **B2** Shot list for every workbench destination and sidebar (agent) → operator captures images → embed in USERGUIDE and in-app help.
- [ ] **B3** Archaeology tab documentation: phase-map auto-detection from client DBC/DB2, and "any map on any map" loading, with walkthrough images.
- [ ] **B4** UI inventory row registered (Spec 223 FR-9).

## Phase C — Operator-directed corrective release v0.6.0.1 (2026-10-14)

- [ ] **C1** Add all executable projects from `wow-viewer/tools/` to the solution and a separate CLI-tools
  archive per platform; compile all nine apphosts into one shared output directory with common runtime/data
  files deduplicated, and verify each tool's apphost/DLL/deps/runtimeconfig. Receipt required for all eight assets (AC-C1).
- [ ] **C2** Apply per-instance terrain MCCV/MCSH tint when rendering ground-effect doodads; focused
  unit tests cover the computed tint. Operator visual smoke remains owed (AC-C3).
- [ ] **C3** Prepare version `0.6.0.1`, release notes, tag/push, and successful release workflow run
  with eight archives (AC-C2).
