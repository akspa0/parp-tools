# Spec Status Router

**Reconciled 2026-09-23** (branch `v0.6.0-dev`). Every earlier spec was audited against the code and
archived; open work now lives in **7 epics** (248–254). Receipt and per-spec ledger:
[archived/reconciliation-2026-09-23/README.md](archived/reconciliation-2026-09-23/README.md).
Previous router: [archived/status-history/2026-09-23-status-pre-reconciliation.md](archived/status-history/2026-09-23-status-pre-reconciliation.md).

## Current step — operator triage

**Operator P1 (2026-09-23):** R-10 modern-data lighting performance (Epic 249) and U-01 god-class
decomposition (Epic 251). Both have a proposed approach awaiting approval; R-10 lands before U-01 step E2.


Nothing is scheduled. The operator marks each backlog item **Want / Drop / Later** in
[TRIAGE.md](TRIAGE.md); only then does each epic get phased `tasks.md` and a chosen implementation
approach. TRIAGE.md opens with the measured "we thought we had it" gaps and a quick-win shortlist.

## Active epics

| Epic | Scope | Backlog | Status |
|---|---|---|---|
| [248 Formats, Readers, Writers & Conversion](248-epic-formats-and-conversion/spec.md) | Legacy + modern (CASC/FDID/DAT) readers and writers; modern→legacy conversion; converter harness | F-01…F-31, F-9x | Triage pending |
| [249 Renderer Performance, Lighting & Correctness](249-epic-renderer-performance-and-correctness/spec.md) | Benchmarks and receipts, batching/instancing, WMO admission, streaming, lighting, correctness | R-01…R-38, R-9x | **R-10 P1** (approach pending approval); rest pending |
| [250 Map Reconstruction, Composition & Editor Platform](250-epic-reconstruction-and-editor-platform/spec.md) | Map save, undo/session, cartography composition, generator, editor plugins | E-01…E-33 | Triage pending |
| [251 Viewer UX, Shell & Code Health](251-epic-viewer-ux-and-code-health/spec.md) | UI consolidation, god-class extraction, governance, automation surface | U-01…U-34, U-90 | **U-01 P1** (approach pending approval); rest pending |
| [252 World Simulation, Audio & Interaction](252-epic-world-simulation-and-audio/spec.md) | Audio, area context, camera, picking, 5.0.1 physics/weather, server data | W-01…W-25 | Triage pending |
| [253 PM4/PD4 Navmesh](253-epic-pm4-navmesh-research/spec.md) | Decode, field naming, matching, placement restore, generation | P-01…P-11 | Triage pending |
| [254 Datasets, Client Datastore & Terrain ML](254-epic-datasets-and-terrain-ml/spec.md) | Multi-build datastore, v50/v60 models, archaeology tooling | D-01…D-30, D-9x | Triage pending |

## Operator verification owed on shipped code

Each epic's `tasks.md` carries a **verification sweep** (V-tasks): proof-only checks on code that
already shipped, including everything v0.6.0-alpha2 shipped unverified (M2 texture-wrap fix,
`LkAdtWriter` chunk fixes, DAT→LK export client load, DAT Cartography layer on screen, new export
buttons).

## Release line

`eng/Version.props` = `0.6.0` / `0.6.0-alpha2` (released 2026-09-21). The v0.6 theme — modern data
access + experimental terrain formats — is now items F-01…F-12 (Epic 248) and R-01, R-02, R-10
(Epic 249). Release scope is re-pinned after triage.

## Status rules

- An epic backlog item is **not** scheduled until marked Want in TRIAGE.md.
- `[x]` requires a §9.2 receipt (files, commands + exit status, criterion→evidence with real output).
  Compilation and unit tests never prove runtime, visual, FPS, audio or client behaviour.
- Archived specs are history, not authority. An archived design document is authority only for the
  epic item whose `plan.md` adopts it by link.
- New work: add an item to the owning epic (operator-approved wording, §9.1) or, for a genuinely new
  area, a new spec number ≥ 255 registered here. Follow AGENTS.md §9–11.
- Monthly `speckit-cleanup` audits the epics' checked tasks; next due 2026-10-01.
