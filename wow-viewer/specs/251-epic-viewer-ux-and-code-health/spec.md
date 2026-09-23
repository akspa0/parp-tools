# Epic 251 — Viewer UX, Shell & Code Health

**Created**: 2026-09-23 (spec reconciliation) · **Branch**: `v0.6.0-dev` · **Status**: Triage pending

> Primary successor of 15 archived specs (split specs also contribute items; see the ledger). **No new scope** (§9.1). Owner of AGENTS.md §9
> (governance, from 224), §10 (god-class freeze, from 228) and §11 (UI standard, from 227/223).
> Evidence: [reconciliation ledger](../archived/reconciliation-2026-09-23/README.md), audits
> [E](../archived/reconciliation-2026-09-23/audit/batch-E.md) · [G2](../archived/reconciliation-2026-09-23/audit/batch-G2.md) ·
> [F1](../archived/reconciliation-2026-09-23/audit/batch-F1.md) · [D1](../archived/reconciliation-2026-09-23/audit/batch-D1.md).

## Goal

An approachable viewer — one authoritative home per data surface, standard sidebar widgets, sane
navigation — built on code that is small enough to change safely.

## Delivered baseline (verified in code 2026-09-23 — do not re-plan)

| Capability | Source |
|---|---|
| Workbench navigator pattern; Editor/Archaeology page IA (Phases 0–4, 6 batch 1, 7) | 231 (supersedes 069/073 tab-bar designs) |
| Viewer reachability/stabilization slices (US1/US5/US7) | 110 |
| Sidebar resize hotfix | 072 |
| Inventory v1/v2 baselines (223 surface inventory; 227 T001/T002) | 223, 227 |
| Governance rules codified and in force (receipts, scope freeze, write containment, spec-sync, monthly cleanup) | 224 → AGENTS.md §9 |
| Spatial-UI scaffolding (`CameraHudRig`, `Enabled=false`, no consumer) | 212 |

## Measured code-health baseline (2026-09-23)

`WorldScene.cs` **17,153** lines (single file) and `ViewerApp.cs` **16,746** lines — each about 8.5× the
2,000-line budget. No extraction has happened yet.

## Backlog (spec-stated residue — each item awaits operator triage in [TRIAGE.md](../TRIAGE.md))

### A. Code health & governance

| ID | Item | Source |
|---|---|---|
| U-01 | Behaviour-preserving extraction: pure Core selection service (+tests), then `WorldScene` extraction, then the next `ViewerApp` candidate | 228 T001–T015 |
| U-02 | Governance Gate 1 operator sign-off; resync 224 ledger; monthly cadence | 224 Gate 1, T201/T202, T301 |
| U-03 | Automation surface for agents/tests: MCP tooling harness + external MCP automation (one design, not two) | 213 FR-001–020; 178 |

### B. UI consolidation (223 → 227 → 231 lineage, merged)

| ID | Item | Source |
|---|---|---|
| U-10 | Sidebar standard: every surface on `SharedUiWidgets`, ≤3-interaction navigation, Archaeology styling fix | 227 US2 |
| U-11 | Deduplicate weak-signal amplifiers, minimaps and Inspector repetition (operator-named) | 227 US3 |
| U-12 | Approachability pass | 227 US4 |
| U-13 | Unified Inspector by object type | 223 T101–T107 |
| U-14 | Retire floating windows; merge duplicates | 223 T201–T203 |
| U-15 | Quick-tab single source of truth for Fog Defaults (operator pain point) | 223 T501/T502 |
| U-16 | Editor/Archaeology merge + Cartography placement; Editor re-split | 223 T301/T302, T401 |
| U-17 | Converters page full scope; removal pass + page convergence on the Data I/O/Quick pattern | 231 T041, T061–T064 |
| U-18 | Hovered-WMO doodad-set combo regression (fix, don't re-plan) | 231 T080 |
| U-19 | Single UI ownership of renderer/lighting controls | 152 Ph 7 |
| U-20 | Tools menu inventory + dependency diagnostics | 110 US3/FR-010 |
| U-21 | Toolbar width scoping (T005 was checked but never took effect) | 072 T005 |

### C. New surfaces

| ID | Item | Source |
|---|---|---|
| U-30 | WoW-style shell + contextual keybind profiles (sequenced after U-10/U-01) | 229 US1–US3 |
| U-31 | Overhead orthographic world view with the three existing grid overlays | 225 Phase 1 |
| U-32 | Selection-silhouette outlines (current selection is a bounding box) | 212 US6/FR-024–029 |
| U-33 | Simple interactive viewer surface | 151 US2/FR-006–007 |
| U-34 | Workspace profiles / editor mode + MK Dataset purge | 197 T101–T108 |

### D. Parked

| ID | Item | Source |
|---|---|---|
| U-90 | Panels-as-3D-objects spatial UI shell, museum profile | 212 (rest) |

## Operator verification owed on shipped code

223 T609/T610 retest (fog, WMO-only global WMO, playback/capture, ffmpeg, spatial UI) with acceptance
recorded separately from build evidence; 231 T050–T052 navigation smoke + inventory v3; v0.5.2.3
checks (slider order, overlay balance, hover occlusion, About credits); alpha2's new export menu and
sidebar buttons have never been clicked.
