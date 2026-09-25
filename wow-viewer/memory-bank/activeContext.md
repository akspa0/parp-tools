# Active Context — wow-viewer

Last updated: 2026-09-25 · Branch: `v0.6.0-dev`

## Fresh-chat route

1. [Spec status](../specs/STATUS.md) — 7 active epics (248–254), all **triage pending**.
2. This compact handoff.
3. [TRIAGE.md](../specs/TRIAGE.md) — the operator's Want / Drop / Later decisions.
4. Only then one epic's `spec.md` → `plan.md` → `tasks.md`, and only the archived design documents
   its `plan.md` adopts for the selected item.

[Progress ledger](progress.md), [memory archive](archive/README.md) and `specs/archived/` are
on-demand history, never default reading.

## Current lanes — operator P1s (both marked Want 2026-09-23)

| Lane | Code | Owed (operator) | Next agent step |
|---|---|---|---|
| **R-10** modern-data lighting perf (Epic 249) | T002, T004–T006 landed 2026-09-23 | T003 baseline + T007 after-capture on `wow_classic_beta` 1.60.1 `Azeroth`; T008 decides R-10e | none until captures exist |
| **U-01** god-class decomposition (Epic 251) | **E1 landed 2026-09-25**: PM4 overlay → `Terrain/Pm4/` (`Pm4OverlayScene` + helpers); `WorldScene.cs` 17,175 → 8,326 | U01-T003 E1 smoke (overlay, colours, selection, OBJ export) | E3 `ViewerApp` menu bar + converter dialogs (independent of R-10). E2 `Render()` split waits for R-10's after-capture |

E1 receipt: [u01-e1-pm4-extraction-2026-09-25.md](../specs/251-epic-viewer-ux-and-code-health/evidence/u01-e1-pm4-extraction-2026-09-25.md).
PM4 overlay code now lives in `Terrain/Pm4/`; callers reach it as `WorldScene.Pm4Overlay.X`. The PM4
draw blocks inside `WorldScene.Render()` stay there until E2.

All other epic items remain untriaged in [TRIAGE.md](../specs/TRIAGE.md); nothing else is scheduled.
Receipt for the 2026-09-23 reconciliation:
[reconciliation README](../specs/archived/reconciliation-2026-09-23/README.md).

**Linux build note (agent sessions):** the viewer builds and tests on Linux with
`-p:EnableWindowsTargeting=true` after `git submodule update --init` of the six `wow-viewer/libs/*`
submodules. 26 tests fail on `HEAD` there for environmental reasons (missing `test_data/`,
Windows-path expectations) — compare against that set, not zero. One Curation test rewrites
`data-harvester/tests/fixtures/spec122_curation_manifest/.../curation_run.json` with the local path;
revert it before staging.

## Release state

`eng/Version.props` = `0.6.0` / `InformationalVersion 0.6.0-alpha2`, released 2026-09-21 (tag
`v0.6.0-alpha2`, all four self-contained binaries). Notes: `docs/releases/v0.6.0-alpha2.md`.

**Shipped unverified in alpha2** (operator verification, tracked as V-tasks in the epics): the M2
texture-wrap fix (every M2 era); `LkAdtWriter` chunk-completeness fixes (17 call sites); no exported
map loaded in a client or Noggit; no DAT Cartography layer composed on screen; new export menu and
sidebar buttons never clicked.

## Measured gaps worth knowing before any work (details: TRIAGE §1)

- No map save pipeline exists (`MapSaveService` absent; `EditorSession.SaveAll()` writes nothing).
- Editor undo reverses only 2 of 5+ operation kinds.
- Zone music is disabled by policy and still reads `ZoneMusic` as a SoundEntries id.
- Modern data: the scene-light gate disables WMO instancing (~5.5 FPS, 16,431 WMO draws).
- v22 DAT blends layer 0 only; `AMAP` codec unidentified.
- `WorldScene.cs` 8,326 lines (after U-01 E1), `ViewerApp.cs` 16,746 lines — no new members (AGENTS.md §10).

## Non-negotiable constraints

- Preserve MPQ/ADT/WMO/M2/MDX readers and `AlphaWdtWriter`; no feature scope rides on a refactor.
- Runtime visual, input, FPS, audio, video and client-data proof are operator-owned.
- Training, harvests, GPU and cloud runs are operator-run; hand over PowerShell-ready commands.
- Preserve unrelated dirty work (`imgui.ini`); stage named files only.
- New scope needs operator-approved wording (AGENTS.md §9.1); receipts per §9.2.

## Handoff

**Immediate:** operator runs R10-T003/T007 captures and the U01-T003 E1 smoke. Agent may start
U-01 E3 (`ViewerApp` menu bar + converter dialogs, same verbatim-move mechanics as E1). Do not start
any other epic item before it is marked Want. Superseded dashboard:
[archive/2026-09-23-pre-reconciliation-active-context.md](archive/2026-09-23-pre-reconciliation-active-context.md).
