# Active Context — wow-viewer

Last updated: 2026-09-23 · Branch: `v0.6.0-dev`

## Fresh-chat route

1. [Spec status](../specs/STATUS.md) — 7 active epics (248–254), all **triage pending**.
2. This compact handoff.
3. [TRIAGE.md](../specs/TRIAGE.md) — the operator's Want / Drop / Later decisions.
4. Only then one epic's `spec.md` → `plan.md` → `tasks.md`, and only the archived design documents
   its `plan.md` adopts for the selected item.

[Progress ledger](progress.md), [memory archive](archive/README.md) and `specs/archived/` are
on-demand history, never default reading.

## Current lane — backlog triage (operator-owned)

On 2026-09-23 every open spec (137) was audited against the code and archived; open residue now lives
in 7 epics. **Nothing is scheduled.** The operator marks each item Want / Drop / Later in TRIAGE.md;
then each epic gets phased tasks and a chosen implementation approach (operator directive: flag
wanted/unwanted first, approach second). Receipt:
[reconciliation README](../specs/archived/reconciliation-2026-09-23/README.md).

**Next bounded action (agent):** once TRIAGE.md has decisions, record them as dated amendments in each
epic's `spec.md` and run `speckit-tasks` for the Want items only. Start with the quick-win shortlist
if the operator marks those Want.

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
- `WorldScene.cs` 17,153 lines, `ViewerApp.cs` 16,746 lines — no new members (AGENTS.md §10).

## Non-negotiable constraints

- Preserve MPQ/ADT/WMO/M2/MDX readers and `AlphaWdtWriter`; no feature scope rides on a refactor.
- Runtime visual, input, FPS, audio, video and client-data proof are operator-owned.
- Training, harvests, GPU and cloud runs are operator-run; hand over PowerShell-ready commands.
- Preserve unrelated dirty work (`imgui.ini`); stage named files only.
- New scope needs operator-approved wording (AGENTS.md §9.1); receipts per §9.2.

## Handoff

**Immediate:** wait for triage decisions in `specs/TRIAGE.md`. Do not start implementation on any
epic item before it is marked Want. Superseded dashboard:
[archive/2026-09-23-pre-reconciliation-active-context.md](archive/2026-09-23-pre-reconciliation-active-context.md).
