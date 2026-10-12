# Quickstart: Legacy MDX/M2 Rendering (1.0.0-3.0.1) & Fuckported-Asset Compatibility

Operator-facing commands for each phase's exit gate. All commands are PowerShell 7-ready per
AGENTS.md §5. None of these claim runtime/visual proof by themselves — build/test output gates
code correctness; the operator's own viewer session gates visual acceptance (AGENTS.md §6).

## Phase 0 — Reconciliation (no code change, research only)

No commands yet — Phase 0 is tracing callers and reading code/decompiled evidence. Its output is
`research.md` filled in, not a runnable artifact.

## Phase 1 — Survey (once built)

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect -- `
  m2 survey --client-root "<configured client library root>" --output "wow-viewer/output/spec235-survey.json"
```

Read the resulting survey records; every row must have no `NotPresent`/`Failed` section left
without a `Detail`, and no build/model pair should be missing.

## Phase 2/3 — Geometry, skeleton, and extended-range verification

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~M2Era100|FullyQualifiedName~M2Era1121"
```

Then, operator-run: load a `0x100`-era model (e.g. the 2.0.0.5610 Blood Elf/Night Elf pair
previously failing at bone index 10) and a `0x107` model (3.0.1.8303) in the viewer; confirm visible
geometry or an accurate bounding box, never nothing.

## Phase 4 — Fuckported-asset parity

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect -- `
  m2 fuckport-check --asset "<path to the identified fuckported asset>" --reference warcraftnet
```

Operator-run: load the same asset in the viewer and confirm it renders.

## Phase 5 — Light effects

Operator-run: load a torch-bearing MDX model (identified in Phase 0 step 8) in the viewer and
confirm its light/glow effect is visible.

## Regression check (run at every phase exit)

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

Compare the failing-test **set** (not just the count) against the baseline recorded in Phase 0 —
per plan.md's Regression Protection section, a changed set is a regression until proven otherwise.
