# Workspace Instructions

These instructions apply to `I:/parp/parp-tools` and its active projects. Keep this file
operational and short. Historical findings belong in specs, architecture notes, or the memory-bank
workstream files—not here.

## Start here

For `wow-viewer` work, read in this order:

1. `wow-viewer/AGENTS.md`
2. `wow-viewer/specs/STATUS.md`
3. `wow-viewer/memory-bank/activeContext.md`
4. The selected spec's `spec.md`, `plan.md`, and `tasks.md`
5. Only the linked research/workstream files needed for the current task

Use Spec Kit for non-trivial work: specify → plan → tasks → implement one validated phase at a
time. If the request is a small fix, use the existing spec/checklist when one applies.

## Ownership and safety

- All new code, tests, tools, and docs for the viewer go under `wow-viewer/`.
- `gillijimproject_refactor/` is read-only reference code unless the user explicitly requests a
  bounded legacy hotfix.
- Do not rewrite or duplicate existing game-file readers. Extend shared `WowViewer.Core.*` owners
  only when the current contract lacks the required behavior.
- Preserve unrelated dirty worktree changes. Stage named files only; never use broad staging.
- Do not delete, reset, or overwrite user data without explicit approval and verified targets.
- Do not claim runtime, visual, FPS, GPU, audible, or real-client proof from compilation or unit
  tests alone.

## Execution boundaries

- The user runs training, GPU jobs, data harvests, long/heavy/billed runs, and real-client visual
  proof. Prepare exact commands and stop before launching those operations.
- Read-only inspection, focused tests, and quick builds are allowed when needed to implement safely.
- Client roots are runtime configuration. `H:\CLIENTS` is an approved library; never hardcode a
  machine-local client path into source or portable docs. Record the exact build/root/fingerprint
  for validation.
- Python work belongs under `wow-viewer/data-harvester/`, uses its `uv` environment, and must not
  be launched from the repository root when package imports depend on that project.
- Every command handed to the user must be PowerShell 7 compatible: use backticks for continuation,
  PowerShell variables, and no bash heredocs, `export`, `/tmp`, or POSIX-only command syntax.

## Code ownership

- Shared data models: `wow-viewer/src/core/WowViewer.Core/`
- Shared format I/O: `wow-viewer/src/core/WowViewer.Core.IO/`
- Runtime/M2/world contracts: `wow-viewer/src/core/WowViewer.Core.Runtime/`
- PM4: `wow-viewer/src/core/WowViewer.Core.PM4/`
- Viewer shell and rendering: `wow-viewer/src/viewer/WoWViewer/`
- CLI tools: `wow-viewer/tools/`
- C# tests: `wow-viewer/tests/`
- Python ML/data tooling: `wow-viewer/data-harvester/`

Keep readers library-first, tools thin, and UI out of core libraries. Maintain existing Alpha vs
Standard terrain adapter separation. Keep `AlphaWdtWriter.cs` frozen unless explicitly reopened by
the user or a proven compatibility regression requires it.

## Validation

Preferred viewer checks:

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

Use focused tests for the changed owner first. For real-client checks, state the configured client
root, build, and proof level. Do not use the legacy `MdxViewer` as the active viewer test target.

## Documentation and continuity

- Specs are the source of truth for feature behavior and phase order.
- `wow-viewer/memory-bank/activeContext.md` is a short dashboard: active lanes, next task, proof
  owner, main gap, and out-of-scope items.
- `wow-viewer/memory-bank/progress.md` is a short newest-first ledger: one compact entry per
  completed session or phase. Move durable findings to a workstream file.
- Keep `coding_standards.md`, `data-paths.md`, `projectbrief.md`, `systemPatterns.md`, and
  `techContext.md` stable; change them only when a durable rule changes.
- Archive superseded detail under `wow-viewer/memory-bank/archive/` and index it in that archive's
  README. Do not delete negative results that prevent repeated mistakes.
- Update the relevant spec and continuity dashboard in the same pass as non-trivial code changes.

## Communication

Lead with the result. State what changed, what was validated, what remains user-owned, and the next
bounded step. Do not bury an unresolved proof gap under a long history recap.
