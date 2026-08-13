# wow-viewer Agent Guide

`wow-viewer/` is the active, standalone development target. Keep this guide operational; feature
history belongs in specs and opt-in workstream notes.

## Required handoff order

1. Read this file.
2. Read `specs/STATUS.md`.
3. Read `memory-bank/activeContext.md`.
4. Open the selected spec's `spec.md`, `plan.md`, and `tasks.md`.
5. Read only the linked research/workstream note needed for the current task.

For non-trivial work use Spec Kit: specify → plan → tasks → implement one phase at a time. Do not
start a later phase until the current phase has focused validation.

## Sub-agents

- Use sub-agents by default for independent, safely parallelizable discovery, analysis, focused
  verification, or bounded implementation slices. Keep delegated work inside the active Spec Kit
  phase and run independent slices in parallel when practical.
- Give each sub-agent one concrete deliverable and an explicit read/write scope. Use disjoint write
  sets; do not ask multiple agents to edit the same unresolved files or to perform unrelated cleanup.
- The primary agent owns spec selection, architecture decisions, integration, conflict resolution,
  final validation, commits, and the user-facing handoff. Review every delegated patch and result
  before treating it as complete.
- Do not delegate training, GPU/heavy jobs, broad harvests, long captures, billed operations, or
  real-client visual/FPS/audio proof. Prepare PowerShell-ready commands for the user instead.

## Non-negotiable boundaries

- New code and tests stay in `wow-viewer/`.
- `gillijimproject_refactor/` is read-only reference. Do not add features or refactor it.
- Reuse existing ADT, WDT, WMO, M2, MDX, BLP, DBC/DB2, MPQ, PM4, and chunk readers. Add missing
  shared behavior in `src/core/` rather than cloning parser logic in the viewer or tools.
- The user owns training, GPU/heavy jobs, broad harvests, long-running captures, and real-client
  visual/FPS/audio proof. Do not launch them; prepare PowerShell-ready commands instead.
- Client roots are configured at runtime. Never hardcode `H:\CLIENTS` or another machine-local path
  into source or portable docs. Report root, build, and fingerprint for validation.
- Preserve unrelated dirty changes. Do not use destructive git commands or broad staging.
- Build/test proof is not runtime/rendering/performance/audio proof.

## Ownership map

| Surface | Owner |
|---|---|
| Shared models | `src/core/WowViewer.Core/` |
| Format I/O | `src/core/WowViewer.Core.IO/` |
| Runtime/M2/world contracts | `src/core/WowViewer.Core.Runtime/` |
| PM4 | `src/core/WowViewer.Core.PM4/` |
| Viewer shell/rendering | `src/viewer/WoWViewer/` |
| CLI tools | `tools/` |
| C# tests | `tests/` |
| Python ML/data tooling | `data-harvester/` |

Keep core libraries free of UI ownership; keep tools thin; preserve the Alpha/Standard terrain
adapter split. `src/core/WowViewer.Core.IO/Maps/AlphaWdtWriter.cs` is frozen unless explicitly
reopened for a proven compatibility fix.

## Python and client data

- Python environments, scripts, and libraries live only under `data-harvester/` and are managed by
  `uv`.
- Run Python from `data-harvester/`, not the repository root, when package imports require it.
- Use configured client roots directly when available; project-local staging is optional.
- Keep proprietary client data, harvested corpora, model outputs, and weights out of commits.

## Validation

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

Use focused tests before solution-wide checks. For viewer runtime testing use the active WoWViewer
project, not the legacy `MdxViewer`. For PM4 inspection use the existing `WowViewer.Tool.Inspect`
commands and the checked-in development fixtures when the changed surface requires it.

## Continuity rules

- `memory-bank/activeContext.md`: current dashboard only—active specs, next bounded task, proof
  owner, main unproven gap, and explicit out-of-scope items.
- `memory-bank/progress.md`: newest-first compact ledger; one entry per completed slice, with proof
  level and next handoff. Do not append command transcripts or old history.
- Durable findings, traps, measurements, and commands belong in a named workstream file or the
  owning spec's research/quickstart document.
- Update the owning spec and continuity dashboard in the same pass as non-trivial implementation.
- Archive superseded detail under `memory-bank/archive/`; preserve negative results that prevent
  repeated dead ends.

## Handoff format

End a substantive session with:

```text
Current target: <spec and task>
Proof owner: <tests/build/user/runtime>
Completed: <small factual summary>
Unproven: <one or two concrete gaps>
Next step: <one bounded task>
Out of scope: <explicit exclusions>
```
