# T001–T002 Design Receipt — 2026-09-08

## Files authored

- `spec.md`, `plan.md`, `tasks.md`, `research.md`, `data-model.md`, `quickstart.md`
- `contracts/feature-tour-recipe.schema.json`
- `contracts/authoring-handoff.schema.json`
- `checklists/requirements.md`

## Verification

| Command | Exit | Real output |
|---|---:|---|
| `Get-Content -Raw specs/233-marketing-capture-automation/contracts/feature-tour-recipe.schema.json \| ConvertFrom-Json; Get-Content -Raw specs/233-marketing-capture-automation/contracts/authoring-handoff.schema.json \| ConvertFrom-Json` | 0 | `contracts-json=pass` |

`tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj` already references `WowViewer.Core.Runtime`; the feature adds no project/package, `ViewerApp` field, or `ViewerApp_*` partial class.

## Criterion-to-artifact mapping

| Requirement / outcome | Design artifact | Evidence boundary |
|---|---|---|
| FR-001 / SC-001 recipe, warm/run, ordered beats | `spec.md`, recipe schema, `data-model.md` | Contract only; no viewer run yet. |
| FR-006–FR-007 / SC-002 terminal benchmark receipt | `data-model.md`, Phase 4 tasks | Not implemented by T001–T002. |
| FR-008–FR-009 / SC-003–SC-004 safe external handoff | handoff schema, `research.md` Decision 5 | Transport deliberately unimplemented. |
| FR-010 / SC-005 real README assets | `quickstart.md`, T026 | Operator-only publishing gate. |

No runtime, video, UI, FPS, ComfyUI, or README-media claim is made by this receipt.

