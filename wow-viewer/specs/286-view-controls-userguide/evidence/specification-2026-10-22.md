# Specification receipt — 2026-10-22

## Operator scope and existing-state findings

The operator accepts the present icon-bar direction as "very nice looking and pretty functional"
and "feels just right for what we're doing here," while reporting scaling/discoverability gaps.
This is evidence of overall direction acceptance, not proof that every Spec 283 control or runtime
gate passed. No new screenshot or exhaustive interaction matrix was supplied with that feedback.

Observed in existing sources:

- [View menu](../../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs:170)
  already exists, but exposes camera/chrome/sidebar/layout reset rather than direct text scale.
- [Font settings](../../../src/viewer/WoWViewer/Workbench/Services/Settings/SettingsWindowService.cs:157)
  contain text size and presets; the
  [layout reset](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Persistence.cs:86)
  does not reset the user font preference.
- [UI scale application](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs:64)
  combines the saved user preference with automatic display scaling. A user-scale reset and a
  display-scale policy are not interchangeable; the implementation plan must preserve readability.
- [Map discovery](../../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/DataSourceSessionService.cs:228)
  sets an auto-open flag, while the
  [Maps tool window](../../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerActionBarsService.cs)
  currently opens from its icon. The plan must identify one successful-activation path for each
  supported source and consume the signal once, rather than rely on ordinary frame updates.
- [User guide](../../../docs/WoWViewer/USERGUIDE.md:93) still describes the retired sidebar design.
  The guide directory currently contains four Markdown documents and no screenshot assets. The
  guide must not be declared illustrated/current until approved captures are available.

## Artifacts authored

- [Feature specification](../spec.md): 3 user stories, 13 functional requirements, 7 success criteria.
- [Quality checklist](../checklists/requirements.md): content, completeness and feature-readiness review.
- Central registry, owning UX epic and continuity updates link the new bounded follow-up without
  erasing existing Spec 283, 284 or 285 work.

## Workflow execution

1. Ran `git fetch --all --prune`: exit 0.
2. Inspected local/remote branches and spec directories. No exact `view-controls-userguide` feature
   existed; highest current spec was 285. Used global spec number 286 to avoid reuse/collision.
3. Ran the project's PowerShell feature-creation script exactly once from the viewer project:

   `pwsh -NoProfile -File .specify/scripts/powershell/create-new-feature.ps1 -Json -Number 286 -ShortName view-controls-userguide "Preserve the accepted icon-bar UI; expose view, scale and reset controls in View; open Maps automatically after successful client load; provide a real screenshot-based user guide."`

   Exit 0; returned branch `286-view-controls-userguide`, feature number 286, and the spec path;
   updated the active feature pin. Existing dirty worktree changes were retained, not reset/staged.

## Specification review

| Criterion | Evidence | Result |
|---|---|---|
| Operator-origin scope | Spec input quotes and three stories map to View/scale, automatic Maps and screenshot guide. | Pass |
| Testable scenarios | US1 has 4 scenarios, US2 has 5, US3 has 4; edge cases distinguish client activation, refresh, cancellation and no-map outcomes. | Pass |
| Measurable user outcomes | SC-001–007 cover menu depth, reset correctness/persistence, display readability, one-shot Maps and real guide illustrations. | Pass |
| No invented feature implementation | No runtime source edit, new screenshot, viewer launch or converter work was performed in this specification pass. | Pass |
| Screenshot/runtime evidence boundary | FR-012 and assumptions keep actual client/visual capture operator-owned; build/test output cannot satisfy those gates. | Pass |

This is a specification receipt, not an implementation receipt. Next: implementation plan and
dependency-ordered tasks, beginning with shared scale/reset ownership and View-menu access.

## Validation result

Specification structure, branch/pin consistency, placeholder checks, scoped whitespace and local
file/source-line links were validated with PowerShell; exit **0**. [Actual output](spec-quality-validation.txt):
3 stories, 13 requirements, 7 criteria, 16 checked specification-quality items, 12 local links checked
at validation time. The output explicitly records implementation not run and screenshots not captured.
The existing source worktree remains the earlier Spec 283 implementation, not code delivered by this
specification pass.
