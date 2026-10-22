# Spec 283 UI shell correction receipt — 2026-10-21

> **Superseded in part:** the Spellbook prototype described here was subsequently rejected by the
> operator and removed. See [the action-bars correction receipt](action-bars-and-popup-windows-2026-10-22.md).

## Scope

Applied the operator's follow-up to the WoW-style shell: use a compact minimap footprint, reduce
the chat dock height, replace the chat's general-purpose tool list with a Spellbook entry point,
and repair the generated UI reference warning for the `Minimap` Options section.

## Files changed

- `src/viewer/WoWViewer/Workbench/Services/UiManager/UiLayout.cs` — square minimap footprint and
  compact chat height.
- `src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs` — Spellbook launch points
  replace the chat tools list; spellbook is drawn by the UI manager.
- `src/viewer/WoWViewer/Workbench/Services/Spellbook/SpellbookService.cs` — owned tool-book surface.
- `src/viewer/WoWViewer/ViewerApp.cs` — instantiate the Spellbook service in the composition root.
- `src/viewer/WoWViewer/ViewerApp_Host.cs`,
  `src/viewer/WoWViewer/Workbench/Services/IViewerAppHost.cs` — expose the service to UI providers.
- `eng/ui/ui-controls.json` — document the `Minimap` Options section and Spellbook launchers.
- `docs/WoWViewer/UI-REFERENCE.md` — regenerated from the registry.
- `specs/283-v070-ui-presets-and-explore/plan.md`, `tasks.md`, and `specs/STATUS.md` — operator
  amendment, bounded task receipts, and active spec registry row.
- `memory-bank/activeContext.md` and `memory-bank/progress.md` — current handoff and session ledger.

## Verification commands

| Command | Exit | Actual result |
|---|---:|---|
| `pwsh -NoProfile -File wow-viewer/eng/ui/Update-UiReference.ps1 -Write` | 0 | Wrote `wow-viewer/docs/WoWViewer/UI-REFERENCE.md`. |
| `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | 0 | `inventory_rows=130 distinct_labels=120 registry_entries=121`; `PASSED errors=0 warnings=0`. |
| `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --no-restore -v:m -clp:ErrorsOnly` | 0 | Build succeeded; 384 warnings, 0 errors. |

## Criterion → evidence

| Criterion | Evidence | Result |
|---|---|---|
| Every visible control has a registry entry and the generated reference matches it. | `Verify-UiReference.ps1` output above; registry contains the `Minimap` Options section and Spellbook controls. | Pass (static check). |
| Viewer solution compiles with the new service wired into the composition root. | Solution build output above; `_spellbook` is assigned in `ViewerApp`'s constructor. | Pass (compile only). |
| Minimap/chat proportions and Spellbook behavior match the operator's visual intent in a running viewer. | No viewer launch or visual inspection was performed for this receipt. | **Open; not claimed.** |
| Spellbook tool actions reach the intended destinations in runtime. | No interactive action walkthrough was performed. | **Open; not claimed.** |

## Remaining validation

The spellbook currently presents named tool buttons grouped into tabs; this is an implementation
slice, not visual acceptance of the requested game-like treatment. Runtime interaction, screen-size
visual review, converter readiness, and 0.5.3 DBC automation remain unverified/out of this receipt's
scope.
