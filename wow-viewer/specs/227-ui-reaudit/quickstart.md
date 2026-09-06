# Validation Quickstart — Spec 227

## Source-audit check

From `I:\parp\parp-tools`:

```powershell
rg -n -g "*.cs" "GetBottomTabLabels|GetArchaeologyWorkbenchLabels|DrawMinimapWindow|DrawUtilitiesMinimap|DrawUnifiedInspectorContent|DrawTerrainControlsAdjustmentWeakSignalContent|DrawTerrainLabSubTab" wow-viewer/src/viewer/WoWViewer
```

Confirm every result is represented in `surface-inventory-v2.md`, or add an inventory row before
changing its route or presentation.

## Per-source-slice gate

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

Run focused tests only for the changed owner. A build/test result is source evidence; it does not
pass a visual, input, or teleport acceptance criterion.

## Operator walkthrough gate

For each `UIV2-*` row, record the client root, viewer build, profile, path to the surface, and one
screenshot in `evidence/screenshots/`. For duplicate families additionally record:

- weak-signal: the authoritative destination and any converted link;
- minimap: which host received the click, tile target, and resulting camera position;
- Inspector: object type, authoritative page, and the route/link used by a former duplicate.

Stop the phase and update its receipt if the actual build exposes an unlisted surface or a route
does not remain reachable within three interactions.
