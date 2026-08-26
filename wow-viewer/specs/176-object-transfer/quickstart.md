# Quickstart: PM4-Guided Museum Placement Repair

This is the current validation path. It does not modify a game install, PM4 file, or source Museum map.
Use a configured client root. Outputs land under the project output root, not the client tree.

## Prerequisites

- Specs 166–168, 173, and 175 library checkpoints are present (`WowViewer.Core.Editor`, `AdtPlacementEditor`).
- A Museum map whose PM4 overlay tiles load in the viewer, with matching unpadded `_obj0.adt` companions on disk.
- A configured client build and recorded fingerprint when claiming corpus identity.
- An output directory outside the client installation (the viewer creates `output/projects/<map>/<yyyyMMdd_HHmmss>` if none is set).

## Source validation

From PowerShell 7 at the repository root:

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Editor.Tests/WowViewer.Core.Editor.Tests.csproj -c Debug --no-restore
```

Focused editor tests must cover adapter conversion, tile association (including `AlreadyAligned`), apply round-trips, substitute ID chronology, and staleness refusal. Existing baseline failures outside this assembly must be recorded rather than attributed to this feature.

## Current operator flow (committed through `6fa1adbb`)

1. Load the Museum map so the PM4 overlay tiles are resident. Experimental → PM4 → **Reconcile**.
2. On the last committed build, PM4/`_obj0.adt` paths still have text + Browse (`ImGuiPathPicker`) and a **Use Camera Tile** prefill. Output defaults to a timestamped project folder; **New folder** forces a new one. **Do not type a client-install path.**
3. Run **Preview**. The panel lists proposals (reviewable / conflict / already aligned / accepted). Nothing is written.
4. Review rows: inline Accept/Reject, **Accept all reviewable**, details tree for residuals/evidence. `AlreadyAligned` is a result, not work. Confidence is display-only (FR-012).
5. **Apply Accepted (n)** writes loose ADT copies plus `.reconciliation.json` under the project output folder. Source ADTs and PM4 files stay untouched.
6. Manual placement edits (Editor tab or Scene → Placements) use the same staged-save queue: **Save Current Source** / **Save All Pending**.

Working-tree WIP (not compiled): hide path fields entirely and discern every loaded PM4 tile + `_obj0.adt` pair from the scene. That slice is incomplete — `ViewerApp_Sidebars.cs` still calls the deleted prefill helper.

## Still user-owned

- In-scene overlay of current vs proposed transforms (Phase 3 step 3).
- Reload of written ADTs in a fresh viewer session and an independent reader.
- Undo after apply on a real pair.
- One accepted align, one accepted clone, and one ambiguous/conflict case that mutates nothing.

Build/test success alone is not visual or runtime proof.

## PowerShell handoff for user-owned visual proof

Launch the active `WoWViewer` project with the configured client root, load one small Museum/PM4 pair, open Experimental → PM4 → Reconcile, Preview → accept a known-good subset → Apply Accepted, then reload the written ADT from `output/projects/<map>/<timestamp>/`. Heavy corpus sweeps, long captures, and real-client proof remain user-owned and must not be launched by the implementation agent.
