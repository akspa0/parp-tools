# WoWViewer Core Libraries

This folder contains the library layer for `wow-viewer`. Core projects own file formats,
runtime data contracts, editor operations, and reusable rendering primitives. The desktop
viewer and CLI tools consume these libraries; core projects do not depend on the viewer UI.

## Project Map

| Project | Role |
|---|---|
| `WowViewer.Core` | Shared domain models, map coordinates, tensors, buffers, and common contracts. |
| `WowViewer.Core.IO` | Format and archive I/O for MPQ, BLP, WDT, ADT, WDL, WMO, M2/MDX, LIT, and related generated outputs. |
| `WowViewer.Core.Runtime` | Runtime scene contracts for world state, terrain visibility, M2 animation, skinning, and object residency. |
| `WowViewer.Core.Anim` | Animation-specific helpers and compatibility surfaces. |
| `WowViewer.Core.PM4` | PM4/PD4 decode, geometry reconstruction, grouping, reconciliation, and analysis algorithms. |
| `WowViewer.Core.Editor` | Editor sessions, placement operations, undo/redo, guarded writes, and provenance tracking. |
| `WowViewer.Core.Renderer` | Reusable renderer primitives, headless capture support, and scene-building helpers. |
| `WowViewer.Core.Curation` | Dataset and corpus curation helpers used by tooling lanes. |

## Boundaries

- Keep UI code in `src/viewer/WoWViewer/`; core libraries must stay usable by CLI tools and tests.
- Keep CLI argument parsing in `tools/`; core APIs should expose explicit library contracts.
- Do not hardcode local client paths. Client roots are runtime inputs.
- Do not change proven format readers, writers, terrain loading, camera behavior, or renderer behavior just to make a new tool pass. Add opt-in adapters, probes, tests, or generated-artifact logic above the base layer first.
- `AlphaWdtWriter.cs` is frozen unless a separate, evidence-backed compatibility bug explicitly reopens it.
- Generated outputs should carry enough provenance for a later reader or viewer session to explain their source.

## Validation

Run from the repository root:

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug
```

For broader shared-contract changes, run the whole solution test suite:

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

Compilation and unit tests are source proof only. They do not establish runtime visual, FPS,
audio, GPU, or real-client proof.
