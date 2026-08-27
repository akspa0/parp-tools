# WoWViewer Desktop App

`src/viewer/WoWViewer/` is the active desktop viewer and editor shell. It hosts the
OpenGL viewport, ImGui workbench, client source selection, world navigation, scene inspection,
performance panels, camera tools, audio diagnostics, and PM4/editor workflows.

The current app is editor and inspection tooling. Future player-model camera or game-mode work
belongs beside this shell as explicit SpecKit work; it should not rewrite working editor camera,
terrain loading, or renderer behavior as a side effect of new tooling.

## Run

From the repository root:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

Launch into a configured client root and world:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug -- `
  --game-path "H:\CLIENTS\WoW-0.5.3.3368-Client" `
  --world "World\Maps\Kalimdor\Kalimdor.wdt"
```

`H:\CLIENTS` is an approved local library path for this workspace, but it is still runtime
configuration. Do not bake it into source code, tests, or portable documentation examples that
are meant to run elsewhere.

## Main Surfaces

| Surface | Role |
|---|---|
| Navigator | Client roots, file tree, world maps, phase maps, and source selection. |
| Viewport | Streaming world render, fly camera, selection, overlays, terrain, WMO, M2/MDX, lighting, and audio markers. |
| Quick | Compact status and common operational shortcuts. |
| Inspect | Selected model, terrain, ADT/MCNK, PM4, scene investigation, animation, and action details. |
| Scene | Placements, level-of-detail controls, and scene-scoped state. |
| Utilities | Minimap, audio, performance counters, frame history, captures, and diagnostics. |
| Experimental | Terrain lab, archaeology, and bounded prototype surfaces. |

See [the user guide](../../../docs/WoWViewer/USERGUIDE.md) for controls and end-user workflows.

## Development Boundaries

- Keep shared logic in `src/core/`; viewer code should compose library APIs instead of duplicating readers or algorithms.
- Keep new tools opt-in. Do not alter protected base renderer, terrain, camera, or format behavior to make a generated artifact look correct.
- Treat frame history and capture artifacts as evidence, not assumptions. A build is not visual or FPS proof.
- Real-client visual, audio, FPS, and long capture validation remain operator-owned unless a spec says the proof is automated.

## Validation

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

There is no dedicated viewer test assembly today. Viewer changes usually need source inspection,
focused core tests for extracted logic, and a real viewer or capture proof when behavior is visual.
