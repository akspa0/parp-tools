# Quickstart: UI Overhaul Slice

## Build

```powershell
dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

## Focused source checks

- Confirm Help > Keyboard Shortcuts is present.
- Confirm Capture authoring handling checks the active Capture context.
- Confirm the right workbench uses the vertical Model/World/Tools rail.
- Confirm the log child uses wrapped text and no horizontal scrollbar.

## User-run visual checks

1. Launch the viewer against a configured client root and load a world.
2. Open Help > Keyboard Shortcuts; switch Model, World, Tools, and Utilities pages and confirm the active section changes.
3. Open Capture > Camera Path, enable keyboard authoring, leave the Capture page, and press `K`, `U`, `Delete`, `Z`, or `X`; verify no path/camera authoring state changes.
4. Return to Capture and verify the same keys work only when the Capture page is active.
5. Load a map with a minimap and discovered map list; verify both remain accessible in the left navigator.
6. Open Log, generate a long message, and verify wrapping plus vertical scrolling.
7. Open a promoted utility window, click the viewport, and close it with its title-bar X.

## Proof boundary

The build proves compilation only. FPS stability, client asset visibility, terrain correctness, and capture playback remain manual real-client checks owned by the user.
