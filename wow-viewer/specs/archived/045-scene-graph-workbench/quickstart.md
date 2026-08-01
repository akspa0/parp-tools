# Quickstart: Scene Graph Workbench

## Goal

Validate that `WoWViewer` can expose one whole-scene hierarchy for a loaded world session and keep it synchronized with existing viewer selection.

## Preconditions

- A staged client exists under `I:\parp\parp-tools\output\tmp\wowarchive-clients\`
- A world map can be loaded in `WoWViewer`
- PM4 data is available for the chosen validation map
- The dockable shell path from spec 044 is enabled

## Manual Validation Flow

1. Launch `WoWViewer`.
2. Load a staged client and open a world map with terrain, objects, and PM4 available.
3. Open the `Scene Graph` panel from the right sidebar or dockable shell.
4. Confirm the tree shows one shared root and separate top-level branches for:
   - terrain
   - world objects
   - PM4
5. Expand the terrain branch until a tile and chunk are visible.
6. Expand the PM4 branch until a region, object, and sub-object are visible.
7. Expand the object branch until at least one placed WMO or M2 instance is visible.
8. Click one node from each domain and confirm:
   - existing selection/inspector state updates
   - the tree remains open and coherent
9. Select an element from the viewport or existing inspector and confirm the tree highlights the matching node.
10. Filter by:
    - a tile coordinate
    - a PM4 region id or CK24-like identifier
    - an object label or uid
11. Confirm matching nodes remain reachable without the whole tree expanding permanently.

## Suggested Focused Test Commands

```powershell
dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter SceneGraph
dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

## Expected Outcomes

- The scene graph behaves like a real outliner, not three unrelated debug lists.
- Terrain, placed objects, and PM4 can be browsed from one hierarchy.
- Selection is synchronized both directions for supported node types.
- Large branches remain lazy and filterable instead of dumping every leaf at once.
