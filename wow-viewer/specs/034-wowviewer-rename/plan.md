# Phase 1: Defunct Old App
- Rename src/viewer/WowViewer.App to src/viewer/WowViewer.App.Defunct.
- Update project references to the new paths and remove it from the solution.

# Phase 2: Elevate WoWViewer
- Rename src/viewer/MdxViewer to src/viewer/WoWViewer.
- Rename project and assembly to WoWViewer.
- Deep string and namespace replacement of MdxViewer to WoWViewer across the codebase.
- Bump version strings in GUI to 0.5.0.
