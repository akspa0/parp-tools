# Spec 298 — PM4 Authentic Pairing Telemetry, Markdown Elimination of CK24 & Multi-Object GLB Export

## 1. Problem Statement

1. **Pairing Telemetry Deficiency**: The pairing rules (`TaughtPairing`) and exported manifests currently store model name, yaw offset, and bounding box dimensions, but omit authentic object spatial coordinates (`PlacementPosition`, `PlacementRotation`, `PlacementScale`), authentic placement `UniqueId`, tile coordinates, and PM4 hull centroids/bounds. This discards critical placement ground truth.
2. **Persistent CK24 Exposure in Markdown Exports**: While recent UI panels removed user-facing `CK24` labels, exported markdown files (`<map>-pm4-shape-match-summary.md`, `pm4_llm_bundle.md`, `pm4_overlay_*.md`) still render `CK24` tables and columns instead of genuine object coordinates, dimensions, unique IDs, and model paths.
3. **Absence of PM4 GLB 3D Exports**: The viewer currently provides OBJ export for PM4, but lacks GLB scene export. Users need a modern GLB output where every PM4 collision object is an individual, named node alongside the textured/MCCV terrain geometry, allowing full inspection and separation in modern 3D tools (Blender, Godot, etc.).

## 2. User Requirements

- **FR-001 (Authentic Pairing Telemetry in Taught Rules)**: Enrich `TaughtPairing` to capture the complete ground-truth telemetry of both the placed object and the PM4 collision hull:
  - Placed object telemetry: `PlacementPosition` (Vector3 X, Y, Z in yards), `PlacementRotation` (Vector3 Pitch, Roll, Yaw degrees), `PlacementScale` (float), and `UniqueId` (uint).
  - Paired PM4 hull telemetry: `TileX`, `TileY`, `ObjectPartId`, `LinkGroupObjectId` (uint), `HullCentroid` (Vector3), `HullBoundsMin` (Vector3), and `HullBoundsMax` (Vector3).
  - Maintain backwards compatibility with existing pairing project files while serializing all telemetry to `<map>.pairings.json` and `<map>.manifest.json`.
- **FR-002 (Purge CK24 from All Markdown Outputs)**: Update markdown summary generators across the PM4 workbench:
  - In `Pm4ShapeMatchBatchService.FormatMarkdownSummary`: Replace the raw `CK24` table column with `Object ID`, `Position (X, Y, Z)`, `Rotation`, `Scale`, `Dimensions (LxWxH)`, `Matched Model`, and `Unique ID`.
  - In `Pm4WorkbenchService.Collections.BuildPm4LlmBundleMarkdown`: Replace raw `CK24` text lines with placement height, part ID, and world bounds.
  - In `Pm4WorkbenchService.SceneGraph.ExportPm4OverlayReport`: Replace `CK24` column with `Placement Z (yd)` and part ID.
- **FR-003 (PM4 Multi-Object GLB Exporter Engine)**: Create `Pm4GlbExporter` in `WoWViewer.Export`:
  - Export every PM4 collision object as an independent rigid mesh and scene node in the GLB scene graph.
  - Node naming convention: `[{TileX}_{TileY}] Obj #{PartId} - {ModelNameOrSurfaceClass}`.
  - Materials: Assign double-sided PBR collision materials color-coded by surface class (doodad, floor, solid wall, ramp, etc.).
  - Terrain integration: When terrain is available, include the terrain tile mesh with its blended layer textures / minimap image and vertex MCCV as a separate base node (`Terrain_{TileX}_{TileY}`).
  - Provide both single-tile export (`ExportTileGlb`) and multi-tile combined export (`ExportTilesCombinedGlb`).
- **FR-004 (UI Action Bar & Export Workbench Integration)**:
  - In `ExportWorkbenchService` and `Pm4WorkbenchService.Exports.cs`, add one-click action buttons: `[ Export PM4 Tile to GLB... ]` and `[ Export All Loaded PM4 to GLB... ]`.
  - When a single PM4 object is selected in the inspector, provide an `[ Export Selected PM4 to GLB... ]` button.
- **FR-005 (Verification & Non-Destructive Integrity)**: Full solution builds with 0 errors, unit tests pass 100%, `Verify-UiReference.ps1` passes, and all changes are backed by a §9.2 receipt.

## 3. Success Criteria

- **SC-001**: Harvested and taught pairings serialize full object position, rotation, scale, unique ID, and PM4 hull dimensions/centroid to project and manifest JSON files.
- **SC-002**: Generated markdown summaries contain zero `CK24` columns/labels, presenting clean object coordinates, IDs, dimensions, and model paths.
- **SC-003**: GLB exports load in standard 3D viewers (Windows 3D Viewer, Blender) with all PM4 collision objects as separate selectable meshes and textured terrain geometry underneath.
- **SC-004**: Full solution build succeeds with 0 errors, unit tests pass, and UI reference check passes with 0 errors/warnings.
