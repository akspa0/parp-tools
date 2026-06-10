# PM4 Consolidation Audit — 2025-07-21

This document records the Phase 1 audit of all PM4-related services found under `src/parpToolbox/Services/PM4/`. The goal is to decide which classes will be **Kept**, **Merged**, or **Removed** to achieve a single authoritative pipeline centred on `Pm4Adapter`.

| File/Class | Purpose | Action |
|------------|---------|--------|
| `Pm4Adapter` | Core single-tile loader + analysis helpers | **Keep & Extend** (cross-tile load, MSCN remap) |
| `Pm4RegionLoader` | Cross-tile region loader | **Merge** into `Pm4Adapter` (becomes `LoadRegion`) |
| `Pm4GlobalTileLoader` | Early experimental region loader | **Remove** (superseded by RegionLoader) |
| `Pm4MsurObjectAssembler` | Current best object grouping (SurfaceGroupKey) | **Keep** |
| `Pm4ObjectAssembler` | Legacy IndexCount-based grouper | **Remove** |
| `Pm4MprlObjectGrouper` | MPRL-based grouper | **Remove** |
| `Pm4HierarchicalObjectAssembler` | MPRR hierarchy experiment | **Remove** |
| `Pm4SmartGrouper` | Auto-heuristic grouper prototype | **Remove** |
| `Pm4GroupObjExporter` | Per-group OBJ exporter | **Merge** into unified `Pm4Exporter` |
| `Pm4ObjExporter` | Whole-scene OBJ exporter | **Keep** (may rename to `Pm4SceneExporter`) |
| `Pm4SceneExporter` | Duplicate of above (faces toggle) | **Remove** |
| `Pm4TileBasedExporter` | Tile-prefix OBJ exporter | **Remove** |
| `Pm4RawGeometryExporter` | Vertex-only debug exporter | **Keep** as debug helper (rename `Pm4DebugRawExporter`) |
| `Pm4OptimizedObjectExporter` | Performance variant of object exporter | **Remove** |
| `Pm4SurfaceGroupExporter` | Groups by SurfaceGroupKey (overlaps Assembler) | **Remove** (functionality superseded) |
| `Pm4CsvDumper` | CSV chunk dumps | **Keep** (used by `pm4-analyze` when CSV requested) |
| `Pm4BulkDumper` | Binary chunk dumps | **Keep** |
| `Pm4IndexPatternAnalyzer` | Index pattern analysis | **Keep** (integrated into Adapter) |
| `Pm4ChunkCombinationTester` | Chunk correlation diagnostics | **Keep** for research |
| `Pm4GroupingTester` | CLI cmd for grouping experiments | **Remove** |
| `Pm4DataAnalyzer` | Stand-alone analyzer wrapper | **Remove** (logic in Adapter) |
| `Pm4UnknownFieldAnalyzer` | Raw field correlation | **Keep** (research) |

Legend:
* **Keep** – remains in solution; may be renamed or refactored.
* **Merge** – logic will be absorbed into another class; file will be deleted after merge.
* **Remove** – delete the file entirely to reduce clutter.

Next Steps:
1. Extend `Pm4Adapter` with logic from `Pm4RegionLoader` (and delete that file).
2. Migrate `GroupObjExporter` logic into `Pm4Exporter` and delete duplicates.
3. Delete all files marked **Remove** in a separate commit after successful build.
