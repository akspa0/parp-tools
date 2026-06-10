# PM4 Exporter Consolidation Plan — 2025-07-21

## Objective
Unify the fragmented PM4 OBJ exporters into a single, maintainable `Pm4Exporter` service while preserving all existing export capabilities (whole-scene, per-group, assembled objects).

## Rationale
Eight separate exporter classes duplicate large swaths of OBJ writing logic and make future maintenance error-prone. Consolidation reduces surface area, eliminates code drift, and aligns with the overall PM4/PD4 tooling cleanup.

## Scope
Consolidate the following legacy exporters:
1. `Pm4ObjExporter`
2. `Pm4GroupObjExporter`
3. `Pm4SceneExporter`
4. `Pm4OptimizedObjectExporter`
5. `Pm4TileBasedExporter`
6. `Pm4RawGeometryExporter`
7. `Pm4SurfaceGroupExporter`
8. Any ad-hoc OBJ writers inside test commands

## Implementation Steps
1. **Audit Existing Exporters**
   • Catalogue unique behaviours and flags.
2. **Design Unified API** (`Pm4Exporter`)
   ```csharp
   static class Pm4Exporter
   {
       void ExportWholeScene(Pm4Scene scene, string outputFile, bool writeFaces);
       void ExportGroups(Pm4Scene scene, string outputDir, bool writeFaces);
       void ExportAssembledObjects(IReadOnlyList<MsurObject> objects, Pm4Scene scene, string outputDir);
   }
   ```
3. **Refactor Shared Logic**
   • Move common OBJ-writing routines (vertex/face output, material stubs) into private helpers.
4. **Update CLI Call Sites**
   • `ExportCommand`, `Program.cs`, and test commands call the new API.
   • Map existing flags: `--exportchunks` → `ExportGroups`, `--objects` → `ExportAssembledObjects`, default → `ExportWholeScene`.
5. **Remove Redundant Classes**
   • Delete superseded exporter files after successful build.
6. **Validation**
   • Build solution.
   • Run exports on real multi-tile data to verify parity with legacy output.
7. **Documentation**
   • Update `progress.md` and audit docs.
   • Note that CLI semantics remain unchanged for end-users.

## Success Criteria
- Build passes with no references to legacy exporters.
- OBJ outputs from new exporter match legacy results byte-for-byte (excluding ordering differences).
- CLI flags remain unchanged and work as before.
- Code coverage for exporter logic reaches 80% via regression tests.

---
*Document authored automatically by Cascade on 2025-07-21 22:41 EDT.*
