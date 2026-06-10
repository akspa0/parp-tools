# Usage Guidelines and Best Practices

Detailed guidelines, code examples, best practices for parpToolbox (all formats: PM4/PD4 phasing MSLK/MSUR/MSCN, WMO/ADT tiles MOHD/MHDR/MODF/MDDF/MCNK, WDT maps MAIN/MDNM, M2 models MD21 verts/skins, Alpha legacy conversion). Priority: PM4FacesTool/PM4NextExporter (PM4 blueprint: DSU/grouping/CLI); extend to unified tool (single CLI --formats=pm4,adt,wmo,m2,wdt,alpha --mode=parse/export/analyze/convert). Derived from code; .NET 9 focus (preview C# records/patterns/spans for efficiency), chunk handling (ChunkedFile base), workflows (WDT→ADT→WMO/M2→PM4 match via WmoMatcher, Alpha→LK gillijimproject WdtAlpha.ToWdt). Best practices: Duplicate-free (shared IChunkedFile/ObjWriter/CoordinateTransforms), efficient (DSU batch async, validation --audit-only), mergeable (subprojects into parpToolbox Services/Formats).

## General Usage Principles (Unified Tool)
- **Build and Run**: `dotnet build` from root (single .sln post-merge); run `dotnet run --project src/parpToolbox -- [args]` or `dotnet tool install parp-tool --global` for unified CLI. parpToolbox first (Warcraft.NET NuGet replaces wow.tools.local). .NET 9 SDK; preview LangVersion for all (records/spans async).
- **Dependencies**: Ref parpToolbox for all (ChunkedFile parsing PM4File/ADTFile/WDTFormat/M2File/WmoFile, services Pm4BatchProcessor/AdtService/WmoRootLoader/M2ModelHelper); Warcraft.NET (chunks/M2), EF Core 9.0.8 (SQLite metadata), System.CommandLine beta (CLI --formats=pm4,adt,wmo,m2,wdt,alpha). Unify: Single deps list, no external wow.tools.local.
- **Input/Output**: Multi-format binaries (*.pm4/*.pd4/*.adt/*.wdt/*.wmo/*.m2/Alpha.wdt); inputs: single/dir (`--batch --input=zone.wdt --load-adts`). Outputs: project_output/<timestamp>/<format>/ (OBJ/GLTF/GLB per-tile/object, CSVs surface_coverage/mcnk_terrain/mohd_groups, JSON indexes, Alpha convert reports). ProjectOutput.GetPath("multi") for unified sessions.
- **Error Handling**: Nullable/chunk-optional (e.g., no MSPV/MCNK→stub OBJ in BatchProcessor/AdtService); validate (TryMap bounds, WdtFormat.IsValid MVER+MAIN); try-catch non-fatal (MSCN/MODF failures log/continue). Alpha: Handle sparse MAIN (--compare for diff).
- **Exclusions/Maintenance**: Exclude legacy (CompositeHierarchyInstanceAssembler, GltfExporter); favor DSU (PM4FacesTool for all assemblies). Update: Deprecated→removed post-merge; version all (e.g., WMO V14/V17 via converters).
- **Unification Best Practices**: Use IChunkedFile (FromFile/Serialize for all); shared exporters (ObjWriter.Write for PM4/WMO/M2/ADT verts/tris); batch MultiFormatBatchProcessor (--formats=all); CLI System.CommandLine (subcommands parse/export/analyze/convert-alpha); test WoWToolbox.Tests (PM4FileTests + AdtServiceTests + Alpha validation --compare).

## Priority Projects (Blueprints for Unified Tool)

### PM4FacesTool (src/PM4FacesTool/PM4FacesTool.csproj) - PM4 Exporter Blueprint
Minimal PM4 "faces" (polygons) handler; reference for multi-format export.

- **Structure**: Minimal .csproj; 2175-line Program.cs:1 standalone CLI; records Options:16 (25+ fields).
- **Deps**: parpToolbox (Pm4GlobalTileLoader/Pm4Scene/MsurChunk.Entry); custom ObjWriter/GltfWriter.
- **Best Practices**:
  - Entry: Main:48 (ParseArgs→Options, LoadRegion(dir/pattern/remap)→ToStandardScene, ProcessOne for export/diag).
    ```csharp
    var opts = ParseArgs(args); // --input/--group-by/--batch
    if (opts.Batch && Directory.Exists(opts.Input)) {
        var pm4s = Directory.EnumerateFiles(opts.Input, "*.pm4").ToList();
        foreach (var pm4 in pm4s) ProcessOne(pm4, opts);
    } else ProcessOne(opts.Input, opts);
    ```
  - Handling: Pm4Scene (Verts/Indices/Surfaces/TileOffset); group ExportCompositeInstances (DSU CK24-shared verts); AssembleAndWrite local geo.
    ```csharp
    var dsu = new DSU(n); // Program.cs:1062 union-find
    foreach (var kv in vertToSurfs) if (list.Count > 1) for (int i=1; i<list.Count; i++) dsu.Union(list[0], list[i]);
    // compToSurfaces[r].Add(surfaceSceneIndices[i]);
    ```
  - Export: AssembleAndWrite (EmitTriMapped/Projected tris, TryMap dedup, ApplyProjectLocal/GlobalTransform, ObjWriter.Write/GltfWriter.WriteGltf).
    ```csharp
    ApplyProjectLocal(localVerts, opts.ProjectLocal); // center
    ApplyGlobalTransform(localVerts, opts); // flip/rot/trans
    if (opts.FlipXEnabled) { /* winding swap */ }
    ObjWriter.Write(objPath, localVerts, localTris, legacyParity: false);
    ```
  - Resilience: TryMap bounds (skip g<0||g>=Verts.Count), degenerate filter (la==lb skipped), try-catch non-fatal (MSCN log/continue).
- **Usage Example** (PrintHelp CLI):
  1. Build: `dotnet build src/PM4FacesTool`
  2. Basic: `dotnet run --project src/PM4FacesTool --input sample.pm4 --out output/ --group-by composite-instance` → objects/tiles/ CSVs/JSON.
  3. GLTF/MSCN: `--gltf --mscn-sidecar --mscn-pre-rotz 90 --mscn-pre-flip xy` → .gltf + mscn OBJ.
  4. Snap: `--snap-to-plane --height-scale 0.02777778` → MSUR planes.
  5. Batch: `--batch --input pm4_dir/ --render-mesh-merged` → all *.pm4, render_mesh.obj.
- **Why Blueprint?**: Comprehensive (parse→export), extensible (Export[Strategy]), efficient (DSU O(n)/dedup), WoW-aware (tile/CK24). Unification: Generalize to MultiFormatFacesTool (--formats=pm4,wmo,m2: DSU for MOGP groups/M2 skins/MCNK terrain).

### PM4NextExporter (src/PM4NextExporter/PM4NextExporter.csproj) - CLI Exporter Blueprint
Advanced export with exclusions; prioritize for production/multi-format CLI.

- **Structure**: .csproj preview LangVersion/exclusions; thin wrapper (no Program.cs, System.CommandLine via parpToolbox); 30+ options orchestrate Pm4BatchProcessor/MslkHierarchyAnalyzer.
- **Deps**: parpToolbox (PM4File/exporters Pm4ObjExporter/LegacyObjExporter), EF logs.
- **Best Practices**:
  - Entry: pm4next-export input [options] (README): Wrap Pm4BatchProcessor.Process/custom assembly (mslk-parent MslkLinkGraphBuilder).
    ```csharp
    var processor = new Pm4BatchProcessor(buildingService, wmoMatcher);
    var result = processor.Process(pm4Path); // buildings/WMO match
    if (assembly == "composite-hierarchy") { /* DSU PM4FacesTool */ }
    LegacyObjExporter.ExportAsync(pm4, objPath, filename); // parity OBJ
    ```
  - Logic: PM4File.FromFile; strategy --assembly mslk-parent (MslkParentExporter); transforms CoordinateTransforms/Pm4CoordinateTransforms; preview records BatchProcessResult.
    ```csharp
    var mslk = pm4.MSLK;
    var exporter = new MslkExporter(); // IMslkExporter
    exporter.Export(mslk, outputPath, minTris: opts.MslkParentMinTris);
    ```
  - Hierarchy: --assembly composite-hierarchy (DSU default); MslkHierarchyAnalyzer --include-adjacent cross-tile.
  - Integration: ProjectOutput.CreateOutputDirectory (sessions), EF run.log, CsvExporter --csv-diagnostics.
- **Usage Example** (README):
  1. Build: `dotnet build src/PM4NextExporter`
  2. Basic: `dotnet run --project src/PM4NextExporter -- development.pm4 --format obj` → project_output/timestamp/ OBJ/MTL/run.log.
  3. Advanced: `--include-adjacent --export-tiles --csv-diagnostics --assembly mslk-parent --mslk-parent-min-tris 500` → cross-tile tiles/ CSV.
  4. Audit: `--audit-only` → validation no export.
  5. MSCN: `--export-mscn-obj` → MSCN OBJ.
- **Why Blueprint?**: Flexible CLI (30+ options), production (batch/audit/diag), extensible (--assembly add), scripted complement to PM4FacesTool. Unification: Unified CLI core (--formats=all --assembly=dsu --export=obj for PM4/WMO/ADT/M2/WDT; --convert-alpha for gillijimproject).

## Applying Guidelines to Other/Subprojects (Merging)
- **parpDataHarvester**: Refactor minimal (PM4FacesTool style, drop RootNamespace/AssemblyName); integrate MultiFormatBatchProcessor (--formats=pm4,adt,wmo,m2 --output=csv/json); add --batch like PM4NextExporter. Merge: Into unified CLI subcommand harvest.
- **PM4MscnAnalyzer**: Preview LangVersion/exclusions; use ExportMscnSidecar + MscnMeshComparisonAnalyzer (MSUR/WMO validation). Generalize ChunkAnalyzer (--chunk=mscn,mcnk,mohd). Merge: Services/Analysis.
- **wow.tools.local (Missing)**: Restore Warcraft.NET wrapper (chunks MVER/MSHD/MHDR/MOHD); test PM4File.FromFile/AdtParser/WmoRootLoader. Merge: parpToolbox/Formats/WarcraftNETAdapter.
- **Subprojects (WoWToolbox.Core.v2/PM4Rebuilder/AnalysisTool)**: Template PM4FacesTool .csproj/preview; ref parpToolbox IChunkedFile/IObjExporter. Test --audit-only (PM4NextExporter); DSU for components (all assemblies). Merge: Core.v2/Foundation→Formats, Services/PM4→Services, PM4Rebuilder→Services/Export (MultiFormatBatchObjExporter), AnalysisTool→Services/Analysis.
- **Legacy (AlphaWDTReader/gillijimproject)**: Adopt patterns (ChunkedFile for AlphaWDT, async convert); integrate AlphaToLkConverter (--mode=convert-alpha). Test validation --compare. Merge: Services/Legacy.
- **New/Unified Tools**: PM4FacesTool template; preview for DSU/records; ref parpToolbox; IObjExporter custom (GLTF via PM4FacesTool). Test multi-format (--formats=all --audit-only); DSU PM4FacesTool for WMO/ADT/M2. CLI: System.CommandLine subcommands (export --assembly=dsu --formats=pm4,wmo).

## Troubleshooting (Multi-Format)
- **Build**: Missing wow.tools.local → Warcraft.NET NuGet; EF mismatch → 9.0.8 update. Subproject merge: Resolve refs (e.g., PM4Rebuilder→parpToolbox Services/Export).
- **Compatibility**: Invalid chunks (no MSPV/MCNK/MODF) → stub OBJ (Pm4BatchProcessor/AdtService/WmoRootLoader); validate PM4File/AdtParser/WdtFormat.IsValid/WmoFile counts. Specs: PM4Tool/docs (MSUR/MCNK/MOHD).
- **Performance**: Large files (PM4/ADT/WDT) → DSU O(n) PM4FacesTool, spans EmitTriMapped/ReadVector3; batch --batch avoid OOM, async exports profile. Alpha convert: --compare for diff.
- **Export**: Degenerate tris → AssembleAndWrite filter (skipped); parity --legacy-obj-parity PM4NextExporter. Missing MSCN/MODF/MDDF → empty CSVs/placements. M2 load fail → Warcraft.NET fallback.
- **CLI**: Unknown --assembly/formats → defaults (composite-hierarchy/pm4); invalid --height-scale → 1.0f. Alpha: Missing ADTs → log, --load-adts=dir.
- **Unification**: Dup code → shared IChunkedFile; test multi (--formats=all --audit-only); legacy errors → gillijimproject --report-mclq liquids.

For overviews, overview.md; configs, project-details.md; specs, formats-overview.md.