# Project Details

Per-project details (.csproj configs, deps, code analysis, functionality) for parpToolbox solution, subprojects, and legacy tools. All target .NET 9.0, SDK-style .csproj (implicit usings/nullable enable). Unified focus: WoW formats (PM4/PD4 phasing, WMO/ADT tiles/buildings, WDT maps, M2 models, Alpha legacy). Enables merging into definitive tool: Duplicate-free (ChunkedFile base, shared exporters like ObjWriter), efficient (DSU/batch for all), comprehensive (CLI for parse/export/analyze/convert). Priority: PM4FacesTool/PM4NextExporter (PM4 blueprint, extend to multi-format).

## parpToolbox (src/parpToolbox/parpToolbox.csproj) - Unified Core Library
- **Type**: Library (shared for all formats/tools)
- **Output**: DLL
- **Target Framework**: net9.0
- **C# Version**: Default (latest)
- **Key Properties**: OutputType=Library, ImplicitUsings/Nullable=enable
- **Dependencies**:
  - ProjectRef: wow.tools.local (missing; replace with Warcraft.NET NuGet for chunks)
  - Packages: EF Core.Sqlite/Design 9.0.8 (SQLite metadata), Microsoft.Data.Sqlite 9.0.8, System.CommandLine 2.0.0-beta4.22272.1 (CLI)
- **Exclusions**: Services/PM4/GltfExporter.cs (deprecated; use PM4FacesTool GLTF)
- **Purpose**: Foundation for WoW formats: Parsing (ChunkedFile for PM4/PD4/WMO/ADT/WDT/M2), services (batch extraction/matching/export), utilities (ProjectOutput, CoordinateTransforms). Integrates Warcraft.NET (chunks), EF (metadata). Unification: Central hub post-merge (all subprojects here); extend BatchProcessor to Adt/Wmo/M2.
  - **Key Components**:
    - Foundation/PM4/PM4File.cs: ChunkedFile impl for PM4/PD4 (MVER/MSHD/MSLK/MSUR/MSCN/MSVI/MSPV/MSPI/MPRL/MPRR/MDBH/MDOS/MDSF); FromFile:127 loads binary.
    - Services/PM4/Pm4BatchProcessor.cs: IPm4BatchProcessor for batch (IBuildingExtractionService.ExtractBuildings, IWmoMatcher.Match, LegacyObjExporter.ExportAsync); BatchProcessResult with stats (MSPV/MSVT/MSVI counts). Extend: AdtBatchProcessor for tiles.
    - Foundation/WMO/WmoRootLoader.cs & V14WmoFile.cs: MOHD/MOTX/MOGP/MOVT/MOVI/MOPY/PORT/MLIQ parsing/conversion.
    - Core/ADT/AdtService.cs: AdtParser for MHDR/MCNK/MTEX/MMDX/MODF/MDDF/MWMO; placements (ModelPlacement for M2).
    - Core.v2/Foundation/WDT/WdtFormat.cs: MAIN 64x64, MDNM/MONM names.
    - Helpers/M2ModelHelper.cs: Warcraft.NET M2 load (MD21 verts/skins/BoundingTriangles), M2Mesh for export.
    - Infrastructure/ProjectOutput.cs: Timestamped outputs (project_output/<format>/).
- **API Reference**:
  - Parsing: PM4File.FromFile(path) → MSUR.Entries; ADTFile.FromFile → MCNK/MODF; WdtFormat(string path) → MAIN.
  - Batch: Pm4BatchProcessor.Process(path) → fragments/matches; extend for AdtService.ProcessAdts(wdtPath).
  - Export: LegacyObjExporter.ExportAsync(file, path); M2ModelHelper.LoadMeshFromFile(m2Path, pos/rot/scale) → M2Mesh (verts/tris).
  - Unification: IChunkedFile for all (FromFile, Serialize); MultiFormatBatchProcessor (PM4/ADT/WMO/M2/WDT).

## wow.tools.local (src/lib/wow.tools.local/wow.tools.local.csproj) - Format Wrapper (Missing)
- **Type**: Library
- **Output**: DLL
- **Target Framework**: net9.0 (inferred)
- **Status**: Missing; critical dep for parpToolbox.
- **Dependencies**: Unknown
- **Purpose**: wow.tools adaptation for PM4/ADT/WMO/M2/WDT parsing (chunk readers via Warcraft.NET). Integrated into PM4File/AdtParser/WmoRootLoader. Restore: Git submodule/NuGet (Warcraft.NET.Files for M2/ADT/WMO).
- **Unification**: Merge into parpToolbox/Formats namespace (e.g., WarcraftNETAdapter); eliminate external dep for standalone tool.

## PM4FacesTool (src/PM4FacesTool/PM4FacesTool.csproj) - **Priority: High** (PM4 Exporter Blueprint)
- **Type**: Console App
- **Output**: Exe
- **Target Framework**: net9.0
- **C# Version**: Default
- **Key Properties**: OutputType=Exe, ImplicitUsings/Nullable=enable
- **Dependencies**: ProjectRef: parpToolbox (PM4 parsing/ProjectOutput)
- **Exclusions**: None
- **Purpose**: Advanced PM4 mesh export/analysis ("latest/greatest"): DSU components (vertex-shared surfaces), grouping (composite/type-attr/surface/groupkey/render-mesh), tile batching, OBJ/GLTF/GLB, MSCN sidecar, plane snapping, diagnostics (CSVs/height-fit). WoW transforms (flipX/rot/trans/local project). 2175-line Program.cs:1 standalone CLI; uses Pm4GlobalTileLoader/Pm4Scene. Extend to WMO/ADT/M2 (generalize AssembleAndWrite).
  - **Key Features**:
    - Grouping: ExportCompositeInstances (CK24-DSU), ExportTypeInstances (GroupKey+CK24); AssembleAndWrite (TryMap dedup, EmitTriMapped tris).
    - Exports: OBJ/GLTF per-object/tile/merged; MSCN sidecar (ApplyMscnPreTransform: basis/rotZ/flip).
    - Transforms: ApplyGlobalTransform (flip/rot/trans), snap-to-plane (--height-scale).
    - Diagnostics: surface/tile/walkable/m2_coverage.csv, objects_index.json, msur_height_overview.csv, msur_plane_fit.csv.
    - Filtering: --ck-min-tris, --ck-monolithic, --ck-merge-components.
  - **Entry Point**: Main:48 (ParseArgs→Options:16, LoadRegion/ToStandardScene, ProcessOne, Export[Strategy], AssembleAndWrite).
  - **Data Structures**: Pm4Scene (Verts/Indices/Surfaces/TileIndexOffsetByTileId); MsurChunk.Entry (GroupKey/CompositeKey/MsviFirst/IndexCount/Height/NxNyNz); ObjectIndexEntry:2014/TileIndexEntry:2033 (JSON).
  - **DSU**: Internal DSU:1062 (union-find on shared verts).
- **API Reference**:
  - Parsing: Pm4GlobalTileLoader.LoadRegion(dir, pattern, remap)→Pm4Scene; ToStandardScene.
  - Export: AssembleAndWrite (TryMap/EmitTriMapped, ApplyProjectLocal/GlobalTransform, ObjWriter.Write/GltfWriter.WriteGltf).
  - Grouping: ExportCompositeInstances (CK24-DSU), ExportTypeInstances (GroupKey+CK24).
  - Utils: CK24(key) (24-bit extract), DominantTileIdFor (bucketing), SanitizeFileName/EscapeCsv.
  - Transforms: ApplyMscnPreTransform (canonical/rotZ/flip), EmitHeightFitReport (MSUR residuals).
- **Unification**: Blueprint for MultiFormatExporter (extend to WMO groups/M2 skins/ADT terrain via shared DSU/ObjWriter); integrate with PM4NextExporter CLI (--format=pm4,wmo).

## parpDataHarvester (src/parpDataHarvester/parpDataHarvester.csproj) - Batch Asset Collector
- **Type**: Console App
- **Output**: Exe
- **Target Framework**: net9.0
- **C# Version**: latest
- **Key Properties**: OutputType=Exe, ImplicitUsings/Nullable=enable, RootNamespace=ParpDataHarvester, AssemblyName=parpDataHarvester
- **Dependencies**: ProjectRef: parpToolbox
- **Exclusions**: None
- **Purpose**: Extracts/collects assets from PM4/PD4/ADT/WMO/M2/WDT (batch ingestion, stats MSPV/MSVT/MCNK/MOHD, CSV/JSON). Uses Pm4BatchProcessor; extend to AdtService/WmoRootLoader/M2ModelHelper. Explicit naming for scripts; latest C# efficiency.
  - **Integration**: Pm4BatchProcessor.Process for fragments/matches/OBJ stubs; directory scan, missing chunks→stubs. Unification: Generalize to MultiFormatHarvester (--formats=pm4,adt,wmo --output=csv,json).
- **API Reference**: Program.Main orchestrates Process on dirs; stats (MSPV/MSVT/MSVI/MCNK counts). Add: AdtService.GetPlacements (MODF/MDDF), WdtFormat.GetPresentTiles.

## PM4MscnAnalyzer (src/PM4MscnAnalyzer/PM4MscnAnalyzer.csproj) - Chunk Inspector
- **Type**: Console App
- **Output**: Exe
- **Target Framework**: net9.0
- **C# Version**: preview
- **Key Properties**: OutputType=Exe, ImplicitUsings/Nullable=enable
- **Dependencies**: ProjectRef: parpToolbox
- **Exclusions**: None
- **Purpose**: Analyzes PM4/PD4 MSCN (exterior verts; preview C# patterns/records). Inspects scenes, correlates with MSUR/WMO bbox (MscnMeshComparisonAnalyzer). Extend to general chunks (MCNK/MOHD).
  - **Key Features**: CSV dumps (CsvChunkDumper), MSUR correlation, WMO validation. Unification: Merge into AnalyzerTool (--chunk=mscn,mcnk --format=pm4,adt,wmo).
- **API Reference**: MSCNChunk in PM4File; MscnMeshComparisonAnalyzer from AnalysisTool. Add: AdtService.GetMscnEquiv (terrain bbox).

## PM4NextExporter (src/PM4NextExporter/PM4NextExporter.csproj) - **Priority: High** (CLI Exporter Blueprint)
- **Type**: Console App
- **Output**: Exe
- **Target Framework**: net9.0
- **C# Version**: preview
- **Key Properties**: OutputType=Exe, ImplicitUsings/Nullable=enable
- **Dependencies**: ProjectRef: parpToolbox
- **Exclusions**: Assembly/CompositeHierarchyInstanceAssembler.cs (deprecated; use DSU from PM4FacesTool)
- **Purpose**: CLI PM4 exporter (30+ options: assembly strategies, batch/audit, CSV, per-tile/MSCN OBJ). Thin wrapper (no Program.cs; System.CommandLine inferred); README usage. Focus: Production (legacy parity, cross-tile). Extend to unified exporter (--formats=pm4,wmo,adt,m2 --assembly=dsu).
  - **Key Features** (README):
    - Assembly: composite-hierarchy (default DSU cross-tile), msur-indexcount, mslk-parent/instance (MslkLinkGraphBuilder), parent16/container-hierarchy.
    - Grouping: parent16/object, surface/flags/type, tile; CK-split-by-type.
    - Exports: OBJ/MTL (--export-tiles), MSCN (--export-mscn-obj); CSV (--csv-diagnostics).
    - Options: --include-adjacent (cross-tile), --legacy-obj-parity, --project-local, --mslk-parent-min-tris N, --correlate keyA:keyB.
    - Output: project_output/<timestamp>/run.log.
  - **Integration**: PM4File/Pm4BatchProcessor/custom exporters (Pm4ObjExporter); System.CommandLine CLI.
- **API Reference** (README/core):
  - CLI: pm4next-export input [options] → Pm4BatchProcessor.Process/custom assembly (MslkHierarchyAnalyzer/WmoMatcher).
  - Strategies: composite-hierarchy (DSU), mslk-parent (linkage).
  - Diagnostics: --audit-only (validation), --csv-out dir (reports).
- **Unification**: Core for merged CLI (dotnet tool --formats=pm4,wmo,adt --export=obj --batch); integrate PM4FacesTool strategies, extend to WDT/ADT batch (--input=wdt --load-adts).

## Subprojects and Legacy Tools (For Merging)
- **WoWToolbox.Core.v2/Foundation**: PM4 infrastructure (MSHDChunk/Foundation/PM4/Chunks/MSHDChunk.cs, MSLK/Chunks/MSLK.cs:20-byte, MSURChunk/Chunks/MSURChunk.cs:32-byte, MSCNChunk/Chunks/MSCNChunk.cs, MSVIChunk/Chunks/MSVIChunk.cs, MSPVChunk/Chunks/MSPVChunk.cs, MSPIChunk/Chunks/MSPIChunk.cs, MVER/Chunks/MVER.cs). Models (BatchProcessResult/Models/PM4/BatchProcessResult.cs, BuildingFragment/Models/PM4/BuildingFragment.cs, MDBHChunk/Chunks/MDBHChunk.cs). Utils (CoordinateTransforms/Foundation/Transforms/CoordinateTransforms.cs, OutputManager/Foundation/Utilities/OutputManager.cs). Merge: Into Formats/PM4; extend ADT/WMO chunks.
- **WoWToolbox.Core.v2/Services/PM4**: Pm4BatchProcessor/Services/PM4/Pm4BatchProcessor.cs (extraction/WMO match/OBJ), Pm4ObjExporter/Services/PM4/Pm4ObjExporter.cs, CsvChunkDumper/Services/Export/CsvChunkDumper.cs, MslkHierarchyAnalyzer/Services/PM4/MslkHierarchyAnalyzer.cs, WmoMatcher/Services/PM4/WmoMatcher.cs (bbox/vert), RenderMeshBuilder/Services/PM4/RenderMeshBuilder.cs, IPm4FileLoader/Services/PM4/IPm4FileLoader.cs, IRenderMeshBuilder/Services/PM4/IRenderMeshBuilder.cs. Merge: Services/PM4 → Services (extend to Adt/Wmo).
- **WoWToolbox.Core.v2/Foundation/WMO**: WmoRootLoader/Foundation/WMO/WmoRootLoader.cs, WmoMeshExporter/Foundation/WMO/WmoMeshExporter.cs, V14WmoFile/Foundation/WMO/V14/V14WmoFile.cs (MOHD/MOGP). Merge: Formats/WMO.
- **PM4Rebuilder**: Pm4BatchObjExporter/PM4Rebuilder/Pm4BatchObjExporter.cs:613 (UnifiedMapScene:389, ExtractSurfaceGeometry fan MSUR/MSVI), BuildingLevelExporter/PM4Rebuilder/BuildingLevelExporter.cs, DirectPm4Exporter/PM4Rebuilder/DirectPm4Exporter.cs, MscnAnalyzer/PM4Rebuilder/MscnAnalyzer.cs. Merge: Services/Export as MultiFormatBatchObjExporter.
- **WoWToolbox.AnalysisTool**: MprrHypothesisAnalyzer/WoWToolbox/WoWToolbox.AnalysisTool/MprrHypothesisAnalyzer.cs, MscnMeshComparisonAnalyzer/WoWToolbox/WoWToolbox.AnalysisTool/MscnMeshComparisonAnalyzer.cs, MslkAnalyzer/WoWToolbox/WoWToolbox.AnalysisTool/MslkAnalyzer.cs, PM4CorrelationUtility/WoWToolbox/WoWToolbox.AnalysisTool/PM4CorrelationUtility.cs. Merge: Services/Analysis.
- **Other**: FileDumper DTOs (Pm4FileDto/WoWToolbox/WoWToolbox.FileDumper/DTOs/Pm4FileDto.cs, AdtObj0Dto), MSCNExplorer (Pm4MeshExtractor/WoWToolbox/WoWToolbox.MSCNExplorer/Pm4MeshExtractor.cs), PM4WmoMatcher, SpatialAnalyzer. Merge: Tools/Analyzer.
- **Legacy (Integrate)**: AlphaWDTReader (ADTPreFabTool: MPHD/MAIN/AlphaToLkConverter for WDT/ADT), gillijimproject-csharp (WdtAlpha.ToWdt: embed/ remap Mcrf.UpdateIndicesForLk, AdtAlpha.ToAdtLk: index/AreaID). Merge: Services/Legacy/AlphaConverter (CLI --mode=alpha-to-lk --input=wdt).

## Interdependencies and Merge/Build Order
- Deps: Tools → parpToolbox; parpToolbox → wow.tools.local (replace Warcraft.NET).
- Build Order: Warcraft.NET NuGet → parpToolbox (merge subprojects) → tools.
- Merge Plan: Duplicate-free (IChunkedFile for parsers, shared exporters); single .sln/CLI (dotnet tool --formats=all --mode=export/analyze/convert); test all (WoWToolbox.Tests + legacy validation --compare).
- Configs: Debug/Release (AnyCPU/x64/x86).

For usage (priority + unification examples), see usage-guidelines.md; formats, formats-overview.md.