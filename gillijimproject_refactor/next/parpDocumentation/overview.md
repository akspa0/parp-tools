# parpToolbox Solution Overview

The parpToolbox solution is a unified .NET 9 ecosystem for processing all major World of Warcraft (WoW) file formats: PM4/PD4 (phasing/mesh/scene), WMO (buildings V14/V17), ADT (tiles), WDT (maps), and M2 (models). It features shared utilities in parpToolbox for parsing/export, specialized CLI tools for analysis/batch processing, and legacy support via AlphaWDTReader/gillijimproject-csharp (Alpha→LK conversion). Modular architecture enables easy merging into a single definitive tool: core parsing via ChunkedFile base (PM4File, ADTFile, WDTFormat, etc.), services like Pm4BatchProcessor/WmoMatcher for extraction/matching, and exporters using DSU/grouping for OBJ/GLTF. All leverage SDK-style .csproj, implicit usings/nullable, net9.0, preview C# (records/patterns).

Structure: "src" for tools/subprojects, "lib" for deps; parpToolbox central with EF Core (metadata), System.CommandLine (CLI), Warcraft.NET (chunks). Unification path: Abstract parsers (e.g., IChunkedFile), shared exporters (ObjWriter for PM4/WMO/M2), batch via Pm4BatchProcessor extended to ADT/WDT.

## Introduction
parpToolbox unifies PM4/PD4 (phasing via MSLK/MSUR/MSCN chunks), WMO/ADT (chunked parsing MOHD/MHDR/MODF/MDDF), WDT (MAIN indexing), M2 (verts/skins via Warcraft.NET), and legacy Alpha formats. Code: Modular (ChunkedFile base for all), extensible (IWmoMatcher for PM4-WMO, AdtParser for tiles), duplication-free (shared CoordinateTransforms, OutputManager).

## Solution Projects and Subprojects

parpToolbox.sln includes core library + tools; subprojects (WoWToolbox.Core.v2, PM4Rebuilder) as building blocks. For unification: Merge subprojects into parpToolbox (e.g., PM4Rebuilder exporters), resolve deps (wow.tools.local → Warcraft.NET NuGet).

1. **parpToolbox** (src/parpToolbox/parpToolbox.csproj) - Core library for all formats.

1. **parpToolbox** (src/parpToolbox/parpToolbox.csproj) - Unified core for PM4/PD4/WMO/ADT/WDT/M2.
   - Type: Library
   - Purpose: Shared parsing (ChunkedFile for all: PM4File, ADTFile, WDTFormat, M2File via Warcraft.NET), services (Pm4BatchProcessor extended to AdtService/WmoRootLoader), exporters (ObjWriter for OBJ/GLTF, shared CoordinateTransforms). EF Core metadata, System.CommandLine CLI. Key: [`PM4File`](src/WoWToolbox/WoWToolbox.Core.v2/Foundation/PM4/PM4File.cs) (MSLK/MSUR/etc.), [`AdtParser`](src/WoWToolbox/WoWToolbox.Core/ADT/AdtService.cs) (MHDR/MCNK/MODF/MDDF), [`WmoRootLoader`](src/WoWToolbox/WoWToolbox.Core.v2/Foundation/WMO/WmoRootLoader.cs), `ProjectOutput`.
   - Features: Format-agnostic parsing (`FromFile`), batch (Process for PM4/ADT), matching (IWmoMatcher for PM4-WMO), export (LegacyObjExporter for stubs). Unification: Extend to single CLI command handling all formats.

2. **wow.tools.local** (src/lib/wow.tools.local/wow.tools.local.csproj) - WoW format wrapper (missing).
   - Type: Library
   - Purpose: wow.tools adaptation for PM4/ADT/WMO/M2; chunk readers (Warcraft.NET integration). Status: Missing; replace with NuGet (Warcraft.NET) for unification.
   - Merge: Integrate into parpToolbox as FormatParsers namespace.

3. **PM4FacesTool** (src/PM4FacesTool/PM4FacesTool.csproj) - **Priority: High** PM4 exporter blueprint.
   - Type: Console App
   - Purpose: Advanced PM4 mesh export (DSU assembly, grouping strategies, tile batching, OBJ/GLTF, MSCN sidecar, diagnostics). 2175-line Program.cs; refs parpToolbox (Pm4GlobalTileLoader/Pm4Scene). Extendable to WMO/M2 export.
   - Features: ExportCompositeInstances (CK24-DSU), AssembleAndWrite (dedup/tris), transforms (ApplyGlobalTransform), CSVs/JSON. Unification: Generalize to MultiFormatExporter (PM4/WMO/ADT).

4. **parpDataHarvester** (src/parpDataHarvester/parpDataHarvester.csproj) - Batch asset collector.
   - Type: Console App
   - Purpose: Extracts assets from PM4/PD4/ADT/WMO (directory scan, stats MSPV/MSVT, CSV/JSON). Uses Pm4BatchProcessor; extend to AdtService.
   - Features: Batch ingestion, stub handling. Unification: Merge into unified CLI (--mode=harvest --formats=pm4,adt).

5. **PM4MscnAnalyzer** (src/PM4MscnAnalyzer/PM4MscnAnalyzer.csproj) - MSCN inspector.
   - Type: Console App
   - Purpose: Analyzes PM4/PD4 MSCN (exterior verts); preview C#. Integrate with WMO bbox comparison.
   - Features: CSV dumps (CsvChunkDumper). Unification: Add to AnalyzerTool suite.

6. **PM4NextExporter** (src/PM4NextExporter/PM4NextExporter.csproj) - **Priority: High** CLI exporter.
   - Type: Console App
   - Purpose: PM4 export with 30+ options (assembly strategies, batch/audit, CSV, per-tile/MSCN OBJ). Thin wrapper; README usage. Excludes legacy; extend to WDT/ADT batch.
   - Features: --assembly (DSU/MSLK), --export-tiles, transforms (--legacy-obj-parity). Unification: Core for merged tool CLI (dotnet run --formats=pm4,wmo --export=obj).

**Legacy/External Subprojects** (for merging):
- **WoWToolbox.Core.v2/Foundation**: Chunks (MSHDChunk, MSLK, MSURChunk, MSCNChunk, MSVIChunk, MSPVChunk, MSPIChunk, MVER for PM4; extend to ADT MHDR/MCNK, WMO MOHD/MOGP).
- **WoWToolbox.Core.v2/Services/PM4**: Pm4BatchProcessor (extraction/matching/export); analyzers (MslkHierarchyAnalyzer, WmoMatcher, RenderMeshBuilder).
- **WoWToolbox.Core.v2/Foundation/WMO**: WmoRootLoader, WmoMeshExporter, V14WmoFile/V17 converters.
- **PM4Rebuilder**: Pm4BatchObjExporter (unified OBJ, ExtractSurfaceGeometry fan tris); BuildingLevelExporter, DirectPm4Exporter, MscnAnalyzer.
- **WoWToolbox.AnalysisTool**: MprrHypothesisAnalyzer, MscnMeshComparisonAnalyzer, MslkAnalyzer, PM4CorrelationUtility.
- **Other**: FileDumper (DTOs Pm4FileDto/AdtObj0Dto), MSCNExplorer (Pm4MeshExtractor), PM4WmoMatcher, SpatialAnalyzer.
- **External (Integrate)**: AlphaWDTReader (ADTPreFabTool: Alpha WDT parse/convert), gillijimproject-csharp (Alpha→LK: WdtAlpha.ToWdt, AdtAlpha.ToAdtLk index remap).

Merge plan: Consolidate into parpToolbox (e.g., Rebuilder→Services/Export), resolve wow.tools.local (Warcraft.NET), add unified CLI (handle --input=*.{pm4,pd4,adt,wdt,wmo,m2}).

## Prioritization and Unification
- **Primary**: PM4FacesTool/PM4NextExporter (PM4 blueprint: DSU/grouping, CLI strategies); extend to WMO/ADT/M2 (e.g., WmoMeshExporter + ObjWriter).
- **Core**: parpToolbox (essential; update for all formats: add ADTFile, WDTFormat, M2 wrapper).
- **Secondary/Legacy**: parpDataHarvester (batch all formats), PM4MscnAnalyzer (generalize to chunk analyzer); integrate AlphaWDTReader/gillijimproject (Alpha conversion service).
- **Merge Strategy**: Duplicate-free (shared ChunkedFile/IChunkedFile); single CLI (System.CommandLine --formats=pm4,adt --mode=export,analyze); test suite (WoWToolbox.Tests for all).

## Build and Configuration
- Framework: .NET 9.0; preview C# for all (records, spans).
- Configs: Debug/Release (AnyCPU/x64/x86).
- Deps: parpToolbox refs Warcraft.NET (chunks), EF Core 9.0.8 (SQLite metadata), System.CommandLine beta (CLI). Unify: Single .sln with all subprojects.

## Architecture Overview (Unified)
- **Parsing Layer**: ChunkedFile base loads all (PM4File: MSUR/etc.; ADTFile: MHDR/MCNK/MODF; WDTFormat: MAIN/MDNM; M2File: MD21/Warcraft.NET; WmoFile: MOHD/MOGP).
- **Processing Layer**: Batch services (Pm4BatchProcessor for PM4/ADT/WDT: extract buildings/WMO/M2 via IBuildingExtractionService/IWmoMatcher/AdtService; match PM4-WMO).
- **Export Layer**: DSU/grouping (PM4FacesTool patterns) for all (AssembleAndWrite: verts/tris; ObjWriter/GLTF for PM4/WMO/M2/ADT; shared transforms flip/rotate/project).
- **Data Flow**: Input multi-format (PM4/ADT/WMO/M2/WDT) → Parse/LoadRegion → Process (match/extract) → Assemble/Transform → Output (timestamped: objects/tiles/CSVs/JSON; unified via ProjectOutput).
- **Legacy**: AlphaWDTReader/gillijimproject integrated as services (AlphaToLkConverter for WDT/ADT).

## Building Block Subprojects (For Merging)
Subprojects provide format-specific code; merge into parpToolbox for unification (e.g., Foundation→Formats, Services→Services).

- **WoWToolbox.Core.v2/Foundation**: Core infrastructure. PM4 chunks ([MSHDChunk](src/WoWToolbox/WoWToolbox.Core.v2/Foundation/PM4/Chunks/MSHDChunk.cs), [MSLK](Chunks/MSLK.cs: 20-byte linkages), [MSURChunk](Chunks/MSURChunk.cs: 32-byte surfaces), [MSCNChunk](Chunks/MSCNChunk.cs: verts), [MSVIChunk](Chunks/MSVIChunk.cs: indices), [MSPVChunk](Chunks/MSPVChunk.cs: path verts), [MSPIChunk](Chunks/MSPIChunk.cs: path indices), [MVER](Chunks/MVER.cs)). Models ([BatchProcessResult](Models/PM4/BatchProcessResult.cs), [BuildingFragment](Models/PM4/BuildingFragment.cs)). Utilities ([CoordinateTransforms](Foundation/Transforms/CoordinateTransforms.cs: PM4/PD4/WMO), [OutputManager](Foundation/Utilities/OutputManager.cs)). Extend: ADT/WMO chunks (MHDR/MOHD).

- **WoWToolbox.Core.v2/Services/PM4**: PM4 processing. [Pm4BatchProcessor](Services/PM4/Pm4BatchProcessor.cs: extraction/matching/OBJ), exporters ([Pm4ObjExporter](Services/PM4/Pm4ObjExporter.cs), [CsvChunkDumper](Services/Export/CsvChunkDumper.cs)), analyzers ([MslkHierarchyAnalyzer](Services/PM4/MslkHierarchyAnalyzer.cs), [WmoMatcher](Services/PM4/WmoMatcher.cs: bbox/vert similarity), [RenderMeshBuilder](Services/PM4/RenderMeshBuilder.cs)), interfaces ([IPm4FileLoader](Services/PM4/IPm4FileLoader.cs), [IRenderMeshBuilder](Services/PM4/IRenderMeshBuilder.cs)). Unify: Extend to AdtBatchProcessor/WmoProcessor.

- **WoWToolbox.Core.v2/Foundation/WMO**: WMO support. [WmoRootLoader](Foundation/WMO/WmoRootLoader.cs), [WmoMeshExporter](Foundation/WMO/WmoMeshExporter.cs: OBJ from groups), converters ([V14WmoFile](Foundation/WMO/V14/V14WmoFile.cs: MOHD/MOGP/MOVT/MOVI)).

- **PM4Rebuilder**: Batch OBJ. [Pm4BatchObjExporter](PM4Rebuilder/Pm4BatchObjExporter.cs:613 lines, UnifiedMapScene:389 offsets, ExtractSurfaceGeometry fan tris from MSUR/MSVI). Others: [BuildingLevelExporter](PM4Rebuilder/BuildingLevelExporter.cs), [DirectPm4Exporter](PM4Rebuilder/DirectPm4Exporter.cs), [MscnAnalyzer](PM4Rebuilder/MscnAnalyzer.cs). Merge: Into Services/Export as MultiFormatBatchExporter.

- **WoWToolbox.AnalysisTool**: Analyzers. [MprrHypothesisAnalyzer](WoWToolbox/WoWToolbox.AnalysisTool/MprrHypothesisAnalyzer.cs), [MscnMeshComparisonAnalyzer](WoWToolbox/WoWToolbox.AnalysisTool/MscnMeshComparisonAnalyzer.cs), [MslkAnalyzer](WoWToolbox/WoWToolbox.AnalysisTool/MslkAnalyzer.cs), [PM4CorrelationUtility](WoWToolbox/WoWToolbox.AnalysisTool/PM4CorrelationUtility.cs). Unify: General ChunkAnalyzer for all formats.

- **Other Tools**: FileDumper (DTOs [Pm4FileDto](WoWToolbox/WoWToolbox.FileDumper/DTOs/Pm4FileDto.cs)/AdtObj0Dto), MSCNExplorer ([Pm4MeshExtractor](WoWToolbox/WoWToolbox.MSCNExplorer/Pm4MeshExtractor.cs)), PM4WmoMatcher, SpatialAnalyzer. Merge: Into unified AnalyzerTool.

- **Legacy (Integrate)**: AlphaWDTReader (ADTPreFabTool: MPHD/MAIN parse, AlphaToLkConverter), gillijimproject-csharp (WdtAlpha.ToWdt: embed ADTs, index remap Mcrf.UpdateIndicesForLk). Add as Services/Legacy.

These enable full workflows (e.g., PM4FacesTool + AdtService for PM4- ADT-WMO-M2). For details, see project-details.md; usage, usage-guidelines.md; specs, formats-overview.md.