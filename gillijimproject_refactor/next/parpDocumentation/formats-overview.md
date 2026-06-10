# Formats Overview

Central index of WoW formats in parpToolbox (PM4/PD4 phasing, WMO/ADT/WDT world, M2 models, Alpha legacy), with specs and code paths for easy porting/merging. Interrelations: Hierarchical (WDT indexes ADTs → MODF WMO/MDDF M2; PM4/PD4 matches WMO via bbox/vert in WmoMatcher). Unified tool vision: Single CLI (dotnet tool --formats=pm4,adt,wmo,m2,wdt --mode=parse/export/analyze/convert-alpha), duplicate-free (ChunkedFile base, shared exporters ObjWriter/GLTF, batch Pm4BatchProcessor extended to Adt/Wmo).

## Formats (Parsing/Export Code Paths)

- **PM4**: Phasing/mesh/scene (MSLK linkages, MSUR surfaces GroupKey=0 M2/ non0 walkable, MSVI indices, MSVT verts offset-X/Y Z-unscaled, MSCN exterior). Spec: [pm4-specification.md](pm4-specification.md). Code: src/WoWToolbox/WoWToolbox.Core.v2/Foundation/PM4/PM4File.cs (FromFile:127, MSUR.Entries). Inter: WmoMatcher.Services/PM4/WmoMatcher.cs (bbox similarity); export PM4FacesTool/Program.cs: AssembleAndWrite (DSU CK24).
  
- **PD4**: Single-WMO PM4 superset (+MCRC=0 checksum). Spec: [pd4-specification.md](pd4-specification.md). Code: Core.v2/Foundation/PD4/PD4File.cs (inherits PM4File, MCRCChunk). Inter: Parsed as PM4 for WMO phasing; PD4FileTests.cs validates MSVT transforms.

- **WMO (V14/V17)**: Buildings (MOHD header nGroups/nPortals, MOTX textures, MOGP groups MOVT verts/MOVI faces/MOPY flags/textures, PORT portals, MLIQ liquids). Spec: [wmo-specification.md](wmo-specification.md). Code: Core.v2/Foundation/WMO/WmoRootLoader.cs, V14WmoFile.cs (MOHD.FromSpan, MOGP subchunks MOVT/MOVI/MOPY). Inter: ADT MODF placements; match PM4 via WmoMatcher; export WmoMeshExporter.cs (OBJ from groups).

- **ADT**: Tiles (MHDR offsets nMCNK=64, MCNK terrain MCVT 145x145 heights/MCNR normals/MCLQ liquids, MTEX/MMDX textures/models, MODF WMO 64B placements name_id/pos/rot/bbox/flags, MDDF M2 36B placements name_id/pos/rot/scale/flags). Spec: [adt-specification.md](adt-specification.md). Code: Core/ADT/AdtService.cs/AdtParser (manual chunk scan REVM/MHDR, br.ReadUInt32 flags/offsets, ReadNullTerminatedString MTEX/MMDX, MODF/MDDF arrays). Inter: WDT MAIN loads; prefab AdtFlatPlateBuilder (129x129 flat verts); Alpha via gillijimproject AdtAlpha.ToAdtLk.

- **WDT**: Map index (MAIN 64x64 flags/offsets, MDNM/MONM M2/WMO names). Spec: [wdt-specification.md](wdt-specification.md). Code: Core.v2/Foundation/WDT/WdtFormat.cs (MAIN 8192B grid), AlphaWDTReader (ADTPreFabTool: MPHD offsets/sparse MAIN flags&1 present). Inter: Indexes ADTs for loading; Alpha→LK convert gillijimproject WdtAlpha.ToWdt (embed ADTs, remap Mcrf.UpdateIndicesForLk).

- **M2**: Models (MD21 header nVertices/nSkins, verts XYZ, skins indices/bone_weights/textures, bones hierarchy pos/rot/scale, anim tracks). Spec: [m2-specification.md](m2-specification.md). Code: Helpers/M2ModelHelper.cs (Warcraft.NET.Files.M2.M2File.Load, MD21.Vertices/BoundingTriangles, M2Mesh for export). Inter: ADT MDDF placements (name_id→MMDX), WMO doodads MODD.

## Interrelations and Workflows (Code for Unified Tool)
Formats form world graph: WDT MAIN→ADTs (AdtParser)→MCNK terrain (AdtFlatPlateBuilder), MODF WMO (WmoRootLoader), MDDF M2 (M2ModelHelper.LoadMeshFromFile); PM4/PD4 match WMO (WmoMatcher bbox/vert Pm4BatchProcessor.Process). Legacy: AlphaWDTReader/gillijimproject-csharp convert (WdtAlpha.ToWdt: parse MPHD/MAIN, discover ADTs AdtAlpha.ToAdtLk remap indices/AreaIDs, embed offsets via offset_builder; liquids mclq_to_mh2o connected_components).

- **World Loading**: WdtFormat(path)→MAIN present tiles→AdtParser.LoadAdt (MCNK/AdtFlatPlateBuilder terrain, MODF→WmoRootLoader/V14WmoFile, MDDF→M2ModelHelper).
- **PM4 Integration**: Pm4BatchProcessor.Process (extract buildings IBuildingExtractionService, match WMO IWmoMatcher), PM4FacesTool ExportCompositeInstances (CK24-DSU MSUR groups).
- **Legacy Refactoring**: AlphaWDTReader.Read (MPHD/MAIN/MDNM/MONM)→gillijimproject WdtAlpha.ToWdt (convert ADTs McnkAlpha.ToMcnkLk index remap Mcrf.UpdateIndicesForLk, MH2O mclq_to_mh2o, AreaIDs areatable_mapper; embed payloads offset_builder).
- **Multi-Format Workflow Example** (Unified CLI): dotnet tool --input=zone.wdt --mode=export --formats=adt,wmo,m2,pm4 → WdtFormat→AdtParser extract MODF/MDDF→WmoRootLoader/M2ModelHelper→match PM4 (Pm4BatchProcessor/WmoMatcher)→merged GLB (WmoMeshExporter + PM4FacesTool AssembleAndWrite).

For implementation (code porting/merging), see project-details.md; usage/best practices (unified CLI/extensions), usage-guidelines.md.