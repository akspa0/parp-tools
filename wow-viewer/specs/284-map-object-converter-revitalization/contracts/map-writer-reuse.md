# Contract: Reuse Viewer New-Map Writers

**Operator addition, 2026-10-21. Planning only.** Refresh conversion output around the current
viewer/library new-map writers, rather than the obsolete Converter project or a second serializer.

## Current code-backed creation path

```text
Workbench/Services/NewMapCreatorService.CreateAndLoadMap
  → Core.IO/Terrain/TemplatedTerrainGenerator.GenerateMap
  → LkAdtData/LkMcnkData
  → Core.IO/Maps/LkAdtWriter.Write
  → Core.IO/Maps/LkWdtWriter.Write
  → Core.IO/Maps/WdlWriter.ExtractTileHeightsFromLk + Write
```

The generator supplies synthetic new-map content. Conversion uses native v65 records to populate
target documents; it must not replace imported terrain with a fresh template. Current harvest
`relief-to-map`, Inspect's blank/templated-map generation and `DatToLkAdtFolderExporter` also call
these libraries. Those are discovery/orchestration references, not parallel conversion authorities.

## Shared materialization boundary

`Core.IO/Conversion/Maps/V65TerrainExportAdapter` maps validated v65 records into target documents.
`Core.IO/Conversion/Maps/MapTargetMaterializer` owns writer selection and ADT/WDT/WDL emission through
the existing canonical writers. `MapConversionService` owns scope, dependencies, placement IDs and
publication. CLI and Editor use the same services. No renderer behavior or new god-class member.

Reuse `LkAdtWriter` for fresh LK v18 content, `LkWdtWriter` for its supported map contract and
`WdlWriter` with explicit native-versus-derived WDL policy. Alpha uses the existing public
`AlphaWdtWriter` input boundary after target conversion; it remains frozen. Current DAT/split-object
writers are distinct capabilities, not evidence of a complete modern split-map writer.

If consolidating New Map orchestration onto this facade, make only a behavior-preserving delegation
to the same writer defaults. It keeps the existing synthetic document generation and viewer load
callback. Do not require a UI redesign or claim its current map-load behavior was witnessed.

## Capability matrix before acceptance

For every retained v65 field record: source field → target field/DTO → canonical writer → actual
readback → preserved/transformed/reduced/blocked. Record named writer/version and missing fields.
Present source and post-write MCAL/MCLY opacity reports and an explicit layer mapping.

Known source-inspected limits requiring explicit handling:

- `LkAdtWriter` currently emits zero MTXF, empty MCSE and zero MDDF flags; the DTO cannot express
  all retained native header/material/unknown metadata. Existing defaults are not preservation.
- LK heights are stored relative to `BaseHeight`; normalize native absolute Z deliberately.
- `LkWdtWriter` emits empty global MWMO/MODF and does not preserve every source WDT field; a
  `HasMaid` flag alone does not write MAID content.
- `WdlWriter` derives rounded/clamped heights when extracting from LK and writes zero MAHO holes.
  A derived WDL must not be labelled preserved source WDL.
- `AdtTerrainWriter.ApplyHeightmap` clones existing ADT bytes and patches MCVT/MCNR only. It is
  an explicit same-family source-template height-patch option, never a new-map/metadata serializer.

Use bounded opt-in target-document/writer field support where this refresh requires it, preserving
existing new-map defaults and proving output readback. Do not patch working readers, force raw
unknown chunks into a different target version, or reopen the frozen Alpha writer. If a field
cannot be represented, report a reviewed reduction or block preserve-features output. Retain the
source evidence in v65 regardless of target limitations.

## Writer verification

Verify actual ADT/WDT/WDL output from the same canonical writer path for a small current New Map
control and a v65-native conversion selection. Check tile coordinates, native heights/normals,
holes/liquids, metadata and placement references, with per-layer alpha/contribution readback.
Compare viewer/control defaults without treating synthetic output as real-client fidelity proof.
Prepare operator visual/target commands; compilation alone never proves correct material opacity.
