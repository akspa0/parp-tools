# WoWFileFormatDefs: Declarative Schema Definitions for WoW Binary Formats

**Created**: 2026-06-16 | **Status**: Vision / Architecture Direction

## The Problem

WoW's binary file formats (ADT, WDT, WMO, M2, MDX, PM4, BLP, etc.) are compressed relational databases. Each format contains:
- **Tables**: Chunks like MCNK (terrain), MDDF/MODF (placements), MTEX/MMDX/MWMO (string tables), MCIN (index), MH2O (liquid)
- **Indices**: MCIN offsets → MCNK locations, MMID/MWID → MMDX/MWMO offsets
- **Foreign keys**: MCLY textureId → MTEX entry, MCRF doodad refs → MDDF indices, MDDF nameId → MMDX offset
- **Inter-file references**: WDT MAIN flags → which ADTs exist, _obj0.adt MDDF entries → M2 model files

The current state of WoW tooling (ours included) treats these as opaque binary blobs with hand-coded readers/writers per format. Every writer is a bespoke implementation that hallucinates byte offsets and gets foreign keys wrong. The Pm4AdtWriter disaster is proof — it patched placement chunks incorrectly because it didn't understand the relational schema.

## The Working Pattern: DBC/DB2 + WoWDBDefs

We already have a proven pattern for this exact problem:

1. **WoWDBDefs** (`wow-viewer/libs/wowdev/WoWDBDefs/definitions/*.dbd`) — hundreds of declarative schema files that define column names, types, build-specific layouts, and relationships. Each `.dbd` file is a `CREATE TABLE` statement for a DBC table.

2. **DBCD** — a generic reader that takes binary DBC data + `.dbd` schema definition and produces a typed `IDBCDStorage` you can query by column name and row ID. No hand-coded per-table readers.

3. **Typed accessors** — `AreaIdMapper`, `LightService`, `MapDiscoveryService`, `ReplaceableTextureResolver`, etc. consume DBCD storage and provide domain-specific typed access.

This pattern is **correct**. It works for hundreds of DBC tables across dozens of build versions. The schemas are declarative, version-aware, and machine-readable.

## The Vision: WoWFileFormatDefs

Extend the WoWDBDefs pattern from flat DBC tables to nested chunked binary formats:

### What WoWDBDefs does for DBC:
```
AreaTable.dbd → declares columns (ID, AreaName, ContinentID, ...)
DBCD reads .dbc + .dbd → IDBCDStorage
AreaIdMapper queries IDBCDStorage
```

### What WoWFileFormatDefs would do for ADT/WDT/WMO/M2/PM4:
```
adt_v18.schema → declares chunks (MHDR, MCIN, MTEX, MMDX, ..., MCNK), 
                  their columns, offsets, foreign keys, and sub-table relationships
AdtFormatReader reads .adt + schema → IFormatStorage
LkAdtData / BlankAdtFactory / AdtInspector queries IFormatStorage
```

### Key Differences from DBC

| Aspect | DBC/DB2 | ADT/WDT/WMO/M2 |
|--------|---------|-----------------|
| Structure | Flat rows, fixed columns | Nested chunks, variable-size sub-tables |
| FK style | Column value → row in another DBC | Byte offset → chunk in same file, or file offset |
| Versioning | Build-specific column layout | Build-specific chunk presence/absence and field sizes |
| Nesting | None | MCNK contains MCVT, MCNR, MCLY, MCAL, MCRF, etc. |
| Inter-file | None (each DBC is independent) | WDT → ADT filenames, _obj0.adt references M2/WMO paths |
| String tables | DBCD handles inline | Separate MTEX, MMDX, MWMO chunks with offset tables (MMID, MWID) |

### Schema Definition Format

A `.wffd` (WoW File Format Definition) file for LK ADT would look like:

```yaml
# adt_v18.wffd — LK 3.3.5 monolithic ADT schema
format: ADT
version: 18  # MVER value
name: LichKingADT

chunks:
  - id: MVER
    columns:
      - name: version
        type: UINT32
        value: 18

  - id: MHDR
    columns:
      - name: flags
        type: UINT32
        offset: 0x00
      - name: ofsMcin
        type: UINT32
        offset: 0x04
        fk: MCIN  # foreign key → MCIN chunk offset
      - name: nMtex
        type: UINT32
        offset: 0x08
        description: "Count of MTEX entries"
      - name: ofsMtex
        type: UINT32
        offset: 0x0C
        fk: MTEX
      # ... remaining MHDR fields

  - id: MCIN
    row_count: 256
    columns:
      - name: offset
        type: UINT32
        fk: MCNK  # → absolute file offset of MCNK[i]
      - name: size
        type: UINT32
      - name: flags
        type: UINT32
      - name: pad
        type: UINT32

  - id: MTEX
    type: string_table  # null-terminated string sequence

  - id: MMDX
    type: string_table

  - id: MMID
    type: offset_table
    target: MMDX  # each entry is an offset into MMDX

  - id: MWMO
    type: string_table

  - id: MWID
    type: offset_table
    target: MWMO

  - id: MDDF
    columns:
      - name: nameId
        type: INT32
        fk: MMID  # → offset into MMDX via MMID[nameId]
      - name: uniqueId
        type: INT32
      - name: position
        type: FLOAT3  # Vector3
      - name: rotation
        type: FLOAT3
      - name: scale
        type: FLOAT16  # UINT16 / 1024.0

  - id: MODF
    columns:
      - name: nameId
        type: INT32
        fk: MWID  # → offset into MWMO via MWID[nameId]
      - name: uniqueId
        type: INT32
      - name: position
        type: FLOAT3
      - name: rotation
        type: FLOAT3
      - name: boundsMin
        type: FLOAT3
      - name: boundsMax
        type: FLOAT3
      - name: flags
        type: UINT16
      - name: doodadSet
        type: UINT16
      - name: nameSet
        type: UINT16
      - name: scale
        type: FLOAT16

  - id: MCNK
    row_count: 256  # one per MCIN entry
    index_source: MCIN.offset
    sub_chunks:
      - id: MCVT
        columns:
          - name: heights
            type: FLOAT
            count: 145  # 9×9 + 8×8 vertex grid
      - id: MCNR
        columns:
          - name: normals
            type: BYTE
            count: 448  # 3 bytes per normal × ~149
      - id: MCLY
        row_count: variable  # NLayers
        columns:
          - name: textureId
            type: UINT32
            fk: MTEX  # → offset into MTEX string table
          - name: flags
            type: UINT32
          - name: alphaOffset
            type: UINT32
            fk: MCAL  # → offset within MCNK's MCAL data
          - name: effectId
            type: UINT32
      - id: MCRF
        columns:
          - name: doodadRefs
            type: INT32_ARRAY
            fk: MDDF  # → MDDF entry indices
          - name: worldModelRefs
            type: INT32_ARRAY
            fk: MODF  # → MODF entry indices
      - id: MCAL
        type: alpha_mask  # compressed or uncompressed alpha data
      - id: MCSH
        type: shadow_map
      - id: MCLQ
        type: liquid_chunk
      - id: MCCV
        type: vertex_colors
      - id: MCLV
        type: vertex_lighting

  - id: MH2O
    type: liquid_data
    description: "Per-chunk liquid information"

  - id: MFBO
    type: flight_bounds
    description: "3×3 float array for flight boundary"

  - id: MTXF
    columns:
      - name: textureId
        type: UINT32
        fk: MTEX
      - name: flags
        type: UINT32
```

### What This Enables

1. **Generic format reader**: Read any chunked binary file using its `.wffd` schema → produces structured data without hand-coded per-format readers
2. **Generic format writer**: Write any chunked binary file from structured data using its `.wffd` schema → produces correct byte offsets, FK references, and chunk sizes automatically
3. **Format inspection/diff**: Compare two ADT files structurally — which chunks differ, which FK references changed, which placements moved
4. **Cross-format navigation**: Follow FK references from WDT → ADT → MDDF → MMDX → M2 model path, all through schema-aware typed access
5. **Blank generation**: Given a schema, generate minimal valid files with zeroed defaults
6. **Round-trip verification**: Read → schema → write → compare bytes. Any difference is a bug in the schema or the writer.
7. **PM4 matching output**: Instead of patching ADT bytes, emit placement data that a schema-aware writer can correctly insert using proper FK resolution

### Why This Is The Holy Grail

No existing tool does this. The WoW modding ecosystem has:
- **WoWDBDefs/DBCD**: Perfect for DBC, but doesn't touch chunked formats
- **WoW.Tools/ExportTools**: Per-format readers, no unified schema model
- **Noggit/ModernWoW tooling**: C++ readers/writers, opaque binary, no schema awareness
- **WoWDev wiki**: Human-readable documentation, not machine-readable
- **Our own readers**: Hand-coded per-format, correct but fragile and duplicative

A unified `.wffd` schema system would be the first tool that:
- Reads AND writes every chunked format through the same schema-driven pipeline
- Validates FK integrity across chunks automatically
- Handles version differences declaratively (same way `.dbd` handles build-specific column layouts)
- Enables a universal viewer/inspector that knows the schema of every file it opens
- Makes PM4 → ADT restoration a matter of "insert these rows into the MDDF table, resolve the MMDX FK, fix up the MCRF references" instead of "patch these bytes at these offsets and pray"

## Relationship to Spec 064

Spec 064 (Blank Map Generation) is Phase 1 of this vision. The steps are:

1. **P1**: Generate blank ADT/WDT/WDL that loads → proves our writer model is correct
2. **P2**: Document ADT relational schema → this is the first `.wffd` file
3. **P3**: Build schema-driven reader/writer → generalizes beyond ADT to WDT, WMO, M2, PM4

Each phase proves the model before the next one starts.

## Roadmap (Very Long-Term)

| Format | Schema | Reader | Writer | Blank Gen | Round-Trip |
|--------|--------|--------|--------|------------|-------------|
| ADT v18 (LK 3.3.5) | P2 | Exists (partial) | Exists (LkAdtWriter) | P1 | P2 |
| WDT (LK) | P2 | Exists | Exists (LkWdtWriter) | P1 | P2 |
| WDL | P2 | Exists | Exists (WdlWriter) | P1 | P2 |
| WDT v18 (Alpha 0.5.3) | Later | Exists (AlphaWdtReader) | Exists (AlphaWdtWriter, frozen) | Later | Later |
| ADT v23+ (Cata+) | Later | Exists (partial) | TBD | Later | Later |
| WMO | Later | Exists (partial) | TBD | Later | Later |
| M2 v11+ | Later | Exists (partial) | TBD | Later | Later |
| PM4 | Later | Exists | N/A (not written) | N/A | N/A |
| BLP | Later | Exists | TBD | Later | Later |

The "Later" items are explicitly not in scope for spec 064. They are listed here to show the long-term direction — the same schema-driven pattern applies to every format.

## Naming

**WoWFileFormatDefs** — by analogy with WoWDBDefs. Could also be called:
- `.wffd` files (WoW File Format Definitions)
- `WowViewer.Core.Schema` — the C# namespace for schema-driven reading/writing
- `WowViewer.Tool.SchemaDump` — tool to dump any file's schema-validated structure

The `.wffd` extension mirrors `.dbd` and makes the relationship clear.

## Constitution Alignment

| Principle | Status |
|-----------|--------|
| Repo Independence | ✅ All under `wow-viewer/` |
| Library-First | ✅ Schema → Core library, tools are thin wrappers |
| Real-Data Validation | ✅ Every phase validates against real client files |
| Format Reader/Writer Ownership | ✅ One canonical owner per format, driven by schema |
| One Phase at a Time | ✅ P1 (blank ADT) before P2 (schema) before P3 (Zarr/datastore) |