# WDT File Format Specification

This document provides a comprehensive specification of the WDT (World Definition Table) file format, reverse-engineered from the parpToolbox codebase and related tools like AlphaWDTReader and gillijimproject-csharp. WDT files index ADT map tiles for a continent/zone (64x64 grid), containing offsets/flags in MAIN, and optional name lists (MDNM for doodads, MONM for WMOs). Alpha-era WDTs (pre-WotLK) have variants (e.g., MPHD header, sparse MAIN), analyzed/converted to LK (3.x) format. WDT is chunked IFF-style, similar to ADT/WMO.

Spec derived from WdtFormat.cs (ProjectArcane), AlphaWDTReader (ADTPreFabTool), and gillijimproject (conversion logic). Why: Efficient tile loading (presence flags, offsets); how: Chunk scan, MAIN grid parsing. Tools like AlphaWDTReader parse Alpha WDT for MAIN/MDNM/MONM, convert to LK via gillijimproject-csharp.

## Overall Structure
WDT chunks (order: MVER, MAIN, then optional MDNM/MONM):
- MVER (version: 18 for LK/Alpha)
- MAIN (8192 bytes: 64x64 entries of flags/offsets)
- MDNM (variable: doodad names, null-terminated)
- MONM (variable: WMO names, null-terminated)
- ... (Alpha: MPHD header with offsets)

Size: ~10KB (small, index-only). Validation: MVER present; MAIN fixed 8192 bytes; name chunks match counts. Alpha: Variable MAIN (e.g., 64x64x8 for flags/pad).

### Chunk Header
```c
struct ChunkHeader {
    char signature[4];  // e.g., "MAIN" (reversed on disk)
    uint32_t size;      // Payload
};
```

## Key Chunks

### MVER: Version (4 bytes)
**Purpose**: WDT version (18 for both Alpha/LK).

**Plain English**: Simple tag for format (18=standard). Code: `br.ReadInt32()`; logs if !=18.

**C Struct**:
```c
struct MVER {
    int32_t version;  // 18
};
```

**Usage**: `WdtFormat.MVER`; required for parsing.

### MAIN: Tile Index (8192 bytes)
**Purpose**: 64x64 grid of ADT tiles: flags (presence, liquid, etc.), offsets to ADT files.

**Plain English**: Rows tx=0..63, cols ty=0..63; each entry (16/8 bytes Alpha): flags (bit0=present, bit1=liquid), offset (to ADT or 0). Alpha sparse: flags+pad. Code: 2D array parse, filter present tiles.

**C Struct** (LK, 16 bytes/entry):
```c
struct MAINEntry {
    uint32_t flags;     // Bit0=ADT present, bit1=liquid, bit2=terrain, etc.
    uint32_t offset;    // File offset to ADT (0=absent)
    uint32_t pad1, pad2; // Padding/unknown
};  // 64x64 array
```
Alpha variant: 8 bytes (flags+pad), offset inferred.

**Usage**: `WdtFormat.MAIN`; `AlphaWDTReader` scans for present (flags&1), converts to LK MAIN with offsets.

### MDNM: Doodad Names (variable, null-terminated strings)
**Purpose**: List of M2 doodad names referenced in ADTs.

**Plain English**: Null-terminated strings (e.g., "tree.m2"); count from size. Code: Loop `ReadNullTerminatedString` until end.

**C Struct**:
```c
struct MDNM {
    char doodad_names[variable][];  // Null-terminated .m2 paths
};  // Size = sum(strlen+1)
```

**Usage**: `WdtFormat.MDNM`; used in gillijimproject for index remapping (Alpha→LK).

### MONM: WMO Names (variable, null-terminated strings)
**Purpose**: List of WMO names referenced in ADTs.

**Plain English**: Similar to MDNM but for .wmo files. Code: Same as MDNM.

**C Struct**:
```c
struct MONM {
    char wmo_names[variable][];  // Null-terminated .wmo paths
};
```

**Usage**: `WdtFormat.MONM`; conversion tools remap indices.

## Alpha-Specific: MPHD (16 bytes)
**Purpose**: Alpha header with offsets to MAIN/MDNM/MONM.

**Plain English**: Offsets/counts for sparse layout. Code: `AlphaWDTReader.MPHD` reads uints.

**C Struct**:
```c
struct MPHD {
    uint32_t main_offset, main_size;
    uint32_t mdnm_offset, mdnm_size;
    uint32_t monm_offset, monm_size;
    uint32_t unknown;
};  // 16 bytes
```

## Alpha WDT to LK WDT Conversion Process
The Alpha (0.5.3-era) to LK (3.3.5/WotLK) WDT conversion is complex due to format differences: Alpha uses sparse MPHD/MAIN (flags+pad, no offsets), while LK embeds full ADT payloads with absolute offsets. Tools like AlphaWDTReader and gillijimproject-csharp handle this via multi-step remapping. Process overview (from WdtAlpha.cs, AlphaWdtReader.cs):

1. **Parse Alpha WDT**:
   - Read MVER (confirm v18).
   - Read MPHD: Offsets/sizes for MAIN (sparse 64x64x8 flags/pad), MDNM/MONM (names).
   - Scan MAIN: Identify present tiles (flags&1 !=0); no offsets (ADTs external).
   - Extract names: MDNM (.m2), MONM (.wmo) for index tables.

2. **Discover/Convert ADTs**:
   - For each present tile (tx,ty): Locate external Alpha ADT (e.g., "{map}_xx_yy.adt").
   - Convert ADT: Remap internals (e.g., McnkAlpha.ToMcnkLk: update M2/WMO indices via Mcrf.UpdateIndicesForLk, using MDNM/MONM for name→id; handle MH2O liquids via mclq_to_mh2o connected components; AreaID normalization via areatable_mapper).
   - Embed raw LK ADT bytes into new WDT (after names).

3. **Build LK MAIN**:
   - Create 64x64 MAIN: For present tiles, set flags (bit0=1, bit1=liquid if MH2O), offset=absolute pos in WDT (post-MAIN/MDNM/MONM).
   - Absent tiles: offset=0, flags=0.
   - Pad to 8192 bytes.

4. **Remap Indices**:
   - Build global M2/WMO tables: Collect unique names from all ADTs, assign sequential ids.
   - For each ADT: Update MDDF/MODF (name_id→new id), MCRF (doodad/WMO refs via GetDoodadsIndices/GetWmosIndices).
   - Handle Alpha quirks: M2Number/NLayers in MCNK, offset math (chunk_origin_math).

5. **Write LK WDT**:
   - MVER (18), MAIN (offsets), MDNM/MONM (global names), then concatenated LK ADT payloads.
   - Output: "{map}_new.wdt" with embedded ADTs (~MBs if many tiles).

**Challenges/Complexity**:
- Sparse→embedded: Alpha external ADTs become internal offsets; requires full ADT conversion first.
- Index remapping: Cross-ADT unique names; collisions resolved by normalization (e.g., case-insensitive).
- Liquids/AreaIDs: Alpha MCLQ→LK MH2O (connected_components for layers); AreaTable remap (areatable_mapper for 0.5.3→3.3.5 IDs).
- Validation: Post-conversion, verify offsets (MAIN→ADT MHDR), indices (no out-of-bounds in MDDF/MCRF).

**Code Usage**:
- `AlphaWDTReader.Read(path)`: Parses MPHD/MAIN, returns present tiles/names.
- `gillijimproject-csharp`: `WdtAlpha.ToWdt(outputDir)`: Orchestrates ADT conversion (AdtAlpha.ToAdtLk), embedding via offset_builder.
- Snippets: mclq_to_mh2o (liquids), alpha_mcvt_index_map (heights), mcnr_unpack (normals).

For failures: Log mismatches (e.g., missing ADTs); use --compare for diff reports.

## Data Arrangement and Usage in Tools
WDT: MVER→MAIN (grid flags/offsets)→names (MDNM/MONM for ADT refs). Alpha: MPHD→sparse chunks.

- **AlphaWDTReader**: Parses Alpha WDT (MPHD/MAIN), lists present tiles, converts to LK MAIN (offsets to embedded ADTs).
- **gillijimproject-csharp**: `WdtAlpha.ToWdt()` builds LK WDT from Alpha; remaps names/indices for ADTs.
- **ADTPreFabTool**: Loads WDT MAIN for tile coords; prefabs ADTs with WDT flags.
- **parpToolbox**: Inferred via WoWFormatLib; `Pm4BatchProcessor` uses WDT for region loading (WDT→ADTs→MODF WMO/MDDF M2).

Interrelation: WDT MAIN→load ADTs; ADT uses MDNM/MONM indices for names. Validate with `WdtFormat.IsValid` (MVER+MAIN present).

For ADT/M2/WMO integration, see respective specs.