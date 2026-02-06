# System Patterns

## FourCC Handling (CRITICAL)

### The Rule
```
READ:  Reverse on disk → Forward in memory (XETM → MTEX)
WRITE: Forward in memory → Reverse on disk (MTEX → XETM)
```

## MDX Alpha 0.5.3 Patterns

### GEOS Sub-Chunk Parsing
- **Padding Aware**: Always scan for 4-byte UTF-8 tags (`VRTX`, `TVRT`, `NRMS`, `PTYP`, `PCNT`, `PVTX`, `GNDX`, `MTID`).
- **Null Safety**: Avoid fixed-offset jumps between sub-chunks. Padding can be 0-12 bytes.
- **UVAS (v1300)**: If `VERS` is 1300, `UVAS` Count=1 contains raw UV data (8 bytes per vertex) immediately. There is no `UVBS` tag.

### Texture Resolution (DBC + Fallback)
- **Primary Source**: `DbcService` (Resolves `ModelID` -> `TextureVariation` strings via `CreatureDisplayInfo`).
- **Baked Skins**: Queries `CreatureDisplayInfoExtra` for `BakeName` column when `ExtraId > 0`.
- **ID Offset Rule**: `ReplaceableId 11+n` or `1+n` maps to variation index `n`.
- **Legacy Fallback**: If DBC lookup fails, default to `<ModelName>Skin.blp` or local directory scan.

## ADT Structure

### Alpha 0.5.3 (Monolithic)
```
<map>.wdt — Monolithic tileset:
  MPHD, MAIN, MDNM, MONM, then per-tile MHDR+MCIN+MCNKs
```

### LK 3.3.5 (Split)
```
<map>_XX_YY.adt      — Root (terrain)
<map>_XX_YY_obj0.adt — Objects
<map>_XX_YY_tex0.adt — Textures
```

## Offset Conventions

### Alpha WDT
- **MCIN offsets**: Absolute file positions.
- **MAIN offsets**: Absolute file positions.

### LK ADT
- **MCIN offsets**: Read 256 entries to locate non-sequential MCNK chunks.
