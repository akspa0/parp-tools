# System Patterns

## FourCC Handling (CRITICAL)

### The Rule
```
READ:  Reverse on disk → Forward in memory (XETM → MTEX)
WRITE: Forward in memory → Reverse on disk (MTEX → XETM)
```

### In Code
- **Never** use reversed literals (`XETM`, `KNCM`) except in the lowest-level writer
- **Always** normalize to readable form immediately after reading
- **Only** reverse at the moment of writing bytes to disk

## ADT Structure

### LK 3.3.5 (Split)
```
<map>_XX_YY.adt      — Root (terrain, MCNK headers)
<map>_XX_YY_obj0.adt — Objects (MDDF, MODF placements)
<map>_XX_YY_tex0.adt — Textures (MTEX, MCLY layers)
```

### Alpha 0.5.3 (Monolithic)
```
<map>.wdt — Contains EVERYTHING:
  MPHD, MAIN, MDNM, MONM, then per-tile MHDR+MCIN+MCNKs
```

## Merge Priority
When merging split → monolithic:
1. `_tex0` data **overwrites** root texture data
2. `_obj0` data **overwrites** root placement data
3. Root provides base terrain (MCVT, MCNR)

## Offset Conventions

### Alpha WDT
- **MHDR offsets**: Relative to `MHDR.data` start (after 8-byte header)
- **MCIN offsets**: Absolute file positions
- **MAIN offsets**: Absolute file positions

### LK ADT
- **MHDR offsets**: Relative to file start + 0x14 (Noggit convention)
- **MCNK subchunk offsets**: Relative to MCNK chunk start
