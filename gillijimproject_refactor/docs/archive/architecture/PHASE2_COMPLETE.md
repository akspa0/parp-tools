# Phase 2: Extraction - COMPLETE ✅

## Summary

Phase 2 (MCNK Extraction) has been successfully implemented in AlphaWDTAnalysisTool.

**Time Spent**: ~2.5 hours  
**Status**: ✅ Ready for testing

---

## Files Created

### Core Classes (AlphaWdtAnalyzer.Core/Terrain/)

1. **`McnkTerrainEntry.cs`** (23 properties)
   - Complete terrain data record for a single chunk
   - Includes: all flags, liquids, holes, AreaID, positions

2. **`McnkTerrainExtractor.cs`** (197 lines)
   - Extracts all MCNK data from Alpha WDT files
   - Parses 32-bit flags, liquid types, holes, AreaID
   - Calculates chunk world positions

3. **`McnkTerrainCsvWriter.cs`** (62 lines)
   - Writes 23-column CSV output
   - Proper CSV escaping and formatting

4. **`McnkShadowEntry.cs`** (8 properties)
   - Shadow map data record
   - Includes base64-encoded 64×64 bit shadow bitmap

5. **`McnkShadowExtractor.cs`** (175 lines)
   - Extracts MCSH shadow map data
   - Base64 encodes 512-byte shadow bitmaps
   - Handles missing shadow data gracefully

6. **`McnkShadowCsvWriter.cs`** (55 lines)
   - Writes shadow CSV output
   - Handles large base64 strings

### Integration Changes

7. **`AnalysisPipeline.cs`** (Modified)
   - Added `ExtractMcnkTerrain` option
   - Added `ExtractMcnkShadows` option
   - Calls extractors when flags are set

8. **`BatchAnalysis.cs`** (Modified)
   - Added extraction options to batch processing
   - Extracts terrain/shadows for all maps

9. **`Program.cs`** (Modified)
   - Added `--extract-mcnk-terrain` CLI flag
   - Added `--extract-mcnk-shadows` CLI flag
   - Updated usage help text
   - Passes flags through to pipelines

---

## CSV Output Format

### `<map>_mcnk_terrain.csv` (23 columns)

```csv
map,tile_row,tile_col,chunk_row,chunk_col,
flags_raw,has_mcsh,impassible,lq_river,lq_ocean,lq_magma,lq_slime,
has_mccv,high_res_holes,areaid,num_layers,
has_holes,hole_type,hole_bitmap_hex,hole_count,
position_x,position_y,position_z
```

**Example Row**:
```csv
Azeroth,31,34,0,0,0x00000001,true,false,false,false,false,false,false,false,1519,2,false,none,0x0000,0,17066.67,34133.33,0.00
```

### `<map>_mcnk_shadows.csv` (8 columns)

```csv
map,tile_row,tile_col,chunk_row,chunk_col,has_shadow,shadow_size,shadow_bitmap_base64
```

**Example Row**:
```csv
Azeroth,31,34,0,0,true,512,AAAA//8AAAD//wAAf/8AAH//AAB//wAA...
```

---

## CLI Usage

### Single Map
```bash
AlphaWdtAnalyzer \
  --input "World/Maps/Azeroth/Azeroth.wdt" \
  --listfile "listfile.csv" \
  --out "output" \
  --extract-mcnk-terrain \
  --extract-mcnk-shadows
```

### Batch Processing (All Maps)
```bash
AlphaWdtAnalyzer \
  --input-dir "World/Maps" \
  --listfile "listfile.csv" \
  --out "output" \
  --extract-mcnk-terrain \
  --extract-mcnk-shadows
```

**Output Location**:
- Single: `output/csv/<map>/<map>_mcnk_terrain.csv`
- Batch: `output/csv/maps/<map>/<map>_mcnk_terrain.csv`

---

## Technical Details

### MCNK Terrain Extraction

**Data Sources**:
- MCNK header flags (offset 0x00, 32 bits)
- AreaID from Unknown3 field (offset 0x38)
- Holes bitmap (offset 0x40, 16 bits)
- Number of layers (offset 0x10)
- Chunk indices (IndexX, IndexY)

**Position Calculation**:
- Uses `McnkLk.ComputePositionFromAdt()` for consistency
- Calculates world X/Y/Z from tile + chunk coordinates

**Flag Parsing** (32-bit flags):
- `0x1` - has_mcsh (shadow map present)
- `0x2` - impassible
- `0x4` - lq_river
- `0x8` - lq_ocean
- `0x10` - lq_magma
- `0x20` - lq_slime
- `0x40` - has_mccv (vertex colors)

### MCSH Shadow Extraction

**Data Source**:
- MCSH subchunk (offset from MCNK header + 0x30)
- Size from MCNK header + 0x34

**Bitmap Format**:
- 64×64 bits = 512 bytes
- LSB-first encoding
- 0 = shadowed (dark), 1 = lit (bright)

**Base64 Encoding**:
- 512 bytes → ~684 character base64 string
- Stored in CSV for easy transport

---

## Known Limitations

### Alpha Format Specifics

1. **No High-Res Holes**: Alpha only has low-res holes (16-bit), not high-res (64-bit)
   - `high_res_holes` column always `false`
   - `hole_type` is either `none` or `low_res`

2. **AreaID Encoding**: Uses Unknown3 field
   - This is the Alpha AreaID encoding
   - May need mapping to LK AreaTable (handled in Phase 3)

3. **No MCCV Data**: While flag exists, MCCV subchunk not extracted yet
   - `has_mccv` reports flag status only
   - Actual vertex color data not in CSV

4. **Position Estimation**: Chunk positions calculated, not read from header
   - Alpha MCNK doesn't store PosX/Y/Z like LK
   - Uses standardized calculation for consistency

---

## Testing Checklist

### Before Moving to Phase 3

- [ ] Build AlphaWDTAnalysisTool project successfully
- [ ] Test extraction on single map (e.g., Azeroth)
- [ ] Verify CSV output format (23 columns for terrain)
- [ ] Check shadow CSV output (base64 strings present)
- [ ] Validate AreaID values against known areas
- [ ] Test batch processing on multiple maps
- [ ] Verify flags parse correctly (impassible, liquids)
- [ ] Check hole detection and bitmap output
- [ ] Confirm shadow maps only for chunks with has_mcsh=true
- [ ] Test with missing/empty chunks (should handle gracefully)

### Expected Output Size

**Per Continent** (64×64 tiles × 256 chunks = 1,048,576 total chunks):
- Terrain CSV: ~32MB uncompressed
- Shadow CSV: ~128MB uncompressed (with base64 overhead)

**Optimization**: Shadow CSV is optional, can skip if not needed for overlay.

---

## Next Steps: Phase 3

Phase 3 will transform these CSVs into viewer-ready JSON overlays in WoWRollback.Core.

**Required Components**:
1. `McnkTerrainCsvReader.cs` - Read terrain CSV
2. `AreaTableReader.cs` - Read existing AreaTable CSVs
3. `TerrainPropertiesOverlayBuilder.cs` - Group by properties
4. `LiquidsOverlayBuilder.cs` - Group by liquid types
5. `HolesOverlayBuilder.cs` - Decode hole bitmaps
6. `AreaIdOverlayBuilder.cs` - Detect boundaries
7. `ShadowMapCompositor.cs` - Composite 16×16 shadow maps to 1024×1024 PNG

**Estimated Time**: 6 hours

---

## Phase 2 Status: ✅ COMPLETE

All extraction code is implemented and integrated. Ready to test and move to Phase 3 (Transformation).

**To continue**: Type `CONTINUE` to begin Phase 3 implementation, or `TEST` to validate Phase 2 first.
