# LK ADT/WDT to Alpha Converter - Implementation Plan

## Goal
Convert Lich King (3.3.5a) ADT/WDT files to Alpha (0.5.3-0.6.0) format for testing "rollback" scenarios in genuine Alpha clients. Strip away post-Alpha features while preserving core terrain, textures, and model placements.

## Use Case
- Parse stable, well-documented LK ADT files
- Convert to Alpha format (simpler, single-file ADTs)
- Test in real Alpha client to see "what it would have looked like"
- Preserve: terrain geometry, textures, model placements
- Strip: advanced features added after Alpha

---

## File Format Differences

### WDT (World Data Table)
| Feature | LK (3.3.5a) | Alpha (0.5.3-0.6.0) | Conversion |
|---------|-------------|---------------------|------------|
| Structure | MVER + MPHD + MAIN + MWMO + MODF | MVER + MAIN | Strip MPHD/MWMO/MODF |
| Tile flags | 64x64 grid in MAIN | 64x64 grid in MAIN | Direct copy |
| WMO support | Yes (MWMO/MODF) | Limited/None | Strip or warn |

### ADT (Area Data Tile)
| Feature | LK (3.3.5a) | Alpha (0.5.3-0.6.0) | Conversion |
|---------|-------------|---------------------|------------|
| File split | 3 files (root/obj/tex) | Single file | Merge all data |
| MCNK chunks | 16x16 per tile | 16x16 per tile | Direct copy structure |
| Vertices (MCVT) | 145 per chunk (9x9+8x8) | 145 per chunk | Direct copy |
| Texture layers (MCLY) | Up to 4 layers | Up to 4 layers | Direct copy |
| Alpha maps (MCAL) | Compressed | Uncompressed? | May need decompression |
| Models (MDDF) | M2 format (.m2) | MDX format (.mdx) | Rename references |
| WMOs (MODF) | Supported | Supported | Direct copy |
| Liquids (MH2O) | Advanced chunk | Simpler MCLQ | Convert or strip |

---

## Project Structure

```
WoWRollback.AdtConverter/
├── Readers/
│   ├── LkAdtReader.cs           // Read LK ADT (root/obj/tex split files)
│   ├── LkWdtReader.cs           // Read LK WDT
│   └── IAdtReader.cs            // Interface for future formats
├── Converters/
│   ├── TerrainConverter.cs      // MCNK/MCVT/MCNR conversion
│   ├── TextureConverter.cs      // MCLY/MCAL layer simplification
│   ├── PlacementConverter.cs    // MDDF/MODF + M2→MDX rename
│   ├── LiquidConverter.cs       // MH2O → MCLQ (or strip)
│   └── WdtConverter.cs          // Strip advanced WDT features
├── Writers/
│   ├── AlphaAdtWriter.cs        // Write Alpha ADT (single file)
│   ├── AlphaWdtWriter.cs        // Write Alpha WDT
│   └── IAdtWriter.cs            // Interface for future formats
├── Models/
│   ├── LkAdtData.cs             // In-memory representation of LK ADT
│   ├── AlphaAdtData.cs          // In-memory representation of Alpha ADT
│   └── ConversionResult.cs      // Conversion metadata/warnings
└── AdtConverterService.cs       // Main orchestrator
```

---

## Conversion Pipeline

### Phase 1: Read LK Files
```
Input: LK ADT files (mapname_xx_yy.adt, _obj0.adt, _tex0.adt)
       LK WDT file (mapname.wdt)

Steps:
1. Parse LK WDT → extract tile flags
2. For each tile:
   a. Parse root ADT → terrain chunks (MCNK)
   b. Parse obj ADT → model placements (MDDF/MODF)
   c. Parse tex ADT → texture layers (MCLY/MCAL)
3. Merge into unified in-memory structure
```

### Phase 2: Convert Data
```
For each ADT tile:
1. Terrain (MCNK/MCVT/MCNR):
   - Copy vertex heights directly (145 vertices)
   - Copy normals (if present)
   - Preserve chunk flags (holes, etc.)

2. Textures (MCLY/MCAL):
   - Copy up to 4 texture layers
   - Decompress alpha maps if needed
   - Simplify blending modes (Alpha has fewer)
   - Map texture IDs (may need texture name mapping)

3. Models (MDDF):
   - Copy position, rotation, scale
   - Rename .m2 → .mdx in filename references
   - Preserve uniqueID

4. WMOs (MODF):
   - Copy position, rotation, scale
   - Preserve uniqueID
   - Warn if WMO not available in Alpha

5. Liquids:
   - Option A: Convert MH2O → MCLQ (complex)
   - Option B: Strip liquids entirely (simpler)

For WDT:
1. Copy MAIN chunk (tile flags)
2. Strip MPHD, MWMO, MODF chunks
3. Write minimal Alpha WDT
```

### Phase 3: Write Alpha Files
```
Output: Alpha ADT files (mapname_xx_yy.adt - single file)
        Alpha WDT file (mapname.wdt)

Steps:
1. Write Alpha WDT:
   - MVER chunk (version)
   - MAIN chunk (tile flags)

2. For each tile, write Alpha ADT:
   - MVER chunk
   - MHDR chunk (header with offsets)
   - MCIN chunk (chunk index)
   - MTEX chunk (texture filenames)
   - MMDX chunk (model filenames - .mdx!)
   - MMID chunk (model ID offsets)
   - MWMO chunk (WMO filenames)
   - MWID chunk (WMO ID offsets)
   - MDDF chunk (model placements)
   - MODF chunk (WMO placements)
   - 256x MCNK chunks (terrain data)
```

---

## Key Challenges

### 1. Texture References
- **Problem:** LK uses texture IDs, Alpha uses texture filenames
- **Solution:** Need texture ID → filename mapping (from DBC or listfile)

### 2. Model Format (M2 → MDX)
- **Problem:** Alpha uses .mdx extension, LK uses .m2
- **Solution:** Simple string replacement in MMDX chunk

### 3. Alpha Map Compression
- **Problem:** LK MCAL may be compressed, Alpha may expect uncompressed
- **Solution:** Decompress if needed, or copy raw if compatible

### 4. Liquids (MH2O → MCLQ)
- **Problem:** Completely different liquid systems
- **Solution:** Phase 1 - skip liquids entirely. Phase 2 - attempt conversion

### 5. Missing Assets
- **Problem:** Models/textures may not exist in Alpha client
- **Solution:** Log warnings, continue conversion (client will show placeholder)

---

## Implementation Phases

### Phase 1: Minimal Viable Converter (MVP)
**Goal:** Convert terrain geometry only, no textures/models

- [x] Read LK ADT root file (MCNK chunks)
- [x] Extract vertex heights (MCVT)
- [x] Write Alpha ADT with terrain only
- [x] Test in Alpha client (should show gray terrain)

**Deliverable:** Terrain-only converter

---

### Phase 2: Add Textures
**Goal:** Convert texture layers and alpha maps

- [ ] Read LK ADT tex file (MCLY/MCAL)
- [ ] Map texture IDs to filenames
- [ ] Decompress alpha maps if needed
- [ ] Write MTEX/MCLY/MCAL to Alpha ADT
- [ ] Test in Alpha client (should show textured terrain)

**Deliverable:** Textured terrain converter

---

### Phase 3: Add Models
**Goal:** Convert model placements with M2→MDX rename

- [ ] Read LK ADT obj file (MDDF/MODF)
- [ ] Rename .m2 → .mdx in model references
- [ ] Write MMDX/MMID/MDDF to Alpha ADT
- [ ] Test in Alpha client (should show models)

**Deliverable:** Full terrain + models converter

---

### Phase 4: Add WMOs
**Goal:** Convert WMO placements

- [ ] Copy MODF chunk
- [ ] Write MWMO/MWID/MODF to Alpha ADT
- [ ] Test in Alpha client (should show WMOs if available)

**Deliverable:** Complete converter (terrain + textures + models + WMOs)

---

### Phase 5: Polish & Optimization
**Goal:** Handle edge cases, add validation

- [ ] Validate converted files
- [ ] Add progress reporting
- [ ] Handle missing assets gracefully
- [ ] Add CLI with options (skip liquids, etc.)
- [ ] Batch convert entire maps

**Deliverable:** Production-ready converter

---

## CLI Interface

```bash
# Convert single tile
dotnet run --project WoWRollback.AdtConverter -- convert-tile \
  --lk-adt "path/to/Azeroth_32_48.adt" \
  --out "output/Azeroth_32_48.adt" \
  --format alpha

# Convert entire map
dotnet run --project WoWRollback.AdtConverter -- convert-map \
  --lk-dir "path/to/lk/World/Maps/Azeroth" \
  --out-dir "output/Azeroth" \
  --format alpha \
  --skip-liquids

# Options:
#   --skip-liquids        Don't convert liquid data
#   --skip-wmos           Don't convert WMO placements
#   --texture-mapping     Path to texture ID→name mapping file
#   --validate            Validate output files
```

---

## Testing Strategy

### Unit Tests
- Test each converter independently (terrain, textures, models)
- Verify chunk structure matches Alpha format
- Test M2→MDX renaming

### Integration Tests
- Convert known LK tiles
- Load in Alpha client
- Verify terrain renders correctly
- Verify models appear (if assets exist)

### Validation
- Compare converted ADT chunk structure to known-good Alpha ADTs
- Check file sizes (Alpha ADTs should be smaller - single file)
- Verify no crashes in Alpha client

---

## Success Criteria

1. ✅ Converted ADT loads in Alpha client without crash
2. ✅ Terrain geometry matches LK source (visually similar)
3. ✅ Textures display correctly (if assets exist in Alpha)
4. ✅ Models appear in correct positions (if assets exist in Alpha)
5. ✅ Can convert entire map (64x64 tiles) in reasonable time (<5 min)

---

## Future Enhancements

- **Reverse converter:** Alpha → LK (for testing)
- **Texture downsampling:** Reduce texture resolution for Alpha
- **Model simplification:** Reduce poly count for Alpha performance
- **Liquid conversion:** MH2O → MCLQ (complex, low priority)
- **GUI tool:** Visual converter with preview

---

## References

- [WoWDev Wiki - ADT Format](https://wowdev.wiki/ADT)
- [WoWDev Wiki - WDT Format](https://wowdev.wiki/WDT)
- Alpha ADT samples in `test_data/alpha_adts/`
- LK ADT samples in `test_data/lk_adts/`

---

## Notes

- **Preserve as much as possible:** Don't simplify unless necessary
- **Log everything:** Warn about missing assets, unsupported features
- **Fail gracefully:** Continue conversion even if some data is missing
- **Document assumptions:** Alpha format is not fully documented, may need experimentation
