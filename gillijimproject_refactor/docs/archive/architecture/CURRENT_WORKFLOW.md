# Current Workflow (Before Consolidation)

## Status: ✅ Working (with fixes applied)

This documents the current multi-tool workflow until the consolidation plan is implemented.

---

## Prerequisites

### 1. Generate AreaTable CSVs (One-time per version)

```bash
cd DBCTool.V2

# Extract and compare AreaTables
# Output: dbctool_outputs/session_{date}/compare/v2/AreaTable_dump_0.5.5.csv
# Output: dbctool_outputs/session_{date}/compare/v2/AreaTable_dump_3.3.5.csv
```

**Note**: rebuild-and-regenerate.ps1 now automatically copies these to rollback_outputs

---

## Multi-Map Workflow

### Step 1: Extract Terrain for All Maps

```bash
cd WoWRollback

# Option A: Extract specific maps
.\rebuild-and-regenerate.ps1 `
  -Maps @("Azeroth","Kalimdor","PVPZone01") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -Serve

# Option B: Auto-discover all maps
.\rebuild-and-regenerate.ps1 `
  -Maps @("auto") `
  -Versions @("0.5.3.3368") `
  -AlphaRoot ..\test_data\ `
  -Serve
```

**What Happens**:
1. Extracts MCNK terrain CSVs for each map
2. Copies CSVs to `rollback_outputs/0.5.3.3368/csv/{map}/`
3. Copies AreaTable CSVs to `rollback_outputs/0.5.3.3368/`
4. Generates viewer overlays for all maps
5. Starts web server

### Step 2: View in Browser

```
http://localhost:8080
```

- Select map from dropdown
- Enable terrain overlays
- **Areas now show correct names** (not "Unknown Area 1234")!

---

## Output Structure (Current)

```
rollback_outputs/
├── 0.5.3.3368/
│   ├── AreaTable_Alpha.csv       ← Copied from DBCTool.V2
│   ├── AreaTable_335.csv         ← Copied from DBCTool.V2
│   ├── csv/
│   │   ├── Azeroth/
│   │   │   └── Azeroth_mcnk_terrain.csv
│   │   ├── Kalimdor/
│   │   │   └── Kalimdor_mcnk_terrain.csv
│   │   └── PVPZone01/
│   │       └── PVPZone01_mcnk_terrain.csv
│   └── Azeroth/
│       └── id_ranges_by_map_alpha_Azeroth.csv
└── comparisons/
    └── 0_5_3_3368/
        └── viewer/
            ├── overlays/
            │   └── 0.5.3.3368/
            │       ├── Azeroth/
            │       │   └── terrain_complete/
            │       │       ├── tile_r31_c34.json  (with area names!)
            │       │       └── ...
            │       ├── Kalimdor/
            │       │   └── terrain_complete/
            │       │       └── ...
            │       └── PVPZone01/
            │           └── terrain_complete/
            │               └── ...
            └── (HTML/JS assets)
```

---

## Map/Version Validation (Viewer)

### Current Implementation

The viewer loads overlays based on URL pattern:
```javascript
/overlays/{version}/{map}/terrain_complete/tile_r{row}_c{col}.json
```

### JSON Structure with Validation Data

```json
{
  "map": "Azeroth",
  "tile": { "row": 31, "col": 34 },
  "layers": [
    {
      "version": "0.5.3.3368",
      "terrain_properties": { ... },
      "liquids": { ... },
      "holes": { ... },
      "area_ids": {
        "boundaries": [
          {
            "area_id": 1519,
            "area_name": "Stormwind City",  ← Real name!
            "neighbor_id": 1537,
            "neighbor_name": "Ironforge"    ← Real name!
          }
        ]
      }
    }
  ]
}
```

### Future Validation (Phase 2)

Add validation in `overlayManager.js`:
```javascript
loadAndValidate(data, expectedMap, expectedVersion) {
    if (data.map !== expectedMap) {
        console.warn(`Map mismatch: got ${data.map}, expected ${expectedMap}`);
        return null;
    }
    if (!data.layers.some(l => l.version === expectedVersion)) {
        console.warn(`Version mismatch: expected ${expectedVersion}`);
        return null;
    }
    return data;
}
```

---

## Known Limitations (To Be Fixed)

### 1. AreaTable Source
- ✅ **Fixed**: Now copies from DBCTool.V2 outputs
- ⏳ **Future**: Generate during WDT analysis (consolidation)

### 2. Path Inconsistencies
- ⏳ Cached maps in `cached_maps/`
- ⏳ CSVs in `rollback_outputs/{version}/csv/{map}/`
- ⏳ Viewer data in `rollback_outputs/comparisons/{key}/viewer/`
- ⏳ **Future**: Use DataPaths utility

### 3. Multiple Map Support
- ✅ Script generates all requested maps
- ✅ Viewer can switch between maps
- ⏳ **Future**: Add validation to prevent cross-map data loading

### 4. Performance
- ⏳ Loads all overlay JSON for visible tiles (can be slow)
- ⏳ **Future**: Implement tile batching, caching

---

## Testing Checklist

### Before Each Release

- [ ] Extract terrain for at least 2 maps (Azeroth, Kalimdor)
- [ ] Verify AreaTable CSVs exist in rollback_outputs/{version}/
- [ ] Generate viewer with `--viewer-report`
- [ ] Check terrain overlay JSON files contain area names (not "Unknown")
- [ ] Test in browser:
  - [ ] Switch between maps
  - [ ] Enable/disable terrain overlays
  - [ ] Verify area labels show correct names
  - [ ] Check no cross-map contamination

---

## Migration Path

As we implement the consolidation plan:

### Phase 1 (This Week)
- ✅ Fix AreaTable reader (parse 5-column CSV)
- ✅ Update script to copy AreaTable CSVs
- ⏳ Test multi-map viewer
- ⏳ Document current state

### Phase 2 (Next)
- Create WoWRollback.Data project
- Move DBC readers
- Move terrain extractors
- Update all references

### Phase 3 (Future)
- Centralize paths with DataPaths
- Add viewer validation
- Deprecate duplicate code in other tools

---

## Summary

**Current State**: Multi-map terrain overlays work with real area names!

**Workflow**:
1. DBCTool.V2 generates AreaTable CSVs (one-time)
2. rebuild-and-regenerate.ps1 extracts terrain and copies AreaTables
3. Viewer displays overlays with correct area names
4. Can switch between multiple maps

**Next Steps**: Implement consolidation plan to eliminate path issues and duplicate code.
