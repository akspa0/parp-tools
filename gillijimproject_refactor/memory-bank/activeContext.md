# Active Context

## Current Focus: PM4 Object Injection into WoWMuseum ADTs (Dec 11, 2025)

### Problem Statement
We need to inject PM4-reconstructed WMO placements (MODF) into WoWMuseum 3.3.5 ADTs. Previous approaches failed:
- **AdtModfInjector**: Appends chunks to end → invalid structure
- **Warcraft.NET**: Corrupts MCNK data during roundtrip (loses ~2KB)

### Data Sources (NEVER CHANGE)
| Source | Path | Contents |
|--------|------|----------|
| **Split Cata ADTs + PM4** | `test_data/development/World/Maps/development` | 466 root ADTs, 616 PM4 files |
| **Minimap tiles** | `test_data/minimaps/development` | 2252 PNG files for MCCV |
| **WoWMuseum 3.3.5 ADTs** | `test_data/WoWMuseum/335-dev/World/Maps/development` | 2303 complete monolithic ADTs |
| **WMO Library (legacy)** | `pm4-adt-test12/wmo_library.json` | Pre-computed WMO geometry stats (test12 snapshot) |
| **MODF Reconstruction (legacy)** | `pm4-adt-test12/modf_reconstruction/` | 1101 MODF entries, 352 WMO names (test12 snapshot) |
| **WMO Collision (per-group/flag)** | `pm4-adt-test13/wmo_flags/` | Split WMO collision OBJ library used for current PM4 matching |
| **MODF Reconstruction (current)** | `pm4-adt-test13/modf_reconstruction/` | Latest MODF entries + MWMO names from `pm4-reconstruct-modf` |

---

## NEXT SESSION: Chunk-Preserving ADT Patcher

### Core Principle
**WoWMuseum ADTs are the baseline** - they contain all terrain, textures, MCNK subchunks (MCAL, MCLY, MCCV, MCSH, etc.). We parse chunks as **raw bytes** and preserve them exactly. Only modify MWMO/MWID/MODF, then rebuild with recalculated offsets.

### Implementation Plan

#### Step 1: Create `MuseumAdtPatcher` class
Parse WoWMuseum ADT into raw chunk bytes:
```
struct ParsedAdt {
    byte[] mver;      // Keep as-is
    byte[] mhdr;      // Will recalculate offsets
    byte[] mcin;      // Will recalculate MCNK offsets
    byte[] mtex;      // Keep as-is
    byte[] mmdx;      // Keep as-is
    byte[] mmid;      // Keep as-is
    byte[] mwmo;      // APPEND new WMO names
    byte[] mwid;      // APPEND new offsets
    byte[] mddf;      // Keep as-is
    byte[] modf;      // APPEND new placements
    byte[] mh2o;      // Keep as-is (if present)
    byte[][] mcnks;   // Keep as-is (256 raw MCNK chunks)
    byte[] mfbo;      // Keep as-is (if present)
    byte[] mtxf;      // Keep as-is (if present)
}
```

#### Step 2: Chunk Parser
```csharp
// Parse ADT file into chunks by scanning for FourCCs
// Store each chunk as raw bytes including header (8 bytes) + data
// For MCNK: store all 256 as raw bytes (preserves all subchunks)
```

#### Step 3: MWMO/MWID/MODF Modification
```csharp
// Append new WMO names to MWMO (null-terminated strings)
// Append new offsets to MWID (uint32 offsets into MWMO)
// Append new MODF entries (64 bytes each)
// NameId in new MODF entries = existingWmoCount + index
```

#### Step 4: Rebuild with Correct Offsets
Use `WdlToAdtGenerator` pattern:
```csharp
// Write chunks in order: MVER, MHDR, MCIN, MTEX, MMDX, MMID, MWMO, MWID, MDDF, MODF, [MH2O], MCNK×256, [MFBO], [MTXF]
// Track positions as we write
// Go back and fix MHDR offsets (relative to MHDR data start)
// Go back and fix MCIN offsets (absolute file positions)
```

#### Step 5: Only Patch Tiles That Need It
```csharp
// Check if tile has PM4 MODF entries for its coordinates
// If no entries for this tile → copy file unchanged
// If has entries → parse, modify, rebuild
```

### Key Files to Reference
- `WoWRollback.PM4Module/WdlToAdtTest.cs` - Shows correct MHDR offset calculation
- `src/gillijimproject-csharp/WowFiles/LichKing/AdtLk.cs` - Shows chunk parsing pattern
- `pm4-adt-test12/modf_reconstruction/modf_entries.csv` - PM4 MODF data to inject
- `WoWRollback.AdtModule/` - End-to-end Alpha→LK ADT conversion using known-good WowFiles writers (baseline for LK ADT structure)
- `WoWRollback.LkToAlphaModule/` - LK↔Alpha ADT/WDT models and writers (reference for placements and liquids wiring, even if coords need tuning)

### MHDR Offset Calculation (from WdlToAdtTest.cs)
```csharp
int mhdrDataStart = (int)mhdrPos + 8;  // After MHDR chunk header
// Offsets are relative to 0x14 (mhdrDataStart)
BitConverter.GetBytes((uint)(mcinPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x04);
BitConverter.GetBytes((uint)(mtexPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x08);
// ... etc for each chunk
```

### Future Work (After WMO Pipeline Works)
- M2/MDX placements from PM4 data
- Multiple matching passes for unmatched PM4 objects
- MCCV blending for tiles without textures

---

## Completed This Session (Dec 10-11, 2025)
- ✅ Fixed MCCV vertex layout (interleaved format matching MCVT)
- ✅ Created `.windsurf/rules/data-paths.md` with all fixed paths
- ✅ Verified PM4 reconstruction pipeline works end-to-end
- ✅ Added `ExportVerificationJson()` to `Pm4ModfReconstructor.cs`
- ✅ Added `verify-pm4-data` and `csv-to-json` CLI commands
- ✅ Generated full verification: **1101 matched placements**, 351 WMOs, 163 tiles
- ✅ Output: `pm4_full_verification.json` proves data generation works
- ✅ Tested AdtModfInjector (binary append) - FAILED

## Next Session: Test MuseumAdtPatcher Outputs
The PM4→WMO matching is **proven working** (1101 placements). The chunk-preserving ADT patcher is now implemented (`WoWRollback.PM4Module/MuseumAdtPatcher.cs`) and wired into the `inject-modf` CLI using `AdtPatcher`'s manual write path.

Next step is to run `inject-modf` over a small tile subset, diff patched ADTs against known-good LK outputs (AdtModule) and Noggit, and confirm MWMO/MWID/MODF are correct while MCNK and other chunks remain byte-stable.
