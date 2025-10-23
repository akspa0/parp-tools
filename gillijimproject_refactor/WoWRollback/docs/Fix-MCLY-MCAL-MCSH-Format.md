# Fix: MCLY/MCAL/MCSH Chunk Format Correction

## Date: 2025-10-16

## Problem
Textures were appearing in wrong chunks - road textures showing up as disjointed patches instead of following the road bed. This indicated chunk attribution problems where MCLY/MCAL/MCSH data was being written in the wrong format or to the wrong tiles.

## Root Cause
The LK→Alpha converter was incorrectly handling sub-chunk formats:

1. **Incorrect Format**: MCSH, MCAL, and MCSE were being written with chunk headers (FourCC + size), but Alpha format expects them as **raw data only**
2. **Strip-then-rewrap**: Code was extracting chunks from LK, stripping headers, then creating NEW wrappers - potentially losing tile attribution
3. **Offset Miscalculation**: Offsets in MCNK header were calculated assuming all chunks had headers

## Alpha Format Specification

Based on `Alpha.md` and `McnkAlpha.cs`:

| Chunk | Has Header? | Format |
|-------|-------------|--------|
| MCVT  | ❌ No       | Raw data (580 bytes) |
| MCNR  | ❌ No       | Raw data (448 bytes) |
| MCLY  | ✅ Yes      | FourCC + size + data |
| MCRF  | ✅ Yes      | FourCC + size + data |
| MCSH  | ❌ No       | Raw data (size from header) |
| MCAL  | ❌ No       | Raw data (size from header) |
| MCSE  | ❌ No       | Raw data (size from header) |

## Changes Made

### 1. Fixed Chunk Writing Format (`AlphaMcnkBuilder.cs`)

**Before:**
```csharp
// All chunks wrapped with headers
var mcshChunk = new Chunk("MCSH", mcshRaw.Length, mcshRaw);
var mcalChunk = new Chunk("MCAL", mcalRaw.Length, mcalRaw);
var mcseChunk = new Chunk("MCSE", mcseRaw.Length, mcseRaw);

byte[] mcshWhole = mcshChunk.GetWholeChunk();  // Includes 8-byte header
byte[] mcalWhole = mcalChunk.GetWholeChunk();  // Includes 8-byte header
byte[] mcseWhole = mcseChunk.GetWholeChunk();  // Includes 8-byte header
```

**After:**
```csharp
// Only MCLY and MCRF have headers in Alpha format
var mclyChunk = new Chunk("MCLY", mclyRaw.Length, mclyRaw);
var mcrfChunk = new Chunk("MCRF", mcrfRaw.Length, mcrfRaw);

byte[] mclyWhole = mclyChunk.GetWholeChunk();  // Has header
byte[] mcrfWhole = mcrfChunk.GetWholeChunk();  // Has header
// MCSH, MCAL, MCSE written as raw data (no wrappers)
```

### 2. Fixed Offset Calculations

**Before:**
```csharp
int offsShadow = offsRefs   + mcrfWhole.Length;
int offsAlpha  = offsShadow + mcshWhole.Length;  // Wrong - includes header
int offsSnd    = offsAlpha  + mcalWhole.Length;  // Wrong - includes header
```

**After:**
```csharp
int offsShadow = offsRefs   + mcrfWhole.Length;
int offsAlpha  = offsShadow + mcshRaw.Length;  // Correct - raw data only
int offsSnd    = offsAlpha  + mcalRaw.Length;  // Correct - raw data only
```

### 3. Fixed Write Operations

**Before:**
```csharp
ms.Write(mcshWhole, 0, mcshWhole.Length);  // Writes header + data
ms.Write(mcalWhole, 0, mcalWhole.Length);  // Writes header + data
ms.Write(mcseWhole, 0, mcseWhole.Length);  // Writes header + data
```

**After:**
```csharp
if (mcshRaw.Length > 0) ms.Write(mcshRaw, 0, mcshRaw.Length);  // Raw data only
if (mcalRaw.Length > 0) ms.Write(mcalRaw, 0, mcalRaw.Length);  // Raw data only
if (mcseRaw.Length > 0) ms.Write(mcseRaw, 0, mcseRaw.Length);  // Raw data only
```

### 4. Added Attribution Logging

Added `VerboseLogging` option to track MCNK chunk attribution:
- Logs which IndexX/IndexY is being built from which offset
- Logs MCLY extraction (number of layers, texture IDs)
- Helps debug if chunks are being written to wrong tiles

## Testing Instructions

### 1. Build and Pack
```powershell
cd WoWRollback

# Build
dotnet build WoWRollback.LkToAlphaModule\WoWRollback.LkToAlphaModule.csproj

# Pack with verbose logging
dotnet run --project WoWRollback.AdtConverter -- pack-monolithic `
  --lk-dir ..\test_data\0.6.0\tree\World\Maps\Kalidar\ `
  --lk-wdt ..\test_data\0.6.0\tree\World\Maps\Kalidar\Kalidar.wdt `
  --map Kalidar `
  --verbose-logging
```

### 2. Verify Output
```powershell
# Check file size (should be ~40-41 MB, not 17 MB)
Get-ChildItem "project_output\Kalidar_*\Kalidar.wdt" | Select-Object Name, Length

# Inspect structure
dotnet run --project WoWRollback.AdtConverter -- inspect-alpha `
  --wdt project_output\Kalidar_20251016_XXXXXX\Kalidar.wdt --tiles 3
```

### 3. Test in Client
1. Copy `Kalidar.wdt` to `Data\World\Maps\Kalidar\`
2. Launch Alpha 0.5.3 client
3. Verify:
   - ✅ No ERROR #132 crash
   - ✅ Textures appear in correct locations
   - ✅ Roads follow the road bed properly
   - ✅ No disjointed texture patches

### 4. Compare with Real Alpha Files
```powershell
# Compare structure with real 0.5.3 Kalidar
dotnet run --project WoWRollback.AdtConverter -- compare-alpha `
  --wdt1 ..\test_data\0.5.3\tree\World\Maps\Kalidar\Kalidar.wdt `
  --wdt2 project_output\Kalidar_20251016_XXXXXX\Kalidar.wdt
```

## Expected Results

### File Size
- **Before**: ~17 MB (missing texture data)
- **After**: ~40-41 MB (complete with MCLY/MCAL/MCSH)

### MCNK Structure
Each MCNK should now have:
- MCVT: 580 bytes (raw)
- MCNR: 448 bytes (raw)
- MCLY: 8 + N*16 bytes (with header, N layers)
- MCRF: 8 + data bytes (with header)
- MCSH: 0-512 bytes (raw, shadow map)
- MCAL: 0-4096 bytes (raw, alpha maps)
- MCSE: variable bytes (raw, sound emitters)

### Offsets in MCNK Header
- `offsHeight` (0x18): Points to MCVT raw data
- `offsNormal` (0x1C): Points to MCNR raw data
- `offsLayer` (0x20): Points to MCLY FourCC (has header)
- `offsRefs` (0x24): Points to MCRF FourCC (has header)
- `offsShadow` (0x30): Points to MCSH raw data (no header)
- `offsAlpha` (0x28): Points to MCAL raw data (no header)
- `offsSndEmitters` (0x5C): Points to MCSE raw data (no header)

## Reference Files

### Source of Truth
- `src/gillijimproject-csharp/WowFiles/Alpha/McnkAlpha.cs` - Shows how Alpha format reads chunks
- `z_wowdev.wiki/Alpha.md` - Alpha format specification
- `z_wowdev.wiki/ADT_v18.md` - MCLY/MCAL/MCSH details

### Modified Files
- `WoWRollback.LkToAlphaModule/Builders/AlphaMcnkBuilder.cs` - Fixed chunk writing
- `WoWRollback.LkToAlphaModule/Models/LkToAlphaOptions.cs` - Added VerboseLogging flag

## Next Steps

If textures still appear incorrect after this fix:

1. **Check MTEX chunk**: Verify texture list is correct
2. **Check MCLY texture IDs**: Use verbose logging to see which textures are referenced
3. **Check MCAL alpha maps**: Verify alpha map data is being extracted correctly
4. **Compare byte-by-byte**: Use `compare-alpha` to find remaining differences with real Alpha files

## Notes

- This fix aligns with how `McnkAlpha.ToMcnkLk()` converts Alpha→LK (it passes chunks directly without stripping/rewrapping)
- The key insight was that Alpha format is **inconsistent**: some sub-chunks have headers (MCLY, MCRF) while others don't (MCSH, MCAL, MCSE)
- Offsets in the MCNK header must point to the correct location: FourCC for chunks with headers, raw data for chunks without
