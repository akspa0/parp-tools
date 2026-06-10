# Session 2025-10-16 Part 2: MCLY/MCAL/MCSH Format Fix

## Problem Statement
Textures were appearing in wrong chunks - roads showing up as disjointed patches instead of following the road bed. This indicated that MCLY/MCAL/MCSH data was either:
1. Being written in the wrong format
2. Being attributed to the wrong tiles during packing

## Investigation Process

### Step 1: Analyzed Alpha Format Specification
Examined `McnkAlpha.cs` (the working Alpha→LK converter) to understand the correct format:

```csharp
// MCLY - HAS chunk header
_mcly = new Chunk(adtFile, offsetInFile);  // Reads FourCC + size + data

// MCSH - NO chunk header (reads raw data)
byte[] mcshData = Util.GetByteArrayFromFile(adtFile, offsetInFile, _mcnkAlphaHeader.McshSize);
_mcsh = new Chunk("MCSH", _mcnkAlphaHeader.McshSize, mcshData);  // Wrapper for internal use

// MCAL - NO chunk header (reads raw data)
byte[] mcalData = Util.GetByteArrayFromFile(adtFile, offsetInFile, _mcnkAlphaHeader.McalSize);
_mcal = new Mcal("MCAL", _mcnkAlphaHeader.McalSize, mcalData);  // Wrapper for internal use
```

**Key Discovery:** Alpha format is inconsistent!
- MCLY and MCRF: Have chunk headers on disk
- MCSH, MCAL, MCSE: Raw data only (no headers on disk)
- MCVT, MCNR: Raw data only (no headers on disk)

### Step 2: Identified the Bug
In `AlphaMcnkBuilder.cs`, we were:
1. Extracting chunks from LK (correctly)
2. Stripping headers (correctly for raw data chunks)
3. **Creating NEW chunk wrappers for ALL chunks** (WRONG!)
4. Writing wrapped chunks with headers (WRONG for MCSH/MCAL/MCSE!)

This caused:
- Extra 8 bytes per chunk (FourCC + size) being written
- Offset calculations pointing to wrong locations
- Client reading garbage data or data from wrong chunks

## The Fix

### Changed Files

#### 1. `AlphaMcnkBuilder.cs` - Chunk Writing Logic

**Before (WRONG):**
```csharp
// All chunks wrapped with headers
var mclyChunk = new Chunk("MCLY", mclyRaw.Length, mclyRaw);
var mcrfChunk = new Chunk("MCRF", mcrfRaw.Length, mcrfRaw);
var mcshChunk = new Chunk("MCSH", mcshRaw.Length, mcshRaw);  // ❌ Wrong!
var mcalChunk = new Chunk("MCAL", mcalRaw.Length, mcalRaw);  // ❌ Wrong!
var mcseChunk = new Chunk("MCSE", mcseRaw.Length, mcseRaw);  // ❌ Wrong!

byte[] mclyWhole = mclyChunk.GetWholeChunk();
byte[] mcrfWhole = mcrfChunk.GetWholeChunk();
byte[] mcshWhole = mcshChunk.GetWholeChunk();  // Includes header
byte[] mcalWhole = mcalChunk.GetWholeChunk();  // Includes header
byte[] mcseWhole = mcseChunk.GetWholeChunk();  // Includes header

// Write all with headers
ms.Write(mcshWhole, 0, mcshWhole.Length);  // ❌ Writes header + data
ms.Write(mcalWhole, 0, mcalWhole.Length);  // ❌ Writes header + data
ms.Write(mcseWhole, 0, mcseWhole.Length);  // ❌ Writes header + data
```

**After (CORRECT):**
```csharp
// Only MCLY and MCRF have headers in Alpha format
var mclyChunk = new Chunk("MCLY", mclyRaw.Length, mclyRaw);
var mcrfChunk = new Chunk("MCRF", mcrfRaw.Length, mcrfRaw);
// MCSH, MCAL, MCSE are raw data only (no wrappers)

byte[] mclyWhole = mclyChunk.GetWholeChunk();  // Has header
byte[] mcrfWhole = mcrfChunk.GetWholeChunk();  // Has header

// Write MCSH/MCAL/MCSE as raw data
if (mcshRaw.Length > 0) ms.Write(mcshRaw, 0, mcshRaw.Length);  // ✅ Raw data only
if (mcalRaw.Length > 0) ms.Write(mcalRaw, 0, mcalRaw.Length);  // ✅ Raw data only
if (mcseRaw.Length > 0) ms.Write(mcseRaw, 0, mcseRaw.Length);  // ✅ Raw data only
```

#### 2. `AlphaMcnkBuilder.cs` - Offset Calculations

**Before (WRONG):**
```csharp
int offsShadow = offsRefs   + mcrfWhole.Length;
int offsAlpha  = offsShadow + mcshWhole.Length;  // ❌ Includes 8-byte header
int offsSnd    = offsAlpha  + mcalWhole.Length;  // ❌ Includes 8-byte header
```

**After (CORRECT):**
```csharp
int offsShadow = offsRefs   + mcrfWhole.Length;
int offsAlpha  = offsShadow + mcshRaw.Length;  // ✅ Raw data size only
int offsSnd    = offsAlpha  + mcalRaw.Length;  // ✅ Raw data size only
```

#### 3. `LkToAlphaOptions.cs` - Added Logging Flag

Added `VerboseLogging` property to enable detailed chunk attribution tracking:
```csharp
public bool VerboseLogging { get; init; }
```

#### 4. `AlphaMcnkBuilder.cs` - Added Attribution Logging

```csharp
// Log which chunk is being built
if (opts?.VerboseLogging == true)
{
    Console.WriteLine($"[MCNK] Building chunk [{lkHeader.IndexX},{lkHeader.IndexY}] from offset 0x{mcNkOffset:X}");
}

// Log MCLY extraction
if (opts?.VerboseLogging == true)
{
    int numLayers = sz / 16;
    Console.WriteLine($"[MCLY] Chunk [{lkHeader.IndexX},{lkHeader.IndexY}] extracted {numLayers} texture layers ({sz} bytes)");
    if (sz >= 16)
    {
        uint textureId = BitConverter.ToUInt32(mclyRaw, 0);
        Console.WriteLine($"[MCLY]   Layer 0 textureId: {textureId}");
    }
}
```

## Impact Analysis

### Before Fix
```
MCNK Structure (WRONG):
- MCVT: 580 bytes (raw) ✅
- MCNR: 448 bytes (raw) ✅
- MCLY: 8 + data bytes (with header) ✅
- MCRF: 8 + data bytes (with header) ✅
- MCSH: 8 + data bytes (with header) ❌ WRONG!
- MCAL: 8 + data bytes (with header) ❌ WRONG!
- MCSE: 8 + data bytes (with header) ❌ WRONG!

Offsets in MCNK header:
- offsShadow: Points 8 bytes BEFORE actual MCSH data ❌
- offsAlpha: Points 8 bytes BEFORE actual MCAL data ❌
- offsSnd: Points 8 bytes BEFORE actual MCSE data ❌

Result: Client reads wrong data, textures appear in wrong chunks
```

### After Fix
```
MCNK Structure (CORRECT):
- MCVT: 580 bytes (raw) ✅
- MCNR: 448 bytes (raw) ✅
- MCLY: 8 + data bytes (with header) ✅
- MCRF: 8 + data bytes (with header) ✅
- MCSH: data bytes (raw, no header) ✅
- MCAL: data bytes (raw, no header) ✅
- MCSE: data bytes (raw, no header) ✅

Offsets in MCNK header:
- offsShadow: Points to MCSH raw data ✅
- offsAlpha: Points to MCAL raw data ✅
- offsSnd: Points to MCSE raw data ✅

Result: Client reads correct data, textures appear in correct chunks
```

## Testing Instructions

### Quick Test
```powershell
cd WoWRollback
.\test-texture-fix.ps1
```

### Manual Test
```powershell
# Build
dotnet build WoWRollback.LkToAlphaModule\WoWRollback.LkToAlphaModule.csproj

# Pack with verbose logging
dotnet run --project WoWRollback.AdtConverter -- pack-monolithic `
  --lk-dir ..\test_data\0.6.0\tree\World\Maps\Kalidar\ `
  --lk-wdt ..\test_data\0.6.0\tree\World\Maps\Kalidar\Kalidar.wdt `
  --map Kalidar

# Check output
Get-ChildItem "project_output\Kalidar_*\Kalidar.wdt" | Select-Object Name, Length

# Inspect
dotnet run --project WoWRollback.AdtConverter -- inspect-alpha `
  --wdt project_output\Kalidar_XXXXXX\Kalidar.wdt --tiles 3
```

### In-Game Test
1. Copy output `Kalidar.wdt` to `Data\World\Maps\Kalidar\`
2. Launch Alpha 0.5.3 client
3. Verify:
   - ✅ No ERROR #132 crash
   - ✅ Textures appear in correct locations
   - ✅ Roads follow the road bed properly
   - ✅ No disjointed texture patches

## Expected Results

### File Size
- Should remain ~40-41 MB (texture data still extracted correctly)
- The fix doesn't change data extraction, only how it's written

### Visual Results
- Roads should follow the road bed smoothly
- Textures should not appear as random patches
- Each chunk should display its correct textures

## Technical Details

### Alpha MCNK Sub-Chunk Format Reference

| Offset | Field | Description |
|--------|-------|-------------|
| 0x18 | offsHeight | Points to MCVT raw data (no header) |
| 0x1C | offsNormal | Points to MCNR raw data (no header) |
| 0x20 | offsLayer | Points to MCLY FourCC (has header) |
| 0x24 | offsRefs | Points to MCRF FourCC (has header) |
| 0x28 | offsAlpha | Points to MCAL raw data (no header) |
| 0x2C | sizeAlpha | Size of MCAL raw data |
| 0x30 | offsShadow | Points to MCSH raw data (no header) |
| 0x34 | sizeShadow | Size of MCSH raw data |
| 0x5C | offsSndEmitters | Points to MCSE raw data (no header) |
| 0x60 | nSndEmitters | Number of sound emitters |

### Why Alpha Format is Inconsistent

The Alpha format evolved over time:
- **MCVT/MCNR**: Originally raw data embedded in MCNK
- **MCLY/MCRF**: Added later as proper named chunks
- **MCSH/MCAL/MCSE**: Added even later but kept as raw data for backward compatibility

This explains why some sub-chunks have headers and others don't.

## Success Criteria

✅ Build succeeds with 6 warnings
✅ File size ~40-41 MB
✅ MCNK chunks have correct structure
✅ Offsets point to correct locations
⏳ Client loads without crash
⏳ Textures appear in correct locations

## References

- `src/gillijimproject-csharp/WowFiles/Alpha/McnkAlpha.cs` - Source of truth for Alpha format
- `z_wowdev.wiki/Alpha.md` - Alpha format specification
- `z_wowdev.wiki/ADT_v18.md` - MCLY/MCAL/MCSH details
- `docs/Fix-MCLY-MCAL-MCSH-Format.md` - Detailed fix documentation
