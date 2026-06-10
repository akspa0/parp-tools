# MCNK Header Field Comparison: Real vs Packed

## Summary

Comparing MCNK[0] headers from:
- **Real Alpha**: RazorfenDowns.wdt (authentic 0.5.3 file)
- **Our Packed**: Kalidar/UnderMine.wdt (our generated file)

## Side-by-Side Comparison

| Field | Real Alpha | Our Packed | Status | Notes |
|-------|------------|------------|--------|-------|
| **Flags** | 0x00000000 | 0x00000000 | ✅ MATCH | |
| **IndexX** | 0 | 0 | ✅ MATCH | |
| **IndexY** | 0 | 0 | ✅ MATCH | |
| **Radius** | 23.570227 | 0.0 | ❌ WRONG | We're writing 0! |
| **NLayers** | 0 | 1 | ❌ WRONG | We have 1, real has 0 |
| **NDoodadRefs** | 0 | 0 | ✅ MATCH | |
| **McvtOffset** | 0 | 0 | ✅ MATCH | MCVT starts immediately |
| **McnrOffset** | 580 | 580 | ✅ MATCH | 145 floats * 4 = 580 |
| **MclyOffset** | 1028 | 1028 | ✅ MATCH | |
| **McrfOffset** | 1036 | 1052 | ❌ WRONG | Off by 16 bytes |
| **McalOffset** | 1044 | 1068 | ❌ WRONG | Off by 24 bytes |
| **McalSize** | 0 | 0 | ✅ MATCH | |
| **McshOffset** | 1044 | 1060 | ❌ WRONG | Off by 16 bytes |
| **McshSize** | 0 | 0 | ✅ MATCH | |
| **AreaId** | 0x00000000 | 0x00000000 | ✅ MATCH | |
| **NMapObjRefs** | 0 | 0 | ✅ MATCH | |
| **Holes** | 0x0000 | 0x0000 | ✅ MATCH | |
| **PredTex[8]** | all 0 | all 0 | ✅ MATCH | |
| **NoEffectDoodad[8]** | all 0 | all 0 | ✅ MATCH | |
| **McseOffset** | 1044 | 1076 | ❌ WRONG | Off by 32 bytes |
| **NSndEmitters** | 0 | 0 | ✅ MATCH | |
| **MclqOffset** | 1044 | 0 | ❌ WRONG | We're writing 0! |

## Critical Issues Found

### 1. **Radius = 0** ❌
**Problem**: We're writing 0.0 for radius, real Alpha has 23.570227

**Impact**: Client might use this for culling/LOD calculations

**Fix**: Calculate proper bounding sphere radius from MCVT heights

### 2. **NLayers Mismatch** ❌
**Problem**: We have NLayers=1 (because we add MCLY), real Alpha has NLayers=0

**Impact**: This might be causing the crash! Client expects 0 layers but we say 1

**Analysis**: 
- Real Alpha MCLY has size=0 (empty chunk)
- Our MCLY has size=16 (one layer entry)
- But NLayers in header says 0 vs 1

**Hypothesis**: Maybe Alpha terrain-only tiles should have NLayers=0 with empty MCLY?

### 3. **Sub-chunk Offset Cascade** ❌
**Problem**: All offsets after MCLY are shifted

**Real Alpha offsets**:
```
McvtOffset: 0
McnrOffset: 580
MclyOffset: 1028
McrfOffset: 1036  (MCLY + 8 header + 0 data)
McalOffset: 1044  (MCRF + 8 header + 0 data)
McshOffset: 1044  (same as McalOffset - both empty)
McseOffset: 1044  (same - all empty chunks point to same spot)
MclqOffset: 1044  (same)
```

**Our packed offsets**:
```
McvtOffset: 0
McnrOffset: 580
MclyOffset: 1028
McrfOffset: 1052  (+16 bytes - we have 16-byte MCLY data!)
McalOffset: 1068  (+24 bytes)
McshOffset: 1060  (+16 bytes)
McseOffset: 1076  (+32 bytes)
MclqOffset: 0     (WRONG - should be 1076 or same as others)
```

**Root Cause**: We're writing a 16-byte MCLY entry when we should write empty MCLY!

### 4. **MclqOffset = 0** ❌
**Problem**: We're writing 0 for MclqOffset, real Alpha writes 1044 (pointing to end)

**Impact**: Client might try to read liquid data at offset 0!

## The Smoking Gun 🔫

**The crash is likely caused by**:

1. **NLayers=1 but MCLY is wrong** - Client tries to read 1 layer but data is malformed
2. **MclqOffset=0** - Client tries to read liquid at offset 0 (invalid!)
3. **Radius=0** - Client might reject chunk as invalid

## Required Fixes

### Fix 1: Empty MCLY for Terrain-Only
```csharp
// In AlphaMcnkBuilder.cs
// Don't write MCLY data for terrain-only chunks
// Write empty MCLY chunk (just 8-byte header, no data)
var mcly = new Chunk("MCLY", 0, Array.Empty<byte>());
```

### Fix 2: Set NLayers=0
```csharp
// In MCNK header
hdr.NLayers = 0; // Not 1!
```

### Fix 3: Calculate Radius
```csharp
// Calculate bounding sphere from MCVT min/max
float radius = CalculateRadius(mcvtHeights, chunkX, chunkY);
hdr.Radius = radius;
```

### Fix 4: Fix MclqOffset
```csharp
// Point to end of sub-chunks, not 0
hdr.MclqOffset = totalSubChunkSize;
```

### Fix 5: Recalculate All Offsets
After fixing MCLY to be empty, all subsequent offsets will shift back to match real Alpha.

## Next Steps

1. Fix AlphaMcnkBuilder to write empty MCLY
2. Set NLayers=0 in header
3. Calculate proper Radius
4. Fix MclqOffset to point to end
5. Repack and test

**Expected Result**: Offsets will match real Alpha exactly, crash should be fixed!
