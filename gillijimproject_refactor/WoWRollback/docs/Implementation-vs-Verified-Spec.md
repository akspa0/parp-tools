# Implementation Comparison: Our Code vs Verified Spec

## Summary

Comparing `AlphaWdtMonolithicWriter.cs` against verified RazorfenDowns.wdt structure.

## ✅ CORRECT Implementation

### 1. MAIN.offset (Line 139)
```csharp
mhdrAbsoluteOffsets[tileIndex] = checked((int)(mhdrAbsolute));
```
**Status**: ✅ CORRECT - Points to MHDR letters

### 2. MAIN.size (Line 202)
```csharp
mhdrToFirstMcnkSizes[tileIndex] = checked((int)(firstMcnkAbsolute - mhdrAbsolute));
```
**Status**: ✅ CORRECT - Distance from MHDR letters to first MCNK

### 3. MHDR Offset Base (Line 179-197)
```csharp
long mhdrDataStart = mhdrAbsolute + 8;
// All offsets written relative to mhdrDataStart
```
**Status**: ✅ CORRECT - Offsets relative to MHDR.data

## ❌ CRITICAL BUG FOUND

### MHDR.offsInfo Calculation (Line 180)

**Our Code**:
```csharp
int offsTexRel = 64 + mcinChunkLen;  // Line 180
// offsInfo is NOT written! It's missing!
```

**Problem**: We calculate `offsTexRel` but **never write offsInfo field**!

**Verified Spec**:
```
offsInfo = 64 (always)
```

**Impact**: The MHDR.data[0..3] field (offsInfo) is likely zero or garbage, causing the client to fail when trying to locate MCIN!

### Missing Code

We need to add BEFORE line 182:
```csharp
ms.Position = mhdrDataStart + 0; // offsInfo
ms.Write(BitConverter.GetBytes(64));
```

## ✅ Other Calculations CORRECT

### offsTex Calculation (Line 180)
```csharp
int offsTexRel = 64 + mcinChunkLen;
// mcinChunkLen = 8 + 4096 = 4104
// offsTexRel = 64 + 4104 = 4168 ✓
```
**Status**: ✅ CORRECT - Matches verified value

### offsDoo Calculation (Line 187)
```csharp
int offsDooRel = offsTexRel + mtexChunkLen;
// With minimal MTEX: 4168 + 16 = 4184
// But we're using longer texture path...
```
**Status**: ⚠️ DEPENDS on mtexData.Length

### offsMob Calculation (Line 193)
```csharp
int offsMobRel = offsDooRel + mddfChunkLen;
```
**Status**: ✅ CORRECT logic

## 🔍 Secondary Issue: MTEX Size

**Our Code** (Line 151-152):
```csharp
var baseTexturePath = "Tileset\\Generic\\Checkers.blp";
var mtexData = Encoding.ASCII.GetBytes(baseTexturePath + "\0");
// Length = 31 bytes
```

**Verified Spec**:
```
sizeTex = 8 bytes (minimal)
```

**Analysis**: RazorfenDowns uses a very minimal MTEX. Our texture path is longer, which is fine, but means our offsets will differ:

```
Our offsTex  = 64 + 4104 = 4168 ✓ (same)
Our sizeTex  = 31 bytes (vs 8 in real file)
Our offsDoo  = 4168 + 8 + 31 = 4207 (vs 4176 in real file)
Our offsMob  = 4207 + 8 = 4215 (vs 4184 in real file)
```

This is **acceptable** as long as the chunks are actually written in the correct order.

## 🎯 ROOT CAUSE OF CRASH

**The crash at index 0x4D484452 ('RDHM') is caused by**:

The client reads MHDR.data[0..3] expecting `offsInfo=64`, but we never wrote it!

The client then tries to use this garbage/zero value to locate MCIN, fails, and crashes with the MHDR letters value in the crash dump.

## 🔧 Required Fix

### File: AlphaWdtMonolithicWriter.cs

**Add after line 179**:
```csharp
long mhdrDataStart = mhdrAbsolute + 8;

// FIX: Write offsInfo field (CRITICAL!)
ms.Position = mhdrDataStart + 0;
ms.Write(BitConverter.GetBytes(64));

int offsTexRel = 64 + mcinChunkLen;
long save = ms.Position;
```

This single missing field write is likely the entire cause of the client crash!

## Verification Plan

After fix:
1. Repack WDT with fix
2. Run inspector on our packed WDT
3. Compare MHDR fields with real RazorfenDowns
4. Verify offsInfo=64 in output
5. Test in client
