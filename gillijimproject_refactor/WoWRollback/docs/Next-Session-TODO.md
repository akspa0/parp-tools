# Next Session TODO - MCNK Sub-Chunk Data Fix

## Current Status (Session 2025-10-15 End)

### ✅ WDT Structure - COMPLETELY FIXED
- ✅ MAIN chunk properly patched with tile offsets
- ✅ MHDR offsets use correct Alpha convention (point to FourCC, not data)
- ✅ MDNM/MONM/MODF written as empty chunks (client expects them)
- ✅ All validation passes
- ✅ RazorfenDowns loads successfully in client

### ❌ MCNK Sub-Chunks - CURRENT ISSUE
**Error:** `ERROR #132 (0x85100084) index (0x4D48452), array size (0x00000004)`

**Root Cause Identified:** MCNR (normals) sub-chunk is all zeros
- **File:** `AlphaMcnkBuilder.cs` line 66
- **Problem:** `mcnrRaw = new byte[448]` creates empty array but never populates it
- **Impact:** Client reads zeros/garbage → offset calculation fails → crash

### Why RazorfenDowns Works But Kalidar Fails
- **RazorfenDowns:** Terrain is flat/simple, zeros might be acceptable
- **Kalidar:** Complex terrain with mountains, needs real normal data

---

## 🎯 Action Plan for Next Session

### PRIORITY 1: Extract and Convert MCNR from LK MCNK

**File:** `WoWRollback.LkToAlphaModule/Builders/AlphaMcnkBuilder.cs`
**Location:** Around lines 36-66

#### Current Code (BROKEN):
```csharp
// Line 66 - creates empty array
var mcnrRaw = new byte[448]; // 145*3 + 13 pad
```

#### What We Need:
1. **Extract MCNR chunk from LK MCNK** (similar to MCVT extraction at lines 36-43)
2. **Convert normal vertex order:**
   - LK format: Interleaved outer/inner (9 outer, 8 inner, 9 outer, 8 inner...)
   - Alpha format: All outer first (81 normals), then all inner (64 normals)
3. **Add 13-byte padding** at end
4. **Verify total size:** 145 normals × 3 bytes + 13 pad = 448 bytes

#### Code Template:
```csharp
// After line 43 where MCVT is extracted, add:

// Find MCNR chunk
if (fcc == "RNCM") // 'MCNR' reversed
{
    mcnrLkWhole = new byte[8 + size + ((size & 1) == 1 ? 1 : 0)];
    Buffer.BlockCopy(lkAdtBytes, p, mcnrLkWhole, 0, mcnrLkWhole.Length);
    break;
}

// Then convert MCNR order
if (mcnrLkWhole != null)
{
    var lkData = new byte[BitConverter.ToInt32(mcnrLkWhole, 4)];
    Buffer.BlockCopy(mcnrLkWhole, 8, lkData, 0, lkData.Length);
    mcnrRaw = ConvertMcnrLkToAlpha(lkData);
}
else
{
    mcnrRaw = new byte[448]; // fallback to zeros if MCNR missing
}
```

#### Add Conversion Function:
```csharp
private static byte[] ConvertMcnrLkToAlpha(byte[] lkData)
{
    // LK: 145 normals interleaved (9-8-9-8...)
    // Alpha: 81 outer, then 64 inner + 13 pad
    var alpha = new byte[448]; // 145*3 + 13
    
    // TODO: Implement vertex reordering
    // For now, direct copy as first attempt:
    Buffer.BlockCopy(lkData, 0, alpha, 0, Math.Min(lkData.Length, 435));
    
    return alpha;
}
```

### PRIORITY 2: Verify All Sub-Chunk Formats

**Reference:** `z_wowdev.wiki/Alpha.md` lines 109-163

#### Sub-chunks WITHOUT chunk header (raw data only):
- ✅ **MCVT** (heights) - 145 floats - DONE
- ❌ **MCNR** (normals) - 145×3 bytes + 13 pad - **FIX THIS**
- ❓ **MCSH** (shadows) - verify if present/needed
- ❓ **MCAL** (alpha maps) - currently empty, verify format
- ❓ **MCLQ** (liquid) - not implemented
- ❓ **MCSE** (sound emitters) - not implemented

#### Sub-chunks WITH chunk header (8 bytes + data):
- ✅ **MCLY** (texture layers) - Currently one empty layer
- ✅ **MCRF** (references) - Currently empty
- ✅ **MCAL** (alpha map chunk wrapper) - Currently empty

### PRIORITY 3: Test Incremental Changes

1. **Test 1:** Extract MCNR but don't reorder → does it work?
2. **Test 2:** Add proper vertex reordering → lighting correct?
3. **Test 3:** If still crashes, check MCLY (texture layers)
4. **Test 4:** Compare byte-by-byte with working Alpha WDT

---

## 🛠️ Implementation Steps

### Step 1: Add MCNR Extraction
**Time estimate:** 15 minutes

1. Open `AlphaMcnkBuilder.cs`
2. Find the MCVT extraction code (lines 36-43)
3. Duplicate and adapt for MCNR (look for 'RNCM' FourCC)
4. Build and test

### Step 2: Add Conversion Function
**Time estimate:** 30 minutes

1. Create `ConvertMcnrLkToAlpha()` function
2. Implement vertex reordering logic
3. Add 13-byte padding
4. Test with simple direct copy first

### Step 3: Test and Validate
**Time estimate:** 15 minutes

1. Pack Kalidar with new code
2. Test in Alpha client
3. If crashes, add debug output to verify MCNR data
4. Compare with working RazorfenDowns

### Step 4: Refine if Needed
**Time estimate:** Variable

- Add proper vertex reordering if direct copy doesn't work
- Verify normal vector calculations
- Check for endianness issues

---

## ✅ Success Criteria

1. ✅ Kalidar loads in Alpha 0.5.3 client without crash
2. ✅ Terrain visible with mountains/valleys
3. ✅ Lighting looks correct (normals working)
4. ✅ All 55 tiles render properly
5. ✅ RazorfenDowns still works

---

## 📚 Key References

### Documentation
- **Alpha format spec:** `z_wowdev.wiki/Alpha.md` lines 101-163
- **MCNK structure:** Line 106 shows header format
- **MCNR details:** Lines 119-127 (no header, 145×3 + 13 pad)
- **Vertex ordering:** Line 113 explains outer-first, then inner

### Code Files
- **Main builder:** `AlphaMcnkBuilder.cs`
- **WDT writer:** `AlphaWdtMonolithicWriter.cs`
- **Validation:** `AlphaWdtInspector.cs`

### Tools
- `validate-wdt` - Structure validation
- `inspect-alpha` - Deep WDT inspection
- `compare-alpha` - Byte-level comparison

---

## 🎉 Expected Outcome

After fixing MCNR:
- Kalidar should load successfully
- Terrain will be visible with proper lighting
- We can then add texture layers (MCLY) in a future session
- Full Alpha WDT conversion pipeline complete!
