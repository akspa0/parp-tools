# MCAL Alpha Map Data Orientation Bug

## Date: 2025-10-20

## Observed Symptom

**Test Case**: Gray dot painted in center of depressed terrain area

**Expected Result**: Gray dot appears as circular texture in center

**Actual Result**: Gray dot appears as **two half-circles with arcs back-to-back**

## Analysis

This is a **classic data orientation bug** - the alpha map is being read/written with incorrect row/column ordering or byte sequence.

### Visual Representation

```
Expected (gray dot):        Actual (two half-circles):
    --------                    --------
    ---XX---                    X------X
    --XXXX--                    XX----XX
    --XXXX--                    XX----XX
    ---XX---                    X------X
    --------                    --------
```

The pattern suggests the 64×64 alpha map data is being:
1. **Split in half** (at row 32 or column 32)
2. **Each half flipped** horizontally
3. **Recombined** in wrong order

### Possible Root Causes

#### 1. Row Order Reversal
Alpha maps are stored "left to right, top to bottom" but might be read/written in opposite order:

```csharp
// WRONG: Reading bottom-to-top instead of top-to-bottom
for (int row = 63; row >= 0; row--) {
    for (int col = 0; col < 64; col++) {
        alpha[row * 64 + col] = data[offset++];
    }
}

// CORRECT: Reading top-to-bottom
for (int row = 0; row < 64; row++) {
    for (int col = 0; col < 64; col++) {
        alpha[row * 64 + col] = data[offset++];
    }
}
```

#### 2. Byte Order Issues (4-bit packed format)

For 2048-byte format, each byte contains TWO 4-bit values:

```csharp
// WRONG: Reading high nibble first when should read low nibble first
byte b = data[offset];
int value1 = (b >> 4) & 0x0F;  // High nibble
int value2 = b & 0x0F;          // Low nibble

// CORRECT: LSB first order
byte b = data[offset];
int value1 = b & 0x0F;          // Low nibble FIRST
int value2 = (b >> 4) & 0x0F;  // High nibble SECOND
```

#### 3. Offset Calculation Error

If MCAL offset is wrong by 2048 bytes (half the alpha map), we'd read from middle of data:

```
Correct:  [----LAYER1----][----LAYER2----]
           ^start here

Wrong:    [----LAYER1----][----LAYER2----]
                         ^start here (middle of Layer1!)
```

#### 4. Alpha Map "Fix" Flag Confusion

The `FLAG_DO_NOT_FIX_ALPHA_MAP` affects whether map is 63×63 or 64×64:

**Fixed (63×63):**
```c
struct { 
    uint4_t alpha_map[63]; 
    uint4_t ignored; 
}[63]; 
uint4_t ignored[64];

// Last row/column replicated:
alpha_map[x][63] == alpha_map[x][62]
alpha_map[63][x] == alpha_map[62][x]
```

If we read 63×63 data as 64×64, or vice versa, we get misaligned data causing visual distortion.

### Most Likely Culprit

Given the "two half-circles back-to-back" pattern, the most likely cause is:

**MCLY `offsetInMCAL` calculation is wrong**, causing each layer's alpha map to start at the wrong position in the MCAL chunk.

```csharp
// WRONG: Not accounting for previous layer sizes correctly
layer[0].offsetInMCAL = 0;           // ✅ Correct
layer[1].offsetInMCAL = 2048;        // ❌ WRONG if layer 0 is actually 4096 bytes!
// Now layer 1 reads from MIDDLE of layer 0's data!

// CORRECT: Calculate based on actual sizes
layer[0].offsetInMCAL = 0;
layer[1].offsetInMCAL = layer[0].actualSize; // Could be 2048 OR 4096!
```

## Code Areas to Check

### 1. AlphaMcnkBuilder.cs - MCLY offsetInMCAL Calculation

When building MCLY, verify each layer's `offsetInMCAL` accounts for:
- Previous layers' actual sizes (not assumed sizes)
- Compression flags affecting size (2048 vs 4096)
- Padding between layers (must be even-byte aligned)

### 2. MCAL Extraction from LK ADT

When extracting MCAL from LK:
```csharp
// Check: Are we stripping the 8-byte header correctly?
byte[] mcalWhole = ExtractSubChunk(lkData, header.ofsAlpha);
int size = BitConverter.ToInt32(mcalWhole, 4);
byte[] mcalRaw = new byte[size];
Buffer.BlockCopy(mcalWhole, 8, mcalRaw, 0, size); // Skip 8-byte header

// Check: Is 'size' correct? Should match header.sizeAlpha
if (size != header.sizeAlpha) {
    // PROBLEM: Size mismatch indicates header/data confusion
}
```

### 3. MCAL Write to Alpha WDT

When writing MCAL to Alpha:
```csharp
// Check: Are we writing raw data at correct offset?
int offsAlpha = offsShadow + mcshRaw.Length;  // Should be raw size, not whole size!

// Check: Are we writing the data itself correctly?
if (mcalRaw.Length > 0) {
    ms.Write(mcalRaw, 0, mcalRaw.Length);  // Must be raw data, NO header!
}
```

### 4. MCLY Texture Layer Order

Verify layers are written in correct order:
```csharp
// Layer 0: Base texture (no alpha map)
// Layer 1: First blend texture (has alpha map at offset 0)
// Layer 2: Second blend texture (has alpha map at offset [layer1_size])
// Layer 3: Third blend texture (has alpha map at offset [layer1_size + layer2_size])
```

## Testing Strategy

### 1. Create Minimal Test Case

Build a LK ADT with:
- 1 base texture (layer 0)
- 1 blend texture (layer 1) with simple pattern:
  - Top half: fully opaque (255)
  - Bottom half: fully transparent (0)

Convert to Alpha and verify the split is horizontal, not scrambled.

### 2. Dump MCAL Binary Data

Add logging to dump raw MCAL bytes:
```csharp
File.WriteAllBytes($"mcal_lk_{x}_{y}.bin", mcalRawFromLK);
File.WriteAllBytes($"mcal_alpha_{x}_{y}.bin", mcalRawToAlpha);
```

Compare byte-by-byte to see if data is identical or transformed.

### 3. Visual Pattern Test

Create test patterns:
- **Horizontal stripes**: Tests row order
- **Vertical stripes**: Tests column order  
- **Checkerboard**: Tests both axes
- **Gradient**: Tests byte order within pixels

### 4. Real Alpha Comparison

Extract MCAL from authentic 0.5.3 Alpha WDT:
```csharp
var realAlphaMCAL = ExtractMCALFromRealAlphaWDT("RazorfenDowns.wdt", tileX, tileY);
var ourAlphaMCAL = ConvertLKtoAlphaMCAL(lkADT, tileX, tileY);

// Compare byte-by-byte
if (!realAlphaMCAL.SequenceEqual(ourAlphaMCAL)) {
    // Find first difference
    for (int i = 0; i < Math.Min(realAlphaMCAL.Length, ourAlphaMCAL.Length); i++) {
        if (realAlphaMCAL[i] != ourAlphaMCAL[i]) {
            Console.WriteLine($"First difference at byte {i}");
            Console.WriteLine($"Real: {realAlphaMCAL[i]:X2}, Ours: {ourAlphaMCAL[i]:X2}");
            break;
        }
    }
}
```

## Immediate Action

Since we already identified the header issue in `AlphaMcnkBuilder.cs`, let's verify the MCAL data is actually being copied correctly:

### Check Current Code (Lines 220-232)

```csharp
// Build MCAL raw - use extracted LK data or create empty fallback
byte[] mcalRaw;
if (mcalLkWhole != null && mcalLkWhole.Length > 8)
{
    int sz = BitConverter.ToInt32(mcalLkWhole, 4);
    mcalRaw = new byte[sz];
    Buffer.BlockCopy(mcalLkWhole, 8, mcalRaw, 0, sz);
    DumpMcalData("lk", lkHeader.IndexX, lkHeader.IndexY, mcalRaw, opts);
}
else
{
    mcalRaw = Array.Empty<byte>();
}
```

**Potential Issues:**
1. Is `mcalLkWhole` extracted with correct offset from LK ADT?
2. Is the size `sz` matching what we expect?
3. Are we checking if the extracted data is actually valid alpha map data?

### Add Validation

```csharp
// After extraction, validate the data
if (mcalRaw.Length > 0) {
    // Check if size is reasonable for alpha maps
    // Typical sizes: 2048 (4-bit), 4096 (8-bit), or compressed (128-4160)
    if (mcalRaw.Length != 2048 && mcalRaw.Length != 4096 && 
        (mcalRaw.Length < 128 || mcalRaw.Length > 4160)) {
        Console.WriteLine($"[WARNING] Unusual MCAL size: {mcalRaw.Length} bytes at tile {lkHeader.IndexX},{lkHeader.IndexY}");
    }
    
    // If we have MCLY data, verify offsetInMCAL values are within bounds
    if (mclyRaw.Length > 0) {
        int numLayers = mclyRaw.Length / 16;
        for (int i = 1; i < numLayers; i++) { // Skip layer 0 (base, no alpha)
            int offsetInMCAL = BitConverter.ToInt32(mclyRaw, i * 16 + 8);
            if (offsetInMCAL >= mcalRaw.Length) {
                Console.WriteLine($"[ERROR] MCLY layer {i} offsetInMCAL ({offsetInMCAL}) exceeds MCAL size ({mcalRaw.Length})");
            }
        }
    }
}
```

## Expected Fix

Once we fix the header issue (MCSH/MCAL/MCSE should not have headers in Alpha), we should also verify:

1. **MCAL data is byte-identical** between LK source and Alpha output
2. **MCLY offsetInMCAL values** are correct for each layer
3. **Data orientation** is preserved (no flips, no reversals)

If the gray dot still appears as two half-circles after fixing headers, then we have a **data orientation** bug that requires deeper investigation of the coordinate system and read/write order.

## Next Debugging Step

Enable verbose logging and capture:
```powershell
dotnet run --project WoWRollback.AdtConverter -- pack-monolithic \
  --lk-dir path\to\test\map \
  --lk-wdt path\to\test.wdt \
  --map TestMap \
  --verbose-logging \
  2>&1 | Tee-Object -FilePath conversion.log
```

Then examine:
1. MCAL sizes being extracted
2. MCLY offsetInMCAL values
3. Any warnings about unusual data

The log will tell us if the problem is in extraction or in writing.
