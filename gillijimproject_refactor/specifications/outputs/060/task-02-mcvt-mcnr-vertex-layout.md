# Task 2: MCVT/MCNR Vertex Layout

**Binary**: WoWClient.exe (Alpha 0.6.0 build 3592)
**Analysis Date**: 2026-02-09

---

## Goal

Determine if heights/normals are interleaved or non-interleaved.

---

## Key Findings

### Format Determination: **Non-Interleaved (Alpha-Style)**

---

## Evidence

### 1. MCVT Processing Function

**Function**: [`FUN_006a7d20`](0x006a7d20)
**Address**: `0x006a7d20`

This function processes the MCVT (height map) data. Analysis of the loop structure:

```c
// Outer loop: 9 iterations (rows 0-8)
local_c = 0;
do {
    // Inner loop: 9 iterations (columns 0-8)
    local_8 = 0;
    do {
        *pfVar9 = fVar2;
        pfVar9[1] = (float)local_8 * fVar4;
        pfVar9[2] = *pfVar8;  // Read height value
        // ...
        local_8 = local_8 + 1;
        pfVar8 = pfVar8 + 1;  // Sequential read
        pfVar9 = pfVar9 + 3;
    } while (local_8 < 9);
    
    // Inner vertices (8x8) between outer rows
    if (local_c < 8) {
        local_8 = 0;
        do {
            // Process inner row vertices
            pfVar8 = pfVar7 + 1;  // Sequential read continues
            // ...
        } while (local_8 < 8);
    }
    local_c = local_c + 1;
} while (local_c < 9);
```

The key observation is `pfVar8 = pfVar8 + 1` - heights are read **sequentially** as floats, not interleaved.

### 2. MCNR Processing Function

**Function**: [`FUN_006a7490`](0x006a7490)
**Address**: `0x006a7490`

```c
iVar3 = 0x91;  // 145 iterations (81 outer + 64 inner)
pcVar1 = *(char **)(param_1 + 0xef4);  // MCNR data pointer
pfVar2 = (float *)(param_1 + 0x14c);   // Output buffer
do {
    iVar3 = iVar3 + -1;
    *pfVar2 = (float)(int)*pcVar1 * _DAT_0081cd3c;       // X normal
    pfVar2[1] = (float)(int)pcVar1[1] * _DAT_0081cd3c;   // Y normal
    pfVar2[2] = (float)(int)pcVar1[2] * _DAT_0081cd3c;   // Z normal
    pcVar1 = pcVar1 + 3;  // Sequential read, 3 bytes per normal
    pfVar2 = pfVar2 + 3;
} while (iVar3 != 0);
```

The MCNR data is read **sequentially** - 145 normals of 3 bytes each = 435 bytes total.

### 3. Chunk Validation in MCNK Parser

**Function**: [`FUN_006a6d00`](0x006a6d00)
**Address**: `0x006a6d00`

The MCNK chunk parser validates separate MCVT and MCNR chunks:

```c
// MCVT chunk at offset 0x14 in MCNK header
if (*(int *)(iVar1 + *(int *)(iVar1 + 0x1c)) != 0x4d435654) {  // 'MCVT'
    // Error: expected MCVT
}
*(int *)(param_1 + 0xef0) = *(int *)(param_1 + 0xee4) + 0x14 + 8 + *(int *)(param_1 + 0x138);

// MCNR chunk at offset 0x18 in MCNK header  
if (*(int *)(*(int *)(param_1 + 0x138) + *(int *)(*(int *)(param_1 + 0xee4) + 0x18)) != 0x4d434e52) {  // 'MCNR'
    // Error: expected MCNR
}
*(int *)(param_1 + 0xef4) = *(int *)(param_1 + 0xee4) + 0x18 + 8 + *(int *)(param_1 + 0x138);
```

The MCVT and MCNR are stored as **separate, contiguous chunks** - not interleaved.

### 4. Vertex Count Analysis

| Component | Count | Size | Total |
|-----------|-------|------|-------|
| Outer vertices (9x9) | 81 | 4 bytes | 324 bytes |
| Inner vertices (8x8) | 64 | 4 bytes | 256 bytes |
| **MCVT Total** | 145 | 4 bytes | **580 bytes** |
| MCNR normals | 145 | 3 bytes | **435 bytes** |

---

## Corresponding 0.5.3 Functions

| 0.6.0 Address | Function Purpose | 0.5.3 Reference |
|---------------|------------------|-----------------|
| 0x006a7d20 | MCVT Processing | `CMapChunk::ProcessHeights` |
| 0x006a7490 | MCNR Processing | `CMapChunk::ProcessNormals` |
| 0x006a6d00 | MCNK Validation | `CMapChunk::ValidateChunks` |

---

## Conclusion

**0.6.0 uses non-interleaved vertex layout (Alpha-style)**:
- MCVT: 145 floats stored sequentially (81 outer + 64 inner heights)
- MCNR: 145 normals stored sequentially (3 bytes each = 435 bytes)

The interleaved format (where 9-row and 8-row data alternates) had **NOT** been adopted yet in 0.6.0.

---

## Confidence Level: **HIGH**

- Clear sequential read patterns in decompiled code
- Loop counters match expected non-interleaved counts (145 total)
- Separate chunk validation confirms non-interleaved storage
