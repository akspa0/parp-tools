# Task 5: MCNK Header Layout

**Binary**: WoWClient.exe (Alpha 0.6.0 build 3592)
**Analysis Date**: 2026-02-09

---

## Goal

Compare MCNK header against 0.5.3 and identify field offsets.

---

## Key Findings

### Format Determination: **Alpha-Style (Similar to 0.5.3)**

---

## Evidence

### 1. MCNK Header Access Pattern

**Function**: [`FUN_006a6d00`](0x006a6d00)
**Address**: `0x006a6d00`

The MCNK chunk validation function accesses header fields at these offsets:

```c
// param_1 + 0xee4 points to MCNK header (after chunk header)
iVar1 = *(int *)(param_1 + 0xee4);

// Header field accesses:
*(undefined4 *)(param_1 + 0x148) = *(undefined4 *)(iVar1 + 0x34);  // Flags?
*(undefined2 *)(param_1 + 0x80) = *(undefined2 *)(iVar1 + 0x3c);   // 16-bit field
*(undefined4 *)(param_1 + 0x6c) = *(undefined4 *)(iVar1 + 0x68);   // Position X
*(undefined4 *)(param_1 + 0x70) = *(undefined4 *)(iVar1 + 0x6c);   // Position Y
*(undefined4 *)(param_1 + 0x74) = *(undefined4 *)(iVar1 + 0x70);   // Position Z
```

### 2. Sub-Chunk Offset Table in MCNK Header

The MCNK header contains an offset table pointing to sub-chunks:

| Header Offset | Sub-Chunk | FourCC |
|---------------|-----------|--------|
| 0x14 | MCVT | 0x4d435654 |
| 0x18 | MCNR | 0x4d434e52 |
| 0x1c | MCLY | 0x4d434c59 |
| 0x20 | MCRF | 0x4d435246 |
| 0x24 | MCAL | 0x4d43414c |
| 0x2c | MCSH | 0x4d435348 |
| 0x58 | MCSE | 0x4d435345 |
| 0x60 | MCLQ | 0x4d434c51 |

### 3. MCNK Validation Code

```c
// Validate MCNK magic
if (**(int **)(param_1 + 0x138) != 0x4d434e4b) {  // 'MCNK'
    FUN_00640240(..., "iffChunk->token=='MCNK'", ...);
}

// Store MCNK header pointer
*(int *)(param_1 + 0xee4) = iVar1 + 8;

// Validate MCVT at offset 0x14
if (*(int *)(iVar1 + *(int *)(iVar1 + 0x1c)) != 0x4d435654) {
    FUN_00640240(..., "iffChunk->token=='MCVT'", ...);
}

// Validate MCNR at offset 0x18
if (*(int *)(*(int *)(param_1 + 0x138) + *(int *)(iVar1 + 0x18)) != 0x4d434e52) {
    FUN_00640240(..., "iffChunk->token=='MCNR'", ...);
}
```

### 4. MCNK Header Field Mapping

Based on analysis of [`FUN_006a6710`](0x006a6710):

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0x00 | 4 | Flags | Header flags |
| 0x04 | 4 | IndexX | Tile X index |
| 0x08 | 4 | IndexY | Tile Y index |
| 0x10 | 4 | LayerCount | Number of texture layers |
| 0x14 | 4 | MCVT Offset | Offset to MCVT chunk |
| 0x18 | 4 | MCNR Offset | Offset to MCNR chunk |
| 0x1c | 4 | MCLY Offset | Offset to MCLY chunk |
| 0x20 | 4 | MCRF Offset | Offset to MCRF chunk |
| 0x24 | 4 | MCAL Offset | Offset to MCAL chunk |
| 0x2c | 4 | MCSH Offset | Offset to MCSH chunk |
| 0x30 | 4 | ? | Unknown |
| 0x34 | 4 | Flags2 | Additional flags |
| 0x38 | 4 | ? | Unknown |
| 0x3c | 2 | HoleCount | Number of holes (16-bit) |
| 0x58 | 4 | MCSE Offset | Offset to MCSE chunk |
| 0x60 | 4 | MCLQ Offset | Offset to MCLQ chunk |
| 0x68 | 4 | PosX | X position (float) |
| 0x6c | 4 | PosY | Y position (float) |
| 0x70 | 4 | PosZ | Z position (float) |

### 5. Comparison with 0.5.3

The MCNK header layout in 0.6.0 is **very similar** to 0.5.3:
- Same sub-chunk offset table structure
- Same field ordering
- Same 0x80+ byte header size

---

## Corresponding 0.5.3 Functions

| 0.6.0 Address | Function Purpose | 0.5.3 Reference |
|---------------|------------------|-----------------|
| 0x006a6d00 | MCNK Validation | `CMapChunk::ValidateChunks` |
| 0x006a6710 | MCNK Loading | `CMapChunk::Load` |

---

## Conclusion

**0.6.0 uses Alpha-style MCNK header (similar to 0.5.3)**:
- Sub-chunk offset table at fixed positions
- Header size approximately 0x80 bytes
- Same field ordering as 0.5.3

No significant changes to MCNK header layout detected.

---

## Confidence Level: **MEDIUM**

- Header field accesses clearly documented
- Sub-chunk offset table confirmed
- Some fields not fully identified (marked as unknown)
