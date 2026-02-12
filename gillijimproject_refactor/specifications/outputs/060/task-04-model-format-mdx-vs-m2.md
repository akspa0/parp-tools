# Task 4: Model Format (MDX vs M2)

**Binary**: WoWClient.exe (Alpha 0.6.0 build 3592)
**Analysis Date**: 2026-02-09

---

## Goal

Determine which model format 0.6.0 uses - MDX (Alpha) or M2 (LK+).

---

## Key Findings

### Format Determination: **MDX (Alpha-Style)**

---

## Evidence

### 1. Model Magic Number Check

**Function**: [`FUN_00421250`](0x00421250)
**Address**: `0x00421250`

```c
piVar1 = *(int **)(*(int *)(param_1 + 8) + 8);  // File data pointer
if (piVar1 == (int *)0x0) {
    // Error: fileData is null
}
if (*piVar1 != 0x584c444d) {  // 'MDLX' magic
    // Error: *((ULONG *) (fileData)) == 'XLDM'
}
```

The magic number `0x584c444d` is the FourCC for **'MDLX'** (stored little-endian as XLDM when read as bytes).

### 2. Model File Extension

**String References**:
- `"%s%s_%s%s.mdx"` at `0x00848190` - Model path construction
- Various hardcoded `.mdx` paths in strings

No references to `.m2` file extensions were found in the binary.

### 3. Model Loading Flow

```c
// FUN_00421250 - Model async completion handler
local_14 = piVar1 + 1;  // Data after magic
local_10 = *(int *)(*(int *)(param_1 + 8) + 0xc) + -4;  // File size - 4

iVar2 = FUN_00420230();  // Check if hardware vertex processing
if (iVar2 == 0) {
    // Software T&D path - allocate 0x148 bytes
    puVar3 = FUN_0063d9e0(0x148, ...);
    // Initialize software vertex processing structures
}
else {
    // Hardware T&D path - allocate 0x120 bytes  
    puVar3 = FUN_0063d9e0(0x120, ...);
    // Initialize hardware vertex processing structures
}
```

### 4. No M2 Magic Detection

Searched for M2 format magic numbers:
- `'MD20'` (0x3032444d) - Not found
- `'MD21'` (0x3132444d) - Not found
- `'MDLX'` (0x584c444d) - **Found** at `0x0083f09c`

### 5. String Evidence

**String**: `"*((ULONG *) (fileData)) == 'XLDM'"` at `0x0083f09c`

This assertion string confirms the expected magic number for model files is MDLX.

---

## MDX Format Structure (0.6.0)

Based on the loading code:

| Offset | Size | Purpose |
|--------|------|---------|
| 0x00 | 4 | Magic 'MDLX' |
| 0x04 | ... | Model header and data |

---

## Corresponding 0.5.3 Functions

| 0.6.0 Address | Function Purpose | 0.5.3 Reference |
|---------------|------------------|-----------------|
| 0x00421250 | Model Async Complete | `CModel::AsyncComplete` |
| 0x00420230 | Check Hardware T&D | `IsHardwareVertexProcessing` |

---

## Conclusion

**0.6.0 uses MDX format (Alpha-style)**:
- Magic number: 'MDLX' (0x584c444d)
- File extension: `.mdx`
- No M2 format support detected

The M2 format (MD20/MD21 magic) had **NOT** been adopted yet in 0.6.0.

---

## Confidence Level: **HIGH**

- Explicit magic number check for MDLX
- No M2 magic references found
- Consistent with 0.5.3 Alpha format
