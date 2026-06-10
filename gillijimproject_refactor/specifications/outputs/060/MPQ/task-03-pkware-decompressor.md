# Task 3: PKWARE DCL Decompressor Analysis (Compression Type 0x08)

## Overview

This document details the PKWARE DCL implode decompression algorithm as implemented in WoWClient.exe 0.6.0.3592.

## Critical Discovery

**The WoW 0.6.0 client does NOT include standard PKWARE DCL header bytes in the compressed data stream.**

Standard PKWARE DCL format expects:
- Byte 0: Compression type (0=binary, 1=ASCII)
- Byte 1: Dictionary size bits (4=1KB, 5=2KB, 6=4KB)

WoW 0.6.0 format:
- **No header bytes** - dictionary size is calculated from compressed data length
- Compressed data starts immediately after the MPQ compression type byte (0x08)

## Function Chain

### 1. PKWARE Wrapper - [`FUN_00651550`](specifications/outputs/060/MPQ/task-03-pkware-decompressor.md:17)

**Address**: 0x00651550

This wrapper function:
1. Allocates working memory (0x8dd8 = 36312 bytes)
2. Calculates dictionary size from compressed data length
3. Initializes the PKWARE decompressor
4. Calls the main decompression loop

**Decompiled Code**:
```c
void __fastcall FUN_00651550(undefined4 param_1, undefined4 *param_2, undefined4 param_3,
                             uint param_4, int *param_5)
{
    undefined4 uVar1;
    undefined1 local_24[4];
    undefined4 local_20;
    undefined4 local_1c;
    undefined4 local_18;
    undefined4 local_14;
    uint local_10;
    uint local_c;
    int local_8;

    // Allocate working memory (36312 bytes)
    uVar1 = FUN_0063d9e0(0x8dd8, s_C__build_buildWoW_Storm_Source_S_008a6580, 0x447, 0);
    local_1c = *param_2;
    local_18 = param_3;
    local_c = (uint)(*param_5 == 2);  // Compression mode (0=binary, 1=ASCII)
    local_20 = 0;
    local_14 = 0;
    local_10 = param_4;

    // === DICTIONARY SIZE CALCULATION ===
    // This is the KEY difference from standard PKWARE DCL!
    if (param_4 < 0xc00) {
        // For small files (< 3072 bytes compressed)
        // Formula: (-(compressedSize < 0x600) & 0xfffffc00) + 0x800
        // If compressedSize < 0x600 (1536): dictSize = 0x400 (1024)
        // If compressedSize >= 0x600: dictSize = 0x800 (2048)
        local_8 = (-(uint)(param_4 < 0x600) & 0xfffffc00) + 0x800;
    } else {
        // For larger files (>= 3072 bytes compressed)
        local_8 = 0x1000;  // 4096 bytes
    }

    // Initialize PKWARE decompressor
    // Parameters: readCallback, writeCallback, workMem, outPtr, &compMode, &dictSize
    FUN_006646d0(FUN_00651610, FUN_00651660, uVar1, local_24, &local_c, &local_8);

    // Free working memory
    FUN_0063f2a0(uVar1, s_C__build_buildWoW_Storm_Source_S_008a6580, 0x47e, 0);

    *param_2 = local_20;
    *param_5 = 0;
    return;
}
```

### 2. PKWARE Initialization - [`FUN_006646d0`](specifications/outputs/060/MPQ/task-03-pkware-decompressor.md:77)

**Address**: 0x006646d0

This function initializes the PKWARE decompression state, including building Huffman tables.

**Decompiled Code**:
```c
undefined4 FUN_006646d0(undefined4 param_1, undefined4 param_2, int param_3, undefined4 param_4,
                        int *param_5, int *param_6)
{
    int iVar1;
    short *psVar2;
    uint uVar3;
    uint uVar4;
    uint uVar5;
    uint uVar6;
    short sVar7;
    short *psVar8;
    short *psVar9;
    ushort *puVar10;

    // Store callbacks and parameters
    *(undefined4 *)(param_3 + 0x9b4) = param_1;  // Read callback
    *(undefined4 *)(param_3 + 0x9b8) = param_2;  // Write callback
    *(int *)(param_3 + 0x18) = *param_6;         // Dictionary size
    *(int *)(param_3 + 0x14) = *param_5;         // Compression mode
    *(undefined4 *)(param_3 + 0x9b0) = param_4;  // Output pointer

    // Initialize bit buffer
    *(undefined4 *)(param_3 + 0xc) = 4;
    *(undefined4 *)(param_3 + 0x10) = 0xf;

    // === Validate dictionary size ===
    iVar1 = *param_6;
    if (iVar1 != 0x400) {   // 1024 bytes
        if (iVar1 != 0x800) { // 2048 bytes
            if (iVar1 != 0x1000) { // 4096 bytes
                return 1;  // Invalid dictionary size!
            }
            // 4KB dictionary: set bits to 5, mask to 0x2F
            *(int *)(param_3 + 0xc) = *(int *)(param_3 + 0xc) + 1;
            *(uint *)(param_3 + 0x10) = *(uint *)(param_3 + 0x10) | 0x20;
        }
        // 2KB dictionary: set bits to 5, mask to 0x1F
        *(int *)(param_3 + 0xc) = *(int *)(param_3 + 0xc) + 1;
        *(uint *)(param_3 + 0x10) = *(uint *)(param_3 + 0x10) | 0x10;
    }
    // 1KB dictionary: bits=4, mask=0x0F (default)

    // === Build Huffman tables based on compression mode ===
    if (*param_5 == 0) {
        // Binary mode - use default tables
        sVar7 = 0;
        uVar4 = 0;
        psVar2 = (short *)(param_3 + 0x3a2);
        do {
            *(undefined1 *)(param_3 + 0x9c + uVar4) = 9;
            *psVar2 = sVar7;
            uVar4 = uVar4 + 1;
            sVar7 = sVar7 + 2;
            psVar2 = psVar2 + 1;
        } while (uVar4 < 0x100);
    }
    else {
        if (*param_5 != 1) {
            return 2;  // Invalid compression mode!
        }
        // ASCII mode - use pre-defined tables
        uVar4 = 0;
        psVar2 = (short *)(param_3 + 0x3a2);
        psVar8 = &DAT_008a79d8;
        do {
            psVar9 = psVar8 + 1;
            *(char *)(param_3 + 0x9c + uVar4) = (&DAT_008a78d8)[uVar4] + '\x01';
            uVar4 = uVar4 + 1;
            *psVar2 = *psVar8 * 2;
            psVar2 = psVar2 + 1;
            psVar8 = psVar9;
        } while (psVar9 < s_PKWARE_Data_Compression_Library_f_008a7bd8);
    }

    // Build length code tables
    uVar3 = 0;
    do {
        if (1 << ((&DAT_008a78a8)[uVar3] & 0x1f) != 0) {
            uVar5 = 0;
            puVar10 = (ushort *)(param_3 + 0x3a2 + uVar4 * 2);
            do {
                uVar6 = uVar5 + 1;
                *(char *)(param_3 + 0x9c + uVar4) = 
                    (&DAT_008a78a8)[uVar3] + (&DAT_008a78b8)[uVar3] + '\x01';
                uVar4 = uVar4 + 1;
                *puVar10 = (short)uVar5 << ((&DAT_008a78b8)[uVar3] + 1 & 0x1f) |
                           (ushort)(byte)(&DAT_008a78c8)[uVar3] * 2 | 1;
                uVar5 = uVar6;
                puVar10 = puVar10 + 1;
            } while (uVar6 < (uint)(1 << ((&DAT_008a78a8)[uVar3] & 0x1f)));
        }
        uVar3 = uVar3 + 1;
    } while (uVar3 < 0x10);

    // Initialize distance tables
    FUN_00664fc0(param_3 + 0x5c, &DAT_008a7868, 0x40);
    FUN_00664fc0(param_3 + 0x1c, &DAT_008a7828, 0x40);

    // Start decompression
    FUN_00664870(param_3);
    return 0;
}
```

### 3. PKWARE Main Decompression Loop - [`FUN_00664870`](specifications/outputs/060/MPQ/task-03-pkware-decompressor.md:177)

**Address**: 0x00664870

This is the main decompression loop that processes the compressed data.

**Decompiled Code** (simplified):
```c
void FUN_00664870(uint *param_1)
{
    byte bVar1;
    bool bVar2;
    int iVar3;
    uint uVar4;
    uint uVar5;
    uint uVar6;
    int iVar7;
    int iVar8;
    byte *pbVar9;
    byte *pbStack_10;
    int local_8;
    int local_4;

    local_4 = 0;
    bVar2 = false;

    // Initialize output state
    *(char *)((int)param_1 + 0x1fca) = (char)param_1[5];
    pbVar9 = (byte *)(param_1[6] + 0x29d0 + (int)param_1);
    *(char *)((int)param_1 + 0x1fcb) = (char)param_1[3];
    param_1[1] = 2;

    // Clear working buffer
    FUN_00665000(param_1 + 0x7f3, 0, 0x800);
    param_1[2] = 0;

    do {
        // Read 0x1000 (4096) bytes of compressed data
        iVar8 = 0x1000;
        iVar7 = 0;
        do {
            local_8 = iVar8;
            iVar3 = (*(code *)param_1[0x26d])(
                        (int)param_1 + param_1[6] + iVar7 + 0x29d0,
                        &local_8,
                        param_1[0x26c]);
            if (iVar3 == 0) {
                // End of input - flush output
                if ((iVar7 == 0) && (local_4 == 0)) {
                    FUN_00664f30(param_1, 
                                 *(undefined1 *)((int)param_1 + 0x3a1),
                                 (short)param_1[0x26b]);
                    if (param_1[2] != 0) {
                        param_1[1] = param_1[1] + 1;
                    }
                    (*(code *)param_1[0x26e])(
                        (int)param_1 + 0x1fca,
                        param_1 + 1,
                        param_1[0x26c]);
                    return;
                }
                bVar2 = true;
                break;
            }
            iVar7 = iVar7 + iVar3;
            iVar8 = iVar8 - iVar3;
        } while (iVar8 != 0);

        // Process compressed data
        uVar4 = param_1[6];
        pbStack_10 = (byte *)((int)param_1 + uVar4 + iVar7 + 0x27cc);

        if (bVar2) {
            pbStack_10 = pbStack_10 + 0x204;
        }

        // Initialize on first pass
        if (local_4 == 0) {
            FUN_006650f0(param_1, pbVar9, pbStack_10 + 1);
            local_4 = local_4 + 1;
            if (param_1[6] != 0x1000) goto LAB_00664986;
        }
        else if (local_4 == 1) {
            FUN_006650f0(param_1, pbVar9 + (0x204 - uVar4), pbStack_10 + 1);
            LAB_00664986:
            local_4 = local_4 + 1;
        }
        else {
            FUN_006650f0(param_1, (int)pbVar9 - uVar4, pbStack_10 + 1);
        }

joined_r0x0066498e:
        // Main decode loop
        if (pbVar9 < pbStack_10) {
            uVar4 = FUN_00664bf0(param_1, pbVar9);
            while (true) {
                if ((uVar4 == 0) || ((uVar4 == 2 && (0xff < *param_1)))) {
                    goto LAB_00664a3f;
                }
                if ((bVar2) && (pbStack_10 < pbVar9 + uVar4)) break;
                if ((7 < uVar4) || (pbStack_10 <= pbVar9 + 1)) goto LAB_00664b0f;

                uVar6 = *param_1;
                uVar5 = FUN_00664bf0(param_1, pbVar9 + 1);
                if ((uVar5 <= uVar4) || ((uVar5 <= uVar4 + 1 && (uVar6 < 0x81)))) {
                    *param_1 = uVar6;
                    goto LAB_00664b0f;
                }

                // Decode literal byte
                bVar1 = *pbVar9;
                pbVar9 = pbVar9 + 1;
                FUN_00664f30(param_1,
                             *(undefined1 *)(bVar1 + 0x9c + (int)param_1),
                             *(undefined2 *)((int)param_1 + (uint)bVar1 * 2 + 0x3a2));
                uVar4 = uVar5;
            }

            // Handle back-reference
            uVar4 = (int)pbStack_10 - (int)pbVar9;
            if ((1 < uVar4) && ((uVar4 != 2 || (*param_1 < 0x100)))) {
                LAB_00664b0f:
                // Decode length and distance
                FUN_00664f30(param_1,
                             *(undefined1 *)((int)param_1 + uVar4 + 0x19a),
                             *(undefined2 *)((int)param_1 + uVar4 * 2 + 0x59e));
                if (uVar4 == 2) {
                    uVar4 = *param_1 >> 2;
                    FUN_00664f30(param_1,
                                 *(undefined1 *)((int)param_1 + uVar4 + 0x1c),
                                 *(undefined1 *)((int)param_1 + uVar4 + 0x5c));
                    pbVar9 = pbVar9 + 2;
                    FUN_00664f30(param_1, 2, *param_1 & 3);
                }
                else {
                    uVar6 = *param_1 >> ((byte)param_1[3] & 0x1f);
                    FUN_00664f30(param_1,
                                 *(undefined1 *)((int)param_1 + uVar6 + 0x1c),
                                 *(undefined1 *)((int)param_1 + uVar6 + 0x5c));
                    pbVar9 = pbVar9 + uVar4;
                    FUN_00664f30(param_1, param_1[3], param_1[4] & *param_1);
                }
                goto joined_r0x0066498e;
            }
            LAB_00664a3f:
            // Output literal byte
            bVar1 = *pbVar9;
            pbVar9 = pbVar9 + 1;
            FUN_00664f30(param_1,
                         *(undefined1 *)(bVar1 + 0x9c + (int)param_1),
                         *(undefined2 *)((int)param_1 + (uint)bVar1 * 2 + 0x3a2));
            goto joined_r0x0066498e;
        }

        // Check for end of stream
        if (bVar2) {
            FUN_00664f30(param_1,
                         *(undefined1 *)((int)param_1 + 0x3a1),
                         (short)param_1[0x26b]);
            if (param_1[2] != 0) {
                param_1[1] = param_1[1] + 1;
            }
            (*(code *)param_1[0x26e])(
                (int)param_1 + 0x1fca,
                param_1 + 1,
                param_1[0x26c]);
            return;
        }

        // Continue with next block
        pbVar9 = pbVar9 + -0x1000;
        FUN_00664fc0(param_1 + 0x9f3, param_1 + 0xdf3, param_1[6] + 0x204);
    } while (true);
}
```

### 4. Read Callback - [`FUN_00651610`](specifications/outputs/060/MPQ/task-03-pkware-decompressor.md:347)

**Address**: 0x00651610

Reads data from the compressed input buffer.

```c
void FUN_00651610(undefined4 *param_1, uint *param_2, int param_3)
{
    uint uVar1;
    uint uVar2;
    undefined4 *puVar3;

    // Calculate available bytes
    uVar1 = *(int *)(param_3 + 0x14) - *(int *)(param_3 + 0x10);
    if (*param_2 < uVar1) {
        uVar1 = *param_2;
    }

    // Copy from input buffer
    puVar3 = (undefined4 *)(*(int *)(param_3 + 0xc) + *(int *)(param_3 + 0x10));
    for (uVar2 = uVar1 >> 2; uVar2 != 0; uVar2 = uVar2 - 1) {
        *param_1 = *puVar3;
        puVar3 = puVar3 + 1;
        param_1 = param_1 + 1;
    }
    for (uVar2 = uVar1 & 3; uVar2 != 0; uVar2 = uVar2 - 1) {
        *(undefined1 *)param_1 = *(undefined1 *)puVar3;
        puVar3 = (undefined4 *)((int)puVar3 + 1);
        param_1 = (undefined4 *)((int)param_1 + 1);
    }

    // Update read position
    *(uint *)(param_3 + 0x10) = *(int *)(param_3 + 0x10) + uVar1;
    return;
}
```

### 5. Write Callback - [`FUN_00651660`](specifications/outputs/060/MPQ/task-03-pkware-decompressor.md:383)

**Address**: 0x00651660

Writes decompressed data to the output buffer.

## PKWARE Copyright String

Found at address 0x008a7bd8:
```
"PKWARE Data Compression Library for Win32\r\n
Copyright 1989-1995 PKWARE Inc.  All Rights Reserved\r\n
Patent No. 5,051,745\r\n
PKWARE Data Compression Library Reg. U.S. Pat. and Tm. Off.\r\n
Version 1.11"
```

## Lookup Tables

The PKWARE decompressor uses several lookup tables stored in the binary:

| Address | Size | Purpose |
|---------|------|---------|
| 0x008a7828 | 0x40 | Distance table 1 |
| 0x008a7868 | 0x40 | Distance table 2 |
| 0x008a78a8 | 0x10 | Length code bits |
| 0x008a78b8 | 0x10 | Length code extra bits |
| 0x008a78c8 | 0x10 | Length code base values |
| 0x008a78d8 | 0x100 | ASCII mode literal lengths |
| 0x008a79d8 | 0x200 | ASCII mode literal codes |

## Algorithm Summary

1. **No header bytes** - Dictionary size is determined by compressed data length
2. **Dictionary sizes**: 1024, 2048, or 4096 bytes based on input size
3. **Compression modes**: Binary (0) or ASCII (1)
4. **Huffman coding** for literals and length codes
5. **LZ77-style backreferences** with sliding window dictionary

## Implementation Recommendation

To fix the decompression failure:

```c
// Instead of reading header bytes:
// uint8_t compType = data[0];
// uint8_t dictBits = data[1];
// uint32_t dictSize = 1 << dictBits;

// Calculate dictionary size from compressed length:
uint32_t dictSize;
if (compressedLength < 0x600) {
    dictSize = 1024;  // 0x400
} else if (compressedLength < 0xC00) {
    dictSize = 2048;  // 0x800
} else {
    dictSize = 4096;  // 0x1000
}

// Assume binary mode (0) for MPQ files
uint8_t compType = 0;

// Start decompression from byte 0 (after MPQ compression type byte)
pkware_decompress(output, outputSize, compressedData, compressedLength, compType, dictSize);
```
