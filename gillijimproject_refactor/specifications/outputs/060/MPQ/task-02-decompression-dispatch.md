# Task 2: Decompression Dispatch Analysis

## Overview

This document details the decompression dispatch mechanism in WoWClient.exe 0.6.0.3592, specifically how compression type 0x08 (PKWARE DCL) is handled.

## Main Decompression Dispatch Function - [`FUN_006525d0`](specifications/outputs/060/MPQ/task-02-decompression-dispatch.md:9)

**Address**: 0x006525d0

This is the central decompression dispatch function that handles all compression types.

### Function Signature

```c
int SCompDecompress(
    byte* destBuffer,      // param_1 - Output buffer
    uint* destSize,        // param_2 - Pointer to output size (in/out)
    byte* srcBuffer,       // param_3 - Input buffer (compressed data)
    uint srcSize,          // param_4 - Input size
    undefined4 param_5     // param_5 - Additional flags/parameters
);
```

### Decompiled Code

```c
undefined4 FUN_006525d0(byte *param_1, uint *param_2, byte *param_3, byte *param_4, undefined4 param_5)
{
    uint uVar1;
    int iVar2;
    uint uVar3;
    uint *puVar4;
    uint uVar5;
    uint uVar6;
    uint uVar7;
    byte *pbVar8;
    byte *pbVar9;
    char *pcVar10;
    undefined **local_8;

    uVar3 = *param_2;  // Destination buffer size
    if (param_1 == (byte *)0x0) {
        // Error: null destination buffer
        FUN_006414b0(s_C__build_buildWoW_Storm_Source_S_008a6580, 0x752);
        pcVar10 = &DAT_008a5adc;
    }
    else if (uVar3 < param_4) {
        // Error: destination too small
        FUN_006414b0(s_C__build_buildWoW_Storm_Source_S_008a6580, 0x753);
        pcVar10 = s_destbuffersize_>__sourcesize_008a662c;
    }
    else if (param_3 == (byte *)0x0) {
        // Error: null source buffer
        FUN_006414b0(s_C__build_buildWoW_Storm_Source_S_008a6580, 0x754);
        pcVar10 = s_source_0083febc;
    }
    else {
        if (param_4 != (byte *)0x0) {
            // If source size == dest size, just copy (no compression)
            if (param_4 == (byte)uVar3) {
                if (param_1 == param_3) {
                    return 1;  // Same buffer, nothing to do
                }
                // Copy data
                for (uVar3 = (uint)param_4 >> 2; uVar3 != 0; uVar3 = uVar3 - 1) {
                    *(undefined4 *)param_1 = *(undefined4 *)param_3;
                    param_3 = param_3 + 4;
                    param_1 = param_1 + 4;
                }
                for (uVar3 = (uint)param_4 & 3; uVar3 != 0; uVar3 = uVar3 - 1) {
                    *param_1 = *param_3;
                    param_3 = param_3 + 1;
                    param_1 = param_1 + 1;
                }
                return 1;
            }

            // === CRITICAL: Read compression type byte ===
            uVar7 = (uint)*param_3;  // First byte is compression type bitmask
            pbVar9 = param_3 + 1;     // Skip compression type byte
            uVar6 = (int)param_4 - 1; // Remaining size after type byte
            param_3 = (byte *)0x0;

            // Count how many compression methods are used
            puVar4 = &DAT_0081b2ec;  // Compression mask table
            uVar5 = uVar7;
            do {
                uVar1 = *puVar4;
                if ((uVar7 & uVar1) != 0) {
                    param_3 = (byte *)((int)param_3 + 1);
                }
                puVar4 = puVar4 + -2;
                uVar5 = uVar5 & ~uVar1;
            } while (0x81b2c3 < (int)puVar4);

            if (uVar5 == 0) {
                param_4 = (byte *)0x0;

                // If multiple compression methods, allocate temp buffer
                if ((1 < param_3) || 
                    (iVar2 = FUN_00652430(uVar6), iVar2 != 0 && (param_3 != (byte *)0x0))) {
                    param_4 = (byte *)FUN_00652460();
                }

                // === Iterate through compression handlers ===
                local_8 = &PTR_FUN_0081b2f0;  // Function pointer table
                do {
                    pbVar8 = pbVar9;
                    if (((uint)local_8[-1] & uVar7) != 0) {
                        param_3 = (byte *)((int)param_3 - 1);
                        pbVar8 = param_4;
                        if (((uint)param_3 & 1) == 0) {
                            pbVar8 = param_1;
                        }
                        iVar2 = FUN_00652430(uVar6);
                        if ((iVar2 != 0) && (iVar2 = FUN_00652430(uVar6), pbVar8 = param_1, iVar2 == 0)) {
                            pbVar8 = param_4;
                        }
                        // Call the decompression handler
                        (*(code *)*local_8)(pbVar9, uVar6, param_5);
                        uVar6 = uVar3;
                    }
                    local_8 = local_8 + -2;
                    pbVar9 = pbVar8;
                } while (0x81b2c7 < (int)local_8);

                // Copy from temp buffer if needed
                if (pbVar8 != param_1) {
                    for (uVar3 = uVar6 >> 2; uVar3 != 0; uVar3 = uVar3 - 1) {
                        *(undefined4 *)param_1 = *(undefined4 *)pbVar8;
                        pbVar8 = pbVar8 + 4;
                        param_1 = param_1 + 4;
                    }
                    for (uVar3 = uVar6 & 3; uVar3 != 0; uVar3 = uVar3 - 1) {
                        *param_1 = *pbVar8;
                        pbVar8 = pbVar8 + 1;
                        param_1 = param_1 + 1;
                    }
                }
                *param_2 = uVar6;
                FUN_00652560();
                return 1;
            }
            return 0;
        }
        FUN_006414b0(s_C__build_buildWoW_Storm_Source_S_008a6580, 0x755);
        pcVar10 = s_sourcesize_>__sizeof_BYTE__008a6610;
    }
    FUN_00640150(pcVar10);
    FUN_00641690(0x57);
    return 0;
}
```

## Compression Type Tables

### Mask Table at 0x0081b2ec

This table contains the compression type bitmasks in descending order:

| Address | Mask | Description |
|---------|------|-------------|
| 0x81b2ec | 0x10 | BZip2 |
| 0x81b2e4 | 0x08 | PKWARE DCL Implode |
| 0x81b2dc | 0x02 | zlib/deflate |
| 0x81b2d4 | 0x01 | Huffman |

### Function Pointer Table at 0x0081b2f0

This table contains function pointers for each compression type:

| Address | Function | Compression Type |
|---------|----------|-----------------|
| 0x81b2f0 | FUN_00651a90 | BZip2 (0x10) |
| 0x81b2e8 | FUN_00651550 | PKWARE DCL (0x08) |
| 0x81b2e0 | FUN_00661930 | zlib (0x02) |
| 0x81b2d8 | FUN_006517e0 | Huffman (0x01) |

## Compression Type 0x08 Handler - [`FUN_00651550`](specifications/outputs/060/MPQ/task-03-pkware-decompressor.md)

**Address**: 0x00651550

This is the PKWARE DCL implode wrapper function.

### Key Finding: Dictionary Size Calculation

**CRITICAL**: The function does NOT read header bytes from the compressed data. Instead, it calculates the dictionary size based on the compressed data size:

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

    uVar1 = FUN_0063d9e0(0x8dd8, s_C__build_buildWoW_Storm_Source_S_008a6580, 0x447, 0);
    local_1c = *param_2;
    local_18 = param_3;
    local_c = (uint)(*param_5 == 2);  // Compression type (0=binary, 1=ASCII)
    local_20 = 0;
    local_14 = 0;
    local_10 = param_4;

    // === CRITICAL: Dictionary size calculation ===
    if (param_4 < 0xc00) {
        // For small files: calculate dict size based on compressed size
        // If param_4 < 0x600 (1536): local_8 = 0x400 (1024)
        // If param_4 >= 0x600: local_8 = 0x800 (2048)
        local_8 = (-(uint)(param_4 < 0x600) & 0xfffffc00) + 0x800;
    } else {
        // For larger files: use 4096 byte dictionary
        local_8 = 0x1000;
    }

    // Initialize PKWARE decompressor with calculated values
    FUN_006646d0(FUN_00651610, FUN_00651660, uVar1, local_24, &local_c, &local_8);
    FUN_0063f2a0(uVar1, s_C__build_buildWoW_Storm_Source_S_008a6580, 0x47e, 0);

    *param_2 = local_20;
    *param_5 = 0;
    return;
}
```

### Dictionary Size Logic Explained

```c
// Pseudocode for dictionary size calculation:
if (compressedSize < 0x600) {       // < 1536 bytes
    dictSize = 0x400;               // 1024 bytes
} else if (compressedSize < 0xC00) { // < 3072 bytes
    dictSize = 0x800;               // 2048 bytes
} else {
    dictSize = 0x1000;              // 4096 bytes
}
```

## Disassembly of Key Section

The compression type byte reading and dispatch:

```asm
00652699: MOVZX ESI, byte ptr [EDI]     ; Read compression type byte
0065269c: INC EDI                        ; Skip to compressed data
0065269d: DEC EBX                        ; Decrement remaining size
0065269e: MOV dword ptr [EBP - 0x14], ESI
006526a1: MOV dword ptr [EBP + 0x14], EBX
006526a4: MOV dword ptr [EBP + 0x10], 0x0
006526ab: MOV EDX, ESI
006526ad: MOV ECX, 0x81b2ec             ; Load mask table address
006526b2: MOV EAX, dword ptr [ECX]      ; Read mask value
006526b4: TEST ESI, EAX                 ; Check if compression type set
006526b6: JZ 0x006526be
006526b8: INC dword ptr [EBP + 0x10]    ; Count compression methods
006526bb: MOV EBX, dword ptr [EBP + 0x14]
006526be: NOT EAX
006526c0: SUB ECX, 0x8                  ; Move to next table entry
006526c3: AND EDX, EAX
006526c5: CMP ECX, 0x81b2c4             ; Check if done
006526cb: JGE 0x006526b2

; ... later, call the handler ...

0065270c: MOV EAX, 0x81b2f0             ; Load function table address
00652711: MOV dword ptr [EBP - 0x4], EAX
00652714: TEST dword ptr [EAX - 0x4], ESI  ; Check mask
00652717: JZ 0x0065276c
; ... setup parameters ...
0065275f: CALL dword ptr [EAX]          ; Call decompression handler
```

## Summary

1. **Compression type byte**: First byte of compressed data is a bitmask of compression methods
2. **Multiple compression**: Can chain multiple methods (applied in reverse order)
3. **PKWARE (0x08)**: Calls [`FUN_00651550`](specifications/outputs/060/MPQ/task-03-pkware-decompressor.md)
4. **Critical finding**: Dictionary size is calculated from compressed data size, NOT read from header bytes

This explains why our implementation fails - we were trying to read PKWARE header bytes that don't exist in the MPQ compressed data stream!
