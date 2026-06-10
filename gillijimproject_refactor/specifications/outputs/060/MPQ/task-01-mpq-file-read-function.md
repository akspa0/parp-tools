# Task 1: MPQ File Read Function Analysis

## Overview

This document details the MPQ file reading functions found in WoWClient.exe 0.6.0.3592.

## Key Functions Identified

### 1. MPQ Archive Opening - [`FUN_006479f0`](specifications/outputs/060/MPQ/task-01-mpq-file-read-function.md:10)

**Address**: 0x006479f0

This function reads the `(listfile)` from an MPQ archive during initialization.

**Decompiled Code**:
```c
undefined4 FUN_006479f0(int param_1)
{
    undefined4 uVar1;
    int iVar2;
    
    // Open "(listfile)" from the archive
    iVar2 = FUN_00646f70(*(undefined4 *)(param_1 + 4), "(listfile)", 0, &uVar1);
    if (iVar2 != 0) {
        // Read the listfile contents
        FUN_00658c20(uVar1, *(undefined4 *)(param_1 + 0x10), *(uint *)(param_1 + 0xc), 
                     (uint *)(param_1 + 0xc), 0);
        FUN_00646d70(uVar1);  // Close file
    }
    return uVar1;
}
```

### 2. SFileOpenFileEx - [`FUN_00646f70`](specifications/outputs/060/MPQ/task-01-mpq-file-read-function.md:30)

**Address**: 0x00646f70

Opens a file from an MPQ archive by name.

**Function Signature**:
```c
BOOL SFileOpenFileEx(HANDLE hMpq, const char* fileName, DWORD searchScope, HANDLE* phFile);
```

**Decompiled Code**:
```c
undefined4 FUN_00646f70(undefined4 param_1, undefined4 param_2, uint param_3, int *param_4)
{
    uint uVar1;
    int iVar2;
    
    if (param_4 == (int *)0x0) {
        FUN_006414b0(s_C__build_buildWoW_Storm_Source_S_008a6580, 0x3c6);
        FUN_00640150(s_phFile_008a6424);
        FUN_00641690(0x57);
        return 0;
    }
    *param_4 = 0;
    if (param_1 == 0) {
        // Search all open archives
        uVar1 = FUN_00646b90();
        while (uVar1 != 0) {
            iVar2 = FUN_00647240(uVar1, param_2, param_3, param_4);
            if (iVar2 != 0) {
                return 1;
            }
            uVar1 = FUN_00646c20(uVar1);
        }
        return 0;
    }
    // Search specific archive
    return FUN_00647240(param_1, param_2, param_3, param_4);
}
```

### 3. File Read Dispatcher - [`FUN_00646d70`](specifications/outputs/060/MPQ/task-01-mpq-file-read-function.md:70)

**Address**: 0x00646d70

Dispatches file operations based on file type.

**Decompiled Code**:
```c
void FUN_00646d70(int param_1)
{
    int iVar1;
    
    if (param_1 != 0) {
        iVar1 = *(int *)(param_1 + 8);  // Get file type
        switch(iVar1) {
            case 0:
                FUN_00647640(param_1);  // Regular MPQ file
                break;
            case 1:
                FUN_00647740(param_1);  // Local file
                break;
            case 2:
                FUN_00647640(param_1);  // Another MPQ type
                break;
            case 3:
                // Handle type 3
                break;
            case 4:
                // Handle type 4
                break;
        }
    }
    return;
}
```

### 4. SFileReadFile - [`FUN_00658c20`](specifications/outputs/060/MPQ/task-01-mpq-file-read-function.md:110)

**Address**: 0x00658c20

Main file reading function that handles sector-based reading and decompression.

**Function Signature**:
```c
BOOL SFileReadFile(HANDLE hFile, void* buffer, DWORD toRead, DWORD* read, LPOVERLAPPED overlapped);
```

**Key Operations**:
1. Validates file handle and parameters
2. Calculates sector positions from sector offset table
3. Reads compressed sectors from MPQ
4. Decompresses sectors using [`FUN_006525d0`](specifications/outputs/060/MPQ/task-02-decompression-dispatch.md)
5. Copies decompressed data to output buffer

**Decompiled Code** (simplified):
```c
undefined4 FUN_00658c20(int param_1, undefined4 param_2, uint param_3, uint *param_4, undefined4 param_5)
{
    uint uVar1;
    uint uVar2;
    uint uVar3;
    int iVar4;
    
    if (param_1 == 0) {
        // Error: invalid handle
        return 0;
    }
    
    // Get file info from handle
    uVar1 = *(uint *)(param_1 + 0x10);  // Current position
    uVar2 = *(uint *)(param_1 + 0x14);  // File size
    
    // Calculate bytes to read
    if (uVar1 + param_3 > uVar2) {
        param_3 = uVar2 - uVar1;
    }
    
    // Read and decompress sectors
    uVar3 = 0;
    while (uVar3 < param_3) {
        // Get sector offset from table
        iVar4 = FUN_00658740(param_1, uVar1);
        
        // Read compressed sector
        FUN_00658920(param_1, sectorBuffer, iVar4, sectorSize);
        
        // Decompress sector
        FUN_006525d0(outputBuffer, &outputSize, sectorBuffer, sectorSize, flags);
        
        uVar3 += bytesDecompressed;
        uVar1 += bytesDecompressed;
    }
    
    *param_4 = uVar3;
    return 1;
}
```

### 5. Hash Table Lookup - [`FUN_00647240`](specifications/outputs/060/MPQ/task-01-mpq-file-read-function.md:170)

**Address**: 0x00647240

Looks up a file by name in the MPQ hash table.

**Decompiled Code**:
```c
undefined4 FUN_00647240(undefined4 param_1, undefined4 param_2, uint param_3, int *param_4)
{
    uint uVar1;
    uint uVar2;
    int iVar3;
    
    // Calculate hash values
    uVar1 = FUN_00644a60(param_2, 0);  // Name hash
    uVar2 = FUN_00644a60(param_2, 1);  // Locale hash
    
    // Search hash table
    iVar3 = FUN_00647080(param_1, uVar1, uVar2);
    if (iVar3 != 0) {
        *param_4 = iVar3;
        return 1;
    }
    return 0;
}
```

## MPQ Header Processing

The MPQ header is read and validated in the archive opening function:

**MPQ v1 Header Structure** (32 bytes):
```c
struct MPQHeader {
    uint32_t dwID;           // 0x1A51504D ('MPQ\x1A')
    uint32_t dwDataOffset;   // Offset to file data
    uint32_t dwArchiveSize;  // Total archive size
    uint16_t wVersion;       // Version (0 for MPQ v1)
    uint16_t wBlockSize;     // Block size shift (typically 3 = 4096 byte sectors)
    uint32_t dwHashTablePos; // Hash table offset
    uint32_t dwBlockTablePos;// Block table offset
    uint32_t dwHashTableSize;// Hash table entry count
    uint32_t dwBlockTableSize;// Block table entry count
};
```

## String References

Key strings found in the binary:
- `"(listfile)"` at 0x008a6548 - Internal file name for archive file list
- `"(hash table)"` at 0x008a6554 - Used for decryption key generation
- `"(block table)"` at 0x008a6560 - Used for decryption key generation
- `"C:\build\buildWoW\Storm\Source\SFile.cpp"` - Source file path for error messages

## Summary

The MPQ file reading system follows the standard StormLib architecture:
1. [`SFileOpenArchive`](specifications/outputs/060/MPQ/task-01-mpq-file-read-function.md:10) opens the MPQ and reads header/tables
2. [`SFileOpenFileEx`](specifications/outputs/060/MPQ/task-01-mpq-file-read-function.md:30) looks up files by name in hash table
3. [`SFileReadFile`](specifications/outputs/060/MPQ/task-01-mpq-file-read-function.md:110) reads and decompresses file data sector by sector
4. Decompression is handled by [`FUN_006525d0`](specifications/outputs/060/MPQ/task-02-decompression-dispatch.md) (see Task 2)
