# Task 7: Encryption Key Derivation

## Overview

MPQ encryption uses a custom stream cipher developed by Blizzard. The encryption is applied to sector data after compression (if any).

## Decrypt Function

The decryption function is located at `0064e0b0`.

```c
void __fastcall Decrypt(uint *data, uint length, uint key)
{
  uint seed = 0xEEEEEEE; // -0x11111112
  uint ch;
  
  for (uint i = length >> 2; i != 0; i--) {
    // Update seed
    seed += Storm::SFile::s_hashsource[(key & 0xFF) + 0x400];
    
    // Decrypt block
    ch = *data ^ (seed + key);
    
    // Update key
    key = ((~key << 21) + 0x11111111) | (key >> 11);
    
    // Update seed
    seed = *data + seed + (seed << 5) + 3; 
    
    // Store decrypted block
    *data = ch;
    data++;
  }
}
```

*Note: The decompiled code shows a slightly optimized/different structure but the logic is mathematically equivalent to the standard MPQ decryption.*

## Key Derivation

The key used for decryption is derived from a base file key and the sector index.

In `InternalReadAligned` (`0064da90`):
```c
Decrypt(buffer, length, baseKey + sectorIndex);
```

*   **Base Key**: The file's encryption key, usually derived from the file name using the `HashString` function (Hash Type `MPQ_HASH_FILE_KEY`).
*   **Sector Index**: The zero-based index of the sector being read.
*   **Adjustment**: The key passed to `Decrypt` is simply `BaseKey + SectorIndex`.

## Crypt Table

The function uses a global table `Storm::SFile::s_hashsource` (often referred to as the "crypt table"). This table contains precomputed values used for hashing and encryption. The offset `0x400` corresponds to the encryption section of the table.
