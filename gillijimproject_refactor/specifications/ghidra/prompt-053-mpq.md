# Ghidra LLM Prompt — WoWClient.exe 0.5.3.3368: MPQ Implementation

**Binary**: WoWClient.exe (Alpha 0.5.3 build 3368)
**PDB**: Available — full function names and type information
**Architecture**: x86 (32-bit)

---

## Context for the LLM

You are reverse engineering WoW Alpha 0.5.3 (build 3368) using Ghidra. This binary has a PDB loaded, so you have **named functions and types**. The goal is to document the complete MPQ file reading and decompression pipeline. These function names will also help us identify the same functions (unnamed) in the 0.6.0 binary.

### Why This Matters

- 0.5.3 uses per-asset `.ext.MPQ` archives (e.g., `BigKeep.wmo.MPQ` wraps a single WMO file)
- 0.6.0 uses standard large MPQ archives (`wmo.MPQ`, `terrain.MPQ`, etc.) with internal listfiles
- Both versions share the same underlying Storm.dll / MPQ reading code
- 0.5.3 has PDB symbols; 0.6.0 does not — so names found here will map directly to 0.6.0

### What We Already Know

1. **MPQ v1 format**: magic `MPQ\x1A` (0x1A51504D), 32-byte header
2. **Hash table and block table** are encrypted with keys derived from `"(hash table)"` and `"(block table)"`
3. **Compression types** (per-sector byte): 0x01=Huffman, 0x02=zlib, 0x08=PKWARE DCL, 0x10=BZip2
4. **Block flags**: 0x80000000=EXISTS, 0x00000200=COMPRESSED, 0x00000100=IMPLODED, 0x00010000=ENCRYPTED, 0x01000000=SINGLE_UNIT
5. **Our problem**: Compression type `0x08` fails in our implementation for 0.6.0 — data doesn't match standard PKWARE DCL format

---

## Research Tasks

Use Ghidra to investigate the following. Since we have PDB symbols, search by function name first.

### Task 1: Find All Storm/MPQ Functions

**Goal**: List all MPQ-related functions with their addresses and signatures.

**How to find them**:
1. Search the function list for names containing `SFile`, `SComp`, `Storm`, `MPQ`, `Decomp`, `Explode`, `Implode`, `Compress`
2. Search for names containing `Pkware`, `PKLIB`, `pklib`, `huffman`, `Huffman`
3. List every match with: address, full name, parameter types if available

**What to document**:
- All `SFile*` functions (SFileOpenArchive, SFileOpenFileEx, SFileReadFile, SFileCloseFile, etc.)
- All `SComp*` functions (SCompDecompress, SCompExplode, SCompImplode, etc.)
- Any internal decompression functions (explode, implode, decompress, inflate, etc.)
- Any hash/encrypt functions used for MPQ tables

### Task 2: Document the Decompression Dispatch Function

**Goal**: Find and fully document the function that dispatches to different decompression algorithms based on the compression type byte.

**How to find it**:
1. Look for `SCompDecompress` or similar named function
2. If not found by name, search for a function with a switch/if-chain on byte values 0x01, 0x02, 0x08, 0x10
3. This function takes compressed data (with type byte prefix) and outputs decompressed data

**What to document**:
- Full decompiled code of the dispatch function
- Every compression type it handles
- The function it calls for each type
- Whether types can be combined (bitmask) or are exclusive (enum)
- The order of decompression for combined types

### Task 3: Document the 0x08 Decompression Function

**Goal**: Fully reverse the decompression function called for type 0x08.

**What to document**:
1. The function name (from PDB)
2. The function signature (parameters, return type)
3. The full decompiled code
4. What the first few bytes of input mean:
   - Does it read a compType byte (0=binary, 1=ASCII)?
   - Does it read a dictShift byte (4, 5, or 6)?
   - Or does it use a completely different header format?
5. What lookup tables it uses (provide the table contents if small, or first/last entries if large)
6. The core algorithm: LZ77 with Huffman? Pure Huffman? Shannon-Fano?
7. Dictionary/window size

### Task 4: Document the Huffman Decompression (0x01)

**Goal**: Document the Huffman-only decompression for type 0x01.

**Why**: If 0x08 doesn't match PKWARE DCL, maybe these MPQs are actually using Huffman encoding labeled as 0x08.

**What to document**:
- Function name and address
- Input format (any header bytes?)
- Tree structure (static or dynamic Huffman?)
- Output: literal bytes only, or length-distance pairs too?

### Task 5: Document the Zlib Decompression (0x02)

**Goal**: Confirm how type 0x02 is handled (this already works, but for completeness).

**What to document**:
- Function name and address
- Does it expect a zlib header (2 bytes) or raw deflate?
- Any wrapper logic

### Task 6: Document MPQ File Read Pipeline

**Goal**: Trace the complete path from "open file by name" to "return decompressed bytes."

**Steps to trace**:
1. `SFileOpenFileEx` (or equivalent) — hash lookup in hash table → block table entry
2. `SFileReadFile` (or equivalent) — read block data from archive
3. Sector handling — how are sector offsets read? How is single-unit detected?
4. Decryption — how is encryption key derived from filename?
5. Decompression — per-sector type byte → dispatch → decompress

**What to document**:
- The complete call chain with function names
- How `FLAG_COMPRESSED` (0x200) vs `FLAG_IMPLODED` (0x100) are handled differently
- Whether single-unit files (fileSize ≤ sectorSize) skip sector offset tables
- The sector size calculation: `512 << sectorSizeShift` from MPQ header

### Task 7: Find Encryption Key Derivation

**Goal**: Document how the file encryption key is calculated.

**How to find**:
1. Look for `HashString` or similar
2. The key derivation uses the filename: `key = HashString(basename, HASH_FILE_KEY)`
3. With `FLAG_FIX_KEY`: `key = (key + blockOffset) ^ fileSize`

**What to document**:
- The hash function implementation
- The crypto table generation (0x500 entries)
- The block decryption function

---

## String References to Search For

These strings appear in Storm.dll / MPQ code and can help locate functions:

| String | Purpose |
|--------|---------|
| `"(listfile)"` | Internal listfile read |
| `"(hash table)"` | Hash table decryption key |
| `"(block table)"` | Block table decryption key |
| `"(attributes)"` | MPQ attributes file |
| `"Storm.dll"` | Library identification |
| `"PKLIB"` or `"pklib"` | PKWARE compression library |
| `"SFile"` | Storm file API prefix |

---

## Key Constants to Search For

| Value | Meaning |
|-------|---------|
| `0x1A51504D` | MPQ magic ("MPQ\x1A" LE) |
| `0x7FED7FED` | Hash seed1 initial value |
| `0xEEEEEEEE` | Hash/decrypt seed2 initial value |
| `0x11111111` | Decrypt key rotation constant |
| `0x00100001` | Crypto table generation seed |
| `0x2AAAAB` | Crypto table modulus |
| `0x80000000` | FLAG_EXISTS |
| `0x00000200` | FLAG_COMPRESSED |
| `0x00000100` | FLAG_IMPLODED |
| `0x00010000` | FLAG_ENCRYPTED |
| `0x01000000` | FLAG_SINGLE_UNIT |

---

## Output Format

For each task, provide:
1. **Function name** (from PDB) and **address**
2. **Decompiled C code** (cleaned up)
3. **Analysis** of what the code does
4. **Cross-reference notes** — what other functions call this, and what this calls

**Priority order**: Task 1 (get the function list first), then Task 2 + Task 3 (decompression dispatch + 0x08 handler), then the rest.

The primary deliverable is: **the exact algorithm and data format used by compression type 0x08**, with enough detail to write a correct C# reimplementation.
