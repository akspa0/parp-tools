# Ghidra LLM Prompt — WoWClient.exe 0.6.0.3592: MPQ Decompression

**Binary**: WoWClient.exe (Alpha 0.6.0 build 3592)
**Architecture**: x86 (32-bit)
**PDB**: None available — rely on string references, known constants, and call patterns.

---

## Context for the LLM

You are reverse engineering WoW Alpha 0.6.0 (build 3592) using Ghidra. We need to understand exactly how this client reads files from its MPQ archives.

### What We Already Know

1. **0.6.0 uses standard MPQ v1 archives** — header magic `MPQ\x1A` (0x1A51504D), 32-byte header
2. **Archive list** (from the Data folder):
   - base.MPQ, dbc.MPQ, fonts.MPQ, interface.MPQ, misc.MPQ, model.MPQ, sound.MPQ, speech.MPQ, terrain.MPQ, texture.MPQ, wmo.MPQ
3. **All archives have internal (listfile) entries** — so file lookup uses standard MPQ hash-based naming
4. **Block flags observed**: `0x80000200` = EXISTS (0x80000000) | COMPRESSED (0x00000200)
5. **Compression type byte 0x08** is used for small files (≤512 bytes decompressed), especially WMO root files
6. **Compression type byte 0x02** (zlib) works fine for larger files — we can already decompress those
7. **Our PKWARE DCL (implode) implementation for type 0x08 fails** — the data after the 0x08 type byte does NOT match standard PKWARE DCL format (first two bytes should be compType=0/1 and dictShift=4/5/6, but we see dictShift=0)

### The Specific Problem

When reading a compressed file from wmo.MPQ:
- Block: offset=435912, size=318 (compressed), fileSize=472 (decompressed), flags=0x80000200
- The first byte of compressed data = `0x08` (per-sector compression type)
- The remaining 317 bytes should be the compressed payload
- Our PKWARE DCL decompressor rejects it: "Invalid dict size bits: 0"
- This affects ALL small compressed files in 0.6.0 MPQs (WMO root files, some DBC internal data, etc.)

### MPQ Compression Background

In standard MPQ implementations (StormLib), the per-sector compression type byte is a bitmask:
- `0x01` = Huffman encoding
- `0x02` = zlib/deflate
- `0x08` = PKWARE DCL implode
- `0x10` = BZip2
- `0x40` = IMA ADPCM stereo
- `0x80` = IMA ADPCM mono

Multiple bits can be set for chained compression (applied in reverse order for decompression).

---

## Research Tasks

Use Ghidra to investigate the following. For each task, document the function address, decompiled code, and your analysis.

### Task 1: Find the MPQ File Read Function

**Goal**: Locate the function that reads and decompresses a file from an MPQ archive.

**How to find it**:
1. Search for the string `"(listfile)"` — this is read from every MPQ on load
2. Search for the string `"(hash table)"` or `"(block table)"` — these are MPQ internal names used for decryption keys
3. Search for the MPQ magic constant `0x1A51504D`
4. Search for strings like `".MPQ"` or `"MPQ"` to find archive loading code
5. Look for functions that call `CreateFileA`/`ReadFile` with MPQ-related paths

**What to document**:
- The function that opens an MPQ archive and reads its header/hash table/block table
- The function that looks up a file by name in the hash table
- The function that reads file data from a block entry
- The function that decompresses sector data

### Task 2: Identify the Decompression Dispatch

**Goal**: Find exactly how compression type `0x08` is handled.

**How to find it**:
1. From the file read function (Task 1), trace into the decompression path
2. Look for a switch/if-chain on a byte value that branches to different decompression routines
3. The switch will check values like 0x01, 0x02, 0x08, 0x10 etc.
4. Follow the 0x08 branch to find the actual decompression function

**What to document**:
- The address of the decompression dispatch function
- The exact branching logic for each compression type
- Whether type 0x08 calls PKWARE DCL implode or something else
- The function signature and parameters of the 0x08 decompressor
- The first few lines of the 0x08 decompression function (does it read compType and dictShift like PKWARE, or something different?)

### Task 3: Reverse Engineer the 0x08 Decompressor

**Goal**: Fully understand the decompression algorithm used for type 0x08.

**What to look for**:
1. Does the function read two header bytes (PKWARE DCL style: compression type + dict size)?
2. Or does it use a different format?
3. What lookup tables or constants does it reference? (PKWARE uses specific Huffman tables)
4. Does it use a sliding window dictionary? What size?
5. Is it actually Huffman-only encoding (no LZ77 back-references)?

**Key constants to look for** (that would confirm PKWARE DCL):
- Dictionary sizes: 1024 (0x400), 2048 (0x800), 4096 (0x1000)
- Length table: {3, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7}
- Distance table with 64 entries
- CRC table or bit manipulation patterns

**Key constants for Huffman-only**:
- 256-entry frequency or code table
- Bit-reading loop without back-reference copy

**Deliverable**: Pseudocode or C equivalent of the 0x08 decompression function, enough detail to reimplement.

### Task 4: Check for Encryption in Small Files

**Goal**: Verify whether small file blocks use encryption.

**How to check**:
- Block flags `0x80000200` do NOT include `0x00010000` (FLAG_ENCRYPTED), but verify the client agrees
- Look at the file read function: does it check any additional flags or conditions for encryption?
- Is there a case where encryption is applied even without the flag bit?

### Task 5: Verify Sector Size and Single-Unit Handling

**Goal**: Confirm how single-sector files are read.

**What to check**:
- What is the default sector size shift for these archives? (typically 3, meaning 512 << 3 = 4096)
- For files smaller than one sector (fileSize ≤ sectorSize), does the client read BlockSize bytes and decompress directly?
- Or does it still read a sector offset table first?
- Is there any special handling for very small files?

---

## Known Function Signatures (from StormLib for reference)

These are the standard MPQ API signatures the client likely implements internally:

```c
// Archive operations
HANDLE SFileOpenArchive(const char* mpqName, DWORD priority, DWORD flags);
BOOL SFileOpenFileEx(HANDLE hMpq, const char* fileName, DWORD searchScope, HANDLE* phFile);
BOOL SFileReadFile(HANDLE hFile, void* buffer, DWORD toRead, DWORD* read, void* overlapped);
BOOL SFileCloseFile(HANDLE hFile);

// Decompression (internal)
int SCompDecompress(void* outBuf, int* outSize, void* inBuf, int inSize);
int SCompExplode(void* outBuf, int* outSize, void* inBuf, int inSize);
```

### String References to Search For

- `"(listfile)"` — read during archive initialization
- `"(hash table)"` — hash table decryption key name
- `"(block table)"` — block table decryption key name
- `"Storm.dll"` — the MPQ library name (Blizzard's internal name)
- `"PKLIB"` — PKWARE compression library identifier
- `"SFile"` — prefix for Storm file API functions
- `"Unable to"` or `"Error"` — error messages near file I/O

---

## Output Format

For each task, provide:
1. **Function address(es)** found
2. **Decompiled C code** (cleaned up with meaningful variable names)
3. **Analysis** — what the code does, how it relates to our problem
4. **Recommendation** — what we need to change in our implementation

**Priority**: Task 2 and Task 3 are the most critical. If type 0x08 is NOT standard PKWARE DCL in this binary, we need to know exactly what algorithm it IS so we can implement it correctly.
