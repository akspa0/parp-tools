# Ghidra Definitive Guide Workflow (All Versions)

This guide defines how to build small, tidy analysis outputs and then compose them into definitive format specifications for any WoW client version (including 0.5.3 and later).

## Goals
- Produce small, focused Markdown outputs that each answer one question.
- Capture hard evidence (function addresses, checks, offsets, sizes).
- Keep outputs tidy and consistent so they can be merged into a definitive reference.
- Repeat the same workflow for other versions to enable accurate comparisons.

## Recommended folder layout
Use a consistent versioned layout under specifications/outputs:

```
specifications/outputs/
  053/
    <topic folders>
    <definitive files>
  060/
  335/
  4.0.0.11927/
```

Recommended naming:
- Version root: 053, 060, 335, 4.0.0.11927
- Topic folders: ADTs, lighting, MPQ, mdx-alignment, etc.
- Definitive file: `alpha-wdt-definitive.md` (or `<topic>-definitive.md`).

## Tiny output file standard (one topic per file)
Each topic file should follow a fixed template so it is easy to merge later.

**Required headings:**
- `# <Title>`
- `## <Question>`
- `### Function addresses`
- `### Decompiled evidence`
- `### Definitive answer`
- `### Constants` (optional)
- `### Confidence`

**Template:**
```markdown
# <Title>

## <Question>

### Function addresses
- `FunctionName` @ `0x00000000`

### Decompiled evidence
```c
// Short, focused excerpts only
```

### Definitive answer
- Clear, testable statement derived from evidence.

### Constants
- `CONST_NAME = 0x00`

### Confidence
- High | Medium | Low (with a one-line rationale)
```

**Rules:**
- One question per file.
- Show only the minimal code excerpt needed to prove the claim.
- Put offsets and sizes in the answer even if they appear in evidence.
- Avoid speculation in the final answer section.

## Ghidra setup (repeatable)
1. Import the target binary and load symbols (PDB if available).
2. Verify base address and image format are correct.
3. Run auto-analysis with default options.
4. Save the project and export function list once.

## Core evidence collection loop
For each question:
1. Identify the entry function (e.g., loader or parser).
2. Confirm token checks or magic constants.
3. Record struct field offsets accessed in decompiled code.
4. Record any pointer arithmetic that defines size or layout.
5. Extract only the minimal code snippet needed to justify the answer.
6. Write the tiny output file using the template above.

## Checklist by data type
Use these checklists to ensure full coverage in each topic.

### File headers and chunk order
- Token check order (exact sequence).
- Fixed size reads (exact bytes).
- Optional chunk detection logic.
- Endianness handling (how tokens are compared).

### Tables and entry layouts
- Total table size.
- Entry size and count.
- Indexing order (row-major vs column-major).
- Per-entry fields and offsets.

### Subchunk layouts
- Fixed-size blocks vs variable-length blocks.
- Relative vs absolute offsets.
- Stride values for arrays.
- Loop bounds (counts) tied to fields.

### Rendering data
- Alpha or shadow unpacking rules.
- Texture mip sizes based on runtime flags.
- Combine paths (shader textures, packed data).

### Placement data
- MDDF/MODF or equivalent placement structures.
- Rotation order and axis.
- Scale encoding.
- Bounds usage and inverse matrix creation.

### Liquids
- Presence or absence of chunk headers.
- Per-instance size and layout.
- Per-vertex layout and grid dimensions.
- Type mapping to flags or enums.

## Expanded subsystem checklist (use when no PDB exists)
This checklist is designed for versions without symbols. It focuses on discoverable patterns, strings, and call sequences that reliably identify key functions.

### WDT / ADT / MCNK parsing
- Look for token checks using string literals like `"iffChunk->token=='MVER'"`, `"MPHD"`, `"MAIN"`, `"MHDR"`, `"MCIN"`, `"MCNK"`.
- Identify `SFile::Read` or equivalent wrapper in tight loops with `8`-byte reads (IFF header) and fixed-size reads (0x80, 0x10000).
- Confirm MAIN entry size via `0x10000 / 4096` and indexing expression (`y * 0x40 + x`).
- Find MCNK parser by a sequence of fixed-size pointer advances (MCVT + MCNR) before a `MCLY` check.

### MDDF / MODF placements
- Search for `size >> 6` (MODF count) and a magic divide (MDDF count).
- Follow placement creation: translation + Z-axis rotation using a constant axis vector `(0,0,1)`.
- Confirm scale encoding (uint16 / 1024.0f) by spotting float conversion and division.

### MCAL (alpha) and MCSH (shadow)
- Look for loops over 0x1000 (4096 pixels) with nibble extraction from a byte stream (MCAL).
- Look for mip-size branches (32 vs 64) using a global like `shadowMipLevel`.
- Find a shader texture combine path that merges alpha with shadow into a GX/D3D texture.

### Liquids
- Search for liquid type checks with `type < 4` or `LQ_LAST` patterns.
- In Alpha, find a fixed-size inline copy loop (0x324 bytes) tied to MCNK flags bits 2..5.
- Identify 9x9 vertex grid loops or 81/162 element loops in liquid parsing.

### WMO loading / rendering
- Identify WMO name table loads (`MONM` token check) and placement creation.
- Look for group render calls with a group index loop (up to 0x180 groups in Alpha).
- Find liquid handling in WMO by searching for `"MLIQ"` token checks.

### MDX loading
- Find a central model build routine with sequential chunk readers (MODL/TEXS/GEOS/etc.).
- Use strings like `".mdx"`, `".mdl"`, or known chunk names in the binary.

### MPQ / file IO
- Identify MPQ open/read functions by strings like `"SFile"`, `"MPQ"`, `"Storm"`.
- Look for decompression dispatch tables and magic constants for compression types.

### BLP / texture loading
- Search for `"BLP"` magic, `.blp` strings, and DXT format checks.
- Confirm fallback paths when DXT is unsupported.

### Lighting / fog
- Look for `.lit` strings or a loader that reads a version + count + groups.
- Find per-object fog setters and world fog query functions.

## Pattern library (no PDB heuristics)
Use these patterns to locate functions by behavior when names are missing.

### Token check pattern
```
Read 8 bytes -> compare token -> error string "iffChunk->token=='XXXX'"
```

### MCNK parse pattern
```
Read MCNK chunk -> skip 0x08 + 0x80 -> read 0x244 -> read 0x1C0 -> check MCLY
```

### Alpha map unpack pattern
```
Loop 0..0x1000 -> nibble extraction -> alpha in high byte | 0xFFFFFF
```

### Shadow combine pattern
```
Select mip size 0x20/0x40 -> unpack shadow -> merge alpha+shadow -> GxTexUpdate
```

## Cross-version symbol transfer (no direct PDB)
You cannot apply a PDB from one binary directly to a different version, but you can still transfer knowledge effectively:

### Use Function ID matching
- Enable Ghidra's Function ID and generate signatures for the known build.
- Run Function ID on the target build and accept high-confidence matches.
- This is the fastest way to carry over names when code is largely unchanged.

### Diff and merge labels
- Use Ghidra program diffing to compare the known build against the target.
- Apply matched function names and comments manually or via the diff tool.
- Prioritize functions with unique string references and large basic blocks.

### Signature-based renaming
- For key functions, build custom signatures from instruction sequences.
- Apply these signatures to the target binary and rename on match.

### Structure offset validation
- Even if a function matches, re-verify field offsets and sizes.
- Only copy struct layouts once the offset usage pattern is confirmed.

## Version-specific notes
Do not assume later-version behavior is present in Alpha (or vice versa). Always verify:
- FourCC order comparisons in the binary.
- Presence of header fields and their offsets.
- Differences in chunk ordering or optional chunks.

Record version deltas in a dedicated file, e.g.:
- `specifications/outputs/060/adt-deltas-from-053.md`

## Building a definitive guide (merge phase)
Once the tiny files are complete for a topic, build a definitive file with:
- A full, verbose narrative of load flow and parsing steps.
- All struct layouts in one place.
- All offsets/sizes listed in a quick-reference block.
- All runtime behavior notes (alpha/shadow combine, lighting, liquids).

**Definitive file structure:**
1. Scope and evidence
2. High-level runtime flow
3. File-level structure and chunk order
4. Tables and indices
5. Subchunk layouts
6. Runtime behavior (rendering, placement, lighting)
7. Quick-reference offsets and sizes
8. Open questions

## Keeping outputs tiny and tidy
- Keep each output file under ~200 lines unless necessary.
- Prefer multiple small files over one large draft.
- Use consistent headings and structure across versions.
- Avoid duplicating evidence; link to the definitive file for summary.

## Suggested workflow for a new version
1. Create version folder under specifications/outputs.
2. Write a function map file with addresses and names.
3. Create tiny files for each major subsystem:
   - WDT/ADT/MCNK parsing
   - Alpha/shadow unpacking
   - MDDF/MODF placements
   - Liquids
   - WMO/MDX loading
4. Merge into definitive specs once the tiny files are done.
5. Create a delta file comparing with the previous known version.

## Evidence style conventions
- Always include function addresses.
- Use short code excerpts with a comment above describing the purpose.
- State offsets in hex and sizes in bytes.
- Keep the "Definitive answer" concise and unambiguous.

## Common pitfalls
- Mixing reversed and forward FourCC without normalization.
- Assuming table indexing order without validating the indexing expression.
- Treating offsets as relative to the end of the header when they are relative to chunk start.
- Failing to account for fixed-size blocks that skip chunk headers in Alpha.

## Output example index
For each version, keep a simple index file listing all tiny outputs and the definitive guide(s). This helps future merges and comparisons.

```
# Index (053)
- WDT/ADT: terrain-loading-053-mcnk-deep-dive.md
- Liquids: u005-mclq-complete-format.md
- Placements: terrain-loading-053-mddf-modf.md
- Definitive: alpha-wdt-definitive.md
```
