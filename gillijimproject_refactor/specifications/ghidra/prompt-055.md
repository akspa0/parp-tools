# Ghidra LLM Prompt — WoWClient.exe 0.5.5.3494

**Binary**: WoWClient.exe (Alpha 0.5.5 build 3494)
**PDB**: NOT available. Use 0.5.3 PDB symbols as reference for function identification.
**Architecture**: x86 (32-bit)

---

## Context for the LLM

You are reverse engineering WoW Alpha 0.5.5 (build 3494) using Ghidra. This binary does NOT have a PDB file. However, it is only ~126 builds newer than 0.5.3.3368 which DOES have a PDB. Many functions will be at similar addresses or have identical byte patterns.

### Strategy: Diff Against 0.5.3

1. Use 0.5.3 PDB function names as a reference
2. Search for identical byte sequences to locate the same functions
3. Focus on functions that DIFFER — these reveal format changes between 0.5.3 and 0.5.5
4. String constants are your friend — error messages and chunk signatures don't change

### What We Know About 0.5.3 (Reference)

- WDT: Monolithic with embedded ADTs, column-major MAIN indexing
- Chunks: Forward FourCC byte order
- MCVT/MCNR: Non-interleaved (81 outer + 64 inner)
- WMO: v14 monolithic format
- Models: MDX format
- Coordinate system: X=North, Y=West, Z=Up, file stores (X, Z, Y)

---

## Research Tasks

### Task 1: Identify Format Changes from 0.5.3

**Goal**: Find any chunk or structure differences between 0.5.3 and 0.5.5.

**Method**:
1. Find the WDT/ADT loading functions (search for "MAIN", "MCNK", "MCVT" strings)
2. Compare the chunk reader dispatch table against 0.5.3
3. Look for new chunk signatures that don't exist in 0.5.3
4. Check if any chunk sizes or header layouts changed

**Key questions**:
- Did the MCNK header size change?
- Are there any new subchunks?
- Did the WDT MAIN entry format change?
- Did MCVT/MCNR switch to interleaved format?

### Task 2: WMO Version Check

**Goal**: Determine if 0.5.5 still uses WMO v14 or has started transitioning.

**Method**:
1. Find the WMO loader
2. Check the MOHD version field handling
3. Look for any v17 split-file logic (loading "_000.wmo" group files)

### Task 3: MDX vs M2 Transition Check

**Goal**: Determine if 0.5.5 supports M2 format in addition to MDX.

**Method**:
1. Find the model loading function
2. Check if it looks for "MD20" or "MD21" magic in addition to "MDLX"
3. Check file extension handling (.mdx vs .m2)

### Task 4: MCLQ Liquid Verification

**Goal**: Verify MCLQ format matches 0.5.3 or document changes.

**Method**:
1. Find MCLQ chunk reader
2. Compare structure layout against 0.5.3 findings
3. Note any additional fields or changed sizes

---

## Tips for This Binary

- **No PDB** — rely on string searches, constant matching, and byte-pattern comparison with 0.5.3
- The vtable layout should be very similar to 0.5.3 — find `CMap`, `CMapChunk`, `CMapObj` classes
- Check the PE header timestamp/version to confirm the build number
- If a function is identical to 0.5.3, note it as "unchanged" and move on
- Focus effort on functions that DIFFER — those are the valuable findings

---

## Output Format

For each finding, provide:
1. **Function address** in this binary
2. **Corresponding 0.5.3 function** (address and PDB name if matched)
3. **Status**: Unchanged / Modified / New / Removed
4. **Details** of any changes found
5. **Confidence level** (high/medium/low)
