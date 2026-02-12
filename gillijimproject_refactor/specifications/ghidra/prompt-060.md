# Ghidra LLM Prompt — WoWClient.exe 0.6.0.3592

**Binary**: WoWClient.exe (Alpha 0.6.0 build 3592)
**PDB**: NOT available. Use 0.5.3 PDB symbols as reference.
**Architecture**: x86 (32-bit)

---

## Context for the LLM

You are reverse engineering WoW Alpha 0.6.0 (build 3592) using Ghidra. This is a **transitional** build — it sits between the Alpha format (0.5.x) and the format used in later versions (1.x+). This build likely contains format changes that were precursors to the Beta/Release client formats.

### Why 0.6.0 Matters

This is the last "Alpha" build before major format changes. Key transitions that may appear here:
- WMO v14 → v17 (monolithic → split root + group files)
- MDX → M2 model format transition
- MCVT/MCNR non-interleaved → interleaved vertex layout
- WDT MAIN column-major → row-major indexing
- Chunk FourCC forward → reversed byte order

### Strategy

1. Use 0.5.3 PDB function names to identify functions via byte-pattern matching
2. Focus on **transition points** — functions that handle BOTH old and new formats
3. Look for version checks, format detection branching, or dual-path code
4. Document which changes have already happened vs still pending

---

## Research Tasks

### Task 1: WDT/ADT Format Detection

**Goal**: Determine if 0.6.0 uses monolithic WDT or separate ADT files.

**Method**:
1. Find WDT loading code
2. Check if it opens separate `MapName_XX_YY.adt` files or reads embedded data
3. Check MAIN chunk entry format — does it store file offsets (Alpha) or flags (LK)?
4. Check FourCC byte order — forward or reversed?

**Key question**: Is this the build where WDT/ADT split happened?

### Task 2: MCVT/MCNR Vertex Layout

**Goal**: Determine if heights/normals are interleaved or non-interleaved.

**Method**:
1. Find MCVT reader function
2. Check if it reads 81+64 sequentially (non-interleaved) or alternates 9/8 rows (interleaved)
3. Same for MCNR

### Task 3: WMO Format Version

**Goal**: Determine WMO version used.

**Method**:
1. Find WMO loading function
2. Check MOHD version handling
3. Look for group file loading logic (`*_000.wmo`)
4. Check if v14 parsing code still exists (backward compat?)

### Task 4: Model Format (MDX vs M2)

**Goal**: Determine which model format 0.6.0 uses.

**Method**:
1. Find model loader
2. Check for "MDLX" (MDX) vs "MD20"/"MD21" (M2) magic detection
3. Check if both are supported or only one

### Task 5: MCNK Header Layout

**Goal**: Compare MCNK header against 0.5.3.

**Method**:
1. Find MCNK parser
2. Trace all header field accesses
3. Compare offsets and field sizes against 0.5.3
4. Note any new fields or changed semantics

### Task 6: Chunk FourCC Byte Order

**Goal**: Determine if FourCCs are forward or reversed on disk.

**Method**:
1. Find chunk parser loop (the main dispatch that reads FourCC + size)
2. Check if it reads bytes forward and compares against "MCNK" or reversed "KNCM"
3. This is a simple but critical check

---

## Tips for This Binary

- **No PDB** — use 0.5.3 symbol names as reference
- This is the most important binary for understanding format transitions
- Look for `if (version >= X)` style branching — indicates format evolution
- Pay special attention to any #ifdef or conditional compilation artifacts
- Error strings may reference new chunk names not present in 0.5.3
- The binary may contain dead code paths for deprecated formats

---

## Output Format

For each finding, provide:
1. **Function address** in this binary
2. **Corresponding 0.5.3 function** (if identifiable)
3. **Format determination**: Alpha-style / LK-style / Transitional (both)
4. **Evidence** supporting the determination
5. **Confidence level** (high/medium/low)
