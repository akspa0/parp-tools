# Ghidra LLM Prompt — Wow.exe 4.0.0.11927 (Cataclysm Alpha)

**Binary**: Wow.exe (Cataclysm Alpha 4.0.0 build 11927)
**PDB**: NOT available.
**Architecture**: x86 (32-bit)

---

## Context for the LLM

You are reverse engineering WoW Cataclysm Alpha (4.0.0 build 11927) using Ghidra. This is an **early Cataclysm build** that still uses **3.3.5-style data file formats** — unified ADTs (not split), same WDT/MCNK structure, same WMO v17, same M2 format. It does NOT have split ADTs (_tex0/_obj0). Use 3.3.5 findings as a baseline and focus on what's different or new in this transitional build.

### Key Facts About This Build

1. **Data formats are 3.3.5-compatible** — unified ADT files, no split _tex0/_obj0
2. **Chunk FourCCs** are **reversed** on disk (same as 3.3.5)
3. **MCVT/MCNR** are **interleaved** (same as 3.3.5)
4. **WDT MAIN** is **row-major** (same as 3.3.5)
5. **WMO v17** split root + group files (same as 3.3.5)
6. **M2** format with MD20/MD21 header (same as 3.3.5)
7. This build may contain **early implementations** of Cataclysm features that didn't ship until later builds

### Why This Build Matters

Being a Cataclysm Alpha, it sits between the well-known 3.3.5 format and the later Cataclysm changes (split ADTs, MAID chunk, etc.). It's valuable for:
- Finding **early traces** of upcoming format changes (new flags, new optional chunks)
- Understanding **what changed in the client code** even if the data format stayed the same
- Verifying that our 3.3.5 parsing handles this build's data correctly
- Identifying any **new rendering features** (water, lighting, terrain shading)

### Coordinate System

Same as 3.3.5: X=North, Y=West, Z=Up. File positions stored as (X, Z, Y).
- MapOrigin = 17066.66666
- ChunkSize = 533.33333

---

## Research Tasks

### Task 1: ADT Loading — Confirm Unified Format

**Goal**: Verify this build uses unified ADTs (no split files).

**Method**:
1. Find ADT loading function (search for ".adt" string references)
2. Check if there's ANY code trying to load `_tex0.adt` or `_obj0.adt`
3. If found, is it active or dead code?
4. Check WDT loading for MAID chunk handling — does it exist yet?

**Key question**: Is split ADT support present in code but not yet used by data files?

### Task 2: New MPHD Flags

**Goal**: Check if MPHD has new flag bits compared to 3.3.5.

**Method**:
1. Find MPHD reader in WDT loading
2. Trace all flag bit checks
3. Compare against known 3.3.5 flags:
   - 0x0001: WdtUsesGlobalMapObj
   - 0x0002: AdtHasMccv
   - 0x0004: AdtHasBigAlpha
   - 0x0008: AdtHasDoodadRefsSortedBySizeCat
4. Look for checks of bits 0x0010+ that don't exist in 3.3.5

### Task 3: MCNK Subchunk Inventory

**Goal**: Check for new MCNK subchunks not present in 3.3.5.

**Method**:
1. Find MCNK chunk dispatcher
2. List all handled FourCCs
3. Flag any that don't appear in 3.3.5 (MCLV, MTXF, etc.)
4. For new chunks, document their reader structure

**Expected candidates**:
- MCLV — vertex lighting (Cata feature, may already be present)
- MTXF — height-based texturing (Cata feature)

### Task 4: M2 Header Version

**Goal**: Check if M2 version field has incremented from 3.3.5's 264.

**Method**:
1. Find M2 loader (search for "MD20" = 0x3032444D)
2. Check what version value it expects/accepts
3. Look for any new chunk handlers in the MD21 wrapper path
4. Check for TXID (texture file data IDs) or SFID (skin file data IDs) handling

### Task 5: Water/Liquid Rendering Changes

**Goal**: Check if liquid rendering has changed from 3.3.5.

**Method**:
1. Find MCLQ reader in MCNK handling
2. Check if there's a new water system (MH2O is the LK water chunk in the ADT header area)
3. Compare water vertex format against 3.3.5
4. Look for new water shaders or rendering paths

### Task 6: Rendering Pipeline Changes

**Goal**: Identify any rendering changes (shaders, techniques).

**Method**:
1. Look for new shader string constants or HLSL references
2. Check terrain rendering path for new uniform variables or texture bindings
3. Look for deferred rendering or new lighting model code
4. Check for new blend modes or material flags

---

## Tips for This Binary

- **No PDB** — use 3.3.5 function patterns and byte-sequence matching as reference
- Since data formats are 3.3.5-compatible, many functions will be identical — focus on what DIFFERS
- String constants are the fastest way to locate functions — search for chunk FourCCs and file extensions
- The vtable structure should be very similar to 3.3.5 — `CMap*`, `CM2Model*`, `CWorldModel*`
- Check for `#if`/`#ifdef` artifacts — preprocessor-excluded code sometimes leaves traces
- New features may be present but gated behind flags that are never set in the data files

---

## Output Format

For each finding, provide:
1. **Function address** in this binary
2. **Corresponding 3.3.5 function** (if identifiable)
3. **Status**: Unchanged / Modified / New / Removed
4. **Structure definitions** for any new or changed formats
5. **Confidence level** (high/medium/low)
