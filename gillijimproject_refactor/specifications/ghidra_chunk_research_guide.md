# WoW Chunk Format Research Guide — Ghidra Analysis

This guide describes how to systematically reverse-engineer WoW file format chunk structures across all game versions using Ghidra. The goal is to document every chunk variation for every known build, since WoW's file formats evolved continuously and no single reference covers all versions.

---

## Why This Is Needed

Existing documentation (wowdev.wiki) covers primarily:
- **1.12.1** (Vanilla final)
- **2.4.3** (TBC final)
- **3.3.5** (WotLK final)

The following versions have **little to no documentation**:
- **0.5.3** (Alpha) — Monolithic WDT, reversed FourCCs, packed MCLQ
- **0.6.0** (Alpha) — Split ADTs, transitional chunk layouts
- **0.7.x–0.12.x** (Closed Beta) — Rapidly evolving formats
- **1.0.0–1.11.x** (Vanilla pre-final) — Minor chunk additions
- **2.0.0–2.3.x** (TBC pre-final) — MH2O introduction period

---

## Setup

### Tools Required
- **Ghidra** (latest, with Decompiler)
- **WoW client executable** for the target build (e.g., `WoW.exe`, `WoWT.exe`)
- **Hex editor** (HxD, 010 Editor, or ImHex)
- **Sample data files** (WDT, ADT, WMO, MDX/M2) from the target build

### Ghidra Project Setup
1. Create a new Ghidra project per WoW build (e.g., `WoW_0.10.0.3892`)
2. Import the client executable
3. Run auto-analysis with default settings
4. Enable "Aggressive Instruction Finder" for better coverage

---

## Research Methodology

### Step 1: Find FourCC References

Search for FourCC chunk tags as 4-byte integer constants. WoW chunks use ASCII FourCCs stored as little-endian uint32:

| FourCC | Hex (LE) | Context |
|--------|----------|---------|
| `MVER` | `0x5245564D` | Version chunk (all files) |
| `MHDR` | `0x5244484D` | ADT header |
| `MCNK` | `0x4B4E434D` | ADT terrain chunk |
| `MCVT` | `0x5456434D` | Height values |
| `MCNR` | `0x524E434D` | Normals |
| `MCLY` | `0x594C434D` | Texture layers |
| `MCAL` | `0x4C41434D` | Alpha maps |
| `MCLQ` | `0x514C434D` | Liquid data |
| `MH2O` | `0x4F32484D` | Liquid data (post-TBC) |
| `MOBJ` | `0x4A424F4D` | WMO object |
| `MOGP` | `0x50474F4D` | WMO group |
| `MOTV` | `0x56544F4D` | WMO texture UVs |
| `MOVT` | `0x54564F4D` | WMO vertices |
| `MLIQ` | `0x51494C4D` | WMO liquid |

**Note:** Alpha 0.5.3/0.6.0 uses **reversed FourCCs** in some files (e.g., `KNMC` instead of `MCNK`). Search for both orientations.

### Step 2: Trace the Parsing Functions

1. Find cross-references to the FourCC constant
2. The referencing function is typically a chunk parser or dispatcher
3. Look for the pattern:
   ```c
   uint32_t tag = *(uint32_t*)ptr;
   uint32_t size = *(uint32_t*)(ptr + 4);
   switch(tag) {
       case 'MCVT': parse_mcvt(ptr + 8, size); break;
       case 'MCLQ': parse_mclq(ptr + 8, size); break;
       ...
   }
   ```
4. Follow each `parse_*` function to document the structure

### Step 3: Document the Structure

For each chunk, record:

```
CHUNK: [FourCC]
BUILD: [X.Y.Z.NNNN]
SIZE:  [fixed or variable]
PARENT: [containing chunk, e.g., MCNK]

STRUCTURE:
  offset 0x00: [type] [name] — [description]
  offset 0x04: [type] [name] — [description]
  ...

NOTES:
  - [Any version-specific behavior]
  - [Differences from wowdev.wiki documentation]
  - [Conditional fields based on flags]
```

---

## Priority Chunks to Research

### Tier 1: Critical (cause crashes/hangs)

These chunks have known format variations that cause viewer hangs or crashes:

#### MCLQ (Terrain Liquid)
- **Known issue**: Format changed between 0.6.0 and 3.3.5
- **Research**: Document the exact MCLQ header and per-vertex layout for each build
- **Key questions**:
  - When did the 9×9 vertex grid become standard?
  - When did flow direction data get added?
  - Are heights absolute or relative in each version?
  - What is the tile flag format per version?

#### MCNK (Terrain Chunk Header)
- **Known issue**: Header size and field offsets vary across versions
- **Research**: Document every field offset for each build
- **Key questions**:
  - When did `AreaID` move from offset 0x38 to its 3.3.5 position?
  - When were `MH2O` offsets added?
  - What flags exist per version?

#### MH2O (New Liquid System)
- **Known issue**: Introduced during beta, replaced MCLQ
- **Research**: Find the exact build where MH2O first appears
- **Key questions**:
  - What build introduced MH2O?
  - Did MCLQ and MH2O coexist in any builds?
  - What is the header format in early MH2O builds?

### Tier 2: Important (cause visual bugs)

#### MCLY (Texture Layers)
- **Research**: Layer entry size and flag meanings per build
- **Key questions**:
  - When did compressed alpha maps become standard?
  - What texture flags exist per version?

#### MCAL (Alpha Maps)
- **Research**: Compression format per build
- **Key questions**:
  - When did 2048-byte uncompressed become 4096-byte?
  - When was big-alpha (4096 uncompressed) introduced?
  - Compression algorithm differences?

#### MOTV (WMO Texture UVs)
- **Research**: Can there be multiple MOTV chunks per group?
- **Key questions**:
  - When was the second MOTV (lightmap UVs) introduced?
  - Is the V coordinate flipped in any version?

#### MOGP (WMO Group Header)
- **Research**: Header size and field offsets per build
- **Key questions**:
  - When was `groupLiquid` added?
  - What do the flag bits mean per version?
  - When did the `transBatchCount` field appear?

### Tier 3: Nice to Have

#### MDDF / MODF (Placement Data)
- **Research**: Entry size and fields per build
- **Key questions**:
  - When was `uniqueId` added?
  - When did scale become a float vs uint16?

#### MCCV (Vertex Colors)
- **Research**: When introduced, format per build

#### MCLV (Vertex Lighting)
- **Research**: When introduced, relationship to MCCV

---

## File Organization

**Save each chunk's documentation in a separate markdown file:**

```
specifications/chunks/
├── ADT/
│   ├── MCNK.md
│   ├── MCVT.md
│   ├── MCNR.md
│   ├── MCLY.md
│   ├── MCAL.md
│   ├── MCLQ.md
│   ├── MH2O.md
│   ├── MCCV.md
│   ├── MDDF.md
│   ├── MODF.md
│   └── ...
├── WMO/
│   ├── MOHD.md
│   ├── MOGP.md
│   ├── MOTV.md
│   ├── MOVT.md
│   ├── MOPY.md
│   ├── MOBA.md
│   ├── MLIQ.md
│   └── ...
├── MDX/
│   ├── VERS.md
│   ├── GEOS.md
│   ├── UVAS.md
│   ├── UVBS.md
│   └── ...
└── WDT/
    ├── MPHD.md
    ├── MAIN.md
    └── ...
```

### Per-Chunk File Template

```markdown
# [FOURCC] — [Full Name]

## Summary
[One-line description of what this chunk contains]

## Parent Chunk
[e.g., MCNK, MOGP, or root-level]

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | ... | ... |
| 0.6.0.3592 | ... | ... |
| 0.10.0.3892 | ... | ... |
| 3.3.5.12340 | ... | ... |

## Structure — [Build X.Y.Z.NNNN]

| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | uint32 | ... | ... |
| 0x04 | float | ... | ... |

## Structure — [Build X.Y.Z.NNNN] (if different)
...

## Version Differences
- **0.5.3 → 0.6.0**: [what changed]
- **0.6.0 → 0.10.0**: [what changed]
- **0.10.0 → 3.3.5**: [what changed]

## Ghidra Notes
- **Function address**: `0x00XXXXXX` in [build]
- **Parser pattern**: [describe the decompiled logic]
- **Key observations**: [anything unusual]

## References
- wowdev.wiki: [link if applicable]
- [Other sources]
```

---

## Workflow

1. **Pick a chunk** from the priority list
2. **Open the target build's executable** in Ghidra
3. **Search for the FourCC** constant (both forward and reversed)
4. **Trace the parser** function and document the structure
5. **Save to the per-chunk markdown file** immediately
6. **Compare across builds** — open multiple Ghidra projects side-by-side
7. **Note any conditional logic** (flag-based field inclusion, version checks)
8. **Save often** — the amount of information per chunk can be massive

### Tips

- **Save after every chunk.** Don't try to batch — you will lose work.
- **Screenshot the Ghidra decompiler output** for complex parsers.
- **Note the function address** so you can return to it later.
- **Check for version checks in code** — the client often has `if (version >= X)` guards.
- **Cross-reference with hex dumps** of actual data files to verify your structure.
- **Start with MCLQ** — it's the most likely cause of viewer hangs on transitional builds.

---

## Delivering Results

When delivering chunk research results:

1. **One chunk per message** — keeps things manageable
2. **Use the template above** — consistent format speeds integration
3. **Include the build number** — always specify the exact X.Y.Z.NNNN
4. **Note confidence level** — "confirmed via hex dump" vs "inferred from decompiler"
5. **Flag unknowns** — mark fields you couldn't identify with `???`
6. **Include raw offsets** — absolute offsets from the Ghidra listing

The results will be integrated into the MdxViewer's terrain adapters and format parsers to support all WoW versions correctly.
