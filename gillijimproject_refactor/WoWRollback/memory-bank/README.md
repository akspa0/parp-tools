# Memory Bank - WoWRollback.RollbackTool

## Overview
This memory bank contains complete documentation of the WoWRollback.RollbackTool project state as of **October 21, 2025**.

## Core Achievement
✅ **PROVEN WORKING**: Core rollback functionality successfully tested on Alpha 0.5.3 WDT files!
- Kalimdor: 951 tiles, 126,297 placements modified
- Azeroth: Multiple successful tests
- MD5 checksum generation confirmed

## Files in This Memory Bank

### 1. projectbrief.md
**What it contains**: High-level project vision and goals
- Project purpose: Time-travel WoW maps by UniqueID threshold
- Key features: Works on 0.5.3-3.3.5, pre-generated overlays, terrain hole management
- Success criteria and use cases

### 2. productContext.md
**What it contains**: User perspective and workflow
- Why this exists (problem solved)
- How it works (three-phase workflow)
- User experience flow (analyze → visualize → roll back)
- Technical approach and known limitations

### 3. activeContext.md
**What it contains**: Current work focus and recent accomplishments
- **Session 2025-10-21**: Core rollback tested and working!
- Technical breakthroughs (AdtAlpha integration)
- Verified file formats (MDDF/MODF layouts)
- Architecture decision (split into 3 tools)
- Next steps for fresh session

### 4. systemPatterns.md
**What it contains**: Architecture patterns and design decisions
- Three-tool separation (Analysis / Modification / Visualization)
- In-memory modification pattern
- Chunk access patterns
- Spatial MCNK mapping
- Pre-generation strategy
- Command structure
- Testing strategy

### 5. techContext.md
**What it contains**: Technical implementation details
- Runtime environment (.NET 9.0)
- Project structure (planned vs current)
- Critical dependencies (GillijimProject.WowFiles)
- Verified file format layouts (MDDF 36 bytes, MODF 64 bytes, MCNK 128 bytes)
- Output formats (JSON schemas)
- Proven implementation details
- Performance characteristics

### 6. progress.md
**What it contains**: Completed work and to-do list
- ✅ Core rollback functionality (TESTED!)
- ✅ MD5 checksum generation
- ⏳ MCNK terrain hole management
- ⏳ MCSH shadow disabling
- ⏳ Overlay generation
- ⏳ Lightweight viewer
- Test results and success metrics

## Quick Start for New Session

When starting a new session, read these files in order:

1. **projectbrief.md** - Understand the vision
2. **activeContext.md** - See what was just accomplished
3. **progress.md** - Check what's done and what's next
4. **systemPatterns.md** - Learn the architecture patterns
5. **techContext.md** - Get technical implementation details

## Key Locations in Code

### Current Implementation (Temporary)
```
WoWRollback/WoWDataPlot/Program.cs
  Lines ~1980-2180: Rollback command implementation
```

### Modified Library Files
```
src/gillijimproject-csharp/WowFiles/Alpha/AdtAlpha.cs
  - Added GetMddf() / GetModf() accessors
  - Added GetMddfDataOffset() / GetModfDataOffset()
  - Added _adtFileOffset field
```

## Test Data Locations
```
test_data/0.5.3/tree/World/Maps/
  ├── Azeroth/Azeroth.wdt  (tested successfully)
  └── Kalimdor/Kalimdor.wdt (tested successfully)
```

## Next Steps Summary

1. **Create WoWRollback.RollbackTool project**
2. **Extract rollback logic** from WoWDataPlot
3. **Implement MCNK hole management** (spatial calculations)
4. **Add MCSH shadow disabling** (optional feature)
5. **Generate overlay images** (pre-rendered PNGs)
6. **Build lightweight viewer** (HTML+JS slider)

## Git Branch
- Current: `wrb-poc5`
- Last commit: `58d0aae` - "WoWDataPlot - now with Rollback support (Tested on 0.5.3, it works!)"

---

**Note**: This memory bank is maintained to ensure continuity across sessions. Update it whenever significant progress is made or architecture decisions change.
