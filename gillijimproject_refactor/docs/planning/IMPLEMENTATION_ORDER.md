# WoWRollback Implementation Order 🗺️

**Priority-ordered implementation plan for WoWRollback project.**

---

## 📋 Phase Priorities

### ✅ **FIRST: Fix AreaTable Source** (1-2 days)
**Document**: `FIX_AREATABLE_SOURCE.md`

**Why First**: 
- Critical bug (reading from wrong source)
- Prerequisite for Phase 0
- Quick win

**Tasks**:
- Add MCNK reading to `LkAdtReader.cs`
- Create `LkAdtTerrainReader.cs`
- Update overlay builders
- Test area ID accuracy

---

### 🎯 **Phase 0: WoWRollback Core Feature** (4 weeks) ⏮️
**Document**: `PHASE_0_ROLLBACK_FEATURE.md`

**Why Before Consolidation**:
- ✅ This IS the namesake feature
- ✅ Proves the concept
- ✅ Can demo immediately
- ✅ Informs consolidation architecture

**Deliverables**:
- UniqueID analysis (CSV reports)
- Layer detection algorithm
- Timeline slider in viewer
- Asset filtering (global + per-tile)
- Export filtered LK ADTs

---

### 🔧 **Phases 1-5: Tool Consolidation** (5 weeks)
**Document**: `MASTER_PLAN.md`

**Goals**:
- Merge 3 tools → 1 unified tool
- Multi-threading (6.5x speedup)
- Beautiful CLI (Spectre.Console)

**Phases**:
1. Core format readers
2. Multi-threaded processing
3. CLI integration
4. Testing
5. Cleanup

---

### 🎨 **Phase 6: 3D Export & Visualization** (10-15 weeks)
**Documents**:
- `PHASE_6_3D_EXPORT.md` (terrain)
- `PHASE_6B_MDX_AND_TEXTURES.md` (MDX + textures)
- `PHASE_6C_WMO_SUPPORT.md` (WMOv14)

**Goals**:
- GLB/glTF export
- Alpha MDX support
- Texture baking
- WMOv14 conversion
- Complete 3D asset pipeline

---

## 🎯 What to Work On Next

### Today: Fix AreaTable Source
```
See: FIX_AREATABLE_SOURCE.md
Add ReadMcnkChunks() to LkAdtReader.cs
Create LkAdtTerrainReader.cs
Test on Dun Morogh
```

### This Week: Phase 0 - Rollback Feature
```
See: PHASE_0_ROLLBACK_FEATURE.md
Implement UniqueID analysis
Build timeline slider
Test time-travel visualization
```

### This Month: Tool Consolidation
```
See: MASTER_PLAN.md
After Phase 0 is complete and demoed
Merge AlphaWDTAnalysisTool + DBCTool.V2
Apply multi-threading optimizations
```

### This Quarter: 3D Export
```
See: PHASE_6_*.md
After consolidation is stable
Implement complete 3D asset pipeline
GLB export for modern tools
```

---

## 📊 Timeline Summary

| Phase | Duration | Priority | Status |
|-------|----------|----------|--------|
| **Fix AreaTable** | 1-2 days | 🔥 CRITICAL | ⏳ Next |
| **Phase 0** | 4 weeks | 🎯 HIGH | ⏳ Planned |
| **Phases 1-5** | 5 weeks | ⚡ MEDIUM | 📋 Planned |
| **Phase 6** | 10-15 weeks | 💎 NICE-TO-HAVE | 📋 Planned |
| **Total** | ~20 weeks | | |

---

## 🚀 Quick Start for Next Session

```
I'm working on the WoWRollback project. 

Next task: Fix AreaTable source (see docs/planning/FIX_AREATABLE_SOURCE.md)

Problem: Currently reading area IDs from CSV (Alpha WDT data).
Solution: Read directly from cached LK ADT files.

Starting with: Add ReadMcnkChunks() method to LkAdtReader.cs
```

---

## 📚 All Planning Documents

1. `IMPLEMENTATION_ORDER.md` (this file) - Priority overview
2. `FIX_AREATABLE_SOURCE.md` - AreaTable bug fix
3. `PHASE_0_ROLLBACK_FEATURE.md` - Time-travel feature
4. `MASTER_PLAN.md` - Consolidation strategy
5. `SESSION_CONTEXT.md` - Quick resume guide
6. `ARCHITECTURE_CHANGES.md` - Before/after architecture
7. `05-wowrollback-consolidation.md` - Detailed design
8. `ALPHAWDT_MULTITHREADING.md` - Performance strategy
9. `PHASE_6_3D_EXPORT.md` - 3D export core
10. `PHASE_6B_MDX_AND_TEXTURES.md` - MDX + textures
11. `PHASE_6C_WMO_SUPPORT.md` - WMOv14 support

---

**Work flows top-to-bottom. Start with AreaTable fix, then Phase 0!** 🎯
