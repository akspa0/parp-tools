# 🚀 START HERE - WoWRollback Implementation Order

**Read this first! Priority-ordered plan for WoWRollback project.**

> **Note**: All planning documents are now organized by priority (see `README.md`)

---

## 📋 Phase Priorities

### 🚨 **FIRST: Fix Missing Terrain Overlays** (1 day)
**Document**: `01_CRITICAL_Missing_Terrain_Overlays.md`

**Why First**:
- CRITICAL: Blocks viewer for most maps
- Shadow maps don't work
- Only Azeroth/Kalimdor have overlays

**Tasks**:
- Investigate why CSVs missing for instances
- Add logging to overlay generation
- Fix shadow map generation
- Test on DeadminesInstance

---

### ✅ **SECOND: Fix AreaTable Source** (1-2 days)
**Document**: `02_CRITICAL_AreaTable_Fix.md`

**Why Second**:
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
**Document**: `03_Rollback_TimeTravel_Feature.md`

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
**Document**: `04_Tool_Consolidation.md`

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
**Documents** (in `future/` folder):
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

### Today: Fix Missing Terrain Overlays
```
See: 01_CRITICAL_Missing_Terrain_Overlays.md
Investigate CSV generation
Add logging to overlay builders
Test on DeadminesInstance
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
| **Fix Terrain Overlays** | 1 day | 🚨 **CRITICAL** | ⏳ Next |
| **Fix AreaTable** | 1-2 days | 🔥 CRITICAL | ⏳ Planned |
| **Phase 0** | 4 weeks | 🎯 HIGH | ⏳ Planned |
| **Phases 1-5** | 5 weeks | ⚡ MEDIUM | 📋 Planned |
| **Phase 6** | 10-15 weeks | 💎 NICE-TO-HAVE | 📋 Planned |
| **Total** | ~21 weeks | | |

---

## 🚀 Quick Start for Next Session

```
I'm working on the WoWRollback project. 

Next task: Fix missing terrain overlays (see docs/planning/01_CRITICAL_Missing_Terrain_Overlays.md)

Problem: Only Azeroth/Kalimdor have terrain overlays; instances missing.
Problem: Shadow maps don't work.

Starting with: Investigate why CSVs aren't generated for all maps.
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
