# WoWRollback Planning Documents 📋

**Organized by priority - work from top to bottom!**

---

## 🚀 Quick Start

1. Read **`00_START_HERE.md`** for overview
2. Work on **`01_CRITICAL_AreaTable_Fix.md`** (immediate bug fix)
3. Then **`02_Rollback_TimeTravel_Feature.md`** (the namesake feature!)
4. Continue with consolidation (files `03_*`)

---

## 📂 Current Planning Documents

### Priority 1: Critical Bug Fixes 🚨
- **`00_START_HERE.md`** - Implementation order & priorities
- **`01_CRITICAL_Missing_Terrain_Overlays.md`** - Fix missing overlays & shadow maps (1 day)
- **`02_CRITICAL_AreaTable_Fix.md`** - Fix area ID sourcing (1-2 days)

### Priority 2: Core Feature 🎯
- **`03_Rollback_TimeTravel_Feature.md`** - Time-travel visualization (4 weeks)

### Priority 3: Tool Consolidation & Performance 🔧
- **`04_Tool_Consolidation.md`** - Merge 3 tools → 1, multi-threading (5 weeks)
- **`04_Tool_Consolidation_Details.md`** - Detailed architecture
- **`04_MultiThreading_Strategy.md`** - Performance optimization strategy
- **`04_Architecture_Changes.md`** - Before/after architecture diagrams

### Utilities
- **`SESSION_CONTEXT.md`** - Quick resume guide for fresh chat sessions

---

## 📁 Folder Structure

### `completed/`
Finished features and milestones:
- `SHADOW_MAPS_COMPLETE.md` - Shadow map overlay implementation
- `SMART_CACHE_OPTIMIZATION.md` - Caching strategy
- `TILE_COUNT_VALIDATION.md` - WDT validation
- `PHASE0_PROGRESS.md` - Early progress tracking

### `archive/`
Historical planning documents (reference only):
- Early area table mapping explorations
- Viewer pipeline improvements (v1)
- Terrain feature plans (v1)
- 3D viewer vision (v1)

### `future/`
Long-term enhancements (after core features):
- `PHASE_6_3D_EXPORT.md` - Terrain 3D export
- `PHASE_6B_MDX_AND_TEXTURES.md` - MDX models + texture baking
- `PHASE_6C_WMO_SUPPORT.md` - WMOv14 support

---

## 🎯 Current Focus

**TODAY**: Fix missing terrain overlays & shadow maps
**This Week**: Fix AreaTable source + start Phase 0  
**This Month**: Phase 0 - Rollback time-travel feature  
**This Quarter**: Tool consolidation & multi-threading

---

## 📊 Implementation Timeline

| Phase | Document | Duration | Priority | Status |
|-------|----------|----------|----------|--------|
| **Missing Overlays** | `01_CRITICAL_*.md` | 1 day | 🚨 **NOW** | ⏳ Next |
| **AreaTable Fix** | `02_CRITICAL_*.md` | 1-2 days | 🔥 **CRITICAL** | 📋 Planned |
| **Phase 0** | `03_Rollback_*.md` | 4 weeks | 🎯 **HIGH** | 📋 Planned |
| **Phases 1-5** | `04_Tool_*.md` | 5 weeks | ⚡ MEDIUM | 📋 Planned |
| **Phase 6** | `future/*.md` | 10-15 weeks | 💎 LATER | 📋 Future |

---

## 💡 Tips for Fresh Sessions

### Starting Work
```
Read: 00_START_HERE.md
Current task: [check file]
Branch: wowrollback-refactor
```

### Resuming Work
```
Check SESSION_CONTEXT.md for quick context
Review progress in current phase document
Continue where left off
```

### Lost Context
```
Read 00_START_HERE.md
Check completed/ folder for what's done
Follow priority order (01 → 02 → 03)
```

---

## 🔄 Document Updates

When adding new planning docs:
1. Use priority prefix (`01_`, `02_`, `03_`)
2. Clear descriptive name
3. Add to this README
4. Update `00_START_HERE.md` if priorities change

When completing work:
1. Move document to `completed/`
2. Update this README
3. Update `00_START_HERE.md`

---

**Work top-to-bottom by file number. Don't skip ahead!** 🎯
