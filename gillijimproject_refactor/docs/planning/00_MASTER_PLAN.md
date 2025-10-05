# 🎯 WoWRollback Master Plan - Unified Architecture

**Status**: Draft for Review  
**Created**: 2025-10-04  
**Purpose**: Reconcile all planning documents into a single, coherent execution plan

---

## Executive Summary

This document unifies:
- **06_Viewer_Implementation_Plan.md** (viewer refactor)
- **04_Overlay_Plugin_Architecture.md** (plugin system)
- **03_Rollback_TimeTravel_Feature.md** (Phase 0 feature)
- **04_Tool_Consolidation_Details.md** (merge 3 tools)
- **05_Unified_Tool_Architecture.md** (audit-first methodology)

**Core Philosophy**:
1. **Audit-First**: No implementation without approved audit
2. **Plugin-Oriented**: Everything is a plugin (overlays, formats, exporters)
3. **Incremental Migration**: One piece at a time, fully tested
4. **Performance by Design**: Multi-threaded from day one

---

## 🔍 Key Conflicts Identified

### Conflict 1: Priority Order
- **00_START_HERE.md**: Fix critical bugs first (terrain overlays, AreaTable)
- **User Request**: Start with viewer refactor (06_Viewer_Implementation_Plan.md)
- **Resolution**: Viewer refactor IS the foundation for fixing bugs properly

### Conflict 2: Overlay Architecture
- **06_Viewer_Implementation_Plan**: Simple migration (move files, create runtime)
- **04_Overlay_Plugin_Architecture**: Complex manifest system with plugin registry
- **Resolution**: Implement both - simple runtime first, then enhance with manifest

### Conflict 3: Rollback Feature Integration
- **03_Rollback_TimeTravel_Feature**: Needs timeline slider in viewer
- **06_Viewer_Implementation_Plan**: Doesn't mention rollback feature
- **Resolution**: Add rollback as a plugin in Phase 5

### Conflict 4: Tool Consolidation
- **04_Tool_Consolidation**: Merge AlphaWDTAnalysisTool + DBCTool.V2
- **Viewer Plan**: No mention of tool consolidation
- **Resolution**: Tool consolidation is Phase 6 (after viewer is stable)

### Conflict 5: Methodology
- **05_Unified_Tool_Architecture**: Audit-first, no code without approval
- **06_Viewer_Implementation_Plan**: Start implementing Phase 1 immediately
- **Resolution**: Phase 0 audit-first, user approval required before Phase 1

---

## 📋 Implementation Phases

### Phase 0: Audit & Foundation (Week 1) ✅ COMPLETE

**Goal**: Understand current system before making changes

**Tasks**:
1. ✅ Audit `ViewerAssets/` structure and dependencies
2. ✅ Document overlay system architecture
3. ✅ Identify migration paths for each overlay type
4. ✅ Create additional audits (gillijimproject, AlphaWdtAnalyzer)

**Deliverables**:
- [x] `docs/planning/00_MASTER_PLAN.md` (this document)
- [x] `docs/planning/07_Incremental_Migration_Strategy.md`
- [x] `docs/audits/Viewer_Audit.md`
- [x] `docs/audits/GillijimProject_Audit.md`
- [x] `docs/audits/AlphaWdtAnalyzer_Audit.md`

**Acceptance Criteria**:
- ✅ All audits complete
- ✅ Clear understanding of migration paths
- ⏳ Awaiting user approval to proceed to Phase 1

---

### Phase 1: Viewer Foundation (Week 2) ⏳ IMPLEMENTATION COMPLETE, AWAITING VALIDATION

**Goal**: Create plugin-ready viewer project structure

**Tasks**:
1. ✅ Create `WoWRollback.Viewer` project
2. ✅ Configure `.csproj` to copy all assets
3. ✅ Copy `ViewerAssets/` → `WoWRollback.Viewer/assets/`
4. ✅ Add feature flag to rebuild script
5. ⏳ Build and verify
6. ⏳ Test both paths (old and new)
7. ⏳ SHA256 validation

**Deliverables**:
- [x] `WoWRollback.Viewer/WoWRollback.Viewer.csproj`
- [x] `WoWRollback.Viewer/ViewerManifest.cs`
- [x] All viewer files in `assets/` subdirectory
- [x] Feature flag in `rebuild-and-regenerate.ps1`
- [x] `docs/planning/08_Phase1_Validation.md`
- [ ] Validation complete (pending user testing)

**Acceptance Criteria**:
- ⏳ Solution builds without errors
- ⏳ Old path works (default)
- ⏳ New path works (with `-UseNewViewerAssets`)
- ⏳ SHA256 validation: old == new outputs
- ⏳ Zero visual differences

---

### Phase 2: Backend Manifest System (Week 2-3)

**Goal**: Implement overlay manifest generation (04_Overlay_Plugin_Architecture)

**Tasks**:
1. Create `OverlayManifestBuilder.cs` in `WoWRollback.Core`
2. Generate `overlay_manifest.json` during viewer report generation
3. Add CLI flag `--viewer-next-objects` for A/B testing
4. Document manifest schema

**Manifest Schema**:
```json
{
  "version": "0.5.3.3368",
  "map": "Azeroth",
  "overlays": [
    {
      "id": "terrain.properties",
      "plugin": "terrain",
      "title": "Terrain Properties",
      "tiles": "sparse",
      "resources": {
        "tilePattern": "overlays/{version}/{map}/terrain_complete/tile_{col}_{row}.json"
      }
    }
  ]
}
```

**Note**: Tile pattern is `tile_{col}_{row}` to match actual WoW tile naming (column_row format, no r/c prefixes).

**Deliverables**:
- [ ] `WoWRollback.Core/Services/Viewer/OverlayManifestBuilder.cs`
- [ ] Manifest generation in `ViewerReportWriter.cs`
- [ ] Sample `overlay_manifest.json` for test map

**Acceptance Criteria**:
- Manifest correctly lists available overlays
- Sparse tiles marked correctly
- CLI flag works for A/B testing

---

### Phase 3: Frontend Runtime Core (Week 3-4)

**Goal**: Implement plugin loader and lifecycle management

**Tasks** (from 06_Viewer_Implementation_Plan Phase 3):
1. Create `WoWRollback.Viewer/js/runtime/` directory
2. Create `WoWRollback.Viewer/js/plugins/` directory
3. Implement `runtime.js` (manifest loader, plugin manager)
4. Implement `plugin-interface.js` (lifecycle contract)
5. Implement `resource-loader.js` (cached fetching)

**Plugin Interface**:
```javascript
class OverlayPlugin {
  async initialize(context) { }
  async loadTile(tileCoord) { }
  render(tileCoord, payload) { }
  teardown() { }
}
```

**Deliverables**:
- [ ] `js/runtime/runtime.js`
- [ ] `js/runtime/plugin-interface.js`
- [ ] `js/runtime/resource-loader.js`
- [ ] Documentation: `docs/Viewer_Plugin_API.md`

**Acceptance Criteria**:
- Runtime loads manifest successfully
- Plugins can be registered and initialized
- Resource loader caches correctly
- No breaking changes to existing viewer

---

### Phase 4: Plugin Migration (Week 4-6)

**Goal**: Migrate existing overlays to plugin system (06_Viewer_Implementation_Plan Phase 4-5)

**4A: Simple Overlays** (Week 4-5)
1. ✅ Terrain Properties Plugin (`js/plugins/terrain.js`)
2. ✅ Area ID Plugin (`js/plugins/areaId.js`)
3. ✅ Holes Plugin (`js/plugins/holes.js`)
4. ✅ Liquids Plugin (`js/plugins/liquids.js`)

**4B: Complex Overlays** (Week 5-6)
1. ✅ Shadow Map Plugin (`js/plugins/shadow.js`)
2. ⏳ Objects Plugin (existing, compatibility mode)
3. 🔄 Objects-Next Plugin (`js/plugins/objects-next.js`) - A/B test

**Migration Process per Plugin**:
1. Create `js/plugins/{name}.js` implementing plugin interface
2. Move rendering logic from `js/overlays/{name}Layer.js`
3. Update `main.js` to use runtime for this plugin
4. Test plugin loads and renders correctly
5. Delete old `js/overlays/{name}Layer.js`

**Deliverables**:
- [ ] 6 migrated plugins (terrain, areaId, holes, liquids, shadow, objects)
- [ ] 1 new plugin (objects-next)
- [ ] Old `js/overlays/` directory can be deleted (except objects, keep for A/B)

**Acceptance Criteria**:
- All overlays work through plugin system
- A/B testing works for objects-next
- Legacy overlay code removed (except objects)

---

### Phase 5: Rollback Feature Plugin (Week 7-8)

**Goal**: Implement time-travel visualization (03_Rollback_TimeTravel_Feature)

**Tasks**:
1. Create `WoWRollback.Core/Analysis/UniqueIdAnalyzer.cs`
2. Create `WoWRollback.Core/Analysis/LayerDetector.cs`
3. Generate UniqueID analysis CSVs during comparison
4. Create `js/plugins/timeline.js` (timeline slider plugin)
5. Implement global and per-tile filtering modes

**Timeline Plugin Features**:
- Slider showing ID ranges
- Layer markers on slider
- Object count display
- Filter modes (global vs per-tile)
- Export filtered ADTs

**Deliverables**:
- [ ] `UniqueIdAnalyzer.cs` and `LayerDetector.cs`
- [ ] `js/plugins/timeline.js`
- [ ] UniqueID analysis CSVs for test maps
- [ ] Timeline UI in viewer

**Acceptance Criteria**:
- UniqueID analysis runs on all maps
- Timeline slider filters objects correctly
- Layer detection identifies work sessions
- Export filtered ADTs works

---

### Phase 6: Tool Consolidation (Week 9-11)

**Goal**: Merge AlphaWDTAnalysisTool + DBCTool.V2 into WoWRollback (04_Tool_Consolidation)

**6A: Audit Phase** (Week 9)
1. Audit `AlphaWDTAnalysisTool` → `docs/audits/AlphaWDT_Audit.md`
2. Audit `DBCTool.V2` → `docs/audits/DBCTool_Audit.md`
3. Review and approve audits

**6B: Core Readers** (Week 10)
1. Migrate `WdtAlphaReader.cs` → `WoWRollback.Core/Formats/Alpha/`
2. Migrate `AdtAlphaReader.cs` → `WoWRollback.Core/Formats/Alpha/`
3. Migrate `AreaTableReader.cs` → `WoWRollback.Core/Formats/Dbc/`
4. Create `AdtLkWriter.cs` → `WoWRollback.Core/Formats/Lk/`

**6C: Multi-Threaded Processing** (Week 11)
1. Implement `MapConverter.cs` (Alpha → LK, multi-threaded)
2. Implement `TerrainExtractor.cs` (parallel MCNK extraction)
3. Implement `ShadowExtractor.cs` (parallel MCSH extraction)
4. Add `ConvertMapCommand.cs` to CLI

**Expected Performance**:
- **Before**: 45-60 minutes (single-threaded)
- **After**: 7-8 minutes (8 threads) → **6.5x faster**

**Deliverables**:
- [ ] Two audit documents (approved)
- [ ] Format readers in `WoWRollback.Core`
- [ ] Multi-threaded processors
- [ ] New CLI commands
- [ ] Archive old tools in `_archived/`

**Acceptance Criteria**:
- Single command replaces 3 tools
- 6.5x performance improvement
- No functionality lost
- Comprehensive tests

---

### Phase 7: Deprecation & Polish (Week 12)

**Goal**: Clean up legacy code and polish UX

**Tasks**:
1. Delete `js/overlays/` directory completely
2. Remove `overlayManager.js`
3. Remove `--viewer-next-objects` flag (make objects-next default)
4. Update all documentation
5. Add comprehensive tests
6. Polish Spectre.Console UI

**Deliverables**:
- [ ] Clean codebase (no legacy overlay code)
- [ ] Updated documentation
- [ ] 90%+ test coverage
- [ ] Beautiful CLI experience

---

## 🏗️ Final Architecture

### Solution Structure
```
WoWRollback/
├── WoWRollback.Core/
│   ├── Formats/
│   │   ├── Alpha/           ← Alpha WDT/ADT reading
│   │   ├── Lk/              ← LK ADT reading/writing
│   │   └── Dbc/             ← DBC/AreaTable reading
│   ├── Processing/          ← Multi-threaded pipelines
│   │   ├── MapConverter.cs
│   │   ├── TerrainExtractor.cs
│   │   └── ShadowExtractor.cs
│   ├── Analysis/            ← Rollback feature
│   │   ├── UniqueIdAnalyzer.cs
│   │   └── LayerDetector.cs
│   └── Services/
│       ├── Viewer/
│       │   ├── ViewerReportWriter.cs
│       │   └── OverlayManifestBuilder.cs  ← NEW
│       └── Comparison/
├── WoWRollback.Viewer/      ← NEW PROJECT
│   ├── index.html
│   ├── tile.html
│   ├── css/
│   ├── js/
│   │   ├── runtime/         ← Plugin system
│   │   │   ├── runtime.js
│   │   │   ├── plugin-interface.js
│   │   │   └── resource-loader.js
│   │   └── plugins/         ← All overlays
│   │       ├── terrain.js
│   │       ├── shadow.js
│   │       ├── objects-next.js
│   │       └── timeline.js  ← Rollback feature
│   └── WoWRollback.Viewer.csproj
├── WoWRollback.Cli/
│   └── Commands/
│       ├── ConvertMapCommand.cs        ← NEW
│       ├── ExtractTerrainCommand.cs    ← NEW
│       └── CompareVersionsCommand.cs   ← Enhanced
└── WoWRollback.Tests/       ← Comprehensive tests
```

### Viewer Plugin System
```
Viewer Runtime
    ↓
overlay_manifest.json
    ↓
Plugin Loader
    ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Terrain    │   Shadow    │ Objects-Next│  Timeline   │
│  Plugin     │   Plugin    │   Plugin    │   Plugin    │
└─────────────┴─────────────┴─────────────┴─────────────┘
    ↓             ↓             ↓             ↓
resource-loader (cached, sparse-aware)
    ↓
Overlay Data (JSON/PNG)
```

---

## 📊 Timeline & Milestones

| Phase | Duration | Cumulative | Key Deliverable |
|-------|----------|------------|-----------------|
| **Phase 0** | 1 week | Week 1 | Viewer Audit (approved) |
| **Phase 1** | 1 week | Week 2 | `WoWRollback.Viewer` project |
| **Phase 2** | 1-2 weeks | Week 3 | Overlay manifest system |
| **Phase 3** | 1-2 weeks | Week 4 | Plugin runtime core |
| **Phase 4** | 2 weeks | Week 6 | All overlays migrated |
| **Phase 5** | 2 weeks | Week 8 | Rollback feature working |
| **Phase 6** | 3 weeks | Week 11 | Tools consolidated |
| **Phase 7** | 1 week | Week 12 | Clean, polished codebase |
| **Total** | **~12 weeks** | | **Complete refactor** |

---

## 🎯 Success Criteria

### Technical
- ✅ Plugin system supports all current overlays
- ✅ Manifest-driven overlay loading
- ✅ Rollback feature fully functional
- ✅ 6.5x performance improvement (tool consolidation)
- ✅ 90%+ test coverage
- ✅ Zero breaking changes during migration

### User Experience
- ✅ Single command replaces 3 tools + PowerShell script
- ✅ Beautiful progress bars (Spectre.Console)
- ✅ A/B testing for new features
- ✅ Timeline slider for time-travel visualization
- ✅ Fast, responsive viewer

### Code Quality
- ✅ Clean architecture (DDD, DI, SOLID)
- ✅ Audit-first methodology followed
- ✅ Comprehensive documentation
- ✅ Easy to extend (new plugins, formats)

---

## 🚨 Risk Mitigation

### Risk 1: Scope Creep
**Mitigation**: Strict phase boundaries, user approval between phases

### Risk 2: Breaking Existing Functionality
**Mitigation**: A/B testing, keep legacy code until new code proven

### Risk 3: Performance Regression
**Mitigation**: Benchmarks before/after, multi-threading from day one

### Risk 4: Integration Complexity
**Mitigation**: Audit-first, incremental migration, comprehensive tests

---

## 📝 Next Steps

### Immediate (Today)
1. **Review this master plan** - Does it reconcile all conflicts?
2. **Approve or request changes**
3. **Begin Phase 0** - Create viewer audit

### This Week
1. Complete viewer audit
2. Get audit approved
3. Start Phase 1 (viewer project setup)

### This Month
1. Complete Phases 1-3 (viewer foundation + runtime)
2. Begin Phase 4 (plugin migration)

### This Quarter
1. Complete all 7 phases
2. Have fully refactored, consolidated WoWRollback
3. Demo rollback feature

---

## 🔗 Document Cross-References

This master plan supersedes and integrates:
- ✅ `00_START_HERE.md` → Priority order integrated
- ✅ `03_Rollback_TimeTravel_Feature.md` → Phase 5
- ✅ `04_Overlay_Plugin_Architecture.md` → Phases 2-4
- ✅ `04_Tool_Consolidation_Details.md` → Phase 6
- ✅ `05_Unified_Tool_Architecture.md` → Methodology
- ✅ `06_Viewer_Implementation_Plan.md` → Phases 1-4, enhanced

**All planning documents are now reconciled into this single master plan.** 🎯

---

**Status**: ⏳ Awaiting user approval to begin Phase 0
