# AlphaWDT Analysis Pipeline - Output Normalization & Stabilization Plan

**Created**: 2025-10-06  
**Status**: Draft  
**Priority**: Critical - Foundation for all future work  

---

## Executive Summary

The AlphaWDT analysis pipeline has become fragmented across multiple output directories (`dbctool_out/`, `cached_maps/`, `rollback_outputs/`) making debugging difficult and the user experience confusing. Additionally, recent Map.dbc integration work has introduced regressions. This plan outlines a comprehensive refactoring to:

1. **Normalize all outputs** to a single, clear directory structure
2. **Stabilize the pipeline** after Map.dbc changes
3. **Improve verification** layers for AreaTable mapping
4. **Establish architectural foundations** for future enhancements

---

## Current State Assessment

### What Works ✅
- **ADT conversion quality** - Alpha ADTs → LK ADTs conversion produces good terrain
- **AreaTable crosswalk generation** (DBCTool.V2) - Produces accurate mappings
- **Map.dbc parsing** - Successfully extracts MapID → Directory mappings
- **Basic terrain CSV extraction** - MCNK data extraction works

### What's Broken ❌
- **Fragmented outputs** - Multiple root directories cause confusion
- **Viewer integration** - Recent changes broke the Viewer
- **MapID resolution** - Incomplete integration in AdtExportPipeline (banned from editing after 3 failures)
- **AreaID patching** - Not all maps getting AreaIDs patched (e.g., Shadowfang: "0 AreaIDs patched")
- **Cache invalidation** - Unclear when to rebuild vs reuse cached ADTs
- **Log organization** - Session logs scattered, hard to trace issues
- **WMO-only map handling** - Script tries to process maps with no ADT tiles (e.g., MonasteryInstances)

### Technical Debt
- **rebuild-and-regenerate.ps1** - Grown organically, needs restructuring
- **AdtExportPipeline.cs** - Mixing concerns (export + area patching + fixups)
- **DbcPatchMapping** - Complex loading logic with multiple search paths
- **Error handling** - Partial failures (e.g., WMO-only maps) crash entire pipeline

---

## Proposed Output Structure

### New Unified Structure

```
parp_outputs/                          # Root for ALL pipeline outputs
├─ sessions/                           # Time-stamped session runs
│   └─ 20251006_001234/
│       ├─ logs/
│       │   ├─ 00_init.log
│       │   ├─ 01_dbctool.log
│       │   ├─ 02_adt_export.log
│       │   ├─ 03_analysis.log
│       │   └─ 04_viewer.log
│       ├─ dbctool/                    # DBCTool.V2 outputs for this run
│       │   └─ 0.5.3/
│       │       ├─ maps.json
│       │       ├─ compare/v2/*.csv
│       │       └─ alpha_core/*.json   # Future: alpha-core exports
│       ├─ adt_cache/                  # Converted LK ADTs (reusable)
│       │   └─ 0.5.3.3368/
│       │       └─ World/Maps/
│       │           ├─ Azeroth/*.adt
│       │           └─ Kalimdor/*.adt
│       ├─ analysis/                   # CSV extractions
│       │   └─ 0.5.3.3368/
│       │       ├─ Azeroth/
│       │       │   ├─ csv/
│       │       │   │   ├─ Azeroth_mcnk_terrain.csv
│       │       │   │   └─ Azeroth_mcnk_shadows.csv
│       │       │   └─ validation/
│       │       │       └─ area_id_coverage.json
│       │       └─ Kalimdor/...
│       └─ viewer/                     # Viewer deliverables
│           ├─ index.html
│           ├─ tiles/
│           ├─ overlays/
│           └─ data/
│               ├─ maps.json           # Symlink to dbctool/0.5.3/maps.json
│               └─ area_crosswalk.json
├─ stable/                             # Latest verified good outputs
│   ├─ 0.5.3.3368/                     # Per-version stable cache
│   │   ├─ adt_cache/
│   │   ├─ analysis/
│   │   └─ metadata.json               # Validation checksums
│   └─ viewer/                         # Latest stable viewer build
└─ workspace/                          # User scratch area
    └─ custom_mappings/                # User-supplied override CSVs
```

### Migration Path

**Phase 1: Parallel Outputs** (transition period)
- Keep old structure working
- Add new `parp_outputs/` alongside
- Scripts write to both

**Phase 2: Default to New** 
- New structure is default
- Old structure opt-in via flag

**Phase 3: Remove Old**
- Delete old output logic after 1-2 releases

---

## Pipeline Architectural Refactor

### Current Flow (Fragile)
```
rebuild-and-regenerate.ps1
  ├─ DBCTool.V2 (writes to dbctool_out/)
  ├─ AlphaWdtAnalyzer (writes to cached_maps/)
  └─ WoWRollback.Cli (writes to rollback_outputs/)
```

**Problems:**
- Each tool manages its own outputs
- No central orchestration
- Hard to trace data flow
- Failures leave partial state

### Proposed Flow (Robust)

```
PipelineOrchestrator (new C# CLI or enhanced PS script)
  ├─ Stage 1: Validation
  │   ├─ Check user paths exist
  │   ├─ Verify Map.dbc present
  │   ├─ Validate map names against maps.json
  │   └─ Detect WMO-only maps → skip gracefully
  ├─ Stage 2: DBC Processing (DBCTool.V2)
  │   ├─ Parse Map.dbc → maps.json
  │   ├─ Parse AreaTable.dbc → crosswalks
  │   └─ Write to session/dbctool/
  ├─ Stage 3: ADT Conversion (AlphaWdtAnalyzer)
  │   ├─ Load MapIdResolver from maps.json
  │   ├─ Convert Alpha ADTs → LK ADTs
  │   ├─ Patch AreaIDs using crosswalks
  │   ├─ Write to session/adt_cache/
  │   └─ Validation: Check AreaID coverage per map
  ├─ Stage 4: Analysis Extraction
  │   ├─ Extract MCNK terrain → CSV
  │   ├─ Extract shadow maps → CSV
  │   ├─ Write to session/analysis/
  │   └─ Validation: Compare to expected tile counts
  ├─ Stage 5: Viewer Generation
  │   ├─ Generate tiles from ADTs
  │   ├─ Create overlays from CSVs
  │   ├─ Copy/link metadata files
  │   └─ Write to session/viewer/
  └─ Stage 6: Promotion
      ├─ If validation passes → copy to stable/
      └─ Generate session report
```

### Key Improvements

1. **Fail-fast validation** - Catch errors before expensive work
2. **Atomic stages** - Each stage completes or rolls back
3. **Progress tracking** - Clear indication of current stage
4. **Validation gates** - Don't proceed if previous stage failed
5. **Session isolation** - Each run is self-contained
6. **Stable promotion** - Only validated outputs become "stable"

---

## Critical Bug Fixes Required

### 1. Complete MapIdResolver Integration

**File**: `AlphaWdtAnalyzer.Core/Export/AdtExportPipeline.cs`  
**Status**: Partially implemented, editing banned after 3 failures  
**Issue**: Function signature updated but calls not updated  

**Fix Required**:
```csharp
// Line 238 needs manual update
// OLD: int currentMapId = ResolveMapIdFromDbc(wdt.MapName, opts.DbctoolLkDir, opts.Verbose);
// NEW: int currentMapId = ResolveMapIdFromDbc(wdt.MapName, mapIdResolver, opts.DbctoolLkDir, opts.Verbose);
```

**Validation**: Test with Shadowfang (map ID 33) to ensure AreaIDs are patched correctly

### 2. WMO-Only Map Handling

**Files**: 
- `rebuild-and-regenerate.ps1` (Ensure-CachedMap function)
- `AdtExportPipeline.cs`

**Issue**: Script crashes when processing maps like MonasteryInstances that have no ADT tiles

**Fix Required**:
```powershell
# In Ensure-CachedMap, before calling AlphaWdtAnalyzer
if ($expectedTiles -eq 0) {
    Write-Host "[cache] Skipping $Map - WMO-only (no ADT tiles)" -ForegroundColor Yellow
    return $false
}
```

### 3. AreaID Patching Verification

**Issue**: Some maps report "0 AreaIDs patched" even when crosswalks exist

**Root Causes**:
- MapID resolution may return -1 (not found)
- Crosswalk CSV selection may be incorrect
- LK ADT AreaID reading may have issues

**Fix Required**:
1. Add verbose logging to MapIdResolver
2. Add validation: if mapId == -1, log error and skip map
3. Add post-export validation: count patched AreaIDs, fail if 0 for maps that should have data

### 4. Cache Invalidation Logic

**Issue**: Unclear when cached ADTs are reused vs regenerated

**Fix Required**:
```powershell
# Clear cache invalidation rules:
# 1. --RefreshCache → delete everything, rebuild
# 2. --RefreshAnalysis → keep ADTs, regenerate CSVs
# 3. Default → check if ADT cache is complete AND up-to-date
#    - Complete = all expected tiles present
#    - Up-to-date = DBCTool outputs haven't changed (checksum metadata)
```

---

## Verification & Validation Framework

### Pre-Flight Checks
```csharp
public class PipelineValidator
{
    public ValidationResult ValidateInputs(PipelineConfig config)
    {
        // Check paths exist
        // Check Map.dbc exists
        // Check AreaTable.dbc exists
        // Validate version strings
    }
    
    public ValidationResult ValidateMapList(List<string> maps, MapIdResolver resolver)
    {
        // Check each map exists in maps.json
        // Warn about WMO-only maps
        // Detect maps without crosswalks
    }
}
```

### Post-Stage Validation
```csharp
public class StageValidator
{
    public ValidationResult ValidateAdtExport(string mapName, string outputDir, int expectedTiles)
    {
        // Count ADT files
        // Verify MHDR/MCIN integrity
        // Check AreaID coverage (not all zeros)
        // Validate file sizes (not truncated)
    }
    
    public ValidationResult ValidateAnalysisExport(string mapName, string csvDir)
    {
        // Check CSV row counts match expected chunks
        // Validate AreaID distribution (not all "Unknown")
        // Check for coordinate outliers
    }
}
```

### Session Reporting
```json
// session/validation_report.json
{
  "session_id": "20251006_001234",
  "start_time": "2025-10-06T00:12:34Z",
  "end_time": "2025-10-06T00:45:12Z",
  "status": "partial_success",
  "stages": {
    "dbctool": { "status": "success", "duration_sec": 45 },
    "adt_export": { "status": "partial", "warnings": 2, "errors": 1 },
    "analysis": { "status": "success" },
    "viewer": { "status": "failed", "error": "Missing AreaID data for Shadowfang" }
  },
  "maps_processed": [
    {
      "name": "Azeroth",
      "map_id": 0,
      "tiles_expected": 1024,
      "tiles_generated": 1024,
      "area_ids_patched": 6400,
      "status": "success"
    },
    {
      "name": "Shadowfang",
      "map_id": 33,
      "tiles_expected": 25,
      "tiles_generated": 25,
      "area_ids_patched": 0,
      "status": "warning",
      "message": "No AreaIDs were patched - check crosswalk CSV"
    },
    {
      "name": "MonasteryInstances",
      "map_id": -1,
      "status": "skipped",
      "message": "WMO-only map (no ADT tiles)"
    }
  ]
}
```

---

## Testing Strategy

### Regression Test Suite

**Test 1: End-to-End Happy Path**
```powershell
# Input: Clean state + valid 0.5.3 client
# Expected: All stages succeed, stable/ populated
.\rebuild-and-regenerate.ps1 `
  -AlphaRoot test_data/0.5.3.3368 `
  -Maps @("DeadminesInstance") `
  -Versions @("0.5.3.3368")

# Verify:
# - maps.json created with DeadminesInstance (ID=36)
# - Crosswalk CSV exists for map36
# - All ADTs generated
# - AreaIDs patched > 0
# - Viewer index.html exists
```

**Test 2: WMO-Only Map Graceful Skip**
```powershell
# Input: Map with no ADT tiles
.\rebuild-and-regenerate.ps1 `
  -Maps @("MonasteryInstances") `
  -Versions @("0.5.3.3368")

# Expected: Gracefully skipped with clear message
# Verify: No crash, validation_report shows "skipped"
```

**Test 3: Cache Reuse**
```powershell
# Run 1: Full generation
.\rebuild-and-regenerate.ps1 -Maps @("Azeroth")

# Run 2: Same map, no changes
.\rebuild-and-regenerate.ps1 -Maps @("Azeroth")

# Expected: ADTs reused from stable/ cache
# Verify: Run 2 completes in <10% time of Run 1
```

**Test 4: AreaID Patching Verification**
```powershell
# Input: Map with known crosswalk (Shadowfang = map 33)
.\rebuild-and-regenerate.ps1 -Maps @("Shadowfang")

# Verify:
# - MapIdResolver reports: "Resolved 'Shadowfang' -> 33"
# - Crosswalk CSV loaded: Area_patch_crosswalk_map33_0.5.3_to_335.csv
# - validation_report.json: area_ids_patched > 0
```

### Manual Verification Checklist

- [ ] Output structure matches new design
- [ ] All logs consolidated in session/logs/
- [ ] Viewer loads and displays tiles correctly
- [ ] AreaID overlays show zone names (not "Unknown")
- [ ] Session report JSON is valid and complete
- [ ] Stable cache promotion works
- [ ] Error messages are clear and actionable

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal**: Stabilize core without breaking existing workflows

- [ ] Create `parp_outputs/` structure
- [ ] Add parallel output writing (old + new)
- [ ] Fix MapIdResolver integration (manual line 238 fix)
- [ ] Add WMO-only map detection and graceful skip
- [ ] Create PipelineValidator class

**Deliverable**: Script runs without crashes, outputs to both locations

### Phase 2: Validation Framework (Week 2)
**Goal**: Add verification layers

- [ ] Implement StageValidator for each pipeline stage
- [ ] Add AreaID coverage validation
- [ ] Generate validation_report.json
- [ ] Add pre-flight checks (ValidateInputs, ValidateMapList)

**Deliverable**: Clear feedback on what succeeded/failed

### Phase 3: Output Migration (Week 3)
**Goal**: Make new structure the default

- [ ] Default to `parp_outputs/`
- [ ] Add `--legacy-output` flag for old structure
- [ ] Update all documentation
- [ ] Create migration guide

**Deliverable**: New users start with clean structure

### Phase 4: Refinement (Week 4)
**Goal**: Polish and optimize

- [ ] Implement stable/ promotion logic
- [ ] Add cache invalidation checksums
- [ ] Optimize session cleanup (delete old sessions > 7 days)
- [ ] Performance profiling and bottleneck removal

**Deliverable**: Production-ready, fast, reliable pipeline

---

## Success Criteria

### Must Have
- ✅ Single output root (`parp_outputs/`)
- ✅ No crashes on WMO-only maps
- ✅ MapIdResolver integrated and working
- ✅ AreaIDs patched correctly for all maps with crosswalks
- ✅ Validation report generated every run
- ✅ Clear error messages when something fails

### Should Have
- ✅ Session logs consolidated and traceable
- ✅ Stable cache promotion working
- ✅ Pre-flight validation catches issues early
- ✅ Regression test suite passes

### Nice to Have
- ✅ Auto-cleanup of old sessions
- ✅ Progress bar / time estimates
- ✅ Parallel ADT conversion (if safe)
- ✅ Web UI for viewing validation reports

---

## Migration Guide for Users

### Existing Users

**Before Refactor:**
```
gillijimproject_refactor/
├─ dbctool_out/              # DBCTool outputs
├─ cached_maps/              # Converted ADTs
└─ rollback_outputs/         # Viewer files
```

**After Refactor:**
```
gillijimproject_refactor/
├─ parp_outputs/             # Everything in one place!
│   ├─ sessions/20251006.../
│   └─ stable/
└─ (old folders deleted after migration)
```

**Migration Script:**
```powershell
# Provided by us, safe to run
.\scripts\migrate_outputs.ps1

# This:
# 1. Backs up old outputs to parp_outputs/migration_backup/
# 2. Converts stable data to new structure
# 3. Deletes old folders (after confirmation)
```

### Fresh Users
- Clone repo
- Run pipeline
- Everything goes to `parp_outputs/` by default
- No confusion!

---

## Risk Assessment

### High Risk
- **Breaking existing workflows** - Mitigation: Parallel outputs during transition
- **Data loss during migration** - Mitigation: Backup step in migration script

### Medium Risk
- **Performance regression** - Mitigation: Benchmark before/after
- **New bugs in validation** - Mitigation: Comprehensive test suite

### Low Risk
- **User adoption** - Mitigation: Clear migration guide, benefits are obvious

---

## Open Questions

1. **Session retention policy**: How long to keep old sessions? (Proposal: 7 days)
2. **Parallel processing**: Can we safely parallelize ADT conversion? (Needs thread-safety audit)
3. **Viewer auto-reload**: Should viewer auto-refresh when new data available? (Nice-to-have)
4. **Cloud storage**: Support for S3/Azure Blob for outputs? (Future consideration)

---

## Appendix A: File System Layout Reference

### Session Structure Detail
```
parp_outputs/sessions/20251006_001234/
├─ config.json                         # Snapshot of run configuration
├─ validation_report.json              # Overall run status
├─ logs/
│   ├─ 00_init.log                     # Startup and validation
│   ├─ 01_dbctool.log                  # DBCTool.V2 full output
│   ├─ 02_adt_export_Azeroth.log       # Per-map ADT conversion
│   ├─ 02_adt_export_Kalimdor.log
│   ├─ 03_analysis.log                 # CSV extraction
│   └─ 04_viewer.log                   # Viewer generation
├─ dbctool/
│   └─ 0.5.3/
│       ├─ maps.json
│       ├─ areas_hierarchy.json
│       └─ compare/v2/*.csv
├─ adt_cache/
│   └─ 0.5.3.3368/World/Maps/
│       ├─ Azeroth/*.adt
│       ├─ Kalimdor/*.adt
│       └─ DeadminesInstance/*.adt
├─ analysis/
│   └─ 0.5.3.3368/
│       ├─ Azeroth/
│       │   ├─ csv/Azeroth_mcnk_terrain.csv
│       │   └─ validation/area_coverage.json
│       └─ Kalimdor/...
└─ viewer/
    ├─ index.html
    ├─ tiles/
    │   ├─ Azeroth_0_0.png
    │   └─ ...
    ├─ overlays/
    │   └─ areas/Azeroth_areas.json
    └─ data/
        └─ maps.json -> ../../dbctool/0.5.3/maps.json
```

### Stable Structure Detail
```
parp_outputs/stable/
├─ 0.5.3.3368/
│   ├─ metadata.json                   # Checksums, validation status
│   ├─ adt_cache/                      # Symlink to latest good session
│   └─ analysis/                       # Symlink to latest good session
├─ 0.5.5.3494/
│   └─ ...
└─ viewer/
    └─ latest -> ../sessions/20251006_001234/viewer/
```

---

## Appendix B: Validation Report Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["session_id", "start_time", "status", "stages", "maps_processed"],
  "properties": {
    "session_id": { "type": "string" },
    "start_time": { "type": "string", "format": "date-time" },
    "end_time": { "type": "string", "format": "date-time" },
    "status": { "enum": ["success", "partial_success", "failed"] },
    "stages": {
      "type": "object",
      "properties": {
        "dbctool": { "$ref": "#/definitions/stage_status" },
        "adt_export": { "$ref": "#/definitions/stage_status" },
        "analysis": { "$ref": "#/definitions/stage_status" },
        "viewer": { "$ref": "#/definitions/stage_status" }
      }
    },
    "maps_processed": {
      "type": "array",
      "items": { "$ref": "#/definitions/map_result" }
    }
  },
  "definitions": {
    "stage_status": {
      "type": "object",
      "required": ["status"],
      "properties": {
        "status": { "enum": ["success", "partial", "failed", "skipped"] },
        "duration_sec": { "type": "number" },
        "warnings": { "type": "integer" },
        "errors": { "type": "integer" },
        "error": { "type": "string" }
      }
    },
    "map_result": {
      "type": "object",
      "required": ["name", "map_id", "status"],
      "properties": {
        "name": { "type": "string" },
        "map_id": { "type": "integer" },
        "tiles_expected": { "type": "integer" },
        "tiles_generated": { "type": "integer" },
        "area_ids_patched": { "type": "integer" },
        "status": { "enum": ["success", "warning", "failed", "skipped"] },
        "message": { "type": "string" }
      }
    }
  }
}
```

---

**Next Steps**: 
1. Review and approve this plan
2. Create GitHub issues/milestones for each phase
3. Start fresh chat session focused on Phase 1 implementation
4. Track progress in this document

**Document Version**: 1.0  
**Last Updated**: 2025-10-06
