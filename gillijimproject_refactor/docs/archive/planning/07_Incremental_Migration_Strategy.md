# 🦀 Incremental Migration Strategy - "Crab Walk to Success"

**Philosophy**: Never break the viewer. It's our continuous integration test.

---

## Core Principle: Side-by-Side Migration

**Rule**: Old and new code run in parallel until new code is proven identical.

```
┌─────────────────────────────────────────────────────────┐
│  Current Working Pipeline (NEVER REMOVE UNTIL PROVEN)   │
│  AlphaWDTAnalysisTool → LK ADTs → WoWRollback → Viewer  │
└─────────────────────────────────────────────────────────┘
                         ↓
                   (both running)
                         ↓
┌─────────────────────────────────────────────────────────┐
│  New Plugin Pipeline (BUILD ALONGSIDE, TEST, VALIDATE)  │
│  WoWRollback.Plugins → Same LK ADTs → Same Viewer Data  │
└─────────────────────────────────────────────────────────┘
```

**Only when**: SHA256(old output) == SHA256(new output)  
**Then**: Remove old code

---

## Migration Phases (Incremental)

### Phase 0: Audit & Foundation (Week 1) ⏳ CURRENT
**Status**: In Progress  
**Goal**: Understand what we have before changing it

**Tasks**:
1. ✅ Create master plan
2. ✅ Create incremental strategy
3. ⏳ Audit viewer
4. ⏳ Audit gillijimproject-csharp
5. ⏳ Audit AlphaWdtAnalyzer

**Deliverables**:
- [x] `docs/planning/00_MASTER_PLAN.md`
- [x] `docs/planning/07_Incremental_Migration_Strategy.md`
- [ ] `docs/audits/Viewer_Audit.md`
- [ ] `docs/audits/GillijimProject_Audit.md`
- [ ] `docs/audits/AlphaWdtAnalyzer_Audit.md`

---

### Phase 1: Viewer Project Setup (Week 2)
**Goal**: Create plugin-ready viewer WITHOUT breaking existing viewer

**Feature Flag**: `--UseNewViewerAssets` in rebuild-and-regenerate.ps1

**Implementation**:
```powershell
# Line 496 of rebuild-and-regenerate.ps1
if ($UseNewViewerAssets) {
    $assetsSrc = Join-Path $PSScriptRoot 'WoWRollback.Viewer\bin\Release\net9.0'
} else {
    $assetsSrc = Join-Path $PSScriptRoot 'ViewerAssets'  # Still works!
}
```

**Validation**:
```powershell
# Test old (default):
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance

# Test new (opt-in):
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance -UseNewViewerAssets

# Both must work identically
```

---

### Phase 2: Backend Manifest (Week 3)
**Goal**: Add manifest generation WITHOUT changing overlay behavior

**Tasks**:
- Create `OverlayManifestBuilder.cs`
- Generate `overlay_manifest.json` (viewer ignores it)
- No changes to overlay loading

**Validation**: Manifest exists, viewer works as before

---

### Phase 3: Frontend Runtime (Week 4)
**Goal**: Add plugin runtime with feature flag

**URL Flag**: `?use_runtime=1`

**Implementation**:
```javascript
// index.html
const useRuntime = new URLSearchParams(window.location.search).get('use_runtime') === '1';

if (useRuntime) {
    console.log('[NEW] Plugin runtime');
    await initPluginRuntime();
} else {
    console.log('[STABLE] Legacy overlays');
    await initLegacyOverlays();
}
```

**Validation**:
- Default URL: old system works
- `?use_runtime=1`: new system works identically

---

### Phase 4: Plugin Migration (Weeks 5-6)
**Goal**: Migrate overlays ONE AT A TIME

**Order** (simple → complex):
1. Terrain Properties
2. Area IDs
3. Holes
4. Liquids
5. Shadow Maps
6. Objects

**Per-Plugin Process**:
1. Create `js/plugins/{name}.js` (isolated)
2. Add URL flag `?plugin_{name}=1` (opt-in)
3. Test side-by-side (1 week)
4. SHA256 validation (must match)
5. Flip default to new plugin
6. Monitor 1 week
7. Remove old code when stable

---

### Phase 5: Rollback Feature (Weeks 7-8)
**Goal**: Add timeline slider as plugin

**URL Flag**: `?timeline=1`

---

### Phase 6: Tool Consolidation (Weeks 9-15)
**Goal**: Migrate tools to WoWRollback plugins

**Feature Flag**: `--UseNewConverter` in rebuild-and-regenerate.ps1

**Strategy**: Library-first (keep gillijimproject-csharp as dependency)

**Weeks 9**: Audits  
**Weeks 10-11**: Plugin wrappers (side-by-side)  
**Weeks 12-13**: Multi-threading  
**Weeks 14-15**: Deprecation

**Validation**:
```powershell
# Test old:
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance
Get-FileHash rollback_outputs/**/*.json > old.txt

# Test new:
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance -UseNewConverter
Get-FileHash rollback_outputs/**/*.json > new.txt

# Must be identical:
Compare-Object (Get-Content old.txt) (Get-Content new.txt)
```

---

## Feature Flag Management

### PowerShell Flags
```powershell
param(
    [switch]$UseNewViewerAssets,   # Phase 1
    [switch]$UseNewConverter,      # Phase 6
    [switch]$EnableTimeline        # Phase 5
)
```

### Viewer URL Flags
```
?use_runtime=1           # Phase 3: plugin runtime
?plugin_terrain=1        # Phase 4: per-plugin testing
?plugin_shadow=1
?plugin_objects=1
?timeline=1              # Phase 5: rollback feature
?debug=1                 # always available
```

---

## Validation Checklist (Every Phase)

Before promoting new code to default:

1. **Functional Parity**
   - [ ] Old viewer works
   - [ ] New implementation works
   - [ ] Visual output identical

2. **Data Parity**
   - [ ] SHA256(old outputs) == SHA256(new outputs)

3. **Performance**
   - [ ] New ≥ old performance
   - [ ] No memory leaks

4. **Testing Period**
   - [ ] 1 week opt-in testing
   - [ ] 1 week default testing
   - [ ] 2 weeks stable → remove old

5. **Documentation**
   - [ ] Update CHANGELOG.md
   - [ ] Update README.md

---

## Emergency Rollback

**If anything breaks**:

```powershell
# Immediate (<1 minute):
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance

# Or remove URL flags:
http://localhost:8080/index.html

# Old system works, zero data loss
```

---

## Timeline Summary

| Phase | Duration | Feature Flag | Status |
|-------|----------|--------------|--------|
| 0: Audits | 1 week | N/A | ⏳ In Progress |
| 1: Viewer Project | 1 week | `--UseNewViewerAssets` | Pending |
| 2: Manifest | 1 week | Always generate | Pending |
| 3: Runtime | 2 weeks | `?use_runtime=1` | Pending |
| 4: Plugins | 2 weeks | Per-plugin | Pending |
| 5: Rollback | 2 weeks | `?timeline=1` | Pending |
| 6: Tools | 6 weeks | `--UseNewConverter` | Pending |
| **Total** | **15 weeks** | | |

---

**"Crab walk to success" - sideways, incrementally, never falling.** 🦀
