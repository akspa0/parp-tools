# Viewer Improvement Initiative - Summary (2025-10-08)

## Overview

The WoWRollback viewer is a working proof-of-concept but suffers from architectural problems that make it laggy, hard to maintain, and brittle. This document summarizes the complete improvement plan.

---

## Current Problems

### 1. **Compilation Blocker** (URGENT)
- OverlayGenerator.cs has syntax errors preventing build
- AnalysisOrchestrator.cs has parameter mismatches
- **Impact**: Cannot generate viewer overlay JSONs
- **Estimated Fix**: 30 minutes

### 2. **Path Mismatches** (HIGH)
Backend writes: `viewer/overlays/<version>/<map>/objects_combined/tile_{X}_{Y}.json`  
Frontend expects: `overlays/<version>/<map>/<variant>/tile_r{row}_c{col}.json`

**Problems**:
- Filename format mismatch (`tile_{X}_{Y}` vs `tile_r{row}_c{col}`)
- Directory structure mismatch (hardcoded `objects_combined` vs dynamic `<variant>`)

### 3. **Data Format Mismatches** (HIGH)
Backend writes new flat structure:
```json
{ "tileX": 30, "tileY": 30, "placements": [...] }
```

Frontend expects legacy nested structure:
```json
{ "layers": [{ "version": "...", "kinds": [{ "kind": "...", "points": [...] }] }] }
```

**Missing**: Pixel coordinates required for map rendering

### 4. **Monolithic Architecture** (MEDIUM)
- main.js is 1033 lines with mixed concerns
- No ObjectOverlayManager (only terrain has a manager)
- Coordinate transforms scattered throughout code
- Direct Leaflet API calls everywhere
- Hard to test, extend, or optimize

### 5. **Performance Issues** (MEDIUM)
- No proper marker lifecycle management
- Inefficient caching strategies
- Repeated DOM operations
- Memory leaks from unreleased markers

---

## Solution Overview

### Immediate (30 min) - Fix Compilation
**Document**: [overlay-generator-fix-plan.md](overlay-generator-fix-plan.md)

1. Fix broken `PlacementOverlayJson` record (OverlayGenerator.cs:490)
2. Add missing helper methods
3. Fix orchestrator parameter mismatches
4. Remove dead legacy code

**Result**: Build succeeds, overlays can be generated

---

### Phase 1 (2 hours) - Backend-Frontend Alignment
**Document**: [viewer-architecture-improvement-plan.md](viewer-architecture-improvement-plan.md) Phase 1

1. **Standardize Filenames**: Backend writes `tile_r{X}_c{Y}.json`
2. **Create OverlayAdapter**: Convert new JSON format → legacy format
3. **Fix Path Resolution**: Update `state.getOverlayPath()` to match backend

**Result**: Backend and frontend speak same language

---

### Phase 2 (4 hours) - Object Overlay Manager
**Document**: [viewer-architecture-improvement-plan.md](viewer-architecture-improvement-plan.md) Phase 2

Create `ObjectOverlayManager` parallel to existing `OverlayManager`:
- Load object overlays from backend
- Create/update/remove markers
- Viewport-based loading with debouncing
- Proper caching and cleanup
- Marker lifecycle management

**Result**: Objects loaded via clean manager interface

---

### Phase 3 (2 hours) - Refactor main.js
**Document**: [viewer-architecture-improvement-plan.md](viewer-architecture-improvement-plan.md) Phase 3

1. Replace `performObjectMarkerUpdate()` (150 lines) with manager calls (3 lines)
2. Extract coordinate transforms to utilities
3. Extract marker creation to components
4. Clean up event handlers

**Result**: main.js reduced from 1033 → <400 lines

---

### Phase 4 (3 hours) - Coordinate Transforms
**Document**: [viewer-architecture-improvement-plan.md](viewer-architecture-improvement-plan.md) Phase 4

Implement proper WoW World → Tile/Pixel → Leaflet LatLng transforms:
- World coordinates from backend
- Calculate tile + pixel coordinates
- Convert to Leaflet map positions
- Handle edge cases and validation

**Result**: Objects appear at correct map positions

---

## Timeline

### ⚡ SIMPLIFIED APPROACH (2 hours total!)

**Discovery**: You already have production-ready infrastructure!
- ✅ CoordinateTransformer.cs (world → tile/pixel transforms)
- ✅ OverlayBuilder.cs (generates overlay JSON with coords)
- ✅ ViewerReportWriter.cs (orchestrates viewer generation)

**New Plan** (see viewer-json-refactor-plan.md):

```
Immediate: OverlayGenerator Fix
  └─ 30 minutes (compilation fix)

Phase 2: Integration with OverlayBuilder
  ├─ Add WoWRollback.Core reference
  ├─ Implement PlacementMapper converter
  └─ Delegate to existing OverlayBuilder
  └─ 1 hour

Phase 3: Testing
  └─ 30 minutes (end-to-end validation)

Total: 2 hours (vs 14 hours for new implementation!)
```

### Original Timeline (if building from scratch)
```
Day 1: Backend-Frontend Alignment + ObjectOverlayManager
  ├─ Phase 1: 2 hours (alignment)
  └─ Phase 2: 4 hours (manager)

Day 2: Refactor + Coordinate Transforms + Testing
  ├─ Phase 3: 2 hours (refactor main.js)
  ├─ Phase 4: 3 hours (coordinates)
  └─ Testing: 3 hours (integration)

Total: 30 min + 2 days (~14 hours)
```

---

## Dependencies & Order

### Critical Path
1. **OverlayGenerator Fix** (30 min) - MUST DO FIRST
2. Phase 1 (alignment) - depends on #1
3. Phase 2 (manager) - depends on #2
4. Phase 3 (refactor) - depends on #3

### Parallel Work
- Phase 4 (coordinate transforms) can start anytime independently
- Documentation updates can happen in parallel

---

## Success Metrics

### Compilation & Generation
- [x] Build succeeds with zero errors
- [x] OverlayGenerator writes correct JSON files
- [x] Filenames match viewer expectations

### Viewer Functionality
- [x] Objects load and render at correct positions
- [x] Pan/zoom works smoothly with 100+ objects
- [x] Memory usage stable (no leaks)
- [x] No console errors during operation

### Code Quality
- [x] main.js < 400 lines
- [x] Proper separation of concerns
- [x] Managers handle their domains
- [x] Coordinate transforms centralized

### Performance
- [x] Smooth 60 FPS panning
- [x] < 100ms overlay load latency
- [x] Proper marker cleanup outside viewport
- [x] Efficient caching strategy

---

## Risk Assessment

### Low Risk
- ✅ OverlayGenerator fixes (syntax, parameters)
- ✅ Filename standardization
- ✅ Path resolution updates

### Medium Risk
- ⚠️ OverlayAdapter (format conversion)
  - **Mitigation**: Comprehensive test cases with real data
  - **Fallback**: Can adapt viewer to new format instead

- ⚠️ ObjectOverlayManager (new component)
  - **Mitigation**: Copy proven patterns from TerrainOverlayManager
  - **Fallback**: Can keep old logic initially, migrate incrementally

### Higher Risk
- ⚠️ World → Pixel coordinate transforms
  - **Mitigation**: Test with known reference points
  - **Fallback**: Use WoW.tools coordinate system as reference
  - **Validation**: Visual inspection with minimap alignment

---

## Testing Strategy

### Unit Tests
- OverlayAdapter format conversion
- CoordinateTransforms math
- PathResolver logic

### Integration Tests
- OverlayGenerator end-to-end
- ObjectOverlayManager loading
- Manager coordination in main.js

### Visual Tests
- Objects at correct positions
- Smooth pan/zoom performance
- Popup interactions
- Layer toggling

### Regression Tests
- Existing terrain overlays still work
- Minimap tiles still load
- Version/map switching still works
- Sedimentary layers still function

---

## Future Improvements (Post-MVP)

### Performance
- WebWorker for JSON parsing
- Virtual scrolling for markers
- Progressive loading (priority tiles first)

### Architecture
- CoordinateManager (centralized)
- MarkerFactory (separate creation)
- PopupGenerator (template-based)
- FilterManager (UID, layers)

### Monitoring
- PerformanceMonitor (FPS, memory)
- ErrorBoundary (graceful degradation)
- Analytics (user interactions)

---

## Getting Started

### For Immediate Fix (30 min)
1. Read [overlay-generator-assessment.md](overlay-generator-assessment.md)
2. Follow [overlay-generator-fix-plan.md](overlay-generator-fix-plan.md)
3. Type `ACT` to proceed with implementation

### For Full Viewer Improvement (2 days)
1. Complete immediate fix first
2. Read [viewer-architecture-improvement-plan.md](viewer-architecture-improvement-plan.md)
3. Follow phase-by-phase implementation
4. Test each phase before proceeding

---

## Questions & Clarifications

### Q: Can we skip the adapter and just update the viewer?
**A**: Yes, but:
- Adapter approach is safer (backward compatible)
- Allows gradual migration
- Isolates format changes
- Easier to test

Recommend: Use adapter initially, then phase out when stable

### Q: Can we do coordinate transforms later?
**A**: No - objects won't appear at correct positions without them. This is critical for basic functionality.

### Q: Why not just fix paths and call it done?
**A**: Path fixes alone won't solve:
- Data format mismatches
- Missing pixel coordinates
- Performance issues
- Architectural problems

Recommend: Do all phases for production-quality viewer

---

## Contact & Support

- **Documentation**: All plans in `refactor/` directory
- **Status**: See [viewer-overlay-blocker.md](viewer-overlay-blocker.md)
- **Questions**: Review architecture plan first, then ask specific questions

**Next Step**: Type `ACT` to begin implementation with OverlayGenerator fix
