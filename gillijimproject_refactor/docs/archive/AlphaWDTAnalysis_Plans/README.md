# AlphaWDT Analysis Plans - Quick Reference

This directory contains planning documents for major refactoring and enhancement efforts.

## Active Plans

### 001: Output Normalization & Stabilization ⚠️ **CRITICAL - START HERE**
**File**: `001_output_normalization_and_stabilization.md`  
**Status**: Draft - Ready for implementation  
**Priority**: Critical  

**What it fixes**:
- Fragmented outputs (dbctool_out/ vs cached_maps/ vs rollback_outputs/)
- Broken Viewer after Map.dbc integration
- WMO-only map crashes
- Missing AreaID patching on some maps
- Unclear cache invalidation

**Quick summary**:
1. New unified output structure: `parp_outputs/`
2. Session-based runs with validation reports
3. Stable cache promotion
4. Fix MapIdResolver integration (line 238 in AdtExportPipeline.cs)
5. Add validation framework

**Estimated effort**: 4 weeks (phased)

---

## Current Known Issues (As of 2025-10-06)

### Blocking 🔴
1. **AdtExportPipeline.cs line 238** - MapIdResolver parameter not passed (banned from auto-editing after 3 failures)
   - Manual fix required before testing
   
2. **Viewer broken** - Recent changes broke viewer generation
   - Root cause unknown, needs investigation

### High Priority 🟡
3. **WMO-only maps crash pipeline** (e.g., MonasteryInstances)
   - Need graceful skip logic
   
4. **Shadowfang AreaID patching** - Reports "0 AreaIDs patched" despite crosswalk existing
   - MapID resolution or crosswalk loading issue

5. **Output confusion** - Users don't know where to find generated files

### Medium Priority 🟢
6. **Cache invalidation unclear** - When to rebuild vs reuse?
7. **Log organization** - Hard to trace issues across pipeline
8. **No validation framework** - Silent partial failures

---

## What Works (Don't Break!)

✅ **DBCTool.V2 Map.dbc parsing** - Successfully generates maps.json  
✅ **AreaTable crosswalk generation** - Produces accurate mapping CSVs  
✅ **ADT conversion quality** - Alpha → LK ADTs are good  
✅ **MCNK terrain extraction** - CSV generation works  
✅ **DeadminesInstance full pipeline** - This map works end-to-end  

---

## Recommended Workflow for Next Session

### Session Goal: Stabilize Foundation

**Phase 1A: Critical Fixes (Day 1)**
1. Manually fix line 238 in `AdtExportPipeline.cs`
2. Test with DeadminesInstance (known working)
3. Test with Shadowfang (currently failing AreaID patch)
4. Add WMO-only map skip logic

**Phase 1B: Output Structure (Day 2-3)**
5. Create `parp_outputs/` structure
6. Update scripts to write to both old + new (parallel)
7. Add session validation reporting

**Phase 1C: Testing (Day 4-5)**
8. Regression test suite
9. Document migration guide
10. Create Phase 2 plan based on findings

---

## Session Checklist Template

Copy this for each work session:

```markdown
## Session: [Date] - [Goal]

### Pre-Session
- [ ] Read relevant plan document
- [ ] Review current known issues
- [ ] Check what works (don't break list)
- [ ] Set specific, achievable goal

### During Session
- [ ] Make incremental changes
- [ ] Test after each change
- [ ] Document new findings
- [ ] Update known issues list

### Post-Session
- [ ] Run full regression test
- [ ] Update plan document status
- [ ] List what's ready for next session
- [ ] Note any new issues discovered
```

---

## File Naming Convention

- `NNN_descriptive_name.md` - Plan documents (numbered sequentially)
- `NNN_a_subtask.md` - Sub-plans (if needed)
- `README.md` - This file (quick reference)

---

## How to Use These Plans

**Starting a new session:**
1. Read the active plan (001 currently)
2. Copy session checklist
3. Pick ONE concrete task from the plan
4. Focus on completion, not perfection

**During implementation:**
- Update plan status sections
- Add "Actual Implementation Notes" if you deviate
- Document new issues discovered

**After completion:**
- Mark plan as "Implemented"
- Create new plan for next major feature
- Archive old plans to `archive/` folder

---

## Quick Links

- Main Refactor Plan: [001_output_normalization_and_stabilization.md](./001_output_normalization_and_stabilization.md)
- Project Root: `../../`
- Current Outputs: 
  - DBCTool: `../../dbctool_out/`
  - Cached ADTs: `../../cached_maps/` (via WoWRollback/)
  - Rollback: `../../rollback_outputs/` (via WoWRollback/)

---

**Remember**: We're building a research tool for preserving WoW history. Quality and accuracy matter more than speed. Take time to understand before changing.
