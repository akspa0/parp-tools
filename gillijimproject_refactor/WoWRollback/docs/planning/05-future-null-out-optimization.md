# Future Optimization: Null-Out vs Model Replacement

**Status**: 🔬 RESEARCH PHASE - Manual testing required first

**Current Approach**: Replace model paths with invisible models (same-length replacement)  
**Potential Optimization**: Null out model names entirely and recalculate offsets

---

## 🎯 Hypothesis

Instead of replacing unwanted model paths with invisible models, we may be able to:
1. **Null out the model name entries** completely
2. **Recalculate chunk offsets** as needed
3. **Update reference indices** in MDDF/MODF chunks

This could be simpler and more efficient than maintaining length-matched invisible models.

---

## 📋 Research Tasks

### Phase 0: FLAGS INVESTIGATION ⚠️ **PRIORITY**

**CRITICAL DISCOVERY**: Noggit has "show hidden models" feature - models might have visibility flags!

**Hypothesis**: Instead of replacing/deleting model paths, we might be able to:
1. Set a "hidden" flag on model placements
2. Keep model paths intact
3. Client respects the flag and doesn't render the model

**Investigation Steps**:
1. **Examine MDDF/MODF Structure**:
   - Locate flags field in placement data
   - Document bit flags and their meanings
   - Identify "hidden" or "disabled" flag

2. **Noggit "Show Hidden Models" Feature**:
   - How does Noggit implement this?
   - What flag does it check/set?
   - Is this WoW client behavior or Noggit-only?

3. **Test in Alpha Client**:
   - Set various flags on test objects
   - See if client respects them
   - Document which flags cause invisibility

4. **Flag Types to Track**:
   - Visibility/hidden flags
   - Particle emitter flags
   - Shader-only objects (no geometry)
   - Collision-only objects
   - LOD flags

**If This Works**: We might not need model replacement OR null-out at all! Just set a flag! 🎉

---

### Phase 1: Manual ADT Testing (USER-LED)

**Goal**: Understand how ADT structure changes when objects are deleted.

**Steps**:
1. **Create test ADTs** with single models using WoW map editor
2. **Delete objects** and resave the ADT
3. **Binary comparison** of before/after files
4. **Analyze changes**:
   - Which chunks are affected?
   - Do offsets get recalculated?
   - Are chunks resized or just zeroed?
   - How are MMID/MWID indices updated?
   - **What happens to flags field?**

**Test Cases**:
- [ ] Single M2 object
- [ ] Single WMO object
- [ ] Multiple M2 objects (delete one)
- [ ] Multiple WMO objects (delete one)
- [ ] Mixed M2 + WMO (delete various combinations)

**Deliverables**:
- Before/after ADT pairs for each test case
- Binary diff analysis notes
- Findings document

---

### Phase 2: Noggit Research

**Goal**: Learn from existing map editor implementation.

**Source**: [Noggit GitHub](https://github.com/Marlamin/noggit-red) or similar WoW map editors

**Investigation**:
1. **Locate object deletion code**
   - How does Noggit handle MMDX/MWMO chunk updates?
   - Does it null out entries or remove them entirely?
   - Are offsets recalculated or preserved?

2. **Chunk update logic**
   - Which chunks get modified when deleting objects?
   - How are MDDF/MODF indices updated?
   - Is there special handling for MHDR offsets?

3. **File integrity checks**
   - Does Noggit validate chunk sizes?
   - Are there CRC or checksum updates?
   - How does it maintain backward compatibility?

**Deliverables**:
- Code snippets from Noggit
- Implementation notes
- Comparison with our approach

---

## 🔍 Key Questions to Answer

1. **Offset Recalculation**:
   - Q: Do we need to update MHDR chunk offset table?
   - Q: Do MCIN offsets change when objects are removed?
   - Q: Are there other offset references we're missing?

2. **Chunk Resizing**:
   - Q: Can we shrink MMDX/MWMO chunks, or must they stay same size?
   - Q: If we shrink, do all subsequent chunks need offset updates?
   - Q: Is there a performance benefit to resizing vs zeroing?

3. **Client Compatibility**:
   - Q: Will the client accept resized chunks?
   - Q: Are there minimum chunk size requirements?
   - Q: Do different client versions (Alpha vs LK) have different rules?

4. **Index Management**:
   - Q: When we remove a model name, do we shift all indices down?
   - Q: Or do we leave gaps and update MDDF/MODF to skip removed indices?
   - Q: How does this affect MMID/MWID offset arrays?

---

## 💡 Potential Approaches

### Approach A: Null-Out In-Place (Current Method) ✅
**Pros**:
- Simple implementation
- No offset recalculation needed
- File size preserved
- Low risk of corruption

**Cons**:
- Wasted space (null-filled model names)
- Requires invisible model assets
- Not "true" deletion

### Approach B: Remove Entries + Recalculate Offsets (Future)
**Pros**:
- True deletion, no wasted space
- Smaller file sizes
- No dependency on invisible models
- More "correct" from format perspective

**Cons**:
- Complex implementation
- Must update all offset references
- Higher risk of corruption
- Requires deep understanding of ADT structure

### Approach C: Hybrid (Best of Both?)
**Possible Strategy**:
1. Identify which chunks actually use offsets
2. Remove entries from end of lists (no offset shift needed)
3. Only recalculate if removing from middle
4. Fall back to null-out if recalculation is risky

---

## 🧪 Test Plan (After Manual Research)

### Test 1: Binary Diff Analysis
Compare before/after ADTs from manual testing:
```powershell
# Hex diff to see exact byte changes
fc /b original.adt modified.adt > diff.txt

# Or use better hex diff tool
BeyondCompare original.adt modified.adt
```

**Look for**:
- Chunk size changes
- Offset updates in MHDR
- MMID/MWID index shifts
- Null-filled regions

### Test 2: Parse & Compare
Use our ADT parser to analyze structure:
```csharp
var original = AdtParser.Parse("original.adt");
var modified = AdtParser.Parse("modified.adt");

// Compare chunk counts
Console.WriteLine($"MMDX entries: {original.MmdxCount} → {modified.MmdxCount}");
Console.WriteLine($"MDDF entries: {original.MddfCount} → {modified.MddfCount}");

// Compare offsets
Console.WriteLine($"MMDX offset: {original.MmdxOffset} → {modified.MmdxOffset}");
```

### Test 3: Client Validation
Test each modified ADT in WoW client:
- [ ] File loads without errors
- [ ] Objects properly removed
- [ ] No crashes or visual glitches
- [ ] Performance unchanged

---

## 📊 Implementation Complexity Estimate

| Approach | Complexity | Risk | Benefit | Priority |
|----------|-----------|------|---------|----------|
| Current (Null-Out) | Low | Low | Medium | ✅ Now |
| Remove + Recalc | High | Medium | High | 🔮 Later |
| Hybrid | Medium | Medium | High | 🔮 Later |

---

## 🚨 Reminder System

**Trigger**: When waiting for long-running operations (cache regeneration, viewer generation, etc.)

**Message**:
```
💡 REMINDER: During downtime, consider testing ADT null-out optimization!

Steps:
1. Create test ADTs with single objects
2. Delete objects and resave
3. Binary diff analysis
4. Review Noggit implementation

See: docs/planning/05-future-null-out-optimization.md
```

**Downtime Opportunities**:
- Waiting for `rebuild-and-regenerate.ps1` (5-10 min)
- Waiting for DBCTool.V2 analysis (2-5 min)
- Waiting for AlphaWdtAnalyzer conversion (3-7 min)
- Waiting for large map processing (10-30 min)

---

## 📝 Current Status: DEFERRED

**Current Implementation**: Using invisible model replacement (safe, proven approach)

**Future Work**: After manual testing confirms offset recalculation is feasible

**Decision Point**: When we have:
1. ✅ Test ADT pairs (before/after deletion)
2. ✅ Binary diff analysis results
3. ✅ Noggit code review findings
4. ✅ Proof of concept working in test client

**Until then**: Continue with current approach, document findings as we go.

---

## 🎯 Success Criteria (Future)

When this optimization is implemented, we should have:
- [ ] True object deletion (no invisible models needed)
- [ ] Automatic offset recalculation
- [ ] Smaller patched file sizes
- [ ] Same or better client compatibility
- [ ] Comprehensive test coverage
- [ ] Fallback to null-out if recalc fails

---

**Keep this document updated as research progresses!**

---

## 🔗 Related Documents

- `01-rollback-feature-plan.md` - Main rollback feature design
- `03-alphawdt-patching-sprint.md` - Current implementation approach
- `04-per-tile-ui-implementation.md` - UI for selecting objects to remove

---

**Next Update**: After manual ADT testing is complete
