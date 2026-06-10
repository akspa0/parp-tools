# ⏰ Downtime Reminder

**Show this during long-running operations!**

---

## 💡 Research Opportunity: ADT Null-Out Optimization

While waiting for generation/processing, consider working on:

### **Manual ADT Testing** (User-led)

**Goal**: Understand how ADT structure changes when objects are deleted

**Quick Test**:
1. Open a small map in WoW map editor (Noggit, etc.)
2. Add a single M2 object
3. Save the ADT
4. Delete the object
5. Save the ADT again
6. Binary compare the two files

**What to look for**:
- Did chunk sizes change?
- Were offsets recalculated?
- How were model name entries handled?
- Are there patterns we can exploit?

**Files to create**:
- `test_single_m2_before.adt`
- `test_single_m2_after.adt`
- `test_single_wmo_before.adt`
- `test_single_wmo_after.adt`

---

## 📚 Noggit Code Review

**Source**: https://github.com/Marlamin/noggit-red

**Search for**:
- Object deletion functions
- ADT save/export logic
- Chunk update code
- Offset recalculation methods

**Questions to answer**:
- How does Noggit handle removing model names?
- Does it recalculate offsets or just null out?
- What chunk validation does it do?

---

## 📊 Current Status

**Current Approach**: Invisible model replacement (safe, proven)

**Future Optimization**: Null out model names + recalculate offsets

**Decision Point**: After manual testing proves it's safe

**See**: `docs/planning/05-future-null-out-optimization.md`

---

**Keep this research in mind during downtime!** 🦀
