# WoWRollback Active Context

## Current Blocker (Dec 2025)
**Output writers produce corrupted files despite working input parsers.**

### The Bug
- Merged ADTs are half expected size
- Chunks dropped silently
- FourCC confusion (`MTEX` vs `XETM`) throughout codebase

### Immediate Fix
1. Audit for reversed FourCC literals — consolidate to single write point
2. Merge `AdtPatcher.cs` + `SplitAdtMerger.cs` into one tested class
3. Add verification against known-good reference files
4. Regenerate test data with fixed code

### Key Insight
> Split files (`_obj0`, `_tex0`) are overlay patches. Their data MUST take precedence over root ADT.

## Do NOT
- Add features until writers work
- Trust `PM4ADTs/combined/` (corrupted)
- Use reversed FourCC outside write path