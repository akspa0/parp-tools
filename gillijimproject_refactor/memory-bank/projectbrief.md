# Project Brief

## Mission
Bidirectional WoW format conversion between **Alpha 0.5.3** and **LK 3.3.5** clients.

## Scope
- **Read/Write**: WDT, ADT, WMO, M2/MDX, BLP formats for both versions
- **Convert**: Modern → Alpha (retroporting) and Alpha → Modern (analysis)
- **Validate**: Roundtrip integrity, in-game client testing

## Current Reality (Dec 2025)
**Input parsers work. Output writers are broken.**

We have standardized libraries that correctly read both Alpha and LK formats. Yet our output files fail despite having working writers in standalone tools (BlpResizer, AlphaWdtInspector). The bug is in the write path integration, not the format understanding.

## Success Criteria
1. **Merged ADTs** match expected size (not half)
2. **Roundtrip** Alpha → LK → Alpha preserves all data
3. **In-game** loading without crashes
4. **No reversed FourCC** literals in code (normalize on read, reverse only on write)

## Key Constraint
Fix the output path before adding features. No new functionality until writers work.
