# Feature Specification: LIT Documentation Update

## Overview

Create updated documentation for LIT (Lighting) files based on the original wowdev.wiki documentation (https://wowdev.wiki/LIT) and our implementation's findings for v2 areatest.lit files. The v2 format is currently undocumented on wowdev.wiki.

## Requirements

### Functional Requirements

1. **FR-001**: Document the LIT file format structure for all known versions (v2, v83, v84, v85)
2. **FR-002**: Preserve original wowdev.wiki terminology and structure
3. **FR-003**: Add implementation findings as clearly labeled comments/interpretations
4. **FR-004**: Document the v2 pre-alpha partial layout (areatest.lit) which has no wiki documentation
5. **FR-005**: Document spatial coordinate handling (client fixed-point XZY at 1/36 scale)
6. **FR-006**: Document color track layout (BGRX packed format, 2880 time units per day)
7. **FR-007**: Document float bands (fog end, fog start scalar, sky bands, parameter bands)
8. **FR-008**: Document light group kinds (Clear, Storm, ClearWater, StormWater, Partial, LegacyPartialAlternate)

### Non-Functional Requirements

1. **NFR-001**: Output must be suitable for submission to wowdev.wiki
2. **NFR-002**: Implementation comments must be clearly distinguishable from original wiki content
3. **NFR-003**: No changes to existing wowdev.wiki structure or terminology
4. **NFR-004**: All findings must be traceable to implementation code

## Scope

### In Scope
- LIT file header structure
- Light header structure (64 bytes)
- Light group structure (version-dependent stride)
- Color tracks (keyframe format, interpolation)
- Float bands (fog, sky, parameter bands)
- Version-specific layouts (v2, v83, v84, v85)
- Spatial coordinate conversion (XZY fixed-point → world XYZ)
- v2 pre-alpha partial layout specifics

### Out of Scope
- Runtime lighting evaluation (covered in other specs)
- LIT spatial application/blending (known bug in MdxViewer)
- MDX LITE chunk (different format)
- WMO MOLT lighting
- DBC Light* tables

## Acceptance Criteria

1. Documentation covers all 4 known LIT versions
2. v2 areatest.lit layout is fully documented
3. Implementation findings are marked as `[Implementation Note]` or similar
4. Original wowdev.wiki terminology is preserved
5. Spatial coordinate conversion is documented with the 1/36 scale factor
6. Color track BGRX format and 2880 time units/day are documented
7. Output file is ready for community review/submission