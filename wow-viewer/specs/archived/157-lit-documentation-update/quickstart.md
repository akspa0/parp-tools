# Quickstart: LIT Documentation Update

## Goal

Produce an updated LIT file format documentation file suitable for submission to wowdev.wiki, incorporating implementation findings from wow-viewer.

## Steps

### 1. Review Source Materials
- Original wowdev.wiki LIT page: https://wowdev.wiki/LIT
- Implementation: `wow-viewer/src/core/WowViewer.Core.IO/Lit/`
- Test fixtures: `wow-viewer/tests/WowViewer.Core.Tests/LitProfileReaderTests.cs`

### 2. Generate Documentation Draft
Create `wow-viewer/docs/wowdev-wiki/lit-draft.md` with:
- Original wiki structure preserved
- Implementation findings as `[Implementation Note: ...]` comments
- v2 areatest.lit fully documented
- All version-specific details

### 3. Validate Against Implementation
- Cross-reference every claim with source code
- Verify test fixture data matches documentation
- Ensure coordinate conversion formulas are correct

### 4. Review Checklist
- [ ] All 4 versions documented (v2, v83, v84, v85)
- [ ] v2 partial layout complete
- [ ] Spatial coordinates (XZY, 1/36, Y/Z swap) documented
- [ ] Color tracks (BGRX, 2880 units/day) documented
- [ ] Float bands (fog, sky, parameter) documented
- [ ] Group kinds enumerated
- [ ] Implementation notes clearly marked
- [ ] Original terminology preserved

## Output Location

`wow-viewer/docs/wowdev-wiki/lit-draft.md`

## Validation Commands

```powershell
# Run LIT tests to verify implementation
dotnet test wow-viewer/tests/WowViewer.Core.Tests/LitProfileReaderTests.cs -c Debug

# Run LIT summary tests
dotnet test wow-viewer/tests/WowViewer.Core.Tests/LitSummaryReaderTests.cs -c Debug

# Run LIT terrain day/night tests
dotnet test wow-viewer/tests/WowViewer.Core.Tests/LitTerrainDayNightProfileTests.cs -c Debug
```

## Submission Process

1. Review `lit-draft.md` against wowdev.wiki LIT page
2. Prepare diff or annotated version for community review
3. Submit via wowdev.wiki contribution process