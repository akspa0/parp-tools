# Implementation Plan: LIT Documentation Update

**Branch**: `157-lit-documentation-update` | **Date**: 2026-08-16 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/157-lit-documentation-update/spec.md`

## Summary

Create updated documentation for LIT files based on the original wowdev.wiki documentation (https://wowdev.wiki/LIT) and our implementation's findings for v2 areatest.lit files. The goal is to produce a community-ready documentation update that preserves original terminology while adding our implementation findings as clearly labeled comments/interpretations.

## Technical Context

**Language/Version**: Markdown documentation (no code implementation)

**Primary Dependencies**: 
- wowdev.wiki LIT page (reference)
- wow-viewer/src/core/WowViewer.Core.IO/Lit/ (implementation)
- wow-viewer/src/viewer/WoWViewer/Terrain/LitLoader.cs (runtime usage)
- wow-viewer/tests/WowViewer.Core.Tests/LitProfileReaderTests.cs (test fixtures)

**Storage**: Documentation files in `wow-viewer/docs/wowdev-wiki/`

**Testing**: Manual review against implementation and wowdev.wiki

**Target Platform**: GitHub/wiki documentation

**Project Type**: Documentation update

**Performance Goals**: N/A

**Constraints**: 
- Preserve original wowdev.wiki terminology
- Add implementation findings as comments only
- Do not modify existing wowdev.wiki structure
- Focus on v2 areatest.lit which has no documentation on wowdev.wiki

**Scale/Scope**: Single documentation file update with implementation findings

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [ ] Documentation follows existing wowdev.wiki structure
- [ ] Implementation findings are clearly separated as comments/interpretations
- [ ] Original terminology is preserved
- [ ] v2 areatest.lit findings are documented (currently undocumented on wiki)

## Project Structure

### Documentation (this feature)

```text
specs/157-lit-documentation-update/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (N/A for docs)
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
wow-viewer/
├── docs/wowdev-wiki/
│   └── lit-draft.md          # Updated LIT documentation (output)
├── src/core/WowViewer.Core.IO/Lit/
│   ├── LitProfileReader.cs   # Reference implementation
│   ├── LitProfileModels.cs   # Data models
│   └── LitSummaryReader.cs   # Summary reader
├── src/viewer/WoWViewer/Terrain/
│   └── LitLoader.cs          # Runtime loader
└── tests/WowViewer.Core.Tests/
    └── LitProfileReaderTests.cs  # Test fixtures with v2 data
```

**Structure Decision**: Documentation-only feature. Output goes to `wow-viewer/docs/wowdev-wiki/lit-draft.md` for community submission.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |