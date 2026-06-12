# Implementation Plan: M2 MD20 v0x109+ Cataclysm Support

**Branch**: `059-m2-md20-v109-cata-support` | **Date**: 2026-06-11 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/059-m2-md20-v109-cata-support/spec.md`

## Summary

Split the current `>= 0x108` dispatch into two explicit era tags: `Md20_3X_V108` (3.3.5 Wrath) and `Md20_4X_V109` (4.0.0+ Cataclysm and beyond). Both use the same `M2ModelReader` (3.3.5 reader with same feature support). The Ghidra project at `output/tmp/wowarchive-clients/4_0_0_11927/World of Warcraft/ghidra/` is the reference for the native binary, and real `.m2` files from the staged 4.0.0.11927 client are the validation ground truth.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: `WowViewer.Core.IO` M2 reader family (`M2ModelReaderDispatcher`, `M2Era1121EraTag`, `M2Era1121Version`)

**Testing**: Existing 3.3.5, 1.12.1, and chunked-MDLX tests must pass unchanged

**Constraints**: No changes to `M2ModelReader` behavior. No new M2 features. Era tag split only.

## Project Structure

```text
wow-viewer/src/core/WowViewer.Core.IO/
├── M2Era1121/
│   ├── M2Era1121EraTag.cs      # Add Md20_4X_V109
│   └── M2Era1121Constants.cs   # No changes
└── M2Chunked/
    └── M2ModelReaderDispatcher.cs  # Update DetectEra + dispatch

wow-viewer/tests/WowViewer.Core.Tests/
└── M2Era1121ModelReaderTests.cs    # Add 0x109/0x10A dispatch tests
```

## Implementation Phases

### Phase 1: Era Tag Split + Dispatch Update

**Goal**: Add `Md20_4X_V109` era tag, update `DetectEra` to route 0x109+ to it, update dispatch to use 3.3.5 reader for the new tag.

**Approach**:
1. Add `Md20_4X_V109 = 4` to `M2Era1121EraTag` enum with label `"4.x / Cata+ (MD20 v0x109)"`.
2. Split the `>= 0x108` condition in `DetectEra`: version 0x108 → `Md20_3X_V108`, version >= 0x109 → `Md20_4X_V109`.
3. Add `Md20_4X_V109` case to dispatch switch — uses same `M2ModelReader`.
4. Add `Md20_4X_V109` to the `[InlineData]` in existing `DetectEra_ReturnsExpectedTag` tests.

**Validation**: `dotnet test` passes. `m2 inspect` on a 4.0.0 .m2 shows `ERA: 4.x / Cata+ (MD20 v0x109)`.

### Phase 2: Ghidra Confirmation

**Goal**: Use Ghidra to confirm the version gate in the 4.0.0.11927 native binary.

**Approach**:
1. When Ghidra analysis completes, locate the M2 loader function.
2. Read the version check — what range does 4.0.0.11927 accept?
3. Document the finding in the architecture doc.

**Validation**: Ghidra decompilation shows the version gate. Documented in spec.

### Phase 3: Real-Data Validation

**Goal**: Load a 4.0.0.11927 .m2 file through the dispatcher and confirm it parses without error.

**Approach**:
1. Extract a `.m2` file from the staged 4.0.0.11927 MPQ archives.
2. Run `WowViewer.Tool.Inspect m2 inspect` on the extracted file.
3. Verify the era tag is `4.x / Cata+` and the model parses successfully.

**Validation**: `m2 inspect` output shows correct era tag, no parse errors.
