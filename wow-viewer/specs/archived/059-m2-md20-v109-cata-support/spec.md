# Feature Specification: M2 MD20 v0x109+ Cataclysm Support

**Feature Branch**: `059-m2-md20-v109-cata-support`
**Created**: 2026-06-11
**Status**: Draft
**Owner**: wow-viewer (Core.IO M2 reader)
**Related**: spec 043 (chunked MDLX), spec 048 (1.12.1 MD20 reader), architecture doc `m2-native-client-research-2026-03-31.md`, Ghidra prompt `prompt-400.md`

**Input**: User wants the M2 reader to accept version 0x109 and higher (4.0.0+ Cataclysm and beyond), reusing the 3.3.5 0x108 reader for supported features. The 4.0.0.11927 Ghidra project at `output/tmp/wowarchive-clients/4_0_0_11927/World of Warcraft/ghidra/` is the source of truth for the M2 loader function.

## Context

The current `M2ModelReaderDispatcher.DetectEra` routes ALL `MD20` versions `>= 0x108` to `M2Era1121EraTag.Md20_3X_V108`, which dispatches to the 3.3.5 `M2ModelReader`. This already handles 0x109+ by accident — the reader has no upper-bound version gate.

What's missing:
1. **A distinct era tag** for 0x109+ so the dispatch is explicit, not accidental.
2. **Ghidra-confirmed version gate** from the 4.0.0.11927 native binary — does it accept 0x108 only, or 0x108+?
3. **Real-data validation** against 4.0.0.11927 `.m2` files to confirm the format is compatible with the 3.3.5 reader.

The 3.3.5 native binary rejects 0x109 (`0x107 < ver < 0x109`), and the 3.0.1 binary does the same (`0x103 < ver < 0x109`). This means 0x109 represents a version increment that older clients cannot handle. The 4.0.0 client may accept 0x109, 0x108, or both.

## Ghidra Investigation (Task 5 from prompt-400.md)

The Ghidra project at `output/tmp/wowarchive-clients/4_0_0_11927/World of Warcraft/ghidra/` already has:
- Static anchors for `%02d.skin`, `%04d-%02d.anim`, skin choose/load/init, section materialization, effect builder
- A Ghidra prompt (`gillijimproject_refactor/specifications/ghidra/prompt-400.md`) that asks Task 5: "Check if M2 version field has incremented from 3.3.5's 264 (0x108)"

What we need from Ghidra:
1. Find the M2 loader function (search for `MD20` = `0x3032444D` in the binary)
2. Read the version check — what range does 4.0.0.11927 accept?
3. Identify the minimum and maximum accepted versions
4. Determine if any new chunk handlers exist in the MD21 wrapper path

## User Scenarios

### User Story 1 - Load 4.0.0+ M2 models (Priority: P1)

Load `.m2` files from staged 4.0.0.11927 client. The model loads using the 3.3.5 reader with the same feature set.

**Independent Test**: Load a known `.m2` from the staged 4.0.0.11927 client, verify the model renders in the viewer without errors.

### User Story 2 - Clear era tag in inspect output (Priority: P1)

`WowViewer.Tool.Inspect m2 inspect` shows `ERA: Cataclysm / 4.x (MD20 v0x109)` for 4.0.0+ files, not `ERA: 3.3.5 (MD20 v0x108)`.

## Requirements

- **FR-001**: Add `Md20_4X_V109` to `M2Era1121EraTag` enum.
- **FR-002**: Update `DetectEra` in `M2ModelReaderDispatcher` to route version 0x109+ to the new era tag.
- **FR-003**: Existing `M2ModelReader` (3.3.5 reader) handles the new era tag with same feature support.
- **FR-004**: `WowViewer.Tool.Inspect m2 inspect` prints the correct era label for 0x109+ files.
- **FR-005**: Ghidra analysis confirms the version gate in the 4.0.0.11927 binary.
- **FR-006**: Pre-existing 3.3.5, 1.12.1, and chunked-MDLX tests unchanged.
- **FR-007**: Unit test proves a synthetic 0x109 header dispatches correctly.
- **FR-008**: Unit test proves a synthetic 0x10A header also dispatches correctly (open-ended upper bound).

## Out of Scope

- New 4.0.0-specific M2 features (TXID, SFID, new chunk types). The 3.3.5 reader provides baseline support only.
- Runtime validation against 4.0.0 client (x64dbg). This spec is static/Ghidra only.
- Changes to the `M2ModelReader` itself beyond version acceptance.
