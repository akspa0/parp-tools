# Feature Specification: MH2O / MCLQ Liquid Type Determination Fix

**Feature Branch**: `041-mh2o-mclq-liquid-type-determination-fix`
**Created**: 2026-06-02
**Status**: Draft (Implementation Slice)
**Predecessor**: `specs/040-mh2o-mclq-liquid-type-determination/` (Research Slice, approved)
**Input**: User description: "write spec 041 (liquid type fix)" — closing the loop on the "lava for everything" rendering bug in 1.x+ data. The 3.3.5 Ghidra evidence and wow-viewer bug catalog are in spec 040's `research.md`.

## Status Notice

This is the **implementation slice** for spec 040. It targets exactly the 5 files in spec 040's "Files to Fix" table plus 2 new helper types, validates against real 1.x/2.x/3.x data from `output/tmp/wowarchive-clients/`, and ships an inspection command for empirical proof. It honors spec 040's `FR-009` and the protected surfaces from RULE 10 + spec 037.

## User Scenarios & Testing

### User Story 1 - Unified MCNK flag decoder (Priority: P1)

As a wow-viewer developer, I want a single canonical helper that maps MCNK flags to `AdtLiquidBasicType`, so every code path that touches MCNK flags uses the same bit mapping and stops disagreeing.

**Why this priority**: Spec 040's table shows 4 wow-viewer functions (`AlphaToLkConverter.ResolveLiquidBasicType`, `AlphaTensorPackBuilder.McnkFlagsToLiquidType`, `LkToAlphaConverter.ClassifyLkLiquid`, `AlphaTileData.ClassifyLiquid`) all use a broken `((mcnkFlags>>4)&0x3)` switch and disagree with each other and with the 3 correct functions (`LiquidConverter.GetLiquidTypeFromMcnkFlags`, `AlphaLiquidTypeCodec.ResolveBasicType`, `LkToAlphaConverter.MapAlphaLiquidFlags`). The 4-way drift is the root cause of the "lava for everything" symptom.

**Independent Test**: Open the new helper, run the unit tests in `wow-viewer/tests/WowViewer.Core.Tests/Maps/McnkFlagDecoderTests.cs`, verify all 16 possible MCNK flag combinations produce the documented `AdtLiquidBasicType`.

**Acceptance Scenarios**:

1. **Given** an MCNK flag of `0x00`, **When** `McnkFlagDecoder.Decode(uint mcnkFlags)` is called, **Then** it returns `AdtLiquidBasicType.Water`.
2. **Given** an MCNK flag of `0x04` (River bit), **When** the helper is called, **Then** it returns `AdtLiquidBasicType.River` (or `Water` if the renderer treats River as water).
3. **Given** an MCNK flag of `0x08` (Ocean bit), **When** the helper is called, **Then** it returns `AdtLiquidBasicType.Ocean`.
4. **Given** an MCNK flag of `0x10` (Magma bit), **When** the helper is called, **Then** it returns `AdtLiquidBasicType.Magma`.
5. **Given** an MCNK flag of `0x20` (Slime bit), **When** the helper is called, **Then** it returns `AdtLiquidBasicType.Slime`.
6. **Given** an MCNK flag of `0x30` (Magma+Slime bits set simultaneously), **When** the helper is called, **Then** it returns `Magma` (per `AlphaLiquidTypeCodec.ResolveBasicType` precedence: check Magma first, then Slime) and the precedence is documented in the helper's XML doc.
7. **Given** an MCNK flag of `0xFF` (all bits set), **When** the helper is called, **Then** it returns `Magma` (highest-priority set bit wins) without throwing.

---

### User Story 2 - DBC-backed LiquidType lookup (Priority: P1)

As a wow-viewer developer, I want a DBC reader for `LiquidType.dbc` that exposes the `type` field at offset `0x38`, so 3.x+ `AdtLiquidReader.MapLiquidTypeId` does not have to hardcode `17/19/20` and can resolve any LiquidTypeId to its actual type.

**Why this priority**: Spec 040 FR-005 establishes the priority `MH2O LiquidTypeId → DBC type field at offset 0x38` as the authoritative first step. The 3.3.5 Ghidra `FUN_00439760` confirms this is what the live client does. Without DBC, every 3.x ADT with a non-17/19/20 LiquidTypeId (rows 1, 2, 3, 13, 14, etc.) defaults to `Water` and the type is silently wrong.

**Independent Test**: Load `LiquidType.dbc` from a staged 3.3.5 client (or use a `DbcLiquidTypeTable` built from a JSON test fixture), verify `ResolveBasicType(17)` returns the DBC row's `type` field (which should be `1` for Water, per the convention table in spec 040 §4.3), verify fallback to `param_1 = 1` (water) on miss.

**Acceptance Scenarios**:

1. **Given** a 3.3.5 `LiquidType.dbc` file with row 17 (Ocean) having `type=1` (Water) and row 19 (Magma) having `type=2` (Magma), **When** `DbcLiquidTypeTable.ResolveBasicType(19)` is called, **Then** it returns `AdtLiquidBasicType.Magma`.
2. **Given** a LiquidTypeId that does not exist in the DBC (e.g. `9999`), **When** the table is queried, **Then** it returns `AdtLiquidBasicType.Water` (per 3.3.5 `FUN_00439760` `param_1 = 1` fallback) and logs a warning at most once per LiquidTypeId.
3. **Given** a 3.3.5 `LiquidType.dbc` file, **When** the DBC reader parses row 1, **Then** the parsed record's `Type` field (offset `0x38`, uint) is `1` (Water), `Name` (offset `0x04`, string ref) is the localized name, and `Flags` (offset `0x08`, int) is preserved.
4. **Given** a `DbcLiquidTypeTable` built from a JSON test fixture, **When** the inspect command runs against a 3.3.5 ADT, **Then** the resolution is deterministic and offline-replayable (no MPQ access required).

---

### User Story 3 - Fix the 5 buggy sites (Priority: P1)

As a wow-viewer developer, I want the 5 buggy functions in spec 040's "Files to Fix" table rewritten to use the canonical `McnkFlagDecoder` and (where applicable) the `DbcLiquidTypeTable`, so 1.x+ ADTs render the correct liquid color.

**Why this priority**: This is the user-visible bug. The 4-way drift and the DBC gap together produce the "lava for everything" symptom. After this story, a 1.12 ADT with `MCLQ tile nibble=0x04` (Slime) and `MCNK flag=0x20` (Slime) renders green, not orange.

**Independent Test**: Run `WowViewer.Tool.Inspect map inspect --adt <path> --dump-liquid-types` against a real 1.12 ADT (extracted from `output/tmp/wowarchive-clients/1.X_Retail_Windows_enUS_1.12.1.5875/`), verify per-tile `BasicType` matches the MCNK flag bit assignments and MCLQ tile nibbles per spec 040 §3 canonical table.

**Acceptance Scenarios**:

1. **Given** a 1.12 ADT with `McnkFlags=0x00` and MCLQ tile nibble `0x00`, **When** the fix runs, **Then** the resolved `BasicType` is `Water` (no MCNK bit set, no tile nibble set, default).
2. **Given** a 1.12 ADT with `McnkFlags=0x20` and MCLQ tile nibble `0x00`, **When** the fix runs, **Then** the resolved `BasicType` is `Slime` (was: `Magma` per the broken switch).
3. **Given** a 1.12 ADT with MCLQ tile nibble `0x04` (Slime per `AlphaLiquidTypeCodec.GetWriterTileTypeNibble`), **When** the fix runs, **Then** the resolved `BasicType` is `Slime` (tile nibble wins over MCNK flag, per `AlphaLiquidTypeCodec` intent).
4. **Given** a 3.3.5 ADT with MH2O `LiquidTypeId=13` and no DBC, **When** the fix runs, **Then** the resolved `BasicType` is `River` (per documented WL→MH2O mapping row 13 = River in spec 040 §4.2) or `Water` (per 3.3.5 default) — must be one of these, not `Magma`.
5. **Given** a 3.3.5 ADT with MH2O `LiquidTypeId=19` and a loaded DBC where row 19's `type=2`, **When** the fix runs, **Then** the resolved `BasicType` is `Magma` (DBC-backed, no hardcoding).

---

### User Story 4 - Empirical validation via inspect command (Priority: P2)

As a wow-viewer developer, I want `WowViewer.Tool.Inspect map inspect --dump-liquid-types` to emit a JSON file with per-tile raw and resolved liquid data, so the fix can be proven against real 1.x/2.x/3.x ADTs, not just unit tests.

**Why this priority**: Spec 040 US-4 + FR-006 + §5 establish the validation path. Unit tests for the helper are necessary but not sufficient — the bug is in the integration between 4 files, and the only way to prove the integration works is on real data.

**Independent Test**: Run the inspect command against one ADT per staged client version (1.12, 2.4.3, 3.3.5), each with at least one water, one magma, and one slime tile. Compare the resolved `BasicType` per tile against the source MCNK flags / MCLQ tile nibbles / MH2O LiquidTypeId.

**Acceptance Scenarios**:

1. **Given** a 1.12 ADT extracted from `1.X_Retail_Windows_enUS_1.12.1.5875/.../World/Maps/`, **When** the inspect command runs, **Then** the output JSON includes per-tile: `mcnkFlags`, `mclqTileNibble` (lower 4 bits), `mh2oLiquidTypeId` (0 for 1.x), and `resolvedBasicType` (string enum name).
2. **Given** a 3.3.5 ADT with MH2O data, **When** the inspect command runs with `--dbc <path-to-LiquidType.dbc>`, **Then** the output JSON includes the DBC `type` field for each `LiquidTypeId` and the resolved `BasicType` reflects the DBC lookup.
3. **Given** the JSON output, **When** a developer diffs the `resolvedBasicType` against the expected (per spec 040 §3 canonical table), **Then** the diff is empty (no misclassified tiles).
4. **Given** a 2.4.3 ADT with both MCLQ and MH2O (TBC transition era), **When** the inspect command runs, **Then** the output shows MH2O wins (per 3.x+ priority), MCLQ is logged but not used.

---

### User Story 5 - 1.x/2.x ADT extraction pipeline (Priority: P2)

As a wow-viewer developer, I want a scriptable pipeline to extract ADT files from the staged 1.12 and 2.4.3 MPQs, so the empirical validation has a known fixture set.

**Why this priority**: Spec 040's "Empirical Validation Plan" §5 requires extraction. The staged clients have MPQ files only (`output/tmp/wowarchive-clients/1.X_Retail_Windows_enUS_1.12.1.5875/606/World of Warcraft/Data/common.MPQ` etc.). Without extraction, the empirical validation in US-4 is blocked.

**Independent Test**: Run the extraction script against one map per staged client (e.g. `Kalimdor\Orgrimmar.adt` for 1.12 with magma at the lava pool, `Outland\HellfirePeninsula.adt` for 2.4.3 with MH2O, `Kalimdor\Ashenvale.adt` for 3.3.5 with MH2O), verify the extracted ADT loads in `AdtLiquidReader.Read` without errors.

**Acceptance Scenarios**:

1. **Given** the staged 1.12 MPQ files, **When** the extraction script runs against `World\Maps\Kalimdor\Orgrimmar*.adt`, **Then** at least one `*_obj0.adt` and one `*_tex0.adt` is produced in `output/tmp/extracted-clients/1.12.1/Kalimdor/`.
2. **Given** the staged 2.4.3 MPQ files, **When** the extraction script runs against `World\Maps\Outland\HellfirePeninsula*.adt`, **Then** MH2O-bearing ADTs are extracted.
3. **Given** an extracted ADT, **When** `AdtLiquidReader.Read` parses it, **Then** the resulting `AdtLiquidFile` has the expected number of `Layers` per `Mh2oInstance.LayerCount` and no `null` `LiquidTypeId` values.

---

### Edge Cases

- What happens when MCNK flag is `0x30` (Magma+Slime both set)? `McnkFlagDecoder` returns `Magma` per `AlphaLiquidTypeCodec` precedence (Magma check first, then Slime). Documented in helper XML doc.
- What happens when MH2O `LiquidTypeId=0`? `DbcLiquidTypeTable.ResolveBasicType(0)` returns `Water` (default fallback). Matches 3.3.5 behavior.
- What happens when DBC file is missing or corrupt? `DbcLiquidTypeTable` returns `Water` for every query and logs a single warning. No exception propagates to the ADT reader.
- What happens when MCLQ tile flags are all `0x0F` (DontRender)? `AlphaLiquidTypeCodec.GetVisibleTileTypeNibble` returns `0`, falls through to MCNK flag check. Unchanged.
- What happens when 1.12 ADT has `MclqLiquidType` enum values that conflict with `AlphaLiquidTypeCodec.GetWriterTileTypeNibble`? The two encodings coexist in wow-viewer; the fix preserves both and only the new `McnkFlagDecoder` enforces the canonical MCNK bit mapping. The `MclqLiquidType` enum is NOT modified.
- What happens when the same wow-viewer code path is hit by both Alpha and LK data in a single run (e.g. a tool that processes a folder of mixed-version ADTs)? `McnkFlagDecoder` and `DbcLiquidTypeTable` are version-agnostic; the dispatch (which decoder to use) happens at the `AdtLiquidReader` / `AlphaToLkConverter` boundary. The 5 fixed files all use the same canonical helper, so the behavior is consistent.
- What happens if MPQ extraction produces 0-byte ADT files? The extraction script validates size and fourCC and re-extracts on failure. The inspect command validates `MapNameLength` and `Mh2oOffset` sanity.

## Requirements

### Functional Requirements

#### Core helper

- **FR-001**: System MUST provide a single static helper `McnkFlagDecoder` in a new file `wow-viewer/src/core/WowViewer.Core/Maps/McnkFlagDecoder.cs` (placed in `WowViewer.Core` rather than `WowViewer.Core.IO` because `WowViewer.Core` cannot reference `WowViewer.Core.IO` and the helper is pure data-logic with no I/O) that exposes `static AdtLiquidBasicType Decode(uint mcnkFlags)` and `static AdtLiquidBasicType DecodeWithMclqTileNibble(uint mcnkFlags, byte mclqTileNibble)`.
- **FR-002**: `McnkFlagDecoder.Decode` MUST use canonical bit checks in this precedence: `0x10=Magma` (first), `0x20=Slime`, `0x08=Ocean`, `0x04=River` (or `Water`), default `Water`. Magma-vs-Slime precedence: `0x10` (Magma) checked before `0x20` (Slime), matching `LiquidConverter.GetLiquidTypeFromMcnkFlags` at `wow-viewer/src/core/WowViewer.Core.IO/Liquids/LiquidConverter.cs:240-248`. For a `0x30` flag (both bits set, malformed data), Magma wins. **Note**: `AlphaLiquidTypeCodec.ResolveBasicType` uses the opposite order (Slime first); the canonical helper picks Magma-first because it matches the only function that returns `MclqLiquidType` and is used in the bidirection conversion path `LiquidConverter.MclqToMh2o`. Documented in helper XML doc.
- **FR-003**: `McnkFlagDecoder.DecodeWithMclqTileNibble` MUST check the MCLQ tile nibble first (per `AlphaLiquidTypeCodec.GetWriterTileTypeNibble`: `0x02=Ocean`, `0x03=Magma`, `0x04=Slime`, `0x01=Water`) and fall through to `McnkFlagDecoder.Decode(mcnkFlags)` if nibble is `0` or `0x0F` (DontRender).
- **FR-004**: `McnkFlagDecoder` MUST be a single source of truth — all 4 buggy functions in spec 040 §3 MUST call it and MUST NOT contain their own `(mcnkFlags>>4)&0x3` switches.

#### DBC LiquidType reader

- **FR-005**: System MUST provide `DbcLiquidTypeTable` in a new file `wow-viewer/src/core/WowViewer.Core.IO/Dbc/DbcLiquidTypeTable.cs` that exposes `static DbcLiquidTypeTable Load(string dbcPath)`, `static DbcLiquidTypeTable LoadFromBytes(byte[] data)`, and `AdtLiquidBasicType ResolveBasicType(ushort liquidTypeId)`.
- **FR-006**: `DbcLiquidTypeTable` MUST parse the 3.3.5 `LiquidType.dbc` schema: row count at offset `0x04` (uint), string-block at end, `ID` at offset `0x00` (int), `Name` at `0x04` (string ref), `Flags` at `0x08` (int), `Type` at `0x38` (uint) with values `1=Water`, `2=Magma`, `3=Slime`.
- **FR-007**: `DbcLiquidTypeTable.ResolveBasicType` MUST return `AdtLiquidBasicType.Water` for any LiquidTypeId that is not found in the table (matches 3.3.5 `param_1 = 1` fallback). It MUST log a single warning per unknown LiquidTypeId via the existing `ILogger` injection point.
- **FR-008**: `DbcLiquidTypeTable` MUST support loading from a JSON test fixture (format: `{ "rows": [ { "id": 17, "name": "Ocean", "type": 1, "flags": 0 }, ... ] }`) so unit tests are offline-replayable.

#### File fixes (the 5 buggy sites from spec 040 §3)

- **FR-009**: `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtLiquidReader.cs:275-284` `MapLiquidTypeId` MUST accept an optional `DbcLiquidTypeTable` parameter (default `null` for 1.x data). When the DBC is provided, it MUST call `DbcLiquidTypeTable.ResolveBasicType(liquidTypeId)`. When the DBC is `null`, it MUST fall back to the documented `WlToLiquidConverter.MapWlTypeToMh2oTypeId` mapping (`17→Ocean, 19→Magma, 20→Slime, else→Water`).
- **FR-010**: `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaToLkConverter.cs:547-558` `ResolveLiquidBasicType` MUST be replaced with a call to `McnkFlagDecoder.Decode(mcnkFlags)`. The function signature MAY stay the same for caller compatibility.
- **FR-011**: `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaTensorPackBuilder.cs:217-223` `McnkFlagsToLiquidType` MUST be replaced with a call to `McnkFlagDecoder.Decode(mcnkFlags)`. The function signature MUST change return type from raw `int` (current bug) to `AdtLiquidBasicType` (or the tensor pack's internal liquid-type index if it differs — investigate during planning).
- **FR-012**: `wow-viewer/src/core/WowViewer.Core/Maps/AlphaTileData.cs:243-255` `ClassifyLiquid` MUST be replaced with a call to `McnkFlagDecoder.Decode(mcnkFlags)`. The function signature MUST stay the same.
- **FR-013**: `wow-viewer/src/core/WowViewer.Core.IO/Maps/LkToAlphaConverter.cs:543-549` `ClassifyLkLiquid` MUST be replaced with a call to `McnkFlagDecoder.Decode(flags)`. The function signature MUST stay the same.

#### Round-trip integration fixes (discovered during T-05 empirical validation)

- **FR-018**: `wow-viewer/src/core/WowViewer.Core.IO/Maps/LkToAlphaConverter.cs:599` `BuildAlphaTileFlags` MUST set `tileFlags[(globalY * 8) + globalX] = AlphaLiquidTypeCodec.GetWriterTileTypeNibble(layer.BasicType)` for tiles that exist (passes the `TileExists` check). The original code set the byte to `0`, which strips the type information and forces consumers to fall back to the MCNK flag — lossy for round-trip scenarios where the WDT format normalizes MCNK flags to `0x3C`. The fix preserves the per-tile type information through the LK→Alpha→WDT→Alpha→LK round trip.
- **FR-019**: `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaToLkConverter.cs:520` `BuildLiquidData` MUST use `AlphaLiquidTypeCodec.ResolveBasicType(liquidChunk.TileFlags, liquidChunk.McnkFlags)` instead of `ResolveLiquidBasicType(liquidChunk.McnkFlags)` alone. The original code only consulted the MCNK flag, which is normalized to `0x3C` (all 4 bits set) by `AlphaWdtWriter.NormalizeAlphaLiquidFlags` (line 1038). The 0x3C normalization causes `McnkFlagDecoder.Decode(0x3C)` to return `Magma` (since 0x10 is checked first), so a WDT round-trip of an Ocean liquid layer returns Magma. Using the tile-nibble-aware resolver (which checks the per-tile nibble first) preserves the original type through the round trip.

#### Inspection command

#### Inspection command

- **FR-014**: `WowViewer.Tool.Inspect` MUST add a new subcommand `map inspect --adt <path> --dump-liquid-types [--dbc <path-to-LiquidType.dbc>] [--out <path-to-json>]` that emits a JSON file with per-tile: `mcnkFlags`, `mclqTileNibble`, `mh2oLiquidTypeId`, `dbcTypeField` (if DBC loaded), `resolvedBasicType` (string enum name). Default output: `<adt-basename>.liquid-types.json` next to the ADT.
- **FR-015**: The inspect subcommand MUST handle missing files gracefully (exit code `2` with a clear error message) and MUST NOT throw on malformed ADT (exit code `3` with a partial JSON report).

#### Empirical validation pipeline

- **FR-016**: A new PowerShell script `wow-viewer/scripts/Extract-StagedClientAdts.ps1` MUST extract at least one ADT per staged client (1.12, 2.4.3, 3.3.5) with known water/slime/magma tiles into `output/tmp/extracted-clients/<client>/<continent>/`. The script MUST use `wowmpq` (or `StormLib`-based equivalent) and MUST validate the extracted file's fourCC and size.
- **FR-017**: A new PowerShell script `wow-viewer/scripts/Validate-LiquidTypes.ps1` MUST run `WowViewer.Tool.Inspect map inspect --dump-liquid-types` against the extracted ADTs and diff the output against expected per-tile types (encoded as a sidecar `.expected.json`). Pass criteria: zero diffs.

### Key Entities

- **McnkFlagDecoder** (NEW): `wow-viewer/src/core/WowViewer.Core.IO/Maps/McnkFlagDecoder.cs`. Static helper, no state. Inputs: `uint mcnkFlags`, optional `byte mclqTileNibble`. Output: `AdtLiquidBasicType`.
- **DbcLiquidTypeTable** (NEW): `wow-viewer/src/core/WowViewer.Core.IO/Dbc/DbcLiquidTypeTable.cs`. Loads `LiquidType.dbc` (binary or JSON fixture). Inputs: `ushort liquidTypeId`. Output: `AdtLiquidBasicType` (with `Water` default on miss). Logs warnings on miss.
- **AdtLiquidBasicType** (UNCHANGED): enum in `wow-viewer/src/core/WowViewer.Core/Maps/AdtLiquidFile.cs:160-166`. The canonical liquid type enum (`Water=0, Ocean=1, Magma=2, Slime=3`).
- **AdtLiquidLayer** (UNCHANGED): record in `wow-viewer/src/core/WowViewer.Core/Maps/AdtLiquidFile.cs:51`. `LiquidTypeId` (ushort, DBC row index) and `BasicType` (the resolved type).
- **LiquidType.dbc record** (NEW visibility): per 3.3.5 schema, `Type` field at offset `0x38` is the actual type (1=Water, 2=Magma, 3=Slime). Parsed by `DbcLiquidTypeTable`.

## Success Criteria

### Measurable Outcomes

- **SC-001**: `McnkFlagDecoderTests` covers all 16 MCNK flag combinations (0x00..0xFF) and passes on first run. Each test asserts the expected `AdtLiquidBasicType` per spec 040 §3 canonical table.
- **SC-002**: `DbcLiquidTypeTableTests` parses a 3.3.5 `LiquidType.dbc` (or a JSON fixture) and resolves LiquidTypeId `17`, `19`, `20`, `1`, `13`, `14`, `9999` to the expected `AdtLiquidBasicType` (`Ocean`, `Magma`, `Slime`, `Water`, `River`/`Water`, `Water`/`River`, `Water` default).
- **SC-003**: Running `WowViewer.Tool.Inspect map inspect --dump-liquid-types` against a real 1.12 ADT (extracted from staged 1.12 MPQ) produces a JSON file where every tile's `resolvedBasicType` matches the canonical interpretation. Specifically, a tile with `mcnkFlags=0x20` and no MCLQ data shows `resolvedBasicType=Slime` (was: `Magma` before the fix).
- **SC-004**: Running `WowViewer.Tool.Inspect map inspect --dump-liquid-types` against a real 3.3.5 ADT with `LiquidTypeId=13` and a loaded DBC shows `resolvedBasicType` matching the DBC `type` field, not the hardcoded `else→Water` default.
- **SC-005**: `dotnet build I:\parp\parp-tools\wow-viewer\WowViewer.slnx -c Debug` succeeds. `dotnet test I:\parp\parp-tools\wow-viewer\WowViewer.slnx -c Debug` passes all liquid-related tests (existing + new). No new warnings introduced.
- **SC-006**: `Validate-LiquidTypes.ps1` produces a zero-diff report when run against the extracted 1.12/2.4.3/3.3.5 ADT set.
- **SC-007**: The pre-existing `LkToAlphaRoundTripTests.ConvertTile_ThroughAlphaWdt_BackToLkAdt_PreservesOceanLiquidTypeViaTileFlags` test PASSES after FR-018 + FR-019. This test was failing before this spec was written (test on line 899 expected `TileFlags[0] & 0x0F == 0x02` but got `0`; after the fix on line 906 it expected `BasicType == Ocean` but got `Magma` due to the WDT `0x3C` normalization + MCNK-only `ResolveLiquidBasicType` interaction). The fix lands both behaviors.

## Assumptions

- The wow-viewer project layout matches what's in spec 040's Files Inventory. If any file path has changed since spec 040 was written, the implementing slice must update the path.
- A `DbcReader` or `DbcFile` type already exists in `wow-viewer` (per the Shared I/O skill description: "DBC, DB2"). If not, the fix slice creates a minimal binary DBC parser scoped to `LiquidType.dbc`'s schema. **Verify during planning.**
- The `ILogger` injection point for `DbcLiquidTypeTable` warnings is the existing `ILogger<T>` pattern used elsewhere in `WowViewer.Core.IO`. If absent, use `Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance` as a fallback.
- The 3.3.5 `LiquidType.dbc` type-field offset is `0x38` per Ghidra `FUN_00439760`. This is the only client version where the offset is probed. If 1.12/2.4.3 use a different offset, the fix only requires the 3.3.5 offset (1.x data has no DBC-backed material lookup; 2.4.3 is a transition client — investigate during planning).
- The empirical validation pipeline uses PowerShell + `wowmpq` (or a `.NET` MPQ reader). If `wowmpq` is not installed, the script's `--skip-extract` mode loads pre-extracted ADTs from a known fixture path.
- The 1.12.1 and 2.4.3 staged clients have `World\Maps\<Continent>\<Map>.adt` files inside their `Data/common.MPQ` or `Data/expansion.MPQ` archives, per the standard WoW client layout.
- The `MclqLiquidType` enum (in `wow-viewer/src/core/WowViewer.Core.IO/Liquids/MclqChunk.cs:223-231`) and the `AlphaLiquidTypeCodec.GetWriterTileTypeNibble` (in `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaLiquidTypeCodec.cs:11-20`) are two different encodings; the fix preserves both and only introduces the canonical `McnkFlagDecoder` for MCNK flag interpretation. No enum refactor.

## Out of Scope

- Refactoring `MclqLiquidType` to share an enum with `AlphaLiquidTypeCodec.GetWriterTileTypeNibble`.
- Changes to `LiquidRenderer.cs:85` color palette (already correct per spec 040 §2.5).
- Changes to `AlphaWdtWriter.cs` (RULE 10).
- Changes to the 3.3.5 M2 reader family (`M2ModelReader.cs`, `M2SkinReader.cs`, `M2ToMdxConverter.cs`, `M2StaticRenderModelBuilder.cs`, `M2SkinnedRenderModelBuilder.cs`, `M2TrackSampler.cs`, `M2SkinProfileRuntime.cs`) per RULE 10 + spec 037.
- Probing the 1.12 or 2.4.3 binaries in Ghidra (the 3.3.5 evidence is sufficient).
- Adding WMO liquid type support (3.3.5 `FUN_00739e00` WMO lookup is a separate concern; deferred to a future spec).
- Adding animation/sound/material parameter resolution for `CMaterialWater`/`CMaterialMagma` (per spec 040 §1.6, 3.3.5 uses distinct shader names — the wow-viewer renderer is intentionally simple and uses color+opacity only).
- Multi-layer MH2O refactoring (already handled by `foreach (var layer in chunk.Layers)`).
- Spec 038's M2 culling/render-perf work.

## Open Questions

- **OQ-1**: Does the 1.12 `LiquidType.dbc` (if it exists) have the same `Type` field at offset `0x38`? Likely yes, but unprobed. The fix only uses the 3.3.5 DBC. If 1.12 has no DBC-backed material lookup, the priority order naturally degrades to MCLQ tile nibble → MCNK flag (the 1.x path).
- **OQ-2**: Should the 5 buggy functions be modified in-place, or should they be marked `[Obsolete]` and replaced with new helpers? Spec 041's FR-009..FR-013 specify in-place modification for `AlphaToLkConverter.ResolveLiquidBasicType`, `AlphaTensorPackBuilder.McnkFlagsToLiquidType`, `AlphaTileData.ClassifyLiquid`, `LkToAlphaConverter.ClassifyLkLiquid`, and `AdtLiquidReader.MapLiquidTypeId`. The in-place approach is the bite-sized choice.
- **OQ-3**: Does `AlphaTensorPackBuilder.McnkFlagsToLiquidType` need to return `AdtLiquidBasicType` directly, or does the tensor pack use a different internal liquid-type index (e.g. `0=Water, 1=Magma, 2=Slime, 3=Ocean`)? Investigate during planning. If different, wrap `McnkFlagDecoder.Decode` with a tensor-pack-specific mapping.
- **OQ-4**: Should `DbcLiquidTypeTable` be thread-safe (concurrent reads from multiple ADT files)? The current `LiquidConverter` is a static class; suggest `DbcLiquidTypeTable` be instance-based with one instance per DBC file (loaded once, queried many times).
- **OQ-5**: For 2.4.3 (TBC), does MH2O exist? Per wowdev.wiki, MH2O was introduced in WotLK (3.0+). 2.4.3 uses MCLQ. The fix's 1.x path handles 2.4.3 correctly. The 3.x path with DBC lookup applies to 3.0+ only. **Verify by extraction during planning.**
- **OQ-6**: What is the actual default of `WowViewer.Tool.Inspect`'s logger? Does it write to stdout, stderr, or a log file? If logging is needed for `DbcLiquidTypeTable` warnings, the inspect command must wire the logger correctly.
- **OQ-7**: Does the `Mh2oInstance` struct in `wow-viewer/src/core/WowViewer.Core.IO/Liquids/Mh2oChunk.cs` have a `LiquidTypeId` field, or is it parsed from offset 0 in `AdtLiquidReader.ParseLayer` line 153? Spec 040 §1 assumes the latter (parsed, not struct). **Verify during planning.**
- **OQ-8**: The `WlToLiquidConverter.MapWlTypeToMh2oTypeId` (line 286-295) maps WL types to DBC row IDs `13/14/17/19/20`. Is this the same set of IDs the actual 3.3.5 `LiquidType.dbc` uses? Spec 040 §4.3 says yes. If a staged 3.3.5 DBC has different row IDs, the fallback chain in `AdtLiquidReader.MapLiquidTypeId` (FR-009) must be updated. **Verify by parsing a real DBC during planning.**
- **OQ-9** (discovered during T-01): The grep for `ClassifyLiquid`/`ClassifyLiquidType` revealed 2 additional buggy functions NOT in spec 040's "Files to Fix" table: `AlphaWdtReader.cs:988` `ClassifyLiquid` and `AlphaTerrainAdapter.cs:276` `ClassifyLiquidType`. Both use the same broken `((mcnkFlags>>4)&3)` switch. Both MUST be added to the fix list. The spec 040 research did not enumerate these because the grep that produced spec 040 §3 was scoped to the file list in spec 040's Files Inventory; a broader project-wide grep (used during T-01) found the additional copies. **Action**: spec 041 "Files to Fix" table updated to include both.
- **OQ-10** (T-02 deferral): The spec 041 task list proposed refactoring `LiquidConverter.GetLiquidTypeFromMcnkFlags` (line 240-248) to call `McnkFlagDecoder.Decode`. This refactor is NOT a 1:1 replacement because `LiquidConverter` returns `MclqLiquidType` (which has `River=4` and `Magma=6` values that do not exist in `AdtLiquidBasicType`) and `McnkFlagDecoder.Decode` returns `AdtLiquidBasicType` (with `Water=0, Ocean=1, Magma=2, Slime=3`). A correct refactor would need a `DecodeToMclqLiquidType` adapter on the helper, which is out of scope for the 4-way drift fix. **Decision**: skip T-02 in the initial fix. `LiquidConverter.GetLiquidTypeFromMcnkFlags` is already correct and its output is consumed only by `LiquidConverter.MclqToMh2o` which handles the MclqLiquidType→AdtLiquidBasicType conversion downstream. Document the skip here.
- **OQ-11** (discovered during T-05 round-trip validation): Two additional pre-existing bugs surfaced when running the WDT round-trip test after T-05. The T-05 fix correctly maps raw MCNK flags to `AdtLiquidBasicType`, but the WDT round-trip test exposed that (a) `BuildAlphaTileFlags` was setting tile nibble to `0` instead of the type-specific value, and (b) `BuildLiquidData` was only consulting MCNK flags (which get normalized to `0x3C` by the WDT format). Both bugs were pre-existing — they had been masked by the broader 4-way MCNK drift because the test was already failing on the tile-nibble assertion before the helper existed. **Action**: FR-018 + FR-019 added to spec 041. Test now passes.

## Files Inventory (read for this spec, listed for cross-reference)

- Spec 040: `wow-viewer/specs/040-mh2o-mclq-liquid-type-determination/spec.md` (200 lines, 4 stories, 10 FRs, 4 SCs, 9 OQs)
- Spec 040 research: `wow-viewer/specs/040-mh2o-mclq-liquid-type-determination/research.md` (379 lines, full Ghidra decompilations of `FUN_00439760` and `FUN_0043a730`)

## Files to Touch (8 modified, 2 new, 1 test, 1 inspect subcommand, 2 scripts)

| File | Action | Spec 040 row |
|------|--------|--------------|
| `wow-viewer/src/core/WowViewer.Core/Maps/McnkFlagDecoder.cs` | **NEW** | spec 041 FR-001..FR-004 (placed in `WowViewer.Core` not `WowViewer.Core.IO` to honor the project dependency direction: `WowViewer.Core` cannot reference `WowViewer.Core.IO`; pure data-logic helpers live in `WowViewer.Core`) |
| `wow-viewer/src/core/WowViewer.Core.IO/Dbc/DbcLiquidTypeTable.cs` | **NEW** | spec 041 FR-005..FR-008 |
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtLiquidReader.cs:275-284` | MODIFY | FR-009 (spec 040 §3) |
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaToLkConverter.cs:547-558` | MODIFY | FR-010 (spec 040 §3) |
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaTensorPackBuilder.cs:217-223` | MODIFY | FR-011 (spec 040 §3) |
| `wow-viewer/src/core/WowViewer.Core/Maps/AlphaTileData.cs:243-255` | MODIFY | FR-012 (spec 040 §3) |
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/LkToAlphaConverter.cs:543-549` | MODIFY | FR-013 (spec 040 §3) |
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaWdtReader.cs:988-1000` | MODIFY | discovered during T-01 grep: identical `((mcnkFlags>>4)&3)` switch bug. Not in spec 040's table. |
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaTerrainAdapter.cs:276-287` | MODIFY | discovered during T-01 grep: identical `((mcnkFlags>>4)&3)` switch bug. Not in spec 040's table. |
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/LkToAlphaConverter.cs:599` | MODIFY | FR-018 (round-trip tile nibble fix, discovered during T-05 validation) |
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaToLkConverter.cs:520` | MODIFY | FR-019 (round-trip tile-nibble-aware resolver, discovered during T-05 validation) |
| `wow-viewer/tests/WowViewer.Core.Tests/Maps/McnkFlagDecoderTests.cs` | **NEW** | spec 041 SC-001 |
| `wow-viewer/tests/WowViewer.Core.Tests/Dbc/DbcLiquidTypeTableTests.cs` | **NEW** | spec 041 SC-002 |
| `wow-viewer/src/tools/WowViewer.Tool.Inspect/Commands/MapInspectCommand.cs` | EXTEND | FR-014..FR-015 |
| `wow-viewer/scripts/Extract-StagedClientAdts.ps1` | **NEW** | FR-016 |
| `wow-viewer/scripts/Validate-LiquidTypes.ps1` | **NEW** | FR-017 |

**Not touched** (per RULE 10 + spec 037):
- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaWdtWriter.cs`
- `wow-viewer/src/core/WowViewer.Core.IO/M2/M2ModelReader.cs`
- `wow-viewer/src/core/WowViewer.Core.IO/M2/M2SkinReader.cs`
- `wow-viewer/src/core/WowViewer.Core.IO/M2/M2ToMdxConverter.cs`
- `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2StaticRenderModelBuilder.cs`
- `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2SkinnedRenderModelBuilder.cs`
- `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2TrackSampler.cs`
- `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2SkinProfileRuntime.cs`

## Validation Targets

- **Staged 1.12**: `output/tmp/wowarchive-clients/1.X_Retail_Windows_enUS_1.12.1.5875/606/World of Warcraft/Data/` (MPQ only, requires `wowmpq` extraction). Expected: MCLQ ADTs, no MH2O, no DBC-backed material lookup.
- **Staged 2.4.3**: `output/tmp/wowarchive-clients/2.X_Retail_Windows_enUS_2.4.3.8606/606/World of Warcraft/Data/` (MPQ only, requires extraction). Expected: MCLQ-only per OQ-5 (verify during planning).
- **Staged 3.3.5**: `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft/Data/` (already extracted). Expected: MH2O-only, DBC-backed material lookup.
- **3.3.5 DBC fixture**: `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft/Data/misc.MPQ` (or `dbc.MPQ`) — contains `LiquidType.dbc`. Extract once into `output/tmp/extracted-clients/3.3.5/dbclua/LiquidType.dbc` for tests + inspect command.
- **Pre-fix baseline**: capture per-tile `BasicType` for 1.12/2.4.3/3.3.5 ADTs into `output/tmp/extracted-clients/<client>/<map>/<adt>.pre-fix.json` before the fix ships. This becomes the proof that the fix changed the correct tiles.
- **Post-fix expected**: capture the same ADTs into `<adt>.post-fix.json`. Diff tool (custom or `jq`) shows the tiles that flipped from `Magma` to `Water`/`Slime`.

## Suggested Bite-Sized Task Order (for `plan.md`)

1. **T-01**: Create `McnkFlagDecoder` helper with `Decode` and `DecodeWithMclqTileNibble` (no callers yet). Add `McnkFlagDecoderTests` (16 test cases, one per MCNK flag combination). Validate: tests pass, `dotnet build` clean.
2. **T-02**: Refactor `LiquidConverter.GetLiquidTypeFromMcnkFlags` (line 240-248) to call `McnkFlagDecoder.Decode`. Validate: existing tests pass, no behavior change (this function was already correct, just deduplicates).
3. **T-03**: Refactor `AlphaLiquidTypeCodec.ResolveBasicType` (line 32-56) to call `McnkFlagDecoder.Decode` for the MCNK branch (keep the MCLQ tile nibble branch). Validate: existing tests pass.
4. **T-04**: Refactor `LkToAlphaConverter.MapAlphaLiquidFlags` (line 573-582) to call `McnkFlagDecoder.Decode`. Validate: existing tests pass.
5. **T-05**: Fix the 4 buggy functions: `AlphaToLkConverter.ResolveLiquidBasicType`, `AlphaTensorPackBuilder.McnkFlagsToLiquidType`, `AlphaTileData.ClassifyLiquid`, `LkToAlphaConverter.ClassifyLkLiquid`. Each becomes a 1-line call to `McnkFlagDecoder.Decode`. Validate: each fix has a unit test.
6. **T-06**: Create `DbcLiquidTypeTable` with `Load(string)`, `LoadFromBytes(byte[])`, `LoadFromJson(string)`, and `ResolveBasicType(ushort)`. Add `DbcLiquidTypeTableTests` with 7 test cases (rows 17, 19, 20, 1, 13, 14, 9999).
7. **T-07**: Update `AdtLiquidReader.MapLiquidTypeId` to accept optional `DbcLiquidTypeTable` and consult it before falling back to the hardcoded mapping. Validate: existing tests pass.
8. **T-08**: Add `map inspect --dump-liquid-types` subcommand to `WowViewer.Tool.Inspect`. Run against a staged 3.3.5 ADT (no fix yet) to capture pre-fix JSON. Save to `output/tmp/extracted-clients/3.3.5/.../*.pre-fix.json`.
9. **T-09**: Write `Extract-StagedClientAdts.ps1`. Run against 1.12, 2.4.3, 3.3.5. Capture pre-fix JSON for all 3.
10. **T-10**: Write `Validate-LiquidTypes.ps1`. Define expected JSON for the extracted ADTs (manual sidecar authoring for 1-2 ADTs per client). Run. Fix any misclassified tiles (likely already fixed by T-05/T-07).

Each task is independently testable. Tasks 1-5 close the bit-mapping bug. Tasks 6-7 close the DBC gap. Tasks 8-10 close the empirical validation gap.
