# Feature Specification: MH2O / MCLQ Liquid Type Determination

**Feature Branch**: `040-mh2o-mclq-liquid-type-determination`
**Created**: 2026-06-02
**Status**: Research complete — consumed by spec 041 (liquid type fix)
**Input**: User description: "we do not handle liquids from MH2O chunks properly. I think we just aren't using the MCNK chunk flags in conjunction with the MCLQ or MH2O data properly, to determine what is meant to render from the data. the renderer just draws lava for everything in 1.x+ version data, for some reason, even after adding liquidType dbc lookups to try and resolve the issue."

## Status Notice

This is a **research slice**, following the precedent of `specs/038-m2-301-renderer-perf-research/`. It captures authoritative 3.3.5 Ghidra evidence, the wow-viewer code path that is producing the "lava for everything" symptom, the conflicting bit-mapping interpretations, and the priority order required for a future fix. **No code changes are made in this slice.** A follow-on implementation spec (e.g. `041-mh2o-mclq-liquid-type-determination-fix/`) will be created only after the user reviews the research and approves a fix direction.

## User Scenarios & Testing

### User Story 1 - Diagnose "lava for everything" in 1.x+ data (Priority: P1)

As a wow-viewer developer, I want a single, evidence-backed description of how liquid type is determined across MH2O, MCLQ, and DBC, so I can find the exact source of the "lava for everything" rendering bug and stop guessing.

**Why this priority**: Without a correct understanding of the data flow, every fix attempt is a coin flip. The current wow-viewer code has at least four functions that all read the same bit field and produce different interpretations. The 3.3.5 Ghidra evidence in this spec is the authoritative reference.

**Independent Test**: Open the spec and `research.md`, follow the data flow diagrams, and verify that each wow-viewer file referenced has been read and the bit mappings have been compared against the 3.3.5 binary.

**Acceptance Scenarios**:

1. **Given** a developer reading this spec, **When** they look up how a tile's `BasicType` is set, **Then** they can trace the path from MH2O payload → `AdtLiquidLayer.BasicType` → `LiquidRenderer` color in ≤5 minutes.
2. **Given** the MCNK flag bit assignments in the spec, **When** the developer compares them against the canonical wowdev.wiki MCLQ table, **Then** the wow-viewer interpretations and the spec's chosen interpretation agree.
3. **Given** the 3.3.5 `FUN_00439760` decompilation in `research.md`, **When** the developer reads the `param_1 = 1` fallback, **Then** they confirm 3.3.5 defaults to water on DBC miss, not magma.

---

### User Story 2 - Catalog the 4 conflicting wow-viewer interpretations (Priority: P1)

As a wow-viewer developer, I want a table showing every wow-viewer function that converts MCNK flags or LiquidTypeId to a liquid type, so the fix slice can target exactly the files that are wrong.

**Why this priority**: The same 2-bit MCNK field is interpreted in at least 4 different ways across `LiquidConverter`, `AlphaToLkConverter`, `AlphaTensorPackBuilder`, and `LkToAlphaConverter`. The fix must update all of them; missing one leaves the bug intact.

**Independent Test**: For each of the 4 functions listed in this spec, verify the file path, line range, and the bit interpretation matches what the function actually does.

**Acceptance Scenarios**:

1. **Given** the 4 function table in this spec, **When** the developer opens each file, **Then** the function body matches the table's "current interpretation" column.
2. **Given** the table's "canonical interpretation" column, **When** the developer compares to wowdev.wiki MCLQ tile flags, **Then** the canonical interpretation matches.
3. **Given** the table's "consistent with canonical?" column, **When** the developer runs the build, **Then** the fix slice knows exactly which functions need changes.

---

### User Story 3 - Establish priority order for liquid type resolution (Priority: P2)

As a wow-viewer developer, I want a documented priority order (MH2O LiquidTypeId DBC lookup → MCLQ tile flag → MCNK flag → default to water), so any future fix slice has a single decision tree to implement.

**Why this priority**: The 3.3.5 binary uses the DBC record's `type` field at offset `0x38` (per `FUN_00439760`), not the LiquidTypeId row index. The wow-viewer code currently shortcuts this by hardcoding 17/19/20. The fix must use the DBC record when present, and fall back through a documented chain.

**Independent Test**: For a 3.3.5 ADT with a non-17/19/20 LiquidTypeId (e.g. row 1, "Slime - Green" or row 13, "River - Dark"), the priority order produces the correct `BasicType`. For a 1.x ADT with MCLQ tile flag `0x04` (Slime nibble per AlphaLiquidTypeCodec) and MCNK flag `0x10`, the priority order produces Slime, not Magma.

**Acceptance Scenarios**:

1. **Given** a 3.3.5 ADT with MH2O LiquidTypeId=13, **When** the priority order runs, **Then** the resolved `BasicType` is the DBC record's `type` field value (mapped per 3.3.5 switch: 1=Water, 2=Magma, 3=Slime), not a hardcoded guess.
2. **Given** a 1.x ADT with MCLQ tile nibble `0x04` and MCNK flag `0x20`, **When** the priority order runs, **Then** the resolved `BasicType` is Slime (tile nibble wins over MCNK flag per AlphaLiquidTypeCodec intent).
3. **Given** an ADT with no DBC, no MCLQ, no MCNK flag set, **When** the priority order runs, **Then** the resolved `BasicType` is Water (3.3.5 default behavior).

---

### User Story 4 - Document the empirical validation path (Priority: P2)

As a wow-viewer developer, I want a scriptable validation path that uses real staged client data, so the future fix slice can prove it works on 1.x, 2.x, and 3.x data, not just unit tests.

**Why this priority**: The bug is observed in 1.x+ data. A correct fix must be validated against real MCLQ/MH2O data from `output/tmp/wowarchive-clients/1.X_Retail_Windows_enUS_1.12.1.5875/`, `2.X_Retail_Windows_enUS_2.4.3.8606/`, and `3_3_5_12340/`. Staged clients are inside MPQs and require extraction.

**Independent Test**: Run the validation path against at least one ADT per staged client version that contains water, slime, and magma tiles; verify the output JSON shows correct `BasicType` per tile.

**Acceptance Scenarios**:

1. **Given** the staged 1.12 client MPQ files, **When** an ADT is extracted, **When** `WowViewer.Tool.Inspect map inspect` runs on it, **Then** the output includes per-tile `BasicType` values that match the MCLQ tile flag nibble.
2. **Given** the staged 2.4.3 client MPQ files, **When** an ADT is extracted (and a WL file if present), **When** the same inspect runs, **Then** WL-derived tiles show the correct `BasicType` (River/Water for WL 13/14, Ocean for WL 17, Magma for WL 19, Slime for WL 20).
3. **Given** the staged 3.3.5 client (already extracted in `output/tmp/wowarchive-clients/3_3_5_12340/`), **When** an ADT with a non-default LiquidTypeId (e.g. 13, 14, 1) is inspected, **Then** the resolved `BasicType` matches the DBC record's `type` field, not a hardcoded lookup.

---

### Edge Cases

- What happens when both MCNK flag bit 0x10 (Magma) and bit 0x20 (Slime) are set? Canonical interpretation: it's invalid, but `AlphaLiquidTypeCodec` and `LiquidConverter.GetLiquidTypeFromMcnkFlags` both check Magma first, so Magma wins. The fix slice should document this precedence.
- What happens when an MH2O instance has LiquidTypeId=0? 3.3.5 still tries DBC lookup; if 0 is not a valid DBC row, fallback to `param_1 = 1` (water). The wow-viewer code currently has `0 → Water` via the default branch, which matches.
- What happens when MCLQ tile flags are all 0x0F (DontRender)? `AlphaLiquidTypeCodec.GetVisibleTileTypeNibble` returns 0 → falls through to MCNK flag check. Correct.
- What happens when 1.x ADT has both MCNK flag 0x20 (Slime) AND a 0x10 MCNK flag set on adjacent chunks? Each chunk's MCNK is independent; the per-tile MCLQ tile flag nibble is the per-tile source of truth.
- What happens when WL type is `FastWater` (5)? `MapWlTypeToMh2oTypeId` maps it to 13 (River). `MapWlTypeToMclqType` maps it to `River`. Consistent.
- What happens when `Mh2oInstance` is parsed with offset/length that put `LiquidTypeId` past payload end? `AdtLiquidReader.ParseLayer` line 135 guards with `offsetInstances + LayerSize <= payload.Length` before parsing. Safe.

## Requirements

### Functional Requirements

- **FR-001**: This spec MUST NOT modify any wow-viewer source file. It is research-only.
- **FR-002**: This spec MUST capture, in `research.md`, the decompiled 3.3.5 `FUN_00439760` (Material Bank liquid type lookup) and `FUN_0043a730` (Settings Bank liquid type lookup) functions in full, with the `param_1 = 1` fallback and the `*(uint *)(iVar4 + 0x38)` type field read.
- **FR-003**: This spec MUST enumerate every wow-viewer function that maps MCNK flags or MH2O LiquidTypeId to a liquid type, with file path, line range, current bit interpretation, and canonical interpretation.
- **FR-004**: This spec MUST establish a single source of truth for MCNK flag bit assignments, documented as `MclqFlagBitToBasicType` table in `research.md`.
- **FR-005**: This spec MUST establish the priority order: `MH2O LiquidTypeId → DBC type field at offset 0x38` > `MCLQ tile flag nibble` > `MCNK flag bits` > `default = Water`.
- **FR-006**: This spec MUST list the empirical validation path: which staged clients, which ADT files to extract, and which `WowViewer.Tool.Inspect` command to run, to prove a future fix works against real 1.x/2.x/3.x data.
- **FR-007**: This spec MUST list the 5 strings and 2 xref functions in 3.3.5 that anchor the liquid type lookup: `Material Bank: Liquid type [%d] not found, defaulting to water!` (0x00b6b918), `Settings Bank: Liquid type [%d] not found, defaulting to water!` (0x00b6b964), `WMO: Liquid type [%d] not found, defaulting to water!` (0x00b9b2b8), `Map Object Liquid` (0x00b9b2a3), `WCHUNKLIQUID` (0x00b96e21), and the callers `FUN_00439760`, `FUN_0043a730`.
- **FR-008**: This spec MUST document the four LiquidType/MH2O/MCLQ DBC ID conventions in use (3.3.5 row index 1, 2, 3 = Water/Magma/Slime; 1.x row index 13/14/17/19/20 = River/StillWater/Ocean/Magma/Slime per WlToLiquidConverter; MCLQ tile nibble 0x01/0x02/0x03/0x04/0x06 = Water/Ocean/Slime/?/Magma per `AlphaLiquidTypeCodec`; 3.3.5 DBC record offset 0x38 = type field).
- **FR-009**: This spec MUST defer any code changes to follow-on spec `041-mh2o-mclq-liquid-type-determination-fix/` and call out the file list that fix will touch.
- **FR-010**: This spec MUST NOT touch `AlphaWdtWriter.cs` per RULE 10 from the workspace AGENTS.md.

### Key Entities

- **AdtLiquidBasicType**: wow-viewer enum (Water=0, Ocean=1, Magma=2, Slime=3) defined at `wow-viewer/src/core/WowViewer.Core/Maps/AdtLiquidFile.cs:160-166`. The renderer's color pick key.
- **AdtLiquidLayer**: wow-viewer model with `LiquidTypeId` (ushort, the MH2O DBC row index) and `BasicType` (the rendered type) at `wow-viewer/src/core/WowViewer.Core/Maps/AdtLiquidFile.cs:51`.
- **MclqLiquidType**: wow-viewer enum (None=0, Ocean=1, Slime=3, River=4, Magma=6, DontRender=0x0F) defined at `wow-viewer/src/core/WowViewer.Core.IO/Liquids/MclqChunk.cs:223-231`. **NOTE**: The nibble values here do NOT match `AlphaLiquidTypeCodec.GetWriterTileTypeNibble` (which uses 0x01=Water, 0x02=Ocean, 0x03=Magma, 0x04=Slime per `wow-viewer/src/core/WWowViewer.Core.IO/Maps/AlphaLiquidTypeCodec.cs:11-20`). The two encodings conflict.
- **Mh2oInstance.LiquidTypeId**: ushort field in MH2O instance header, the LiquidType.dbc row index. Read correctly at `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtLiquidReader.cs:153`.
- **DBC record type field**: per 3.3.5 Ghidra `FUN_00439760`, at offset 0x38 in the LiquidType.dbc record, a uint with values 1=Water, 2=Magma, 3=Slime. The actual type.
- **MCNK flag bit assignments**: canonical per wowdev.wiki MCLQ tile flags and MCNK flags: 0x04=River, 0x08=Ocean, 0x10=Magma, 0x20=Slime.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A developer can read this spec in ≤15 minutes and produce a written fix proposal that correctly addresses the bit mapping conflict and the DBC lookup gap.
- **SC-002**: The follow-on fix spec (041) touches exactly the files in this spec's "files to fix" table and no others.
- **SC-003**: After the future fix lands, an ADT from staged 1.12/2.4.3/3.3.5 with mixed water/slime/magma tiles produces a `BasicType` distribution that matches the source MCNK flags / MCLQ tile nibbles / MH2O LiquidTypeId.
- **SC-004**: The fix does NOT modify any of: `AlphaWdtWriter.cs`, `M2ModelReader.cs`, `M2ToMdxConverter.cs`, `M2StaticRenderModelBuilder.cs`, `M2SkinnedRenderModelBuilder.cs`, `M2TrackSampler.cs`, `M2SkinProfileRuntime.cs` (per RULE 10 + spec 037 protected surfaces).

## Assumptions

- The user has staged 1.12 (`1.X_Retail_Windows_enUS_1.12.1.5875/`) and 2.4.3 (`2.X_Retail_Windows_enUS_2.4.3.8606/`) clients under `output/tmp/wowarchive-clients/` (verified 2026-06-02). The 1.12 client has no extracted ADT/WDT/WL files; both clients have MPQ files only. Extraction via an MPQ tool (e.g. `MPQEditor`, `wowmpq`, or `StormLib`-based) is out of scope for this research spec.
- The 3.3.5 client is fully staged at `output/tmp/wowarchive-clients/3_3_5_12340/` and its `WoW.exe` is currently loaded in Ghidra.
- The 3.3.5 Ghidra findings (`FUN_00439760`, `FUN_0043a730`, the `param_1 = 1` fallback) are the authoritative reference for liquid type determination. The 1.x binary (when probed) is expected to use a similar pattern but without DBC lookup at runtime (1.x had fewer DB rows and used MCNK flags + MCLQ tile nibble).
- The wowdev.wiki canonical MCNK flag bit assignment (0x04=River, 0x08=Ocean, 0x10=Magma, 0x20=Slime) is the ground truth for the MCNK flag interpretation.
- The `MclqLiquidType` enum values (Ocean=1, Slime=3, River=4, Magma=6) and the `AlphaLiquidTypeCodec.GetWriterTileTypeNibble` values (0x01=Water, 0x02=Ocean, 0x03=Magma, 0x04=Slime) are two different encodings used in different parts of the wow-viewer code. The fix must reconcile them or document which one is canonical.
- The 3.3.5 DBC record type field at offset 0x38 is a uint with values 1=Water, 2=Magma, 3=Slime. Other values (e.g. 0) may exist but the 3.3.5 code only switches on 1, 2, 3 and falls through to `iVar4 = 0` (no material assigned).

## Out of Scope

- Any code changes (per Status Notice above).
- MPQ extraction tooling for the 1.12 and 2.4.3 staged clients.
- Probing the 1.12 or 2.4.3 binaries in Ghidra to confirm the same bit-mapping convention (the 3.3.5 evidence is sufficient for this research slice).
- Implementation of the DBC lookup table or extension of the LiquidType.dbc reader in wow-viewer.
- Changes to the renderer color palette in `LiquidRenderer.cs:85` (the colors are correct, the upstream `BasicType` is wrong).
- Refactoring `MclqLiquidType` and `AlphaLiquidTypeCodec` to share a single enum.

## Open Questions

- **OQ-L1**: What value of `BasicType` actually arrives at the renderer for water chunks in 1.x+ data — is the type getting clobbered somewhere between `AlphaLiquidTypeCodec.ResolveBasicType` and `LiquidRenderer.BuildLayer`? (Needs empirical dump from `WowViewer.Tool.Inspect` on a real 1.12 ADT.)
- **OQ-L2**: Does the MH2O instance header `LiquidTypeId` field's byte offset match the 16-bit field at offset 0 in `Mh2oInstance`? Per wowdev.wiki MH2O v18: `Offset0=liquidTypeId, Offset2=liquidVertexFormat, Offset4=...`. `AdtLiquidReader.ParseLayer` line 153 reads `ushort` at offset 0. **Likely correct** but unverified against a real 3.3.5 ADT.
- **OQ-L3**: Does `MclqChunk.Parse` correctly populate per-tile `LiquidType` from the MCLQ tile flags? Line 58: `chunk.Tiles[i] = new MclqTile(data[offset++])`. The raw byte is stored; `MclqTile.LiquidType` returns lower 4 bits. **Likely correct.**
- **OQ-L4**: Should MCNK flag bit mapping be unified into a single `McnkFlagDecoder` helper shared by `LiquidConverter`, `AlphaToLkConverter`, `AlphaTensorPackBuilder`, `LkToAlphaConverter`, `AlphaLiquidTypeCodec`, and `AlphaTileData.ClassifyLiquid`? Strong yes — the current 4-way drift is the root cause.
- **OQ-L5**: For 3.3.5+ data, should MCNK flags be consulted at all for liquid type, or is `MH2O LiquidTypeId → DBC record type field at 0x38` sufficient? Per 3.3.5 Ghidra, MCNK flags are not consulted in the material lookup path; only the DBC record is. The fix should treat MCNK flags as a fallback only when MH2O is absent (1.x data).
- **OQ-L6**: What is the actual `vertex_format` enum mapping for Magma in MH2O? Per `WlToLiquidConverter` line 179, Magma (LiquidTypeId=19) uses `Mh2oVertexFormat.HeightUv`. 3.3.5 Ghidra shader names `vsLiquidMagma`/`psLiquidMagma` confirm magma uses UV-driven rendering. **Likely correct.**
- **OQ-L7**: Are there any `Mh2oInstance` layers in real 3.3.5 data where the same chunk has multiple layers (e.g. water on top of slime)? Per wowdev.wiki, yes — `LayerCount` can be > 1. The current `AdtLiquidReader` and `LiquidRenderer` handle this via `foreach (var layer in chunk.Layers)`. **Likely correct.**
- **OQ-L8**: How should the fix handle a 1.12 ADT where the MCNK flag is `0x30` (bits 0x10 and 0x20 set, which is Magma+Slime simultaneously)? Per `AlphaLiquidTypeCodec.ResolveBasicType`, Magma check runs first (line 49-50), so Magma wins. Document this precedence.
- **OQ-L9**: Does the 1.12 binary (when probed) have the same `FUN_00439760`/`FUN_0043a730` material-bank structure? Likely yes (WoW 1.x had a DBC-backed material system), but the type enum values may differ.

## Files Inventory (read for this spec)

| File | Lines | Role |
|------|-------|------|
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtLiquidReader.cs` | 285 | Reads MH2O payload, `MapLiquidTypeId` (line 275-284) hardcodes 17/19/20 → Ocean/Magma/Slime. **Does not consult DBC record type field.** |
| `wow-viewer/src/core/WowViewer.Core.IO/Liquids/Mh2oChunk.cs` | 298 | MH2O struct model. `Mh2oInstance.LiquidTypeId` is the DBC row index. |
| `wow-viewer/src/core/WowViewer.Core.IO/Liquids/MclqChunk.cs` | 280 | MCLQ struct model. `MclqLiquidType` enum (line 223-231) uses nibble values Ocean=1, Slime=3, River=4, Magma=6. |
| `wow-viewer/src/core/WowViewer.Core.IO/Liquids/LiquidConverter.cs` | 267 | Bidirectional MCLQ↔MH2O. `GetLiquidTypeFromMcnkFlags` (line 240-248) uses 0x10=Magma, 0x20=Slime — **correct**. |
| `wow-viewer/src/core/WowViewer.Core.IO/Liquids/WlToLiquidConverter.cs` | 361 | WL→MCLQ/MH2O for missing water planes. `MapWlTypeToMh2oTypeId` (line 286-295) maps WL types to DBC row IDs 13/14/17/19/20. |
| `wow-viewer/src/core/WowViewer.Core/Maps/AdtLiquidFile.cs` | 174 | `AdtLiquidBasicType` enum (line 160-166) and `AdtLiquidLayer` model. |
| `wow-viewer/src/core/WowViewer.Core/Maps/AlphaTileData.cs` | 360 | `AlphaLiquidChunk` record (line 23-31) and `ClassifyLiquid` (line 243-255) with broken `((mcnkFlags>>4)&0x3)` switch. |
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaLiquidTypeCodec.cs` | 72 | **Most correct** function: `ResolveBasicType` (line 32-56) reads MCLQ tile nibble first (0x02=Ocean, 0x03=Magma, 0x04=Slime), then MCNK flag (0x08=Ocean, 0x10=Magma, 0x20=Slime — **correct order**). |
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaToLkConverter.cs` | 676 | `ResolveLiquidBasicType` (line 547-558) uses `((mcnkFlags>>4)&0x3)==2 → Magma` (**WRONG** — 0x20 bit is Slime per canonical). `MapLiquidTypeId` (line 560-568) emits LiquidTypeId 17/19/20/0. |
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/LkToAlphaConverter.cs` | 788 | `ClassifyLkLiquid` (line 543-549) uses `(flags>>4)&3` switch 1=Water, 2=?, 3=?. `MapAlphaLiquidFlags` (line 573-582) is **correct** (0x08=Ocean, 0x10=Magma, 0x20=Slime). |
| `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaTensorPackBuilder.cs` | 743 | `McnkFlagsToLiquidType` (line 217-223) returns raw 2-bit field (0..3) as type mask — **not mapped to AdtLiquidBasicType**, propagates the bit-position error into the tensor pack. |
| `wow-viewer/src/core/WowViewer.Core.Renderer/Liquid/LiquidRenderer.cs` | 161 | `BuildLayer` line 85 color pick: Magma=`(0.9, 0.4, 0.05)`, Slime=`(0.2, 0.5, 0.1)`, else=`(0.1, 0.3, 0.6)` blue. Opacity: Water=0.45, others=0.7. **The renderer is correct; the upstream `BasicType` is wrong.** |

## Files to Fix (deferred to spec 041)

| File | Function | Current | Should be |
|------|----------|---------|-----------|
| `AdtLiquidReader.cs:275-284` | `MapLiquidTypeId` | Hardcodes 17/19/20 | Use DBC lookup with fallback to documented mapping |
| `AlphaToLkConverter.cs:547-558` | `ResolveLiquidBasicType` | `(mcnkFlags>>4)&3` switch with 0x20→Magma | Direct bit checks: 0x20=Slime, 0x10=Magma, 0x08=Ocean, 0x04=River |
| `AlphaTensorPackBuilder.cs:217-223` | `McnkFlagsToLiquidType` | Returns raw 2-bit field | Map to AdtLiquidBasicType using canonical bit order |
| `AlphaTileData.cs:243-255` | `ClassifyLiquid` | Same broken `(mcnkFlags>>4)&3` switch | Use canonical bit checks |
| `LkToAlphaConverter.cs:543-549` | `ClassifyLkLiquid` | Same broken `(flags>>4)&3` switch | Use canonical bit checks |

## 3.3.5 Ghidra Anchors (this session)

| Address | Length | What |
|---------|--------|------|
| `0x00b6b918` | 47 bytes | `"Material Bank: Liquid type [%d] not found, defaulting to water!"` |
| `0x00b6b964` | 49 bytes | `"Settings Bank: Liquid type [%d] not found, defaulting to water!"` |
| `0x00b9b2b8` | 46 bytes | `"WMO: Liquid type [%d] not found, defaulting to water!"` |
| `0x00b9b2a3` | 17 bytes | `"Map Object Liquid"` |
| `0x00b96e21` | 13 bytes | `"WCHUNKLIQUID"` (struct tag for liquid chunk pool, 0x444 bytes) |
| `0x00b96f4c` | 92 bytes | `".../WorldClient/MapChunkLiquid.cpp"` (source path) |
| `0x00439760` | function | `FUN_00439760` Material Bank lookup: DBC `param_1` lookup, miss → `param_1 = 1` (water), read type at `record + 0x38` |
| `0x0043a730` | function | `FUN_0043a730` Settings Bank lookup: same pattern, calls `FUN_0043a610` for DBC load |
| `0x006b6940` | function | Vertex/index buffer pool constructor (`MapChunkLiquid.cpp:0x89`) |
| `0x006b6de0` | function | Free-list deallocator (`MapChunkLiquid.cpp:0x7d`) |
| `0x006ad110` | function | CMapMgr::Initialize (allocates `WCHUNKLIQUID` pool of 0x40 × 0x444 bytes) |

The full decompilations of `FUN_00439760` and `FUN_0043a730` and the cross-reference table are in `research.md`.

## Validation Targets (deferred to spec 041)

- **Staged 1.12**: `output/tmp/wowarchive-clients/1.X_Retail_Windows_enUS_1.12.1.5875/606/World of Warcraft/Data/` (MPQ only, requires extraction). Expected data: MCLQ-format ADTs, no MH2O, no DBC-backed material bank.
- **Staged 2.4.3**: `output/tmp/wowarchive-clients/2.X_Retail_Windows_enUS_2.4.3.8606/606/World of Warcraft/Data/` (MPQ only, requires extraction). Expected data: MH2O introduced late in 2.x; mix of MCLQ and MH2O; WL data present.
- **Staged 3.3.5**: `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft/` (already extracted). Expected data: MH2O-only, DBC-backed material lookup.
- **Inspection command (proposed for 041)**: `WowViewer.Tool.Inspect map inspect --adt <path> --dump-liquid-types` to emit per-tile `BasicType` for empirical comparison against the source MCNK flags / MCLQ tile nibbles / MH2O LiquidTypeId.
