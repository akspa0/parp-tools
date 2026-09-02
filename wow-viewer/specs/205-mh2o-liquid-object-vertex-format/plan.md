# Implementation Plan: MH2O LiquidObject Vertex-Format Resolution

**Branch**: `205-mh2o-liquid-object-vertex-format` | **Date**: 2026-09-01 | **Spec**: [spec.md](spec.md)

## Summary

Resolve `liquid_object_or_lvf` values ≥ 42 through the `LiquidObject` → `LiquidType` →
`LiquidMaterial` DBC chain instead of casting them to a vertex-format enum that has no such case.
Read the river heightmaps that are currently discarded. Make an unresolved format a reported
condition rather than silent flat water. Consolidate the two decoders that both carry the defect.

Diagnosis is complete and measured before any code is written — see [research.md](research.md).
This plan is implementation only; there is no discovery phase left.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: existing `DbcLiquidTypeTable` and the DBC reading infrastructure in
`Core.IO/Dbc`; the corpus probe `inspect adt liquid-formats`

**Storage**: N/A

**Testing**: xUnit against fixture MH2O payloads for the decode logic; the corpus probe for
whole-client verification; operator flight for the visual seam check

**Target Platform**: Windows desktop, OpenGL

**Project Type**: Format-decoder fix inside an existing solution

**Performance Goals**: none — this is correctness. Decode cost is not on the hot path.

**Constraints**: pre-Cata behaviour unchanged (FR-003); ocean unchanged (FR-006); no heuristic as
the primary decode path (FR-008); missing DBCs degrade rather than crash (FR-009)

**Scale/Scope**: 17,461 measured layers on 80 ADTs; two decoders; three DBCs, two of which have no
reader yet.

## Constitution Check

| Principle | Status | Note |
|---|---|---|
| I. Repo Independence | **PASS** | all inside `wow-viewer/` |
| II. Library-First | **PASS — restores it.** Two independent MH2O decoders currently carry the same defect (research R5). Phase 4 gives the format one owner. |
| III. Real-Data Validation | **PASS** | every success criterion is measured against the operator's own client, and the diagnosis already was |
| IV. Evidence vs Architecture | **N/A** (no model training) — in spirit: FR-004 forbids the silent fallback that let a 100%-unhandled encoding look like working output |
| V. Streaming-First Dataset | **N/A** — though note harvested liquid data has been wrong for every MoP tile, which is a dataset-correctness issue as much as a rendering one |
| VI. No Client Path Assumptions | **PASS** | FR-009 handles absent DBCs |
| VII. Containers Are Inputs | **PASS** |

**No Complexity Tracking entries.**

## Project Structure

```text
specs/205-mh2o-liquid-object-vertex-format/
├── spec.md
├── research.md          # the histogram, the two decoders, why the probe is not the fix
├── plan.md              # this file
├── contracts/
│   └── liquid-vertex-format.md
├── checklists/
│   └── requirements.md
└── tasks.md
```

```text
wow-viewer/src/core/WowViewer.Core.IO/Dbc/
├── DbcLiquidTypeTable.cs        # gains MaterialID; currently reads only Type at 0x38
├── DbcLiquidObjectTable.cs      # NEW — id -> LiquidTypeID
└── DbcLiquidMaterialTable.cs    # NEW — id -> LVF

wow-viewer/src/core/WowViewer.Core.IO/Liquids/
└── Mh2oChunk.cs                 # RENDER PATH — used by StandardTerrainAdapter

wow-viewer/src/core/WowViewer.Core.IO/Maps/
└── AdtLiquidReader.cs           # HARVEST PATH — used by converter/dataset builders

wow-viewer/tools/inspect/WowViewer.Tool.Inspect/
└── AdtLiquidFormatSupport.cs    # the SC-001 verification command (already exists)

wow-viewer/tests/WowViewer.Core.Tests/
├── Mh2oLiquidObjectTests.cs
└── Mh2oDecoderParityTests.cs
```

**Structure Decision**: the DBC tables join the existing `Core.IO/Dbc` family rather than becoming
liquid-specific one-offs, so the resolution chain is reusable and testable without a client.

## Phasing

| Phase | Delivers | Gate |
|---|---|---|
| **1. Verify the chain** | Confirm `LiquidObject` and `LiquidMaterial` field offsets against the **MoP DBCs**, not the wiki (research R7). | The three ids 2325/2333/2372 and id 42 resolve to formats consistent with the measured vertex blocks |
| **2. Resolve and report** | The chain is wired into the **render-path** decoder. Unresolved formats are counted and logged (FR-004). | SC-001: zero unresolved layers on the MoP corpus |
| **3. Read the heights** | Height-bearing layers decode their per-vertex heights. | SC-002: decoded spreads match 11.90 / 70.15 / 163.25. SC-003: ocean unchanged |
| **4. One decoder** | Consolidate `Mh2oChunk` and `AdtLiquidReader` onto one implementation, or enforce their agreement by test. | SC-005 |
| **5. Verify** | Operator flight; era regression. | SC-004, SC-006 |

**Why Phase 1 is first and separate.** The chain is wiki-documented and unverified against this
client. If the offsets are wrong, everything built on them is wrong in a way that still *looks*
plausible — the exact failure mode that produced this defect. The gate is cheap: the corpus already
tells us what the answer must be. Id 42 must resolve to a depth-only format and 2325/2333/2372 must
resolve to height-bearing ones. **If the DBC chain disagrees with that, the chain is being read
wrong** — and finding that out in Phase 1 costs an hour instead of a week.

**Why the render path is fixed before the harvest path.** Research R5: `StandardTerrainAdapter`
calls `Mh2oChunk`, not `AdtLiquidReader`. Fixing the other one first would produce a change with no
visible effect, which is how the previous day was lost twice.

## Risks

- **The wiki chain is wrong for 5.0.1.** Phase 1 gate, with a corpus-derived expected answer to
  check against.
- **The DBCs are absent or differently named in this build.** FR-009 degrades to current behaviour
  with the degradation reported; the probe cross-check (research R6) then says how much is affected.
- **Sub-rectangle layers.** All 17,461 measured layers are 8×8, so `x_offset`/`y_offset`/`width`/
  `height` handling is untested by this corpus. Keep the existing handling and do not "simplify" it
  on the strength of a corpus that cannot see the case.
- **Regressing pre-Cata.** FR-003 plus SC-006; 0.5.3 and LK are verified in Phase 5 as their own
  gate rather than assumed to come along.
- **Fixing the decoder nobody renders with.** Named explicitly in research R5 and handled by phase
  ordering.
