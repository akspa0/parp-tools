# Implementation Plan: MCAL Alpha Map Decode Correctness

**Branch**: `199-mcal-decode-correctness` | **Date**: 2026-09-01 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/199-mcal-decode-correctness/spec.md`

## Summary

Collapse four independent MCAL decoders into one canonical decoder in `WowViewer.Core.IO`,
drive its rule selection from era profile and format flags rather than span inference, make
every decode outcome reportable, and delete the fabrication that turns decode failures into
opaque chunk-sized blocks. Renderer and dataset harvest then read the same bytes.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: existing `AdtMcalDecodeProfile`, `AdtFormatProfile`,
`AdtMcalAlphaEncoding`, `AdtMcalSummary` — this feature fills them in rather than replacing them

**Storage**: N/A. Reports are in-memory; corpus aggregation writes a report file the operator
runs and reads.

**Testing**: xUnit in `WowViewer.Core.Tests`, no GL context required (FR-008); operator-run
corpus sweep and capture for SC-002/SC-004/SC-006

**Target Platform**: Windows desktop viewer plus offline harvest tooling

**Project Type**: Shared format library with viewer and harvester as consumers

**Performance Goals**: No regression in tile load time. Decoding is already per-tile work;
consolidation must not add a pass.

**Constraints**: 0.5.3 behaviour frozen (FR-007, SC-007); no fabricated data anywhere
(FR-002); must run headless (FR-008)

**Scale/Scope**: Four decode sites removed or delegated; corpus sweep spans 0.5.3, LK, Cata,
MoP builds from the configured client library.

## Constitution Check

*GATE: evaluated before Phase 0 and re-evaluated after design.*

| Principle | Status | Note |
|---|---|---|
| I. Repo Independence | **PASS** | All work inside `wow-viewer/`. |
| II. Library-First | **PASS — and this feature exists to restore it.** Four duplicate decoders across `Core.IO` and the viewer currently violate "one canonical owner per format surface". FR-001 removes the violation. |
| III. Real-Data Validation | **PASS** | SC-002/004/005/006 are all real-corpus or real-client gated. Unit tests pin behaviour; they do not constitute signoff. |
| IV. Evidence vs Architecture | **N/A** | No model training. Relevant only in spirit: a decode rule that cannot be justified must be reported as unexplained (FR-005), not hidden inside an aggregate that looks fine. |
| V. Streaming-First Dataset | **PASS** | Harvest keeps its streaming path; only the decoder behind it changes. |
| VI. No Client Path Assumptions | **PASS** | Corpus sweep takes a configured client root. |
| VII. Containers Are Inputs | **PASS** | Read-only. |

**Post-design re-check**: unchanged, and Principle II moves from violated to satisfied.

**No Complexity Tracking entries.**

## Project Structure

### Documentation (this feature)

```text
specs/199-mcal-decode-correctness/
├── plan.md
├── research.md          # four decoders, the fabrication, the open native question
├── data-model.md
├── quickstart.md
├── contracts/
│   └── alpha-decoder.md
├── checklists/
│   └── requirements.md
└── tasks.md
```

### Source Code

```text
wow-viewer/src/core/WowViewer.Core.IO/Maps/
├── AdtMcalDecoder.cs             # becomes THE decoder (FR-001)
├── AdtAlphaDecodeRule.cs         # rule selection from era profile + flags (FR-004)
└── AdtAlphaDecodeReport.cs       # layer/chunk/corpus outcomes (FR-003, FR-009)

wow-viewer/src/core/WowViewer.Core.IO/Lk/
└── Mcal.cs                       # delegates, or is deleted if it has no other role

wow-viewer/src/viewer/WoWViewer/Terrain/
├── StandardTerrainAdapter.cs     # DecodeLayerBySpan + fallback loop + synthesis REMOVED
└── Vlm/AlphaMapService.cs        # delegates or is deleted

wow-viewer/src/viewer/WoWViewer/Terrain/Vlm/
└── VlmDatasetExporter.cs         # stops hardcoding bigAlpha:false (FR-006)

wow-viewer/tools/inspect/WowViewer.Tool.Inspect/
└── AlphaDecodeSweepCommand.cs    # corpus sweep producing the report (SC-002, SC-006)

wow-viewer/tests/WowViewer.Core.Tests/
├── AdtAlphaDecodeRuleTests.cs
├── AdtMcalDecoderTests.cs
└── AdtAlphaDecodeReportTests.cs
```

**Structure Decision**: `AdtMcalDecoder` is promoted rather than a new type introduced — it
already lives in the right project and already takes an explicit `maxLength`, which is closer
to correct than the span-inferring implementations. The viewer keeps no decode logic at all.

## Phasing

| Phase | Delivers | Gate |
|---|---|---|
| **1. Report first, change nothing** | Decode outcome types; the current behaviour instrumented, still using today's rules. The sweep command runs and produces per-era counts. | Sweep runs on the real corpus; today's failure rate is a **number**, not an impression |
| **2. Native rule (T101)** | Isolate the client's alpha consumer and establish the per-era rule. If it cannot be isolated, record the negative result and proceed on file-side proof per research.md R5. | Evidence recorded either way |
| **3. Canonical decoder** | `AdtMcalDecoder` becomes the single owner; rule selection from era + flags; all other sites delegate. Fabrication still present. | SC-001; sweep failure rate equal or better than Phase 1 |
| **4. Delete the fabrication** | Synthesis and edge-stitch-over-synthesis removed. | SC-003, SC-004; 0.5.3 capture pixel-identical (SC-007) |
| **5. Harvest parity** | Harvest reads through the canonical decoder with real era inputs; enumerate corpus tiles whose alpha changes. | SC-005, US4 enumeration |

**Phase 1 before Phase 3 is deliberate.** Consolidating first would make the before/after
comparison impossible — there would be no baseline failure rate to improve on.

**Phase 4 after Phase 3 is deliberate.** The blocks are currently the only visible indicator
that decode fails; deleting them before the decode is trusted removes the signal and the
symptom together.

## Risks

- **The native decoder cannot be isolated.** Mitigated by R5's fallback: the rule is then
  backed by exact-payload-accounting across the corpus, with exceptions enumerated. Weaker
  evidence, explicitly labelled as such rather than quietly asserted.
- **Removing the fabrication makes terrain look worse before it looks better.** Expected and
  intended; Phase 4 is gated on Phase 3's failure rate first. The operator sees the honest
  state, and SC-006 turns the remainder into a list.
- **A previously harvested corpus becomes invalid.** Out of scope to rebuild; US4's
  enumeration tells the operator exactly which tiles changed so the call is theirs.
- **Regressing 0.5.3.** FR-007 plus SC-007's pixel comparison. 0.5.3 has no alpha at all, so
  the correct outcome is "untouched and not counted as failure" — the edge case list names it.
