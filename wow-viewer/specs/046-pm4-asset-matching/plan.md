# Implementation Plan: PM4 Asset Matching

**Branch**: `046-pm4-asset-matching` | **Date**: 2026-06-03 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/046-pm4-asset-matching/spec.md`

## Summary

Replace the freeze-prone PM4 object export workflow with a library-first automation lane that exports deterministic PM4 object segments, derives comparable signal corpora for PM4 segments and staged WMO/M2 assets, ranks candidate matches automatically, and emits replacement-placement proposals for missing development tiles. The design uses Zarr-backed signal stores, shared PM4 matching libraries, thin CLI/report surfaces, and bounded viewer review rather than manual matching as the primary workflow owner.

## Technical Context

**Language/Version**: C# / .NET 10 for PM4 libraries and CLI surfaces; Python 3.11+ / `uv` for corpus building and Zarr-backed signal tooling

**Primary Dependencies**: `WowViewer.Core.PM4`, `WowViewer.Core.IO`, `WowViewer.Tool.Inspect`, `wow-viewer/data-harvester` Python environment, `zarr`, `numpy`, `pyarrow`, staged-client asset access and existing capture/inspect helpers

**Storage**: Zarr v3 signal stores for PM4 segments and asset references; JSON and Parquet manifests/reports for match and placement outputs

**Testing**: `dotnet test` for PM4 library/inspect coverage, `dotnet build` for CLI/viewer integration points, `uv run` smoke commands for corpus tooling, and bounded real-data validation against staged clients and known development tiles

**Target Platform**: Windows desktop + CLI tooling in `wow-viewer`; offline corpus generation against staged client roots under `output/tmp/wowarchive-clients/`

**Project Type**: Shared library + CLI/report workflow + Python dataset tooling

**Performance Goals**:

- PM4 export must avoid blocking/freezing the primary viewer shell
- corpus-scale export and matching should stream through signal stores rather than loading the whole corpus eagerly into memory
- validation-scale matching should produce ranked candidate lists and placement proposals in one reproducible run

**Constraints**:

- `wow-viewer` is the only implementation owner
- no new PM4 or client-file parser stacks; reuse existing readers
- staged clients only; never use `H:\CLIENTS`
- first slice is automation-first, not an interactive manual matching workflow
- replacement placement output is proposal-grade first, not direct map mutation

**Scale/Scope**:

- development PM4 corpus scale: hundreds of tiles, thousands of PM4 object segments
- eligible asset families: WMO and M2 reference entries from staged clients
- first validation target: known tiles with existing placements plus bounded missing-tile replacement synthesis runs

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Repo Independence**: Pass. All planned code and tooling live under `wow-viewer/`.
- **Library-First**: Pass. Segmentation, scoring, and placement synthesis live in shared PM4/core surfaces first; tools and viewer review are thin hosts.
- **Real-Data Validation**: Pass. Validation is explicitly tied to staged clients and known PM4/development tiles.
- **Streaming-First / Zarr Policy**: Pass. Signal storage uses the repo’s Zarr-first data conventions instead of ad hoc file piles.
- **No Duplicate Readers**: Pass. Existing PM4 and client asset readers remain the only decode owners.
- **Bite-Sized Planning**: Pass. The implementation is split into export, corpus, matching, synthesis, and review slices.

## Project Structure

### Documentation (this feature)

```text
specs/046-pm4-asset-matching/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── pm4-asset-match-report.schema.json
├── checklists/
│   └── requirements.md
└── tasks.md
```

### Source Code (repository root)

```text
wow-viewer/src/core/WowViewer.Core.PM4/
└── Matching/
    ├── Pm4ObjectSegment.cs
    ├── Pm4ObjectSegmentBuilder.cs
    ├── Pm4SegmentSignalExtractor.cs
    ├── Pm4AssetMatchCandidate.cs
    ├── Pm4AssetMatchScorer.cs
    ├── Pm4ReplacementPlacementProposal.cs
    └── Pm4ReplacementPlacementSynthesizer.cs

wow-viewer/src/tools-shared/WowViewer.Tools.Shared/
└── Pm4Matching/
    ├── Pm4MatchRunOptions.cs
    └── Pm4MatchReportWriter.cs

wow-viewer/tools/inspect/WowViewer.Tool.Inspect/
└── Program.cs   # Add PM4 export/match/synthesis/report commands

wow-viewer/src/viewer/WoWViewer/
└── [bounded review wiring only if a viewer review surface is kept]

wow-viewer/data-harvester/src/harvester/
└── pm4_asset_matching/
    ├── asset_signal_corpus.py
    ├── pm4_signal_store.py
    └── match_validation.py

wow-viewer/data-harvester/scripts/
├── build_pm4_asset_signal_corpus.py
├── export_pm4_segment_signals.py
└── validate_pm4_asset_matching.py

wow-viewer/tests/WowViewer.Core.PM4.Tests/
├── Pm4ObjectSegmentBuilderTests.cs
├── Pm4SegmentSignalExtractorTests.cs
├── Pm4AssetMatchScorerTests.cs
└── Pm4ReplacementPlacementSynthesizerTests.cs
```

**Structure Decision**: Keep segmentation, scoring, and synthesis in `WowViewer.Core.PM4`, use `WowViewer.Tool.Inspect` as the primary automation host, and put Zarr corpus-generation logic in the single `data-harvester` Python environment. This preserves repo independence, library-first ownership, and the one-environment Python rule.

## Complexity Tracking

No constitution violations currently require justification.
