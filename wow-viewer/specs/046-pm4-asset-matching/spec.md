# Spec 046: PM4 Asset Matching

**Status**: Active — C# library + CLI + tests landed. Python/Zarr lane unstarted.

**Created**: 2026-06-03 | **Last updated**: 2026-06-14

## What Exists

All C# code in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/`:
- `Pm4ObjectSegmentBuilder` — deterministic segment builder (CK24+TypeFlags grouping, 4110→18 segments on dev tile)
- `Pm4SegmentSignalExtractor` — v2 signal contract (bounds, footprint hull, height stats, surface-family histogram)
- `Pm4AssetMatchScorer` — type-profile matching (typed overlap 35% + type profile 15% + shape 50%)
- `Pm4ReplacementPlacementSynthesizer` — proposal-grade placement synthesis with provenance
- `Pm4SegmentExportService` — aggregate segment exports from PM4 files
- `Pm4AssetSignalCorpusSupport` — durable asset corpus with seeded placement path

Inspect CLI commands:
- `pm4 export-segments`, `pm4 match-assets`, `pm4 export-asset-signals`, `pm4 synthesize-placements`

14 unit tests pass. 7 smoke proof JSONs in `output/tmp/`.

## What's Missing

- Python/Zarr signal-store lane (`data-harvester/src/harvester/pm4_asset_matching/`) — completely unimplemented
- Schema validation contracts
- Known-tile validation of proposal quality
- Viewer TypeFlags filter improvements
- Polish/doc updates

## History

Was split from 050 (WMO group matching) + 052 (signature matcher) → consolidated into 046 during 2026-06-09 spec consolidation pass. The C# side was completed 2026-06-08.
