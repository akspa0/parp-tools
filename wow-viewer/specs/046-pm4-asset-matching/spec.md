# Spec 046: PM4 Asset Matching

**Status**: Active — C# library + CLI + tests landed. Python/Zarr lane implemented (Phases 1-4). Phase 5 (validation scripts) created, real-data validation pending.

**Created**: 2026-06-03 | **Last updated**: 2026-06-16

## What Exists

All C# code in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/`:
- `Pm4ObjectSegmentBuilder` — deterministic segment builder (CK24+TypeFlags grouping, 4110→18 segments on dev tile)
- `Pm4SegmentSignalExtractor` — v2 signal contract (bounds, footprint hull, height stats, surface-family histogram)
- `Pm4AssetMatchScorer` — type-profile matching (typed overlap 35% + type profile 15% + shape 50%)
- `Pm4ReplacementPlacementSynthesizer` — proposal-grade placement synthesis with provenance
- `Pm4SegmentExportService` — aggregate segment exports from PM4 files
- `Pm4AssetSignalCorpusSupport` — durable asset corpus with seeded placement path

Inspect CLI commands:
- `pm4 export-segments`, `pm4 match-assets`, `pm4 export-asset-signals`, `pm4 synthesize-placements`, `pm4 match-report`

14 unit tests pass. 7 smoke proof JSONs in `output/tmp/`.

## What's Missing

- Known-tile validation of proposal quality against real C# exports (T016)
- Viewer TypeFlags filter improvements
- Polish/doc updates

## Goal: PM4 → Human-Readable Match Report

The end-to-end pipeline is:
1. Read PM4 file → extract segments (CK24 grouping, TypeFlags classification)
2. Read _obj0.adt → get placement catalog (MDDF + MODF entries)
3. Match PM4 segments to placements (shape, footprint, TypeFlags scoring)
4. Output a markdown report listing every matched placement with PM4 candidate details

The `pm4 match-report` command produces a single markdown file per PM4 tile with:
- Tile metadata (coordinates, archive root, object counts)
- PM4 object match summary table
- WMO placement table with positions, rotations, bounds, asset resolution, and candidate count
- M2 placement table with positions, rotations, scale, bounds, asset resolution, and candidate count
- Per-placement PM4 candidate detail tables (CK24, type, overlap ratios, gaps, distances)

**ADT patching is explicitly out of scope.** The matcher provides data; writing that data into ADT files is a separate concern that must not touch `LkAdtWriter`.

## History

Was split from 050 (WMO group matching) + 052 (signature matcher) → consolidated into 046 during 2026-06-09 spec consolidation pass. The C# side was completed 2026-06-08.
ADT-writing code (`Pm4AdtWriter`, `Pm4BinaryAdtPatcher`, `write-adt` CLI command) was removed on 2026-06-16. It corrupted output ADTs by patching placement chunks incorrectly. The matcher now produces human-readable markdown reports instead.