# Spec 046: PM4 Asset Matching

**Status**: Active — C# library + CLI + tests landed. Python/Zarr lane implemented (Phases 1-4). Phase 5 (validation scripts) created, real-data validation pending.

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

- **ADT placement writing** — take PM4-matched M2/WMO placements and write them into LK ADT files (MDDF/MODF chunks) so the viewer can render the restored objects
- Known-tile validation of proposal quality against real C# exports (T016)
- Viewer TypeFlags filter improvements
- Polish/doc updates

## Goal: PM4 → ADT Restoration

The end-to-end pipeline is:
1. Read PM4 file → extract segments (CK24 grouping, TypeFlags classification)
2. Read _obj0.adt → get placement catalog (MDDF + MODF entries)
3. Match PM4 segments to placements (shape, footprint, TypeFlags scoring)
4. Write new ADT file with matched placements (MDDF for M2, MODF for WMO)
5. Open written ADT in viewer to verify restored objects

Existing writer: `LkAdtWriter` in `WowViewer.Core.IO/Maps/` — writes complete LK ADT with MVER, MHDR, MCIN, MTEX, MMDX, MMID, MWMO, MWID, MDDF, MODF, MCNK, MH2O, MFBO chunks.

User guide: `docs/PM4-ADT-RESTORATION.md`

## History

Was split from 050 (WMO group matching) + 052 (signature matcher) → consolidated into 046 during 2026-06-09 spec consolidation pass. The C# side was completed 2026-06-08.
