# Implementation Plan: 046 — PM4 Asset Matching + ADT Restoration

**Branch**: `v0.5.0-dev` | **Date**: 2026-06-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/046-pm4-asset-matching/spec.md`

## Summary

Two-lane PM4 asset matching system:

1. **C# lane** (done): PM4 segment extraction, TypeFlags-based matching, LK ADT writing
2. **Python/Zarr lane** (done): signal store, scorer, placement synthesizer, validation

The end-to-end pipeline: PM4 file → segment extraction → placement matching → LK ADT output → viewer playback.

## ADT Restoration Pipeline (Phase 6 — Landed 2026-06-15)

### What it does

Takes PM4 files + _obj0.adt placement catalogs from a staged game client, matches PM4 collision surfaces to M2/WMO placements, and writes a new LK ADT file with those placements restored.

### Files

| File | Purpose |
|------|---------|
| `Core.PM4/Matching/Pm4AdtWriter.cs` | Converts match results to `LkAdtData` |
| `Core.IO/Maps/LkAdtWriter.cs` | Writes complete LK ADT binary |
| `Core/Maps/LkAdtData.cs` | Data model for LK ADT |
| `tools/inspect/Program.cs` | `pm4 write-adt` CLI command |

### CLI

```powershell
pm4 write-adt --input <file.pm4> --archive-root <client> [--placements <obj0.adt>] [--output <out.adt>] [--map-name <name>]
```

### Batch workflow (all tiles)

See `docs/PM4-ADT-RESTORATION.md` for the full guide.

```powershell
# For each PM4 file in a directory:
Get-ChildItem *.pm4 | ForEach-Object {
    $tile = $_.BaseName -replace '.*_(\d+)_(\d+)$', '$1_$2'
    dotnet run ... -- pm4 write-adt --input $_.FullName --archive-root <client> --output "output_$tile.adt"
}
```

## Python/Zarr Lane (Phases 1-5 — Complete)

### What exists

All Python code in `wow-viewer/data-harvester/src/harvester/pm4_asset_matching/`:

- `models.py` — dataclasses mirroring C# models
- `json_import.py` — import C# JSON exports
- `signal_store.py` — Zarr v3 read/write
- `scorer.py` — port of `Pm4AssetMatchScorer`
- `placement_synthesizer.py` — port of `Pm4ReplacementPlacementSynthesizer`

Scripts in `wow-viewer/data-harvester/scripts/`:

- `pm4_import_segment_signals.py`
- `pm4_import_asset_corpus.py`
- `pm4_validate_proposals.py`

Validation: 65/65 segments match C# scores to 0.005 tolerance. Proposal IDs match.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**: numpy, zarr (v3), numcodecs (blosc), scipy, pyarrow

**Storage**: Zarr v3 LocalStore for segment signals and asset reference signals; JSON for C# interop

**Testing**: pytest

**Target Platform**: Windows (dev), cross-platform Python

**Project Type**: Library (data-harvester/src/harvester/pm4_asset_matching/)

**Performance Goals**: Handle 10k+ segments and 1k+ asset references per store

**Constraints**: Must round-trip with C# JSON export format exactly; must not require staged client at match time

**Scale/Scope**: ~6 Python modules, ~4 test files

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| Repo Independence | ✅ | All code under `wow-viewer/data-harvester/` |
| Library-First | ✅ | Library in `harvester/pm4_asset_matching/`, scripts in `scripts/` |
| Real-Data Validation | ✅ | Will validate against dev tile JSON exports |
| Zarr v3 Storage | ✅ | Uses existing zarr_store.py patterns |
| No H:\CLIENTS | ✅ | Only reads JSON exports, no client paths |

## Project Structure

### Documentation (this feature)

```text
specs/046-pm4-asset-matching/
├── spec.md              # Feature specification
├── plan.md              # This file
└── tasks.md             # Task breakdown
```

### Source Code

```text
wow-viewer/data-harvester/src/harvester/pm4_asset_matching/
├── __init__.py                  # Public API
├── models.py                    # Python data models mirroring C# Pm4MatchingModels
├── signal_store.py              # Zarr signal store read/write
├── json_import.py               # Import C# JSON exports → Python models
├── scorer.py                    # Port of Pm4AssetMatchScorer logic
└── placement_synthesizer.py     # Port of Pm4ReplacementPlacementSynthesizer logic

wow-viewer/data-harvester/scripts/
├── pm4_import_segment_signals.py    # CLI: JSON → Zarr store
├── pm4_import_asset_corpus.py       # CLI: JSON → Zarr store
└── pm4_validate_proposals.py        # CLI: validate proposals against ground truth

wow-viewer/data-harvester/src/harvester/
└── test_pm4_asset_matching.py       # Tests
```

## Implementation Phases

### Phase 1: Data Models and JSON Import

**Goal**: Python data models that mirror the C# `Pm4MatchingModels` records, plus JSON import functions that deserialize C# export files into these models.

**Approach**: Define dataclasses for `Pm4SegmentSignalRecord`, `Pm4AssetReferenceSignalRecord`, `Pm4SegmentMatchResult`, `Pm4ReplacementPlacementProposal`. Write JSON import that handles the exact schema C# exports (camelCase keys, nested objects).

**Validation**: Round-trip test — import a C# JSON export, verify all fields match expected values.

### Phase 2: Zarr Signal Store

**Goal**: Read/write segment signals and asset reference signals to Zarr stores, keyed by segment ID or asset ID.

**Approach**: Extend existing `zarr_io.py` patterns. Each store contains: segment/asset ID array (string), bounds (float32 [N,2,3]), footprint hulls (variable-length), height stats (float32 [N,3]), surface family histograms (int32 sparse), topology stats (int32 [N,4]), anchor signals (float32 [N,8]). Store metadata as JSON attrs.

**Validation**: Write a store, read it back, compare all fields to original data.

### Phase 3: Python Scorer

**Goal**: Port the C# `Pm4AssetMatchScorer.ScoreSegment` logic to Python, producing identical scores for the same inputs.

**Approach**: Implement typed-overlap (35%) + type-profile (15%) + shape (50%) scoring. Key functions: `compute_bounds_overlap_ratio`, `score_ratio`, `score_distance`, `evaluate_typed_candidate`. Must handle TypeFlags profile matching and sub-part bounds.

**Validation**: Import C# match report JSON, re-score in Python, verify scores match to 0.001 tolerance.

### Phase 4: Placement Synthesizer

**Goal**: Port `Pm4ReplacementPlacementSynthesizer.Synthesize` to Python.

**Approach**: For each matched segment, build a placement proposal with provenance. Must produce identical proposal IDs for the same inputs (SHA256-based).

**Validation**: Import C# proposal JSON, re-synthesize in Python, verify proposal IDs match.

### Phase 5: Validation Script and Known-Tile Proof

**Goal**: End-to-end validation script that imports C# exports, runs Python scorer, and compares results against C# ground truth.

**Approach**: `pm4_validate_proposals.py` takes segment JSON + asset corpus JSON + C# match report JSON, runs Python scorer, reports pass/fail with diff details. Use dev tile (33_32) as primary validation target.

**Validation**: Script passes on dev tile data. All matched/ambiguous/unresolved statuses match C# output.

## Complexity Tracking

No constitution violations. All phases are straightforward library + script work within existing patterns.
