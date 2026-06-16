# Implementation Plan: 046 — PM4 Asset Matching

**Branch**: `v0.5.0-dev` | **Date**: 2026-06-16 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/046-pm4-asset-matching/spec.md`

## Summary

Two-lane PM4 asset matching system:

1. **C# lane** (done): PM4 segment extraction, TypeFlags-based matching, human-readable match reports
2. **Python/Zarr lane** (done): signal store, scorer, placement synthesizer, validation

The end-to-end pipeline: PM4 file → segment extraction → placement matching → markdown report output.

## Match Report (Phase 6 — Revised 2026-06-16)

### What it does

Takes PM4 files + _obj0.adt placement catalogs from a staged game client, matches PM4 collision surfaces to M2/WMO placements, and outputs a single human-readable markdown file per PM4 tile.

### CLI

```powershell
pm4 match-report --input <file.pm4> --archive-root <client> [--placements <obj0.adt>] [--max-matches <n>] [--search-range <units>] [--output <report.md>]
```

### Output

A markdown file containing:
- Tile metadata (coordinates, archive root, object counts, search range)
- PM4 object match summary table (CK24, type, part ID, surface count, footprint area, candidate counts)
- WMO placement table (uniqueID, model path, position, rotation, bounds, asset resolution, candidate count)
- M2 placement table (uniqueID, model path, position, rotation, scale, bounds, asset resolution, candidate count)
- Per-placement PM4 candidate detail tables (CK24, type, gaps, overlaps, distances)

### Removed: ADT Patching (2026-06-16)

`Pm4AdtWriter`, `Pm4BinaryAdtPatcher`, `Pm4AdtM2Placement`, `Pm4AdtWmoPlacement`, and the `pm4 write-adt` CLI command were removed. These produced corrupted ADT files by incorrectly patching placement chunks. The matcher now produces human-readable markdown reports — ADT writing is a separate concern that must not touch `LkAdtWriter`.

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

### Phase 1: Data Models and JSON Import ✅

### Phase 2: Zarr Signal Store ✅

### Phase 3: Python Scorer ✅

### Phase 4: Placement Synthesizer ✅

### Phase 5: Validation Script and Known-Tile Proof ✅

### Phase 6: Match Report CLI ✅ (revised from ADT writing)

Replaced `pm4 write-adt` (ADT binary patching, removed 2026-06-16) with `pm4 match-report` (human-readable markdown output).

## Complexity Tracking

No constitution violations. All phases are straightforward library + script work within existing patterns.