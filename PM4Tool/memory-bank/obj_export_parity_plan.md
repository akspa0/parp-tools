# PM4 OBJ Export Parity Plan

**Generated:** 2025-07-08 23:22-04:00

---

## Overall Objective
Restore full-parity PM4 → OBJ export in `WoWToolbox.Core.v2`, matching the legacy exporter in geometry, grouping and metadata (no materials), while writing all results to `project_output/{timestamp}`.

## Current Status
* MSVT vertices and orientation verified correct; geometry matches legacy output.
* Build is clean; per-chunk debug exporter operational.
* Memory-bank updated with MSVT-parity milestone.
* SHA comparison deemed unnecessary – focus is on functional parity.

## Task Checklist
| Task | Status |
| ---- | ------ |
| Review core memory-bank files & summarise context | ✅ |
| Fix all build errors (type mismatches, braces, etc.) | ✅ |
| Restore canonical transforms for all chunks | ✅ |
| Finalise MSVT X/Y mapping & validate | ✅ |
| Per-chunk OBJ debug export implemented & verified | ✅ |
| Remove material references / `.mtl` files | ✅ |
| Ensure MSLK **not** exported as geometry (debug-only) | ⏳ |
| Export faces for MSVT mesh where MSVI indices reference them | ⏳ |
| Include MSCN collision data as in legacy (main vs debug decision) | ⏳ |
| Add legacy `g` group statements & header comments | ⏳ |
| Update `pm4_obj_export_feature_matrix.md` as features complete | ⏳ |
| Extend `ObjExportCoverageTests` to assert new parity rules | ⏳ |
| Final validation with real PM4 samples & user sign-off | ⏳ |

## Immediate Next Steps
1. Visually / line-diff current OBJ vs legacy (ignore byte-hash).
2. Address high-priority gaps:
   * MSVI-driven faces on MSVT vertices.
   * Exclude MSLK from main geometry; keep in debug.
   * Add missing `g` group lines and header comments.
3. Update `LegacyObjExporter.cs` accordingly.
4. Expand `ObjExportCoverageTests` to cover these features using real PM4 files in `test_data/`.

---

## Feature Coverage Matrix (snapshot)

| # | Feature / Data Element | Legacy | Core.v2 | Parity |
|---|------------------------|--------|---------|--------|
| 1 | Vertex positions (`v`) | ✅ | ✅ | ✅ Complete |
| 2 | Triangle faces (`f`)   | ✅ | ✅ | ✅ Complete |
| 3 | `mtllib` line          | ✅ | ❌ | ❌ Missing |
| 4 | `usemtl` assignments   | ✅ | ❌ | ❌ Missing |
| 5 | `g` / `o` group names  | ✅ (`g`+`o`) | ✅ (`o` only) | ⚠️ Partial |
| 6 | Separate objects for position-data points | ✅ | ❌ | ❌ |
| 7 | Separate objects for command-data points  | ✅ | ❌ | ❌ |
| 8 | Bounding-box & dimension comments | ✅ | ❌ | ❌ |
| 9 | Terrain-coordinate comment header        | ✅ | ❌ | ❌ |
|10 | Consolidated multi-file exporter         | ✅ | ❌ | ❌ |

_(See `memory-bank/pm4_obj_export_feature_matrix.md` for the authoritative, always-updated version.)_
