# Implementation Plan: 097-v18-to-wdl-adt

**Status**: Draft
**Created**: 2026-07-10
**Parent spec**: [`spec.md`](spec.md)
**Parent lane**: Spec 096 (WDL prior + minimap-only deployment)

---

## Goal

Close the full-map deployment gap: point one command at a per-map V18 Zarr store, get back (1) a single stitched OBJ + atlas with edge alignment, (2) real `.wdl` files per map, (3) minimal `.adt` files per tile, and (4) a round-trip report that proves the data survives the C# reader re-read. The user opens the result in the WoWViewer app.

---

## Architecture Sketch

```
V18 Zarr (per map, via index.parquet)
  │
  │  harvester.v24.tiles.TileSource (Spec 094 substrate)
  ▼
Per-tile WDL prior (17,17) outer + (16,16) inner
  │
  │  upsample to 257x257 (Spec 094 lattice.upsample_prior_257)
  ▼
Per-tile 257x257 heightmap
  │
  │  Slice 1 edge alignment: average the 16-pixel border between adjacent tiles
  ▼
Stitched (rows*257, cols*257) heightmap
  │
  ├───► Slice 1:  v24_export_map.py writes
  │       (rows*257, cols*257) OBJ + atlas.png
  │
  ├───► Slice 2:  v24_write_wdl.py writes
  │       per-map .wdl (MAOF + 17x17 + 16x16 int16 per MARE)
  │
  └───► Slice 3:  v24_write_adt.py writes
          per-tile _tex0.adt (minimal MCNK + MCAL + MCNR + MCSH)

Slice 4:
  - run the C# WdlSummaryReader over the .wdl, assert MARE grids match ±1 world unit
  - run the C# ADT reader over the _tex0.adt, assert parseable
  - emit round_trip_report.json
```

The WDL and ADT writers are the only new C#-adjacent surface; everything else reuses Spec 096 / 094 code. The C# side is not modified.

---

## Slice Breakdown (RULE 8 — one phase at a time, validated before next)

### Slice 1 — Per-map stitched OBJ + baked atlas with edge alignment

**Touches**:
- New script: `wow-viewer/data-harvester/scripts/v24_export_map.py`. ~150 lines. CLI: `--v18-store --map [--build] [--output] [--curation-manifest] [--device] [--seed]`.
- New test: `wow-viewer/data-harvester/tests/v24/test_export_map.py`. 2-3 tests using the existing V18 store's Northrend map (32×32 = 1,024 tiles, fast to test).
- Optional: pre-stage script `v24_precompute_map_priors.py` to batch the per-tile Stage A calls and cache the priors as a Zarr.

**Algorithm**:
1. Read `index.parquet`, filter by `map == <name>`, build `(rows, cols)` of tile coordinates.
2. For each tile: load V18 record, compute the WDL prior (minimap-only if no height, cheat regime if V18 height exists — use the same code path as `train_v24_stage_a.py`).
3. Upsample each (17,17)+(16,16) prior to 257×257.
4. Build a `(rows, cols, 257, 257)` tensor of per-tile heights.
5. **Edge alignment**: for every (r, c) and its east neighbour (r, c+1), average the 16 rightmost columns of (r, c) with the 16 leftmost columns of (r, c+1). Same for north/south. Corners (4-way average). This produces a continuous heightmap.
6. Build a single OBJ + atlas covering all tiles. Atlas layout: tile minimaps placed in row-major order, indexed by per-vertex UVs.
7. OBJ vertices: `((r * 257 + y) * cols * 257 + (c * 257 + x))` with X = `(c * 257 + x) * tile_size`, Y = `(r * 257 + y) * tile_size`, Z = `heightmap[r, c, y, x]`. Default tile_size = 533.333 (one WoW tile).

**Validation gate**:
- `tests/v24 -m v24 -q` passes (39+ tests).
- Manual smoke: `v24_export_map.py --v18-store <3_3_5_12340> --map Northrend` runs to completion and produces a single OBJ + atlas.

**Honest failure mode**: If edge alignment doesn't fully eliminate seams (e.g. the two adjacent priors disagree by a lot), the slice ships anyway with the averaging approach. A future spec adds a low-pass or learned seam.

---

### Slice 2 — WDL file writer (Python)

**Touches**:
- New module: `wow-viewer/data-harvester/src/harvester/v24/wdl_writer.py`. ~80 lines. `write_wdl(priors, output_path, build)` writes a single `.wdl`.
- New test: `wow-viewer/data-harvester/tests/v24/test_wdl_writer.py`. 2 tests: (a) write a small WDL, (b) read it back via the existing C# shim and assert MARE grids match (the second test requires the C# shim to be built; skip if not).

**Algorithm** (LK 3.3.5a format):
1. Header: `WDL5` (4 bytes) + version (4 bytes LE = 5) + 0x00 padding to 12 bytes (per wowdev.wiki WDL_v18 + C# reader audit).
2. MAOF: 64×64 (or N×M) offsets, one per tile, 4 bytes LE each. MARE index 0 = top-left in (tile_x, tile_y) order.
3. Per-MARE payload: 17×17 outer int16 (289 × 2 = 578 bytes) + 16×16 inner int16 (256 × 2 = 512 bytes) = 1,090 bytes per MARE.
4. Layout matches the C# `WdlSummaryReader`'s output when run against the same input.

**Validation gate**:
- Test: write a 2×2 WDL with known MARE grids, read it back via a small in-Python parser, assert byte-for-byte round-trip.
- If the C# shim is built: smoke test via `WowViewer.Tool.WdlRead read --wdl <file>` and assert the MARE grids match the prior grids within ±1 world unit.

**Honest failure mode**: If the C# reader's WDL layout diverges from the wowdev.wiki spec, the test will catch it. The fix is to align the writer with whatever the reader expects. If the format is too subtle for Python to match exactly, the slice moves the writer into the C# shim (a small bounded extension). That is documented as a follow-up.

---

### Slice 3 — Minimal ADT writer (Python)

**Touches**:
- New module: `wow-viewer/data-harvester/src/harvester/v24/adt_writer.py`. ~150 lines. `write_adt_minimal(record, output_path, build)`.
- New test: `wow-viewer/data-harvester/tests/v24/test_adt_writer.py`. 2 tests: (a) write a small ADT, (b) parse the MCNK + MCAL + MCNR chunks in Python and assert the minimap is in MCAL layer 0.

**Algorithm** (LK 3.3.5a format):
1. MCNK (256×16-byte entries, one per 16×16 chunk sub-grid). For a 1-chunk-per-tile layout, one MCNK entry. Flags = 0x001 (MCSH present). Liquid level = 0. Offsets to other chunks filled in below.
2. MCAL: 4 layers × 16×16 uint8 alpha. Layer 0 = the V18 minimap downsampled to 16×16. Layers 1-3 = 0.
3. MCNR: 16×16 × 3 int8 vertex normals. Placeholder (0, 0, 127) = up.
4. MCSH: 16×16 uint8 shadow map. Placeholder (128 = neutral shadow).
5. Other chunks (MCLY, MCSE, MCLQ, MDDF, MODF, M2 references): omit. The viewer should fall back to defaults.
6. The output is the concatenation of all present chunks prefixed with their 4-byte FourCC + 4-byte size (per chunk header), per the ADT format the existing C# reader expects.

**Validation gate**:
- Test: write a 1-chunk ADT with a known minimap, parse the MCNK header + MCAL layer 0 in Python, assert it matches the input.
- If the C# reader is available: load the ADT, assert no parse errors. (The viewer-side rendering is not tested in this slice — that's Slice 4.)

**Honest failure mode**: If the minimal ADT layout doesn't open in the C# reader, the slice ships the in-Python round-trip test as proof the bytes are coherent, and the C# compatibility is recorded as a follow-up.

---

### Slice 4 — Round-trip smoke + viewer proof

**Touches**:
- New script: `wow-viewer/data-harvester/scripts/v24_round_trip.py`. ~100 lines. CLI: `--v18-store --map [--output]`. Runs Slice 1 + Slice 2 + Slice 3, then runs the C# `WdlSummaryReader` over the produced `.wdl` and asserts per-MARE match within ±1 world unit. Emits `round_trip_report.json`.
- New doc: `wow-viewer/docs/architecture/v24-round-trip-2026-07-XX.md`. Architecture summary + reproduce commands.
- Memory bank + progress.md update.

**Validation gate**:
- `v24_round_trip.py --v18-store <3_3_5_12340> --map Northrend` exits 0.
- `round_trip_report.json` has `all_pass: true` and per-tile MARE diff stats.
- The user can run the WoWViewer app against `<output>/wdl` + `<output>/adt` and see the prior as a viewable surface (manual proof).

---

## Risk Mitigations (Spec 097 risks re-stated with action)

| Risk | Severity | Mitigation in this plan |
| --- | --- | --- |
| Edge alignment smoothing visible at high-disagreement seams | High | Honest recording; the average-based alignment is the first cut. Future spec adds a low-pass or learned term. |
| WDL format details diverge from C# reader's contract | High | The first test is a byte-level round-trip in Python; if that fails, the format details are wrong. The fix is a small C# shim extension (bounded). |
| ADT minimal layout doesn't open in the C# reader | Medium | In-Python chunk-level round-trip test ships as proof of byte coherence. C# compatibility is a follow-up. |
| 64×64 map takes >10 minutes | Medium | Use the minimap-only model for fast inference; the V18 height is too slow. The 10-minute budget is the SC-007 gate. |
| Edge alignment with audit-empty tiles | Low | Fallback is per-map mean height; the seam is visible but documented. |

---

## Constitution Re-Check (post-plan, AGENTS.md rules)

- **RULE 1 (no edits to `gillijimproject_refactor`)**: ✓
- **RULE 2 (all new code in `wow-viewer`)**: ✓ all four slices land under `wow-viewer/data-harvester/`.
- **RULE 3 (no rewrite of game client reading tooling)**: ✓ the new writers are net-new; the existing C# WDL/ADT readers are unchanged.
- **RULE 4 (`wow-viewer` repo-independent)**: ✓
- **RULE 5 (one Python environment)**: ✓ all new code under `wow-viewer/data-harvester/`.
- **RULE 6 (no mutation of training scripts without a plan)**: ✓ the V24 trainer is unchanged.
- **RULE 7 (small modular residual-predicting models)**: ✓ no new models.
- **RULE 9 (no `H:\CLIENTS`)**: ✓ reads from `output/datasets/v18/...` and `output/tmp/wowarchive-clients/...` only.
- **RULE 10 (`AlphaWdtWriter` frozen)**: ✓ not touched.
- **RULE 11 (doc hygiene, plans bite-sized)**: ✓ 4 slices, each independently validatable.
- **RULE 8 (one phase at a time)**: ✓ each slice ends with a validation gate.

---

## Out of Scope (from the spec, re-asserted)

- Multi-map batch (one map at a time).
- Full-fidelity ADT (4-5 chunks ship; 20+ in the real format).
- RunPod work.
- New training.
- WDL/ADT format changes (we match the existing C# contract).
- C# changes (we wrap with the existing shim; only add C# if the format is too subtle for Python).
- Spec 095 (learned minimap cleaner) is a separate lane; not blocked by this spec.

---

## End of Plan

Each slice is small, testable, and ends with a concrete validation gate. The full round-trip is the next major milestone for V24 and the user is the proof owner.
