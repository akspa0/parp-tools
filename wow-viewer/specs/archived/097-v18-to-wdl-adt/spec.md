# Feature Specification: V18 → WDL/ADT Round-Trip (Spec 097)

**Feature Branch**: `097-v18-to-wdl-adt`
**Created**: 2026-07-10
**Status**: Draft
**Owner**: wow-viewer
**Parent**: Spec 096 `096-v24-minimap-deploy` (the WDL prior + OBJ mesh pipeline)

**Input**: User description (collected, in order) — "we need a way to process a whole big set of files for a whole map - make sure to fix up the grid lines that may form across the tiles, as the tiles need to join up with the other ADT tiles adjacent to them, to form a proper single obj map object (and to bake a proper texture for the whole map, too. It would be great to be able to point at the zarr datastores for inputs, but get actual WDL files formatted properly as outputs, and maybe also some real ADT files, so we can take a look at what we've got, in our Viewer, as a round-trip test of all the tooling and models against a 3.3.5 client. use speckit."

---

## Problem Statement

Spec 096 ships a minimap-to-WDL-prior inference path and a single-tile OBJ exporter. The user can now run `v24_run_on_png.py some.png` and get a 257×257 mesh back, but the deployment story has three open problems for full-map use:

1. **Per-tile, not per-map.** The wrapper runs on one PNG at a time. A real map is 64×64 = 4,096 tiles. The user wants to point at a V18 Zarr store (per-map) and get the whole map back.
2. **No edge alignment.** Each tile's WDL prior is a 17×17 + 16×16 grid over a 256×256 minimap. Adjacent tiles share an edge in world space, but the prior grids are independent — without stitching, the seam between two tiles shows up as a visible height discontinuity in the stitched mesh. The mesh is broken at the tile boundaries.
3. **No real .wdl or .adt output.** The deployment only writes NPZ + OBJ. The viewer reads real `.wdl` and `.adt` files. A round-trip test (load the output in the WoWViewer app) requires writing proper WDL files at minimum, and ADT files for a true terrain round-trip.

This spec closes the gap end-to-end so the user can run one command, point it at a per-map V18 Zarr, and load the result in the viewer.

---

## What This Spec Does

Four bounded slices, each validated before the next:

1. **Slice 1 — Per-map V18 Zarr → single stitched OBJ + baked atlas with edge alignment.** New CLI `v24_export_map.py` reads a V18 Zarr store, applies the V24 prior (minimap-only path, augmented with V18 height in the cheat regime if available — see Slice 2 note), and writes one OBJ per map whose tiles join at the edges. Edge alignment is done by computing the prior on a per-tile basis and then snapping the 16-pixel border of each tile to the prior of the tile on the other side of the shared edge. A single atlas.png covers all tiles.
2. **Slice 2 — WDL file writer.** New Python module `harvester/v24/wdl_writer.py` (or a small C# shim if that's faster; the existing `WdlWriter.ExtractTileHeightsFromAlpha` is the template) writes one `.wdl` per map matching the layout the existing C# `WdlSummaryReader` expects (17×17 outer + 16×16 inner int16 per MARE, MAOF offsets, version-compatible header). This is the round-trip key: the viewer can now load the output.
3. **Slice 3 — ADT file writer (minimal).** New module `harvester/v24/adt_writer.py` writes one `_tex0.adt` per tile with the V18 minimap (downsampled to the ADT layer grid) and a minimal MCNK/MCAL/MCNR layout that the existing C# ADT reader can open. This is "minimal" because a full ADT file is a 20+ chunk format; we ship the chunks the viewer needs to display the minimap as terrain, with placeholders for the rest. The user explicitly said "maybe also some real ADT files" — this slice ships the minimum to make the round-trip a real one, not a fake.
4. **Slice 4 — Round-trip smoke + viewer proof.** A `v24_round_trip.py` script that runs the full Slice 1-3 pipeline on a known-good V18 map, then opens the resulting .wdl + .adt files via the existing C# `WdlSummaryReader` and `AdtReader` to confirm the round-trip is lossless. The user's WoWViewer app can then load the result and the user sees their own pre-alpha prior as a viewable surface.

---

## What This Spec Does NOT Do (Explicit Out of Scope)

- **No minimap-only training of any kind.** The prior path is whatever Spec 096 ships today: minimap-only (3-channel, ~190 world-unit L1) for tile export, or cheat regime (13-channel, ~1.2 world-unit L1) when a V18 height is available. Slice 1 picks the better of the two per tile automatically.
- **No full-fidelity ADT writing.** A complete ADT is ~20 chunks. We ship the 4-5 the viewer needs for a basic terrain display (MCNK + MCAL + MCNR + minimap layer). MCNK flags, liquid (MCLQ), doodads (MDDF/MODF), M2 references — all placeholders. A future spec (098) handles "real" ADT writing.
- **No MAHO.** The existing C# WDL reader does not expose MAHO (see Spec 094 amendment A1). The WDL writer matches the reader's contract.
- **No C# changes.** The new WDL/ADT writers are Python, calling the same C# shim pattern as Spec 094. We do not edit the existing C# WDL/ADT readers.
- **No new minimap-only training.** The minimap-only regime's 190-world-unit L1 is a known issue (Spec 095 is the path forward). This spec ships the deployment wiring, not a new model.
- **No multi-map batch.** One map at a time. Multi-map is a future spec.
- **No RunPod work.** Everything is local.

---

## User Scenarios & Testing

### User Story 1 — Stitch a whole map from a V18 Zarr (Priority: P1)

As a V24 owner, I can point a CLI at a per-map V18 Zarr store and get back a single OBJ mesh for the whole map, with all 4,096 tiles stitched and the seams aligned.

**Why this priority**: This is the actual deployment story the user is asking for. Slice 1 is the entry point.

**Acceptance Scenarios**:

1. **Given** a `3_3_5_12340` V18 Zarr store with at least one full map (e.g. `Kalimdor` 64×64 = 4,096 tiles), **When** `v24_export_map.py --v18-store <path> --map Kalimdor --output <dir>` runs, **Then** `<dir>/kalimdor.obj` exists, `<dir>/atlas.png` exists, and the OBJ references the atlas via per-vertex UV coordinates.
2. **Given** the same map export, **When** the OBJ is opened in any 3D viewer, **Then** the user sees a single 64×64-tile mesh whose surface is continuous across tile boundaries (no visible seams at the 256-pixel-tile borders). Edge-alignment may be a slight blur rather than a hard seam; the explicit constraint is "no visible hard step" at tile boundaries.
3. **Given** a map with a partial tile footprint (e.g. `Northrend` is 32×32), **When** the script runs, **Then** the script does not crash on missing tiles — missing tiles are skipped with a clear per-tile message and the rest of the map is exported.
4. **Given** a map where some tiles are missing height/minimap (audit-empty V18 tiles), **When** the script runs, **Then** audit-empty tiles are reported, the corresponding grid cells in the OBJ are filled with the per-map mean height, and the atlas covers them with the global mean colour.
5. **Given** the same V18 store and the same map, **When** the script is run twice with different `--seed` values, **Then** the output is bit-identical (SC-004 determinism contract).

### User Story 2 — Round-trip via real WDL files (Priority: P1)

As a V24 owner, I can run a script that takes the per-map V24 prior coverage and writes one `.wdl` per map. The existing C# WDL reader (already in the project, no changes) opens the file and returns the same MARE grid the Python pipeline produced.

**Why this priority**: This is the round-trip the user explicitly asked for. The viewer needs real `.wdl` files.

**Acceptance Scenarios**:

1. **Given** a per-map V24 prior grid (from Slice 1), **When** `v24_write_wdl.py --v18-store <path> --map <name> --output <dir>` runs, **Then** `<dir>/<map>.wdl` exists, has the same byte layout the existing C# `WdlSummaryReader` produces from a real game-client WDL, and the C# reader returns the same MARE grids that the Python pipeline wrote (within ±1 world unit of int16 rounding).
2. **Given** the same V24 prior, **When** the script runs on a `3_3_5_12340` map, **Then** the WDL is written in the LK format (header bytes `WDL5` or `WDL4` as appropriate, MAOF offsets, 17×17 outer + 16×16 inner int16 per MARE). The `wow-viewer` C# shim (`WowViewer.Tool.WdlRead read`) returns the same MARE data when run against the output.
3. **Given** the WDL is loaded by the existing C# reader, **When** the user opens the same map in the WoWViewer app, **Then** the minimap on the WDL tab matches the V18 minimap. (The viewer-side WDL display is a min-app feature; we are not adding new viewer code, just confirming the data round-trips.)

### User Story 3 — Minimal ADT writing (Priority: P2)

As a V24 owner, I can run a script that takes the per-tile V18 data and writes one `_tex0.adt` per tile that the existing C# ADT reader can open. The ADT carries the minimap as the first texture layer and a basic MCNK/MCAL/MCNR layout.

**Why this priority**: The user said "maybe also some real ADT files" — a full ADT is too much for this spec, but a minimal one closes the round-trip.

**Acceptance Scenarios**:

1. **Given** a per-tile V18 record (minimap, alpha, normal, height), **When** `v24_write_adt.py --v18-store <path> --map <name> --output <dir>` runs, **Then** `<dir>/<map>/<tile_x>_<tile_y>_tex0.adt` exists for every V18 tile in the map, the file opens with the existing C# ADT reader, and the minimap renders correctly.
2. **Given** a tile with audit-empty V18, **When** the script runs, **Then** the corresponding ADT is either skipped or written with a placeholder layout that the viewer can open without crashing. A clear per-tile message is recorded.
3. **Given** the same V18 data, **When** the script is run twice, **Then** the ADT files are bit-identical (determinism contract).

### User Story 4 — Round-trip smoke + viewer proof (Priority: P1)

As a V24 owner, I can run one command that takes a per-map V18 Zarr, runs the full pipeline (Slice 1 + Slice 2 + Slice 3), opens the resulting WDL files with the existing C# reader, and asserts the round-trip is lossless. The same command writes a `round_trip_report.json` with the per-tile diff.

**Why this priority**: The user wants to load the result in the viewer. The smoke is the proof the round-trip works before the user invests time in the viewer.

**Acceptance Scenarios**:

1. **Given** a `3_3_5_12340` V18 Zarr with a known map, **When** `v24_round_trip.py --v18-store <path> --map <name> --output <dir>` runs end-to-end, **Then** it writes: (a) the stitched OBJ, (b) per-map `.wdl`, (c) per-tile `_tex0.adt`, (d) `round_trip_report.json` with per-tile diff stats.
2. **Given** the round-trip report, **When** reviewed, **Then** for every tile the WDL MARE grids match the prior grids within ±1 world unit (int16 quantization bound), and the ADT minimap matches the V18 minimap within 1 unit per channel.
3. **Given** the round-trip is clean, **When** the user runs the WoWViewer app against the output directory, **Then** the map renders without errors and the user's pre-alpha prior is visible as a terrain surface.

### Edge Cases

- A map with a non-64-aligned footprint (e.g. 32×32, 16×16). Handled.
- A map with a missing or corrupt V18 Zarr. The script exits non-zero with a clear error.
- A map with a custom curated corpus (filtered tile list). The script accepts `--tile-ids <list>` and operates on that subset.
- A build other than 3_3_5_12340 (e.g. 0_5_3_3368, 4_0_0_11927). The WDL format changes per build; the script uses the build's known layout.
- A tile at the map edge (no neighbour on one side). Edge alignment skips that side; no crash.

---

## Functional Requirements

### Slice 1: Per-map V18 Zarr → single stitched OBJ + baked atlas

- **FR-101**: A new script `wow-viewer/data-harvester/scripts/v24_export_map.py` exists. CLI: `--v18-store <path> --map <name> [--build <build>] [--curation-manifest <parquet>] [--output <dir>] [--device <dev>] [--seed <n>]`. Defaults: `--build 3_3_5_12340`, `--output ./output/v24_maps/<map>`.
- **FR-102**: The script reads the V18 store's `index.parquet`, filters by `map == <name>`, applies the curation manifest if given, and iterates over the per-tile records.
- **FR-103**: For each tile, the script computes the WDL prior using the same code path as Spec 096 (minimap-only if no height available, cheat regime if V18 height is available), up-samples the (17,17) + (16,16) prior to 257×257, and stages the heightmap in a `(rows, cols, 257, 257)` tensor.
- **FR-104**: After all tiles are computed, the script applies edge alignment: for every shared 16-pixel border between two adjacent tiles, the script averages the two tiles' prior values along that border (a 16×257 vertical strip for east-west seams, a 257×16 horizontal strip for north-south seams). Corner cells (4-way) are the average of all four contributing tiles. The result is one continuous `(rows, cols, 257, 257)` heightmap with no visible hard step at the seams.
- **FR-105**: The script writes one OBJ with `(rows*257)×(cols*257)` vertices and one atlas.png covering all tile minimaps. Per-vertex texture coordinates index into the atlas.
- **FR-106**: A `preprocess_v24_adt.py` exists (or is inlined into the export script) that pre-stages the per-tile output of the inference call so the heavy inference runs once and the stitching / WDL / ADT phases are fast.

### Slice 2: WDL file writer

- **FR-201**: A new module `wow-viewer/data-harvester/src/harvester/v24/wdl_writer.py` exists. The function `write_wdl(prior_grids: dict[(tile_x, tile_y), (outer, inner)], output_path: Path, build: str)` writes a single `.wdl` per map.
- **FR-202**: The output matches the existing C# `WdlSummaryReader` byte-for-byte contract for the given build. For `3_3_5_12340`, the file starts with the `WDL5` header, then 64×64 MAOF offsets (or N×M for the actual map footprint), then per-MARE 17×17 outer + 16×16 inner int16 (the prior grids). The MAHO chunk is not emitted (matches Spec 094 amendment A1).
- **FR-203**: A test loads the produced `.wdl` via the existing C# shim and asserts the MARE grids match the prior grids within ±1 world unit.

### Slice 3: Minimal ADT writer

- **FR-301**: A new module `wow-viewer/data-harvester/src/harvester/v24/adt_writer.py` exists. The function `write_adt_minimal(record: TileRecord, output_path: Path, build: str)` writes a single `_tex0.adt` file per tile.
- **FR-302**: The output contains the minimum chunks for the viewer to open and display the minimap: MCNK (16×16 chunk grid, 1 chunk per tile), MCAL (alpha pack with 4 layers, the first layer carrying the V18 minimap), MCNR (vertex normals placeholder = (0, 0, 1) per vertex), and a MCSH stub. Other chunks (MCLY, MCSE, MCLQ, MDDF, MODF, M2 references) are placeholders.
- **FR-303**: A test opens the produced ADT with the existing C# reader and asserts the MCNK header is parseable and the MCAL first layer matches the V18 minimap.

### Slice 4: Round-trip smoke + viewer proof

- **FR-401**: A new script `wow-viewer/data-harvester/scripts/v24_round_trip.py` exists. CLI: same as `v24_export_map.py` plus `--report <json>`. Runs Slice 1 + Slice 2 + Slice 3, then opens the WDL files with the existing C# shim and asserts the round-trip.
- **FR-402**: The script writes `round_trip_report.json` with per-tile MARE diff stats and a final `all_pass` boolean.
- **FR-403**: A `docs/architecture/v24-round-trip-2026-07-XX.md` summary doc is written.

---

## Success Criteria

- **SC-097-001**: `v24_export_map.py` runs end-to-end on a real V18 map and produces a single stitched OBJ + baked atlas. The mesh is continuous at tile boundaries (no visible hard step at any 256-pixel seam).
- **SC-097-002**: The OBJ's per-vertex height values match the V18 heightmap at the per-tile grid points (16, 32, 48, ..., 240) within ±1 world unit (int16 quantization).
- **SC-097-003**: The output `.wdl` file is round-tripped through the existing C# `WdlSummaryReader` without errors. The MARE grids returned by the C# reader match the prior grids within ±1 world unit per cell.
- **SC-097-004**: The output ADT files open in the existing C# ADT reader without errors. The minimap renders correctly.
- **SC-097-005**: `v24_round_trip.py` exits 0 and reports `all_pass: true` on a known-good V18 map.
- **SC-097-006**: The user's WoWViewer app loads the output and the user can see their pre-alpha prior as a viewable terrain surface.
- **SC-097-007**: The pipeline runs locally on a 6 GB consumer GPU. A 64×64-tile map runs in well under 10 minutes.

---

## Key Entities

- **Per-map V18 Zarr store** (input): the existing `wow-viewer/output/datasets/v18/<build>.zarr` (Spec 001 / Spec 088 substrate). Per-map filtering via `index.parquet`.
- **Per-tile WDL prior** (intermediate): the (17,17) outer + (16,16) inner grid from the V24 prior store or from a fresh Stage A run.
- **Stitched per-map heightmap** (intermediate): the `(rows, cols, 257, 257)` tensor after Slice 1's edge alignment.
- **Per-map OBJ + atlas** (output): a single mesh covering the whole map, with one atlas.png holding the per-tile minimaps.
- **Per-map `.wdl`** (output): the file the viewer reads to know the WDL grid.
- **Per-tile `_tex0.adt`** (output): the file the viewer reads to know the per-tile terrain detail.
- **Round-trip report** (output): per-tile MARE diff stats and `all_pass` boolean.

---

## Risks

- **Risk 1 (high):** Edge alignment via averaging may produce visible smoothing at the seams if the two adjacent tiles' priors disagree by a lot (e.g. one is at the top of a hill, the neighbour at the bottom). A more sophisticated alignment (a low-pass on the seam, or a learned consistency term) is future work. Spec ships the averaging approach; the SC-001 "no visible hard step" is the gate.
- **Risk 2 (high):** WDL format details per build. The wowdev.wiki spec is the starting point, but the C# reader's actual byte layout is the source of truth. Slices 2 will read the existing C# reader's WDL writer (`WdlWriter.ExtractTileHeightsFromAlpha`) and match its output byte-for-byte. If the C# reader's WDL writer does not write a `.wdl` file (it only writes the inner heights), we will need a small extension to the C# shim — that is a separate bounded spec if it comes up.
- **Risk 3 (medium):** ADT format is complex. The Slice 3 "minimal" ADT is enough to display the minimap; the viewer may need additional chunks for other features. Out of scope for this spec.
- **Risk 4 (medium):** A 64×64 map is 4,096 tiles. Each tile's inference takes ~200 ms on a 12 GB GPU; the whole map takes ~14 minutes. SC-007 sets the budget at 10 minutes; if we miss it, the slice ships anyway with a clear "this is the wall-time you can expect" note. A future spec can use a learned cleaner (Spec 095) to cut inference time per tile.
- **Risk 5 (low):** Edge alignment with audit-empty tiles. The fallback (per-map mean height) is documented; the seams between real and audit-empty tiles may be visible. Acceptable for the first version.

---

## Assumptions

- The V18 store per build carries the `index.parquet` with the per-tile map, tile_x, tile_y, height, minimap, etc. Verified on `3_3_5_12340.zarr` in prior slices.
- The existing C# `WdlSummaryReader` reads WDL files and the C# `WdlWriter` extracts heights. The Python WDL writer in this spec does the reverse: takes a height grid and writes a WDL. The C# side is not modified.
- The existing C# ADT reader reads ADT files. The Python ADT writer in this spec writes a minimal ADT that the reader can open. The C# side is not modified.
- The user has access to a 3.3.5a staged client at `I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft` (per Spec 094 amendment A2).
- The WoWViewer app can be launched against a directory of `.wdl` + `_tex0.adt` files without code changes. Verified by the existing viewer's existing WDL/ADT display.
- Per-map footprint is 64×64 by default, but may be smaller (e.g. Northrend is 32×32). The script accepts arbitrary footprint sizes.

---

## Open Questions (For User Review Before Plan)

1. **Edge alignment algorithm.** Slice 1 averages the two tiles' values along the shared border. An alternative is a low-pass on a 32-pixel band (half the seam, half inside each tile). The simpler averaging is recommended for the first version; the low-pass is a future improvement.
2. **WDL writer location.** The Python WDL writer could live in Python (faster to iterate, no C# changes) or as an extension to the existing C# `WdlWriter` (matches the C# contract exactly). Recommended: Python first, with a small C# shim extension if the format details diverge. Slice 2 will start in Python and only add C# if the format is too subtle.
3. **ADT minimal chunks.** Slice 3 ships MCNK + MCAL + MCNR + MCSH stub. The user may need additional chunks (MCLY for liquid, MDDF/MODF for doodads). Recommended: ship minimal first, follow up with whatever the user actually needs to see in the viewer.

---

## End of Spec

This spec closes the deployment gap from "one minimap at a time" to "a whole map, loadable in the viewer." Each slice is bounded, validated, and either succeeds or has a documented honest failure mode. The full round-trip — V18 Zarr → WDL prior → stitched mesh + WDL + ADT → viewer — is the next major milestone for V24.
