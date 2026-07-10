# Tasks: 097-v18-to-wdl-adt

**Status**: Draft
**Created**: 2026-07-10
**Parent spec**: [`spec.md`](spec.md)
**Parent plan**: [`plan.md`](plan.md)

Each task is one focused, testable commit. Do not bundle. Do not skip validation.

---

## Slice 1 — Per-map stitched OBJ + baked atlas with edge alignment

- [ ] **T-097-001**: Write `wow-viewer/data-harvester/scripts/v24_export_map.py` per the Slice 1 design in `plan.md`. CLI: `--v18-store --map [--build] [--output] [--curation-manifest] [--device] [--seed]`. Default build = `3_3_5_12340`, default output = `./output/v24_maps/<map>`. ~150 lines.
- [ ] **T-097-002**: Write `wow-viewer/data-harvester/tests/v24/test_export_map.py`. 2 tests: (a) smoke with a small map (e.g. Northrend 32×32 or a 4×4 test fixture), (b) determinism across seeds.
- [ ] **T-097-003**: Run `v24_export_map.py` against the real V18 store for Northrend (3_3_5_12340). Confirm the OBJ + atlas land. Confirm the OBJ opens in any 3D viewer and tiles join at the seams.
- [ ] **T-097-004**: Validation gate for Slice 1: `tests/v24 -m v24 -q` passes (≥ 38 tests). Manual smoke output documented.

## Slice 2 — WDL file writer

- [ ] **T-097-005**: Write `wow-viewer/data-harvester/src/harvester/v24/wdl_writer.py`. Function `write_wdl(priors, output_path, build)`. ~80 lines. Matches the C# `WdlSummaryReader` byte layout for `3_3_5_12340` (WDL5 header + MAOF + 17×17 + 16×16 int16 per MARE). MAHO not emitted (per Spec 094 amendment A1).
- [ ] **T-097-006**: Write `wow-viewer/data-harvester/tests/v24/test_wdl_writer.py`. 2 tests: (a) write a 2×2 WDL with known MARE grids, parse it back in Python, assert byte-for-byte round-trip. (b) If the C# shim is built, round-trip through the C# reader and assert MARE grids match within ±1 world unit.
- [ ] **T-097-007**: Validation gate for Slice 2: tests pass. The C# round-trip test is the proof.

## Slice 3 — Minimal ADT writer

- [ ] **T-097-008**: Write `wow-viewer/data-harvester/src/harvester/v24/adt_writer.py`. Function `write_adt_minimal(record, output_path, build)`. ~150 lines. MCNK + MCAL (minimap in layer 0) + MCNR (placeholder normals) + MCSH (placeholder shadow). Other chunks omitted.
- [ ] **T-097-009**: Write `wow-viewer/data-harvester/tests/v24/test_adt_writer.py`. 2 tests: (a) write a small ADT, parse the chunks in Python, assert MCAL layer 0 matches the input minimap. (b) If the C# reader is available, load the ADT, assert no parse errors.
- [ ] **T-097-010**: Validation gate for Slice 3: tests pass.

## Slice 4 — Round-trip smoke + viewer proof

- [ ] **T-097-011**: Write `wow-viewer/data-harvester/scripts/v24_round_trip.py`. ~100 lines. Runs Slice 1 + Slice 2 + Slice 3. Then runs the C# `WdlSummaryReader` over the produced `.wdl` and emits `round_trip_report.json` with per-tile MARE diff stats and an `all_pass` boolean.
- [ ] **T-097-012**: Run `v24_round_trip.py --v18-store <3_3_5_12340> --map Northrend` end-to-end. Confirm exits 0 and `all_pass: true`.
- [ ] **T-097-013**: Write `wow-viewer/docs/architecture/v24-round-trip-2026-07-10.md` with the architecture, the reproduce commands, and the round-trip metrics.
- [ ] **T-097-014**: Update `wow-viewer/memory-bank/activeContext.md` and `progress.md` with the Spec 097 entry. Update the project README to announce the round-trip.
- [ ] **T-097-015**: Validation gate for Slice 4: `tests/v24 -m v24 -q` passes (≥ 42 tests). The user can run the WoWViewer app against the output and see the prior as a viewable surface (manual proof).

---

## End of Tasks

Each slice is bounded, single-commit, single-validation-gate. The full round-trip is the proof that the V24 deployment path is real and useful.
