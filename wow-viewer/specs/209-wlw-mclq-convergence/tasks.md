# Tasks: WLW / MCLQ Liquid Convergence

**Spec**: [spec.md](./spec.md)  
**Implementation Plan**: [NEXT-DAY-PLAN.md](../NEXT-DAY-PLAN.md)

Status key: `[ ]` not started · `[x]` done · `[~]` in progress · `[O]` operator-owned

---

## Phase 1 — Diagnosis & Measurement (US1)

- [x] **209-T1** Build `inspect adt liquid-convergence --client <dir> --map Azeroth`, reporting per tile: cells MCLQ-only / WL-only / both / neither, the height-difference distribution where both, partially-present MCLQ quad count (Mechanism A), and terrain-culled WL* cells (Mechanism B).
- [x] **209-T2** Run on the Wetlands coast (`H:\CLIENTS\Vanilla\0.x\0_5_3_3368\World of Warcraft`, `Azeroth`).
- [x] **209-T3** Identify root mechanism and verify union invariant (SC-002 / FR-004):
  - **Union verified**: 0 cells missing from unified array across all 500 liquid tiles.
  - **Mechanism A reach measured**: Quad edge interpolation validated.
  - **Mechanism B confirmed**: 585,108 WL* cells culled by terrain across Azeroth; **459,374 of those culled cells had no MCLQ coverage**, creating shoreline gaps (23,192 cells on `Azeroth_31_29` alone).
  - See [evidence/phase1-convergence-measured.md](./evidence/phase1-convergence-measured.md).

---

## Phase 2 — Shoreline Convergence Remediations (US2)

- [ ] **209-T4** Soften shoreline waterline culling in `WlLiquidRasterizer.KeepOnlyAboveTerrain`:
  - When a WL* cell is within coastal proximity or near terrain-water intersection, avoid hard-culling WL* liquid where no other liquid representation provides coverage.
- [ ] **209-T5** Unit tests verifying shoreline convergence preserves waterline connectivity without visual popping.
- [ ] **209-T6** Re-run `inspect adt liquid-convergence` and verify reduction in un-covered culled shoreline cells.
