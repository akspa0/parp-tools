# Spec 270 — Tasks: PM4 $\leftrightarrow$ ADT Object Association Ledger & Unauthored Placement Pipeline

**Owner**: Epic 253 · **Status**: Active (2026-10-10) · **Branch**: `v0.6.0-dev`

---

## Phase 1 — Ledger Engine & 399-Tile Ground-Truth Extraction

- [ ] **T01**: Implement `Pm4ObjectLedgerHarvester` (C# / CLI tool) harvesting paired tiles in `original_development/`.
- [ ] **T02**: Implement multi-feature signature extractor (extents, vertex count, surface count, Z-key `MSUR._0x1C`, `CK24`, and `MSLK.GroupObjectId`).
- [ ] **T03**: Run harvest across all 399 paired tiles and serialize `output/datasets/pm4_object_association_ledger.json`.
- [ ] **T04**: Verify ledger statistics: record count, asset catalog coverage, and match fraction $\ge 90\%$.

---

## Phase 2 — Placement Resolver & Unauthored Tile Synthesis

- [ ] **T05**: Implement `Pm4UnauthoredPlacementResolver` using ledger geometric distance scoring.
- [ ] **T06**: Implement ADT placement builder writing authentic `MMDX`, `MWMO`, `MDDF`, `MODF` chunks.
- [ ] **T07**: Run placement reconstruction on unauthored development tiles (e.g. `development_17_13`, `17_14`).
- [ ] **T08**: Round-trip check: ensure generated `_obj0.adt` reads back cleanly with `AdtPlacementReader.Read()`.

---

## Phase 3 — Verification & Visual Outliner Smoke

- [ ] **T09**: Write automated unit tests for ledger serialization, matching score, and placement round-trip.
- [ ] **T10**: Visual inspection in `WoWViewer` outliner to confirm 3D alignment between PM4 collision surfaces and restored geometry.
- [ ] **T11**: Author completion receipt `evidence/receipt-spec270.md` with real outputs and verification results.
