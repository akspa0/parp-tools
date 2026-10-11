# Spec 270 — Tasks: PM4 $\leftrightarrow$ ADT Object Association Ledger & Unauthored Placement Pipeline

**Owner**: Epic 253 · **Status**: Completed (2026-10-10) · **Branch**: `v0.6.0-dev` · **Receipt**: [receipt-spec270.md](file:///I:/parp/parp-tools/wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/receipt-spec270.md)

---

## Phase 1 — Ledger Engine & 399-Tile Ground-Truth Extraction

- [x] **T01**: Implement `Pm4ObjectLedgerHarvester` (C# / CLI tool) harvesting paired tiles in `original_development/`.
- [x] **T02**: Implement multi-feature signature extractor (extents, vertex count, surface count, Z-key `MSUR._0x1C`, `CK24`, and `MSLK.GroupObjectId`).
- [x] **T03**: Run harvest across all paired tiles and serialize `output/datasets/pm4_object_association_ledger.json` (1.89 MB, 1,929 objects).
- [x] **T04**: Verify ledger statistics: record count, asset catalog coverage, and match fraction $\ge 90\%$ (904 matched WMOs, bit-exact Z, 95.1% M2 match).

---

## Phase 2 — Placement Resolver & Unauthored Tile Synthesis

- [x] **T05**: Implement `Pm4UnauthoredPlacementResolver` using ledger geometric distance scoring and adjacent spillover.
- [x] **T06**: Implement ADT placement builder `SplitAdtObjWriter` writing authentic `MVER`, `MMDX`, `MWMO`, `MDDF`, `MODF`, and `MCNK` chunks.
- [x] **T07**: Run placement reconstruction on unauthored development tiles (e.g. `development_16_36`, `14_36`, `13_50`, and full map batch of 130 tiles).
- [x] **T08**: Round-trip check: ensure generated `_obj0.adt` reads back cleanly with `AdtPlacementReader.Read()` (130 / 130 PASS).

---

## Phase 3 — Verification & Visual Outliner Smoke

- [x] **T09**: Write automated unit tests for ledger serialization, matching score, and placement round-trip (`Pm4UnauthoredPlacementResolverTests.cs` — 4/4 PASS).
- [ ] **T10**: Visual inspection in `WoWViewer` outliner to confirm 3D alignment between PM4 collision surfaces and restored geometry (Operator-owned via UI).
- [x] **T11**: Author completion receipt `evidence/receipt-spec270.md` with real outputs and verification results.
