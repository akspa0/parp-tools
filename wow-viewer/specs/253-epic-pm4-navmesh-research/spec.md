# Epic 253 — PM4/PD4 Navmesh Decoding, Matching & Generation

**Created**: 2026-09-23 (spec reconciliation) · **Branch**: `v0.6.0-dev` · **Status**: Triage pending

> Primary successor of 10 archived specs (split specs also contribute items; see the ledger) plus the old `epic-pm4-restoration`. **No new
> scope** (§9.1). Evidence: [reconciliation ledger](../archived/reconciliation-2026-09-23/README.md),
> audit [B](../archived/reconciliation-2026-09-23/audit/batch-B.md); durable findings live in
> [workstream-pm4-decode.md](../../memory-bank/workstream-pm4-decode.md).

## Goal

Decode PM4/PD4 completely, name fields only by measurement, identify the source assets and
placements they encode, and eventually generate navmesh from source geometry.

## Delivered baseline (verified in code 2026-09-23 — do not re-plan)

| Capability | Where | Source |
|---|---|---|
| Segment builder, signal extractor, asset-match scorer, replacement-placement synthesizer; `pm4 match-assets/match-report/correlate-models`; Python mirror | `Core.PM4/Matching/`, `data-harvester/.../pm4_asset_matching/` | 046 |
| Surface-triangle correlation + collision fingerprints; `build-wmo-surface-db`, `extract-pm4-surfaces`, `match-surfaces`, `validate-matches`, `identify-models` | `Core.PM4/Services/` | 065 |
| PM4 object library (904 objects → 243 source assets), shape-only asset scoring (top-5 77.52% held-out), **`pm4 restore-placements`** writing verified `_obj0.adt` (73/73 round-trip; rotation not recovered) | `Pm4ObjectLibrarySupport`, `Pm4AssetScoringSupport`, inspect `pm4 object-library` / `restore-placements` | memory-bank workstream (delivered outside a spec; 065 Phase 9 intent) |
| Rosetta PM4 matching (`rosetta-pm4-match`, `rosetta-build-library`) | inspect | 190 |
| Field sweep instrument (`Pm4FieldSweepAnalyzer`, `pm4 field-sweep`) | `Core.PM4/Research/` | 188/189 |
| Measured field facts (MSVT placement space, MPRL permutation, CK24 = MODF Z float, MSLK surface adjacency, MSPV wall mesh, MSCN ordered chain, MPRR 4n+3 runs) | workstream note | 185/188/189 |
| MCSE/MCNK coordinate normalization; resident Zone/SubZone area overlay | `AreaOverlayRegion` | 149 US4 |

## Backlog (spec-stated residue — each item awaits operator triage in [TRIAGE.md](../TRIAGE.md))

| ID | Item | Source |
|---|---|---|
| P-01 | Remaining decode: grouping-rule harness, evidence register, canonical object-identity service, MSPV/MSCN discriminator, MPRR domain sweep | 130 Ph 2–9 |
| P-02 | Field map close-out: `MSLK._0x04` measured role; verify `AdtPm4MaskBuilder` space | 189 SC-003 |
| P-03 | Terminology restoration: rename names the data refutes (`AttributeMask` is a count, `_0x18` indexes MSLK); PM4 and PD4 docs | 185 FR-001–014; 188 US2 |
| P-04 | Live grouping-quality readout; mode retirement; doodad grouping | 188 US1–US3 |
| P-05 | Top-1 disambiguation (P@1 is 1.2%) + WMO root enumeration beyond ~502 roots; decide on unused merge code | 065 T034–T036 |
| P-06 | Placement **rotation** recovery (unevaluated: 0 of 4,000 dev pairs differ in rotation) | 065 Ph 9 |
| P-07 | Negative-BSP object matching (after P-01) | 128 |
| P-08 | Object-primary PM4 Zarr dataset (after P-01) | 129 |
| P-09 | Generate PM4/PD4 from source WMO/M2 geometry (PD4 first) | 184 |
| P-10 | PM4 region browser; remove the Correlation UI tab (64 refs); audio trigger enablement | 149 US1/US2 |
| P-11 | PM4 confirmed-match library in the viewer | 156 |

## Operator verification owed

149 SC-008 build + focused tests note; 188 SC-005 (readout distinguishes fields unprompted).
Publishing corrected docs to wowdev.wiki is an operator action.
