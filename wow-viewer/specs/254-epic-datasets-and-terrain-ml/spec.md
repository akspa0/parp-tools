# Epic 254 — Datasets, Client Datastore & Terrain ML

**Created**: 2026-09-23 (spec reconciliation) · **Branch**: `v0.6.0-dev` · **Status**: Triage pending

> Primary successor of 25 archived specs plus the old `epic-client-datastore` (split specs also contribute items; see the ledger). **No new
> scope** (§9.1). Evidence: [reconciliation ledger](../archived/reconciliation-2026-09-23/README.md),
> audits [A1](../archived/reconciliation-2026-09-23/audit/batch-A1.md) ·
> [A2](../archived/reconciliation-2026-09-23/audit/batch-A2.md) · [F1](../archived/reconciliation-2026-09-23/audit/batch-F1.md) ·
> [C2](../archived/reconciliation-2026-09-23/audit/batch-C2.md). Durable findings:
> [workstream-terrain-ml.md](../../memory-bank/workstream-terrain-ml.md) ·
> [weak-signal-tile-archaeology.md](../../memory-bank/weak-signal-tile-archaeology.md).

## Goal

Reproducible, provenance-carrying datasets from client data (one store, many builds, processed once),
and terrain reconstruction models that beat the tile-mean baseline on held-out maps.

## Delivered baseline (verified in code + on-disk artifacts 2026-09-23 — do not re-plan)

| Capability | Source |
|---|---|
| v50 clean-room dataset (4-map 0.5.3 corpus, strict + object-inclusive manifests) | 109 |
| Minimap lighting calibration + synthesized-minimap DXT1/MCCV options | 111 |
| Terrain-feature classifier + geometry retrain (road-region height error −21%) | 115 |
| Per-object occlusion-aware instance masks: C# emission (`TensorPack` builders, `mask-validate`) → Python segmenter + loss weighting | 118 |
| `terrain_shadow_256` end-to-end (C# → NPZ → v60 store → training) | 133 |
| Image-only WDL prior code; v50 height model dataset corrections; direct reconstruction geometry chain | 108, 112, 114 |
| v60 clean-signal lane (`v60/clean_signal_*`), with real CUDA receipts incl. negative results | 134, 139 |
| Brush-signature classification US1 | 132 |
| Weak-signal tile inventory / synthesize / composite / version-diff scripts | 127 (tooling) |

**Measured negatives (do not retry):** RGB→coarse WDL prior (123 closes 117's premise); minimap object
identity does not survive minimap scale (119/120); MCSH is not in minimaps; Depth-Anything family
blacklisted; VLM terrain analysis failed.

## Backlog (spec-stated residue — each item awaits operator triage in [TRIAGE.md](../TRIAGE.md))

### A. Client datastore (old epic 179–183 + 206)

| ID | Item | Source | Today |
|---|---|---|---|
| D-01 | Canonical MPQ patch-chain resolver (landing gate for the rest) | 179 | `MpqArchiveCatalog` + `NativeMpqService` still both present |
| D-02 | Multi-build content-addressed datastore | 180 | absent |
| D-03 | Incremental processing (compare `InputSha256`, reuse derivations) | 181 | hash computed, never compared |
| D-04 | Adaptive per-type encoding (one codec policy; zstd-5 vs lz4-1 defaults coexist today) | 182 | absent |
| D-05 | Viewer: load a Zarr datastore | 183 | Rosetta Zarr v3 maps already load (`RosettaDatastoreTerrainAdapter`, `ViewerApp.cs` ~13066); the general multi-build datastore load is absent |
| D-06 | Zarr-first asset residency | 206 | absent |

### B. Terrain models

| ID | Item | Source |
|---|---|---|
| D-10 | Ground-up v50 height model: real WDL prior + residual detailer | 123 |
| D-11 | Clean-signal lane: fix the cross-tile family regression, then real albedo-normalized transfer promotion (SC-004) | 139 |
| D-12 | Later-client adapters for the v60 lane | 134 US6 |
| D-13 | Texture-family selection + alpha-stack reconstruction | 114 Ph 6/7 |
| D-14 | Research-lead ledger | 141 US4 |

### C. Archaeology tooling

| ID | Item | Source |
|---|---|---|
| D-20 | Weak-tile explorer UI: inspection views, tile listing, neighbour auto-amplify wiring | 127 US1–US3 |
| D-21 | Brush/motif archaeology (132 + 140 merged): nested tiers, brush-scar correlation, cross-map fragment alignment, rescale boundaries, paste retrieval, tileset separation, paint-order inference, predictive model | 132 US2–US6; 140 |

### D. Infrastructure

| ID | Item | Source |
|---|---|---|
| D-30 | Legacy Python lane detangle + C# RunPod tooling for v50 | 124 |

### E. Parked

| ID | Item | Source |
|---|---|---|
| D-90 | Minimap super-resolution; restoration hallucination gate; residual-extractor confidence | 125 FR-021, FR-010/FR-012 |
| D-91 | Minimap texture-tier decode; multi-tile seam stitching | 126 US5/US6/US8 |

## Operator-run training / data gates owed (user-owned runs only)

108 T012/T013/T015 mixed-corpus training; 111 T009 bucketing + T019 training; 118 real paired
comparisons (US2/US3); 139 SC-004 held-out-family promotion; 109 T050/T051 doc verification.
