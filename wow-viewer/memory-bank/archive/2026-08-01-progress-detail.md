# Progress archive — pre-2026-08-01 (condensed out of progress.md)

This file preserves the detailed per-spec progress log that was condensed out of the live
`memory-bank/progress.md` on 2026-08-01 (feature-complete declaration + spec audit). Read it
for forensic detail only. The live files carry the current state.

## Spec 122 — Dataset curation (shipped 2026-07-30)

Full speckit docs + implementation same session. `WowViewer.Core.Curation` (repo's first Parquet
writer) + `WowViewer.Tool.Harvest curate`: buckets (difficulty/coverage/lighting) + findings
(height-normal mismatch, non-finite, has-flag, synthetic-fidelity gap) per tile. Partitioned,
never filtered. Real-data validated twice on PVPZone02 (64/64 tiles); 22/25 fidelity-evaluated
tiles flagged high-severity gap. 39/39 C# tests, 9/9 curation-store Python tests, full suite
1154 passed / 45 skipped / 3 pre-existing failures. US4 corrected mid-session: 14+3 live legacy
callers kept as documentation-only pointers (5th script `spec103_curate_dataset.py` found, the
real producer of on-disk curation).

## Spec 121 — WDL-prior height reconstruction (CLOSED 2026-07-25)

RGB→WDL prediction is fundamentally wrong for this project. Three architectures (LatticeNet
v2/v5, MitB0LatticeNet) hit the same wall: zone-local mapping that doesn't transfer cross-region
(val −73% vs tile-mean). Detailer survived (5% improvement). Salvageable: DetailerMitB0Net,
within-map split, object-mask tile loss. Next (when ready): merged WDL prior → detailer refines,
no RGB→WDL model.

## Specs 119/120 — Object-library classifier/segmenter + retrieval (ARCHIVED 2026-07-24)

Minimap object identity is a measured dead end: retrieval PoC p50=10px instances, ~0.99 cosine
to unrelated blobs. Classifier 0.9137 (majority 0.8562), segmenter IoU 0.9921. Precise masks
repurposed loss-side in Spec 121. Full detail preserved in archived spec dirs.

## Spec 118 — Object occlusion masks (US1–US3 implemented)

Strict object-geometry target already existed (`TerrainVisibleObjectMaskRasterizer`); only new
C# was `object_geometry_visible_instance_257` + metadata table. `--object-mask-weight` on both
geometry trainers; ObjectSegmentNet U-Net-lite segmenter; feature bridge into
`v115-feature-map-v1`. `--feature-store` became repeatable (terrain + object priors side by side).
48/48 tests.

## Spec 117 — WDL-lattice coarse prior (US1–US3(i) implemented)

Lattice was already streamed (`wdl_outer_17`/`wdl_inner_16`) — only frozen catalog rows added.
`LatticeNet` v2 U-Net-lite 675K params; masked 545-pt contract; bridge into `--feature-store`.
Scheduling bug found: OneCycleLR warmup vs patience-15 early stop killed detailer runs — fixed
via shared `lr_schedule.py` warmup-aware stale counter.

## Spec 116 — Relational terrain layers (FULLY IMPLEMENTED)

All 35 tasks. Relational schema framing (MCLY.textureId = foreign key into MTEX). StructureSlotNet
per detail slot, held-out 8-neighbour split, structure→feature bridge. Key rescore finding on the
honest split: every geometry checkpoint beats tile-mean (v3-deconfounded best −40.7%); old "no
model beats tile-mean" was a leaky-split artifact. 125/125 tests.

## Spec 114 — Direct minimap-to-terrain

Unauthorized universal-raster reset reverted (commit 06151357). Coarse `mit_b0` (visually
strongest, SC-001 false) + residual detailer chain. T061 detailer 9.1%, T063 bandsplit 11.2%
relative, user visual verdict positive, promotion pending. All heavy training user-run.

## Spec 115 — Terrain-feature deconfounding

Road-region MAE −21.35% (v3-deconfounded 8ch). Liquid cell classifier river IoU 0.82 at quad
resolution. Normal gradient supervision loss-only. Lessons: classify at the authoring unit
(128×128 quads), target must be visible in RGB.

## Spec 113 — Minimap super-res / detail

Production 8×/chunk UVs, footprint-selected mips. Authored vs synthetic are different domains
(NCC p50 0.211); intentional terrain-only cross-domain supervision. ComfyUI-native RealPLKSR
decision. Cross-map analyzer, tile-list + authored-reference support.

## Spec 112 — V50 height model

Relative-height contract follows published floor formula; dual-source curriculum
(2,990 rows: 1,629 authored + 1,361 synthetic). `curriculum-0_5_3_3368-dual_v1/v3.zarr`.

## Spec 111 — Minimap lighting calibration

`MinimapShadingMatch` sweeps 24 hourly candidates, tint-invariant luma Pearson correlation.
0.05 elevation floor makes hours 0–6/18–23 render identically. Implemented through T019 gate;
user-run bucketing + training remain.

## Spec 110 — Viewer stabilization (detail archived 2026-07-18)

Global time-of-day light unconditional (default noon); exact-build DBC locals blend over it.
Fog Start/End user overrides. MCLY/MCAL/MCNR/BLP compositor corrected; MCLQ per-cell types;
WL* requires all three provenance signals. M2 runtime must become native-only (remove M2→MDX
fallback). Full chronological detail in `memory-bank/archive/2026-07-18-spec110-viewer-stabilization-detail.md`.

## Spec 109 — V50 clean-room dataset

Build pipeline complete; `H:\CLIENTS` auto-resolution; parallel synthesis; per-build Zarr writer.
Phase 8 data-loss bug (finalize fed blank template, retry destroyed store) fixed with staged
writes. Phase 9: per-map resilience + second object-inclusive curation manifest. Full corpus
(4 maps) built and curated on disk (Kalimdor 951, Azeroth 685, PVPZone02 64, Kalidar 56).

## Corpus structure findings (2026-07-21)

9.5% L1 alpha block cross-tile reuse ≥0.99 under 8 dihedrals → terrain assembled from reused
fractal brush library. L0 has no alpha map. 99.6% val/train spatial adjacency → leaky split.
`sc001=False` in every run until the honest split fixed evaluation.

## M2/liquid/lighting research

- 1.0.0 M2 = `MD20` v0x100 (NOT pre-256); only 0.11/0.12 are pre-256 and use MDX.
- 0.5.3 normals inverted vs 0.6.0+ (winding); light Z sign build-version-aware.
- `TerrainSolarDirection`: fixed NW bearing (theta=225°), elevation only cycles. +X=North,
  +Y=West, +Z=Up.
