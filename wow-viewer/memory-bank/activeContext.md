# Active Context — wow-viewer

Last updated: 2026-08-01

## Session: v0.5.2 release + spec audit

- **Version** bumped to 0.5.2 (`eng/Version.props`, `ViewerProductName`, `docs/releases/v0.5.2.md`).
- **Release fixes**: hillshade Y-axis inversion (NW→SW), object-capture Z backlighting (DirectX/OpenGL winding), taxi popup removed, `--time-hours` unlocked on `synthetic-minimap`, tone mapping always applied, liquid terrain-height gating.
- **Synthesized minimap**: `--include-wmos` (experimental, off) + `--bake-mcsh` (terrain shadows) flags; GUI checkboxes added.
- **Spec audit**: [`docs/specs-audit-2026-08-01.md`](../docs/specs-audit-2026-08-01.md) finalizes the backlog. Code/viewer/v50 dataset are feature-complete. ~52 old specs archived, 20 stay current. **Direction: lift off from current code with new speckit plans — don't resurrect old specs.**
- **Memory bank condensed** 2026-08-01; detailed per-spec history lives in `memory-bank/archive/`.

## Active specs (front-of-queue, per audit)

| Spec | State |
|------|-------|
| 046 PM4-asset-matching | Active — Ck24ObjectId global id, fingerprint-scan landed; match groups vs WMO archive next |
| 065 PM4-correlation-to-world-assets | Active — surface triangle correlation primary |
| 069 viewer-ui-overhaul | In progress (Phase 15) |
| 080 wow-ui-consolidation | Active UI target |
| 104 legacy-m2-rendering | Active — 1.0.0 M2 slice; gap is builds 1.0.0–3.0.0 (0.11/0.12 = MDX, work; 3.0.1+ work) |
| 105 format-version-profiles | Draft — feeds M2 version dispatch |
| 106 native-daynight-lighting | Planned |
| 107 lighting-quick-inspection | Implementing |
| 108 image-wdl-prior | Implementing |
| 109 v50-clean-room-audit | Active audit; v50.1 0_5_3_3368 full corpus on disk (4 maps) |
| 110 viewer-stabilization | Global light unconditional; fog/LIT fixes landed; detail history archived |
| 111 minimap-lighting-calibration | Implemented through T019 gate; user-run bucketing + training remain |
| 112 v50-height-model | Dual-source curriculum proven; relative-height model + CUDA trainer implemented |
| 114 direct-terrain-reconstruction | Coarse+detailer chain; detailer gate passed (11.2% relative); user visual promotion pending |
| 115 terrain-feature-classifier | Road-region MAE −21.35%; liquid cell classifier IoU 0.82 |
| 117 wdl-lattice-prior | US1–US3(i) implemented; real training verdict user-run |
| 118 object-occlusion-masks | US1–US3 implemented; user-run gates remain |
| 123 real-wdl-detailer | Draft — next lane: real WDL prior + residual detailer |
| 124 legacy-detangle-runpod | Draft — legacy Python detangle + C# RunPod tooling |

## v50 dataset state (feature-complete)

- `curriculum-0_5_3_3368-dual_v3.zarr` = 2,990 rows (1,629 authored + 1,361 synthetic).
- Coarse `mit_b0` + detailer `mit_b0` chain: detailer bandsplit-v2 val MAE 0.166769 vs coarse-only 0.1878 (11.2%), gate passed. User visual verdict positive; promotion gate pending.
- All six legacy geometry checkpoints clear the old leaky-split "tile-mean" bar on the honest Spec 116 split; v3-deconfounded (8ch) is the generalizer (relief MAE −40.7%).
- `synthetic-minimap` composes terrain-only PNGs from BLP+MCLY/MCAL/MCNR/MCSH; solar direction = fixed NW bearing, elevation cycles with `--time-hours` (default noon).

## Durable constraints

- `gillijimproject_refactor` is read-only. New code lives in `wow-viewer`.
- User runs training, capture, client-backed proof, and heavy work. Report client root + build + fingerprint with real-data conclusions.
- `AlphaWdtWriter.cs` is frozen (Rule 10).
- No DepthAnything/multi-head/shared-weight model paths. Ground-truth lighting/time never a model input. Canonical storage = per-build Zarr.
- M2 rendering gap is 1.0.0–3.0.0; 0.11/0.12 (MDX) and 3.0.1+ work. Do not use converted MDX as renderer proof.

## Known open items

- GLB export textures mirrored on Y (unrelated lane, unfixed).
- `--include-wmos` overlay experimental — can fail on some clients; defaults off.
- 0.5.3 normals inverted vs 0.6.0+ (winding); light Z sign is build-version-aware.

## Coroutine / next

- User: promote bandsplit-v2 geometry checkpoint (visual gate), then object-mask phase.
- Next lane per direction: Spec 123 real WDL prior + detailer; Spec 124 legacy detangle.
