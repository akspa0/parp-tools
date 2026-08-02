# Progress — wow-viewer

Last updated: 2026-08-01

## v0.5.2 release (2026-08-01)

Shipped: hillshade Y-axis fix, object-capture Z backlighting fix, taxi popup removal,
`--time-hours` unlocked on `synthetic-minimap`, tone mapping always applied, liquid
terrain-height gating, `--include-wmos` + `--bake-mcsh` flags, GUI checkboxes. Version
bumped to 0.5.2. Build clean (0 errors).

**Spec audit**: [`docs/specs-audit-2026-08-01.md`](../docs/specs-audit-2026-08-01.md) — code,
viewer, and v50 dataset declared feature-complete; ~52 old specs archived; 20 stay current.
Detailed pre-2026-08-01 history: `memory-bank/archive/2026-08-01-progress-detail.md`.

## Current state by lane

| Lane | State |
|------|-------|
| Viewer | Lighting/taxi/minimap fixed for v0.5.2. Global noon light unconditional. M2 gap = 1.0.0–3.0.0 (0.11/0.12 MDX work; 3.0.1+ work). Spec 104 active. |
| v50 dataset | Feature-complete. `curriculum-0_5_3_3368-dual_v3.zarr` (2,990 rows). Coarse+detailer chain proven; detailer bandsplit-v2 11.2% relative, user promotion pending. |
| Synthetic minimap | Solar direction at `--time-hours` (fixed NW bearing, elevation cycles). Linear-space shading + analytic cast shadows replaced the exposure-20 tone map (2026-08-01, uncommitted); MCSH still optional and separate. |
| PM4 | Asset matching (046) + surface correlation (065) active. |
| UI | 069/080 active; 107 implementing. |
| M2 | 104 active; 105 format profiles feed version dispatch. |

## Test suite state

Full data-harvester suite: ~1150 passed / ~45 skipped / 3 pre-existing unrelated failures
(v24 export-map fixture, 2× v25 h1_coarse) — unchanged across recent sessions. Full C# solution
Debug build: 0 errors.

## User-run gates still open

- Detailer bandsplit-v2 geometry promotion (visual gate) → then object-mask phase.
- Spec 117 lattice predictor learnability verdict; Spec 118 object-mask training comparison.
- Spec 111 real-client bucketing pass.
- Terrain-shadow fix calibration: re-measure `SyntheticMinimapLinearLightGain` (1.166) and
  `DefaultCastShadowStrength` (0.45) against authored tiles, asserting contrast/std **and** mean.
- Spec 110 real 3.x visual proof for global-light/LIT repairs.

## Historical summary (2026-07 → feature-complete declaration)

- Spec 122 curation shipped; Spec 121 closed (RGB→WDL wrong); 119/120 archived (minimap object
  identity dead end).
- Spec 116 relational layers fully implemented; Spec 117/118 implemented (user gates remain).
- Spec 114 coarse+detailer chain proven; Spec 115 road deconfounding −21.35%.
- Spec 109 v50 corpus built + curated (4 maps on disk); Phase 8/9 data-loss fixes landed.
- Spec 110 viewer stabilization detail archived 2026-07-18.
- Full per-spec chronology: `memory-bank/archive/2026-08-01-progress-detail.md`.
