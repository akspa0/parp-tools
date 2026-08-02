# Active Context — wow-viewer

Last updated: 2026-08-02

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
| 125 minimap-dxt1-inversion | Active — DXT1 parity companion + lighting baseline + restoration/reconstruction/super-res |

## v50 dataset state (feature-complete)

- `curriculum-0_5_3_3368-dual_v3.zarr` = 2,990 rows (1,629 authored + 1,361 synthetic).
- Coarse `mit_b0` + detailer `mit_b0` chain: detailer bandsplit-v2 val MAE 0.166769 vs coarse-only 0.1878 (11.2%), gate passed. User visual verdict positive; promotion gate pending.
- All six legacy geometry checkpoints clear the old leaky-split "tile-mean" bar on the honest Spec 116 split; v3-deconfounded (8ch) is the generalizer (relief MAE −40.7%).
- `synthetic-minimap` composes terrain-only PNGs from BLP+MCLY/MCAL/MCNR/MCSH; solar direction = fixed NW bearing, elevation cycles with `--time-hours` (default noon).

## Terrain shadow fix (2026-08-01, uncommitted)

Synthesized minimaps read as flat/shadowless next to authored ones. Two independent causes, both fixed:

1. **Tone map killed contrast.** Exposure-20 Reinhard + ambient 0.25 compressed the whole Lambert
   range into `0.833..0.962` — **12.8% of albedo**. Exposure 20 had been fitted against *mean*
   authored brightness only; a saturating curve matches the mean by destroying the range. Replaced
   with linear-space shading (`SrgbToLinear` → light → `LinearToSrgb`) plus a linear gain.
2. **Nothing ever cast a shadow.** Lambert N·L is slope shading; it can't darken flat ground behind
   a ridge. New `TerrainCastShadowMap` ray-marches the tile's own MCVT heightfield toward the sun.
   Single-tile only — shadows do not cross tile seams.

**Gain must be DERIVED, never hardcoded** (`TerrainLightingMath.ResolveLinearLightGain`). The first
cut hardcoded 1.166, anchored at lambert 0.5 — but flat ground under the noon sun is lambert
**0.894**, so ordinary terrain rendered 19% too bright and still read as washed out. Anchoring at
`FlatGroundNoonLambert` against the legacy calibrated response fixes it, and makes brightness and
contrast independent: lowering ambient deepens shadows while the gain auto-compensates to hold lit
ground steady.

Defaults now: ambient **0.12** (was 0.25), cast strength **0.70** (was 0.45) → shadowed flat ground
~37% darker than lit, vs ~18% at the old values. A *tinted* ambient is the lever for shadow HUE —
shadowed ground is lit by ambient alone, so a cool ambient is what makes shadows read cooler rather
than as a darker copy of lit ground.

## Water palette is era-scoped (2026-08-01, uncommitted)

`MinimapLiquidPalette` replaces the hardcoded liquid colours. Default is **`PreAlpha053`** — a bright
cyan-teal `(0.33, 0.72, 0.80) @ 0.82`, matching the 0.5.3 restoration target. The legacy
`(0.15, 0.35, 0.65) @ 0.55` survives as `ViewerFlatV1` for viewer-matching renders and for
reproducing any pre-existing corpus. The palette name goes into the synthesis manifest's render
profile, so a corpus always states which water it was rendered with.

The miss against authored was overwhelmingly in **green** (0.35 → ~0.72); blue moved much less. The
teal values are **eyeballed off a comparison screenshot, not measured** off authored tiles.

## Era gating (2026-08-01, uncommitted) — the organising principle

Blizzard changed minimap generation across builds: **0.5.3 = Alpha, 0.6.0 = Beta 1, 1.0.0 = different
again**. `MinimapEraProfile` (`WowViewer.Core.IO/Maps/`) is now the single source of every
era-sensitive default — solar azimuth model + provenance, liquid palette, ambient, cast-shadow
strength/softness — resolved from the build (`--era` forces it). Two invariants:

- An **unrecognised build is flagged**, never silently defaulted (`exactEraMatch`; CLI warns).
- Profiles carry **provenance, not just values**. `SolarModelProvenance` marks 1.0.0 as
  `TracedFromClient` and Alpha/Beta 1 as `AssumedFromOtherBuild`. Beta 1's numbers are inherited
  from Alpha and are *not* Beta-1 findings. Era + provenance go into the manifest render profile.

## Solar azimuth: open question, measurement built

The fixed 45° NW source bearing is from an x32dbg trace of **WoW 1.0.0** `SetDirection`; the target is
**0.5.3 Alpha**. User expects Alpha's sun to travel east→west. `TerrainSolarDirection.Evaluate` now
takes an azimuth (no-arg overload unchanged), `EvaluateSweepAzimuthDegrees` provides the E→W model,
and `MinimapShadingMatch.SweepSolarAzimuth` scores bearings × hours against authored tiles —
exposed as `synthetic-minimap --measure-sun --authored-reference`. **Not yet run.** Report caveat:
tight per-tile agreement only proves a fixed sun if the tiles span different capture times.

## LIT spatial records were never scaled (fixed)

LIT positions/radii are client fixed-point at 1/36 — the scale `ClientFixedUnitsPerWorldUnit` was
already named for ("the same 1/36 fixed scale as the outdoor-light spatial records") but only fog
used. Lights plotted ~36× off-map. `LitListEntrySummary` now exposes `RawPosition`/`RawLightRadius`/
`RawLightDropoff` alongside world-unit `Position`/`LightRadius`/`LightDropoff`/`OuterRadius`.
Unverified assumption: all three components scale (user's phrasing was ambiguous about vertical).
Same bug fixed in `LitLightHeaderProfile` (`WorldPosition`/`WorldRadius`/`WorldDropoff`/
`WorldOuterRadius`).

## LIT light overlay (2026-08-02, uncommitted)

`TerrainMinimapLightOverlayCompositor` + `synthetic-minimap --light-overlay` emits a per-tile
`*_lights.png`: each light's influence dome (reusing `LitSpatialSampler`'s falloff, blended toward a
weighted mean so overlaps don't bleach white) plus a colour swatch at its centre, coloured from the
clear-weather LIT track at the render's time of day. **Separate file on purpose** — light annotations
must never enter the terrain RGB corpus. Lights on adjacent tiles are drawn when their radius crosses
the seam. `MinimapTileProjection` holds the world↔tile mapping (world X ↔ ROW, world Y ↔ COLUMN,
both decreasing with index), derived from `AdtTensorPackBuilder` and round-trip tested.

## Authored comparison scorecard (2026-08-02, uncommitted)

`synthetic-minimap --score --authored-reference` measures agreement instead of leaving it to the eye.
Per tile it reports **mean ratio, contrast ratio, luma correlation, MAE, per-channel ratios** and a
combined score, writes `authored-comparison.csv`, and — critically — **re-renders the pre-session
configuration on the same tiles** (tone-mapped exposure 20, ambient 0.25, no cast shadows,
`ViewerFlatV1` water) and prints a BETTER/WORSE/same verdict per metric plus tiles-improved count.

The score is a **product** of brightness × contrast × structure penalties, not an average: a render
that nails the mean while flattening contrast must score ~0, which is exactly what the exposure-20
calibration did while passing its own gate. A test pins that case.

Correlation is the metric that responds to **shadow direction** — a mirrored render matches both
ratios perfectly and only correlation catches it. That makes it the one to watch for the sun-azimuth
question.

## Spec 125 — minimap DXT1 inversion (2026-08-02)

Authored 0.5.3 minimaps are DXT1-compressed; our synthesizer produces pristine 24-bit output, so every
comparison has been scoring clean-vs-lossy. Spec 125 adds encoding awareness:

- **Pure-C# DXT1 codec** (`WowViewer.Core.IO.Blp.Dxt1TileCodec`) — encode/decode cycle + round-trip
  check, zero external codec dependency (BCnEncoder.Net is NOT .NET 10 compatible and was rejected).
- **`--dxt1-parity`** — synthesizer emits a `*_dxt1.png` parity companion per tile (FR-015).
- **`--encoding-survey`** — per-build/map encoding distribution (FR-013).
- **`--lighting-baseline`** — tests the global lighting normalisation hypothesis (FR-016).
- **Strategic direction (user, 2026-08-02)**: because we now know how the minimap terrain shadow is
  created, the shadow in an authored tile is a readable terrain-shape signal. This opens three
  downstream models (all data-harvester, user-run): **US3 restoration** (pre-compression appearance),
  **US4 terrain reconstruction** (minimap RGB → heightmap → 3D mesh, skipping WDL), and **US5
  super-resolution** (real low/high-res pairs, no objects).

## synthetic-minimap tuning knobs (no rebuild needed to sweep)

`--ambient <v|r,g,b>` · `--cast-shadow-strength <0..1>` · `--shadow-softness <world-units>` ·
`--light-gain <v>` · `--liquid-palette <prealpha053|viewer>` · `--water-color <r,g,b[,a]>`.
Every run prints the resolved values, so each render is self-documenting.

`TerrainMinimapLighting.CreateShadedTerrain` is the new export profile; `CreateWhiteTopEdge` /
`CreateNoonWhiteGlobal` are unchanged so Spec 111's `MinimapShadingMatch` hour sweep is unaffected.
CLI: cast shadows default ON, `--no-cast-shadows` opts out; GUI has a matching checkbox.

**Not yet calibrated against real tiles.** Both `SyntheticMinimapLinearLightGain` and
`DefaultCastShadowStrength` (0.45) are analytic/chosen values. The re-check is user-run and **must
assert shading contrast (std), not just mean brightness** — fitting the mean alone is what produced
the flat render.

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
