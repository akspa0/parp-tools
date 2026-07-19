# Synthetic terrain lighting and time-color evidence

**Status**: implemented; real-client capture comparisons remain user-run | **Date**: 2026-07-15

## Why this is a dataset signal

Early minimap tiles do not always share one neutral-white grade. A slightly different outdoor-light
time can tint the entire render off-white, while relief and baked shadow retain local structure. If
synthetic training renders contain only one fixed neutral light, a model can incorrectly bind that
grade to terrain identity. Time-of-day variants therefore belong in the synthetic image generator
and its evidence ledger, not as another deployment input or a new prediction head.

## Evidence boundary

The 1.0.0 client research in
`docs/architecture/wow-1.0.0-world-lighting-shadow-model-2026-07-15.md` supports this terrain model:

```text
lit_rgb = albedo * (ambient + directional_color * max(0, MCNR_normal dot light_direction) * visibility)
visibility = f(MCSH)
```

- `MCNR` supplies 145 terrain normals per chunk. They must be decoded from their file convention and
  transformed into renderer coordinates; a two-sided `abs(N dot L)` is incorrect. Client-faithful
  capture evaluates the clamped Lambert term at MCNR vertices and interpolates it, matching the
  fixed-function vertex-lighting order rather than re-normalizing an interpolated normal per pixel.
- `MCSH` is a 64x64 one-bit baked mask per chunk. It is a distinct spatial signal, not an MCAL layer,
  and its final row/column must not receive MCAL edge-fix copying.
- Outdoor directional color, ambient color, fog, and related values vary with game time.
- Exact timed colors come from the selected map LIT tracks or the exact-build Light* DBC records.
  The native MCSH attenuation coefficient is still unproven in this spec. A fallback palette or
  coefficient is authored evidence, never client-exact evidence.

## Minimap boundary

`synthetic-minimap` is not a runtime-world capture. Every supported era uses one fixed 12:00
achromatic global light so its terrain material colors remain comparable. Map LIT and local/global
Light DBC profiles belong to interactive viewer lighting and MUST NOT color-grade minimap targets.
The profile precedence below applies to runtime/capture experiments outside that minimap command.

## Runtime/capture profile precedence

Synthetic rendering uses a versioned lighting profile with an explicit evidence state:

1. **Explicit exact profile artifact**: an operator-selected, hash-bound artifact wins. It must name
   one source kind and may not silently splice LIT fog into DBC diffuse/ambient values.
2. **Map LIT profile**: for the non-minimap runtime/capture lane, evaluate the selected map's global/default clear
   group at the requested time. Retain build, exact virtual path, file hash, LIT version, light/group,
   track IDs, timed samples, and interpolation result. Local LIT zones remain disabled for capture
   until their `/36` scale and coordinate transform are proven.
3. **Build-scoped Light* DBC profile**: load the applicable `Light`, `LightParams`, `LightIntBand`,
   `LightFloatBand`, and `LightSkybox` database records through DBCD with the exact client build and
   bundled WoWDBDefs definitions. Retain table, record, band, timed-value, schema/build, and source
   hashes. These database tables are not animation keyframes.
4. **Authored fallback**: `wow-1.0.0-authored-day-night-v1` keeps the current analytic sun path and
   approximate colors available when no table artifact exists. Its evidence state is
   `authored_fallback_not_client_light_data`. This profile is useful augmentation, not recovered fact.

The profile artifact must retain at least build, source kind, table/schema or LIT identity, selected
time and contributing record/track samples, light direction source, directional/ambient/fog colors
and intensities, MCSH strength and filter policy, renderer revision, and source hashes. DBCD is the
reader for Light* tables; this work must not create another DBC parser.

Implemented operator surfaces are `wowviewer-inspect lit profile` and
`wowviewer-inspect light profile`. Their hash-bound JSON can be passed directly to
`spec103_build_synthetic_store.py --lighting-profile`; direct/ambient/fog colors are retained as
client evidence while direction and MCSH strength stay authored. Such rows are always private BYOD.

The active viewer uses the exact-build Light DBC resolver for 2.x+ clients without a usable map LIT
source and exposes whether it is active and why it is unavailable. `synthetic-minimap` deliberately
does not call that resolver; `terrain-minimap-synthesis-v6` records its fixed-noon-white profile.
Live 2.4.3.8606 proof also established two recoverable source anomalies: LightIntBand row 360
declares 360 values although six Time/Data slots are populated, and LightParams row 575 references
an absent optional LightSkybox row 18. The resolver records both anomalies while preserving the
usable direct, ambient, and fog bands.

## LIT decode and sky boundary

- Disk order is the 8-byte file header, **all** 64-byte light-list entries, then all per-light data
  groups. The previous viewer interleaved one header with its groups, shifting color payload by the
  remaining headers and turning unrelated bytes into plausible burgundy plus neon-green bands.
- The world renderer is Z-up. The previous sky dome made Y the vertical axis, rotating the gradient
  onto the horizon. The dome is now built as an XY ring with Z height.
- Tracks 2 through 6 are five distinct sky colors. Rendering only top-to-horizon is an explicit
  approximation until the four LIT sky-float arrays' altitude placement is understood; fixed band
  thresholds must not be invented and called client-exact.
- Track 8 remains unknown in `LIT.md`; shadow-opacity semantics are an inference unless separately
  recovered. MCSH terrain visibility is a different signal.

## Synthetic variant contract

- Compute normals from owned/generated height or carry MCNR from an explicitly licensed source.
- Apply one-sided Lambert lighting and optional MCSH modulation to generated/owned albedo.
- Never apply the profile to a captured or already-lit minimap; that would double-light it.
- Record `source_group_id`, `lighting_variant_id`, `game_time`, profile revision/evidence state,
  light direction/colors/intensities, MCSH source/presence/strength, and the source height hash.
- Every lighting variant of one source tile remains in the same train/validation group. Color
  variants cannot become split leakage or inflate the unique-terrain count.
- Lighting/MCSH may be baked into synthetic RGB or used by a teacher/diagnostic. Ground-truth MCNR,
  MCSH, or time is not an allowed input to the final image-only deployment model.

## Rights and provenance boundary

Rendering a client-derived height, MCSH, or texture through this code does not make it legally clean.
The tooling distinguishes two lanes and does not make a legal conclusion:

- `clean_synthetic`: operator-authored or licensed geometry/material sources, source license and
  rights assertion supplied explicitly, hashes verified, no client-derived training arrays.
- `private_byod`: client-derived numeric/image evidence kept for the operator's private workflow;
  no redistribution claim is attached.

The licensed-synthetic gate fails closed on missing/`UNSPECIFIED` rights fields, a changed height
hash, captured minimap input, or client-derived source state. A model's distribution status remains
a question for the operator and counsel; code can preserve evidence, not certify legality.

## Proof checklist

1. Asymmetric MCNR samples prove the file-to-renderer coordinate transform.
2. MCSH bit/corner tests prove all 64 rows and columns survive the mesh bridge.
3. CPU lighting tests prove back-facing normals receive ambient only and MCSH attenuates the declared
   term under the versioned contract.
4. Top-down capture tests prove one ADT tile maps to one image with dataset row/column orientation.
5. Synthetic-store tests prove variant provenance, source grouping, no double-lighting, and the
   fail-closed licensed-synthetic gate.
6. The exact 2.4.3.8606 archive probe resolves global Light row 1 / LightParams row 12 and the timed
   direct, ambient, and fog values with exact source hashes. A user-run one-tile render remains the
   visual proof; no training or broad client harvest is launched by this implementation pass.
