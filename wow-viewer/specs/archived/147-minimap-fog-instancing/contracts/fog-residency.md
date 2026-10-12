# Contract: Fog-Bounded Residency

`WorldScene` resolves the effective fog range. The terrain/object streaming owner consumes a
snapshot of that range for normal camera-driven coverage.

## Coverage inputs

- active fog source, start, and end;
- camera world position and map tile;
- indexed existing tiles and conservative tile bounds;
- existing near-field safety policy;
- optional capture-path preload leases;
- explicit diagnostic full-load mode.

## Coverage outputs

Each candidate tile receives a deterministic state/reason:

- `OutsideMap`
- `WithinFogCoverage`
- `NearFieldSafety`
- `DirectionalPriority`
- `RetainedOnly`
- `CapturePreload`
- `DiagnosticFullLoad`
- `OutsideFogCoverage`
- `InvalidFogFallback`

Normal detailed terrain/object work is eligible only for a ready tile with
`WithinFogCoverage`/`NearFieldSafety` (and the existing visibility checks). Preloaded and diagnostic
states are exceptions visible in diagnostics.

## Rules

1. Derive coverage from active `FogEnd` in renderer world units; do not create a second lighting
   truth in `TerrainManager`.
2. Use conservative tile bounds intersection, including tile-edge safety, not tile-center distance
   alone.
3. Directional ordering may prioritize loads but cannot reject a nearby tile that intersects the
   fog window.
4. Apply stable release hysteresis so small camera/fog changes do not thrash residency.
5. A preload lease can protect a tile outside normal coverage only while its lease is active.
6. Invalid fog uses the existing bounded fallback and never means whole-map admission.
7. Report selection, retention, residency, readiness, drawable state, and submission separately.

## Required tests

- tile bounds intersecting fog radius are included;
- nearby side/rear tiles are not removed by directional ordering;
- outside-fog tiles are excluded without a lease;
- capture lease and full-load exceptions are named;
- fog revision updates invalidate targets without oscillation;
- invalid fog remains bounded;
- map edges and missing tiles remain deterministic.
