# MDX/M2 0.9.0 Render Path — Part 05 (Parser↔Render Contract Cross-check)

## Scope
Cross-check parser-proven invariants against now-anchored renderer-side functions.

## Parser-side proven invariants (existing run)
- `SEQS` strict record stride (`0x8C`) and count contract.
- `GEOS` two-pass parse with hard overrun fail.
- Structured stream extraction for geoset components (not opaque bytes).

## Renderer-side anchored points (new)
- `FUN_0043cb90` (`ModelRender.cpp`): builds render-side entries, attaches material, calls geoset payload writer.
- `FUN_0043cea0` (`ModelRender.cpp`): copies geoset shared arrays with index guards.

## Contract implications
1. Parser guarantees must hold before render-side copy/assembly:
   - count/stream lengths from `GEOS` must be coherent, or render assembly becomes invalid.
2. Material linkage is renderer-first-class:
   - material object creation/association occurs in render module (`FUN_0043cb90`).
3. Render path repeats strict bounds assertions:
   - `index < array_size` checks exist both parser-side and renderer-side.

## Updated confidence
- Parser contract: **High** (already proven).
- Renderer function localization: **High** for `ModelRender.cpp` anchoring.
- Exact `AddGeosetToScene materialId >= numMaterials` branch bind: **Medium** (literal unresolved but adjacent flow anchored).

## Practical tooling rule (0.9.0 profile)
- Treat geoset-material consistency as mandatory pre-render validation input.
- Do not downgrade parser hard-fails into warnings for this profile.
- Preserve renderer-compatible per-stream cardinality assumptions (vertex/aux arrays aligned by geoset index domain).
