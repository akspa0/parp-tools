# WLW-based Liquid Rescue

Purpose: Rebuild missing/broken MH2O water from WLW source files by rasterizing WLW surface triangles into chunk masks and heights. Additive-only; never overwrite valid water.

## Availability
- 0.5.3: WLW only (authoritative).
- 0.6.0–3.0: WLW + WLQ/WLM may exist. Use WLQ/WLM only if validated against WLW; otherwise skip.

## Knobs (minimal)
- `--rescue-liquids-from-wlw` (default off)
- `--wlw-rescue-tri-threshold N` (ignore tiny/degenerate faces; default small)

## Inputs
- WLW decoded blocks → triangles in world space (see `3D_wlw2obj.py` for reference).
- ADT tile context: `MapId`, tile indices `(x,y)`, chunk bounds.
- Existing MH2O (if any) per chunk.

## Outputs
- MH2O patch per tile:
  - Zero or more layers per chunk generated from connected components of covered cells.
  - Per-layer masks (compressed), height arrays (uniform or per-cell), min/max.
  - Liquid type when confidently inferred; else default "generic".

## Algorithm
1. Decode WLW blocks → generate 3D triangles.
2. Transform triangles into the tile’s world space.
3. Clip by tile AABB and MCNK AABB.
4. Rasterize per chunk to grid cells:
   - Coverage → occupancy mask.
   - Height per cell = median of triangle plane samples at cell center (or robust average).
5. Connected components on the mask → one MH2O layer per component; compute bbox and compress masks.
6. Integrate patch additively: only fill where MH2O missing/broken; do not overwrite valid existing layers.

## Output Mode Note

- With `--output-adt-version 3x` (default): WLW rescue generates MH2O layers (additive-only, does not overwrite valid water).
- With `--output-adt-version pre3x`: WLW rescue maps to legacy MCLQ generation; see `snippets/mclq_emit_from_wlw.md`. MH2O is not emitted in this mode.

## Pitfalls
- WLW structure variance: enforce header/magic/version and bounds checks; skip on mismatch.
- Coordinates may be stored as ints/floats across versions: prefer interpretation with plausible AABBs.
- Liquid type inference from WLW may be incomplete; default conservatively.
- Don’t introduce both MCLQ and MH2O; we remain MH2O-only.

## Validation (real data)
- Use `test_data/0.5.3` and `test_data/0.6.0` samples.
- Compare masks before/after; emit CSV deltas.
- Visualize OBJ overlays or NDJSON of rescued components.
- Open ADTs in noggit3 to confirm water surfaces and MH2O correctness.
