# Viewer Stabilization Contract

## Active fog contract

1. The renderer receives exactly one finite range with `0 <= start < end`.
2. The range source is visible: fallback, lighting recommendation, or user override.
3. LIT/DBC updates may change colors and the recommendation, but must not overwrite a user override.
4. Invalid/missing lighting values resolve to a visible fallback before any shader or culling work.
5. Reset returns control to the current lighting recommendation or fallback.

## Native M2 contract

1. An M2 source is read and rendered as M2.
2. `M2Renderer` is the only successful runtime renderer for M2 source data.
3. M2→MDX is explicit export tooling only; it is never called by world-object or WMO-doodad runtime loading.
4. A native route failure reports source path, detected profile, missing capability, and failure reason.

## LIT map inspection contract

1. Minimap markers are an opt-in diagnostic visualization of the already loaded LIT entries.
2. The regular minimap and full-screen minimap use the same entry positions, visibility setting, and selected index.
3. Default or non-finite-position entries remain visible in the Lighting list as non-navigable but produce no map marker.
4. A list or marker selection changes only the shared selected index; it does not activate an entry's lighting, fog, or terrain state.
5. Double-clicking a navigable list entry positions the camera above its renderer coordinate and aims down; this is navigation, not lighting selection.

## Synthesized minimap export contract

1. The terrain baseline is a derived artifact built from decoded BLP texture pixels plus MCLY/MCAL
   composition; it never requires a shipped minimap BLP/PNG. Each emitted terrain artifact has an
   aligned `_liquid` companion derived from decoded unified liquid coverage/type evidence. The
   companion rasterizes complete source cells, not individual coverage vertices; Alpha MCLQ's 8×8
   tile flags determine whether a source cell exists. A visible cell's raw MCLQ type nibble selects
   its palette class (`0x01=Ocean`, `0x03=Slime`, `0x04=River/Water`, `0x06=Magma`); MCNK liquid
   flags are used only when that cell lacks a type nibble.
2. Production minimap synthesis is fixed at 12:00. `--time-hours` is optional compatibility syntax
   and may only specify noon. Every client era selects the shared terrain direction with a
   pure-white direct term and achromatic ambient; map LIT and Light DBC colors are not inputs.
3. For 2.x+ clients without a usable map LIT source, the interactive viewer uses the exact-build
   `Light` → `LightParams` → band chain and reports its records/recovery evidence. The generator
   never loads or evaluates that chain. Raw ADT normals are transformed to renderer space before
   the synthetic minimap Lambert dot.
4. The recovered 0.5.3.3368 world-light ray remains independent diagnostic research. It is never
   transformed or applied by synthesized minimaps. Raw MCNR/MCVT world axes are +X = North,
   +Y = West, +Z = Up, so positive terrain X is raster north in the MCNR/minimap contract,
   preventing terrain hillshade inversion.
5. Per-tile output and whole-map output are independently selectable. Each selected mode emits
   both liquid-free terrain and `_liquid` artifacts. Whole-map canvases cover only the inclusive
   bounds of successfully emitted tiles and preserve unoccupied positions as transparent pixels.
6. The manifest records client build identity, map, time, lighting source/evidence, tile results,
   terrain/liquid paths, liquid pixel count/render profile, bounds, scaled output dimensions, and
   the pipeline stage plus first relevant source frame for every failed tile. `--tile-x` and
   `--tile-y` select one occupied coordinate only when supplied together. The liquid overlay uses
   the current flat viewer type palette and opacity, not a claim of native water texture, animation,
   or reflection parity. It is the source of truth for how each PNG was derived.
7. A readable tile is not skipped solely because a declared MTEX BLP does not decode. After the
   same-stem and related-diffuse tiers, the exporter may use a successfully decoded deterministic
   catalog RGB material chosen by source-folder and terrain-family affinity, or a previously decoded
   catalog material when early-client listfile discovery is incomplete. It records
   `catalog_rgb_last_resort_proxy`, the original reference, and the resolved BLP. If no non-empty
   MTEX name is declared, the compositor instead emits an unlit solid-white empty baseline.
8. Alpha object and roof masks may have different dimensions. Every painter validates its target
   buffer dimensions; the 257² terrain grid is never used as a bounds check for a 256² roof mask.
   WMO placements that cross a tile boundary are clipped per destination tile rather than treated
   as a terrain-decode failure.
9. Any WL*-derived `wl_liquid_*` or unified fallback mask MUST carry all of
   `wl_liquid_surface_quads_v1`, `wl_liquid_above_terrain_v1`, and
   `wl_liquid_basic_type_header_v1`. They prove contiguous 4x4-block surface geometry,
   per-raster-sample terrain-height visibility, and a resolved per-pixel liquid class. WLW/WLQ
   use the parsed header class where valid; WLM is magma and WLL is lava represented through the
   canonical magma palette. Dataset builders MUST reject a WL* fallback lacking any marker rather
   than treating sparse, below-terrain, or default-water pixels as a liquid-aware fact. Earlier
   datasets are invalid for liquid-aware use and require re-harvest; their stored masks cannot be
   repaired after the fact.

## Conversion contract

1. WMO v14→v17 and v17→v14 are independent operations with separate capability statements.
2. M2→MDX export includes a source identity and profile/result summary.
3. A direction is called reliable only after its declared fixture and real-client validation level is met.
