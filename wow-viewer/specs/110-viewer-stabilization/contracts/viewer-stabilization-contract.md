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
   tile flags determine whether a source cell exists.
2. `--time-hours` accepts minute-precise `HHmm` (`1215`) or `HH:mm` (`12:15`) clock input as
   well as legacy decimal hours. It is normalized to one client day and recorded as both canonical
   `HH:mm` and decimal hours in the manifest. A readable global clear-weather LIT profile supplies
   direct, ambient, and fog color tracks; the manifest records its source and evidence.
3. If global LIT evaluation is unavailable, the command uses the versioned authored day/night
   profile and records `authored_fallback_not_client_light_data`. It does not claim exact client
   lighting or use unproven local LIT zones.
4. LIT global-clear colors do not supply a directional vector. The shared authored direction is
   vertical at noon and uses the terrain-world/minimap-raster axis transform after noon; its
   evidence remains explicitly authored rather than client-exact.
5. Per-tile output and whole-map output are independently selectable. Each selected mode emits
   both liquid-free terrain and `_liquid` artifacts. Whole-map canvases cover only the inclusive
   bounds of successfully emitted tiles and preserve unoccupied positions as transparent pixels.
6. The manifest records client build identity, map, time, lighting source/evidence, tile results,
   terrain/liquid paths, liquid pixel count/render profile, bounds, scaled output dimensions, and
   the pipeline stage plus first relevant source frame for every failed tile. `--tile-x` and
   `--tile-y` select one occupied coordinate only when supplied together. The liquid overlay uses
   the current flat viewer type palette and opacity, not a claim of native water texture, animation,
   or reflection parity. It is the source of truth for how each PNG was derived.
7. A readable tile is not skipped solely because no referenced MTEX BLP decodes. After the
   same-stem and related-diffuse tiers, the exporter may use a successfully decoded deterministic
   catalog RGB material chosen by source-folder and terrain-family affinity, or a previously decoded
   catalog material when early-client listfile discovery is incomplete. It records
   `catalog_rgb_last_resort_proxy`, the original reference, and the resolved BLP. If MCLY/MTEX is
   absent entirely, the compositor uses that proxy as its base material.
8. Alpha object and roof masks may have different dimensions. Every painter validates its target
   buffer dimensions; the 257² terrain grid is never used as a bounds check for a 256² roof mask.
   WMO placements that cross a tile boundary are clipped per destination tile rather than treated
   as a terrain-decode failure.

## Conversion contract

1. WMO v14→v17 and v17→v14 are independent operations with separate capability statements.
2. M2→MDX export includes a source identity and profile/result summary.
3. A direction is called reliable only after its declared fixture and real-client validation level is met.
