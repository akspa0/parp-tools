# WoW 2.0.0 Beta M2, Light, Particle, And Terrain Follow-Up Guide

## Purpose

This guide captures the current high-confidence findings from a static Ghidra pass against a beta `2.0.0` `WoW.exe` so future sessions do not need to repeat the same initial archaeology.

The immediate target is not to claim full `2.x` parity. The target is to preserve what the client demonstrably does in the engine, identify which seams are safe to implement in `MdxViewer`, and avoid widening support in ways that break `0.5.x`, `0.6.0`, or later clients.

## Scope

This document covers four related tracks:

1. `Model2` / BLS shader usage in the `2.0.0` client
2. Light-table loading and runtime light projection/management
3. Particle runtime behavior relevant to the current smoke/effects bug
4. A terrain follow-up seam for moving/sliding terrain textures such as the Shadowmoon-style slime effect

It does not claim that every field or every renderer branch is fully named. It records only what was directly supported by strings, xrefs, decompilation, and nearby runtime-type evidence.

## Evidence Base

- Binary: beta `2.0.0` `WoW.exe` loaded in Ghidra
- Method: string search, xref tracing, function decompilation, and runtime-type anchor inspection
- Validation boundary:
  - reverse engineering only
  - no repo code changes were made as part of this investigation
  - no automated tests were added or run
  - no runtime validation against a live `2.0.0` client session was performed in this slice

## Executive Summary

1. `Model2` in the `2.0.0` engine is not using a generic one-size-fits-all material path. It explicitly loads dedicated `Model2.bls` vertex and pixel shaders.
2. Map objects preload multiple dedicated pixel BLS programs, including translucent diffuse/specular variants. That makes shader/material selection a real parity seam for `2.x`.
3. Light-table loading is conventional, strict `WDBC` ingestion with ID-index lookup tables. The dangerous seam is not raw table loading but how those IDs are consumed at runtime.
4. `M2Light` objects are spatial/runtime-managed, not passive records. They are inserted into bucketed structures or general linked lists and relinked when their state changes.
5. The particle system is a true runtime subsystem with bootstrap pools and runtime `CParticle2` / `CParticle2_Model` objects. That is why parser-only fixes are unlikely to solve the current smoke projection issue.
6. The terrain side still needs a separate follow-up, but one ambiguity is now resolved: `terrainp*` belongs to the terrain pixel-shader path, while `XTextures\slime\slime.%d.blp` resolves into a separate animated `WCHUNKLIQUID` surface path. The current viewer terrain shader still uses fixed world-space UVs with no time-based or per-layer scroll offset path, so it cannot reproduce any true terrain-layer glide effect today.

## 1. BLS Shader Usage In The 2.0.0 Client

### 1.1 `Model2` Has A Dedicated BLS Bootstrap

The strongest `Model2` shader anchor is `FUN_00717b00`.

Confirmed strings referenced from this path:

- `shaders\vertex\Model2.bls`
- `shaders\pixel\Model2.bls`

High-confidence reading of the function:

- it belongs to a `Model2` initialization/cache path (`M2Cache.cpp` source-path string is nearby in the binary)
- it loads both the vertex and pixel BLS programs for `Model2`
- it allocates/initializes render-related state around that shader setup

Practical implication:

- later `2.x` support should not assume that an M2-family asset can be safely routed through a generic early-model material path and still produce parity
- a separate `2.x` `Model2` shader/material branch is justified by engine evidence, not just by guesswork

### 1.2 Map Objects Preload A Dedicated Pixel-Shader Bank

`FUN_006b3b20` preloads a fixed map-object pixel-shader bank:

- `Shaders\Pixel\MapObjOverbright.bls`
- `Shaders\Pixel\MapObjSpecular.bls`
- `Shaders\Pixel\MapObjMetal.bls`
- `Shaders\Pixel\MapObjEnv.bls`
- `Shaders\Pixel\MapObjEnvMetal.bls`
- `Shaders\Pixel\MapObjExtWater0.bls`
- `Shaders\Pixel\MapObjTransDiffuse.bls`
- `Shaders\Pixel\MapObjTransSpecular.bls`

This matters because it proves the engine is already splitting map-object materials by behavior class. In other words, translucency, specular behavior, environment behavior, and related map-object cases are not all being flattened into one program.

Practical implication:

- if the viewer collapses these families into one simplified shader path, it should expect regressions in lighting, translucency, and special-material behavior
- the fact that `MapObjTransDiffuse` and `MapObjTransSpecular` are distinct is especially relevant when diagnosing odd translucent behavior that also shows up in smoke-like effects

### 1.3 Terrain Uses More Than The Basic `terrain1..4` Family

The world/map initialization path `FUN_006a2360` preloads two terrain shader families:

Basic terrain variants:

- `Shaders\Pixel\terrain1.bls`
- `Shaders\Pixel\terrain2.bls`
- `Shaders\Pixel\terrain3.bls`
- `Shaders\Pixel\terrain4.bls`
- their `_s` shadowed variants

Special `terrainp*` variants:

- `Shaders\Pixel\terrainp.bls`
- `Shaders\Pixel\terrainp_s.bls`
- `Shaders\Pixel\terrainp_us.bls`
- `Shaders\Pixel\terrainp_u.bls`

The current pass now closes a more specific part of the terrain shader split than earlier notes did.

Relevant terrain-side functions:

- `FUN_006a4ab0`
  - computes runtime capability flags including `DAT_00cb3594` and `DAT_00ca31b8`
  - both are gated by pixel-shader-style capability checks (`FUN_00599010()` pattern) rather than by terrain-file decode
  - also computes `DAT_00cb359c`, which later shows up in the animated liquid/surface path as well
- `FUN_006bf760`
  - chunk/layer setup path
  - calls `FUN_006c0210()` only when the `terrainp*`-path flags are enabled
- `FUN_006c00f0` and `FUN_006c01a0`
  - allocate small fallback textures only when both `DAT_00ca31b8` and `DAT_00cb3594` are off
  - this is strong evidence for a fixed-function fallback path
- `FUN_006ceaa0`
  - terrain chunk draw caller
  - routes into `FUN_006cee30(...)`
- `FUN_006cee30`
  - terrain chunk draw selector
  - when runtime shader-capability flags are present, chooses a cached one-pass terrain program by layer count
  - otherwise falls back to a slower manual terrain path
- `FUN_006cf590`
  - consumer for the cached one-pass terrain programs chosen by `FUN_006cee30`

What the latest pass now supports directly:

1. `FUN_006a2360` loads and stores three distinct terrain-program families:
   - `terrain1.bls` .. `terrain4.bls` into `DAT_00caf304` .. `DAT_00caf310`
   - `terrain1_s.bls` .. `terrain4_s.bls` into `DAT_00caf548` .. `DAT_00caf554`
   - the separate special programs `terrainp`, `terrainp_s`, `terrainp_us`, `terrainp_u` into dedicated globals
2. `FUN_006cee30` uses the contiguous `DAT_00caf300` block as a layer-count indexed table when `DAT_00cb3594 == 0` and `DAT_00ca31b8 != 0`:
   - layer count `1` -> `terrain1`
   - layer count `2` -> `terrain2`
   - layer count `3` -> `terrain3`
   - layer count `4` -> `terrain4`
3. `FUN_006cee30` uses the contiguous `DAT_00caf544` block as the alternate layer-count indexed table when `DAT_00cb3594 != 0`:
   - layer count `1` -> `terrain1_s`
   - layer count `2` -> `terrain2_s`
   - layer count `3` -> `terrain3_s`
   - layer count `4` -> `terrain4_s`
4. If those cached layer-count programs cannot be used, `FUN_006cee30` drops into a slower manual terrain path that references `terrainp` / `terrainp_s` instead.
5. In this beta build, `terrainp_u` and `terrainp_us` are loaded during startup and released during shutdown, but this pass has not found terrain draw-path consumers for them outside init/teardown.

Practical reading:

- the client has a cached layer-count fast path built around `terrain1..4` and `terrain1_s..4_s`
- `terrainp` / `terrainp_s` are still part of the live terrain shader path, but they belong to the slower manual fallback path rather than the cached layer-count table
- the engine keeps a separate fixed-function-style fallback below both of those shader-enabled paths when the capability flags are unavailable
- terrain mode selection is decided in the chunk draw path, not by a one-time map-level mode switch alone

Practical implication:

- do not assume the entire terrain pipeline in `2.0.0` is only “sample layer textures by alpha and stop there”
- the terrain shader split is more specific than “one special `terrainp` path”: there are cached one-pass `terrain1..4` / `terrain1_s..4_s` programs plus a slower `terrainp` / `terrainp_s` fallback path that the active viewer still lacks
- however, the current evidence does **not** tie `terrainp*` directly to the animated slime surface sequence

### 1.4 Implementation Guidance For `MdxViewer`

Do:

1. Keep the current explicit `2.x` model-profile routing.
2. Treat shader/material parity as a first-class follow-up seam, not as a cosmetic polish step.
3. Separate `Model2`, map-object, and terrain shader concerns instead of forcing them through the same simplification.

Do not:

1. Claim later `2.x` parity based only on parser success.
2. Widen format routing and assume the renderer will “just work”.
3. Flatten translucent/specular/material families into one generic path without expecting regressions.

## 2. Light Data Read Path And Runtime Projection

### 2.1 The `Light*.dbc` Family Is Loaded Through Strict `WDBC` Loaders

The following path helpers and loader functions were confirmed:

- `FUN_00596a30` -> `DBFilesClient\LightFloatBand.dbc`
- `FUN_00596b70` -> `DBFilesClient\LightIntBand.dbc`
- `FUN_00596cb0` -> `DBFilesClient\LightParams.dbc`
- `FUN_00596ea0` -> `DBFilesClient\Light.dbc`
- `FUN_00597070` -> `DBFilesClient\LightSkybox.dbc`

Their corresponding loaders:

- `FUN_00552480` -> `LightFloatBand.dbc`
- `FUN_005526d0` -> `LightIntBand.dbc`
- `FUN_00552920` -> `LightParams.dbc`
- `FUN_00552b50` -> `Light.dbc`
- `FUN_00552d90` -> `LightSkybox.dbc`

Each loader follows the same pattern:

1. open the file
2. verify `WDBC` header magic
3. verify expected column count
4. verify expected row size
5. allocate one contiguous block for row data plus string-table bytes
6. read per-row records through a record reader helper
7. build an ID-index map for constant-time lookup by record ID

Confirmed row shapes from the decompiled readers:

- `LightFloatBand.dbc`
  - expected columns: `0x22`
  - row size: `0x88`
  - record reader copies:
    - 4 bytes at `+0x00`
    - 4 bytes at `+0x04`
    - `0x40` bytes at `+0x08`
    - `0x40` bytes at `+0x48`
- `LightIntBand.dbc`
  - same `0x22` columns and `0x88` row size
  - same record shape as `LightFloatBand`
- `LightParams.dbc`
  - expected columns: `9`
  - row size: `0x24`
  - record reader copies nine 4-byte fields
- `Light.dbc`
  - expected columns: `0x0c`
  - row size: `0x30`
  - record reader copies seven 4-byte fields plus one trailing `0x14`-byte block
- `LightSkybox.dbc`
  - expected columns: `2`
  - row size: `8`
  - record reader copies an ID and a string-table offset that resolves to a path/name pointer

Practical implication:

- the raw table-read side is not mysterious
- the engine is not doing ad-hoc lossy parsing here
- if viewer lighting is wrong, it is much more likely to be a consumer/runtime issue than a `WDBC` loader issue

### 2.2 The Light Tables Are Bootstrapped As One Coherent Group

The shared caller cluster leads into `FUN_00548170`, which loads the light-related database families as part of a larger bootstrap.

Practical implication:

- Blizzard treated the light tables as one coherent subsystem, not a scattered collection of unrelated files
- future viewer work should follow that same mental model: `Light`, `LightParams`, the band tables, and `LightSkybox` belong to one connected runtime pipeline

### 2.3 `M2Light` Is Runtime-Managed And Spatially Bucketed

The key `M2Light` anchor is `FUN_0072d1a0`.

High-confidence behavior from decompilation:

- when a light is in one runtime mode/type, it is inserted into a spatial bucket structure based on transformed position
- otherwise, it is inserted into a more general linked list

Related mutators:

- `FUN_0072cc60`
- `FUN_0072cc90`
- `FUN_0072cdc0`

These mutators change enabled/state/position-like data and then relink the light into the correct runtime structure.

Practical implication:

- model lights are not just passive parsed blobs attached to a model file
- the engine expects them to participate in runtime spatial/light-link management
- any viewer-side “just read the light struct and apply it directly” approach is likely to miss behavior

### 2.4 The Runtime Also Has Explicit Light-Management Types

Relevant runtime-type anchors confirmed by strings/xrefs:

- `CMapLight`
- `CLightList`
- `CGxuLight`
- `CGxuLightLink`
- `LightRef`

Useful decompilation anchors:

- `FUN_006a9d20` grows arrays of `CMapLight*`
- `FUN_006e7ee0` grows arrays of `LightRef`
- `FUN_005cddf0` allocates and links `CGxuLightLink` objects
- `FUN_005cd280` tears down light lists, `CGxuLight`, and `CGxuLightLink` structures

Practical implication:

- there is a real runtime graph/list layer between raw light records and final render-state use
- this supports the earlier `M2Light` result: the engine is building and maintaining live light structures

### 2.5 The Renderer Still Uses Fixed-Function OpenGL Light Calls

Confirmed imported GL symbols include:

- `glLightf`
- `glLightfv`
- `glLightModeli`
- `glLightModelfv`

This matters because the `2.0.0` renderer is not purely a modern shader-only light path. Shader toggles exist, but the binary still imports and presumably uses fixed-function GL light calls in the wider runtime.

Practical implication:

- do not assume all lighting behavior can be inferred from shader filenames alone
- some visual differences may come from fixed-function light state and how it is bridged into shader-enabled paths

### 2.6 Implementation Guidance For `MdxViewer`

Do:

1. Treat light-data ingestion and light runtime behavior as separate problems.
2. Keep the current viewer `LightService` work clearly labeled as an approximation until it is checked against the engine contracts more closely.
3. Prioritize tracing how `Light`, `LightParams`, and the band tables resolve into actual world/model light state.

Do not:

1. Assume `Light.dbc` alone is the whole lighting story.
2. Assume correct DBC parsing implies correct runtime light projection.
3. Assume the engine’s light behavior is shader-only.

## 3. Particle Runtime And The Smoke Bug

### 3.1 `ParticleSystem2` Is A Dedicated Runtime Subsystem

Strong string anchors:

- `E:\build\buildWoW\ENGINE\Source\Services\ParticleSystem2.h`
- `CParticleEmitter2_idx`
- runtime-type strings for `CParticleEmitter`, `CParticleEmitter2`, `CParticle2`, and `CParticle2_Model`

Key runtime bootstrap:

- `FUN_007c26c0`

High-confidence behavior:

- initializes `CParticleEmitter2_idx`
- allocates global emitter/index/record pools
- maintains a global initialization/refcount pattern

Practical implication:

- particles in this era are not a simple draw-what-the-file-says layer
- there is a real engine subsystem allocating and managing particle runtime state

### 3.2 Runtime Particle Objects Are Built From Copied Emitter Payloads

Key constructors/copy paths:

- `FUN_007ca960`
- `FUN_007ca390`
- `FUN_007c9eb0`
- `FUN_007ca9d0`
- `FUN_007ca400`
- `FUN_007c9f20`

The deeper path `FUN_007ca9d0` is the most informative:

- it copies emitter-related fields from offsets roughly in the `0x290..0x2CC` region
- it allocates and populates internal arrays/buffers/lists for the runtime object

Practical implication:

- there is a distinction between on-disk emitter definitions and the engine’s runtime particle objects
- this is exactly the sort of seam where weird smoke projection or billboarding bugs can survive even if the parser is “mostly right”

### 3.3 `CParticle2_Model` Uses Explicit Runtime Arrays

Useful anchors:

- `FUN_007c79d0` grows contiguous arrays of `CParticle2_Model`
- `FUN_007c3180` tears them down and frees related buffers/resources

That means `CParticle2_Model` is stored as a runtime object family with constructor/destructor loops, not merely as borrowed pointers into model file memory.

Practical implication:

- the engine is building model-specific particle runtime state on top of the parsed emitter payload
- this strengthens the case that the current smoke issue likely lives in runtime interpretation, transform/billboard math, or render-state handling, not only in file parsing

### 3.4 Why This Matters For The Current Smoke Bug

Current symptom in the active viewer context:

- smoke/effects look projected or oriented incorrectly across both MDX and M2-family assets

The RE evidence supports the following conclusion:

- this is unlikely to be solved by a narrow parser-only adjustment

Why:

1. the engine has a dedicated runtime particle subsystem
2. particle emitters are copied into runtime objects with their own buffers and state
3. map-object translucent shader families are also split into multiple specialized paths
4. runtime lights are spatially managed and may interact with particle/material rendering in ways the viewer does not yet match

The likely bug classes to inspect next are:

1. billboard orientation and camera-facing math
2. local-to-world transform application for emitter/model space
3. particle blend/state selection
4. interactions with the translucent/specular shader path

### 3.5 Implementation Guidance For `MdxViewer`

Do:

1. Treat the smoke bug as a renderer/runtime parity bug until proven otherwise.
2. Compare emitter transform handling and billboard behavior across both MDX and M2-family renderers.
3. Keep particle investigation separate from model-format-routing work.

Do not:

1. Treat a parser-side change as verified just because it changes the visual symptom.
2. Conflate `2.x` model-format support with particle/runtime correctness.
3. Assume the same fix will apply blindly to both old MDX and later M2 paths without checking shared runtime math.

## 4. Terrain Follow-Up: Moving Terrain Textures And Shadowmoon-Style Slime

### 4.1 What The Current Evidence Actually Supports

This pass did not yet fully close the terrain texture-animation path, so the claims here must stay narrower than the BLS/light/particle findings.

Current concrete anchors:

1. `FUN_006a2360` loads a special `terrainp*` pixel-shader family in addition to the basic `terrain1..4` family.
2. `FUN_006a4ab0`, `FUN_006bf760`, `FUN_006ceaa0`, `FUN_006cee30`, and `FUN_006cf590` now support a more precise split:
  - `terrain1..4` and `terrain1_s..4_s` are the cached one-pass terrain programs indexed by layer count
  - `terrainp` and `terrainp_s` belong to the slower manual terrain fallback path
  - `terrainp_u` and `terrainp_us` are loaded at startup but not yet tied to an active terrain draw branch in this pass
3. The binary contains the explicit strings:
  - `SLIME`
  - `XTextures\slime\slime.%d.blp`
4. That slime string no longer floats unanchored. `FUN_0069b310` uses a format-string table rooted at `PTR_s_XTextures_river_lake_a__d_blp_0088c3a0`, loads a 30-frame animated texture family, and returns a frame based on runtime cycling.
5. The animated-surface caller chain is:
  - `FUN_0069b310`
  - `FUN_0069e300`
  - `FUN_0069e3e0`
  - `FUN_0069e4f0`
  - `FUN_0069e690`
  - `FUN_006c65b0`
6. The object/type context around that chain points to `WCHUNKLIQUID`, not to the main terrain diffuse-layer path:
  - `DAT_00cc50b8` is initialized from the `WCHUNKLIQUID` world type in `FUN_006a2360`
  - `FUN_006b19e0` / `FUN_006b1a90` are create/destroy-style functions for that `WCHUNKLIQUID` object family
  - `FUN_006c4390` routes into `FUN_006c65b0`, which calls the animated texture loader
7. The active viewer terrain shader in `src/MdxViewer/Terrain/TerrainRenderer.cs` uses fixed world-space UVs for diffuse terrain sampling:
   - `vec2 worldUV = vWorldPos.xy * texScale;`
8. The active viewer terrain shader has no terrain-layer time uniform, no per-layer scroll velocity, and no terrain texture offset source.
9. The active viewer does animate liquids in `LiquidRenderer`, but that is a separate liquid pass and does not animate terrain layer textures.

Additional high-confidence `WCHUNKLIQUID` findings from the latest pass:

10. `FUN_006cac40` reads the low nibble of per-cell type bytes and returns the first non-`0xF` mode value for the active chunk-liquid object.
11. `FUN_006c65b0` dispatches rendering based on that mode value:
  - modes `0`, `4`, and `8` go through the texture-family surface path (`FUN_006c66f0` or `FUN_006c6900`)
  - modes `2`, `3`, `6`, and `7` go through `FUN_006c6bc0`, which uses per-vertex short pairs scaled like direct UV-style coordinates
12. `FUN_0069b310` is the animated texture-family loader:
  - loads `30` frames for one family from an `XTextures\...%d.blp` format string table
  - picks a current frame based on runtime cycling
13. `FUN_0069e200` builds strip/index ranges over an `8x8` cell grid for specific low-nibble cell modes, and the three animated-surface parents use values `1`, `4`, and `6`.
14. The animated liquid/surface caller chain splits into three family renderers:
  - `FUN_00695810` -> `FUN_0069e690(1)`
  - `FUN_006959b0` -> `FUN_0069e690(0)`
  - `FUN_00695da0` -> `FUN_0069e690(2)`
15. Those three family renderers ultimately use:
  - `FUN_0069dee0`
  - `FUN_0069dfe0`
  - `FUN_0069e0e0`
  and they are not identical:
  - `FUN_0069dee0` and `FUN_0069dfe0` use palette/lookup-driven texture-frame values
  - `FUN_0069e0e0` uses direct short-pair coordinate values instead
16. Disassembly tightens the relationship between low-nibble liquid mode and animated texture family:
  - `FUN_006c65b0` puts the raw mode from `FUN_006cac40()` into `ECX` and calls `FUN_0069b310`
  - this means the chunk-liquid mode value is also the texture-family index used by the animated frame loader
17. The currently recoverable `FUN_0069b310` texture-family table is:
  - family `0` -> `XTextures\river\lake_a.%d.blp`
  - family `1` -> `XTextures\ocean\ocean_h.%d.blp`
  - family `2` -> `XTextures\lava\lava.%d.blp`
  - family `3` -> `XTextures\slime\slime.%d.blp`
  - family `4` -> `XTextures\river\lake_a.%d.blp` again
18. The traced active global animated-surface branches now line up as:
  - `FUN_0069e690(0)` -> `FUN_0069e3e0` -> `FUN_0069b310(4)` -> duplicate `lake_a` family
  - `FUN_0069e690(1)` -> `FUN_0069e300` -> `FUN_0069b310(1)` -> `ocean_h` family
  - `FUN_0069e690(2)` -> `FUN_0069e4f0` -> `FUN_0069b310(6)` -> unresolved/empty table slot in this pass
19. The `WCHUNKLIQUID` mode split can now be read more concretely:
  - textured animated path: modes `0`, `4`, `8`
    - mode `0` currently maps to `lake_a`
    - mode `4` currently maps to a second `lake_a` slot
    - mode `8` still points at an unresolved table slot in this pass
  - direct-coordinate path: modes `2`, `3`, `6`, `7`
    - mode `2` currently maps to `lava`
    - mode `3` currently maps to `slime`
    - modes `6` and `7` still point at unresolved table slots in this pass
20. Novelty / possible dead-content findings from the same table walk:
  - `XTextures\river\fast_a.%d.blp` exists in the binary strings but is not referenced by the traced `FUN_0069b310` family pointer table
  - the family slots beyond the currently recovered pointers do not yet resolve to named strings through data xrefs
  - `terrainp_u` and `terrainp_us` still look similar on the terrain side: loaded at startup, freed at shutdown, but not yet tied to a live draw branch
21. A viewer-side source audit now sharpens where the current gap really is:
  - `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` reads `MPHD` as a raw `uint`, but actively uses it only for the profile-specific big-alpha mask
  - the same adapter treats any non-zero `MAIN` entry as generic tile presence and does not preserve or interpret per-entry flag meaning
  - `TerrainChunkData` preserves raw `MCNK` and texture-source flags, but the active terrain pipeline only branches on holes, liquid-related bits, and the `do-not-fix-alpha` bit
  - `TerrainLayer.Flags` preserves raw `MCLY` flags, but `TerrainRenderer` only consults `0x100` as the implicit-full-alpha hint when no alpha texture is bound
22. Repo-local format docs and writers already name richer standard-terrain flags that the active viewer is not modeling yet:
  - `MPHD`: `0x1` global WMO, `0x2` MCCV, `0x4` big alpha, `0x8` doodad-ref sort, `0x10` MCLV/lighting vertices, `0x20` upside-down ground, `0x40` unknown, `0x80` height texturing, `0x200` MAID
  - `MAIN`: `0x1` has ADT, `0x2` all water
  - `MCLY`: animation mask `0x001..0x040`, `0x080` overbright, `0x100` use alpha, `0x200` compressed alpha, `0x400` cubemap reflection
23. The beta `2.0.0` terrain renderer now shows a second motion seam that is separate from `WCHUNKLIQUID` animated texture families:
  - `FUN_006c00f0` builds a per-layer runtime object and copies a source layer flag field directly into the layer object's flag word at `+0x08`
  - both terrain draw paths, `FUN_006cee30` and `FUN_006cf590`, check that runtime flag word and, when bit `0x40` is present, use the low bits as an index into `DAT_00c72978` / `DAT_00c7297c` / `DAT_00c72980` before applying an extra transform/state change
  - `FUN_006804b0` updates those `DAT_00c72978..80` tables every world tick rather than treating them as fixed constants
24. Practical read of the new terrain-side evidence:
  - the beta client has a real time-varying terrain-layer transform path in addition to the separate `WCHUNKLIQUID` animated-frame path
  - that makes it much less likely that true moving terrain textures are driven by `WDT` global flags alone
  - the more plausible split is:
    - `WDT/MPHD` and `MAIN` flags gate parse/render capabilities and tile class
    - per-layer terrain motion comes from layer-local flags/state
    - chunk-liquid motion comes from the `WCHUNKLIQUID` mode nibble and animated family loader

Practical implication:

- two separate seams now exist and should not be conflated:
  - terrain shader behavior (`terrain1..4`, `terrain1_s..4_s`, `terrainp`, `terrainp_s`, and the capability-gated chunk draw selector)
  - animated chunk-liquid surfaces (`XTextures\slime\slime.%d.blp` and the `WCHUNKLIQUID` path)
- the current viewer cannot reproduce either kind of historical motion correctly today:
  - terrain layers have no client-faithful layer-count shader split and no proven motion/projection path
  - liquids are animated by a viewer-side approximation, not by the original client's mode-dispatched animated texture family system
- the client also appears to contain at least some partial or dormant liquid/surface content:
  - one traced global family branch resolves into an unresolved `FUN_0069b310(6)` table slot
  - `fast_a` exists as a texture family string but is not part of the traced active pointer table

### 4.2 What Is Still Open

The current pass does **not** yet prove all of the following:

1. what exact semantic mode `DAT_00cb3594` represents beyond “choose the `_s` layer-count table and extra render state”
2. whether `DAT_00ca31b8` is best read as the non-`_s` cached one-pass capability gate or something more specific
3. whether `terrainp_u` and `terrainp_us` are dead/unfinished in this beta build or used by a still-untraced terrain branch
4. whether any true terrain-layer glide/projection effect exists in the terrain path, as opposed to the separate animated chunk-liquid path
5. where the terrain-side scroll direction, speed, or projection basis would come from if such an effect exists
6. whether the famous Shadowmoon visual specifically belongs to:
  - animated chunk liquid
  - terrain-layer projection/motion
  - or both interacting together
7. how the terrain-layer runtime flag word used by `FUN_006c00f0` / `FUN_006cee30` / `FUN_006cf590` maps back to on-disk layer flags, and whether it is a direct `MCLY` projection or a normalized runtime encoding
8. which named surface families correspond to the animated liquid/surface mode groups currently seen only as dispatcher values `0/1/2` and cell-mode values `1/4/6`
9. whether the unresolved `FUN_0069b310(6)` branch and the unused `fast_a` family are dead code, disabled content, or simply routed from a caller that has not been found yet
10. which currently ignored `WDT` / `MAIN` / `MPHD` / `MCLY` flags the active viewer should eventually preserve as first-class terrain metadata even before it implements motion/render parity

Those are the next investigation steps, not solved facts.

### 4.3 Why This Seam Matters

If the original beta/TBC engine could display terrain textures that visibly slid or flowed over slopes, and the modern re-releases still fail to reproduce that behavior, then this is exactly the kind of place where the viewer can surpass the modern client in historical fidelity.

This is also a good fit for the current project because:

1. it is grounded in concrete engine evidence rather than wishful feature creep
2. it now breaks into two tractable implementation tracks instead of one fuzzy “slime terrain” story
3. it does not require widening model-format heuristics or destabilizing `0.5.x`/`0.6.0` support just to investigate it

### 4.4 Recommended RE Plan For The Terrain Texture-Animation Track

1. Continue from the now-confirmed terrain-side path:
  - `FUN_006a4ab0`
  - `FUN_006bf760`
  - `FUN_006ceaa0`
  - `FUN_006cee30`
  - `FUN_006cf590`
2. Tighten the semantic meaning of the terrain capability flags and shader families:
  - what `DAT_00cb3594` and `DAT_00ca31b8` mean in public rendering terms
  - whether `_s` is definitively the shadowed path and whether the non-`_s` table is the matching unshadowed one-pass path
  - whether `terrainp_u` / `terrainp_us` are reachable in another terrain branch
3. Continue from the now-confirmed animated-surface path:
  - `FUN_0069b310`
  - `FUN_0069e200`
  - `FUN_0069e300`
  - `FUN_0069e3e0`
  - `FUN_0069e4f0`
  - `FUN_0069e690`
  - `FUN_006c65b0`
4. Determine whether the unresolved liquid family slots are dead or merely still-untraced:
  - `FUN_0069e690(2)` currently resolves into `FUN_0069b310(6)`, but the family slot is not yet backed by a recovered string pointer
  - `XTextures\river\fast_a.%d.blp` exists in the binary but is not yet part of the traced active pointer table
5. Determine whether the Shadowmoon effect is primarily the `WCHUNKLIQUID` animated surface path, a terrain path, or a composite of both.
6. Recover whether the motion is:
   - time-based UV offset
   - projected texture coordinates
   - normal/slope-direction projection
   - screen/world-space perturbation in shader code
7. Trace the runtime layer descriptor that feeds `FUN_006c00f0` so the animated terrain transform bit and its low-bit direction selector can be mapped back to a specific on-disk field with evidence instead of inference.
8. Treat `WDT` / `MAIN` / `MPHD` parsing as a separate viewer-audit task:
  - preserve `MAIN` entry semantics instead of flattening them to boolean tile existence
  - preserve known `MPHD` bits beyond big-alpha
  - preserve known `MCLY` animation/material bits even if render support lands later

### 4.5 Recommended Viewer Implementation Shape If The RE Confirms It

If deeper RE confirms time-based or projected terrain-layer motion, the safest viewer shape is probably:

1. keep the existing terrain alpha/decode path unchanged
2. treat terrain motion and animated chunk-liquid surfaces as separate implementation tracks until the RE proves they are the same thing
3. add an optional terrain material stage only if the terrain-side `terrainp*` work proves true terrain-layer projection/motion exists
4. add a separate animated-surface texture-family path if the `WCHUNKLIQUID` chain proves to be the real Shadowmoon/slime behavior
5. keep all of that version/profile-gated so `0.5.x`, `0.6.0`, and non-animated layers stay untouched
6. make both effects inspectable/toggleable during bring-up so parity work can be validated against real tiles

That approach targets the real likely seam, which is shader/material behavior, not MCAL decode.

## What This Means For The Active `MdxViewer` Roadmap

The current `2.x` profile-routing work was a reasonable structural first step, but it is only that: a structural first step.

The next high-value implementation priorities are:

1. shader/material parity for `Model2` and map-object translucent/specular paths
2. runtime-light interpretation rather than just light-table parsing
3. particle transform/billboard/render-state parity for the smoke bug
4. a dedicated terrain-material follow-up for moving/sliding terrain textures

In other words:

- parser support is necessary
- parser support is not sufficient

## 5. 0.5.3 Render Path And Why The Active Viewer Is Still Slow

The current 0.5.3 pass tightens an important point that was only partially inferred earlier: the old client is not fast because it is "doing less rendering." It is fast because it precomputes more of the terrain/object render setup, keeps specialized render paths alive, and avoids flattening everything into one generic material model.

### 5.1 High-Confidence 0.5.3 Terrain Render Findings

Recent decompilation anchors:

- `CreateRenderLists` at `0x00698230`
- `RenderLayers` at `0x006a5d00`
- `RenderLayersDyn` at `0x006a64b0`
- `RenderLayersColor` at `0x006a6bf0`

What these now support directly:

1. `CreateRenderLists` is a real startup/precompute step, not a trivial wrapper.
  - It builds terrain texcoord tables (`texCoordList`, `texCoordList2`) for the chunk mesh topology.
  - It populates prebuilt runtime render-list / batch tables (`rmTexCoordList`, `rmTexCoordList2`, `rmGxBatchList`-adjacent data) instead of recomputing that layout per draw.
  - It also sorts several index/offset tables up front, which is a strong signal that chunk draw submission is meant to be table-driven at runtime.
2. `RenderLayers` and `RenderLayersDyn` both use locked GX buffers with prebuilt batches rather than rebuilding terrain geometry/material state from scratch every frame.
3. Both terrain paths set up generated/projected texture coordinate transforms (`GxXform_Tex0`, `GxXform_Tex1`) from camera-relative chunk state before layer submission.
4. Terrain in 0.5.3 is not purely fixed-function.
  - When terrain shader support is enabled, the chunk draw path binds `this->shaderGxTexture` and selects `CMap::psTerrain` or `CMap::psSpecTerrain`.
  - When the shader path is unavailable, it falls back to the older alpha/lightmap-style path.
5. Terrain layer count is reduced by distance.
  - `RenderLayers` clamps `nLayersTest` down to `1` when `CWorld::textureLodDist` is exceeded by a large enough margin.
  - `RenderLayersDyn` also fades material diffuse alpha before collapsing to one layer, which is a more nuanced distance transition than the active viewer currently performs.
6. The moving-terrain seam is real in the terrain path itself, not only in `WCHUNKLIQUID`.
  - In both `RenderLayers` and `RenderLayersDyn`, if a layer runtime flag has bit `0x40`, the client applies an extra texture transform using the low flag bits as an index into time-varying world transform tables.
  - This matches the earlier `FUN_006804b0` / `DAT_00c72978..80` evidence and strengthens the conclusion that true moving terrain layers are controlled by per-layer runtime flags.
7. Shadowing is its own terrain pass.
  - If `shadowGxTexture` exists and world shadow rendering is enabled, the client draws an extra modulation pass instead of baking that behavior into one generic terrain fragment path.

Practical read:

- `0.5.3` terrain rendering is already specialized, batch-oriented, and capability-gated.
- The client keeps both a shader-assisted terrain path and a fallback path alive.
- The active viewer's terrain renderer is currently much flatter than the original client.

### 5.2 High-Confidence 0.5.3 Object And Lighting Findings

Recent decompilation anchors:

- `RenderMapObjDefGroups` at `0x0066e030`
- `CreateLightmaps` at `0x006adba0`
- `CalcLightColors` at `0x006c4da0`

What these now support directly:

1. World object rendering is list-driven and group-oriented.
  - `RenderMapObjDefGroups` walks a visible `CMapObjDefGroup` list, sets a world transform once for the group, and dispatches `CMapObj::RenderGroup(...)`.
  - This is more structured than the active viewer's generic per-instance renderer loops.
2. Map-object lighting is not just "sample one directional light."
  - `CreateLightmaps` allocates per-group lightmap textures (`256x256`) and registers an update callback (`UpdateLightmapTex`).
  - This strongly suggests a dedicated lightmap-backed lighting path for map-object groups.
3. The object lightmap path is a real dedicated render seam.
  - `RenderGroupLightmap` locks a dedicated group/lightmap vertex stream, walks group batches, binds the batch lightmap texture, and draws only batches that survive local cull checks.
  - The draw path uses per-group `lightmapVertexList` data rather than reusing the same generic object UV/material path for everything.
4. Lightmap texture submission is callback/latch-based rather than rebuilt ad hoc in the draw loop.
  - `UpdateLightmapTex` exposes row-stride plus CPU lightmap memory on the `GxTex_Latch` command, which fits the earlier `CreateLightmaps` result: the client owns and updates dedicated lightmap textures instead of folding this into one generic fragment shader.
  - `RenderGroupLightmapTex` is also split into internal/external subpasses with lighting forced off and `GxRs_Texture1` cleanup between them, which is more evidence that map-object lightmap composition is its own pipeline.
5. Global/world lighting is richer than the current viewer approximation.
  - `CalcLightColors` computes direct light, ambient light, six sky colors, five cloud colors, four water colors, fog end, fog-start scalar, and additional cloud/storm-related values.
  - It also blends storm/light-override data into the base light state when an override table is active.
6. Combined with the earlier `M2Light` / `CGxuLight` / `CLightList` evidence, this means the client's lighting stack is layered:
  - world light tables
  - runtime light selection/blending
  - object/lightmap integration
  - terrain shader/specular paths

Practical read:

- terrain lighting and object lighting are not one uniform system in the old client.
- map-object rendering is not just "WMO mesh + one shared shader"; it has a group-local lightmap path with separate texture submission/combine work.
- The active viewer's current "one ambient + one directional + fog" model is structurally too thin for parity.

### 5.3 Why The Active Viewer Is Slower And Less Accurate

The current active-tree audit lines up with the RE in an uncomfortable but useful way.

1. `StandardTerrainAdapter` preserves only a narrow slice of WDT semantics.
  - `MPHD` is actively consumed only for big-alpha/profile behavior.
  - `MAIN` entries are flattened to `flags != 0` tile presence instead of preserving `has ADT` vs `all water` semantics.
2. `TerrainRenderer` is a generic four-layer renderer, not a client-faithful terrain material system.
  - It renders base + overlay passes in a simple loop.
  - It only interprets `MCLY 0x100` as the implicit-alpha hint.
  - It has no terrain shader-family split, no per-layer transform/motion support, no layer-count LOD collapse, and no separate specular terrain path.
3. `LightService` is much thinner than the client's runtime light pipeline.
  - It does nearest-zone selection plus interpolation from `Light.dbc` / `LightData.dbc`.
  - It does not model runtime light bucketing, storm/light override blending beyond the DBC approximation, light links, or the richer sky/cloud/water channels seen in `CalcLightColors`.
4. `WmoRenderer` and `MdxRenderer` still flatten renderer specialization heavily.
  - Each renderer uses one shared static shader program family instead of the client's multiple terrain / map-object / translucent / specular / env / lightmap paths.
  - The object side therefore cannot yet reproduce the original specialization that the client appears to rely on for both correctness and speed.
5. `WorldScene` is still doing a lot of expensive generic work on the hot path.
  - WMO and MDX rendering are walked instance-by-instance each frame.
  - transparent MDX instances are re-collected and sorted every frame.
  - all unique MDX renderers are animation-updated before passes.
6. Optional debug/forensics systems can still dominate frame cost when enabled.
  - PM4 overlay budgets are currently `int.MaxValue` for lines, triangles, and position refs.
  - That is useful for forensic work, but it is the opposite of the original client's conservative draw budgeting.
7. GPU resource creation still happens on the viewer side in places where the client appears to rely on longer-lived prepared state.
  - Terrain alpha/shadow textures are uploaded per chunk.
  - Terrain/material/light behavior is then rebuilt through a generic pass loop rather than a client-like prepared render-list path.

### 5.4 Practical Priority Order For Viewer Recovery

If the goal is both performance and historical accuracy, the most defensible order is:

1. Preserve the missing terrain/world metadata first.
  - Keep `MAIN` entry semantics.
  - Preserve useful `MPHD` bits beyond big-alpha.
  - Preserve the important `MCLY` animation/material bits as first-class runtime metadata.
2. Split the terrain renderer into at least two conceptual paths.
  - a simpler fallback path
  - a shader/material path that can eventually model the old client's layer-count, shadow/specular, and animated-layer behavior
3. Stop treating object lighting as the same problem as terrain lighting.
  - Map-object/lightmap behavior needs its own investigation and implementation seam.
  - `WmoRenderer` should not be expected to converge just by tweaking the current generic shader.
4. Tighten hot-path render work before adding more fidelity features.
  - reuse more prepared draw state
  - reduce generic per-frame sorting/state churn where possible
  - keep heavy PM4/debug overlays outside the normal world-render budget unless explicitly enabled
5. Keep lighting parity scoped as a real subsystem task.
  - The client computes more channels than the current viewer exposes.
  - The viewer should not claim lighting parity until terrain, object, and sky/light data paths are treated separately.

### 5.5 Concrete Recovery Order In The Active Viewer

If the next work is implementation rather than more archaeology, the most defensible file order is:

1. Metadata preservation slice.
  - `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
  - goal: stop collapsing `MAIN`, stop treating `MPHD` as only big-alpha/profile guidance, and surface the important `MCLY` bits as explicit runtime metadata instead of leaving them as opaque integers.
  - reason: every later terrain/material decision depends on this information surviving decode.
2. Terrain path split slice.
  - `src/MdxViewer/Terrain/TerrainRenderer.cs`
  - `src/MdxViewer/Terrain/TerrainChunkData.cs`
  - `src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs`
  - goal: separate the current generic fallback loop from a client-faithful prepared/material path that can later absorb layer-count collapse, shadow/specular branches, and animated-layer transforms.
  - reason: this is the closest structural analogue to `CreateRenderLists` plus `RenderLayers` / `RenderLayersDyn`.
3. Hot-path render-loop cleanup slice.
  - `src/MdxViewer/Terrain/WorldScene.cs`
  - `src/MdxViewer/Rendering/ModelRenderer.cs`
  - `src/MdxViewer/Rendering/WmoRenderer.cs`
  - goal: reduce per-frame resorting/state churn, keep optional PM4/debug work bounded, and stop paying full unbatched costs except where the asset really requires it.
  - evidence from the active tree:
    - `WorldScene` still rebuilds/sorts `_transparentSortScratch` every frame and runs two MDX visibility walks around `BeginBatch(...)`
    - `ModelRenderer.RenderGeosets(...)` still sorts `_transparentGeosetOrder` per transparent draw and keeps material/state selection local to each renderer instance
    - `WmoRenderer` still does its own `_transparentGroupSortScratch` sort and per-batch material binding, while doodad rendering stays on the expensive per-instance `RenderWithTransform(...)` path
  - note: `src/MdxViewer/Rendering/RenderQueue.cs` exists but is not yet the active world-render submission path, so it should not be treated as proof that submission/state grouping is solved.
4. Object/lightmap parity slice.
  - `src/MdxViewer/Rendering/WmoRenderer.cs`
  - `src/MdxViewer/Terrain/LightService.cs`
  - goal: treat map-object lightmaps as their own seam instead of expecting the current generic WMO shader plus one ambient/directional light model to converge.
  - reason: the `RenderGroupLightmap` / `RenderGroupLightmapTex` / `UpdateLightmapTex` cluster is a stronger object-lighting specialization than the active viewer currently models.
5. Asset/cache policy slice.
  - `src/MdxViewer/Terrain/WorldAssetManager.cs`
  - goal: keep using the existing read/path-probe stats, but turn them into an explicit scene residency/prefetch policy rather than a passive cache.
  - reason: the viewer already measures read requests, resolved-path cache hits, and path probes; the next step is to use those counters to cut redundant work in the active scene.

### 5.6 Immediate Engineering Consequences

The current viewer is not "just missing a few flags." It is missing several structural choices the old client already made:

1. prepared terrain render lists instead of generic per-layer pass logic
2. terrain shader/material specialization instead of one flattened terrain shader path
3. object-group lightmaps and richer world-light blending instead of one simplified light service
4. stronger separation between terrain, object, liquid, and debug rendering costs

That is the real reason the comparison to a 23-year-old client looks bad right now. The old client is specialized; the active viewer is still too generic.

## Fresh-Chat Entry Points

Any future `2.0.0` / `2.x` parity session should read, in order:

1. `memory-bank/activeContext.md`
2. `memory-bank/progress.md`
3. this guide
4. `src/MdxViewer/memory-bank/activeContext.md`

Then it should answer these questions before touching code:

1. Is the problem format routing, shader/material selection, light runtime behavior, particle runtime behavior, or terrain material behavior?
2. Is the current change localized enough to avoid regressions in `0.5.x` / `0.6.0` / later `3.x` / `4.x` paths?
3. What real data or runtime check will be used before claiming the result is correct?