# Phase 0 Research: World Context And Lighting Parity

## Decision 1: Treat the current AreaName failure as a context-contract bug first

**Evidence**

- `src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs` already stores the low 16 bits of Alpha
  MCNK `Unknown3` as `AreaId`.
- `src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs` already stores standard MCNK
  `Header.AreaId`.
- `src/core/WowViewer.Core.IO/Lk/Mcnk.cs` reads the standard header area field at its existing
  format-defined offset.
- `TerrainRenderer.GetChunkAt(float worldX, float worldY)` accepts viewer coordinates, while the
  status path directly passes `_camera.Position.X/Y` and then applies a map-filtered AreaTable
  lookup.
- `AreaTableService` already loads DBCD storage and detects logical columns, but the status path
  collapses a missing chunk, zero ID, map mismatch, and unresolved row into an empty name.
- The Alpha adapter previously truncated `Unknown3` to its low 16 bits, while the checked-in Alpha
  reference describes the full packed `AreaNumber` as zone high-word plus subzone low-word.
- `TerrainRenderer.GetChunkAt` previously returned the nearest resident chunk after exact/bounds
  lookup failed, which could assign an unrelated area's name while the camera was over an unloaded
  tile. The status path also queried that legacy GPU-mesh list even though the active batched terrain
  path stores resident chunk metadata in `_chunkInfosByTile`, producing `NoTerrainChunk` over visible
  terrain.

**Decision**: Phase 1 introduces a structured lookup result and instruments the coordinate,
tile/chunk, raw ID, map ID, table row, and unresolved reason before changing lookup semantics. The
existing parsers remain the source of raw IDs. The status path consumes resident chunk metadata so
batched tile rendering and context selection share the same residency authority. The map filter will
be treated as validation/context, not as permission to erase a valid row; a map mismatch will remain
visible as a diagnostic state.

**Alternatives considered**

- Hardcode known map or zone names: rejected because it violates the client-data contract.
- Add another AreaTable parser: rejected because DBCD and existing shared readers already provide
  the input.
- Remove map validation entirely: rejected because it would hide cross-build ID collisions; instead,
  expose both the resolved row and map-validation result.

## Decision 2: Use logical DBD fields and retain source metadata

**Evidence**: `AreaTableService` already discovers `AreaName_lang`, `AreaName`, `Name`, `ID`,
`AreaID`, `AreaNumber`, parent, map, and flag columns from `storage.AvailableColumns`. Existing
`AreaIdMapper` also demonstrates DBCD-backed field access and cross-era fallback behavior.

**Decision**: The runtime contract will carry the loaded build, locale, logical ID/name/map/parent
column names, raw table key, canonical ID, and alias/collision status. Numeric field positions and
machine-local client paths are not part of the implementation contract.

**Alternatives considered**

- Make `AreaTableService.GetAreaDisplayNameForMap` return only a string: rejected because it cannot
  distinguish no chunk, zero ID, map mismatch, row miss, and missing localized text.
- Use the historical crosswalk for all builds: rejected because this viewer is resolving the active
  client's own MCNK ID, not converting between unrelated source eras.

## Decision 3: Do not invent WMOAreaID until the field is proven per WMO profile

**Evidence**

- The current shared WMO root summary decodes MOHD counts, bounds, and flags, but does not expose a
  WMO area field.
- The current WMO group summary decodes MOGP name offsets, flags, bounds, portal ranges, batches,
  liquid, and other existing fields, but does not expose a proven area field.
- The WMO render document is assembled from those existing readers; no current model contract
  carries `WMOAreaID`.
- Legacy and modern WMO root/group layouts differ, and the repository explicitly forbids speculative
  parser rewrites.

**Decision**: Phase 0 must identify the exact source chunk/offset and version/profile scope using
  existing reference code, DBD/client evidence, or a real WMO fixture before a field is added. The
  design will represent this as `WmoAreaIdEvidence` with source kind, raw value, profile, and
  confidence. If a profile has no proven field, the runtime reports `UnavailableForProfile` and uses
  ADT context. A filename, WMO display name, group index, or guessed header offset is never accepted
  as an area ID.

**Alternatives considered**

- Assume WMOAreaID is always in the same MOGP header slot: rejected until real fixtures prove the
  slot across supported versions.
- Derive the area from the nearest terrain chunk: retained only as the explicit ADT fallback, never
  as WMO provenance.
- Add a second speculative WMO parser: rejected; extend the existing reader only after the field
  contract is proven.

## Decision 4: Make context camera-owned and same-frame

**Evidence**: `Rendering.Camera` already owns position, yaw, pitch, forward vector, and view matrix.
The viewer separately consumes the camera for rendering and uses the position for terrain area
lookup. Taxi/cockpit camera code provides an existing precedent for explicit modes and offsets, but
it is not a player-head contract.

**Decision**: Add an explicit camera/head state around the existing camera rather than replacing all
input behavior. It contains eye position, yaw, pitch, mode, and an explicit head offset. The frame
coordinator obtains one snapshot and passes it to view construction, context selection, visibility,
fog, and lighting. Museum view is a named mode with a visible offset; it is not an implicit hidden
translation.

**Alternatives considered**

- Replace the free-fly camera with a gameplay controller: rejected as scope creep.
- Compute context from the prior frame while rendering from the current frame: rejected because it
  causes visible one-frame identity/lighting transitions.
- Add collision now: rejected; collision is a future feature and not needed to prove head semantics.

## Decision 5: Start lighting with attributable existing inputs

**Evidence**

- `WmoRenderer` already uploads directional, ambient, and light color uniforms and consumes WMO
  vertex-light/baked-weight attributes.
- WMO data structures already carry root ambient color, light records, group light references,
  vertex colors, and lightmap-related data in the render path, but the shader currently uses generic
  half-Lambert defaults when no scene inputs are supplied.
- `M2Renderer` already receives ambient/directional uniforms, but the current contract does not yet
  prove build-specific local-light or effect selection.
- Spec 106 owns native day/night lighting and Spec 138 owns the 4.x evidence/profile renderer
  matrix. Memory research identifies client-derived `LightFloatBand.dbd` and `LightService` as
  relevant evidence, not permission to guess values.

**Decision**: Phase 4 first creates a named lighting-input selection and diagnostics for WMO and
MDX/M2. It uses actual scene/profile values when available, otherwise an explicitly labeled
equivalent fallback. Shader changes are limited to consuming selected inputs and preserving baked
contributions; BLS porting is a separate evidence gate, not a prerequisite or hidden claim.

**Alternatives considered**

- Increase ambient brightness to make interiors look better: rejected because it changes appearance
  without restoring source lighting.
- Apply one global directional light everywhere: rejected because it is the current flat-lit failure
  mode.
- Port all BLS shaders immediately: rejected because the supported effect contract and inputs are not
  yet proven for every era.

## Decision 6: Measure performance at frame-stage and ownership boundaries

**Evidence**: Recent user captures showed scene-maintenance, WMO/MDX visibility, and overlay owners
can dominate frames independently. The graph path is an investigation path, not the default renderer.

**Decision**: Context lookup is evaluated through existing frame statistics and new context/lighting
diagnostics, with no whole-map residency changes in this feature. User-run profiles compare the flat
baseline and the feature path by phase and p95; the agent only performs focused builds/tests.

**Alternatives considered**

- Use visual smoothness as the performance proof: rejected because it cannot attribute regressions.
- Re-enable the hierarchical graph as part of context work: rejected; Spec 142 owns that boundary.

## Decision 7: Model the native UI display as `SubzoneText`, not as an arbitrary parent-chain label

**Evidence**

- The local 3.3.5 client reference inventory exposes `lua_GetSubZoneText` at the same UI API layer
  as `lua_GetZoneText` and `lua_GetMinimapZoneText`.
- The current viewer only stores `_currentAreaName` and formats a synthetic parent chain through
  `AreaTableService.GetAreaDisplayName`; it has no explicit subzone/zone display roles.
- The repository's AreaTable DBD contains `AreaName_lang`, `ParentAreaID`/`ParentAreaNum`,
  `ContinentID`, `Flags`, and in older layouts `AreaNumber`. The checked-in flag definitions include
  `IS_SUBZONE`, confirming that subzone-ness is client data rather than a presentation guess.
- The Alpha reference decode documents that 0.5.x `AreaNumber` and `ParentAreaNum` can pack zone and
  subzone halves. It also warns that naive parent traversal or trusting an unanchored continent can
  select the wrong hierarchy.

**Decision**: The world-context contract exposes both logical display roles: `ZoneText` for the
resolved parent zone and `SubzoneText` for the resolved leaf/subzone, with a deterministic fallback
to the zone when no valid leaf is present. The status bar shows `SubzoneText` as the primary
native-style area label and retains raw IDs, parent chain, and source diagnostics beside it. This
does not claim that a UI API call alone reveals the AreaID; ADT/WMO and AreaTable resolution remain
the source of truth.

**Alternatives considered**

- Display `Parent > Child` as the only value: rejected because it is an editor-specific format, not
  the game's `SubzoneText` role.
- Use only the leaf row name: rejected because zone-only records and missing subzone data need the
  same fallback behavior as the client UI.
- Treat `SubzoneText` as a new DBC column: rejected; it is a UI result derived from area context,
  parent relationships, flags, and build-specific packing.

## Decision 8: Decode the observed pre-alpha version-2 partial LIT shape separately

**Evidence**

- The configured `H:\\053-client` archive contains
  `World\\Maps\\Azeroth\\areatest.lit` with a 5,324-byte payload, version `0x00000002`, and raw
  light count `-1`.
- The first 64 bytes after the file header are not a modern group-length array. They contain an
  embedded `Global Light` header: three signed chunk fields, four scalar fields, a 32-byte name,
  and a reserved word. The first two chunk fields and the file count are `-1`; the third chunk
  field is `0` in this observed file.
- The remaining `0x1484` bytes contain a 60-byte legacy prefix followed by two consecutive
  `0xA24` data sets. Each data set has nine 32-slot time/BGRX tracks and two 32-sample float
  arrays. The primary set has track lengths `4,4,4,4,4,4,4,4,3`; the second set is retained but
  its semantic selector is not established.
- The previous reader interpreted the embedded header's first `-1` as track 0's length, which
  produced the reported `expected 0..32` parse failure before `LitLoader.Version` was assigned.

**Decision**: Add a version-2 negative-count layout profile in `LitProfileReader` rather than
loosening modern track validation. Decode the embedded header with its observed pre-alpha field
shape, skip the legacy prefix without inventing field names, expose the primary set as
`LitLightGroupKind.Partial`, and expose the second as `LegacyPartialAlternate` for inspection.
The primary set alone drives the global partial-light selection. This is an observed compatibility
slice, not a claim that every version-2 client uses the same payload; additional v2 variants require
their own evidence and fixture before acceptance.

## Decision 9: Keep the interactive 0.5.3 clock separate from minimap synthesis

**Evidence**

- LightService already documents the native time domain as 0..2880, with 1440 at noon in a
  24-minute cycle.
- LitProfileReader independently defines 2880 time units per day for LIT keyframes. These existing
  contracts agree on the conversion used by this slice.
- The harvest tool accepts an explicit --time-hours value and uses a separate achromatic minimap
  lighting profile; it does not run the viewer frame loop or consume LightService.

**Decision**: Add a pure WorldTimeCycle conversion/advance contract and drive it from the interactive
WorldScene monotonic frame clock. TerrainLighting is the same-frame authority for global lighting,
Light DBC overlays, LIT samples, sky, and audio time. Manual UI input freezes that clock. Synthetic
minimap generation remains frozen at its requested time and writes timeOfDayMode=frozen to its manifest.

**Open evidence**: The suspicion that shipped authored minimaps were captured while their time-of-day
clock was moving remains unproven. The manifest boundary prevents new synthesis nondeterminism; an
authored-minimap tint audit still requires user-run client/output comparison and is not inferred from
this implementation.

**Validation**

- `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore --filter FullyQualifiedName~LitProfileReaderTests`: 8 passed.
- `dotnet build wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug --no-restore --no-dependencies`: passed with the existing warning set.
- `lit profile --archive-root H:\\053-client --virtual-path World\\Maps\\Azeroth\\areatest.lit`: decoded
  `Global Light`, version `2`, raw count `-1`, track count `9`, stride `0x1484`, and primary
  `Partial` samples successfully. Viewer visual/runtime proof remains user-owned.


---

## Native lighting and M2 shader evidence — 5.0.1.15464 (recorded 2026-09-01)

Read from `Wow.exe` (MoP Beta 5.0.1.15464) via GhidraMCP. Addresses are image-base
`0x00400000`.

### 1. The engine has indexed lights, not one sun

The Lua binding signatures name the per-light parameter set exactly:

```
SetLight(enabled[, omni, dirX, dirY, dirZ, ambIntensity[, ambR, ambG, ambB],
                              dirIntensity[, dirR, dirG, dirB]])          @ 0x00d630b8
AddLight(index, enabled[, omni, ...])                                     @ 0x00d937a0
AddCharacterLight(index, enabled[, omni, ...])                            @ 0x00d93828
```

So each light carries: enabled, an **omni/directional flag**, a direction, an ambient
intensity + RGB, and a diffuse intensity + RGB. `AddLight` is **indexed** — a set, not a
singleton — and characters get their **own** light set on top.

Our M2 path has one directional light and one ambient term
(`M2Renderer.cs:844-846`: `uLightDir`, `uLightColor`, `uAmbientColor`), with no omni
support, no light index, and no character light set.

### 2. Zone lights are polygonal regions — we do not implement them at all

`DNZoneLight.cpp` builds a sorted, binary-searchable table of **40-byte (0x28) zone light
records** (`FUN_00c26090` searches it; `FUN_00c26900` builds it):

| Offset | Meaning |
|---|---|
| `+0x00` | zone light id (sort key) |
| `+0x04` | point count |
| `+0x08`, `+0x0c` | running min X, min Y |
| `+0x10`, `+0x14` | running max X, max Y |
| `+0x1c` | point array capacity |
| `+0x20` | point array pointer |

It is populated from two client DBs: the zone light table, then a **zone light point**
table whose records carry a parent zone-light id at `+0x04`, an X/Y at `+0x08`/`+0x0c`, and
a 1-based **point order** at `+0x10` (asserted `>= 1` and `<= capacity`,
`DNZoneLight.cpp:0x99` "invalid point order for zone lights"). Points are written into the
parent's array at `order - 1`, and the parent's AABB is accumulated as each point lands.

That is an **ordered polygon per zone with a derived bounding box** — spatial lighting
regions that override the global day/night light. `ZoneLight`/`ZoneLightPoint` have **zero
references anywhere in our codebase**, which is the single largest missing piece of "the
real lighting."

We do already model the DBC side of the global lighting: `BuildScopedLightDbcModels`
carries `Light.dbc`, `LightParams` (`DBFilesClient\\LightParams.dbc` @ `0x00e0628c`),
18 `LightIntBand` colour tracks and 6 `LightFloatBand` float tracks.

### 3. The M2 shader table — 17 vertex x ~30 pixel permutations

The client selects a shader **pair** per batch. Both halves are named in the binary:

**Vertex (`Diffuse_*`, @ `0x00d722c4`-`0x00d72858`)** — `T1`, `T2`, `T1_T2`, `T1_Env`,
`Env_T1`, `Env_Env`, `Env`, `T1_T1`, `T1_T1_T1`, `T1_Env_T1`, `T1_Env_T2`, `T1_T2_T1`,
`T1_T1_T1_T2`, `EdgeFade_T1`, `EdgeFade_Env`, `EdgeFade_T1_T2`.

**Pixel (`Combiners_*`, @ `0x00d7286c`-`0x00d72bcc`)** — `Mod`, `Opaque`, `Mod_Mod`,
`Mod_Mod2x`, `Mod_Add`, `Mod_AddNA`, `Mod_Mod2xNA`, `Mod_Opaque`, `Mod_Depth`,
`Mod_AddAlpha`, `Mod_AddAlpha_Alpha`, `Mod_AddAlpha_Wgt`, `Mod_Add_Alpha`,
`Mod_Dual_Crossfade`, `Mod_Masked_Dual_Crossfade`, `Opaque_Alpha`, `Opaque_Alpha_Alpha`,
`Opaque_Opaque`, `Opaque_Mod`, `Opaque_Mod2x`, `Opaque_Mod2xNA`, `Opaque_ModNA_Alpha`,
`Opaque_AddAlpha`, `Opaque_AddAlpha_Alpha`, `Opaque_AddAlpha_Wgt`, `Opaque_Mod_Add_Wgt`,
`Opaque_Mod2xNA_Alpha`, `Opaque_Mod2xNA_Alpha_Add`, `Opaque_Mod2xNA_Alpha_Alpha`,
`Opaque_Mod2xNA_Alpha_3s`, `Opaque_Mod2xNA_Alpha_UnshAlpha`.

WMO has its own smaller set: `MapObjDiffuse`, `MapObjSpecular`, `MapObjTwoLayerDiffuse`,
`MapObjDiffuseEmissive`, `MapObjTwoLayerDiffuseOpaque`, `MapObjTwoLayerDiffuseEmissive`
(@ `0x00dee4b4`-`0x00dee5c8`).

Our renderers carry **no permutation system at all** — a search for `Combiners_` or
`Diffuse_T1` across `src/` returns nothing. Every M2 batch goes through one program, so the
per-batch texture-combiner work the client resolves at shader-select time is either done on
the CPU or not done. This is the concrete form of "more shader programs to reduce CPU
compute."

### 4. What this means for sequencing

- Spec 136 (doodad performance) is **9/11**, and both open tasks (T008, T011) are
  user-owned measurement, not code. GPU instancing exists and is wired
  (`IGpuInstancedModelRenderer`, used from `WorldScene.cs:10940`).
- The shader-permutation gap is a **separate, unspecified lane** from 136's batching work.
  136 reduces draw calls; permutations reduce per-batch CPU material work and are what make
  doodads look right.
- Zone lights are a data + spatial-query feature, largely independent of both.
