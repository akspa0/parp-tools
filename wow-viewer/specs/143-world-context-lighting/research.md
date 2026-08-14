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

**Validation**

- `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore --filter FullyQualifiedName~LitProfileReaderTests`: 8 passed.
- `dotnet build wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug --no-restore --no-dependencies`: passed with the existing warning set.
- `lit profile --archive-root H:\\053-client --virtual-path World\\Maps\\Azeroth\\areatest.lit`: decoded
  `Global Light`, version `2`, raw count `-1`, track count `9`, stride `0x1484`, and primary
  `Partial` samples successfully. Viewer visual/runtime proof remains user-owned.
