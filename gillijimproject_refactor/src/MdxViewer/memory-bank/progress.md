# Progress — AlphaWoW Viewer (MdxViewer)

## Status: v0.4.0 Released — 0.5.3 Rendering Improvements + 3.3.5 Groundwork (In Progress)

**Supported client versions: 0.5.3 through 0.12** — fully usable
**3.3.5 WotLK: IN PROGRESS** — scaffolding exists but MH2O liquid and terrain texturing are broken

## What Works Today

| Feature | Status |
|---------|--------|
| Terrain rendering + AOI lazy loading | ✅ (AOI radius=3, 7×7 tiles, 4 uploads/frame) |
| Terrain MCSH shadow maps | ✅ Applied on ALL layers (not just base) |
| Terrain alpha map debug view | ✅ Show Alpha Masks toggle, Noggit edge fix |
| Terrain fog-based chunk culling | ✅ Skip chunks beyond FogEnd+200 |
| Terrain liquid rendering | ✅ Water/lava/slime (WMO MLIQ + terrain + MH2O) |
| **WotLK 3.3.5 terrain support** | 🔧 In progress — split ADT, MPHD flags parsed. **MH2O broken, texturing broken** |
| WDL parser (MVER/MAOF/MARE, v0x12) | ✅ Strict parsing + version validation |
| WDL terrain tile scale | ✅ Uses TileSize (8533.3333), not ChunkSize |
| WDL preview window reliability | ✅ Error reporting + `.wdl.mpq` fallback |
| WDL spawn chooser | ❌ Reported non-functional across tested versions (Mar 20 runtime report) |
| WDL runtime visibility toggle | ✅ UI checkbox for testing overlap issues |
| Async tile streaming | ✅ Background parse, render-thread GPU upload, max 2/frame |
| Standalone MDX rendering | ✅ MirrorX for LH→RH, front-facing, textured |
| **MDX skeletal animation** | ✅ Compressed quats, GPU skinning, standalone + terrain |
| MDX pivot offset correction | ✅ BB center pre-translation for correct placement |
| MDX blend modes + depth mask | ✅ Transparent layers don't write depth |
| MDX fog blending | ✅ Models blend into fog like terrain |
| MDX doodads in WorldScene | ✅ Position + animation + particles working |
| **MDX particle effects (PRE2)** | ✅ Billboard quads, texture atlas, bone-following, per-emitter blend |
| **Geoset animation alpha (ATSQ)** | ✅ Per-frame keyframe evaluation, alpha modulation |
| **M2 (MD20) model loading** | ✅ WarcraftNetM2Adapter: MD20→MdxFile conversion |
| **Half-Lambert lighting** | ✅ Softer shading on MDX + WMO (no harsh black shadows) |
| **Improved ambient lighting** | ✅ Day/night with WoW-like brightness levels |
| WMO v14 loading + rendering | ✅ Groups, BLP textures per-batch |
| WMO fog blending | ✅ WMOs blend into fog like terrain |
| WMO liquid rendering (MLIQ) | ✅ Semi-transparent water surfaces |
| WMO intermittent render race | ✅ Fixed via shared static shaders + ref counting |
| WMO doodad sets | ✅ Loaded and rendered with WMO modelMatrix |
| WMO rotation/facing in WorldScene | ✅ Fixed — `-rz` negation for handedness |
| MDDF/MODF placements | ✅ Position + pivot correct |
| Bounding boxes | ✅ Actual MODF extents with correct min/max swap |
| Batched overlay rendering | ✅ POI pins + taxi paths in single draw call |
| Minimap zoom (4 tiles around camera) | ✅ |
| TaxiPath visualization | ✅ DBC-loaded flight paths as 3D lines |
| Taxi path selection (sidebar) | ✅ |
| POI + Taxi lazy-load UI | ✅ Load buttons → toggle checkboxes after load |
| AreaID/MapID-aware area names | ✅ Filters by current map, warns on mismatch |
| NoCullRadius (150 units) | ✅ Nearby objects skip frustum cull |
| VLM terrain loading | ✅ JSON dataset → renderer |
| VLM minimap | ✅ Works for VLM projects |
| VLM dataset generator | ✅ File > Generate VLM Dataset (background export) |
| BLP2 texture loading | ✅ DXT1/3/5, palette, JPEG |
| MPQ data source | ✅ Listfile, nested WMO/WDT/WDL archives |
| DBC integration | ✅ DBCD, CreatureModelData, CreatureDisplayInfo, TaxiPath, AreaPOI, AreaTable |
| Camera | ✅ Free-fly WASD + mouse look |
| ImGui UI | ✅ File browser, model info, visibility toggles |
| Live minimap + click-to-teleport | ✅ WDT + VLM |
| AreaPOI system | ✅ DBC loading, 3D markers, minimap markers, UI list |
| Object picking/selection | ✅ |
| PM4 debug overlay | 🔧 In progress — viewer-side PM4 surfaces + color modes + MPRL/centroid markers + CK24 split controls + parity-aware winding correction; runtime visual signoff still pending |
| GLB export | ✅ MDX + WMO, Z-up → Y-up conversion |
| Thread safety | ✅ ConcurrentDictionary for TileTextures, locks for placement dedup |
| **Asset Catalog** | ✅ SQL dump parser (no MySQL), browse/search/filter, JSON+GLB+screenshot export |
| **Loading screen** | ✅ BLP-based with progress bar |

## 2026-03-20 - WDL Spawn Chooser Regression Handoff

- Runtime status correction: WDL heightmap spawn chooser is currently reported non-functional across tested versions.
- This is now an active blocker for the WDL spawn workflow.
- Prior implementation notes about `Spawn` gating/fallback should be treated as historical code intent until runtime behavior is revalidated.
- Required next-pass investigation focus:
	- map list `Spawn` enablement state versus warm-state lifecycle
	- chooser open path and spawn-commit callback
	- preview-failure fallback behavior (`OpenWdlPreview`/dialog path) and whether it bypasses chooser state incorrectly
- Required closure evidence:
	- real-data runtime verification on at least one Alpha-era map and one 3.x map
	- proof that selected spawn is actually applied
- Validation limits in this handoff update:
	- no code changes made
	- no automated tests added or run

## 2026-03-20 — PM4 MSLK Grouping + Per-Object Keying Follow-up

- `WorldScene` PM4 object assembly now uses `MSLK` relationships to split CK24 buckets into linked sub-groups before optional MDOS/connectivity splitting.
- PM4 object identity is no longer `tile + CK24` only:
	- selection, lookup, and per-object translation keys now include a per-component `objectPart` id
	- this avoids key collisions/overwrites when one CK24 resolves to multiple linked components
- Planar/orientation solve now uses linked `MPRL` anchors at CK24 scope with one shared transform per CK24:
	- gathers linked `MPRL` refs from `MSLK` entries across the CK24 surface set
	- keeps nearest-anchor distance scoring plus MPRL-rotation-aware tie-breaking using the candidate's planar principal axis
	- keeps `MSLK` surface linkage preferring `MsurIndex` over ambiguous `RefIndex` surface matching
	- avoids per-linked-component transform divergence that can spin parts of one CK24 object differently
- Selected PM4 debug metadata now includes:
	- `ObjectPartId`
	- dominant `MSLK.GroupObjectId` (when available)
	- linked `MPRL` ref count used by the orientation solve
- `ViewerApp` PM4 UI/pick text now displays the per-component part id and MSLK group id for triage.
- Build validation:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
- Validation limits:
	- no automated tests were added or run
	- no runtime real-data PM4 visual signoff yet on the reported split/rotated structures

## 2026-03-20 — PM4 Tile Assignment Normalization + Sparse Merge Guardrail

- Addressed PM4 tile drift/collision risk in `WorldScene` overlay loading:
	- removed MPRL-driven tile reassignment from parsed PM4 filenames
	- filename coordinates now map directly by index (`x_y` -> `tileX=x`, `tileY=y`)
- Added duplicate-tile merge behavior:
	- PM4 objects append instead of replacing prior tile payload
	- PM4 tile stats and position-ref markers accumulate instead of overwrite
	- merged object part ids are rebased to preserve `(tile, ck24, objectPart)` uniqueness
- Build validation:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
- Validation limits:
	- no automated tests were added or run
	- no runtime real-data validation yet for the specific tile mismatch report (`00_00` / `01_00` / `1_1`)

## 2026-03-20 — PM4 Post-Restart Runtime Validation Handoff

- User requested system restart before runtime checks due to performance constraints.
- Pending first-pass runtime validation order after restart:
	1. confirm PM4 for `00_00.pm4` aligns to ADT tile `(0,0)`
	2. confirm tile directly below (`01_00`) no longer appears shifted into `01_01`
	3. confirm sparse/missing PM4 tiles stay empty rather than causing neighbor drift
	4. confirm no data-loss behavior from duplicate tile mapping (merge path active)
- Implementation state at handoff:
	- filename tile mapping normalized to direct filename indices (`x_y -> tileX=x, tileY=y`)
	- MPRL-based tile reassignment removed in PM4 viewer load path
	- duplicate tile payloads merge instead of overwrite

## 2026-03-20 — PM4 Runtime Orientation Consolidation Checkpoint

- Runtime update after reverting per-linked-subgroup transform solving:
	- CK24 object parts now stay together as coherent objects (major regression rollback success)
	- residual orientation is now a consistent global offset of about 90 degrees counter-clockwise in tested scenes
- Interpretation:
	- grouping/splitting and object-part keying are materially improved
	- remaining issue is likely a single yaw-basis correction, not random per-part transform divergence
- Documentation artifact added:
	- `documentation/pm4-current-decoding-logic-2026-03-20.md`
	- captures active PM4 chunk usage, assembly pipeline, transform logic, and current residual offset status

## 2026-03-20 — PM4 Yaw Basis Tie-Break Fallback

- Applied a minimal orientation-solver follow-up in `WorldScene.ResolvePlanarTransform(...)`:
	- yaw tie-break now keeps direct MPRL yaw comparison as primary
	- adds an explicit quarter-turn fallback (`expectedYaw +/- 90°`) only when it clearly out-scores the direct comparison
- Scope intentionally stays narrow:
	- no change to CK24 grouping, MSLK splitting, object-part identity, or tile mapping
	- no change to centroid/footprint distance scoring terms
- Goal of this slice:
	- handle the currently observed global ~90° PM4 orientation offset without reintroducing part-fragmentation regressions
- Validation status:
	- build/runtime validation still pending in this checkpoint entry
	- no automated tests were added or run

## 2026-03-20 — PM4 MPRL Orientation Weighting Follow-up

- Follow-up solver adjustment in `WorldScene.ResolvePlanarTransform(...)` after runtime report that objects were still rotated wrong:
	- yaw-basis fallback now evaluates direct, sign-flipped, and quarter-turn variants (`expected`, `-expected`, `expected +/- 90°`, `-expected +/- 90°`)
	- when linked-footprint scoring is active, strong yaw agreement can now override modest distance-score differences
- Intent of this slice:
	- treat MPRL as authoritative orientation linkage more strongly for transform selection
	- keep prior CK24 cohesion/object-part/tile-mapping guardrails unchanged
- Validation status:
	- no automated tests were added or run
	- runtime signoff still pending

## 2026-03-20 — PM4 MPRL Yaw Decode Basis Correction

- Follow-up after runtime report that PM4 objects were still 90 degrees clockwise:
	- `TryComputeExpectedMprlYawRadians(...)` now decodes packed MPRL low-16 rotation as clockwise and rebases it by +90 degrees before circular averaging
	- this directly targets the observed consistent global yaw-basis mismatch while keeping CK24 grouping/tile mapping logic unchanged
- Validation status:
	- no automated tests were added or run
	- runtime signoff still pending

## 2026-03-20 — PM4 Tile Drift Follow-up (Per-CK24 Coordinate Mode)

- Follow-up after runtime report that many PM4 tiles/object payloads appeared shifted southeast/off-map:
	- `WorldScene.BuildPm4TileObjects(...)` no longer relies on one file-level tile-local/world-space decision
	- coordinate mode is now selected per CK24 via `ResolveCk24CoordinateMode(...)` using MPRL fit quality
	- the chooser compares tile-local vs world-space placement by weighted footprint/centroid score against linked/all MPRL refs
- Intent:
	- prevent incorrect tile-offset application when PM4 coordinate mode differs across objects/files
	- keep deterministic filename tile assignment and no MPRL tile-reassignment heuristic
- Validation status:
	- no automated tests were added or run
	- runtime signoff still pending

## 2026-03-20 — PM4 Continuous CK24 Yaw Correction Follow-up

- Follow-up after runtime report that PM4 orientation was still coherently off by about 80 degrees:
	- `WorldScene` now computes a signed yaw delta at CK24 scope (`TryComputeWorldYawCorrectionRadians(...)`) from principal-axis yaw vs MPRL-derived yaw using basis/parity fallback
	- PM4 geometry conversion now applies this as a world-space rotation around the CK24 world centroid before renderer conversion (`RotateWorldAroundPivot(...)`)
	- edge/triangle generation paths share the same correction inputs so line and solid overlays stay coherent
- Intent:
	- keep deterministic tile assignment and per-CK24 coordinate mode logic unchanged
	- resolve the residual coherent yaw offset without reintroducing per-part divergence
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
	- no automated tests were added or run
	- runtime visual signoff still pending

## 2026-03-20 — PM4 Interchange Export + Tile 22_18 Data Confirmation

- Added PM4 interchange export in viewer:
	- `WorldScene.BuildPm4OverlayInterchangeJson(...)` emits summary + tile/object metadata and optional geometry payload (lines/triangles)
	- `ViewerApp` PM4 Alignment window now includes `Dump PM4 Objects JSON` to write this payload for offline comparison
- External data sanity check run on known-problem tile:
	- `WoWRollback.PM4Module` `dump-pm4-geometry` executed for `test_data/development/.../development_22_18.pm4`
	- exports were non-trivial (`development_22_18_msvt_wmo.obj`, `development_22_18_msvt_m2.obj`, `development_22_18_mscn.obj`)
	- observed payload magnitude: ~65k WMO face lines, ~126k WMO vertex lines, ~68k MSCN points
- Interpretation:
	- source PM4 tile `22_18` contains substantial geometry
	- remaining "missing tile" symptom is likely in viewer reconstruction/visibility state, not empty PM4 input data
- Validation status:
	- no automated tests were added or run
	- runtime viewer-side signoff still pending

## 2026-03-20 — PM4 Overlay Diagnostics/Grouping/Winding Update

- Added active PM4 overlay reconstruction/rendering in `WorldScene` with controls in `ViewerApp`.
- Visualization/inspection additions:
	- PM4 color modes (`CK24` type/object/key, tile, dominant group/attribute, height)
	- optional PM4 solid overlay + edge lines
	- optional `MPRL` ref pins and PM4 object centroid pins
- Object decomposition additions:
	- split CK24 groups by shared-vertex connectivity
	- optional split by dominant `MSUR.MdosIndex` before connectivity split
- Orientation/winding additions:
	- per-object planar transform solve across swap/invert U/V candidates, scored against nearest `MPRL` anchors
	- mirrored transform parity now flips triangle winding to avoid backward-wound surfaces
- Selected-object diagnostics now include dominant group key, attribute mask, `MdosIndex`, planar transform flags, and winding inversion status.
- Validation status:
	- repeated `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
	- no automated tests were added or run
	- runtime real-data visual signoff is still pending for disjoint/merged PM4 object edge cases

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Foundation | ✅ Complete |
| 3 | Terrain | ✅ Complete (shadow fix, alpha seam fix, async streaming, fog culling) |
| 4 | World Scene | ✅ WMOs, MDX placement, rotation. Particles deferred. |
| VLM | VLM Dataset Support | ✅ Load + Generate + Minimap |
| Overlays | POI, Taxi, Minimap Zoom | ✅ Complete (batched rendering, lazy-load UI) |
| Loading | Loading Screen | ✅ Complete |
| Catalog | Asset Catalog | ✅ SQL dump reader, ImGui browse/filter, JSON+GLB+screenshot export |
| 1 | **MDX Animation** | ✅ Complete (compressed quats, GPU skinning, terrain doodads) |
| — | **Per-object folders + multi-angle screenshots** | 🔧 Next up |
| 2 | Particles (PRE2/RIBB) | ✅ PRE2 complete — billboard quads, texture atlas, blend modes. RIBB still pending. |
| WL | WL loose liquids transform alignment | 🔧 In progress — matrix tuning UI added, values not finalized |
| LK | **WotLK 3.3.5 Support** | 🔧 In progress — scaffolding exists. MH2O + texturing broken. **Not usable yet** |
| 5-7 | Liquids, Detail Doodads, Polish | ⏳ Lava type mapping still broken (green) |
| MCP | MCP Server | ⏳ Designed — GLB terrain, NPC spawn, click-to-chat, audio |

## 2026-02-15 — v0.4.0 Release: 0.5.3 Rendering Improvements + 3.3.5 Groundwork

**Rendering improvements for 0.5.3. Initial 3.3.5 scaffolding added (NOT ready for use).**

**3.3.5 WotLK support (IN PROGRESS — NOT USABLE):**
- StandardTerrainAdapter: split ADT loading, MPHD bigAlpha flag — but **MH2O broken, texturing broken**
- WarcraftNetM2Adapter: MD20→MdxFile conversion for M2 models (works in isolation)
- WMO v17: multi-MOTV/MOCV, strict validation
- Fixed terrain regression from initial 3.3.5 commit (surgical revert of shared renderer code)
- **Only client versions 0.5.3 through 0.12 are currently usable**

**Lighting overhaul:**
- Half-Lambert diffuse shading on MDX + WMO (wraps light, no harsh black shadows)
- Raised ambient: day 0.4→0.55, night 0.08→0.25 (WoW-like brightness)
- WMO shader: proper vec3 lighting instead of lossy scalar average
- Reduced specular intensity (0.3→0.15)

**Particle system (PRE2):**
- Rewrote ParticleRenderer with per-particle uniforms + texture atlas + per-emitter blend
- Wired into MdxRenderer: emitters created from parsed data, bone-following, transparent pass

**Geoset animation alpha (ATSQ):**
- Per-frame alpha keyframe evaluation with global sequence support
- Alpha modulates layer alpha in RenderGeosets; invisible geosets skipped

**Key files:** TerrainLighting.cs, ModelRenderer.cs, WmoRenderer.cs, ParticleRenderer.cs, StandardTerrainAdapter.cs, TerrainRenderer.cs, WarcraftNetM2Adapter.cs, WorldAssetManager.cs

## 2026-02-13 — MDX Animation System Complete

**Three bugs fixed:**
1. **KGRT Compressed Quaternions** — Rotation keys are 8-byte `C4QuaternionCompressed`, not 16-byte float4. Ghidra-verified decompression formula.
2. **Animation Never Updated** — `ViewerApp` called `RenderWithTransform()` directly, bypassing `Render()` which contained `Update()`. Extracted `UpdateAnimation()` as public method.
3. **PIVT Chunk Order** — PIVT comes after BONE in MDX files. All bone pivots were (0,0,0). Added deferred pivot assignment after all chunks parsed.

**Terrain animation** — Added `UpdateAnimation()` for unique MDX renderers in `WorldScene.cs` before render passes.

**Key files:** `MdxTypes.cs`, `MdxFile.cs`, `MdxAnimator.cs`, `ModelRenderer.cs`, `ViewerApp.cs`, `WorldScene.cs`

## 2026-02-13 — WDL/WL/WMO Rendering Fix Pass

**WDL parser + rendering fixes:**
1. `WdlParser` rewritten for strict chunk parsing and version check (`0x12`)
2. `MARE` chunk header handling corrected before reading 545 heights
3. WDL renderer now uses `WoWConstants.TileSize` (8533.3333)
4. WDL preview improved: `.wdl` / `.wdl.mpq` fallback + explicit failure reason
5. WDL tile overlap mitigation: hide preloaded ADT tiles + depth polygon offset
6. Added UI toggle to disable WDL rendering for overlap testing

**WMO render reliability fix:**
- Converted WMO main shader and liquid shader to static shared programs with ref-counted lifetime.
- Fixes intermittent WMO disappearance caused by per-instance shader program deletion/reuse.

**WL liquids iteration tooling:**
- Replaced hardcoded axis swap with matrix-based configurable transform (rotation + translation).
- Added UI controls and hot-reload path (`Apply + Reload WL`) for rapid empirical alignment.
- Final transform values not yet locked/hard-wired.

## MDX Magenta Textures — RESOLVED (Particle System Implemented)

The magenta quads on MDX doodads were unimplemented particle emitter geometry (PRE2 chunks). Now resolved: ParticleRenderer rewritten, emitters wired into MdxRenderer, particles render with proper textures, atlas UV mapping, and per-emitter blend modes.

## Upcoming: MCP Server for LLM-Orchestrated 3D

New feature planned: turn MdxViewer into an MCP server that external applications (LLMs, procedural generators) can push content into at runtime.

### MCP Server Tools (planned)
| Tool | Description |
|------|-------------|
| `load_world` | Load GLB file as terrain (single mesh + texture) |
| `spawn_npc` | Place NPC at position with GLB model, texture, name, personality |
| `remove_npc` | Remove NPC by ID |
| `move_npc` | Move/animate NPC to new position |
| `play_audio` | Play sound file |
| `set_npc_dialog` | Set chat response for NPC |
| `get_scene_state` | Return camera pos, visible NPCs, selected NPC |
| `get_click_events` | Poll for NPC click events |

### Architecture
- MCP server runs on background thread (stdio transport)
- Commands queued and executed on render thread
- GLB → OpenGL via SharpGLTF (already in deps) + ImageSharp
- Entity system: `Dictionary<string, NpcEntity>` with position, mesh, click AABB
- Click detection via ray-AABB on mouse click
- Chat UI via ImGui overlay
- Phased: (1) server + GLB terrain + spawn, (2) click + chat, (3) audio + movement

## Mar 18, 2026 - Current Model Rendering Handoff

- The empty-fallback guardrail remains in place: converted M2-family fallback models are rejected when they contain no renderable geometry.
- User runtime verification now narrows the remaining format issue:
	- most unresolved M2 failures are specific to the pre-release `3.0.1` model family
	- treat that family as a possible hybrid or transitional `MDX` + `M2` path rather than assuming later `3.3.5` semantics
- Separate shared rendering defect still remains open:
	- neon-pink transparent surfaces still reproduce on both `MDX` and M2-family assets
	- treat that as shared material / texture / blend / shader work, not just parser work
- Next split for follow-up work:
	1. version-aware pre-release `3.0.1` model routing and parsing
	2. shared transparent-surface parity audit in renderer code

## Mar 19, 2026 - Runtime Confirmation Update

- User runtime validation now confirms classic Alpha `0.5.3` MDX rendering is fixed again.
- The repaired classic path includes three scoped renderer corrections in `ModelRenderer.cs`:
	- replaceable fallback kept M2-only
	- wrap/clamp semantics kept version-family-specific
	- classic `Layer 0 + Transparent` restored to unconditional alpha-cutout behavior
- A reusable direct-asset diagnostic path was added:
	- `MdxViewer --probe-mdx <gamePath> <modelVirtualPath> [--listfile <path>]`
	- this was used against `DuskwoodTree07.mdx` on the real `H:\053-client` data to prove the remaining failure was renderer-side, not parser-side
- Current model status is now narrower and more defensible:
	- classic `0.5.3` MDX is restored
	- pre-release `3.0.1` is still buggy
	- do not describe the remaining model issue as a shared MDX + M2 transparency problem anymore without new evidence

## Detailed Fix Log

### 2026-02-09 Late Evening — Performance, Fog, Culling, Failed MDX Fix

**Completed (working):**
- Batched overlay rendering — POI pins + taxi paths in single draw call (BoundingBoxRenderer rewrite)
- AreaID/MapID-aware area name lookup — filters by current map, warns on mismatch
- Minimap zoom — 4 tiles around camera, scrollable
- TaxiPath visualization — DBC-loaded flight paths as 3D lines with selection
- POI + Taxi disabled by default
- NoCullRadius (150 units) — nearby objects skip frustum cull to prevent pop-in
- Fog added to MDX shader — models blend into fog like terrain
- Fog added to WMO shader — WMOs blend into fog like terrain
- Fog-based terrain chunk culling — skip chunks beyond FogEnd+200
- AOI radius reduced 3→2 (49→25 tiles)
- Doodad cull distance reduced 1500→1200, WMO cull distance 5000→2000
- README renamed to AlphaWoW Viewer

**Failed (reverted or ineffective):**
- `.blp.MPQ` scan — WRONG, these files don't exist. Reverted.
- ResolveReplaceableTexture 4-strategy rewrite — did not fix magenta textures
- Alpha test threshold change (0.3→0.1, conditional) — did not fix magenta textures

**Key files modified:**
- `BoundingBoxRenderer.cs` — Complete rewrite for batched rendering
- `WorldScene.cs` — Batched overlays, fog passing, reduced cull distances, NoCullRadius
- `TerrainManager.cs` — AOI radius 3→2
- `TerrainRenderer.cs` — Fog-based chunk distance culling
- `ModelRenderer.cs` — Fog shader, alpha test fix, ResolveReplaceableTexture rewrite
- `WmoRenderer.cs` — Fog shader
- `AreaTableService.cs` — MapID-aware lookup
- `ViewerApp.cs` — MapID tracking, area name display
- `MpqDataSource.cs` — Reverted bad .blp.MPQ change

### 2026-02-08 Evening — Shadow, Pivot, VLM Generator

**MCSH Shadow Blending Fix** (TerrainRenderer.cs):
- Shadow was only applied on base layer (`uIsBaseLayer == 1`). Overlay texture layers drawn on top with alpha blending would wash out shadows.
- Fix: Removed `isBaseLayer` guard from both C# shadow texture binding and GLSL shader shadow application.
- C#: `bool hasShadow = isBaseLayer && chunk.ShadowTexture != 0` → `bool hasShadow = chunk.ShadowTexture != 0`
- GLSL: `if (uShowShadowMap == 1 && uIsBaseLayer == 1 && uHasShadowMap == 1)` → `if (uShowShadowMap == 1 && uHasShadowMap == 1)`

**MDX Bounding Box Pivot Offset** (WorldScene.cs, WorldAssetManager.cs):
- MDX geometry is offset from origin. MODL bounding box center = effective pivot.
- Added `WorldAssetManager.TryGetMdxPivotOffset()` → `(BoundsMin + BoundsMax) * 0.5f`
- Transform chain: `pivotCorrection * mirrorX * scale * rotX * rotY * rotZ * translation`
- `pivotCorrection = Matrix4x4.CreateTranslation(-pivot)`
- Applied in both `BuildInstances()` and `OnTileLoaded()`.
- WMO models do NOT need pivot correction.
- Note: PIVT chunk is for per-bone skeletal pivots, NOT placement pivot.

**VLM Dataset Generator** (ViewerApp.cs):
- Menu: `File > Generate VLM Dataset...`
- Uses `VlmDatasetExporter.ExportMapAsync()` from WoWMapConverter.Core on ThreadPool.
- Progress log via `IProgress<string>`, "Open in Viewer" button on completion.

### 2026-02-08 Afternoon — VLM Terrain, Async Streaming, Thread Safety

**VLM Terrain Rendering Fixes** (VlmProjectLoader.cs, TerrainRenderer.cs):
- GLSL shader em-dash → ASCII hyphen (compilation error).
- NullReferenceException in DrawTerrainControls → null-conditional access.
- VLM coordinate conversion: swapped posX/posY, removed MapOrigin subtraction.

**Minimap for VLM** (ViewerApp.cs, VlmTerrainManager.cs):
- `DrawMinimap()` refactored to support both `_terrainManager` and `_vlmTerrainManager`.
- Added `IsTileLoaded()` to VlmTerrainManager.

**Async Tile Streaming** (TerrainManager.cs, VlmTerrainManager.cs):
- `UpdateAOI()` queues tiles to ThreadPool for background JSON/ADT parsing.
- `ConcurrentQueue<TileLoadResult>` for render-thread submission.
- `SubmitPendingTiles()` uploads max 2 tiles/frame.
- `_disposed` flag prevents post-dispose background access.

**Thread Safety** (VlmProjectLoader.cs, AlphaTerrainAdapter.cs, TerrainRenderer.cs):
- `TileTextures` → `ConcurrentDictionary` in both adapters.
- `_placementLock` protects `_seenMddfIds`, `_seenModfIds`, placement lists.
- `CollectMddfPlacements()` and `CollectModfPlacements()` also locked.
- `TerrainRenderer.AddChunks()` parameter: `Dictionary` → `IDictionary`.

### 2026-02-08 Morning — MDX, BIDX, MTLS, GLB

- Fixed standalone MDX rendering — MirrorX model matrix for LH→RH conversion
- Fixed BIDX parsing — 1 byte per vertex (not 4)
- Reverted MTLS dual-format heuristic — back to stable count-header format
- Fixed WMO WorldScene regression — reverted to stable commit a1b0b41
- Fixed GLB export — Z-up → Y-up conversion for MDX and WMO
- Added terrain alpha mask debug view (Show Alpha Masks toggle)
- Applied Noggit edge fix for terrain alpha map seams

### 2026-02-07 — World Scene, Minimap, AreaPOI

- Phase 4 — World Scene: lazy tile loading, MDDF/MODF placements, object rendering
- Fixed MDX holes — disabled backface culling
- Fixed WMO textures — BLP loaded per-batch from materials
- Fixed MODF bounding boxes — actual extents with min/max swap
- Added live minimap with click-to-teleport
- Added AreaPOI system (DBC loading, 3D/minimap markers, UI list)
- Fixed MDX blend modes — depth mask off for transparent layers, alpha discard 0.1
- WMO doodads now rendered with WMO's modelMatrix

### 2026-02-06 — Foundation + Terrain

- Phase 0 COMPLETE — 6 foundation files
- Phase 3 COMPLETE — 7 terrain files + ViewerApp integration
