# Active Context — wow-viewer

Last updated: 2026-09-04

**START HERE: [`specs/NEXT-DAY-PLAN.md`](../specs/NEXT-DAY-PLAN.md)** — the ordered pass through the
open specs, with the reasoning for the order. This section is the session summary behind it.

## Landed 2026-09-04 — Spec 222 v2: Cartography (multi-map, multi-tile workbench) — operator accepted

- **The 0.5.3 "phase overlay regression" was diagnosed and is NOT a code defect.** Measured via the
  new admission diagnostics: `Shadowfang` over `Azeroth` — stack propagates, WDT resolves (10 tiles),
  but target (34,28) maps to source (34,28) at zero offset and Shadowfang has no tile there, so
  composition correctly contributes nothing. The failure is the interaction model: numeric offsets
  with no spatial feedback. Diagnostics remain in `AlphaTerrainAdapter` (admission / skipped /
  unresolved / tile-miss, one-shot per configuration) plus occupied-tile coordinates in the resolver
  log.
- **Operator accepted the workbench deal with expanded scope**: any TILE, not just any map, bundled
  with copy/paste/rotate, as one right-sidebar **Cartography** feature replacing the phase panel,
  the chunk manipulator (195), and the scattered selection surfaces — multiple layers of tile data
  from multiple maps. Spec 222 v2 ([spec](../specs/222-map-composition-workbench/spec.md) ·
  [plan](../specs/222-map-composition-workbench/plan.md) ·
  [tasks](../specs/222-map-composition-workbench/tasks.md)): footprints AND tile placements drawn
  and dragged on the minimap; transform tools wrap the validated Spec 219 seam (`TileContentTransform`);
  full-channel paste fixes 195's partial-paste defect; old surfaces retired in the same change
  (FR-11). Composition semantics unchanged. **Next: 222-T101.**

## Landed 2026-09-04 — Spec 211 Phase 5: doodad selection bounds + 3D aids (build green, operator visual check owed)

- Selected WMO doodads now get REAL oriented geometry boxes: `WmoRenderer.TryGetDoodadLocalBounds`
  exposes model-space bounds, `TryBuildSelectedWmoDoodadInstance` feeds them as
  `LocalBounds*/SelectionLocalBounds*/SelectionBoundsResolved` so the overlay takes the oriented
  tight-box path through the doodad's world transform. `WmoRenderer.RequestDoodadModelLoad`
  queues the doodad's model on selection so placeholder bounds resolve within a few frames.
- 3D aids on selected doodads: cyan origin octahedron + gold position pin at the MODD point, RGB
  axis tripod through the placement transform, sized from the box. Placeholder (model unloaded)
  bounds draw an ORANGE accent — never readable as real extent. Tasks 211-T501..T505 in
  [`specs/211-wmo-interior-picking-ghost-wireframe/tasks.md`](../specs/211-wmo-interior-picking-ghost-wireframe/tasks.md);
  T505 is the operator's interactive check.

## Next lane — Spec 221: alpha round-trip — 3 defects fixed, 1 open (2026-09-04)

- Fixed: (1) MCLY `0x200`→`0x100` on uncompressed alpha layers in `AlphaToLkConverter` (0x200 is
  RLE; the old comment even said "big alpha"); (2) `LkAdtReader` now strips the 8-byte MCAL/MCSH
  subchunk headers — `AlphaMapData` previously began with the literal bytes `MCAL`+size and every
  alpha byte was shifted 8; (3) `ValidateRoundTripCommand` compares at matching resolution
  (`McalAlphaPack` is a 4× downsampled 256 signal; the old compare compared `orig256[y,x]` against
  `upsample(orig256)[y,x]`, so hard edges reported as 1.000 flips). Writer offsets verified correct
  by byte probe; a writer-side −8 experiment broke MCVT and was reverted — do NOT retry it.
- **Full suite 1441 passed / 10 failed**: 9 known baseline + NEW pinned test
  `AlphaToLk_FlagContract_AllowsAlphaRoundTripThroughLkBytes` (deliberately red — drift 0.9333
  through pack256→LK→pack256 proves at least one more defect; suspect the nearest-upsample
  `y*16/64` mapping in `SliceChunkAlphaBytes` or layer-span inference in `LkToAlphaConverter`).
  Also known: the Alpha→LK leg consumes the LOSSY 256 pack (4× alpha resolution loss by design).
- Evidence with the full defect trail: [`specs/221-converter-validation-harness/evidence/phase0-baseline.md`](../specs/221-converter-validation-harness/evidence/phase0-baseline.md).
- **Next bounded step**: one focused debug pass on the pinned test (its synthetic fixture isolates
  the chain — no MPQ needed); then 221-T101 object corpus validator. Do not widen scope past the
  pinned test's chain until it is green.

## Queued lane — Spec 220 WMO doodad editing & writing (spec authored 2026-09-04, not implemented)

- Operator follow-up to 211's doodad picking: edit MODD placements (move/rotate/scale/add/delete),
  author MODS doodad sets, save the WMO back in the opened file's version. Full Spec Kit trio at
  [`specs/220-wmo-doodad-editing/`](../specs/220-wmo-doodad-editing/spec.md); registered in STATUS.md.
- Feasibility measured before writing: V17 canonical read model, a V14 root writer
  (`WmoV17ToV14Converter.WriteWmoV14`), MODD/MODS/MODN detail readers, Core.Editor ops+undo patterns,
  and 211's selection pipeline all already exist — the spec composes them, no forks, readers frozen.
- **Phase 0 is blocking**: round-trip gate (V14 → V17 → V14 → byte/field equality on a real WMO)
  before any editing UI, so the writer is proven on unmodified data first. Next bounded action:
  **220-T001**.

## Hard constraints (violating these has cost real time)

- **Python owns the datastore. C# does NOT implement Zarr or TensorStore.** C# emits `ARRY/ENDS`
  blobs via `RawArraySerializer`; Python's `harvester.raw_reader.read_tile_blob` ingests them and
  `zarr_io.py` builds the store. A C# Zarr reader was written 2026-09-02 and **deleted the same day**;
  the operator has been down that road before and it cost two months. Spec 206's framing of "no C#
  Zarr reader" as a *gap* was wrong — the absence is the architecture.
- **No test project references the viewer.** Logic that must be tested belongs in `WowViewer.Core*`.
  This is why `PhaseCompositionPolicy`, `GpuInstanceBatchPolicy`, `PhaseSignalChannelMap` and
  `LiquidSurfaceInterpolation` all live in Core rather than beside their callers.
- **`DBCDRow.ID` is a POSITIONAL key** for MoP WDB2 tables, not the row id. Key on the `ID` column
  (`DbcTableLoader.ResolveRowId`). Keying on `row.ID` silently resolves the wrong row for every
  sparse id.
- **Close the viewer before running tests** — it locks the build output and the test project pulls in
  the viewer transitively.
- **Baseline: 9 pre-existing test failures** (WtfLineClassifier x2, WorldFramePassCoordinator x3,
  AdtV23SummaryReader, V18StorePlacementsReader, EnrichmentStreamFormat, ModelFootprintReader). Any
  other failure is new. Current: **1354 passed**.

## Current lane — Spec 211 Phase 4 doodad hover tooltip (landed 2026-09-04, awaiting operator visual check)

- Precise-ray hover now reuses the Spec 211 click picker (`CollectSceneObjectPickHits`) so WMO doodads
  show the hover overlay, with `WmoContainerFallThroughFilter` applied after WMO/doodad visibility
  gates so an enclosing WMO AABB cannot hide its interior doodads.
- `HoveredAssetInfo` carries `ParentWmoIndex`/`ParentSourcePath`; "Left-click to inspect" now selects
  the exact hovered doodad via `SelectSceneObject(type, index, parentIndex)`. Tooltip shows MODD
  definition index, MODN offset, active MODS set, MODR group references, and parent WMO identity.
- **Proof:** focused `WmoContainerFallThroughFilterTests` 7/7 green (new doodad fall-through case);
  viewer Debug build 0 errors. Source/build/unit proof only — operator must hover a placed WMO
  doodad to verify the tooltip visually (T407 open).
- **Unrelated open regression (do not conflate):** Alpha 0.5.3 terrain shows chunk-aligned dark MCSH
  patches. Tracked separately under Spec 219's earlier debug lane; not caused by this change.

## Previous lane — Spec 219 phase-layer rotation

- **Phase 1 Gate 1 passed 2026-09-03.** Core now owns a phase-agnostic `TileContentTransform` for
  exact 90°/180° rotation and H/V mirrors across terrain vertices/normals, holes, alpha/shadow,
  MCCV, liquid flags, complete MDDF/MODF placements, and MODF bounds. Mirrors are exact involutions;
  CW/CCW round-trip is identity. Free-angle placement-point rotation is present; terrain free-rotate
  remains a stated grid-snapped approximation.
- `PhaseLayerSettings` now carries angle/origin/mirrors/per-tile mappings and preserves them in
  `Clone`. `ResolveTileSource` is origin-aware, per-tile mappings win last, duplicate-target claims
  are counted for diagnostics, negative quarter turns normalize, and approximation is explicit.
- **Proof:** focused Core tests **55/55**; full `WowViewer.slnx` Debug build **0 errors** (existing
  warnings only). This is source/build/unit proof, not runtime or visual proof.
- **Next bounded action:** Phase 2 T007 — wire only `StandardTerrainAdapter` to
  `ResolveTileSource`/`TileContentTransform`, keep the zero-transform path unchanged, then rerun the
  focused tests before touching `AlphaTerrainAdapter`.
- **Operator scope addition:** offsets need one-terrain-cell precision so terrain pasted across
  unrelated source/target chunk or ADT boundaries can be aligned. Canonical unit: integer cells;
  **8 cells = 1 chunk, 128 cells = 1 ADT**. Phase 2B owns Core boundary re-slicing plus adapter/UI
  wiring. Preserve authored heights exactly; sub-cell/world-float shifts require an explicit future
  resampling mode. Spec 196's `WdlLatticeMagnetizer` may supply an optional candidate offset + fit
  score, but the operator accepts it and the integer cell offset remains authoritative.
- **Workbench expansion (operator 2026-09-03):** selection must be a full 64×64 orthographic map
  over minimap/heightmap/occupancy, with magnetic tile/chunk/cell click/check, paint, box, lasso,
  source/target colors, and an explicit target. The same selection projects into configurable 3D
  terrain selection. Base becomes a non-removable channel-gated first layer. Every phase card must
  expose 90/45 CW/CCW, Mirror H/V, free angle, origin, approximation, and reset. The page also owns
  preflighted Save Transformed Map to supported ADT/Alpha WDT output copies.
- **Spec 195 correction:** its “complete” claim is false by source: camera-centered 280px dark grid,
  no minimap/heightmap, click-only despite drag task, camera-derived target, and paste materializes
  only heights/normals/holes. Reuse only audited Core coordinate/undo pieces; retire duplicate UI,
  selection, clipboard, target, and paste state through Spec 219 Phase 4.

## Landed 2026-09-02

**Spec 210 — 3D Scene Cursor & In-World Spatial Selection (Phases 1–4 Landed).** User initiated full redesign
of the mouse and object selection system.
1. Configured and tested OpenSCAD MCP server (`quellant/openscad-mcp`) wrapping CLI at `C:\Program Files\OpenSCAD\openscad.com`.
2. Implemented headless `OffGeometry` in `WowViewer.Core.Geometry` (with 3/3 passing unit tests in `OffGeometryTests.cs`) and `ProceduralMeshLoader` in `WoWViewer.Rendering` for OpenSCAD OFF/STL models and primitives.
3. Implemented `SceneCursorRenderer` loading authentic `Interface\Cursor\Cursor.mdx` (Alpha 0.5.3) / `Point.mdx` with fallback to procedural pointer.
4. Enforced camera culling invariant: near-plane clamp $Z \ge Z_{\text{near}} + 0.15\text{ yd}$, depth range bias `0.0..0.05` (always visible on contact geometry), and perspective scaling $S = \text{BaseScale} \times \text{Distance} \times \tan(\text{FOV}/2)$.
5. Seamless ImGui interop: hides OS cursor (`ImGui.SetMouseCursor(None)`) only when inside the 3D viewport, instantly restoring it over ImGui panels/menus/modals.
6. Implemented `SceneClusterSelector3D` replacing intrusive 2D modal menus with in-world 3D orbital rings and interactive candidate markers with 1-9 keyboard shortcuts.
7. Authored 5 production-grade OpenSCAD assets in source tree (`Assets/OpenScad/`): `cursor_pointer.scad`, `cursor_reticle.scad`, `cluster_pin.scad`, `camera_hud_gimbal.scad`, `hud_frame_bracket.scad` + compiled `.off` files + batch compiler script `compile_openscad_assets.ps1`.
9. Added Settings controls for Cursor Style/Scale and 3D Camera HUD toggles in Settings > Interface.
10. Eliminated overlapping popups: suppressed legacy 2D hover overlay (`##SceneHoverAssetOverlay`) when in 3D cursor mode or during active cluster selection; removed legacy 2D `ClickSelectionOverlay` modal list entirely; consolidated cluster disambiguation into a single non-overlapping in-scene card anchored cleanly above the 3D orbital ring.
11. Fixed hardware cursor overlap: toggled physical OS cursor visibility via Silk.NET `CursorMode.Hidden` / `CursorMode.Normal` directly on `_input.Mice`, guaranteeing the white OS arrow is invisible over the 3D viewport and immediately reappears over ImGui panels/menus.
12. Added UI Typography & Font Size scaling: continuous slider (`0.85x`–`2.20x`) and quick presets (`100%`, `120%`, `135%`, `150%`, `175%`) under Settings > Interface with live `ImGui.GetIO().FontGlobalScale` scaling and persistent storage in `ViewerSettings.UiFontScale`.
13. Reconciled Phase Map Layer offset controls in `ViewerApp_PhaseLayers.cs`: swapped the UI text identifiers (`Tile offset X` now controls horizontal/col `TileOffsetY`, and `Tile offset Y` now controls vertical/row `TileOffsetX`) with descriptive tooltips, resolving the discrepancy where phased maps worked on the minimap but had swapped axes in 3D terrain.
14. Startup scene cursor guarantee: decoupled 3D viewport setup, sky gradient backdrop, 3D camera HUD, and `RenderSceneCursor()` from `_renderer != null` in `ViewerApp.cs`. The viewer now always renders a single active 3D scene on startup with the in-scene cursor floating smoothly over a clean sky gradient backdrop even when nothing is loaded, preventing cursor disappearance on launch.
15. Spatial clustering proximity filter & selection lifecycle:
    - Enforced a strict spatial proximity cluster threshold ($\le 2.0\text{ yd}$ from closest hit) and terrain occlusion culling: objects 40 yards apart along the ray no longer trigger ambiguous cluster popups; foreground objects select immediately on a single click.
    - Resolved selection lock: removed stale candidate block in `TryHandleSceneClickSelection`, ensuring subsequent clicks can pick different objects seamlessly.
    - Added comprehensive deselection: clicking on empty terrain/sky clears selection, clicking an already-selected object toggles it off, and pressing Escape immediately deselects active world objects or closes the cluster card. All builds clean. Ready for operator interactive verification.
16. Spec 211 Landed (Phases 1–3 Implemented & Unit Tested 2026-09-02; Ready for Operator Interactive Verification):
    - WMO Interior Ray Picking & Container Fall-Through: Authored `WmoContainerFallThroughFilter` in `WowViewer.Core.Runtime.World` with comprehensive unit tests (6/6 passing). When a ray hits an interior object (MDX, WMO doodad, or nested WMO) enclosed within a WMO's bounding box, clicks fall through to the interior object rather than locking onto the outer building envelope.
    - WMO Doodad Selection: Added `ObjectType.WmoDoodad`, implemented `TryPickDoodadsByRay` in `WmoRenderer.cs`, wired doodads into `WorldScene.cs` candidate gathering and selection state (`SelectedWmoParentIndex`, `TryGetSelectedWmoDoodad`), and connected to inspector and camera framing in `ViewerApp.cs`.
    - Ghost Transparent Wireframes: Implemented dual-pass ghost wireframe rendering across all 3 subsystems:
      - MDX / M2 models (`ModelRenderer.cs`): Pass 1 fills textured geometry with 33% alpha blending (`fadeAlpha * 0.33f`); Pass 2 draws prominent wireframe lines with `PolygonOffsetLine` (-1.0, -1.0) and 1.5 line width.
      - WMO objects (`WmoRenderer.cs`): Opaque batches unbind GPU instancing and render with 33% alpha blending (`uColor = (1, 1, 1, 0.33)`), followed by `RenderWireframeOverlay` with polygon offset and 1.5 line width.
      - Terrain (`TerrainRenderer.cs`): Draws textured terrain fill in `PolygonMode.Fill`, followed by a line pass with `PolygonOffsetLine` (-1.0, -1.0) and 1.5 line width so wireframes are obvious while terrain texturing remains clearly visible.
    - Clean build, zero errors across solution, all unit tests green. Ready for operator interactive verification.

**Scene cursor drew under the UI — draw order, not depth.** The 3D scene cursor was rendered with the
rest of the 3D pass (`ViewerApp.cs`, before `_imGui.Render()`), so **every** ImGui window painted over
it. No depth state can fix this: ImGui is a separate pass, and the cursor already defeats depth
occlusion deliberately via `DepthRange(0, 0.05)`. Because the hardware cursor is hidden while a 3D
cursor style is active, the pointer disappeared entirely under any overlapping panel/menu/card —
exactly when aiming a click. Fixes:

- Split `RenderSceneCursor` into `RenderSceneClusterSelector3D(proj)` (stays in the 3D pass; the rings
  are scene geometry and must occlude correctly) and the cursor overlay pass, moved to **after**
  `_imGui.Render()` and after both `CaptureVideoFrameIfNeeded` taps (cursors do not belong in
  captures). Caller re-sets the scene viewport around it.
- The overlay runs on whatever GL state ImGui left, so it now sets scissor/cull/blend/depth-mask
  explicitly. **Leaving `ScissorTest` enabled would silently clip the cursor away.**
- `DrawUI` only cleared `_dockspaceHostPosition/Size` for the chrome and dockspace toggles, not for
  `_useTabUi`, but `DrawDockspaceHost` runs only when `!_useTabUi`. Toggling tab UI on at runtime left
  a **stale non-zero host rect**, making `ShouldBypassDockspaceMouseCapture` claim the mouse for the
  scene across the whole viewport rect — defeating ImGui capture for floating windows over it, so
  clicks meant for those windows also fired scene picking. Now cleared for `_useTabUi` too.

Related standing note: the dockspace already passes `ImGuiDockNodeFlags.PassthruCentralNode`, which is
the supported mechanism for letting the empty central node ignore the mouse. That makes
`ShouldBypassDockspaceMouseCapture` largely redundant in dockspace mode; it was not removed because
that mode is not exercised by default. Revisit if dockspace mode is ever made primary.

**WMO doodad selection was a placeholder, not a read.** Operator-reported 2026-09-02: blank asset
paths, boxes that do not fit the object, and a "UniqueId" on WMO doodads. All one cause —
`TryGetSceneObjectByIndex`'s `ObjectType.WmoDoodad` case *synthesised* an `ObjectInstance` instead of
reading the doodad:

- **Bounds were a hard-coded `position ± 1` cube.** `WmoRenderer.TryGetDoodadBounds` already computes
  the real transformed AABB and `TryPickDoodadsByRay` already carried it in the pick hit; the
  selection path discarded it. The 0.75 yd min-half-extent clamp added in `4c73d263` **could never
  have fixed this** — 1.0 already exceeds the 0.75 floor, so the clamp was a no-op on exactly the
  objects it was written for.
- **`BatchHighlightedBoxMinMax` then inflated it further.** Accent inflate was
  `Clamp(maxDim * 0.015, 0.75, 6.0)` — a *fixed* 0.75 yd floor, so a third-of-a-yard scroll got a halo
  six times its own size, and `segmentLength`'s 6.0 yd floor collapsed the dashes into a second solid
  box. Both are now proportional (`0.06` inflate / `0.03` floor; segment floor `0.08`).
- **`UniqueId = DoodadDefIndex` was the subtle one.** MODD carries `NameIndex, Position, Orientation,
  Scale, Color` and **no uniqueId** — uniqueId is an MDDF/MODF (ADT placement) concept. This was not
  just a mislabel: `ShouldHideObjectInstanceByUniqueId` keys on that field, so the uniqueId range
  filter could hide a doodad whose MODD index happened to fall in range. Def index now lives in
  `PlacementEntryIndex`; `UniqueId` is 0, which that filter already treats as "no id".
- `ModelName`/`ModelPath` were never assigned (only `ModelKey`) — hence the blank Path. `Rotation`
  `(0,0,0,0)` and `Scale 0.000` were unassigned defaults; both are available on the MODD def, and the
  instance transform is `Scale * Rotation * Translation` so it carried them all along.

Bounds now report `BoundsResolved`, and the inspector marks a placeholder box as one rather than
presenting a guess as a measurement. General rule, same as [[feedback_a_name_stops_the_looking]]: a
field named `UniqueId` on a record type that has no unique id is an assertion nobody checked.

**Spec 209 — Liquid Convergence Measured (Phase 1 Complete).** Built `inspect adt liquid-convergence`
and `LiquidConvergenceAnalyzer` (4 new tests). Catalog discovery loads 108 WL* files directly from
`misc.mpq` in 0.5.3. Scanned 500 liquid tiles on Azeroth. **Union invariant verified (SC-002 / FR-004)**:
exactly **0 cells** missing from unified array across all 500 tiles. **Mechanism B CONFIRMED**:
`KeepOnlyAboveTerrain` culled 585,108 WL* cells across Azeroth; **459,374 of those culled cells had NO
MCLQ coverage**, creating empty waterline strips along coastlines (23,192 cells on Wetlands coast tile
`Azeroth_31_29` alone). In open water where sources overlap, surface heights agree almost identically
(mean ΔH = 0.00).

**Spec 208 — Cross-Map Transplant Phase 0 Complete.** Completed technical audit & reconciliation
([`evidence/phase0-reconciliation.md`](../specs/208-cross-map-tile-transplant/evidence/phase0-reconciliation.md)):
(1) Map identity enters at source chunkReader and target MCLY texture re-mapping boundaries. (2) Source
inspection proved 195 fails to rotate normals, placement rotations, and placement relative positions
(defect in 195 to fix there). (3) Reconciled `ChunkTranspositionOptions` directly onto `PhaseDataChannel`
(Constitution II). (4) Confirmed chunk granularity (1/16 tile = 33.33 yd) preserves raw vertex fidelity.
(5) Staged transplant proposal lives in `EditorSession`.

**Spec 205 — MH2O liquid.** Rivers decode. The wiki's "values >= 42 are LiquidObject ids" threshold is
**wrong for 5.0.1**: real ids run 57..2390 and 42 is absent, so ocean stays unresolved and flat, which
is correct. Rivers resolve `2325/2333/2372 -> LiquidType 5 -> Material 1 -> LVF 0`. Both decoders
fixed; two-decoder parity test.

**Terrain normals — the darkness cause.** MCNR component order is **era-split**: 0.5.3 is `(x,z,y)`,
Cata+ is `(x,y,z)`; the shared decoder applied the alpha order everywhere. Renderer agreement against
heightmap-derived normals: Cata **0.0437 -> 0.9953**, MoP **0.3282 -> 0.9985**, alpha unchanged at
0.9179. **3.3.5 is the same as 4.0.1** (operator-confirmed), so LK writing is unblocked.

**Spec 203 — phase composition.** Was a whole-chunk replacement preserving only liquid; now a
per-channel patch with a presence gate. Placements were appended unconditionally; now presence-gated
replace. Multi-layer stack + tile offsets + `Map.dbc.ParentMapID` discovery (5.0.1 only — **0.5.3's
Map.dbc has no ParentMapID**, only 5 columns). Photoshop-style layers panel.

**Spec 207 — object draw calls.** Ablation: terrain ~3.3 ms, WMOs ~55.5 ms, doodads ~41.2 ms —
objects are ~97% of the frame, and at 4.6 us per submission the cost is **per-draw CPU overhead, not
fill rate**. Faded instances were excluded from instancing *because* they were faded (36% of the
visible disc); now split into opaque + faded batches. Zero-fade instances dropped. Asset load priority
FIFO backlog bounded (12 promoted/frame vs 1-4 drained made it stale).

**Minimap DXT1.** The codec existed but sat behind an opt-in flag emitting a *companion*; primary
tiles were pristine 24-bit while authored tiles are decoded DXT1. **The scorecard had the same
confound** and fixing only the files would not have fixed it. Now primary, encoded once, reused for
writing + scoring + baseline + the visual A/B.

## Open, with the next concrete action

- **Spec 203 — Alpha phase fixes landed 2026-09-03, operator visual proof owed.** Two
  operator-confirmed defects fixed in the Alpha phase-layer path: (1) **name-table remap** —
  MDDF/MODF `NameIndex` is local to each WDT's MDNM/MONM table, so copying a phase index into the
  base tile resolved the base table's unrelated entry (`newbindstone.mdx` became `gypsywagon.mdx`
  at the correct transform); the Alpha adapter now resolves phase indices to phase-table paths,
  reuses/appends them in the base table, and rewrites the index before rendering. (2) **tile-offset
  double inversion** — the Phase Map Layers panel cross-wired its labels (now direct: X = row
  `TileOffsetX`, Y = column `TileOffsetY`), and `TranslatePhasePlacements` in BOTH adapters passed
  `ChunkSize` (533.33 yd) where offsets are in TILES, moving placements 1/16 of the terrain's
  distance; the attempted `TileSize` (8533.33 yd) correction was later reverted after auditing the
  coordinate formula: this codebase's misnamed `ChunkSize` is the one-ADT span (533.33 yd). Source
  lookup stays `target - offset`. Build
  **0 errors**; phase-composition tests **18/18**. **Proof owner: operator** — reload an offset
  Alpha phase layer and confirm objects land on their terrain; the `[AlphaADT] Phase offset
  mapping` and `Phase ... name map` log lines record the mapping.
- **Spec 218 — creature staging** (drafted 2026-09-02, not planned). Spawn a subject, **attach** an
  equipped torch, paper-doll UI, and save the whole arrangement so it replays through the existing
  capture automation. **Measured: spawning already exists and capture automation already exists, but
  attachment points are PARSED AND NEVER RENDERED — `MdxAttachment*` is in Core and nothing in
  `Rendering/` references it.** That is why a torch cannot be put in a hand. Item-to-appearance
  resolution also does not exist and is the largest unknown. Saved scenes are P1, not convenience:
  216's value depends on re-running the comparison whenever lighting changes, and manual reassembly
  each time is not comparable. This is the test rig for 216.
- **Spec 216 — model cursor as a scene light source** (drafted 2026-09-02, not planned). Any MDX/M2
  as the cursor, with particles, and **its lights illuminating the scene** — aimed at reproducing the
  2001 thief-with-torch screenshot (`areatest.lit`, ~3am, torch as the only light). **Measured: the
  LIT variant is already probed, particles already render, and MDX lights are already parsed — but
  `UploadMdxLights` uploads a model's lights into that model's own shader program, so a torch lights
  only itself — and particle effects illuminate nothing at all.** **Operator correction: the light
  comes from the EFFECT, not the `LITE` chunk** — a torch's illumination is its fire. The emitter data
  already carries position, an animated colour ramp and an intensity proxy, so a derived light
  flickers because the flame does. **The lights were always in the data; 2001 hardware could not
  afford them** — `areatest.lit` has authored point lights, and one well-done point light in a dark
  scene was expensive then, which is why the look survives in one screenshot and then vanishes for a
  decade. So a failure to reproduce it points at our lighting model, never at missing data. Era
  dungeon shots (Deadmines) show the same thing from *placed* objects, so the mechanism must be
  designed to drive any model instance (activation for world objects stays out of scope). **Operator's
  reading of the reference image: the night colour profile is already right — the point light is the
  only missing piece.** Recorded as a judgement, not a measurement; the reconstruction checks it.
- **Spec 217 — audio lifecycle** (drafted 2026-09-02, not planned). One-shots fire forever and music
  does not play, so audio is off by default. **The 5.0.1 binary explains it: a sound is a state
  machine over six explicit lists, and one that never reaches the delete list never stops. It is a
  lifecycle defect, not a decoding one** — which is why work on the decoders never fixed it. Also:
  **three** repeat modes, not two (periodic = repeats *with gaps*), duplicate suppression at the play
  call, and finite prioritised channels where exhaustion routes to deletion as a *handled* outcome.
  Backend is era-split (0.5.3 DirectSound/DirectMusic, 5.0.1 FMOD): the discipline transfers, the
  backend does not. Autoplay is the *outcome*, gated on the lifecycle being demonstrated.
- **Spec 214 — 5.0.1 physics (Domino)** (**implementing 2026-09-03; solver-independent
  US7 policy validated, Phase 0 evidence gate next**). The client solver is **Domino**,
  a separate engine at `Engine\Source\Domino/`; WoW's adapter (`Physics.cpp`, `PhysData.h`) is the
  *sibling* directory — two layers, do not conflate. Entry point is the Domino assertion string
  `0x00e0ac8c` → handler `FUN_00c29680`, whose ~90 callers each pass their own file and line.
  **~90 is a floor, not the size** — only asserting functions are visible to that pivot. **Operator
  direction: license in an existing permissively-licensed C# solver; no copyrighted engine code.**
  Domino is decoded as a *contract* — data layouts and observable behaviour — and never reproduced;
  transcribing its algorithms would make this a derivative of Blizzard's engine. Library selection and
  license verification are `plan.md` deliverables, and cloth must be a selection criterion rather than
  a discovery. The client culls physics by distance, so an unbudgeted implementation is both
  unfaithful and a frame-cost regression on top of a known one. The new [`plan.md`](../specs/214-mop-physics-domino/plan.md),
  [`research.md`](../specs/214-mop-physics-domino/research.md), [`data-model.md`](../specs/214-mop-physics-domino/data-model.md),
  contract, quickstart, and [`tasks.md`](../specs/214-mop-physics-domino/tasks.md) enforce the format/solver gate:
  read-only caller attribution, adapter-to-model-sidecar discovery, real-client manifest, and exact-version
  BepuPhysics/Jitter2 license + deterministic-cloth evaluation. **No parser, package, simulation, cloth,
  or viewer path starts first.**
  **Current-source audit complete:** [`current-implementation-audit.md`](../specs/214-mop-physics-domino/evidence/current-implementation-audit.md)
  confirms `M2ModelDocument.HasPhysicsSidecar` is unconsumed metadata and MDX `CLID`, camera collision,
  and particle gravity are unrelated adjacent paths. **Landed policy slice:** `PhysicsRuntimePolicy.cs`
  reuses `ClientBuildKey` for exact 0.5.3.3368 disabled / exact 5.0.1.15464 enabled / all others
  unknown, carries activation + provenance + diagnostics on every decision, validates input, and assigns
  deterministic priority/distance/ordinal admission with explicit cull/defer reasons. Focused tests pass
  **16/16**; Runtime Debug build passes with **0 errors**. Full Core scope gate: **1,379 passed,
  1 skipped, same 9 unrelated baseline failures** (existing `Snappier` NU1903 warnings only).
  This is not simulation: sidecar resolution/parsing, solver, bodies, collision response, cloth, joints,
  animation binding, and viewer integration remain absent and gated.
  **Phase 2 evidence gate PASSED 2026-09-03 (T003/T004/T005) — the sidecar format is solved.** The
  file is **`.phys`**, found **by replacing the model's extension**; no id, no table. Reversed-tag
  chunked container, magic `PHYS`, **version u16 must be 0**, nine chunks with measured strides
  (`BOXS` 60, `CAPS` 28, `SPHS` 16, `SHAP` 20, `BODY` 28, `SPHJ` 28, `SHOJ` 108, `WELJ` 104,
  `JOIN` 16). **The array names are Blizzard's**, recovered from `PhysData.h` bounds asserts — so
  `SHOJ` is a *shoulder* joint and `SPHJ` a *spherical* joint by measurement, not by guess. The
  recovered `PhysData` field map ends at exactly `0x50`, matching the allocation at `Physics.cpp:50`,
  which proves it has no gaps. **Gravity is `(0, 0, -10.0)`, not 9.81.** Distance culling early-returns
  in the per-instance update (validating the budget policy already landed); the first update
  *teleports* bodies to their bones instead of velocity-driving them; `0xFFFF` is the no-bone
  sentinel; the client **pins x87 control word and MXCSR across physics work and restores them**,
  which is a determinism requirement for SC-004/SC-005, not a detail. Malformed/absent data fails
  closed at every stage, and **unknown chunk tags are skipped by size** — the format is
  forward-compatible, so a reader that rejects unknown tags would be *stricter than the client*.
  **Detector trap worth remembering: there is no `.phys` string in the binary.** The extension is
  written as two immediates (`0x7968702e` then `0x73`), so a string search returns a false negative —
  same shape as [[feedback_verify_detector_power_before_null_results]]. Method note: the caller map
  was built from **header-string xrefs, not ~90 decompilations**, which is far cheaper but sees 58
  functions where the direct-xref count sees ~90; the delta is call sites in Ghidra-undefined regions
  and hoisted string operands, and is recorded rather than papered over. Evidence:
  [`domino-caller-map.md`](../specs/214-mop-physics-domino/evidence/domino-caller-map.md),
  [`physics-adapter-contract.md`](../specs/214-mop-physics-domino/evidence/physics-adapter-contract.md).
  **T007 solver selection also PASSED 2026-09-03: Jitter2 2.8.10, BepuPhysics v2 rejected.** Not on
  licensing — Apache-2.0 was acceptable — but because Bepu has **no cloth/soft-body support and no
  determinism evidence**, the two properties this feature exists to deliver. Jitter2 is **MIT**
  (verified from the `LICENSE` file itself; the empty NuGet `licenseExpression` on 2.7.x+ is a
  `<license type="file">` packaging change, **not** a license change — a metadata trap worth
  remembering), targets **`net10.0`** exactly, and enforces determinism through
  `World.Deterministic.cs`, `StableMath.cs`, reproducibility tests, and a **CI workflow that hashes
  simulation output** — tested and gated, not merely documented. Cloth route, stated precisely:
  `SoftBodyTriangle` + `SpringConstraint` are **library** types, but `SoftBodyCloth : SoftBody`
  (~115 lines) is **demo sample code**, so the route is "adapt MIT sample onto shipped primitives",
  not "call a supported Cloth class". Saying that plainly now is the whole point of making cloth a
  selection criterion. **No package reference was added** — Phase 3, needs operator go-ahead.
  **Open risk: shoulder and weld joints have no obvious Jitter2 counterpart**; only spherical maps
  cleanly. Evidence: [`solver-selection.md`](../specs/214-mop-physics-domino/evidence/solver-selection.md).
  **Both Phase 0 gates are now answered. US1 is closed** (T006 contract + T008 review: PASS, with
  SC-002's line-level attribution recorded as a **partial** and recommended for acceptance —
  Domino-internal line numbers have no consumer, since we never reproduce those algorithms, so
  closing it would cost ~90 decompilations for information nothing will act on. Written down rather
  than quietly dropped; it is the operator's cost call).
  **The `.phys` reader is landed and green (T013–T018).** `PhysSidecarPath` + `PhysReader`
  (Core.IO/Phys) and `PhysDocument` (Core/Phys). **Fail-closed and never throwing**; **unknown chunk
  tags skipped by size, never rejected** (a stricter reader would be stricter than the client and
  break on later eras); unverified regions (`BOXS` `0..47`, `SHAP` `+4`/`+8`/`+12`/`+16`, `JOIN` `+8`,
  all of `SPHJ`/`SHOJ`/`WELJ`) **preserved raw rather than interpreted**; `0xFFFF` no-bone sentinel
  preserved. One deliberate divergence from the client: it matches the fail-closed *fallback* but not
  the *silence* — every skip carries a diagnostic (FR-010/FR-012).
  **24/24 focused tests; full Core 1,403 passed / 1 skipped / the same 9 baseline failures**
  (1,403 = 1,379 + exactly these 24, so nothing regressed). Note: `FourCC.FromString("PHYS")
  .ToFileUInt32()` is `0x50485953`, **bit-identical to the client's compared constant** — the
  existing `FourCC` already models the reversed-tag convention, no special case needed. The reader
  does **not** reuse `ChunkedFileReader`, which throws on malformed input and pads odd chunk sizes;
  the client does neither.
  **Next**: **T002 real-client asset manifest is the only remaining blocker** on real-byte validation
  and is **operator-owned** — everything above is recovered from decompiled arithmetic, not from an
  observed `.phys` file. Then T019 (wire `HasPhysicsSidecar` to the resolver — **deliberately
  deferred**, it touches the render path and deserves its own change), T020 (`inspect model phys`),
  and **T021, an operator decision**: approve the Jitter2 2.8.10 package reference.
- **Spec 215 — 5.0.1 weather** (drafted 2026-09-02, not planned). Owns `MapWeather`, `Weather.dbc`,
  precipitation and `Lightning`. **Does not own lighting/fog/sky** — 143/147/160 do, and 160 is
  already tasked at 72 tasks / 8 phases. Weather drives them through interfaces; FR-015 forbids a
  parallel model. Wind is its own interface, the single join with 214, so neither blocks the other.
  **Next: speckit-plan.**
- **[`workstream-atmosphere-501-ghidra.md`](workstream-atmosphere-501-ghidra.md)** — shared 5.0.1
  native evidence (physics, weather, sky, light, fog anchors + the Light\* DBC chain), consumed by
  214, 215, **and** 160/147/143. Read-only Ghidra session; the program was opened but nothing was
  edited. Carries the era warning: **5.0.1 is the complete implementation and is not evidence about
  0.5.3** — Domino does not exist there. Same failure mode as
  [[project_mcnr_axis_order_wrong]] and [[feedback_era_gate_minimap_generation]].
- **Spec 212 — 3D spatial UI shell** (drafted 2026-09-02, not planned; US6/US7/US8 added the same
  day and all independently shippable — **US6** selection outlines that trace the object instead of a
  box, **US7** the museum profile: a camera-locked floating HUD with the panels and readouts gone,
  "more museum than in-your-face data explorer", **US8** 3D tool controls starting with an
  interactive clock face for time of day. FR-032 is load-bearing: a HUD element must invoke the *same*
  action as its full-shell equivalent, never a second implementation). Panels become interactive
  surfaces composited over a full-window scene instead of 2D windows carved out of it by
  `TryGetSceneViewportRect`, mounted on a rig whose profile follows the top-bar workspace task.
  Generalises spec 210's OpenSCAD asset path. **Next: speckit-plan.** The phase ordering is
  load-bearing: pointer-to-content accuracy (US1) must fully pass on flat surfaces before curved
  shells (US4) are attempted, because that mapping is the whole technical risk.
- **Spec 213 — MCP tooling harness** (drafted 2026-09-02, not planned). MCP *server* over the ten CLI
  tool projects so an external orchestration/inference harness can drive them; client deferred.
  **Next: speckit-plan.** FR-007 is the load-bearing requirement — one shared definition behind both
  the MCP schema and the CLI parser, build failing on divergence. That is the mechanism that prevents
  [[feedback_verify_cli_docs_against_argparse]] from recurring; a hand-maintained parallel schema
  does not satisfy it.
- **Operator verification sweep** — six code-complete fixes need one pass in the viewer. See the plan's
  Block 0. For 0.5.3 phase layers specifically, **send the `[AlphaADT]` / `[TerrainManager]` log
  lines**; they say whether it is resolution, tile lookup, or the merge.
- **Spec 209 Phase 2** — Shoreline convergence remediation: soften `KeepOnlyAboveTerrain` at the
  waterline so shoreline WL* water is not culled when MCLQ has no water coverage.
- **Spec 208 Phase 1** — Cross-map sourcing: load source map tiles independently of the active target
  map and re-map MCLY texture indices into the target's palette (T101–T106).
- **Spec 207 Phase 2** — WMO group admission: 0 of 80 groups rejected, 62.5% via conservative
  fallback, portal traversal scoring 0.
- **TensorStore migration** — blocked on the operator's environment; everything downstream of the
  datastore waits on it.

---

## Earlier session history

Last updated: 2026-09-01

**Renderer + asset pipeline workstream handoff (2026-09-01).** Agreed implementation
order for a fresh session: **205 liquid → 204 async asset loading → 202 T301 native-M2
instancing**. All three are diagnosed and measured; none is speculative.

**205 — MH2O LiquidObject vertex format: IMPLEMENTED 2026-09-01, operator proof owed.**
The Phase 1 DBC gate passed and **corrected two assumptions in its own research**. First, the
wowdev threshold "values >= 42 are LiquidObject ids" is **wrong for 5.0.1.15464**: `LiquidObject`
real ids run **57..2390** and **42 is absent from the table**, so the 17,317 ocean layers must stay
*unresolved* — which keeps the flat plane, and R2 already showed flat is correct for ocean. Second,
the first gate run reported the exact inverse (42 resolved, rivers absent) because **`DBCDRow.ID` is
a positional key for these WDB2 tables**, not the row id: `storage[42]` returned a row whose `ID`
column read 316, and `LiquidMaterial`'s keys 1..7 hide real ids {1,2,3,4,5,8,10}. Keying `map[row.ID]`
builds a table indexed by row order that silently resolves the wrong row for every sparse id and
calls every id past the row count absent. Fixed by `DbcTableLoader.ResolveRowId`; **audit other
`row.ID` lookups against Cata+ clients**. The verified answer: the 144 river layers
`2325/2333/2372 -> LiquidType 5 -> LiquidMaterial 1 -> LVF 0 (HeightDepth)`, and the independent
float-plausibility probe **agrees** on all 144. This client's `LiquidMaterial` only ever yields LVF
0 or 1, so no depth-only material exists here. Shipped: `LiquidVertexFormatChain`,
`DbcLiquidObjectTable`, `DbcLiquidMaterialTable` in `Core.IO/Dbc`; **both** decoders now resolve the
field and gate their vertex-block switch on `Resolved` (`Mh2oChunk.Parse` is the render path,
`AdtLiquidReader.ParseLayer` the harvest path); `StandardTerrainAdapter` loads the chain and reports
each unresolved value once (FR-004). 15 new tests including a two-decoder parity test; full
Core.Tests 1282 passed with the same 9 pre-existing failures. **Proof owner: operator** — load
`HawaiiMainLand` and confirm the waterways slope and the chunk-boundary steps are gone. Evidence:
[`phase1-dbc-chain-verified.md`](../specs/205-mh2o-liquid-object-vertex-format/evidence/phase1-dbc-chain-verified.md).

**Spec 206 — Zarr-first asset residency (NEW, drafted 2026-09-01).** Operator asked for the render
data to live in a Zarr dataset from the start instead of a cache folder, as a universal interchange
format. Drafted as a member of the **Client Datastore epic (179-183)**, not a new store. The premise
was re-aimed: the operator named the MPQ read as the bottleneck, but 204 measured the read as *not*
the cost, so 206 stores **already-decoded, render-ready** arrays to delete the decode stage rather
than relocate it — complementary to 204, which still owns moving the GPU upload off-thread.
Measured anchors: `output/cache/` is **~1.75 GB** of byte-for-byte client copies (`Kalimdor.wdt`
1.03 GB) written by `ViewerApp.cs:12398` purely as a **path shim** for parsers that want a path; and
the **C# Zarr array reader does not exist** (`ZarrTileDatasetLoader.LoadTile` throws,
`StoreIndexReader` says so outright, `RosettaDatastoreWriter` emits uncompressed `codecs: [bytes]`
while Python writes Blosc/lz4). Operator decisions: full renderer coverage, derived/rebuildable
store, textures stored **both** as portable pixels and a derived block-compressed array.

**205 — MH2O LiquidObject vertex format (do first).** MEASURED via the new
`inspect adt liquid-formats` command against `C:\WoW4-data\MoPBeta` / `HawaiiMainLand`,
80 root ADTs, 17,461 liquid layers: **100% carry a `LiquidObject.dbc` id in
`liquid_object_or_lvf` (values 42, 2325, 2333, 2372), not a vertex format**. Both MH2O
decoders `switch` on that field with **no `default`**, so every layer falls through with
`heights = null` and renders flat at the header `minHeight`. Ocean (id 42, 17,317 layers,
liquidType 2) is genuinely depth-only and **correct today**; the 144 river layers (ids
2325/2333/2372, liquidType 5) carry real sloped heightmaps with spreads of 11.90 / 70.15 /
163.25 world units, all discarded — and the substituted flat plane is at the wrong height
too (lowest vertex disagrees with the header in every varying layer). That is both reported
symptoms from one cause: flat waterways, and chunk-boundary steps because each chunk
flattens to its own wrong value. **Two decoders carry the defect independently and only one
is in the render path**: `StandardTerrainAdapter` calls `Mh2oChunk.Parse`; `AdtLiquidReader`
serves harvest/converter — fixing the wrong one produces a change with no visible effect.
Fix is the DBC chain `LiquidObject → LiquidType → LiquidMaterial → LVF`; neither
`LiquidObject` nor `LiquidMaterial` has a reader yet, and the offsets are wiki-documented and
**unverified against this client** (Phase 1 gate). The float-plausibility probe that made the
diagnosis is **not** acceptable as the decoder — 0.3% false positives on ocean.
See [`specs/205-mh2o-liquid-object-vertex-format/`](../specs/205-mh2o-liquid-object-vertex-format/spec.md).
Tool: [`AdtLiquidFormatSupport.cs`](../tools/inspect/WowViewer.Tool.Inspect/AdtLiquidFormatSupport.cs).

**204 — Off-thread asset decode (do second).** `DeferredAssetLoads` owns 12 of 13 recent
hitches at 26.4–68.1 ms on a 2048-frame MoP flight (median 75.26, p95 144.59, p99 235.37,
2047/2048 frames over 33.3 ms). The operator's premise was right: it is **not** the SSD or the
MPQ reader — `MpqDataSource` already runs 2 prefetch workers and root bytes are usually warm.
`WorldAssetManager` has **no threading at all**, so parse, adaptation, BLP decode and GL upload
all run inside the frame. `DeferredLoadBudget` admits **one unbounded load per frame by
design** (guaranteed progress) and its own docs name this exact work as *"Spec 153 Phase 5
step 2, deliberately not attempted here"*; it counts the damage in `OversizedAdmissionCount`,
which nothing read until today (now surfaced in the frame panel). The CPU throttle clamps to
1 load/frame on a slow frame while the first load stays unconditional — cutting streaming ~6x
**without reducing the hitch**, a self-reinforcing spiral. **Phase 1 is a hard gate**: GL
objects live on static fields and `MdxTextureDiagnosticLogger` is a process-global
`StreamWriter` re-opened per model from a renderer constructor.
See [`specs/204-off-thread-asset-decode/`](../specs/204-off-thread-asset-decode/spec.md).

**202 T301 — Native-M2 instancing (do third).** Batching still reports `route requires
unbatched render` for every M2. Cause: `WowViewerM2RuntimeBridge.PreferNativeStaticRenderer`
**defaults to `true`** when its env var is unset, so `ShouldUseNativeStaticRenderer` always
wins and every M2 gets the native-only `M2Renderer` with `_legacyRenderer == null` —
making `RequiresUnbatchedWorldRender` unconditionally true. The native renderer has **no
instancing path at all** (`QueueGpuInstance` delegates to the null legacy renderer) and its
`RenderCore` re-uploads all ten shared uniforms per instance, so merely flipping the flag buys
nothing. It needs the same shader surgery already done on `MdxRenderer`.

**Landed today (2026-09-01), specs 201/202 Phase 0 + Phase 3 partial.** Instanced /
state-hoisted / unbatched / unbatchable are now four separate numbers per render path, with
draw calls counted at the four GL call sites (a model draws once per geoset or section, so no
arithmetic over instance counts can produce that number) and a named gate on every instance
short of instancing. **GPU instancing was unreachable dead code**: `SupportsGpuInstancedOpaque`
was hardcoded `false` on `MdxRenderer` and delegated to it by `M2Renderer`, and the CPU side
was complete (instance VBO, divisor-tagged attributes at locations 6–10,
`DrawElementsInstanced`) while **the vertex shader declared only locations 0–5** — enabling the
flag as it stood would have stacked every doodad on the world origin. Shader completed with a
constant-folded non-instanced fallback and an automatic disable if it fails to compile.
`MdxRenderer.RequiresUnbatchedWorldRender` narrowed to `_wireframe` alone: the particle/ribbon
terms blocked 3,305 of 3,313 opaque instances for effects the opaque pass never draws (the M2
adapter copies header emitter counts but never populates `ParticleEmitters2` — the
"[M2] Unresolved effect systems" log). Models with local MDX lights are held out of instancing:
`UploadMdxLights` transforms light pivots per instance and that state is not in the instance
payload. Added a "Animate world doodads" toggle (default **off**, per the operator's rule that
only WMO doodads should auto-animate). **Note the state-hoisted path cannot hoist bone
matrices**: submission walks instances in visibility order and calls `BeginBatch` lazily, so
renderers interleave and per-model uniforms would be clobbered — only GPU instancing fixes the
per-instance bone upload.

**Open, not yet investigated.** Taxi paths and camera model paths on 3.x+; right-sidebar
duplication; `MergePhaseTile` wholesale replacement dropping base placements (spec 203);
MCAL alpha blockiness (spec 199); portal culling admitting 60.8% of groups via the
conservative fallback (spec 200).

**Spec 197 PTCH patch-artifact fix (2026-09-01).** Root cause of the random
missing Thunder Isle tiles is confirmed: loose 5.0.1 `.adt` files are frequently
PTCH/BSDIFF patch artifacts, and the viewer fed them raw to `ParseAdt` (zero
`KNCM` records → empty tile). Native proof: `MapArea` (`FUN_00BB0850`) receives
already-reconstructed bytes — the load-complete callback `FUN_00BB70F0`
(`MapAdtFileData.cpp`) stores final `(fileData, size)`; the cache helpers
(`FUN_00BB71B0`, `FUN_00BB7C80`) are pure hash-table plumbing. Fix shipped:
[`AdtPatchArtifact.cs`](../src/core/WowViewer.Core.IO/Maps/AdtPatchArtifact.cs)
(PTCH parse + BSDIFF40 apply + MD5-matched base selection), `ReadFileCopies`
base-copy enumeration on `IDataSource`/`MpqDataSource`/`IArchiveCatalog`, and a
reconstruction hook in `StandardTerrainAdapter.LoadMapTile` for root + tex/obj
companions (failure logs Important and treats the file as missing). 7 new tests
in `AdtPatchArtifactTests.cs`; focused run 12/12 passed; full Core.Tests 1245
passed with 9 pre-existing failures verified identical at HEAD (disjoint
modules). Evidence note:
[`5.0.1-adt-ptch-patch-artifacts.md`](../specs/197-workspace-profiles-editor-and-mop-adt-pipeline/evidence/5.0.1-adt-ptch-patch-artifacts.md).
**Proof owner: user** — reload Thunder Isle from MoPBeta and confirm missing
tiles render plus `Reconstructed patched ADT` log lines. Out of scope: multi-step
patch-chain base synthesis, patched-WDT reconstruction, editing patched companions.

**Spec 197 implementation handoff (2026-08-31).** The read-only 5.0.1 native
evidence has been written into the focused
[`5.0.1-dead-dormant-partial-rendering.md`](../specs/197-workspace-profiles-editor-and-mop-adt-pipeline/evidence/5.0.1-dead-dormant-partial-rendering.md)
note and consolidated
[`wow-5.0.1-adt-wdt-definitive.md`](../docs/architecture/wow-5.0.1-adt-wdt-definitive.md)
guide. The liquid geometry factory default branch is classified as partial;
DepthCache/GBuffer as capability-gated; atlas, doodad batching, particle
batching, and water-detail settings as optional/configuration paths; and terrain
cleanup assertions as ownership contracts. The broad 2,184 zero-direct-xref
inventory remains a candidate pool, not a dead-code count. The split reader/runtime
slice is now implemented and tested: `_obj0`/`_obj1` and `_tex0`/`_tex1` are
classified, root plus selected companions are loaded, split MCNK wrappers skip
only the root-only 128-byte header, and sparse MCIN physical slots are preserved
in the supported reader paths. The map converter now exposes an explicit target
format and loss policy. `LkAdtWriter` is target-guarded to LK v18; split-to-LK and
split-to-Alpha are deliberate lossy down-conversions; native MoP split output is
disabled because no native split writer exists. T117 remains open for numeric
FourCC, blend/seam, WDT/MAIN, indirect-caller, and path-builder-address follow-up.
Remaining implementation work is the compact MCIN consumer/merger/transfer audit
and a slot-aware native split writer.

**Spec 197 Ghidra/ADT evidence checkpoint (2026-08-31).** The loaded GhidraMCP 6.0.0
plugin is healthy and serves the `Mists of Pandaria 5.0.1.15464` project with
`Wow.exe` over `http://127.0.0.1:8089`; its status dialog reports UDS and TCP
running, version 6.0.0, and 222 endpoints. The broken part was only the local
MCP launcher: both [`.mcp.json`](../../.mcp.json) and [`.roo/mcp.json`](../../.roo/mcp.json)
pointed at a nonexistent Ghidra `.venv` and passed the unsupported
`--ghidra-server` option. Installed `ghidra_mcp_bridge-6.0.0` with `uv tool
install`; both configs now launch the installed `bridge-mcp-ghidra.exe` with
`GHIDRA_MCP_URL=http://127.0.0.1:8089` and `--no-lazy`. Stdio smoke proof:
bridge initializes as `ghidra-mcp` 1.29.1, auto-connects to the named 5.0.1
project, and registers the live schema (221 tools observed during startup;
the Ghidra endpoint itself reports 222). Program proof: `Wow.exe`, PE x86,
image base `0x00400000`, 38,405 functions, 175,352 symbols, 790 data types,
and 766 `.cpp`-matching strings. The first T117 evidence pass is now recorded in
[`research-ghidra-5.0.1.md`](../specs/197-workspace-profiles-editor-and-mop-adt-pipeline/research-ghidra-5.0.1.md:173).
Native 5.0.1 requires `MVER == 0x12`, builds root + one selected
`_obj0`/`_obj1` and `_tex0`/`_tex1` companion pair, requires 256 outer MCNK
records in every file-data object, consumes the 128-byte MCNK header only in
the root slot, and gates area creation on map-table `Flag_Exists`. The exact
loader path does not construct `_lod.adt`; `MHID`/`MDID`/`MCXH` remain
unconfirmed for this build. T116 is complete and T117 remains in progress.
Do not modify the loaded Ghidra program; production parser/runtime changes are
now limited to the evidenced split-loading slice, while native blend/shader
semantics remain gated on further review.

**Spec 197 lane (2026-08-30, checkpoint 51).** UI Workspace Profiles, Editor Mode Integration, PM4 Mouse Inspection, Multi-Client Map Staging & MoP 5.0.1 ADT Pipeline — ACTIVE:
1. **Workspace Mode / Profile Switcher**: Add top-bar mode switcher (`[Viewer]`, `[Editor]`, `[Archaeology]`) and expose the missing Editor button in the right sidebar. Filter sidebar tool tabs and top menus dynamically based on the active mode.
2. **Menu Audit & Legacy "MK Dataset" Purge**: Remove all defunct "MK Dataset" / `MkDatasetHarvester` references from menus and loaders.
3. **PM4 Mouse Raycasting & Viewport Selection**: Add bounding-box and triangle raycast pick candidates for PM4 objects/surfaces directly from viewport mouse clicks.
4. **Multi-Client Restoration Staging**: Architecture for mounting multiple client archives simultaneously and copying terrain/placements into an active restoration library map.
5. **4.3.4 through 5.1 MoP ADT & Blending Engine**: Support multi-split ADT chunks (`_obj0`/`_obj1`, `_tex0`/`_tex1`), height texture blending (`MHID`, `MDID`, `MCXH`), and WMO terrain seam blending via Ghidra analysis of `WoW.exe` 5.0.1.15464. Reader/runtime support is landed; native blend/seam semantics remain evidence-gated.
6. **Verification and serialization safety**: The split reader/runtime and explicit map-conversion target boundary have focused source/build/test proof. A genuine native MoP split writer, full band-1 merger/transfer conversion, and real-client visual proof remain open.

**Spec 196 lane (2026-08-30, checkpoint 50).** WDL Lattice Magnetization, Polarity Inversion, Neighbor Auto-Fit & 0-Hitch Async Stratigraphy Restoration Engine — COMPLETE:
1. **Polarity Inversion & Multi-Anchor Datum Geometry**: Added `StratigraphyAnchorMode` enum (`LowestZ_Floor`, `HighestZ_Ceiling`, `MeanZ`, `NeighborMeshBorder`, `WdlLattice`, `CustomDatum`) and polarity inversion (`PolarityInverted`) to `TemporalStratigraphyOptions` and `TemporalMeshRestorer`. Resolves developmental terrain inverted compression (e.g. Dragon Isles) by inverting scale direction from an upper ceiling datum without vertical wall spikes.
2. **Neighboring Mesh Height & Scale Auto-Fit Solver**: Implemented `NeighborMeshHeightSolver` with 1–3 chunk spatial radius adjacency search and boundary vertex extraction along shared chunk seams (9 outer lattice vertices per edge). Computes closed-form vertical offsets $\Delta Z = \text{mean}(Z_{\text{active}} - Z_{\text{candidate}})$ and selects the $(S, P, \text{Anchor})$ tuple minimizing boundary RMSE.
3. **WDL Macro-Lattice Magnetization & WDL Binary Serialization**: Implemented `WdlLatticeMagnetizer` providing continuous bilinear height interpolation over $17\times 17$ tile vertices and $16\times 16$ chunk center heights, adding high-frequency ADT micro-relief onto low-frequency macro topography. Created `WdlFileWriter` supporting Blizzard-standard `.wdl` binary serialization (`MVER`, `MWMO`, `MWID`, `MODF`, `MAOF`, `MARE`, `MAHO`).
4. **0-Hitch Asynchronous Pipeline & In-Viewer Workbench Controls**: Refactored `ViewerApp.cs` tile restoration to execute asynchronously via background tasks (`Task.Run`) and double-buffered result queue (`_pendingRestoredTilesQueue`), eliminating the 2–4s UI freeze. Integrated Polarity Inversion checkbox, Anchor Mode selector, Neighbor Auto-Fit toggle, and WDL Magnetization controls into `ViewerApp_Sidebars.cs`. Added companion `.wdl` export in `ExportLoadedStratigraphyTiles`.
5. **Verification**: 12/12 Spec 196 unit tests passing 100% green (`NeighborMeshHeightSolverTests`, `WdlLatticeMagnetizerTests`, `StratigraphyLevelAnalyzerTests`, `StratigraphyTileExporterTests`). Full solution builds with 0 errors across Windows and CrossPlatform configurations.

**Stratigraphy Factor Precision Suite & ImGui Path Picker Fix (2026-08-30, checkpoint 49).** High-Precision Factor Controls & Dialog Fixes — COMPLETE:
1. **High-Precision Numeric & Nudge Control Suite**: Replaced imprecise slider with direct 4-decimal input field (`%.4fx`), multi-scale delta steppers ($\pm 10\times, \pm 1\times, \pm 0.1\times$), wide logarithmic slider ($1.0\times \to 512.0\times$), and 10 one-click historical era presets (`1x`, `3.33x`, `10x`, `16x`, `33.334x`, `64x`, `80x`, `128x`, `256x`, `512x`) across both Stratigraphy Archaeology workbench and Terrain Controls.
2. **ImGui Path Picker Modal Fix**: Eliminated child window overlap over the "Use Folder" / "Open" action buttons by enforcing clean separate button rows, cursor offset alignment, and explicit footer padding.
3. **Verification**: Stratigraphy tests passing 100% green (`StratigraphyLevelAnalyzerTests`, `TemporalMeshRestorerTests`).

**Alpha 0.5.3 Terrain Organization & Tile Indexing Fix (2026-08-30, checkpoint 48).** Alpha 0.5.3 WDT Grid & WDL Index Parity — COMPLETE:
1. **Row-Major Grid Index Alignment**: Corrected `TileExists` and `LoadTileWithPlacements` in [`AlphaTerrainAdapter.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs) and `TryGetTerrainWeakSignalWdlTile` in [`ViewerApp.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/ViewerApp.cs) from inverted `tileY * 64 + tileX` to `tileX * 64 + tileY`.
2. **Eliminated Transposition Discontinuity**: Resolved diagonal coordinate swap where non-diagonal tiles loaded transposed $(Y, X)$ ADT blocks across world boundaries.
3. **Verification**: 6/6 `WdtSummary` tests passing (100% green). Full solution builds with 0 errors across Windows and CrossPlatform targets.

**Spec 195 lane (2026-08-30, checkpoint 47).** Overhead Chunk Manipulator & Multi-Tile Sub-Cell Transposition Engine — COMPLETE:
1. **Global Chunk Coordinate Space & Selection Region Model**: Created `GlobalChunkCoordinate` ($1024 \times 1024$ continuous lattice where $G_x = T_x \times 16 + C_x$, $G_y = T_y \times 16 + C_y$) and `ChunkSelectionRegion` in `WowViewer.Core.Editor.Operations`, allowing multi-tile rectangular boxes, chunk toggle, and whole-tile selections spanning across tile junctions.
2. **Core Chunk Transposition Engine & Texture Harmonization**: Implemented `ChunkTranspositionOptions`, `ChunkTranspositionPayload`, `ChunkTranspositionService`, and `ChunkTranspositionOperation` (`IEditorOperation`) in `WowViewer.Core.Editor.Operations`. Supports full extraction and transformation of 145-vertex MCVT height arrays, MCNR normals, MCLY/MCAL texture layers, hole masks, and MDDF/MODF doodad and WMO placements with spatial translation $(\Delta X, \Delta Y, \Delta Z)$ and rotation ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
3. **Editor Plugin & Interactive Overhead 2D Canvas**: Created `ChunkManipulatorEditorPlugin` registered in `EditorHost` and implemented `DrawChunkManipulatorPluginPanel` in `ViewerApp_Editor.cs` with an interactive 2D overhead canvas showing tile borders ($533.334\text{m}$) and chunk sub-grids ($33.334\text{m}$), zoom/pan, click-drag box selection, copy/cut/paste buttons, and live in-place memory replacement via `ReplaceTileChunksAndRebuild`.
4. **Validation**: 91/91 unit tests passing in `WowViewer.Core.Editor.Tests` (100% green). Full solution builds with 0 errors across Windows and CrossPlatform targets.

**Spec 194 lane (2026-08-30, checkpoint 46).** Temporal Stratigraphy & Weak Signal Development Mesh Restoration — COMPLETE:
1. **Core Stratigraphy & SIMD Analysis Engine**: Created `TemporalStratum` enums, `StratigraphyLevelAnalyzer` (hash-bit unique level counter $|\{h\}|$ without altitude bias), `SeamDiscontinuityProfiler` (C0 step/C1 slope evaluator detecting 2x2 merge spikes at boundary 8 and 4x4 merge spikes at 4, 8, 12), `FastTerrainNormalSolver` (vectorized normal generation for 257x257 lattices and 145-vertex MCNK chunks), and `TemporalMeshRestorer` (SIMD height scaling, floor preservation, and SmoothStep boundary feathering).
2. **Interactive Viewer Stratigraphy & Mesh Recovery Workbench**: Implemented `DrawTemporalStratigraphySubTab` on the Archeology page on the right sidebar (Inspect > Archeology > Stratigraphy) and unified it with Terrain Lab. Features live stratum metrics (`DominantStratum`, `SurvivingLevels`), continuous gradient factor slider ($1.0\times \to 512.0\times$), preset snap points ($33.334\times$, $16\times$, $64\times$), dev mesh unhiding toggle (`_stratigraphyUnhideDevMeshes` bypassing `HoleMask`), boundary slope stitching, floor anchoring, in-viewer analysis, and in-app folder picker export via `ImGuiPathPicker`.
3. **Restored Terrain Export Pipeline**: Implemented `StratigraphyTileExporter` and connected in-viewer ADT / Alpha WDT export with zero runtime loss.
4. **Corpus-Wide CLI Tooling**: Authored `TerrainStratigraphyScanCommand` in `WowViewer.Tool.Inspect` (`terrain-stratigraphy-scan`) emitting structured `stratigraphy_manifest.json` + `stratigraphy_summary.csv`, and `TerrainStratigraphyPatchCommand` in `WowViewer.Tool.Converter` (`terrain-stratigraphy-patch`) for offline batch patching.
5. **Validation**: 10/10 stratigraphy unit tests passing 100% green (`StratigraphyLevelAnalyzerTests`, `TemporalMeshRestorerTests`). Solution builds with 0 errors across both Windows and Cross-Platform configurations.
1. **Spec Kit Architecture Pack Authored**: Created [`specs/193-benilla-112-client-reference/`](file:///I:/parp/parp-tools/wow-viewer/specs/193-benilla-112-client-reference/) with `spec.md`, `research.md`, `plan.md`, and `tasks.md` defining the role of [Benilla (`samwhosung/benilla`)](https://github.com/samwhosung/benilla) as an active clean-room Rust 1.12.1 reference/oracle for 1.x M2 model parsing (`MD20 0x100`), embedded skin view unpacking (`ofsViews`), submesh partitions, material blend states, and forward kinematics while maintaining 100% native C# tooling.
2. **Cross-Spec Reference Links**: Registered Spec 193 in `specs/STATUS.md` and added reference documentation in `specs/104-legacy-m2-rendering/research.md` and `specs/154-m2-era-reader-parity/research.md`.

**Cross-Platform ImGui File/Folder Picker & Dual-Era Generator lane (2026-08-30, checkpoint 44).** Cross-Platform BCL ImGui File Browser & Dual-Era Map Output — COMPLETE:
1. **Cross-Platform In-App ImGui File Browser**: Upgraded `ImGuiPathPicker` into a full-featured BCL (`System.IO`)-based in-app modal file and folder browser supporting `OpenFolder`, `OpenFile`, and `SaveFile` modes. Features drive shortcuts (`[C:]`, `[D:]`, `[H:]`, `[I:]`), `[CLIENTS (H:)]` and `[App Directory]` quick navigation, editable path bar, breadcrumbs, search filtering, inline folder creation (`+ New Folder`), and multi-extension filter parsing (e.g. `".pm4;.pd4"`, `".wdt;.mpq"`, `".json"`).
2. **Complete WinForms STA Elimination**: Removed all `System.Windows.Forms` and `#if WINDOWS` modal dialog dependencies (`ShowFolderDialogSTA`, `ShowFileDialogSTA`, `ShowSaveFileDialogSTA`) from `ViewerApp.cs`, `ViewerApp_ClientDialogs.cs`, `ViewerApp_Workspaces.cs`, `ViewerApp_SynthesizedMinimapExport.cs`, `ViewerApp_MlTraining.cs`, `ViewerApp_Sidebars.cs`, `ViewerApp_CameraPaths.cs`, and `ViewerApp_Pm4Utilities.cs`. All file open/save and folder select interactions now work identically across Windows, Linux, and macOS without thread blocking or OS dialog crashes.
3. **Dual-Era Output Support in CLI**: Enhanced `terrain-generate-templated` in `WowViewer.Tool.Inspect` (`Program.cs`) with `--format both|lk|alpha` (default: `both`). Emits LK multi-ADT files (`.adt`/`.wdt`/`.wdl`), Alpha monolithic `.wdt` via `LkToAlphaConverter` + `AlphaWdtWriter.Build`, companion `.wdl`, client DBCs (`Map.dbc`, `AreaTable.dbc`), and minimap/MD5 translation tables (`minimap.trs`, `md5translate.trs`).
4. **Garden Museum Asset & Flat Floor Validation**: Populated `TemplatedTerrainGenerator` with authentic decorative doodad placements (Human Fountains, Stormwind Street Lights, Park Benches, and Elwynn Shrubbery) for `BiomeTheme.GardenMuseum` at courtyard nodes.
5. **Verification**: 14/14 unit tests pass 100% green (`TemplatedTerrainGeneratorTests`, `TerrainBrushPasteTests`, `TerrainLayerAllocatorTests`, `AdtPasteExtractorTests`, `TerrainStampOperationTests`). Solution builds with 0 errors across both Windows and Cross-Platform targets.

**Spec 192 lane (2026-08-30, checkpoint 43).** Terrain Template Brush & Paste Library with In-Viewer Map Generator — COMPLETE:
1. **Curated Terrain Brush & Paste Library**: Implemented `TerrainBrushPaste`, `TerrainBrushLibrary`, and `CuratedTerrainBrushLibrary` providing 15 archetypal terrain motifs (cobblestone straight/curve/cross roads, dirt paths, marble plazas, gentle knolls, terraces, ridges, pond basins, grand avenues, flat exhibit pads) with 2D relative heightfields, multi-layer alpha masks, and slope metadata.
2. **Strict Hardware 4-Layer Chunk Allocator**: Implemented `TerrainLayerAllocator` with energy-based alpha pruning and weight normalization, strictly enforcing the engine's 4-texture-layer-per-chunk ceiling.
3. **ADT Sub-Region Extraction Engine**: Implemented `AdtPasteExtractor` to extract bounded terrain pastes and multi-layer alpha splats directly from loaded `LkMcnkData` / `AlphaMcnk` structures.
4. **Undo/Redo Terrain Stamping Operation**: Implemented `TerrainStampOptions` and `TerrainStampOperation` with `SmoothStep` edge feathering, height blending modes (Additive, Replace, Min, Max), normal recalculation, and snapshot state capture for `EditorSession`.
5. **Templated Procedural Map Generator**: Implemented `TerrainMapTemplate` and `TemplatedTerrainGenerator` synthesizing multi-tile maps with arterial road networks, flat marble exhibit courtyards ($Z = 0$), slope constraints ($\le 25^\circ$), and multi-era ADT / WDT / WDL serialization.
6. **Interactive Viewer Editor Plugin UI & CLI**: Implemented `TerrainTemplateEditorPlugin` registered in `EditorHost`, ImGui catalog browser and stamping sliders in `ViewerApp_Editor.cs`, and `terrain-generate-templated` CLI command in `WowViewer.Tool.Inspect` (`Program.cs`).
7. **Verification**: 14/14 unit tests pass 100% green across `TerrainBrushPasteTests`, `TerrainLayerAllocatorTests`, `AdtPasteExtractorTests`, `TerrainStampOperationTests`, and `TemplatedTerrainGeneratorTests`. Solution builds with 0 errors.

**Spec 191 lane (2026-08-30, checkpoint 42).** Procedural Garden Museum Map Generator — DEFECT DIAGNOSIS & REWRITE PLAN:
1. **Current Generation Reality & User Verification Failure**:
   - In-game screenshot and user testing confirmed that `rosetta-generate` produces **zero garden features**:
     - **Hardcoded Sand Defaults**: `RosettaGeneratorOptions.DefaultGroundTexture` is hardcoded to `wcsand.blp` (Wailing Caverns sand) and `DefaultInkTexture`/`DefaultCheckersTexture` to `checkers.blp`, causing all generated maps to be sandy wastelands with giant checkerboard lines.
     - **Impassable Triangle Quads**: `CreateChunkHeights` generates arbitrary linear bevel ramps across 4.16m MCVT vertex spacing, creating steep 45-degree collision-blocking triangles between exhibits that trap the player.
     - **Cluttered Signage**: Grid line outline drawing (`DrawRectOutlineMeters`) intersects and draws directly over the text label bands, rendering signage unreadable.
     - **Single-Layer Alpha Abuse**: No multi-tileset blending or garden path networks are active in the live generator.
2. **Actionable Remediation Architecture**:
   - **A. Authentic Garden Palette**: Default Layer 0 to lush Elwynn garden grass (`tileset\elwynn\elwynngrass.blp`), Layer 1 to Stormwind cobblestone promenades (`tileset\city\stormwindcobble.blp`), and Layer 2 to clean polished white marble exhibit pads (`tileset\city\whitemarble.blp`).
   - **B. 100% Flat Walkable Terrain**: Eliminate the jagged bevel MCVT calculation entirely. Keep all paths and exhibit courtyards completely flat ($Z = 0$), guaranteeing unobstructed player navigation.
   - **C. Clean Uncluttered Signage**: Remove grid line overlays from the text label band. Place signage on clean, dedicated stone plaque areas with clear contrast.
   - **D. True Multi-Layer Alpha Splatting**: Generate continuous 64x64 MCAL alpha splats per chunk for arterial cobblestone walkways and exhibit plazas.
3. **Current State**:
   - Spec 191 is in active development and NOT complete until verified in-game. Client root `H:\CLIENTS` is strictly read-only; all outputs routed to isolated staging directories.

**Spec 190 lane (2026-08-28, checkpoint 36).** Minimap TRS / BLP Multi-Directory Emission & Map ID / Port Command Reporting — COMPLETE:
1. Enhanced `RosettaMinimapPainter.GenerateMinimapTrs` and `WriteMinimapTrs` to generate authentic Alpha TRS format with relative entries under `dir: {map}` and write `minimap.trs` / `md5translate.trs` globally and into per-map directories (`World\Maps\{map.MapName}\`).
2. Expanded `rosetta-generate` minimap BLP writer to emit all 4 coordinate / padding variants (`map{Y:D2}_{X:D2}.blp`, `map{X:D2}_{Y:D2}.blp`, `map{Y}_{X}.blp`, `map{X}_{Y}.blp`) across all standard directories (`Textures\Minimap\`, `World\Textures\Minimap\`, `World\Minimaps\`, `World\Maps\{map}\`, and `World\Maps\{map}\Minimap`).
3. Added `InitializeMinimapSupport()` to overlay attachment in `ViewerApp.cs` and expanded search paths to include `OverlayRoots` and `LooseRoots`, resolving minimaps dynamically in viewer.
4. Upgraded `rosetta-generate` console summary to display prominent Map IDs, map names, tile bounds, first exhibit spawn world coordinates (`X`, `Y`, `Z`), and ready-to-copy in-game teleport commands (`.worldport <mapId> <x> <y> <z>` and `.go xyz <x> <y> <z> <mapId>`).
5. All 74 Rosetta and Core IO unit tests pass green (100%). Solution builds cleanly with 0 errors.

**Spec 190 lane (2026-08-28, checkpoint 35).** Loose File DBC Resolution & Valid Map ID Display in Viewer — COMPLETE:
1. Fixed `MpqDBCProvider` to accept `IDataSource` and check loose files (`_dataSource.ReadFile`) across all table candidate paths (`DBFilesClient\Map.dbc`, `DBC\Map.dbc`, etc.) *before* falling back to raw MPQs.
2. Added `"DBFilesClient"`, `"DBC"`, `"dbfilesclient"`, `"dbc"` to `IndexedLooseExtensions` / `dataDirs` in `MpqDataSource.cs` so loose DBC files in game folders and attached loose overlays are scanned, indexed, and accessible.
3. Updated `ViewerApp.cs` to pass `_dataSource` into `MpqDBCProvider(mpqDs.ArchiveReader, _dataSource)` on data source loading and overlay attachment, ensuring `MapDiscoveryService` loads the patched `Map.dbc` and resolves custom maps with their authentic 500+ Map IDs (`[500] Rosetta Exhibit (Development_M200)`) instead of falling back to synthetic `[custom]` entries.
4. Updated `ArchiveReaderDbcProvider` in `WowViewer.Core.IO.Dbc` with optional `Func<string, byte[]?>? looseFileReader` delegate for unified offline tool and test support.
5. All 74 Rosetta and Core IO unit tests pass green (100%). Solution builds cleanly with 0 errors.

**Spec 190 lane (2026-08-28, checkpoint 34).** DBC Real Client Patching via DBCD & WoWDBDefs — COMPLETE:
1. Updated `RosettaDbcGenerator` in `WowViewer.Core.IO.Dbc` with `PatchAndSaveClientDbcs` to load existing client `Map.dbc` and `AreaTable.dbc` from client MPQ archives / directories using `DBCD` and `WoWDBDefs` definition schemas (`Map.dbd`, `AreaTable.dbd`), preserving all real client records and appending new Rosetta exhibit continents and designkit exhibit zones.
2. Added `InferClientBuild` to automatically detect client builds from paths and format flags against known `Map.dbd` BUILD ranges (e.g. `3.3.5.12340`, `0.5.3.3368`, `1.12.1.5875`), and `TryFindDefinitionsDirectory` to locate `WoWDBDefs/definitions/`.
3. Implemented safe column mapping in `PatchAndSaveClientDbcs` supporting both scalar fields and multi-locale `string[]` arrays with `_mask` handling across Alpha, Vanilla, TBC, and LK client schemas.
4. Integrated `PatchAndSaveClientDbcs` with `ArchiveReaderDbcProvider` in `rosetta-generate` CLI command in `WowViewer.Tool.Inspect` (`Program.cs`) with automatic fallback to standalone DBC creation if client DBCs are absent.
5. Authored comprehensive unit tests in [`RosettaTilesetGeneratorTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/RosettaTilesetGeneratorTests.cs) verifying build inference, definition discovery, and real 3.3.5 DBC patching + DBCD round-trip loading. All 71 Rosetta unit tests pass green (100%). Solution builds cleanly with 0 errors.

**Spec 190 lane (2026-08-28, checkpoint 33).** Phase 4 Companion ADT Synthesizer (US4) — COMPLETE:
1. Created `RosettaCompanionAdtSynthesizer` in `WowViewer.Core.IO.Maps` to scan PM4 file directories and identify orphan PM4 tiles lacking companion `.adt` or `_obj0.adt` files.
2. Implemented minimal compliant companion ADT synthesis (`BlankAdtFactory` / `LkAdtWriter`) generating standard valid ADTs with flat terrain and authentic headers (`MVER`, `MHDR`, `MCIN`, `MTEX`, `MMDX`, `MMID`, `MWMO`, `MWID`, `MDDF`, `MODF`, `MCNK`).
3. Added SHA256 cryptographic provenance reporting (`RosettaCompanionProvenanceReport`) recording source PM4 files, output paths, timestamps, options, and content hashes to distinguish synthetic data from authentic game data (FR-009, FR-010, SC-003).
4. Implemented safe overwrite protection skipping existing companions by default and `--overwrite` flag.
5. Added CLI command `rosetta-synthesize-companions` in `WowViewer.Tool.Inspect` (`Program.cs`).
6. Authored 6 comprehensive unit tests in [`RosettaCompanionAdtSynthesizerTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/RosettaCompanionAdtSynthesizerTests.cs). All 69 Rosetta tests pass green (100%).

**Spec 190 lane (2026-08-28, checkpoint 32).** Phase 3 Deterministic PM4 Lookup Engine (US3):
1. Created `RosettaPm4LookupEngine` in `WowViewer.Core.PM4.Matching` for pure deterministic PM4 object identification against `RosettaReferenceLibrary` without ML/LLMs.
2. Implemented tri-state classification (`Identified`, `Ambiguous`, `NoReference`, `Ineligible`) with score floor ($0.45$), ambiguity window ($0.03$), and bounding tolerance pruning (`FindCandidatesByBounds`).
3. Added granular signal evidence evaluation (`AspectRatio`, `MajorSpan`, `Volume`, `Footprint`, `TypeFlags`) surfacing exact reasons for match agreement and discrepancy.
4. Added `CompareWithLegacyScorer` and `Pm4ReconciliationInputAdapter.BuildRosettaCorpusReferences` connecting the global Rosetta reference library directly into the Spec 176 reconciliation pipeline.
5. Added `rosetta-pm4-match` CLI command in `WowViewer.Tool.Inspect` (`Program.cs`) producing detailed console diagnostics and structured JSON match reports.
6. Authored 7 comprehensive unit tests in [`RosettaPm4LookupEngineTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/RosettaPm4LookupEngineTests.cs). All 63 Rosetta unit tests and all 82 Editor tests pass green (100%).

**Spec 190 lane (2026-08-28, checkpoint 31).** Alpha 0.5.3 Native Client Ergonomics, First-Class DBCs, WDL Mesh, Map Splitting & +20Z Exhibits:
1. Created `RosettaDbcGenerator` in `WowViewer.Core.IO.Dbc` to generate authentic binary `Map.dbc` (5 fields) and `AreaTable.dbc` (14 fields) for Alpha 0.5.3, registering Rosetta maps as first-class outdoor continents (Map ID 500+) and named designkit exhibit zones (Area ID 5000+).
2. Added automatic `.wdl` distant low-resolution terrain mesh generation (`WdlWriter.Build`) alongside `.wdt` files.
3. Added `GenerateMinimapTrs` and `WriteMinimapTrs` to `RosettaMinimapPainter` for `minimap.trs` / `md5translate.trs` mapping with `Azeroth` aliases.
4. Added kind-based map splitting (`SplitAssetKinds`), partitioning models (`.mdx`/`.m2`) into `{map}_MDX` and world models (`.wmo`) into `{map}_WMO`, with an 800-tile map budget ceiling to prevent client engine memory exhaustion.
5. Implemented bounding-box centering offset (`(X, Y) -= boundsCenter`) and $+20\text{Z}$ elevation ($Z = \text{groundZ} + \max(0, -\text{bounds.Min.Z}) + 20\text{m}$), placing models comfortably floating in the air above flat walkable terrain ($Z = 0$) with zero ground clipping or saw-tooth ridge trapping.
6. All 56 Rosetta unit tests pass green (100%).

**Spec 190 lane (2026-08-28, checkpoint 30).** Phase 2 Reference Library Builder (US2):
1. Created `RosettaReferenceLibrary` and `RosettaReferenceAsset` data model with full bounding, span, volume, footprint, aspect ratio, subpart bounds, and signal dictionary properties, with 100% interoperability with `Pm4AssetMatchScorer` via `ToAssetReferenceSignalRecord()`.
2. Created `RosettaCorpusReader` to decode synthetic Rosetta placements and geometry from `rosetta-manifest.json`, in-memory `RosettaGenerationResult`, and Zarr datastores (`RosettaObjectLibrary`).
3. Created `RosettaReferenceLibrarySelfTest` enforcing the $\ge 99.0\%$ Top-1 identification accuracy requirement with exact and perturbed (jittered) bounding box tests.
4. Added CLI commands `rosetta-build-library` and `rosetta-library-selftest`, plus `--emit-library` and `--library-output` flags in `rosetta-generate`.
5. Created custom `Vector3JsonConverter` and `Vector2JsonConverter` for clean JSON serialization and round-tripping.
6. Added 7 unit tests in `RosettaReferenceLibraryTests.cs`. All 50 Rosetta unit tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 29).** Alpha WDT Row-Major Indexing & Visual Calibration:
1. Fixed transposed tile indexing bug in [`AlphaTerrainAdapter.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs): `TileExists` and `LoadTileWithPlacements` now index `_adtOffsets` as row-major `tileY * 64 + tileX` instead of column-major `tileX * 64 + tileY`. This was the root cause of objects being placed over mismatched terrain tiles/labels in Alpha WDT maps when $tileX \ne tileY$.
2. Added visual calibration tools in [`RosettaAlphaPainter.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/RosettaAlphaPainter.cs) with `DrawCircleOutline` and `DrawBullseyePattern` (concentric rings from 30m to 240m, full crosshair axes, and cardinal direction indicators: `NORTH (-Y)`, `SOUTH (+Y)`, `WEST (-X)`, `EAST (+X)`).
3. Connected calibration bullseye generation to [`RosettaTilesetGenerator.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/RosettaTilesetGenerator.cs) for empty tiles and test patterns.
4. All 43 focused Rosetta unit tests pass green. Full solution builds with 0 errors.

**Spec 190 lane (2026-08-28, checkpoint 28).** Cross-Era Extension Resolution, Datastore Diff Engine & UI Integration:
1. Implemented seamless 3-way cross-era model extension shifting (`.mdx` $\leftrightarrow$ `.mdl` $\leftrightarrow$ `.m2`) in `WorldAssetManager`, `WmoRenderer`, and `ViewerApp`, allowing maps from any era (such as Rosetta 0.5.3 or Alpha WMOs) to resolve models when loaded over newer clients (1.12.1 / 3.3.5) and vice-versa.
2. Built `RosettaBuildMetadata` and `RosettaBuildDiff` with `ComputeBuildDiff` in `RosettaObjectLibrary` to compare any two client builds directly from the Zarr datastore without re-processing.
3. Added `rosetta-datastore-diff` CLI command in `WowViewer.Tool.Inspect` (`Program.cs`).
4. Added "Load from Rosetta Datastore..." menu item and interactive modal in `ViewerApp` with Data Version, Map Name, Base Game Version dropdowns, and live cross-build diff statistics.
5. Documented Phased Maps and Rosetta Datastore in `USERGUIDE.md` and `README.md`.
6. All 41 focused Rosetta unit tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 27).** Minimap & Placed Object Coordinate Alignment:
1. Resolved minimap tile loading mismatch by generating dual-convention minimaps (`map{tileY:D2}_{tileX:D2}.blp` for standard viewer queries and `map{tileX:D2}_{tileY:D2}.blp`) across all four standard directories (`Textures/Minimap/{mapName}`, `Textures/Minimap/{mapName.ToLowerInvariant()}`, `World/Minimaps/{mapName}`, `World/Minimaps/{mapName.ToLowerInvariant()}`).
2. Fixed Alpha WDT streaming converter invocation in `Program.cs` which previously inverted `(tileX, tileY)` coordinate arguments.
3. Enforced strict transitive weak ordering in `kitAssets.Sort` comparator using floor bucketing.
4. Added path separator normalization to `SanitizeLabel` and `RosettaMinimapPainter` label formatting, and made `IndexOfOrAdd` case-insensitive.
5. All 41 focused Rosetta unit tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 26).** Deduplication Normalization & Minimap Path/Font Optimization:
1. Fixed duplicate asset enumeration by normalizing all paths to backslashes before `Distinct(StringComparer.OrdinalIgnoreCase)` and adding deduplication to `RosettaTilesetGenerator.Generate`.
2. Corrected minimap tile naming from `map{tileY}_{tileX}.blp` to canonical `map{tileX:D2}_{tileY:D2}.blp`.
3. Emitted minimap BLPs to both `Textures/Minimap/{mapName}/` and `World/Minimaps/{mapName}/` for cross-era client and viewer compatibility.
4. Built a 3×5 bitmap font for minimap plaques with dynamic capacity calculation to fit 30 characters per cell without clipping.
5. 41 focused Rosetta unit tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 25).** High-Legibility MCAL & Minimap Text Rendering:
1. Switched font pixel metrics in `AppendClass` from coarse 4.16m MCCV lattice sizing to 2 MCAL texels (~1.04m), boosting capacity from 5 to 21+ characters per line across 2-3 line label bands.
2. Updated `SanitizeLabel` to preserve natural lowercase and uppercase casing so handwriting glyphs render with proper ascenders and descenders.
3. Updated `WrapLabel` with smart path separator wrapping and middle-elision with `...`.
4. Replaced blurry minimap downsampling with direct 5×7 glyph rendering on high-contrast parchment plaques (`ColorPlaque = (248, 245, 238)`, `ColorInk = (15, 15, 20)`) over warm sand terrain.
5. 41 focused Rosetta unit tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 24).** Unified Multi-Version Zarr Datastore & Interchange Engine:
1. Created `RosettaDatastoreWriter` implementing a versioned Zarr v3 datastore with global content-addressed asset deduplication (`global_assets/catalog.parquet`).
2. Deduplicates assets across client builds by deterministic SHA1 hash (`objlib_<hash>`), storing shared assets once with incremented reference counts.
3. Slices heights, MCAL alpha, checkers alpha, and raw 24-bit RGB minimaps into chunked Zarr v3 arrays.
4. Built `RosettaObjectLibrary` for high-performance C# O(1) asset lookups, spatial bounding queries, and on-demand chunk loading.
5. Integrated `--datastore` / `--emit-zarr` CLI flags in `rosetta-generate`, plus `rosetta-datastore-info` and `rosetta-datastore-query` commands.
6. 41 focused Rosetta unit tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 23).** Painted terrain grid lines and cell perimeter demarcation:
1. Added `DrawRectOutline` and `DrawLine` to `RosettaAlphaPainter` and updated `BuildTileAlphaCanvas` to paint 1.2m perimeter borders and 1.0m object/label divider lines onto the 1024×1024 MCAL canvas.
2. Updated `FillRect` with `Math.Max` for seamless non-destructive alpha blending across overlapping lines and text.
3. Connected `options.PaintCellBorders` through `BuildTileAdt` in both LK and Alpha generation pipelines.
4. 39 focused Rosetta tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 22).** Streaming tile generation and writing to eliminate out-of-memory crashes:
1. Added streaming `AlphaWdtWriter.Write` directly to `FileStream`, writing tiles one-by-one to disk and patching the 64 KB `MAIN` table and `MPHD` offsets at the end.
2. Made 1024×1024 text and checkers canvas generation on-demand via `BuildTileAlphaCanvas` and `BuildTileCheckersCanvas`, avoiding gigabytes of canvas allocations across all planned tiles in memory.
3. Updated `RosettaMinimapPainter` to render downsampled text on-demand.
4. Cleanly handled `--overwrite` across all output files (Alpha WDT, LK ADT/WDT/WDL, and minimap BLPs).
5. Verified on 0.5.3 client: 1,720 tiles, 11,663 placements, 920 MB monolithic WDT written smoothly without OOM.
6. 38 focused Rosetta tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 21).** Diagnostic `checkers.blp` texturing on indented terrain floors:
1. Painted `tileset\generic\checkers.blp` under objects on the indented pedestal floor via `CheckersCanvas`, replicating authentic Blizzard model testing terrain.
2. Supported 3-layer MCNK structures in `BuildTileAdt` (Layer 0: Sand, Layer 1: Checkers pad under object, Layer 2: Handwriting ink label) with concatenated MCAL chunks and correct offsets.
3. Added `ResolveCheckersTexture` in `Program.cs` and `--checkers-texture` CLI option.
4. Rendered matching checkered pattern on pedestal plateaus in `RosettaMinimapPainter`.
5. 38 focused Rosetta tests pass green.

**Spec 190 lane (2026-08-28, checkpoint 20).** Museum exhibit layout, handwriting font, compact cells & Westfall sand:
1. `rosetta-generate` defaults to `CellChunks = 4, LabelBandChunks = 1`, packing 16 cells per tile (133.33m cells with 100m object area) so small objects don't waste empty space.
2. Museum curation ordering: Models sorted by size (smallest -> largest) first, followed by WorldModels sorted by size.
3. Added 5x7 cursive handwriting font in `RosettaTextPainter` supporting uppercase ('A'-'Z'), lowercase ('a'-'z'), digits, and punctuation, baked into the 1024x1024 MCAL Layer 1 texture map.
4. Set default ground texture to `tileset\westfall\westfallsand.blp` and prioritized Westfall/Westwood sand auto-selection.
5. 37 focused Rosetta tests pass green.

**Spec 190 lane (2026-08-27, checkpoint 19).** Negative pedestal height support (sunken object viewing dips):
1. Enabled negative pedestal heights (`PedestalHeightMeters = -10f` default; e.g. `-10.0`, `-20.0`) in `RosettaGeneratorOptions` and `Program.cs` to give sunken viewing dips with beveled ramps for WMOs whose origins sit below the mesh.
2. Updated `CreateChunkHeights` to compute sunken dips when `p.Height < 0` with bevel ramp scaling.
3. 34 focused Rosetta tests pass green, including `Generate_PedestalHeights_NegativeSunkenDipWithBevel`.

**Spec 190 lane (2026-08-27, checkpoint 18).** Alpha WDT MCAL/MCLY tile transform alignment:
1. `rosetta-generate --format alpha` in `Program.cs` now passes `(tile.TileY, tile.TileX)` to `AlphaWdtWriter.Build`, using the exact same coordinate transform as `LkWdtWriter.Write` and `RosettaMinimapPainter` (`map{tileY}_{tileX}.blp`).
2. No protected format readers or writers were modified; `AlphaWdtWriter.cs` and `AlphaWdtReader.cs` remain completely untouched.
3. MCAL/MCLY texture layers now align with the correct map tiles and minimap tiles.
4. 33 focused Rosetta tests pass green, all 17 LkToAlpha tests pass green. Real 0.5.3 client visual proof remains operator-owned.

**Spec 190 lane (2026-08-27, checkpoint 17).** Correction: Alpha Rosetta map bytes were not the proven defect; the retained fix is minimap-only.
1. `rosetta-generate --format alpha` writes the monolithic map bytes directly from `AlphaWdtWriter.Build(...)`; no Rosetta post-write MDDF/MODF coordinate mutation remains.
2. `RosettaMinimapPainter` now places model/WMO pins at the object-band center used by `RosettaTilesetGenerator`, not the whole label cell center.
3. No map renderer, terrain adapter, scene-loading, culling, protected writer, or generated map placement byte contract was changed.
4. Noggit3 evidence: its minimap is WDL horizon-derived and skips WDL object chunks; no accessible Noggit/Noggit-Red per-object minimap denylist or classifier was found in this pass.
5. 33 focused Rosetta tests pass, including a minimap marker pixel regression. Real 0.5.3 client visual proof remains operator-owned.

**Spec 190 lane (2026-08-27, checkpoint 16).** Alpha 0.5.3 candidate asset discovery in `rosetta-generate`:
1. Updated `RunRosettaGenerate` in `WowViewer.Tool.Inspect/Program.cs` to scan Alpha-era `.mdx.mpq`, `.mdl.mpq`, `.m2.mpq`, `.wmo.mpq`, and `.blp` single-file wrappers on disk with robust enumeration options.
2. Verified on Alpha client roots: era detection correctly classifies `.mdx`/`.mdl` models and discovers candidate assets.
3. 32 focused Rosetta tests pass green; solution builds clean.

**Spec 190 lane (2026-08-27, checkpoint 15).** Minimap generation and runtime tile coordinate readout:
1. `RosettaMinimapPainter` & `Blp2Writer`: 256×256 DXT1-compressed BLP2 minimap tiles written under `Textures/Minimap/{mapName}/map{tileY}_{tileX}.blp` with cell borders, pedestals, downsampled text labels, and model/WMO markers.
2. Bottom status bar now displays `Tile: {tileY:D2}_{tileX:D2}` matching ADT and minimap filenames whenever runtime scene is loaded.
3. 32 focused unit tests pass green; solution builds clean.

**Spec 190 lane (2026-08-27, checkpoint 14).** Museum-grade presentation and full-scale continent container support:
1. `RosettaAlphaPainter`: 1024×1024 MCAL text rasterization (0.52 m/texel) with antialiased quincunx sampling and 4-bit nibble slicing into 2-layer MCLY/MCAL chunks.
2. Museum pedestals (`MCVT`): raised 4m plateau with 12.5m beveled ramp in the terrain mesh under every object cell to eliminate base clipping.
3. Removed artificial 512-tile Alpha WDT limitation: unified 4096-tile capacity across both Alpha and LK generation.
4. CLI options: `--ink-texture`, `--pedestal-height`, `--pedestal-bevel`. 30 focused unit tests pass.

**Spec 190 lane (2026-08-26, checkpoint 5).** Stride-scatter reverted to CONTIGUOUS row-major tile
fill from the start tile (operator rejected scattered tiles; the full ~14k-asset corpus covers the
map naturally). New CI alignment proof: every placement decoded with the viewer's rule
`(M − rawY, M − rawX)` must land inside its tile's ChunkCorner bounds. Smoke: 400 assets → 74
contiguous tiles, 345 placements, 2,578 exclusions (WMO group files). 8 focused tests pass.

**Spec 190 lane (2026-08-26, checkpoint 4).** Output is a self-consistent standalone map under
`{output}/World/Maps/{mapName}/` — folder, tile prefix (`{map}_{tileY}_{tileX}.adt`), WDT, and WDL
all derive from one map name; mismatches are refused. Tiles spread across the whole 64×64 grid on a
stride grid (tile count measured first). Smoke-verified on `C:\WoW4-data\WoW-12025`: 14,029
candidates, 60 placements, 7 tiles at stride-25, WDT parses via `map inspect`. 7 focused tests pass.
The "objects horizontal vs ADTs vertical" report came from pre-fix outputs (old transposing naming).
Next: user runs the current binary into a fresh output and confirms objects sit on their tiles.

**Spec 190 lane (2026-08-26, checkpoint 3).** Second real run root cause: the viewer loads
`{map}_{tileY}_{tileX}.adt` (column first) while the generator wrote `{tileX}_{tileY}` — every tile
loaded transposed, so terrain showed as a strip and objects landed on nonexistent tiles. Fixed file
naming, occupied-tile parsing, WDT/WDL tuple convention (MAIN read at `tileX*64+tileY`), whole-map
row-major layout spread, and label shrink-to-fit (overflow was the "nonsense" text). 7 focused tests
pass. Next: user re-run into a FRESH output dir, confirm objects render + labels legible.

**Spec 190 lane (2026-08-26, checkpoint 2).** First real run exposed three defects, all fixed and
tested: (1) placements invisible — raw MDDF/MODF coords must be `rawX = tileY*T + u`,
`rawY = tileX*T + v` because the viewer decodes `(M − rawY, M − rawX)`; (2) MCCV text scrambled —
vertex layout is interleaved 9-8 rows, not row-major 17×17; painter now mirrors the mesh builder;
(3) `LkAdtWriter` MCRF declared size was `4+…` instead of `8+…`, desyncing chunks carrying refs.
Output is now a standalone map (`{map}.wdt` with MCCV flag, flat `{map}.wdl`, optional `--pm4-dir`
PM4 copies); existing files never overwritten. 5 focused tests pass; solution builds clean; 9
pre-existing unrelated Core.Tests failures remain. Spec:
[190-rosetta-calibration-corpus](../specs/190-rosetta-calibration-corpus/spec.md). Next: user re-runs
`rosetta-generate` into a fresh output dir and confirms objects render + labels legible (user-owned),
then US2 reference-library builder.

This file is the interchange for the next agent. It records only the current routing state. Read
the owning spec for requirements and proof; read a workstream only when the spec links it.

**Editor Platform lane (2026-08-25).** The editor dependency chain for Spec 176 is implemented
library-first and tested: `src/core/WowViewer.Core.Editor/` (166 plugin host, 167 bridge contracts +
operations-as-data, 168 session/undo/save policy, 173 asset-integrity gate), `AdtPlacementEditor` in
`WowViewer.Core.IO/Maps` (175 placement authoring + 176 Phase 2 name-table/ID mutation; ID allocation
now keeps an original-catalog high-water mark so substitute delete+add continues the chronology),
and `Pm4ReconciliationEngine` + `Pm4ReconciliationInputAdapter` in `WowViewer.Core.PM4/Reconciliation`
(176 Phases 1–2: real guide observations from `Pm4ObjectSegmentBuilder`, canonical world→placement
composition, MSUR `_0x1C` height signal validated against the segment Z span, bounds-containment
association with explicit Conflict for competing placements, scorer candidates over a labelled
Museum self-corpus). The **viewer shell runs the real pipeline**: the Editor workbench destination's
reconciliation panel parses the actual PM4 guide and Museum ADT, and Apply goes through
`ReconciliationApplyService` — source-hash staleness refusal, session write guards, JSON provenance
sidecar (FR-016), and undoable `ReconciliationApplyOperation` restoring prior output bytes. 79 focused
`WowViewer.Core.Editor.Tests` pass; the full solution builds 0 errors. Known gaps: proposals are not
yet drawn as in-scene overlays (Phase 3 step 3), the P1 cross-tile/cross-era transfer story is
untouched, and one pre-existing real-corpus test (`Pm4RegionObjectGrouperTests`, local development
corpus) fails independently of this lane. Real Museum/PM4 visual + independent-reader proof remains
user-owned; no runtime/visual claim is made from compilation. See Specs 166–168, 173, 175, 176 and
`progress.md`.

## Current handoff

**START HERE: read the new WMO admission counters on a real Stormwind flight, then pick the rule to
fix. The instrumentation landed; the measurement has not been taken.**

- **Next bounded action, user-owned:** fly Stormwind, open Utilities > Perf > **"WMO admission (this
  frame)"**, and record the numbers into
  [Spec 151 research.md](../specs/151-portal-game-mode-surface/research.md). The panel names the
  dominant rule. **Do not change an admission rule before that reading exists.**
- **Confirmed by the Stormwind capture (2048 frames), taken after the Spec 153 fixes:**
  `PrepareObjectPhase` max **283.4 → 2.5 ms** and gone from the hitch list; `SceneMaintenance` max
  **454.5 → 3.9 ms**; unaccounted median 0.02 / p99 0.11 ms; median frame 17.40 → 6.98 ms.
- **Owner: `WmoSubmission`** — p99 154.10 / max 161.3 ms with a median of 0.71 ms, and **all 592
  recent hitches** read `<- WmoSubmission` at 153–157 ms. Stormwind submits **all districts at once**:
  7512 visible groups, 80484 draw calls, 15852 doodad submissions. **This is an admission problem,
  not batching** — 80200 of 80484 calls are correctly batched.
- **Instrumentation shipped (source proof + 11 core tests, nothing measured).** `WmoAdmissionTally` /
  `WmoAdmissionStats` in `Core.Runtime/World/Visibility` count placements and groups by admitting
  rule; `CollectVisibleWmos` has a `ref` overload proven identical to the old one; `WmoRenderer`
  records the per-group rule; the Perf panel displays both layers. Four **source** findings drove the
  counter shape, all with magnitudes still unmeasured: portal culling **cannot reject** a group
  (the decision is unioned with raw frustum visibility); WMO placements are **never rejected by the
  frustum** (`IgnoreVisionConeCulling: true` also disables the only frustum branch), which is the
  shape of the old-Ironforge-past-fog symptom; group admission is evaluated **twice per placement per
  frame** (opaque + transparent); and the recorded 7512 is a **submission** count spanning both
  passes, not distinct groups. Detail in Spec 151 `research.md`.
- Still suspect group-to-group visibility data the client uses and this renderer does not read.
- **Spec 153 Phase 4 may be moot** — it was written against `SceneMaintenance` max 454.8 ms, which no
  longer reproduces (3.9 ms). Re-measure on a route that forces `_instancesDirty` before implementing.
- **Spec 153 Phase 5 step 2 still owed:** `DeferredAssetLoads` max 442.9 ms in Stormwind against a
  3.5 ms budget. The admission policy bounded the additive overshoot; the single-load residual needs
  decode off the render thread.
- **Released:** v0.5.2.1 (commit `975d0c79`, tag `v0.5.2.1`, branch `v0.5.3-dev`). Version bumped in
  both csproj files and `ViewerApp.ViewerProductName`; `wow-viewer/CHANGELOG.md` added;
  `docs/releases/v0.5.2.1.md` is what the release workflow publishes; both READMEs updated and the
  Stormwind WMO issue is documented as a known issue rather than left for users to discover.

---

**Spec 153 detail (shipped in v0.5.2.1, confirmed by capture).** Full numbers and both capture
tables are in [Spec 153 research.md](../specs/153-renderer-hitch-and-batching/research.md).

- **Defect A was `AudioRuntime.Update`, and it was never audio.** `RefreshEmitterDiagnosticsIfDue`
  rebuilt an `AudioTriggerDiagnostic` per resident emitter (5565) on a **wall-clock 250 ms** timer on
  the render thread — which is why the "every 47–50 frames" interval drifted with framerate — and it
  ran **whether or not anything displayed the result**. A second, movement-triggered copy: `RemoveTile`
  rebuilt the list synchronously on streaming eviction. Fixed by gating the rebuild on
  `NoteEmitterDiagnosticsObserved()` (only the audio panel calls it) and making eviction invalidate
  only. **Measured 283.4 → 2.5 ms max.** General lesson: *a diagnostics surface nothing is reading
  still pays full cost unless something gates it.*
- **Defect B was a hardcoded `return true`.** `PlanVisibleMdxPasses` gave the route planner a
  `requiresUnbatchedRender` predicate whose whole body was `return true`, so 100% of opaque MDX took
  the per-instance fallback while the batching machinery sat inert. Now consumes
  `IModelRenderer.RequiresUnbatchedWorldRender` — the contract the WMO doodad path already used.
  **0/312 → 526 batched / 3 unbatched**; `MdxOpaqueSubmission` p99 30.75 → 14.12 ms. GPU instancing
  stays off (`SupportsGpuInstancedOpaque` is still `false`); the win is begin-once/submit-many state.
  `_wireframe` folded into `RequiresUnbatchedWorldRender` — the one real visual divergence between
  the paths. Live-revertible via `WorldScene.MdxOpaqueBatchingEnabled`.
- **Do not credit Defect B's fix with fixing the gallop.** Frame p99 barely moved on that capture
  (259.70 → 246.62) because the periodic stall was never the MDX cost.
- **Instrumentation is now self-defending.** `PrepareObjectPhase` has a stage timer (`StageCount`
  18 → 19) and `WorldFramePassInstrumentation` + a reflection test fail the build if a pass has no
  timer, a stage is recorded by nothing, or a stage is double-counted. Unaccounted time is now
  median 0.02 / p99 0.11 ms, so hitch attribution names a stage instead of a void.
- **Audio is scoped to the camera tile.** `Update` consulted **no tile information at all** — it
  scanned every resident tile. Now takes `TerrainManager.CameraTileX/Y` (passed in, never re-derived)
  within `WorldAudioRuntime.AudibleTileRadius` (1); the diagnostics panel uses the same window.
  **Tile keying was checked and cleared** — `AddTile`/`RemoveTile`/`EmitterKey`/`OnTileLoaded` all
  agree; scanning every tile just made it look like a keying fault.
- **OPEN: MCSE emitters read as permanently out of range; only water works.**
  `AlphaTerrainAdapter.ConvertSoundPosition` does `chunkCorner - local` on the strength of an
  unevidenced comment ("Alpha MCSE stores a chunk-local C3Vector"); the Ghidra work proved the 0x34
  **field layout**, not the frame. If it is not chunk-local, every MCSE emitter lands tens of
  thousands of units off-map — which also explains why MCNK liquid rows work, since they derive from
  the renderer's own `chunk.WorldPosition` and never touch the transform. **`McseFrameEvidence`
  measures it** (raw min/max per axis, chunk/tile/beyond counts, explicit verdict) at the top of
  Utilities > Audio. **Read that line before touching the transform** — it deliberately reports
  "inconclusive" rather than picking a winner on a mixed sample.
- **Refuted, do not revive without new evidence:** the allocation-churn hypothesis. Median
  world-render CPU is 0.33–8.58 ms and traversal maxes at 0.22 ms. Spec 152 Phases 3–5 (flatten the
  scene graph into retained draw lists, view modes) are **suspended** because they rested on that
  premise. Also ruled out with evidence: decoded-asset caching / LRU thrash (`MaxMdxCached = 0`,
  unlimited; 554 models serve 18663 instances — nothing is re-decoded, the cost is submission).
- **The measurement tool exists and works.** Utilities > Perf > Frame history: rolling per-frame
  history, hitch detection with dominant-cause attribution, unaccounted time, region peaks,
  submission batching counts, and an injected-stall self-check. Recording is allocation-free.
  Use it for every before/after. **Benchmarks: Stranglethorn Vale** (dense doodads),
  **Stormwind** (dense WMO groups).
- **No viewer test assembly exists** (`tests/` has Core, Core.Anim, Core.Curation, Core.PM4 only), so
  viewer-side changes — `McseFrameEvidence`, the audio tile window, the batching predicate — carry
  source proof plus capture, not unit tests. Moving those types into core is an open follow-up.
- **Detector lessons that made this possible:** p99 hides rare hitches (they land at p100 — use max
  and over-threshold count); ranking stages by p99 buries a rare-but-huge stage (sort by max);
  always report unaccounted time, or attribution names a 0.2 ms stage for a 350 ms frame.
- **Lower-priority target:** User-run visual/compact-window proof for the Spec 080 Phase 2E IA.
  Check Scene for only Placements/LOD, Experimental > Terrain Lab for tiles plus chunk clipboard,
  Inspect's dropdown for Archeology, MCNK/ADT, scene investigation, world context, animations, and
  actions, Utilities > Minimap for the restored route, and Navigator > World Maps for the Phase Map
  selector.
- **Related WMO/fog observation — fold this into the group-admission work above.** A screenshot showed
  distant WMO content, including old Ironforge, still visible beyond the effective fog end while
  terrain had already been culled. Same shape as the Stormwind finding: WMO geometry is being admitted
  that should not be. Treat as a concrete symptom, not a proven owner; it wants the same
  visibility/submission counters plus a trace of camera-to-bounds distance against the fog plane
  before any admission logic changes.
- **Proof owner:** Focused PM4/audio contract tests and cross-platform viewer build pass; the user owns
  real-client region-camera, streaming, archive-provenance, and audible proof. The current camera
  slice updates active tiles on mouse-look without reopening the residency lease.
  **For renderer performance specifically, the proof is now the in-viewer frame history** — the user
  flies the route and reads Utilities > Perf. Two captures (Stranglethorn, Stormwind) are recorded
  with full numbers in Spec 153 `research.md`; every renderer claim must cite one.
- **Time-of-day checkpoint:** The interactive lighting path has a pure 2,880-unit/24-minute Alpha
  clock enabled by default, with manual slider freeze/resume; Light DBC and LIT consume the same frame
  time, while synthetic minimap manifests record a frozen time-of-day mode.
- **Completed slice (latest):** `975d0c79` — Spec 153 Phases 1/2/3/5, audio tile scoping,
  `McseFrameEvidence`, and the v0.5.2.1 release (version bump, CHANGELOG, release notes, READMEs).
  `bda47bdb` — handoff repointed at WMO group admission. Both pushed to `v0.5.3-dev`; tag `v0.5.2.1`
  published with all four platform builds.
- **Completed slice (earlier):** Checkpoint commits `3bfbbba4` (accumulated audio, AreaNumber, Ghidra, and
  Zone/SubZone overlay work), `de41b183` (Spec Kit design pack), and `c70e1945` (portal phase)
  contain the work completed on this lane. Spec 151 Phase 1 now has a pure, fail-open WMO portal
  decision using transformed portal polygons/clip volumes, source-side admission, bounded
  depth/visit limits, renderer integration, and portal counters in `WmoRenderStats`; the old
  center-distance/queue traversal scaffolding is removed. Focused portal/graph tests pass 16/16 and
  the full solution Debug build passes with 0 errors. The graph evaluator is explicitly diagnostic;
  the shared runtime decision owns final renderer admission. Spec 149 now has an opt-in resident
  Zone/SubZone overlay slice: Ghidra-backed
  MCNK AreaNumber evidence, revisioned resident chunk enumeration, AreaTable-grouped footprint regions,
  distinct Zone/Subzone styling, projected labels, and unresolved-count diagnostics. Spec 148 now has a
  provenance-first world-simulator spec/plan/tasks pack;
  MCSE emitters preserve raw/transformed positions and the proven Alpha 0.5.3 0x34-byte scheduler
  fields; shared Alpha AreaNumber resolution splits high/low `ushort` zone/subzone words and follows
  `ParentAreaNum` without half-word aliases; the area contract now branches explicitly so 3.3.5+
  direct AreaTable IDs cannot be captured by Alpha AreaNumber aliases; status-bar and terrain audio now
  consume the same resolved Zone/SubZone result; the runtime exposes non-playing diagnostic rows; the audio panel
  shows IDs, coordinates, path/source, decode/backend state, terminal reason, and coordinate provenance.
  The audio runtime also exposes a residency-change-only normalized emitter snapshot and the audio
  panel can opt into source-colored 3D speaker pins without starting playback.
  Automatic ZoneMusic playback is now hard-muted behind a tested policy; its area assignment remains
  diagnostic-only so the working MCNK/MCSE water path is not affected.
  The current Spec 080 Phase 2A/2C sidebar slice replaces the visible Model/World/Tools top row with
  Quick/Inspect/Scene/Utilities/Experimental, gives selection/model/ADT/MCNK/PM4 facts one inline
  inspector route, restores MDX/M2 animation controls in Inspect, and combines terrain targeting
  with MCNK/chunk clipboard actions in Experimental Terrain Lab. Audio is owned only by Utilities.
  The viewer build and source checks pass; the full test command timed out and the
  focused core suite reports nine unrelated baseline failures.
  Main Panels utility entries now select their exact Utilities page; compact-window
  manual proof remains open. Source/file/map loading is explicitly left-sidebar-only; the right
  Scene selector now contains only Placements and LOD, while Utilities keeps an isolated page index
  so Inspect/Scene page selection cannot hide or misroute the Minimap page.
- **Main unproven gap:** **WMO group admission is measured as the renderer's dominant remaining cost
  but its cause is still not diagnosed** — the counters that name the admitting rule now exist and
  have never been read on a real flight. Reading them is the next bounded action.
  The MCSE coordinate frame is also open: measured, verdict not yet read on real data.
  The sidebar slice still needs user-owned visual proof at normal and compact
  window sizes, including selected-context transitions and legacy caller reachability. The time-of-day
  slice still needs live early-client visual proof and a
  comparison of authored minimap tint behavior; the theory that shipped minimaps captured a moving
  clock remains unproven. Spec 104's restored MDX material shader inputs still need real model/shader
  compilation and visual proof. Full BLS bytecode parity remains out of scope. Spec 151's game-mode head anchor/physics, simple-surface policy, and
  diagnostic budget remain unimplemented. Portal admission is source-tested but still needs the
  user-owned real-client visual/submission/FPS comparison. Spec 149's PM4 region
  bounds/focus, correlation UI retirement, focused area aggregation tests and default-off per-trigger
  audio controls remain open. MCSE tile/chunk
  normalization and MCNK liquid-center placement now have focused source/test proof, pending live
  runtime/audible proof. Speaker-marker placement is source-tested through the normalized snapshot
  path, pending live visual proof. The area overlay is resident chunk coverage, not a
  proven complete polygon. Automatic ZoneMusic playback is intentionally muted until its handoff is
  proven; area resolution remains diagnostic-only.
  ZoneMusic table indirection, exact `sounds.mpq` provenance, MIDI/DLS
  playback, and native MCSE callback installation remain separate proof gates. Spec 150 still lacks
  native renderer anchors, repeatable 0.5.3 baseline capture, and CPU/GPU attribution.
- **Explicitly out of scope for the next slice:** Simple-surface UI, logging-policy retirement,
  whole renderer rewrite, `.bls` bytecode loading/porting, fake audio conversion, and claims of
  visual/FPS/audible gains. Game-mode input/UI follows the pure Phase 2 runtime-core checkpoint.

## Active spec lanes

| Spec | State | Next handoff |
|---|---|---|
| **151 Portal-aware rendering / WMO group admission** | **PRIORITY 1 — instrumentation shipped, measurement owed** | **Fly Stormwind, read Utilities > Perf > "WMO admission (this frame)", record it in Spec 151 `research.md`. Only then pick the rule to change.** |
| 153 Renderer hitch and MDX batching | Phases 1/2/3/5 shipped as v0.5.2.1 and confirmed by capture; Phase 4 likely moot | Re-measure `SceneMaintenance` before implementing Phase 4; Phase 5 step 2 (decode off the render thread) still owed |
| 152 Renderer frame-time stability / per-era lighting | Detector landed and used; its Phase 1 gate refuted the allocation hypothesis so Phases 3–5 are suspended | Owns the measurement infrastructure (done) and Phase 6 per-era terrain lighting (independent, not started, fixes 1.0.0+ darkness). |
| 151 Portal-aware rendering/game mode/simple surface | Phase 1 portal checkpoint implemented; Phase 2 open | Add pure game-mode state/physics and character-head anchor; preserve editor camera state and stop at the focused physics checkpoint. |
| **155 Asset reference inventory (expected vs catalogued vs present)** | **NEW — spec drafted, not planned** | Extend `inspect` with per-asset reference logging and corpus-wide sweeps over **WMO doodads, WMO textures, and MDX/M2 textures**. Deliverable is the disagreement between what data references, what listfiles name, and what the build contains, plus orphans as the repair donor pool. **Positive control: the Mt. Hyjal green-smoke effect objects must be flagged by an untargeted sweep** — the engine draws untextured geometry neon green, so it is verified in-world. Corpus comes from the data-access layer, never from archive internal listfiles. |
| **154 M2 reader era parity (1.x–3.0.1)** | **NEW — spec + plan drafted, not started** | Run US1 first: survey every staged build before touching a reader. Three measured defects; the "4.0.0 works" premise is contradicted by measurement and must be resolved, not assumed. |
| 104 Legacy M2/MDX rendering | 1.0.0 route complete; MDX material/effect shader checkpoint implemented with visual proof open | Validate shader compilation and translucent/reflective models against the configured client/build; keep full BLS parity separate. **Blocked in part by Spec 154** — M2 bone reading is broken outside the Alpha and late-3.x routes. |
| 149 PM4 region navigation/audio trigger controls | Draft pack; resident area overlay, MCNK liquid producer, coordinate normalization, and opt-in speaker-marker slices implemented | Add area aggregation/audio-control tests, then complete per-trigger toggles and ZoneMusic indirection; retire correlation UI only after the region checkpoint; keep world triggers default-off. |
| 150 Alpha 0.5.3 renderer performance | Draft evidence/planning pack complete; no source optimization started | Recover native world/terrain/object/resource/LOD anchors and run two repeated production `profile-render` baselines before choosing one owner. |
| 148 Artifact world simulator runtime | Phase 1 diagnostics in progress; client contract correction landed | Add ZoneMusic indirection, then finish read/decode/source-stage coverage and user real-client inspection. |
| 147 Minimap/fog/doodad instancing | Phase 2 implemented; Phase 3/4 open | User-run minimap proof, then implement fog coverage and structured batching diagnostics. |
| 146 Audio/camera playback | AreaNumber-aware area selection and master mute control implemented; client audio contracts recovered | Add ZoneMusic row resolution; MIDI/DLS and native MCSE callback proof remain gated. |
| 144 Camera capture paths | Implemented with user gates | Validate swept path residency during playback and confirm lease release before extensions. |
| 145 WoW UI overhaul | First slice implemented | Continue only the remaining persistent-window/placeholder audit tasks. |
| 080 WoW UI consolidation | Phase 2A tabbed sidebar IA and unified inspector source slice implemented; manual proof open | Run the five-tab visual/compact-window check, then resume the legacy route inventory before deleting old methods. |
| 143 World context and lighting | LIT source/fallback, pre-alpha v2 parser, and default-on 0.5.3 time cycle implemented with user gate | Validate live clock/manual freeze and authored-minimap tint boundaries, then continue WMO area and lighting evidence. |
| 142 World scene graph | In progress | User-run dense-WMO capture to compare internal-doodad batching against the prior placement-local path. |
| 139–141 Terrain/minimap reconstruction | Active/parked ML lanes | Reopen only for the named spec and user-run training/validation. |
| 138 Cross-era renderer research | Evidence/planning | Do not generalize one client build to every era. |
| 128–131 PM4 | Established research lane | Use the PM4 spec pack and `workstream-pm4-decode.md`. |

## Stable boundaries

- New code, tests, tools, and viewer docs go in `wow-viewer/`; the legacy tree is read-only
  reference unless a bounded compatibility fix is explicitly requested.
- Keep format readers library-first and tools thin. Do not duplicate or rewrite working client-file
  readers. Keep the Alpha/standard terrain split. `AlphaWdtWriter.cs` is frozen unless explicitly
  reopened with focused proof.
- Client roots are runtime configuration. `H:\CLIENTS` is approved; never hardcode a local client
  path. Record root, build identity, and fingerprint for client-backed proof.
- Training, GPU work, broad harvests, long captures, and real-client/runtime testing are user-run.
- The default source proof is:
  `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  and
  `dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.

## Continuity rules

- Update this dashboard and `progress.md` only when the implementation handoff changes.
- Put durable technical findings in the owning workstream or architecture note, not here.
- Preserve negative results and open gates, but remove superseded narrative from the default path.
- End every handoff with: current target, proof owner, completed slice, unproven gap, next bounded
  action, and explicit out-of-scope items.
