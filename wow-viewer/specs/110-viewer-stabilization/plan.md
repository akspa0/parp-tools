# Implementation Plan: Viewer Stabilization

**Branch**: `110-viewer-stabilization` | **Date**: 2026-07-16 | **Spec**: [spec.md](spec.md)

**Input**: Viewer-stabilization specification and the existing Specs 032, 080, 073b, and 104.

## Summary

Stabilize the world viewer before expanding renderer scope. First give fog a single validated range,
a user-owned override, a no-LIT fallback, a spatial LIT-inspection surface, and a direct
terrain-to-minimap export path so time-of-day evaluation cannot hide terrain and missing authored
minimap assets do not block inspection.
Then make every M2 route either native or explicitly diagnostic, remove MDX conversion from runtime
fallbacks, clean the Tool menu, and expose only conversion capabilities that have evidence separate
from renderer proof.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Silk.NET OpenGL, ImGui.NET, existing `WowViewer.Core*` libraries

**Storage**: `viewer_settings.json` for persistent viewer preferences; no new persistent asset store

**Testing**: xUnit focused Core tests; targeted Debug build; user-run real-client visual proof

**Target Platform**: Windows desktop viewer

**Project Type**: Desktop application plus Core/IO/runtime libraries and CLI tools

**Performance Goals**: Fog-range resolution has no per-frame allocation and does not add a draw pass;
the active range is resolved once before scene submission. The opt-in minimap overlay performs a
bounded scan of loaded LIT entries only while a minimap is visible and adds no lighting evaluation.
Minimap export runs as a user-started background process, emits tiles incrementally, and keeps the
combined PNG within the documented output dimension guard.

**Constraints**: No writes in `gillijimproject_refactor`; no parser rewrite; client roots are runtime
configuration; conversion is an explicit export only and never a renderer fallback.

**Scale/Scope**: Four independently validatable stabilization stories. Phase 1 is only fog/terrain
visibility; no later phase starts until it is build- and test-validated.

## Constitution Check

| Gate | Status | Evidence / handling |
|---|---|---|
| Repo independence | Pass | All proposed code and tools stay inside `wow-viewer/`. |
| Library-first | Pass | Fog-range normalization belongs in `WowViewer.Core`; viewer UI only selects a source. |
| Real-data validation | Pending user proof | Unit/build proof is local; visual M2 and map proof records configured root, build, and fingerprint. |
| Read-only reference repo | Pass | Reference code is read-only; no legacy edits are planned. |
| One phase at a time | Pass | Complete Phase 1 before starting native M2, tool, or conversion work. |
| M2 native ownership | Pass | The plan removes M2-to-MDX runtime fallback instead of extending it. |
| Derived-artifact provenance | Pass | The export writes the selected time, LIT/fallback evidence, and client build into a manifest; it is never called ground truth. |

## Research Decisions

See [research.md](research.md). The key decisions are:

1. Normalize every active fog range before shader, terrain-culling, or object-visibility submission;
   invalid LIT/DBC values fall back to a visible range.
2. Preserve a user override separately from the lighting recommendation. Lighting supplies colors and
   recommended distances; it cannot overwrite an active user override every frame.
3. Route 1.0.0 M2 through `BuildEra100StaticRenderModel` and native `M2Renderer`; delete all MDX
   conversion and adapter-backed renderer fallback paths from M2 loading.
4. WMO v14↔v17 already has library converters and fixture tests in both directions. M2→MDX has
   synthetic conversion coverage only, so it remains export-only and not yet real-client reliable.
5. Terrain minimap synthesis is a Core.IO compositor over existing MCLY/MCAL/BLP decode contracts.
   It preserves terrain layer ordering, but intentionally does not sample the renderer's repeated
   screen diffuse UVs: it uses phase-independent BLP material averages at minimap scale so the
   viewer dialog invokes a stable export rather than moire-prone rendering behavior.

## Project Structure

### Documentation

```text
specs/110-viewer-stabilization/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── viewer-stabilization-contract.md
└── tasks.md
```

### Source Code

```text
src/core/WowViewer.Core/
└── Terrain/TerrainLightingMath.cs

src/core/WowViewer.Core.IO/
└── Maps/
    ├── TerrainMinimapCompositor.cs
    └── TerrainMinimapStitcher.cs

src/viewer/WoWViewer/
├── ViewerApp_Lighting.cs
├── ViewerApp_MinimapAndStatus.cs
├── MinimapHelpers.cs
├── ViewerApp_Sidebars.cs
├── Terrain/WorldScene.cs
├── Terrain/WorldAssetManager.cs
├── Rendering/WmoRenderer.cs
└── Rendering/WowViewerM2RuntimeBridge.cs

src/tools/
├── harvest/WowViewer.Tool.Harvest/
├── WowViewer.Tool.Inspect/
└── WowViewer.Tool.Converter/

tests/WowViewer.Core.Tests/
├── TerrainLightingMathTests.cs
└── TerrainMinimapCompositorTests.cs
```

**Structure Decision**: Extend the existing Core terrain-math contract and the viewer's existing
scene/UI ownership. Existing Core.IO converters remain the only conversion implementations; viewer
work must invoke or describe them, not duplicate format logic.

## Phase 0 - Evidence and contract recovery

1. Record the current fog writer order and prove why user-edited values are overwritten.
2. Record every M2 route that builds or renders MDX, including world objects and WMO doodads.
3. Inventory the Tools menu and its executable-resolution seam.
4. Record WMO v14↔v17 and M2→MDX test coverage separately from runtime rendering.

**Exit criteria**: `research.md`, data model, and user-facing contract name exact owners and no
unresolved technical ambiguity remains.

## Phase 1 - Fog, terrain visibility, and terrain-derived minimaps (current implementation phase)

1. Add a Core-tested fog-range normalizer with a visible fallback and minimum non-zero span.
2. Add a `WorldScene` user fog-range override with source/status reporting and a reset to the
   current lighting recommendation.
3. Resolve the active range after LIT/DBC/no-source lighting selection and before all terrain,
   object, WDL, and shader consumers receive it.
4. Put active-range controls in the Lighting surface; retain Settings as defaults only.
5. Add a shared opt-in LIT marker overlay to both minimap surfaces and selection handling that does
   not alter lighting state.
6. Add a virtualized LIT list in the Lighting surface; double-click focuses a safe camera point.
7. Add a Core.IO compositor and stitcher that reuse decoded MTEX/MCLY/MCAL, MCNR, and MCSH data;
   verify their weighted blend, lighting, and transparent-hole behavior with focused tests.
8. Replace the `synthetic-minimap` stub with a map-aware Harvest command that emits tile PNGs,
   a stitched PNG, and a provenance manifest at an exact selected clock minute.
9. Add a Tools > Export dialog that resolves the in-repository Harvest command (built output or
   source-project fallback) instead of assuming an external binary exists.
10. Run focused test/build proof, then hand the user one LIT and one no-LIT real-client visual/export
    proof command.
11. Keep Alpha MAIN coordinate enumeration aligned with its reader contract. Synthesized minimaps
    use north/top-edge white terrain light independent of LIT/native-world-light data; record that
    exclusion and pipeline failure stages for residual real-data tiles.

**Exit criteria**: zero-width/invalid ranges cannot reach a shader; manual UI controls cannot be
overwritten by a lighting update; no-LIT fallback remains visible; LIT markers and selection agree
between both minimap views and the list without altering the active lighting contract; the current
Tools menu can launch a terrain-only per-tile and combined-map export with explicit lighting
provenance.

## Phase 1c - Control reachability repair (current implementation phase)

1. Replace drag-only fog range fields with true visible slider controls in the active and default
   fog surfaces while retaining the normalized override contract.
2. Give Tools > Archeology its own nested-tab state, remove duplicate World controls, and preserve
   a dedicated legacy Archaeology window route.
3. Make playback pause/stop reachable while active on every Archeology subtab; stop safely when
   its world or scoped range disappears. Build the viewer without terminating a running user app.

**Exit criteria**: every fog range has a visible grab; selecting Playback cannot redirect the
parent Tools tab; an active UniqueId range can always be paused or stopped from Archeology.

## Phase 1d - Terrain minimap fidelity correction (current implementation phase)

1. Keep the CPU minimap contract separate from the terrain renderer's repeated diffuse UVs: derive
   a cached phase-independent BLP material average for each texture, with no renderer-UV/mip
   sampling, so output does not retain sub-pixel repeat phase or interpolation artifacts.
2. Replace normalized MCAL weight summation with base-plus-ordered-overlay composition, matching
   the terrain fragment shader when alpha layers overlap.
3. Normalize Alpha MCLY's native column-major chunk layout at the tensor-pack boundary and enforce
   `MclyLayerMask` during composition so each MCAL alpha map is applied to the matching texture layer.
4. Add focused material-average, layer-order, phase-invariance, and Alpha coordinate/presence tests; build the Harvest command and have
   the user re-export one bounded real-client tile before whole-map proof.
5. Make the bounded export limit count emitted PNGs rather than skipped WDT candidates, preserving
   skipped/failed tile diagnostics in the manifest.
6. Treat Alpha MCNR as its native staggered vertex lattice: interpolate vertex Lambert values across
   terrain triangles and retain MCSH occupancy separately, rather than introducing a dense-raster
   checkerboard into the synthesized RGB output.
7. Keep MCSH out of default synthesized minimap RGB. Allow it only as an explicitly labeled
   exceptional-history preview, while retaining the decoded mask as an independent training signal.
8. For raw authored minimap ingestion, derive versioned lighting provenance from a neutral terrain
   baseline: global tint, MCSH darkening correlation, and a conservative LIT-chroma time bucket that
   never claims a capture time. Promote this analysis to V22/full streams and preserve MTEX texture
   payload identity by omitting incomplete sidecar tables rather than shifting their indices.
9. Treat missing MCAL on a readable MCLY tile as base-layer-only and a mismatched MCNR validity
   mask as incomplete evidence. Do not fabricate overlay alpha or abort whole-map export; bound-check
   the mask, retain normal/white-top-edge evaluation, and use a neutral normal outside valid mask bounds.
10. When a referenced terrain BLP cannot decode, try only successfully decoded deterministic RGB
    proxies: its same-stem `_s` companion, then at most sixteen ordinary color BLPs scanned from the
    loaded archive/listfile catalog. Rank exact/strong basename matches before directory-theme tokens
    so moved assets can recover; retain original MTEX identity and the resolution kind in metadata;
   apply the identical recovery order in the terrain viewer without claiming native material-semantics
   parity.
11. Emit a paired `_liquid` target for every terrain minimap tile and stitched map. Normalize Alpha
    MCLQ type coverage to its terrain-resolution surface before building unified liquid, then render
    decoded coverage with the current flat viewer liquid palette/opacity. Respect Alpha's 8×8 MCLQ
    cell flags and rasterize only complete covered source cells so liquid cannot bleed along dry
    terrain-cell boundaries. Decode raw MCLQ `0x04` as a river/water rather than slime. Record
    render profile and pixel count without claiming water-material or animation parity. Retain a
    one-tile CLI selection and first relevant source frame in failed
    diagnostics so residual client-data decode faults are reproducible rather than opaque. Make
    Alpha footprint painters use their destination-buffer dimensions so 256² roof-mask edges cannot
    abort terrain decode; after same-name/related BLP recovery, use a decoded folder/terrain-family
    catalog RGB proxy (then a verified previously decoded BLP) rather than skip a readable tile
    whose named stale MTEX linkage cannot resolve. A tile with no declared material name stays an
    unlit solid-white empty baseline. Record every such substitution.

**Exit criteria**: focused tests prove terrain-equivalent layer order, phase-independent material
averaging, MCNR triangle interpolation, unshadowed default RGB, trustworthy lighting sidecar
states, and aligned terrain/liquid outputs; a real-client tile export is visually free of the
reported static, projection, shadow, and interpolation artifacts.

## Phase 2 - Native M2 route recovery

1. Make native static renderer selection unconditional for M2 runtime models.
2. Route 1.0.0 embedded divisions to `BuildEra100StaticRenderModel` in world-object and WMO-doodad
   loaders before external-skin probing.
3. Remove M2→MDX conversion and adapter-backed `MdxRenderer` construction from all M2 runtime
   fallback paths.
4. Make unsupported native capability diagnostics explicit and attach format/version/route data.
5. Run focused reader/runtime tests and a Debug build; hand the user exact real-client visual tests.

**Exit criteria**: no M2 load path constructs converted MDX for rendering; all failures are
diagnostic rather than silent.

## Phase 3 - Tools menu and modern entry points

1. Audit every main-menu Tools item in both UI modes and classify it as current, replaceable, or
   dead.
2. Remove MK Dataset/VLM Dataset and unsupported legacy launchers.
3. Route Inspect/Converter actions to in-repository project outputs or show an actionable missing
   dependency state.
4. Verify the menu inventory with a focused source/UI test where practical.

**Exit criteria**: no obsolete dataset menu entry and no launcher that assumes an unreported binary.

## Phase 4 - Explicit conversion capability publication

1. Run fixture tests for WMO v14→v17 and v17→v14; publish preservation/failure limits.
2. Audit M2→MDX core conversion input profiles and add missing capability evidence.
3. Add explicit CLI export contracts; conversion result never enters M2 runtime loading.
4. Schedule user-run real-client export checks before marking any profile reliable.

**Exit criteria**: capability table labels each direction as fixture-proven, real-client-proven, or
unsupported; renderer proof is never substituted with conversion proof.

## Complexity Tracking

No constitution exception is required.
