# Phase 8J: Overlay Recovery — Fresh-Session Handoff

**Feature**: Spec 142 World Scene Graph and Spatial Partitioning
**Status**: Ready to implement, starting with 8J.1 only
**Evidence**: `output/diagnostics/azeroth-32-32-full-post-tile-cull.json`
**Proof owner**: Production `profile-render`, then focused unit/build proof

## Why this phase exists

The post-cull, full-Azeroth capture still has a P95 CPU frame of 44,178.7 ms. Its `overlay` stage
alone takes 39.5-44.0 seconds on alternating samples. The current `WorldScene.Render` groups
unrelated work under that one timer: object wireframe/reveal, selected and global bounds, PM4
geometry/bounds/markers, POI pins, taxi paths, area triggers, and other visual diagnostics.

The first implementation must not guess that PM4 is the culprit or “optimize overlays” globally.
It must make every owner visible in the report, then fix the proven owner without changing terrain,
WMO, M2, minimap, dual-map, or visual-fidelity behavior.

## Scope and non-goals

In scope:

- Owner-level overlay timings, counts, enabled state, cache/rebuild state, and deferred-work state.
- Separating CPU geometry preparation from GPU submission for PM4 and other overlays.
- Event-driven invalidation, viewport/tile culling, cache reuse, and bounded preparation after the
  owner is proven.

Out of scope for this phase:

- PM4 format decoding/research semantics or visual redesign.
- Terrain `SecondaryOverlayMap`, minimap texture overlay, liquid correctness, or WMO portal work.
- Whole-map tile residency (Phase 8K), shader/lighting fidelity, and modern GPU submission (8L/
  Spec 138).
- Any user-run full-map capture by the agent.

## Overlay owner taxonomy

Every timed overlay operation belongs to exactly one owner. Disabled owners report zero work rather
than disappearing into a generic stage.

| Owner ID | Existing WorldScene surface | Expected expensive work |
|---|---|---|
| `object_wireframe` | `RenderVisibleObjectWireframeOverlay` / reveal | per-visible M2/WMO wireframe generation/submission |
| `selection_bounds` | selected/global bounds | placement iteration and line-batch generation |
| `pm4_bounds` | PM4 object/CK24 bounds | PM4 object traversal and dynamic bounds aggregation |
| `pm4_geometry_prepare` | PM4 fill/line/centroid loop | transform, object visibility, triangle/line/pin batching |
| `pm4_geometry_submit` | PM4 batch flushes | GL buffer upload and draw submission |
| `pm4_nodes` | MSCN/MSPV markers | lazy data ensure and cube/pin batching |
| `poi_taxi` | POI and taxi block | pin/line/bounds batch creation |
| `area_triggers` | AreaTriggers block | procedural debug shape generation |
| `other_overlay` | any remaining visual aid | explicit temporary owner; must be split before promotion |

## 8J.1 — Attribution (first fresh-chat slice)

**Goal**: Replace the opaque `overlay` timer with owner-level evidence without changing rendering
output or work scheduling.

1. Add a library-owned serializable `WorldOverlayOwnerFrameStats`/aggregate contract beside
   `WorldRenderFrameStats` in `src/core/WowViewer.Core.Runtime/World/`. It records owner ID,
   duration, enabled, prepared/submitted primitive counts, cache status, and deferred count.
2. Extend the production diagnostic schema in `WorldRenderDiagnostics` and its focused tests so
   owner records are retained per frame and summarized in the final JSON. Keep the existing coarse
   `overlay` stage for compatibility, defined as the sum of owner durations.
3. Wrap the existing `WorldScene.Render` overlay blocks at their current boundaries using a small
   local timing helper. Do not move rendering code yet. Every taxonomy owner must emit exactly one
   record per frame, including disabled owners.
4. Add a focused viewer/runtime test or extraction-friendly helper test proving known records sum
   to the coarse overlay duration and that disabled records have zero counts/duration.
5. Build the validation-capture project and run focused diagnostic tests. Hand the user one
   3-warmup/10-frame full-map command; stop after recording the dominant owner.

**Done means**: a report identifies the owner, enabled state, primitive count, and cache/deferred
state responsible for each multi-second overlay frame. No optimization claim is allowed yet.

## 8J.2 — Owner isolation and invalidation (only after 8J.1)

**Goal**: Ensure an unchanged camera/settings/content frame does not rebuild static overlay CPU
geometry.

1. Extract the proven owner behind a narrow `IWorldOverlayOwner` seam in the viewer layer; it owns
   `IsEnabled`, `InvalidationKey`, `Prepare`, `Submit`, and diagnostic snapshot only.
2. Define the owner’s invalidation key from named inputs (map/source content revision, overlay
   settings, selection/highlight state, and camera/tile window only if that owner requires it).
3. Cache CPU-prepared primitives or GPU-ready batches keyed by that value. Never cache ambiguous
   transformed output across a changing input.
4. Add a no-change-frame test: same key causes zero prepare work and preserves submitted geometry;
   each named key change rebuilds once.
5. Preserve a disable switch and compare visual output/primitive counts against the pre-extraction
   path on a bounded scene.

**Done means**: unchanged frames reuse the proven owner’s prepared output and diagnostics call it
`cache_hit`, not `rebuild`.

## 8J.3 — Bounded admission and viewport culling (only after 8J.2)

**Goal**: A changed, large overlay never turns one frame into an unbounded CPU job.

1. Add an owner-specific preparation queue with a documented CPU budget and a visible deferred
   count; submission continues to use the last complete valid batch until replacement is ready.
2. Cull owner inputs by active tile/camera window before transforms or primitive generation where
   visual semantics permit it.
3. For PM4 geometry, cache per-tile prepared batches and invalidate only affected tiles, not the
   full map.
4. Add focused queue/cull tests for budget enforcement, cancellation on invalidation, and stale
   batch disposal.
5. Re-run the user-owned full-map capture and compare owner timings, deferred state, visible counts,
   and output parity.

**Done means**: no overlay owner consumes unbounded work in one frame; the report makes incomplete
work explicit instead of reporting a silent stall.

## Fresh-session start order

1. Read `AGENTS.md`, `memory-bank/activeContext.md`, Spec 142 `spec.md`, this file, and
   `WorldRenderDiagnostics.cs` plus the overlay portion of `WorldScene.cs`.
2. Confirm the dirty tree and preserve unrelated `.gitignore`, `.specify/feature.json`, `imgui.ini`,
   and generated catalog changes.
3. Implement **8J.1 only**. Do not add queues, caching, residency changes, or GPU instancing until
   the first owner-attributed report exists.
4. Update Spec 142, this handoff, `activeContext.md`, and `progress.md`; run focused tests/build;
   commit only the slice.

## Commands and evidence

Focused checks after 8J.1:

```powershell
cd I:\parp\parp-tools\wow-viewer
dotnet test tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~WorldRenderDiagnosticsTests"
dotnet build tools\validation-capture\WowViewer.Tool.ValidationCapture\WowViewer.Tool.ValidationCapture.csproj -c Debug
```

User-run full-map capture after code proof:

```powershell
cd I:\parp\parp-tools\wow-viewer
dotnet run --project tools\validation-capture\WowViewer.Tool.ValidationCapture\WowViewer.Tool.ValidationCapture.csproj -- profile-render --client-root "H:\CLIENTS\World of Warcraft Cata beta 11927" --map-input "World\Maps\Azeroth\Azeroth.wdt" --output "output\diagnostics\azeroth-32-32-overlay-owners.json" --build "4.0.0.11927" --tile-x 32 --tile-y 32 --load-all-tiles --warmup-frames 3 --frames 10
```

The report must name the dominant overlay owner. It does not authorize a performance claim unless
the scene is settled enough for the comparison and the owner timing is no longer opaque.
