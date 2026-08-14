# Alpha 0.5.3 Renderer Performance Workstream

Last updated: 2026-08-14

This is an evidence/planning note for Spec 150. It does not claim native-client or viewer FPS parity.

## Current viewer evidence

- The interactive user capture showed approximately `23 FPS`, `41.2 ms` CPU frame time, `356/14895`
  visible/total M2 or MDX placements, and `8/179` visible/total WMO placements. This motivates the
  lane but is not a repeatable benchmark.
- `profile-render` already drives the production `WorldScene.Render` path in a hidden OpenGL context
  and emits CPU stage summaries, workload counts, overlay-owner timings, pending work, and findings.
- `WorldScene` separates lighting, sky, WDL, terrain, WMO visibility/submission, MDX visibility/
  animation/opaque submission, liquid, transparent sorting/submission, overlays, scene maintenance,
  and deferred asset work.
- `TerrainRenderer.RenderTiles` already has retained tile VAOs, indexed tile draws, texture arrays,
  distance/frustum admission, tile-key admission, and per-frame draw/uniform/active-texture/texture-
  bind counters. Do not assume a terrain rewrite is the first win.
- Object admission already performs bounds distance, frustum, vision-cone, projected-size, asset-ready,
  and pending-load decisions. Opaque WMO/MDX grouping and GPU-instancing seams exist, while
  transparent/material-sensitive work remains on fallback paths.
- `WorldRenderDiagnostics` explicitly reports that GPU timer-query attribution is not yet available.
  CPU submission time must not be reported as GPU time.

## Native 0.5.3 evidence status

The open read-only program is:

`H:\CLIENTS\Vanilla\0.x\0_5_3_3368\World of Warcraft\WoWClient.exe`

The existing Ghidra workstream proves audio contracts only. No renderer behavior is promoted to
`proven` until an anchor is added below. Use this ledger shape:

```text
Build: 0.5.3.3368
Anchor: <function/symbol/address/xref/data reference>
Area: world | terrain | object | resource | state | lod | unknown
Observation: <behavior seen in Ghidra>
Confidence: proven | inferred | unknown
Viewer implication: <bounded candidate constraint or experiment>
```

### Open renderer questions

1. What is the world render entry and pass order for terrain, objects, liquids, and overlays?
2. What inputs reject an MCNK/chunk before vertex submission: frustum, distance, portal/area, or a
   combination?
3. How are WMO/M2/MDX instances admitted, grouped, and distance-limited?
4. Which resources are created once and reused, and which render-state/material transitions are
   intentionally changed per batch?
5. Does the Alpha client have a reduced terrain/far-horizon path, and how is it selected?
6. Is the native bottleneck comparable to the viewer's CPU stage, GPU/driver work, or asset I/O?

## Working hypotheses (not native facts)

- Coarse tile/chunk admission before per-object work may matter more than another generic cache.
- Stable retained resources and compatible opaque state buckets should reduce driver/state pressure;
  transparent, animated, particle/ribbon, and material-sensitive work must retain fallbacks.
- Existing current-path batching may already solve part of the problem; profile counters must show
  whether its eligible set is actually large and whether GPU instancing is used in the target scene.
- WDL/far-distance reduction is valuable only if the 0.5.3 client evidence and fixed captures support
  the same behavior; do not borrow later-build LOD assumptions.

## Proof boundary

- No source optimization has been implemented under Spec 150.
- No native renderer anchor has been recorded yet.
- No repeatable 0.5.3 `profile-render` baseline has been run in this session.
- Focused tests/build may prove contracts; only user-run real-client comparison can prove interactive
  visuals, native FPS comparison, GPU driver behavior, or a net frame-rate gain.
