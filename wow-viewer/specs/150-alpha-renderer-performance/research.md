# Research: Alpha 0.5.3 Renderer Performance Evidence and Optimization

**Feature**: [Alpha 0.5.3 Renderer Performance Evidence and Optimization](spec.md)

**Date**: 2026-08-14

## Decision 1: Use the production profile before changing the renderer

The existing `profile-render` command already drives the production `WorldScene.Render` path in a
hidden OpenGL context and emits stage/workload diagnostics. The first performance artifact will use
that path with a fixed 0.5.3 control scene, settled warmup, and repeated samples.

Rationale: the interactive screenshot is useful evidence of user impact (`23 FPS`, `41.2 ms` CPU
frame time, `356/14895` M2 and `8/179` WMO counters), but one camera frame cannot identify whether
the limit is CPU submission, GPU shading, driver synchronization, asset settling, or overlays.

Rejected: selecting batching or LOD from the screenshot alone.

## Decision 2: Treat current retained terrain work as a foundation, not a performance claim

`TerrainRenderer.RenderTiles` already uses retained tile VAOs, one indexed draw per admitted tile,
texture arrays, frustum/distance checks, and counters for draws, uniforms, active texture calls, and
texture binds. `TerrainManager` bounds detailed tiles and retained residency, and `WorldScene` passes
the selected tile set into terrain rendering.

Implication: the first terrain experiment should not assume that replacing tile geometry or adding
another generic cache is useful. It should measure whether terrain is actually dominant and whether
the remaining cost is tile iteration, shader/texture state, liquid, or GPU fragment work.

## Decision 3: Treat object submission and per-frame collection as separate costs

The production world frame separates WMO visibility/submission, MDX visibility/animation/opaque
submission, transparent sorting/submission, and liquid. `WorldObjectVisibilityCollector` performs
distance, frustum, vision-cone, projected-size, and asset-readiness checks. The opaque pass already
has WMO and MDX grouping/instancing contracts, while transparent work remains distance ordered.

Implication: the report must compare visible count, culled count, batch count, fallback count, and
draw/state pressure. A higher visible count after an optimization is not automatically a win, and a
lower draw count is not safe if it changes transparent/material behavior.

## Decision 4: Existing M2/WMO batching work remains unproven until profile evidence exists

Spec 136 and Spec 142 already contain bounded opaque M2/WMO grouping and GPU-instancing seams. The
current source includes `IGpuInstancedModelRenderer`/`IGpuInstancedWmoRenderer` paths and the world
pass coordinator. These are reusable candidates, but no native-0.5.3 performance or real-client
visual claim is inferred from their presence.

Rejected: rewriting the renderer or moving all renderer files into the shared library as the first
response. That is the broader Draft Spec 056 lane and would obscure the first measurable owner.

## Decision 5: Native Ghidra evidence must be build-scoped and behavior-only

The open program is the exact `WoWClient.exe` 0.5.3.3368 binary. The rendering audit should record
anchors for:

1. world/camera render entry and pass ordering;
2. terrain/chunk visibility and submission;
3. WMO/M2/MDX admission, distance, or LOD decisions;
4. resource creation/reuse and render-state/material grouping;
5. far-horizon/WDL or reduced-detail behavior, if present.

Each row records an address/function label, observed inputs/outputs or control flow, confidence, and
the viewer-side implication. The ledger does not copy disassembly or client implementation code.

The existing 0.5.3 Ghidra note proves audio contracts only; it does not yet prove renderer behavior.
That is an explicit open evidence gap, not permission to substitute 3.3.5 or 4.x findings.

## Decision 6: Separate CPU, GPU, driver, and I/O proof

The current `WorldRenderDiagnostics` report explicitly says GPU timing is not yet attributed. The
performance lane will retain that limitation and add optional backend timing only if it can be
collected without changing the render path. CPU stage time, GPU/driver time, asset-read time, and
settled-frame state are separate fields.

Rejected: calling CPU submission time "renderer FPS" or treating a clean Debug build as a performance
result.

## Decision 7: One reversible experiment per phase

The first optimization may be one of:

- reuse of existing per-frame scratch collections;
- visibility/admission ordering that avoids repeated work;
- compatible opaque grouping or state/uniform reduction;
- retained resource reuse;
- build-scoped terrain/object distance reduction or far-horizon selection.

The choice is made only after the baseline. It must keep the current path as a switchable fallback,
record fallback counts/reasons, and compare the same scene before and after.

## Evidence inventory

| Evidence | Current state | Use |
|---|---|---|
| `profile-render` | Existing production OpenGL capture | Baseline and post-change stage/workload reports |
| `WorldRenderFrameStats` | CPU stages and visible/submitted counts | Stable report foundation |
| `TerrainRenderer` counters | Draw/uniform/texture-state counters | Terrain state-pressure attribution |
| `WorldObjectVisibilityCollector` | Distance/frustum/cone/projected-size admission | Object workload attribution |
| Spec 136 | M2/WMO batching and deferred I/O decisions | Candidate/fallback context, not proof |
| Spec 142 | Scene graph, profile-render, overlay attribution | Measurement and residency context |
| 0.5.3 Ghidra program | Open, read-only; renderer anchors not yet recorded | Native evidence source |
| User screenshot | 23 FPS / 41.2 ms interactive frame | Motivation only, not benchmark |

## Open proof gates

- Exact 0.5.3 native renderer functions and state/resource strategy remain to be recovered in Ghidra.
- Current production CPU/GPU split has not been captured on the fixed control scene.
- The dominant owner is not selected until that capture exists.
- No source optimization, FPS gain, or native parity claim is made by this planning artifact.
