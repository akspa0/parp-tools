# Implementation Plan: World Scene Graph and Spatial Partitioning

**Branch**: `142-world-scene-graph` | **Date**: 2026-08-10 | **Spec**: [spec.md](spec.md)

## Summary

Phase 1 establishes the library-owned runtime contract that later renderer migration will consume:
a nestable scene graph with stable node identity, transforms, conservative bounds, attach/detach
ownership, and deterministic synthetic-world workload manifests. The first slice is deliberately
below the renderer: it proves graph structure, replayability, and invariant enforcement before
moving the existing `WorldScene` type collections behind the new traversal.

The synthetic fixture is a workload generator, not a second renderer. It produces the same node
and render-pass descriptors that a real client scene will use, while keeping payloads generated and
client-independent. Image-only synthetic minimaps remain outside this benchmark contract.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: `System.Numerics`, `System.Text.Json`, existing `WowViewer.Core.Runtime`
  contracts, xUnit test stack

**Storage**: In-memory graph and JSON-serializable workload manifest; no client data or generated
  scene assets are committed

**Testing**: `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug`
  plus focused graph/manifest tests and `dotnet build` for the runtime project

**Target Platform**: Windows desktop viewer runtime; the core graph and fixture contracts remain
  platform-neutral

**Project Type**: Library-first runtime model with a later viewer integration

**Performance Goals**: Phase 1 must make resident/visible counts, graph depth, subtree rejection
  potential, and workload identity measurable without a benchmark-only renderer. It does not claim
  an FPS improvement until the graph is wired into real frame traversal.

**Constraints**: Do not rewrite existing format readers, do not launch GPU captures or long runs,
  do not hardcode client roots, do not make image-only minimap data satisfy renderer proof, and do
  not modify the legacy reference repository.

**Scale/Scope**: Support sparse maps, nested asset sub-graphs, four replayable synthetic scales,
  repeated asset references, portal descriptors, and explicit render-pass/material descriptors.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Repo independence**: PASS. New source and tests stay under `wow-viewer/`.
- **Library first**: PASS. The graph and workload model live in `WowViewer.Core.Runtime`; no
  viewer-only duplicate is introduced.
- **Real-data validation**: PASS for this phase. No real-data claim is made; real-client parity is
  a later gate and remains user-run.
- **Format ownership**: PASS. No parser or writer is changed.
- **Streaming/user-run heavy work**: PASS. The slice is in-memory and deterministic; it launches no
  training, harvest, GPU capture, or broad benchmark.
- **One phase at a time**: PASS. This plan stops after graph/manifest proof; renderer integration,
  spatial indexing, portal culling, and migration remain later phases.
- **Documentation/memory**: PASS. Spec 142, plan/tasks, quickstart, and memory-bank are updated
  with the slice.

## Project Structure

### Documentation

```text
wow-viewer/specs/142-world-scene-graph/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── synthetic-world-workload.schema.md
├── checklists/requirements.md
└── tasks.md
```

### Source and Tests

```text
wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/
├── WorldSceneNode.cs
├── WorldSceneNodeKind.cs
├── WorldSceneGraph.cs
├── WorldSceneGraphSnapshot.cs
├── WorldSceneTraversal.cs
├── WorldSceneGraphObjectAdapter.cs
├── SyntheticWorldWorkload.cs
└── SyntheticWorldWorkloadBuilder.cs

wow-viewer/tests/WowViewer.Core.Tests/World/
├── WorldSceneGraphTests.cs
└── SyntheticWorldWorkloadTests.cs
```

**Structure Decision**: Keep graph ownership in the existing runtime library so the viewer,
validation capture, future spatial queries, and future renderer migration consume one contract.
The fixture builder stays beside the contract but has no OpenGL or client-file dependency. A later
CLI may serialize/replay manifests, but this phase does not create a parallel renderer tool.

## Phase 0 — Research Decisions

1. Preserve existing `System.Numerics.Matrix4x4` and min/max bounds conventions used by runtime
   world objects.
2. Use stable caller-provided node IDs rather than generated object references; this makes replay
   and current/new path comparison possible.
3. Treat a node's bounds as conservative world-space rejection bounds. A parent that cannot prove
   containment is marked non-rejectable instead of silently shrinking to its children.
4. Represent shared asset structure as a reusable fixture descriptor plus per-placement nodes; do
   not copy the entire asset tree for every placement in the first contract.
5. Keep portal links as graph metadata in Phase 1. Portal frustum construction and culling belong
   to the later traversal phase.

## Phase 1 — Graph and Workload Contract

**Status**: Complete for the bounded foundation slice on 2026-08-10. The runtime graph and
deterministic manifest build; focused proof is 8 passing tests. No renderer integration or GPU
performance claim is made.

1. Add immutable node identity/kind/render metadata and guarded parent/child attachment.
2. Add graph enumeration, lookup, subtree detach, world-transform propagation, and invariant
   validation.
3. Add the versioned synthetic-world workload manifest and deterministic builder for sparse tiles,
   nested WMO/M2/PM4 nodes, repeated assets, render-pass mix, and portal descriptors.
4. Add snapshot/count output sufficient to compare current and future traversal without running a
   GPU renderer.
5. Add focused tests for identity, bounds containment, cycle prevention, detach cleanup,
   deterministic replay, and explicit separation from image-only minimap workloads.

## Phase 2 — Conservative Shared Traversal

**Status**: Complete for the library slice on 2026-08-10. The traversal rejects a complete subtree
after one failed node test, preserves non-rejectable nodes, and reports visited/tested/skipped/
visible counts. Its graph-validation pass is optional so a graph proven at rebuild time does not
pay a full invariant walk on every frame.

1. Traverse one graph with an injected visibility predicate and a renderable-node selector.
2. Attribute the rejected region and count descendants skipped without visiting them.
3. Preserve unknown/incomplete bounds as visible-but-non-rejectable.
4. Prove the behavior with synthetic fixed-camera-style tests before viewer integration.

## Phase 3 — Runtime Object Adapter and Opt-In WorldScene Traversal

**Status**: Complete for the bounded selector slice on 2026-08-10. Existing `WorldScene` object
lists can be adapted to `map -> tile/external bucket -> placement`, and the opt-in selector uses
one graph traversal before the existing WMO/MDX visibility and asset-readiness checks. The legacy
path remains the default; no FPS, GPU, portal, pass-order, or real-client parity claim is made.

1. Adapt resolved `WorldObjectInstance` placements without reopening format readers or inventing
   WMO group bounds.
2. Rebuild the graph only when object residency or resolved bounds change.
3. Use one conservative frustum traversal to feed the existing WMO and M2 collectors when
   `UseHierarchicalSceneTraversal` is enabled.
4. Expose graph snapshot and traversal diagnostics for later validation-capture reporting.
5. Prove the adapter with stable IDs, tile grouping, unknown-bound fail-open, and replay tests;
   compile the viewer project to verify the integration seam.

## Later Phases (Not Started In This Slice)

- **Phase 4**: WMO portal-restricted nested view volumes and fallback diagnostics.
- **Phase 5**: Per-pass visible/non-visible queues, shared animation update ownership, and query
  reuse.
- **Phase 6**: Incremental terrain/chunk graph migration, synthetic four-scale measurements, and named
  real-client parity captures.

## Complexity Tracking

No constitution violations are required. The workload manifest is additional contract surface,
not a second runtime or renderer; it exists because synthetic image data and synthetic 3-D scene
data answer different performance questions.
