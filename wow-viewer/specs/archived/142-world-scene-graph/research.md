# Research: World Scene Graph and Synthetic Workload Foundation

**Feature**: [World Scene Graph and Spatial Partitioning](spec.md)
**Date**: 2026-08-10

## Decision: Put the first graph contract in `WowViewer.Core.Runtime`

**Rationale**: The existing runtime library already owns world-object records, visibility frames,
pass coordinators, and frame-stage statistics. A graph introduced in the viewer app would create a
second authority and make validation capture unable to exercise the same contract.

**Alternatives considered**:

- Viewer-only graph: rejected because picking, validation capture, and future tools would diverge.
- New renderer project: rejected because Phase 1 has no GPU ownership and should not duplicate the
  current runtime pass contracts.

## Decision: Use stable IDs and explicit parent ownership

**Rationale**: Performance comparisons need replayable node identities, and tile/asset detach must
  be provably complete. Caller-provided IDs also let real-client adapters bind decoded placement
  identity later without changing the graph model.

**Alternatives considered**:

- Object-reference identity: rejected because it cannot survive serialization or current/new path
  comparison.
- Implicit path IDs only: rejected because reparenting and shared asset instances need explicit
  placement identity.

## Decision: Conservative bounds are first-class and non-rejectable is explicit

**Rationale**: A parent-level false negative is a rendering correctness failure. If containment
cannot be proven after a transform or incomplete asset load, the graph must preserve visibility by
declaring the node non-rejectable and reporting that state.

**Alternatives considered**:

- Recompute a tight aggregate on every attach: rejected for the first slice because it hides the
  cost model and is not safe for incomplete streaming payloads.
- Silently use child bounds: rejected because it can discard descendants when the child set is
  incomplete.

## Decision: Keep synthetic minimap data outside the world-scene fixture

**Rationale**: A generated RGB/height tile displayed as a texture is not exercising 3-D graph
traversal, WMO portals, object submission, or spatial queries. It receives its own evidence label
and cannot close renderer performance gates.

**Alternatives considered**:

- Treat every synthetic image as a synthetic scene: rejected because it produces false renderer
  evidence.
- Require client assets for all performance work: rejected because deterministic generated scene
  scales are valuable for isolating resident-versus-visible cost.

## Decision: JSON is the replay interchange, not the runtime hot path

**Rationale**: A human-readable manifest makes workload provenance and replay reviewable. The graph
itself remains in-memory; no JSON parsing is placed inside frame traversal.

**Alternatives considered**:

- Binary-only fixture: rejected for the first phase because it makes evidence review harder.
- Runtime reflection serialization: rejected because stable schema fields and validation are needed.

## Decision: Eliminate the observed overlay stall before full-residency redesign

**Rationale**: The real `Azeroth 32_32` full-map capture on the production path establishes a
clear first limiter: full residency took 66,388.8 ms, but steady frames were also catastrophically
blocked by `overlay` at 39.5-44.0 seconds on alternating samples. That is not a visibility or GPU
claim; it is repeatable CPU work hidden behind one broad stage. Streaming every ADT first would
leave the viewer unusable whenever that overlay path runs.

**Alternatives considered**:

- Begin GPU-driven instancing immediately: deferred. It cannot compensate for a 40-second CPU
  overlay rebuild, and requires a settled CPU submission contract first.
- Treat `--load-all-tiles` as normal startup behavior: rejected. It is an explicit stress mode;
  normal viewer startup remains camera-first streaming.

## Decision: Use a capability-gated modern submission ladder after CPU work admission

**Rationale**: Cataclysm's dense-detail direction and modern OpenGL offer compatible tools:
immutable shared asset buffers, texture/material arrays, per-instance buffers, and multi-draw or
indirect submission. They must be introduced behind a renderer capability record with an explicit
legacy fallback, after overlay and residency work prove which content is ready for a stable batch.

**Alternatives considered**:

- One universal GPU path: rejected because animated, transparent, particle, ribbon, WMO-group,
  and unsupported-driver paths require correctness-preserving fallbacks.
- CPU batching only as the end state: rejected. It remains useful as a fallback and staging
  contract, but cannot be the long-term dense-scene path.

## 2026-08-14 implementation checkpoint

- The opaque WMO path now collects internal doodad transforms across visible WMO placements before
  submitting them. Shared IModelRenderer instances use one GPU instance batch where supported or
  one renderer-level CPU batch otherwise.
- WMO shell batching and internal doodad batching remain separate passes. Transparent and
  correctness-sensitive doodads stay on the existing placement-aware path.
- This is not yet a performance claim: the next proof is a user-run dense-WMO/Stormwind capture
  comparing unique doodad renderers, instance counts, draw submissions, and CPU stage time.
