# Current Implementation Audit — Physics (2026-09-03)

## Scope and method

This is a source audit of `src/` and `tests/`, not a claim about the 5.0.1 binary or a request to
add new subsystem surfaces. It distinguishes **implemented adjacent capabilities** from the missing
5.0.1 physics path so follow-up work starts from an actual gap.

## What exists

| Surface | Current implementation | Does not provide |
|---|---|---|
| M2 metadata | [`M2ModelDocument.HasPhysicsSidecar`](../../../src/core/WowViewer.Core/M2/M2ModelDocument.cs) exposes bit `0x20` from the parsed M2 flags. | No sidecar name/path resolution, byte read, parser, diagnostic, or caller consumes this property. The bit is a signal, not an implemented load path. |
| MDX collision geometry | [`MdxCollisionReader`](../../../src/core/WowViewer.Core.IO/Mdx/MdxCollisionReader.cs) reads classic `CLID`; [`WorldAssetManager`](../../../src/viewer/WoWViewer/Terrain/WorldAssetManager.cs) caches only summary counts, bounds, and sampled vertices. | No collision world, broad phase, contacts, rigid bodies, or collision response. This is asset/PM4 inspection geometry, not runtime physics. |
| Camera collision | [`WorldScene.TryResolveCameraPathCollision`](../../../src/viewer/WoWViewer/Terrain/WorldScene.cs) clamps a camera sample to terrain height and uses loaded WMO placement AABBs. | No triangle collision or physics; its own source calls the WMO approximation a conservative exterior shell and keeps it opt-in. It must not be treated as a physics solver input without an evidence-backed adapter. |
| Particle motion | [`ParticleEmitter.Update`](../../../src/viewer/WoWViewer/Rendering/ParticleSystem.cs) advances viewer particles with gravity and emitter-local random spawning. | No physicalised-model data, collision, cloth constraints, wind contract, fixed timestep, sleep, budget, or determinism. The `Random` instance and unbounded emission loop make this unsuitable as a physics foundation. |
| Runtime policy | [`PhysicsRuntimePolicy.cs`](../../../src/core/WowViewer.Core.Runtime/World/Physics/PhysicsRuntimePolicy.cs) resolves exact measured builds, carries provenance/diagnostics, validates budgets and candidates, and deterministically assigns distance/capacity outcomes without depending on a solver or sidecar layout. | No asset discovery, parse, body creation, simulation, collision response, cloth, joints, animation binding, or viewer consumption. An enabled build means only that candidates may enter this policy boundary. |
| Focused verification | [`PhysicsRuntimePolicyTests.cs`](../../../tests/WowViewer.Core.Tests/PhysicsRuntimePolicyTests.cs) covers measured alpha disabled, exact 5.0.1.15464 enabled, unknown builds fail-closed, deterministic ordering, culling, capacity deferral, provenance, diagnostics, duplicate identities, forged availability, and invalid input. | No real-client, visual, solver, file-format, or performance proof. |

## Confirmed missing implementation

1. There is no physics asset resolver/parser, simulation host, rigid-body/shape/joint/cloth model, or
   solver adapter under `src/`; only the solver-independent era/admission policy exists.
2. No project references BepuPhysics, Jitter, Bullet, PhysX, or another general-purpose physics solver;
   [`Directory.Packages.props`](../../../Directory.Packages.props) confirms no central package version.
3. No runtime code references `HasPhysicsSidecar`; setting the M2 flag has no visible or diagnostic
   effect today.
4. Policy tests now exist, but no test covers `HasPhysicsSidecar`, physics asset bytes, rigid bodies,
   soft bodies, solver integration, or the 5.0.1 adapter boundary.
5. `TransportPhysics` exists only as vendored DBC-definition/listfile material. It has no reader,
   runtime consumer, or link to model movement, and remains explicitly outside the model-cloth route.

## Requirement disposition

| Requirement group | Status | Evidence-backed gap owner |
|---|---|---|
| FR-009–FR-012 data discovery and safe parse | Missing | Canonical M2/MDX Core.IO path; the discovery contract is still unmeasured. |
| FR-013–FR-020 simulation, determinism, cloth, joints | Missing | New Core-owned solver adapter only after the sidecar and license gates. Existing particle code is unrelated. |
| FR-021–FR-024 culling and budget | Partial | Core.Runtime now has deterministic enable/distance/capacity admission and explicit outcomes. No scene scheduler or measured runtime defaults are wired. |
| FR-025–FR-027 era/provenance | Partial | Exact 0.5.3.3368, exact 5.0.1.15464, and unknown-build behavior is implemented with provenance. No asset/viewer integration consumes it yet. |
| Existing collision readers | Implemented for their present purpose | Remain owned by Core.IO/PM4 inspection and must not be repurposed by assumption. |

## Correct next boundary

The smallest next action is to prove whether the unconsumed `0x20` flag actually corresponds to the
5.0.1 adapter's model-sidecar discovery path, using read-only adapter evidence and real candidate
models. Only that result can say whether the M2 flag is the correct join point or merely an unrelated
metadata bit.

The new policy slice does not weaken that gate: it is independent of the flag, file layout, and
solver. Parser, package, simulation, cloth, and viewer implementation remain blocked until the
sidecar and solver-selection evidence tasks pass.
