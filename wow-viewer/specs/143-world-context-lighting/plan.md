# Implementation Plan: World Context And Lighting Parity

**Branch**: `143-world-context-lighting` | **Date**: 2026-08-11 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/143-world-context-lighting/spec.md`

**Branch note**: Speckit branch creation was attempted once and was blocked because the shared
`.git/index.lock` cannot be created in this environment. These artifacts are therefore authored on
the current `142-world-scene-graph` branch; implementation must not be treated as branch-isolated
until the user resolves that repository permission issue.

## Summary

Create one camera-owned world-context contract that resolves the current ADT area and WMO interior
area from the active client data, exposes raw IDs and provenance, and feeds the same player-head
camera state to visibility, fog, and lighting. The first implementation slice repairs the existing
AreaTable lookup boundary and adds diagnostics before any shader change. WMO area decoding, camera
rig behavior, and lighting are separate gated slices so a visual improvement cannot conceal a bad
identity or coordinate contract.

Lighting work consumes build/profile evidence from Specs 106 and 138. It may reuse the existing WMO
vertex colors, baked weights, light references, lightmaps, M2 scene-light inputs, and current
uniform plumbing, but it must not claim original BLS parity without observed client evidence. An
equivalent shader path is acceptable only when its inputs, limitations, and active profile are
reported.

## Technical Context

**Language/Version**: C# on .NET 10

**Primary Dependencies**: Existing WowViewer.Core/IO readers, DBCD plus WoWDBDefs DBD schemas,
Silk.NET OpenGL, ImGui.NET, System.Numerics, and the existing xUnit test projects

**Storage**: No new persistent storage. Runtime state is in-memory; diagnostics and validation
captures use the existing viewer output paths and user-selected client roots.

**Testing**: Focused xUnit tests for read-model and lookup contracts; isolated Debug builds; user-run
real-client viewer/capture validation for native OpenGL, cross-era data, frame time, and visual
acceptance. The agent must not launch the heavy real-client capture.

**Target Platform**: Windows desktop viewer with the existing cross-platform library build kept
compilable

**Project Type**: Desktop renderer/application with shared format and runtime libraries

**Performance Goals**: Context evaluation is bounded and allocation-light on the render loop; no
per-frame full placement scan or table parse. The feature must preserve the flat-path baseline and
stay within the spec's documented <=10% p95 frame-time regression budget unless a profile report
records the reason and a compensating improvement.

**Constraints**: Build/profile-aware behavior; no hardcoded area names, numeric DBC column positions,
or client-local paths; no rewrite of existing ADT/WMO/M2 readers; no silent fallback that is reported
as parity; preserve WMO batching and scene-residency boundaries; one phase at a time with a real-data
exit gate.

**Scale/Scope**: Early Alpha through 4.x world maps, terrain chunks, loaded WMO roots/groups, placed
MDX/M2 assets, and the existing interactive camera/status/diagnostic surfaces. Full gameplay,
collision, character animation, and a complete BLS compiler are out of scope.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Gate | Status | Evidence / boundary |
|---|---|---|
| Repo independence | PASS | All new code and docs stay under `wow-viewer`; game clients are runtime inputs only. |
| Library-first ownership | PASS | Area/WMO read contracts belong in `src/core`; viewer wiring stays thin. |
| No reader rewrite | PASS | Existing MCNK/WMO/M2 readers are inspected and extended only for a proven missing field. |
| Client provenance | PASS | DBC/DBD, WMO, light, and shader claims carry build/profile/source diagnostics. |
| No hardcoded names or IDs | PASS | Logical DBD columns and decoded source fields are required. |
| Performance boundary | PASS | Context lookup is bounded; renderer work remains gated by measured evidence. |
| Phase discipline | PASS | Phase 0 research precedes Phase 1 design and implementation. |
| User-owned heavy work | PASS | Real-client captures, broad map sweeps, and GPU profiling are handed to the user. |
| Branch isolation | BLOCKED/RECORDED | Speckit could not create the feature branch because `.git/index.lock` is read-only here. |

## Project Structure

### Documentation

```text
specs/143-world-context-lighting/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/world-context-contract.md
├── checklists/requirements.md
└── tasks.md
```

### Source Code

```text
src/
├── core/
│   ├── WowViewer.Core/                  # profile-neutral world-context records and math
│   ├── WowViewer.Core.IO/               # AreaTable/WMO/DBD-backed decoding and provenance
│   └── WowViewer.Core.Runtime/           # camera/world context and lighting input contracts
└── viewer/
    └── WoWViewer/                       # status, camera controls, WMO/M2 shader consumers

tests/
├── WowViewer.Core.Tests/                # lookup, ID, WMO, and camera contract tests
└── WowViewer.Core.Runtime.Tests/         # runtime context and lighting selection tests when present
```

Likely implementation paths are `src/core/WowViewer.Core.IO/Dbc/`,
`src/core/WowViewer.Core.IO/Wmo/`, `src/core/WowViewer.Core.Runtime/World/`,
`src/viewer/WoWViewer/Terrain/AreaTableService.cs`,
`src/viewer/WoWViewer/Rendering/Camera.cs`, `src/viewer/WoWViewer/Rendering/WmoRenderer.cs`, and
`src/viewer/WoWViewer/Rendering/M2Renderer.cs`. Exact new filenames remain subject to Phase 0/1
ownership decisions.

**Structure Decision**: Keep decoding and immutable contracts in the shared core libraries, keep
camera/world-context orchestration in the runtime layer, and keep ImGui/status/OpenGL adaptation in
the viewer. This avoids placing build-specific DBC logic inside the renderer and avoids making the
legacy viewer or a shader path the authority for world identity.

## Delivery Phases And Gates

### Phase 0 — Evidence and contract audit

Resolve the current AreaID coordinate/map mismatch, establish the exact WMOAreaID field and version
variants from existing readers and client-backed samples, inventory lighting inputs, and record the
unsupported-BLS policy. Exit only when every implementation field has a source, version scope, and
diagnostic representation. No renderer behavior changes in this phase.

### Phase 1 — ADT area context

Introduce a structured context result instead of a nullable name. It preserves raw ADT ID,
coordinate/chunk source, map identity, DBD logical columns, parent chain, resolved name, and an
explicit unresolved reason. Fix the camera-to-chunk lookup and map semantics with focused synthetic
tests before wiring the status bar.

### Phase 2 — WMO interior context

Extend the proven WMO read model with the evidence-backed WMO/group area identifier and containment
source. Select WMO context deterministically, expose candidate/confidence data, and fall back to ADT
context when the WMO field or volume is unavailable. Do not infer an area from a filename or WMO
name.

### Phase 3 — Player-head camera rig

Make eye position, orientation, mode, and explicit head offset a serializable runtime state. Feed one
same-frame state to view construction, WMO containment, terrain area lookup, fog, and lighting. Keep
museum/elevated inspection as an explicit reversible mode; do not add gameplay collision or input
systems in this feature.

### Phase 4 — Evidence-backed lighting slice

Create a profile-scoped lighting input selection for WMO and MDX/M2. First consume existing baked,
vertex, lightmap, local-light, directional, ambient, and fog inputs. Then implement the smallest
shader/effect change that makes contributions attributable and non-flat. Original BLS behavior is
used only where the active build evidence and compatible inputs support it; otherwise diagnostics say
`equivalent fallback` with the missing contract.

The first bounded lighting correction is the LIT spatial coordinate contract. `lights.lit` list
headers use client fixed-point XZY positions: divide by 36, decode semantic game XYZ, and then apply
the active map-origin transform for renderer-space consumers. Raw, decoded WoW, and renderer values
remain separately diagnosable; this correction does not claim local-light visual parity.

### Phase 5 — Cross-era and performance proof

Run focused tests/builds in the workspace, then hand the user exact PowerShell commands for the
real-client matrix. Compare early, 1.x/3.x, and 4.x context results; WMO entry/exit transitions;
lighting sources; camera traces; and p95 frame stages. Fix failures before any release claim.

## Phase Task Boundaries

Each phase has at most ten implementation concerns and one exit gate. No phase may begin while the
previous phase's evidence and focused validation are incomplete. Broad shader rewrites, full BLS
translation, gameplay, collision, and whole-map residency changes remain outside this feature.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Branch isolation unavailable | The environment cannot create `.git/index.lock`. | Silently continuing as if the feature branch existed would make later commits ambiguous. |
