# Implementation Plan: 5.0.1 Physics — Decode Data, Drive a Licensed Solver

**Branch**: `v0.5.3` | **Date**: 2026-09-03 | **Spec**: [spec.md](spec.md)

**Input**: [spec.md](spec.md), the measured native evidence in
[workstream-atmosphere-501-ghidra.md](../../memory-bank/workstream-atmosphere-501-ghidra.md), and
[research.md](research.md).

## Summary

Recover only the external physics-data contract and observable budget behaviour of the 5.0.1 client,
then map that contract onto an independently licensed solver. The first delivery is evidence: an
address-cited contract, era-safe capability resolution, and an inspectable model-sidecar discovery
path. Solver selection is a hard gate: the exact version's license, deterministic stepping support,
and viable cloth route must be verified before a package reference is introduced. No Domino algorithm,
implementation detail, or transcribed proprietary code enters the repository.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Existing `WowViewer.Core*` libraries; the third-party physics solver is
not selected yet and is not a dependency.

**Storage**: Read-only configured game-client assets; checked-in evidence and configuration only.

**Testing**: `WowViewer.Core.Tests` for policy/parser diagnostics; operator-owned real-client model
and visual validation.

**Target Platform**: Cross-platform .NET library; Silk.NET desktop viewer binding after Core gates.

**Project Type**: Shared-library-first desktop viewer with a thin CLI inspector.

**Performance Goals**: Disabled physics has no measurable added frame cost; simulations are bounded
by distance and an explicit active-object budget before viewer integration.

**Constraints**: Ghidra read-only; data and observable behaviour only; unknown eras never default to
5.0.1; client paths are configuration; no archive is written; no solver package before verification.

**Scale/Scope**: One 5.0.1 era profile first; unknown model-sidecar format; later rigid bodies,
cloth, joints, culling, and viewer binding delivered in independent phases.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Pre-design: PASS with explicit gates.**

- **Repo independence**: source remains below `wow-viewer/`; client roots remain runtime configuration.
- **Library-first**: evidence models, era resolution, parser diagnostics, time-step policy, and budget
  policy are Core-owned; viewer and CLI layers consume them only.
- **Real-data validation**: parser and behaviour claims require an operator-configured 5.0.1 client,
  build identity, and hashes. Synthetic bytes test safety only.
- **Read-only/legal boundary**: existing readers are extended only where evidence proves a missing
  surface. This feature records layouts, data flow, and observable policy; it never imports or
  translates Domino algorithms. A solver needs a version-pinned permissive license record.
- **One phase at a time**: no parser, package, or viewer path begins before its preceding gate passes.

## Project Structure

### Documentation (this feature)

```text
specs/214-mop-physics-domino/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
src/
├── core/
│   ├── WowViewer.Core/Physics/             # era, data contract, budget, solver facade
│   ├── WowViewer.Core.IO/M2/                # evidence-gated model-sidecar reader
│   └── WowViewer.Core.Runtime/World/        # pure scheduling contracts
├── viewer/WoWViewer/                        # later model binding and settings
└── tools/inspect/                           # thin, read-only physics inspection command

tests/
└── WowViewer.Core.Tests/Physics/            # deterministic policy and parser tests
```

**Structure Decision**: Preserve the existing library-first layout. No test project references the
viewer, therefore testable capabilities and scheduling live in Core. The viewer consumes only a
validated facade after Core gates complete.

## Phase Roadmap

### Phase 0 — Evidence and selection gates

1. Enumerate assertion-handler caller clusters in a read-only Ghidra session, retaining source
   header/line attribution and external data/budget observations only.
2. Trace `Physics.cpp` and `PhysData.cpp` to the model-sidecar discovery boundary and collect actual
   5.0.1 candidate assets.
3. Produce an address-cited contract separating measured facts, inferences, and unknown fields.
4. Evaluate BepuPhysics v2 and Jitter2 at exact versions for permissive license, deterministic
   fixed-step support, collision coverage, and a viable cloth route; record a selection or blocker.
5. Record an evidence manifest containing configured client root, build fingerprint, asset hashes, and
   reproducible commands.

**Gate**: US1 and license evidence pass. No parser or solver package is added while sidecar location
or legal selection remains unverified.

### Phase 1 — Core capability, provenance, and diagnostics

**Status 2026-09-03**: Complete for the current solver-independent contract. Exact-build era
resolution, provenance, diagnostic admission outcomes, validated budgets/candidates, and deterministic
capacity assignment are implemented in Core.Runtime and covered by 16 focused passing tests.

1. Add a pure era-profile resolver for known 5.0.1, explicitly-disabled 0.5.3, and unknown builds.
2. Add immutable provenance and unsupported-construct diagnostic records.
3. Add deterministic budget policy: enabled state, cull distance, and active-object cap.
4. Unit-test every state and boundary condition in `WowViewer.Core.Tests`.

**Gate**: Alpha remains disabled, unknown builds are flagged, and every decision carries evidence.

### Phase 2 — Sidecar discovery and lossless inspection

1. Implement the evidence-defined sidecar resolver alongside the canonical M2/MDX reader.
2. Parse only verified records; diagnose malformed bytes and each unsupported construct.
3. Expose a thin inspector reporting bodies, shapes, joints, and provenance.
4. Validate against the Phase 0 real-asset manifest.

**Gate**: Models without sidecars retain current rendering, malformed data fails safely, and no
construct is silently dropped.

### Phase 3 — Solver adapter and rigid-body baseline

1. Add the selected, license-recorded solver behind a Core-owned adapter.
2. Map only Phase 2 verified primitive shapes and material parameters.
3. Add fixed timestep, bounded catch-up, sleep/reactivation, and reproducibility tests.
4. Validate resting penetration, energy, and no-tunnelling on a real-data-backed fixture.

**Gate**: SC-004 and SC-005 pass before joints, cloth, or viewer scheduling.

### Phase 4 — Cloth and animated attachment binding

1. Implement the selected library's cloth route without proprietary algorithm logic.
2. Map verified attachment constraints and consume weather-owned wind as an input contract.
3. Bind deformation to the model skeleton in the viewer adapter.
4. Obtain operator-owned side-by-side reference evidence for a physicalised flag/banner.

**Gate**: Cloth hangs, settles, remains attached, and responds to controlled wind.

### Phase 5 — Budgeted world scheduling

**Status 2026-09-03**: Item 1 is complete as a pure candidate-admission policy. Items 2–4 remain
blocked on an actual parsed asset and selected solver; no viewer scheduling or performance claim exists.

1. Apply Core budget policy to candidates in deterministic priority order.
2. Preserve/resume state across cull transitions without accumulated impulses.
3. Add viewer toggles and profiler counters for disabled/cull/budget reasons.
4. Measure a physicalised crowd through the operator-owned frame-time workflow.

**Gate**: Disabled cost matches the pre-feature baseline and all deferrals are inspectable.

### Phase 6 — Joints and hardening

1. Map each evidence-verified joint family and limits.
2. Add deterministic degree-of-freedom and limit fixtures.
3. Run malformed-data, unknown-era, and real-client regression sweeps.

**Gate**: All declared constructs are parsed or diagnosed, and release evidence meets SC-001–SC-011.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | — | — |
