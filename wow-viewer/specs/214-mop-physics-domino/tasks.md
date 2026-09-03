# Tasks: 5.0.1 Physics — Decode Data, Drive a Licensed Solver

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md), [physics runtime contract](contracts/physics-runtime-contract.md), and
[quickstart.md](quickstart.md).

**Rule**: Sidecar parsing, package references, simulation, cloth, and viewer binding may not start
until T005 proves the sidecar discovery boundary and T007 records the exact selected solver's
permissive license and cloth route. Solver-independent era/provenance and admission policy may proceed
because it does not assume a file layout, solver API, or Domino algorithm.

**Phase 2 evidence gate PASSED 2026-09-03.** T003/T004/T005 are complete and recorded in
[domino-caller-map.md](evidence/domino-caller-map.md) and
[physics-adapter-contract.md](evidence/physics-adapter-contract.md). The headline results:
the sidecar is **`.phys`**, associated with its model **by filename with the extension replaced**
(no id, no table); the container is a standard reversed-tag Blizzard chunked file with magic `PHYS`,
**version `u16` must be 0**, and nine record chunks whose strides are measured, not guessed; every
array's field name is recovered from Blizzard's own `PhysData.h` bounds asserts; malformed and
absent data **fail closed at every stage** and unknown chunk tags are **skipped by size**, so the
format is forward-compatible by construction. Gravity is measured as **`(0, 0, -10.0)`**, not 9.81.

**Solver selection gate PASSED 2026-09-03 (T007).** **Jitter2 2.8.10** is selected and BepuPhysics v2
rejected — not on licensing (Apache-2.0 was acceptable) but because it has **no cloth or soft-body
support and no determinism evidence**, the two properties this feature is chosen for. Jitter2 is
**MIT** (verified from the `LICENSE` file; the empty NuGet `licenseExpression` on 2.7.x+ is a
`<license type="file">` packaging change, not a license change), targets **`net10.0`**, ships
`SoftBodyTriangle` + `SpringConstraint` in the library with a ~115-line first-party
`SoftBodyCloth : SoftBody` **demo** built on them (sample code to adapt, *not* public API — state it
that way), and enforces determinism through `World.Deterministic.cs`, `StableMath.cs`,
reproducibility tests, and a **CI workflow that hashes simulation output**. See
[solver-selection.md](evidence/solver-selection.md).

**Both Phase 0 gates are now answered on evidence.** Sidecar parsing (Phase 2) is unblocked but still
waits on **T002** for real-byte validation. **No package reference has been added** — that is Phase 3
and needs the operator's go-ahead, being a permanent third-party dependency.

**T012 is now eligible**: its precondition was T005 and T007, and both are complete.

**Current implementation audit**: [current-implementation-audit.md](evidence/current-implementation-audit.md)
confirms that `HasPhysicsSidecar` is unconsumed metadata, classic MDX `CLID` is inspection geometry,
camera collision is an opt-in navigation approximation, and particle movement is not the requested
physics system. A bounded Core.Runtime policy slice now supplies exact-build era resolution,
provenance, validated budgets, deterministic candidate admission, and explicit outcomes. No sidecar
resolver/parser, solver, body simulation, collision response, cloth, joints, or viewer integration
exists yet.

## Dependency graph

`T001 + T002 -> T003 -> T004 -> T005 -> T006 + T007 -> T008 -> parser/solver task generation`

`T002a -> T009 -> T010 -> T011` is the completed, solver-independent US7/policy path.

US1 and US7 are the initial independent outcomes. The safe US7 policy subset is complete. US2–US6
remain intentionally blocked by the Phase 0 evidence/selection gate; this prevents speculative format
parsing and a solver decision that cannot deliver cloth.

## Phase 1 — Setup

- [x] T001 Confirm the 5.0.1 Ghidra program identity, image base, function count, and read-only policy in `memory-bank/workstream-atmosphere-501-ghidra.md`
- [ ] T002 Capture the configured real-client build identity and candidate asset hashes in `specs/214-mop-physics-domino/evidence/real-client-manifest.md`
- [x] T002a Audit existing physics-adjacent implementation and record confirmed gaps in `specs/214-mop-physics-domino/evidence/current-implementation-audit.md`

## Phase 2 — Foundational evidence gate

- [x] T003 Enumerate callers of `FUN_00c29680` and record source-path/header/line attribution only in `specs/214-mop-physics-domino/evidence/domino-caller-map.md`
- [x] T004 Trace the `Physics.cpp` / `PhysData.cpp` adapter boundary without decoding Domino algorithms in `specs/214-mop-physics-domino/evidence/physics-adapter-contract.md`
- [x] T005 Trace model-sidecar discovery from the confirmed adapter and record the actual model association plus malformed-data behaviour in `specs/214-mop-physics-domino/evidence/physics-adapter-contract.md`

## Phase 3 — User Story 1: address-cited physics contract (P1)

**Goal**: Publish a reviewable data and observable-behaviour contract without reproducing Domino.

**Independent test**: A reviewer can select any structural claim, follow its cited address, and verify
that the contract labels it as measured or inferred.

- [ ] T006 [US1] Consolidate verified layouts, data flow, update observations, adapter boundary, and explicit unknowns into `specs/214-mop-physics-domino/evidence/physics-contract.md`
- [x] T007 [P] [US1] Evaluate exact BepuPhysics v2 and Jitter2 versions for license, deterministic stepping, collision scope, and cloth route in `specs/214-mop-physics-domino/evidence/solver-selection.md`
- [ ] T008 [US1] Review `specs/214-mop-physics-domino/evidence/physics-contract.md` against FR-001 through FR-005 and record the Phase 0 pass/block decision in `specs/214-mop-physics-domino/tasks.md`

## Phase 4 — User Story 7: era gating, provenance, and admission policy (P1, complete)

**Goal**: Ensure physics is enabled only by measured profile, disabled for alpha, and unknown for
unrecognized builds.

**Independent test**: Unit tests verify disabled 0.5.3, enabled exact-build 5.0.1.15464, fail-closed
unknown builds, validated budgets/candidates, deterministic priority/distance/stable-id ordering,
distance culling, capacity deferral, and one explicit provenance-carrying result per candidate.

- [x] T009 [US7] Implement exact-build era resolution, evidence provenance, validated budgets, and deterministic explicit admission outcomes in `src/core/WowViewer.Core.Runtime/World/Physics/PhysicsRuntimePolicy.cs`
- [x] T010 [US7] Add focused policy coverage in `tests/WowViewer.Core.Tests/PhysicsRuntimePolicyTests.cs`
- [x] T011 [US7] Validate 16/16 focused tests and a zero-error `WowViewer.Core.Runtime` Debug build; full Core scope gate: 1,379 passed, 1 skipped, and the same 9 unrelated baseline failures; existing `Snappier` NU1903 warnings remain unrelated

## Phase 5 — Format and solver implementation gate (blocked)

- [ ] T012 Generate sidecar-parser and solver-adapter implementation tasks only after T005 and T007 are marked complete in `specs/214-mop-physics-domino/tasks.md`

## Parallel opportunities

- T001 and T002 are independent environment/provenance records.
- After T005, T006 (contract consolidation) and T007 (license/capability evaluation) can proceed in
  parallel because they modify distinct evidence files.

## Implementation strategy

The delivered first slice is the format- and solver-independent US7 policy boundary. The next minimum
delivery remains the Phase 0 evidence pack: caller map, adapter/sidecar boundary, contract,
exact-version solver selection, and real-client manifest. Parser, package, simulation, cloth, joint,
and viewer tasks must be generated only after that pack passes review. This preserves the evidence,
era, legal, and cloth constraints instead of prematurely locking an API around guessed physics data.
