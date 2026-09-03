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

- [x] T006 [US1] Consolidate verified layouts, data flow, update observations, adapter boundary, and explicit unknowns into `specs/214-mop-physics-domino/evidence/physics-contract.md`
- [x] T007 [P] [US1] Evaluate exact BepuPhysics v2 and Jitter2 versions for license, deterministic stepping, collision scope, and cloth route in `specs/214-mop-physics-domino/evidence/solver-selection.md`
- [x] T008 [US1] Review `specs/214-mop-physics-domino/evidence/physics-contract.md` against FR-001 through FR-005 and record the Phase 0 pass/block decision in `specs/214-mop-physics-domino/tasks.md`

## Phase 4 — User Story 7: era gating, provenance, and admission policy (P1, complete)

**Goal**: Ensure physics is enabled only by measured profile, disabled for alpha, and unknown for
unrecognized builds.

**Independent test**: Unit tests verify disabled 0.5.3, enabled exact-build 5.0.1.15464, fail-closed
unknown builds, validated budgets/candidates, deterministic priority/distance/stable-id ordering,
distance culling, capacity deferral, and one explicit provenance-carrying result per candidate.

- [x] T009 [US7] Implement exact-build era resolution, evidence provenance, validated budgets, and deterministic explicit admission outcomes in `src/core/WowViewer.Core.Runtime/World/Physics/PhysicsRuntimePolicy.cs`
- [x] T010 [US7] Add focused policy coverage in `tests/WowViewer.Core.Tests/PhysicsRuntimePolicyTests.cs`
- [x] T011 [US7] Validate 16/16 focused tests and a zero-error `WowViewer.Core.Runtime` Debug build; full Core scope gate: 1,379 passed, 1 skipped, and the same 9 unrelated baseline failures; existing `Snappier` NU1903 warnings remain unrelated

## T008 — Phase 0 review against FR-001..FR-005

**Verdict: PASS, with one partial recorded rather than waived.**

| Req | Verdict | Basis |
|---|---|---|
| **FR-001** structural claims cite an address | **PASS** | Every layout, offset, stride and update-order claim in [physics-contract.md](evidence/physics-contract.md) carries a function or global address |
| **FR-002** measured vs inferred; name for what it is | **PASS** | Every claim is labelled **M** or **I**. Array names come from Blizzard's own `PhysData.h` asserts. `BOXS` `0..47` is explicitly left **I, unverified** instead of being filled in from the community layout |
| **FR-003** state the adapter/Domino boundary | **PASS** | §1, with the `dmMath.h` string-region evidence for where the boundary actually sits |
| **FR-004** read-only Ghidra session | **PASS** | Read-only; no exception taken, and that is recorded in the contract header |
| **FR-005** layouts/behaviour only, no algorithms | **PASS** | No Domino function was decompiled for the caller map. The large-motion blend arithmetic was **deliberately not reproduced** even though it was visible |

**SC-002 is PARTIAL and this is the honest finding of the review.** It asks that every function
reached from the assertion handler be attributed to a source header **and line**, with the count
reported against the ~90 floor. The count *is* reported (58 named, with the delta explained). Headers
*are* attributed. **Line numbers are not**, for Domino internals — the header-string pivot recovers
the file but not the line, and only the adapter-side lines (`dmMath.h:127`, the `PhysData.h` and
`Physics.cpp` lines) were obtained.

**Recommendation: accept the partial rather than close it.** Line-level attribution of Domino
internals has **no consumer** — we never reproduce those algorithms, so the line a solver-internal
assert fires on has no downstream use. Closing it costs ~90 decompilations for information the
project has committed never to act on. If the operator disagrees, the work is well-defined and can be
commissioned as its own task; it is a cost decision, not a blocker.

**Gate decision: Phase 0 PASSES.** Phase 2 (parsing) and Phase 3 (solver adapter) may begin.
**T002 remains open** and gates real-byte validation, not implementation.

## Phase 5 — Format and solver implementation tasks (T012 output)

- [x] T012 Generate sidecar-parser and solver-adapter implementation tasks only after T005 and T007 are marked complete in `specs/214-mop-physics-domino/tasks.md`

**Generated 2026-09-03.** Ordering rule: **the reader lands before the package reference.** The
parser is solver-independent and testable on its own, so it carries no dependency risk; adding
Jitter2 is a permanent commitment and waits for operator go-ahead.

### Phase 2 — `.phys` reader (US2/US3, solver-independent, may start now)

- [ ] T013 Add `PhysSidecarPath` in Core.IO: model path → `.phys` by extension replacement, matching `FUN_005a29a0`. Must be pure and testable with no file access
- [ ] T014 Add `PhysChunkReader` in Core.IO: reversed-tag chunk walk, `PHYS` magic, **version `u16` must be 0**, `8 + size` advance, and **unknown tags skipped by size** (never rejected — the client is permissive here and a stricter reader breaks on later eras)
- [ ] T015 Add record readers for `BOXS`/`CAPS`/`SPHS`/`SHAP`/`BODY`/`JOIN` at the measured strides (60/28/16/20/28/16). Leave `BOXS` `0..47`, `SHAP` `+4`/`+8`/`+12`/`+16` and `JOIN` `+8` as **raw preserved bytes**, not interpreted fields — they are unverified
- [ ] T016 Add `SPHJ`/`SHOJ`/`WELJ` as **length-validated opaque records** (28/108/104). Per-field semantics are undecoded; preserve bytes and report presence rather than inventing a layout
- [ ] T017 Add `PhysDocument` + diagnostics: one explicit diagnostic per malformed, unsupported, or skipped construct (FR-010/FR-012). **Match the client's fail-closed fallback, not its silence**
- [ ] T018 [P] Add focused tests in `WowViewer.Core.Tests`: bad magic, version ≠ 0, truncated chunk, **unknown tag skipped**, absent chunk, `0xFFFF` bone sentinel preserved, and a stride-mismatch case. Synthetic bytes only — these test safety, not fidelity
- [ ] T019 Wire `M2ModelDocument.HasPhysicsSidecar` to the real resolver, replacing the unconsumed metadata identified in [current-implementation-audit.md](evidence/current-implementation-audit.md)
- [ ] T020 Add a thin `inspect model phys` command reporting bodies, shapes, joints, provenance and diagnostics

**Gate**: models without a sidecar render exactly as today; malformed data fails closed **with a
diagnostic**; no construct is silently dropped. Fidelity against real assets is **T002's** gate, not
this one.

### Phase 3 — Jitter2 adapter (US5, needs operator go-ahead for the package reference)

- [ ] T021 **Operator decision**: approve the `Jitter2 2.8.10` package reference, per [solver-selection.md](evidence/solver-selection.md). Nothing below starts first
- [ ] T022 Add a Core-owned solver facade so no viewer or test project takes a direct Jitter2 dependency
- [ ] T023 Map `BOXS`/`CAPS`/`SPHS` onto Jitter2 shapes; report any shape the reader saw but the adapter cannot build
- [ ] T024 Implement the adapter obligations that are ours regardless of solver: **gravity `(0,0,-10.0)`**, pinned/restored FP state, build order **bodies → shapes → joints**, **teleport on first update**, `0xFFFF` skipped, and kinematic driving as `(current − previous) × 1/dt`
- [ ] T025 Enable Jitter2's deterministic solver mode and add a reproducibility test (SC-005: identical inputs, zero divergence)
- [ ] T026 Add resting/energy/no-tunnelling fixtures (SC-004)
- [ ] T027 Bind `PhysicsAdmissionPolicy` to real instances so cull distance is enforced **at the per-instance update**, matching `FUN_005a40d0` rather than inside the solver

### Deferred, with reasons

- [ ] T028 **Joints.** Only `SPHJ` maps cleanly to a Jitter2 constraint. **`SHOJ` (shoulder) and `WELJ` (weld) have no obvious counterpart** — the open risk from T007. Needs its own investigation; do not assume a mapping
- [ ] T029 **Cloth (US4).** Adapt the ~115-line MIT `SoftBodyCloth` sample onto `SoftBodyTriangle` + `SpringConstraint`, with attribution. Consumes wind as an input contract owned by spec 215
- [ ] T030 **Viewer binding + profiler counters.** After Core gates pass
- [ ] T031 **`BOXS` `0..47`.** Second measurement to confirm or refute the 4×3-transform inference before any reader interprets those bytes

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
