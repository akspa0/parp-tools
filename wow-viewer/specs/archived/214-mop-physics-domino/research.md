# Research: 5.0.1 Physics Contract and Solver Selection

## Decision: keep Phase 0 evidence-only

**Rationale:** The binary reconnaissance proves a complete 5.0.1 subsystem and a separate
`Physics`/`PhysData` adapter, but does not establish the model-sidecar path, record layout, or update
order. Writing a reader from the M2 `0x20` flag alone would violate the evidence gate.

**Alternatives considered:** Guessing a `.phys` sibling or copying a community layout was rejected:
both would present an assumption as a measured format claim.

## Decision: keep the solver unselected until exact license and cloth route are verified

**Rationale:** The operator requires an existing permissively licensed C# solver, and flag/cloth
motion is the primary visible outcome. A rigid-only selection that discovers missing cloth later
would strand US4.

**Candidates:** BepuPhysics v2 and Jitter2. Native bindings are secondary because they complicate
cross-platform deployment. The evaluation must cite exact version and license text; this file makes
no unverified licensing claim.

## Decision: enforce era gating before data parsing or simulation

**Rationale:** The 5.0.1 evidence explicitly does not generalize to 0.5.3. The existing M2 flag is a
local format observation, not authorization to activate physics for every era.

**Alternatives considered:** Defaulting unknown builds to 5.0.1 was rejected because it recreates
the documented era-blind decoder failures.

## Decision: Core owns policy and data contracts

**Rationale:** The test suite cannot reference the viewer. Capability resolution, diagnostics,
fixed-step policy, and culling selection can be complete and unit-tested without OpenGL.

**Outstanding Phase 0 evidence:** read-only caller map, verified sidecar discovery, exact solver
license/cloth evaluation, and a real-client asset manifest with build fingerprint.
