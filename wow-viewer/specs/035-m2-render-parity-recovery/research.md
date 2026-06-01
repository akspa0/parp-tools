# Research: M2 Render Parity Recovery

## Decision 1: Treat 3.3.5 world parity as the first closure target

- **Decision**: Scope the first recovery slice to staged `3.3.5.12340` world rendering, with explicit parity samples for tree-heavy and transparent-material assets.
- **Rationale**: The reported regression is blocking day-to-day world validation now. Narrowing to one build baseline prevents spreading fixes across incompatible profile semantics too early.
- **Alternatives considered**:
  - Multi-build simultaneous parity closure: rejected for this slice due to higher diagnostic ambiguity.
  - Runtime-only closure first: rejected because compatibility route still owns most stable behavior today.

## Decision 2: Introduce a deterministic M2 route contract

- **Decision**: Add an explicit route-decision contract for world M2 handling: primary route, fallback route, skin source, and fallback reason.
- **Rationale**: Current behavior can drift between adapter/runtime/converter paths without a single enforceable boundary, which caused repeated regressions.
- **Alternatives considered**:
  - Keep implicit route selection in scattered call sites: rejected; hard to audit and validate.
  - Hard-disable all fallback paths: rejected; would break unknown model families.

## Decision 3: Separate material-pass classification from draw submission

- **Decision**: Standardize world M2 material pass classification (opaque, cutout, blended) before draw submission and expose that state in probes.
- **Rationale**: Trees and transparent assets fail primarily when layer semantics and pass routing diverge. Making classification explicit isolates the failure seam.
- **Alternatives considered**:
  - Keep per-renderer ad-hoc pass heuristics: rejected; difficult to reason about cross-path parity.
  - Push all pass semantics into shader-only behavior: rejected; routing errors happen before shader execution.

## Decision 4: Establish parity sample evidence as release gate

- **Decision**: Require a fixed parity sample set with headless probe evidence and runtime validation logs before route changes are accepted.
- **Rationale**: Prevents “looks fixed on one model” regressions and supports reproducible debugging.
- **Alternatives considered**:
  - Manual screenshot-only QA: rejected; too slow and inconsistent.
  - Unit tests only: rejected; cannot prove real staged-client behavior alone.

## Decision 5: Keep ownership in wow-viewer runtime/core surfaces

- **Decision**: Land route and parity logic in `wow-viewer` viewer/runtime seams, with `gillijimproject_refactor` as reference only.
- **Rationale**: Matches repo-independence and migration goals.
- **Alternatives considered**:
  - Re-open legacy viewer as design owner: rejected by workspace policy and prior drift history.
