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

## Decision 6: Treat 3.0.1.8303 prototype `MD20`/`.mdx` behavior as a deferred build-profile boundary

- **Decision**: Record staged `3.0.1.8303` `.mdx` load failures as a separate research boundary and do not force them through the current `3.3.5.12340` parity assumptions.
- **Rationale**: Current logs suggest some `3.0.1.8303` `.mdx` assets may actually be prototype `MD20` / `Model2` family files with different bootstrap expectations, and a wrong assumption here would contaminate cross-build renderer work and downstream dataset capture.
- **Alternatives considered**:
  - Treat all `3.0.1.8303` `.mdx` assets as normal classic MDX: rejected because current converter and skin-fallback failures do not match a healthy classic MDX path.
  - Treat all `3.0.1.8303` `.mdx` assets as normal later-Wrath M2 files: rejected because numbered `.skin` and current runtime assumptions are not proven for this build.

## Decision 7: Unify 3.3.5 `M2CompQuaternion` payload reads across runtime and converter paths

- **Decision**: Read staged `3.3.5.12340` `M2CompQuaternion` payloads in direct little-endian component order (`x, y, z, w`) and centralize that read path so the runtime sampler and M2-to-MDX converter cannot drift.
- **Rationale**: The native runtime path had a local swizzle (`y, -x, z, w`) that did not match the converter-backed compatibility route. That kind of decode mismatch is a plausible direct cause of the observed wrong-axis rotation bug in animated world doodads.
- **Alternatives considered**:
  - Preserve the runtime-only swizzle and compensate later in bone composition: rejected because it leaves two incompatible interpretations of the same on-disk payload.
  - Change only the converter path: rejected because the current visible regression is on the native runtime route, not the restored compatibility route.

## Decision 8: Prefer `.skin` bone-entry remap over raw M2 vertex bone indices in the runtime path

- **Decision**: Build runtime render vertices from the `.skin` bone-entry table when available, instead of assuming raw `M2Vertex.bone_indices` are already the final skinnable indices.
- **Rationale**: wowdev/native notes describe the skin profile as the active owner of submesh-local bone remap through `boneComboIndex`, and the runtime had been parsing `skin.BoneEntries` without ever applying them. That makes section motion coupling bugs likely on animated doodads with distinct visual submeshes.
- **Alternatives considered**:
  - Continue using raw M2 vertex bone indices only: rejected because it ignores a parsed skin-owned remap seam and does not match the stronger native interpretation.
  - Flatten skin bone entries into geometry at parse time: rejected for now because it hides a runtime ownership boundary we still need visible while recovering parity.
