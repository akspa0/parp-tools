# Requirements Quality Checklist: Minimap, Fog, and Doodad Instancing

## Content

- [x] No implementation language or framework is required to understand the user behavior.
- [x] Full-screen minimap drag behavior is independently testable.
- [x] Triple-click teleport behavior and reset rules are explicit.
- [x] Fog-bounded tile/object residency is separated from capture/full-load exceptions.
- [x] Doodad batching preserves correctness fallbacks for incompatible materials/effects.
- [x] Diagnostics distinguish residency, visibility, batching, fallback, and draw work.
- [x] Existing readers and Alpha/Standard terrain ownership are preserved.
- [x] Real-client visual/FPS proof is explicitly user-owned.

## Completeness

- [x] User stories are prioritized and independently testable.
- [x] Acceptance scenarios cover success and failure/reset paths.
- [x] Edge cases cover map bounds, fog changes, WMO containment, and unsupported doodads.
- [x] Functional requirements are numbered and testable.
- [x] Success criteria are measurable without promising unperformed runtime proof.
- [x] Assumptions and out-of-scope work are stated.

## Scope

- [x] The feature composes Specs 136, 137, and 142 without silently replacing them.
- [x] The request does not expand into audio, shader, WDL, or camera-path work.
- [x] No proprietary client data or machine-local path is embedded.
