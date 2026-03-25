# v0.4.5 Release Stabilization Prompt

Use this prompt in a fresh planning chat when the goal is to define what must be fixed, validated, or explicitly deferred before a `v0.4.5` GitHub release of `parp-tools WoW Viewer` can reasonably be called stable enough.

## Prompt

Design a concrete release-stabilization plan for `gillijimproject_refactor/src/MdxViewer` targeting `v0.4.5`.

The plan must separate:

- release blockers
- strong should-fix issues
- acceptable post-release follow-ups

The plan must assume the active tree already has meaningful in-progress work for taxi visualization, minimap interaction, WMO baked-light prototyping, and render-quality controls, but most of that work has only build validation rather than runtime signoff.

## The Plan Must Produce

1. A short release-blocker list.
2. A prioritized stabilization backlog.
3. A validation checklist for the release candidate.
4. A deferred-items list that clearly moves non-critical work to `v0.5.0`.
5. A release-readiness statement format that does not over-claim.

## Required Current Assumptions

- fullscreen minimap behavior is still reported broken by runtime user feedback and must be treated as unresolved even though an earlier tile-scale patch compiled successfully
- the earlier fullscreen minimap work is only build-validated, not runtime-signed-off
- taxi route actor work is promising but not yet release-critical unless it actively destabilizes the viewer
- enhanced renderer and shader-family work are explicitly `v0.5.0` scope, not `v0.4.5` scope
- branding and workflow packaging for `v0.4.5` already exist in the active tree

## Required Constraints

- Do not define `v0.4.5` so broadly that it becomes a disguised `v0.5.0` milestone.
- Treat runtime user-facing navigation and inspection regressions as higher priority than speculative fidelity work.
- Be explicit about build-only validation versus runtime validation.
- Favor small surgical fixes over broad viewer rewrites.
- Use the fixed real-data paths already documented in the repo.

## Minimum Areas The Plan Must Review

1. fullscreen minimap correctness and interaction feel
2. minimap teleport correctness
3. minimap pan/zoom consistency between docked and fullscreen modes
4. any obvious release-facing UX instability introduced by recent viewer-side interaction slices
5. packaging/readme/release-note honesty around what is and is not validated

## Suggested Deliverable Structure

1. Release scope boundary
2. Release blockers
3. Should-fix stabilization items
4. Explicit `v0.5.0` deferrals
5. Validation checklist
6. Release-readiness language

## Validation Rules

- If the plan says something is release-ready, define the runtime check that makes that statement defensible.
- If no automated tests exist for a seam, say that explicitly.
- If a fix only builds, say that explicitly.
- Do not describe fullscreen minimap as fixed without runtime confirmation on the real minimap dataset.

## Fixed Data Reminder

Use the existing fixed paths, especially:

- `test_data/development/World/Maps/development`
- `test_data/minimaps/development`

Do not ask for alternate paths unless those are genuinely missing.