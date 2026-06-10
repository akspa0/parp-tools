# v0.4.5 Release Stabilization Prompt

Use this prompt in a fresh planning chat when the goal is to define the final release-readiness pass before a `v0.4.5` GitHub release of `parp-tools WoW Viewer` is cut.

Mar 25 update: the fullscreen minimap blocker was closed by runtime user confirmation after the final transpose-only repair. This prompt should now treat minimap work as recently stabilized, not as the primary open blocker.

## Prompt

Design a concrete final release-readiness plan for `gillijimproject_refactor/src/MdxViewer` targeting `v0.4.5`.

The plan must separate:

- release blockers
- strong should-fix issues
- acceptable post-release follow-ups

The plan must assume the active tree already has meaningful in-progress work for taxi visualization, minimap interaction, WMO baked-light prototyping, and render-quality controls, and that only some of that work has runtime signoff.

## The Plan Must Produce

1. A short release-blocker list.
2. A prioritized stabilization backlog.
3. A validation checklist for the release candidate.
4. A deferred-items list that clearly moves non-critical work to `v0.5.0`.
5. A release-readiness statement format that does not over-claim.

## Required Current Assumptions

- fullscreen minimap behavior has user-confirmed runtime signoff for the previously broken top-right Designer Island scenario after the final transpose-only repair
- minimap work should now be reviewed for release-note honesty and regression risk, not treated as the main unresolved blocker by default
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

1. regression spot-check of fullscreen and docked minimap behavior after the repair
2. minimap teleport correctness and interaction-feel wording in release notes and docs
3. any obvious release-facing UX instability introduced by recent viewer-side interaction slices
4. packaging/readme/release-note honesty around what is and is not validated
5. final scope boundary between `v0.4.5` and deferred `v0.5.0` renderer work

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
- Do not extend the minimap signoff beyond the runtime-confirmed real-data scenario that was actually checked.

## Fixed Data Reminder

Use the existing fixed paths, especially:

- `test_data/development/World/Maps/development`
- `test_data/minimaps/development`

Do not ask for alternate paths unless those are genuinely missing.