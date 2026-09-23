# Contract: Doodad Asset and Instance Submission

The submission owner receives visible placements from fog-bounded object admission and groups them
by a compatibility key. The contract is additive to existing renderer interfaces.

## Batch eligibility

A placement may enter a shared static instance batch only when its asset backend, render pass,
material/texture state, fade class, animation/effect requirements, and required WMO context are
compatible with the bucket. Otherwise it uses the existing fallback route with a reason code.

## Submission contract

For each compatible bucket:

1. resolve/load immutable asset geometry once;
2. append placement transforms and permitted instance values;
3. bind shared state once;
4. submit the instance set;
5. report asset, batch, instance, fallback, animation, and draw counts.

Transparent content retains distance ordering. Particle/ribbon/effect-heavy or placement-local
animated content is not merged merely to reduce draw calls. WMO-internal doodads preserve group,
portal, placement, and transform semantics.

## Required tests

- identical static placements group into one compatible batch;
- material/pass/fade/effect mismatches split deterministically;
- unsupported/transparent paths remain fallbacks;
- shared asset geometry is loaded once across tiles;
- releasing one tile does not destroy an asset still referenced by another;
- WMO placement-local transforms and group visibility remain attached to their placement;
- diagnostic counts reconcile placements, batches, fallbacks, and submissions.
