# Physics Runtime Contract

## Input boundary

The current policy boundary accepts an explicit client build identity for era resolution, then an era
decision, validated `PhysicsBudget`, and solver-independent candidates with unique stable identities,
distance, and priority. Future asset/simulation boundaries also require model identity. Wind remains
a future value supplied by the weather owner; this runtime does not generate weather.

## Required result boundary

Every resolution and scheduling result returns an activation status (`enabled`, `known-disabled`, or
`unknown-build`), profile identifier and evidence source, explicit diagnostics, and—for eligible
scheduling—a decision (`active`, `distance-culled`, or `budget-deferred`) with reason.

No API silently promotes an unknown build, drops an unsupported construct, or discards an eligible
object without a reason.

Candidate results are deterministic by ascending priority, distance, and ordinal stable identity,
while the returned list preserves original input order. Missing/invalid evidence, invalid candidates,
duplicate stable identities, and unknown enum values fail closed.

## Safety boundary

The API contains only independently authored types, decoded file-data declarations, and observable
behaviour policy. It does not expose Domino source, algorithm descriptions, or proprietary data
structures. Client files remain read-only inputs.
