# Lighting Profile and Capture Contract

## Client-exact eligibility

A profile may claim `client_exact` only when all conditions hold:

1. `build_identity`, direction model revision, and calibrated coordinate-transform revision are present.
2. Exactly one exact-build color source is selected: global-clear LIT **or** resolved Light* DBC records.
3. Native light ray and viewer direction are finite normalized vectors.
4. MCSH and dynamic-shadow evidence states are explicit and remain distinct.
5. Capture sidecar binds renderer contract, source and output hashes, camera orientation, time, and all lighting revisions.

Otherwise the profile is `partially_proven` or `authored`; Capture must reject an attempted `client_exact` output.

## Direction semantics

`native_light_ray` travels from the source toward the surface. `viewer_light_direction` is the normalized vector expected by the viewer's Lambert calculation. The profile declares whether the viewer expects ray or source semantics and applies a single signed axis permutation only through the calibrated transform.

## Calibration artifact

Each transform revision must name:

- exact client build and renderer revision;
- one lock-time comparison and two held-out-time comparisons;
- map/tile, camera/framing, normalized time, source IDs, image hashes, and result;
- explicit decision: `calibrated` or `rejected`.

No per-time transform override is permitted.

## Dataset contract

The store builder rejects rows that lack complete lighting provenance, mix LIT and DBC sources, name an uncalibrated client-exact profile, relight captured RGB, or split a `source_group_id` across partitions.
