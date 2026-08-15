# Portal Visibility Contract

`WmoPortalVisibilityDecision` is a bounded, fail-open decision for one WMO placement and one camera
frame.

## Inputs

- Decoded WMO group bounds and flags.
- Decoded portal vertices, portal plane, and group/reference side data when available.
- Placement transform and camera position/view/projection.
- Configured traversal depth and scratch capacity.

## Required behavior

1. If transforms, bounds, portal geometry, or side data are invalid, return the conservative visible
   set with a stable fallback reason.
2. If the camera is in a known group, begin traversal there and carry a clipped portal view volume.
3. A destination group may be admitted only if its bounds intersect the carried volume or if the
   conservative fallback is selected.
4. Use visited state and a hard depth/capacity bound; cycles must terminate.
5. Use the resulting group set consistently for group geometry, doodads, and liquid. A pass that
   cannot honor it must report that it used a conservative fallback.

## Outputs

- Visible group indices.
- Source group, tested/rejected portal counts, depth, clip-use flag, and fallback reason.
- Counters integrated into the existing WMO visibility/submission diagnostics.
