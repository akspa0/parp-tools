# Contract: PM4 Region Navigation

## Producer

`WorldScene` is the producer. It aggregates the existing resident `_pm4TileObjects` and exposes a
deterministic list of `Pm4RegionNavigationItem` values plus selection/focus data. It does not read WMO,
M2, or saved correlation records to build the list.

## Consumer

`ViewerApp_Pm4Utilities.cs` renders the rows. A normal click selects a row; a double-click requests a
focus operation through the viewer camera/residency owner. The UI must not mutate PM4 transforms or
construct an asset-match candidate.

## Invariants

1. One row per non-empty `RegionId` in the current resident PM4 snapshot.
2. Region totals are calculated from decoded object/surface records.
3. A focus request is emitted only for finite, available bounds.
4. Focusing a region does not force whole-map residency.
5. Selection is cleared when the source snapshot no longer contains the region.
6. Tooltip and workbench presentation data contain no match/correlation fields.

## Failure contract

`Unavailable`, `Pending`, and `Stale` states remain visible with an actionable message. A missing or
malformed source never produces a guessed coordinate or an external asset association.
