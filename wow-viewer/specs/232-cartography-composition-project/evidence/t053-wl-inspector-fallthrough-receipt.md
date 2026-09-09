# Receipt — Spec 232 T053: WL inspector fall-through repair

**Date**: 2026-09-08 · **Spec**: [232](../spec.md) · **Task**: T053

## Audit result

The minimap footprint path is not on the WL viewport-selection path: its click handler only
arms footprint dragging, panning, LIT selection, or minimap teleport. `GetLayerFootprints` is
only consumed for minimap rendering/footprint hit-tests. Neither route creates or clears a WL
selection.

The viewport path is `UpdateWorldSceneHoveredAssetInfo` → `TryHandleSceneClickSelection`.
`TryHandleSceneClickSelection` creates a WL candidate solely from the current
`HoveredAssetInfo`. The terrain-occlusion guard added in `c75a16669` cleared every precise hover
whose bounds were behind the terrain ray hit. Once composition supplies a placed layer, its
terrain can be in front of the original WL bounds, which clears the WL identity before the click
handler gets a chance to open the data inspector.

## Files changed

| File | Change |
|---|---|
| [ViewerApp.cs](../../../src/viewer/WoWViewer/ViewerApp.cs) | Keeps the terrain-occlusion guard for physical scene selections while exempting the `WL liquid` source-data inspection target, so a composed terrain layer cannot erase the hover identity consumed by the WL click inspector. |
| [tasks.md](../tasks.md), [activeContext.md](../../../memory-bank/activeContext.md), [progress.md](../../../memory-bank/progress.md) | Records the audited route, repair, and outstanding viewport witness. |

## Verification

| Command | Exit | Real output |
|---|---:|---|
| `dotnet build WowViewer.slnx -c Debug --no-restore` | 0 | **0 errors**; 543 pre-existing warnings, including NU1903 advisories. |
| `git diff --check` | 0 | No whitespace errors. Git emitted unrelated user-config access and CRLF notices. |

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| Map-layer minimap handling does not consume a WL viewport click | Audit traces minimap handling to drag/pan/LIT/teleport only; WL candidate creation occurs separately in the viewport click path. | Pass (source audit) |
| Placed terrain cannot clear a WL data-inspection hover before click selection | `UpdateWorldSceneHoveredAssetInfo` now exempts only `WL liquid`; the adjacent terrain occlusion rule remains for physical scene assets. | Implemented (solution build) |
| WL inspector works identically with and without an active layer stack | Requires an operator viewport witness with actual WL data and a composed layer; no synthetic or compile-only claim is sufficient. | Pending operator witness |

## Remaining operator witness

Open a map that has WL liquid data, confirm clicking a visible WL body opens its inspector, then
enable and place a phase layer that puts composed terrain at the same target. Click the same WL
body again and capture its inspector identity/source path. The second click must still open the
same WL inspector. T053 remains unchecked until that witness is recorded.
