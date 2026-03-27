# Fullscreen Minimap Repair Prompt

Use this prompt only if the fullscreen minimap regresses again after the Mar 25, 2026 repair.

Mar 25 resolution note: the original `v0.4.5` blocker was closed after the final transpose-only repair and runtime user confirmation on the fixed development minimap dataset. Keep this prompt as regression archaeology and a fallback investigation checklist, not as a statement that the bug is still open.

## Prompt

Design a concrete investigation-and-fix plan for a renewed fullscreen minimap regression in `gillijimproject_refactor/src/MdxViewer`.

The plan must assume the original bug was eventually fixed by reverting the bad `WoWConstants.TileSize` hypothesis, backing out the later broad world-axis swap, and keeping only the narrower marker/grid transpose repair. Treat those facts as the last known-good baseline unless new runtime evidence proves otherwise.

## The Plan Must Produce

1. A likely root-cause shortlist.
2. A file-by-file investigation order.
3. A minimal fix strategy.
4. A runtime validation checklist.
5. A rollback/debugging strategy if the first hypothesis is wrong.

## Required Investigation Areas

The plan must inspect and reason about these specific seams:

- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- `DrawMinimapWindow()`
	- `DrawFullscreenMinimap()`
	- `HandleMinimapInteraction(...)`
	- `TryGetMinimapClickTarget(...)`
	- `RegisterMinimapTeleportClick(...)`
- `src/MdxViewer/MinimapHelpers.cs`
	- camera-marker projection
	- tile projection
	- POI/taxi overlay projection
	- row/column orientation
- `src/MdxViewer/Rendering/MinimapRenderer.cs`
	- tile file selection and coordinate ordering
	- `GetTileTexture(mapName, ty, tx)` call-site assumptions

## High-Priority Hypotheses To Evaluate

1. Tile lookup/display ordering mismatch:
	- tile storage, tile drawing, and tile texture lookup may no longer agree on whether `(tx, ty)` means `(row, column)` or `(x, y)`.
2. Fullscreen-only interaction drift:
	- fullscreen minimap may share rendering helpers with the docked minimap but still differ in input hit-testing or panning behavior enough to look broken.
3. Camera marker versus tile content disagreement:
	- the marker may again be mathematically consistent with one coordinate system while the tile imagery is being fetched or arranged in another.
4. Teleport/click mapping drift:
	- click-to-tile and tile-to-world conversion may have regressed away from the direct world-axis mapping restored in the final fix.
5. Regression from reintroducing the wrong scale:
	- a later patch may have reintroduced `WoWConstants.TileSize` or another mismatched spacing assumption into one of the minimap paths.

## Required Constraints

- Do not change terrain decode or world-tile loading code as part of this minimap repair unless the planner can prove the minimap bug originates there.
- Prefer one or two small fixes over a wholesale minimap rewrite.
- Keep docked minimap and fullscreen minimap behavior aligned through shared logic where practical.
- Treat this as a `v0.4.5` release blocker.

## Expected Deliverables

1. Root-cause hypotheses ranked by likelihood.
2. Concrete instrumentation or debug-observation steps.
3. Minimal file-by-file fix scope.
4. Runtime validation steps using the fixed minimap and terrain paths.
5. Release-blocker acceptance criteria.

## Validation Rules

- Build success alone does not close this issue.
- Require runtime confirmation that:
	- the camera marker stays inside the valid `64x64` map space
	- tile imagery and marker position agree
	- click/teleport lands on the expected tile
	- docked and fullscreen minimap behave consistently enough to trust both
- If no automated tests are added, say that explicitly.