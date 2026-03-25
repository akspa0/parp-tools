# Fullscreen Minimap Repair Prompt

Use this prompt in a fresh planning chat when the goal is to diagnose and fix the still-broken fullscreen minimap in `MdxViewer`.

## Prompt

Design a concrete investigation-and-fix plan for the fullscreen minimap in `gillijimproject_refactor/src/MdxViewer`.

The plan must assume an earlier patch changed the minimap math from `WoWConstants.ChunkSize` to `WoWConstants.TileSize`, but runtime user feedback still says the fullscreen minimap is broken. Treat the earlier patch as a partial hypothesis, not as a confirmed fix.

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

1. Axis swap mismatch:
	- camera world-to-tile conversion may still disagree with the rest of the viewer's world-coordinate conventions.
2. Row/column mismatch:
	- tile storage, tile drawing, and tile texture lookup may not agree on whether `(tx, ty)` means `(row, column)` or `(x, y)`.
3. Fullscreen-only interaction drift:
	- fullscreen minimap may share rendering helpers with the docked minimap but still differ in input hit-testing or panning behavior enough to look broken.
4. Camera marker versus tile content disagreement:
	- the marker may be mathematically consistent with one coordinate system while the tile imagery is being fetched or arranged in another.
5. Teleport/click mapping drift:
	- click-to-tile and tile-to-world conversion may still be transposed even after the scale fix.

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