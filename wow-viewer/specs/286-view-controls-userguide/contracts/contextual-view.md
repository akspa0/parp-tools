# Local interaction contracts — Spec 286

Desktop callbacks/state only; no REST/GraphQL endpoints are appropriate.

## View/reset
- View and Settings mutate one finite user-scale preference. UI manager owns display composition.
- Layout-only delegates existing reset; scale-only restores 1.0; combined explicitly performs both.
  None unloads a client/map. Existing visual groups stay within the one-submenu requirement.
- Register changed routes in [control inventory](../../../eng/ui/ui-controls.json), not new settings copies.

## Activation/Maps
- Session commits an activation identity, then its current discovery outcome; consume once to open
  existing Maps, including honest empty state. Failure/cancel does not emit success.
- Stale completion cannot display old data as new. Refresh/user close do not create an activation.

## Footer/disclosure
- Both bottom regions prohibit scrolling and fit essential controls/FPS to actual client geometry.
- FPS hover reads existing diagnostic snapshot, including pending count, inside client bounds.
  Interactive long detail uses the existing popup owner. No disk/DBC/model work on hover.
- Tooltip/popup input blocks scene actions where appropriate; no extra telemetry pipeline.

## Selected identity/type colors
- Existing selection/hover owners supply data; presentation neither repicks nor creates scene state.
- Larger bright/bold identity bar is top-middle of scene viewport; honest identifier fallback,
  never HP/progress. Clearing selection/client replacement clears it.
- One category resolver/palette feeds header/tooltip/enabled bounds. Known trees share a category;
  unknown semantic evidence remains Unknown. Do not infer tree semantics from M2/WMO family alone.
- Preserve bounds toggles/culling and selected emphasis; category styling does not add hidden draws.

## Proof
- Tests support pure layout/state policy. Real input/layout/screenshots require operator receipts
  with version, client/build when relevant, display/scale and criterion→real-output mapping.
