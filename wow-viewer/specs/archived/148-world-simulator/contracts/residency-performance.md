# Contract: Residency and Performance Attribution

## Residency selection

The resident set is the union of independently justified requests:

- actor/fog-visible coverage;
- camera-path swept warmup and hold interval;
- explicit inspection or selected-asset leases;
- mandatory context lease for the WMO/terrain containing the actor.

Selecting a map is not a residency reason. Every resident tile/object must be attributable to at
least one owner, and release must record the reason.

## Per-frame attribution

The renderer records selection/culling, resource preparation, batch preparation, draw submission,
terrain, WMO-internal doodads, and audio work separately. It reports unique model preparation and
instance submissions separately from total draw calls. This permits comparison of terrain MDX and
WMO-internal doodad paths without requiring a generic scene-graph traversal.

## Performance gate

Before an optimization is accepted, a fixed capture records the same map/build, camera/path,
resolution, fog/time settings, and diagnostic counters. User-run FPS/visual proof remains required;
automated tests prove attribution and lease behavior only.
