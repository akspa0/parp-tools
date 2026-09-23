# Contract: Dynamic Point Light

Shape of one active dynamic light, as uploaded to the renderer each frame.

```json
{
  "sourceId": "attachment:m2:1234:hand",
  "position": { "x": 100.2, "y": 55.0, "z": 13.5 },
  "color": { "r": 1.0, "g": 0.55, "b": 0.15 },
  "radius": 8.0
}
```

**Contract rules**:

- `sourceId` uniquely identifies what this light is attached to. Removing that source removes exactly
  this entry from the active-light list the next frame — FR-015's "no residual glow" is a structural
  property of this list (a light not present is a light not contributing), not a runtime flag any light
  entry carries.
- The active-light list is bounded (research.md §7) and is expected to silently drop the least-relevant
  entries beyond that bound (e.g. farthest from camera) rather than error — a torch far off-screen not
  currently affecting anything visible is expected to not consume a slot, per spec.md's own edge case
  ("does not need to be computed at a cost that scales with total scene size").
- This contract describes the light itself, not how it combines with existing static LIT lighting in the
  shader — that combination is additive per FR-014, verified in Phase 5b's tests, not part of this data
  shape.
