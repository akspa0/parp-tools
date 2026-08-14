# Contract: Minimap Interaction

The UI adapter supplies pointer events and map-space conversion to a pure interaction decision
surface. Docked and full-screen minimaps use the same contract.

## Inputs

- `SurfaceId`
- pointer position and button transition (`Down`, `Move`, `Up`)
- map rectangle and current pan/zoom transform
- valid map bounds
- current UTC/monotonic time

## Decisions

The interaction state returns one or more of:

- `PanStarted`
- `PanUpdated(deltaMapSpace)`
- `PanCompleted`
- `TeleportArmed(target, count)`
- `TeleportExecuted(target)`
- `TeleportReset(reason)`
- `IgnoredOutsideMap`

## Rules

1. Pointer movement beyond the click threshold changes the gesture to a drag.
2. While dragging, map pan is updated continuously and click confirmation is not updated.
3. On release after a drag, the pending teleport sequence is reset.
4. On release without a drag, a valid target advances the same-target sequence.
5. A different target or expired confirmation window starts a new sequence at `1/3`.
6. The third same-target click emits exactly one `TeleportExecuted` decision.
7. The UI adapter performs the existing world-coordinate conversion and camera update only for
   `TeleportExecuted`.

## Required tests

- drag updates pan and never emits teleport;
- click one/two arm and advance, but do not teleport;
- click three executes once;
- target change and timeout reset;
- invalid/out-of-bounds target is ignored;
- fullscreen and docked adapters feed equivalent events.
