# Game Mode Contract

Game mode is an opt-in shell/runtime feature. It does not replace the editor camera.

## State transitions

- `Editor -> Game`: save the editor pose once, resolve a character anchor, initialize body state.
- `Game -> Editor`: restore the saved editor pose and stop applying player physics to the camera.
- `Game + missing model/anchor`: retain a finite fallback camera and expose the reason; do not throw
  from the frame loop.

## Update order

1. Read input intent and clamp it.
2. Integrate a clamped finite time step.
3. Apply gravity/jump and horizontal movement.
4. Resolve terrain/WMO collision through existing world collision seams.
5. Resolve the head anchor and project it to the camera.

The first slice supports walking, running, grounding, jumping, and simple collision response. It does
not provide networking, AI, combat, navmesh, or gameplay rules.
