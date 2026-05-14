# GV-17C WebGL Component And Web Delivery Surface

## Intent

Define a WebGL-facing component as a delivery and embedding surface without replacing the primary engine backend strategy.

## Scope

- WebGL component role
- browser/embed preview role
- runtime-to-web scene projection boundary
- capability and degradation rules

## Touched Surfaces

- render/backend planning docs
- host/editor planning docs
- future export/preview contracts
- future web host or embedded panel work

## Inputs And Assumptions

- Vulkan remains the primary engine backend
- OpenGL remains the native fallback backend
- WebGL is valuable as a universal output experience, embedded preview surface, and future bridge into the user's other project
- WebGL should not force the engine core to become browser-shaped

## Outputs

- one rule that WebGL is a component/delivery surface, not the primary engine backend
- one candidate contract where runtime or content scenes can be projected into a WebGL-safe representation
- one capability/degradation story for when a native frame can only be partially expressed on the WebGL side

## Dependencies

- GV-14
- GV-17
- GV-21

## Proof

- future planning can describe how a native runtime scene or asset preview reaches a browser/embed surface without redefining backend ownership

## Stop Conditions

- a smaller model can tell whether a task belongs in Vulkan/OpenGL engine work or in the WebGL delivery layer

## Non-Goals

- no browser renderer implementation yet
- no promise that WebGL reaches feature parity with the native renderer
