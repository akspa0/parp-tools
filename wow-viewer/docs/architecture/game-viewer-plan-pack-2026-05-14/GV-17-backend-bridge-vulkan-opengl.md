# GV-17 Backend Bridge Vulkan OpenGL

## Intent

Keep Vulkan primary and OpenGL fallback behind one small backend bridge.

## Scope

- backend interface
- frame lifecycle hooks
- packet submission entrypoints
- diagnostics and capability reports

## Outputs

- `IRenderBackend`
- backend capability record
- packet-to-backend submission contract
- one explicit rule that WebGL, if added, is a delivery/component surface layered beside this bridge rather than a replacement for the native backend plan

## Dependencies

- GV-14 through GV-16

## Proof

- one frame can be described without choosing Vulkan or OpenGL at the runtime layer

## Non-Goals

- no advanced frame graph yet
