# GV-15 Terrain And Liquid Render Packets

## Intent

Break terrain and liquid submission into explicit runtime packets that Vulkan and OpenGL can both consume.

## Scope

- terrain tile packet
- terrain chunk packet
- liquid tile packet
- material/alpha/light inputs
- visibility and LOD fields

## Outputs

- terrain and liquid packet records
- required source-data list
- debug summary rules

## Dependencies

- GV-14

## Proof

- one bounded world frame emits terrain/liquid packets without backend-specific types

## Non-Goals

- no final textured renderer
