# GV-14 Render Layer Contracts

## Intent

Define the render layers as explicit packet families before backend work deepens.

## Scope

- terrain layer
- liquid layer
- sky layer
- WMO layer
- M2/MDX layer
- overlay layer

## Outputs

- `RenderLayerId`
- per-layer packet interface boundary
- layer diagnostics contract

## Dependencies

- GV-06

## Proof

- runtime can enumerate active layer families for a frame

## Non-Goals

- no shader implementation
