# GV-02 WoW Alpha Constants Pack

## Intent

Capture stable `0.5.x` constants in one reusable pack instead of scattering them across readers, writers, runtime, and renderer code.

## Scope

- map origin and tile sizes
- ADT/WDT/MCNK dimensions
- Alpha liquid/tile flags
- placement transform constants
- Alpha-specific FourCC/chunk ids that must not live in core
- version labels and known build ids

## Outputs

- `WowViewer.Core.Constants.WowAlpha`
- one short source-of-truth note per constant group

## Dependencies

- GV-01

## Proof

- no duplicate literal values remain in the highest-risk Alpha seams

## Non-Goals

- no writer refactor beyond constant extraction rules
