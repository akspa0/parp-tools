# GV-06A Profile Personality Library Contract

## Intent

Define what a supported profile/personality library is so WoW, Warcraft 3, Museums, and future families all plug into the engine the same way.

## Scope

- profile/personality identity
- required capability surfaces
- optional extension surfaces
- ownership split between `BASE` and a profile library

## Touched Surfaces

- compatibility/profile docs
- future shared contracts in engine-neutral libraries
- future WoW/WC3/Museums integration points

## Inputs And Assumptions

- a profile library is not only a constants pack
- a profile library may bring readers, schema sources, import/export adapters, render adapters, and audio resolvers
- the core engine must not become a disguised WoW profile

## Outputs

- one contract for what every profile/personality library may provide:
  - identity and metadata
  - constants packs
  - root detection hooks
  - asset-family declarations
  - schema providers
  - import/export adapters
  - render-layer adapters
  - audio-resolution adapters
- one rule for what stays in `BASE`:
  - engine-neutral runtime
  - engine-neutral rendering/audio/backend interfaces
  - artifact/provenance contracts
  - shared workspace and diagnostics seams

## Dependencies

- GV-00B
- GV-00C
- GV-06

## Proof

- a new profile family can be described as "implement these contract surfaces" instead of inventing its own shape

## Stop Conditions

- BASE vs profile ownership is clear for content, render, audio, and editor seams

## Non-Goals

- no plugin loader implementation
- no package manager design
