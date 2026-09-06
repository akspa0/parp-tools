# GV-01 Core Constants Registry

## Intent

Create one engine-neutral constants library pattern that all game/profile packs can build on without assuming WoW-style storage or runtime semantics.

## Thesis

The core constants layer must not assume that every game:

- uses chunk ids
- uses FourCCs
- uses MPQ/CASC-like archives
- uses tile grids
- uses WoW-like world origins
- uses Blizzard-era model/material semantics

Those are profile-level concerns.

The core layer is only allowed to define constants and ids that remain meaningful across radically different data models, including forward-native GLB + metadata pipelines.

## Scope

- shared numeric tolerances
- engine-neutral unit helpers
- coordinate-system descriptor ids
- engine-neutral asset-family ids
- metadata/provenance tier ids
- package format/version ids for engine-owned interchange
- profile-neutral asset-family ids

## Explicit Exclusions

- no FourCC wrappers in core
- no chunk-id wrappers in core
- no WoW tile-size assumptions in core
- no archive-format assumptions in core
- no game-specific world-origin rules in core

## Outputs

- `WowViewer.Core.Constants` namespace
- naming rules for constants packs
- separation rules between universal constants and profile-specific constants
- tests proving no duplicate ids or contradictory unit definitions

## Dependencies

- GV-00
- GV-00A

## Proof

- compile-safe constants surface
- focused tests for uniqueness, unit conversions, and profile-neutral coordinate-system descriptors

## Non-Goals

- no game-specific values in this first slice
