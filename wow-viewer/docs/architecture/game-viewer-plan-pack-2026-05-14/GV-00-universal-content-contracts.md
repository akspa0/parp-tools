# GV-00 Universal Content Contracts

## Intent

Define the engine's truly game-neutral content contract before any WoW-, Warcraft-, or custom-profile constants packs sit on top of it.

## Thesis

The engine core must not assume:

- MPQ archives
- FourCC chunked binaries
- tile grids
- WoW coordinate systems
- Blizzard-era asset semantics

The engine core must be able to host:

- archival artifact profiles like WoW and Warcraft 3
- forward-native profiles like GLB + textures + sidecar metadata
- future generated content packages

## Scope

- engine-neutral asset identity
- engine-neutral scene/object/material concepts
- content package identity and versioning
- engine-neutral coordinate-system descriptor
- engine-neutral metadata attachment model

## Outputs

- `UniversalContentId`
- `UniversalAssetKind`
- `CoordinateSystemDescriptor`
- `MetadataAttachment`
- `ContentPackageDescriptor`

## Dependencies

- none

## Proof

- one WoW-style asset and one forward-native GLB + metadata asset can both be described without the core contract pretending they share the same storage shape

## Non-Goals

- no file parser work
- no renderer work
