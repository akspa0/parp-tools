# GV-04 Warcraft 3 Constants Pack

## Intent

Establish the first explicit Warcraft 3 constants pack so support lands as a profile seam instead of a WoW-side hack.

## Scope

- MPQ/archive assumptions
- MDX/BLP-era asset-family ids
- Warcraft 3-specific chunk/tag ids where needed
- version labels for early supported Warcraft 3 profiles
- basic coordinate and unit notes where they differ from WoW

## Outputs

- `WowViewer.Core.Constants.Warcraft3`
- explicit unsupported/unknown areas list

## Dependencies

- GV-01

## Proof

- profile registry can bind a Warcraft 3 root without pretending it is a WoW root

## Non-Goals

- no full Warcraft 3 renderer parity
