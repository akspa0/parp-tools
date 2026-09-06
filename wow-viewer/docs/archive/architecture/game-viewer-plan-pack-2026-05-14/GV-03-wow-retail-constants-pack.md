# GV-03 WoW Retail Constants Pack

## Intent

Create a shared constants pack for pre-release, LK, and early Cata-era WoW data that the runtime and tools can query by profile.

## Scope

- tile/chunk dimensions
- known ADT-family version markers
- common world-space conventions
- retail-era FourCC/chunk ids that must not live in core
- profile feature flags for `0.7.0`, `3.x`, `4.0.0`

## Outputs

- `WowViewer.Core.Constants.WowRetail`
- profile-keyed constants records

## Dependencies

- GV-01

## Proof

- inspect/harvest/runtime code can resolve profile constants without raw switch ladders

## Non-Goals

- no DBC editing logic
