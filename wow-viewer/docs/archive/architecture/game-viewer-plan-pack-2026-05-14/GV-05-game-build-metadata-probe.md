# GV-05 Game Build Metadata Probe

## Intent

Detect game family, version, and build metadata from a root with as little manual input as possible.

## Scope

- executable version metadata
- archive layout signals
- key file existence probes
- fallback manual confirmation path

## Outputs

- `GameBuildMetadata`
- `GameFamily`
- `ProfileDetectionResult`

## Dependencies

- GV-01 through GV-04

## Proof

- one probe command can distinguish at least Alpha WoW, LK WoW, and Warcraft 3 candidates

## Non-Goals

- no UI yet
