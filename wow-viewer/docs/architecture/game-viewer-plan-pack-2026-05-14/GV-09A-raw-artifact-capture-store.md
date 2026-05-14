# GV-09A Raw Artifact Capture Store

## Intent

Define where and how raw artifact captures are recorded so provenance does not collapse into normalized-only workflows.

## Scope

- raw capture record format
- byte-hash rules
- source-location references
- immutable capture vs derived cache distinction

## Outputs

- `RawArtifactCaptureRecord`
- capture storage rules
- link rules into provenance chains

## Dependencies

- GV-00A

## Proof

- one preserved artifact can be referenced without confusing it with decoded or converted outputs

## Non-Goals

- no storage backend implementation
