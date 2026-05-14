# GV-11 Import Job Pipeline

## Intent

Define import as a first-class job system rather than ad hoc button handlers.

## Scope

- import request model
- source/target validation
- dry-run summary
- progress and diagnostics
- artifact outputs

## Outputs

- `ImportJobRequest`
- `ImportJobResult`
- shared job logging contract

## Dependencies

- GV-07, GV-09, GV-10

## Proof

- one asset import flow runs through the common job pipeline

## Non-Goals

- no UI polish
- no forward-native GLB specifics in this generic slice; those belong in `GV-11A`
