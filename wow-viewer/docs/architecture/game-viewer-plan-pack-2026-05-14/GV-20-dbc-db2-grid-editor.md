# GV-20 DBC DB2 Grid Editor

## Intent

Define the first bounded metadata-table editor workflow.

## Scope

- row grid model
- typed cell editors
- change tracking
- import/export of table deltas
- profile-safe validation

## Outputs

- table editor session model
- change-set format
- validation message contract

## Dependencies

- GV-19, GV-21

## Proof

- one supported table can be opened, edited, validated, and saved through a schema-driven path

## Non-Goals

- no every-table parity promise
