# GV-09 Archive And Filesystem Adapter Seam

## Intent

Unify archive-backed, loose-file, and mixed-overlay asset access behind one small adapter contract.

## Scope

- MPQ root adapter
- loose filesystem adapter
- overlay precedence rules
- existence/read/list operations

## Outputs

- `IGameDataSource`
- source-kind diagnostics
- profile-aware path normalization rules

## Dependencies

- GV-06, GV-07

## Proof

- one caller can request assets without caring whether they came from MPQ or loose overlay

## Non-Goals

- no catalog UI
