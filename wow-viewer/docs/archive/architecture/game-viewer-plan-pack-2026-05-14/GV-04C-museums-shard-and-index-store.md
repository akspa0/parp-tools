# GV-04C Museums Shard And Index Store

## Intent

Carve the Museums storage question into a tiny plan instead of letting it stay as a vague dream.

## Scope

- shard-style payload thinking inspired by current NPZ tooling
- separation between payload storage and index storage
- compression pluggability
- future chromaDB-style/indexed backing-store seam
- portability rules

## Outputs

- `MuseumsShardStore` concept
- `MuseumsIndexStore` concept
- backend-neutral store interface notes
- explicit undecided-backend boundary

## Dependencies

- GV-04A
- GV-09A

## Proof

- the Museums profile can describe stored objects/data without committing the project to zip-only or one database choice too early

## Non-Goals

- no concrete database implementation
