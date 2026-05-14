# GV-22 Selection And Copy Paste Semantics

## Intent

Avoid vague editor behavior by defining what "copy", "paste", and "selection" mean for assets and world content.

## Scope

- asset selection ids
- world selection ids
- transform payloads
- profile compatibility checks
- conflict and remap rules

## Outputs

- selection model
- copy/paste operation contract
- unsupported-operation messages

## Dependencies

- GV-13, GV-18

## Proof

- one bounded selection can round-trip through copy/paste semantics without silent data loss

## Non-Goals

- no undo stack yet
