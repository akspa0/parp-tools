# GV-04B Museums Object Package Contract

## Intent

Define the smallest forward-native object package for Museums content.

## Scope

- GLB payload rules
- external texture references
- per-object metadata sidecar
- package identity and version tag
- raw-artifact vs normalized-object distinction

## Outputs

- `MuseumsObjectPackage`
- per-object package folder/layout contract
- required metadata fields list

## Dependencies

- GV-04A
- GV-10A

## Proof

- one object can be represented as a Museums package without borrowing WoW-family assumptions

## Non-Goals

- no collection/index store yet
