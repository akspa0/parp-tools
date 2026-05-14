# GV-00A Artifact Provenance And Preservation

## Intent

Make artifact preservation a first-class engine concern instead of an emotional side-note.

## Thesis

The project values the genuine artifact:

- buggy shipped data
- accidental exports
- version-specific mistakes
- archive-specific edge cases
- historically real outputs that modern clones smooth over

The engine must preserve the ability to witness those artifacts, not just sanitized reinterpretations.

## Scope

- source artifact identity
- exact-source vs normalized vs converted representations
- provenance chains
- hash and version capture
- immutable raw-capture references

## Outputs

- `ArtifactRecord`
- `ArtifactRepresentationTier`
- `ProvenanceChain`
- preservation rules for raw bytes, decoded semantics, and converted outputs

## Representation Tiers

1. raw artifact
2. decoded artifact semantics
3. normalized engine representation
4. converted/exported derivative
5. generated derivative

## Dependencies

- GV-00

## Proof

- one asset can be traced from raw source bytes to decoded semantics to normalized engine use without losing provenance identity

## Non-Goals

- no database UI yet
