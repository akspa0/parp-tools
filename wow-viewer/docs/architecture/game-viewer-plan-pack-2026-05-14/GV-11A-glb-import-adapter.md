# GV-11A GLB Import Adapter

## Intent

Carve out the forward-native import path as a tiny story instead of hiding it inside a generic import pipeline, with Museums as the first host profile.

## Scope

- GLB source intake
- external texture discovery
- sidecar metadata association
- normalized asset-package output

## Outputs

- `GlbImportRequest`
- `GlbImportResult`
- failure codes for missing textures or metadata mismatch

## Dependencies

- GV-04A
- GV-10A
- GV-11

## Proof

- one GLB + metadata object can enter the common import pipeline without pretending it is a WoW-family asset

## Non-Goals

- no runtime rendering parity
