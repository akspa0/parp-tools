# GV-23A Distilled Portable Model Packages

## Intent

Define the future package seam for portable distilled models that may become part of Museums-backed content workflows.

## Scope

- tiny model package identity
- CPU-first inference assumption
- metadata-driven regeneration hooks
- provenance between source artifacts and distilled model outputs
- portability rules

## Outputs

- `DistilledModelPackage`
- package metadata fields
- relation to Museums stores and generated-content packages

## Dependencies

- GV-00A
- GV-04C
- GV-23

## Proof

- the architecture can talk about portable distilled models as content packages without pretending the compression/runtime scheme is already final

## Non-Goals

- no actual ML architecture choice
