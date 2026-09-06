# GV-08 Feature Gate Capability Matrix

## Intent

Define exactly how the app decides which tools and render layers are available for a profile.

## Scope

- profile capabilities
- feature ids
- required schemas/files
- enable, disable, degraded states

## Outputs

- `FeatureCapabilityMatrix`
- rules for honest disabled-state messaging

## Dependencies

- GV-06, GV-07

## Proof

- the app can explain why a feature is on, off, or degraded for a root

## Non-Goals

- no editor widgets
