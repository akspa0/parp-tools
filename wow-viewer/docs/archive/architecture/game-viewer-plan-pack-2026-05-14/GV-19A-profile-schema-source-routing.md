# GV-19A Profile Schema Source Routing

## Intent

Separate "which schema source do we use?" from the broader schema catalog plan.

## Scope

- DBCD/WoWDBDefs routing for WoW profiles
- sidecar schema routing for forward-native profiles
- unsupported-schema reporting for other profiles

## Outputs

- `SchemaSourceRoutingRule`
- profile-to-schema-source map

## Dependencies

- GV-06
- GV-10A
- GV-19

## Proof

- the engine can explain why one profile uses DBCD/WoWDBDefs and another uses sidecar schema documents

## Non-Goals

- no row editor behavior
