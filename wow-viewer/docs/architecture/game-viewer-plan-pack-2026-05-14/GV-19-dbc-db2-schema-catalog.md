# GV-19 DBC DB2 Schema Catalog

## Intent

Create the schema catalog that all metadata-table viewing and editing flows depend on.

## Scope

- table identity
- column schema
- profile support matrix
- definition-source provenance

## Outputs

- `SchemaCatalogEntry`
- `SchemaCatalog`
- missing-schema diagnostics

## Dependencies

- GV-05, GV-06

## Proof

- one profile can enumerate editable and readable table schemas

## Non-Goals

- no row editor UI yet
