# GV-12 Export Job Pipeline

## Intent

Mirror imports with a shared export pipeline so conversion and extraction flows are predictable.

## Scope

- export request model
- output packaging rules
- overwrite and provenance rules
- dry-run and summary output

## Outputs

- `ExportJobRequest`
- `ExportJobResult`
- common export audit format

## Dependencies

- GV-07, GV-09, GV-10

## Proof

- one export flow uses the shared pipeline instead of a tool-specific custom path

## Non-Goals

- no batch scheduler yet
