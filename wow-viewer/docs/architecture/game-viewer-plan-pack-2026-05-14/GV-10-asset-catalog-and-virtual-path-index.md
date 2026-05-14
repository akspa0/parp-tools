# GV-10 Asset Catalog And Virtual Path Index

## Intent

Create the catalog contract that powers asset browsing, search, and import/export actions.

## Scope

- asset-family classification
- virtual path index
- profile-aware path casing rules
- optional discovered metadata summaries

## Outputs

- `AssetCatalogEntry`
- `AssetCatalogIndex`
- minimal search/filter contract

## Dependencies

- GV-07, GV-09

## Proof

- one root can produce a searchable asset inventory by family

## Non-Goals

- no render preview yet
