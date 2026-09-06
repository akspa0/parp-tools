# GV-07 Game Root Records And Manager Service

## Intent

Build the service contract behind the future `Game Manager` workspace.

## Scope

- registered root record
- root labels and notes
- active root selection
- multi-root session list
- persisted root metadata

## Outputs

- `GameRootRecord`
- `GameRootManager`
- app settings storage contract

## Dependencies

- GV-05, GV-06

## Proof

- roots can be registered, reloaded, and switched without re-detection every launch

## Non-Goals

- no asset browsing yet
