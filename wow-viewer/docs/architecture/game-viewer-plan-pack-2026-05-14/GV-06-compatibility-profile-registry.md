# GV-06 Compatibility Profile Registry

## Intent

Turn supported game families into explicit compatibility profiles rather than scattered conditional behavior.

## Scope

- profile ids
- profile display names
- profile capability flags
- profile-to-constants binding
- profile-to-schema binding

## Outputs

- `CompatibilityProfileRegistry`
- profile records for WoW Alpha, WoW Retail-era, Warcraft 3 Classic

## Dependencies

- GV-02 through GV-05

## Proof

- one root resolves to one explicit profile record

## Non-Goals

- no renderer implementation
