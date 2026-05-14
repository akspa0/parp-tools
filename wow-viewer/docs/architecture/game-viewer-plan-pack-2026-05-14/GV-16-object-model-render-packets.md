# GV-16 Object Model Render Packets

## Intent

Separate WMO, M2, MDX, and future Warcraft 3 model submission into clear packet families.

## Scope

- WMO instance packets
- M2/MDX instance packets
- animation state fields
- material/effect routing fields
- compatibility-profile tags

## Outputs

- object/model packet records
- family-specific diagnostics

## Dependencies

- GV-06, GV-14

## Proof

- runtime can emit object/model packets with profile-aware family tagging

## Non-Goals

- no deep material parity yet
