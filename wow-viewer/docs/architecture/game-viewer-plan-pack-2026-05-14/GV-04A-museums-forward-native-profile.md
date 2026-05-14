# GV-04A Museums Forward Native Profile

## Intent

Define `Museums` as the first forward-native profile family for the user's own simpler data style: GLB assets, external textures when needed, one metadata file per object, and later shard/index-backed data stores.

## Thesis

`Museums` is not just "custom content."

It is a named supported data type for:

- forward-native assets
- artifact-conscious metadata
- evolving package/index formats
- future distilled-model-backed content stores

The exact on-disk specification is intentionally still evolving. The plan must support that uncertainty without pretending the format is already final.

## Scope

- profile id for Museums roots
- GLB-first asset assumptions
- external texture attachment rules
- per-object metadata sidecar expectations
- NPZ-like shard thinking for structured data payloads
- future non-zip compression/index options
- future chromaDB-style or other indexed backing-store possibility
- explicit non-WoW/non-archive workflow boundary

## Outputs

- `Museums.ForwardNative.v0` profile record
- minimal sidecar metadata contract summary
- capability flags for import, preview, export, and future runtime use
- explicit "format still evolving" rule so implementation slices stay bounded

## Dependencies

- GV-00
- GV-01
- GV-06

## Proof

- the registry can represent a Museums root that has no MPQ, no FourCC chunks, no tile-grid assumptions, and no fixed Blizzard-style datastore shape

## Non-Goals

- no final sidecar schema yet
- no final shard/index backend choice yet
