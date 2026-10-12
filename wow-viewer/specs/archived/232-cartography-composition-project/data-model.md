# Data Model — Spec 232

## `PhaseLayerSettings`

An ordered layer over a base map. Key persisted state: source map, enabled flag, channel mask, presence gate, tile offset, cell offset, quarter-turn rotation and origin, horizontal/vertical mirrors, tile placements, footprint colour, and lock state. Tile and cell resolution remains confined to 0..63 and 0..15 grid axes respectively.

## `PhaseLayerProjectFile`

Per-base-map JSON document: `BaseMap`, base channel mask, and every `PhaseLayerProjectEntry`. Entries preserve unresolved and disabled layers. Save is explicit; map construction auto-loads the matching base map. A mismatched nonempty project is rejected.

## `AlphaTileData`

Parsed Alpha ADT content, including the 257x257 height lattice, optional normal/shadow/alpha lattices, 16x16 chunk metadata, liquid chunks, textures and placements. T015 adds a derived, transform-safe tile representation; it does not mutate parsed input.

## `PhaseCompositionPolicy`

Pure owner of target-to-donor tile/chunk mapping, transform ordering, placements, 64x64 admission, and later export parity. Adapters may provide existence/parse functions but may not reimplement coordinate rules.

## State transitions

`editable` → `locked` rejects channel and transform changes; explicit unlock returns to `editable`. `saved` is a snapshot of all layer states and does not imply the map source is currently resolvable.

