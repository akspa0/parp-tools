# Visualization Notes: Alpha Brush Library

**Date**: 2026-06-23

## Contact Sheet Legend

The contact sheets render alpha-mask component crops from `catalog.jsonl`.

Color meaning:

| Color | Layer | Working interpretation |
|-------|-------|------------------------|
| Gray | L0 | Base/fill layer. Often empty in sampled alpha masks. |
| Blue | L1 | Primary brush stroke / broad terrain motif. |
| Green | L2 | Transition/detail brush stroke. |
| Orange | L3 | Highlight/detail brush stroke. |

Each row is one embedding cluster. Each cell is one representative alpha component crop from that cluster. The text inside each cell shows the full build ID, source map tile coordinate, tile-local component bbox, layer, and area.

## Human Review Notes

- The current clusters are too atomic to represent the full authored terrain units. The real building blocks appear closer to **sprites/prefabs/pastes** that can span multiple ADT tiles and contain multiple alpha components across layers.
- Many brush clusters look like hand-placed terrain sprites rather than arbitrary masks. This matches the working theory that terrain was painted from a reusable library of authored shapes.
- Cluster `C35` in `cluster_contact_sheet_000.png` appears to contain very low-resolution heightmap-like shapes. These may be legacy terrain stamps from an older toolchain, plausibly inherited from Warcraft 3 editor-era content and reused as building blocks for later WoW terrain.
- User terminology: these micro-level alpha-mask edits are still best described as **scars**. They include exact reused brush marks plus hand-modified/fixup variants used to blend local terrain better.
- Exact binary dedupe over the full two-build catalog found `263,188` exact scar patterns from `320,368` components. This confirms that exact reuse exists, but most scar instances are unique or modified.
- `pattern_neighbors.jsonl` ranks non-exact scar variants by embedding similarity so the review surface can show "same brush idea, edited differently" instead of treating every scar as unrelated.

## Next Interpretation Step

The current `BrushComponent` catalog should be treated as an **atomic stroke library**, not the final prefab library.

The next useful layer is a multi-component/multi-tile grouping pass:

1. Group nearby components by tile adjacency, layer stack, and cluster co-occurrence.
2. Reconstruct larger sprite/paste candidates from those grouped components.
3. Render contact sheets for the grouped sprites, not only individual connected components.
4. Compare those grouped sprites against the older V18 paste outputs to see where the two libraries overlap.

This should be a follow-up task after the current component-level visualization is reviewed.
