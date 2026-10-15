# Operator Addendum — Technical Planning Inputs

**Current-scope note (2026-10-14):** The operator subsequently prioritized WC3 tilesets/files in the existing File Browser, explicitly not world rendering, and directed reuse of existing MPQ/listfile support. [Current spec](spec.md) supersedes the WC2-first and restoration-first sequencing below. Retain this addendum as future technical input, not current-phase implementation scope.

**Received**: 2026-10-14 during specification discovery.

**Authority**: Operator-supplied extension of the [original draft](../../docs/wc2-3-support/gemini-code-1791995880062.md), followed by the explicit correction that the operator owns original DOS WC2, not Battle.net Edition. This document retains the requested technical scope; it is **not** a validated format reference or an implementation plan.

## Operator WC1/GDS Lineage Clarification

The operator subsequently stated: “WC2's source code shows us that WC2 is just a slightly scaled up WC1, as part of the Game Development System that Blizzard called the WC1 engine. It's all in there, every iota of WC1 is hidden in the WC2 code!”

Treat this as the operator's concrete source-lineage/reuse basis: build WC2's shared foundation with a WC1 follow-on in mind, and trace retained GDS/WC1 behavior in the local source rather than proposing an unrelated engine. This session did not exhaustively audit that source or validate WC1 data, so it records the statement as operator-provided evidence rather than an independently completed compatibility result.

## Requested Model and I/O Alignment

The operator requested alignment with the existing [authoring-map models](../../src/core/WowViewer.Core/Maps/) and [DAT converter](../../src/core/WowViewer.Core.IO/Maps/DatToLkAdtConverter.cs:1), rather than the draft's obsolete legacy-project layout.

Requested names for future planning, **not existing/implemented symbols**:

```text
WowViewer.Core.Maps.Wc2
  Wc2PudMap, Wc2Tileset, Wc2GrpSprite
WowViewer.Core.Maps.Wc3
  Wc3W3eMap, Wc3Tilepoint, Wc3TerrainSlkRow, Wc3AtlasSubTile
WowViewer.Core.IO.Wc3
  Blp1Reader, Wc3TileExtractor, Wc3ToBlp2Converter, W3eToLkAdtConverter
```

Shared models belong in core, format I/O in core I/O, and viewer interactions in owned viewer services. The names above do not authorize replacing proven readers or adding members to the frozen shell/scene classes.

## Supplied Coordinate Table — Assertions Awaiting Cross-Engine Proof

| Metric | WC3 terrain | WoWEdit authoring / 2000 HUD | Compiled WoW terrain |
|---|---|---|---|
| Base unit | Approximately 1.152 inches per WC3 unit | One inch; 36 per yard | One yard |
| Up axis | +Z; decoded ground height uses an 8192 bias and divisor 4 | +Y in viewport/authoring placements; column, vertical, row | +Z |
| Terrain quad spacing | 128 units, stated as 4.096 yards | 150 inches | 4.16667 yards |
| Chunk | 8×8 WC3 cells, 1024 units | 1200 inches | 33.33333 yards |
| Full WoW tile comparison | 128×128 WC3 cells, 16384 units | 19200 inches | 533.33333 yards |
| Layer weights | Discrete corner masks | 64×64 authoring weights | 64×64 sequential terrain blending |

**Supplied examples**: Goldshire client HUD (-19582, -2277, 0); Loch Modan WoWEdit viewport HUD (64480, 11807, 35270).

Requested helper interpretations:

- Client HUD: divide X/Y/Z inches by 36 and add an explicitly supplied local-map origin in yards.
- WoWEdit viewport HUD: divide by 36, then map source X to horizontal X, source Y to vertical Z, and source Z to horizontal Y. Any sign changes, map origin, and viewer-axis conversion still need evidence.

**Verification limits**:

- The existing [height-divisor option](../../src/core/WowViewer.Core.IO/Maps/DatToLkAdtConverter.cs:11) documents inches-to-yards for DAT heights. This does not prove the screenshot claims, all horizontal coordinates, or a WC3 physical unit.
- The table's 4.096-yard WC3 cell and 4.16667-yard WoW cell are not equal. Matching cell topology would instead imply 150/128 = 1.171875 inches per WC3 unit. A physical-scale conversion and a topology-preserving conversion are different experiments; neither may be selected implicitly.
- The screenshots and their camera/map origins were not supplied or independently inspected in this session. The two tuples are operator-provided examples, not verified camera destinations.
- Existing authoring-weight uncertainty is not resolved by placing a proposed interpretation in this table.

## Requested Candidate Extraction and Ancestry Comparison

1. Decode mip 0 of WC3 terrain atlases to color-plus-alpha pixels. The supplied layout rule proposes four columns for square sheets, eight columns for double-width sheets, and four rows; validate supported asset layouts before relying on dimensions alone.
2. Extract proposed solid interiors at (0,0), (3,3), and the sixteen right-hand cells of an extended sheet, preserving variation numbering and source crop. Normalize 64×64 cells to 128×128 with wrap-aware resampling; retain native 128×128 cells.
3. For each 256×256 WoW 0.5.3 texture, produce four 128×128 quadrant views plus a full-tile 128×128 reduction.
4. Compare those views with WC3 candidates at 0°, 90°, 180°, and 270°, using zero-mean normalized cross-correlation (ZNCC) on luminance. Requested luminance weights: 0.299 red, 0.587 green, 0.114 blue.
5. Emit ranked JSON and Markdown reports retaining source/target identity, crop, variation, rotation, dimensions, and score. Treat a zero denominator as unscorable; define deterministic tie handling during planning. Scores are hypotheses about similarity, not provenance certificates.

Requested normalization:

```text
ZNCC(A,B) = sum((A-mean(A))*(B-mean(B))) /
            sqrt(sum((A-mean(A))^2) * sum((B-mean(B))^2))
```

The operator named Westfall straw/grass, Elwynn dirt/grass, and Durotar cracked earth as comparison targets, not already established matches. No WC3 corpus or WoW target corpus was compared in this session.

## Requested 256×256 Synthesis and Live Comparison

Requested future operations, **not implemented APIs**:

```text
SynthesizeQuilted256(varTL, varTR, varBL, varBR, crossSeamBlendRadius = 8)
SynthesizeToroidalUpscale256(singleVar128)
```

- Quilt four 128×128 variations into a 256×256 canvas. The operator requested cosine/Hermite cross-fading at the inner X/Y=128 seams and outer wrap edges, with default blend radius 8.
- Upscale a single 128×128 or 64×64 variation with a Lanczos-3 kernel and modulo boundary wrapping. The supplied bit-mask example is valid only for a 128-wide input; a 64-wide input requires its own dimensions or prior explicit normalization.
- Retain the original draft's nearest-neighbor-plus-box/Gaussian comparison mode rather than silently dropping it when adding the two detailed modes above.
- Generate a nine-level mip chain from 256×256 to 1×1 and export the requested BLP2 terrain encodings, including BC1 and palettized color with the selected alpha semantics. Actual target-consumer compatibility must be verified; do not assume every WoW Alpha build accepts the same encoding.
- Temporary viewer swapping is an explicit, reversible experiment. No existing map/texture or frozen Alpha writer is implicitly modified.

Wrap-aware filtering avoids clamped-filter boundary artifacts but does not prove that arbitrary art repeats seamlessly. All seam-removal and historical-authenticity claims require the corresponding visual/source evidence.

## Routing and Boundaries

- WC2 original DOS discovery and terrain preview come first; WC2 sprites are a separately demonstrable extension.
- Classic WC3 viewing precedes its restoration experiments; beta compatibility needs its own real samples.
- Coordinate helpers and WC3-to-LK terrain comparison are later explicit tools, not changes to the normal camera or terrain base.
- WC1 is a reuse-driven follow-on to the shared GDS/WC1 foundation identified by the operator; trace retained behavior and validate specific data variants. Battle.net is not required and remains unscheduled; PlayStation source is reference-only.
- Technical file sets, algorithms' exact error handling, conversion sampling, UI registration, and receipt-gated task phases belong in the subsequent Spec Kit plan/tasks workflow, not this addendum.
