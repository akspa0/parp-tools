# Bounded Discovery — 2026-10-14

**Scope**: Read-only local inventory, two archive headers/hashes, eight small loose-map structural scans/hashes, and focused PlayStation source excerpts. No archive entry decompression, sprite decoding, image generation, client launch, broad harvest, or format-reader change occurred.

**Related documents**: [Specification](../spec.md), [operator addendum](../operator-addendum.md), [unverified original draft](../../../docs/wc2-3-support/gemini-code-1791995880062.md).

## Operator-Local Inputs — Not Portable Defaults

These exact paths record the operator-supplied discovery environment only. Applications/tests must accept configured roots and must not depend on these paths.

| Alias | Operator-supplied root | Provenance/limitations |
|---|---|---|
| WC2_DATA | C:\WarcraftResearch\WC2\War2 | Operator identifies original DOS WC2; physical CD is scratched. Exact executable build not established. Successful structural reads are not proof of intact assets. |
| WC2_PSX_REFERENCE | G:\SourceLeak\WC2PSX2 #1\Playstation | Read-only local source reference. Exact source revision not established. No source/assets copied into the repository. |

No WC1, Battle.net, or WC3 data root was supplied or validated.

**Operator source-lineage evidence**: After discovery, the operator identified the WC2 source as retaining WC1 throughout Blizzard's shared GDS engine. This is the basis for a reuse-driven WC1 follow-on; the bounded source excerpts below are not an exhaustive independent audit of that claim or a WC1 file-compatibility test. See the [recorded clarification](../operator-addendum.md).

## Actual DOS Data Inventory

The data subdirectory contains the following [draft-named archive families](../../../docs/wc2-3-support/gemini-code-1791995880062.md:44):

| Resource | Bytes |
|---|---:|
| MAINDAT.WAR | 8,409,933 |
| REZDAT.WAR | 1,894,026 |
| SNDDAT.WAR | 23,650,176 |
| STRDAT.WAR | 51,550 |
| MUDDAT.CUD | 50,740,192 |
| SFXDAT.SUD | 4,302,671 |

Also observed: two installer-resource files. No Battle.net MPQ requirement is justified by this inventory. Audio archives were inventoried only, not decoded or added to scope.

### Header Evidence Contradicts the Draft's Constant Type Field

| Resource | First 12 bytes (hex) | Candidate count at byte 4 | 16-bit field at byte 6 | SHA-256 |
|---|---|---:|---:|---|
| MAINDAT.WAR | 19 00 00 00 B6 01 E8 03 E0 06 00 00 | 438 | 1000 | 2039C6DEFC73392C5EA826671EEFA25A54BF954E335D2E47EA8702F2DC4CE126 |
| REZDAT.WAR | 19 00 00 00 5D 00 B8 0B 7C 01 00 00 | 93 | 3000 | DF3F03405BD1B9BF219D349891836AC61578621EC034561F2704FBA9C3E9DAA0 |

The [draft header](../../../docs/wc2-3-support/gemini-code-1791995880062.md:54) labels the byte-6 field as a constant 1. Both real headers contradict that assertion. Resource-ID-base semantics are a **hypothesis**, not verified by these bytes alone. The first candidate offsets (1760 and 380) are consistent with an eight-byte header plus four bytes per candidate entry; entry contents/compression remain unverified.

### Loose Map Evidence

All eight loose maps structurally consumed exactly their file length. Each reports version 17, and each terrain payload length equals width × height × 2. The final column is the observed placement payload length divided by eight, **not proof of placement semantics**.

| Map | Bytes | Width × height | Era ID | Terrain bytes | Eight-byte records |
|---|---:|---|---:|---:|---:|
| ALAMO.PUD | 71,902 | 96×96 | 0 | 18,432 | 71 |
| CHANNEL.PUD | 71,486 | 96×96 | 2 | 18,432 | 19 |
| DEATH.PUD | 37,086 | 64×64 | 2 | 8,192 | 199 |
| DRAGON.PUD | 122,006 | 128×128 | 2 | 32,768 | 62 |
| ICEBRDGE.PUD | 122,510 | 128×128 | 1 | 32,768 | 125 |
| ISLANDS.PUD | 71,518 | 96×96 | 1 | 18,432 | 23 |
| LAND_SEA.PUD | 71,806 | 96×96 | 1 | 18,432 | 59 |
| MUTTON.PUD | 18,790 | 32×32 | 0 | 2,048 | 600 |

Input hashes:

| Map | SHA-256 |
|---|---|
| ALAMO.PUD | DBF65ECB7842906EDD5590475AFB3C91D51F13A9B5265EC35EEBDEC3724A7308 |
| CHANNEL.PUD | AE6BC5015D50225B096700049E0FD613611985775DC70759F3A2F96F15BD5389 |
| DEATH.PUD | 75F695029C50A76699FDD2E143CC1B15D61C34C3F60219D5371CE57BB4167A1B |
| DRAGON.PUD | 13E2A0EA96602086733CC4A34073142F47C4D6D0545E2851774FC971145B258E |
| ICEBRDGE.PUD | 955C0AB945AF0F46C0A29AC0BC3B994EA253C81D4E0E5F8CBD50229DB163ABE5 |
| ISLANDS.PUD | 4FFE6C96C25D11846F330C98BEF4E3552485143B6FDFB991924488C49E26CE7E |
| LAND_SEA.PUD | A62A84A33E49BFC2937477CAF2409F61EFD31378856E43125144B8C2B6FF2B79 |
| MUTTON.PUD | 2D9E1FF3583E5D672A6472255CFAD0C0229FF56EFB306527491282AAAE0EBB41 |

All observed maps contain type, version, description, ownership, era, dimensions, unit-data/upgrade/side/resource/AI sections, terrain, square/oil/region maps, and unit placement sections. This corroborates the chunk envelope and selected field sizes, **not** the draft's complete tile-resolution formula or sprite format. No extended-era chunk was observed in these eight maps.

## Read-Only PlayStation Source References

Locations below are relative to WC2_PSX_REFERENCE, not repository files. They are recorded as source locations rather than links to nonexistent repository copies.

| Local source location | Observed evidence | Limitation |
|---|---|---|
| MAPHDR.H, lines 35–41 | Map type has a ten-byte type string, a two-byte test field, and a four-byte checksum. | More specific than the draft's generic six-byte suffix; do not import native struct packing. |
| MAPHDR.C, lines 386–414 | Checks the WC2 map type and version. | Does not validate DOS archives. |
| MAPHDR.C, lines 656–657 | Terrain chunk reads directly into the cell-map buffer. | Does not establish group-versus-direct tile-ID resolution. |
| MAPHDR.C, lines 875–919 | Header/map dispatch tables; extended-era override is conditionally compiled, with an editor-compatibility comment. | Expansion/platform-dependent; no real extended-era map validated. |
| MAINDAT.H, lines 3–37 | Named terrain palette, tile, cell, group, and PSX VRAM resources with numeric IDs. | PSX IDs are not automatically DOS entry indices. |
| maindat.mak, lines 38–40, 54–56, 70–72, 645–647 | Terrain resource build references include the draft's tile/cell/group file families. | Shared names do not prove record sizes, flips, or pixel encoding. |
| TILE.C, line 114 | Loads platform-specific group/VRAM-era tile data. | Concrete warning against treating PSX drawing as the DOS pipeline. |
| LZSS.BAK, first 64 bytes | Starts with a binary LNK signature, not text source. | Do not treat a backup extension as readable compression source. |

An initial bounded search mistakenly included a nonexistent map source filename and exited 1; it was corrected to the actual map-header files and the next probe exited 0. No source was changed or executed.

## Draft Claims That Still Need Proof Before Implementation

- Archive compression flags, size masks, dictionary initialization, cursor, and back-reference semantics.
- DOS archive resource identities and their relationship to reference-source numeric IDs.
- DOS palette precision, transparent/team/cycling entries, tile bitfields, group record sizes, and terrain-ID resolution. Never hide uncertainty behind an automatic alternate formula.
- Sprite format variant, offsets, row tables, and repeat/skip/literal opcodes.
- Classic/beta WC3 texture and model compatibility; atlas variation ordering, cliff/ramp naming, water math, and variable placement tails.
- Early WoW accepted texture encoding, pixel/alpha interpretation, and the draft's historical art-lineage narrative.
- Cross-engine scale/axes/origin claims and screenshot coordinates from the addendum. The unequal WC3/WoW cell spacing is explicitly tracked in the [addendum](../operator-addendum.md).

## Existing-Code Evidence and Safe Boundaries

A bounded read-only sub-agent search found no WC2/WC1-specific archive/map/sprite support in core, Workbench, tests, or viewer user docs. This is a scoped search result, not proof about every external library.

- [Archive contracts](../../../src/core/WowViewer.Core.IO/Files/IArchiveReader.cs:3) and [MPQ catalog](../../../src/core/WowViewer.Core.IO/Files/MpqArchiveCatalog.cs:7) exist; an MPQ catalog is not DOS archive support.
- [BLP summary reader](../../../src/core/WowViewer.Core.IO/Blp/BlpSummaryReader.cs:9) recognizes multiple signatures but currently uses a common 156-byte BLP0/1 header assumption. Signature recognition is not evidence of external-mip beta decoding. The primary agent inspected this file; no change is authorized here.
- [MDX texture-layout handling](../../../src/core/WowViewer.Core.IO/Mdx/MdxFile.cs:705) documents WC3 overlap, and the [standalone loader](../../../src/viewer/WoWViewer/Workbench/Services/ModelLoading/StandaloneModelLoaderService.cs:513) has an MDLX path. These are reuse candidates, not a real WC3 compatibility receipt.
- [Workbench navigation](../../../src/viewer/WoWViewer/Workbench/WorkbenchNavigator.cs:117) and the [WDL preview service](../../../src/viewer/WoWViewer/Workbench/Services/WdlPreview/WdlPreviewService.cs:46) show an owned preview-service boundary. A future WC2 preview must not route through the WoW map loader merely because it displays an image.
- [DAT converter height option](../../../src/core/WowViewer.Core.IO/Maps/DatToLkAdtConverter.cs:11) documents a height divisor of 36. This is narrow code evidence, not a global cross-engine coordinate proof.

## Verification Ownership / Current Outcome

| Intended outcome | Evidence now | Remaining proof |
|---|---|---|
| DOS source selection and map discovery | Actual local inventory, two archive headers/hashes, eight chunk-envelope scans | Entry/resource decode and viewer integration |
| WC2 terrain and sprites | No image produced | Correct resource mapping, decoder tests, real output, operator comparison |
| WC3 viewing, matching, synthesis, terrain bridging | Operator draft/addendum and scoped existing-code inspection only | Real WC3 inputs and all functional/visual/consumer checks |
| Coordinate helper | Supplied examples plus existing DAT height-divisor code | Screenshots, origins, independent anchors, explicit profiles |
| Existing WoW behavior preserved | No runtime source/test file changed | Runtime regression smoke when future implementation exists |

The read-only archive/map/source probe exited 0 after the corrected source lookup. Metadata/hashes are actual output, not generated preview proof. No feature acceptance criterion is closed by this document.
