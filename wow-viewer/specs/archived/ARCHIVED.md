# Archived Specs

Specs moved here are obsolete (superseded by later work, replaced by refactored pipelines, or V16/V17 model architectures that proved impractical).

## Per-spec rationale

| Spec | Reason for archiving |
|------|----------------------|
| 001-v16 | V16 model era — superseded by V18 pipeline |
| 002 | V16-era liquid supervision — superseded by V18 |
| 003 | V16 dataset quality fixes — superseded by V18 contracts |
| 004 | MdxViewer sidebar — legacy viewer, not part of WoWViewer |
| 006 | V16.1 model family — too complex, never converged |
| 007 | V16.1.1 acceleration — never converged |
| 008 | Synthetic guidance model — V16 era, never practical |
| 010 | V16.1.2 no-object guidance — V16 era |
| 011 | V16.2 patch expansion — V16 era |
| 015 | V16.1.2 refiner — dead code, never trained |
| 016 | V16.1.3 height-channel model — plateaued, superseded |
| 017-mdxviewer | Headless capture port — replaced by WoWViewer's built-in automation |
| 017-v16 | V16.1.4 combined model — never converged |
| 021 | Cross-signal curation — V16/V17 era |
| 022 | V17 refiner — never trained |
| 023 | V17.1 global minimap — V17 era |
| 027 | Object multi-angle LoRA — V16 era |
| 050 | PM4 WMO group matching — consolidated into 046 |
| 052 | PM4 signature matcher — consolidated into 046 |
| 086 | V22 consolidated dataset (per-tile stream) — never produced a populated store; C# three-message-class producer was never written; superseded by 088 (V18 substrate + separate enrich tool) |
| 087 | V22 asset library payloads (per-tile, `Path.GetHashCode()` keys) — non-deterministic keys break cross-run dedup; per-tile design duplicates payloads; superseded by 088 (stable canonical path keys + build-wide library) |
| 119 | Object-library classifier/segmenter — trained and passed its own gates, but the minimap retrieval PoC proved object identity does not survive minimap scale (p50=10px instances, ~0.99 cosine to unrelated blobs); minimap object segmentation/classification abandoned; masks repurposed loss-side in 121 (see CLOSED.md) |
| 120 | Minimap OBB detector / DINOv2 placement retrieval — inherits 119's measured scale-physics failure; retrieval/detection from minimaps abandoned; superseded by 121 (see CLOSED.md) |

| 005 | PM4 workbench cleanup — targets legacy MdxViewer; all PM4 work now in wow-viewer specs (046/058) |
| 020 | Renderer culling — subsumed by 056 (GPU/LOD modernization) |
| 026 | Capture batch tuning — no tasks defined; concerns owned by 056 / validation-capture |
| 036 | Renderer improvements — explicitly superseded by 056 (GPU/LOD modernization) |
| 033 | MdxViewer migration — complete per user 2026-06-14. All viewer/renderer work now in wow-viewer; gillijimproject_refactor is read-only reference. |
| 037 | M2 3.0.1 embedded-views adapter — implemented per user. |
| 041 | MH2O/MCLQ liquid type fix — implemented per user (McnkFlagDecoder + tests exist). |
| 043 | M2 chunked MDLX classic support — implemented per user (MDLX reader/dispatcher landed). |
| 059 | M2 MD20 v109 Cata support — implemented 2026-06-11; Cataclysm M2 objects verified working |

## Completed specs (referenced from archive, available in active specs directory)

Specs 012, 014, 024, 025, 033, 034, 037, 041, 043, 047, 048, 059, 060 are fully complete and live in the active specs directory (or archived after completion).

## Supersession note for 086/087

The contents of `specs/086-v22-consolidated-dataset/` and `specs/087-v22-asset-library-payloads/` remain on disk with a `SUPERSEDED.md` redirect. They are NOT physically moved into `archived/` because:

1. They reference an active branch state and git history for the partial work that was done.
2. Spec 088 directly supersedes them and cites both as predecessors in its `spec.md` "Supersedes" header.
3. The redirect `SUPERSEDED.md` makes the supersession visible to anyone navigating the spec list.

If the user wants physical archival, run:
```
mv specs/086-v22-consolidated-dataset specs/archived/086-v22-consolidated-dataset
mv specs/087-v22-asset-library-payloads specs/archived/087-v22-asset-library-payloads
```
and update the directory paths in `088-v22-enrichment-from-v18/spec.md`. This is intentionally deferred to the user because it requires a git history decision.
