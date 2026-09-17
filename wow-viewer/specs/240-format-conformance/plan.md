# Implementation Plan: Format Conformance Pass

**Branch**: `v0.5.4-dev` (v0.6 release line) | **Date**: 2026-09-17 | **Spec**: [spec.md](spec.md) | **Release**: v0.6

## Summary

Turn the one-off `inspect casc wmo-survey` into a per-format conformance survey, then close the reader and
renderer gaps in [research.md](research.md) in order of how much real data each touches. Survey first,
fix second, resurvey third; every fix carries before/after counts.

## Technical Context

**Language**: C# / .NET 10 · **Libraries**: TACTSharp (CASC), DBCD + WoWDBDefs (DB2), Warcraft.NET (M2 chunks, `.skel`), SereniaBLPLib (BLP; BLPSharp is its MIT successor). Recheck upstream freshness before each phase (research.md R1)
**Readers touched**: `WowViewer.Core.IO` — `Converters/WmoV17ToV14Converter.cs`, `Lk/Mcnk.cs`, `Maps/AdtTextureReader.cs`, `Maps/MopAdtChunkParser.cs`, `M2/M2ChunkedFileIds.cs`
**Renderers touched**: `src/viewer/WoWViewer/Rendering/WmoRenderer.cs`, `Terrain/TerrainRenderer.cs`, `Rendering/WowViewerM2RuntimeBridge.cs`
**Survey surface**: `tools/inspect/WowViewer.Tool.Inspect/CascCommandSupport.cs` (existing `wmo-survey`, `map-survey`, `m2`)
**Measurement build**: `wow_classic_beta` 1.60.1.69876 at `I:\wow12\World of Warcraft`, CDN fill cache `output/cache/casc`
**Testing**: xUnit in `tests/WowViewer.Core.Tests` (synthetic fixtures for each new field); survey counts as integration evidence
**Constraints**: pre-6.0 paths unchanged (FR-004); no code copied from unlicensed repositories (FR-011)

## Constitution Check

- Library-first: readers and survey logic in `src/core`; the inspect tool stays a thin CLI. ✅
- Evidence over assumption: every row MEASURED or CODE; wiki claims are hypotheses. ✅
- Base tooling preservation: fixes layer onto the existing converter/renderer paths; no parallel reader. ✅

## Phases

### Phase 0 — Survey foundation (US1)

Move survey logic out of `CascCommandSupport.cs` into `WowViewer.Core.IO/Survey/`:
- `ChunkInventory` (reusable chunk walker + known/skipped/unknown classification per format).
- Field tallies: WMO (batch flags, material shapes, MOTV/MOCV set counts, MOHD flags, split-group headers, MOMX value distributions, MGI2 raw values), ADT (MCNK flags incl. 0x10000, nLayers histogram, MCLY flag bits, MTXP/MHID presence, MODF 0x80), M2 (chunk inventory, SKID/BFID/AFID/LDV1 presence and resolution), BLP (encoding × pixel format × alpha depth), WDT (MPHD flags and FileDataID fields, companion WDT presence).
- CLI: `inspect casc survey --format wmo|adt|m2|blp|wdt|all [--map <wdt id>] [--limit n]`, writing a JSON report plus a text summary into the spec's evidence folder.
- Reproduce research.md WMO counts (SC-001) before any further fix.

### Phase 1 — WMO correctness (US2)

1. MPY2 material ids above 0xFE survive (face-material fallback becomes 16-bit).
2. Second MOTV and MOCV sets carried in the converter model.
3. Per-shader material path in `WmoRenderer`: base + second texture + MOCV-alpha blend for the two-layer family; explicit logged fallback for others (FR-005). Shader-23 blend rule decided from data: compare texture_2/texture_3 coverage against MOCV alpha on the sample WMOs before coding.
4. Split-group parent/child indices read; visibility treats children as reachable through the parent.
5. MOHD flags (do-not-fix vertex colour alpha, liquid type DBC id) applied where the renderer already has the inputs.
6. Record MGI2 and MOMX value distributions; do not render from them until meaning is established.

### Phase 2 — Terrain fields (US3)

1. `high_res_holes`: read the uint64 at the documented offset when flag 0x10000 is set; hole mask becomes 8×8 per chunk in `TerrainTileMeshBuilder.BuildIndices`.
2. MTXP/MHID height blending behind MPHD 0x80, scored with the synthetic-minimap scorecard against authored minimaps before becoming default.
3. MTXF/MCLY texture scale and MCLY animation.
4. MODF 0x80 + MWDR/MWDS doodad sets.

### Phase 3 — M2 renderer audit (US4)

Warcraft.NET (vendored, identical to upstream) parses all modern M2 chunks and `.skel`; animation works.
1. Survey which chunks each model carries and which the render path (`WarcraftNetM2Adapter`, `WowViewerM2RuntimeBridge`) consumes.
2. LDV1 skin LOD selection if the survey shows the renderer ignores it.
3. Investigate the 15 "0 sections" models from map-survey.

### Phase 4 — BLP and WDT companions (US5)

1. BLP pixel-format tally; if BC5 (11) or others appear, evaluate BLPSharp (MIT) as the decoder or add the missing decode.
2. Read MPHD FileDataID fields; survey `_lgt`/`_occ`/`_fogs`/`_mpv` and `_lod.adt` presence; decide rendering per measured use.

### Phase 5 — Reference study (FR-011)

Study WTL (MIT) and WoWFormatLib (no license: behaviour only) for: CASC product/branch handling, listfile and naming (WoWNamingLib), file linking, WMO LOD fallback, M2 chunk handling, ADT `_lod` reading. Record each comparison and outcome in `evidence/reference-study.md`.

## Project Structure

```text
specs/240-format-conformance/
├── spec.md
├── plan.md
├── research.md          # audit table, kept current (FR-010)
├── tasks.md
└── evidence/            # survey reports (JSON + summary), before/after per fix, reference-study.md
src/core/WowViewer.Core.IO/Survey/   # new: ChunkInventory, per-format tallies, report model
```

## Risks

- **Wiki/data disagreement** (MCMT 4-layer note, MGI2 layout): resolved by measurement; recorded in research.md.
- **Shader semantics unknown** (23 and newer): fallback must be explicit and visible in the survey, so wrong guesses cannot hide.
- **Survey cost**: full-product WMO survey takes ~110 s locally; keep per-format `--limit` and map-scoped modes.
