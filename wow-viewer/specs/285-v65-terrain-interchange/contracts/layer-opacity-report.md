# Contract: MCAL/MCLY Per-Layer Opacity Report

**Operator addition, 2026-10-21. Planning only.** Every declared layer matters, including base,
zero-alpha, repeated-texture, fully overpainted, missing and invalid layers. Retain source evidence.

## Three distinct data levels

1. Source evidence: exact MCLY records, MCAL bytes/ranges, file/band/offset/hash and raw flags.
2. Decoded layer alpha: native-resolution map under the recorded decoder/profile/heuristics/fixups.
3. Effective contribution: the contribution remaining after later layers, under a named supported
   composition model. This is not texture-image alpha or a guarantee of final rendered opacity.

Report metadata for every data level, linking each derived field to its source/decode/model version.
Do not normalize MCAL channels as if they were independent weights. Preserve per-layer identity
before optional aggregation by texture; two layers using one texture still have separate rows.

## Report entities

| Record | Required metadata |
|---|---|
| Report | Schema/report version, source revision/build/backend, file/band/hash, map/tile/basis, decoder/profile/version and composition model. |
| Texture table | Every declared MTEX/MDID slot, including unused slots, raw index/path/FDID, resource resolution, referring layer IDs. |
| Chunk | Chunk key, declared/observed layer counts, payload/header references, raw flags and diagnostics. |
| Layer | Stable source-revision/map/tile/chunk/ordinal key; base/overlay role; texture reference; MCLY flags/effect/alpha offset; source range/hash; encoding and available/consumed bytes. |
| Decode custody | Raw flags versus effective decode decision/reason; complete/partial/missing/failed; source-observed sample mask; edge/nibble/RLE fixups and affected samples. |
| Alpha statistics | Valid/unknown/derived sample counts; min/max/mean/std; zero/full/intermediate counts; defined-bin histogram; explicit-threshold coverage and sample-area semantics. |
| Contribution statistics | Separate effective min/max/mean/sum/histogram/coverage and occlusion status under the named model; never substitute decoded alpha statistics. |
| Maps | Native decoded alpha, effective contribution and validity/provenance masks, all hashed and linked; optional previews are versioned views. |
| Cell summaries | Native cell key/sample footprint and aggregation method; each layer's valid/unknown coverage and decoded/effective statistics; chunk/cell/texel are distinct grids. |
| Completeness | One row per declared layer; reported reasons for unavailable stats; counts of layer records rather than pixel-loop failure counts. |

Histogram bins/normalization/thresholds, quantization and numeric precision are explicit. Integer
alpha maps use source quantization metadata; encoded compressed-byte histograms are not opacity
histograms. Do not average unavailable samples as zero or conflate decoder-filled values with observed
zero. Pre-fixup statistics are unavailable unless independently retained; do not invent their values
from an already-fixed map. Report both source-observed validity and the canonical decoded view.

## Ordinary ordered alpha-over model

For a known opaque base `T0`, overlays in source MCLY order `1..n`, and normalized decoded alpha `a`:

```text
C = T0
C = (1-a1)*C + a1*T1
...
w0 = product(1-aj), j=1..n
wi = ai * product(1-aj), j=i+1..n
```

The base has **implicit** alpha semantics and generally no stored MCAL map. Its effective weight
is the residual, not always one. A one-layer chunk has base contribution one only where the base
and profile are valid. Preserve empty/invalid layer tables as diagnostics, not invented base layers.

Use a named numeric method: floating reference composition or the existing byte-rounding recurrence
with its algorithm version. Complete supported samples must sum to one within declared numeric/
quantization tolerance. Unknown/missing overlays make affected effective results partial/unknown;
do not silently assume zero. Height blending, unresolved flags or other shader semantics keep the
decoded/source report but mark final effective contribution unsupported, or expose a clearly labelled
ordinary-alpha reference view. Never present that reference view as measured shader behavior.

## Retention, queries and export checks

Persist report records/arrays in v65 under versioned derived metadata with native source digests;
index by source/map/tile/chunk/layer/texture/cell. JSON/Parquet and optional grayscale previews are
readable exports of the same records. `report-opacity --store <v65> --selection <manifest>
--output <project-dir>` and `query-cell` expose every layer and its completeness state.

Source reporting has no dependency on a target writer. After target output exists, independently
read it and produce a target report, joining through the explicit source-layer→target-layer map.
Compare layer/texture/flags/effects, alpha/contribution statistics and spatial maps. Every source
layer has a preserved/merged/reduced/blocked disposition, including zero-contribution layers.
Target layer reduction uses known effective contributions and re-encodes target alpha semantics;
never normalize raw channels or drop layers silently. Quantization, mask error and unavailable
semantics are explicit. Keep source data/report unchanged.

## Validation scope

Known ordinary-alpha fixtures verify base-only, overlapping translucent/opaque overlays, repeated
texture slots, zero/full values and layer-order sensitivity. Real MPQ/loose/CASC records verify
decode/profile/offset/fixup provenance, partial payloads and all declared layers. Unknown modern
blend cases must stay honest. Build/tests do not prove source/target rendered appearance.
