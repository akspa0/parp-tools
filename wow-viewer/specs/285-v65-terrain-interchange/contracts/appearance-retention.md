# Contract: Durable Terrain Appearance Reference

**Operator-directed addition, 2026-10-21. Proposed, not implemented.** v65 is retained reference
data for conversion, later viewing and future models. Keep placements; exclude object meshes/masks.

## Retained evidence and metadata homes

Retain every selected terrain source file/chunk/header/unknown region with file family/band,
original bytes, ordering/duplicate occurrence, offsets/sizes, coordinates, path/FileDataID/backend,
source build/config and hashes. Today's raw collector skips chunks it considers decoded even when
tensorization loses their metadata; v65 must preserve those bytes too. Include all contributing
tile-family bands and relevant WDT/WDL records, not only one selected companion.

Keep raw evidence and decoded/native records separately linked. Unknown chunks still have a home
and searchable provenance inside v65. A new parser creates a versioned interpretation over the
same bytes; it does not replace them. Opaque data is never counted as decoded conversion support.
Known object geometry/masks are excluded; placement/reference metadata remains included.

Committed source/native/resource records survive export and derived-view generation. No automatic
eviction, destructive resume or cleanup-after-export. Stage new revisions and link interpretations
to old evidence. Existing v50/v60 stores remain read-only.

## Native cell/layer/material graph

```text
source → map → tile → chunk → terrain cell → ordered layers → texture identities → PNG resources
                       ↘ masks / UV / blend / effect / material metadata / unknown parameters
texture identity → all layer/cell usages (inverse index)
```

Derive cell topology/basis from the source profile (commonly 8×8 cells and 64×64 alpha samples per
chunk). A cell references every contributing ordered layer and its exact native alpha sample region,
not one authoritative texture ID. Preserve source local texture indices, path/FDID aliases, layer
flags/effect/alpha offsets, original MCAL bytes, decode profile/encoding/fixup status and decoded masks.
Source Alpha versus standard array orientation is explicit. Boundary mappings must agree.

Sequential alpha and effective blend contributions are different. Per-cell averages/dominant IDs
are optional versioned sampling views with source-mask digest; they never replace native masks.
Height blending can require nonlinear shader inputs; retain those parameters and unknown meanings.

Retain all observed terrain appearance inputs:

- MTEX/MDID/MHID identity/slot order and MTXF/MTXP/MCMT/MAMP values/bytes/associations.
- UV basis/repeat/scale/orientation/animation/transform metadata, layer flags and effect/material IDs.
- Referenced diffuse/height/normal/specular/auxiliary maps, channel/color-space/alpha conventions.
- Native MCCV/MCLV/MCSH, liquid appearance/coverage/type, terrain flags and area metadata.
- Relevant terrain/ground-effect/liquid/material/environment table rows with build/schema identity,
  source rows/bytes and dependencies. Ground-effect model IDs are metadata, not stored mesh payloads.
- Lighting/time/config context for observations, separate from authored static data. No claim one
  image reproduces every dynamic renderer state. Unknown parameters stay raw/indexed.

Existing inferred MTXF/MTXP interpretations remain evidence-qualified; no guessed stride/meaning.

## Source texture resources plus PNGs

Resolve **every referenced terrain appearance texture** in selected maps, including auxiliary,
height/liquid maps and relevant table dependencies, through MPQ/loose/CASC. Retain source bytes,
build/content hash, raw path/FDID aliases, roles/usage, dimensions, format/header/compression/palette,
mip inventory, channels/color-space/alpha metadata and unavailable states.

Derive native-resolution RGBA PNGs for available source mips using existing proven decode
primitives behind a preservation adapter. Keep alpha/data-map channels; no forced opaque alpha,
resize, proxy substitutions or unrecorded display/color correction. Store decoder/version/settings,
pixel and PNG hashes. PNG is convenient for models/viewers; source bytes/mips/material metadata
remain authoritative. Missing/failed decode records prevent complete texture-coverage acceptance.

Store source bytes and encoded PNG byte resources inside v65. Deterministic PNG file export and
an identity/usage manifest read those same retained bytes. Use content identity plus source aliases,
not flattened basenames or run order. Deduplicate bytes without merging distinct build/usage metadata.

## Query and future-model contract

Versioned reader APIs and thin `v65_dataset.py` commands provide:

```text
query-cell       --store <v65> --map <id> --tile <x,y> --chunk <x,y> --cell <x,y>
query-texture    --store <v65> --texture <stable-id>
query-chunks     --store <v65> [source/tag/state filters]
export-textures  --store <v65> --output <project-dir>
model-view       --store <v65> --selection <manifest> --view-contract <version>
```

Results preserve source relationships, unknown values and availability. Model views explicitly
declare selected fields, coordinate mapping, normalization/sampling/resolution and native/resource
digests, and never overwrite canonical records. No trainer/new model or viewer UI is implemented now.

## Acceptance and target output

Verify real MPQ/CASC raw byte readback, indexed occurrence coverage, RGBA/mip/dimension/pixel hashes,
basename/alias safety, complete dependency closure, native layer masks/order, cell↔texture lookup,
and unchanged source records after view/export generation. Pixel checks are not visual proof.

Target exporters use this common reference with version-specific reductions/loss reporting.
Retained data does not create every target writer or make all modern effects representable in
older clients. Incomplete or unknown inputs stay explicit and available for later interpretations.
