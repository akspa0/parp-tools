# Contract: v65 Fields, Coverage and Transport

This is a local pipeline contract, not an HTTP service. Proposed; no profile or reader exists yet.

## Initial code-grounded coverage ledger

| Family | Current decoded/transport state | v65 required change |
|---|---|---|
| MCVT/native lattice | Pack and V22 carry native145; dense257 has interpolated nonvertices | Preserve native145, base/coordinates/presence and source topology as authority. |
| MCNR | Dense normals/validity emitted | Retain native sample/packing provenance; no invented normals presented as source. |
| MCLY/MCAL/MTEX/MTXF | Pack already caps layers at four; V22 reduces alpha to256 and metadata can be lost | Native ragged layers/full alpha, ordered flags/effects/material names/FDIDs, exact table mapping. |
| Hole metadata | Pack/V22 keep only chunk-has-hole boolean | Exact low/high-resolution per-chunk bitmaps and source version. |
| MCLQ/MH2O/WL | Pack has raster families; V22 carries unified cropped256 | Preserve each native layer, dimensions, source types/coverage/heights/depth/flags; derived rasters optional. |
| MCCV/MCLV/MCSH | Pack has decoded values; V22 omits some | Persist faithful native values/availability, separate authored vs derived shadows. |
| MFBO/MCMT/MAMP | Decoded pack fields, omitted by V22 | Add typed profile/store/export fields and target dispositions. |
| Area/MCNK flags/WDT/WDL | Some fields survive; convenience writer sets defaults | Preserve terrain metadata/native WDL; rebuild structural layout only at target output. |
| MCSE | Decoded counts/IDs/positions/raw entries, omitted V22 | Terrain emitter records with source layout/availability; no sound asset/audio harvest scope. |
| MMDX/MMID/MWMO/MWID/MDDF/MODF | Placement state/names exist, float projection loses integer identity | Typed placement metadata inside v65, extracted before float conversion; no geometry. |
| MCRF/MCRD/MCRW + global WMO refs | Pack has some refs; V22 omits them | Stored typed relationship metadata; retain index/uniqueID semantics and ownership for re-itemization. |
| Model/geometry/object target arrays | V22 embeds partial model payloads and masks | Exclude from stream/store; skip object enrichment work for v65. |
| Additional observed chunks | Selected raw blobs exist; not a complete source archive | Retain/index all terrain chunk/header/unknown occurrences; native and opaque states stay distinct; known geometry/masks excluded. |

Every row above is a **coverage requirement**, not proof of completed v65 support. Concrete tags such
as modern texture/height-blend companions, LOD/offset metadata and special liquid chunks enter the
ledger when actually observed. Do not invent a parser result for an uninspected tag.

## Field and exclusion rules

- Native terrain records are interchangeable source semantics; target layer limits, coordinate
  serialization and lossy simplification happen in target adapters, not ingestion.
- Exclusion covers model geometry payloads, mesh libraries, object/instance/footprint/roof masks and
  fragment traces. Typed MDDF/MODF/reference/asset identity metadata remains in v65; equivalent renamed
  geometry/mask fields remain excluded.
- Raw terrain snapshots/chunks retain unknown source evidence without claiming decoded support.
  Placement fields are typed metadata; offsets remain provenance/regenerated output structure.
  Known object geometry/masks cannot be hidden in raw resources.
- A nullable/masked field with reason preserves tile row alignment. No shape substitution, silent
  float dtype fallback, missing-row omission or dedup pointer to a valid row for absent data.

## Versioned bidirectional transport

Existing transport has outer `ARRY + int32 blobLength + innerBlob`, ending `ENDS + 0`. Inner blob
has `ARRY + JSON length/metadata + typed named arrays + ENDS`. Reuse these primitives with v65
schema/profile/layout and record-kind declarations; this is not the separate V22E enrichment format.

Specify checked lengths, dtype widths/endianness (including uint64/int64 where required), shapes,
availability, source/tile keys, field hashes, record ordering and terminal counts/digest. Reject
unsupported dtypes/truncation/schema mismatch. Do not rely on today's Python unknown-dtype→float32
fallback. Forward and reverse directions use the same native semantics.

Python `v65_dataset.py` proposed commands:

Reverse export also carries selected appearance/table metadata and source/PNG resource records
or requested resource frames with checked lengths/hashes, so C# needs neither direct Zarr access
nor the original terrain client for complete retained-resource export. Geometry remains separate.

```text
inspect   --store <v65.zarr>
ingest    --client-root <root> --build <identity> --map <name> --tiles <x,y;x,y>
          [--source-backend <auto|mpq|loose|casc>] [--casc-product <product>]
          --output <new-v65.zarr>
validate  --store <v65.zarr> --report <file>
migrate   --source <old-store> --output <new-v65.zarr> --report <file>
stream    --store <finalized-v65.zarr> --tiles <selection> --schema v65-terrain-interchange-v1
```

The script launches the existing current harvest host/profile for ingestion; it does not duplicate
game format decoding in Python. C# forward harvest and reverse consumers are library services;
CLI/Editor may supervise Python within the declared uv project environment. Structured progress
is separate from binary stdout. Cancellation leaves staging; export requires finalized metadata.

CASC ingestion binds the selected install/product/build and resolves FileDataID dependencies and
companions through the current source catalog. Missing local data or ambiguous product identity
is a structured failure, never an unrelated-client fallback. No CASC output; CDN fill defaults off.

## Dataset-to-converter handoff

The [appearance/retention contract](appearance-retention.md) additionally requires retained raw/native
metadata, cell-layer-material indexes and all referenced terrain source/PNG textures. Unclassified
terrain chunks have a permanent indexed home; missing PNGs/semantics are explicit coverage gaps.

Spec 284's Map request includes store identity/finalization digest, source revision, selection,
coverage requirements, target profile and stored placement-table digests. Source-client
selection resolves an existing validated store or prepares a bounded ingestion job through this
tooling. No silent client fallback when v65 lacks a field. A target client establishes compatibility
and declared dependencies; it is never the output directory.
