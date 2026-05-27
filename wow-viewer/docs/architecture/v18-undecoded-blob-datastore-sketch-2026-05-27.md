# V18 Datastore Sketch — Undecoded ADT/alphaWDT Blob Preservation (2026-05-27)

## Scope

Define a bounded V18 extension that preserves undecoded binary payloads from ADT/alphaWDT inputs inside the datastore, without rewriting existing readers and without breaking the current V16 contract.

## Why

- Current V16 extraction is intentionally signal-first (decoded supervision arrays + placement/index tables).
- Some chunks/fields remain partially decoded or format-variant-sensitive across client eras.
- For later research and parity work, we need deterministic access to original bytes at tile/chunk granularity.

## Current Contract Update (Landed)

The dataset build path now also writes per-tile decoded metadata coverage into:

- `decoded_metadata.parquet`

for every harvested tile accepted into `index.parquet`.

### `decoded_metadata.parquet` (current)

One row per tile (`tile_id`) with:

- tile identity: `build`, `map`, `tile_x`, `tile_y`, `tile_name`
- source traces: `source_adt_path`, `source_wdt_path`
- decoded-structure summary: `raw_chunks_count`, `decoded_metadata_keys_json`
- full decoded metadata payload snapshot: `decoded_metadata_json`

Validation is now enforced by build/merge/validate flows via `decoded_metadata_validation.json` parity checks.

## Non-Goals

- No rewrite of existing ADT/WDT/alphaWDT readers.
- No immediate consumer-wide migration to a new training contract.
- No requirement to decode every stored blob in this slice.

## Proposed V18 Blob Surface

### 1) Sidecar blob group (preferred)

Add a sidecar hierarchy under each build store:

- `raw_blobs/manifest.parquet`
- `raw_blobs/chunks/` (content-addressed blob payloads)

This keeps decoded arrays/index paths stable while adding opt-in deep provenance.

### 2) Manifest schema (minimum)

Each manifest row:

- `build` (string)
- `map` (string)
- `tile_id` (int64)
- `tile_x` / `tile_y` (int32)
- `source_kind` (enum string: `adt_root`, `adt_tex0`, `adt_obj0`, `adt_lod`, `alpha_wdt`, `alpha_tile`, etc.)
- `chunk_fourcc` (string)
- `chunk_index` (int32 nullable)
- `byte_offset` (int64 nullable)
- `byte_length` (int64)
- `content_hash` (hex sha256)
- `compression` (string: `none`, `zstd`, etc.)
- `decode_status` (string: `decoded`, `partial`, `undecoded`)
- `decode_notes` (string nullable)

### 3) Blob payload layout

- Store payload by `content_hash` (dedupe identical bytes across tiles/builds).
- Suggested path split:
  - `raw_blobs/chunks/<hash[0:2]>/<hash[2:4]>/<hash>.bin`
- Keep payload immutable once written.

## Migration Path (V16 -> V18)

### Phase A (safe add-on)

1. Keep existing `index.parquet`, `placements.parquet`, `decoded_metadata.parquet`, and fixed-shape arrays as the canonical decoded contract.
2. Add optional blob extraction flag in harvester/build flow (off by default).
3. Write `raw_blobs/manifest.parquet` + hash-addressed payloads.

### Phase B (auditing)

1. Add lightweight audit command to summarize undecoded coverage by FourCC/source_kind.
2. Emit per-build report listing top undecoded byte volume contributors.

### Phase C (targeted decode follow-up)

1. Prioritize decode work based on blob-volume + downstream impact.
2. Promote newly decoded signals into canonical arrays/tables while retaining raw blob provenance links.

## Validation Contract

For a build where blob extraction is enabled:

- `raw_blobs/manifest.parquet` exists and row count > 0.
- Every manifest row’s `content_hash` resolves to exactly one blob file.
- File size matches `byte_length` (post-decompression contract explicit in manifest).
- Hash of stored bytes equals `content_hash`.
- Coverage summary report generated per build.

For all builds (blob extraction enabled or not):

- `decoded_metadata.parquet` must exist.
- `decoded_metadata.parquet` row count must equal `index.parquet` row count.
- `decoded_metadata.parquet.tile_id` must be 1:1 with `index.parquet.tile_id`.
- `decoded_metadata_json` must deserialize to a JSON object per row.

## Integration with Current Spec 025 T002 Work

- T002 object-capture ledger/pose work remains placement-driven from `placements.parquet`.
- Blob preservation is complementary: it enables later per-asset forensic decoding and parity verification without changing current capture-batch behavior.

## Risks / Guardrails

- Storage growth risk: mitigated by hash dedupe and optional enablement.
- Performance risk: mitigate with bounded extraction modes (selected FourCCs first).
- Contract drift risk: keep V16 decoded contract untouched in Phase A.

## Immediate Next Bounded Step

Implement a read-only planning/proof command that scans one staged build and emits a dry-run `raw_blobs` manifest preview (no datastore mutation), then lock the final manifest schema before enabling writes.
