# Phase 1 Data Model: V50-Native Height-First Terrain Model

## Frozen Signal Catalog

The existing markdown table in
`docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md` under "Frozen Signal Catalog
(T002)" remains the single human-authored source of truth (constitution: "Spec Docs Are Source of
Truth") — no new catalog file is introduced. A small shared parser
(`harvester.v50.signal_catalog.parse_catalog_table()`) reads that exact table (fixed column order:
`Signal | dtype | Shape | V50 Policy | Required | Notes`) into a list of entries; both the manifest
generator (Decision 3) and its test import this parser, so there is exactly one reader of the table
and it cannot silently diverge from itself.

| Field | Type | Notes |
|---|---|---|
| `name` | str | signal key, e.g. `mcnk_flags_16` |
| `dtype` | str | numpy dtype string |
| `shape` | tuple[int, ...] | row shape, parsed from the `Shape` column |
| `policy` | `copy-if-verified` \| `fresh-only` \| `blacklisted` | existing `MigrationPolicy` values |
| `required` | bool | parsed from the `Required` column's **yes**/no |
| `era_available` | `all` \| set[str] | new: parsed from Notes when it names a build-era restriction (e.g. "MCCV Vertex Colors" implies WotLK+ only); defaults to `all` when Notes states no restriction |

`era_available` is the only genuinely new field; it is inferred by the generator from known
era-restricted signal names (an explicit allow-list in the generator, not free-text parsing of
prose) rather than guessed from arbitrary Notes text — `mccv_rgb` is the one entry in this allow-list
today.

## Rebuilt Per-Map Store (extends existing `v50-complete-store-v1`)

No schema version bump. The store contract is unchanged; what changes is *content*:

- Every signal the manifest template declares (post Decision 3) is either populated with real data
  or absent from the store's array set with a corresponding `UnavailableSignal` entry whose reason
  uses the new prefix vocabulary (below).
- `mcnk_flags_16` carries real per-chunk flag data on rebuilt Kalimdor/Azeroth stores (Decision 1).
- The set of rows with non-empty `minimap_rgb_1024` equals the set of rows with non-empty
  `minimap_rgb` (Decision 2 fix), verified by the new coverage auditor.

### `UnavailableSignal.reason` prefix vocabulary (additive, Decision 4)

| Prefix | Meaning | Example |
|---|---|---|
| `era_unavailable:` | signal cannot exist for this build's era | `era_unavailable: MCCV introduced WotLK+, build is 0.5.3.3368` |
| `no_source_data:` | signal is real for this build/era but this specific tile has no source data | `no_source_data: no MCLY texture layers on this tile` |
| `not_yet_extracted:` | a genuine extraction gap not yet closed (tracked, not silently accepted) | `not_yet_extracted: see Spec 112 research.md Decision 2` |

Existing free-text reasons from Spec 109 (e.g. "no rows passed audit as copy-if-verified") remain
valid and unchanged; the vocabulary is opt-in for new/corrected code paths, not a breaking migration.

## Full-Catalog Curriculum (extends existing `v50-mixed-curriculum-v1`, Spec 109's `training_curriculum.py`)

Two changes to the existing entity from Spec 109:

- **Map allow-list**: `build_training_curriculum` gains a `allowed_maps: frozenset[str] | None`
  parameter; for this lane it is called with `{"Kalimdor", "Azeroth"}`. A source row from any other
  map raises `CurriculumBuildError` rather than being silently skipped or included.
- **Full field set**: `CURRICULUM_FIELDS` (currently the 7-field legacy spec108 list) is replaced by
  "every signal present in the source stores' manifests," i.e. derived per-build rather than
  hardcoded, so a future catalog addition doesn't require another manual field-list edit.

Row/index schema, split assignment (`_stratified_split`, within-map only per the 2026-07-18
correction), and lineage columns (`source_store`, `source_curation_manifest`, `source_group_id`,
etc.) are unchanged from the existing implementation.

## Relative-Height Target Contract (new)

| Field | Type | Notes |
|---|---|---|
| `contract_version` | str | e.g. `"v112.1"`; bumped on any incompatible change to the encode/decode math |
| `normalized_height` | float32[257,257] (or model's working resolution) | `(h - tile_min) / max(tile_max - tile_min, RANGE_FLOOR)`, clipped to `[0, 1]` |
| `tile_min` | float32 scalar | per-tile minimum world-unit height, stored alongside the normalized field |
| `tile_max` | float32 scalar | per-tile maximum world-unit height |
| `RANGE_FLOOR` | float32 constant | denominator floor (world units) preventing near-flat-tile blowup; a fixed constant in the contract, not per-tile |

`decode(normalized_height, tile_min, tile_max) = normalized_height * max(tile_max - tile_min,
RANGE_FLOOR) + tile_min` exactly inverts the encode for any tile, including the flat-tile floor
case (a flat tile encodes as zero and round-trips to its constant height). This pair is what a checkpoint's
`run_identity.json`/training summary must reference by `contract_version` (FR-010), mirroring how
`wdl_prior_train.py` already records `INPUT_CONTRACT`/`TARGET_CONTRACT`.

## Training Run Summary (extends the existing `training_summary.json` pattern from `wdl_prior_train.py`)

| Field | Type | Notes |
|---|---|---|
| `curriculum_identity` | str (content hash) | binds the run to an exact curriculum store build |
| `split_mode` | str | e.g. `within_map_stratified:0.15`, carried through from the curriculum's own `summary.json` |
| `target_contract_version` | str | Relative-Height Target Contract version used |
| `per_epoch_metrics` | list[dict] | existing pattern: composite/point loss per epoch |
| `tile_mean_baseline` | dict | new: the same validation metric computed against a trivial "predict each tile's own mean height" baseline, so SC-004's "beats baseline" claim is self-contained in the summary rather than requiring a separate run |
| `best_epoch` | int | existing pattern; SC-004 requires this to be `> 1` |

The first trainer uses Smooth-L1 point loss plus `0.25 ×` first-difference L1 (horizontal and
vertical). That weight and purpose are fixed for the first run: fit values while retaining relief
topology without adding a second target or model head.
