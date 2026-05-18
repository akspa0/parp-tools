# PROGRESS — V14 Branch (V11 Reset)

## POSITION
- V10 pipeline: dead. Two-stage nonsense with broken archive path.
- V11: v9 proven pipeline + ConvNeXt backbone + MCAL/MCLY multi-task + proper training infra.
- V14: wow-viewer library completeness effort, modular terrain model system, and a now-proven harvest/tensor-pack lane from Alpha `0.5.x` through `4.0.0`.

## VALIDATED
| What | Status |
|------|--------|
| MpqArchiveCatalog probe fix | DONE — break→continue, 256 limit |
| MCAL/MCLY in v9 BuildDirectCacheEntry | DONE |
| V11 model forward pass | DONE — 35.5M, 6 output heads |
| V11 training loop (GPU) | DONE — AMP, EMA, cosine, uncertainty loss |
| V11 inference + OBJ export | DONE |
| Channel layout audit | DONE — 26ch, shadow removed, MCCV at 3x dropout |
| Cache memory (LRU 2GB cap) | DONE |
| Zero-samples / empty vocab guards | DONE |
| wow-viewer Phase A: Terrain type system | DONE |
| wow-viewer: NativeMpqService port | DONE — pure C#, no StormLib |
| wow-viewer: AlphaTileData.ToTileLoadResult | DONE |
| wow-viewer: TerrainTileTensorPack.ToTileLoadResult | DONE |
| wow-viewer: Harvest tool extract-unified | DONE |
| wow-viewer: MDDF/MODF model name resolution | DONE |
| wow-viewer: Harvest --export-placements | DONE |
| wow-viewer: V16 archive split-ADT placement staging | DONE — archive-backed harvest now stages `_obj0.adt`, restoring placements, object masks, and instance masks on split ADT builds |
| wow-viewer: Alpha object/precise masks | DONE |
| wow-viewer: Alpha shadow residual mask | DONE |
| **Alpha 0.5.3/0.5.5: all 11 signals** | **DONE** |
| **Retail 3.x: all 11 signals** | **DONE** |
| **Cata 4.0.0: 12 signals** | **DONE** — + MCCV vertex colors |
| **0.7.0 extraction** | **DONE** — AdtProfile0703694 |
| **Alpha object mask projection** | **DONE** |
| **Minimap via Md5TranslateResolver** | **DONE** |
| **All staged clients pass** | **DONE** — 0.5.3, 0.5.5, 0.7.0, 3.0.1, 3.3.5, 4.0.0 |
| **Placement flat arrays in NPZ** | **DONE** |
| **Placement model names resolved** | **DONE** |
| **BuildKey provenance** | **DONE** |
| **WL* loose-file liquid fallback** | **DONE** |
| **Unified liquid sea-level presence fix** | **DONE** — `height == 0` no longer suppresses valid MH2O/MCLQ water |
| **Archive temp map-name fallback** | **DONE** — staged archive temp extraction now preserves real `map_name` metadata |
| **V16 builder live progress + fail-loud stream handling** | **DONE** — `build_v16_dataset.py` now forwards harvester stderr live, prints early progress for small maps, and throws explicit errors on stream/process/finalization failures |
| **V16 staged finalization guard** | **DONE** — builds now write to `.zarr.partial` and only promote to final `.zarr` on success, so failed runs no longer silently poison the canonical dataset path |
| **V16 WDT-driven auto map discovery** | **DONE** — hard-coded default map eras are gone; `WowViewer.Tool.Harvest discover-maps` now uses WDT `MWMO/MONM` plus archive probe checks for a real V16-usable tile (`height_257` + `minimap_rgb_256`), so the builder skips WMO-only, zero-tile, missing-WDT transport, and terrain-but-no-V16-usable-tile maps automatically |
| **V16 zero-usable map skip guard** | **DONE** — if a discovered map still yields zero usable V16 tiles during streaming, `build_v16_dataset.py` now warns and skips that map instead of aborting the whole build; the build still fails loud if all requested maps produce zero usable tiles |
| **V16 stats command repair** | **DONE** — `build_v16_dataset.py stats` now uses `pyarrow.compute.sum` for Parquet `has_*` counts and suppresses harmless Zarr sidecar warnings from `index.parquet` / `placements.parquet` |
| **V16 rejected-tile sidecar report** | **DONE** — tiles dropped for missing required dataset keys now land in `wow-viewer/output/datasets/v16/<build>.rejected_tiles.jsonl`, and per-map summaries now print `dropped_missing_required=<n>` instead of forcing operators to recover that truth from scrolled stderr warnings |
| **Harvest WL archive discovery fix** | **DONE** — archive-backed harvest now finds real `*.wl*` virtual files from MPQ listfiles instead of guessing `World\\Maps\\<map>\\<map>.wl*`; focused smoke on staged `3_3_5_12340 / Azeroth` reports no WL entries in the loaded archives |
| **V16 harvest recovery plan recorded** | **DONE** — `docs/architecture/v16-harvest-recovery-plan-2026-05-17.md` now scopes the next bounded fixes: remove archive temp ADT extraction, add map-level resume, and switch future V16 builds to faster Zarr defaults without changing the reader schema |
| **V16 archive in-memory harvest path** | **DONE (UNVALIDATED)** — archive-backed `root` / `_tex0` / `_obj0` ADT families now route through `AdtTensorPackBuilder.BuildFromBytes(...)` instead of `%TEMP%` staging; operator rebuild proof is still pending because this chat did not run builds |
| **V16 partial-build resume state** | **DONE (UNVALIDATED)** — `build_v16_dataset.py build --resume` now reuses `<build>.zarr.partial`, skips completed maps from `_resume_state.json`, and preserves the existing schema/reader contract |
| **V16 resume bootstrap fallback** | **DONE (UNVALIDATED)** — `--resume` now falls back to a fresh staged build when `_resume_state.json` does not exist yet, instead of failing because a just-created partial directory is non-empty |
| **V16 faster default codec profile** | **DONE (UNVALIDATED)** — new builds default to Blosc `lz4` level `1` with `shuffle`; older completed stores remain valid |
| **V16 completed-store restart guard + backfill script** | **DONE (UNVALIDATED)** — build commands now skip already-complete final `<build>.zarr` stores unless `--rebuild-existing` is passed, successful new final stores retain `_resume_state.json`, and `scripts/backfill_v16_resume_state.py` can add `_resume_state.json` to older completed final stores |
| **V16 dataset inspect/backfill tool** | **DONE (UNVALIDATED)** — `scripts/inspect_v16_dataset.py` now writes human-friendly JSON summaries, sample-tile metadata, optional contact-sheet PNGs, and optional `_dataset_summary.json` backfill files for existing V16 stores |
| **V16 trainer-readiness validator** | **DONE (UNVALIDATED)** — `scripts/validate_v16_training_ready.py` now validates finalized stores through `V16Dataset`, a real `DataLoader` batch, and an optional `V15Model` forward pass so trainer-readable truth is separate from dataset-build truth |
| **V16 MCLY training wiring** | **DONE (UNVALIDATED)** — `V16Dataset` now exposes `mcly_ids` / `mcly_mask` from Zarr and `train_v16.py` now uses the same masked cross-entropy supervision path the NPZ-based V15 trainer already used instead of a zero-loss placeholder |
| **V16 index coordinate repair path** | **DONE (UNVALIDATED)** — future streamed NPZ metadata now carries explicit `tile_x` / `tile_y`, and `build_v16_dataset.py repair-index` can rewrite existing `index.parquet` files in place from a metadata-only re-stream instead of forcing full dataset rebuilds |
| **wow-viewer README V16 routing refresh** | **DONE** — the repo-level README now presents V16 dataset generation, `repair-index`, trainer-readiness validation, and `train_v16.py` as first-class workflow surfaces instead of leaving that lane buried under older NPZ/converter framing |
| **V16 inference pairing spec (input->pred.zarr)** | **DONE (DOC SPEC)** — `v16-terrain-model-spec-2026-05-16.md` now defines deterministic paired input/output Zarr contracts (`v16/<build>.zarr` -> `v16_inference/<run>/<build>.pred.zarr`) plus planned ADT patch/synthesis tooling seams |
| **V16 infer_v16 bridge script** | **DONE (FOCUSED PROOF)** — `wow-viewer/data-harvester/scripts/infer_v16.py` now runs deterministic V16 inference from `<build>.zarr`, writes `<build>.pred.zarr`, and emits per-tile patch-ready summaries (`inference_summary.json` + `predicted_height_257.npy`) compatible with `terrain-patch-adt`; syntax proof: `uv run python -m py_compile scripts/infer_v16.py` |
| **V16 tooling-surface truth correction** | **DONE (DOC SPEC)** — V16 spec now marks `terrain-patch-adt`, `convert-lk-to-alpha`, and `convert-alpha-to-lk` as implemented surfaces, with remaining work focused on ergonomic wrappers rather than missing core tooling |
| **V16 training-contract matrix sync** | **DONE (DOC SPEC)** — `v16-terrain-model-spec-2026-05-16.md` now has an explicit spec-vs-code matrix for loader/model/trainer/validator surfaces, supervised targets, unsupervised-but-present signals, and corrected model-file ownership (`v15_model.py` as current V16 architecture host) |
| **V16 focused training-readiness run (3_3_5_12340)** | **DONE (FOCUSED PROOF)** — `uv run python scripts/validate_v16_training_ready.py --build 3_3_5_12340 --train-samples 8 --val-samples 4 --batch-size 2` wrote `wow-viewer/output/datasets/v16/validation/3_3_5_12340.training_readiness.json` with `overall_ok=true`, `issues=0`, and expected model forward output shapes |
| **V16 liquid-height training supervision** | **DONE (IMPLEMENTED + FOCUSED PROOF)** — `V16Dataset` now exposes `liquid_height`, `V15Model` now has a dedicated liquid-height head, and `train_v16.py` now trains liquid height with liquid-presence masking; focused validation rerun passed with `overall_ok=true` and `issues=0` |
| **V16 liquid inference sidecars for ADT liquid follow-up** | **DONE (IMPLEMENTED)** — `infer_v16.py` now emits `liquid_pred_height_256` in `.pred.zarr` plus per-tile `predicted_liquid_mask_256.npy` / `predicted_liquid_height_256.npy` sidecars so downstream MH2O/MCLQ write tooling has prediction inputs |
| **V16 trainer subset curation + evidence chain + labeled validation overview** | **DONE (IMPLEMENTED + SMOKE PROOF)** — `train_v16.py` now curates deterministic subsets from V16 splits (`--train-max-tiles`, `--val-max-tiles`, `--curation-seed`), writes run-local evidence manifests/JSONL selections/per-epoch sampler order logs, and exports `validation_overview.png` with labeled input/output panels; smoke run: `uv run python scripts/train_v16.py --builds 3_3_5_12340 --epochs 1 --batch-size 2 --device cpu --train-max-tiles 8 --val-max-tiles 4 --val-interval 1 --val-snapshots 1 --run-name smoke_v16_curation_evidence` |
| **V16 CPU compile guard** | **DONE (IMPLEMENTED + SMOKE PROOF)** — trainer now enables `torch.compile` only on CUDA by default and supports `--no-compile`, avoiding CPU `cl.exe` hard-fail during local smoke runs |
| **V16 stats compression reporting** | **DONE (UNVALIDATED)** — `build_v16_dataset.py stats` now reports logical raw array bytes versus on-disk Zarr bytes, including per-array ratios and whole-store savings |
| **V16 Windows chunk-write retry hardening** | **DONE (UNVALIDATED)** — `build_v16_dataset.py` now retries transient `WinError 5` / `WinError 32` Zarr `LocalStore` write failures with bounded backoff instead of aborting the build on the first atomic-replace race |
| **V16 batched tile-array writes** | **DONE (UNVALIDATED)** — the Python writer now buffers tile arrays in memory and flushes them to Zarr as small slice batches instead of one row at a time, reducing chunk rewrite churn and filesystem pressure |
| **V16 batch-shape coercion hardening** | **DONE (UNVALIDATED)** — incoming fixed-shape signals are now padded/truncated into canonical Zarr array shapes before batching so resume/build runs do not fail `np.stack(...)` on variable layer-count payloads |
| **V16 placement catalog reuse** | **DONE (UNVALIDATED)** — the C# builder now reads the placement catalog once per tile and reuses it for both object-mask generation and `placement_*_data` export instead of reparsing the same tile placements twice |
| **Coordinate fixes** | **DONE** |
| **Phase C: AlphaToLk writer infrastructure** | **DONE** — WdlWriter, LkWdtWriter, LkAdtWriter, AlphaToLkConverter |
| **Phase C: AlphaToLk CLI command** | **DONE** — convert-alpha-to-lk in WowViewer.Tool.Converter |
| **Phase C: AlphaToLk real-data tile conversion validation** | **DONE** — 755/755 Azeroth (0.5.5), 972/972 Kalimdor, 256/256 EmeraldDream, 25/25 PVPZone01, 25/25 Shadowfang |
| **Phase C: LkToAlpha reverse converter + writer path** | **DONE** — `LkToAlphaConverter`, `AlphaWdtWriter`, `convert-lk-to-alpha` |
| **Phase C: LkToAlpha focused round-trip validation** | **DONE** — `LkToAlphaRoundTripTests` prove Alpha writer structure and `MH2O <-> MCLQ -> MH2O` parity |
| **Phase C: LkToAlpha real-data MdxViewer render** | **DONE** — 839/839 tiles, terrain + WMOs + MCLQ render, ExitCode=0 |
| **Phase C: LkToAlpha asset name fixup** | **DONE** — `--target-client-root` filters 244k+ missing placements |
| **Phase C: LkToAlpha tileset bundling** | **DONE** — `--bundle-tilesets` extracts 327 textures, fixes up paths |
| **Phase C: LkToAlpha model bundling** | **DONE** — `--bundle-m2s` now bundles doodad outputs as local MDX assets, rewrites placement paths, and rewrites bundled MDX TEXS entries to colocated local BLPs |
| **Phase C: LkToAlpha AlphaWdtWriter structural parity** | **DONE** — MAIN row-major per 0.5.3 client, always emit MCNK+MCRF, FourCC I/O convention |
| **Phase C: Placement orientation proof** | **DONE** — Ghidra-confirmed Alpha MDDF/MODF position and rotation transforms; LK writer MODF bounds now round-trip through the shared reader |
| **Phase C: AlphaWDT placement and MCRF stabilization** | **DONE** — reverse AreaID mapping, target-backed asset filtering, top-level chunk contiguity, raw-file rotation convention, and single-owner doodad chunk assignment are all landed in the shared wow-viewer alphaWDT path |
| **Phase C: AlphaWDT MCNK size hardening** | **DONE** — Ghidra-backed client limit `chunkInfo->size < 15000` is now enforced in `AlphaWdtWriter` by trimming duplicate non-anchor WMO refs before emit and throwing on irreducible overflow |
| **Phase C: ADT shard raw-chunk preservation** | **DONE** — unconsumed ADT-family chunks now persist into NPZ shards as raw uint8 blobs with metadata instead of being dropped outright |
| **Phase C: ADT preservation signal promotion** | **DONE** — `MAMP`, `MFBO`, `MCMT`, `MCLV`, `MCSE`, `MCRF`, `MCRD`, and `MCRW` now persist as first-class NPZ entries rather than raw-only fallback blobs; `MCSE` currently retains raw fallback beside the typed signals, and staged real-data `MCSE`/`MCRF` coverage is broadened smoke rather than a pinned positive regression |
| **Phase C: Alpha shard raw-chunk preservation** | **DONE** — Alpha tile tensor packs now preserve raw embedded tile chunks under `raw_chunks/alpha/...` alongside decoded signals |
| **Phase C: Wmo v17->v14 converter** | **DONE** — shared `wow-viewer` downgrade path plus converter CLI and focused regression coverage are landed; large sources now merge past the Alpha-era `384` group ceiling with spatial buckets and portal remap instead of blunt overflow collapse |
| **Phase C: Wmo v14->v17 converter** | **DONE** — shared `wow-viewer` upgrade path plus converter CLI and focused tests are landed |
| **Phase C: M2->MDX minimal converter** | **DONE (STRUCTURAL, ANIMATED BONE TRACKS LANDED)** — shared `wow-viewer` downgrade lane is landed and reader-validated for geometry, materials, textures, sequence headers, bones, pivots, skin-fed index data, classic bone transform tracks (`KGTR`/`KGRT`/`KGSC`), `GLBS`, and companion external M2 `.anim` ingestion, but Alpha-runtime parity beyond bone transforms remains open |
| **Phase C: MDX->M2 minimal converter** | **DONE** — shared `wow-viewer` minimal upgrade lane is landed, emits a strict `MD20` root plus `00.skin`, has focused reader-backed regression coverage, and is wired into `WowViewer.Tool.Converter` |

## IN PROGRESS
| What | Status |
|------|--------|
| Multi-client full shard dataset prep | SWITCHED TO HARVEST PATH — use `WowViewer.Tool.Harvest harvest-map-mpq` on staged clients, not converter `dataset-scan` manifests |
| V16 harvest recovery implementation | PROOF PENDING — code is landed for in-memory archive harvest, map-level resume, resume bootstrap fallback, completed-store skip guards, backfill tooling, dataset inspection/backfill, stats compression reporting, Windows chunk-write retries, batched tile writes, batch-shape coercion, placement-catalog reuse, and faster default codec settings, but this chat did not run user-blocked rebuild validation |
| V16 trainer-readiness proof | PROOF PENDING — the validation script is landed, but this chat did not run it against the finished V16 stores because the user asked not to have commands run for them |
| data-harvester launcher posture | CANONICAL UV RESTORED — elevated proof on 2026-05-16 showed `.venv\\Scripts\\python.exe` and `uv run` both work in a real shell; `scripts/run-data-harvester-python.ps1` remains a sandbox/AppData-access fallback, not the primary operator path |
| 3_3_5_12340 V16 final store health | DONE — full rebuild landed at `wow-viewer/output/datasets/v16/3_3_5_12340.zarr` with `5134` tiles, `index.parquet` `5134` rows, `placements.parquet` `1,015,470` rows, and clean `stats` output |
| Phase C: AlphaToLk AreaID crosswalk | NOT YET — `AreaIdMapper` exists in `WowViewer.Core.IO/Dbc/`, not yet wired to converter |
| Phase C: LkToAlpha real-data MdxViewer validation | DONE — 839/839 tiles, terrain renders, WMOs load, ExitCode=0 |
| Phase C: Alpha/LK full chunk preservation | OPEN — current converter lane is still a reduced terrain-domain reconstruction, not chunk-for-chunk spec closure. Gap inventory documented below. |
| Phase C: Wmo v14↔v17 converters | DONE — shared writers, CLI wiring, and focused tests are landed in `wow-viewer` |
| Phase C: M2->MDX CLI wiring | DONE — `convert-m2-to-mdx` is now wired in `WowViewer.Tool.Converter` and the converter project builds cleanly |
| Phase C: Mdx->M2 minimal lane | DONE — shared converter, focused regression, and `convert-mdx-to-m2` CLI wiring are landed; broader real-data parity proof is still open |
| Phase C: Legacy MdxViewer → wow-viewer port | LONG-RANGE — `WowViewer.App` shell exists but needs world-session, terrain, WMO, M2 rendering

## CHUNK PRESERVATION GAP INVENTORY (2026-05-08)

### Both converters (AlphaToLk and LkToAlpha) drop these chunks:

| Source Chunk | Where | Data Survives? | Notes |
|---|---|---|---|
| **MFBO** | ADT root top-level | NO | Flight bounding planes are read by inspect/harvest but the converters never carry MFBO to output |
| **MCCV** | MCNK sub-chunk | NO | Vertex colors are read by inspect/harvest (`mccv_rgb` signal) but converters drop them |
| **MCLV** | MCNK sub-chunk | NO | Baked vertex lighting is read by harvest (`mclv_lighting_bytes`) but converters drop it |
| **AreaId** | MCNK header field | NO | Always defaults to 0 in output. AreaIdMapper exists but is not wired |
| **MCSH** | MCNK sub-chunk (Alpha) | PARTIAL | Alpha MCSH is 1-bit-per-pixel (512 bytes); LK MCSH is 1-byte-per-pixel (4096 bytes). AlphaToLk upsamples; LkToAlpha downsamples. Format is correct but re-encoded, not byte-identical |
| **MCAL** | MCNK sub-chunk | PARTIAL | Alpha uses 4-bit packed (2048 bytes); LK uses big-alpha or compressed. Conversion re-encodes correctly but not byte-identical |
| **MCVT** | MCNK sub-chunk | PARTIAL | Alpha interleaves outer/inner; LK uses sequential 145 floats. Layout is reorganized, not byte-identical |
| **MCNR** | MCNK sub-chunk | PARTIAL | Alpha uses Z-up component order; LK uses Y-up. Components are swapped, not byte-identical |

### LkToAlpha additionally drops these chunks:

| Source Chunk | Where | Data Survives? | Notes |
|---|---|---|---|
| **MCRF** | MCNK sub-chunk | PARTIAL | Per-chunk refs are now written into Alpha MCNK. Exact LK multi-parent doodad provenance is intentionally collapsed to one owner chunk for Alpha stability, while WMOs still keep overlap-based multi-chunk refs. |
| **MH2O fidelity** | ADT root top-level | PARTIAL | Only min/max height, 9x9 vertex heights, and 8x8 tile flags survive. Lost: fishableMask, deepMask, liquidTypeId beyond basic type, vertexFormat, depth data, UV coords |
| **MTXF** | tex0 top-level | NO | Texture animation/transformation flags are never carried |
| **MAMP** | tex0 top-level | NO | Terrain texture sizing parameter (never needed for Alpha conversion which doesn't produce split ADTs) |
| **MMID/MWID** | obj0 top-level | NO | Name offset indexes. The string name data (MMDX/MWMO) IS carried, but the offset index tables are rebuilt from scratch |

### AlphaToLk additionally drops these chunks:

| Source Chunk | Where | Data Survives? | Notes |
|---|---|---|---|
| **MCLQ (Alpha source)** | MCNK sub-chunk | PARTIAL | Alpha liquid heights/flags are carried into MH2O format in LK output, but Alpha MCLQ's per-vertex liquid data is simplified |
| **MDNM/MONM (Alpha source)** | WDT top-level | PARTIAL | Model and WMO names ARE resolved and carried into LK MMDX/MMID/MWMO/MWID, but the original Alpha string table offsets are lost |

### Priority order for fixes:

1. **AreaId** — already have AreaIdMapper, just needs wiring (medium scope, high value)
2. **Exact doodad owner provenance** — the shared alphaWDT path already writes MCRF, but remaining edge work is extent-aware owner choice for large doodads near chunk borders (medium scope)
3. **MFBO** — read by harvest already, just needs pass-through in converters (small scope)
4. **MCCV/MCLV** — read by harvest already, need writer support in both converters (medium scope)
5. **MH2O full fidelity** — significant rewrite of liquid path, carry more fields (large scope)
6. **MTXF** — small chunk, straightforward carry (small scope)
7. **MAMP** — irrelevant for Alpha conversion but matters for split-ADT LK output (small scope)

## ALPHA WDT FORMAT — CURRENT OWNERSHIP AND LIVE RULES (2026-05-11)

### Keep these reverted lessons
Both commits after `47cbb435` were reverted because they broke 0.5.3 client map rendering:
- **`8bcb7045`** "fix: write MCRF as raw uint32 data (not FourCC-wrapped)" — REVERTED. Ghidra confirms MCRF is raw uint32 in the 0.5.3 client, but writing it as raw broke the client. The MCRF FourCC-wrapper path currently works; the raw-uint32 Ghidra finding may apply to a different version or need more investigation.
- **`d52bda9b`** "fix: alpha MCNR normal encoding and liquid flag regeneration" — REVERTED. Two changes in one commit:
  1. **MCNR encoding changed from `(X, Z, Y)` to `(-Y, Z, -X)`**: Ghidra confirms the client decodes MCNR as `(-Y, Z, -X)`, but changing the writer to this format broke rendering. The `(X, Z, Y)` format (which matches the LK convention) currently produces working output. This may indicate the alpha client also reads MCNR in `(X, Z, Y)` order from some code path, or there is a coordinate system mismatch elsewhere that the old format happened to mask.
  2. **Liquid flags changed from `McnkFlags & 0x3C` to `ClassifyAlphaLiquidType` switch**: The new logic produced flags like `0x04` (water only) instead of `0x3C` (all liquid bits), which may have changed liquid rendering behavior.

### Current working state
- MCNR: written as `(X, Z, Y)` per byte — `byte[0]=X, byte[1]=Z, byte[2]=Y`
- MCRF: wrapped in FourCC chunk (`MCRF` + size + data)
- Liquid flags: `McnkFlags & 0x3C` (preserves original 4.x flags with only liquid bits)
- MDDF/MODF position: `file = (MapOrigin - world.Y, world.Z, MapOrigin - world.X)` — Ghidra-verified
- Shared rotation convention in wow-viewer domain models is raw-file-space `Rotation = (fileRotX, fileRotZ, fileRotY)`. `AlphaWdtReader`, `AlphaWdtWriter`, and `LkAdtWriter` all use that convention, and the writer no longer subtracts 180 degrees from yaw.
- MODF bounds: `file.t = (MapOrigin - world.min.Y, world.max.Z, MapOrigin - world.min.X)`, `file.b = (MapOrigin - world.max.Y, world.min.Z, MapOrigin - world.max.X)` — Ghidra-verified
- Top-level alphaWDT chunks are contiguous; odd-byte padding between chunks is invalid.
- Doodad chunk ownership is single-owner only: containing chunk first, preserved LK refs only when they stay in the containing chunk's local `3x3` neighborhood.
- WMO chunk refs still use bounds overlap and can remain multi-chunk.
- Target-client presence checks must be built from target archives, Alpha wrapper scan, and loose files only; external listfiles are not proof of asset existence.
- alphaWDT read/write ownership lives in `wow-viewer` shared I/O. `MdxViewer` is a compatibility consumer, not the format owner.

### Still open
- **Broader LK/Cata corpus validation**: the shared alphaWDT path has focused tests and the staged Azeroth proof, but not broad native LK/Cata batch signoff yet.
- **Exact doodad-border ownership**: the current containing-chunk rule fixed the worst culling failure, but remaining edge cases may still need extent-aware owner choice for large doodads near chunk borders.
- **MCNR explanation gap**: Ghidra still points to `(-Y, Z, -X)` while the working writer uses `(X, Z, Y)`. Keep the current working bytes until the mismatch is explained, not guessed.
- **Forward AlphaToLk AreaID wiring**: reverse LkToAlpha mapping is done; the forward converter still needs its own crosswalk hookup.

### Ghidra-verified MDDF/MODF details (still correct, not dependent on reverted changes)
- SMDoodadDef size = 0x24 (36 bytes): nameId(0x00), uniqueId(0x04), pos(0x08), rot(0x14), scale(0x20), flags(0x22)
- SMMapObjDef size = 0x40 (64 bytes): nameId(0x00), uniqueId(0x04), pos(0x08), rot(0x14), extents.t(0x20), extents.b(0x2C), flags(0x38), doodadSet(0x3A), nameSet(0x3C), scale(0x3E)
- Client position transform: `world = (MapOrigin - file.Z, MapOrigin - file.X, file.Y)`
- Client rotation transform: `world_rot = (file_rot.Z × π/180, file_rot.X × π/180, file_rot.Y × π/180 + π)`
- Client bounds transform: `world_min = (-extents.t.Z + MapOrigin, -extents.t.X + MapOrigin, extents.b.Y)`, `world_max = (-extents.b.Z + MapOrigin, -extents.b.X + MapOrigin, extents.t.Y)`
- Scale: MDDF scale = uint16 × (1/1024), MODF scale = uint16 × (1/1024) at offset 0x3E (alpha padding, not scaling)
- MCRF in `CreateRefs` passed `(17066.666, 17066.666, 0.0)` as position offset to `CreateDoodadDef`/`CreateMapObjDef`
- `wow-viewer` shared alpha placements intentionally store raw file rotation in the round-trip-safe convention `(fileRotX, fileRotZ, fileRotY)`; renderer-space conversion belongs in compatibility/runtime bridges, not in another parser.

## NOT YET
- Explicit Alpha 0.6.0 split ADT validation via `AdtProfile060070Baseline`
- Full extraction run on 6 staged game clients (800-1500+ shards) via `harvest-map-mpq` into `wow-viewer/output/datasets/`
- Production training run (300 epochs)
- Model evaluation on held-out tiles
- DBC/DB2 metadata enrichment (WorldSafeLocs, AreaTable, GroundEffects, LiquidType)
- Exact MCRF doodad provenance for large cross-chunk placements
- MODF doodadSet/nameSet resolution
- PM4 masks for development map build (4.0.0.12304 loose files)
- Development map extraction pipeline (wow-viewer/test_data/original_development)

## BUGS FIXED IN ALPHAtoLK VALIDATION (2026-05-08)
1. **ChunkedFileReader crash on monolithic WDTs** — replaced with `AlphaWdtReader.ReadExistingTiles()`
2. **MHDR/MCIN empty payload** — wrote declared-size chunk headers with 0 data bytes; fixed by writing pre-allocated zero arrays
3. **MPHD size mismatch** — wrote 9 uint32s (36 bytes) but declared 32 bytes; fixed by removing extra `Write(0u)`
4. **MAIN index formula** — `tileX * 64 + tileY` was wrong; fixed to `tileY * 64 + tileX` (row-major with y as row). This caused 420/755 Azeroth tiles to fail before the fix.

## VALIDATED: LKTOALPHA REAL-DATA MDXVIEWER RENDER (2026-05-09)
- 4.0.0 Azeroth (839 tiles) → Alpha WDT → MdxViewer with 0.5.3 staged client
- Terrain renders: up to 11 tiles (2816 chunks) simultaneously
- WMOs resolved via Alpha `.wmo.MPQ` wrappers (v14 format)
- MCLQ liquid data parsed and rendered
- WDL parsed correctly: 839/4096 tiles with MARE data
- ExitCode=0, capture saved

## BUGS FIXED IN LKTOALPHA ALPHAWDTWRITER STRUCTURAL REPAIR (2026-05-09)
1. **MAIN grid order confirmed** — Ghidra shows 0.5.3 `CMap::PrepareArea(x,y)` indexes `areaInfo` as `tileY * 64 + tileX` after `CMap::LoadWdt` reads raw `MAIN`; `AlphaWdtWriter`, `AlphaWdtReader`, and focused tests use that row-major contract.
2. **Empty MCNK omission** — Writer skipped MCNKs with all-zero heights via `BuildEmptyMcnk`, but legacy `McnkAlpha` always reads 256 MCNK entries. Removed `BuildEmptyMcnk` — all 256 MCNKs are now emitted with full subchunk structure.
3. **MCRF conditionally emitted** — Writer only emitted MCRF when `mcrfRaw.Length > 0`. Legacy `McnkAlpha` unconditionally reads MCRF at offset 0x24. Fixed by always wrapping MCRF (even with 0-byte payload).
4. **MDDF/MODF conditionally emitted** — 0.5.3 `CMapArea::Create` unconditionally asserts both embedded tile placement chunks. Fixed `AlphaWdtWriter` to emit empty `MDDF` and `MODF` chunk headers when a tile has no placements of that type.
5. **Placement orientation and MODF bounds** — Ghidra confirms Alpha file positions are `(origin - worldY, worldZ, origin - worldX)` and file rotations are consumed as `(fileZ, fileX, fileY + 180deg)` in client axes. `AlphaWdtWriter` matched this contract; `LkAdtWriter` bounds axes were corrected and covered by `LkAdtWriter_RoundTripsModfBoundsWithReaderOrientation`.
6. **MCLY/MCAL offsets conditionally zeroed** — When `mclyWhole.Length > 0` was false, offset was set to 0, but legacy reader expects non-zero offsets. Fixed by always populating cursor-based offsets.
7. **LkAdtWriter chunk ID byte order** — Used `Encoding.ASCII.GetBytes()` which writes forward-order FourCC. Changed to `FourCC.FromString().ToFileBytes()` which writes the reversed FourCC expected by `wow-viewer`'s I/O boundary convention. This caused `MapFileSummaryReader` to fail detecting ADT family.

## BUGS FIXED IN LKTOALPHA ROUND-TRIP VALIDATION (2026-05-08)
1. **Alpha placeholder chunk payloads** — `AlphaWdtWriter` declared `MPHD`/`MHDR` payload sizes but wrote zero bytes, corrupting all later offsets. Fixed by writing explicit zero-filled payload buffers.
2. **Reverse Alpha terrain heights** — `MCVT` heights were written relative to tile base instead of chunk base. Fixed to match Alpha chunk semantics.
3. **Alpha MCNK offset frame mismatch** — emitted subchunk offsets did not line up with `AlphaWdtReader`. Fixed so written tiles parse as structurally valid Alpha WDTs.
4. **Tile alpha vs chunk alpha mismatch** — the shared `256x256` alpha contract was being reused as if it were already chunk-local `MCAL`. Fixed by resampling during write.
5. **Liquid parity gap** — LK input `MH2O` was dropped before conversion and Alpha return conversion rebuilt only flags. Fixed by carrying structured liquid through the shared models, preserving `MCLQ` 81-sample heights, and emitting real `MH2O` again on the LK writer path.
6. **Heightmap validation gap** — `LkToAlphaConverter` lacked `AlphaWdtReader`'s `FillHeightmapGaps` logic, causing missing heights at chunk borders to fail roundtrip validation. Fixed by porting the logic to the converter pipeline.
7. **Complete Alpha Mask Loss** — `LkAdtWriter` was dropping alpha chunks entirely because `AlphaToLkConverter` extracted data from transposed X/Y indices `[cx, cy, l]` instead of `AlphaWdtReader`'s populated `[cy, cx, l]`. Fixed by correcting array accesses in `AlphaToLkConverter` to perfectly align with `AlphaTileData` layout.
8. **Binary Chunk Tag Corruption** — `LkAdtWriter` used endian-swapping `FourCC.ToFileBytes()` for `MCNK` subchunks (`MCVT`, etc.), breaking binary validation. Fixed to use `ASCII.GetBytes`.

## FOCUSED VALIDATION NOTE (2026-05-11)
- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter LkToAlphaRoundTripTests` currently passes with `13/13` tests.
- Do not describe this as full suite closure: the broader `WowViewer.Core.Tests` run still hits missing `wow-viewer/test_data/development` fixtures and one unrelated pre-existing invalid-data test.

## OBJECT CONVERTER STATUS NOTE (2026-05-11)
- `wow-viewer` now owns both WMO directions in shared I/O: `WmoV17ToV14Converter` and `WmoV14ToV17Converter`, with converter CLI wiring and focused test coverage.
- The `WmoV17ToV14` downgrade path now handles the practical Alpha 0.5.3 ceiling of `384` groups with spatial bucket merging and portal index remap instead of flattening all overflow into one terminal group.
- `wow-viewer/src/core/WowViewer.Core.IO/M2/M2ToMdxConverter.cs` now provides a first minimal `M2 -> MDX` downgrade lane with classic bone transform track export and companion external `.anim` ingestion. Current proof is still reader-backed structure validation rather than broad real-data parity.
- `convert-m2-to-mdx` is now wired in `WowViewer.Tool.Converter` and validated with a focused converter build.
- `wow-viewer/src/core/WowViewer.Core.IO/M2/MdxToM2Converter.cs` now provides the first minimal `MDX -> M2` upgrade lane. Current proof is a synthetic round-trip that re-reads the generated strict `MD20` root and `00.skin` through existing M2 readers, plus converter CLI build validation.
- The next object-converter slice is broad real-data proof for both `M2 -> MDX` and `MDX -> M2` rather than another new ownership seam.

## BUGS FIXED IN NPZ SHARD VALIDATION (2026-05-08)
1. **OverflowException in MCNK sub-chunk locators** — `LocateMcvtDataOffset`, `LocateMcnrDataOffset`, `LocateMccvDataOffset`, and `TryReadSplitMcnkSubchunkPayload` all used `checked((int)header.Size)` casts that threw `OverflowException` when encountering MCNK sub-chunk headers with `uint` sizes > `int.MaxValue`. Fixed by converting all three locators and `TryReadSplitMcnkSubchunkPayload` to `long` arithmetic.
2. **Garbled raw-chunk FourCCs in AdtRawChunkBlobCollector** — `CollectRawMcnkSubchunks` lacked FourCC validation, MCAL/MCSH header-size overrides, and `long` arithmetic. When scanning past valid MCNK sub-chunks, it fabricated 274 garbage "raw chunks" with binary FourCC IDs. Fixed by adding `IsValidAdtFourCC` validation, MCAL/MCSH consumed-size overrides from MCNK header fields at offsets 0x28/0x30, and `long` arithmetic throughout.
3. **Stale AdtTextureFile constructor call** — `WowViewer.Tool.Converter/Program.cs` called `AdtTextureFile(...)` without the `mampValue` parameter added in a prior refactor. Fixed by passing `null`.

## ALPHA RESOLUTION FIX (2026-05-08)
- **NPZ alpha naming mismatch**: `mcal_alpha_pack_256` key stored 1024×1024×4 data (full per-pixel resolution at 64 px/chunk × 16 chunks) but the "256" suffix implied 256×256 resolution. Visualization scripts interpreted it as 256×256, showing a stretched 1024×1024 image with visible 64-pixel chunk-boundary seams.
- **Fix**: Added `McalAlphaPack` (1024×1024 full resolution) as a new property alongside `McalAlphaPack256` (256×256 bilinearly downsampled). `NpzTileSerializer` now writes both `mcal_alpha_pack` and `mcal_alpha_pack_256`. The 256 version is correctly downsampled for minimap-resolution compositing; the full 1024 version preserves per-pixel alpha for training. Updated `AdtTensorPackBuilder`, `AlphaTensorPackBuilder`, `TerrainTileTensorPack.ToTileLoadResult` (now uses `McalAlphaPack` for 64px-per-chunk slice indexing), and harvest tool's synthetic minimap compositor (uses `McalAlphaPack256` for 256×256 compositing, which is correct).
- **Grid artifacts in normals are expected**: MCVT/MCNR store 145 vertices per chunk with shared boundary vertices between neighbors. When rendered as a plain 257×257 image, chunk-boundary edges create visible grid patterns. This is not a bug — it's how the WoW ADT format works. The raw data is correct.

## VALIDATED SHARD STRUCTURE (2026-05-08)
- Azeroth 32,32 from 3.3.5 client via `extract-unified`: 13 typed signals (height_257, mcnr, mcly, mcal, mcsh, hole_mask, object masks, shadow residual, minimap, etc.) plus structurally valid `metadata.json`.
- Raw-chunk collection correctly produces 0 garbled entries (was 274 before FourCC validation fix) for standard LK root MCNKs where all sub-chunks are in the consumed set.
- Build: `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` — 0 errors, 12 warnings (nullable, CA2014).

## WORKFLOW CORRECTION (2026-05-08)
- Do not route full multi-client dataset generation through `WowViewer.Tool.Converter dataset-scan` / `dataset-audit` / `dataset-curate` / `dataset-build-cache`.
- Those commands remain useful as legacy manifest/audit helpers, but they are not the canonical full-signal shard builder and can miss newer harvest/tensor-pack coverage and metrics.
- Use `WowViewer.Tool.Harvest harvest-map-mpq` for staged archive-backed clients and `harvest-map` for loose on-disk maps.
- Default real dataset outputs belong under `wow-viewer/output/datasets/`, not repo-root `output/tmp/`.
