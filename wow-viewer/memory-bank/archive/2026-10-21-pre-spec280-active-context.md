# Pre-Spec-280 active-context detail (archived 2026-10-21)

Compressed out of `activeContext.md` when the Spec 280 UI-refactor entry was added and the
dashboard was brought back under the ~240-line limit. This is history, not live state. The
authority for each item is the linked spec receipt, not this file.

---

## Spec 279 PM4 reference detail (2026-10-20)

- **Spec 279 (operator: full PM4 comprehension + readable writeup, Spec Kit):**
  [Current PM4 reference](../docs/research/pm4-format-reference.md) covers 16 parsed chunks,
  raw fields/evidence levels and 17 unresolved question groups. Spec/plan/tasks/checklist on
  `279-decode-pm4-format`. **2A:** raw MSLK +0x01 is a source-edge ordinal (674,013 boundary matches
  vs shifted control 137). **2B:** +0x04 is not a unique edge ID (889,594 reused values); type-1
  reciprocal records share it 98.92%, type 17 100%. **MSHD:** five of eight words always zero;
  Field00/08 correlate with Mdbh ~0.76–0.79. **M2 partition found:** authored `_obj0.adt` placements
  show `Ck24Type == 0` / `GroupKey == 3` selects 14,489 M2-only surfaces vs 57 WMO-only; WMO is
  64–67 / 16–19. The zero bucket is the M2 class, not a per-object identity (only 12.8% of zero
  surfaces cover an authored M2 origin). Receipts: [2A](../specs/279-decode-pm4-format/evidence/phase-2a-edge-ordinal-2026-10-20.md),
  [2B](../specs/279-decode-pm4-format/evidence/phase-2b-edge-identity-2026-10-20.md),
  [MSHD](../specs/279-decode-pm4-format/evidence/phase-3-mshd-2026-10-20.md),
  [placements](../specs/279-decode-pm4-format/evidence/phase-3-placement-ground-truth-2026-10-20.md).
  **M2 per-object split is negative:** topology, vertex ranges, MSUR order, Padding, +0x18, Normal,
  Height and GroupObjectId all interleave objects (2,000 mixed-model components; 6,576 placements
  share vertices; 9,570 overlapping MSUR ranges; Padding always 0; 7,315 placements multi-GroupObjectId).
  [Receipt](../specs/279-decode-pm4-format/evidence/phase-3-m2-partition-2026-10-20.md). Per-object M2
  ownership remains U. **Path layer:** MPRL 178,588 points, 99.4% inside the mesh, `Unk02 = -1` and
  `Unk06 = 0x8000` constant; MPRR 17,150,640 records at 4 bytes each. **MPRR is a quadtree spatial
  index of the MPRL points:** each run is a node (first two records = bounds, middle = MPRL indices,
  last = terminator), and the whole-map node width/height is 1/2/4/8/16/32 (powers of two). The
  `4k+3` run lengths and ~4:1 decay are the quadtree hierarchy. **Scene-graph hierarchy:** `Unk00`
  (region, 265 values) -> `Unk04` (object, 42,092 values; every group has exactly one `Unk00`) ->
  `Unk14`/`Unk16` (per-point type/flags). **Object partition found:** `MPRL.Unk04` is
  `MSLK.GroupObjectId` (+0x04) for 176,748/178,588 points (98.97%), and 98.8% of MPRR runs group
  MPRL points sharing one `Unk04` (100% share one `Unk00`). The id is a group/model id, not a unique
  placement id. The direct group-to-surface link is not containment (0.4%); the points sit on M2
  surfaces (66.7%), WMO surfaces (29.0%) and other surfaces (6.7%). **Object-to-model join:** the
  nearest-placement join covers every group - 32.2% within 5 units, 59.7% within 10, 85.1% within
  25. **Class-consistency precision:** 19,991 groups (68.5%) have a consistent match (<=25 units and
  model class matches ground-surface class), 5,363 (18.4%) are class-mismatched, 3,662 (12.6%) have
  no match within 25. **Ranked matcher:** 895 models; top are `WINTERSPRINGROCK01.M2` (931 groups),
  `SHADOWMOON_OVERHANGROCK_SMALL_02.M2` (753), `STONETREELOG02.M2` (618). **Viewer:** new
  Experimental > PM4 > Object Groups tab lists the selected tile's groups with filter,
  consistent-only and camera focus. **MPRL points are not MSPV path vertices:** only 0.8% within 0.1
  and 13.1% within 1.0; the path geometry is the separate MSPV/MSPI stream.
  [Receipt](../specs/279-decode-pm4-format/evidence/phase-3-path-nodes-2026-10-20.md). Editor-node
  identity remains unknown. Operator wants checkpoints committed. **Reference consolidated
  (2026-10-20):** [pm4-format-reference.md](../docs/research/pm4-format-reference.md) rewritten top to
  bottom as a standalone reader/renderer specification using the current wowdev.wiki field names
  (`PM4`, `PD4`) as primary names; proposed names live only in §9. No commentary about earlier names
  remains. **Opt-in model prior:** `Pm4AssetMatchScorer` accepts an optional
  `modelPriors` map (exact path, then filename, clamped 0-1) and adds at most +0.10 to the overall
  score; `priorBonus` is 0 and the default path is unchanged when no map is supplied. 5/5 scorer
  tests pass, including `ScoreSegment_ModelPriorBreaksTieOnlyWhenSupplied`.
  [Receipt](../specs/279-decode-pm4-format/evidence/phase-4-model-prior-2026-10-20.md).
- **PM4 structural findings (2026-10-20, newest):** summary probes read all **616 files**; exact
  MSUR->MSLK start/count windows; packed link target is tile X=low byte, Y=next byte + surface index.
  **Not universally file-local:** 15,773 cross-tile links; 14,659 resolve. Same-tile addressed links
  are 1,257,562/1,257,562 reciprocal. Paired wrong-local comparison: 9,748 vs 15 back-links on
  10,222 identical records. Raw 0x01/0x11 links behave as boundary adjacency; other kinds carry
  connection geometry. **32 focused tests pass.** Research audit now retains addressed hypotheses.
  [Measured/refuted/unknown ledger](../specs/270-pm4-object-association-ledger-and-placement/evidence/structural-foundations-2026-10-20.md).
  **Unsolved:** editor-authored path-node chunk identity, remaining raw fields, M2 object partition.
  Screenshot-reported NPC route nodes are a separate hypothesis, not automatically MPRL/MSCN.
  Existing readers/matchers/viewer untouched; ~70% WMO accuracy remains operator-reported.
  Prior [instruments/pools](../specs/270-pm4-object-association-ledger-and-placement/evidence/map-context-instrument-2026-10-20.md)
  remain available; large Museum/raw export, viewer pool UI and WL footprint mapping are still open.

## Committed feature sets on `v0.6.1-dev` (2026-10-20)

1. `fix(terrain,pm4)`: Resolved MCNK MCAL/MCSH subchunk parsing desynchronization, 3.3.5 ADT format profile auto-detection, authentic sequential mix terrain blending, and PM4 M2 surface correlation pipeline.
2. `feat(archaeology,spec-277)`: UniqueId Stratigraphy & Cross-Map Chronological Fork Analysis. Unified `UniqueIdRecord` contract, quantile calculation ($p_{10}..p_{90}$), adaptive layer clustering, spatial hash twin placement matching, divergence boundary ($UID_{div}$) detection, CLI profiler (`map stratigraphy`), and interactive in-viewer Archaeology Time Machine slider. 5/5 focused unit tests pass; CLI divergence report verified on Map 451 vs Map 0 ($UID_{div} = 400111$, 100% matched overworld branch).
3. `perf(population,viewer)`: Resolved 0.5.3 MDX/WMO loading regression. Made SQL spawn auto-streaming opt-in (`_sqlAutoLoadOnMapChange = false`), switched partition loading to non-blocking background `Task.Run`, deduplicated applied spawn sets to avoid spurious scene-graph invalidations on camera movement, replaced synchronous WMO disk parsing with cached mesh summary / MOHD bounds in `ExternalSpawnLayer` and `WorldAssetManager.TryGetWmoPlacementBounds`, and added immediate `.mdx`/`.mdl` priority resolution for classic clients in `WorldAssetPathResolver`.
4. `feat(population,spec-276)`: TrinityCore multi-version SQL population extraction pipeline, full multi-map streaming (all 113 maps in 3.3.5a, all 31 maps in 0.5.3), automatic client-version detection & map loading, Map 451 developer test spawns + alpha-core corner dev transposition, authentic NPC/GameObject nameplates and Info tab inspector display, default Stand/Idle animation selection, and ground-truth 4.3.4 detail doodad alpha blending. 12/12 focused tests pass.
5. `feat(harvester,spec-274)`: Minimap Hillshade Correlation & Domain Translation dataset pipeline across 37 Zarr v3 map revisions; 228/228 pytest tests pass.
6. `feat(inspector)`: Integrated 4.3.4 client ground-truth format flags inspection (`FormatFlagsDescriber.cs`) across Terrain, Object, Model, and Standalone inspection panels.
7. `perf(renderer)`: Optimized spatial M2 lighting collection, terrain streaming MCNK FourCC fast-path, and asset path resolver negative caching.
8. `feat(renderer,spec-273)`: Completed Spec 273 performance pipeline with fail-safe WMO frustum preservation; 48/48 portal tests pass.

## Older release and regression notes

**WC3 file support unified in main browser (2026-10-15):** [Spec 272](../specs/272-wc2-wc3-viewer-support/spec.md) WC3 loose folders and MPQs now load as first-class `IDataSource` instances directly into the main File Browser (`Wc3DataSource`). Textures (BLP, BLP0 with external mips, TGA, PNG) render directly in the main viewer scene viewport (`TextureViewRenderer`) with transparency checkerboard, pan, and zoom, instead of cramped sidebar previews. Models (`.mdx`) render through the main MDX viewport renderer. Arbitrary text/binary files can be browsed with the new "All files" (`*`) filter. Multiple BLP0 mip levels (`.b00` through `.b15`) are grouped into a single `<texture>.blp [N mips]` row with level 0 as the full-resolution primary. Added PNG batch/single export via `TextureExportService` with an option to write all companion mip levels. Viewer build passes 0 errors; focused core tests pass 70/70.

**Immediate release fix:** the release workflow previously published only the viewer `.csproj`. All nine `wow-viewer/tools/**/*.csproj` projects now publish into one shared `publish/<rid>/tools/` directory inside the platform's main build tree; common DBDs, listfiles, runtime DLLs and libraries are emitted once. Viewer and CLI tools are separate ZIPs. `V22Enrich` and `WmoMinimap` are registered in `WowViewer.slnx`. The packaging matrix passed in `38506799386`; the final baseline replacement passed in `38512662293` and release `38513490254`. The replacement is eight assets. Downloaded Windows tool bundle verification from the first split release found nine apphosts, 1,342 DBDs, one common `WowViewer.Core.dll`, and Inspector `--help` exit 0.

**Ground-effect regression response:** the per-instance tint/API changes were removed after the operator reported the viewer could no longer load anything. The affected files match the pre-tint baseline; no crash log or stack trace was available, so the runtime cause is not independently confirmed. Do not reintroduce tint propagation until a separate diagnosed approach exists. The replacement release uses the pre-tint renderer path; operator must confirm a real client/map loads.

**Immediate taxi mount issue (1.60.1):** the gryphon never loaded because
[`TaxiPathLoader.LoadCreatureModelPaths`](../src/viewer/WoWViewer/Terrain/TaxiPathLoader.cs:231)
read only `ModelName`/`ModelPath` - absent from 6.0.0 - so every mount resolved to null and every
actor fell onto the hardcoded `Creature\Gryphon\Gryphon.mdx` default. Fixed with the same both-era
pattern: string first, else `FileDataID` -> listfile. The default candidate list now also probes the
`.m2` variant of each legacy `.mdx` path. 3 new tests; 18/18 focused tests pass.

**Immediate WMO area issue (open):** inside `MD_CryptOneRm.wmo` (header UniqueId 85944) in Tirisfal,
the status bar reads `Area: Crypt84 [Riverglades]`. The 1.60.1 DBD layout declares a relation
`WMOID<u16>` and has no `InlineWMOID`; it does **not** establish that this relation shares the
header UniqueId's numeric domain. The global `WMOGroupID` fallback can collide across WMOs, but the
bad row/match is not yet confirmed. A temporary log probe was removed: operator reports the viewer
log is not selectable/exportable and does not provide usable evidence. The operator's hypothesis is
a client-data remap/copy-link error; treat it as plausible, not proven. Next proof must be available
in-viewer or extracted directly from the relevant client rows.

**M2 texture resolver status (2026-10-12):** the broad FileDataID attempt and ItemDisplayInfo
resource-chain changes were reverted after the operator reported that many character M2 textures
stopped rendering. `ReplaceableTextureResolver.cs` is restored to the proven baseline; its new tests
were removed. No FileDataID correction for creature textures or item-display texture chains is
currently active. Operator must confirm character M2s load again after the rollback. **Do not refactor
`ReadStringArray` or its call sites again** without a narrowly scoped, proven change.

**Immediate M2 camera issue:** M2/MDX camera keys are tile-local until their matching
`CinematicCamera` origin/facing is applied. Warm Path's 64x64 grid spacing was corrected from
`TileSize` (8,533 units) to `ChunkSize` (533.333 units); FlybyUndead's documented origin maps to
tile `(28,28)`. Modern 1.60.1 `CinematicCamera` rows have `FileDataID`, not `Model`; the resolver now
matches the selected CASC asset's ID (community listfile: FlybyUndead is 116912). If lookup fails,
tracks remain local and playback, key captures, and warmup reject them. A missed guard was found in
plain Play and fixed. Focused camera tests pass (12/12), including local-origin-to-world-tile mapping.
Phase layers merge donor content onto base-map target coordinates; camera binding still uses the
base `TerrainManager.MapName`, so there is no code evidence that phase composition itself relocates
the path. Operator must verify the actual client DB2 row/origin and visual playback on
`wow_classic_beta` 1.60.1/Azeroth. Separately, the earlier reported `DD21` (`0x31324444`) magic is
still unexplained; exact bytes remain unavailable, so do not widen the reader based on that report.
