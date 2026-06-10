# Active Context — V16.x Terrain Training Pipeline

## Branch
- `v0.5.0-dev`

## Current Focus
Terrain normal/height prediction from minimap images. Chain of V16.1.x model iterations:
- **V16.1.1**: Normal model (minimap 3ch → normals)
- **V16.1.2**: Abandoned — refiner approach (random refiner used as distillation target, never trained)
- **V16.1.3**: Height-channel normal model (cat(minimap, height) 4ch → normals) — plateaued at epoch 123
- **V16.1.4**: Combined normal+height model (4ch → normals + height, shared backbone) — just implemented, not yet trained
- **V18 object-roof lane**: new Spec Kit draft for object-family roof curation plus learned minimap object sieve (`specs/025-object-roof-mask-library-and-minimap-sieve/`)
  - includes MdxViewer improvements for one-at-a-time object asset capture with pose metadata
  - stores per-asset object visual outputs in a separate Zarr datastore for roof exemplars and object-family recognition
  - separate object-identification model is intended to live in the Python `uv` stack and use the Hugging Face transformers library as the first host
  - SAM2 is the first promptable mask-generation host; SAM3 is allowed later if the Hugging Face token has approved gated access
- **V18 dataset canonical contract**: Spec Kit draft expanded in `specs/001-v18-dataset-spec/` 🧭
  - now frames V18 as the direct versioned successor to the V16 dataset creation flow
  - promotes decoded metadata plus currently patched-on V16 signal families into the main V18 build contract
  - defines finalized dataset status, mandatory decoded metadata parity, merge fallback coverage, and additive raw-blob sidecar boundaries
  - initial implementation slice now exists in `data-harvester/scripts/build_v18_dataset.py`
  - landed so far: V18 output root, finalization report writing, optional renderer-truth capture promotion during `build`, and upstreamed object-roof arrays/provenance in the shared harvest/tensor-pack contract ✅
  - renderer-truth promotion is now explicitly gated as experimental until object-loading/capture proof is refreshed ⚠️
  - dry-run readiness passes on staged `0_5_3_3368 / Azeroth_30_48` and `3_3_5_12340 / Azeroth_30_48`
  - non-dry-run `gpu-viewer-style` capture on staged `3_3_5_12340 / Azeroth_30_48` completed 4/4 variants but still produced flat/uniform renders and an all-black object-visibility artifact
  - current conclusion: command-path proof exists, but real object-rendering proof is still blocked 🧪
  - proof-owner correction: bounded `gillijimproject_refactor/src/MdxViewer` validation capture remains the only credible full-layer terrain + object-inclusive proof lane until wow-viewer visual parity is actually demonstrated 📌
  - explicit scope guard: the parser → decoded → dataset direct-pipeline redesign is deferred to a future V20 dataset effort, not V18

## What Exists (Completed)
- All model classes in `wow-viewer/data-harvester/src/harvester/v16_1_models.py`:
  `V161NormalModel`, `V161HeightModel`, `V161NormalHeightModel`, `V161NormalHeightCombinedModel`,
  `V161NormalRefiner`, `V161HolesModel`, `V161LiquidModel`, `V161TexcompModel`
- Training loop in `train_v16_1_common.py` with autotune, curation, hard-region weighting
- `_combined_loss` + `combined` task registered in TASKS (V16.1.4)
- Working export script: `export_terrain_obj.py`
- V16 dataset: 5134 tiles across 6 builds in Zarr stores
- Spec 017 (V16.1.4): spec.md, plan.md, tasks.md (Phase 1 done)

## What's Next (Immediate)
- bounded real-data proof for `build_v18_dataset.py`
- fix the real object-rendering/capture lane before widening any renderer-truth promotion claims
- simplify `patch_v18_object_roof_masks.py` usage now that the shared C# contract emits roof arrays and provenance directly
- use bounded `MdxViewer` compatibility proof for visual validation before calling roof/object data trustworthy 🧭
- **Spec 020**: Fix renderer culling (renderer doesn't see objects) → tile-level capture → batch capture → V16.2 object mask generation
- V16.1.4 combined model training (waiting on smoke test/launch)
- V16.2 model architecture (V16.1.x lessons + proper object-aware loss weighting)

## MotherShip Direction
The long-range target is a universal game engine (`game-engine`) with a plugin architecture. WoW data support lives in `GameEngine.Plugin.WoW`. The current `wow-viewer` work feeds into this. The repo structure is placeholder but the direction is real: `theMothership/` with `game-engine/` core + plugins + viewer + tools. See `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` for the engine-modernization context.

## Known Issues
- Renderer culls all objects — coordinate system bug in `ComputeTilePlanarMin/Max` (spec 013 diagnosed, not fixed)
- MdxViewer loads whole map for single-tile capture (performance bottleneck for V16.2 dataset gen)
- Object mask weighting in training is ineffective (needs V16.2 precise masks)
- Bounded wow-viewer non-dry-run validation capture can finish 4/4 variants while still producing flat/uniform renders and all-black object visibility output; treat artifact emission as insufficient proof of real object rendering
- V16.1.2 refiner is dead code (never trained, random distillation)
- Current context-drift guard: if route, proof owner, or scope changes, restate them explicitly before continuing work ⚠️

## Relevant Files
- `wow-viewer/data-harvester/src/harvester/v16_1_models.py` — all model classes
- `wow-viewer/data-harvester/scripts/train_v16_1_common.py` — training loop, all loss functions
- `wow-viewer/data-harvester/scripts/train_v16_1_combined.py` — V16.1.4 entrypoint
- `wow-viewer/data-harvester/scripts/export_terrain_obj.py` — OBJ export
- `wow-viewer/data-harvester/src/harvester/v16_1_dataset.py` — dataset loader
- `wow-viewer/specs/017-v16-1-4-combined-normal-height-model/` — V16.1.4 spec
- `wow-viewer/specs/020-renderer-culling-and-tile-capture/` — renderer fix spec (new)
- `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` — MotherShip context

## 051 Status (2026-06-09 — Text + Doc correction landed, then rename landed)

- **15/33 tasks complete** — Visualization (Phase 1) + in-program text + canonical semantics doc (Phase 4 partial).
- **2026-06-09 rename**: the `MdosIndex` → `MscnRefIndex` rename landed across the entire stack. The PM4 overlay cache version was bumped to `7` (cache format breaking — old cached overlays will be invalidated on next load). Renamed:
  - `Pm4MsurEntry.MdosIndex` alias → removed; canonical name is `MscnRefIndex`.
  - `Pm4OverlayObject.DominantMdosIndex` → `DominantMscnRefIndex`.
  - `Pm4SelectedObjectGraphMdosNode` → `Pm4SelectedObjectGraphMscnRefNode` (and inner `MdosIndex` → `MscnRefIndex`).
  - `MdosGroups`/`MdosIndices`/`MdosGroupCount`/`MdosCount`/`UniqueMdosCount`/`SameMdosCount`/`SameMdosIndex`/`SplitByMdos`/`splitCk24ByMdos`/`dominantMdosIndex`/`SameMdosIndex`/`AddPm4MdosGroupToCollection`/`Pm4SplitCk24ByMdos`/`SplitSurfaceGroupByMdos` → `MscnRef…` / `SplitByMscnRef` / `…ByMscnRef` / `AddPm4MscnRefGroupToCollection` / `Pm4SplitCk24ByMscnRef` / `SplitSurfaceGroupByMscnRef`.
  - `Pm4MscnClusterExample.InvalidMdosRefCount`/`ValidMdosRefCount` → `…MscnRefCount`; `TopInvalidMdosClusters` → `TopInvalidMscnRefClusters`.
  - `Pm4BadMdosCluster` → `Pm4BadMscnRefCluster`; local `badMdosClusters`/`badMdosTypeCounts`/`badMdosSurfaceCounts` → `badMscnRefClusters`/`…TypeCounts`/`…SurfaceCounts`; `topBadMdosClusters` → `topBadMscnRefClusters`. The public report fields `FilesWithBadMdos` / `TopBadMdosClusters` / `EntriesInFilesWithBadMdos` are kept for the public report contract (the test asserts `FilesWithBadMdos`).
  - Cache signature key: `splitByMdos=…` → `splitByMscnRef=…`.
- **What stayed on `Mdos`**: the **real `MDOS` chunk** (destructible object state) is untouched. `Pm4MdsfEntry.MdosIndex` (a real `MDOS` chunk reference) is correctly named. The Linkage analyzer's destructible edge (`MDSF.MdosIndex -> MDOS`) and the `FilesWithBadMdos` public report field are about the real `MDOS` chunk and stay as-is.
- **Massive moment**: 2026-06-09 visual analysis combined with code inspection of the actual mesh walk and TypeFlags classification produced an accurate reading of the PM4 streams that we did not have before. After ~2 years, the cyan/magenta cubes now have a definitive meaning.
- **Authoritative doc**: `wow-viewer/docs/architecture/pm4-chunk-semantics.md` is now the single source of truth for what each PM4 stream is. The earlier "navmesh graph nodes / polygon centroids / shared vertices" reading is dead.
- **What MSCN is**: scene-graph connector anchor. `MSUR.MscnRefIndex` indexes `KnownChunks.Mscn`. The client uses each `MSCN[i]` as a placement/connector anchor for that surface. Cyan cube = one anchor per surface.
- **What MSPV is**: path-vertex chain of an MSLK link. `MSPI[link.MspiFirstIndex..link.MspiFirstIndex+link.MspiIndexCount]` → `MSPV[indices]`. Magenta cube = one vertex per `MSPI` index reached from a link. Only present when surfaces are connected via `MSLK`.
- **What MSLK.TypeFlags is**: the walkable/structural classifier. `0x03` = M2 top (walkable), `0x10` = WMO interior floor (walkable), `0x12` = WMO exterior solid (structural wall). This unlocks walkable-mask and roof-mask ground-truth sources for the V18 lane.
- **Build verified**: `dotnet build wow-viewer/WowViewer.slnx -c Debug` → 0 errors. PM4 tests: 10/10 pass (`Pm4AssetMatchScorerTests` 4/4, `Linkage` 2/2, `Unknown` 1/1, `MscnDirectory_DevelopmentCorpus_ProducesExpectedHighLevelSignals` 1/1, plus 2 more).
- **Biggest remaining unproven gap**: Status counter still shows `MSUR: 0 raw` (Phase 2 not yet implemented — root cause is `RecalculatePm4OverlayRuntimeTotals` resetting the counter). Will be revealed on first PM4 reload after the cache is rebuilt with version 7.
- **Out of scope**: walkable-mask extraction lane (future V18 spec), TypeFlags-tinted color sub-family, spatial cluster visualization (Phase 7 optional).

## 054 Status (2026-06-09 — Spec written, awaiting fresh chat implementation)

- **0/18 tasks complete** — fresh spec written; no implementation yet
- **Goal**: fix the "every camera jump is slow" PM4 UX bug. The current on-disk cache key is the SHA-256 of the camera-window file set, so every camera jump produces a brand-new cache key and pays the full-decode cost again. Fix is per-file caching at two levels (in-memory + on-disk), keyed on content not on window.
- **Real user workflow** (validated in chat): load a map, jump far, jump back, jump far again. Today: every jump takes minutes. With 054: first jump slow, subsequent jumps effectively free.
- **Root cause** (from code inspection of `WorldScene.cs:38-60` and `Pm4OverlayCacheService.cs:38-60`): the `BuildCandidateSignature` method includes every PM4 file's `(path|length|writeTicks)` for the **current camera window**. When the window changes, the candidate list changes, the signature changes, the cache key changes, and the old cache is unreachable. Even re-visiting the start area only helps if the new window's file set exactly matches the original window's file set.
- **3 phases, 18 tasks, library-first**:
  - Phase 1 (in-memory cache + per-file progress status) — 6 tasks
  - Phase 2 (on-disk per-file cache + version bump 7 → 8) — 6 tasks
  - Phase 3 (real-data smoke + doc sync + manual end-to-end UX) — 6 tasks
- **Cache version bump**: 7 → 8 (intentional format break; the old per-window cache is invalidated and rebuilt on next use).
- **Spec**: `wow-viewer/specs/054-pm4-camera-window-cache/spec.md`
- **Out of scope**: hard LRU eviction tuning, networked/shared PM4 cache, predictive pre-fetch, compression changes, migration of the old per-window cache blob to the new per-file format.

## 054 Status Update (2026-06-09 — Viewer wiring + critical stamp-folding fix landed; 16/18 tasks done)

- **Wired** (T004-T006, T008-T012, T014, T016 all done in this session):
  - `_pm4PerFileInMemoryCache = new(capacity: 256)` field in `WorldScene.cs:1010` (T004)
  - `LoadPm4OverlayAsync` per-file loop checks `_pm4PerFileInMemoryCache.TryGet` then `_pm4PerFileDiskCache.TryRead` then fresh `BuildPm4TileObjects` then writes to both caches (T005, T009, T010)
  - `StorePerFileInMemoryCache` (T005 return path) builds a `CorePm4PerFileCacheEntry` with the split-flag stamp folded into `LastWriteTicks`
  - `TryReadPerFileDiskCache` (T009) gates on `FileLength` mismatch
  - `EnsurePerFileDiskCache` (T007) creates the on-disk service once per `(dataSource, mapName)` and shares the `Pm4OverlayCacheService.CacheRoot + "/files"` parent
  - `BuildCachedObjectsForDiskWrite` (T010) and `BuildPm4OverlayCacheData` mirror the in-memory and on-disk formats
  - `Pm4OverlayCacheService.CacheVersion = 8` (T008) — old v7 blobs are now invalid; readers gate on version mismatch
  - `ReloadPm4Overlay` (T014) clears in-memory cache + on-disk cache for the map + nulls the field
  - Status format (T006/T013) now shows `(mem-cache N hit, disk-cache M hit)`
  - Architecture doc (T016) has a new "PM4 overlay cache layout (spec 054)" section explaining the two layers
- **Critical bug fixed**: the in-memory cache read at `WorldScene.cs:3703` was passing `lastWriteTicks = 0L`, but the writer at `WorldScene.cs:4528` stores a folded split-flag stamp (e.g. `0x100000001L` for both flags on, `0L` for both off). With the default config (`_pm4SplitCk24ByMscnRef = true`, `_pm4SplitCk24ByConnectivity = false`), the stored stamp was `0x100000000L` and the read stamp was `0L`, so the in-memory cache **always missed** in the default config. Fixed by computing `memStamp` at the read site from the same split-flag formula the writer uses. Without this fix, every camera shift paid the full `BuildPm4TileObjects` cost even for files already decoded in the session. The PM4 tile navigation hard crash the user reported was this combined with a slow decode path.
- **Tests added** (T003 + T011): `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4PerFileCacheServiceTests.cs` has 10 round-trip tests. `wow-viewer/tests/WowViewer.Core.PM4.Tests/pm4PerFileCacheTests.cs` has 9 in-memory tests including a stamp-folding regression test. All 18 pass.
- **Build verified**: `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug` → 0 errors. Anim farm tests: 21/21 pass. PM4 per-file cache tests: 18/18 pass.
- **Outstanding (deferred)**: T015 (real-data test gated on staged 3.3.5 client); T018 (manual end-to-end UX timings — needs interactive viewer).
- **Follow-up fix (2026-06-09)**: The MSCN/MSPV lazy path (`EnsurePm4MscnData` / `EnsurePm4MspvData` at `WorldScene.cs:10680` and 10703) had an inverted early-return guard. The condition was `if (_pm4TileMscnPoints.Count > 0 || _pm4TileObjects.Count == 0) return;` which bailed out as soon as ANY tile was populated, so subsequent camera-window shifts that added new tiles never repopulated the new tiles' points. The fix drops the `Count > 0` short-circuit and trusts the inner per-tile guard (`if (_pm4TileMscnPoints.ContainsKey(tileKey)) continue;`) for dedup. Both functions now iterate all current tiles and populate only the missing ones. **The "Reload PM4" requirement for MSCN/MSPV nodes after a camera shift is resolved.**
- **Out of scope (unchanged)**: LRU tuning, network cache, pre-fetch, compression changes, migration of v7 blob.

## 053 Status (2026-06-09 — Spec + Plan + Tasks + Research written; Phase 0 + Phase 1 done; Phases 2-9 deferred pending 054 wrap-up)

- **Phase 0 (research) ✅ + Phase 1 (library skeleton + loaders) ✅ complete; 21/21 tests pass**
- **Phase 2-9 deferred** (anim farm) — pivoted to spec 054 per user direction; 054 is now substantially complete (16/18 tasks), see below
- **What landed**:
  - `wow-viewer/specs/053-m2-animation-pose-farm/spec.md` (full spec with 7 user stories, 24 FRs, 5 NFRs, out-of-scope)
  - `wow-viewer/specs/053-m2-animation-pose-farm/plan.md` (9 phases, 104 tasks, library-first)
  - `wow-viewer/specs/053-m2-animation-pose-farm/tasks.md` (all 104 tasks enumerated, Phase 0+1 marked done)
  - `wow-viewer/specs/053-m2-animation-pose-farm/research.md` (8 decisions: dispatcher, MDX, listfile cache, anim resolution, JSON schema, BVH format, FBX scope, spec deltas)
  - `wow-viewer/src/core/WowViewer.Core.Anim/` (new library: `PathNormalizer`, `M2AnimationPoseSource`, `M2PoseSourceLoader` using `M2ModelReaderDispatcher.ReadDetailed` per R-0.1, `MdxAnimationPoseSource`, `MdxPoseSourceLoader`)
  - `wow-viewer/tests/WowViewer.Core.Anim.Tests/` (21 passing tests: 12 PathNormalizer + 5 M2 loader + 4 MDX loader)
  - `wow-viewer/WowViewer.slnx` updated with new `/src/core/Anim/` and `/tests/anim/` folders
- **Key research finding (R-0.1)**: `M2ModelReaderDispatcher` at `src/core/WowViewer.Core.IO/M2Chunked/M2ModelReaderDispatcher.cs:10` is the canonical entry point (auto-detects MDLX/MD20-1x/MD20-3x via magic). The plan's T014 was revised to use this instead of `M2ModelReader.Read` directly.
- **Pre-existing build error fixed (one line)**: `wow-viewer/src/core/WowViewer.Core.PM4/Caching/pm4PerFileCacheService.cs:7` was missing `using WowViewer.Core.PM4.Models;` — added it. This unblocked the whole `dotnet build`. The PM4 `Caching/` dir is untracked (work-in-progress for spec 054).
- **Next session, if user wants to resume 053**: start at Phase 2 (alias resolution + bone track stream extraction). The spec/plan/tasks are complete and the library is ready for the next slice.
