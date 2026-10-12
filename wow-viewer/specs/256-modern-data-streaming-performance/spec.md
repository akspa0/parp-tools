# Spec 256 — Modern-Data Streaming & M2 Submission Performance

**Created**: 2026-09-27 · **Branch**: `v0.6.0-dev` · **Status**: Plan approved 2026-09-27 (T001) incl. minimap upload-budget amendment; P0, P1, minimap budget and P2a/P2b/P2c landed 2026-09-27 (operator capture owed: T073)
**Parent**: Epic 249 (renderer performance). Governance: AGENTS.md §4 (renderer/reader boundaries), §9, §10.

## Operator direction (2026-09-27)

- *"the performance for the modern rendering is atrocious, under 1fps most of the time."*
- *"it wasn't improved, it was a lot worse. the sitting idle fps was higher, but loading became sluggish as
  hell, as if we are reading from hdd, but I know we are reading from the fastest ssd in the system!"*
- Asked which specs to write after the measured baseline below, the operator selected all three:
  **"Off-thread model loading"**, **"Native M2 instancing"**, **"Find the hidden ~115 ms first"**.

## Measured baseline (operator screenshot, 2026-09-27, build `9ef30ec`)

`wow_classic_beta` 1.60.1 via CASC, `Azeroth`, tile (48,41), 25 ADT tiles loaded, Runtime Stats panel:

| Measure | Value |
|---|---|
| FPS / world render CPU | 4 FPS / 266.6 ms |
| Deferred model loading this frame | 98.2 ms |
| Predicted (average) cost of one load | M2 54.5 ms · WMO 21.3 ms |
| Oversized admissions (loads larger than the frame budget) | 137 |
| M2 opaque instanced / state-hoisted / unbatched | 0 / 0 / 5,169 |
| M2 anim / visibility / opaque submission | 4.3 / 12.3 / 18.2 ms |
| WMO visibility / opaque / transparent | 0.1 / 13.0 / 1.6 ms |
| Managed heap live / allocated since start | 4.53 GiB / 59.67 GiB |
| Asset I/O requests / cache hits | 783 / 191 |
| Stages listed in the visible panel lines | ≈ 150 ms of 266.6 ms — **≈ 115 ms unattributed** |

## Code findings (read 2026-09-27; not measured individually)

1. **Skin lookup scans every `.skin` file** — `WorldAssetManager.ResolveBestSkinPath` → `WarcraftNetM2Adapter.FindSkinInFileList`
   walks `IDataSource.GetFileList(".skin")` with ~5 string allocations per entry, once per unique M2 (cached),
   on the render thread, even when the model names its skin by FileDataID. On CASC the list is every present
   `.skin` in the community listfile. Also called from `PrefetchModelBytes` when a model is queued.
2. **Every skinned M2 is parsed twice** — the native runtime model *and* the M2→MDX adapter model are built;
   `WowViewerM2RuntimeBridge.PreferNativeStaticRenderer` defaults to true (env `PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER`
   unset), so the adapter result is discarded.
3. **Native M2s are never instanced** — `M2Renderer.RequiresUnbatchedWorldRender` is true without a legacy
   renderer ("keep it isolated until a backend-specific batch key exists"); that is every M2 by default.
4. **Loading is paced by frame time** — `WorldScene.ProcessDeferredAssetLoads`: previous frame ≥ 33 ms → at
   most 1 load and 1 ms per frame; one M2 costs ~55 ms, so loading runs at about one model per frame.
5. **~115 ms of the frame is not named** by the visible Runtime Stats lines; GC pause time is not shown.
6. Runtime Stats shows read/prefetch counters for MPQ but not for `CascDataSource` (`Stats` exists, unread).

## User stories

- **US1 (P1)** — As the operator, models stream in on a CASC client as fast as the CPU can parse them, not one
  per rendered frame.
- **US2 (P1)** — As the operator, a dense modern map draws its repeated M2s instanced, not 5,000 draws apiece.
- **US3 (P1)** — As the operator and agents, every millisecond of the frame and every GC pause is named in
  Runtime Stats, so changes are chosen from numbers, not guesses.

## Acceptance criteria

1. Runtime Stats names ≥ 95 % of world CPU per frame and shows GC pause time per frame, M2 batch-gate counts,
   per-load phase times (read / skin resolve / parse / GPU create) and CASC read/prefetch/cache counters.
2. Skin resolution returns the same path as `FindSkinInFileList` for every model (equivalence test), without a
   per-model scan of the file list.
3. No M2 is parsed through a route whose result is discarded.
4. Model parsing runs off the render thread; the render thread only creates GL objects. Loaded models,
   renderer choice, textures and draw output are unchanged.
5. Static opaque native M2s can be GPU-instanced; lit/animated/fading cases follow the existing gates.
6. Every phase: `dotnet build` 0 errors, test failure set unchanged, receipt. FPS, load time and visuals are
   claimed only from operator captures at the same map/camera as the baseline.

## Constraints and out of scope

- Format readers (M2/WMO/BLP/MPQ/CASC) are not modified (AGENTS.md §4); work sits in `WorldAssetManager`,
  renderer construction, and the M2 renderer's draw path.
- The WMO per-placement instancing gate stays reverted (Epic 249 amendment 2026-09-27) — not reopened here.
- The minimap (Epic 249 R-39) is not part of this spec.
- No change to the frame-time throttle until off-thread loading lands (it is re-evaluated then).
