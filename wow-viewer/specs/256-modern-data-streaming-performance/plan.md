# Plan — Spec 256 Modern-Data Streaming & M2 Submission Performance

**Status**: approved 2026-09-27 (T001; operator: "Approve Spec 256 plus the minimap budget", replying to the offer to
start with P0 and P1). P0 and P1 land as **separate commits** so the operator can capture the P0-only build
before P1; the revert-on-regression rule applies to each.

Phases run in order; each ends with an operator capture on the baseline map/camera
(`wow_classic_beta` 1.60.1, `Azeroth`, tile (48,41)) and a receipt. A phase whose capture shows no gain or a
regression is reverted before the next starts.

## P0 — Name the frame (operator: "Find the hidden ~115 ms first")

| Change | Where | Notes |
|---|---|---|
| Missing world-frame stages timed (object-phase prepare, scene maintenance, overlays, taxi, UI/ImGui, swap) and an "unattributed" remainder line | existing `WorldRenderFrame` / `WorldRenderFrameStats` counters; Runtime Stats panel | remainder must fall under 5 % |
| GC pause ms per frame (`GC.GetTotalPauseDuration()` delta) and gen0/1/2 counts per second | Runtime Stats | read-only |
| Per-load phase times: read, skin resolve, native parse, adapter parse, GPU create — MDX/M2 and WMO | `WorldAssetManager` load functions (timing only) + `DeferredLoadBudget` | averages and worst |
| M2 opaque batch-gate counts (batching disabled / route requires unbatched / fade / lights) | existing `WorldModelSubmissionTally` gates, surfaced in Runtime Stats | already recorded, not shown |
| CASC read/prefetch/cache counters | `CascDataSource.Stats` in Runtime Stats | read-only |

Operator task with P0: one capture as the baseline, plus an A/B with the existing switch
`PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER=0` (legacy, instanceable M2 route) to see what instancing is worth and
whether the legacy route looks right on modern data.

## P1 — Exact load-cost cuts (only if P0 confirms them)

| Change | Where | Equivalence proof |
|---|---|---|
| Skin lookup index built once per data source: `.skin` paths grouped by directory and by basename prefix; the score/tie rules of `FindSkinInFileList` applied to the candidate subset | new small owned class beside `WorldAssetManager`; `ResolveBestSkinPath` calls it | unit test: index result == `FindSkinInFileList` for every model path in a large synthetic list and in a real listfile subset |
| Adapter parse skipped when the native renderer will be chosen (`PreferNativeStaticRenderer` true); unchanged when false | `WorldAssetManager.LoadMdxModel` | the discarded object is never observed; renderer choice unchanged (`ShouldUseNativeStaticRenderer`) |

## P2 — Off-thread model loading

- Split each model load into **prepare** (read bytes, resolve skin, parse M2/WMO into CPU-side data — no GL)
  on worker threads (normal priority; count from P0 data) and **create** (GL buffers/renderer object) on the
  render thread, which drains prepared results under the existing per-frame budget.
- Queue order, cache/LRU, failure suppression, route decisions and deferred texture loads behave as today;
  a model requested twice is prepared once.
- Renderer constructors gain a prepared-data entry point; readers are not modified. `WmoRenderer`,
  `MdxRenderer`, `M2Renderer` are each over 1,600 lines, so the prepare step lives in new owned classes, not
  inside them (AGENTS.md §10 budget).
- Throttle (finding 4) re-measured after P2; any change to it is a separate amendment.

## P3 — Native M2 instancing

- A backend-specific batch key and an instanced opaque path in the native `M2Renderer` for static opaque
  sections, following the existing gates (model lights, fade handling, wireframe) and the legacy renderer's
  contract C2 (no approximated per-instance state).
- Scope decided from P0: if the legacy route A/B looks correct and is faster, the alternative is to make it
  the modern default instead of building a new path — operator decides.

## Risks

| Risk | Mitigation |
|---|---|
| Off-thread parsing races shared caches or GL | prepare step touches no GL and only thread-safe caches; create step unchanged on the render thread |
| Instanced native M2s look different | per-model A/B captures; gates keep uncertain cases unbatched |
| Another regression like R-10/R-39 | every phase is measured by the operator before the next; revert on regression |

## Amendment 2026-09-27 — minimap upload budget (operator: "plus the minimap budget")

`ViewerApp` drains decoded minimap tiles with `MinimapRenderer.ProcessPendingLoads(maxLoads: 1, maxBudgetMs: 1.5)`
unless the Minimap window or fullscreen minimap is open, so tiles appear at one per frame (≈ 4/s at 4 FPS;
5,285 pending in the operator's capture). Change: upload by time budget instead — up to 16 tiles within 2 ms
per frame normally, 32 within 6 ms with the minimap window/fullscreen open. Tile decode, reads and display
are unchanged; only how many ready tiles are uploaded per frame.

## Amendment 2026-09-27 — P2 refined from the P1 capture (operator: "Approve P2a and P2b and P2c")

Capture (`evidence/capture-p1-2026-09-27.md`): an M2 load averages 64.5 ms, of which ≈ 57 ms is renderer
creation; the native `M2Renderer` decodes and uploads every texture synchronously in its constructor, with a
texture cache per renderer (a texture shared by 40 models is decoded and uploaded 40 times). P2 becomes:

| Step | Change | Identical output? |
|---|---|---|
| **P2a** | One reference-counted texture cache shared by all native M2 renderers (key: resolved path + clamp flags) | yes — same pixels, decoded once |
| **P2b** | Texture bytes read and decoded on background threads; the render thread only uploads; a renderer draws a section once its texture is uploaded | pixels yes; textures can appear a few frames after the geometry |
| **P2c** | DXT BLP textures uploaded compressed with the BLP's own mip levels instead of CPU decode + `GenerateMipmap` | **no** — distant mips come from the file (operator accepted) |

Order P2a → P2b → P2c, one commit each, each revertible. The original P2 text (off-thread model parsing)
is deferred: parse is ≈ 6 ms per M2, so textures come first.
