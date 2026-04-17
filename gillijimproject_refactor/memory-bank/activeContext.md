# Active Context

## Apr 17, 2026 - wow-viewer viewer-app cutover is now staged through a dedicated plan, and slice 01 app settings persistence is landed

- after the first wow-viewer desktop shell landed, the next risk was slipping back into broad ad hoc viewer work; that is now corrected with a dedicated continuity plan at `gillijimproject_refactor/plans/wow_viewer_viewer_app_cutover_plan_2026-04-17.md`
- the active ordered sequence is now explicit:
	- slice 01 app state/settings persistence
	- slice 02 viewer session boundary
	- slice 03 standalone asset workspaces
	- slice 04 GPU M2 preview consumer
	- slice 05 world session bootstrap
	- slice 06 world runtime consumer bridge
	- later shell-surface expansion and legacy cutover review
- slice 01 is now implemented in `wow-viewer/src/viewer/WowViewer.App/`:
	- `WowViewerAppSettings.cs` owns wow-viewer-local persisted app settings under `output/settings/wowviewer_app_settings.json`
	- `WowViewerDesktopApp.cs` now loads those settings on startup and saves them on dispose, keeping source selection, preview parameters, and app window toggles inside the new repo instead of falling back to old `ViewerApp` state
- proof in this chat remains build-level for the settings slice:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the usual workspace `LIB` warnings only
- current boundary:
	- the plan exists and slice 01 is real, but the new app still does not have a typed viewer session object, standalone workspace split, GPU preview path, or world-session bootstrap yet

## Apr 16, 2026 - wow-viewer now has its first real desktop viewer shell, but it is still an M2 preview host rather than a full world-scene consumer

- the user explicitly redirected the work away from keeping `ViewerApp` trapped in `MdxViewer`; the active cutover target is now a true viewer app inside `wow-viewer`
- `wow-viewer/src/viewer/WowViewer.App` is no longer only the `m2-frame` console harness:
	- it now owns a Silk.NET + ImGui desktop host
	- the desktop app loads M2 assets only through shared `wow-viewer` runtime code via a shared `M2PreviewLoader`
	- the same app shows the deterministic software visual snapshot plus runtime hashes and submission diagnostics inside the new repo, with no `MdxViewer` project dependency
- `m2-frame` still exists and now shares the exact same loader path as the desktop preview shell instead of duplicating that app-local runtime bootstrap
- bounded proof in this chat:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the usual workspace `LIB` warnings only
	- real fixed-root loader proof on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `Creature/Wolf/Wolf.m2`, through `WowViewer.App m2-frame` succeeded and reported runtime hash `9e9586068a443468ccec1abd62b3d717c0455e08999bd03beb21427a9df4ec30` plus visual hash `b2fabb6da814c393ea149fb7321cbd3e05d24db8852f59dca35c755c29bfb177`
- current boundary:
	- this proves the first wow-viewer-owned desktop app shell plus shared M2 preview loading, not world-scene cutover, GPU renderer parity, or active interactive runtime signoff for the new windowed shell
	- `WorldScene` extraction and a true wow-viewer world-runtime consumer still remain later slices under the existing world-runtime plan

## Apr 17, 2026 - MdxViewer weak-signal terrain restore is back on a simpler camera-neighbor whole-tile path with range-based per-cell clamp

- the user rejected the newer workbench-scope, loaded-tile, per-chunk, and shadow-heavy restore behavior and explicitly redirected the active design back toward the older reference around commit `336894c7c3a8c51f94da2efe6ad1accacc883352`
- active `MdxViewer` weak-signal restore eligibility is again limited to the camera tile plus its four direct neighbors, but the gate now accepts either a full weak-range ADT or partial weak-signal evidence from per-cell observed ranges
- the active restore application still routes through the whole-tile factor path, but the actual deformation is now clamped to every weak sub-cell detected from the configured source-height range across the ADT instead of narrowing the mask to chunk or texture buckets
- the sidebar copy now describes one active mode (`whole-tile factor, per-cell weak-signal clamp`), and persisted settings now also force `EnableWeakSignalTerrainRestore=false` on load and save so the feature never auto-enables across launches
- validation in this chat is still build-only:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`
	- full `dotnet build` is currently blocked by the already-running `ParpToolsWoWViewer` process locking the active output binaries, so this chat does not yet have a fresh unlocked build pass after the latest per-cell restore change
- current boundary:
	- no fresh real-data runtime proof has been captured yet for the simplified camera-neighbor restore path with the new per-cell mask
	- seam continuity is still not solved at the shared tile-grid level; if restored chunks still drift along boundaries, the next fix should apply the masked restore on `TileHeightmap257` before converting back to MCNK chunks

## Apr 16, 2026 - MdxViewer weak-signal terrain restore now uses a persisted source-height band instead of a hard-coded `<= 10` gate

- the user clarified that buried terrain compression is era-dependent rather than universally sea-level-based:
	- early-era weak-signal tiles tend to live around `-10..10`
	- later-era ocean-floor-compressed tiles can live closer to `-5000..10`
- `MdxViewer` weak-signal restore now exposes persisted `Restore Range Min Z` and `Restore Range Max Z` controls so the viewer gate is user-driven instead of hard-coded to `source max Z <= 10`
- the manual restore multiplier and shared auto-factor clamp now both allow values above `64x` so deeper later-era compression can still be amplified from the cached original tile data
- current boundary:
	- this is still viewer-side live restore gating only, not chunk-level saved-bundle export
	- no new real-data runtime validation was completed in this chat yet; build validation still needs to be reported as build-only unless followed by a live viewer check

## Apr 16, 2026 - MdxViewer native static M2 path now uses primary runtime textures, and MPQ-backed `*_cam.m2` loads no longer fall through to dead geometry fallback

- the user asked to finish the remaining native static M2 consumer slice after the earlier camera-path extraction work, so the active follow-up stayed bounded to `MdxViewer` as a `wow-viewer` consumer rather than reopening ownership
- `Rendering/WowViewerM2RuntimeBridge.cs` now passes `IDataSource` and `ReplaceableTextureResolver` into the pure runtime `M2Renderer` path, and `Rendering/M2Renderer.cs` now uploads UVs, resolves primary-stage texture bindings, samples a bound texture in the shader, and disposes owned GL textures instead of staying flat-tinted only
- real Wrath startup-automation proof on fixed local root `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft` for `Creature/Wolf/Wolf.m2` now writes a textured native-static capture at `output/build-validation/m2-native-static-texture-path/standalone/3.3.5.12340/20260416_040737484_current_20260416_040737_no_ui.png`
- the same validation pass exposed a real camera-only regression in the MPQ-backed data-source route: `Cameras/Scry_cam.m2` still fell through to the old adapter path because `WowViewer.Core.Runtime/M2/M2CameraPathOverlayBuilder.CanBuild(...)` required `ViewCount == 0`, while the real asset carries one dummy skin profile and one helper bone
- candidate detection is now widened in `wow-viewer` for camera assets with parsed cameras, no ribbons or particles, and canonical camera-style names such as `*_cam.m2` or `Cameras\...`, so the fix remains runtime-owned instead of becoming a `MdxViewer`-only exception
- rerunning the same MPQ-backed startup automation now produces a camera-path overlay capture at `output/build-validation/m2-native-static-texture-path/standalone/3.3.5.12340/20260416_042648038_current_20260416_042648_no_ui.png` instead of failing on missing renderable geometry
- focused validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "M2FoundationTests|M2RuntimeTests"` passed `34/34`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` succeeded with existing workspace `LIB` warnings only
- current boundary:
	- the native static consumer path now has primary-stage runtime textures plus simple lighting, but it is still not full native material, animation, or shader parity
	- camera-only `*_cam.m2` support is now proven on the active MPQ-backed viewer path as sampled-path visualization, not as a mesh renderer

## Apr 16, 2026 - standalone camera-only MD20 assets now have a bounded viewer path instead of failing on missing `.skin`

- the user redirected the current M2 work away from generic texture parity and toward `*_cam.m2` handling after `Cameras\\Scry_cam.m2` proved that the standalone loader was still assuming every `.m2` must resolve a `.skin`
- local format references confirmed that flyby camera M2s can be valid geometry-less assets, so the correct fix shape was to treat them as structured camera data rather than weakening mesh validation globally
- strict `wow-viewer` MD20 ownership now parses `M2CameraDefinition` data into `M2ModelDocument`, and `WowViewer.Core.Runtime/M2/M2CameraPathOverlayBuilder` now owns camera-only candidate detection plus sampled camera or target overlay generation and bounds
- `MdxViewer` now probes the strict model before `.skin` loading, then consumes the wow-viewer-owned `M2CameraPathVisualization` result through a thin `M2CameraPathRenderer` GL adapter instead of owning the sampling logic locally
- the standalone renderer still draws the same sampled paths plus pin markers, and the sidebar frame-model action was widened to all `IModelRenderer` implementations so the new path remains operable in the existing UI
- focused validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "M2FoundationTests|M2RuntimeTests"` passed `33/33`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` is still blocked by the earlier unfinished `M2Renderer` texture-path slice (`SelectBestReplaceableDisplayIndex`, `LoadSectionTextures` missing), not by the new camera-path code
- current boundary:
	- camera-only flyby assets now have a bounded standalone visualization path with ownership centered in `wow-viewer`, but the separate textured pure-runtime `M2Renderer` work remains incomplete and should not be conflated with this fix

## Apr 15, 2026 - wow-viewer Wolf M2 geometry corruption was fixed by stopping the blind `globalVertexOffset` shift on strict skin vertex lookup

- continued the active `wow-viewer` M2 correctness debugging after the user rejected the previous Wolf proof as random geometry:
	- added inspect-side `--static-visual-output` proof so the same real asset could be compared before and after skinning
	- confirmed the fault was already present in the static mesh path, not introduced by pose math
	- isolated the main corruption to `WowViewer.Core.Runtime/M2/M2StaticRenderModelBuilder.TryGetVertex`, where the strict skin header field at `0x2C` was being applied as a blind vertex base offset before trying the direct lookup entry
	- real Wolf proof showed that field value as `53`, which matches documented `boneCountMax` values and not a sane LOD0 vertex base for a skin whose local lookup count already matches the full model vertex count (`557`)
	- runtime vertex resolution now uses the direct skin lookup entry first and only falls back to the extra header field if the direct lookup is out of range
	- strict skin parsing now also rejects bogus optional shadow-batch spans instead of reporting garbage counts from payload bytes
- bounded real-data proof for the active Wrath baseline used the fixed local root `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft` on `Creature/Wolf/Wolf.m2`, sequence `0`, time `0`:
	- `WowViewer.Tool.Inspect m2 inspect --visual-output output/build-validation/wow-viewer-m2-wolf-idle-skinned-visual-fixed.bmp --static-visual-output output/build-validation/wow-viewer-m2-wolf-idle-static-visual-fixed.bmp` now produces recognizable quadruped silhouettes in both static and skinned proof images instead of the earlier collapsed wedge
	- the same proof now reports `shadowBatches=0` instead of the earlier bogus `393221`
	- render-frame hash changed to `86048f9de460bb5e75a557d526609700f4292b61ccc0f8eae4b4bd6206f012bb`
	- software visual hash changed to `71aff63b3d0fba7e1eba03bcad894f2af0f2c87448fc9d706a976506b9f17ee5`
	- the earlier `H:/CLIENTS/World of Warcraft Cata beta 11927` Wolf run hit the same corrected counts and hashes, but that was cross-build side validation and not the intended task baseline
	- focused M2 tests pass `31/31` and full `wow-viewer` build still passes with the existing invalid `LIB` path warnings only
- proof boundary:
	- this fixes one concrete mesh-assembly bug in the first-party `wow-viewer` runtime and materially improves the real Wolf silhouette
	- it is still software-proof validation, not final GPU renderer signoff or full native visual parity

## Apr 15, 2026 - wow-viewer M2 app and inspect now share the same frame pipeline, and inspect can emit render-frame plus visual proof outputs too

- continued the M2 runtime implementation in the user's requested order (`3, 1, 2`) after reconstructing the stop point:
	- extracted the app-side frame assembly into shared `WowViewer.Core.Runtime/M2/M2RuntimeFramePipeline`
	- the shared pipeline now owns animated state, bone pose, skinned vertices, render-consumer state, particle/ribbon runtime state, submission plan, render frame, software visual snapshot, and golden frame as one reusable result
	- `WowViewer.App m2-frame` now consumes that shared pipeline instead of carrying its own local orchestration path
	- `WowViewer.Tool.Inspect m2 inspect` now consumes the same shared pipeline and supports `--render-frame-output` plus `--visual-output` in addition to `--golden-output`
	- inspect output now also prints `RENDER.VISUAL` hash and lit-pixel proof, so app and inspect expose matching golden/render/visual state over the same runtime build path
- bounded real-data proof again used the fixed local root `H:/CLIENTS/World of Warcraft Cata beta 11927`:
	- `WowViewer.App m2-frame --virtual-path Creature/Wolf/Wolf.m2 --sequence-index 20 --time-ms 500` wrote:
		- `output/build-validation/wow-viewer-m2-wolf-runtime-golden.json`
		- `output/build-validation/wow-viewer-m2-wolf-runtime-frame.json`
		- `output/build-validation/wow-viewer-m2-wolf-runtime-visual.bmp`
	- `WowViewer.Tool.Inspect m2 inspect` on the same asset wrote:
		- `output/build-validation/wow-viewer-m2-wolf-inspect-golden.json`
		- `output/build-validation/wow-viewer-m2-wolf-inspect-frame.json`
		- `output/build-validation/wow-viewer-m2-wolf-inspect-visual.bmp`
	- matching proof hashes on that asset now include:
		- golden/runtime hash: `113f55daaad3e996476eeff4c9e6fe37aa4c4d3cc364a48e38c6a86bc6fb980e`
		- render-frame hash: `a285c8ef68b0d3304a55d93a30a34f4722fea7c9ed9d429fd5bf1db903932988`
		- software visual hash: `8880ba87d37662a59c8b07d040a7eeb40b1a1060585c9593b197712db6ccf5ec`
	- focused M2 tests now pass `31/31`
	- full `wow-viewer` build passed with existing invalid `LIB` path warnings only
- proof boundary:
	- this is stronger consumer parity between app and inspect plus a reusable visual-proof harness, not active GPU renderer cutover or native screenshot parity
	- shader backend wiring, particle/ribbon parser and simulation, and final visual renderer signoff remain open

## Apr 15, 2026 - wow-viewer M2 now has resolved effect objects, particle/ribbon submission descriptors, app consumption, and golden-frame proof

- continued the M2 runtime implementation in the user's requested order (`3, 1, 2`):
	- residual shader/effect parity now has a runtime-owned `M2EffectRegistry` / `M2ResolvedEffect` seam that exposes native-style effect object keys such as `Model2_Diffuse_T1Combiners_AlphaKey`, depth-write, alpha-test, additive, lighting, two-sided, projected, and stable state-bucket decisions
	- particle and ribbon scene work now has typed `M2ParticleSubmissionDescriptor` / `M2RibbonSubmissionDescriptor` inputs plus family policies and named handlers (`particle-dispatch`, `ribbon-direct`, `core-batch`, etc.) instead of only a generic entry list
	- `M2SceneSubmissionEntryBuilder` is now the shared render-entry builder used beyond the inspect-local helper path
	- `M2RuntimeGoldenFrameBuilder` now emits deterministic runtime golden snapshots and SHA-256 hashes over model, animation, pose, effect, and submission state
	- `WowViewer.App` now has a real `m2-frame` consumer command over the first-party M2 runtime frame instead of only printing the old skeleton banner
	- `WowViewer.Tool.Inspect m2 inspect` now supports `--golden-output` / `-g` and prints resolved effect object, native-family, depth-write, alpha-test, handler, and state-scope proof lines
- bounded real-data proof used the fixed local root `H:/CLIENTS/World of Warcraft Cata beta 11927`:
	- `WowViewer.App m2-frame --virtual-path Creature/Wolf/Wolf.m2 --sequence-index 20 --time-ms 500` loaded `Wolf00.skin` and `Wolf0096-00.anim`, wrote `output/build-validation/wow-viewer-m2-wolf-runtime-golden.json`, and produced hash `113f55daaad3e996476eeff4c9e6fe37aa4c4d3cc364a48e38c6a86bc6fb980e`
	- `WowViewer.Tool.Inspect m2 inspect` on the same asset wrote `output/build-validation/wow-viewer-m2-wolf-inspect-golden.json` with the same hash and printed the resolved effect/submission lines
	- full `wow-viewer` build passed with existing invalid `LIB` path warnings only
	- full `wow-viewer` tests passed: `263` `WowViewer.Core.Tests` and `36` `WowViewer.Core.PM4.Tests`
- proof boundary:
	- this is now stronger than inspect-only ownership because `WowViewer.App` consumes the runtime frame, but it is still not an active visual renderer, shader backend, particle simulation, ribbon geometry, or old `MdxViewer` parity signoff
	- the particle/ribbon work is a submission contract and handler-policy closure, not final emitter/ribbon parser or GPU behavior

## Apr 15, 2026 - wow-viewer M2 now has first-party pose, skinning, render-consumer, and submission-planning seams

- extended the already-landed `wow-viewer` M2 runtime baseline instead of restarting slices 01 or 02:
	- `M2ModelReader` now parses typed bone definitions from the strict `MD20` root
	- `WowViewer.Core.Runtime/M2` now owns a shared track sampler, compressed M2 quaternion sampling, bone-pose evaluation, CPU-side skinned render vertices, render-consumer frame state, and an explicit M2 scene-submission/batching coordinator
	- `M2SkinReader` and active/static render sections now preserve submesh bone lookup metadata (`BoneComboIndex`, `BoneCount`, `BoneInfluences`, `CenterBoneIndex`) so skinning does not have to guess that data later
	- `WowViewer.Tool.Inspect m2 inspect --time-ms` now prints `ANIM.POSE`, `RENDER.CONSUMER`, and `SCENE.SUBMISSION` proof lines in addition to the earlier animated material/light evaluator output
- validation completed in this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter "M2FoundationTests|M2RuntimeTests"` passed `24/24`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with existing `LIB` path warnings only
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with `260` `WowViewer.Core.Tests` and `36` `WowViewer.Core.PM4.Tests`
	- fixed local client proof on `H:/CLIENTS/World of Warcraft Cata beta 11927` for `Creature/Wolf/Wolf.m2`, sequence `20`, time `500`, loaded exact external `Creature/Wolf/Wolf0096-00.anim` and printed the new pose/skinning/consumer/submission lines
- proof boundary:
	- this is still library/runtime/inspect ownership proof, not active `WowViewer.App` or old `MdxViewer` visual runtime signoff
	- particles, ribbons, hit testing, final shader parity, and app renderer cutover remain open follow-on work

## Apr 15, 2026 - wow-viewer M2 now owns first-pass animated material/light state, and the next chat should resume after that baseline instead of re-planning it

- landed first-party `wow-viewer` M2 seams now include:
	- strict `MD20` root parsing and exact `%02d.skin` choose/load/init runtime
	- first-party geometry/material tables plus structured section or pass or material routing
	- effect-recipe classification in `WowViewer.Core.Runtime/M2`
	- external `%04d-%02d.anim` selection/load plus alias ready-state ownership
	- first-party animated block definitions and parsing for colors, texture weights, texture transforms, and lights
	- first-pass animated runtime evaluation for material/pass and light state from root or external payloads
	- `WowViewer.Tool.Inspect m2 inspect --time-ms` output for evaluated animated runtime state
- bounded proof completed in this chat:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter M2FoundationTests` passed `19/19`
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug` succeeded
	- real asset proof on fixed client root `H:/CLIENTS/World of Warcraft Cata beta 11927` for `Creature/Wolf/Wolf.m2`, sequence `20`, loaded external `Creature/Wolf/Wolf0096-00.anim` and printed `ANIM.RUNTIME` from the first-party evaluator
- remaining M2 seam for the next fresh chat is now narrower and should be treated as such:
	- bone pose solve and animated skinning application
	- applying evaluated material/light state into a real renderer or consumer path instead of inspect-only ownership
	- remaining model-local lighting or emissive semantics not yet consumed by rendering
	- particle or ribbon or other family-specific runtime ownership as needed before submission work
	- explicit scene submission or batching coordinator and a consumer cutover beyond inspect
- important boundary:
	- this does not mean active-viewer runtime parity or full first-party M2 renderer closure; it is parser/runtime/evaluator ownership plus inspect proof only

## Apr 15, 2026 - uv-managed training bootstrap is now implemented, and train_v7 no longer silently falls back to CPU

- implemented new dedicated training bootstrap scripts:
	- `gillijimproject_refactor/scripts/setup_training_env.ps1`
	- `gillijimproject_refactor/scripts/setup_training_env.sh`
	- shared deps file `gillijimproject_refactor/scripts/requirements_train_v7.txt`
- bootstrap behavior now:
	- uses `uv` to install a pinned Python runtime (default `3.11`), create a dedicated training venv (default `.venv-train`), install common deps, then install torch/torchvision/torchaudio from backend-specific indexes (`cu128`, `rocm6.2.4`, `cpu`, or standard PyPI for `mps`)
	- supports explicit backend selection plus `auto` backend resolution
	- runs post-install runtime validation for requested accelerator capability unless `DryRun`/`--dry-run` is used
- `src/WoWMapConverter/scripts/train_v7.py` now fails fast when CUDA is unavailable unless `--allow-cpu` is explicitly provided:
	- new resolver reports Python and torch diagnostics (`torch.__version__`, `torch.version.cuda`, `torch.version.hip`) and exits with remediation guidance
	- explicit CPU runs remain possible for debug-only usage via `--allow-cpu`
	- this removes the old silent `torch.cuda.is_available()` -> CPU fallback behavior that triggered the long-running accidental CPU training run
- docs updated:
	- `gillijimproject_refactor/docs/VLM_Training_Guide.md` now includes the uv bootstrap workflow and the explicit `--allow-cpu` note

## Apr 15, 2026 - training env drift was real, and M2 ownership is now explicitly back on the wow-viewer first-party path

- the CPU training fallback was an environment failure, not an acceptable runtime choice:
	- configured interpreter was `i:/parp/parp-tools/.venv/Scripts/python.exe`
	- that environment currently reports `torch 2.11.0+cpu`, `torch.version.cuda = None`, `torch.cuda.is_available() = False`
	- the host GPU is visible and healthy through `nvidia-smi` (`NVIDIA GeForce RTX 4070 Ti SUPER`, driver `595.97`)
	- `src/WoWMapConverter/scripts/train_v7.py` still does a silent capability gate: `use_cuda = torch.cuda.is_available()` then `device = torch.device("cuda" if use_cuda else "cpu")`, so a CPU-only torch build falls straight through into live CPU training instead of failing fast
- active workflow correction from the user:
	- stop treating ad hoc `.venv` reuse or random conda fallback as acceptable for training
	- future training environment work should move to an explicit `uv`-managed bootstrap with deployment scripts that install the correct hardware-specific torch/runtime stack and verify the target accelerator before training starts
	- trainer entrypoints should fail loudly when a GPU-required run lands in a CPU-only environment instead of quietly training on CPU
- active M2 ownership correction from the same user directive:
	- stop spending continuation budget on more `MdxViewer` bandaid fixes as the design owner for M2 rendering
	- treat the new corrective path as full first-party M2 parser plus runtime plus renderer ownership in `wow-viewer`
	- use `MdxViewer`, `WarcraftNetM2Adapter`, wowdev docs, native-client notes, and `noggit-red` only as extraction/reference inputs unless a bounded compatibility proof is explicitly requested
	- new planning surface for that reset is `.github/prompts/wow-viewer-full-m2-parser-renderer-plan.prompt.md`, with the existing staged runtime prompts staying as the narrower follow-on slices

## Apr 15, 2026 - corpus export now has resume-aware map completion instead of unconditional full reruns

- the active corpus workflow bug was not imagined:
	- `scripts/export_ml_corpus.ps1` had been deliberately changed to re-export every configured map so stale partial roots would not survive just because `dataset/` existed
	- that also meant completed roots kept being re-exported on every broad run, including expensive fixed-client jobs the user already had on disk
- active behavior now:
	- `scripts/export_ml_corpus.ps1` supports `-Resume`
	- `WoWMapConverter.Cli ml-corpus` supports `--resume`
	- both paths now write `.ml-corpus-resume-state.json` inside each dataset map root and use it together with `ml_dataset_manifest.json` freshness to decide whether a map is complete, needs harvest only, or needs a full export
	- resume skips fully completed roots, reruns only incomplete roots, and can do harvest-only recovery when export already finished but manifest coverage is stale
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` succeeded
	- `scripts/export_ml_corpus.ps1 -DryRun -Resume` now cleanly skips `original_development/development` without printing export or harvest commands for that same map
- important boundary:
	- wrapper dry-run still does not execute live `ml-list-maps`, so `all_maps` clients appear empty there by design; use a real `-Resume` run when you want actual map scheduling instead of a syntax or routing smoke

## Apr 14, 2026 - `original_development` now stages onto a real 4.0.0.11927 base client instead of exporting from the sparse loose tree directly

- the old `original_development` corpus entry was still pointing straight at `gillijimproject_refactor/test_data/original_development`, which left the exporter on a sparse loose root with no real client `Data` surface behind it
- a reusable staging helper now exists at:
	- `gillijimproject_refactor/scripts/stage_original_development_overlay.ps1`
- current staged overlay root is:
	- `i:/parp/parp-tools/output/tmp/original_development_client_4_0_0_11927`
- that staged root is composed from:
	- base client `H:\CLIENTS\World of Warcraft Cata beta 11927\Data`
	- loose development map `gillijimproject_refactor/test_data/original_development/World/Maps/development`
	- development minimaps `gillijimproject_refactor/test_data/development/World/Textures/Minimap`
- `scripts/ml_corpus_fixed_clients.json` now points the `original_development` client entry at the staged 11927-backed root instead of the raw loose tree
- bounded real-data proof succeeded with:
	- `dotnet run --project ...WoWMapConverter.Cli.csproj -- ml-export --client i:/parp/parp-tools/output/tmp/original_development_client_4_0_0_11927 --map development --tile 31_36 --out i:/parp/parp-tools/output/build-validation/original_development_11927_overlay_probe --listfile i:/parp/parp-tools/gillijimproject_refactor/test_data/community-listfile-withcapitals.csv`
- proof outcome on `development_31_36`:
	- loose `development.wdt` / `development.wdl` plus split `_tex0` / `_obj0` were found from the overlay root
	- hashed minimap resolution succeeded against the 11927 base data and wrote `images/development_31_36.png`
	- exporter produced `heightmap_local`, `heightmap_global`, `object_visibility_mask`, `pm4_mask`, and semantic rasters under `output/build-validation/original_development_11927_overlay_probe`
	- stitched full-map outputs for the bounded one-tile probe were also written under the same root
- important boundary:
	- this is proven against 4.0.0.11927 because it is the closest fixed local base client currently on disk to the PM4 build hints (`11927` / `12304`)
	- if a real `4.0.1.12304` client is found later, rebuild the overlay root against that client and prefer it over `11927`

## Apr 14, 2026 - 4.0.0.11927 world-root recovery now uses fast core export and no longer preserves partial map roots silently

- the user-reported Cataclysm beta corpus gap was real:
	- `datasets/4_0_0_11927` only had `Azeroth`, `EmeraldDream`, and `LostIsles` when checked
	- `Azeroth/ml_dataset_manifest.json` showed only `1` processed tile before recovery
	- fixed local client discovery on `H:\CLIENTS\World of Warcraft Cata beta 11927` proved that missing worlds like `Kalimdor` were present in the client and that the dataset state, not the client, was incomplete
- workflow fixes landed to stop partial roots from being treated as complete:
	- `gillijimproject_refactor/scripts/export_ml_corpus.ps1` now uses the correct map-discovery client path during dry-run/staged-client cases, removes the accidental duplicate discovery override, and no longer skips a map just because `dataset/` already exists
	- `-Force` on the wrapper now clears an existing map root explicitly instead of leaving stale partial exports behind
	- `WoWMapConverter.Cli ml-export` now accepts `--skip-derived-assets`, and `WoWMapConverter.Core/VLM/VlmDatasetExporter.ExportMapAsync(...)` uses it to skip tilesets, stitched outputs, and semantic postprocess while still writing tile JSON, local heightmaps, and global-normalized heightmaps
- bounded real-data recovery on the fixed local 4.0.0.11927 client now produced these root-level manifests under `datasets/4_0_0_11927`:
	- `Azeroth`: `839` tiles processed, including a follow-up fast rerun that backfilled the previously missing `839/839` global heightmaps after an interrupted earlier run
	- `Kalimdor`: `1011` tiles processed
	- `Deepholm`: `100` tiles processed (archive directory resolved as `Deephome` during export)
	- `EmeraldDream`: `256` tiles processed
	- `LostIsles`: `149` tiles processed
- important boundary:
	- this recovery closed the missing world-root / missing-global-heightmap problem for the targeted 4.0.0.11927 roots, but it did not make every per-channel surface uniformly dense; current manifests still show partial source-minimap coverage on some maps (for example `Kalimdor 1006/1011`, `Azeroth 835/839`, `EmeraldDream 91/256`), so do not describe the whole 4.0.0.11927 corpus as fully dense across every harvested channel without map-by-map proof

## Apr 14, 2026 - Terrain-only rebake now has bounded real-data proof on a fixed 3.0.1 client tile

- bounded exporter proof was run against the fixed local `3.0.1.8303` client root:
	- command shape: `dotnet run --project .../WoWMapConverter.Cli.csproj -- ml-export --client "H:/CLIENTS/3.X_Pre-Release_Windows_enUS_3.0.1.8303/World of Warcraft" --map EmeraldDream --tile 26_26 --out i:/parp/parp-tools/output/build-validation/emeralddream_26_26_terrain_rebake_probe`
	- tile `EmeraldDream_26_26` was chosen because the existing harvested corpus already had a legacy `terrain_only_minimap`, real `chunk_layers`, real tileset textures, and an object mask
- proof outcome on the fresh export:
	- exporter loaded and converted `4` real tileset textures from the client archives
	- fresh tile JSON under `output/build-validation/emeralddream_26_26_terrain_rebake_probe/dataset/EmeraldDream_26_26.json` now includes `terrain_only_minimap` plus the new semantic rasters `holes_mask`, `area_id_map`, `chunk_flags_map`, `liquid_type_map`, and `dominant_effect_id_map`
	- cropped comparison against the legacy dataset `terrain_only_minimap` showed the strongest changes localized to the object-mask footprint, with `782 / 861` masked pixels changing by more than `8` intensity levels and masked-region RGB MAE about `19.49`
	- the updated `src/WoWMapConverter/scripts/audit_v7_signals.py` run on the fresh probe reported `1/1` coverage for `terrain_only_minimap`, `holes_mask`, `area_id_map`, `chunk_flags_map`, `liquid_type_map`, and `dominant_effect_map`
	- bounded real-data brush proof also now exists for the new archetype pass: `ml-harvest-brushes --dataset-root i:/parp/parp-tools/datasets/original_development/development --output-dir i:/parp/parp-tools/output/build-validation/brush-imprints/original_development_archetype_probe_20260414 --limit 6 --write-previews` processed `6` tiles, wrote `117` group files, and emitted `brush_archetype_manifest.json` with `115` archetypes plus per-group `archetype_id` / `archetype_key` / `archetype_label` / `shape_fingerprint`
- important boundary:
	- the checked-in `test_data/original_development` split root still lacks the source BLP payloads needed for honest texture-rebake proof; a bounded probe there reached the export path but logged `Exported 0 textures`, so real rebake validation should use a fixed local client root or a staged archive-backed copy when texture evidence matters

## Apr 14, 2026 - V7.6 docs now distinguish the paired-output branch from the active harvested-corpus terrain line

- added a separate V7.6 doc set instead of overloading the V7.5.1 terrain docs:
	- `gillijimproject_refactor/docs/v76-model-architecture-guide.md`
	- `gillijimproject_refactor/docs/v76-output-dataset-spec.md`
- active documentation boundary is now explicit:
	- `v75-model-architecture-guide.md` still owns the active grounded multichannel terrain-regressor story over harvested `datasets/`
	- V7.6 is documented as a separate paired-output image-to-height+albedo branch built around `cache_v7_6_data.py`, `train_v7_6.py`, `inference_v7_6.py`, and `stitch_full_map.py`
	- V7.6 predicted outputs are now documented as a structured derivative dataset surface, not as harvested truth and not as a replacement label for `datasets/`
- shared docs were updated to route readers correctly:
	- `README.md`
	- `docs/ML_DATASET_GROUNDING.md`
	- `docs/VLM_DATASET_EXPORTER.md`
	- `docs/VLM_Training_Guide.md`
	- `docs/V7_HEIGHT_REGRESSOR.md`
	- `docs/v75-model-architecture-guide.md`
- important boundary:
	- V7.6 remains a documented code branch with loose-file current outputs and heuristic world-scale assumptions; the new spec defines the intended structured output package but does not mean the inference scripts already emit it

## Apr 14, 2026 - Dataset grounding docs now make the real-data provenance and channel policy explicit

- documentation now has a dedicated grounding surface at:
	- `gillijimproject_refactor/docs/ML_DATASET_GROUNDING.md`
	- linked from `gillijimproject_refactor/README.md`, `docs/VLM_DATASET_EXPORTER.md`, `docs/VLM_Training_Guide.md`, and `docs/v75-model-architecture-guide.md`
- active public framing is now explicit:
	- the terrain corpus is harvested from real client roots or the checked-in real `original_development` split-root seam
	- GAN is a training-time refinement objective, not a dataset generator and not a source of ground-truth supervision
	- deterministic derived channels are allowed only when they are reproducible transforms over harvested real tile assets
- active channel policy is now explicit too:
	- brush harvesting remains the trusted patch-scale archaeology channel and part of the current grounded training story
	- prefab work remains in the repo as research or review tooling, but it is now explicitly deferred from the trusted active supervision narrative until it is validated to the same standard as brush harvest
- important boundary:
	- `terrain_only_minimap` is still a derived cleaned surface rather than a raw capture, but it is documented as a deterministic cleanup over real exported minimap plus mask channels rather than as synthetic label generation

## Apr 14, 2026 - Archive-backed corpus export now stages mounted clients through both the PowerShell workflow and direct `ml-corpus`

- the archive workflow is no longer just documentation:
	- `scripts/wowarchive_client_staging.ps1` now contains reusable mount or stage or prune helpers for WoWArchive-backed client roots
	- `scripts/stage_wowarchive_client.ps1` is the standalone helper for staging one mounted client and pruning stale staged copies
	- `scripts/export_ml_corpus.ps1` now prefers fixed local roots when present and otherwise stages archive-backed client roots into `output/tmp/wowarchive-clients` before map discovery or export
	- `src/WoWMapConverter/WoWMapConverter.Cli/Program.cs` `ml-corpus` now honors the same `mount_root` or `mount_script` or `staging_root` or `prune_staged_clients` policy directly instead of relying on the PowerShell wrapper
- current config surface for that workflow now includes:
	- top-level `mount_root`, `mount_script`, `staging_root`, and `prune_staged_clients`
	- per-client `local_client_path`, `archive_client_path`, `local_minimap_root`, `archive_minimap_root`, and `all_maps`
	- `scripts/ml_corpus_fixed_clients.json` now uses explicit `local_client_path` plus `archive_client_path` entries for the verified WoWArchive-backed 0.x or 3.x clients, while `4_0_0_11927` stays local-only because the mounted `0.X-3.X` archive does not appear to contain that build
- validated state in this chat:
	- synthetic mounted-client smoke proved stage plus prune behavior through `stage_wowarchive_client.ps1`
	- synthetic `export_ml_corpus.ps1 -DryRun` proved archive-backed config entries resolve to the staged working root before `ml-export` would run
	- real-data direct CLI dry-run against mounted `3.X_Pre-Release_Windows_enUS_3.0.1.8303/World of Warcraft` proved `archive_client_path` resolves to the staged working root and reports the mounted source path explicitly
	- synthetic direct CLI `--harvest-only` proof with `all_maps: true` staged a fake archive-backed client, discovered `SynthMap`, and pruned a stale staged client directory end to end
- important boundary:
	- live `all_maps` discovery is still expensive when you force a dry-run directly against a mounted archive source because the dry-run intentionally avoids the actual copy step; the intended fast path remains the real staged run

## Apr 14, 2026 - WoWArchive should be treated as a mounted source plus staged-client workflow

- user provided a new canonical client-access rule for broad multi-build work:
	- source large client coverage from `G:\WoW\WoWArchive-0.X-3.X`
	- mount it with `G:\WoW\WoWArchive-0.X-3.X\MountAll.bat`
	- treat the mounted archive as the source surface only, not the preferred processing root
	- for repeated or wide export or audit or inspect or training-prep work, copy the required client into `output/tmp/wowarchive-clients/` first
	- delete staged clients that are no longer needed after the run
- practical implication for continuations:
	- stop treating direct mounted-archive reads as the default path for large archive-backed jobs
	- keep validation notes explicit about whether a proof used fixed `H:\CLIENTS\...` roots, direct mounted paths, or staged copies
	- route future client-root staging questions through `.github/skills/wowarchive-client-staging/SKILL.md`

## Apr 14, 2026 - Dataset-builder convergence is now an explicit workflow rule

- user directive is now explicit:
	- all shared dataset or export or terrain-supervision seams should converge into `wow-viewer`
	- the intended long-range dataset-builder surface is a new `wow-viewer` tool over shared libraries, not more architecture inside `WoWMapConverter`
	- the user-facing target should be shared library plus CLI plus viewer/editor workflows plus dataset explorer plus supervised training tooling over the same contracts
	- the workflow must stay Bring Your Own Data; do not ship copyrighted corpora, trained models, or model outputs
	- CUDA can remain an early host, but the long-range orchestration shape should not hard-lock the design away from Vulkan or OpenCL or MLX or other local runners
- practical implication for the next continuation:
	- stop treating `WoWMapConverter.Core/VLM` fixes as the default path when the real issue is shared ownership, new capability, or artifact semantics
	- route dataset-builder planning through `.github/prompts/wow-viewer-dataset-builder-plan.prompt.md` and the continuity file `plans/wow_viewer_dataset_builder_tool_plan_2026-04-14.md`
	- use `WoWMapConverter` and `MdxViewer` dataset/export code as extraction or compatibility references only unless a bounded legacy hotfix is explicitly requested
- important boundary:
	- this is a workflow and ownership correction only
	- no `wow-viewer` dataset-builder tool implementation or full shared-contract cutover is landed yet

## Apr 13, 2026 - Fresh-chat continuation should resume with the full multi-client dataset refresh and training run, not more narrow probes

- explicit user directive for the next chat:
	- run the dataset extract across all already-indicated fixed client roots and configured corpus maps
	- harvest everything, re-audit the outputs, and then train the terrain model on the refreshed corpus
	- do not spend the next continuation on more narrow one-tile archaeology unless a concrete exporter blocker stops the full run
- practical next-run sequence:
	- rerun the corpus export/harvest flow against the fixed configured roots under `datasets/`
	- make sure the rerun picks up the latest exporter corrections, including:
		- MCSH no longer participating in `terrain_only_minimap`
		- MCCV inverse cleanup parity
		- loose override precedence
		- geometry-derived object masks
	- rerun dataset signal audit on the refreshed roots
	- launch the intended V7.5.1 training pass with the corrected schedule, then inspect the real training outputs instead of treating the older GAN-off rerun as closure
- known unresolved blocker to keep visible, but not to let sprawl again:
	- Cataclysm `MH2O` / liquid-loss behavior is still unresolved for the failing `LostIsles_23_24` path and likely sits in the current MH2O parse path rather than the later stitching stage

## Apr 13, 2026 - `terrain_only_minimap` no longer treats stitched MCSH shadows as removable contamination

- the over-mask bug was real in the active V7.5 exporter path:
	- `VlmDatasetExporter` built the `terrain_only_minimap` removal surface from stitched alpha masks plus the stitched shadow map
	- real exported 3.0.1.8303 corpus tiles such as `datasets/3_0_1_8303/EmeraldDream/dataset/EmeraldDream_24_25.json` showed the bad pattern clearly: `shadow_maps` present, `alpha_masks` empty, `no_liquid_minimap` null, `object_visibility_mask` null, `pm4_mask` null, yet `terrain_only_minimap` still existed
- active behavior now:
	- `terrain_only_minimap` only unions stitched alpha masks plus object, PM4, and liquid masks
	- stitched `MCSH` shadow output is still exported for diagnostics, but it is no longer fed into the minimap inpaint-removal path
- focused validation completed:
	- `dotnet test i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/WoWMapConverter.Core.Tests.csproj -c Debug --filter VlmDatasetExporterTests` passed (`12/12`)
	- bounded real-data re-export on `H:\CLIENTS\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft` for `EmeraldDream --tile 24_25` now keeps `shadow_maps`, leaves `alpha_masks` empty, leaves `no_liquid_minimap` / object / PM4 masks null, and writes `terrain_only_minimap: null` under `output/build-validation/emeralddream_tile_24_25_shadow_rule/dataset/EmeraldDream_24_25.json`
- important boundary:
	- this fixes the shadow-only false-positive cleanup path in code, focused tests, and bounded real-data export proof
	- it does not change shadow export itself, only whether MCSH participates in terrain-only minimap masking

## Apr 13, 2026 - Shared md5translate and exporter asset reads now prefer loose overrides before archive-backed copies

- the loose-file patching gap was real in two different places:
	- `WowViewer.Core.IO.Files.Md5TranslateResolver` loaded archive `md5translate` candidates before loose files, so a loose patched `.trs` or `.txt` could not override an archive-backed mapping once the index was built
	- `VlmDatasetExporter` still had several archive-first virtual asset reads for mapped minimaps, model bounds, WMO split-group footprint reads, tileset BLP export, and LK tile scoring
- active behavior now:
	- `Md5TranslateResolver.TryLoad(...)` checks loose disk candidates before archive candidates for both shared defaults and map-specific extra candidates
	- `VlmDatasetExporter` now routes minimap hash mappings, tileset texture export, model bounds reads, model-footprint reads, split WMO group reads, and LK tile-content scoring through a loose-first virtual asset helper before falling back to archive reads
- focused validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter Md5TranslateResolverTests` passed (`3/3`)
	- `dotnet test i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/WoWMapConverter.Core.Tests.csproj -c Debug --filter VlmDatasetExporterTests` passed (`6/6`)
	- bounded real-data baseline on the untouched `H:\CLIENTS\World of Warcraft Cata beta 11927` root exported `LostIsles_29_32` from archive-backed minimap path `textures/minimap/807183b22bf2ba9e1f0305a2d345c015.blp`, with logged center pixel `79,142,255,255`
	- bounded overlay proof on `output/tmp/cata_loose_override_overlay_20260413` reused the real 11927 archives via a `Data` junction plus a loose `World/Maps/LostIsles/md5translate.trs` override mapping to `textures/minimap/override.png`; the exported `LostIsles_29_32.png` center pixel changed to `255,0,255,255`, proving the loose override path actually won end-to-end over the archive-backed baseline
- important boundary:
	- this is real-archive validation plus a temporary overlay proof, not evidence that the stock 11927 client already contains natural loose minimap overrides
	- WL fallback for missing Cataclysm liquid masks is still deferred until there is concrete evidence that those client roots actually ship usable `WL*` payloads

## Apr 13, 2026 - The broader cleaned-input rerun completed on current dataset roots, but this particular run stayed GAN-off and should be treated as the non-adversarial cleaned-input baseline

- the larger rerun requested after the exporter refresh completed under `output/ml-training/v7_5_1_cleaned_inputs_20260413_rerun`
- that run used the current harvested dataset-root set including:
	- `datasets/0_7_0_3694/EmeraldDream`
	- `datasets/3_0_1_8303/{EmeraldDream,Northrend,PVPZone01..04}`
	- `datasets/3_3_5_12340/{Azeroth,EmeraldDream}`
	- `datasets/4_0_0_11927/{Azeroth,Deepholm,EmeraldDream,LostIsles}`
	- `datasets/original_development/development`
- outcome from `training_log.json`:
	- run stopped after `12` epochs
	- best validation loss was about `0.0691` at epoch `7`
	- final epoch was about train `0.0643`, val `0.0794`
	- metadata recorded `921` loaded samples before split/curation, `573` train, `98` val, and `13` input / `2` output channels
- important caveat:
	- this run was not the GAN-enabled V7.5.1 cadence proved in the refreshed-root smoke
	- the saved metadata still shows `start_gan_epoch = 101`, `gan_enabled = false` for all `12` epochs, and `gan_burst_after_best = 0`, so treat this run as a cleaned-input non-adversarial baseline rather than the final intended trainer schedule
- audit context captured alongside the rerun in `output/build-validation/ml-audit/v7_5_1_dataset_signal_audit_20260413_rerun.txt`:
	- `datasets/original_development/development` now reports `terrain_only_minimap` on `198/352` tiles and nonzero object masks on `49/352`
	- `datasets/4_0_0_11927/Deepholm` reports `terrain_only_minimap` on `25/100` tiles and nonzero object masks on `39/100`
	- `datasets/4_0_0_11927/LostIsles` reports `terrain_only_minimap` on `77/149` tiles and nonzero object masks on `12/149`

## Apr 13, 2026 - Exporter object masks now use geometry-derived footprints, and refreshed roots prove the new path is trainable but not fully closed for 4.x

- the exporter-side object-mask closure landed in `VlmDatasetExporter`:
	- `VlmObjectPlacement` now carries `model_path`
	- `object_visibility_mask` no longer starts from shadow rectangles plus per-object circles
	- the exporter now caches per-model footprint polygons from real `M2`, `MDX`, and `WMO` geometry, projects those hulls into tile UV space, and only falls back to bounds polygons or ellipses when geometry is unavailable
	- the current `M2` footprint read is already shared-format shaped rather than 4.x-only exporter logic: it builds hulls directly from `M2GeometryReader` vertex positions
- refreshed real-data dataset roots completed after that change:
	- `datasets/original_development/development`: `352` tiles exported and harvested
	- `datasets/4_0_0_11927/Deepholm`: `100` tiles exported and harvested
	- `datasets/4_0_0_11927/LostIsles`: `149` tiles exported and harvested
- mixed-root retrain smoke on those refreshed roots completed successfully:
	- dataset roots: refreshed `development` + `Deepholm` + `LostIsles`
	- usable samples: `466`
	- train/val: `418 / 48`
	- pinned validation refs still printed `development:development_0_0`
	- epoch 1 finished with train `0.2071`, val `0.1754`, GAN `on`, and a saved best model under `output/tmp/v7_5_1_geometry_mask_refresh_smoke_20260413`
- important boundary from refreshed exporter-output audit:
	- the new geometry path materially improved the `original_development` root, where refreshed object-mask PNGs averaged about `3.58%` coverage across `38` mask-bearing tiles and topped out around `17.09%`
	- the Cataclysm roots are not fully closed yet: refreshed `Deepholm` masks still averaged about `33.49%` coverage with a worst tile near `88.69%`, and refreshed `LostIsles` masks averaged about `20.72%` with a worst tile near `71.17%`
	- user requirement correction on Apr 15: for most object-masking work, treat `3.3.5.12340` plus `4.0.0.11927` as the paired validation floor; `4.0.0` is close enough to Wrath to matter, but not close enough to skip a bounded proof on both roots
	- treat the current state as trainable-with-guardrails, not final exporter signoff for those 4.x maps

## Apr 13, 2026 - MCCV export and cleanup now match MdxViewer semantics instead of the old subtraction heuristic

- the earlier MCCV bug was not just channel order:
	- `MdxViewer` exports MCCV PNGs with raw MCCV bytes preserved in PNG channel order for tooling compatibility
	- the terrain renderer itself decodes those raw bytes as BGRA and applies them as a multiplicative tint via `clamp(vertexColor.rgb * 2.0, 0.0, 2.0)`
- active VLM behavior now matches that contract:
	- `VlmDatasetExporter` writes `mccv_map` in the same raw channel order as `MdxViewer.Export.TerrainMccvIo`
	- `VlmMinimapCleanupService.RemoveMccvTint(...)` now decodes the raw-view MCCV PNG back to renderer tint and removes it by dividing by the same multiplicative factor the viewer shader uses, instead of subtracting channel deltas around `127`
- validation completed:
	- focused tests in `WoWMapConverter.Core.Tests/VLM/VlmMinimapCleanupServiceTests.cs` passed after the change
	- bounded real-data probe on `Deepholm` under the 4.0.0.11927 client regenerated `mccv_map` and `no_mccv_minimap` under `output/tmp/deepholm_mccv_inverse_probe_20260413`
- important boundary:
	- older already-exported V7.5/V7.5.1 dataset roots still contain stale MCCV-derived artifacts and need re-export to pick up the corrected cleanup behavior

## Apr 13, 2026 - `Deepholm` now recovers to archive-backed `Deephome`, and corpus harvest skips empty exports instead of aborting the batch

- the current 4.0.0.11927 client issue was not a missing map payload; it was an internal directory-name mismatch:
	- the failing user-facing label was `Deepholm`
	- the archive-backed client actually stores the map under `World/Maps/Deephome/...`
- active behavior now:
	- `VlmDatasetExporter` first tries `Map.dbc` as before, then falls back to archive-known `World/Maps/*/*.wdt` names and can recover near matches like `Deepholm -> Deephome`
	- both `WoWMapConverter.Cli ml-corpus` and `scripts/export_ml_corpus.ps1` now skip `ml-harvest` with a warning when an export produced no tile JSON files instead of aborting the whole batch on an empty dataset root
- real-data proof captured:
	- bounded probe command against `H:\CLIENTS\World of Warcraft Cata beta 11927` with `--map Deepholm --limit 1` resolved `Deephome`, found `World\Maps\Deephome\Deephome.wdt` in MPQ, loaded `100` WDT tiles, and exported `1` tile to `output/tmp/deepholm_alias_probe_20260413`
- important boundary:
	- this proves the failing Deepholm lookup path on real data and the new non-fatal empty-harvest behavior in code/build terms
	- the full forced V7.5.1 corpus refresh and retrain still need to complete after this recovery

## Apr 13, 2026 - Datasets now live under `datasets/`, with HF-style metadata and split terrain/minimap roots

- the active ML dataset workflow now targets `i:/parp/parp-tools/datasets` instead of `output/ml-corpus`
- `ml-corpus` and `scripts/export_ml_corpus.ps1` now support per-client `label` and `minimap_root` config entries, and the fixed-clients config under `scripts/ml_corpus_fixed_clients.json` routes all configured builds/maps into `datasets/<label>/<map>`
- `ml-harvest` now also writes:
	- `metadata.jsonl`
	- `dataset_info.json`
- those files are root-level, HF-friendly imagefolder metadata surfaces on top of the existing JSON/bin/image layout
- conservative cleanup follow-up:
	- legacy `output/ml-corpus` was archived to `output/archive/ml-corpus_legacy_20260413`

## Apr 13, 2026 - Bounded V7.5 proof now works with `original_development` terrain input plus a separate minimap-only root

- bounded real-data proof now exists for the V7.5 export path using:
	- terrain source: `gillijimproject_refactor/test_data/original_development/World/Maps/development`
	- minimap-only root: `gillijimproject_refactor/test_data/development`
	- command shape: `ml-export --client <original_development> --minimap-root <development> --map development --limit 4`
- exporter behavior changed again for sparse loose LK roots:
	- `VlmDatasetExporter` now filters WDT `MAIN`-flagged tiles against actually reachable root ADT files before bounded tile selection
	- on the approved `original_development` root this reduced the working tile set from `1496` WDT-flagged entries to `352` reachable root ADTs
- bounded proof result now lives under `datasets/original_development/development_proof_20260413`:
	- `4` tiles exported, `0` skipped
	- representative sample `development_31_36.json` includes `no_mccv_minimap`, `object_visibility_mask`, `pm4_mask`, `no_object_minimap`, and `terrain_only_minimap`
	- representative sample `development_34_34.json` includes `terrain_only_minimap` and `no_liquid_minimap` even when object/PM4 mask fields are null
	- `ml-harvest` was rerun there and wrote `ml_dataset_manifest.json`, `metadata.jsonl`, and `dataset_info.json`
- important boundary:
	- this is bounded export proof only, not full-map export proof or model-training proof
	- `test_data/WoWMuseum/335-dev` exposed `md5translate.trs` entries for development minimaps but did not provide the loose tile payloads needed for this bounded run

## Apr 13, 2026 - V7 export proof must keep `original_development` as the terrain source and name any minimap root explicitly

- after an invalid proof attempt pointed `ml-export` at the wrong development tree, the active rule for the V7 terrain-model export path is now explicit:
	- terrain and ADT sampling for this proof path must come from `gillijimproject_refactor/test_data/original_development/World/Maps/development`
	- if source minimaps are needed and they are not present under that root, they must come from an explicit separate minimap root rather than broadening the terrain input root
- active tooling follow-up:
	- `WoWMapConverter.Cli ml-export` now accepts `--minimap-root <dir>`
	- `VlmDatasetExporter.ExportMapAsync(...)` now keeps minimap lookup separate from the terrain client root when that option is provided
- proof boundary:
	- this is source-boundary enforcement plus build-path correction
	- no approved minimap-root rerun has been completed yet after the correction

## Apr 13, 2026 - Fallback object masking is no longer WMO-only, and the converter build blocker is cleared

- after the user called out that fallback masking was missing whole object families, the active object-mask paths were checked and the bug was real:
	- `VlmDatasetExporter.BuildObjectVisibilityMask(...)` only projected `wmo` placements
	- `train_v7.py` and `infer_v7.py` also explicitly skipped non-`wmo` objects in the coarse fallback object-context mask path
- active behavior now:
	- exporter fallback masks include all projected object placements, not just WMOs
	- trainer and inference fallback object masks also include all object families instead of filtering to `wmo`
	- PM4/seeded masks still take precedence where present, but maps without PM4 are no longer blind to M2/doodad occlusion in the fallback path
- separate follow-up from the same slice:
	- the old `VlmMinimapCleanupService.cs` compile blocker was fixed and `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` now succeeds again with existing warnings only
- proof boundary:
	- this is build proof plus Python syntax proof for the touched masking scripts
	- no real-data corpus re-export has been run yet to confirm the wider fallback object masks on tile outputs

## Apr 13, 2026 - V7.5 now means terrain-only minimap precedence, not a wider tensor contract

- after the user pushed for explicit compensation of minimap contamination from alpha overlays, shadow (`MCSH`-style) darkening, lighting, liquids, and object occlusion, the active terrain line was bumped from V7.4 to V7.5 at the dataset-contract level
- active code behavior now:
	- `WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` can emit `terrain_only_minimap`
	- that cleaned image starts from `no_mccv_minimap` when available and then inpaints out the union of available object, PM4, liquid, stitched alpha, and stitched shadow masks
	- `train_v7.py` and `infer_v7.py` now prefer `terrain_only_minimap`, then `no_object_minimap`, then `no_mccv_minimap`, then raw `image`
	- the trainer cache version was bumped so old dataset index caches do not silently hide the new field
	- `audit_v7_signals.py` now reports `terrain_only_minimap` coverage and reflects the new effective minimap precedence
	- docs now include `docs/v75-model-architecture-guide.md`, and the old `v74` guide is explicitly marked superseded
- important boundary:
	- this is code-path proof and contract proof only
	- Python syntax validation passed via `py_compile` for the updated trainer, inference, and audit scripts
	- file-level diagnostics were clean for the touched exporter and script files
	- a full `WoWMapConverter.Core` build is still blocked by pre-existing `VlmMinimapCleanupService.cs` compile errors unrelated to this slice, so there is not yet full-project compile proof for V7.5
	- no real-data dataset re-export or retraining run has been captured yet for the new `terrain_only_minimap` path

## Apr 13, 2026 - Full improved V7.4 run was relaunched with pinned `development_0_0` validation and safer object-mask precedence

- after the user pointed out that coarse object masks were wiping legitimate terrain on the left side of the preview tile, `train_v7.py` and `infer_v7.py` were corrected so object-context precedence is now:
	- precise exported silhouettes first: `object_visibility_mask_cv2`, `pm4_mask`, `pm4_object_mask`, `collision_mask`
	- exported seed mask next: `object_visibility_mask`
	- coarse fallback WMO-box projection only when no exported mask exists
- this avoids unioning a broad coarse WMO box over a tighter exported PM4/CV2 silhouette when one is available
- trainer-side validation selection also changed so the trusted reference tile `development_0_0` is always forced into validation and static previews when present in the loaded dataset roots
- bounded validation proof from a real-data smoke on `output/ml-corpus/4_0_0_12304_original/development`:
	- static preview tiles printed `development:development_0_0` first
	- the bounded run completed one epoch and wrote a best checkpoint under `output/tmp/v7_4_validation_pin_smoke_20260413`
- the first full improved launch under the new architecture was then restarted into `output/ml-training/v7_4_wdl_trestle_reflect_brush_bestburst_pinval_20260413` so the pinned-reference behavior applies from epoch `1`
- live launch facts now confirmed from terminal output:
	- `26` audited dataset roots selected
	- `6070` valid samples loaded
	- raw train/val changed to `5449 / 621` because the pinned development reference group is now held out for validation
	- curated train count changed to `3230`
	- static preview sentinels now start with `development:development_0_0`, then `Northrend:Northrend_20_24`
	- trainer is running on CUDA with AMP `bfloat16`, TF32 enabled, `disc_lr=5e-5`, and `--gan-burst-after-best 2`
- important upstream boundary:
	- sampled ML corpora checked during this change still had no exported precise PM4/MPRL mask payloads on representative development tiles, including `development_31_36` and `development_0_0`
	- the trainer and inference path are now ready to honor those precise masks, but the exporter still needs a real PM4 or MPRL-driven silhouette seam before that signal materially changes current corpora
- proof boundary:
	- this is code-path proof, validation-selection proof, and fresh launch proof; it is not yet retrained-model proof on the relaunched full run

## Apr 13, 2026 - Trainer-side object masks now reject pathological coverage, and validation grouping no longer collides across same-name dataset roots

- after the user called out that current training previews were still letting object masks dominate large parts of some tiles and that `development_0_0` had fallen out of active validation attention again, the active `train_v7.py` path was checked against real dataset roots and the bug was real:
	- several current exported `object_visibility_mask` payloads were still coarse enough to cover most of a tile, especially on `Deepholm`, because the trainer trusted seeded masks before any sanity bound
	- validation grouping was still keyed by short `dataset_name`, which can collide across multiple loaded roots that share names such as `EmeraldDream`
- active trainer behavior after this slice:
	- oversized object masks are now rejected instead of being passed through as context
	- the current caps are stricter for coarse paths than precise ones: precise masks may cover up to `50%`, seeded exported masks up to `25%`, and trainer-built fallback masks up to `20%`
	- the fallback trainer rasterizer now uses ellipses instead of axis-aligned rectangles for per-object footprints, which materially reduces over-coverage on broad AABB cases
	- validation grouping now keys by full dataset-root path plus map/block coordinates instead of short dataset name, so cross-version roots with the same leaf name do not merge into one validation group by accident
	- `development_0_0` is now explicitly re-forced into validation after the split as a belt-and-suspenders check, and the trainer prints pinned validation refs at startup
- real-data validation captured in this chat:
	- worst object-mask coverage on the audited `Deepholm` / `LostIsles` smoke subset dropped from near-full-tile seeded masks to a top observed coverage of about `10%`
	- a one-epoch mixed-root smoke on `original_development/development` + `4_0_0_11927/LostIsles` printed `Pinned validation refs: development:development_0_0` and kept that tile first in static previews
- important boundary:
	- this is trainer-side guardrail proof and startup proof only; the exporter still needs a true geometry/silhouette mask seam so the dataset can stop relying on seeded coarse masks in the first place

## Apr 13, 2026 - Development-map inference side-quest exposed tile-edge curl, and the active V7 path now anchors exported borders plus trains against them explicitly

- after running the epoch-51 `output/ml-training/v7_4_brush_channel_bestburst_20260413/best.pt` checkpoint against `output/ml-corpus/400_12304/development`, the first exported OBJs showed two distinct failure modes:
	- soft ramping across terrain transitions that should stay sharper
	- tile-border curl, where all four outer edges could slope away instead of staying stitchable for a quilted whole-map export
- active code changes now in place:
	- `gillijimproject_refactor/src/WoWMapConverter/scripts/infer_v7.py` now supports the live `13`-channel checkpoint contract including brush masks, and it now anchors the outer tile band back to the WDL prior with `--edge-anchor-width`
	- `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` now adds two new loss components:
		- `transition` loss: increases reconstruction pressure around sharp target terrain changes so cut or dry segments do not blur into ramps as easily
		- `tile_edge` loss: increases reconstruction pressure on the outer border band so the model learns stitchable tile edges instead of curled borders
- bounded real-data proof from the re-exported development batch in `output/tmp/v7_4_dev_infer_full_edgeanchored_20260413`:
	- adjacent seam mismatch `development_31_36 south` vs `development_31_37 north` dropped from about `118.8` mean 16-bit units to about `13.4`
	- adjacent seam mismatch `development_30_36 east` vs `development_31_36 west` dropped from about `406.2` to about `29.2`
	- this is inference/export proof plus trainer-objective proof, not yet a retrained-model proof

## Apr 13, 2026 - The first full V7.4 best-triggered run finished close to target, with best val `0.0506` at epoch `51`

- completed `output/ml-training/v7_4_brush_channel_bestburst_20260413`
- exact summary from `training_log.json`:
	- `112` epochs completed before early stop
	- best epoch: `51`
	- best val loss: `0.05059416059936796`
	- final epoch: train `0.04502828886709463`, val `0.05751752530454428`
	- dataset shape: `6070` valid samples, curated train `3237`, val `613`, launched roots `26`
	- input channels: `13`
- practical read:
	- this is the closest full audited-corpus run so far and materially better than the previous audited `0.1256` result
	- late epochs still did not beat the epoch-51 checkpoint, so it remains the best legacy-semantics reference point
	- do not resume that checkpoint into the new WDL-trestle and reflect-padding variant; fresh improved runs should start clean under the new semantics unless a checkpoint was written by that same variant

## Apr 13, 2026 - Validation previews now include mixed held-out tiles and an explicit object-mask context sheet

- `train_v7.py` preview behavior changed again after the user asked for more confidence that object-masked minimap regions are being conveyed to the model
- active preview behavior now:
	- each epoch uses a mixed validation preview set by default: `2` fixed high-signal held-out tiles plus `2` random held-out tiles
	- each epoch writes `val_epoch_XXXX.json` listing the exact tiles used in that preview
	- each epoch also writes `val_epoch_XXXX_context.png` with columns:
		- minimap
		- object-mask overlay
		- masked-minimap diagnostic
		- object mask
		- liquid mask
		- brush mask
- bounded real-data proof:
	- verified on `output/tmp/v7_objectmask_preview_smoke_20260413/previews/val_epoch_0001_context.png`
	- verified sidecar metadata at `output/tmp/v7_objectmask_preview_smoke_20260413/previews/val_epoch_0001.json`
- important boundary:
	- this improves validation visibility only; the active trainer still uses raw minimap plus separate mask channels rather than hard-zeroing minimap pixels under object masks

## Apr 13, 2026 - Discriminator stabilization controls are now in the trainer and were finally proven on a real multi-step GAN smoke

- after the user called out earlier `Disc: 0` collapse/un-collapse behavior, `train_v7.py` gained conservative discriminator stabilizers:
	- real/fake target smoothing (`--disc-real-target`, `--disc-fake-target`)
	- small target jitter (`--disc-label-noise`)
	- discriminator input noise (`--disc-input-noise-std`)
	- discriminator gradient clipping (`--disc-grad-clip`)
	- discriminator real/fake mean logging in the epoch summary
- meaningful proof required a multi-step smoke because earlier tiny development smokes never hit a discriminator update when `disc_every=2`
- bounded real-data proof now exists from `output/tmp/v7_disc_real_smoke_20260413` using `LostIsles`, `--disc-every 1`, and `--gan-burst-after-best 1`:
	- epoch 2 GAN-on summary reported `Disc: 0.9667`
	- discriminator real/fake means: `0.4209 / 0.3878`
	- this proves the stabilized discriminator path is actually executing and no longer only syntactically present
- next action preserved:
	- relaunch the audited-corpus run from `v7_4_brush_channel_bestburst_20260413/best.pt` into a new folder with the discriminator stabilizers enabled

## Apr 13, 2026 - Negative validation loss in the brush-channel geometry-first run was a trainer numerics bug, not a real best checkpoint

- investigated the impossible `best val = -0.0060` reported by `output/ml-training/v7_4_brush_channel_geomfirst_20260413`
- root cause narrowed to the structural-loss path in `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py`:
	- validation and training loss were being computed under AMP autocast
	- SSIM had no numeric guardrails on variance/denominator terms
	- the trainer accepted any lower validation scalar, including impossible negative values, as a new `best.pt`
- active fix now in `train_v7.py`:
	- structural losses are computed in float32 even when model forward stays under AMP
	- SSIM now clamps variance terms non-negative, clamps the denominator, and clamps the SSIM map to `[-1, 1]`
	- validation loss must be finite and non-negative before it can drive the LR scheduler or overwrite `best.pt`
	- geometry-first epochs with GAN off no longer emit `numpy` empty-mean warnings for discriminator telemetry
- validation completed:
	- `C:\Users\akspa\anaconda3\python.exe -m py_compile i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` passed
	- tiny real-data smoke passed on `output/ml-corpus/400_12304/development` with `--epochs 1 --limit 4 --batch-size 1 --no-augment --train-workers 0 --val-workers 0`; result stayed sane (`Val Loss: 0.1466`) with no negative-loss or empty-discriminator warnings
- important boundary:
	- this fixes the trainer numerics going forward; it does not rehabilitate the previously written `output/ml-training/v7_4_brush_channel_geomfirst_20260413/best.pt`
	- treat that negative-loss checkpoint as invalid and resume only from a last sane checkpoint or restart the run under the fixed trainer

## Apr 13, 2026 - V7 trainer now supports periodic GAN detail bursts with cooldown after GAN-assisted best checkpoints

- landed new scheduling controls in `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py`:
	- `--gan-cycle-length`
	- `--gan-cycle-on-epochs`
	- `--gan-cooldown-after-best`
- active behavior:
	- GAN can stay off through the long geometry warmup as before
	- after `--start-gan-epoch`, GAN can run only for selected epochs within a repeating cadence instead of staying on continuously
	- if a new best checkpoint is achieved while GAN is on, the trainer can force GAN back off for a configurable cooldown stretch before allowing another detail pass
	- cooldown state is persisted in `checkpoint.pt` and `best.pt` so resume behavior matches the live run state
- bounded real-data proof:
	- tiny smoke on `output/ml-corpus/400_12304/development` with `--start-gan-epoch 1 --gan-cycle-length 3 --gan-cycle-on-epochs 1 --gan-cooldown-after-best 2` showed the intended pattern in live output:
		- epoch 1: GAN on
		- epoch 2: GAN off (`cooldown(2)`)
		- epoch 3: GAN off (`cooldown(1)`)
		- epoch 4: GAN on again
- important boundary:
	- this proves schedule control behavior and checkpoint continuity, not full-corpus convergence quality
	- cadence values still need real-run tuning against the audited trusted corpus

## Apr 13, 2026 - Best-triggered GAN refinement bursts replaced the arbitrary fixed warmup idea as the active schedule strategy

- user explicitly rejected the `100`-epoch GAN warmup rule as arbitrary and asked for GAN to auto-run at any and every new best model instead
- `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` now supports `--gan-burst-after-best <epochs>`
- active behavior when `--gan-burst-after-best > 0`:
	- GAN stays off while waiting for the next best checkpoint
	- any new best checkpoint arms GAN for the next configured number of epochs
	- if a GAN-assisted epoch also becomes the new best, the burst rearms again
	- this mode overrides the older epoch-calendar GAN cadence controls
- bounded real-data proof:
	- smoke on `output/ml-corpus/400_12304/development` with `--gan-burst-after-best 2` showed:
		- epoch 1: GAN off (`waiting-for-best`)
		- epoch 1 saved best and armed a 2-epoch burst
		- epoch 2: GAN on (`best-burst(2)`)
		- epoch 2 saved best and rearmed the burst
		- epoch 3: GAN on (`best-burst(2)`)
- important boundary:
	- this proves best-triggered arming works on real data; it does not yet prove the optimal burst length for the full audited corpus
	- the earlier fixed `start_gan_epoch=101` idea should now be treated as a fallback knob, not the preferred path

## Apr 13, 2026 - The current run evidence says `100` epochs is the practical ceiling, not `140+`

- the best-triggered audited-corpus run reached best val `0.0506` at epoch `51`
- later training continued until epoch `112`, then early-stopped after `12` non-improving patience steps with no better result than the epoch-51 checkpoint
- preserved user guidance:
	- epoch `30..40` already looked close
	- `140` is longer than needed for this regime
	- treat `100` epochs as the upper bound unless future evidence shows real late validation gains
- active training-policy change:
	- `train_v7.py` default epochs reduced to `100`
	- `train_v7.py` default `early_stop_start_epoch` reduced to `1` so validation can actually stop a best-triggered run instead of being artificially delayed past the new horizon

## Apr 13, 2026 - Trainer defaults now bias toward a long geometry-first phase before GAN pressure

- after the brush-conditioned full run finished at best val `0.1225` with continued late-epoch wobble, the training code in `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` was retuned so the default schedule no longer turns GAN on immediately
- new default training behavior:
	- `--adversarial-scale` default is now `0.20`
	- `--start-gan-epoch` default is now `101`
	- `--disc-every` default is now `2`
	- `--early-stop-start-epoch` default is now `101`
	- ReduceLROnPlateau patience is now configurable and defaults to `8` instead of the prior hardcoded `2`
- practical intent of this change:
	- let geometry training stabilize for a long warmup window before adversarial pressure begins
	- stop early-stop patience from killing the run during the geometry-only phase
	- avoid the previous pattern where GAN or scheduler pressure began too early and validation never settled into the low-loss regime we want
- important boundary:
	- this is a training-schedule correction only; it does not claim the brush channel or current corpus has solved the broader reconstruction problem
	- the next proof step is to run the new defaults on a real corpus and compare against the prior audited brush-conditioned run (`best 0.1225`) and the non-brush audited run (`best 0.1256`)

## Apr 13, 2026 - Brush-imprint harvest now runs corpus-wide and the terrain trainer can consume a first brush mask channel

- ran `ml-harvest-brushes` across the trusted `output/ml-corpus` set into `output/build-validation/brush-imprints/trusted/`
- corpus-wide brush harvest summary:
	- `27` manifests
	- `10,541` processed tiles
	- `259,216` grouped candidates
	- `51,741,807` patch cells written
	- one zero-group root so far: `400_11927_Uldum`
	- largest roots by grouped output currently include `400_11927_Kalimdor`, `301_8303_Kalimdor`, `335_12340_Kalimdor`, `400_11927_Azeroth`, and `335_12340_Azeroth`
- exporter/harvester behavior change:
	- `ml-harvest-brushes` now also writes per-tile brush masks under `brush_imprints/tile_masks/`
	- each tile summary in `brush_imprint_manifest.json` now carries `brush_mask_path`
- first trainer integration landed in `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py`
	- `MODEL_INPUT_CHANNELS` increased from `12` to `13`
	- `TileSample` now carries `brush_mask_path`
	- dataset loader resolves `brush_imprints/brush_imprint_manifest.json` and tile-level mask paths per dataset root
	- `__getitem__` now loads the tile brush mask as an additional binary conditioning channel appended after the current object mask
- validation captured:
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed after the harvester updates
	- `C:\Users\akspa\anaconda3\python.exe -m py_compile i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` passed
	- a dry trainer smoke with `--epochs 0 --batch-size 1 --limit 4 --no-augment --dataset-root output/ml-corpus/400_12304/development` loaded `3` valid samples successfully after the brush channel integration and reached CUDA startup without input-shape failure
- important boundary:
	- current trainer integration is intentionally minimal: a tile-level brush mask channel only
	- it does not yet use grouped candidate geometry directly, patch-group embeddings, or a separate brush model
	- this is the first safe path to let the terrain regressor see where harvested brush-like imprints cluster while the separate brush dataset/model path is still being built

## Apr 13, 2026 - First brush-imprint harvester landed for patch-scale WoWEdit archaeology

- landed a new wow-viewer command surface in `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs`:
	- `ml-harvest-brushes --dataset-root <path> [--output-dir <dir>] [--limit <count>] [--write-previews]`
- implementation now lives in `wow-viewer/tools/converter/WowViewer.Tool.Converter/MlBrushImprintHarvester.cs`
- current scope of the harvester:
	- reads ML dataset tile JSONs from `dataset/`
	- loads `heightmap_global` (resizing `512x512` exports down to the terrain-native `257x257` lattice when needed)
	- treats each tile as `16x16` chunks and each chunk as `16x16` candidate patch cells
	- scores patch cells from local relief / slope / diagonal change over the `257x257` terrain lattice
	- flood-groups adjacent high-score patch cells into candidate patch-group imprints
	- emits one JSON per grouped candidate plus a manifest and optional preview masks
- important boundary:
	- this is not the final brush-dedupe system and does not yet prove recovered WoWEdit brush identity
	- it is a first archaeology seam that isolates repeated patch or patch-group terrain imprints into a separate dataset for later analysis/modeling
	- texture-layer evidence is currently weak because active corpora mostly carry `texture_path` ordering without live `alpha_bits`; the first harvester therefore leans on terrain-shape imprints first and carries texture signatures only when available
- first real-data validation captured:
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-harvest-brushes --dataset-root i:/parp/parp-tools/output/ml-corpus/400_12304/development --output-dir i:/parp/parp-tools/output/build-validation/brush-imprints/development_40012304 --limit 6 --write-previews` passed
	- output summary from `output/build-validation/brush-imprints/development_40012304/brush_imprint_manifest.json`:
		- `6` tiles processed
		- `250` grouped brush candidates written
		- `17,699` patch cells written across those groups
	- example candidate file `output/build-validation/brush-imprints/development_40012304/groups/development_34_34_g0001.json` shows a grouped imprint spanning patch bounds `64..80 x 213..225`, `94` active patch cells, and an `18x14` normalized height micro-grid suitable for later clustering/model work
- preserved next requirement from user correction:
	- the real goal is identifying patch or patch-group imprints left by the original WoWEdit 3D brush workflow and sorting them into their own dataset
	- actual dedupe/retrieval/classification should happen later over this harvested imprint dataset, likely with a separate model family

## Apr 12, 2026 - Full trusted-corpus signal audit completed in wow-viewer and the next audited V7 run is live

- ran [`ml-audit-signals`](../../wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs) across the trusted corpus roots into `output/build-validation/ml-audit/trusted/`
- current bounded truth summary from the trusted audit set:
	- `27` audit reports
	- `10,541` audited tiles total
	- `1,180` tiles missing source minimaps
	- `1,018` tiles missing global heightmaps, all concentrated in `output/ml-corpus/301_8303/Kalimdor`
	- `0` audits currently report stitched alpha-mask coverage (`tiles_with_any_alpha_mask = 0` across the trusted set)
	- `16` tiles flagged as `below-terrain-likely` liquid
	- `158` tiles flagged as `uncertain` liquid
- current training gate decision:
	- geometry-only V7 training can still proceed on the audited trusted set because the active trainer in `train_v7.py` hard-requires minimap + normalmap + local/global heightmaps, not alpha atlases
	- `301_8303/Kalimdor` was excluded from the launched run because its audit shows `0/1018` global heightmaps
	- quarantined roots remain excluded even though the first broad audit command also proved they can be scanned
- launched new run from `C:\Users\akspa\anaconda3\python.exe` into `output/ml-training/v7_4_audited_all_trusted_20260412`
	- profile: manual audited roots only
	- launch settings: `--epochs 16 --learning-rate 8e-5 --disc-learning-rate 5e-5 --adversarial-scale 0.20 --start-gan-epoch 6 --disc-every 2 --patience 8 --amp-dtype auto --train-workers 4 --val-workers 2 --log-every 5`
	- current live terminal state shows preview-tile selection completed and training is active on CUDA (`NVIDIA GeForce RTX 4070 Ti SUPER`, AMP `bfloat16`, TF32 on/on)
- user correction to preserve for the next curation slice:
	- dedupe and future brush/prefab comparison should not stop at tile-level grouping
	- treat each tile as `16x16` chunks, and each chunk as `16x16` candidate patch cells for the next prefab/brush dedupe surface
	- this means the current audit is only the tile-scale gate; patch-scale prefab archaeology remains the next deeper ownership seam

## Apr 12, 2026 - wow-viewer now has a first `ml-audit-signals` corpus-truth audit command for V7.4 curation work

- landed the first bounded wow-viewer-owned ML audit seam in `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs`
- new command surface:
	- `wowviewer-converter ml-audit-signals --dataset-root <path> [--output <report.json>] [--limit <count>]`
- active audit behavior:
	- reads legacy ML dataset tile JSONs from `<dataset-root>/dataset`
	- emits a machine-readable audit report with:
		- dedupe groups
		- concept clusters
		- per-tile retention recommendation (`canonical` vs `review-duplicate`)
		- liquid semantic classification (`visible-surface`, `below-terrain-likely`, `uncertain`, `none`)
		- source/minimap/height/liquid/object/alpha presence counts
	- uses a bounded heuristic first pass rather than claiming final semantic truth:
		- dedupe groups are built from source/alpha/textures/object-count/liquid-class signatures
		- concept clusters are built from coarser perceptual/hash buckets plus content buckets
		- liquid sanity compares source minimap vs `no_liquid_minimap` under `liquid_mask` and marks low-delta cases as `below-terrain-likely`
- real validation captured for this slice:
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-audit-signals --dataset-root i:/parp/parp-tools/output/ml-corpus/301_8303/Northrend --output i:/parp/parp-tools/output/build-validation/ml-audit/northrend_signal_audit.json --limit 32` passed
	- first real audit output reported:
		- `32` tiles processed
		- `21` concept clusters
		- `24` dedupe groups
		- `26` `visible-surface` liquid tiles
		- `3` `below-terrain-likely` liquid tiles
		- `1` `uncertain` liquid tile
- important boundary:
	- this is a first curation/audit seam for V7.4 dataset truth ownership, not full canonical ML-contract cutover
	- concept clustering and liquid semantics are heuristic-first and intended to gate review/reruns, not to claim final authoring truth yet

## Apr 12, 2026 - Post-epoch-5 drift response: V7.3 fine-tune controls added and best-checkpoint continuation path defined

- after the epoch 6..10 continuation reported sustained val regression (`0.1433 -> 0.1758`) with best still pinned at epoch 5 (`0.0493`), the trainer was extended for controlled fine-tuning instead of continuing the same GAN pressure profile
- active behavior now in `train_v7.py`:
	- checkpoint resume can restore optimizer/discriminator/scheduler/scaler state (`--no-resume-optimizer` to disable)
	- adversarial influence is now tunable (`--adversarial-scale`)
	- GAN objective can be delayed to later epochs (`--start-gan-epoch`)
	- discriminator updates can be throttled (`--disc-every`)
	- discriminator LR is now configurable (`--disc-learning-rate`)
	- checkpoint payload now stores optimizer/discriminator/scheduler/scaler state and patience counter for true continuity
- updated the training guide with a dedicated fine-tune recipe that resumes from `best.pt` into a new output folder and uses reduced GAN pressure
- important boundary:
	- this slice lands control/sequencing changes and a concrete fine-tune recipe; quality outcome depends on the new continuation run results, not code changes alone

## Apr 12, 2026 - Geometry-first recovery run launched after GAN-tuned continuation still drifted at epochs 7-8

- observed in-flight fine-tune drift from user-reported/live metrics: epoch 7 val `0.1813`, epoch 8 val `0.1706`, best still `0.0493`
- stopped the active GAN-tuned continuation and pivoted to a conservative geometry recovery profile from `best.pt`
- launched new run in `output/ml-training/v7_3_all_trusted_maps_geom_recover_20260412` with:
	- `--learning-rate 1e-5`
	- `--disc-learning-rate 1e-5`
	- `--adversarial-scale 0.0`
	- `--start-gan-epoch 999`
	- `--disc-every 4`
	- `--no-augment`
	- `--no-resume-optimizer`
	- trust filter still enforced across all 31 non-quarantined roots
- current status:
	- run is active on CUDA with AMP bfloat16 and TF32 on/on, awaiting first post-pivot epoch summary

## Apr 12, 2026 - Fresh full-corpus restart launched from epoch 0 to avoid cross-run architecture/training-setting drift

- accepted the restart rationale after multiple resume/fine-tune experiments stayed above the old best (`0.0493`)
- stopped the geometry-recovery resume and started a true fresh run with no checkpoint resume in a new folder:
	- `output/ml-training/v7_3_all_trusted_maps_fresh_20260412`
	- trusted roots: 31 (quarantine filter still enforced)
	- `--epochs 16 --learning-rate 8e-5 --disc-learning-rate 5e-5`
	- `--adversarial-scale 0.20 --start-gan-epoch 6 --disc-every 2 --patience 8`
	- `--amp-dtype auto --train-workers 4 --val-workers 2 --log-every 5`
- current status:
	- run bootstrapped successfully on CUDA (RTX 4070 Ti SUPER), AMP bfloat16, TF32 on/on; training loop is active and awaiting first epoch summary

## Apr 12, 2026 - V7.3 now has live metric updates and a validated Tensor Core training profile (+8.8% on measured subset)

- followed the request to show live CLI values, verify real GPU usage, and make practical speed improvements before continuing training
- active behavior and proof this session:
	- `train_v7.py` now prints richer live tqdm telemetry (`g`, `d`, `lr`, `vram`) plus per-epoch throughput (`steps/s`, `samples/s`)
	- CUDA path now explicitly enables TF32 (`torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True` unless disabled) and exposes `--amp-dtype auto|bfloat16|float16`
	- AMP instability from FFT frequency loss was repaired by forcing frequency-loss FFT inputs to float32 under autocast
	- launched a real full trusted-corpus continuation run to epoch 10 with the tuned profile (`--amp-dtype auto --train-workers 4 --val-workers 2 --log-every 5`) from `v7_3_all_trusted_maps_20260411_235624/checkpoint.pt`; run is currently in progress
	- benchmark evidence on `NVIDIA GeForce RTX 4070 Ti SUPER` (`Northrend`, `limit=640`, batch 4, one epoch):
		- baseline (`--no-amp --no-tf32 --no-cudnn-benchmark`, workers 4/2): `1.47 steps/s`, `72.15s`
		- Tensor Core profile (`--amp-dtype auto`, TF32 on, workers 4/2): `1.60 steps/s`, `69.10s`
		- measured improvement: `+8.8%`
	- loader defaults were set to the measured faster profile (`train_workers=4`, `val_workers=2`) rather than the prior over-threaded dynamic default on this machine
- important boundary:
	- benchmark was a controlled one-epoch subset run, not full-corpus convergence proof
	- Tensor Core path improves throughput here but can still alter optimization dynamics; quality tracking remains val-loss-led

## Apr 12, 2026 - Full trusted-corpus V7.3 resume completed through epoch 5 with improved best validation loss

- followed the explicit continuation request to update documentation/memory and resume the all-trusted corpus run to epoch 5
- active behavior and proof this session:
	- resumed from `output/ml-training/v7_3_all_trusted_maps_20260411_235624/checkpoint.pt` at epoch 1 and completed epochs `2..5`
	- trust gating held for the resumed run: all 31 dataset roots were non-quarantined (`__UNTRUSTED_DO_NOT_USE` roots excluded)
	- corpus/sample shape remained stable during resume: `6070` valid samples (`2708` blank skipped), train/val `5451/619`, curated train `3233`
	- best validation loss improved from `0.0979` (epoch 1) to `0.0493` at epoch 5
	- per-epoch validation losses during resume: epoch 2 `0.0807`, epoch 3 `0.0529`, epoch 4 `0.0678`, epoch 5 `0.0493`
	- artifacts were updated in-place under `output/ml-training/v7_3_all_trusted_maps_20260411_235624` (`best.pt`, `checkpoint.pt`, `training_log.json`, previews)
	- `output/v73-model-architecture-guide.html` now reflects completed epoch-5 baseline status instead of "resume planned"
- important boundary:
	- this is broad real-data training-baseline proof on trusted roots, not final terrain-restoration quality signoff
	- no new automated model-quality benchmark suite was introduced in this slice; validation is run-log + artifact continuity

## Apr 11, 2026 - Object masking is not trustworthy in older ml-corpus roots; fresh mask-gated smoke exports passed and V7.3 training was restarted on that validated subset

- followed the direct request to stop trusting stale assumptions and prove object masking before resuming training
- active behavior and proof this session:
	- trusted legacy roots used in earlier manual training (`output/ml-corpus/...`) were sampled for WMO-bearing tiles and still showed `object_visibility_mask = null` / `no_object_minimap = null` on the checked tiles; this was captured in `output/build-validation/mask-audit/few_tile_mask_check.json`
	- fresh real-data exports were generated with the current exporter for `Northrend` (`3.0.1.8303`) and `LostIsles` (`4.0.0.11927`) using `--limit 12` each, under:
		- `output/build-validation/mask-audit/fresh-northrend-12`
		- `output/build-validation/mask-audit/fresh-lostisles-12`
	- mask audit on those fresh roots (`output/build-validation/mask-audit/fresh_mask_check.json`) showed 11 tiles with WMO objects, with 8 tiles producing non-empty object mask/no-object artifacts; the 3 misses were investigated and were out-of-footprint placements (no projectable WMO in tile space), not in-tile detection failures
	- projection-aware gating report (`output/build-validation/mask-audit/fresh_mask_check_projectable.json`) showed `8/8` pass on tiles where at least one WMO projected into tile footprint
	- V7.3 was restarted on only the fresh validated roots and completed a one-epoch smoke run:
		- `python .../train_v7.py --profile manual --dataset-root fresh-northrend-12 --dataset-root fresh-lostisles-12 --include-map Northrend --include-map LostIsles --epochs 1`
		- usable samples: `19` (`17/2` train/val), best validation loss: `0.1949`
- important boundary:
	- this is smoke-level mask gating and training proof only on two small fresh roots; it is not full-corpus signoff for all existing `output/ml-corpus/*` roots
	- old corpus roots should be treated as requiring regeneration/validation for object masking before being trusted in larger runs

## Apr 11, 2026 - The active LK exporter now materializes MH2O liquids again, and wow-viewer has a first shared root-ADT MH2O reader

- followed the direct correction after the V7 signal audit and the user pushback narrowed the real regression: WotLK or modern `ml-export` was still dropping `terrain_data.liquids` entirely even though the current codebase already had MH2O-aware consumer paths
- active behavior after this slice:
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtLiquidReader.cs` plus `wow-viewer/src/core/WowViewer.Core/Maps/AdtLiquidFile.cs` now provide the first shared root-ADT MH2O payload reader in `wow-viewer`, with synthetic coverage in `wow-viewer/tests/WowViewer.Core.Tests/AdtLiquidReaderTests.cs`
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` now captures root `MH2O`, parses it through the active `Formats/Liquids/Mh2oChunk.cs` seam, and emits non-null `terrain_data.liquids` for LK tiles instead of hardcoding `Liquids: null`
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDataModels.cs` now preserves MH2O rectangle metadata on each liquid layer (`x_offset`, `y_offset`, `width`, `height`, `exists_bitmap`) so the dataset contract can carry partial sub-rect liquid coverage instead of only whole-chunk approximations
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/TileStitchingService.cs` and `src/MdxViewer/Terrain/VlmProjectLoader.cs` now respect that metadata when building stitched liquid masks/heights and viewer-side `TileFlags`, rather than assuming any liquid means a full 8x8 chunk is wet
	- `src/WoWMapConverter/WoWMapConverter.Core.Tests/VLM/TileStitchingServiceLiquidTests.cs` now adds a focused first-party regression seam for partial MH2O mask placement in the active converter tree
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter AdtLiquidReaderTests` passed
	- `dotnet test i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/WoWMapConverter.Core.Tests.csproj -c Debug --filter TileStitchingServiceLiquidTests` passed
	- real-data smoke export succeeded with `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj --configuration Debug -- ml-export --client "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --out i:/parp/parp-tools/output/tmp/mh2o-smoke-335-azeroth --limit 1`; the live run logged `Parsed 256 MH2O liquid layers for Azeroth_35_20`, emitted `liquids/Azeroth_35_20_liq_mask.png` plus `liquids/Azeroth_35_20_liq_height.png`, and the tile JSON now has non-null `terrain_data.liquids`
- important boundary:
	- the real-data proof so far is a one-tile `3.3.5.12340` smoke on `Azeroth_35_20`, which appears to be an ocean-heavy full-coverage tile; this proves the dead-liquid-signal regression is repaired, not that partial-rect MH2O coverage has been revalidated across broader real corpora
	- stitched `liquid_min` or `liquid_max` on that smoke tile remained `0`, so treat this slice as export-signal recovery rather than final liquid-height semantic signoff

## Apr 10, 2026 - LK ML exports now emit normalmaps, and the first honest V7 smoke ran on a real 3.3.5 dataset root

- followed the reprioritized "basic thing first" direction after auditing the existing corpus and proving the current blocker was not theoretical: live `3.3.5.12340` exports were still writing `terrain_data.normalmap = null`, so `train_v7.py` could not consume the exported roots in strict mode
- active behavior after this slice:
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` now preserves LK/modern `MCNR` and `MCCV` payloads in `chunk_layers` instead of dropping them, and the LK export branch now calls the existing normal-map and MCCV-map generators before constructing `VlmTerrainData`
	- fresh real-data `ml-export` output for `Azeroth` on the fixed `3.3.5.12340` client now writes `images/<tile>_normal.png` and points `terrain_data.normalmap` at that file instead of leaving it null
	- `src/WoWMapConverter/scripts/train_v7.py` no longer depends on `scipy.ndimage.zoom` for heightmap resizing; it now uses `torch.nn.functional.interpolate`, which keeps the V7 sample loader aligned with the trainer's existing torch dependency instead of failing on a broken SciPy/NumPy ABI combo
- validation completed:
	- `dotnet test i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/WoWMapConverter.Core.Tests.csproj -c Debug` still passed after the LK exporter change
	- real-data export smoke succeeded with `ml-export --client "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --out i:/parp/parp-tools/output/tmp/v7-normal-smoke-335-azeroth --limit 1`, and the emitted tile JSON now references `images/Azeroth_35_20_normal.png`
	- a four-tile real dataset root under `i:/parp/parp-tools/output/tmp/v7-train-smoke-335-azeroth-4` loaded successfully in `train_v7.py` strict mode, and a one-epoch CUDA smoke completed with `Train/val samples: 3 / 1` and `Best validation loss: 0.3213`
- important boundary:
	- this is proof that the base minimap-to-terrain training path now runs end-to-end on a tiny real `3.3.5.12340` dataset root; it is not proof yet that the broader checked-in corpora under `output/ml-corpus/` have all been re-exported with the new normalmap behavior
	- the V7 smoke used the existing machine-wide `C:\Users\akspa\anaconda3\python.exe` environment because the workspace `.venv` only had `pip` and was not usable for torch-based training

## Apr 10, 2026 - The ML exporter now emits first-pass explained-vs-residual shadow-scar labels, but dataset/runtime proof is still missing

- followed the next requested step after writing down the `MCSH` object-recovery rationale: make the dataset contract explicit instead of leaving downstream code to recompute scar labels from raw `shadow_bits`
- active behavior after this slice:
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDataModels.cs` now extends `shadow_analysis` with chunk-level explained/residual shadow counts plus ratios and per-region `explained_by_current_objects`, `explained_overlap_ratio`, `nearest_candidate_distance_px`, `scar_candidate_score`, and `scar_type`
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmShadowAssociationService.cs` now rasterizes projected current-object footprint masks into chunk-shadow space and derives first-pass explained-vs-residual labels directly from the exported `MCSH` regions
	- `src/WoWMapConverter/WoWMapConverter.Core.Tests/` now exists as the first active first-party test project for this converter seam, with focused tests covering an explained region and an orphan scar region
- important boundary:
	- this is additive dataset-labeling work only; it does not prove that the exported ML corpus is correct end-to-end, that the selected real corpora are clean, or that `train_v7.py` has been trained successfully on them
	- the current scar labels are heuristic pseudo-labels based on projected current-object footprints, not retrieval-backed recovered placements

## Apr 10, 2026 - ML reconstruction scope now explicitly includes a third `shadow scar` model family for missing-object recovery

- followed the new direction that `MCSH` should not be treated as only generic shadow supervision; it can also act as object-history evidence when a shadow footprint survives but the matching placement no longer exists
- active workflow guidance after this note:
	- keep terrain height recovery, texture/alpha decomposition, and missing-object recovery as three separate model families over the same exported dataset root
	- frame the third model narrowly as `shadow scar` recovery: minimap + `MCSH` evidence + surviving placements -> unexplained shadow regions, missing-object candidate masks, and later restored placement hypotheses
	- `gillijimproject_refactor/docs/SHADOW_SCAR_OBJECT_RECOVERY.md` now records the fuller rationale: treat `MCSH` as historical object evidence, use current placements only as the explanation baseline, and expect retrieval from repeated object patterns elsewhere in copied/pasted world data to help recover orphan scars
	- use `terrain_data.shadow_maps`, raw `shadow_bits`, `shadow_analysis`, and `objects` as the base supervision surface; do not collapse this back into the main V7 terrain regressor
- important boundary:
	- this is a problem-definition and dataset-contract clarification only; no `shadow scar` extractor or training script exists yet in the active tree

## Apr 10, 2026 - wow-viewer ml-corpus now resolves the fixed-client config, scans archive-backed maps through WDT, and prefers split ADT companions

- followed the regression review on the initial `wow-viewer` ML command-surface port after real fixed-client dry-run validation proved the first version was filesystem-only and the legacy wrapper no longer matched the checked-in config field names
- active behavior after this slice:
	- `gillijimproject_refactor/scripts/export_ml_corpus.ps1` now tolerates the current config shape under `Set-StrictMode`, prefers `default_output_root`, resolves relative `client_path` values against `archive_root`, resolves optional `listfile_path` relative to the config, and defaults `harvest_after_export` to enabled when the field is absent
	- `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs` `ml-corpus` path now creates archive catalogs per client root, reads `World\Maps\<map>\<map>.wdt` to enumerate occupied tiles, and no longer requires extracted `Data/World/Maps/<map>` directories to exist on disk
	- the same `ml-corpus` path now prefers split `_tex0.adt` and `_obj0.adt` companions over root ADTs when building tile reports, while still falling back to root ADT summaries/placements when split companions are absent
	- `wow-viewer/src/core/WowViewer.Core/Maps/WdtTileCoordinate.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Maps/WdtTileIndexReader.cs` now provide a shared occupied-tile seam over WDT `MAIN` instead of leaving that parsing tool-local inside the converter
	- repeated tile-level texture/placement failures are now aggregated into one warning summary per map so real archive-backed runs stay readable instead of flooding thousands of lines of duplicate warnings
- validation completed:
	- `pwsh ./gillijimproject_refactor/scripts/export_ml_corpus.ps1 -DryRun` succeeded again and printed the expected fixed-client `ml-export` + `ml-harvest` command sequence with resolved archive-rooted client paths
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter WdtSummaryReaderTests` passed after the new shared WDT occupied-tile coverage landed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-corpus --config i:/parp/parp-tools/gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json --dry-run` now completes and reports `maps=16 tiles=7409`
- important boundary:
	- this is still report-oriented `ml-corpus` proof, not full wow-viewer parity with the old ML export pipeline
	- shared `AdtTextureReader` still overflows on parts of the real `3.0.1.8303` corpus, so the current report aggregates those as map-level warnings and falls back to empty texture-name lists for affected tiles instead of aborting the whole run

## Apr 10, 2026 - ML corpus export now has a fixed-client wrapper, V7 terrain training is geometry-only, and texture decomposition has its own first trainer seam

- followed the latest terrain-model correction that alpha-mask or tileset decomposition should not stay folded into the V7 terrain regressor, and the explicit request to operationalize the fixed local `3.0.1.8303`, `3.3.5.12340`, and `4.0.0.11927` clients as a reusable export corpus
- active behavior after this slice:
	- `gillijimproject_refactor/scripts/export_ml_corpus.ps1` now reads the checked-in `gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json` config and emits a reproducible per-client/per-map ML corpus under `output/ml-corpus/...`, then runs `ml-harvest` for each exported dataset root by default
	- the fixed config currently targets the machine-local `3.0.1.8303`, `3.3.5.12340`, and `4.0.0.11927` client roots with a deliberately narrow checked-in subset: `Northrend` plus `PVPZone01..04` from `3.0.1.8303`, `Azeroth` from `3.3.5.12340`, and `LostIsles` from `4.0.0.11927`
	- `src/WoWMapConverter/scripts/train_v7.py` and `infer_v7.py` now keep V7 focused on terrain geometry only: global/local height prediction plus bounds, with the old alpha auxiliary head removed from the active terrain model contract
	- `src/WoWMapConverter/scripts/train_texture_v1.py` now provides the first separate minimap-to-texture-layer seam, predicting three overlay alpha masks plus chunk-slot texture classes from minimap supervision using exported `chunk_layers`, `alpha_masks`, and palette data
	- `src/MdxViewer/ViewerApp_MlTraining.cs`, `docs/V7_HEIGHT_REGRESSOR.md`, `docs/VLM_Training_Guide.md`, `docs/VLM_DATASET_EXPORTER.md`, and `src/MdxViewer/USERGUIDE.md` now describe the split explicitly instead of continuing to imply that one V7 model owns both terrain and texture decomposition
- validation completed:
	- `get_errors` returned clean for `train_v7.py`, `infer_v7.py`, `train_texture_v1.py`, and `ViewerApp_MlTraining.cs`
	- `pwsh ./gillijimproject_refactor/scripts/export_ml_corpus.ps1 -DryRun` succeeded and printed the expected export/harvest command sequence for all configured clients and maps
	- `i:/parp/parp-tools/.venv/Scripts/python.exe -m py_compile ... train_v7.py infer_v7.py train_texture_v1.py` completed with no output, indicating syntax success
- important boundary:
	- the corpus wrapper has been dry-run validated only; no full multi-client export was executed in this chat
	- `train_texture_v1.py` is the first separate texture trainer seam, not proof yet that the full minimap-to-base-texture-plus-alpha decomposition problem is solved across all palettes or expansions

## Apr 10, 2026 - MdxViewer validation minimaps now render with a dedicated orthographic top-down projection

- followed the user report that live `MdxViewer` validation minimaps were still offset from the true tile borders even after the earlier settle or doodad or WL cleanup work
- active behavior after this slice:
	- `src/MdxViewer/ViewerApp.cs` now swaps the normal scene perspective projection for a dedicated orthographic top-down view and projection whenever an active capture request is part of an MdxViewer validation batch
	- `src/MdxViewer/ViewerApp_CaptureAutomation.cs` now provides the validation-only view/projection matrices, using a straight-down look with `Vector3.UnitX` as the up vector so the output keeps the existing minimap orientation while letting the requested ADT tile span fill the square capture exactly
	- the same validation batch path now forces a deterministic validation-only terrain light direction and restores the prior lighting override state afterward, so generated tiles no longer depend on whichever live world-light direction happened to be active when the batch started
	- `src/MdxViewer/ViewerApp.cs` now queues two validation output families from the ML finalize flow: the primary `viewer_validation_minimaps/` set keeps terrain liquids while still suppressing WL liquids, and a matching `viewer_validation_minimaps/noliquids/` sub-folder disables terrain liquids too so same-tile training inputs can be grouped by basename plus variant folder
	- `src/MdxViewer/ViewerApp_CaptureAutomation.cs` now stitches both validation output families after capture, writing full-map composites under `stitched/` inside the root validation folder and inside the `noliquids/` sub-folder
	- `src/MdxViewer/ViewerApp_CaptureAutomation.cs` still samples tile-center terrain height for the validation eye point, but the shot builder itself is back to tile-center positioning because the batch no longer relies on a tilted perspective workaround
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_CaptureAutomation.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` succeeded with existing workspace warnings only
- important boundary:
	- no new real-data capture rerun has been recorded yet in this chat, so the fix is build-validated only until the regenerated viewer output is compared against the source minimap tiles again

## Apr 09, 2026 - ML dataset finalize no longer generates baked 4k reference minimaps and MdxViewer validation captures now hide doodads

- followed the workflow correction that the ML dataset surface should stop generating baked `reference_minimaps` entirely and should rely only on live `MdxViewer` validation captures for rendered minimap output
- active behavior after this slice:
	- `src/MdxViewer/ViewerApp.cs` ML finalize UI no longer exposes baked-reference generation controls and now always runs the harvester in manifest-only mode while keeping optional `MdxViewer` validation capture output
	- the same finalize flow now explicitly tells users that baked 4k reference minimaps are disabled on the ML dataset surface and that only live viewer validation captures are produced
	- `src/MdxViewer/ViewerApp_CaptureAutomation.cs` validation capture batches now temporarily force `WorldScene.DoodadsVisible = false` during the batch and restore the previous doodad visibility afterward so the saved validation minimaps are terrain or world-shape captures without doodad clutter
	- `src/WoWMapConverter/WoWMapConverter.Cli/Program.cs` `ml-harvest` no longer advertises baked-reference output on the ML-facing help surface and ignores legacy baked-reference flags if they are still passed
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs`, `src/MdxViewer/ViewerApp_CaptureAutomation.cs`, and `src/WoWMapConverter/WoWMapConverter.Cli/Program.cs`
- important boundary:
	- no build or real-data validation has been captured yet for this slice in the current chat
	- the underlying harvester still carries legacy reference-minimap fields for compatibility, but the active ML-facing workflow no longer requests generation of those 4k outputs

## Apr 10, 2026 - Wrath Silverpine/Tirisfall lamp M2 texture collapse was fixed in the active adapter by honoring strict SKIN batch layout during fallback parsing

- followed the active M2 compatibility investigation on the real Wrath `World\Generic\Human\Passive Doodads\Lamps\TirisfallStreetLamp01.m2` repro after proving the adapted texture table already held the correct three lamp textures while material assignment still collapsed to texture `0`
- active behavior after this slice:
	- `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs` legacy `.skin` fallback parsing now treats strict `SKIN` headers as fixed-layout records instead of inferring texture-unit stride from the end of the file, and it now preserves `globalVertexOffset` plus uses the optional shadow-batch offset boundary when present
	- the same fallback path now correctly reads the Tirisfall lamp's three batch records as `textureComboIndex=0/1/2` instead of collapsing later entries, so the adapted materials/geosets resolve to the expected top-post, post-body, and glow textures
- validation completed:
	- isolated build validation passed with `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir="i:/parp/parp-tools/output/build-validation/mdxviewer-m2-texture-fix-6/"`
	- real-data probe validation passed with `ParpToolsWoWViewer.exe --probe-m2-adapter "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" "World/Generic/Human/Passive Doodads/Lamps/TirisfallStreetLamp01.m2" --build 3.3.5.12340 --listfile "i:/parp/parp-tools/gillijimproject_refactor/test_data/community-listfile-withcapitals.csv"`, which now reports three batches/materials and `[M2-DIAG-MAT]` mappings `geoset0->tex0`, `geoset1->tex1`, `geoset2->tex2`
- important boundary:
	- no automated tests were added or run
	- this is real-data adapter/probe proof for the Silverpine/Tirisfall lamp seam, not broad viewer runtime signoff for all later-era M2s or animations

## Apr 09, 2026 - Chunk tool can now invert selected chunk Z and export edited tile heightmaps into the project output folder without pretending terrain ADT save exists

- followed the request to let the chunk manipulator invert terrain vertically and then save the results somewhere useful without claiming a nonexistent general terrain persistence pipeline
- active behavior after this slice:
	- `src/MdxViewer/ViewerApp.cs` now lets the chunk tool invert the current chunk target or the active multi-chunk selection by negating chunk height samples, rebuilding normals, and tracking edited tiles inside the chunk-tool workflow
	- the same chunk-tool path now records dirty tile/chunk sets across chunk pastes and invert-Z edits, and can export the current edited tile state as reusable `257x257` L16 heightmaps plus per-tile JSON metadata and a manifest under the timestamped editor project output folder (`chunk-tool-heightmaps/...`)
	- `src/MdxViewer/ViewerApp_Sidebars.cs` now exposes `Invert Z Chunk` / `Invert Z Selection`, `Save Edited Heightmaps`, dirty-count status, and last-output-folder text inside the existing `Chunk Clipboard` window
	- `src/MdxViewer/ViewerApp_Workspaces.cs` now reports dirty chunk-tool heightmap-output availability in the workspace save summary instead of implying placement-save is the only staged output surface
- validation completed:
	- `get_errors` returned clean for the touched viewer files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir="i:/parp/parp-tools/output/build-validation/mdxviewer-chunktool/"` passed on Apr 09, 2026 with existing workspace warnings only
- important boundary:
	- the default `MdxViewer.sln` build still fails when a live `ParpToolsWoWViewer` process holds the normal `bin/Debug` outputs open, so the isolated `OutDir` build is the honest proof captured for this slice
	- this slice exports edited heightmap outputs only; it does not add a general terrain ADT save/write pipeline, and source terrain files are still left untouched

## Apr 09, 2026 - ML dataset exports now emit packed alpha atlases, and the viewer has a terrain analysis window for local-vs-global heightmap inspection

- followed the terrain tooling request to stop leaving alpha masks only as separate per-layer tile outputs when the viewer already had a single-atlas export pattern, and to add in-viewer terrain heightmap inspection aimed at finding squished or hidden geometry
- active behavior after this slice:
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/TileStitchingService.cs` now emits `*_alpha_atlas.png` alongside the stitched per-layer alpha outputs, packing alpha layers 1-3 into RGB with no stitched `MCSH` shadow data mixed into the atlas, and `VlmDatasetExporter.cs` now also stitches a map-wide `*_full_alpha_atlas.png` beside the existing full-map alpha-layer outputs
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDataModels.cs`, `VlmDatasetExporter.cs`, and `MkDatasetHarvester.cs` now carry that atlas on the dataset surface as `terrain_data.alpha_atlas`, keep stitched `shadow_maps` separate, and expose atlas or shadow presence plus compact image signatures (`sha256` + 64-bit average hash) in the manifest so later dedupe or coverage selection can happen without a separate post-process pass
	- `src/MdxViewer/ViewerApp_TerrainAnalysis.cs`, `ViewerApp.cs`, and `ViewerApp_Sidebars.cs` now expose a floating `Terrain Analysis` window that shows the current tile heightmap in per-tile normalization, the same tile remapped against loaded-tile or whole-map bounds, and the packed alpha/shadow atlas
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Apr 09, 2026 with existing workspace warnings only after the atlas and analysis window landed
	- `get_errors` returned clean for the touched viewer and converter files before the full build
- important boundary:
	- this slice is still build validated only; no real-data viewer runtime retest has been captured yet for the new `Terrain Analysis` window, and no real exported dataset root was re-harvested to inspect the updated alpha-only `alpha_atlas` payload on actual tiles
	- the exporter still keeps the per-layer stitched alpha masks because downstream ML or bake paths may still depend on them; the atlas is additive, not a schema-breaking replacement

## Apr 09, 2026 - Viewer dataset workflow is now one `Build ML Dataset` flow with inline manifest and validation work

- followed the latest naming and workflow complaint by removing the separate harvest viewer modal and folding manifest or baked-reference or MdxViewer-validation work into the existing dataset build dialog
- active viewer behavior after this slice:
	- `src/MdxViewer/ViewerApp.cs` now exposes `Build ML Dataset...` in the tools menu and renames the main export dialog to `Build ML Dataset`
	- the same dialog now includes an inline `ML Dataset Manifest + Validation` section with dataset-root, manifest-path, reference-output, viewer-validation-output, and resolution controls instead of bouncing into a second harvest dialog
	- post-build manifest plus validation can now auto-start after export in the same dialog flow, while `src/MdxViewer/ViewerApp_CaptureAutomation.cs` still owns the deterministic one-PNG-per-tile viewer-validation capture queue, settle waits, and temporary capture-state override/restore behavior
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Apr 09, 2026 with existing workspace warnings only after the dialog merge
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs`
- important boundary:
	- this slice is still build validated only; no real-data viewer batch was run yet against `test_data/development/World/Maps/development`, so there is still no proof yet for framing accuracy, settle timing, or output quality on actual captured tiles
	- the active viewer and converter CLI surfaces now say `ML`, and the default manifest filename is now `ml_dataset_manifest.json`; internal implementation names still remain under `WoWMapConverter.Core.VLM` or `Mk*` type names for continuity
	- alpha-mask completeness and shared `MdxViewer` or `wow-viewer` read-path ownership are still open and should not be treated as solved by this UI/workflow slice

## Apr 09, 2026 - ML Dataset surface and first harvest manifest command are now in place

- followed the request to stop treating the terrain supervision pipeline as a generic `VLM Dataset` surface and move the user-facing workflow onto `ML Dataset` naming before deeper reconstruction work
- active converter and viewer behavior after this slice:
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/MkDatasetHarvester.cs` now scans an exported dataset root, audits per-tile source minimap or local/global heightmap or alpha-mask or object or chunk-layer coverage, and writes the default manifest file `ml_dataset_manifest.json`
	- the same harvester can optionally generate 4096x4096 baked reference minimaps into `reference_minimaps/` using the existing `MinimapBakeService` instead of inventing a second bake path
	- `src/WoWMapConverter/WoWMapConverter.Cli/Program.cs` now exposes `ml-export`, `ml-decode`, `ml-bake`, `ml-bake-heightmap`, `ml-synth`, `ml-batch`, and `ml-harvest` as the primary command surface while keeping `mk-*` and `vlm-*` names as compatibility aliases
	- `src/MdxViewer/ViewerApp.cs`, `ViewerApp_MinimapAndStatus.cs`, `Terrain/VlmProjectLoader.cs`, and the active user docs now use `ML Dataset` wording for the visible menu, loader, status, and guide surface instead of continuing to foreground `VLM Project` in the UI
	- `src/MdxViewer/ViewerApp.cs` now exposes one `Build ML Dataset` dialog with inline manifest and validation controls instead of a separate harvest modal
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` passed on Apr 09, 2026 with existing workspace warnings only
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` passed on Apr 09, 2026 with existing workspace warnings only
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- ml-harvest` reached the new usage surface and printed the expected help text
- important boundary:
	- no real exported dataset root under `dataset/*.json` was available in the workspace during this slice, so `ml-harvest` is command-surface and build validated only, not yet real-data verified against a checked-in export
	- the new viewer harvest dialog is also build validated only; no focused interactive UI retest was captured in this slice
	- internal implementation namespaces and types still remain under `WoWMapConverter.Core.VLM` and `Mk*` types for continuity; this slice only moves the public workflow surface to `ML Dataset`

## Apr 09, 2026 - Viewer now exposes exact hovered object paths plus WMO doodad asset inspection for examination work

- followed the request to turn the earlier one-off asset-path idea into a reusable examination surface for WMOs and their MDX/M2 doodads instead of leaving exact paths trapped in transient hover overlays or raw selection text
- active `src/MdxViewer` behavior after this slice:
	- `ViewerApp_Investigation.cs` now surfaces the existing hovered world-object path as an interactive `Hovered Asset` panel inside the investigation toolbox, with copy-path, load-asset, and inspect-in-scene actions for hovered WMO and MDX/M2 placements
	- `ViewerApp_Sidebars.cs` now exposes explicit selected-world-object asset actions and adds a `WMO Doodad Inspector` for both selected world WMOs and standalone loaded WMOs, including exact doodad model paths, visibility/load state, def index, and local position; selecting a doodad row now frames that doodad in-scene and the detail pane exposes an explicit `Frame Doodad` action
	- `Rendering/WmoRenderer.cs` now exposes a narrow public doodad metadata seam (`WmoDoodadInfo` plus indexed lookup) plus doodad bounds lookup instead of forcing future examination work to read private renderer state
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug --no-restore` passed on Apr 09, 2026 with existing workspace warnings only
	- `get_errors` returned clean for `Rendering/WmoRenderer.cs`, `ViewerApp.cs`, `ViewerApp_Sidebars.cs`, and `ViewerApp_Investigation.cs`
- important boundary:
	- no automated tests were added or run
	- no live viewer runtime retest was captured yet for the new hover panel or WMO doodad inspector, so treat this as build-validated UI/tooling work rather than runtime signoff

## Apr 09, 2026 - Raw classic character probe diagnostics now expose replaceable candidate paths, and the tested 0.5.3 Human/Tauren variation overrides are still primarily geoset swaps rather than proven hair-texture swaps

- followed the next likely raw-character seam after the variation-id override landed: determine whether the remaining non-default cases were failing on hair or facial replaceable texture resolution or whether the tested assets simply did not expose those texture paths
- active viewer/probe behavior after this slice:
	- `gillijimproject_refactor/src/MdxViewer/Rendering/ReplaceableTextureResolver.cs` now exposes ordered replaceable-resolution candidates for raw-character debugging and broadens the character-directory fallback for replaceable ids `6`, `7`, and `10` from two fixed names to an explicit same-directory scan with candidate scoring
	- the same resolver now reports diagnostic misses when there is no matching `CharSections` entry and when there are no matching same-directory character textures, so future raw-character probes do not silently collapse to `Decode: not found` without context
	- `gillijimproject_refactor/src/MdxViewer/AssetProbe.cs` now prints the attempted replaceable candidates and their existence state before decode, so raw-character replaceable failures can be tied to actual file availability instead of only inferred from missing output textures
- validation completed:
	- isolated build validation passed with `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir="i:/parp/parp-tools/output/build-validation/mdxviewer-charprobe/"`; the default debug output remained file-locked by a live PowerShell process, so the isolated output was used for real-data probing instead of claiming the normal bin path was rebuilt
	- `ParpToolsWoWViewer.exe --probe-mdx "H:\053-client" "Character/Human/Male/HumanMale.mdx" --build 0.5.3.3368 --character-hair-variation 1` still reports the expected geoset swap `SelectionGroup=1 -> 2`, but replaceable id `6` now explicitly reports `char-section-hair[var=1]/missing-section` plus two missing same-directory hair-name candidates and still ends in `Decode: not found`
	- `ParpToolsWoWViewer.exe --probe-mdx "H:\053-client" "Character/Tauren/Male/TaurenMale.mdx" --build 0.5.3.3368 --character-hair-variation 1` still reports the expected geoset swap `SelectionGroup=2 -> 3`, but the raw model only surfaced replaceable ids `1` and `8` in this probe, which means the tested Tauren male hair-variation case is currently geoset-only in practice rather than a demonstrated replaceable-hair-texture swap
	- `ParpToolsWoWViewer.exe --probe-mdx "H:\053-client" "Character/Human/Male/HumanMale.mdx" --build 0.5.3.3368 --character-facial-variation 1` also did not produce a new facial-hair replaceable slot in the tested raw model output, reinforcing that the current proof is still about selection-group switching more than confirmed texture-family swapping for these specific 0.5.3 assets
- important boundary:
	- no automated tests were added or run
	- this slice materially improves real-data diagnostics and broadens fallback attempts, but it does not prove that every raw classic character variation should have a separate hair or facial replaceable texture on disk
	- for the tested `HumanMale` and `TaurenMale` 0.5.3 raw-character cases, the strongest current evidence is still geoset-selection correctness, not broad closure on variation-specific hair/facial texture resolution

## Apr 09, 2026 - Standalone raw classic character MDX inspection now exposes narrow hair and facial variation overrides without claiming a full paperdoll system

- followed the next approved character-model slice after the default geoset fix: add a narrow override surface so validation can move past only variation `0`
- active viewer behavior after this slice:
	- `gillijimproject_refactor/src/MdxViewer/Rendering/ReplaceableTextureResolver.cs` now exposes available classic character hair and facial-hair variation ids per raw `Character/...` model and can build a selection-group set for explicit variation ids instead of only the default `0` case
	- `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs` now exposes a narrow character-selection-group reapply path so the standalone viewer can switch raw classic character variation sets on the live renderer without rebuilding a second render pipeline
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` and `ViewerApp_Sidebars.cs` now add a `Character Variants` section to standalone classic character MDX inspection, with raw DBC `VariationId` combos for hair and facial-hair plus reset-to-default
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp_StartupAutomation.cs` and `AssetProbe.cs` now also accept `--character-hair-variation <id>` and `--character-facial-variation <id>`, so non-default raw-character cases stay scriptable for probe and capture validation instead of being UI-only
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` passed on Apr 09, 2026 with existing workspace warnings only
	- `dotnet run --project "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj" -c Debug -- --probe-mdx "H:\053-client" "Character/Human/Male/HumanMale.mdx" --build 0.5.3.3368 --character-hair-variation 1` now reports a real selected-group change from the default set, including `SelectionGroup=1 DefaultVisible=True SelectedVisible=False` and `SelectionGroup=2 DefaultVisible=False SelectedVisible=True`
	- `dotnet run --project "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj" -c Debug -- --probe-mdx "H:\053-client" "Character/Tauren/Male/TaurenMale.mdx" --build 0.5.3.3368 --character-hair-variation 1` also reports a real selected-group change, including `SelectionGroup=2 DefaultVisible=True SelectedVisible=False` and `SelectionGroup=3 DefaultVisible=False SelectedVisible=True`
	- standalone runtime capture using the new startup option completed for `Character/Human/Male/HumanMale.mdx` at `i:/parp/parp-tools/output/character_variation_validation/human_male_hair1/standalone/0.5.3.3368/20260409_003101393_current_20260409_003101_no_ui.png` and for `Character/Tauren/Male/TaurenMale.mdx` at `i:/parp/parp-tools/output/character_variation_validation/tauren_male_hair1/standalone/0.5.3.3368/20260409_004126420_current_20260409_004126_no_ui.png`, proving the non-default override path survives actual viewer startup and capture without regressing render correctness
- important boundary:
	- no automated tests were added or run
	- this is a narrow raw-character variation-id surface for standalone MDX inspection only; it is not a full character customization, gear, texture-composition, or saved paperdoll system

## Apr 08, 2026 - Raw Alpha character MDX viewing now applies classic default geoset selection instead of rendering every character variant at once

- followed the remaining `Character/Tauren/Female/TaurenFemale.mdx` complaint after the texture fix and confirmed the next seam was classic character geoset selection, not another replaceable-texture miss
- active viewer behavior after this slice:
	- `gillijimproject_refactor/src/MdxViewer/Rendering/ReplaceableTextureResolver.cs` now also loads `CharHairGeosets` and `CharacterFacialHairStyles` and can produce a default classic character geoset selection set for raw `Character\<race>\<sex>\*.mdx` models using the existing DBC context
	- `gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs` now applies that default character-selection set during standalone raw-character MDX initialization, so selection-group variants that should be mutually exclusive are no longer all rendered together
	- `gillijimproject_refactor/src/MdxViewer/AssetProbe.cs` now reports `SelectionGroup` plus `DefaultVisible` on each geoset so live client probes can show the exact classic character visibility policy instead of only texture/material state
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` passed on Apr 08, 2026 with existing workspace warnings only
	- `dotnet run --project .\MdxViewer.csproj -c Debug -- --probe-mdx "H:\053-client" "Character/Tauren/Female/TaurenFemale.mdx" --build 0.5.3.3368` still resolved the repaired body textures and now reported filtered classic selection groups, including `SelectionGroup=0,2,702,401,1301` as visible while mutually exclusive alternates such as `202/204/205/402/403/404/802/803/1302` were suppressed
	- standalone viewer runtime capture at `i:/parp/parp-tools/output/tauren_capture_geoset/standalone/0.5.3.3368/standalone/0.5.3.3368/20260408_235429888_current_20260408_235429_no_ui.png` now shows a coherent textured Tauren female body instead of the earlier broken all-geosets-visible presentation
	- follow-up real-data probes on `Character/Human/Male/HumanMale.mdx`, `Character/SCOURGE/Female/ScourgeFemale.mdx`, `Character/Tauren/Male/TaurenMale.mdx`, and `Character/Troll/Female/TrollFemale.mdx` also resolved their default body textures and reported the same default-selection-group suppression pattern instead of reverting to all-geosets-visible behavior
	- follow-up standalone runtime captures at `i:/parp/parp-tools/output/character_validation/human_male/standalone/0.5.3.3368/20260409_001239225_current_20260409_001239_no_ui.png`, `i:/parp/parp-tools/output/character_validation/scourge_female/standalone/0.5.3.3368/20260409_001319208_current_20260409_001319_no_ui.png`, `i:/parp/parp-tools/output/character_validation/tauren_male/standalone/0.5.3.3368/20260409_001510190_current_20260409_001510_no_ui.png`, and `i:/parp/parp-tools/output/character_validation/troll_female/standalone/0.5.3.3368/20260409_001548840_current_20260409_001548_no_ui.png` show coherent default raw-character renders for those additional race or sex cases
- important boundary:
	- no automated tests were added or run
	- this is now real-data standalone-character proof for several default raw 0.5.3 race or sex cases, not broad signoff for every character customization combination or later-era character pipelines

## Apr 08, 2026 - Alpha raw creature MDX replaceables now have an exact-model-path fallback for 0.5.3 when DBCD mapping is wrong

- followed the active `MdxViewer` render-debugging seam for broken standalone Alpha MDX texturing, using `Creature/Dragon/Dragon.mdx` on the real `H:\053-client` 0.5.3.3368 client as the proof target instead of continuing speculative reflective or specular material guesses
- active viewer behavior after this slice:
	- `gillijimproject_refactor/src/MdxViewer/Rendering/ReplaceableTextureResolver.cs` now broadens `CreatureModelData` lookup aliases for bare creature model names and, for `0.5.3` only, loads an exact-model-path fallback map of creature display texture variations from the checked-in alpha-core SQL data
	- the same resolver now prefers those exact-model-path fallback variants before the DBCD `CreatureDisplayInfo` path when available, and uses the same variant source for `Resolve(...)`, `SelectBestDisplayIndex(...)`, `GetVariantCount(...)`, and `GetVariantDescription(...)`
	- this specifically fixes the raw standalone Alpha dragon case where live DBCD resolution was still choosing a bogus `Helm, Rider` display with no decodable replaceable textures even after the earlier display-coherence fix
	- `gillijimproject_refactor/src/MdxViewer/AssetProbe.cs` now accepts `--build`, instantiates the same replaceable-texture resolver path used by the viewer, and prints selected replaceable display-variant details so standalone MDX texture failures can be validated against real client data without relying only on the interactive UI
- important boundary:
	- validation now includes a real no-UI viewer capture of the repaired dragon render through startup automation, using `--capture-shot current` against the live `H:\053-client` model load path; this is still a narrow standalone-model proof, not broad signoff for all MDX assets
	- the separate foliage or tree backface-looking issue remains open and should not be treated as solved by this dragon-specific replaceable-texture fix

## Apr 08, 2026 - Taxi and POI overlays now win clicks ahead of nearby world-object bounds, and taxi speed is normalized around `0.10 = 100%`

- followed the request to make the active viewer taxi workflow usable again in dense scenes instead of forcing users to fight WMO or MDX bounding boxes and runaway taxi speed values
- active viewer behavior after this slice:
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now checks taxi nodes, taxi routes, and area POIs before hovered scene-object or PM4 hit handling, so overlay picks win when the cursor is near both a world marker and a nearby world-object box
	- the same viewer host now tracks a selected area-POI state, supports viewport area-POI picking, and shows those POIs as real selectable entries in the existing area-POI list instead of leaving them list-only plus double-click camera focus
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now resolves taxi actors through explicit override -> mount-model -> default Gryphon or FelBat fallback, and the taxi speed UI/runtimes now share `0.01..0.50` semantics with `0.10` as normal speed
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp_Sidebars.cs` now keeps taxi controls on the right inspector only, clamps the speed slider to the shared taxi-speed range, and adds one-click Gryphon or FelBat override presets for the selected route
- important boundary:
	- this is build validation only; no focused live retest has been captured yet for taxi pick feel, POI pick feel, or ride pacing on a real route

## Apr 08, 2026 - `v0.4.7.1` release prep now packages the current viewer fixes while keeping the next runtime extraction anchored on `WorldScene` to `wow-viewer`

- followed the request to merge and publish the current viewer state as `v0.4.7.1` instead of leaving the shipped docs and workflow metadata on the earlier `v0.4.7` snapshot
- active release snapshot after this continuity update:
	- `gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` and `MdxViewer.CrossPlatform.csproj` now report `0.4.7.1`
	- the release workflow now packages `src/MdxViewer/docs/releases/v0.4.7.1.md` and ships `CHANGES-v0.4.7.1.md` in the archive
	- repo/viewer docs now foreground the repaired taxi workflow, direct route capture hardening, sticky world-object selection, standalone WMO group inspection, and the larger-range terrain/world-object follow-ups already landed in the active viewer host
	- the next engineering slice is still the staged world-runtime split from `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` into `wow-viewer`, using `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md` as the active plan surface
- important boundary:
	- this is continuity and release-alignment work only
	- it does not claim the `WorldScene` extraction is complete or that the recent viewer fixes are broadly runtime-signed-off

## Apr 08, 2026 - Minimap generation now has a dedicated staged wow-viewer plan surface instead of living only as ad hoc viewer follow-up work

- recorded the integrated continuity and execution surface in `gillijimproject_refactor/plans/wow_viewer_minimap_generation_plan_2026-04-08.md`
- added `.github/prompts/wow-viewer-minimap-generation-plan-set.prompt.md` plus ordered prompts for:
	- deterministic one-PNG-per-ADT capture queue in the active viewer host
	- wow-viewer CLI minimap command design and implementation
	- runtime-owned minimap-generation extraction out of `ViewerApp` and `WorldScene`
- updated the existing tool-suite and world-runtime prompt routers so future chats can route minimap work without overloading the generic `WorldScene` split prompt set
- important boundary:
	- this is planning and workflow continuity only
	- no new minimap-generation implementation landed in this slice beyond the already-recorded filter and large-range viewer groundwork

# Active Context

## Apr 07, 2026 - Standalone WMO viewing now keeps groups loaded during camera movement and uses explicit highlighted labels instead of text soup

- followed the request to stop relying on the giant standalone-WMO visibility checkbox list and add scene-native group inspection instead
- active viewer behavior after this slice:
	- `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs` now exposes standalone WMO render-group metadata needed by the viewer host and disables camera-driven runtime group culling for standalone inspection WMOs, so moving the camera no longer unloads groups out from under the user
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now draws a standalone WMO group overlay immediately after the WMO render pass, using the existing `BoundingBoxRenderer` path for color-coded group bounds
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp_WmoGroups.cs` now keeps boxes visible for all groups but only renders large in-scene labels for groups the user explicitly highlights; left-click selects, shift-click toggles label highlighting, ctrl-click toggles visibility, and right-click isolates
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp_Sidebars.cs` now exposes compact standalone WMO group controls with `Hide/Show`, `Highlight Label` or `Remove Label`, `Isolate`, `Show All`, `Clear Labels`, `Clear Selection`, and `Frame` actions so users do not have to work from the full generic visibility list alone
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` WMO conversion dialog now uses an explicit output-folder field with folder browsing and the maintained converter path only, instead of the old deep default export folder plus dead `Extended` mode
- important boundary:
	- this is build validation only; no live viewer retest has been captured yet for dense multi-group WMOs, highlighted-label readability, or the click/hover feel in a real session

## Apr 07, 2026 - `MdxViewer` input routing now defers scene wheel handling until after ImGui update and blocks scene keyboard controls when UI owns focus

- followed the user-selected next slice after the theme scaffold by fixing the active viewer's UI-to-scene input leakage instead of adding more UI surface first
- active viewer behavior after this slice:
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` no longer moves the camera directly from the raw mouse-wheel callback; wheel deltas are queued and applied in `OnUpdate()` after ImGui refreshes capture state
	- scene keyboard controls now use one consistent gate (`WantCaptureKeyboard || WantTextInput`) for chunk clipboard shortcuts, shell hotkeys, minimap toggle, animation stepping, and free-fly movement
	- scene mouse blocking now respects any real ImGui mouse capture by default, with dockspace bypass still limited to the existing dockspace-central-node case
- important boundary:
	- this is compile-validated only; no live viewer interaction retest has been captured yet for scroll-over-panel, typing-in-input, or overlapping floating-window cases

## Apr 07, 2026 - `MdxViewer` now has a persisted UI theme scaffold, including a pre-alpha-inspired chrome option

- followed the user-selected UI order `3,2,1` by landing theme infrastructure first instead of jumping straight into a full shell rewrite
- active viewer behavior after this slice:
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp_Themes.cs` now owns centralized ImGui theme application instead of keeping style colors hardcoded in startup
	- the active viewer can now switch between the existing `Modern Slate` chrome and a new `Pre-Alpha Brass` chrome that uses square borders and brass/navy styling as the first pre-alpha-inspired pass
	- the selected theme is saved in `viewer_settings.json` and applied at startup through `LoadViewerSettings()` + `ApplyActiveUiTheme()`
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp_Sidebars.cs` now exposes theme selection in the unified viewer settings section
- important boundary:
	- this is theme and chrome infrastructure only; no paperdoll panel, no historical shell layout rewrite, and no shared spell/character services landed in this slice

## Apr 07, 2026 - Planned next wow-viewer expansion is shared DBC-first: spell browsing, paperdoll composition, `WorldSafeLocs` POIs, and converter unification

- captured the new roadmap in `gillijimproject_refactor/plans/wow_viewer_spell_paperdoll_poi_and_converter_plan_2026-04-07.md`
- active planning direction after this note:
	- new shared DBC/DB2 readers and resolvers for `Spell`, `WorldSafeLocs`, `CreatureDisplayInfo`, `CreatureDisplayInfoExtra`, and `ItemDisplayInfo` belong in `wow-viewer/src/core/WowViewer.Core.IO/Dbc`
	- higher-level spell-asset bundles, character/paperdoll composition, and POI caching belong in `wow-viewer` runtime/services, not in viewer panel code
	- `gillijimproject_refactor/src/MdxViewer/Rendering/ReplaceableTextureResolver.cs` is the best current extraction seed for character-display and gear-texture composition; do not treat it as the final architecture
	- `wow-viewer/tools/converter/WowViewer.Tool.Converter` remains the canonical front door for version-aware conversion; the open WMO/model/terrain converter mess should converge into one detect -> plan -> convert surface instead of preserving old executable sprawl
	- `WorldSafeLocs` shared ownership can start now, but active `MdxViewer` graveyard overlay wiring should still be treated as a later consumer slice once current viewer render/input cleanup stops thrashing
- important boundary:
	- this is planning and routing only; no shared spell reader, paperdoll runtime, `WorldSafeLocs` reader, or unified converter plan surface has landed yet

## Apr 07, 2026 - Shared WMO liquid family resolution replaced the old build-only `MLIQ` baseline path

- followed the request to differentiate WMO liquid handling by actual asset version instead of keeping the active viewer on a hardcoded `3.3.5.12340 => 270°` baseline
- active shared/runtime behavior after this slice:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoLiquidLayoutResolver.cs` now owns WMO liquid coordinate-family resolution with asset version first and build string only as a fallback hint
	- the shared resolver currently returns a neutral baseline rotation, so `MdxViewer` no longer adds an automatic `+270°` quarter-turn for 3.3.5 assets
	- `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs` now consumes that shared resolver for baseline rotation instead of its old local build-string switch
	- `wow-viewer/tools/converter/WowViewer.Tool.Converter` `detect` now exposes the same WMO liquid family and baseline rotation for `Wmo` and `WmoGroup` inputs
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --no-restore` passed with 270 tests succeeded and no failures
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug --no-restore` passed with existing warnings only
- important boundary:
	- this is not runtime signoff yet; no real-data viewer retest has been captured for representative 0.5.3 and 3.3.5 WMO liquid scenes
	- the larger converter-modernization request is still open; only the shared WMO liquid policy and the modern `detect` surface were updated in this slice

## Apr 06, 2026 - Taxi routes can now drive a ride camera, and the viewer can stream direct mp4/mov capture through ffmpeg

- followed the request to turn the animated taxi actor into a teaser-making workflow instead of just a debugging overlay
- active `src/MdxViewer` behavior after this slice:
	- `Terrain/WorldScene.cs` now exposes live taxi actor pose data for the currently animated route actor, using the same per-frame sampled path position and forward vector already used for actor rendering
	- `ViewerApp_CaptureAutomation.cs` now adds a taxi ride camera with two modes:
		- `Cockpit`, which places the camera on the animated actor
		- `Chase`, which follows behind the actor with configurable distance, height, and look-ahead
	- the same capture partial now supports direct video capture to `.mp4` or `.mov` by streaming raw framebuffer frames into `ffmpeg` instead of only saving PNG stills
	- `ViewerApp_Sidebars.cs` now adds taxi-sidebar controls for ride-camera attach or detach, ride-camera mode and offsets, and one-click route video capture from the selected taxi route
	- `ViewerApp.cs` now updates the ride camera in the normal update loop, disables free-fly movement while attached, records video frames from the same no-UI or with-UI capture seam as still screenshots, and persists video-capture settings in viewer settings
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 06, 2026 with existing warnings only
- important boundary:
	- this is compile validation only
	- no automated tests were added or run
	- no live taxi-route runtime retest has been captured yet for ride-camera feel, ffmpeg encode success inside a real viewer session, or output quality on an actual teaser-length capture

## Apr 06, 2026 - Detailed ADT residency now follows fog distance, and WDL far terrain can sample minimap tiles instead of flat height tint only

- followed the new viewer direction to stop keeping one fixed detailed ADT footprint regardless of visibility conditions and to make the WDL fallback less obviously fake at distance
- active `src/MdxViewer` terrain behavior after this slice:
	- `src/MdxViewer/Terrain/TerrainManager.cs` no longer hardcodes one `16`-tile detailed AOI for every fog setup
	- the terrain AOI now derives its detailed-target and retention counts from the active terrain fog end distance, clamped into a smaller near-field window when fog is short and expanding back toward the previous larger footprint when fog allows it
	- AOI reevaluation now also happens when the fog-driven streaming target changes, not only when the camera crosses a tile boundary or near-corner bias changes
	- `src/MdxViewer/Terrain/WdlTerrainRenderer.cs` now supports sampling the existing minimap tile cache/loader through `MinimapRenderer`, with a height-color fallback when a tile texture is missing or not uploaded yet
	- `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp.cs` now thread the existing viewer minimap renderer into the WDL far-terrain path instead of creating a separate minimap decode stack
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 06, 2026 with existing warnings only
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started and reached game-folder loading before termination
- important boundary:
	- this is compile plus startup-smoke validation only
	- no live real-data retest has been captured yet for minimap texture orientation on WDL tiles, visual handoff quality between detailed ADT and WDL, or actual FPS impact on the target dense maps

## Apr 06, 2026 - v0.4.7 release prep now ships aligned docs and concise change notes through GitHub Actions

- followed the release request to package the current performance and UI train as `v0.4.7` instead of leaving the repo and release workflow stuck on the older `v0.4.6.1` snapshot
- active release-prep state after this slice:
	- `src/MdxViewer/MdxViewer.csproj` and `src/MdxViewer/MdxViewer.CrossPlatform.csproj` now both report `0.4.7`
	- the packaged release archive now includes the repo-level README, the viewer README, the shipped user guide, and a concise checked-in `v0.4.7` changes file
	- both release workflow copies now read their GitHub release body from the same checked-in `src/MdxViewer/docs/releases/v0.4.7.md` note instead of hardcoding stale `v0.4.6.1` text
	- top-level and viewer README release snapshots now foreground the real themes of this train: shell regrouping, performance work, PM4 ranking repair, and continued `wow-viewer` runtime extraction
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 06, 2026 with existing workspace warnings only
- important boundary:
	- this is release-prep and packaging alignment, not broad new runtime signoff for every viewer subsystem touched during the train

## Apr 06, 2026 - The right sidebar no longer uses tabs; viewer tools are stacked as sequential sections

- followed fresh live feedback that the replacement right-sidebar tabs were still behaving like a broken tab host and effectively pinning the view back to the inspect surface
- active `src/MdxViewer` shell behavior after this slice:
	- the viewer-mode right sidebar no longer uses a tab bar
	- `Inspect`, `Terrain`, `PM4`, `World`, and `Diagnostics` now render as stacked collapsing sections in one continuous sidebar flow, matching the left-sidebar style more closely
	- PM4 and editor-task focus helpers now drive one-shot section expansion in the right sidebar instead of trying to select a persistent tab state
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 06, 2026 with existing warnings only
- important boundary:
	- this is compile validation only
	- no fresh live retest has been captured yet for whether the stacked sections feel right in normal viewer usage

## Apr 06, 2026 - The active shell is now two-sidebar first; the broken bottom drawer path was removed from layout

- followed immediate runtime feedback that the new bottom drawer was not functioning correctly and that the shell felt better as two static sidebars with consolidated panel ownership
- active `src/MdxViewer` behavior after this slice:
	- the bottom drawer is no longer part of the active layout path or viewport reservation math
	- the left sidebar is back to navigation-only ownership instead of duplicating workspace task routing
	- the right sidebar is now the single consolidated tool surface:
		- in `Viewer` mode it shows selection summary once and groups the remaining tools into fixed tabs (`Inspect`, `Terrain`, `PM4`, `World`, `Diagnostics`)
		- in `Editor` mode it shows the current editor-task surface only, with task routing still owned by the top toolbar instead of the left sidebar
	- PM4 focus/open flows now route into the right sidebar instead of trying to revive the removed drawer path
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 06, 2026 with existing warnings only
- important boundary:
	- this is compile validation only
	- no fresh live retest has been captured yet for whether the new two-sidebar grouping feels sane across the active world and PM4 workflows

## Apr 06, 2026 - The active shell is now fixed-frame first: top options bar, left navigator, right tool shelf, and a resizable bottom drawer

- followed direct user feedback that the dockable-panel path still felt structurally wrong even after multiple cleanup passes, and that the viewer should instead use a hard-coded WoWEdit-style frame with static regions that scale predictably
- active `src/MdxViewer` shell behavior after this slice:
	- the viewer now defaults back to the fixed shell path instead of dockspace mode, with the top toolbar always visible again as the primary options bar
	- `P` now toggles the new bottom drawer instead of reopening the old `Workspace Bars` panel concept, and `I` still toggles the right sidebar
	- the right sidebar now behaves as a compact selection/tool shelf instead of trying to host every workflow inline; detailed terrain, PM4, world, and diagnostics content moved into a real bottom drawer
	- the new bottom drawer is resizable, persists its height along with left and right sidebar widths, and groups tools into static tabs: `Workspace`, `Terrain`, `PM4`, `World`, and `Diagnostics`
	- PM4 open/focus flows now route into that bottom drawer instead of forcing dockspace back on
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 06, 2026 with existing workspace warnings only
- important boundary:
	- this is compile validation only
	- no fresh runtime retest has been captured yet for whether the new fixed shell feels materially better in live use on the active world scenes

## Apr 06, 2026 - PM4 object match suggestions no longer rank against the whole loaded scene with a WMO-first bias

- followed direct runtime feedback that the active PM4 hover/selection matcher had become unusable because essentially every PM4 object was surfacing the same OilPlatform WMO as its top suggestion
- active `src/MdxViewer` behavior after this slice:
	- `Terrain/WorldScene.cs` now restricts PM4 object match candidates to the same local tile neighborhood (`±1` tile) already used by the PM4/WMO placement correlation report instead of scoring against every loaded placement in the scene
	- PM4 object match ranking now uses the shared `WowViewer.Core.PM4.Services.Pm4CorrelationMath.CompareCandidateScores(...)` geometry comparator first, so same-tile state, footprint overlap, planar overlap, footprint area similarity, and footprint distance drive the order before any coarse evidence-family tie-break
	- the older `GetPm4ObjectMatchEvidenceRank(...)` path is now only a late tie-break after shared geometric ranking and optional linked-anchor gap, which removes the previous non-zero-family WMO-first dominance that could pin many unrelated PM4 objects to one nearby WMO
- validation completed:
	- editor diagnostics were clean for `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 06, 2026 with existing workspace warnings only
- important boundary:
	- this is still compile validation only in this chat
	- no fresh real-data viewer retest has been captured yet for whether the top-match list now stops collapsing to OilPlatform on the active development map

## Apr 05, 2026 - Active terrain AOI now targets a 16-tile detailed near field, and terrain-world object streaming is less stingy while still staying unique-asset based

- followed the user's direct correction that the detailed ADT footprint should feel more like the real game engine and less like a tiny high-detail cross that pops constantly at the edges
- active terrain/runtime behavior after this slice:
	- `src/MdxViewer/Terrain/TerrainManager.cs` no longer builds the detailed terrain set from the old `8`-tile cross-plus-diagonals rule
	- the AOI selector now ranks a full `5x5` candidate neighborhood around the camera tile and keeps the best `16` detailed ADTs loaded, biased toward the center ring first and then toward camera heading and corner approach
	- retention is now slightly wider than the strict visible target, so the terrain path can avoid dropping a tile the moment it falls out of the top `16` candidates during boundary transitions
	- GPU terrain upload throughput was raised modestly (`6` uploads / about `7 ms` budget) so the larger detailed footprint can fill in faster without requiring full-load mode
	- `src/MdxViewer/Terrain/WorldScene.cs` now uses a less stingy streaming terrain asset policy (`12` visible MDX, `6` visible WMO, `4` deferred loads with a slightly larger budget) so the larger ADT window does not make unique object assets appear materially later than their terrain
	- important proof/ownership detail: `WorldAssetManager` was already de-duplicating model loads by normalized asset path; this slice keeps that unique-asset load model and tunes how quickly those unique requests drain, rather than claiming true instanced world rendering landed
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -v q -nologo`
- proof boundary:
	- this is build validation only
	- no automated tests were added or run
	- no real-data runtime retest has been captured yet for whether the `16`-tile footprint and larger unique-asset load budget feel natural in the live viewer

## Apr 05, 2026 - The dockable shell now follows the WoW-style direction: hotkey-driven workspace bars panel, reclaimed top chrome, and inspector-set toggle

- followed the user's explicit decision to stop iterating on generic freeform docking ergonomics and instead move the active shell toward a WoW-like model with familiar panel hotkeys
- active shell behavior after this slice:
	- the old fixed top options bar is no longer drawn in dockspace mode; that top strip is now reclaimed for the scene/dock host, and the controls live in a real shell panel instead
	- `src/MdxViewer/ViewerApp.cs` now adds `P` as a hotkey for the new `Workspace Bars` panel and `I` as a hotkey for the right-side inspector/workflow panel set
	- `src/MdxViewer/ViewerApp_Sidebars.cs` now exposes `Workspace Bars` as a real panel with workspace selection plus the former quick terrain/world display toggles from the toolbar
	- dockspace mode now treats the workspace bars panel as part of the grouped quadrant fallback and saved shell layout path
	- the fixed toolbar remains only as the legacy non-dock fallback path; it is no longer the primary shell surface
- important boundary:
	- this is still a panel-based shell, not a full bottom-bar or action-bar recreation of the retail WoW UI
	- `I` currently toggles the existing right-side inspector/workflow set rather than a newly split standalone info-only window
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -v q -nologo`
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started normally and loaded the configured game folder before shutdown
- proof boundary:
	- no automated tests were added or run
	- no live retest has been captured yet for actual `P`/`I` usability or whether the workspace-bars panel feels better than the old fixed toolbar in real usage

## Apr 05, 2026 - Dockable shell panels now persist their own layout, and the default fallback is a quadrant-stack layout instead of ad hoc first-use placement

- followed fresh live feedback after the dockable-panel extraction landed:
	- panel positions were not surviving restarts reliably
	- the default dockable layout still felt chaotic and did not scale cleanly across window sizes
- active shell behavior after this slice:
	- `src/MdxViewer/ViewerApp.cs` now persists dockable shell panel rectangles in `output/settings/viewer_settings.json` instead of relying only on ImGui first-use placement or ambient `.ini` behavior
	- the saved panel rectangles are normalized against the dockspace host, so they rescale across later window sizes instead of restoring as raw absolute pixels only
	- the shell now has a concrete quadrant-stack fallback layout for dockable mode:
		- top-left: `Navigator`, `Selection`
		- top-right: `Runtime Stats`, `Model Info`
		- bottom-left: `PM4 Workbench`, `Minimap`
		- bottom-right: `Terrain Controls`, `World Objects`
	- `View -> Reset Panel Layout` now clears saved panel rectangles, forces dockable mode on, and reapplies that quadrant fallback so the user can recover from a bad arrangement without hand-dragging every window back into place
	- `UseDockspaceUi`, left/right shell visibility, and panel rectangle state now save with the viewer settings payload
- important boundary:
	- this is explicit floating-layout persistence plus a default grouped fallback; it does not restore true dock-node topology because the current ImGui.NET binding in this workspace still does not expose DockBuilder seeding
	- this improves persistence and startup organization, but it is not yet the WoW-style hotkey popout shell idea the user mentioned as a possible later direction
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -v q -nologo`
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started normally and loaded the configured game folder before shutdown
- proof boundary:
	- no automated tests were added or run
	- no live restart-and-rearrange retest has been captured yet for actual panel persistence or user comfort with the new grouped fallback

## Apr 05, 2026 - First viewer-shell extraction slice landed: shared panel registry plus resize-safe sidebar clamping

- the first concrete implementation step from `plans/mdxviewer_ui_panel_and_prefab_library_plan_2026-04-05.md` is now in the active `src/MdxViewer` shell instead of remaining planning-only
- active shell behavior after this slice:
	- `ViewerApp.cs` now defines a small shared shell-panel model for the current core surfaces (`Navigator`, `Inspector`, `Minimap`) instead of handling dock state as three unrelated special cases
	- dock-panel capture, dock-layout validation, scene-viewport inset logic, and viewport hit exclusion now all route through that shared panel registry rather than duplicating panel assumptions in multiple methods
	- the fixed-sidebar width clamp no longer falls back to `SidebarMaxWidth` when the window is too narrow; sidebars now shrink toward a compact width first, preserving a hard minimum scene viewport instead of overflowing the shell on non-maximized startup/resizes
	- when the window is too narrow to keep both sidebars at compact width, the shell now suppresses lower-priority side panels for layout instead of forcing an invalid combined width
	- docked navigator/inspector/minimap windows now share default-size and size-constraint metadata from the same panel model, which gives the next extraction slice one canonical place to grow panel ownership from
- important boundary:
	- this is still a bridge slice, not the full dock-lane/drawer system
	- `ViewerApp_Sidebars.cs` and `ViewerApp_Investigation.cs` still own too much monolithic UI content; this slice only centralizes current panel metadata/state and stabilizes resize behavior
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -v q -nologo`
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started and reached normal data-source initialization before the process was stopped
- proof boundary:
	- no automated tests were added or run for this slice
	- no live user retest has been captured yet for actual non-maximized resize behavior or the new docked-panel ergonomics

## Apr 05, 2026 - Dockable panels are now the active shell path again, and the bad tabbed-sidebar detour was removed

- corrected the intermediate shell mistake from the same day: the active path is now dockable panel windows in dockspace mode, not tabs embedded inside fixed sidebars
- active shell behavior after the correction:
	- `_useDockspaceUi` now defaults on, so the current shell comes up in the actual dockable-panel mode by default instead of falling back to fixed sidebars unless the user asks for it
	- the right-lane workflow split remains intact as explicit registered panels: `Selection`, `PM4 Workbench`, `Terrain Controls`, `Runtime Stats`, `World Objects`, and `Model Info`
	- dockspace mode opens those panels as independent dockable windows, which is now the intended shell path for the panel work
	- the non-dock fallback no longer uses the tabbed lane-host experiment; it is back to a plain legacy sidebar layout only as a fallback path
	- `OpenPm4Workbench(...)` now forces dockable panel mode and focuses the registered PM4 panel instead of reopening a monolithic inspector section
- important boundary:
	- top/bottom lanes and drawer fallback are still not implemented yet
	- `World Objects` still carries some older investigation and PM4-adjacent content, so workflow ownership is improved but not fully cleaned yet
	- the fallback path still exists for safety, but it is no longer the primary shell direction
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -v q -nologo`
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started normally, loaded the configured game folder and loose overlay, and was then stopped
- proof boundary:
	- no automated tests were added or run for this slice
	- no live retest has been captured yet for actual docked panel usability or constrained-window behavior

## Apr 05, 2026 - Next viewer-shell direction is a panel-based dock/drawer UI, not more UI profiles or duplicated surfaces

- recorded the latest user direction after the streaming and MDX stabilization work reached a better checkpoint: the active viewer UI should stop accumulating duplicated workspace/profile-style surfaces and instead move to one coherent panel model
- concrete implementation surface for this direction now exists in `plans/mdxviewer_ui_panel_and_prefab_library_plan_2026-04-05.md`
- current shell direction to preserve for the next UI pass:
	- split the existing sidebar-heavy and duplicated UI into individual panels instead of monolithic left/right shell buckets
	- support panel drawers or dock lanes on the left, right, top, and bottom of the screen
	- allow multiple panels to stack within each dock area instead of scattering the same controls across `ViewerApp_Sidebars`, investigation surfaces, and one-off windows
	- allow panels to be popped out when the user wants, but keep the default workflow understandable without requiring profile switching or preset hunting
	- treat UI profiles/presets for shell organization as low value for the active viewer; the immediate problem is duplicated controls and unclear ownership, not lack of another shell mode
	- treat `Viewer` and `Editor` as workspaces over one editor-capable app, not as two separate product identities
	- make non-maximized startup and later window resize first-class shell requirements so panels clamp, stack, or collapse instead of disappearing off-screen
	- fold terrain alpha archaeology into the same shell direction: restore separate alpha-layer inspection and give prefab or brush harvesting its own real panel workflow
- implementation bias to preserve:
	- prefer one canonical home per workflow or data family (`Navigator`, `Inspector`, terrain/runtime stats, minimap, PM4 workbench, object tools, lighting/tools) instead of repeating the same toggles in multiple sidebars or investigation panes
	- use docking and stacked panel containers as the primary organization primitive, with drawers as the constrained-screen fallback
	- avoid adding more UI duplication before this shell regrouping lands
	- prefer machine-assisted alpha brush harvesting and dedupe over manual collection; future terrain-analysis UI should plan around a brush or prefab library, not just one debug image panel
- important boundary:
	- this is continuity and direction only; no panel extraction or shell rewrite has landed yet from this note alone

## Apr 05, 2026 - Active terrain streaming is now near-field and camera-biased again, WDL is restored as the far-terrain fallback, and object range can drop below 1.00x

- followed the user's direct correction after another live screenshot still near `18 FPS`: the active viewer was still behaving unlike the retail engine by keeping too many detailed ADTs and too many far objects resident instead of leaning on WDL for distance terrain
- landed the new active-viewer streaming policy slice:
	- `src/MdxViewer/Rendering/Camera.cs` now exposes `Forward`, and `src/MdxViewer/ViewerApp.cs` passes that forward vector into terrain AOI updates so terrain residency can bias toward what the camera is actually facing
	- `src/MdxViewer/Terrain/TerrainManager.cs` no longer uses the old broad `AoiRadius = 4` square; the detailed ADT working set is now a much smaller near-field set, and the latest follow-up raises that active detailed footprint to a stable 8-tile neighborhood by preferring three useful diagonals around the camera instead of only the older cross-plus-lookahead strip
	- `src/MdxViewer/Terrain/WorldScene.cs` no longer hides WDL for every ADT-backed tile at startup, and `OnTileUnloaded(...)` now restores the matching WDL tile so low-res terrain fills back in when detailed ADTs stream out
	- active object visibility/load range is now intentionally tighter by default: `WorldScene` defaults `ObjectStreamingRangeMultiplier` to `0.50x` and allows lowering it to `0.25x`; the shared runtime floor in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs` was lowered to match
	- `src/MdxViewer/ViewerApp_Investigation.cs` now exposes the lower object-stream floor and updated default text in the live investigation UI
	- follow-up visual handoff polish landed in `src/MdxViewer/Terrain/WdlTerrainRenderer.cs`:
		- WDL tiles no longer pop instantly on ADT load/unload transitions
		- `HideTile(...)` and `ShowTile(...)` now drive a short alpha fade, and the WDL pass renders with blend-enabled opacity so distant fallback terrain can ease in/out instead of hard switching
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started normally with no immediate startup errors before the process was stopped
- important proof boundary:
	- this proves the near-field AOI, WDL handoff, and tighter object-range policy compile and do not immediately break viewer startup
		- this also proves the WDL fade transition compiles and still startup-smokes cleanly
	- no automated tests were added or run for this slice
	- no new real-data live FPS capture has been recorded yet, so do not claim the `30 FPS` target or final streaming parity until the user retests the same dense map scene

## Apr 05, 2026 - Active world-object visibility now broad-phases streamed chunk buckets before per-instance MDX/WMO culling, and WMO doodads no longer pay transparent redraw cost when the model has no transparent pass

- followed a fresh live retest screenshot after the terrain batching and MDX sidedness fixes where the viewer still settled around `14 FPS`, with the dominant remaining frame costs now clearly object-side instead of terrain:
	- `WMO vis/draw` about `17.39 ms`
	- `MDX vis` about `16.97 ms`
	- `MDX opaque` about `18.26 ms`
- landed a new active-viewer broad-phase in `src/MdxViewer/Terrain/WorldScene.cs`:
	- streamed chunk object buckets now keep aggregate bounds in `_tileMdxBounds` and `_tileWmoBounds`
	- deferred asset-bound refresh now updates those bucket bounds when newly loaded models resolve real geometry bounds
	- regular MDX visibility no longer scans the flat `_mdxInstances` list first; it now tests each streamed chunk bucket against a coarse frustum/cone/range gate before running the existing per-instance collector inside eligible buckets only
	- WMO visibility uses the same bucket-level gate before per-instance WMO admission, while external WMO spawns still go through the direct path
- landed a narrow WMO doodad hot-path trim in `src/MdxViewer/Rendering/WmoRenderer.cs`:
	- visible doodad sorting now reuses `_visibleDoodadsScratch` instead of allocating a new list every render
	- transparent doodad replay now skips renderers whose shared model reports `HasTransparentWorldPass == false`, avoiding a second no-op world-pass traversal for opaque-only doodads inside visible WMOs
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build validation only
	- no live runtime FPS capture has been recorded yet after the new object-bucket broad-phase and WMO doodad trim
	- do not claim the `30 FPS` target is met until the user retests the same dense world scene

## Apr 05, 2026 - Active terrain rendering is back on a tile-batched submission path for streamed ADT worlds

- followed the user's correction that normal continent-sized maps remain near `5 FPS` regardless of scene composition, while tiny maps with only about a dozen tiles normalize near `60 FPS`; that invalidated the earlier object-heavy interpretation as the primary answer
- confirmed the active terrain-world path had regressed to per-chunk terrain submission:
	- `TerrainManager` was keeping loaded tiles as `List<TerrainChunkMesh>`
	- `TerrainRenderer` was issuing terrain draw work chunk-by-chunk and layer-by-layer across the loaded AOI
	- rollback branches still contained a tile-batched terrain path using `TerrainTileMeshBuilder`, `TerrainTileMesh`, and one terrain draw per tile
- landed the restore in the active `src/MdxViewer` path:
	- added `Terrain/TerrainTileMesh.cs` and `Terrain/TerrainTileMeshBuilder.cs`
	- `Terrain/TerrainManager.cs` now uploads one batched terrain tile mesh per loaded ADT tile instead of uploading hundreds of per-chunk terrain meshes, while preserving the current hole-visibility rebuild hooks by zeroing hole masks only for the render-time tile build when the user disables holes
	- `Terrain/TerrainRenderer.cs` now renders the tile-batched terrain path when tile meshes are present, but still preserves the chunk path for consumers like VLM that still upload chunk meshes directly
	- `ViewerApp_Sidebars.cs` now exposes `Terrain draw/uniform/tex-bind` counters in `Renderer Stats` so the next live screenshot can confirm whether terrain submission actually collapsed as expected
- terrain guardrail follow-up completed:
	- compared the touched terrain batching file set against baseline commit `343dadfa27df08d384614737b6c5921efe6409c8`
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build validation only
	- no real-data runtime retest has been captured yet for the restored tile-batched terrain path
	- because this slice touches alpha/shadow packing again through the tile array path, do not claim terrain safety until a real development-map runtime check confirms both FPS behavior and blend correctness

## Apr 05, 2026 - WorldScene now delegates MDX route planning to wow-viewer runtime instead of deciding batching and transparent ordering inline

- followed live retest feedback after the UI click fix: the active UI felt somewhat less laggy, but scene performance was still catastrophic at roughly `2-5 FPS`, and the next explicit user direction was to shrink `WorldScene` through `wow-viewer` instead of stacking more host-only patches
- landed a new runtime-owned object-pass seam in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes`:
	- `WorldObjectPassFrame` now carries planned opaque and transparent MDX route lists plus the first batched opaque visible index
	- `WorldObjectPassCoordinator` now owns opaque-route planning, transparent-route planning, and execution over those planned routes instead of leaving `WorldScene` to rediscover batching eligibility and transparent ordering inline during submission
	- new `WorldVisibleMdxPassRoute` is the reusable contract for one visible-MDX route decision
- active `MdxViewer` consumption changed in `src/MdxViewer/Terrain/WorldScene.cs`:
	- `WorldScene` now asks the runtime seam to plan MDX pass routes once per frame after visibility and before submission
	- the host still resolves concrete renderers and issues GL draw calls, but it no longer owns the per-frame decision logic for first batched opaque MDX selection or transparent route ordering
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldObjectPassCoordinatorTests|WorldFramePassCoordinatorTests|WorldObjectVisibilityCollectorTests"`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is focused runtime-test plus build validation only
	- no live runtime FPS capture has been recorded yet for the new route-planning seam

## Apr 05, 2026 - UI click loss is not just scene overload; the active ImGui path was dropping short mouse clicks at very low FPS

- followed live feedback that the active viewer UI itself was laggy enough to need multiple clicks for buttons, which pointed to a deeper frame/input path problem than scene draw cost alone
- confirmed a concrete backend issue in the current `MdxViewer` host path:
	- the Silk.NET OpenGL `ImGuiController` used by `src/MdxViewer/ViewerApp.cs` polls mouse button state once per `Update()` via `CaptureState()` instead of queueing mouse down/up transitions, so at `~5 FPS` short clicks can disappear between frames
- landed a narrow host-side mitigation in `src/MdxViewer/ViewerApp.cs`:
	- raw Silk mouse down/up events are now latched into a pending queue and flushed into ImGui with explicit mouse-button events immediately after `_imGui.Update(...)`
	- this keeps short clicks visible to ImGui even when the scene frame time is bad enough that pure per-frame polling misses them
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build validation only
	- no live runtime retest has been captured yet for same-scene button responsiveness after the backend fix

## Apr 05, 2026 - Terrain liquid no longer submits every loaded mesh blindly; the active viewer now frustum- and fog-culls liquid chunks

- followed a new live screenshot showing the active scene still around `5 FPS` with `World CPU` near `89 ms`, where the liquid pass alone was still costing roughly `16.7 ms`
- found a direct renderer-side over-submission bug in `src/MdxViewer/Terrain/LiquidRenderer.cs`:
	- terrain liquid and loose WL liquid bodies were being drawn by iterating every loaded mesh with no frustum cull and no fog-range distance gate
- landed a narrow hot-path fix:
	- `LiquidRenderer` now keeps bounds per liquid mesh and uses `FrustumCuller` plus a fog-range distance gate before issuing liquid draw calls
	- the renderer now also exposes `LastVisibleTerrainMeshCount` and `LastVisibleWlMeshCount`, and `ViewerApp_Sidebars.cs` shows those live counts in `Renderer Stats` so the next screenshot can confirm whether liquid submission actually dropped
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build validation only
	- no live runtime retest has been captured yet for the new liquid cull, so no FPS improvement is claimed yet

## Apr 05, 2026 - Active LIT sampling no longer treats the first local light as a global base, and scene doodad visibility now gates WMO-internal doodads too

- followed fresh live feedback that the active viewer's LIT result still looked globally wrong and that world performance remained unusable even after earlier object-admission trims
- landed two narrow correctness/perf fixes in the active `MdxViewer` host path:
	- `src/MdxViewer/Terrain/LitLoader.cs` no longer picks the first light with any groups as the base LIT sample; the base sample now comes only from an actual default light, so local lights no longer tint the whole scene just because they appear first in file order
	- `src/MdxViewer/Rendering/WmoRenderer.cs` now has a runtime doodad-visibility gate, and `src/MdxViewer/Terrain/WorldScene.cs` forwards the scene-level doodad visibility into visible WMO renderers so `Show Doodads` also suppresses WMO-internal doodad rendering instead of only standalone scene doodads
- tightened the live investigation wording in `src/MdxViewer/ViewerApp_Investigation.cs` so the LIT panel explicitly says table selection is inspection-only while runtime sampling remains camera-driven and group-0-only
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build validation only
	- no automated tests were added or run for these active-viewer changes
	- no live LIT-correctness or same-scene FPS signoff has been captured yet

## Apr 04, 2026 - Active renderer path now has FOV-aware object detail profiles and explicit object-family controls, but live FPS signoff is still pending

- followed new live feedback that even after earlier loading and transparent-pass trims, the active viewer was still stuck around `5 FPS` and remained unusable on dense terrain scenes
- landed a new runtime-owned object-visibility policy slice instead of only another distance/range tweak:
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityContext.cs` now carries `VerticalFieldOfViewRadians` plus a new `WorldObjectVisibilityProfile` (`Quality`, `Balanced`, `Performance`)
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs` now uses that policy to:
		- cull tiny far objects by projected on-screen size, not only raw distance
		- skip queueing pending assets that are both tiny and low-value for the current view
		- keep near-hold and meaningful front-view assets eligible so the viewer still fills the scene in front of the camera
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now derives vertical FOV from the live projection matrix and passes it into the shared collector; the active viewer default is now the new `Performance` detail profile
- active viewer controls now surface the new efficiency seam directly:
	- `ViewerApp_Sidebars.cs` and `ViewerApp_Investigation.cs` now expose `Show Scene Objects`, `Show WMOs`, `Show Doodads`, and `Object Detail` controls instead of leaving object pressure management as an implicit code path only
	- renderer stats now also show the current `Object detail` profile alongside the existing object-stream range and render timings
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldObjectVisibilityCollectorTests|WorldObjectPassCoordinatorTests|WorldFramePassCoordinatorTests"`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build plus focused runtime-unit proof only
	- no new live runtime FPS signoff has been captured yet
	- the immediate next step is a same-scene retest to compare visible counts, pending loads, and world CPU under the new `Balanced` and `Performance` object-detail modes

## Apr 04, 2026 - New integrated wow-viewer reference-renderer performance plan now treats wow-viewer as the canonical cross-version C# engine target

- added `gillijimproject_refactor/plans/wow_viewer_reference_renderer_performance_plan_2026-04-04.md`
- the new plan explicitly reframes the renderer goal as:
	- one version-aware engine for Alpha-era and 3.x-era data
	- artifact-preserving runtime ownership, not just parser ownership
	- real performance architecture work in `wow-viewer`, not endless isolated `MdxViewer` hotfixes
- the plan unifies the previously separate world-runtime and M2-runtime directions under one staged renderer program:
	- measurement and fixed-scene proof harness
	- version-profile system for runtime rules
	- world frame and visible-set extraction
	- spatial index and residency ownership
	- WMO runtime decomposition
	- M2 runtime completion plus real batching/submission work
	- shared lighting and era parity
	- `WowViewer.App` consumer cutover
- current recommended next slices from that plan:
	- visible-set runtime extraction in `wow-viewer`
	- M2 scene-submission and batching design in `wow-viewer`
	- fixed Alpha/3.3.5 performance proof harness work
- important proof boundary:
	- this is planning and continuity only
	- no new runtime behavior or performance result is claimed from the plan itself

## Apr 04, 2026 - World runtime slice 02 now has an implementation-ready build boundary instead of only an umbrella label

- tightened `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md` so `Slice 02 - Visible Set Runtime Extraction` is now a concrete build plan rather than only a phase heading
- the current agreed first extraction seam is narrow on purpose:
	- move pure WMO/MDX/taxi visibility admission and visible-bucket scratch ownership into `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility`
	- keep renderer lookup, pending visible-load queueing, animation advance, transparent sort, and actual draw submission in `MdxViewer.WorldScene`
- important compatibility-bridge rule recorded in that plan:
	- do not let `WowViewer.Core.Runtime` take dependencies on `WmoRenderer`, `IModelRenderer`, `WorldAssetManager`, or raw GL-state ownership just to force the extraction
	- if frustum checks need host help, use a tiny adapter seam rather than smuggling viewer renderer types into the runtime layer
- expected first file set from the build plan:
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldObjectInstance.cs`
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/*`
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
- important proof boundary:
	- this is still continuity and execution-surface work only
	- no slice-02 runtime extraction code has landed yet from the plan edit itself

## Apr 04, 2026 - First world-runtime visible-set bridge is now landed in wow-viewer and consumed by MdxViewer

- the first real `wow-viewer` world-runtime slice beyond telemetry is now in code:
	- `WowViewer.Core.Runtime.World.WorldObjectInstance`
	- `WowViewer.Core.Runtime.World.Visibility.WorldObjectVisibilityContext`
	- `WowViewer.Core.Runtime.World.Visibility.WorldVisibleWmoEntry`
	- `WowViewer.Core.Runtime.World.Visibility.WorldVisibleMdxEntry`
	- `WowViewer.Core.Runtime.World.Visibility.WorldVisibilityFrame`
	- `WowViewer.Core.Runtime.World.Visibility.WorldObjectVisibilityCollector`
- active `MdxViewer` integration now uses that seam in `src/MdxViewer/Terrain/WorldScene.cs`:
	- nested visible-entry structs were removed from `WorldScene`
	- `WorldRenderFrame` now delegates visible-bucket ownership to runtime-owned `WorldVisibilityFrame`
	- WMO and MDX visible-set collection now routes through `WorldObjectVisibilityCollector`
	- `WorldScene` still owns renderer lookup, pending visible-load queueing, animation update, transparent sort, and actual draw submission
- focused proof landed:
	- `wow-viewer/tests/WowViewer.Core.Tests/WorldObjectVisibilityCollectorTests.cs`
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WorldObjectVisibilityCollectorTests`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build plus focused runtime-unit proof only
	- no real-data viewer capture or performance signoff was done in this slice
	- slice 03 pass extraction is still open

## Apr 04, 2026 - First world-runtime object-pass coordinator bridge is now landed, but capture automation is still not non-interactive

- the next world-runtime slice beyond visible-set ownership now exists in code:
	- `WowViewer.Core.Runtime.World.Passes.WorldObjectPassFrame`
	- `WowViewer.Core.Runtime.World.Passes.WorldObjectPassCoordinator`
- active `MdxViewer` usage after this slice:
	- transparent MDX sort scratch no longer lives only in `WorldScene`
	- MDX animation dedup and concrete object-pass iteration for WMO opaque, MDX opaque, and MDX transparent now route through runtime-owned pass helpers
	- `WorldScene` still owns GL state, renderer lookup, batch begin, and actual renderer invocation
- focused proof landed:
	- `wow-viewer/tests/WowViewer.Core.Tests/WorldObjectPassCoordinatorTests.cs`
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldObjectPassCoordinatorTests|WorldObjectVisibilityCollectorTests"`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- capture automation clarification from the same investigation:
	- queued capture execution is automatic once a request exists, but queue creation is still UI-driven in `ViewerApp_CaptureAutomation`
	- current startup args for `MdxViewer` still only open a file path; they do not queue a capture or drive the split development-map world path non-interactively
	- do not claim a real-data capture pass was completed from this environment unless a startup capture hook or equivalent non-UI path is added first

## Apr 04, 2026 - Startup capture automation and a frame-order coordinator are now landed, but proof is still build-and-unit only

- active `MdxViewer` now has a narrow non-interactive startup path in `src/MdxViewer/ViewerApp_StartupAutomation.cs`:
	- `--game-path <clientRoot>` loads a base MPQ client directly
	- `--build <buildVersion>` pins the selected client build
	- `--loose-map-overlay <folder>` reuses the existing loose-overlay attach path without UI
	- `--world <path-or-virtual-path>` loads a world or asset after startup setup
	- `--capture-shot <shotName>`, `--capture-output <folder>`, `--capture-with-ui`, and `--exit-after-capture` now reuse the existing queued capture pipeline without manual queue creation
- world-runtime slice 03 is now widened beyond the object-family helpers:
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs` now owns the current top-level frame order and visibility gating from lighting/sky/skybox/WDL/terrain through object-phase preparation, liquid, transparent MDX, and overlay
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now provides concrete host callbacks into that coordinator instead of keeping the whole stage order inline
	- `wow-viewer/tests/WowViewer.Core.Tests/WorldFramePassCoordinatorTests.cs` adds focused order/gating proof on top of the existing visibility and object-pass tests
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldFramePassCoordinatorTests|WorldObjectPassCoordinatorTests|WorldObjectVisibilityCollectorTests"`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- the new startup hook was not exercised against a real fixed capture in this session
	- the frame-order seam is runtime-owned, but actual GL state and renderer calls are still host-owned in `WorldScene`
	- no real-data capture or performance signoff is claimed from this slice

## Apr 04, 2026 - UniqueId archaeology now filters to a visible range, and terrain streaming no longer prewarms whole tile object sets

- followed immediate live feedback that the two UniqueId archaeology sliders were behaving like a split hide mask instead of a normal `[start..end]` range selector
- current active behavior in `gillijimproject_refactor/src/MdxViewer`:
	- the top UniqueId slider now sets the start of the visible range and the bottom slider sets the end of the visible range
	- enabling the filter now keeps placements inside the selected range visible within the chosen scope instead of hiding the selected band and leaving the outside ranges visible
	- detected archaeology-layer actions now apply that visible band directly instead of a `Hide` action label
- followed the same feedback that terrain-world performance was still collapsing while the scene `settled` under asset loading pressure
	- `Terrain/WorldScene.cs` no longer prewarms every streamed tile's MDX/WMO asset set as soon as the tile enters AOI
	- the active terrain-streaming asset policy was cut back to lower visible/deferred load throughput and now further throttles deferred work when the previous frame CPU time is already high
	- `Terrain/WorldAssetManager.cs` also no longer double-enqueues first-time priority MDX/WMO loads into the priority queues
- validation completed:
	- file diagnostics were clean for `ViewerApp.cs`, `Terrain/WorldScene.cs`, and `Terrain/WorldAssetManager.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build validation only
	- no live runtime signoff has been captured yet for the revised frame stability or the corrected UniqueId archaeology workflow

## Apr 04, 2026 - Steady-state world performance is still limited by object visibility/submission; visible-only MDX animation update and 1.00x default range are now active

- followed live runtime screenshots showing that load speed improved but standstill frame rate was still collapsing with large visible-object counts
- current active findings in `gillijimproject_refactor/src/MdxViewer`:
	- the current MDX `batched` path is not true GPU instancing; it shares shader/frame state but still renders each visible instance through its own geoset draw loop
	- WMO rendering is also per-visible-instance and each visible WMO still runs its internal opaque/transparent/doodad passes independently
	- one clear avoidable cost was still present: `WorldScene` was advancing animation by scanning all placed MDX/taxi instances every frame instead of only the renderers that survived visibility admission
- landed follow-up:
	- `Terrain/WorldScene.cs` now updates animation only for visible MDX renderers in the current frame
	- the default `ObjectStreamingRangeMultiplier` is now `1.00x` instead of `2.00x`
	- the investigation/sidebar UI text now explicitly says the MDX `batched` counters are shared-shader submissions, not true GPU instancing
- validation completed:
	- file diagnostics were clean for the touched files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build validation only
	- no live runtime signoff has been captured yet for the new default range or the visible-only animation update

## Apr 04, 2026 - Fullscreen minimap teleport hold experiment was rolled back; teleports are immediate again

- the short-lived fullscreen minimap `teleport loading` hold added earlier on Apr 04 was removed the same day after live feedback showed it made teleports feel blocked and turned the fullscreen minimap into a chore
- current active behavior:
	- triple-click teleport still moves the camera immediately
	- if fullscreen minimap is open, teleport now drops straight back to the world instead of waiting on destination tile settle logic
	- the extra minimap hold-state fields and overlay flow in `ViewerApp` were removed from the active viewer path
- related active minimap state after the same follow-up:
	- `Rendering/MinimapRenderer.cs` was also narrowed to a cheaper tile path by parallelizing background decode, preferring canonical tile paths first, removing first-hit PNG writeback from the critical path, and avoiding mipmap generation for UI tile textures
- validation completed:
	- file diagnostics were clean for the touched viewer files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build validation only
	- no live runtime signoff was captured yet for the revised fullscreen minimap responsiveness on real teleports

## Apr 04, 2026 - Active MdxViewer LIT path now preserves full group metadata, but only sky/fog override is trusted live

- active `gillijimproject_refactor/src/MdxViewer` LIT handling was tightened after real runtime feedback that the first parser no longer crashed but still produced implausible terrain lighting colors
- current landed state:
	- `src/MdxViewer/Terrain/LitLoader.cs` now keeps `highlightSky`, the four sky float bands, `cloudMask`, and the version-`0x80000005` parameter float bands instead of discarding most of the non-color payload
	- the same viewer path now discovers alternate per-map LIT filenames (`lights.lit`, `areatest.lit`, `light.lit`) and lets the user reload a specific source from the LIT investigation panel instead of hardwiring `lights.lit`
	- the live viewer toggle now says `Use LIT Sky/Fog Override`, and `WorldScene` only applies LIT sky/fog values while leaving direct/ambient lighting on the pre-existing scene path until those light-color semantics are proven on real map data
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
	- one real archive-shape check against repo test data confirmed `world/maps/azeroth/lights.lit` is `0x80000004` with `57` entries and `57 * 4 * 0x1550` bytes of light-group payload
	- a second archive-backed check confirmed `world/maps/azeroth/areatest.lit` exists and is an older partial-form file (`version=0x00000002`, `lightCount=-1`, payload `5316` bytes) rather than another full `57`-light list
- important proof boundary:
	- this is still not runtime signoff for exact LIT direct/ambient color semantics
	- the viewer now has a more honest parse boundary and a safer live override boundary, but real-map visual confirmation is still required

## Apr 04, 2026 - WL scene picks now target one loose-liquid body and expose a selected-body wiremesh overlay in MdxViewer

- `gillijimproject_refactor/src/MdxViewer` now treats hovered `WL liquid` hits as a first-class inspect target instead of hover-only metadata:
	- left-clicking a WL plane now selects that exact WL body, switches the UI into the `Editor -> Inspect` task, and isolates the WL body list to the picked entry so the user does not have to search the full grouped table manually
	- the selected-object inspector is now populated with the picked WL body summary, and the WL investigation panel can render a selected-body wiremesh overlay over the translucent WL mesh data
	- `HoveredAssetInfo` now carries the exact `WlBodyKey` for WL hits so scene picks resolve against stable body identity instead of only display text
- validation completed:
	- file diagnostics were clean for the touched `ViewerApp.cs`, `ViewerApp_Investigation.cs`, `Terrain/LiquidRenderer.cs`, and `Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build validation only
	- no live runtime signoff was captured yet for WL click-selection, list isolation, or the wiremesh overlay on real scene data

## Apr 04, 2026 - MDX click selection now follows hovered instance and terrain worlds prewarm streamed object assets

- narrowed the remaining selected-object mismatch to the MDX scene-selection path rather than WMO selection:
	- `HoveredAssetInfo` in `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now carries the exact hovered scene object type/index, not just display text
	- `ViewerApp` now prefers the hovered scene object identity on left-click before falling back to the broader ray picker, which should stop dense-foliage MDX clicks from selecting a different overlapping instance than the tooltip target
- landed a cross-version terrain-world object-load policy in `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`:
	- streamed terrain tiles now queue their MDX/WMO assets as soon as the tile enters the AOI instead of waiting for the later visible-object prioritization pass to be the first load signal
	- deferred asset-load throughput is now scene-policy driven for terrain worlds instead of being hardcoded to the old `6 MDX / 3 WMO / 2 loads / 6 ms` bottleneck
	- the policy is terrain-streaming vs WMO-only, not a new exact-build branch, so `0.5.x`, `0.6.0`, `0.11+`, and later terrain worlds share the same warmup path instead of drifting by accident
- validation completed:
	- file diagnostics were clean for the touched `WorldScene.cs`, `ViewerApp.cs`, and `ModelRenderer.cs` paths
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build validation only
	- no live runtime signoff was captured yet for the MDX click-selection fix or the terrain-world loading improvement

## Apr 04, 2026 - MdxViewer inspector build is repaired and UniqueId hide controls are more usable

- repaired the broken selected-object inspector slice in `gillijimproject_refactor/src/MdxViewer/ViewerApp_Sidebars.cs`:
	- `DrawModelInfoContent()` is valid again and no longer embeds the selected-object helper in the wrong place
	- the fixed right inspector now exposes an explicit `Inspector Width` slider so the user can widen it even if the edge splitter is awkward to grab
- tightened the UniqueId archaeology UI in `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`:
	- changing the hide-range sliders now enables the UniqueId filter immediately instead of leaving a selected range inert until the checkbox or layer buttons are used
	- the detected-layer table is now more compact so the `Hide` action remains reachable in narrower inspector widths
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- this is build-verified UI/runtime plumbing only
	- no live runtime validation was captured yet for selected-object inspector behavior, UniqueId hide behavior, or fixed-sidebar usability

## Apr 04, 2026 - MdxViewer now stages grouped dirty ADT placement saves by source

- `gillijimproject_refactor/src/MdxViewer` now extends the first selected-placement save seam into a real grouped dirty-source queue:
	- translation-only MDDF and MODF moves are now staged across selection changes instead of being limited to one current selection state
	- pending placement moves are grouped by source ADT so one save can write multiple staged moves from the same tile file
	- the `Publish` workspace now surfaces the same pending dirty-source queue, so `Save All Pending` is available even when nothing is currently selected
- current save boundary after this slice:
	- still translation-only for existing ADT placements only
	- grouped save packaging now exists for multiple staged moves, but add/remove placement support, terrain writes, and full packaged map-save workflow are still open
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important proof boundary:
	- no automated tests were added or run for this `MdxViewer` slice
	- no live runtime or real-data editor workflow signoff was captured yet for the new grouped save queue

## Apr 04, 2026 - MdxViewer now stages and saves selected existing ADT object moves through wow-viewer

- `gillijimproject_refactor/src/MdxViewer` now wires the first live UI consumer onto the shared `wow-viewer` placement writer seam:
	- selected tile-backed MDDF and MODF placements can now be translated in-session from the `Objects` workspace
	- the moved preview updates the live `WorldScene` instance, tile cache, and adapter placement list instead of remaining a placeholder-only status surface
	- save output is explicit: the user chooses an `.adt` path unless a writable loose source path already exists
- shared/data-source seams added to support that wiring:
	- `IDataSource.TryResolveWritablePath(...)`
	- `ITerrainAdapter.TryGetPlacementSourceData(...)`
	- `ITerrainAdapter.TryGetPlacementWritablePath(...)`
	- `TerrainManager.TryUpdateCachedPlacementPosition(...)`
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- important boundary:
	- this is still only translation-only save for one selected existing ADT placement at a time
	- no add/remove placement support, dirty-map aggregation, terrain writes, or full packaged map-save workflow landed

## Apr 03, 2026 - wow-viewer now has a first save-capable ADT object move seam

- `wow-viewer` now owns the first narrow shared object-edit transaction boundary for editor work:
	- `WowViewer.Core/Maps/AdtPlacementEditTransaction.cs`
	- `WowViewer.Core.IO/Maps/AdtPlacementWriter.cs`
- landed behavior:
	- translation-only moves for existing `MDDF` and `MODF` entries in ADT or ADTOBJ files
	- `MODF` bounds are translated with the moved placement instead of being left stale
	- the seam is shared-library ownership in `wow-viewer`, not UI-panel-local state
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter "AdtPlacementReaderTests|AdtPlacementWriterTests"`
	- proof includes a real-data roundtrip on `gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_obj0.adt`
- important boundary:
	- this is not general map-save closure
	- no add/remove placement support yet
	- no path-table rebuilds for new models or WMOs yet
	- no terrain writes, dirty-map pipeline, or save packaging workflow yet

## Apr 03, 2026 - wow-viewer editor-transition workflow surface is now explicit

- new workflow assets now exist for the viewer-to-editor push:
	- `.github/prompts/wow-viewer-editor-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-map-editing-foundation-plan.prompt.md`
	- `.github/prompts/wow-viewer-editor-ui-surface-plan.prompt.md`
	- matching `.codex/prompts/` mirrors plus the new continuity file `gillijimproject_refactor/plans/wow_viewer_editor_plan_2026-04-03.md`
- immediate routing intent:
	- use the editor plan set for broad editor-transition asks
	- use the map-editing foundation prompt for PM4 `MPRL` terrain conform, saved object choices, moved-object persistence, dirty-map state, and map-save planning
	- use the editor UI surface prompt for viewer/editor workspace switching and editor task clustering
- prompt output rule:
	- these editor prompts should now emit implementation-ready build plans with exact slice order, repo or file scope, validation, and explicit next actions instead of stopping at architecture commentary
	- this now applies across the whole editor-transition companion set: map-editing foundation, editor UI surface, CLI or GUI surface, and tool-migration sequence
- important boundary:
	- this is workflow and continuity structure only
	- no wow-viewer editor runtime, map save pipeline, or UI reorganization code is landed yet

## Apr 03, 2026 - `wow-viewer` now emits per-build ADT UniqueId reports for timeline work

- `WowViewer.Tool.Inspect` now supports `map uniqueid-report --input <file.wdt|file.adt|directory> [--build <label>] [--output <report.json>]`
- current real-data proof:
	- `map uniqueid-report` against `gillijimproject_refactor/test_data/development/World/Maps/development/development.wdt` produced `wow-viewer/output/reports/map-uniqueids/development.json`
	- that report currently records `64435` placements, `62490` distinct `UniqueId` values, `1701` reused IDs, and `maxReuse=30`
- important boundary:
	- this is the first per-build evidence artifact for later cross-build add/remove timeline work; it does not yet compute the timeline itself
	- `114` scanned development files currently stay in the report `Failures` list because they still classify as `Unknown` rather than `Adt` or `AdtObj`

## Apr 03, 2026 - `wow-viewer` now persists trusted MPQ listfile caches per client/build

- the shared archive bootstrap can now persist and reload a per-client known-file manifest keyed explicitly by client/build strings such as `0.6.0.3592`
- trust boundary for MPQ-era roots:
	- internal listfiles extracted from the client archives are treated as the trusted primary source
	- external/community listfiles are supplemental gap-fill input only
- current consumer effect:
	- archive-backed `mdx chunk-carriers` now enumerates the shared bootstrap `AllFiles` universe, so trusted internal entries and cached supplemental entries actually influence carrier scans
- real-data proof:
	- `archive build-listfile-cache` against `wow-viewer/testdata/0.6.0/World of Warcraft/Data` produced `wow-viewer/output/cache/archive-listfiles/0.6.0.3592.json`
	- that manifest currently records `56742` trusted internal entries, `1291033` supplemental external entries, and `1347773` merged known files
- important boundary:
	- this is a shared known-file-universe cache for MPQ-era discovery workflows; it is not proof that every cached virtual path exists in every local archive root or that downstream format ownership is complete

## Apr 03, 2026 - `wow-viewer` WMO flag typing now names the exterior bits but still leaves `0x2` unresolved

- the broader Alpha Ironforge `--flag-correlation` run confirmed the existing chunk-gating reads at scale:
	- `0x00000200` stays aligned with `MOLR` light refs
	- `0x00000800` stays aligned with `MODR` doodad refs
	- `0x00001000` stays aligned with `MLIQ` liquid
- the current shared `WmoGroupFlags` layer now also names two repo-documented rendering bits:
	- `0x00000008` -> exterior
	- `0x00000040` -> exterior-lighting
- important boundary:
	- `0x00000002` still does not have a safe shared semantic name from the current corpus and remains intentionally raw
	- this improves inspect and shared summary interpretation only; it is not runtime collision or lighting closure

## Apr 03, 2026 - WMO root skybox presence and per-file `MOGP` correlation now exist in wow-viewer

- `wow-viewer` no longer requires the dedicated `MOSB` reader just to know whether a root WMO advertises an explicit skybox; `WmoSummary.HasSkybox` now carries that root capability signal directly
- `WowViewer.Tool.Inspect wmo inspect` now supports `--flag-correlation`, which uses embedded group summaries to show how the flag bits seen in one real root WMO line up with actual BSP or doodad-ref or light-ref or liquid or vertex-color or extra-UV signals
- current real-data evidence on `castle01.wmo.MPQ`:
	- `0x00000001` correlates cleanly with BSP presence
	- `0x00000800` correlates with doodad refs on the flagged group
	- `0x00000002` still needs further corpus evidence before naming
- practical next step from this state:
	- keep ranking `0x00000002` and higher-era residual bits across more real WMOs before turning them into runtime behavior claims

## Apr 03, 2026 - Active LK to Alpha converter is working again on the fixed development-map repro

- the immediate `WoWMapConverter.Core` LK→Alpha regression is no longer in the `0/2303 tiles` false-success state
- confirmed repair points in the active path:
	- `AlphaMcnkBuilder` now uses the real 128-byte Alpha MCNK header contract
	- `LkToAlphaConverter` now fails honestly when zero tiles convert instead of returning success
	- MCIN offsets are validated against real `MCNK` payloads before being trusted
	- the top-level chunk walker now handles odd-sized chunks without assuming an always-present pad byte, which was the reason tiles like `development_0_0.adt` drifted off after `MTEX`
	- string-table extraction for `MMDX` / `MWMO` is bounded so malformed scans do not throw `startIndex` range failures
- current proof level:
	- real-data CLI repro on `gillijimproject_refactor/test_data/WoWMuseum/335-dev/World/Maps/development/development.wdt` now completes at `2303/2303` tiles
	- this closes the immediate converter regression for the old compatibility path only; it does not replace the longer-term requirement to port format ownership into `wow-viewer`
- practical next step from this state:
	- use the repaired old converter as reference input while moving LK/Alpha format ownership into `wow-viewer`, and keep WMO collision/skybox investigation separate from claims about converter closure

## Apr 02, 2026 - M2 docs are now consolidated into one canonical wow-viewer doc set

- the implementation-facing M2 documentation surface now lives under:
	- `wow-viewer/docs/architecture/m2/README.md`
	- `wow-viewer/docs/architecture/m2/implementation-contract.md`
	- `wow-viewer/docs/architecture/m2/native-build-matrix.md`
	- `wow-viewer/docs/architecture/m2/consumer-cutover.md`
- practical routing change:
	- start new M2 implementation work from that folder first
	- treat `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` as the raw evidence log underneath it
	- treat `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md` as migration history and staged prompt context rather than the primary handoff
- `wow-viewer/README.md` now points at the consolidated M2 doc set instead of only the raw native note

## Apr 02, 2026 - Next cross-build slot is Cataclysm `4.0.0.11927`, but only static evidence is reproducible here

- for the post-Wrath step, the nearest actually documented Cataclysm-era build in this repo is Win32 `4.0.0.11927`, so that is the honest substitution for the default ladder's `4.0.6a.13623` until the exact client is available
- repo-local Cataclysm evidence already exists for:
	- `MD20` / `MD21` M2-family continuity
	- M2 still active in the client
	- explicit shader/effect stack strings (`ShaderEffectManager`, `.bls`, shader directories, `M2Cache.cpp`)
- the current chat materially improved that Cataclysm slot with a real static M2 anchor set:
	- `FUN_007242d0` exact `%02d.skin`
	- `FUN_00724270` exact `%04d-%02d.anim`
	- `FUN_0072a740` choose skin profile
	- `FUN_0072a620` exact skin load + async job setup
	- `FUN_0072a4e0` strict init + callback rebuild drain
	- `FUN_00725e00` active section/effect materialization
	- `FUN_00724320` `Diffuse_*` + `Combiners_*` effect builder
	- `FUN_00402390` runtime option registration with Cataclysm default mask `0x2008`
- current blocker:
	- a direct filesystem search under `I:\parp` only found `WoW.exe` under `wow-viewer/testdata/0.5.5/World of Warcraft/` and `wow-viewer/testdata/0.6.0/World of Warcraft/`
	- despite the user's manual loaded-session handoff, the x64dbg session dropped during the first rebasing attempt before a live Cataclysm breakpoint chain could be captured
- practical reading:
	- Cataclysm is no longer just a placeholder substitution; the static anchor map is now real
	- do not claim live Cataclysm `%02d.skin`, `%04d-%02d.anim`, `Combiners_*`, or world-path runtime proof until the debugger session is stable again

## Apr 02, 2026 - Win32 `0x20` shared-record semantics are now partially closed by exact size matches

- follow-up decompilation against the Wrath Win32 bootstrap helpers now ties the repeated `0x20`-guarded relocation pattern to real track-bearing root-model record families rather than leaving it as a generic unknown section bit
- exact size matches recovered from the in-repo wowdev M2 docs:
	- `0x14` -> `M2Track<T>` / single-track families such as `M2TextureWeight`
	- `0x28` -> `M2Color`
	- `0x3c` -> `M2TextureTransform`
	- `0x9c` -> `M2Light`
- practical reading for future `wow-viewer` work:
	- `0x20` marks a shared-record class with nested animated payloads that gets special relocation handling and stays out of the compact runtime render list
	- the exact user-facing label is still open, but it should no longer be treated as an entirely opaque section flag or conflated with `.skin` texture-unit flag notes

## Apr 02, 2026 - First Win32 world-path M2 choose-load capture landed in native notes

- live x64dbg sampling against Win32 `3.3.5.12340` now includes a real in-world doodad choose-load chain instead of only UI-model traffic
- confirmed world-path model:
	- `world\expansion02\doodads\generic\barbershop\barbershop_mirror_01.m2`
- confirmed exact numbered companion skin output:
	- `world\expansion02\doodads\generic\barbershop\barbershop_mirror_0100.skin`
- confirmed runtime outcome:
	- post-load success at `0x0083cd32` with `EAX=1`
	- downstream callback rebuild hits at `0x00832ea0`
- remaining boundary:
	- explicit `0x00838490` skin-init stop did not surface cleanly in the noisy in-world pass
	- world-path combiner/effect-family capture is still pending

## Apr 02, 2026 - Second world-path M2 skin capture plus init-path reachability recorded

- a second in-world doodad chain is now confirmed in the Win32 notes:
	- `world\expansion02\doodads\generic\barbershop\barbershop_shavecup.m2`
	- exact skin output `world\expansion02\doodads\generic\barbershop\barbershop_shavecup00.skin`
- after deleting noisy choose/load breakpoints, the active in-world session also re-hit:
	- `0x00838490`
	- `0x00838561`
	- `0x00836600`
- remaining boundary tightened but is still open:
	- sampled downstream init and combiner callers after cleanup were still UI-heavy rather than cleanly attributed to one of the confirmed world doodads
	- x64dbg MCP timed out and the debug session dropped before a world-attributed init or combiner sample was harvested

## Apr 02, 2026 - Reattach pass closed world-path combiner and init-completion gaps

- after restarting x64dbg and reattaching with only narrow downstream stops, the Win32 `3.3.5.12340` session produced the first clean world-attributed downstream samples:
	- world model `world\generic\human\passive doodads\beds\duskwoodbed.m2`
	- world combiner sample at `0x00836600`
	- resolved effect recipe for that sample: `Diffuse_T2` + `Combiners_Mod2x`
	- world init-completion sample at `0x00838561` on the same object
- practical result:
	- world-path choose/load, effect routing, and init completion are now all grounded in direct Win32 runtime evidence for Wrath

## Apr 02, 2026 - Wrath static M2 contract now decompiled, not just sampled

- `M2_ChooseAndLoadSkinProfile`, `M2_InitializeSkinProfileAndRebuildInstances`, `M2_BuildCombinerEffectName`, `FUN_00836c90`, `FUN_00837a40`, `M2_RegisterRuntimeFlags`, and `M2_NormalizeModelPathAndProbeSkins` are now backed by direct Ghidra decompilation in the canonical note
- recovered concrete choose thresholds from live data at `0x00A45644`:
	- `0x100`
	- `0x40`
	- `0x35`
	- `0x15`
- recovered the actual combiner decision tree and special alpha-route wrapper, not just individual strings
- recovered the exact flag-word bit model for startup, batching, additive sort, and `M2Faster` high-bit optimization masks

## Apr 02, 2026 - Extension gate and external anim contract now decompiled on Win32

- `FUN_0081c390` now directly confirms the strict Win32 cache-open behavior:
	- accepts `.mdl`, `.mdx`, `.m2`
	- rewrites legacy extensions to `.m2`
	- rejects unsupported extensions with the `Model2: Invalid file extension` path
- `M2_FormatAnimFilename_04d_02d` now directly confirms external animation naming as `basename + %04d-%02d.anim`
- `FUN_00837ee0` confirms that animation-track relocation is part of the strict root-model bootstrap, not a separate late pass

## Apr 02, 2026 - Workflow asset location correction for prompt continuity

- canonical workspace workflow assets for this repo remain under `.github/` with Codex mirrors under `.codex/`
- do not route these workflow prompts to a `.copilot/` folder for this repo
- current cross-build native M2 investigation asset location is:
	- `.github/prompts/m2-cross-build-native-investigation.prompt.md`
	- `.codex/prompts/m2-cross-build-native-investigation.md`

## Mar 31, 2026 - wow-viewer M2 Foundation Slice 01 Landed

- `wow-viewer` now has its first dedicated M2 library/runtime seam for slice 01 instead of only prompt-routing coverage
- landed code areas:
	- `wow-viewer/src/core/WowViewer.Core/M2/*`
	- `wow-viewer/src/core/WowViewer.Core.IO/M2/*`
	- `wow-viewer/src/core/WowViewer.Core.Runtime/M2/*`
	- `wow-viewer/tests/WowViewer.Core.Tests/M2FoundationTests.cs`
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now exposes `m2 inspect`
- landed behavior:
	- canonical `.mdl` / `.mdx` / `.m2` identity normalization into `.m2`
	- strict `MD20` root validation plus typed model metadata summary
	- strict external `SKIN` parsing with exact numbered `%02d.skin` ownership
	- explicit choose/load/initialize skin-profile staging in `WowViewer.Core.Runtime`
	- compatibility-only embedded root-profile hints remain metadata, not the primary ownership model
- validation completed:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- proof level is library/build/test coverage plus thin inspect CLI ownership; no real extracted M2 asset was available in-repo for runtime signoff

## Mar 31, 2026 - Ordered M2 Runtime Prompt Set Landed For wow-viewer

- the workspace now has a dedicated staged prompt surface for M2 runtime/rendering recovery instead of routing that work through only PM4/shared-I/O/world-runtime prompts
- landed workflow assets:
	- `.github/prompts/wow-viewer-m2-runtime-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/01-md20-and-skin-runtime-foundation.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/02-section-classification-and-material-routing.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/03-animation-lighting-and-effect-runtime.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/04-scene-submission-and-batching.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/05-consumer-cutover-and-parity-harness.prompt.md`
	- Codex mirrors under `.codex/prompts/wow-viewer-m2-runtime*`
	- continuity plan `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`
- routing decision locked:
	- M2 parser/runtime/rendering work should now route through the dedicated M2 prompt set, not be forced into the broader world-runtime split prompt unless the real problem is still `WorldScene` ownership
	- `MdxViewer` remains a compatibility/reference input for proof and diagnostics, not the design owner of future M2 seams
- important boundary:
	- this entry was workflow/continuity work only at the time it landed
	- slice 01 has since landed separately in `wow-viewer`; use the newer Mar 31 foundation entry above as the current state

## Mar 31, 2026 - Remaining Giant-Root M2 Failures Now Point At The Shaded Draw Path, Not Placement

- after the build-mismatch correction landed, live viewer feedback still showed more than half of the world M2 set missing, especially the giant root structures that should visibly cover the development terrain
- the new screenshots narrowed the remaining seam further:
	- tooltip selection still resolves the missing root models
	- world bounding-box overlays still draw in the right places for those objects
	- `WorldScene` currently draws those debug boxes directly from instance bounds, so this proves scene registration and placement metadata are present, but it does not prove that shaded geoset layers made it through the actual triangle pass
- practical reading of the latest evidence:
	- the earlier build-resolution fix was real and necessary, but it was not the last blocker
	- the remaining failure class is now more honestly described as a world adapted-M2 shaded render-path failure rather than another asset lookup or placement failure
- highest-probability next seams:
	- `src/MdxViewer/Terrain/WorldScene.cs` opaque/transparent submission for `IsM2AdapterModel`
	- `src/MdxViewer/Rendering/ModelRenderer.cs` layer/pass routing inside `RenderGeosets(...)`
	- `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs` blend/flag mapping for large world M2 materials
- next honest diagnostic step:
	- add targeted runtime diagnostics or a temporary solid-color/no-texture force-draw path for adapted M2s so the viewer can distinguish "triangle submission missing" from "material state/texturing makes submitted geometry invisible"
- important boundary:
	- no new code fix landed for this remaining seam yet in this chat
	- runtime signoff is still open even though the stale-build correction remains valid

## Mar 31, 2026 - M2 Runtime Loading Now Prefers The Actual Client Build Over A Stale Selected Build

- follow-up investigation of the still-failing `World\Expansion02\Doodads\Azjol-Nerub\AzjolRoofGiant.m2` disproved the earlier guess that the remaining blocker was necessarily malformed adapter output
- direct `AssetProbe` comparison on the same real 3.3.5 client root showed the actual seam:
	- with build `3.3.5.12340`, `AzjolRoofGiant.m2` adapted cleanly with sane bounds and geometry (`574` verts / `1063` tris)
	- with stale build `3.0.1.8303`, the same asset collapsed to degenerate geometry (`1` vert / `1` tri) with broken bounds even though the `.skin` and texture resolved from MPQ
- landed a narrow runtime correction across `src/MdxViewer/Terrain/BuildVersionCatalog.cs`, `src/MdxViewer/ViewerApp.cs`, `src/MdxViewer/Terrain/WorldAssetManager.cs`, and `src/MdxViewer/Rendering/WmoRenderer.cs`:
	- M2-family loaders now resolve an effective build from the actual client/game path and prefer that over a stale persisted selection when they disagree
	- standalone M2 open, world M2 load, and WMO doodad M2 load now all use that effective build for profile validation, direct adaptation, and M2-to-MDX fallback conversion
	- the shared build-path helper now lives in `BuildVersionCatalog` instead of keeping model-family build inference trapped inside `ViewerApp`
- validation completed:
	- `get_errors` returned clean for the edited files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Mar 31, 2026 with existing workspace warnings only
- important boundary:
	- this is still build/probe validation, not viewer runtime signoff
	- the next honest check is a real viewer reopen of the development map and the failing standalone asset under the saved 3.3.5 base client

## Mar 31, 2026 - Active Viewer World M2s Now Bypass The Batched RenderInstance Path Again

- followed direct runtime feedback after slice 01: repeated asset-miss churn improved somewhat, but many world M2s were still tooltip-visible / pickable while remaining visually absent in the scene
- landed a narrow active-viewer correction in `src/MdxViewer/Terrain/WorldScene.cs`:
	- M2-adapted world doodads now route through `RenderWithTransform(...)` again for both opaque and transparent passes instead of the generic batched `RenderInstance(...)` path
	- classic MDX world doodads stay on the lighter batched path
	- world batch setup now selects the first renderer that will actually use the batched path instead of blindly seeding batch state from the first visible doodad
- rationale:
	- current live `ModelRenderer.RequiresUnbatchedWorldRender` only covered particles and ribbons, not `IsM2AdapterModel`
	- that let Warcraft.NET-adapted world M2s drift back onto the generic instanced world path even though earlier continuity and user symptoms both pointed at that path as the likely invisibility seam
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Mar 31, 2026 with existing workspace warnings only
- important boundary:
	- this is still compile validation only in this chat
	- no real-data viewer flythrough or capture proof has been completed yet, so this should be treated as the next targeted runtime fix candidate, not final closure on the remaining WMO/M2 hiccups

## Mar 31, 2026 - Adapted M2s No Longer Vanish Completely When Their Base Texture Lookup Fails

- followed fresh runtime evidence after the world-path M2 submission fix: more objects rendered, but another class of MPQ-backed M2s still stayed invisible in both world rendering and standalone model viewing
- landed a narrow renderer-side correction in `src/MdxViewer/Rendering/ModelRenderer.cs`:
	- adapted M2s now use the neutral white fallback texture even when the missing texture is the base layer with `Load` blending
	- the renderer no longer suppresses the normal geoset fallback draw just because a pre-release M2 layer missed its texture lookup
- rationale:
	- current `RenderGeosets(...)` could still leave an adapted M2 with zero rendered layers when the primary texture lookup failed, which matches the symptom of “asset exists, tooltip/picking works, but nothing is visible” in both world and model viewer paths
	- this keeps the fix at the shared renderer seam used by both world and standalone M2 viewing instead of adding another world-only workaround
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Rendering/ModelRenderer.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Mar 31, 2026 with existing workspace warnings only
- important boundary:
	- this proves compile integration only in this chat
	- it does not prove that all adapted M2 material/skin semantics are now correct; if some MPQ models still render as partial strips or malformed shells, the next seam is likely in `WarcraftNetM2Adapter` submesh/material extraction rather than in `WorldScene`

## Mar 31, 2026 - Slice 01 Negative Asset Lookup Suppression Landed In The Active Viewer Path

- implemented the first world-runtime stabilization slice in the active `MdxViewer` compatibility path instead of broadening the `WorldScene` split prematurely
- concrete changes landed across `src/MdxViewer/Terrain/WorldAssetManager.cs`, `src/MdxViewer/Rendering/WmoRenderer.cs`, `src/MdxViewer/ViewerApp.cs`, and `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- `WorldAssetManager` now treats cached failed MDX loads as terminal residency for the current session instead of retrying them through `EnsureMdxLoaded(...)`, deferred queueing, and deferred drain passes
	- the world asset queue no longer re-enqueues or re-dequeues known-failed MDX paths, which removes the repeated `.skin` candidate walk and failed-load churn from the active world frame
	- world-path missing external `.skin` results are now remembered so prefetch does not keep fanning out across the same companion-skin guesses after the miss is already known
	- missing `.skin` logging is now once-per-path for the active world path, standalone M2 open path, and WMO doodad M2 path instead of repeating the same message every retry/reopen path
	- the terrain sidebar now surfaces the new world-miss telemetry: suppressed failed-MDX retries, known missing M2-skin count, and suppressed duplicate skin-miss logs
- validation completed:
	- `get_errors` returned clean for the four edited `MdxViewer` files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing workspace warnings only
- important boundary:
	- no new automated tests were added or run for this slice
	- no real-data capture or runtime flythrough signoff has been performed yet, so this proves compile integration and retry suppression wiring only, not final frame-time improvement on the development map

## Mar 31, 2026 - World Runtime Prompt Set Added For The Ordered WorldScene Split

- user chose the broader world-runtime decomposition path as the next planning surface: split `WorldScene` into explicit terrain/WMO/MDX/overlay runtime services in `wow-viewer`, then implement those slices in fresh chats
- added a dedicated workflow surface:
	- root router: `.github/prompts/wow-viewer-world-runtime-plan-set.prompt.md`
	- ordered prompt folder: `.github/prompts/wow-viewer-world-runtime/`
	- continuity plan: `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`
- key sequencing decision:
	- slice 01 is not another abstract renderer rewrite; it is explicit suppression of repeated asset-miss churn, especially `.skin` lookup loops and failed MDX retry spam, because the current log indicates that this noise is already distorting runtime behavior and masking other degradations
	- later slices then move visible-set collection, pass ownership, and final `WorldScene` host thinning into `wow-viewer`
- current validation reality:
	- this chat created planning assets only for the next fresh implementation chats
	- no new runtime code changed in this step
	- capture automation appears usable enough to be referenced as a smoke aid in the new prompt set, but it is still not treated as full runtime signoff

## Mar 31, 2026 - First WorldScene Extraction Seam Now Lives In wow-viewer Core.Runtime

- followed the explicit user direction to stop deepening `src/MdxViewer/Terrain/WorldScene.cs` as the long-term design owner and move the first stable seam into `wow-viewer`
- landed a narrow cross-repo extraction around render-frame telemetry and optimization guidance:
	- added `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderFrameStats.cs` and `WorldRenderOptimizationAdvisor.cs` as the canonical runtime-owned contracts for world render stage/frame stats and the `next win` hint
	- updated `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` to consume those `WowViewer.Core.Runtime.World` types instead of defining its own public telemetry contracts and hint logic inline
	- added runtime coverage in `wow-viewer/tests/WowViewer.Core.Tests/WorldRenderOptimizationAdvisorTests.cs`
	- wired both `MdxViewer.csproj` and `MdxViewer.CrossPlatform.csproj` to reference `WowViewer.Core.Runtime`
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 31, 2026 with 226 tests passing; only existing environment `CS1668` LIB-path warnings remained
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing workspace warnings only
- important boundary:
	- this is the first stable `WorldScene` extraction seam, not a full world renderer migration
	- `WorldScene` still owns live render orchestration, visibility collection, and debug/overlay behavior; only the reusable telemetry contract and advisor logic moved into `wow-viewer`

## Mar 31, 2026 - WorldScene Now Captures Per-Frame Renderer Timings And A Next-Win Hint

- started the first implementation slice from the new renderer-first roadmap instead of adding another isolated feature overlay
- landed a narrow frame-instrumentation and frame-contract seam in `src/MdxViewer/Terrain/WorldScene.cs` plus UI exposure in `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- `WorldScene` now owns a reusable per-frame render contract that carries the visible WMO/MDX scratch lists, transparent sort scratch, stage timings, and MDX batched-vs-unbatched submission counts
	- the active world render path now records timings for deferred asset drain, taxi actor update, lighting, sky, skybox backdrop, WDL, terrain, WMO visibility, WMO submission, MDX animation, MDX visibility, MDX opaque submission, liquids, MDX transparent sort, MDX transparent submission, and the late overlay/debug block
	- the terrain sidebar now exposes a compact `Renderer Stats` tree that reports the last captured world-frame CPU timings plus a heuristic `next win` hint derived from the latest measured layer costs
	- this is intentionally phase-1 instrumentation plus the smallest phase-2 scaffolding; it does not yet rewrite MDX batching ownership or pull WMO shell/liquid/transparent sequencing fully out of renderer-local control
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing workspace warnings only
- important boundary:
	- this is compile validation only in this chat
	- no live development-map runtime capture was performed yet, so the new next-win hint exists but has not been manually read against real flythrough numbers in this session

## Mar 31, 2026 - Renderer-First Roadmap Now Prioritizes Explicit World Render Layers Before More Features

- user reprioritized away from more local viewer fixes and toward the biggest blocker: camera-movement performance on real world maps is still unusable
- planning outcome is now explicit and recorded in `plans/mdxviewer_renderer_performance_plan_2026-03-31.md`:
	- the active bottleneck is the current `src/MdxViewer/Terrain/WorldScene.cs` world path, which still mixes lighting resolution, terrain/WDL, WMO visibility/submission, MDX animation/visibility/submission, liquids, and overlay/debug work in one large render routine
	- `src/MdxViewer/Rendering/RenderQueue.cs` exists but is still not the active world-scene submission architecture
	- the next serious renderer slice should start with per-frame measurement plus an explicit world render-frame contract, then move into MDX batching/state reduction, then WMO scene-level pass ownership
	- `LightService` / `TerrainLighting` stay on the roadmap, but as a follow-up after render-layer ownership is explicit instead of as the first performance slice
	- graveyards from `WorldSafeLocs.dbc` are intentionally deferred until after renderer cleanup and should land as a sibling overlay to the existing Area POI / taxi systems
- important boundary:
	- this is planning/continuity work only in this slice
	- no renderer-performance code has been landed yet from this roadmap
	- no runtime performance signoff has been captured yet beyond the already-known user report that movement is still too slow

## Mar 31, 2026 - Fixed Sidebar Shell Now Uses Draggable Split Panels

- user reported that the current fixed-sidebar shell was still a mess because the left/right sidebars were not meaningfully resizable in practice
- root cause in `src/MdxViewer/ViewerApp_Sidebars.cs` was concrete:
	- the fixed-shell path still treated the sidebars like anchored ImGui windows instead of real split panels
	- panel width state existed, but the shell was still force-driving window placement/size in a way that left normal resizing unreliable and visually unclear
- landed viewer-shell correction in `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- fixed sidebars now behave like anchored panels with explicit draggable vertical splitters instead of pseudo-resizable floating windows
	- the shell now renders dedicated left/right splitter bars in fixed mode and updates `_leftSidebarWidth` / `_rightSidebarWidth` directly from splitter drag input
	- fixed panels are now intentionally `NoResize` windows because the supported resize path is the explicit splitter, not hidden ImGui window borders
	- left/right panels stay anchored to the screen edges while their widths persist and drive the scene viewport inset logic consistently
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing workspace warnings only
- important boundary:
	- this is compile validation only in this session
	- no manual runtime signoff has been completed yet to confirm the new splitters feel correct on the development map across different window sizes

## Mar 31, 2026 - Mouse Camera Regression Was Caused By The Splitter Overlay Capturing The Whole Viewport

- immediate runtime fallout after the new fixed-sidebar splitter shell was that mouse-look camera control stopped working
- root cause in `src/MdxViewer/ViewerApp_Sidebars.cs` was concrete:
	- `DrawFixedSidebarSplitters()` used one transparent full-width window across the whole panel height
	- even though only the splitter strips were interactive, that host window still let ImGui treat the viewport as UI-covered, so right-mouse camera drag could stop qualifying as scene input
- landed correction:
	- replaced the full-width splitter host with narrow splitter-only windows, one per active left/right splitter
	- result: only the actual splitter strips capture mouse input; the rest of the viewport no longer sits under a UI window from the fixed-sidebar shell
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp_Sidebars.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing workspace warnings only
- important boundary:
	- this is compile validation only in this session
	- no manual runtime signoff has been completed yet to confirm right-mouse camera look is restored in the live viewer

## Mar 31, 2026 - Hover Tooltips Can Be Disabled And UniqueId Archaeology Filtering Landed

- user shifted from the earlier WL liquid/FOV follow-up to exploration controls for the live viewer:
	- mouse-hover scene tooltips needed a direct off switch
	- world objects needed a `UniqueId` archaeology scrubber that can hide lower-id layers either across the whole map or within the current camera tile
- landed viewer/runtime behavior across `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/Terrain/WorldScene.cs`:
	- `DrawSceneHoverAssetOverlay()` now respects a scene-level `ShowHoveredAssetTooltips` toggle so hover cards can be silenced without removing the underlying hover metadata path
	- `WorldScene` now owns a scoped unique-id range filter with `PerMap` and `CameraTile` modes so selective object hiding targets an inclusive `min..max` span instead of only a single cutoff
	- object instances now preserve tile ownership metadata so the tile-scoped archaeology filter works after instance flattening
	- the hide check now applies consistently to visible-instance collection, hover hit testing, scene picking, and debug bounding-box rendering so hidden ranges stop both rendering and interacting
	- the `World Objects` UI now exposes `Hover Tooltips`, `Hide UniqueId Layers`, scope selection, current camera-tile reporting, explicit min/max range controls, detected archaeology layers for the active scope, one-click `Hide` actions per detected layer, and reset behavior
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing workspace warnings only
- important boundary:
	- this is compile validation only in this session
	- no development-map runtime signoff has been completed yet for hover-tooltip suppression or the new per-map/per-tile unique-id archaeology workflow

## Mar 31, 2026 - Zone Lighting No Longer Overrides User Fog Distance

- user reported that fog could no longer be effectively removed and that far-view distance regressed after the shared `LightService` / `TerrainLighting` lighting pass.
- root cause was in `src/MdxViewer/Terrain/TerrainLighting.cs` and `src/MdxViewer/Terrain/WorldScene.cs`:
	- zone lighting color overrides were also overwriting `FogStart` and `FogEnd` every frame
	- the terrain sidebar fog sliders were therefore no longer authoritative once `LightService` had an active zone
	- WMO/object cull distance was indirectly shrinking too because `WorldScene` derives the far visibility budget from the active fog end
- landed fix:
	- `TerrainLighting.ApplyExternalLighting(...)` now only overrides light or ambient or fog color, not fog distance
	- `WorldScene.Render(...)` still uses DBC-driven zone lighting colors and time-of-day, but keeps the live fog range on the user-controlled `TerrainLighting` values
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- important boundary:
	- this is compile validation only in this session
	- no development-map runtime signoff has been completed yet for restored no-fog / farther-view behavior

## Mar 31, 2026 - VLM Dataset Planning Reset For Real-Map Reconstruction

- user asked to reset the VLM direction around a v7-like reconstruction model built from real map data, not generic training scaffolding.
- current state was clarified before changing anything:
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` already emits more than the old docs imply, including chunk heights, local/global heightmaps, normals, MCCV, raw shadow bits, shadow analysis, alpha masks, liquids, objects, WDL data, and binary tile output
	- `docs/VLM_Training_Guide.md` and `docs/VLM_DATASET_EXPORTER.md` are stale relative to the active exporter surface and still read like an older v6/v30-era workflow
	- there was no dedicated prompt or plan surface forcing future chats to keep dataset provenance and real-map curation ahead of generic finetuning advice
- planning direction now locked:
	- `test_data/development/World/Maps/development` is the reconstruction target and evaluation corpus, not the only teacher corpus
	- future teacher corpora should come from real exported complete maps from matching client profiles, not synthetic data and not museum outputs treated as canonical truth
	- the next production-worthy slice is dataset manifest/per-tile provenance/categorization work, not immediate model tuning
- new continuity assets added:
	- `plans/vlm_dataset_reconstruction_plan_2026-03-31.md`
	- `.github/prompts/vlm-dataset-reconstruction-plan.prompt.md`
- important boundary:
	- this is planning/continuity/prompt work only in this slice
	- no new VLM export run, curation run, or model-training run has been executed yet

## Mar 31, 2026 - Terrain WDT Global WMO Placements Restored And M2 UV Regression Narrowed

- user shifted from continuity recovery into active runtime regressions on the development map:
	- terrain-backed maps were dropping WDT-level global `MWMO` or `MODF` placements that should behave like roof or shell geometry over ADT terrain
	- M2 rendering regressed badly enough that some tree trunks showed leaf transparency behavior and some small detail doodads rendered like giant projected leaf sheets
- landed terrain-side fix in `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`:
	- WDT global WMO parsing no longer depends only on `IsWmoBased`
	- terrain maps now also parse WDT placements when `MPHD` carries the global-map-object bit (`0x0001`) or when `MWMO` and `MODF` are both present
	- terrain-map WDT `MODF` coordinates are now converted into the same renderer-space convention used by ADT placements instead of being treated like raw WMO-map coordinates
- current M2 regression work is narrowed to adapter-side material metadata in `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`, not the main geoset draw loop:
	- old known-good behavior always rendered M2 layers on UV0
	- newer adapter behavior resolved dynamic texture coord ids and treated negative coord ids as special reflective/env-mapped layers
	- active mitigation now clamps negative texture coord ids back to UV0 and removes the `coordId < 0 => SphereEnvMap` inference
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- important boundary:
	- the terrain WDT fix and the current M2 adapter fix are compile-validated only in this session
	- no live viewer/runtime signoff has been completed yet on the development map for restored WDT global WMOs or the oversized-detail M2 regression

## Mar 31, 2026 - Active M2 Tree Regression Trimmed Back By Restoring The Conservative Section Material Path

- user reported the remaining M2 regression in live runtime terms: some trees were still effectively all leaves with no visible trunks
- strongest seam stayed in `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`:
	- the newer adapter path was still richer than the old stable behavior, including per-section multi-layer material assembly and dynamic texture-coordinate lookup
	- the old stable runtime path was much narrower: effectively first batch per section on `UV0`
	- given the visible regression and the existing adapter comments about misidentified material tables hiding tree trunks, the safer short-term fix was to restore the conservative compatibility path instead of guessing at partial modern semantics
- landed correction:
	- `BuildMaterialsFromBatches(...)` now keeps only the first batch/material per section again in the active runtime path
	- active runtime M2 layers are forced back to `CoordId = 0` for that conservative path
	- this is an intentional rollback-to-known-good compatibility behavior, not a claim that full modern multi-layer M2 batch semantics are now solved
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing workspace warnings only
- important boundary:
	- this is compile validation only in this session
	- no manual runtime signoff has been completed yet to confirm that affected trees now render trunks again on the development map

## Mar 30, 2026 - v0.4.6.1 Release Prep Tightened PM4 Tooltip Messaging And New-User Onboarding

- user requested a small release bump and packaging focus for `v0.4.6.1` with clearer PM4 tooltip framing and less confusing first-run UI guidance.
- landed release-prep changes across viewer UI text, docs, and release workflows:
	- bumped `src/MdxViewer/MdxViewer.csproj` version metadata to `0.4.6.1`
	- updated welcome/status onboarding copy so first-run guidance points users to `File > Open Game Folder (MPQ)` first, with standalone `Open File` as a secondary path
	- aligned `src/MdxViewer/README.md` and top-level `gillijimproject_refactor/README.md` to the `0.4.6.1` release snapshot and added clearer beginner quick-start flow
	- added `src/MdxViewer/docs/ui-screenshot-guide.md` plus `src/MdxViewer/docs/screenshots/README.md` so users can drop candidate images for README/release selection
	- updated release workflow notes in both `.github/workflows/release-mdxviewer.yml` and `gillijimproject_refactor/.github/workflows/release-mdxviewer.yml` to foreground better PM4 WoW-styled tooltip display and in-app game-path-first onboarding
- version-support messaging now remains explicit:
	- documented support stays `0.5.3` through `4.0.0.11927`
	- later `4.3.4`/some `5.x` paths are framed as promising/experimental
	- `6.x+` remains future compatibility work, not current signoff
- important boundary:
	- this slice is release-prep/documentation/workflow wiring and build validation, not broad runtime signoff
	- no new automated tests were added

## Mar 30, 2026 - PM4 Workbench Tab Selection No Longer Snaps Back To Overlay

- user runtime report after the earlier PM4 sidebar regression fix:
	- `Selection` and `Correlation` tabs in the right-sidebar PM4 workbench were visible only briefly, then snapped back/disappeared
	- the rest of the PM4 inspector workflow was acceptable, but this made the tabbed workflow unreliable
- strongest code seam found in `src/MdxViewer/ViewerApp_Pm4Utilities.cs`:
	- tab rendering used per-frame local `ref bool` tab-open state and always drove tab focus from shared tab state in a way that could fight normal ImGui tab persistence
	- this made the workbench vulnerable to unintended tab reselection behavior during normal clicking
- landed fix:
	- added one-shot `_pendingPm4WorkbenchTab` state in `src/MdxViewer/ViewerApp.cs`
	- `OpenPm4Workbench(...)` now requests tab focus through that one-shot state instead of continuously re-driving tab focus
	- PM4 workbench tabs now use non-closable `BeginTabItem(...)` calls with one-shot `SetSelected` flags, then clear the pending request after the tab bar draw pass
	- practical effect: manual tab clicks now persist on `Selection`/`Correlation` instead of snapping back
- workflow and docs updates recorded in this slice:
	- `src/MdxViewer/README.md` now calls out fixed sidebars as startup default, keeps dock panels opt-in, and records the missing screenshot-guide follow-up
	- this follows the current packaging reality where no pre-baked `imgui.ini` should be assumed in release output
- important boundary:
	- this is still build/diagnostic validation only in this session
	- no live viewer/runtime signoff has been completed yet for the tab-persistence fix on the development map

## Mar 30, 2026 - PM4 Sidebar Workbench Visibility And Hover Targeting Regression Trimmed Back

- user runtime report after the inspector-first PM4 shell change:
	- the right-sidebar `PM4 Workbench` looked effectively dead because the `Overlay`, `Selection`, and `Correlation` tabs were no longer reachable in normal use
	- saving PM4 match selections from the sidebar became impractical because PM4 click selection could still lose to nearby scene hits
	- `WL*` hover info was hard to reach when PM4 data was nearby and the hover radius felt too large
- strongest active seams found in `src/MdxViewer/ViewerApp_Sidebars.cs`, `ViewerApp_Pm4Utilities.cs`, `ViewerApp.cs`, and `src/MdxViewer/Terrain/WorldScene.cs`:
	- the right sidebar only opened the `PM4 Workbench` by default when overlay or selection or collection state was already active, which made explicit workbench-open flows look broken
	- `PickObjectAtMouse(...)` still let normal scene hits beat PM4 in cases where the hovered PM4 target was the user's obvious intent
	- hover info for `WMO`, `MDX`, `WL*`, and PM4 used the large wireframe-reveal brush, so nearby PM4 content could crowd out narrower hover inspection
- landed fix:
	- `OpenPm4Workbench(...)` now forces the right inspector open for the PM4 workbench path instead of relying on the old default-open heuristic
	- the right sidebar now always renders the `PM4 Workbench` when a world scene exists and honors the explicit forced-open flag
	- normal click selection now prefers the actively hovered PM4 object key before broader scene-hit distance arbitration, and `Shift + Left Click` collection add uses the same hovered-PM4-first behavior
	- `WorldScene` now keeps the large brush for wireframe reveal but uses a smaller dedicated hover-info brush for hover cards and WL/object hit testing
- validation completed:
	- file diagnostics were clean for `src/MdxViewer/ViewerApp.cs`, `ViewerApp_Pm4Utilities.cs`, `ViewerApp_Sidebars.cs`, and `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing project and environment warnings only
- important boundary:
	- this is still build validation only
	- no live viewer/runtime signoff has been completed yet to prove the PM4 sidebar tabs, saved-match workflow, and WL hover behavior now feel correct on the development map

## Mar 30, 2026 - PM4 Camera-Window Load Regression Trimmed Back To One MSLK Partition Pass

- user runtime report: PM4 overlay loads regressed badly again, stalling around statuses like `1/12` or `1/15` camera-window files and effectively never finishing.
- strongest code-level regression seam found in `src/MdxViewer/Terrain/WorldScene.cs`:
	- the Mar 30 zero-`CK24` regrouping pass added `SplitZeroCk24SeedGroup(...)`
	- that path first called `SplitSurfaceGroupByMslk(...)`, then re-scanned each returned group again via `SelectDominantMslkGroupObjectId(...)`, and only then optionally connectivity-split the leftover bucket
	- on large zero-`CK24` seed groups this stacked another full `MSLK` walk on top of an already expensive PM4 object-assembly path and was the likeliest reason loads appeared stuck on one file
- active fix in `src/MdxViewer/Terrain/WorldScene.cs`:
	- added one shared `TryPartitionSurfaceGroupByMslk(...)` helper that partitions linked-vs-unlinked surfaces in a single pass
	- `SplitSurfaceGroupByMslk(...)` now uses that shared partition helper instead of rebuilding the same grouping state locally
	- `SplitZeroCk24SeedGroup(...)` now reuses the same partition result and only connectivity-splits the genuinely unlinked remainder, preserving the newer semantics without the extra whole-group rescan
- validation completed:
	- file diagnostics were clean for `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing dependency/environment warnings only
- important boundary:
	- this is still build validation only
	- no live viewer/runtime signoff has been completed yet to prove the camera-window PM4 load no longer stalls on the development map

## Mar 30, 2026 - PM4 Unknowns Report Now Exposes Dominant MSLK And MSUR Families

- extended the shared `wow-viewer` `pm4 unknowns` seam so it no longer stops at top-value histograms and edge-fit counts.
- the report now also emits:
	- dominant `MSLK` families grouped by `TypeFlags` or `Subtype` or `SystemFlag`
	- dominant `MSUR` families grouped by `AttributeMask` or `GroupKey` or `IndexCount`
	- for each family, measured linkage signals against direct `MSUR` fits, direct `MPRL` fits, `LinkId` patterns, `GroupObjectId -> MPRL.Unk04`, `CK24`, `MDOS`, and incoming-link fanout
- current fixed-corpus evidence from `test_data/development/World/Maps/development`:
	- the dominant `MSLK` families are heavily concentrated in sentinel-tile-link families such as `type=0x01 subtype=2 system=0x8000`, `type=0x01 subtype=1`, `type=0x01 subtype=0`, and the matching `0x02` families
	- those top `MSLK` families are not fringe buckets; they account for the bulk of link rows and still keep `MSLK.RefIndex` mostly `MSUR`-fitting with smaller mismatch tails strongest toward `MPRR`, `MSPI`, `MSVI`, and `MSCN`
	- the dominant `MSUR` families split sharply between large `CK24=0x000000` umbrella families like `attr=0x02 group=3 indices=3` and populated object-bearing families like `attr=0x02 group=18 indices=4` or `attr=0x03 group=18 indices=4`
	- `group=3` families in the current corpus are overwhelmingly zero-`CK24` and still pull heavy incoming `MSLK` traffic, which strengthens the current umbrella/root-family reading without proving final semantics
	- `group=18` and nearby non-zero-`CK24` families show large `CK24` or `MDOS` or incoming-link fanout, which makes them better candidates for object-facing attribution work than the old flat "all MSUR unknowns are equal" framing
- important boundary:
	- this is still research evidence, not a closed semantic decode of `MSLK.TypeFlags` or `Subtype` or `MSUR.AttributeMask` or `GroupKey`
	- the next PM4 step should cluster mismatch-heavy outlier families, not just aggregate one more whole-corpus histogram
	- `MCSH` shadow/object association remains a separate follow-up seam; current repo state has raw shadow extraction for VLM export, but not a shared object-association layer yet

## Mar 30, 2026 - PM4 MSHD Correlation Pass Landed And Weakened The Root-Bucket Header Hypothesis

- followed the user's request to stop hand-waving around `MSHD` and run a dedicated corpus-scale correlation pass in shared `wow-viewer` PM4 code instead.
- landed shared-library and tool-surface support in `wow-viewer`:
	- `Pm4ResearchMshdAnalyzer`
	- `Pm4MshdReport` and related field or metric summary records
	- `WowViewer.Tool.Inspect` verb `pm4 mshd --input <directory> [--output <report.json>]`
- current fixed-corpus evidence from `test_data/development/World/Maps/development`:
	- `616` files scanned and `502` files with `MSHD`
	- `MSHD.Field0C` through `MSHD.Field1C` are zero in all `502` sampled `MSHD` headers
	- `MSHD.Field00` and `MSHD.Field08` match each other in only `233/502` files, so they do not behave like one trivially duplicated slot across the corpus
	- `MSHD.Field00`, `Field04`, and `Field08` did not show the strong exact-match or strong correlation signal that would be expected if they directly owned root-group or split-bucket counts for current `MSLK` or `MSUR` or `MPRL` families
- important interpretation boundary:
	- this does not prove `MSHD` is meaningless; it only weakens the current root-bucket-count hypothesis for this development corpus
	- high-level coupling can still reflect scene density or file size rather than direct semantic ownership, so `MSHD` remains unresolved
	- current best next step is targeted per-tile comparison against known oddballs instead of assigning bucket semantics prematurely

## Mar 30, 2026 - PM4 UI Defaults Hardened And Inspector Workbench Consolidated

- followed the user's latest viewer-shell request to reduce PM4 clutter and make the UI act more like a click-to-inspect object workflow instead of a pile of drifting debug windows
- landed viewer behavior across `src/MdxViewer/ViewerApp.cs`, `ViewerApp_Sidebars.cs`, `ViewerApp_Pm4Utilities.cs`, and `Terrain/WorldScene.cs`:
	- PM4 object bounds now default off on startup
	- PM4 x-ray / ignore-depth now defaults off on startup
	- fixed sidebars are now the default shell mode instead of docked panels, so the right inspector stays anchored unless the user explicitly re-enables dock panels
	- the right inspector now exposes a consolidated `PM4 Workbench` with overlay settings, selection-focused PM4 inspection, and WMO-correlation review instead of routing the normal workflow through separate PM4 object-match or PM4/WMO floating windows
	- the left sidebar now gives a lightweight world overview once a map is loaded instead of opening straight into file-browser-first usage all the time
	- hover cards are now more compact and more explicitly "click to inspect" instead of trying to dump the full PM4 debug story on mouse-over
- important boundary:
	- the old PM4 alignment micro-window still exists as an advanced fallback entry point from the workbench while the fine-grained transform controls remain verbose and not yet fully reworked inline
	- this is still compile validation only in this session, not runtime signoff on the development map

## Mar 30, 2026 - PM4 Hover Tooltip Now Shows WoW-Style Card And Top Match Candidates

- followed the user's next PM4 usability request after the first generic hover overlay landed:
	- the overlay needed a stronger WoW-tooltip-like visual treatment instead of a plain debug box
	- PM4 objects needed hover-time candidate visibility without forcing a selection change first
	- current `CK24` or objectId or zero-bucket findings needed to be preserved in shared continuity instead of staying viewer-local
- landed viewer behavior in `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp.cs`:
	- hovered asset metadata is now PM4-aware and carries the hovered PM4 object key when the PM4 overlay is active
	- PM4 hover detection now runs PM4-first while the overlay is visible, so the tooltip can lock onto PM4 parts instead of only nearby WMO or MDX scene assets
	- the hover card now uses a darker gold-bordered tooltip style with brighter title text, stronger visual hierarchy, and a compact top-candidates section for PM4 objects
	- PM4 hover candidates are cached separately from the selected-object match cache so the overlay can show likely `WMO` or `M2` matches without rebuilding full reports on every frame
- research note preserved for continuity:
	- current development-map evidence still points to derived `CK24` low-16 object values separating many `WMO`-like PM4 meshes, while `CK24=0x000000` remains an unresolved umbrella bucket that still seems to contain many `M2`-like families
	- the hover tooltip and graph candidate lists are evidence surfaces only, not proof that PM4 object ownership or subobject semantics are closed
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 after the hover-tooltip update
- important boundary:
	- this is still build validation only in this session
	- no live viewer/runtime signoff has been completed yet for the new PM4 hover tooltip or candidate list behavior

## Mar 30, 2026 - PM4 Shift-Click Collection Stops Yielding To Scene Hits

- user runtime feedback on the first PM4 collection slice was clear: the additive collection flow was not useful because `Shift + Left Click` still lost to normal scene-object picks and per-item collection removal could leave stale PM4 highlight state behind
- active correction in `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Pm4Utilities.cs`:
	- the `addPm4ToCollection` click branch now treats `Shift + Left Click` as a PM4-first action instead of routing through the normal `nearest visible thing wins` object-pick priority
	- when a PM4 object is hit, the viewer now directly selects that PM4 part, toggles collection membership, and updates the PM4 inspector text without allowing regular WMO or MDX selection to steal the click first
	- when no PM4 hit is found, the viewer now reports that explicitly and points the user back to graph `Collect` buttons for dense overlap cases instead of silently doing nothing
	- per-item `Remove` in the PM4 collection list now resyncs highlighted PM4 objects immediately so stale collection highlight does not survive a removal click
- validation completed:
	- file diagnostics were clean for `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 after the collection fix
- important boundary:
	- this is still build validation only in this session
	- no live viewer/runtime signoff has been completed yet to prove the adjusted Shift-click path is now acceptable on the development map

## Mar 30, 2026 - PM4 Multi-Object Collection Export Added In Viewer Utilities

- the viewer now has a lightweight PM4 collection workflow for comparing one selected part against related parts or suspected duplicate placements
- landed viewer behavior:
	- `Shift + Left Click` on PM4 geometry still exists, but the primary path should now be the graph UI because viewport PM4 picking is not reliable against real scene objects
	- the `PM4 Graph` panel now exposes direct `Collect` buttons at the link-group, MDOS-group, and part levels, plus top-level `Add Part`, `Add Merged Group`, `Export Collection JSON`, and `Clear Collection`
	- collected PM4 parts now render with a distinct in-scene highlight color so the temporary multi-selection is visible without relying on the export list alone
	- collection export now writes per-part debug data plus grouped signature summaries, simple same-signature stack clusters, and a `likelyDuplicateScore` heuristic for sanity-checking overlapping copies
- implementation seams:
	- `src/MdxViewer/ViewerApp.cs` handles additive PM4 pick input and stores the temporary collection
	- `src/MdxViewer/ViewerApp_Pm4Utilities.cs` owns the PM4 collection UI and JSON export
	- `src/MdxViewer/Terrain/WorldScene.cs` now exposes arbitrary PM4 object debug lookup, merged-group lookup, and collection highlight rendering for export-backed multi-selection
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 after the feature landed
- important boundary:
	- this is build validation only in this session
	- no live viewer/runtime signoff has been completed yet on the development map for the new PM4 collection workflow

## Mar 30, 2026 - PM4 Selection Family Split Backed Out After Runtime Regression

- user runtime feedback on the Mar 30 zero-`CK24` selection work was unambiguously negative:
	- first the separate selection-family path over-selected unrelated geometry
	- then the partial rollback still under-selected and visibly split objects in half
- active correction in `src/MdxViewer/Terrain/WorldScene.cs`:
	- removed the separate `_pm4SelectedObjectFamilyGroupKeys` selection map and `_selectedPm4ObjectFamilyGroupKey`
	- selection, highlight, and selected-object graph expansion now use the original `_pm4MergedObjectGroupKeys` path again
	- kept the selected-only PM4 match cache and selected-only match builder changes, since they are separable from the grouping regression
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 after the rollback
- important boundary:
	- this is compile validation only in this session
	- no live viewer/runtime signoff has been completed yet to prove the rollback restores the pre-regression zero-`CK24` selection behavior

## Mar 30, 2026 - Zero-CK24 Same-Tile Family Regrouping Added In WorldScene

- followed the user's latest zero-`CK24` hypothesis past the first seed split and into the later merged-group ownership path in `src/MdxViewer/Terrain/WorldScene.cs`
- strongest concrete runtime cause found in this slice:
	- zero-`CK24` parts already use synthetic per-part group keys (`0x80000000 | objectPart`) by default
	- later regrouping depended on `_pm4MergedObjectGroupKeys`
	- shared `Core.PM4` connector merging only merges across neighboring tiles and explicitly refuses same-tile merges
	- result: same-tile zero-`CK24` families could stay permanently fragmented even when local connector or `MPRL` frame evidence suggested one larger placed-object family
- landed viewer behavior:
	- keep the existing shared cross-tile connector merge pass unchanged
	- add a viewer-local same-tile regrouping pass for synthetic zero-`CK24` keys using connector overlap, expanded-bounds overlap, placement-anchor proximity, linked-`MPRL` floor compatibility, linked-heading compatibility, and umbrella handling for zero-link or high-ref groups
- proof gathered before the patch:
	- real-data forensics on `development_23_18.pm4` showed zero-`CK24` was not one flat unlinked bucket; it contained `204` distinct link groups and `203` of them already had non-zero `MSLK.GroupObjectId`
	- that pushed the current theory away from “missing all link ownership” and toward “same-tile multi-group family evidence is not being regrouped later”
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 after the regrouping patch
- important boundary:
	- this is still compile validation only in this session
	- no live viewer/runtime signoff has been completed yet to prove a selected zero-`CK24` part now expands to the intended broader M2-like family

## Mar 29, 2026 - v0.4.6 Release Target And Renderer-Layer Pivot

- user runtime feedback after the latest PM4 fixes is materially different from earlier sessions:
	- PM4 objects are now described as almost `100%` correct on the active development-map workflow
	- that makes the latest PM4 runtime changes a freeze-worthy release seam rather than another speculative experiment
- `v0.4.6` should now be treated as the active viewer release target in this tree
- the PM4 release-facing wins that need to stay called out together are:
	- ADT-scale PM4 camera-window indexing instead of the earlier wrong WDL-scale indexing
	- transposed PM4 filename tile remap into terrain tile space
	- correct handling for zero-byte PM4 carriers and empty-known PM4 windows
	- removal of terrain-AOI PM4 slicing for already loaded PM4 content
	- linked-group placement resolution for non-zero `CK24` seed groups instead of one shared seed transform
- next rendering/performance priority is no longer another narrow PM4 correctness pass by default
- current renderer direction requested by user:
	- move toward real render layers and explicit submission buckets instead of the current hard-coded `WorldScene` pass order with renderer-local immediate draw behavior
	- reduce draw-call and state churn so debugging overlays and exploration surfaces stop fighting the main scene path
- most likely first architecture seam:
	- keep world visibility/culling collection in `WorldScene`, but build per-frame render-layer submission lists for terrain opaque, WMO opaque, MDX opaque, liquids, transparent world geometry, PM4 overlay solids, PM4 overlay lines, and debug/editor overlays
	- route compatible items through a shared queue or batching surface instead of mixing cull, sort, GL-state changes, and direct draw calls inside one monolithic frame function
- important boundary:
	- `v0.4.6` still does not imply final renderer performance closure
	- the current performance work is a first reduction in waste, while the render-layer/submission redesign is still ahead
	- release packaging currently depends on a workflow-side publish mitigation because `WoWMapConverter.Core` still references `WoWRollback.PM4Module` as an `Exe` project, which causes duplicate dependency publish artifacts during viewer publish unless duplicate publish-output errors are relaxed
	- no bundled `area_crosswalk.csv` should be treated as a release requirement; the intended path is runtime `AreaTable` or `Map`-based mapping from user-provided archives or explicit DBC inputs, with CSV crosswalks remaining optional user-side inputs only

## Mar 29, 2026 - Second Viewer Performance Slice Defers WMO Doodads And Disables Object Fog By Default

- follow-up to the first `WorldScene` MDX classification pass after the user reported the viewer was still hitching hard during tile or data loads and that world objects were appearing inside unwanted fog
- strongest newly confirmed hitch source in the active viewer path:
	- `src/MdxViewer/Rendering/WmoRenderer.cs` was eagerly calling `LoadActiveDoodadSet()` in the constructor
	- that constructor path could recursively build many doodad `MdxRenderer`s on the render thread as soon as a WMO shell became visible
- landed behavior in this slice:
	- `WmoRenderer` now supports deferred initial doodad loading for world-scene WMO usage and incrementally loads queued doodad models during render under a small per-frame budget instead of eagerly expanding the whole doodad set in the constructor
	- `src/MdxViewer/Terrain/WorldAssetManager.cs` now opts world-scene WMO loads into that deferred doodad path
	- `src/MdxViewer/Terrain/WorldScene.cs` now lowers render-thread deferred asset processing from `24 loads / 20 ms` to `6 loads / 4 ms` per frame
	- `WorldScene` now disables object fog by default through a dedicated `ObjectFogEnabled` policy and still keeps WMO culling tied to real terrain fog distance instead of the disabled object-fog range
	- `src/MdxViewer/ViewerApp.cs` now exposes a `Fog Objects` checkbox in the world-objects panel so the old behavior can still be re-enabled for comparison
- validation completed:
	- editor error checks were clean for `WorldScene.cs`, `ViewerApp.cs`, `WmoRenderer.cs`, and `WorldAssetManager.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing solution warnings only
- important boundary:
	- this is still compile validation only in this session
	- no live viewer frame-time capture or real-data runtime signoff was completed yet for the deferred WMO doodad path or the new default no-object-fog policy

## Mar 29, 2026 - Viewer Performance Pivot Started With WorldScene MDX Classification Pass

- user direction has shifted away from more PM4-first work and toward real viewer rendering performance or lighting or shader quality, because current map loads still feel unusable at roughly `1-5 FPS`
- first chosen slice is deliberately CPU-side and narrow:
	- reduce per-frame duplicate object work in `src/MdxViewer/Terrain/WorldScene.cs`
	- do not start with shader-parity or sky or lighting refactors before the main scene loop is cheaper
- landed optimization:
	- `WorldScene` now classifies visible loaded `MDX` or taxi-actor instances once per frame into a reusable scratch list
	- the opaque and transparent doodad passes now reuse that one visibility result instead of redoing AABB distance checks or frustum tests or `TryGetQueuedMdx(...)` lookups in separate passes
	- shared per-instance fade values are also precomputed once and reused across both passes
- validation completed:
	- editor error check on `WorldScene.cs` passed
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing environment warnings only
- important boundary:
	- this is a first hot-path reduction only, not full FPS recovery
	- no live viewer frame-time capture or runtime signoff was completed in this pass yet

## Mar 29, 2026 - PM4 Terminology Reconciliation Locked For `wow-viewer`

- the current PM4 reader or analyzer stack is no longer allowed to blur wowdev field names with local research aliases
- current locked rule:
	- use wowdev PM4 or PD4 names when the docs actually name a field
	- use raw offset-style names first when the docs only expose placeholders
	- mention local names second as explicit aliases with confidence level when semantics are still open
- important current reconciliations:
	- `MSUR.AttributeMask`, `MSUR.GroupKey`, `MSUR.MdosIndex`, `MSUR.PackedParams`, `CK24`, `Ck24Type`, `Ck24ObjectId`, and `MSLK.GroupObjectId` are local research aliases, not original wowdev terminology
	- `CK24` remains a useful derived identity slice from `MSUR._0x1c`, but it should not be described as an official PM4 field name
	- `MSUR.Height` is now known to be a bad name for the final float; current geometry evidence says it behaves like a signed plane-distance term
	- `MSLK.RefIndex` should no longer be spoken about as if the wiki label `msur_index` were fully closed truth across the corpus
- continuity updates landed in:
	- `gillijimproject_refactor/src/Pm4Research.Core/README.md`
	- `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md`
	- `.github/prompts/wow-viewer-pm4-library-implementation.prompt.md`
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
	- `wow-viewer/README.md`
- practical implication for future chats:
	- PM4 work should now default to terminology like `MSUR._0x1c (local alias: PackedParams; derived alias: CK24)` instead of presenting `PackedParams` or `CK24` as if they came from the original documentation

## Mar 29, 2026 - Shared CK24 PM4 Forensics Landed In `wow-viewer`

- `wow-viewer` now has a research-only shared CK24 forensic export path in `Core.PM4` instead of leaving richer PM4 graph evidence trapped in `MdxViewer` JSON only.
- Landed pieces in this slice:
	- `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4ForensicsModels.cs` now carries shared CK24 forensic report contracts for per-component link groups, raw MSLK rows, raw linked MPRL rows, footprint counts, and placement comparison.
	- `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4Ck24ForensicsAnalyzer.cs` now builds component-level CK24 reports using the same current MSLK surface-link semantics as the viewer-side PM4 graph export, while keeping the report labeled as research-only.
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `pm4 export-json --ck24 <decimal|0xHEX>` so the shared inspect surface can emit either the coarse single-file PM4 report or a targeted CK24 forensic JSON without adding another tool-local parser path.
	- PM4 export JSON in inspect now enables field serialization so `System.Numerics` vectors serialize as real coordinates instead of empty objects.
- Validation completed on Mar 29, 2026:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed after the new analyzer and inspect wiring landed.
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter Pm4ResearchIntegrationTests` passed with new real-data CK24 forensic assertions.
	- `pm4 export-json --input .../development_00_00.pm4 --ck24 0x412CDC --output .../pm4_ck24_412CDC_forensics.json` wrote real shared CK24 forensic JSON against the fixed development dataset.
- Important boundary:
	- this is still research or export evidence only; it does not prove final PM4 object semantics or runtime viewer closure.

## Mar 28, 2026 - Default Post-MDX Continuation Target

- With classic `MDX` expansion paused, the default next `wow-viewer` implementation track should move to `Core.PM4` library completion rather than another `MDX` seam.
- Reason:
	- the PM4 continuity plan already names direct library completion as the clean next slice
	- `Core.PM4` is the most substantial real library area in `wow-viewer` today
	- this avoids drifting back into speculative `MDX` ownership or vague non-PM4 planning without a concrete target
- Secondary fallback only if PM4 is not the task:
	- continue non-`MDX` shared-I/O work on ADT/WDT/WMO only when the slice is narrow, tool-thin, and backed by a concrete validation target

## Mar 28, 2026 - MDX Audit: Separate Real Legacy Parity From New Shared Readers

- Audit result: recent `wow-viewer` classic `MDX` work is mixed.
- Grounded parity path:
	- `GEOS` shared summary/payload work is aligned with real legacy `MdxFile` parsing and current `MdxViewer` metadata/probe consumption.
- Not direct classic-parser parity:
	- `TXAN` payload reader in `wow-viewer` is not a direct port of active classic `MdxViewer` parsing; legacy `MdxFile` carries `TransformId` and the renderer can consume texture animations, but the classic `MdxFile` parser does not currently read `TXAN` into `TextureAnimations`
	- `HTST` payload reader in `wow-viewer` currently has no matching active classic `MdxViewer` parser/runtime implementation and should be treated as a new shared-reader seam, not viewer parity
	- `CLID` payload reader in `wow-viewer` also exceeds active classic `MdxViewer` behavior; legacy `MdxFile` skips `CLID`, while active `MdxViewer` only consumes shared collision summary metadata for model-info/probe surfaces
- If `MDX` work is ever resumed, the hotter missed legacy seam is not another cold chunk family. The real parity gap is the already-used classic `ATSQ`/geoset-animation and material-animation behavior that the active renderer consumes but `wow-viewer` still exposes mainly as summary-level metadata.

## Mar 28, 2026 - MDX Chunk Expansion Paused By User Direction

- Do not continue speculative `wow-viewer` `MDX` chunk-summary or payload implementation work by default.
- The user explicitly does not want further `MDX` chunk chasing just because chunks exist in archive data, especially when those seams were not implemented in `MdxViewer` already.
- Treat current `MDX` work as paused unless a future task explicitly asks for:
	- a specific `MdxViewer` compatibility need
	- a concrete consumer requirement already proven necessary in the active viewer/tool path
	- or a narrowly named `MDX` seam the user directly requests
- Default continuation should move back to non-`MDX` priorities instead of using `PREM` or `CORN` or any other remaining `MDX` family as the next automatic slice.

## Mar 29, 2026 - Shared Classic `MDX` `TXAN` Payload Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past unresolved classic `TXAN` chunk discovery into first shared texture-animation payload ownership for actual `KTAT` or `KTAR` or `KTAS` transform keyframes.
- Shared boundary and tool updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx` now also contains typed `MdxTextureAnimationFile` and `MdxTextureAnimation` payload contracts for classic indexed texture-animation entries
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxTextureAnimationReader.cs` now reads classic `TXAN` payloads for `v1300` and `v1400`, including counted sections and actual translation or rotation or scaling keyframe payloads
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxTrackReader.cs` now centralizes shared vector3 and compressed-quaternion keyframe parsing so `HTST` node-track and `TXAN` texture-track readers use one track interpretation
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `mdx export-json --include-texture-animations`, making the new shared `TXAN` payload seam exportable without adding another tool-local parser
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxTextureAnimationReaderTests.cs` now covers a synthetic tracked `TXAN` fixture, a real Alpha negative carrier, and a fixed real standard-era positive carrier on `creature/airelemental/airelemental.mdx`
- Validation completed on Mar 29, 2026:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxTextureAnimationReaderTests` passed after the new payload seam landed
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug` passed after the new export option landed
	- `mdx export-json --archive-root ... --virtual-path creature/airelemental/airelemental.mdx --include-texture-animations --output .../mdx-airelemental-texture-animations.json` wrote real standard-era shared texture-animation payload JSON through the archive path
- Important boundary:
	- this is still payload ownership and export only; it does not add runtime texture-transform evaluation, material playback in the renderer, or `MdxViewer` cutover
	- this is also not the recommended continuation path anymore; further `MDX` chunk expansion is paused unless the user explicitly reopens it

## Mar 29, 2026 - Shared Classic `MDX` `HTST` Payload Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `HTST` summary ownership into first shared hit-test payload ownership for fixed shape fields plus actual `KGTR` or `KGRT` or `KGSC` node-track keyframes.
- Shared boundary and tool updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx` now also contains typed `MdxHitTestFile` and `MdxHitTestShape` payload contracts plus reusable node-track payload contracts for vector3 or compressed-quaternion keyframes and interpolation metadata
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxHitTestReader.cs` now reads classic `HTST` payloads for `v1300` and `v1400`, including fixed box or cylinder or sphere or plane payloads plus actual transform keyframe payloads instead of summary-only counts or time ranges
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `mdx export-json --include-hit-test`, making the new shared `HTST` payload seam exportable without adding another tool-local parser
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxHitTestReaderTests.cs` now covers a synthetic tracked `HTST` fixture plus fixed real Alpha and standard-era hit-test carriers
- Validation completed on Mar 29, 2026:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxHitTestReaderTests` passed after the new payload seam landed
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug` passed after the new export option landed
	- `mdx export-json --archive-root ... --virtual-path creature/anubisath/anubisath.mdx --include-hit-test --output .../mdx-anubisath-hit-test.json` wrote real standard-era shared hit-test payload JSON through the archive path
- Important boundary:
	- this is still payload ownership and export only; it does not add runtime hit detection, animated transform evaluation in the renderer, or `MdxViewer` cutover

## Mar 28, 2026 - Shared Classic `MDX` `CLID` Payload Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `CLID` summary ownership into first shared collision-mesh payload ownership for ordered `VRTX` or `TRI ` or `NRMS` geometry.
- Shared boundary and tool updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxCollisionFile.cs` and `MdxCollisionMesh.cs` now carry the shared top-level classic `MDX` collision payload contract and typed mesh payload ownership
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxCollisionReader.cs` now reads classic `CLID` payloads for `v1300` and `v1400`, while `MdxSummaryReader` now reuses the same shared `MdxCollisionChunkReader` helper instead of maintaining a second independent `CLID` interpretation
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `mdx export-json --include-collision`, making the new shared collision payload seam exportable without adding another tool-local parser
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxCollisionReaderTests.cs` now covers a synthetic classic `CLID` payload fixture plus fixed real Alpha and standard-era collision carriers
- Validation completed on Mar 28, 2026:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "MdxCollisionReaderTests|MdxSummaryReaderTests"` passed after the new payload seam landed
	- `mdx export-json --archive-root ... --virtual-path character/dwarf/female/dwarffemale.mdx --include-collision --output .../mdx-dwarffemale-collision.json` wrote real standard-era shared collision payload JSON through the archive path
- Important boundary:
	- this is still payload ownership and export only; it does not add collision queries, runtime physics, or `MdxViewer` collision rendering cutover

## Mar 28, 2026 - `WowViewer.Tool.Inspect` `mdx export-json` Slice Landed In `wow-viewer`

- `wow-viewer` now has a first reusable JSON export surface for the shared classic `MDX` summary seam, with optional inclusion of the current shared `GEOS` payload seam.
- Shared boundary and tool updates in this slice:
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `mdx export-json` for filesystem or archive-backed inputs, with optional `--output <report.json>`
	- `mdx export-json --include-geometry` now also includes the current shared `MdxGeometryReader` output, so the first shared `GEOS` payload seam is exportable without adding a second tool-local parser
	- the command stays a thin consumer of `WowViewer.Core.IO.Mdx.MdxSummaryReader` and `MdxGeometryReader`; it does not move ownership out of the shared readers
- Validation completed on Mar 28, 2026:
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug` passed after the command landed
	- `mdx export-json --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx --output .../mdx-wisp-summary.json` wrote real Alpha summary JSON
	- `mdx export-json --archive-root ... --virtual-path world/generic/activedoodads/chest01/chest01.mdx --include-geometry --output .../mdx-chest-geometry.json` wrote real standard-era summary-plus-geometry JSON through the shared archive path
- Important boundary:
	- this is export of the current shared summary and `GEOS` payload seams only; it does not add new `MDX` chunk-family ownership or runtime render behavior
	- unresolved chunk families like `PREM` and `CORN` remain out of scope for now

## Mar 28, 2026 - `WowViewer.Tool.Inspect` `mdx chunk-carriers` Workflow Landed In `wow-viewer`

- `wow-viewer` now has a repeatable carrier-discovery workflow for classic `MDX` chunk continuation instead of relying on ad hoc archive probing or filename guesses.
- Shared boundary and tool updates in this slice:
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `mdx chunk-carriers --chunks <FOURCC[,FOURCC...]>` against either a filesystem file or directory or an archive-backed standard dataset with `--archive-root` and optional `--listfile`
	- the command stays a thin consumer of shared `WowViewer.Core.IO.Mdx.MdxSummaryReader`; it does not add tool-local `MDX` parsing or alternate chunk heuristics
	- the command also supports `--path-filter <text>` and `--limit <n>` so archive-backed scans can stay narrow and data-backed instead of brute-forcing the whole listfile blindly
- Validation completed on Mar 28, 2026:
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug` passed after the new command landed
	- `mdx chunk-carriers --chunks LITE --archive-root ... --path-filter braziers --limit 100` found `4` real standard-era `LITE` carriers, including the fixed `dwarvenbrazier01.mdx` validation surface
	- `mdx chunk-carriers --chunks TXAN,PREM,CORN --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree --limit 500` scanned the current unpacked alpha corpus (`229` MDX files) and found no carriers for those chunk ids
- Important boundary:
	- this slice adds a workflow and discovery surface, not new shared `MDX` chunk ownership by itself
	- current next-seam status remains: the bundled alpha corpus still has no fixed `TXAN` or `PREM` or `CORN` carrier, so the next classic `MDX` reader slice should still start from a real carrier search rather than from assumed file names

## Mar 28, 2026 - Viewer UI Resize And Hit-Testing Regression Fixed In `MdxViewer`

- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now explicitly resyncs the Silk `ImGuiController` logical window size so the active viewer shell no longer drifts into broken panel sizing and unusable button hit-testing after resize or maximize.
- Verification completed on Mar 28, 2026:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after the fix
	- a short viewer startup smoke launched cleanly after the patch
	- the user manually retested the UI and reported that it now seems to be working
- Important boundary:
	- this is still manual runtime signoff only for the shell regression; no automated UI regression coverage exists yet
	- the current implementation reflects a private Silk `ImGuiController.WindowResized(Vector2D<int>)` method, so future package upgrades should treat that integration point as fragile and re-check resize behavior first

## Mar 28, 2026 - Shared Classic `MDX` `GEOS` Payload Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `GEOS` summary ownership into first shared classic geoset payload ownership for render-facing vertex or normal or UV or index or skin-table data.
- Shared boundary updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxGeometryFile.cs` now carries the shared top-level classic `MDX` geometry-file contract for payload-level `GEOS` reads
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxGeosetGeometry.cs` now carries shared per-geoset payload ownership for vertices, normals, UV sets, primitive types, face groups, indices, vertex groups, matrix tables, bone tables, and footer metadata
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxGeometryReader.cs` now reads classic counted `GEOS` payloads for `v1300` and `v1400`, including direct `UVAS` sets used in Alpha-era files plus optional explicit `UVBS` data when present
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxGeometryReaderTests.cs` now covers a synthetic classic `GEOS` payload fixture, a fixed real standard-era archive-backed positive carrier, and a real on-disk alpha-era positive carrier from the existing `0.5.3` corpus
- Validation completed on Mar 28, 2026:
	- focused `WowViewer.Core.Tests` geometry and summary reader coverage passed after the new synthetic and real `GEOS` payload tests landed
	- the real standard-era path now uses a fixed archive-backed positive `GEOS` carrier, preferring `Creature/AncientOfWar/AncientofWar.mdx` when present and falling back to the existing `chest01.mdx` validation surface otherwise
	- the real alpha-era path now uses the existing unpacked `0.5.3` creature corpus as the positive payload surface; the user-provided `AncientofWar.mdx` attachment remains a good future fixed carrier once it is committed into `wow-viewer/testdata/0.5.3/tree`
- Scope guardrail:
	- this is still classic `GEOS` payload ownership only; it does not yet build runtime render buffers, bind skeleton state, evaluate geoset animation visibility, or replace `MdxViewer` model loading

## Mar 28, 2026 - Shared Classic `MDX` `LITE` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `GLBS` summary ownership into first shared classic `LITE` light-summary ownership for counted `MDLGENOBJECT`-derived light metadata.
- Shared boundary updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxLightType.cs` now carries the shared classic light kind enum for `Omni`, `Direct`, and `Ambient`
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxLightSummary.cs` now carries shared per-light identity, hierarchy, static attenuation or color or intensity metadata, and summary-only `KLAS`, `KLAE`, `KLAC`, `KLAI`, `KLBC`, `KLBI`, and `KVIS` track metadata
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now exposes `Lights` and `LightCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `LITE` entries for `v1300` and `v1400`, including inherited node metadata plus fixed light payload fields and optional summary-only light-track metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports `lights=` in the header and prints `LITE[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `LITE` fixture plus a fixed real archive-backed `0.6.0` `dwarvenbrazier01.mdx` light regression
- Validation completed on Mar 28, 2026:
	- focused `WowViewer.Core.Tests` `MdxSummaryReaderTests` passed after the new synthetic and real archive-backed `LITE` coverage landed
	- the focused MDX reader suite now also includes a real unpacked `0.5.3` alpha-corpus smoke over `229` MDX files under `wow-viewer/testdata/0.5.3/tree`, proving the new `LITE` summary path does not break current alpha-era parsing and that the bundled `0.5.3` sample set contains no `LITE` chunks today
	- `WowViewer.Tool.Inspect mdx inspect` on `0.6.0` `world/generic/dwarf/passive doodads/braziers/dwarvenbrazier01.mdx` now reports `lights=1`, `CHUNK[7]: id=LITE`, and stable `LITE[0]` light metadata including `Omni02`, static attenuation `0.8333333 -> 0.9722222`, and a `KLAI(keys=26 ... time=[0, 3333])` intensity track
- Scope guardrail:
	- this is still classic `LITE` summary ownership only; it does not evaluate runtime lighting, animation-driven intensity/color playback, or viewer render-light parity

## Mar 28, 2026 - Shared Classic `MDX` `GLBS` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `CLID` summary ownership into first shared classic `GLBS` global-sequence summary ownership for strict counted `uint32` duration tables.
- Shared boundary updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxGlobalSequenceSummary.cs` now carries shared per-index global-sequence duration metadata
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now exposes `GlobalSequences` and `GlobalSequenceCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads `GLBS` as a strict `uint32` table and rejects payload sizes that are not divisible by `4`
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports `globalSequences=` in the header and prints `GLBS[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic `GLBS` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` global-sequence regression
- Validation completed on Mar 28, 2026:
	- focused `WowViewer.Core.Tests` `MdxSummaryReaderTests` passed after the new synthetic and real Alpha `GLBS` coverage landed
	- `WowViewer.Tool.Inspect mdx inspect` on Alpha `0.5.3` `Wisp.mdx` now reports `globalSequences=11`, `CHUNK[3]: id=GLBS`, and stable `GLBS[0..10]` durations `267,133,533,0,567,900,1167,667,467,933,300`
- Scope guardrail:
	- this is still classic `GLBS` summary ownership only; it does not evaluate track playback, resolve `globalSeqId` references into runtime animation state, or claim full animation-system ownership

## Mar 28, 2026 - Shared Classic `MDX` `CLID` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `HTST` summary ownership into first shared classic `CLID` collision-summary ownership for ordered `VRTX` or `TRI ` or `NRMS` collision-mesh metadata.
- Shared boundary updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxCollisionSummary.cs` now carries shared collision counts, max-index coverage, and derived collision bounds
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now exposes nullable `Collision` and `HasCollision`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic `CLID` chunks for `v1300` and `v1400`, including ordered `VRTX` or `TRI ` or `NRMS` subchunks, derived collision bounds, and index coverage
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports `collisionVertices=` and `collisionTriangles=` in the header and prints a `CLID:` line during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `CLID` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` collision regression
- Validation completed on Mar 28, 2026:
	- focused `WowViewer.Core.Tests` `MdxSummaryReaderTests` passed after the new synthetic and real Alpha `CLID` coverage landed
	- `WowViewer.Tool.Inspect mdx inspect` on Alpha `0.5.3` `Wisp.mdx` now reports `collisionVertices=8`, `collisionTriangles=12`, `CHUNK[17]: id=CLID`, and stable `CLID: vertices=8 triIndices=36 triangles=12 facetNormals=12 maxIndex=7 ...`
- Scope guardrail:
	- this is still classic `CLID` summary ownership only; it does not expose full collision geometry payloads, collision queries, export surfaces, or runtime physics behavior

## Mar 28, 2026 - Shared Classic `MDX` `HTST` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `EVTS` summary ownership into first shared classic `HTST` hit-test-shape summary ownership for counted `MDLGENOBJECT` hit-test nodes and fixed box or cylinder or sphere or plane payload metadata.
- Shared boundary updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx` now carries shared `MdxGeometryShapeType` and `MdxHitTestShapeSummary` contracts for classic hit-test shapes beside the earlier event, camera, ribbon, particle, attachment, helper, and bone seams
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now exposes `HitTestShapes` and `HitTestShapeCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `HTST` entries for `v1300` and `v1400`, including inherited node metadata plus fixed `SHAPE_BOX` or `SHAPE_CYLINDER` or `SHAPE_SPHERE` or `SHAPE_PLANE` payload fields
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports `hitTestShapes=` in the header and prints `HTST[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `HTST` fixture across box or cylinder or sphere or plane shapes plus a fixed real Alpha `0.5.3` `Wisp.mdx` sphere regression
- Validation completed on Mar 28, 2026:
	- focused `WowViewer.Core.Tests` `MdxSummaryReaderTests` passed after the new synthetic and real Alpha `HTST` coverage landed
	- `WowViewer.Tool.Inspect mdx inspect` on Alpha `0.5.3` `Wisp.mdx` now reports `hitTestShapes=1`, `CHUNK[16]: id=HTST`, and stable `HTST[0]: name=HIT01 ... shapeType=Sphere(2) shape=center=(0.366, 0.009, 1.890) radius=0.833333`
- Scope guardrail:
	- this is still classic `HTST` summary ownership only; it does not evaluate runtime collision or hit detection, animation-driven shape transforms, or viewer physics behavior

## Mar 28, 2026 - Shared Classic `MDX` `EVTS` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `CAMS` summary ownership into first shared classic `EVTS` event-summary ownership for counted `MDLGENOBJECT` event nodes and optional summary-only `KEVT` time-track metadata.
- Shared boundary updates in this slice:
	- `wow-viewer/src/core/WowViewer.Core/Mdx` now carries shared `MdxEventSummary` and `MdxEventTrackSummary` contracts so classic event nodes live beside the existing summary-only bone, helper, attachment, ribbon, camera, and particle seams
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now exposes `Events` and `EventCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `EVTS` entries for `v1300` and `v1400`, including per-section sizing, inherited node metadata, and optional `KEVT` key-time metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports `events=` in the header and prints `EVTS[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `EVTS` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` event regression
- Validation completed on Mar 28, 2026:
	- focused `WowViewer.Core.Tests` `MdxSummaryReaderTests` passed after the new synthetic and real Alpha `EVTS` coverage landed
	- `WowViewer.Tool.Inspect mdx inspect` on Alpha `0.5.3` `Wisp.mdx` now reports `events=3`, `CHUNK[15]: id=EVTS`, and stable `EVTS[0..2]` node metadata with only the final `$DTH` event carrying `KEVT(keys=1 globalSeqId=-1 time=[1667, 1667])`
- Scope guardrail:
	- this is still classic `EVTS` summary ownership only; it does not evaluate event playback semantics, event lookup tables, particle or sound dispatch, or runtime trigger behavior

## Mar 28, 2026 - Shared Classic `MDX` `CAMS` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `RIBB` summary ownership into first shared classic `CAMS` camera-summary ownership for fixed camera metadata and summary-only camera-track metadata.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxCameraSummary.cs` now owns shared per-camera identity, pivot or target-pivot data, fixed clip values, and optional summary-only `KCTR` or `KCRL` or `KVIS` or `KTTR` metadata
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `Cameras` and `CameraCount` alongside the earlier `Ribbons` seam
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `CAMS` entries for `v1300` and `v1400`, including per-camera section sizing, fixed camera payload fields, and optional summary-only `KCTR` or `KCRL` or `KVIS` or `KTTR` metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `CAMS[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `CAMS` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` camera regression
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `24` passing tests after the new synthetic and real Alpha `CAMS` coverage landed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` passed on Mar 28, 2026 and reported `cameras=1`, `CHUNK[14]: id=CAMS`, and stable `CAMS[0]: name=Portrait ... positionTrack=none ... targetPositionTrack=none`
- Important boundary:
	- this is still classic `CAMS` summary ownership only; it does not evaluate camera playback, target interpolation, render-camera selection, or runtime portrait behavior
	- it does not replace `MdxViewer` camera handling or claim Alpha runtime camera parity

## Mar 28, 2026 - Shared Classic `MDX` `PRE2` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `RIBB` summary ownership into first shared classic `PRE2` particle-emitter summary ownership for `MDLGENOBJECT`-derived effect metadata.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxParticleEmitter2Summary.cs` now owns shared per-emitter identity, hierarchy, flags, classic scalar particle fields, color or alpha or scale signals, optional model-path presence, spline-count metadata, and summary-only track metadata
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxTrackSummary.cs` now owns shared summary-only metadata for classic non-node track families across both `PRE2` and `RIBB`, including `KP2S`, `KP2E`, `KP2L`, `KPLN`, `KP2G`, `KLIF`, `KP2W`, `KP2N`, `KP2Z`, `KRHA`, `KRHB`, `KRAL`, `KRCO`, and `KRTX`
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `ParticleEmitters2` and `ParticleEmitter2Count` alongside the earlier `Ribbons` seam
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `PRE2` entries for `v1300` and `v1400`, including outer emitter sizing, inner `MDLGENOBJECT` node sizing, classic scalar payload fields, spline-block sizing, and summary-only `KVIS` or `KP2V` plus `KP2S`, `KP2R`, `KP2L`, `KPLN`, `KP2G`, `KLIF`, `KP2E`, `KP2W`, `KP2N`, and `KP2Z` metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `PRE2[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `PRE2` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` particle-emitter regression
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `22` passing tests after the new synthetic and real Alpha `PRE2` coverage landed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` passed on Mar 28, 2026 and reported `particleEmitters2=11`, `CHUNK[12]: id=PRE2`, stable `PRE2[0]: name=BlizParticle01 ... visibilityTrack=KVIS(...)`, and `PRE2[5]: name=BlizParticleBlackDeath ... speedTrack=KP2S(...) emissionRateTrack=KP2E(...)`
- Important boundary:
	- this is still classic `PRE2` summary ownership only; it does not evaluate particle spawn/update behavior, billboarding, UV animation playback, spline motion, or runtime render parity
	- it does not replace `MdxViewer` particle handling or claim Alpha runtime playback parity

## Mar 28, 2026 - Shared Classic `MDX` `ATCH` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `HELP` summary ownership into first shared attachment summary ownership for classic `MDLGENOBJECT`-derived attachment metadata.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxAttachmentSummary.cs` now owns shared per-attachment identity, hierarchy, flags, attachment-id, optional path, and transform-track metadata contracts
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxVisibilityTrackSummary.cs` now owns the shared classic attachment-visibility track metadata contract for `KVIS` or `KATV`
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `Attachments` and `AttachmentCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `ATCH` entries for `v1300` and `v1400`, including outer attachment-section sizing, inner `MDLGENOBJECT` node sizing, summary-only `KGTR` or `KGRT` or `KGSC` transform metadata, attachment-id/path fields, and optional `KVIS` or `KATV` visibility metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `ATCH[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `ATCH` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` attachment regression
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `17` passing tests after the new synthetic and real Alpha `ATCH` coverage landed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` passed on Mar 28, 2026 and reported `attachments=9`, `CHUNK[10]: id=ATCH`, and real `ATCH[0]` through `ATCH[8]` lines
- Important boundary:
	- this is still classic `ATCH` summary ownership only; it does not resolve attachment paths into assets, evaluate visibility values, or claim attachment-driven runtime render parity
	- it does not replace `MdxViewer` attachment handling or claim Alpha runtime playback parity

## Mar 28, 2026 - Shared Classic `MDX` `HELP` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `BONE` summary ownership into first shared helper-node summary ownership for classic `MDLGENOBJECT` metadata.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxHelperSummary.cs` now owns shared per-helper identity, hierarchy, flag, and transform-track metadata contracts
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxNodeTrackSummary.cs` now owns the generalized shared node-track contract reused by classic `BONE` and `HELP`
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `Helpers` and `HelperCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `HELP` entries for `v1300` and `v1400`, including `MDLGENOBJECT` name or object-id or parent-id or flag fields plus summary-only `KGTR` or `KGRT` or `KGSC` key-count, interpolation, global-sequence, and time-range metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `HELP[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `HELP` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` helper regression
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `16` passing tests after the new synthetic and real Alpha `HELP` coverage landed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` passed on Mar 28, 2026 and reported `helpers=9`, `CHUNK[9]: id=HELP`, and real `HELP[0]` through `HELP[8]` lines
- Important boundary:
	- this is still classic `HELP` summary ownership only; it does not evaluate node transforms, helper-driven billboards, attachment behavior, or runtime animation playback parity
	- it does not replace `MdxViewer` helper-node handling or claim Alpha runtime playback parity

## Mar 28, 2026 - Shared Classic `MDX` `BONE` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `GEOA` summary ownership into first classic bone/node summary ownership for render-facing skeleton metadata.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxBoneSummary.cs` and `MdxNodeTrackSummary.cs` now own shared per-bone identity, hierarchy, flag, geoset-link, and transform-track metadata contracts
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `Bones` and `BoneCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `BONE` entries for `v1300` and `v1400`, including `MDLGENOBJECT` name or object-id or parent-id or flag fields plus summary-only `KGTR` or `KGRT` or `KGSC` key-count, interpolation, global-sequence, and time-range metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `BONE[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `BONE` fixture plus a fixed real Alpha `0.5.3` `Wisp.mdx` bone regression
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `14` passing tests after the new synthetic and real Alpha `BONE` coverage landed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` passed on Mar 28, 2026 and reported `bones=16`, `CHUNK[8]: id=BONE`, and real `BONE[0]` through `BONE[15]` lines
- Important boundary:
	- this is still classic `BONE` summary ownership only; it does not evaluate node transforms, bind pivots to runtime skeleton state, or claim animation playback parity
	- it does not replace `MdxViewer` model skeleton handling or claim Alpha runtime playback parity

## Mar 28, 2026 - Shared Classic `MDX` `GEOA` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past classic `GEOS` structure ownership into first classic geoset-animation summary ownership for render-facing animation metadata.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxGeosetAnimationSummary.cs` and `MdxGeosetAnimationTrackSummary.cs` now own shared per-entry static color or alpha fields plus `KGAO` or `KGAC` track metadata contracts
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `GeosetAnimations` and `GeosetAnimationCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted `GEOA` entries for `v1300` and `v1400`, including static header fields and summary-only `KGAO` or `KGAC` key-count, interpolation, global-sequence, and time-range metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `GEOA[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic `GEOA` fixture plus an optional real archive-backed `GEOA` probe path across fixed `0.6.0` effect assets
- Current verified validation for this landing:
	- synthetic and real-data `MdxSummaryReaderTests` coverage now includes `GEOA`
	- real Alpha `0.5.3` `MDX` files under `wow-viewer/testdata/0.5.3/tree`, such as `Creature/Wisp/Wisp.mdx`, do carry positive `GEOA` data and are the correct fixed validation surface for this seam
	- the fixed `0.6.0` archive probe set was widened across smoke or torch or brazier or vent assets, but no guaranteed positive `GEOA` carrier was found there
- Important boundary:
	- this is still classic `GEOA` summary ownership only; it does not evaluate animated values, build runtime geoset-visibility state, or claim viewer playback parity
	- it does not replace `MdxViewer` model animation handling or claim Alpha runtime playback parity

## Mar 28, 2026 - Shared Classic `MDX` `GEOS` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past `SEQS` plus `PIVT` internal `MDX` summary ownership: the shared `MDX` seam now also exposes first classic geoset coverage for render-facing mesh structure.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxGeosetSummary.cs` now owns shared per-geoset summary contracts for core render-facing counts, material linkage, selection or flag fields, optional bounds, and animation-extent count
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `Geosets` and `GeosetCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads classic counted tagged `GEOS` entries for `v1300` and `v1400`, including vertex, normal, UV, index, matrix, and bone-table counts plus material or bounds summary fields
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `GEOS[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers a synthetic classic geoset fixture and a real archive-backed `chest01.mdx` geoset case with exact fixed-asset signals
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `8` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- mdx inspect --archive-root "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt" --virtual-path "world/generic/activedoodads/chest01/chest01.mdx"` passed on Mar 28, 2026 and reported `geosets=2`, `CHUNK[5]: id=GEOS`, and real `GEOS[0]` plus `GEOS[1]` lines with stable counts
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `174` passing tests
- Important boundary:
	- this is still classic `GEOS` summary ownership only; it does not yet decode full mesh payloads, skinning semantics, geoset animations, or runtime render buffers
	- it does not replace `MdxViewer` model loading or claim runtime viewer mesh or skeleton parity

## Mar 28, 2026 - Shared `MDX` `PIVT` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past `SEQS`-only deeper `MDX` summary ownership: the shared `MDX` seam now also exposes first pivot-table coverage.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxPivotPointSummary.cs` now owns shared per-pivot summary contracts for pivot index and pivot position
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `PivotPoints` and `PivotPointCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads `PIVT` as strict `12`-byte `Vector3` entries and preserves the legacy hard-fail behavior for invalid `PIVT` payload sizes
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `PIVT[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers synthetic pivot tables, keeps the real `chest01.mdx` archive-backed summary case, and adds an optional real pivot-positive probe path
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `6` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -c Debug -- mdx inspect --archive-root "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt" --virtual-path "world/generic/activedoodads/chest01/chest01.mdx"` passed on Mar 28, 2026 and reported `pivotPoints=6`, `CHUNK[8]: id=PIVT`, and real `PIVT[0]` through `PIVT[5]` lines
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `172` passing tests
- Important boundary:
	- this is still `MDX` pivot-table summary ownership only; it does not yet bind pivots onto bones, helpers, emitters, or runtime node transforms
	- it does not replace `MdxViewer` model loading or claim runtime viewer animation or skeleton parity

## Mar 28, 2026 - Shared `MDX` `SEQS` Summary Slice Landed In `wow-viewer`

- `wow-viewer` has moved one step past `TEXS`-plus-`MTLS`-only `MDX` summary ownership: the shared `MDX` seam now also exposes first sequence/animation-summary coverage.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSequenceSummary.cs` now owns shared per-sequence summary contracts for sequence name, time range, move speed, flags, frequency, replay range, optional blend time, and optional bounds
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSummary.cs` now carries `Sequences` and `SequenceCount`
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads observed `SEQS` summary variants, including counted legacy named `128/132/136/140`-byte records, counted named `0x8C` records, and the numeric-heavy `0x8C` `0.9.0` path as summary-only sequence metadata
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `SEQS[n]` lines during `mdx inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers synthetic legacy-sequence and counted-named-`0x8C` sequence cases, keeps the real `chest01.mdx` archive-backed summary case, and adds an optional real animated-asset probe path
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `4` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --archive-root "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" --virtual-path world/generic/passivedoodads/particleemitters/greengroundfog.mdx --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 28, 2026 and reported `model=GreenGroundFog`, `sequences=1`, `CHUNK[2]: id=SEQS`, and a real `SEQS[0]: name=Stand ... blendTime=150 ...` line
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `170` passing tests
- Important boundary:
	- this is still `MDX` sequence summary ownership only; it does not yet parse animation tracks, bones, pivot tables, geosets, or runtime playback semantics
	- it does not replace `MdxViewer` model loading or claim runtime viewer animation parity

## Mar 28, 2026 - Shared Root-ADT Plus `_tex0` Texture Reader And Broadened JSON Export Landed In `wow-viewer`

- The terrain texture-detail seam is no longer `_tex0`-only. `wow-viewer` now has one shared ADT texture reader for root `ADT` and `_tex0.adt` files.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Maps/` now owns shared `AdtTextureChunkLayer`, `AdtTextureChunk`, and `AdtTextureFile` instead of the earlier `_tex0`-only contract names
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtTextureReader.cs` now reads both root `ADT` and `_tex0.adt` files, carrying file kind, decode profile, chunk coordinates, `DoNotFixAlphaMap`, per-layer table data, and decoded overlay alpha payloads through one shared seam
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtMcalSummaryReader.cs` now aggregates both root and `_tex0` `MCAL` signals through `AdtTextureReader` instead of keeping separate root parsing logic
	- `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs` still uses `export-tex-json`, but it now accepts `--input <file.adt|file_tex0.adt>` and emits readable enum strings in JSON output
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now also routes `map inspect --dump-tex-chunks` through the generalized shared reader for root `ADT` and `_tex0.adt`
	- `wow-viewer/tests/WowViewer.Core.Tests/AdtTextureReaderTests.cs` now covers synthetic root and synthetic `_tex0` layer reads plus real root and real `_tex0` development-dataset reads
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AdtTextureReaderTests|AdtMcalSummaryReaderTests|AdtMcalDecoderTests|AdtSummaryReaderTests|AdtMcnkSummaryReaderTests|MapFileSummaryReaderTests|WowFileDetectorTests"` passed on Mar 28, 2026 with `37` targeted passing tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `168` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- export-tex-json --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0.adt | Select-Object -First 20` passed on Mar 28, 2026 and reported root JSON beginning with `Kind: Adt`, `DecodeProfile: LichKingStrict`, empty `TextureNames`, and `Chunks`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- export-tex-json --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt | Select-Object -First 20` passed on Mar 28, 2026 and reported `_tex0` JSON beginning with `Kind: AdtTex`, `DecodeProfile: Cataclysm400`, populated `TextureNames`, and `Chunks`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- export-tex-json --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0.adt --output $env:TEMP\wowviewer-development_0_0-root.json` passed on Mar 28, 2026 and wrote the expected root JSON file
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0.adt --dump-tex-chunks | Select-Object -First 12` passed on Mar 28, 2026 and reported `ADT texture detail: kind=Adt profile=LichKingStrict textures=0 chunks=256`
- Important boundary:
	- this is a shared root-plus-`_tex0` texture-detail read and export seam only
	- the fixed development root dataset currently proves the root command path and chunk metadata path, but not positive real root-layer payload decode because its texture layers live in `_tex0.adt`
	- it still does not port Cataclysm residual-alpha synthesis or neighbor-edge stitching as first-class shared terrain services

## Mar 28, 2026 - Thin `_tex0` JSON Export Surface Landed In `WowViewer.Tool.Converter`

- The new shared `_tex0` reader is no longer inspect-only; `wow-viewer` now has its first thin converter/export consumer for that seam.
- Landed pieces:
	- `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs` now accepts `export-tex-json --input <file_tex0.adt> [--output <report.json>]`
	- the converter validates file kind through shared `WowFileDetector` and serializes shared `AdtTexReader` output directly instead of owning a second `_tex0` parser or formatter
	- stdout export and file-write export both now run on the fixed development `_tex0` dataset
- Current verified validation for this landing:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `166` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- export-tex-json --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt | Select-Object -First 40` passed on Mar 28, 2026 and printed shared JSON rooted at `SourcePath`, `TextureNames`, and `Chunks`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- export-tex-json --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt --output $env:TEMP\wowviewer-development_0_0_tex0.json` passed on Mar 28, 2026 and wrote the expected JSON file
- Important boundary:
	- this is a thin export surface over the existing shared `_tex0` read seam
	- it does not yet convert terrain into another runtime format, merge root plus split ADT families, or provide a write path back to WoW terrain files

## Mar 28, 2026 - Shared `_tex0` Per-Chunk Layer And Decoded Alpha Reader Landed In `wow-viewer`

- The next terrain ownership slice after split-family routing plus aggregate `MCAL` summary is now landed in `wow-viewer` core/core.io.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Maps/` now also owns shared `AdtTexChunkLayer`, `AdtTexChunk`, and `AdtTexFile` contracts for `_tex0.adt` texture-name tables, per-`MCNK` layer tables, and decoded per-layer alpha payload exposure
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtTexReader.cs` now reads `_tex0.adt` files into typed per-chunk layer data and reuses `AdtMcalDecoder` for decoded overlay alpha ownership instead of leaving that detail trapped in inspect-only output or aggregate counters
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/MapSummaryReaderCommon.cs` now exposes reusable string-table extraction through `ReadStringEntries(...)`
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtMcalSummaryReader.cs` now aggregates `_tex0` `MCAL` signals through the new shared `AdtTexReader` instead of re-parsing `_tex0` chunk payloads locally
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now accepts `map inspect --input <file.adt> --dump-tex-chunks` and prints typed per-chunk `MCNK(tex)[n]` / `LAYER[n]` detail lines sourced from the shared reader
	- `wow-viewer/tests/WowViewer.Core.Tests/AdtTexReaderTests.cs` now locks both a synthetic `_tex0` fixture and the real `development_0_0_tex0.adt` dataset against the new shared reader seam
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AdtTexReaderTests|AdtMcalSummaryReaderTests|AdtMcalDecoderTests|AdtSummaryReaderTests|AdtMcnkSummaryReaderTests|MapFileSummaryReaderTests|WowFileDetectorTests"` passed on Mar 28, 2026 with `35` targeted passing tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `166` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt --dump-tex-chunks | Select-Object -First 20` passed on Mar 28, 2026 and reported:
		- `ADT TEX detail: textures=5 chunks=256`
		- `MCNK(tex)[0]: xy=(0,0) layers=1 alphaBytes=0 decodedLayers=0`
		- real decoded per-layer `Compressed` and `BigAlpha` outputs later in the dump while preserving the earlier aggregate `ADT MCAL semantics: ... decodedLayers=519 ... compressed=515 bigAlpha=4 ...`
- Important boundary:
	- this is deeper shared `_tex0` read ownership for typed per-chunk layer and direct alpha payload exposure
	- it still does not port the full Cataclysm `TerrainBlend` runtime behavior from `StandardTerrainAdapter`, especially residual-alpha synthesis and neighbor-edge stitching semantics as first-class shared-core services
	- it does not replace the active `MdxViewer` terrain runtime or claim full terrain visual parity by itself

## Mar 28, 2026 - Shared `ADT` Split-Family Routing And Direct `MCAL` Decode Summary Seams Landed In `wow-viewer`

- The first terrain-focused shared-I/O tranche under the full-format-ownership reset is now landed in `wow-viewer` core/core.io.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Maps/` now also owns shared `AdtTileFamily`, `AdtTextureLayerDescriptor`, `AdtMcalDecodeProfile`, `AdtMcalAlphaEncoding`, `AdtMcalDecodedLayer`, and `AdtMcalSummary`
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtTileFamilyResolver.cs` now resolves root / `_tex0` / `_obj0` / `_lod` companion paths from any local ADT-family input and exposes preferred texture and placement owners
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtMcalDecoder.cs` now owns a first shared direct-layer `MCAL` decode seam for LK strict and Cataclysm 4.0-style direct payload reads, including compressed alpha, packed 4-bit alpha, direct big-alpha, and the current fixed `63x63 -> 64x64` big-alpha expansion path
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtMcalSummaryReader.cs` now aggregates per-file `MCAL` decode signals across root `ADT` and `_tex0.adt` `MCNK` payloads
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `ADT family:` routing plus `ADT MCAL semantics:` lines during `map inspect`
	- `MapFileKind` and `MapFileSummaryReader` now carry `_lod.adt` through as `AdtLod` instead of dropping it back to `Unknown`
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AdtMcalDecoderTests|AdtMcalSummaryReaderTests|AdtTileFamilyResolverTests|AdtSummaryReaderTests|AdtMcnkSummaryReaderTests|MapFileSummaryReaderTests|WowFileDetectorTests"` passed on Mar 28, 2026 with `35` targeted passing tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `164` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 28, 2026 and reported:
		- `ADT family: root=present tex0=present obj0=present lod=missing textureSource=AdtTex placementSource=AdtObj`
		- `ADT MCAL semantics: profile=Cataclysm400 ... overlayLayers=519 decodedLayers=519 missingPayloadLayers=0 decodeFailures=0 compressed=515 bigAlpha=4 bigAlphaFixed=0 packed4=0`
- Important boundary:
	- this is direct split-family routing plus direct `MCAL` payload decode ownership only
	- it does not yet port the full Cataclysm `TerrainBlend` runtime behavior from `StandardTerrainAdapter`, especially residual-alpha synthesis and neighbor-chunk stitching semantics as first-class shared-core services
	- it does not replace the active `MdxViewer` terrain runtime or claim full terrain visual parity by itself

## Mar 28, 2026 - wow-viewer Full Format Ownership Reset

- User direction is now explicit: `wow-viewer` is expected to become the first-party owner of every active `MdxViewer` format family, fully, not just through detector or summary seams.
- The migration target is no longer "enough shared readers to inspect files". The target is full parse, decode, write, runtime-service, and tool ownership for the formats the active viewer currently handles.
- Current summary seams in `wow-viewer` remain valid, but they are now only stepping stones toward that larger ownership target.
- A dedicated program document now exists at `gillijimproject_refactor/plans/wow_viewer_full_format_ownership_plan_2026-03-28.md`.
- A family-by-family backlog now also exists at `gillijimproject_refactor/plans/wow_viewer_format_parity_matrix_2026-03-28.md`.
- Immediate high-risk ownership gaps called out by the reset:
	- `ADT` alpha decode and split-file routing parity
	- deep `WMO` ownership beyond current summaries
	- deep `MDX` ownership beyond top-level summary
	- first-party `M2` ownership instead of Warcraft.NET-only behavior
	- first-party `BLP` decode/write ownership instead of SereniaBLPLib-only behavior
	- continued `PM4` extraction until `WorldScene` is no longer the hidden owner of active semantics

## Mar 28, 2026 - Shared `MDX` Top-Level Plus `TEXS` And `MTLS` Summary Seams And `MdxViewer` Consumer Validation Landed

- `wow-viewer` now owns its first narrow `MDX` model-family seam instead of stopping model validation at cross-family detection.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/` now owns shared `MdxChunkIds`, `MdxChunkSummary`, `MdxTextureSummary`, `MdxMaterialLayerSummary`, `MdxMaterialSummary`, and `MdxSummary` contracts for top-level `MDX` header-summary work
	- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxSummaryReader.cs` now reads `MDLX` files through an `MDX`-specific top-level chunk path, including `VERS`, `MODL`, `TEXS`, `MTLS`, chunk order, known-vs-unknown chunk counts, model name, bounds, blend time, texture count, replaceable-texture count, material count, material-layer count, per-texture path/flag summary, and narrow per-material layer summary fields
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now accepts `mdx inspect --input <file.mdx>` and `mdx inspect --archive-root <dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>]`
	- `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` now covers both a synthetic `MDX` fixture and a real standard-archive `world/generic/activedoodads/chest01/chest01.mdx` read
	- `gillijimproject_refactor/src/MdxViewer/AssetProbe.cs` now also prints shared `MDX` summary output for probed model bytes, including `textures`, `replaceableTextures`, `materials`, `materialLayers`, the first shared `TEXS` paths, and compact first-layer `MTLS` signals alongside the earlier shared `BLP` texture-summary output
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "MdxSummaryReaderTests|WowFileDetectorTests"` passed on Mar 27, 2026 with `11` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --archive-root "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" --virtual-path world/generic/activedoodads/chest01/chest01.mdx --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 28, 2026 and reported `version=1300`, `model=Chest01`, `textures=2`, `materials=2`, and real `TEXS[...]` plus `MTLS[0].LAYER[0]` / `MTLS[1].LAYER[0]` lines for the chest asset
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 27, 2026 with existing warnings
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --probe-mdx "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" "world/generic/activedoodads/chest01/chest01.mdx" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 28, 2026 and now reports `SharedMDX: ... textures=2 replaceableTextures=0 materials=2 materialLayers=2 ... firstTextures=... firstMaterials=tex0/blend0/alpha1.000,tex1/blend0/alpha1.000`
- Important boundary:
	- this is shared `MDX` top-level plus narrow `TEXS` and `MTLS` summary ownership only; it does not replace `MdxFile.Load(...)`, animation-track parsing, `M2` handling, or any live viewer render-path model loading
	- real `MDX` chunk ids are stored as direct ASCII on disk, so this seam intentionally uses an `MDX`-specific header decode path instead of the generic reversed-FourCC chunk reader used by ADT/WDT/WMO files
	- this is still build plus inspect/probe validation, not runtime viewer signoff

## Mar 27, 2026 - `MdxViewer` Consumer Validation Now Exercises The Shared `BLP` Seam

- The first shared `BLP` seam in `wow-viewer` is no longer validated only inside `wow-viewer` tools and tests; the active viewer now consumes it through the existing non-UI probe path.
- Landed pieces:
	- `gillijimproject_refactor/src/MdxViewer/AssetProbe.cs` now runs shared `WowFileDetector` on the probed model bytes and shared `BlpSummaryReader` on resolved texture bytes classified as `Blp`
	- probe output now shows both shared format signals and the legacy decode-based alpha summary, which keeps the latest library seam visible from the compatibility consumer without forcing a render-path cutover
- Current verified validation for this compatibility step:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 27, 2026 with existing warnings
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --probe-mdx "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" "world/generic/activedoodads/chest01/chest01.mdx" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 27, 2026 and reported real per-texture `SharedBLP` lines for `CHEST1SIDE.BLP` and `CHEST1FRONT.BLP`
- Important boundary:
	- this is compile plus non-UI consumer validation only
	- it does not prove live viewer rendering parity or a full migration away from `SereniaBLPLib` texture decode

## Mar 27, 2026 - Shared `BLP` Header Summary Seam And Inspect Surface Landed

- The broader shared-I/O gap against `MdxViewer` moved past another WMO-only step: `wow-viewer` now owns a first real `BLP` seam instead of stopping at detector-level classification.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Blp/` now owns the shared `BLP` format, compression, pixel-format, mip-entry, and summary contracts
	- `wow-viewer/src/core/WowViewer.Core.IO/Blp/BlpSummaryReader.cs` now exposes the first shared `BLP` reader seam for `BLP1` and `BLP2` header summary coverage, including compression fields, alpha depth, pixel format, image size, palette or JPEG-header presence, and per-mip offset or size bounds checks
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now accepts `blp inspect --input <file.blp>` and `blp inspect --archive-root <dir> --virtual-path <path/to/file.blp> [--listfile <listfile.txt>]`, printing the shared summary plus per-mip lines
	- synthetic regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/BlpSummaryReaderTests.cs`, including both `BLP1` and `BLP2` synthetic headers plus a real standard-archive `BLP` read through `MpqArchiveCatalog`
	- `wow-viewer/tests/WowViewer.Core.Tests/WowFileDetectorTests.cs` now also locks direct synthetic `BLP2` detector coverage
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "BlpSummaryReaderTests|WowFileDetectorTests"` passed on Mar 27, 2026 with `11` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- blp inspect --archive-root i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data --virtual-path interface/minimap/minimaparrow.blp` passed on Mar 27, 2026 and reported a real `BLP2` summary with `size=32x32`, `pixelFormat=Dxt3`, and `6` in-bounds mip levels
- Important boundary:
	- this is now real first-party `BLP` header-summary ownership plus a thin inspect surface, which materially reduces the gap between `wow-viewer` and the active `MdxViewer` read surface
	- it still does not prove full `BLP` pixel decode ownership, write support, or any model-family (`M2` or `MDX`) seam yet
	- if the next chat says to keep broadening shared-library parity after this landing, the clean next family is `M2` or `MDX`, not another narrow WMO-only refinement

## Mar 27, 2026 - Shared `MOLT` Per-Light Detail Seam And Opt-In Inspect Dump Landed

- Followed the settled root-light summary seam with the next narrow shared-I/O step instead of reopening layout offsets again: `wow-viewer` now owns reusable per-entry `MOLT` detail reads for both legacy Alpha and standard later roots.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoLightDetail.cs` now owns the shared per-light contract for `MOLT` entries, including payload offset, entry size, type, attenuation flag, raw BGRA color, position, intensity, attenuation range, and optional standard-layout `headerFlagsWord` plus quaternion rotation fields
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoLightReaderCommon.cs` now centralizes shared `MOLT` entry-size inference and per-entry field decoding so summary and detail reads stay aligned across Alpha `32`-byte and later `48`-byte layouts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoLightDetailReader.cs` now exposes the reusable shared per-light detail seam instead of forcing the inspect tool to parse `MOLT` payloads itself
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoLightSummaryReader.cs` now aggregates through that shared detail decode path instead of duplicating the per-entry layout logic
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now accepts `wmo inspect ... --dump-lights` and prints opt-in `MOLT[n]` lines for each root-light entry while keeping the default report summary-only
	- synthetic regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoLightDetailReaderTests.cs`
	- real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now proves both Alpha `ironforge.wmo.MPQ` legacy entry details and standard `0.6.0` `world/wmo/khazmodan/cities/ironforge/ironforge.wmo` later-layout detail fields
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoLightSummaryReaderTests|WmoLightDetailReaderTests|Read_IronforgeAlphaPerAssetMpq_ProducesExpectedRootLightSummary|Read_IronforgeAlphaPerAssetMpq_RootLightDetails_UseLegacyLayout|Read_IronforgeStandard060_RootLightSummary_UsesStandardTailAttenuationOffsets|Read_IronforgeStandard060_RootLightDetails_ExposeRawStandardLayoutFields"` passed on Mar 27, 2026 with `8` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --archive-root i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data --virtual-path world/wmo/khazmodan/cities/ironforge/ironforge.wmo --dump-lights | Select-String '^(MOLT:|MOLT\[0\]:|MOLT\[1\]:)'` passed on Mar 27, 2026 and now reports real standard per-light lines including `MOLT[0]: ... headerFlagsWord=0x0101 ... rotation=(-0.000, 0.000, -1.000, -0.500) ...`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/KhazModan/Cities/Ironforge/ironforge.wmo.MPQ --dump-lights | Select-String '^(MOLT:|MOLT\[0\]:|MOLT\[1\]:)'` passed on Mar 27, 2026 and now reports real Alpha per-light lines including `MOLT[0]: ... entryBytes=32 ... headerFlagsWord=n/a ... rotation=n/a`
- Important boundary:
	- this proves shared per-entry `MOLT` ownership and an inspect surface that exposes the settled raw fields directly on real Alpha and standard roots
	- it still does not prove the semantic meaning of the later-layout `headerFlagsWord` bits across multiple standard assets or any deeper light rendering behavior
	- if the next chat says to continue the current shared-I/O WMO path without a narrower target, resume from the next standard-root `MOLT` seam: prove whether `headerFlagsWord` varies across additional real `v16` roots now that the raw per-entry dump is available

## Mar 27, 2026 - WMO Group Optional `MOLR`, `MOBN`, `MOBR`, And `MOBN->MOBR` Summary Slice Landed

- The next narrow shared-I/O follow-up stayed inside the existing WMO group summary seam and added ownership for the remaining low-risk optional group chunks instead of jumping into broader group-routing work.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupLightRefSummary.cs`, `WmoGroupBspNodeSummary.cs`, `WmoGroupBspFaceSummary.cs`, and `WmoGroupBspFaceRangeSummary.cs` now own shared group-level summary contracts for `MOLR`, `MOBN`, `MOBR`, and `MOBN -> MOBR`
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupLightRefSummaryReader.cs`, `WmoGroupBspNodeSummaryReader.cs`, `WmoGroupBspFaceSummaryReader.cs`, and `WmoGroupBspFaceRangeSummaryReader.cs` now read those optional group chunks through the existing shared `MOGP` boundary
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupSummaryReader.cs` now expose group-level `lightRefs`, `bspNodes`, and `bspFaceRefs` counts so inspect and embedded-group aggregate paths do not need tool-local chunk scans
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoEmbeddedGroupSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoEmbeddedGroupSummaryReader.cs` now also aggregate embedded-group `lightRefs`, `bspNodes`, and `bspFaceRefs` totals for Alpha monolithic roots
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated group lines for `MOLR`, `MOBN`, `MOBR`, and `MOBN->MOBR`, and the Alpha `MOGP(root)` aggregate line now also includes `lightRefs`, `bspNodes`, and `bspFaceRefs`
	- synthetic regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupLightRefSummaryReaderTests.cs`, `WmoGroupBspNodeSummaryReaderTests.cs`, `WmoGroupBspFaceSummaryReaderTests.cs`, and `WmoGroupBspFaceRangeSummaryReaderTests.cs`
	- real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now validates the optional embedded-group chunk totals on `castle01.wmo.MPQ` and replays the real embedded `MOGP` payloads through the new BSP readers
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoRealDataTests|WmoEmbeddedGroupSummaryReaderTests|WmoGroupSummaryReaderTests|WmoGroupLightRefSummaryReaderTests|WmoGroupBspNodeSummaryReaderTests|WmoGroupBspFaceSummaryReaderTests|WmoGroupBspFaceRangeSummaryReaderTests"` passed on Mar 27, 2026 with `9` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports:
		- `MOGP(root): groups=2 ... doodadRefs=24 lightRefs=0 bspNodes=583 bspFaceRefs=6716 ...`
- Important boundary:
	- this proves shared summary ownership for optional group `MOLR`, `MOBN`, `MOBR`, and narrow `MOBN -> MOBR` range coverage
	- it does not yet expose per-embedded-group inspect routing on root files or deeper BSP topology semantics beyond count and range signals

## Mar 27, 2026 - Alpha Root Per-Embedded-Group Inspect Routing Landed For `MOBN`, `MOBR`, And `MOBN->MOBR`

- Followed the aggregate-only Alpha root work by adding a shared per-embedded-group detail seam instead of leaving root inspect stuck at totals.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoEmbeddedGroupDetail.cs` now owns the per-embedded-group contract for root-embedded `MOGP` details
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoEmbeddedGroupDetailReader.cs` now enumerates root-embedded `MOGP` payloads through shared `WmoGroupSummaryReader`, `WmoGroupLightRefSummaryReader`, `WmoGroupBspNodeSummaryReader`, `WmoGroupBspFaceSummaryReader`, and `WmoGroupBspFaceRangeSummaryReader` without rebuilding temporary group files in the inspect tool
	- the optional group readers now expose internal `ReadMogpPayload(...)` entry points so embedded-root detail routing can reuse the same shared parsing logic directly on real root `MOGP` payloads
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints `MOGP(root)[n]`, `MOBN(root)[n]`, `MOBR(root)[n]`, and `MOBN->MOBR(root)[n]` lines for Alpha monolithic roots with embedded groups
	- synthetic regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupDetailReaderTests.cs`
	- real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now validates the per-group detail reader on `castle01.wmo.MPQ`
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupDetailReaderTests|WmoRealDataTests|WmoEmbeddedGroupSummaryReaderTests|WmoGroupBspNodeSummaryReaderTests|WmoGroupBspFaceSummaryReaderTests|WmoGroupBspFaceRangeSummaryReaderTests"` passed on Mar 27, 2026 with `8` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports real per-group lines including:
		- `MOBN(root)[0]: payloadBytes=2032 nodes=127 ...`
		- `MOBR(root)[0]: payloadBytes=2290 refs=1145 ...`
		- `MOBN->MOBR(root)[0]: nodes=127 faceRefs=1145 zeroFaceNodes=92 coveredNodes=35 outOfRangeNodes=0 maxFaceEnd=1145`
		- `MOBN(root)[1]: payloadBytes=7296 nodes=456 ...`
		- `MOBR(root)[1]: payloadBytes=11142 refs=5571 ...`
		- `MOBN->MOBR(root)[1]: nodes=456 faceRefs=5571 zeroFaceNodes=237 coveredNodes=219 outOfRangeNodes=0 maxFaceEnd=5571`
- Important boundary:
	- this proves real per-embedded-group inspect routing for the existing shared BSP summaries on Alpha `MOMO` roots
	- it still does not expose full per-embedded-group routing for every group subchunk family or deeper BSP traversal semantics

## Mar 27, 2026 - Alpha Root Per-Embedded-Group Inspect Routing Expanded To Existing Shared Group Summaries

- Followed the first BSP-only per-group landing by broadening the shared embedded-group detail seam instead of adding another tool-local root formatter.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoEmbeddedGroupDetail.cs` now also carries per-group shared summaries for `MLIQ`, `MOBA`, `MOPY`, `MOTV`, `MOCV`, `MODR`, `MOVI` or `MOIN`, `MOVT`, and `MONR`
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupLiquidSummaryReader.cs`, `WmoGroupBatchSummaryReader.cs`, `WmoGroupFaceMaterialSummaryReader.cs`, `WmoGroupUvSummaryReader.cs`, `WmoGroupVertexColorSummaryReader.cs`, `WmoGroupDoodadRefSummaryReader.cs`, `WmoGroupIndexSummaryReader.cs`, `WmoGroupVertexSummaryReader.cs`, and `WmoGroupNormalSummaryReader.cs` now expose internal `ReadMogpPayload(...)` entry points so root-embedded `MOGP` detail reads can reuse the same shared parsing logic directly on payload bytes
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoEmbeddedGroupDetailReader.cs` now populates those additional shared group summaries when the per-group `MOGP` header reports the relevant counts or liquid presence
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints additional per-group root lines for `MONR(root)[n]`, `MOVT(root)[n]`, `MOVI(root)[n]` or `MOIN(root)[n]`, `MODR(root)[n]`, `MOCV(root)[n]`, `MOTV(root)[n]`, `MOPY(root)[n]`, `MOBA(root)[n]`, and `MLIQ(root)[n]` when present
	- synthetic regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupDetailReaderTests.cs` now proves those additional detail summaries on embedded synthetic `MOGP` payloads
	- real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now proves the broadened detail reader against `castle01.wmo.MPQ`
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupDetailReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `4` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports additional real per-group lines including:
		- `MONR(root)[0]: payloadBytes=16488 normals=1374 ...`
		- `MOVT(root)[0]: payloadBytes=16488 vertices=1374 ...`
		- `MOIN(root)[0]: payloadBytes=2664 indices=1332 ...`
		- `MODR(root)[0]: payloadBytes=48 refs=24 ...`
		- `MOCV(root)[0]: payloadBytes=5496 primaryColors=1374 ...`
		- `MOTV(root)[0]: payloadBytes=10992 primaryUv=1374 ...`
		- `MOPY(root)[0]: payloadBytes=1832 entryBytes=4 faces=458 ...`
		- `MOBA(root)[0]: payloadBytes=192 entries=8 ...`
		- matching positive lines also appear for root group `1`, with `MODR(root)[1]` correctly absent because that embedded group has zero doodad refs
- Important boundary:
	- this proves the shared embedded-group detail seam can now surface the already-owned geometry or metadata group summaries directly on Alpha root `MOGP` payloads
	- real `castle01.wmo.MPQ` still does not positively prove `MOLR(root)[n]` or `MLIQ(root)[n]`, because its embedded groups report zero light refs and no liquid

## Mar 27, 2026 - `ironforge.wmo.MPQ` Added Positive Real Coverage For `MOLR(root)` And `MLIQ(root)`

- Switched the missing positive real-data proof from `castle01.wmo.MPQ` to `wow-viewer/testdata/0.5.3/tree/World/wmo/KhazModan/Cities/Ironforge/ironforge.wmo.MPQ`, because that real Alpha monolithic root actually exercises the remaining per-group light-ref and liquid seams.
- Landed pieces:
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now includes a real-data regression that proves `WmoEmbeddedGroupDetailReader` sees embedded groups with non-zero `LightRefSummary` and non-null `LiquidSummary` on `ironforge.wmo.MPQ`
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now treats invalid optional `MOLT` payload reads as non-fatal for inspect output, matching the existing optional-chunk behavior used for other root summaries, so the real Ironforge asset can continue through later root and embedded-group lines instead of aborting early
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoRealDataTests"` passed on Mar 27, 2026 with `4` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/KhazModan/Cities/Ironforge/ironforge.wmo.MPQ` passed far enough on Mar 27, 2026 to print positive per-group lines including:
		- `MOLR(root)[120]: payloadBytes=10 refs=5 ...`
		- `MOLR(root)[121]: payloadBytes=20 refs=10 ...`
		- `MOLR(root)[123]: payloadBytes=2 refs=1 ...`
		- `MOLR(root)[124]: payloadBytes=10 refs=5 ...`
		- `MLIQ(root)[127]: payloadBytes=6457 verts=30x24 tiles=29x23 ... liquidType=Magma`
- Important boundary:
	- this proves positive real-data ownership for the remaining per-group `MOLR(root)` and `MLIQ(root)` inspect lines on an Alpha monolithic root
	- the underlying `MOLT` reader still does not claim full compatibility with Ironforge's real root-light payload layout; inspect now simply does not let that optional root-summary failure block later shared outputs

## Mar 27, 2026 - Shared `MOLT` Root-Light Summary Now Reads Real Alpha `ironforge.wmo.MPQ`

- Followed the non-fatal inspect guard with the actual shared-library fix: `WowViewer.Core.IO.Wmo.WmoLightSummaryReader` now supports both the legacy 32-byte Alpha light entries and the later 48-byte root-light entries instead of assuming only the later size.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoLightSummaryReader.cs` now infers `MOLT` entry size from version and payload shape, using 32-byte entries for Alpha `v14` roots and 48-byte entries for later roots
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoLightSummary.cs` and `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now expose and print `attenStartRange`, a raw later-layout `headerFlagsWord` summary from bytes `2..3`, and later-layout rotation metrics (`rotationEntries`, `nonIdentityRotations`, `rotationLenRange`) alongside the existing intensity and `maxAttenEnd` metrics
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoLightSummaryReaderTests.cs` now covers both synthetic `v14` 32-byte `MOLT` payloads and synthetic `v17` 48-byte payloads
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now verifies the real Ironforge root light summary directly, including the exact `218` light count, `6976` payload bytes, and positive attenuation-start range from `ironforge.wmo.MPQ`
	- the same real-data test surface now also loads `world/wmo/khazmodan/cities/ironforge/ironforge.wmo` from the shared `0.6.0` standard MPQ set via `MpqArchiveCatalog` + the vendored `wow-listfile`, proving that 48-byte standard `MOLT` entries carry a non-zero `headerFlagsWord` of `0x0101` at bytes `2..3`, quaternion rotation at offsets `24..39`, and attenuation values at offsets `40` and `44`
	- `wow-viewer/src/core/WowViewer.Core.IO/Files/ArchiveVirtualFileReader.cs` now owns the shared “read a virtual file from standard archive roots” seam, and `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now consumes it for `wmo inspect --archive-root <dir> --virtual-path <world/...wmo>` with default vendored-listfile discovery
- Current verified validation for this landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoLightSummaryReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `7` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/KhazModan/Cities/Ironforge/ironforge.wmo.MPQ | Select-String '^(WMO semantics:|MOLT:|MFOG:)'` passed on Mar 27, 2026 and now reports:
		- `WMO semantics: ... lights=218 ...`
		- `MOLT: payloadBytes=6976 entries=218 distinctTypes=1 attenuated=218 intensityRange=[0.120, 1.000] attenStartRange=[1.306, 8.333] maxAttenEnd=29.611 ...`
		- `MFOG: payloadBytes=96 entries=2 ...`
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoLightSummaryReaderTests|Read_IronforgeAlphaPerAssetMpq_ProducesExpectedRootLightSummary|Read_IronforgeStandard060_RootLightSummary_UsesStandardTailAttenuationOffsets"` passed on Mar 27, 2026 with `4` targeted passing tests, including the real `0.6.0` standard-archive Ironforge root-light case
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --archive-root i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data --virtual-path world/wmo/khazmodan/cities/ironforge/ironforge.wmo | Select-String '^(Version:|WMO semantics:|MOLT:|MFOG:)'` passed on Mar 27, 2026 and now reports the real standard root-light summary through the inspect CLI, including `headerFlagsWordRange=[0x0101, 0x0101]`, `headerFlagsWordDistinct=1`, `headerFlagsWordNonZero=218`, `rotationEntries=218`, `nonIdentityRotations=218`, and `rotationLenRange=[1.118, 1.118]`
- Important boundary:
	- this proves the shared root `MOLT` semantic-summary seam on a real Alpha monolithic root instead of only surviving past a failure
	- it also now proves the real standard `v16` attenuation offsets for 48-byte entries, so the shared reader no longer reports zero attenuation on standard roots
	- it also now proves that `WowViewer.Tool.Inspect` can consume the shared standard-archive seam directly for root WMO virtual paths instead of requiring an extracted loose file or per-asset Alpha MPQ wrapper
	- it still does not prove deeper light rendering semantics beyond the existing count, raw `headerFlagsWord`, attenuation, attenuation-start range, rotation-shape summary, intensity, and bounds contract; the current real proof only locks Ironforge's standard `0x0101` word, not the per-bit meaning or cross-asset variability yet
	- the follow-up per-light inspect dump has now landed; the next standard-root `MOLT` seam is to prove whether `headerFlagsWord` varies across additional real `v16` roots

## Mar 27, 2026 - Alpha `MOGI -> MOGP(root)` Linkage Summary Landed

- After landing the Alpha embedded-group aggregate, the next narrow follow-up linked root `MOGI` entries to the embedded top-level `MOGP` blocks by ordinal pairing instead of jumping straight to full monolithic group routing.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoEmbeddedGroupLinkageSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoEmbeddedGroupLinkageSummaryReader.cs` now own the narrow Alpha `MOGI -> MOGP(root)` linkage seam
	- the linkage summary reports `MOGI` entry count, embedded `MOGP` count, covered pairs, missing/extra groups, flag matches, bounds matches, and maximum bounds delta across paired groups
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints an `MOGI->MOGP(root)` line for Alpha monolithic roots when both surfaces are present
	- synthetic regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupLinkageSummaryReaderTests.cs`
	- real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now validates the linkage summary on `castle01.wmo.MPQ`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `130` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupLinkageSummaryReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `2` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports:
		- `MOGI->MOGP(root): infos=2 groups=2 coveredPairs=2 missingGroups=0 extraGroups=0 flagMatches=0 boundsMatches=2 maxBoundsDelta=0.000`
- Important boundary:
	- this proves count/flag/bounds linkage across paired Alpha root group-info and embedded-group surfaces
	- it does not yet expose standalone per-embedded-group inspect routing or detailed per-group diff output

## Mar 27, 2026 - Alpha Monolithic Root Embedded-Group Aggregate Summary Landed

- With Alpha `MOMO` root support working on real `castle01.wmo.MPQ`, the next narrow follow-up landed on the root file's embedded top-level `MOGP` blocks instead of jumping straight to full monolithic group-consumer cutover.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoEmbeddedGroupSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoEmbeddedGroupSummaryReader.cs` now own an aggregate embedded-group summary for Alpha monolithic root files with top-level `MOGP` chunks
	- the aggregate covers embedded-group count, header-size range, groups with portals, groups with liquid, total faces, vertices, indices, normals, batches, doodad refs, and aggregate bounds
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupSummaryReader.cs` now exposes a reusable internal `MOGP` payload summary helper so the embedded-root aggregate can reuse the same group-header interpretation instead of duplicating it
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints an `MOGP(root)` aggregate line when a root WMO actually contains embedded top-level `MOGP` chunks
	- synthetic regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupSummaryReaderTests.cs`
	- real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` now also validates the embedded-group aggregate against `castle01.wmo.MPQ`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `129` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupSummaryReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `2` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports:
		- `MOGP(root): groups=2 headerBytes=128-128 groupsWithPortals=2 groupsWithLiquid=0 faces=2371 vertices=7113 indices=6195 normals=7113 batches=22 doodadRefs=24 ...`
- Important boundary:
	- this is an embedded-group aggregate seam for Alpha monolithic roots
	- it does not yet expose per-embedded-group detailed mesh summaries or direct monolithic-group selection/routing in inspect

## Mar 27, 2026 - Alpha MOMO Root WMO Support And Real 0.5.3 `.wmo.MPQ` Validation Landed

- Real Alpha-era WMO validation exposed an important boundary gap in the shared root-WMO readers:
	- `castle01.wmo.MPQ` from `wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/` extracts to a v14 Alpha monolithic WMO root
	- the file starts `MVER` then `MOMO`, not the later split-root `MVER` then `MOHD` layout
	- pre-fix `wow-viewer` classified the extracted bytes as `Unknown`, so real 0.5.3 root-WMO validation could not run through the shared root-summary stack
- Landed support:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs` now includes shared `MOMO`
	- `wow-viewer/src/core/WowViewer.Core.IO/Files/WowFileDetector.cs` now recognizes `MVER` + `MOMO` as a root `Wmo`
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoRootReaderCommon.cs` now expands Alpha `MOMO` root subchunks into a flattened root-chunk view so shared root readers can keep using readable FourCC ownership on both Alpha monolithic roots and later split roots
	- shared root readers that previously only scanned top-level chunks now route through `WmoRootReaderCommon`, including `WmoSummaryReader`, `WmoGroupInfoSummaryReader`, `WmoMaterialSummaryReader`, `WmoTextureTableSummaryReader`, `WmoDoodadNameTableSummaryReader`, `WmoDoodadSetSummaryReader`, `WmoDoodadPlacementSummaryReader`, `WmoGroupNameTableSummaryReader`, `WmoSkyboxSummaryReader`, and the portal-root helper in `WmoPortalVertexSummaryReader`
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupInfoSummary.cs` now allows negative `MOGI` name offsets, which real Alpha data exposed as valid sentinel-style values
	- `wow-viewer/src/core/WowViewer.Core.IO/Files/AlphaArchiveReader.cs` now builds broader non-map `World\...` internal-name candidates and uses them even when the input path itself ends with `.MPQ`
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now opens `.MPQ` WMO inputs through shared Alpha archive fallback and runs the shared stream-based readers, so `wmo inspect` works directly on real per-asset Alpha archives
	- real-data regression coverage landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs`
- Concrete real-data proof now available:
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026
	- reported key semantic lines:
		- `Version: 14`
		- `WMO semantics: materials=11/11 groups=2/2 portals=1 ... doodadPlacements=24/24 doodadSets=1/1`
		- `MOPT->MOPV: portals=1 vertices=4 zeroVertexPortals=0 coveredPortals=1 outOfRangePortals=0 maxVertexEnd=4`
		- `MOPR->MOPT: refs=2 portals=1 coveredRefs=2 outOfRangeRefs=0 distinctPortalRefs=1 maxPortalIndex=0`
		- `MOPR->MOGI: refs=2 groups=2 coveredRefs=2 outOfRangeRefs=0 distinctGroupRefs=2 maxGroupIndex=1`
		- `MOMT: payloadBytes=484 entryBytes=44 entries=11 ...`
		- `MOGI: payloadBytes=80 entryBytes=40 entries=2 ... nameOffsetRange=-1--1 ...`
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `128` passing tests after the Alpha `MOMO` support and real-data WMO coverage
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AlphaArchiveReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `7` targeted passing tests
- Important boundary:
	- this adds shared Alpha-root summary ownership for `MOMO`-wrapped root chunks and direct inspect support for `.wmo.MPQ` inputs
	- it does not yet add Alpha monolithic-group mesh summary ownership beyond the root summaries already extracted from `MOMO`

## Mar 27, 2026 - Batched Root WMO Portal Linkage Summary Slices For MOPT->MOPV, MOPR->MOPT, And MOPR->MOGI Landed

- `wow-viewer` now has a portal-linkage focused batched root-WMO landing that builds on the earlier raw portal summaries instead of stopping at count-only payload ownership:
	- `MOPT -> MOPV` portal-vertex range coverage summary
	- `MOPR -> MOPT` portal-ref range coverage summary
	- `MOPR -> MOGI` portal-group range coverage summary
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalVertexRangeSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoPortalVertexRangeSummaryReader.cs` now own narrow `MOPT -> MOPV` linkage semantics for zero-vertex portals, covered portals, out-of-range portals, total visible portal vertices, and max vertex end
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalRefRangeSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoPortalRefRangeSummaryReader.cs` now own narrow `MOPR -> MOPT` linkage semantics for covered refs, out-of-range refs, distinct referenced portals, and max portal index
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalGroupRangeSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoPortalGroupRangeSummaryReader.cs` now own narrow `MOPR -> MOGI` linkage semantics for covered refs, out-of-range refs, distinct referenced groups, and max group index
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoRootReaderCommon.cs` now exposes optional root-chunk reads so root readers can distinguish truly absent chunks instead of accidentally treating the first chunk as a match during optional lookup flows
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated portal-linkage lines for `MOPT->MOPV`, `MOPR->MOPT`, and `MOPR->MOGI`, and it now tolerates missing optional dependency chunks instead of aborting synthetic smoke cases
	- tests landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoPortalVertexRangeSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoPortalRefRangeSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoPortalGroupRangeSummaryReaderTests.cs`, plus a missing-`MOVV` regression in `wow-viewer/tests/WowViewer.Core.Tests/WmoVisibleVertexSummaryReaderTests.cs`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `125` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `94` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-portal-linkage-batch-test.wmo` passed on Mar 27, 2026 and reported:
		- `MOPT->MOPV: portals=2 vertices=6 zeroVertexPortals=0 coveredPortals=1 outOfRangePortals=1 maxVertexEnd=8`
		- `MOPR->MOPT: refs=3 portals=2 coveredRefs=2 outOfRangeRefs=1 distinctPortalRefs=3 maxPortalIndex=4`
		- `MOPR->MOGI: refs=3 groups=3 coveredRefs=2 outOfRangeRefs=1 distinctGroupRefs=3 maxGroupIndex=5`
- Important boundary:
	- these seams prove portal-linkage range coverage only
	- they do not yet prove full portal topology validation, plane correctness, or runtime culling behavior

## Mar 27, 2026 - Batched Root WMO Visibility Summary Slices For MOVV, MOVB, And MOVB->MOVV Landed

- `wow-viewer` now has another batched root-WMO follow-up landing covering the two visibility-owner chunks plus their first narrow linkage seam together:
	- `MOVV` visible-vertex semantic summary
	- `MOVB` visible-block semantic summary
	- `MOVB -> MOVV` visible-block range coverage summary
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoVisibleVertexSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoVisibleVertexSummaryReader.cs` now own narrow `MOVV` semantics for payload size, visible-vertex counts, and computed bounds
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoVisibleBlockSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoVisibleBlockSummaryReader.cs` now own narrow `MOVB` semantics for block counts, total vertex refs, per-block vertex-count range, first-vertex range, and max vertex end
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoVisibleBlockReferenceSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoVisibleBlockReferenceSummaryReader.cs` now own narrow `MOVB -> MOVV` linkage semantics for zero-vertex blocks, covered blocks, out-of-range blocks, visible-vertex counts, and max vertex end
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated `MOVV`, `MOVB`, and `MOVB->MOVV` semantic lines for root WMO files when those chunks are present
	- tests landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoVisibleVertexSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoVisibleBlockSummaryReaderTests.cs`, and `wow-viewer/tests/WowViewer.Core.Tests/WmoVisibleBlockReferenceSummaryReaderTests.cs`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `121` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `90` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-visibility-batch-test.wmo` passed on Mar 27, 2026 and reported:
		- `MOVV: payloadBytes=72 vertices=6 boundsMin=(-4.00, -8.00, -6.00) boundsMax=(7.00, 5.00, 9.00)`
		- `MOVB: payloadBytes=12 blocks=3 vertexRefs=7 blockSizeRange=0-4 firstVertexRange=0-5 maxVertexEnd=8`
		- `MOVB->MOVV: blocks=3 vertices=6 zeroVertexBlocks=1 coveredBlocks=1 outOfRangeBlocks=1 maxVertexEnd=8`
- Important boundary:
	- these seams prove count, bounds, and simple block-to-vertex coverage only
	- they do not yet prove runtime visibility-volume semantics, convexity validation, or any write path

## Mar 27, 2026 - Batched Root WMO Linkage Summary Slices For MODD->MODN, MOGI->MOGN, And MODS->MODD Landed

- `wow-viewer` now has a linkage-focused batched root-WMO landing instead of another raw-payload-only step:
	- `MODD -> MODN` doodad-name reference summary
	- `MOGI -> MOGN` group-name reference summary
	- `MODS -> MODD` doodad-set range summary
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoDoodadNameReferenceSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoDoodadNameReferenceSummaryReader.cs` now own narrow `MODD -> MODN` linkage semantics for resolved-name counts, unresolved-name counts, distinct resolved names, and max resolved-name length
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupNameReferenceSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupNameReferenceSummaryReader.cs` now own narrow `MOGI -> MOGN` linkage semantics for resolved-name counts, unresolved-name counts, distinct resolved names, and max resolved-name length
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoDoodadSetRangeSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoDoodadSetRangeSummaryReader.cs` now own narrow `MODS -> MODD` range semantics for empty-set counts, fully covered sets, out-of-range sets, placement counts, and max range end
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoRootReaderCommon.cs` now centralizes shared root-WMO chunk reads, version reads, root-kind validation, and string-at-offset resolution used by the linkage readers
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated linkage lines for `MODD->MODN`, `MOGI->MOGN`, and `MODS->MODD` when the needed root chunks are present
	- tests landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadNameReferenceSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupNameReferenceSummaryReaderTests.cs`, and `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadSetRangeSummaryReaderTests.cs`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `118` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `87` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-linkage-batch-test.wmo` passed on Mar 27, 2026 and reported:
		- `MODS->MODD: sets=3 placements=3 emptySets=1 coveredSets=0 outOfRangeSets=2 maxRangeEnd=18`
		- `MOGI->MOGN: entries=3 resolvedNames=2 unresolvedNames=1 distinctResolvedNames=2 maxNameLength=9`
		- `MODD->MODN: entries=3 resolvedNames=2 unresolvedNames=1 distinctResolvedNames=2 maxNameLength=7`
- Important boundary:
	- these seams prove narrow cross-chunk linkage and range validation only
	- they do not yet prove full root-name resolution ownership across every consumer path or any write path

## Mar 27, 2026 - Batched Root WMO Metadata Slices For MOLT, MFOG, And MCVP Landed

- `wow-viewer` now has another batched root-WMO metadata landing covering lights, fog, and one opaque trailing root chunk together:
	- `MOLT` light semantic summary
	- `MFOG` fog semantic summary
	- `MCVP` opaque-chunk byte summary
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoLightSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoLightSummaryReader.cs` now own narrow `MOLT` semantics for entry counts, distinct light types, attenuation usage, intensity range, attenuation-end range, and light bounds
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoFogSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoFogSummaryReader.cs` now own narrow `MFOG` semantics for entry counts, non-zero flag counts, radius ranges, fog-end range, and fog bounds
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoOpaqueChunkSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoOpaqueChunkSummaryReader.cs` now provide a thin shared seam for byte-count reporting of opaque root chunks like `MCVP`
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs` now includes shared `MOLT`, `MFOG`, `MCVP`, plus root `MOVV` and `MOVB` ids for continued root-chunk ownership work
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated `MOLT`, `MFOG`, and `MCVP` semantic lines for root WMO files when those chunks are present
	- tests landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoLightSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoFogSummaryReaderTests.cs`, and `wow-viewer/tests/WowViewer.Core.Tests/WmoOpaqueChunkSummaryReaderTests.cs`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `115` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `84` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-meta-batch-test.wmo` passed on Mar 27, 2026 and reported:
		- `MOLT: payloadBytes=96 entries=2 distinctTypes=2 attenuated=1 intensityRange=[4.000, 8.000] maxAttenEnd=20.000 boundsMin=(-4.00, 2.00, -6.00) boundsMax=(1.00, 5.00, 3.00)`
		- `MFOG: payloadBytes=96 entries=2 nonZeroFlags=1 minSmallRadius=1.000 maxLargeRadius=7.000 maxFogEnd=11.000 boundsMin=(-4.00, 2.00, -6.00) boundsMax=(1.00, 5.00, 3.00)`
		- `MCVP: payloadBytes=12`
- Important boundary:
	- these seams prove light or fog count-level semantics plus a byte-count seam for opaque `MCVP`
	- they do not yet prove deeper light/fog rendering semantics, `MCVP` structure ownership, or any write path

## Mar 27, 2026 - Batched Root WMO Portal Summary Slices For MOPV, MOPT, And MOPR Landed

- `wow-viewer` now has a second batched root-WMO landing covering the three portal-owner chunks together:
	- `MOPV` portal-vertex semantic summary
	- `MOPT` portal-info semantic summary
	- `MOPR` portal-ref semantic summary
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalVertexSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoPortalVertexSummaryReader.cs` now own narrow `MOPV` semantics for vertex counts and computed bounds
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalInfoSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoPortalInfoSummaryReader.cs` now own narrow `MOPT` semantics for portal-entry counts, max start vertex, max vertex count, and plane-D range
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoPortalRefSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoPortalRefSummaryReader.cs` now own narrow `MOPR` semantics for ref counts, distinct portal counts, max group index, and side distribution
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs` now includes shared `MOPV`, `MOPT`, and `MOPR` chunk ids
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated `MOPV`, `MOPT`, and `MOPR` semantic lines for root WMO files when portal data is present
	- tests landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoPortalVertexSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoPortalInfoSummaryReaderTests.cs`, and `wow-viewer/tests/WowViewer.Core.Tests/WmoPortalRefSummaryReaderTests.cs`
- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `112` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `81` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-portals-test.wmo` passed on Mar 27, 2026 and reported:
		- `MOPV: payloadBytes=36 vertices=3 boundsMin=(-4.00, -8.00, -6.00) boundsMax=(7.00, 5.00, 9.00)`
		- `MOPT: payloadBytes=40 entries=2 maxStartVertex=10 maxVertexCount=4 planeDRange=[-2.000, 1.000]`
		- `MOPR: payloadBytes=24 entries=3 distinctPortals=2 maxGroupIndex=7 sides(+/-/0)=1/1/1`
- Important boundary:
	- these three seams prove portal-owner count and range semantics only
	- they do not yet prove full root-to-group portal routing behavior or any write path

## Mar 27, 2026 - Batched Root WMO Summary Slices For MODD, MOGN, And MOSB Landed

- `wow-viewer` now has a batched set of three additional narrow root-WMO seams instead of a one-slice landing:
	- `MODD` doodad-placement semantic summary
	- `MOGN` group-name table semantic summary
	- `MOSB` skybox semantic summary
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoDoodadPlacementSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoDoodadPlacementSummaryReader.cs` now own count-level `MODD` semantics for entry counts, distinct name indices, scale range, alpha range, and placement bounds
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupNameTableSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupNameTableSummaryReader.cs` now own narrow `MOGN` string-table semantics for count, longest entry, and max offset
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoSkyboxSummary.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoSkyboxSummaryReader.cs` now own the narrow `MOSB` seam for payload size and resolved skybox name
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs` now includes shared `MOGN` and `MOSB` chunk ids
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints dedicated `MODD`, `MOGN`, and `MOSB` semantic lines for root WMO files when those chunks are present
	- tests landed in `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadPlacementSummaryReaderTests.cs`, `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupNameTableSummaryReaderTests.cs`, and `wow-viewer/tests/WowViewer.Core.Tests/WmoSkyboxSummaryReaderTests.cs`

- Current verified validation for this batched landing:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `109` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `78` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-batch-test.wmo` passed on Mar 27, 2026 and reported:
		- `MOSB: payloadBytes=4 skybox=Sky`
		- `MOGN: payloadBytes=31 names=3 longestEntry=10 maxOffset=21`
		- `MODD: payloadBytes=80 entries=2 distinctNameIndices=2 maxNameIndex=7 scaleRange=[1.250, 2.500] alphaRange=[170, 255] boundsMin=(-4.00, 2.00, -6.00) boundsMax=(1.00, 5.00, 3.00)`
- Important boundary:
	- these three seams prove string-table and placement-summary ownership only
	- they do not yet prove `MODD` linkage back to `MODN`, `MOGN` name resolution against group metadata, or any write path

## Mar 27, 2026 - Shared WMO Root Doodad-Set Semantic Summary Slice Landed

- `wow-viewer` now has the next narrow WMO root seam after `MODN`: a shared `MODS` doodad-set semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoDoodadSetSummary.cs` now owns the typed root `MODS` summary contract for payload size, set count, non-empty-set count, longest set-name length, total doodad refs, max start index, and max range end
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoDoodadSetSummaryReader.cs` now reads `MODS` payload semantics from root WMO files as a narrow doodad-set seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MODS` semantic line for root WMO files that contain doodad sets
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadSetSummaryReaderTests.cs` now covers a synthetic `MODS` table with empty and non-empty sets
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `106` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `75` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-mods-test.wmo` passed on Mar 27, 2026 and reported `MODS: payloadBytes=96 entries=3 nonEmptySets=2 longestName=7 totalDoodadRefs=10 maxStartIndex=12 maxRangeEnd=18`
- Important boundary:
	- this proves shared `MODS` semantic summary for doodad-set counts and range signals only
	- this does not yet prove set-to-`MODD` linkage beyond count-level ranges or any write path

## Mar 27, 2026 - Shared WMO Root Doodad-Name Table Semantic Summary Slice Landed

- `wow-viewer` now has the next narrow WMO root seam after `MOTX`: a shared `MODN` doodad-name-table semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoDoodadNameTableSummary.cs` now owns the typed root `MODN` summary contract for payload size, name count, longest entry length, max string offset, distinct extension counts, and `.mdx` or `.m2` entry counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoDoodadNameTableSummaryReader.cs` now reads `MODN` payload semantics from root WMO files as a narrow string-table seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MODN` semantic line for root WMO files that contain doodad-name tables
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadNameTableSummaryReaderTests.cs` now covers a synthetic `MODN` table with mixed `.mdx` and `.m2` entries plus a nested path
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `105` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `74` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-modn-test.wmo` passed on Mar 27, 2026 and reported `MODN: payloadBytes=31 names=3 longestEntry=15 maxOffset=15 extensions=2 mdxEntries=2 m2Entries=1`
- Important boundary:
	- this proves shared `MODN` semantic summary for string-table counts and extension-shape signals only
	- this does not yet prove offset resolution against `MODD`, path canonicalization, or any write path

## Mar 27, 2026 - Shared WMO Root Texture-Table Semantic Summary Slice Landed

- `wow-viewer` now has the next narrow WMO root seam after `MOMT`: a shared `MOTX` texture-table semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoTextureTableSummary.cs` now owns the typed root `MOTX` summary contract for payload size, texture count, longest entry length, max string offset, distinct extension counts, and `.blp` entry counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoTextureTableSummaryReader.cs` now reads `MOTX` payload semantics from root WMO files as a narrow table-summary seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOTX` semantic line for root WMO files that contain texture tables
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoTextureTableSummaryReaderTests.cs` now covers a synthetic `MOTX` table with mixed texture extensions and nested paths
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `104` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `73` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-motx-test.wmo` passed on Mar 27, 2026 and reported `MOTX: payloadBytes=33 textures=3 longestEntry=16 maxOffset=16 extensions=2 blpEntries=2`
- Important boundary:
	- this proves shared `MOTX` semantic summary for table counts and string-shape signals only
	- this does not yet prove offset resolution against `MOMT`, path canonicalization, or any write path

## Mar 27, 2026 - Shared WMO Root Material Semantic Summary Slice Landed

- `wow-viewer` now has the next narrow WMO root seam after `MOGI`: a shared `MOMT` material semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoMaterialSummary.cs` now owns the typed root `MOMT` summary contract for payload size, inferred entry size, entry count, distinct shader counts, distinct blend-mode counts, non-zero-flag counts, and maximum texture offsets across the first three slots
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoMaterialSummaryReader.cs` now reads standard, legacy, and vintage `MOMT` payload semantics from root WMO files using `MOHD` material-count guidance when available
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOMT` semantic line for root WMO files that contain material entries
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoMaterialSummaryReaderTests.cs` now covers synthetic standard 64-byte and legacy 44-byte `MOMT` payloads
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `103` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `72` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-momt-test.wmo` passed on Mar 27, 2026 and reported `MOMT: payloadBytes=128 entryBytes=64 entries=2 distinctShaders=2 distinctBlendModes=2 nonZeroFlags=1 maxTex1Ofs=24 maxTex2Ofs=20 maxTex3Ofs=88`
- Important boundary:
	- this proves shared `MOMT` semantic summary for material-entry layout and selected top-level fields only
	- this does not yet prove texture-name resolution against `MOTX`, color interpretation, or any write path

## Mar 27, 2026 - Shared WMO Root Group-Info Semantic Summary Slice Landed

- `wow-viewer` now has the next narrow WMO root seam after the group-level payload summaries: a shared `MOGI` group-info semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupInfoSummary.cs` now owns the typed root `MOGI` summary contract for payload size, inferred entry size, entry count, distinct-flag counts, non-zero-flag counts, name-offset range, and union bounds
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupInfoSummaryReader.cs` now reads standard and legacy `MOGI` payload semantics from root WMO files using `MOHD` group-count guidance when available
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOGI` semantic line for root WMO files that contain group info
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupInfoSummaryReaderTests.cs` now covers synthetic standard 32-byte and legacy 40-byte `MOGI` payloads
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `101` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `70` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-mogi-test.wmo` passed on Mar 27, 2026 and reported `MOGI: payloadBytes=64 entryBytes=32 entries=2 distinctFlags=2 nonZeroFlags=1 nameOffsetRange=12-40 boundsMin=(-7.00, -2.00, -3.00) boundsMax=(4.00, 8.00, 9.00)`
- Important boundary:
	- this proves shared `MOGI` semantic summary for root group-info entry counts, flag coverage, name-offset ranges, and union bounds only
	- this does not yet prove root-to-group file linkage beyond raw entry counts, name resolution against `MOGN`, or any write path

## Mar 27, 2026 - Shared WMO Group Normal Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MOVT`: a shared `MONR` normal semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupNormalSummary.cs` now owns the typed normal-summary contract for payload size, normal count, component ranges, length ranges, average length, and near-unit counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupNormalSummaryReader.cs` now reads `MONR` payload semantics from WMO group files as a narrow count-and-range seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MONR` semantic line for WMO group files that contain normal payloads
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupNormalSummaryReaderTests.cs` now covers a synthetic `MONR` payload with two unit-length normals and one shorter vector
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `99` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `68` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-normal-test.wmo` passed on Mar 27, 2026 and reported `MONR: payloadBytes=36 normals=3 rangeX=[0.000, 1.000] rangeY=[-1.000, 0.500] rangeZ=[0.000, 0.500] lengthRange=[0.866, 1.000] avgLength=0.955 nearUnit=2`
- Important boundary:
	- this proves shared `MONR` semantic summary for count, component ranges, and length ranges only
	- this does not yet prove tangent-space ownership, generated-normal fallback logic, or any write path

## Mar 27, 2026 - Shared WMO Group Vertex Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MOVI`: a shared `MOVT` vertex semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupVertexSummary.cs` now owns the typed vertex-summary contract for payload size, vertex count, and computed vertex bounds
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupVertexSummaryReader.cs` now reads `MOVT` payload semantics from WMO group files as a narrow count-and-bounds seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOVT` semantic line for WMO group files that contain vertex payloads
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupVertexSummaryReaderTests.cs` now covers a synthetic `MOVT` payload with mixed positive and negative coordinates
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `98` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `67` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-vertex-test.wmo` passed on Mar 27, 2026 and reported `MOVT: payloadBytes=36 vertices=3 boundsMin=(-4.00, -8.00, -6.00) boundsMax=(7.00, 5.00, 9.00)`
- Important boundary:
	- this proves shared `MOVT` semantic summary for count and computed bounds only
	- this does not yet prove topology linkage, coordinate ownership beyond the payload, or any write path

## Mar 27, 2026 - Shared WMO Group Index Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MODR`: a shared `MOVI` or `MOIN` index semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupIndexSummary.cs` now owns the typed index-summary contract for chunk id, payload size, index count, triangle count, distinct index count, index range, and degenerate-triangle count
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupIndexSummaryReader.cs` now reads either `MOVI` or `MOIN` payload semantics from WMO group files as a narrow count-level seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOVI` or `MOIN` semantic line for WMO group files that contain index payloads
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupIndexSummaryReaderTests.cs` now covers synthetic `MOVI` and `MOIN` payloads including a degenerate-triangle case
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `97` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `66` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-index-test.wmo` passed on Mar 27, 2026 and reported `MOVI: payloadBytes=12 indices=6 triangles=2 distinctIndices=4 indexRange=0-3 degenerateTriangles=1`
- Important boundary:
	- this proves shared `MOVI` or `MOIN` semantic summary for count, range, and degenerate-triangle coverage only
	- this does not yet prove topology ownership, face-material alignment, or any write path

## Mar 27, 2026 - Shared WMO Group Doodad-Ref Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MOCV`: a shared `MODR` doodad-ref semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupDoodadRefSummary.cs` now owns the typed `MODR` summary contract for ref counts, distinct ref counts, min or max ref range, and duplicate-ref counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupDoodadRefSummaryReader.cs` now reads `MODR` payload semantics from WMO group files as a narrow count-level seam
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MODR` semantic line for WMO group files that contain doodad refs
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupDoodadRefSummaryReaderTests.cs` now covers a synthetic `MODR` payload with duplicate refs
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `95` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `64` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-doodadref-test.wmo` passed on Mar 27, 2026 and reported `MODR: payloadBytes=8 refs=4 distinctRefs=3 refRange=3-9 duplicateRefs=1`
- Important boundary:
	- this proves shared `MODR` semantic summary for doodad-ref counts and ranges only
	- this does not yet prove linkage back to root doodad tables, placement ownership, or any write path

## Mar 27, 2026 - Shared WMO Group Vertex-Color Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MOTV`: a shared `MOCV` vertex-color semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupVertexColorSummary.cs` now owns the typed `MOCV` summary contract for primary color payload size, primary color count, BGRA-derived channel ranges, average alpha, and extra color-set counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupVertexColorSummaryReader.cs` now reads `MOCV` payload semantics from WMO group files while keeping the primary set separate from optional extra color sets
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOCV` semantic line for WMO group files that contain vertex colors
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupVertexColorSummaryReaderTests.cs` now covers a synthetic WMO group with one primary and one extra `MOCV` set
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `94` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `63` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-color-test.wmo` passed on Mar 27, 2026 and reported `MOCV: payloadBytes=8 primaryColors=2 rangeR=[30, 70] rangeG=[20, 60] rangeB=[10, 50] rangeA=[40, 80] avgA=60 extraColorSets=1 totalExtraColors=3 maxExtraColors=3`
- Important boundary:
	- this proves shared `MOCV` semantic summary for count, channel-range, and extra-set coverage only
	- this does not yet prove runtime lighting interpretation, second color-set semantics, or any write path

## Mar 27, 2026 - Shared WMO Group UV Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MOPY`: a shared `MOTV` UV semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupUvSummary.cs` now owns the typed `MOTV` UV-summary contract for primary UV payload size, primary UV count, primary U or V ranges, additional UV-set counts, and aggregate extra-UV counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupUvSummaryReader.cs` now reads `MOTV` payload semantics from WMO group files while keeping the primary set separate from optional extra UV sets
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOTV` semantic line for WMO group files that contain UV data
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupUvSummaryReaderTests.cs` now covers a synthetic WMO group with one primary and one extra `MOTV` set
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `93` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `62` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-uv-test.wmo` passed on Mar 27, 2026 and reported `MOTV: payloadBytes=24 primaryUv=3 rangeU=[-0.200, 0.800] rangeV=[0.200, 0.900] extraUvSets=1 totalExtraUv=2 maxExtraUv=2`
- Important boundary:
	- this proves shared `MOTV` UV semantic summary for counts and value ranges only
	- this does not yet prove runtime UV-set selection, secondary-set semantics, or any write path

## Mar 27, 2026 - Shared WMO Group Face-Material Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MOBA`: a shared `MOPY` face-material semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupFaceMaterialSummary.cs` now owns the typed `MOPY` face-material summary contract for face counts, inferred entry size, distinct material ids, highest material id, hidden-face count, and flagged-face count
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupFaceMaterialSummaryReader.cs` now reads `MOPY` payload semantics from WMO group files while respecting v17 two-byte and v16 four-byte entry layouts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupReaderCommon.cs` now exposes shared `MOPY` entry-size inference used by both count-level and face-material readers
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOPY` semantic line for WMO group files that contain face-material entries
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupFaceMaterialSummaryReaderTests.cs` now covers synthetic v17-style and v16-style `MOPY` payloads
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `92` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `61` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-face-v17-test.wmo` passed on Mar 27, 2026 and reported `MOPY: payloadBytes=8 entryBytes=2 faces=4 distinctMaterials=2 highestMaterialId=7 hiddenFaces=1 flaggedFaces=2`
- Important boundary:
	- this proves shared `MOPY` face-material semantic summary for count, flag, and material-id coverage only
	- this does not yet prove face-to-batch reconstruction, material resolution against root tables, or any write path

## Mar 27, 2026 - Shared WMO Group Batch Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after `MLIQ`: a shared `MOBA` batch semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupBatchSummary.cs` now owns the typed `MOBA` batch-summary contract for entry counts, material-id coverage, total index count, first-index range, max index end, and flagged-batch counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupBatchSummaryReader.cs` now reads `MOBA` payload semantics from WMO group files without pretending to own full batch reconstruction
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MOBA` semantic line for WMO group files that contain batches
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupBatchSummaryReaderTests.cs` now covers synthetic v17-style material-bearing batches and v16-style material-less batches
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `90` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `59` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-batch-test.wmo` passed on Mar 27, 2026 and reported `MOBA: payloadBytes=48 entries=2 hasMaterialIds=True distinctMaterials=2 highestMaterialId=7 totalIndexCount=15 firstIndexRange=10-20 maxIndexEnd=29 flaggedBatches=1`
- Important boundary:
	- this proves shared `MOBA` batch semantic summary for top-level batch-entry counts and index or material signals only
	- this does not yet prove full batch reconstruction, bounding-box interpretation, or write-path ownership

## Mar 27, 2026 - Shared WMO Group Liquid Semantic Summary Slice Landed

- `wow-viewer` now has the next deeper WMO group seam after the `MOGP` header summary: a shared `MLIQ` semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoLiquidBasicType.cs` now owns the basic liquid-family enum used by the summary seam
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupLiquidSummary.cs` now owns the typed `MLIQ` semantic-summary contract for liquid dimensions, corner, material id, height range, tile-flag coverage, visible tile count, and inferred liquid family
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupReaderCommon.cs` now centralizes `MOGP` payload reads, header-size detection, subchunk enumeration, and shared helper logic used by both WMO group summary readers
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupLiquidSummaryReader.cs` now reads `MLIQ` payload semantics from WMO group files without pretending to own runtime mesh generation
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupSummaryReader.cs` now consumes the new shared `WmoGroupReaderCommon` helper instead of carrying its own `MOGP` scanning copy
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now prints a dedicated `MLIQ` semantic line for WMO group files that contain liquid
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupLiquidSummaryReaderTests.cs` now covers a synthetic `MLIQ` payload with ocean inference and height-range validation
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `88` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `57` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-liquid-test.wmo` passed on Mar 27, 2026 and reported `MLIQ: payloadBytes=63 verts=2x2 tiles=1x1 ... visibleTiles=1/1 ... liquidType=Ocean`
- Important boundary:
	- this proves shared `MLIQ` payload semantic summary for dimensions, height range, visible tile counts, and basic family inference only
	- this does not yet prove full WMO liquid mesh generation, orientation fitting, or any write path

## Mar 27, 2026 - Shared WMO Group Semantic Summary Slice Landed

- `wow-viewer` now has the next narrow WMO follow-up seam after the root summary: a shared WMO group semantic-summary reader.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoGroupSummary.cs` now owns the typed WMO group semantic-summary contract for `MOGP` header fields, declared batch counts, geometry subchunk counts, optional extra UV-set count, doodad-ref count, and liquid presence
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoGroupSummaryReader.cs` now reads standard `MOGP` group files at count or presence level without pretending to own deep mesh reconstruction
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs` now also owns the shared readable WMO group subchunk ids used by the new seam
	- `wow-viewer/src/core/WowViewer.Core.IO/Files/WowFileDetector.cs` now recognizes `MOGP`-first files as `WmoGroup` instead of treating them as unknown when `MVER` is absent
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now routes `wmo inspect` to either the root-WMO or group-WMO reader based on shared detection and prints a dedicated WMO group report for group files
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupSummaryReaderTests.cs` now covers synthetic `MVER + MOGP` and `MOGP`-first group files
	- `wow-viewer/tests/WowViewer.Core.Tests/WowFileDetectorTests.cs` now locks `MOGP`-first detection as `WmoGroup`
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `87` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `56` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-summary-test.wmo` passed on Mar 27, 2026 and reported `Header: bytes=68 ... Geometry: faces=3 vertices=2 indices=3 ... hasLiquid=False`
- Important boundary:
	- this proves shared WMO group semantic summary for `MOGP` headers and top-level geometry or metadata subchunk counts only
	- this does not yet prove full group mesh decode, batch reconstruction, liquid payload ownership, or any write path

## Mar 27, 2026 - Shared ADT MCNK Semantic Summary And First WMO Root Summary Slices Landed

- `wow-viewer` now has its first chunk-internal ADT semantic seam plus its first shared WMO root semantic-summary seam.
- Landed ADT pieces:
	- `wow-viewer/src/core/WowViewer.Core/Maps/AdtChunkIds.cs` now owns the shared readable `MCNK` subchunk ids used by the ADT MCNK summary seam
	- `wow-viewer/src/core/WowViewer.Core/Maps/AdtMcnkSummary.cs` now owns the typed ADT `MCNK` semantic-summary contract for root-header presence, index coverage, area-id coverage, hole or liquid or `MCCV` flags, subchunk presence, and per-chunk layer-count signals
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtMcnkSummaryReader.cs` now reads those `MCNK` signals from root, `_tex0.adt`, and `_obj0.adt` files while staying at count or presence level instead of deep payload decode
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports the shared ADT `MCNK` semantic summary for `map inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/AdtMcnkSummaryReaderTests.cs` now covers synthetic root, `_tex0.adt`, and `_obj0.adt` buffers plus fixed real-data `development_0_0.adt`, `development_0_0_tex0.adt`, and `development_0_0_obj0.adt`
- Landed WMO pieces:
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoChunkIds.cs` now owns the shared readable root-WMO chunk ids used by the summary seam
	- `wow-viewer/src/core/WowViewer.Core/Wmo/WmoSummary.cs` now owns the typed WMO root semantic-summary contract for `MOHD`-reported counts, string-table counts, top-level entry counts, flags, and bounds
	- `wow-viewer/src/core/WowViewer.Core.IO/Wmo/WmoSummaryReader.cs` now reads those signals from standard chunked WMO root files without pretending to be group-file or deep payload ownership
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now supports `wmo inspect --input <file.wmo>` as a thin shared-reader consumer
	- `wow-viewer/tests/WowViewer.Core.Tests/WmoSummaryReaderTests.cs` now covers a synthetic WMO root summary case
- Current verified validation for these slices:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `84` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `53` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 27, 2026 and now reports the shared ADT `MCNK` semantic line `mcnk=256 zero=0 headerLike=0 distinctIndex=0 duplicateIndex=0 areaIds=0 holes=0 liquidFlags=0 mccvFlags=0 mcly=256 mcal=203 mcsh=174 totalLayers=775 maxLayers=4 multiLayerChunks=203`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-summary-test.wmo` passed on Mar 27, 2026 and reported `materials=2/2 groups=4/4 portals=1 lights=3 textures=2 doodadNames=5/5 doodadPlacements=6/6 doodadSets=2/2 flags=0x00001234`
- Important boundaries:
	- the ADT `MCNK` seam proves count or presence level ownership for root-header signals and split-file subchunk coverage, not full terrain payload decode, alpha decode, shadow decode, liquid decode, or writer support
	- the WMO seam proves root-file semantic summary only; it does not yet prove group-file parsing, material payload ownership beyond entry counts, or any write path

## Mar 27, 2026 - Shared ADT Semantic Summary Slice Landed

- `wow-viewer` now has its first shared ADT semantic-summary seam beyond raw top-level chunk inventory.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Maps/AdtSummary.cs` now owns the typed ADT semantic-summary contract for terrain-chunk counts, texture-name counts, doodad or WMO name counts, placement counts, and top-level MFBO or MH2O or MAMP or MTXF presence
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtSummaryReader.cs` now reads those signals from root, `_tex0.adt`, and `_obj0.adt` files without pretending to be a deep payload parser
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/MapSummaryReaderCommon.cs` now centralizes the shared top-level chunk-payload and string-block helpers used by both `AdtSummaryReader` and `WdtSummaryReader`
	- `wow-viewer/src/core/WowViewer.Core/Maps/MapChunkIds.cs` now includes `MAMP` so texture-parameter presence is expressed as a shared map chunk id instead of a tool-local literal
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports the shared ADT semantic summary for `map inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/AdtSummaryReaderTests.cs` now covers synthetic root, `_tex0.adt`, and `_obj0.adt` buffers plus fixed real-data `development_0_0.adt`, `development_0_0_tex0.adt`, and `development_0_0_obj0.adt`
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `77` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `46` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 27, 2026 and reported `kind=AdtTex terrainChunks=256 textures=5 doodadNames=0 wmoNames=0 doodadPlacements=0 wmoPlacements=0 hasMfbo=False hasMh2o=False hasMamp=True hasMtxf=False`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_obj0.adt` passed on Mar 27, 2026 and reported `kind=AdtObj terrainChunks=256 textures=0 doodadNames=6 wmoNames=12 doodadPlacements=10 wmoPlacements=15 hasMfbo=False hasMh2o=False hasMamp=False hasMtxf=False`
- Important boundary:
	- this proves shared ADT semantic summary for top-level terrain-chunk counts, string-table counts, placement counts, and selected presence flags across root and split ADT-family files
	- this does not yet prove deep root ADT parsing, split-texture payload parsing, split-object payload parsing, chunk-internal MCNK semantics, or any write path

## Mar 27, 2026 - Shared WDT Semantic Summary Slice Landed

- `wow-viewer` now has its first shared WDT semantic-summary seam beyond raw top-level chunk inventory.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core/Maps/WdtSummary.cs` now owns the typed WDT semantic-summary contract for MPHD WMO-based flags, MAIN occupancy, standard `MAIN` flag distributions, string-table counts, and top-level placement counts
	- `wow-viewer/src/core/WowViewer.Core.IO/Maps/WdtSummaryReader.cs` now reads those signals from either Alpha-style or standard WDT top-level chunks, including standard `MAIN` flag summary metadata for `hasAdt`, `allWater`, `loaded`, unknown bits, async-id presence, and distinct non-zero flag values, without pretending to be a full payload parser
	- `wow-viewer/src/core/WowViewer.Core/Maps/MapChunkIds.cs` now includes `MDNM` and `MONM` so the shared reader can treat Alpha name tables as first-class chunk ids instead of tool-local literals
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` now reports both the shared WDT semantic summary and the standard `MAIN` flag distribution line for `map inspect`
	- `wow-viewer/tests/WowViewer.Core.Tests/WdtSummaryReaderTests.cs` now covers synthetic standard WDT flag distributions, synthetic Alpha WDT boundary behavior, and the fixed real-data `development.wdt` standard-flag distribution
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `71` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WdtSummaryReaderTests` passed on Mar 31, 2026 with `3` passing WDT-summary tests after adding standard `MAIN` flag metadata
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development.wdt` passed on Mar 31, 2026 and reported `WDT MAIN flags: any=1496 hasAdt=1496 allWater=0 loaded=0 unknown=0 asyncIds=0 distinct=0x1:1496`
- Important boundary:
	- this proves shared WDT semantic summary for top-level MPHD, MAIN occupancy, standard `MAIN` flag metadata, string-table, and placement-count signals
	- this does not yet prove deep WDT payload parsing, WMO placement semantics beyond counts, or any write path

## Mar 27, 2026 - Shared AreaIdMapper Archive-Backed Loading Replaced Constructor-Time Extracted-Tree Probing

- The primary `AreaIdMapper` load path is now archive-backed instead of constructor-time test-data probing in `WoWMapConverter.Core.Converters.AlphaToLkConverter`.
- Landed pieces:
	- `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs` now exposes `TryLoadFromArchives(...)`, reading `AreaTable` and `Map` through shared `IArchiveReader` plus `DbClientFileReader` and feeding DBCD through an in-memory provider instead of a staged file tree
	- shorthand archive build inputs `0.5.3` and `3.3.5` now normalize to the full WoWDBDefs-compatible build strings the DBCD seam actually needs
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/AlphaToLkConverter.cs` no longer calls `TryAutoLoadFromTestData()` in its constructor; it now initializes the mapper lazily from explicit DBC paths or explicit Alpha and LK archive roots, then falls back to CSV crosswalks only if those inputs fail
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/Program.cs` now accepts `--alpha-client` and `--lk-client` so converter runs can point directly at MPQ roots
	- `wow-viewer/tests/WowViewer.Core.Tests/AreaIdMapperTests.cs` now covers synthetic archive-backed DBCD loading and explicit archive-missing diagnostics
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `37` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` passed on Mar 27, 2026 with the existing warning floor
- Important boundary:
	- this proves the shared area-mapper seam can now consume archive-backed DBC bytes without extracted trees
	- this does not yet include a real client-root converter smoke test against Alpha and LK MPQ inputs in this workspace

## Mar 27, 2026 - Shared AreaIdMapper DBCD Wiring And Explicit Fallback Warning Landed

- `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs` now prefers real schema-aware loading through DBCD when extracted `AreaTable` and `Map` files are present and `WoWDBDefs` definitions can be discovered from the workspace.
- Landed pieces:
	- `WowViewer.Core.IO.csproj` now references the same vendored `gillijimproject_refactor/lib/wow.tools.local/DBCD/DBCD/DBCD.csproj` project the active viewer already uses, and bundles `gillijimproject_refactor/lib/WoWDBDefs/definitions` into output
	- shared `AreaIdMapper` now discovers `WoWDBDefs/definitions` from the bundled `definitions` output first, then from `gillijimproject_refactor/lib/WoWDBDefs/definitions`, `wow-viewer/libs/wowdev/WoWDBDefs/definitions`, `libs/wowdev/WoWDBDefs/definitions`, or legacy `lib/WoWDBDefs/definitions`
	- shared `AreaIdMapper.LoadDbcs(...)` now uses DBCD plus WoWDBDefs for known `0.5.3` and `3.3.5` paths when available, then falls back to the narrow raw `DbcReader` only when schema-backed loading is unavailable
	- shared `AreaIdMapper.TryAutoLoadFromTestData()` and `TryLoadKnownTestDataFromRoot(...)` now prefer `gillijimproject_refactor/test_data/*/tree/DBFilesClient` before legacy `test_data/*/tree/DBFilesClient`, and record explicit diagnostics instead of silently failing when extracted tables are missing
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/AlphaToLkConverter.cs` now surfaces that missing-tree diagnostic as a runtime warning before falling back to crosswalk-only behavior
	- added focused shared-library regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/AreaIdMapperTests.cs` for explicit missing-tree reporting and a synthetic DBCD+WoWDBDefs-backed `AreaTable`/`Map` load path
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `66` tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 27, 2026 with the existing warning floor and no new build break
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- convert i:/parp/parp-tools/gillijimproject_refactor/test_data/0.5.3/alphawdt/World/Maps/PVPZone01/PVPZone01.wdt -o i:/parp/parp-tools/output/pvpzone01-alpha-to-lk-smoke-dbcd-check3 -v` passed on Mar 27, 2026 and now emits one explicit warning that names the preferred `gillijimproject_refactor/test_data/*/tree/DBFilesClient` roots first when extracted DBC trees are absent
- Important boundary:
	- this proves the shared area-mapper seam is now actually wired to DBCD plus WoWDBDefs when the extracted table trees exist
	- the current real-data runtime smoke tests in this workspace still fall back because the extracted `gillijimproject_refactor/test_data/0.5.3/tree/DBFilesClient/*` and `gillijimproject_refactor/test_data/3.3.5/tree/DBFilesClient/*` files are absent here
	- this is still narrow `AreaTable` and `Map` ownership for the mapper seam, not broad general DBC or DB2 format ownership across all tables

## Mar 26, 2026 - Shared AreaIdMapper And Crosswalk Ownership Landed

- `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs` now owns the remaining live old-repo area-mapping seam plus the embedded area-crosswalk resource it depended on.
- Landed pieces:
	- added shared `AreaIdMapper`
	- moved `area_crosswalk.csv` into `wow-viewer/src/core/WowViewer.Core.IO/Resources/area_crosswalk.csv`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/AlphaToLkConverter.cs` now uses shared `WowViewer.Core.IO.Dbc.AreaIdMapper`
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Dbc/AreaIdMapper.cs`
	- deleted dead `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/AreaIdCrosswalk.cs`
	- deleted the old embedded `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Resources/area_crosswalk.csv`
	- added focused regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/AreaIdMapperTests.cs` for embedded-default loading, matching-report CSV parsing, and continent-hinted name matching
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `64` tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with `53` warnings and no new build break
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` passed on Mar 26, 2026 with `3` warnings
- Important boundary:
	- this proves shared ownership of the active old-repo area-ID mapping seam and its embedded crosswalk data plus consumer compile validation
	- this does not prove broader DBC schema ownership beyond the narrow shared mapper or runtime converter signoff on real data
	- no runtime validation was run

## Mar 26, 2026 - Shared Alpha MPQ Old-Repo Caller Cutover Landed

- The shared `wow-viewer/src/core/WowViewer.Core.IO/Files/AlphaArchiveReader.cs` seam now owns the remaining active old-repo per-asset MPQ callers that were still using the deleted duplicate reader in `WoWMapConverter.Core`.
- Landed pieces:
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` now uses shared `AlphaArchiveReader`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17Converter.cs` now uses shared `AlphaArchiveReader`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/WmoV14ToV17ExtendedConverter.cs` now uses shared `AlphaArchiveReader`
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/AlphaMpqReader.cs`
	- added focused regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/AlphaArchiveReaderTests.cs` for per-asset MPQ block selection and companion `.MPQ` fallback
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `61` tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with `53` warnings and no new build break
- Important boundary:
	- this proves shared ownership of the active old-repo Alpha per-asset MPQ caller seam plus consumer compile validation
	- this does not prove broader WMO, MDX, or BLP format ownership beyond read access through the shared Alpha archive seam
	- no viewer runtime validation was run

## Mar 26, 2026 - Dead Old DBC Helper Cleanup Landed

- The old `WoWMapConverter.Core` archive or DBC helper layer left behind after the shared `Core.IO` cutovers has now been narrowed and cleaned up instead of being carried forward as dead compatibility code.
- Landed pieces:
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/NativeMpqService.cs`
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/Md5TranslateResolver.cs`
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/MapDbcService.cs`
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/GroundEffectService.cs`
	- deleted `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Dbc/DbcReader.cs`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Dbc/AreaIdMapper.cs` now reads tables through shared `WowViewer.Core.IO.Dbc.DbcReader`
- Current live-boundary result from the targeted review:
	- the deleted helper files were definition-only in the active `gillijimproject_refactor/src` tree after the earlier `Core.IO` cutovers
	- `AreaIdMapper` remains the only clearly live DBC-backed seam still owned by `WoWMapConverter.Core`
	- the current active consumer of that seam is `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/AlphaToLkConverter.cs`
- Current verified validation for this slice:
	- workspace diagnostics for `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core` reported no errors after the cleanup
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with `54` warnings and no new build break
- Important boundary:
	- this is old-repo cleanup plus dependency-boundary tightening, not a new `wow-viewer` library slice
	- no new `wow-viewer` tests were run in this pass because the shared library code did not change
	- the next real DBC ownership decision is whether `AreaIdMapper` and the Alpha-to-LK area crosswalk should move into `wow-viewer`

## Mar 26, 2026 - Shared DBC Lookup And VLM Archive Cutover Landed

- `wow-viewer/src/core/WowViewer.Core.IO` now owns the next narrow non-PM4 table-backed helper slice that was still stranded in `WoWMapConverter.Core`.
- Landed pieces:
	- shared `DbcReader`
	- shared `DbcHeader`
	- shared `MapDirectoryLookup`
	- shared `GroundEffectLookup`
	- expanded shared `DbClientFileReader` table probing to cover `DBFilesClient`, `DBC`, and root `.dbc` or `.db2` candidates
	- focused regression coverage for shared DBC lookup behavior in `wow-viewer/tests/WowViewer.Core.Tests/DbcLookupTests.cs`
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj` now references `wow-viewer/src/core/WowViewer.Core.IO`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` now uses shared `IArchiveCatalog` or `IArchiveReader` instead of `WoWMapConverter.Core.Services.NativeMpqService`
	- `VlmDatasetExporter` now resolves `Map.dbc` through shared `MapDirectoryLookup`
	- `VlmDatasetExporter` now resolves ground-effect doodads through shared `GroundEffectLookup`
	- `VlmDatasetExporter` now loads MD5 minimap translation through shared callback-based `WowViewer.Core.IO.Files.Md5TranslateResolver`
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `59` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with the existing warning floor plus no new build break
- Important boundary:
	- this proves shared DBC-backed lookup ownership plus active VLM consumer compile validation
	- this does not prove general DBC or DB2 format ownership, write support, or viewer runtime behavior
	- `MdxViewer` was not rebuilt in this slice because the change targeted `WowViewer.Core.IO` plus `WoWMapConverter.Core`
	- the old `MapDbcService`, `GroundEffectService`, local `DbcReader`, old `Md5TranslateResolver`, and `NativeMpqService` helper layer was later deleted from `WoWMapConverter.Core` once the active-path review showed it was dead in the current tree
	- `AreaIdMapper` is now the remaining active DBC-backed seam in `WoWMapConverter.Core`, and it already reads through shared `Core.IO`

## Mar 26, 2026 - Concrete Shared MPQ Catalog Port Landed

- `wow-viewer/src/core/WowViewer.Core.IO` now owns the concrete standard MPQ implementation used by the active `MdxViewer` path, not just the contracts and bootstrap helpers around it.
- Landed pieces:
	- shared `MpqArchiveCatalog`
	- shared `MpqArchiveCatalogFactory`
	- internal `MpqDiagnostics`
	- focused regression coverage for archive priority, patched-delete fallback, internal listfile extraction, and direct file-0 reads in `wow-viewer/tests/WowViewer.Core.Tests/MpqArchiveCatalogTests.cs`
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs` now defaults to shared `MpqArchiveCatalogFactory`
	- deleted the active bridge file `gillijimproject_refactor/src/MdxViewer/DataSources/NativeMpqArchiveCatalog.cs`
	- active `MdxViewer` `.cs` source no longer instantiates or references `WoWMapConverter.Core.Services.NativeMpqService` in its standard MPQ consumer path
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `57` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 with the existing `32` warning floor
- Important boundary:
	- this proves concrete shared MPQ catalog ownership for the active `MdxViewer` path plus consumer compile validation
	- older `WoWMapConverter.Core.Services.NativeMpqService` code still exists for other non-migrated old-repo consumers, but it is no longer the active `MdxViewer` standard MPQ implementation path
	- no viewer runtime validation was run

## Mar 26, 2026 - Shared Archive Bootstrap And Alpha Wrapper Cutovers Landed

- `wow-viewer/src/core/WowViewer.Core.IO` now owns the next two archive-adjacent seams that `MpqDataSource` was still keeping locally or routing directly to old services.
- Landed pieces:
	- shared `ArchiveCatalogBootstrapper`
	- shared `ArchiveCatalogBootstrapResult`
	- shared `AlphaArchiveReader`
	- shared `PkwareExplode`
	- focused regression coverage for external listfile parsing, archive bootstrap aggregation, Alpha internal-name candidate generation, and direct-file fallback behavior in `wow-viewer/tests/WowViewer.Core.Tests`
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs` now uses shared `ArchiveCatalogBootstrapper` instead of owning the standard archive bootstrap or external listfile parsing path locally
	- `MpqDataSource` now uses shared `AlphaArchiveReader` instead of directly calling `WoWMapConverter.Core.Services.AlphaMpqReader`
	- the active `MdxViewer` source no longer references the old Alpha wrapper reader in its MPQ data source path
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `53` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 after the cutovers
- Important boundary:
	- this proves shared archive bootstrap and shared Alpha wrapper helper ownership plus consumer compile validation
	- `NativeMpqService` still remains behind `gillijimproject_refactor/src/MdxViewer/DataSources/NativeMpqArchiveCatalog.cs`; the concrete standard MPQ implementation is not ported yet
	- no viewer runtime validation was run

## Mar 26, 2026 - Shared Archive-Reader MPQ Cutover Landed

- `wow-viewer/src/core/WowViewer.Core.IO` now owns the shared archive-reader or archive-catalog boundary that `MdxViewer` was still expressing directly through `WoWMapConverter.Core.Services.NativeMpqService`.
- Landed pieces:
	- shared `IArchiveReader`
	- shared `IArchiveCatalog`
	- shared `IArchiveCatalogFactory`
	- shared `DbClientFileReader` for `DBFilesClient` DBC or DB2 path probing
	- focused regression coverage for DBC or DB2 candidate ordering and first-match table reads in `wow-viewer/tests/WowViewer.Core.Tests`
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs` now depends on shared archive interfaces instead of `NativeMpqService`
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDBCProvider.cs` now reads tables through shared `IArchiveReader` and `DbClientFileReader`
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now consumes `MpqDataSource.ArchiveReader` instead of `MpqService`
	- direct `NativeMpqService` ownership is isolated to `gillijimproject_refactor/src/MdxViewer/DataSources/NativeMpqArchiveCatalog.cs`
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `49` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 after the cutover
- Important boundary:
	- this proves shared archive-reader ownership plus consumer compile validation
	- `NativeMpqService` itself is not ported into `wow-viewer`; it is still the current implementation behind the compatibility adapter
	- Alpha wrapper reads still call `WoWMapConverter.Core.Services.AlphaMpqReader` directly inside `MpqDataSource`; that is a separate seam from standard MPQ archive access
	- no viewer runtime validation was run

## Mar 26, 2026 - Shared MD5 Minimap Translation Cutover Landed

- `wow-viewer/src/core/WowViewer.Core.IO` now owns the shared MD5 minimap translation seam that `MdxViewer` was still importing from `WoWMapConverter.Core.Services`.
- Landed pieces:
	- shared `Md5TranslateIndex`
	- shared `Md5TranslateResolver.TryLoad(...)` with archive read callbacks instead of direct `NativeMpqService` type ownership
	- shared `MinimapService.GetMinimapTilePath(...)` and `MinimapTileExists(...)`
	- focused regression coverage for map-specific TRS loading and `dir:` directory-context parsing in `wow-viewer/tests/WowViewer.Core.Tests`
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now loads the MD5 minimap translation index through shared `WowViewer.Core.IO.Files.Md5TranslateResolver`
	- `gillijimproject_refactor/src/MdxViewer/Rendering/MinimapRenderer.cs` and `Export/MapGlbExporter.cs` now consume shared `Md5TranslateIndex` and `MinimapService`
	- `MdxViewer.csproj` now references `wow-viewer/src/core/WowViewer.Core.IO`
	- `ViewerApp` no longer pulls the default development-map directory from `WoWMapConverter.Core.Services.DevelopmentMapAnalyzer`; it now uses shared `Pm4CoordinateService.DefaultDevelopmentMapDirectory`
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `47` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 after the cutover
- Important boundary:
	- this proves shared MD5 minimap translation ownership plus consumer compile validation
	- `MdxViewer` still depends on `WoWMapConverter.Core` for broader MPQ, terrain, converter, and VLM subsystems; that wider cutover is still open
	- no viewer runtime validation was run

## Mar 26, 2026 - PM4 Linked-Position-Ref Summary Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now owns the linked MPRL position-ref summary seam that was still being aggregated inside `WorldScene`.
- Landed pieces:
	- shared `Pm4LinkedPositionRefSummary` contract
	- shared `Pm4PlacementMath.SummarizeLinkedPositionRefs(...)`
	- focused regression coverage for mixed normal-or-terminator linked refs and terminator-only fallback behavior
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now maps local `MprlEntry` values into shared `Core.PM4` position-ref entries and delegates linked-ref summary aggregation to `Core.PM4`
	- the viewer-local heading-range, floor-range, and circular-mean aggregation no longer owns that PM4 seam
- Current verified validation for this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `31` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `45` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-linked-position-ref-summary-hookup/` passed on Mar 26, 2026
- Important boundary:
	- this proves shared PM4 linked-position-ref summary ownership plus consumer compile validation
	- no PM4 inspect or viewer runtime validation was run in this slice because analyzer or report output did not change
	- this is not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Placement-Solution Consumer Hookup Landed

- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now delegates PM4 placement-solution assembly to the already-landed `Core.PM4` placement-solution seam.
- Landed pieces:
	- the CK24 overlay path now calls shared `Pm4PlacementMath.ResolvePlacementSolution(...)` instead of resolving planar transform, world pivot, and world yaw correction as separate consumer-owned steps
	- local per-piece consumer wrappers for those already-shared PM4 placement pieces no longer own that path
- Current verified validation for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-placement-solution-hookup/` passed on Mar 26, 2026
- Important boundary:
	- this slice reuses an already-tested `Core.PM4` placement-solution seam; no new `wow-viewer` library code changed
	- the currently recorded `31` PM4-test and `45` total-test floor is the latest library-test proof
	- this is consumer compile validation only, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Connector-Key Consumer Hookup Landed

- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now delegates PM4 connector-key derivation to the already-landed `Core.PM4` connector-key seam.
- Landed pieces:
	- `BuildCk24ConnectorKeys()` now builds a shared `Pm4PlacementSolution` and delegates connector-key derivation to `Pm4PlacementMath.BuildConnectorKeys(...)`
	- local viewer-owned connector-point conversion and quantization logic no longer owns that PM4 grouping input path
- Current verified validation for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-connector-key-hookup/` passed on Mar 26, 2026
- Important boundary:
	- this slice reuses an already-tested `Core.PM4` connector-key seam; no new `wow-viewer` library code changed
	- the currently recorded `31` PM4-test and `45` total-test floor is the latest library-test proof
	- this is consumer compile validation only, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Merge-Map Consumer Hookup Landed

- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now delegates PM4 merged-group resolution to the already-landed `Core.PM4` merge-map seam.
- Landed pieces:
	- `RebuildPm4MergedObjectGroups()` now builds shared `Pm4ConnectorMergeCandidate` inputs and delegates canonical merge resolution to `Pm4PlacementMath.BuildMergedGroupMap(...)`
	- local viewer-owned union-find and merge-heuristic logic no longer owns that PM4 grouping path
- Current verified validation for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-merge-map-hookup/` passed on Mar 26, 2026
- Important boundary:
	- this slice reuses an already-tested `Core.PM4` merge-map seam; no new `wow-viewer` library code changed
	- the currently recorded `31` PM4-test and `45` total-test floor is the latest library-test proof
	- this is consumer compile validation only, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Correlation Geometry-Input Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now owns the next PM4-only correlation seam: geometry-input assembly for shared object-state construction.
- Landed pieces:
	- shared `Pm4GeometryLineSegment` contract
	- shared `Pm4GeometryTriangle` contract
	- shared `Pm4CorrelationGeometryInput` contract
	- `Pm4CorrelationMath.BuildObjectStatesFromGeometry(...)`
	- regression coverage for building shared PM4 correlation object states directly from PM4 line or triangle geometry plus a transform
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now maps PM4 overlay lines and triangles into shared PM4 geometry-input contracts and delegates object-state construction to `Core.PM4`
	- local viewer-specific world-point flattening for PM4 correlation object-state assembly is no longer the owner of that seam
- Current verified validation floor after this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `29` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `45` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-correlation-geometry-hookup/` passed on Mar 26, 2026
- Important boundary:
	- PM4-owned geometry, transforms, and shared object-state construction belong in `Core.PM4`
	- WMO-facing correlation report payloads stay in WMO or consumer space and should not be moved into PM4 just because the report compares PM4 against WMO
	- this proves shared PM4 geometry-input ownership plus consumer compile validation, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Correlation Object-State Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now owns the next reusable correlation object-state seam for PM4 placement or report work.
- Landed pieces:
	- shared `Pm4CorrelationObjectDescriptor` contract
	- shared `Pm4CorrelationObjectInput` contract
	- shared `Pm4CorrelationObjectState` contract
	- `Pm4CorrelationMath.BuildObjectStates(...)`
	- public `Pm4CorrelationMath.BuildFootprintHull(...)`, `BuildTransformedFootprintHull(...)`, and `ComputeFootprintArea(...)`
	- regression coverage for synthetic object-state bounds or footprint derivation, empty-geometry fallback, and transformed footprint-hull construction
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now builds shared correlation object inputs and consumes shared correlation states
	- `BuildPm4WmoPlacementCorrelationReport(...)` now uses shared hull and metric helpers from `Core.PM4` instead of duplicating that scoring path locally
- Current verified validation floor after this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `28` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `42` total tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-correlation-state-hookup/` passed on Mar 26, 2026
- Important boundary:
	- this proves shared object-state, hull, and scoring consumption plus consumer compile validation, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Correlation-Math Library Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now owns the next reusable correlation-scoring seam for PM4 placement or report work.
- Landed pieces:
	- shared `Pm4CorrelationMetrics` contract
	- shared `Pm4CorrelationCandidateScore` contract
	- `Pm4CorrelationMath.EvaluateMetrics(...)`
	- `Pm4CorrelationMath.CompareCandidateScores(...)`
	- library-owned planar-gap, vertical-gap, footprint-distance, polygon-overlap, footprint-area-ratio, planar-overlap, and AABB-overlap helpers extracted from the current `WorldScene` correlation logic
	- regression coverage for synthetic metric calculation and ranking precedence
- Current verified validation floor after this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `25` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `39` total tests
- Important boundary:
	- no active-viewer consumer wiring changed in this slice
	- this proves library-owned correlation metrics and ranking, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Connector-Group Merge Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now owns the first connector-based group-merge seam for PM4 grouping work.
- Landed pieces:
	- shared `Pm4ObjectGroupKey` contract
	- shared `Pm4ConnectorMergeCandidate` contract
	- `Pm4PlacementMath.BuildMergedGroupMap(...)`
	- library-owned connector-overlap, bounds-padding, and center-distance merge heuristics extracted from the current `WorldScene`
	- regression coverage for neighbor-tile merge resolution and same-tile non-merge behavior
- Current verified validation floor after this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `22` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `36` total tests
- Important boundary:
	- no active-viewer consumer wiring changed in this slice
	- this proves library-owned merge heuristics, not viewer runtime PM4 closure

## Mar 26, 2026 - PM4 Connector-Key Library Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now owns the first reusable connector-key extraction seam for PM4 grouping or correlation work.
- Landed pieces:
	- shared `Pm4ConnectorKey` contract
	- `Pm4PlacementMath.BuildConnectorKeys(...)`
	- library-owned conversion of `MSUR.MdosIndex` exterior vertices into quantized world-space connector keys through typed `Pm4PlacementSolution`
	- dedupe and deterministic ordering of connector keys
	- regression coverage for distinct sorted connector extraction and yaw-corrected connector placement
- Current verified validation floor after this slice:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `20` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `34` total tests
- Important boundary:
	- no active-viewer consumer wiring changed in this slice
	- this is library-owned grouping or correlation groundwork, not viewer runtime PM4 signoff

## Mar 26, 2026 - wow-viewer Source-Of-Truth Reset

- The current default rule for `wow-viewer` changed: new implementation work should treat `WowViewer.Core.PM4`, `WowViewer.Core`, and `WowViewer.Core.IO` as the canonical owners, not `MdxViewer`.
- `MdxViewer` is now a historical, extraction, or consumer-compatibility input for `wow-viewer` work rather than the default runtime PM4 reference.
- Default validation for `wow-viewer` work is `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`, `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`, and the relevant inspect or converter command against the fixed development dataset.
- Build `gillijimproject_refactor/src/MdxViewer/MdxViewer.sln` only when a slice intentionally changes consumer compatibility or when the user explicitly asks for it.
- Older sections below that describe `MdxViewer` as the PM4 runtime reference are now historical context, not the live rule for new `wow-viewer` implementation work.

## Mar 26, 2026 - wow-viewer PM4 Fresh-Chat Handoff

- Treat the current `wow-viewer` PM4 state as library-first progress, not PM4 completion.
- What is real in `wow-viewer/src/core/WowViewer.Core.PM4` now:
	- research-seeded PM4 reader and inspect surface
	- working `pm4 inspect`, `pm4 audit`, `pm4 audit-directory`, `pm4 linkage`, `pm4 mscn`, `pm4 unknowns`, and `pm4 export-json`
	- shared placement-contract and placement-math slices for axis detection, planar-transform resolution, world-yaw correction, world-space centroid, pivot rotation, corrected world-position conversion, typed placement solutions, and typed coordinate-mode resolution
	- first reusable connector-key extraction seam for grouping or correlation work through typed placement solutions
	- first connector-based group-merge seam for PM4 grouping work through typed merge candidates and merge-map resolution
	- first narrow active-viewer consumer hookups for `ResolvePlanarTransform(...)`, `TryComputeWorldYawCorrectionRadians(...)`, and `ComputeSurfaceWorldCentroid(...)`
- Current verified validation floor:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `22` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `11` placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `36` total tests
- Boundaries that still matter:
	- `MdxViewer` remains the runtime PM4 reference implementation
	- shared-library and compile validation are not viewer runtime signoff
	- final semantics are still open around `MSLK.RefIndex`, `MPRL.Unk14/16`, `MPRR.Value1`, and full coordinate ownership
	- renderer-space conversion, broader object-group transforms, and final viewer object composition still remain outside `Core.PM4`
- Best next PM4 slice for a fresh chat:
	- hook the already-landed typed coordinate-mode resolver into the active `WorldScene` call site through the same narrow adapter pattern already used for planar-transform, yaw-correction, and centroid seams
	- keep that slice explicit about proof level: consumer hookup and regression preservation, not runtime PM4 closure

## Mar 25, 2026 - wow-viewer Tool Inventory And Cutover Plan

- Added a concrete inventory and cutover document at `plans/wow_viewer_tool_inventory_and_cutover_plan_2026-03-25.md`.
- New planning decisions captured there:
	- first-class survivors are the main viewer shell, one converter CLI, one inspect CLI, one optional catalog CLI, and a real PM4 library plus workspace from day one.
	- do not port duplicate legacy executables as permanent apps; merge WoWMapConverter with still-useful WoWRollback or AlphaLkToAlpha conversion seams, merge the Alpha WDT inspectors, and keep DBCTool.V2 behavior only.
	- PM4 correction: current `MdxViewer` behavior is the de facto PM4 runtime reference implementation, and `Pm4Research` should be ported as the future `Core.PM4` library family because PM4 semantics are still under active research.
	- keep parpToolbox, PM4Tool, ADTPrefabTool, and the legacy WoWRollback GUI or viewer surfaces in `parp-tools` as archaeology or reference unless a specific algorithm is deliberately re-homed.
	- immediate follow-up planning docs now exist for bootstrap layout, CLI or GUI surfaces, and the PM4 library direction:
		- `plans/wow_viewer_bootstrap_layout_plan_2026-03-25.md`
		- `plans/wow_viewer_cli_gui_surface_plan_2026-03-25.md`
		- `plans/wow_viewer_pm4_library_plan_2026-03-25.md`
	- migration emphasis is now effectively `1, 3, 2`: bootstrap layout and project skeleton, then dual-surface tool design, then deeper PM4 library consolidation work.
- This plan refines `plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md` rather than replacing it.
- Validation status:
	- planning and documentation only
	- no viewer, converter, or renderer code changed in this slice

## Mar 25, 2026 - wow-viewer Initial Skeleton Created In Workspace

- A first-pass `wow-viewer/` scaffold now exists directly under the workspace root.
- Created projects:
	- `src/viewer/WowViewer.App`
	- `src/core/WowViewer.Core`
	- `src/core/WowViewer.Core.IO`
	- `src/core/WowViewer.Core.Runtime`
	- `src/core/WowViewer.Core.PM4`
	- `src/tools-shared/WowViewer.Tools.Shared`
	- `tools/converter/WowViewer.Tool.Converter`
	- `tools/inspect/WowViewer.Tool.Inspect`
- Added first-pass repo files:
	- `WowViewer.slnx`
	- `Directory.Build.props`
	- `Directory.Packages.props`
	- `eng/Version.props`
	- `scripts/bootstrap.ps1`
	- `scripts/bootstrap.sh`
	- `scripts/validate-real-data.ps1`
- PM4-specific rule carried into the scaffold:
	- `Core.PM4` exists from day one
	- the placeholder code explicitly treats `MdxViewer` as the PM4 runtime reference and `Pm4Research` as the PM4 library seed
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- this is only a structure lock and placeholder-code build, not a real code-port or runtime signoff

## Mar 25, 2026 - First PM4 Code-Port Slice Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first real PM4 code port from `src/Pm4Research.Core`.
- Landed pieces:
	- typed chunk models for the trusted PM4 chunk set
	- `Pm4ResearchDocument`
	- `Pm4ResearchReader`
	- `Pm4ResearchSnapshotBuilder`
- Important boundary:
	- this is still a raw research-facing PM4 reader layer
	- current `MdxViewer` behavior remains the runtime PM4 reference implementation for reconstruction, grouping, transforms, and viewer-facing semantics
	- no viewer PM4 logic has been re-homed onto `Core.PM4` yet
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 after the PM4 port
	- no runtime validation or app integration has happened yet

## Mar 25, 2026 - PM4 Inspect Verbs Now Work In wow-viewer

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first single-file PM4 analyzer and report layer on top of the earlier reader port.
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect` now has working PM4 commands:
	- `pm4 inspect --input <file.pm4>`
	- `pm4 export-json --input <file.pm4> [--output <report.json>]`
- Smoke-test result on the fixed reference tile:
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` succeeded
	- output included version `12304`, `54` chunks, `6318` `MSVT` vertices, `9990` `MSCN` points, and `2493` `MPRL` refs for `development_00_00.pm4`
- Important boundary:
	- this is still single-file research analysis, not viewer reconstruction or PM4 correctness closure
	- current `MdxViewer` behavior remains the runtime PM4 reference implementation

## Mar 25, 2026 - PM4 Audit And Placement Contracts Follow-Up

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first decode-audit path plus the first extracted MdxViewer-facing PM4 placement-contract seam.
- Landed pieces:
	- `Pm4ResearchAuditAnalyzer` with single-file and directory-level decode or corpus audit entry points
	- `WowViewer.Tool.Inspect` verbs for `pm4 audit --input <file.pm4>` and `pm4 audit-directory --input <directory>`
	- shared `Pm4AxisConvention`, `Pm4CoordinateMode`, `Pm4PlanarTransform`, `Pm4CoordinateService`, and `Pm4PlacementContract`
- New research note captured in the inspect layer:
	- CK24 low-16 object values, read as integers, appear to be plausible `UniqueID` candidates on the development map, but this remains a hypothesis until correlated against real placed-object data
- Important boundary:
	- this is still not the full MdxViewer PM4 reconstruction or transform solver port
	- current `MdxViewer` behavior remains the runtime reference implementation
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 after this slice
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 audit-directory --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026 and scanned `616` PM4 files with no unknown chunks or diagnostics
	- early audit findings worth keeping visible:
		- `MDOS.buildingIndex->MDBH` shows real invalid references in the development corpus
		- `MSLK.RefIndex->MSUR` also shows corpus-level mismatches in nontrivial counts, which supports keeping linkage interpretation labeled as research

## Mar 25, 2026 - First wow-viewer PM4 Tests Landed

- `wow-viewer/tests/WowViewer.Core.PM4.Tests` now exists as the first real-data test project in the new repo.
- Current test coverage locks:
	- reader counts for `development_00_00.pm4`
	- current single-file analyzer summary and the `UniqueID` research note
	- current single-file decode-audit findings for `development_00_00.pm4`
	- current corpus-audit shape for `test_data/development/World/Maps/development`
- Validation status:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 with `6` passing tests
	- this is still fixed-dataset regression coverage only, not broad PM4 correctness closure

## Mar 25, 2026 - PM4 Linkage Slice And Placement-Math Helper Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first linkage-report family ported from `Pm4Research.Core`.
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect` now supports `pm4 linkage --input <directory> [--output <report.json>]`.
- Validated corpus result on the fixed development PM4 directory:
	- `616` files scanned
	- `150` files with ref-index mismatches
	- `58` files with bad `MDOS` refs
	- `4553` total ref-index mismatches
	- only `2` low16 object-id groups reused across multiple full CK24 values in this corpus slice
- Important interpretation boundary:
	- low16 CK24 object values may still align with expected `UniqueID` ranges, but the linkage report shows that range alignment alone is not enough to treat them as globally unique identifiers by themselves.
- First actual `WorldScene` helper port also landed in `Core.PM4`:
	- `Pm4PlacementMath.DetectAxisConventionByRanges`
	- `Pm4PlacementMath.IsLikelyTileLocal`
	- `Pm4PlacementMath.ConvertPm4VertexToWorld`
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 linkage --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 with `7` tests

## Mar 25, 2026 - PM4 MSCN Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first MSCN relationship analyzer ported from `Pm4Research.Core`.
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect` now supports `pm4 mscn --input <directory> [--output <report.json>]`.
- Validated corpus result on the fixed development PM4 directory:
	- `616` files scanned
	- `309` files with MSCN
	- `1,342,410` total MSCN points
	- `MSUR.MdosIndex -> MSCN`: `511,891` fits and `6,201` misses
	- raw MSCN bounds overlap against mesh-backed CK24 groups: `1,162` fits and `724` misses
	- swapped-XY MSCN bounds overlap against mesh-backed CK24 groups: only `10` fits and `1,876` misses
- Important interpretation boundary:
	- this slice weakens the simple XY-swapped MSCN companion-space hypothesis for the fixed development corpus
	- it still does not make MSCN authoritative for final viewer reconstruction by itself
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 mscn --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 with `7` tests

## Mar 26, 2026 - PM4 Unknowns Slice And Normal-Based Axis Scoring Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first unknowns-report family ported from `Pm4Research.Core` plus the next extracted `WorldScene` solver seam for normal-based axis scoring.
- Landed pieces:
	- `Pm4ResearchUnknownsAnalyzer`
	- unknowns report records for relationship summaries, link-id patterns, MSPI interpretation, field distributions, and explicit open-question findings
	- `WowViewer.Tool.Inspect` verb `pm4 unknowns --input <directory> [--output <report.json>]`
	- `Pm4PlacementMath.DetectAxisConventionByTriangleNormals`
	- `Pm4PlacementMath.DetectAxisConventionBySurfaceNormals`
	- normal-based axis scoring helpers on triangles and surfaces
- Validated corpus result on the fixed development PM4 directory:
	- `616` files scanned
	- `309` non-empty geometry or link files
	- `1,273,335` `MSLK.LinkId` values, all currently fitting the sentinel-tile pattern in this corpus
	- `598,882` active `MSLK` path windows with `399,183` indices-only fits and `199,699` dual-fit windows
	- `MSLK.RefIndex -> MSUR` still partial with `1,268,782` fits and `4,553` misses
	- `MPRR.Value1` remains mixed-domain with partial fits against both `MPRL` and `MSVT`
- Important interpretation boundary:
	- this strengthens the decode-evidence base, but it still does not close the final semantics of `MSLK.RefIndex`, `MPRL.Unk14/16`, `MPRR`, or PM4 coordinate ownership.
	- normal-based axis scoring is now reusable in `Core.PM4`, but the full viewer reconstruction and transform solver still live in current `WorldScene`.
- Validation status:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 unknowns --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 26, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `8` tests

## Mar 26, 2026 - PM4 Planar-Transform Resolver Slice Landed

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the next extracted `WorldScene` PM4 solver seam: planar-transform resolution against MPRL anchors.
- Landed pieces:
	- `Pm4PlacementMath.ResolvePlanarTransform`
	- MPRL centroid-distance scoring against planar candidates
	- MPRL footprint scoring for multi-anchor groups
	- MPRL heading/yaw comparison with quarter-turn fallback
	- reusable helpers for MPRL planar-point conversion and principal-yaw estimation
- Validation status:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `9` tests
	- current measured development-tile result for the whole-tile test slice: tile-local `XYPlaneZUp` resolves to planar transform `(swap=false, invertU=false, invertV=false)`
	- synthetic world-space regression case now also locks a quarter-turn candidate selection `(swap=true, invertU=true, invertV=false)`
- Important boundary:
	- this still does not port the full PM4 object-level placement pipeline or viewer yaw-correction layer.
	- active `WorldScene` remains the runtime reference implementation for full PM4 reconstruction behavior.

## Mar 26, 2026 - PM4 World-Yaw Correction Slice And First Viewer Consumer Wiring

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the next extracted `WorldScene` solver seam: world-yaw correction against MPRL heading evidence.
- Landed pieces:
	- `Pm4PlacementMath.TryComputeWorldYawCorrectionRadians`
	- signed basis fallback against expected MPRL yaw
	- synthetic regression coverage for a meaningful non-zero yaw correction case
- Active viewer integration follow-up also landed:
	- `gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` now references `wow-viewer/src/core/WowViewer.Core.PM4`
	- `WorldScene.ResolvePlanarTransform(...)` now delegates to shared `Core.PM4` through a narrow adapter path
	- `WorldScene.TryComputeWorldYawCorrectionRadians(...)` now delegates to shared `Core.PM4` through the same adapter path
- Important boundary:
	- this is still a narrow consumer slice; `WorldScene` continues to own the broader PM4 placement/render path.
	- no runtime signoff has happened yet on viewer-visible PM4 behavior after the shared-library hookup.

## Mar 26, 2026 - PM4 World-Space Centroid Slice And Second Viewer Consumer Hookup

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the next extracted `WorldScene` solver seam above world-yaw correction: world-space surface centroid computation.
- Landed pieces:
	- `Pm4PlacementMath.ComputeSurfaceWorldCentroid(...)`
	- synthetic tile-local regression coverage for the shared centroid helper
	- `WorldScene.ComputeSurfaceWorldCentroid(...)` now delegates to shared `Core.PM4` through the existing adapter path
- Validation status:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `11` tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `4` placement-focused tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-centroid-hookup/` passed on Mar 26, 2026
- Important boundary:
	- this moves the shared world-space pivot helper only; renderer-space centroid handling and the broader PM4 object placement path still remain in `WorldScene`
	- no real-data runtime signoff yet on viewer-visible PM4 behavior after this additional shared-solver hookup

## Mar 26, 2026 - First Non-PM4 Shared Map Reader Slice Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core` now contains the first shared non-PM4 map-format constants and summary contracts:
	- `MapChunkIds`
	- `MapFileKind`
	- `MapChunkLocation`
	- `MapFileSummary`
- `wow-viewer/src/core/WowViewer.Core.IO` now contains the first reusable WDT or ADT top-level reader layer:
	- `ChunkedFileReader`
	- `MapFileSummaryReader`
- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect` now has the first non-PM4 shared-I/O consumer verb:
	- `map inspect --input <file.wdt|file.adt>`
- Fixed-dataset scope for this slice:
	- `development.wdt`
	- `development_0_0.adt`
- Important boundary:
	- this is only top-level chunk order, version, and file-kind summarization for WDT or ADT-family files
	- it is not yet a full ADT or WDT semantic parser, writer, or runtime cutover

## Mar 26, 2026 - First Shared Cross-Family File Detector Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core` now contains the first cross-family file-detection contracts:
	- `WowFileKind`
	- `WowFileDetection`
- `wow-viewer/src/core/WowViewer.Core.IO` now contains the first shared cross-family detector:
	- `WowFileDetector`
- `MapFileSummaryReader` now routes WDT or ADT-family classification through that shared detector instead of owning its own kind heuristics.
- `wow-viewer/tools/converter/WowViewer.Tool.Converter` now has the first non-placeholder non-PM4 command:
	- `detect --input <file>`
- Fixed-dataset smoke coverage for this slice:
	- `development.wdt` -> `Wdt`
	- `development_00_00.pm4` -> `Pm4`
	- `development_0_0_tex0.adt` -> `AdtTex`
	- `development_0_0_obj0.adt` -> `AdtObj`
- Important boundary:
	- this is classification and version detection only
	- it is not yet a shared read or write implementation for WMO, M2, BLP, DBC, or DB2 payload semantics

## Mar 26, 2026 - PM4 World-Space Yaw Helper Slice Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the next library-only PM4 math slice adjacent to the earlier yaw solver and centroid helper.
- Landed pieces:
	- shared `Pm4PlacementMath.RotateWorldAroundPivot(...)`
	- shared `Pm4PlacementMath.ConvertPm4VertexToWorld(...)` overload that can apply yaw correction around a world pivot without any renderer-space dependency
	- synthetic regression coverage for pivot rotation and corrected world-position conversion
- Validation status:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `6` placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `13` tests
- Important boundary:
	- this is a `wow-viewer` library slice only; no new `MdxViewer` consumer hookup was added in this step
	- renderer-space conversion and object-transform composition still remain outside `Core.PM4`

## Mar 26, 2026 - PM4 Placement-Solution Contract Slice Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the first typed placement-result contract that bundles the current library-owned PM4 placement decision into one object.
- Landed pieces:
	- `Pm4PlacementSolution`
	- `Pm4PlacementMath.ResolvePlacementSolution(...)`
	- `Pm4PlacementMath.ConvertPm4VertexToWorld(Vector3, Pm4PlacementSolution)`
	- synthetic end-to-end regression coverage for world-space transform, pivot, and yaw-correction resolution
- Validation status:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `8` placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `15` tests
- Important boundary:
	- this is still a `wow-viewer` library slice only; no new active-viewer consumer wiring was added here
	- object-group transforms, renderer-space conversion, and final viewer object composition still remain outside `Core.PM4`

## Mar 26, 2026 - wow-viewer Copilot Workflow Surface Updated

- The shared Copilot workflow surface now explicitly treats `wow-viewer` as a primary active path alongside `gillijimproject_refactor`.
- New shared continuation assets now live under `.github/`:
	- `.github/skills/wow-viewer-pm4-library/SKILL.md`
	- `.github/skills/wow-viewer-migration-continuation/SKILL.md`
	- `.github/prompts/wow-viewer-pm4-library-implementation.prompt.md`
- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md` now routes implementation-sized PM4 library asks to the dedicated PM4 library prompt instead of only the broader repo-planning prompts.
- Future-session workflow rule:
	- use the PM4 library prompt or skill when the ask is the next `Core.PM4` slice, inspect verb, regression update, or narrow shared-solver extraction
	- use the broader tool-suite prompt set only when the ask is repo-shape, tool inventory, CLI or GUI parity, or migration sequencing
- Important boundary:
	- this workflow update does not change runtime PM4 validation status
	- `wow-viewer` test or build passes are still library validation, not active-viewer runtime signoff

## Mar 26, 2026 - wow-viewer Shared I/O Copilot Workflow Surface Updated

- The shared Copilot workflow surface now has an explicit non-PM4 implementation path in addition to the earlier PM4-only route.
- New shared continuation assets now live under `.github/`:
	- `.github/skills/wow-viewer-shared-io-library/SKILL.md`
	- `.github/prompts/wow-viewer-shared-io-implementation.prompt.md`
- `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md` now records the current shared `Core` or `Core.IO` source-of-truth, landed slices, validation surface, and immediate next seams.
- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md` now routes implementation-sized non-PM4 shared-format work to the dedicated shared-I/O implementation prompt instead of only the broader shared-I/O planning prompt.
- `.github/copilot-instructions.md` now explicitly covers `wow-viewer` shared I/O guardrails and first reads, so new chats can distinguish PM4, shared-I/O implementation, and broader migration planning earlier.
- Future-session workflow rule:
	- use the PM4 library prompt or skill when the ask is the next `Core.PM4` slice
	- use the shared-I/O implementation prompt or skill when the ask is the next `Core` or `Core.IO` non-PM4 format slice
	- use the broader tool-suite prompt set only when the ask is repo-shape, ownership planning, or migration sequencing
	- whenever a new `wow-viewer` skill or implementation prompt is created, update `.github/copilot-instructions.md` and `wow-viewer/README.md` in the same slice so discovery stays automatic in future chats

## Mar 26, 2026 - PM4 Coordinate-Mode Resolver Slice Landed In wow-viewer

- `wow-viewer/src/core/WowViewer.Core.PM4` now contains the next library-only PM4 solver seam adjacent to the earlier placement-solution work: typed coordinate-mode resolution.
- Landed pieces:
	- `Pm4CoordinateModeResolution`
	- `Pm4PlacementMath.ResolveCoordinateMode(...)`
	- internal coordinate-mode score evaluation that reuses the shared planar-transform resolver, footprint score, and centroid score helpers instead of leaving the tile-local versus world-space decision loop only in `WorldScene`
	- regression coverage for the fixed development tile, a synthetic world-space case, and the missing-evidence fallback path
- Validation status:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `11` placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `18` tests
- Important boundary:
	- this is still a `wow-viewer` library slice only; no new active-viewer consumer wiring was added here
	- the active viewer still owns the current coordinate-mode call site until a later narrow consumer slice explicitly re-homes it

## Mar 26, 2026 - wow-viewer Bootstrap And Non-PM4 Core Follow-Up

- The current concern about `wow-viewer` drifting into PM4-only work was valid.
- Verified repo state before correction:
	- `WowViewer.Core`, `WowViewer.Core.IO`, and `WowViewer.Core.Runtime` were still mostly placeholders
	- `libs/` was empty
	- `scripts/bootstrap.ps1` and `scripts/bootstrap.sh` were literal placeholders
- Corrective slice now landed:
	- `wow-viewer/scripts/bootstrap.ps1` and `wow-viewer/scripts/bootstrap.sh` now clone the baseline upstream repos called out in the migration draft
	- `wow-viewer/src/core/WowViewer.Core` now contains first non-PM4 chunk primitives: `FourCC` and `ChunkHeader`
	- `wow-viewer/src/core/WowViewer.Core.IO` now contains `ChunkHeaderReader`
	- `wow-viewer/tests/WowViewer.Core.Tests` now locks the FourCC and chunk-header boundary behavior
- Important boundary:
	- this only starts the non-PM4 shared-core path; it does not mean the broader map, object, terrain, WMO, model, texture, or runtime library families are migrated yet
	- the next corrective slices should target shared I/O ownership, not just more PM4 seams

## Mar 25, 2026 - Post-v0.4.5 Branch And Roadmap Prompt Bundle

- Post-release planning is now intentionally split onto branch `feature/v0.4.6-v0.5.0-roadmap` so the next milestone work can stay isolated from `main` until the first real slices are ready.
- Detailed Copilot prompt assets for the `wow-viewer` tool-suite/library refactor now live under workspace `.github/prompts/`, not under `gillijimproject_refactor/plans`.
- For this tool-suite migration work, treat `gillijimproject_refactor/plans` as scratchpad/archeology notes and `.github/prompts/` as the canonical prompt surface.
- Current dedicated prompt set:
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-bootstrap-layout-plan.prompt.md`
	- `.github/prompts/wow-viewer-shared-io-library-plan.prompt.md`
	- `.github/prompts/wow-viewer-tool-inventory-cutover-plan.prompt.md`
	- `.github/prompts/wow-viewer-cli-gui-surface-plan.prompt.md`
	- `.github/prompts/wow-viewer-tool-migration-sequence-plan.prompt.md`
- New prompt bundle captured under `plans/` for the next branch of work:
	- `post_v0_4_5_plan_set_2026-03-25.md`
	- `v0_4_6_v0_5_0_roadmap_prompt_2026-03-25.md`
	- `wowrollback_uniqueid_timeline_prompt_2026-03-25.md`
	- `alpha_core_sql_scene_liveness_prompt_2026-03-25.md`
	- `viewer_performance_recovery_prompt_2026-03-25.md`
	- `v0_5_0_new_repo_library_migration_prompt_2026-03-25.md`
	- `v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`
- Current intended milestone split:
	- `v0.4.6` should carry the first visible WoWRollback / `UniqueID` timeline filter slice inside the active viewer, plus Alpha-Core SQL caching/fidelity follow-up and an initial performance recovery pass.
	- `v0.5.0` should move into `https://github.com/akspa0/wow-viewer` as the new production repo with one canonical shared library plus split viewer/tool consumers.
- Important boundaries for future sessions:
	- keep WoWRollback integration on the active viewer UI/data-loading path; do not drift back to the older separate web-viewer plan as the primary delivery target.
	- treat `parp-tools` as the R&D / archaeology repo and `wow-viewer` as the intended production home for the next major milestone.
	- external constructive guidance now explicitly supports a sane top-level `wow-viewer` layout: the main renderer app should have one obvious root, with libraries/dependencies/tools split into their own clear folders instead of repeating the current nested sprawl.
	- latest user constraint: fully refactor and re-own the first-party read/parse/write/convert stack, including current base libraries such as `gillijimproject-csharp`; keep upstream projects like `Warcraft.NET`, `DBCD`, `WoWDBDefs`, `Alpha-Core`, `WoWTools.Minimaps`, and `SereniaBLPLib` under `libs/` and track their original repos where practical.
	- repo bootstrap should automatically pull support repos like `wow-listfile` instead of relying on manual setup.
	- possible targeted integrations worth evaluating later include `MapUpconverter`, `ADTMeta`, `wow.export`, and `wow.tools.local`, but they should support the owned-library plan rather than replace it.
	- possible future upstream work on `Noggit` / `noggit-red` alpha-era support is interesting, but should stay an explicit stretch/outreach track rather than replacing the main `wow-viewer` migration target.
	- a concrete first-pass repo tree and migration order draft now exists in `plans/v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`; future planning should refine that draft rather than re-deriving repo shape from scratch.
	- treat Alpha-Core SQL equipment correctness, animation-state handling, and pathing as separate seams.
	- do not assume SQL or PM4 already prove server-like NPC pathing; that remains a later research seam, not an implicit short-term deliverable.
	- performance recovery is now a first-class dependency, but the deeper overhaul should be planned against the new repo/library split instead of indefinite surgery inside the R&D tree.
- Documentation follow-up on the same slice:
	- root `README.md` was refreshed again to make the active support headline, conversion coverage, WMO `v14/v16/v17` handling, and built-in tooling more explicit.
	- screenshot reality remains unchanged: asset-catalog screenshot automation exists already, but a curated world/UI gallery is still future work.
- Validation status:
	- planning/documentation only
	- no viewer, converter, or renderer code changed in this slice

## Mar 24, 2026 - WMO Vertex-Light Prototype In Active Viewer

- First renderer-side object-lighting prototype is now in the active tree at `src/MdxViewer/Rendering/WmoRenderer.cs`.
- Scope of the implementation:
	- WMO group vertex buffers now carry a fourth attribute for baked vertex-light color.
	- `WmoRenderer` now prefers parsed `MOCV` vertex colors when they look usable.
	- if usable `MOCV` is missing but preserved v14 lightmap payloads exist (`MOLV` / `MOLD` / `MOLM`), the renderer now samples those on load into per-vertex baked-light modulation colors.
	- the fragment shader now modulates the existing diffuse/fog path by that baked-light color, so WMOs can show preserved object-light contribution instead of relying only on the generic ambient+directional path.
- Important limit:
	- this is not full `0.5.3` / early-client object-lightmap parity.
	- there is still no client-faithful group/batch lightmap texture pipeline, no recovered batch-to-lightmap index path, and no dedicated `RenderGroupLightmap` / `RenderGroupLightmapTex` analogue in the active renderer.
	- this is a first prototype using the data the active model already preserves.
- Validation status:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed on Mar 24, 2026 after the change.
	- no automated tests were added or run.
	- no real-data runtime signoff yet on affected WMOs.

## Mar 24, 2026 - 0.5.3 Terrain/Object Render Fast-Path And Viewer Perf Gap

- Reverse-engineering follow-up against the symbolized `0.5.3` client materially tightened the current performance/parity story; no viewer code changed in this slice.
- durable write-up extended in `documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
- high-confidence `0.5.3` terrain findings from decompilation:
	- `CreateRenderLists` (`0x00698230`) is a real precompute step that builds terrain texcoord tables and batch/render-list data instead of leaving chunk draw setup entirely to the frame loop
	- `RenderLayers` (`0x006a5d00`) and `RenderLayersDyn` (`0x006a64b0`) use locked GX buffers plus prebuilt chunk batches, not a fully generic per-layer rebuild path
	- terrain already has shader-assisted paths in `0.5.3`: the chunk draw path binds `CMap::psTerrain` / `CMap::psSpecTerrain` plus `shaderGxTexture` when terrain/specular shader support is enabled
	- terrain layer count is reduced by distance (`textureLodDist` can clamp the runtime draw to one layer), and the dynamic path also fades diffuse alpha before collapse
	- per-layer moving-texture behavior is confirmed in the terrain path itself: when runtime layer flag `0x40` is set, `RenderLayers` / `RenderLayersDyn` apply an extra texture transform indexed by low flag bits into the time-varying world transform tables updated by `FUN_006804b0`
	- terrain shadows are drawn as a separate modulation pass rather than being flattened into one generic terrain blend loop
- high-confidence `0.5.3` object/light findings from decompilation:
	- `RenderMapObjDefGroups` (`0x0066e030`) walks visible `CMapObjDefGroup` lists, sets transforms once per group, and dispatches `CMapObj::RenderGroup(...)`; this is more structured than the active viewer's generic instance loops
	- `CreateLightmaps` (`0x006adba0`) allocates per-group lightmap textures (`256x256`) and registers `UpdateLightmapTex`, which strongly supports a dedicated object-lightmap path in the client
	- `RenderGroupLightmap(...)` uses dedicated group lightmap vertex streams and batch-local lightmap texture binding rather than one generic object UV/material path
	- `RenderGroupLightmapTex(...)` splits the lightmap composition work into dedicated subpasses with lighting forced off, and `UpdateLightmapTex(...)` exposes row-stride plus CPU memory on `GxTex_Latch`; taken together, the object lightmap path is a real rendering subsystem, not just a texture on the generic WMO path
	- `CalcLightColors` (`0x006c4da0`) computes a much richer lighting state than the active viewer currently models: direct, ambient, six sky channels, five cloud channels, four water channels, fog end, fog-start scalar, and storm blending
- viewer-side implication from the same slice:
	- the active viewer remains structurally flatter than the client in the exact places that matter for both performance and fidelity:
		- `StandardTerrainAdapter` still actively uses `MPHD` only for big-alpha/profile selection and still flattens `MAIN` entries to boolean tile existence
		- `TerrainRenderer` is still a generic base+overlay pass loop that only interprets `MCLY 0x100`; it has no terrain shader-family split, no per-layer motion support, no layer-count LOD collapse, and no specular terrain path
		- `LightService` remains a simplified nearest-zone DBC interpolator rather than a full terrain/object/sky/runtime-light system
		- `WmoRenderer` / `MdxRenderer` still rely on shared generic shader families instead of the client's stronger specialization
		- `WorldScene` hot paths remain heavy: MDX transparent items are re-collected/sorted every frame, optional PM4 forensic budgets are still `int.MaxValue`, and the current render-queue abstraction is not yet the active world submission path
- practical priority order now supported by evidence:
	1. preserve `MAIN` / `MPHD` / `MCLY` semantics as first-class runtime metadata
	2. split terrain renderer responsibilities into fallback vs client-faithful material/shader path
	3. treat object/lightmap parity as a separate seam from terrain lighting
	4. reduce generic hot-path state churn before layering on more fidelity features
	5. use the existing `WorldAssetManager` read/path-probe counters as the basis for an explicit scene residency/prefetch policy
- validation status:
	- reverse engineering plus code audit only; no viewer build or runtime signoff was produced by this slice

## Mar 24, 2026 - WoW 2.0.0 Beta Ghidra Recon For M2 / Light / Particle Risk

- Static reverse-engineering pass only against a loaded beta `2.0.0` `WoW.exe` in Ghidra. No viewer/converter code changed in this slice.
- durable write-up: `documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
- High-confidence findings from decompilation:
	- `Model2` has an explicit BLS shader bootstrap in `FUN_00717b00` (`M2Cache.cpp` path string present) and loads both `shaders\vertex\Model2.bls` and `shaders\pixel\Model2.bls`.
	- map objects preload a dedicated bank of pixel BLS programs in `FUN_006b3b20`, including `MapObjOverbright`, `MapObjSpecular`, `MapObjMetal`, `MapObjEnv`, `MapObjEnvMetal`, `MapObjExtWater0`, `MapObjTransDiffuse`, and `MapObjTransSpecular`.
	- `M2Light.cpp`-anchored logic in `FUN_0072d1a0` does not treat model lights as a flat passive list: lights are inserted either into a spatial bucket structure or a general linked list depending on runtime mode/type, and companion mutators (`FUN_0072cc60`, `FUN_0072cc90`, `FUN_0072cdc0`) relink them when state/position changes.
	- particle runtime is a real engine-side system, not just file payload playback: `FUN_007c26c0` bootstraps `CParticleEmitter2_idx` and global pools, while `FUN_007ca9d0` / related constructors copy emitter payload regions into runtime `CParticle2` / `CParticle2_Model` objects.
	- the `Light*.dbc` family is loaded through strict `WDBC` schema-checked table loaders with ID-index maps, not ad-hoc parsing. Confirmed table shapes:
		- `LightFloatBand.dbc` and `LightIntBand.dbc`: `0x22` columns, `0x88` row size, two `0x40`-byte band payloads plus two leading scalars.
		- `LightParams.dbc`: `9` columns, `0x24` row size.
		- `Light.dbc`: `0xc` columns, `0x30` row size with a trailing `0x14`-byte block.
		- `LightSkybox.dbc`: `2` columns, `8` byte rows with string-table resolution.
- Practical viewer risk guidance from this RE pass:
	- do not collapse early/later `2.x` materials into one generic shader path if the goal is parity; the client uses distinct BLS programs for `Model2` and multiple map-object material families.
	- do not expect smoke / particle projection issues to close from parser tweaks alone; the particle and light systems are runtime-managed and likely need render-path/state investigation in addition to format parsing.
	- terrain follow-up is now split into two separate engine tracks:
		- cached per-layer terrain programs are now pinned down more precisely:
			- `terrain1..4` at `DAT_00caf304..310` are the one-pass layer-count table used when `DAT_00cb3594 == 0` and `DAT_00ca31b8 != 0`
			- `terrain1_s..4_s` at `DAT_00caf548..554` are the alternate one-pass layer-count table used when `DAT_00cb3594 != 0`
			- `terrainp` / `terrainp_s` belong to the slower manual terrain fallback path in `FUN_006cee30`, not the cached layer-count table
			- `terrainp_u` / `terrainp_us` are loaded at startup but are still untraced in an active draw branch
			- terrain also has a separate time-varying layer-transform path: `FUN_006c00f0` copies a source layer flag field into each runtime layer object, `FUN_006cee30` / `FUN_006cf590` apply an extra transform when bit `0x40` is present, and `FUN_006804b0` updates the transform tables every world tick
		- `XTextures\slime\slime.%d.blp` resolves into an animated `WCHUNKLIQUID` surface path, not yet proven to be a terrain diffuse-layer effect
		- latest `WCHUNKLIQUID` pass shows a real mode dispatcher: `FUN_006c65b0` splits modes `0/4/8` into animated texture-family rendering and modes `2/3/6/7` into a direct-coordinate/UV-style path
		- `FUN_006c65b0` passes the raw mode nibble into `FUN_0069b310`, so the liquid mode is also the animated family index
		- currently recovered family table entries:
			- `0 -> lake_a`
			- `1 -> ocean_h`
			- `2 -> lava`
			- `3 -> slime`
			- `4 -> lake_a` again
		- novelty/dead-content candidates:
			- `FUN_0069e690(2)` currently reaches `FUN_0069b310(6)`, but the family slot is still unresolved via data xrefs
			- `XTextures\river\fast_a.%d.blp` exists in strings but is not in the traced active family table
	- viewer-side audit against the active tree shows terrain flag under-parsing is real:
		- `StandardTerrainAdapter` currently uses `MPHD` only for big-alpha selection
		- `ReadMainChunk(...)` treats any non-zero `MAIN` entry as generic tile presence instead of keeping entry semantics like `has ADT` vs `all water`
		- raw `MCLY` flags are preserved into `TerrainLayer.Flags`, but `TerrainRenderer` only interprets `0x100` as the implicit-alpha hint
	- the dangerous seam for `2.x` support is downstream interpretation of light/material/particle IDs and runtime state, not raw DBC ingestion.
- Validation status:
	- reverse engineering only; no automated tests, no solution build, and no runtime real-data signoff were performed in this slice.

## Mar 24, 2026 - 0.12 Standalone Model Browser Recovery

- The latest standalone-model regression for the `0.12` client split into two separate seams in the active viewer:
	- `MpqDataSource` was no longer indexing Alpha-style nested model wrappers at all (`.mdx.MPQ`, `.mdl.MPQ`, `.m2.MPQ`), and it also skipped loose `.mdl` files entirely.
	- standalone `MD20` / `MD21` routing in `ViewerApp.LoadM2FromBytes(...)` still allowed an unsupported build with no resolved `M2Profile` to continue into the M2-family adapter path instead of failing cleanly.
- Root cause now fixed in the active tree:
	- `src/MdxViewer/DataSources/MpqDataSource.cs`
		- loose-file indexing now includes `.mdl`
		- Alpha nested wrapper scan now includes model wrappers (`.mdx.MPQ`, `.mdl.MPQ`, `.m2.MPQ`)
		- model wrappers now register extension aliases into the file set / Alpha wrapper cache so the browser and path resolver can find the same wrapped asset through `.mdx`, `.mdl`, or `.m2`
	- `src/MdxViewer/ViewerApp.cs`
		- the standalone browser's `.mdx` filter now aggregates early model files from both `.mdx` and `.mdl`
		- disk loads now accept `.mdl` through the same container-probe path already used by the data-source loader
		- `LoadM2FromBytes(...)` now hard-fails with a clear unsupported-build error when no `M2Profile` resolves for the active client build instead of continuing into an unsafe best-effort adapter path
	- `src/MdxViewer/ViewerApp_Sidebars.cs`
		- the file-browser type label now reflects that the early-model bucket is `.mdx/.mdl`
- Scope boundary:
	- this fix restores file discovery/indexing and turns the unsupported `.m2` route into a safe load failure for pre-M2 builds; it is not proof that standalone `0.12` runtime model rendering is fully signed off across a real client dataset.
- Validation status:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed on Mar 24, 2026 after this fix.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on actual `0.12` client browsing/loading because no fixed `0.12` data path is currently recorded in `memory-bank/data-paths.md`.

## Mar 24, 2026 - 0.6.0 Through 2.x Terrain Alpha Grid Regression Fix

- The terrain grid-pattern regression affecting standard ADT clients from `0.6.0` through the `2.x` era was not a newly proven shader/blend-style difference. The active viewer was still decoding that whole legacy band through a naive sequential 4-bit MCAL unpack path in `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`.
- Root cause now fixed in the active tree:
	- `StandardTerrainAdapter.ExtractAlphaMaps(...)` for `TerrainAlphaDecodeMode.LegacySequential` now prefers the relaxed MCAL path (`Mcal.GetAlphaMapForLayerRelaxed(...)`) and preserves `DoNotFixAlphaMap` behavior.
	- the old naive legacy fallback now routes through the existing row-aware 4-bit decode + legacy edge-fix helpers instead of writing raw nibble pairs straight into the `64x64` output.
- Scope boundary:
	- this change is limited to the standard-terrain legacy band (`0.6.0` through `2.x`) and does not change the separate `AlphaTerrainAdapter` path for `0.5.x` or the strict `3.x` / Cataclysm `4.0.0` decode branches.
- Validation status:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed on Mar 24, 2026 after this fix and after correcting unrelated compile breaks in the in-progress minimap candidate-path patch.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on affected `0.6.0` / `0.7.0` / `0.8.0` / `0.9.0` / `1.x` / `2.x` terrain tiles.

## Mar 24, 2026 - v0.4.5 Branding + MH2O LiquidType Classification Fix

- Active viewer branding/release metadata is now aligned toward `parp-tools WoW Viewer` version `0.4.5` without renaming the `MdxViewer` root namespace.
- Current user-facing changes in the active tree:
	- viewer window title now uses `parp-tools WoW Viewer`
	- Help -> About now opens a modal with author + credits instead of only writing a transient status line
	- project metadata now emits `ParpToolsWoWViewer` as the executable/assembly name
	- `.github/workflows/release-mdxviewer.yml` now packages/releases `parp-tools-wow-viewer-<version>-win-x64.zip` and uses the .NET 10 SDK required by the active project target
- MH2O follow-up on the same slice:
	- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` now classifies `MH2O` liquids from `LiquidType.dbc -> Type` when DBC metadata is available for the active client build
	- when DBC loading is unavailable or an ID is missing from the loaded table, the viewer now falls back to an expanded static family map that includes the real 3.3.5 / 4.0 IDs already used elsewhere in the repo (`13`, `14`, `17`, `19`, `20`)
	- `src/WoWMapConverter/WoWMapConverter.Core/Formats/Liquids/LiquidConverter.cs` now recognizes those late-style IDs in the shared `LiquidTypeId -> MCLQ family` fallback path as well
- Validation status:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"` passed on Mar 24, 2026
	- no automated tests were added or run
	- no runtime real-data signoff yet on 3.3.5 / 4.0 liquid visual parity; the build only proves the implementation compiles

## Mar 25, 2026 - Fullscreen Minimap Release Blocker Closed For v0.4.5

- The fullscreen/docked minimap repair is now treated as closed for `v0.4.5` after the final transpose-only follow-up and runtime user confirmation on the fixed development minimap dataset.
- Final landed behavior in the active tree:
	- the bad `WoWConstants.TileSize` minimap hypothesis stays reverted; the active `64x64` minimap grid continues to use `WoWConstants.ChunkSize`
	- the broad world-axis swap attempted during the first Designer Island follow-up was backed out
	- the landed fix instead keeps the direct world/click mapping and only transposes the screen-space marker placement seam that had drifted away from the drawn tile grid
	- docked and fullscreen minimap now agree well enough for the user to describe the bug as fixed after runtime checking the top-right Designer Island scenario
- Practical release consequence:
	- the fullscreen minimap is no longer an open `v0.4.5` blocker
	- remaining minimap work should be treated as future polish or new regressions, not as justification to keep `v0.4.5` open
- Validation status:
	- build plus targeted runtime user signoff: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-transpose-repair/"` passed on Mar 25, 2026 after the final transpose-only repair
	- runtime user feedback then confirmed the repaired minimap behavior on the fixed development minimap dataset
	- no automated tests were added or run
	- this is not broad automated minimap coverage; it is targeted real-data runtime confirmation for the previously broken release-blocker scenario

## Current Focus: v0.4.0 Recovery Branch (Mar 17, 2026)

Working branch is now reset in the main tree, not only in side worktrees.

- Branch: recovery/v0.4.0-surgical-main-tree
- Baseline tag/commit: v0.4.0 / 343dadf
- .github metadata restored from main and committed: 845748b
- .github restore was pushed to origin/recovery/v0.4.0-surgical-main-tree

### Tooling Path Reuse + Unified Format I/O Proposal (Mar 23)

- Viewer tool dialogs should stop forcing repeated folder browsing when the session already knows the active base client and loose overlay roots.
- Current viewer-side behavior now seeds tool inputs from the active session where practical:
	- `Generate VLM Dataset` pulls the active MPQ base client path and current map name.
	- `Terrain Texture Transfer` prefers the attached loose-overlay map directory as source and the base-client map directory as target when those roots exist.
	- `Map Converter` now seeds WDT/map-directory inputs from the currently loaded local WDT when available, otherwise from the current map under the active loose/base roots.
	- `WMO Converter` still seeds from the currently loaded standalone WMO when applicable.
- Important scope limit:
	- this is UI/tool input seeding only, not proof that all downstream conversion paths are correct for Alpha, LK 3.3.5, or 4.x data.
	- after the Mar 23 seeding follow-up, edited-file diagnostics were clean on `src/MdxViewer/ViewerApp.cs`, but no new full viewer build or runtime signoff was recorded yet for this slice.
- Larger project direction requested by the user:
	- consolidate terrain, ADT/WDT, M2/MDX, and WMO read/write knowledge into one shared library used by viewer, converter, and tooling instead of continuing to split capabilities across `MdxViewer` and `WoWMapConverter.Core`.
	- do not assume the existing map converter is already closed for Alpha placement writing: MODF/MDDF downconversion for Alpha WDT remains an explicit open seam until reimplemented and validated.
	- planning prompt captured in `plans/unified_format_io_overhaul_prompt_2026-03-23.md`.
	- new PM4 planning guardrail from Mar 24 viewer forensics/UI work:
		- the practical viewer hierarchy is `CK24 -> MSLK-linked subgroup -> optional MDOS subgroup -> connectivity part`
		- PM4 centroids are useful derived display anchors for those nodes, not proven raw PM4 node records
		- `MSUR.AttributeMask` colors should be surfaced as explicit value legends, but their semantics remain open and must not be hardcoded into format contracts prematurely

### Documentation Refresh + Render Quality Follow-Up (Mar 23)

- Repo-level docs were refreshed, but the first pass still contained bad assumptions.
- The user then rewrote `src/MdxViewer/README.md` to be more grounded and truthful.
- Current documentation/handoff rule:
	- treat the user-corrected viewer README as the authoritative public summary for support and usage claims
	- do not reintroduce speculative platform restrictions or inflated support statements without direct evidence
	- do not write branch-local language into README text intended for eventual `main`
- Important current README claims to preserve in future sessions:
	- support headline: `0.5.3` through `4.0.0.11927`
	- later `4.0.x` ADT support exists
	- later split-ADT support through `4.3.4` exists but remains explicitly untested
	- Alpha-Core SQL world NPC/gameobject support is relevant to the README and should not be dropped casually
	- asset-catalog screenshot automation exists already; broader UI/menu showcase capture is still future work
- Validation status:
	- docs were updated after the Mar 23 viewer build had already passed
	- the documentation update itself adds no runtime validation and should not be read as new visual signoff

### Viewer Debug/Workflow Follow-Up (Mar 22)

- Latest viewer-side work moved away from treating PM4 runtime streaming as the only inspection path.
- Current additions in the active tree:
	- PM4 offline OBJ export from `src/MdxViewer/Terrain/WorldScene.cs`, surfaced through `ViewerApp_Pm4Utilities.cs`, so per-tile/per-object PM4 geometry can be compared outside the live overlay window.
	- minimap interaction/caching follow-up in `ViewerApp_MinimapAndStatus.cs`, `ViewerApp.cs`, and `Rendering/MinimapRenderer.cs`:
		- teleport now requires triple-clicking the same tile instead of a single short click
		- minimap zoom/pan/window state now persist in viewer settings
		- decoded minimap tiles now cache on disk under `output/cache/minimap/<cache-segment>`
	- terrain-hole debug override in `TerrainMeshBuilder`, `TerrainManager`, `VlmTerrainManager`, and `ViewerApp_Sidebars.cs`:
		- viewer can ignore terrain hole masks globally or on the current camera tile by rebuilding loaded chunk meshes only
		- source ADT hole flags are unchanged; this is viewer-side inspection only
- Validation status:
	- file diagnostics were clean on the edited viewer files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 22, 2026 after these viewer-side follow-ups were in the active tree
	- no automated tests were added or run
	- no runtime real-data signoff yet on PM4 OBJ correctness, minimap feel/cache benefit, or terrain-hole rebuild behavior while streaming

### Standalone PM4 Research Library (Mar 21)

- Added a new isolated project at `src/Pm4Research.Core` for fresh PM4 format work outside the current viewer/converter reconstruction path.
- Current scope of that library:
	- raw chunk walking with preserved signatures, offsets, sizes, and payload bytes
	- standalone typed decoding for `MVER`, `MSHD`, `MSLK`, `MSPV`, `MSPI`, `MSVT`, `MSVI`, `MSUR`, `MSCN`, `MPRL`, `MPRR`, `MDBH`, `MDBI`, `MDBF`, `MDOS`, and `MDSF`
	- lightweight exploration snapshot generation for counts and chunk bounds
	- raw decode-audit reporting for per-file and corpus-wide chunk consistency and cross-chunk reference checks
- Important boundary:
	- no viewer/world transform policy
	- no CK24 object reconstruction
	- no dependency on `MdxViewer` PM4 solver code or the current `WoWMapConverter.Core` PM4 models
- Preferred real-data reference tile for PM4 rediscovery:
	- use `test_data/development/World/Maps/development/development_00_00.pm4` first when checking raw chunk assumptions or viewer-forensics hypotheses
	- Mar 21 standalone analysis on that tile showed it is a dense PM4 file, not a degenerate edge case: `54` chunks, `MSPV=8778`, `MSVT=6318`, `MSCN=9990`, `MPRL=2493`
	- new Mar 21 audit result: `00_00` is also the only currently populated destructible-building payload tile in the in-repo development PM4 corpus; `MDBI` and `MDBF` are one-tile only, while `MDBH` / `MDOS` / `MDSF` mostly appear as empty or placeholder stubs elsewhere
	- the matching original ADTs are not present in this repo, so in-repo validation is currently PM4-side only; external visual cross-checks should still prefer this tile because the user has the trusted ADT placements for it
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Core/Pm4Research.Core.csproj -c Debug` PASSED on Mar 21, 2026.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and found zero file-walk/stride diagnostics across the 616-file corpus, but did surface `MSLK.RefIndex -> MSUR` mismatches in aggregate and the Wintergrasp-only destructible payload split described above.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and narrowed that open seam further: `150` files carry `4553` mismatches, `development_00_00.pm4` carries zero mismatches, and the bad values almost never fit `MPRL` counts but often still fit `MSLK`, `MSPI`, `MSVI`, and `MSCN` counts on the affected tiles.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-linkage --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and materially tightened the identity/hierarchy seam: the UI `Ck24ObjectId` is just the low 16 bits of `MSUR.PackedParams -> CK24`, it is almost always one-to-one with a full CK24 within a file (`2` reuse cases out of `1601` analyzed non-zero object-id groups), and `MSLK.GroupObjectId` remains very weak as the missing hierarchy/ownership key for the unresolved `RefIndex` population (`16` low16 matches and `15` low24 matches across `4553` mismatches).
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mscn --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and materially tightened the MSCN seam: `MSUR.MdosIndex -> MSCN` is strong (`511891` fits, `6201` misses), `1886 / 1895` CK24 groups carry MSCN coverage, and in the standalone raw path raw MSCN bounds overlap CK24 mesh bounds far more often than swapped-XY MSCN bounds (`1162` vs `10` fits). Current standalone corpus evidence does not support the older blanket claim that MSCN is simply world-space plus XY swap.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-msur-geometry --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_msur_geometry_report.json` PASSED on Mar 21, 2026 and materially tightened a major decoder-trust seam: all `518092` analyzed `MSUR` surfaces had unit-length stored normals with strong positive alignment to geometry-derived polygon normals, and the trailing float currently named `Height` behaves like the negative plane-distance term along that normal (best candidate mean absolute error `0.00367829`).
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex-classifier --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_mslk_refindex_classifier_report.json` PASSED on Mar 21, 2026 and replaced the old all-or-nothing mismatch story with family buckets: `505` mismatch families are now classified beyond pure ambiguity, covering `2651` of `4553` mismatch rows, with the largest resolved family population currently landing in `probable-MSVT` plus smaller `MSPI` / `MSPV` / `MSVI` / `MSCN` / `MPRL` slices.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-structure-confidence --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and is now the explicit decode-trust guardrail for the standalone PM4 path: `13` tracked chunk families currently land in `high` layout confidence, but field semantics are much weaker (`1` high, `4` medium, `10` low, `4` very-low). The main hallucination-risk zone is semantic over-closure, not raw stride parsing.
	- refreshed `scan-structure-confidence` result after the new audits: field semantics are still weaker than layout confidence, but the picture improved materially (`2` high, `4` medium, `9` low, `4` very-low). Current highest-risk zones are `MSLK.RefIndex`, `MPRR.Value1`, `MPRL.Unk04/14/16`, and sparse destructible fields; `MSUR` bytes `4..19` are no longer in that top-risk bucket.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- pm4-validate-coords --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --json i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_pm4_coordinate_validation_report.json` PASSED on Mar 21, 2026 and materially strengthened `MPRL` against real placement truth on the fixed dataset: `206` tiles validated, `114301 / 114301` refs inside expected tile bounds (`100.0%`), `107907 / 114301` refs within `32` units of a nearest `_obj0.adt` placement (`94.4%`), average nearest placement distance `10.98`. This helps `MPRL`, not `MPRR`.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-unknowns --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` PASSED on Mar 21, 2026 and now serves as the main corpus-scale PM4 unknowns map: it records verified raw edges, partial fits, field distributions, and open proof tasks in one place.
	- structure-confidence highlights to preserve for future PM4 work:
		- strongest byte+semantic anchors: `MSPV`, `MSPI`, `MSVT`, `MSVI`, `MSUR` plane fields, `MSUR -> MSVI`, and `MDSF -> {MSUR, MDOS}`
		- highest hallucination-risk fields: `MSLK.RefIndex`, `MPRR.Value1`, `MPRL.Unk04/14/16`, and sparse destructible payload fields such as `MDOS.buildingIndex`
		- explicit conflict inventory now exists for overstated legacy claims around `MSLK.LinkId`, `MSLK.RefIndex`, `MSUR.MdosIndex`, `MSUR.Normal + Height`, MSCN coordinate frame, and `MPRR.Value1`
	- no automated tests were added or run.
	- no real-data runtime signoff exists yet because this is a standalone decode/exploration foundation, not an integrated viewer fix.

### M2 Material Parity Slice: Explicit Env-Map + UV Selector Recovery (Mar 21)

### Archive I/O Performance Slice: Read-Path Probe Reduction + Useful Prefetch Instrumentation (Mar 21)

### ViewerApp Partial-Class Refactor (Mar 21)

- `src/MdxViewer/ViewerApp.cs` was reduced by extracting cohesive UI domains into partial-class files instead of doing a behavior rewrite:
	- `src/MdxViewer/ViewerApp_ClientDialogs.cs`
	- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- `src/MdxViewer/ViewerApp_Sidebars.cs`
- The goal of this slice is maintainability only: keep existing viewer behavior while shrinking the single 6000+ line shell file and making future UI changes more localized.
- Current limit of the extraction:
	- the large world-objects body still lives behind `DrawWorldObjectsContentCore()` in `ViewerApp.cs`; the refactor did not attempt a full inspector redesign in this pass.
- Validation status for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026 after the split.
	- no automated tests were added or run.
	- no runtime real-data validation was done because this change is structural, not a terrain/data-path behavior fix.

### Viewer UI / Perf Slice: Hideable Chrome + Clipped Long Lists (Mar 21)

### Viewer UI Follow-Up: Dockspace Host + Dockable Side Panels (Mar 21)

### Viewer PM4/WMO Correlation Export (Mar 21)

- `MdxViewer` now exposes a viewer-side PM4/WMO correlation export in the existing `PM4 Alignment` window.
- Current implementation:
	- `ViewerApp_Pm4Utilities.cs` adds `Dump PM4/WMO Correlation JSON` next to the existing PM4 object dump.
	- `WorldScene.BuildPm4WmoPlacementCorrelationJson(...)` exports loaded ADT WMO placements, parsed WMO mesh summaries, and top nearby PM4 overlay object candidates per placement.
	- `WorldAssetManager` now exposes `WmoMeshSummary`, reusing the existing WMO v14/v17 parsing path to capture local bounds plus group/vertex/index/triangle counts without depending on a renderer instance.
- Scope / limit:
	- this is a correlation/export utility, not closure on PM4-to-WMO semantic identity.
	- current matching is still heuristic, but it is no longer AABB-only: ranking now uses transformed WMO footprint samples versus PM4 footprint hulls in addition to bounds-gap / overlap metrics and PM4 object metadata.
- Follow-up now landed on top of the export path:
	- `ViewerApp_Pm4Utilities.cs` now adds a real `PM4/WMO Correlation` window with refresh/filter controls, placement browsing, candidate inspection, PM4 selection, and camera framing actions.
	- `WorldScene` now exposes a typed PM4/WMO correlation report for viewer use instead of forcing the UI to go through JSON only.
	- `WorldScene.SelectPm4Object(...)` lets the panel drive live PM4 selection from a reported candidate row.
	- `WorldAssetManager.WmoMeshSummary` now caches sampled WMO geometry points so the correlation path can compare transformed footprint shape instead of only transformed bounds.
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026 after the interactive panel + footprint follow-up, with existing warnings.
	- no automated tests were added or run.
	- no runtime real-data signoff was performed yet for the new panel workflow or the footprint-based ranking changes.

- Latest user feedback after the clipped-list shell pass: `World Maps` starting collapsed was wrong, and the viewer still did not have a real dock-panel UI.
- Current correction in `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- ImGui docking is now explicitly enabled in source instead of relying on stale layout state in `imgui.ini`.
	- the viewer now creates a real central dockspace host between the menu/toolbar region and the status bar.
	- the old fixed left/right sidebars can now render as normal dockable windows (`Navigator` and `Inspector`) when dock panels are enabled from the `View` menu.
	- `World Maps` now defaults open again on first draw.
	- scene viewport math no longer subtracts fixed sidebar widths, which was incompatible with docked/floating panels.
- Validation status for this follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on the docking workflow or interaction feel; do not over-claim the UI recovery from build success alone.

- Latest user priority shifted from PM4 transform tuning to viewer usability and frame-time friction while debugging PM4.
- Current implementation in `src/MdxViewer/ViewerApp.cs` is intentionally incremental, not a dockspace/UI-shell rewrite:
	- `Tab` now toggles a hide-chrome mode for the menu bar, toolbar, sidebars, status bar, and floating utility windows while keeping modal dialogs available.
	- left/right sidebar sections no longer all default open on first draw; the shell now starts less expanded by default.
	- large UI lists now use clipped child-list rendering instead of drawing every row every frame:
		- file browser
		- discovered maps
		- subobject/group visibility toggles
		- WMO / MDX placement lists
		- POI / taxi node / taxi route lists
- Scope / limit of this slice:
	- this reduces known UI hot spots and improves focus-mode usability, but it is not a full restoration of the older dockable UI and not proof yet of runtime frame-time recovery on the fixed development dataset.
- Validation status for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on actual UI responsiveness or PM4-debugging flow; do not over-claim the perf impact from build success alone.

- Confirmed hot seam on the active viewer path:
	- `WorldAssetManager.ReadFileData(...)` was still issuing repeated alias/fallback `ReadFile(...)` probes on top of `MpqDataSource`, including duplicate lowercase and `.mpq` retries that the MPQ data source already handled internally through case-insensitive normalization and Alpha wrapper resolution.
	- `MpqDataSource` had a raw-byte cache and worker prefetch path already, but it did not expose exact counters for direct read cache behavior, resolution source, or prefetch queue latency.
- Current implementation change:
	- `MpqDataSource` now exposes precise archive-I/O counters through `MpqDataSourceStats`:
		- `FileExists` request/cache/source counters
		- `ReadFile` request/cache/source counters (`loose`, `alpha wrapper`, `MPQ`, `miss`)
		- average uncached read latency
		- prefetch enqueue/dedup/cache-skip/completion counters plus average queue-wait and worker-read latency
	- `WorldAssetManager` now exposes `WorldAssetReadStats` and caches the winning resolved asset path per requested model/WMO read so later retries can jump straight to the known-good candidate instead of replaying the whole fallback chain.
	- Redundant work removed from the active world-asset path:
		- removed duplicate lowercase retry in `WorldAssetManager.ReadFileData(...)`
		- removed duplicate `.mpq` retry there for Alpha wrapper reads because `MpqDataSource.ReadFile(...)` already resolves the wrapper path directly
		- deduped candidate enumeration before trying alternates / stripped-filename / prefixed fallbacks
	- Prefetch policy is now narrower and more scene-aligned:
		- prefetch uses the canonical resolved model path first
		- if that canonical path is known, it no longer fans out across all extension aliases
		- M2 prefetch now warms the best resolved `.skin` path first and only falls back to generic skin candidates when no indexed best match exists
	- Viewer terrain/world stats panel now surfaces both `WorldAssetManager` probe counters and `MpqDataSource` cache/prefetch counters for runtime measurement.
- Validation status for this slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this slice.
	- no runtime real-data validation has been run yet on fixed MPQ-era data; do not claim generalized scene-streaming improvement from build success alone.

- The active M2-family renderer gap was confirmed to be material-state flattening inside `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`, not missing shader hooks in `ModelRenderer` first.
- Current landed slice recovers one explicit source seam instead of adding new transparency heuristics:
	- M2 skin batch metadata now preserves `textureCoordComboIndex` from raw `.skin` data and merges it back into the Warcraft.NET-derived skin path.
	- raw `MD20` vertex decode now preserves both UV sets instead of dropping everything to the first texture coordinate pair.
	- `textureCoordCombos` lookup now drives `MdlTexLayer.CoordId`; lookup value `-1` now marks the layer as `SphereEnvMap`, and lookup value `1` can select UV1 where present.
	- `ModelRenderer` now emits focused debug traces showing pass + resolved material family for M2-adapted batches when MDX debug focus is enabled.
- Scope of this slice:
	- improved family: reflective / env-mapped M2 surfaces, plus UV1-routed layers that were previously flattened to UV0
	- unchanged gaps: texture transform animation, color/transparency tracks, broader per-batch shader/material combo parity, and any runtime sorting issues beyond the existing pass split
- Validation status for this exact slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this slice
	- no runtime real-data signoff yet on reflective/env-mapped assets; do not claim PM4 matching benefit from this change alone

### M2 Material Parity Follow-Up: 4.0.0.11927 Wrap + Blend Correction (Mar 21)

- Follow-up runtime triage on Cataclysm-era M2 assets found two concrete material-state mismatches after the env-map / UV recovery slice:
	- `ModelRenderer` was only treating `WrapWidth` / `WrapHeight` as M2 repeat flags for the pre-release `3.0.1` profile, leaving later M2 builds on the old classic-MDX clamp interpretation.
	- `WarcraftNetM2Adapter.MapBlendMode(...)` was shifted after mode `2`, so M2 blend ids `4`..`7` were routed into the wrong local material families.
- Current correction:
	- all M2-adapted models now interpret wrap X/Y as repeat flags; classic MDX keeps the legacy clamp-flag behavior.
	- M2 blend ids now map as `0=Load`, `1=Transparent`, `2=Blend`, `3=Add` (`NoAlphaAdd`), `4=Add`, `5=Modulate`, `6=Modulate2X`, `7=AddAlpha` (`BlendAdd`).
	- the local renderer still has no distinct `NoAlphaAdd` or `BlendAdd` states, so those cases are now collapsed intentionally into the nearest additive families instead of landing there because of an off-by-one bug.
- Validation status for this exact follow-up slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this slice
	- no runtime real-data signoff yet on `4.0.0.11927` M2 assets; do not claim visual parity from build success alone

### PM4 Orientation Follow-Up: World-Space Solver No Longer Forces Mirrored Swap Fits (Mar 21)

### PM4 Link-Decode Follow-Up: Legacy `MSLK` Surface Index Defaults No Longer Leak As Real Data (Mar 21)

### PM4 MPRL Axis Contract Correction (Mar 21)

- Follow-up after comparing the active viewer path against older PM4 R&D exports and `WoWRollback/Pm4Reader` forensic notes.
- Current correction in `src/MdxViewer/Terrain/WorldScene.cs`:
	- the active viewer restores the older fixed `MSVT` viewer/world basis `(Y, X, Z)` for the common `XY+Zup` path instead of trying to recover that basis later with per-object planar heuristics.
	- axis convention is now held file-level again across CK24 groups instead of being redetected per CK24; this avoids neighboring PM4 pieces drifting into different mesh bases.
	- viewer-side `MPRL` positions are now converted to world as `(PositionX, PositionZ, PositionY)` so they line up with that restored `MSVT` basis during planar scoring, nearest-anchor comparisons, and PM4 position-ref marker rendering.
	- the previous viewer assumption that `MPRL` could be treated as ADT-style planar `X/Z`, vertical `Y` or as raw `Z/X/Y` world output is no longer the active contract.
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet that this closes the reported PM4 placement failure.

### PM4 Render-Derivation Follow-Up: Overlay Objects Now Keep An Explicit Local Frame (Mar 21)

- Follow-up after runtime evidence that PM4 mesh pieces were effectively being treated as if they were already in final placed space, which makes it too easy to conflate object-local shape with world placement.
- Current correction in `src/MdxViewer/Terrain/WorldScene.cs`:
	- `Pm4OverlayObject` now localizes its line/triangle geometry around a preserved pre-split linked-group placement anchor instead of storing only fully placed geometry.
	- each PM4 overlay object now carries a baked base placement transform that restores that anchored object-local geometry into the solved placed frame.
	- when one CK24 is split into linked-group / MDOS / connectivity-derived parts, those parts keep the original linked-group placement anchor instead of rebasing to per-fragment centers.
	- overlay rendering now applies that baked base transform first, then any global PM4 overlay transform and object-local alignment edits on top.
	- PM4 JSON export now rehydrates placed-space geometry from the baked base transform so the interchange dump still matches what the viewer is rendering.
- Scope / limit:
	- this is structural groundwork for the missing “mesh inside stable object frame” layer; it is not a claim that final PM4 natural-rotation decoding is solved.
	- the CK24 placement solve itself is unchanged in this slice.
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet that this resolves the remaining PM4 orientation mismatch.

- Runtime investigation on `test_data/development/World/Maps/development/development_00_00.pm4` found a concrete active-path bug during PM4 rotation forensics:
	- `WoWMapConverter.Core.Formats.PM4.MslkEntry` exposes `MsurIndex`, `MsviFirstIndex`, and `MsviIndexCount`
	- `WorldScene` consults `MsurIndex` when grouping/linking surfaces and linked `MPRL` refs
	- but `Pm4File.PopulateLegacyView(...)` was never populating those legacy fields, so `MsurIndex` defaulted to `0`
- Current correction:
	- legacy `MSLK` entries created from the canonical decoder now explicitly set sentinel values for the unsupported fields (`MsurIndex = uint.MaxValue`, `MsviFirstIndex = -1`, `MsviIndexCount = 0`) instead of leaking fake `0` values into the viewer
	- this keeps `WorldScene` on the existing `RefIndex` fallback path unless a real surface index is available in the future
- PM4 rotation-forensics result from `development_00_00.pm4`:
	- raw `MPRL.Unk04` values only span about `0.01° .. 22.3°` on this tile
	- treat that field as a narrow local heading/placement signal on this file, not as proven absolute object yaw for the whole placed building set
	- `Unk06` is constant `0x8000` on this tile, and `Unk16` still behaves like normal-vs-terminator entry typing
	- `Unk14` continues to look like floor/level bucketing, not pitch/roll
- Viewer debugging follow-up:
	- selected PM4-object debug info now shows linked `MPRL` normal/terminator counts, floor range, and heading min/max/mean so runtime object picks can be compared against raw PM4 placement stats directly
- Validation status for this follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026
	- no automated tests were added or run
	- no runtime signoff yet that the selected `CK24=0x421809` object now has the final correct orientation; this slice improves link-data integrity and observability first

- Use `documentation/pm4-current-decoding-logic-2026-03-20.md` as the authoritative viewer-side PM4 reconstruction contract for the active branch.
- That doc was refreshed on Mar 21, 2026 to capture the current CK24 pipeline, the tile-local versus world-space planar candidate split, and the rollback of the linked-`MPRL` center-translation experiment.

### PM4 Tile-Local Orientation Follow-Up: Quarter-Turn Swap Solve No Longer Rotates Non-Origin Tiles (Mar 21)

- Latest runtime PM4 report narrowed a second orientation seam after the world-space solver fix: tiles beyond `0_0` / `0_1` were coherently rotated about `90°` counter-clockwise while origin-adjacent tiles still aligned.
- Root cause in `src/MdxViewer/Terrain/WorldScene.cs`:
	- the quarter-turn planar transform expansion was also being offered to tile-local PM4
	- tile-local PM4 already has a fixed south-west tile basis, so per-tile `swap` solving could rotate whole non-origin tiles even when the underlying tile basis was correct
- Current correction:
	- tile-local PM4 now tests only non-swapped mirror candidates inside the established tile basis
	- tile-local PM4 world assembly now applies the file tile indices in viewer-world order (`tileY -> worldX`, `tileX -> worldY`) instead of the naive unswapped pairing that only looked right on origin tiles
	- quarter-turn `swap` candidates remain world-space only
- Validation status for this exact follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this follow-up.
	- no runtime real-data signoff yet on the non-origin tile placement/orientation case; do not claim PM4 tile closure from build success alone.

- Runtime PM4 alignment evidence showed some objects resolving to mirrored planar transforms like `swap=True, invertU=False, invertV=False`, which reverses handedness and makes stairs/ramps wind the wrong way around structures.
- Root cause in `src/MdxViewer/Terrain/WorldScene.cs`:
	- world-space PM4 candidate enumeration only tested `identity` and `swap`
	- rigid quarter-turn candidates were never considered, so some world-space objects could only be approximated by mirrored solutions
- Current correction:
	- world-space PM4 now evaluates the rigid planar set first: identity, 180 degree, +90 degree, and -90 degree basis changes
	- mirrored candidates are no longer part of the active PM4 planar solver; the viewer now stays on rigid candidates only to avoid reversed winding and opposite-facing fits
- Validation status for this exact PM4 solver slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this slice
	- no runtime real-data signoff yet on the guardtower staircase case; do not claim closure from build success alone

### PM4 Bounds Overlay Follow-Up: Per-Object PM4 Bounds Are Now Visible In-Scene (Mar 21)

### PM4 MPRL Frame Follow-Up: Linked-Center Translation Experiment Reverted (Mar 21)

- The earlier linked-`MPRL` frame experiment turned out to regress PM4 placement badly in runtime user validation.
- Latest runtime evidence also argues against the broader `MPRL` bounding-box/container paradigm itself: PM4 geometry and PM4 bounds are not conforming to that model in the viewer.
- Root cause in `src/MdxViewer/Terrain/WorldScene.cs`:
	- the viewer-side reconstruction path was translating whole CK24 groups into the linked `MPRL` world-bounds center after geometry pivot/yaw solve.
	- that shared translation was too aggressive and made PM4 alignment worse instead of better.
- Current correction:
	- the linked-center translation path was removed from `BuildPm4TileObjects(...)`.
	- CK24 rendering is back to the prior geometry-pivot path with the existing coarse yaw-correction logic.
	- this keeps the earlier `12°` suppression of small principal-axis yaw deltas, but no longer forces linked PM4 groups into an MPRL-center translation frame.
- Current interpretation:
	- user/domain correction: `MPRL` points are terrain/object collision-footprint intersections where ADT terrain is pierced by object collision geometry.
	- keep rejecting the old `MPRL` center/bounds translation experiment.
	- do not assume PM4 objects should fit inside an `MPRL` bounding box or container frame; use `MPRL` as footprint/collision reference data instead.
- Validation status for this exact follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026 with existing solution warnings only.
	- no automated tests were added or run for this follow-up.
	- no runtime real-data signoff yet on whether PM4 alignment is restored; do not claim placement closure from build success alone.

### PM4 Yaw Follow-Up: Small Principal-Axis Corrections No Longer Override Near-Correct MPRL Rotation (Mar 21)

- Latest runtime user feedback on PM4 overlay alignment: objects were no longer wildly mis-rotated, but many still looked consistently off by roughly `5..10` degrees around the vertical axis.
- Root cause narrowed in `src/MdxViewer/Terrain/WorldScene.cs`:
	- PM4 MPRL yaw decode was already being rebased and then compared against a geometry-derived principal-axis yaw.
	- the follow-up CK24 world-yaw correction stage was still applying small residual deltas (`>= 2°`), which is too aggressive for irregular object footprints and can turn "almost correct" PM4 orientation into a visible small bias.
- Current correction:
	- CK24 continuous yaw correction is now treated as a coarse recovery step only.
	- residual yaw deltas below `12°` are ignored, leaving MPRL-derived orientation authoritative for near-correct objects.
- Validation status for this exact follow-up:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- no automated tests were added or run for this follow-up.
	- no runtime real-data signoff yet after the threshold change; do not claim PM4 rotation closure from build success alone.

- Latest PM4 alignment feedback showed MPRL anchors lining up while other PM4 object extents still felt offset or nested inside the wrong container, making click-and-compare work too opaque.
- Current correction in `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp.cs`:
	- PM4 per-object bounds that were already computed for picking/culling/debug info are now rendered directly in-scene through the existing `BoundingBoxRenderer` path.
	- the PM4 alignment controls now expose a dedicated `PM4 Bounds` toggle beside `PM4 MPRL Refs` and `PM4 Centroids`.
	- selected PM4 object groups get a highlighted bounds color, and the exact selected PM4 object gets a white bounds box.
- Important scope note:
	- these bounds are currently built from the rendered PM4 object geometry (`MSVT`/`MSVI`/`MSUR` path), not from `MSCN`.
	- treat this as a visibility/debugging aid for the current PM4 reconstruction path, not proof yet that the active PM4 extents are sourced from the final correct container.
- Validation status for this exact PM4 bounds slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026 with existing solution warnings only.
	- no automated tests were added or run for this slice.
	- no runtime real-data signoff yet on PM4 bounds usefulness or on the MSCN-versus-MSVT extent question.

### MPQ Base Build Selection Recovery (Mar 21)

- The active viewer no longer relies only on `InferBuildFromPath(...)` for new MPQ loads.
- `ViewerApp` now restores explicit build selection before loading a game folder:
	- MPQ open flow now pauses on a build-selection dialog.
	- build choices come from `Terrain/BuildVersionCatalog.cs` using `WoWDBDefs/definitions/Map.dbd` when available, with a built-in fallback list that includes `4.0.0.11927` and `4.0.1.12304`.
	- path/build tokens are now treated as preselection hints, not authoritative routing.
- Known-good base-client entries now persist `BuildVersion` in viewer settings and reuse it when reopening a saved base or attaching a loose overlay against that base.
- Loose overlay attach now emits a PM4 build hint when the overlay contains PM4 files with known version markers:
	- `12304` => `4.0.1.12304`
	- `11927` => `4.0.0.11927`
	- if that hint disagrees with the active base build, the viewer logs a warning instead of silently continuing with no build-era signal.
- Validation status for this build-routing slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- No automated tests were added or run for this slice.
	- No runtime real-data signoff yet on PM4/world-object matching with a `4.0.1.12304` base client.

### 4.0.0.11927 Terrain Blend Recovery (Mar 21)

- The earlier working assumption that 4.0 terrain texturing was effectively "3.3.5 MCAL decode with split files" is now documented as incomplete.
- Latest wow.exe RE confirms the missing behavior is runtime blend assembly, not only local MCAL byte decode:
	- `CMapChunk_UnpackChunkAlphaSet` stitches the current chunk with three linked neighbor chunks.
	- Neighbor alpha is matched by texture id, not only by local overlay slot index.
	- In 8-bit mode, layers without direct alpha payload can be synthesized as residual coverage `255 - other layer alphas`.
	- Blend textures are rebuilt through the `TerrainBlend` resource path (`CMapChunk_BuildSingleLayerBlendTexture`, `CMapChunk_BuildChunkBlendTextureSet`, `CMapChunk_RefreshBlendTextures`).
- Active viewer implementation now reflects the first verified slice of that model:
	- `FormatProfileRegistry.AdtProfile40xUnknown` routes to `TerrainAlphaDecodeMode.Cataclysm400`.
	- `StandardTerrainAdapter` captures per-layer source flags, synthesizes residual 8-bit alpha for missing direct payloads, and stitches same-tile chunk edges by matching neighbor layer texture ids.
	- `TerrainChunkData` now preserves `AlphaSourceFlags` for runtime post-processing.
- Documentation/handoff files updated for this recovery line:
	- `documentation/wow-400-terrain-blend-wow-exe-guide.md`
	- `docs/archive/WoW_400_ADT_Analysis.md`
	- `docs/archive/WoW_400_DeepDive_Analysis.md`
	- `docs/archive/WoW_301_DeepDive_Analysis.md`
	- `docs/ADT_WDT_Format_Specification.md`
	- `specifications/ghidra/prompt-400.md`
	- `.github/prompts/wow-400-terrain-blend-recovery.prompt.md`
- Validation status for this exact 4.0 recovery slice:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- No real-data runtime signoff yet on the fixed development terrain after residual synthesis + edge stitching.
	- Do not claim 4.0 terrain correctness from build success or diagnostics alone.

### WMO Blend + Loose PM4 Overlay Follow-Up (Mar 21)

- WMO distant "foggy sheen" triage found one concrete renderer mismatch in `src/MdxViewer/Rendering/WmoRenderer.cs`:
	- the active branch had flattened WMO material blend handling into opaque vs generic transparent
	- current code now maps raw WMO `BlendMode` to `EGxBlend` semantics (`Opaque`, `Blend`, `Add`, `AlphaKey`)
	- opaque pass now keeps `AlphaKey` with alpha-test, while transparent pass only handles `Blend` / `Add`
- Loose overlay PM4 resolution now gives precedence to the most recently attached overlay root in `src/MdxViewer/DataSources/MpqDataSource.cs`.
	- this matters when a base path and a later loose overlay both expose the same PM4 virtual path
	- older behavior searched loose roots in insertion order, so base loose files could shadow the attached overlay
	- current resolver searches newest overlay first and now traces PM4 loose-path misses like WMO misses
- Validation status for these viewer fixes:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` PASSED on Mar 21, 2026.
	- No automated tests were added or run for these fixes.
	- No runtime real-data signoff yet for the WMO sheen symptom or the loose-overlay PM4 workflow.

### PM4 Decode Triage + Rendering Parity Program (Mar 21)

- Current PM4 overlay failure state has moved past indexing/attach into decode-or-reconstruction triage:
	- runtime symptom seen by the user: `PM4: 2674 files found, none decoded into overlay data.`
	- this means PM4 candidates are being found, but none produced renderable overlay objects
	- latest `WorldScene.LazyLoadPm4Overlay()` instrumentation now buckets that failure into:
		- tile-parse rejection
		- tile-range rejection
		- read failure
		- decode failure
		- parsed-but-zero-object files
- Working hypothesis for the `4.0` versus `3.3.5` split:
	- PM4 parsing/object assembly itself appears build-agnostic
	- the likely seam is build-dependent map discovery / WDT resolution / candidate-set selection through `_dbcBuild`
	- the observed `2674` candidate count is suspicious versus the fixed development dataset note in `memory-bank/data-paths.md` (`616 PM4 files`) and should be treated as a clue, not normal noise
- Rendering work is now explicitly grouped as one coordinated program because PM4 object-variant matching depends on visually trustworthy output, not only PM4 geometry placement.
- The ordered rendering program is now:
	1. M2 material, transparency, and reflective-surface parity
	2. lighting DBC expansion beyond the current `Light` + `LightData` subset
	3. skybox / environment parity so backdrop and lighting context stop misleading object matching
- Planning artifacts created for this program live under `.github/prompts/`:
	- `m2-material-parity-implementation-plan.prompt.md`
	- `lighting-dbc-expansion-implementation-plan.prompt.md`
	- `sky-environment-parity-implementation-plan.prompt.md`
- Validation status for this planning slice:
	- no rendering code changes landed yet from this program
	- no automated tests were added or run for the planning-only pass
	- no runtime real-data validation yet on the new PM4 failure-bucket diagnostics

### Recovery Work Completed On This Branch

- Re-established v0.4.0 baseline in the primary tree and validated build.
- Restored the project instruction stack from main:
	- copilot-instructions
	- instructions
	- prompts
	- terrain-alpha-regression skill files
- Applied profile-driven terrain alpha decode routing in viewer terrain path:
	- Added TerrainAlphaDecodeMode to AdtProfile in FormatProfileRegistry
	- 3.x profiles route to LichKingStrict
	- 0.x profiles route to LegacySequential
	- StandardTerrainAdapter alpha extraction now routes by profile mode
	- Strict path includes UseAlpha-first decode plus offset/span fallback for mis-set flags
	- Legacy path remains sequential 4-bit nibble expansion

### Validation Status

- Build: dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug PASSED
- Runtime real-data spot-check: PARTIAL PASS
	- user confirmed Alpha-era 0.5.3 terrain renders correctly again after the alpha-edge-fix restoration
	- user confirmed a 3.0.1 alpha build now renders correctly on the current profile-driven 3.x path
	- earlier 3.3.5 spot-check also looked correct, but broader cross-map signoff is still pending
- Do not claim full terrain regression safety beyond the validated samples above.

### Next Integration Queue (Ordered)

1. Commit and push the current profile/decode code slice if not already committed.
2. Broaden runtime-check alpha decode behavior beyond the currently validated 0.5.3 and 3.0.1 samples.
3. Continue commit-by-commit intake from v0.4.0..main with strict triage:
	 - SAFE first
	 - MIXED only with dependencies proven and build gates
	 - RISKY terrain renderer/decode rewrites skipped unless explicitly approved
4. Keep UI changes incremental; avoid broad layout rewrites.
5. Pull selected import/export functionality in small batches after profile/decode stabilization.

### Surgical Intake Triage (Mar 17)

- Commit triage against `v0.4.0..main` is now documented for the current queue:
	- `177f961`: RISKY, skip entire commit (terrain renderer + tile mesh + alpha decode rewrite)
	- `37f669c`: RISKY, skip entire commit (relaxed alpha heuristics + MPQ decompression changes)
	- `d50cfe7`, `326e6f8`, `4e2f681`, `39799bf`, `62ecf64`: MIXED, only extract isolated safe slices
- First SAFE batch selected:
	- take only the corrected `TerrainImageIo` alpha-atlas helper from `62ecf64`
	- do not take the earlier `d50cfe7` version because it reintroduced atlas import/export edge remapping
	- do not take ViewerApp, TerrainRenderer, WorldScene, test-project, or alpha-decode hunks in the first batch
- Required gate after the first SAFE batch: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime terrain validation remains required after any terrain-adjacent batch; build success is not proof of terrain correctness.
- First SAFE batch status:
	- corrected `TerrainImageIo` helper has been applied in the recovery branch
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed after the change
	- runtime real-data validation is still pending

### Rendering Fix Batch (Mar 18)

- Applied the main-branch `WorldAssetManager` residency fix in the recovery branch:
	- MDX/WMO renderer residency now defaults to unlimited
	- only the raw file-data cache remains bounded
	- cached failed model loads are retried instead of becoming permanent null entries
	- lazy `GetMdx` / `GetWmo` lookups can now load on demand
- Applied the minimal main-branch skybox backdrop path without broad ViewerApp/UI churn:
	- skybox-like MDX/M2 placements are routed into a separate skybox instance list
	- nearest skybox placement renders as a camera-anchored backdrop before terrain
	- `ModelRenderer` now has a backdrop path that keeps depth test/write disabled for all layers
- Current branch already had the reflective M2 depth-flag fix and the guarded env-map backface handling, so those were not re-applied.
- Build gate passed again after this rendering batch: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- Runtime validation is still required for:
	- WMO/MDX disappearance when moving away and back
	- skybox model classification and backdrop behavior
	- MH2O liquid rendering on LK data

### MCCV + MPQ Recovery Batch (Mar 18)

- Restored active-branch MCCV terrain support in the chunk renderer path:
	- `StandardTerrainAdapter` now carries MCNK MCCV data into `TerrainChunkData`
	- `TerrainMeshBuilder` uploads per-vertex RGBA alongside position/normal/UV
	- `TerrainRenderer` consumes MCCV in the shader
- Follow-up correction after runtime feedback:
	- MCCV is now treated as BGRA, matching the repo's own `MinimapService.GenerateMccvData` documentation
	- neutral/no-tint MCCV is treated as mid-gray (`127`) rather than white
	- shader tinting now maps mid-gray to neutral and no longer relies on MCCV alpha as terrain tint strength
- Applied the isolated `NativeMpqService` slice from the mixed MPQ recovery commits:
	- broader patch archive priority ordering, including locale/custom patch variants
	- encrypted-file key derivation now tries the full normalized path first, then basename fallback
	- per-sector MPQ decompression now handles bitmask combinations instead of only single-byte cases
	- BZip2 sector decompression added via SharpZipLib
- Follow-up patch-chain fix:
	- `NativeMpqService.LoadArchives(...)` now discovers MPQs recursively instead of only scanning a few top-level directories
	- Alpha-style single-asset wrapper archives (`.wmo.mpq`, `.wdt.mpq`, `.wdl.mpq`) are still excluded from this generic path because `MpqDataSource` handles them separately
- Build gates passed after this batch:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime real-data validation is still required for:
	- 1.x+ patch-chain reads on patched client data
	- later-version encrypted MPQ entries
	- 3.x MCCV highlight/tint behavior on real LK terrain after the BGRA + mid-gray semantic correction

### WDL + Model Compatibility Follow-up (Mar 18)

- Follow-up after runtime feedback on the newly ported WDL preview cache:
	- the main WDL failure on 1.x/3.x was not only path lookup; `WoWMapConverter.Core.VLM.WdlParser` hard-rejected every non-`0x12` WDL
	- parser is now version-tolerant and scans for `MAOF`/`MARE` instead of requiring Alpha-only layout assumptions
	- parser also tolerates MAOF offsets that point either at a `MARE` chunk header or directly at the height payload
- Viewer-side WDL read paths are now unified through `WdlDataSourceResolver`:
	- both preview warmup and 3D WDL terrain now try `.wdl` and `.wdl.mpq`
	- MPQ-backed loads also use `MpqDataSource.FindInFileSet(...)` so listfile/casing recovery works consistently
- Remaining 3.x doodad extension parity gap closed in `WmoRenderer`:
	- canonical doodad resolution now tries `.m2` in addition to `.mdx`/`.mdl`
- Semi-translucent model follow-up in `ModelRenderer`:
	- shared texture cache entries now carry a simple alpha classification (`Opaque`, `Binary`, `Translucent`)
	- classic non-M2 layer-0 `Transparent` now stays on the hard alpha-cutout path only when the loaded texture alpha is binary
	- textures with intermediate alpha values now render through the blended path instead of the old foliage-style cutout heuristic
- Build gate passed after this batch:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime validation is still required before claiming:
	- non-Alpha WDL previews / WDL 3D terrain actually load on the user's real 1.x/3.x map set
	- 3.x `.mdx` WMO doodads now resolve correctly as M2-family assets on real data
	- the semi-translucent material heuristic fixes the reported visuals without regressing classic cutout foliage

### WDL Spawn Chooser Regression Handoff (Mar 20)

- Latest runtime report from the active branch: WDL heightmap spawn chooser is currently non-functional across tested versions.
- Treat earlier notes that framed the spawn chooser path as working as stale until revalidated.
- Scope this as a viewer flow regression, not a parser-complete claim:
	- likely touchpoints are spawn action enablement (`WdlPreviewWarmState` gating), preview readiness transitions, and preview dialog/open fallback routing
	- this may involve both UI state and async warmup timing, not just WDL decode
- Do not close this issue on build success or file-level diagnostics alone.
- Required signoff for closure:
	- real-data runtime verification on at least one Alpha-era map and one 3.x map
	- explicit proof that spawn chooser opens/commits a spawn point and that fallback load behavior still works when preview prep fails

### PM4 Tile Mapping Runtime Handoff (Mar 20)

- PM4 viewer tile assignment now follows direct filename indices (`map_x_y.pm4` maps to `(tileX=x, tileY=y)`).
- The old MPRL-based tile reassignment heuristic has been removed from the PM4 overlay load path.
- Duplicate PM4 files mapping to one tile now merge object payloads/stats/refs instead of replacing prior data.
- Immediate next step after restart is runtime validation on the reported adjacency mismatch (`00_00`, `01_00`, and `01_01`) before further PM4 transform work.
- Do not claim this fixed from build-only validation; runtime signoff is still pending.

### M2 Empty-Fallback Guardrail (Mar 18)

- Follow-up after the standalone 3.x model-load freeze fix: some M2-family assets could still appear to load while producing a blank viewport and model info with zero geometry.
- Current conclusion is narrow:
	- this is at least partly a false-positive success path, not necessarily a valid render of an odd pre-release asset
	- raw `MD20` fallback conversion can yield an `MDX` shell that parses but has no renderable geosets
- Recovery change applied:
	- shared geometry validation added for converted M2 fallback results
	- standalone `ViewerApp`, world `WorldAssetManager`, and WMO doodad `WmoRenderer` now reject empty converted fallback models and keep the real failure path visible in logs
- Validation status:
	- alternate-OutDir build passed: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
	- no runtime real-data validation yet
	- do not over-claim this as a full M2 render fix for pre-release `3.0.1`; it is a guardrail that removes a misleading blank-success outcome

### Pre-release 3.0.1 M2 + Shared Transparency Follow-up (Mar 18)

- User runtime verification now narrows the remaining model issue further:
	- most unresolved M2 failures are specific to the pre-release `3.0.1` model family, not the later `3.3.5` layout
	- the active working assumption is that this pre-release family may be a hybrid or transitional `MDX` + `M2` variant rather than a clean later-WotLK `M2`
- Treat this as a separate compatibility track:
	- do not assume later `3.3.5` `MD20` / `.skin` semantics are sufficient for pre-release `3.0.1`
	- keep profile/version-aware model parsing on the roadmap instead of broadening generic fallback heuristics
	- the empty-fallback guardrail remains useful, but it is only a diagnostics fix
- Separate rendering issue still confirmed by runtime evidence:
	- neon-pink transparent surfaces still reproduce on both classic `MDX` and M2-family assets
	- that means the pink/transparency bug is not only an M2 parser problem; it is likely in shared material, texture binding, blend, or shader behavior
- Practical next investigation split:
	1. pre-release `3.0.1` model-structure compatibility in `WarcraftNetM2Adapter` / profile routing
	2. shared transparent-material shader parity across `ModelRenderer` and any M2-converted runtime path

### Pre-release 3.0.1 wow.exe Guide Handoff (Mar 19)

- Latest Ghidra pass mapped the common model load chain in `wow.exe` build `3.0.1.8303`:
	- `FUN_0077e2c0` -> `FUN_0077d3c0` -> `FUN_0079bc70` -> `FUN_0079bc50` -> `FUN_0079bb30` -> `FUN_0079a8c0`
- High-confidence parser contract now documented in `documentation/pre-release-3.0.1-m2-wow-exe-guide.md`:
	- root must be `MD20`
	- accepted version range is `0x104..0x108`
	- parser layout splits at `0x108`
	- shared typed span validators use strides `1`, `2`, `4`, `8`, `0x0C`, `0x30`, and `0x44`
	- confirmed nested record families include `0x70`, `0x2C`, `0x38`, `0xD4`, and `0x7C`
	- legacy side uses `0xDC` + `0x1F8`; later side uses `0xE0` + `0x234`
- Fresh-chat prompts now exist for implementation, deeper Ghidra follow-up, and runtime triage:
	- `.github/prompts/pre-release-3-0-1-m2-implementation-plan.prompt.md`
	- `.github/prompts/pre-release-3-0-1-m2-ghidra-followup.prompt.md`
	- `.github/prompts/pre-release-3-0-1-m2-runtime-triage.prompt.md`
- Do not treat the guide as proof that viewer support is implemented yet:
	- no new runtime validation happened in this documentation pass
	- Track B pink transparency remains separate

### Pre-release 3.0.1 Profile Routing Broadening (Mar 19)

- Active profile resolution is no longer restricted to exact build `3.0.1.8303`.
- `FormatProfileRegistry` now maps any parsed `3.0.1.x` build to the existing pre-release `3.0.1` ADT, WMO, and M2 profiles.
- Keep the scope narrow:
	- this is profile routing, not full parser completion for every remaining pre-release `3.0.1` model difference
	- other `3.0.x` builds still use the generic `3.0.x` fallback profile unless new binary evidence justifies a tighter mapping
- Validation status:
	- code change applied
	- build/runtime validation still pending for this exact routing update

### Pre-release 3.0.1 Parser + Fallback Alignment (Mar 19)

- `WarcraftNetM2Adapter` now has a dedicated pre-release `MD20` parse path based on the wow.exe contract instead of routing those files through Warcraft.NET's later-layout `MD21` parser.
- Current scope of the fix:
	- standalone model load
	- world doodad load
	- WMO doodad load
	- shared `M2ToMdxConverter` fallback for those entry points
- Important implementation boundary:
	- the prior profile-specific `.skin` parser path was disabled because its `0x70` / `0x2C` record-size assumptions were lifted from model-family validation, not proven `.skin` layout evidence
	- converter fallback now keeps pre-release handling geometry-focused by skipping later-layout animation / bone parsing and by not forcing optional fixed-stride `.skin` submesh / texture-unit parsing
- Current residual risk:
	- runtime validation on real `3.0.1` assets is still outstanding
	- active MPQ build selection still relies on path/build inference unless a more explicit selector is ported later

### 3.x Alpha Follow-up (Mar 18)

- The LK offset-0 fallback experiment in `StandardTerrainAdapter.ExtractAlphaMaps(...)` was reverted after runtime validation showed it was wrong for the active 3.x terrain path.
- Current conclusion:
	- the recent attempt to treat `AlphaMapOffset == 0` as a valid relaxed-LK fallback case was not the correct fix
	- keep that path reverted and continue investigating 3.x alpha sourcing/decode without broadening fallback heuristics blindly
- Alternate-output build validation passed after reverting the tweak because a live `MdxViewer` process still had the normal `bin/Debug` outputs locked.

### 3.x Profile-Driven Alpha Recovery (Mar 18)

- Follow-up investigation confirmed the active recovery branch was still missing two important 3.x inputs that existed in rollback code:
	- WDT/MPHD big-alpha detection should treat `0x4 | 0x80` as the effective big-alpha mask for 3.x profiles
	- 3.x layer/alpha/shadow sourcing may need to come from split `*_tex0.adt` MCNK data rather than the root ADT alone
- Recovery changes now applied:
	- `AdtProfile` carries `BigAlphaFlagsMask` and `PreferTex0ForTextureData`
	- 3.0.1 / 3.3.5 profiles use `0x4 | 0x80` and prefer `*_tex0.adt`
	- `StandardTerrainAdapter` can build a `*_tex0.adt` MCNK index map and source MTEX/layers/MCAL/MCSH from that file when the profile says to
	- `StandardTerrainAdapter` now passes the MCNK `0x8000` do-not-fix-alpha bit into MCAL decode and uses chunk-level big-alpha inference instead of the reverted offset-0 fallback
	- `WoWMapConverter.Core.Formats.LichKing.Mcal` now has the stronger compressed / big-alpha / 4-bit decode split with proper edge-fix suppression for big-alpha and do-not-fix chunks
- Build gates passed after this batch:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build "I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="I:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- Runtime signoff is still pending:
	- confirm real 3.x tiles stop falling back to obvious 4-bit Alpha-style layer-1-only behavior
	- confirm split `*_tex0.adt` sourcing is actually the missing piece on the user’s 3.x client data

### Commit 39799bf Model Slice (Mar 18)

- The M2 load-failure fix associated with `39799bf` was the `NativeMpqService` encrypted-read compatibility slice, which is now already applied.
- The only additional model-renderer change from that commit was also applied:
	- `ModelRenderer` no longer renders particles on the world-scene batched instance path
	- standalone model viewing still renders particles
- Rationale:
	- world-scene batch instancing does not yet propagate per-instance transforms into particle simulation/rendering
	- leaving particles enabled there can produce camera-locked billboard artifacts on placed models
- Build gate passed again after applying this renderer hunk: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`

## Current Focus: MDX Compatibility Port + Rendering Parity (Feb 14, 2026)

MdxViewer is the **primary project** in the tooling suite. It is a high-performance 3D world viewer supporting WoW Alpha 0.5.3, 0.6.0, and LK 3.3.5 game data.

### Recently Completed (Feb 14)

- **GEOS Port (wow-mdx-viewer parity)**: ✅ `MdxFile.ReadGeosets` now routes by version with strict paths for v1300/v1400 and v1500, with guarded fallback.
- **SEQS Name Recovery**: ✅ Counted 0x8C named-record detection broadened so playable models no longer fall into `Seq_{animId}` fallback names in many cases.
- **PRE2 Parser Expansion**: ✅ Particle emitter v2 parser now reads full scalar payload layout, spline block, and skips known anim-vector tails safely for alignment.
- **RIBB Parser Expansion**: ✅ Ribbon parser now processes known tail anim-vector chunks safely for alignment.
- **Specular/Env Orientation Fix (shader)**: ✅ MDX fragment shader now flips normals/view-normals on backfaces before sphere-env UV and lighting/specular, targeting inside-out dome reflections.

### Previously Completed (Feb 11-12)

- **Full-Load Mode**: ✅ `--full-load` (default) / `--partial-load` CLI flags — loads all tiles at startup
- **Specular Highlights**: ✅ Blinn-Phong specular in ModelRenderer fragment shader (shininess=32, intensity=0.3)
- **Sphere Environment Map**: ✅ `SphereEnvMap` flag (0x2) generates UVs from view-space normals for reflective surfaces
- **MDX Bone Parser**: ✅ BONE/HELP/PIVT chunks parsed with KGTR/KGRT/KGSC keyframe tracks + tangent data
- **MDX Animation Engine**: ✅ `MdxAnimator` — hierarchy traversal, keyframe interpolation (linear/hermite/bezier/slerp)
- **Animation Integration**: ✅ Per-frame bone matrix update in MdxRenderer.Render()
- **WoWDBDefs Bundling**: ✅ `.dbd` definitions copied to output via csproj Content items
- **Release Build**: ✅ `dotnet publish -c Release -r win-x64 --self-contained` verified working (1315 .dbd files bundled)
- **GitHub Actions**: ✅ `.github/workflows/release-mdxviewer.yml` — tag-triggered + manual dispatch, creates ZIP + GitHub Release
- **No StormLib**: ✅ Pure C# `NativeMpqService` handles all MPQ access — no native DLL dependency

### Previously Completed (Feb 9-10)

- WMO doodad culling (distance + cap + sort + fog passthrough)
- GEOS footer parsing (tag validation)
- Alpha cutout for trees, MDX fog skip for untextured
- AreaID fix (low 16-bit extraction + fallback)
- Directional tile loading with heading-based priority
- DBC lighting (Light.dbc + LightData.dbc)
- Replaceable texture DBC resolution with MPQ validation

### Mar 19, 2026 - PM4 Coordinate Validation Slice

- Active core PM4 support now has one explicit coordinate-validation path built around `MPRL` refs already stored in ADT placement order.
- New active-core pieces:
	- `WoWMapConverter.Core/Formats/PM4/Pm4CoordinateService.cs` defines the authoritative PM4 placement helpers for this first validation pass.
	- `WoWMapConverter.Core/Formats/PM4/Pm4CoordinateValidator.cs` validates transformed `MPRL` refs against real `_obj0.adt` placements from the fixed development dataset.
	- `WoWMapConverter.Cli` now exposes `pm4-validate-coords`.
- Real-data validation status for this slice:
	- `wowmapconverter pm4-validate-coords --tile-limit 100` validated 100 PM4 tiles with placements from the fixed development dataset
	- 38,133 `MPRL` refs landed in expected tile bounds (100.0%)
	- 36,070 refs landed within a 32-unit nearest-placement threshold (94.6%)
	- average nearest-placement distance was 10.86 units
- Scope boundary:
	- this validates the `MPRL` anchor path only
	- cross-tile CK24 aggregation is still pending
	- MSCN/world-space semantics are still not the validated contract for active core code
- Do not claim PM4 world placement is fully solved beyond this `MPRL` path until CK24 aggregation and MSCN semantics are also validated.

### Mar 20, 2026 - PM4 Viewer Overlay Diagnostics + Grouping/Winding Pass

- PM4 support advanced from coordinate-validation-only into active viewer diagnostics in `src/MdxViewer/Terrain/WorldScene.cs` + `src/MdxViewer/ViewerApp.cs`.
- New viewer PM4 overlay capabilities now include:
	- multi-mode color classification (`CK24` type/object/key, tile, dominant group/attribute, height)
	- optional `MPRL` reference pins and PM4-object centroid pins
	- selected-object PM4 metadata readout (dominant group key, attribute mask, `MdosIndex`, planar transform flags, winding parity)
	- CK24 disjoint-geometry splitting toggles: connectivity split and optional `MdosIndex` pre-split
- Orientation correction changed from translation-first nudging to per-object planar transform solving with parity-aware triangle winding correction.
- Scope boundary for this pass:
	- still a viewer-side PM4 debug/reconstruction layer, not final cross-tile object identity
	- map-wide CK24 registry + MSCN semantics remain pending
- Validation status:
	- repeated `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
	- no automated tests were added or run
	- runtime real-data visual signoff remains pending for merged/disjoint object edge cases

### Working Features

| Feature | Status | Notes |
|---------|--------|-------|
| Alpha 0.5.3 WDT terrain | ✅ | Monolithic format, 256 MCNK chunks per tile |
| 0.6.0 split ADT terrain | ✅ | StandardTerrainAdapter, MCNK with header offsets |
| 0.6.0 WMO-only maps | ✅ | MWMO+MODF parsed from WDT |
| 3.3.5 split ADT terrain | ⚠️ | Loading freeze — needs investigation |
| WMO v14 rendering | ✅ | 4-pass: opaque/doodads/liquids/transparent |
| WMO liquid (MLIQ) | ✅ | matId-based type detection, correct positioning |
| Terrain liquid (MCLQ) | ✅ | Per-vertex sloped heights, absolute world Z |
| MDX rendering | ✅ | Two-pass, alpha cutout, blend modes 0-6 |
| Async tile streaming | ✅ | 9×9 AOI, directional lookahead, persistent cache |
| Frustum culling | ✅ | View-frustum + distance + fade |
| DBC Lighting | ✅ | Zone-based ambient/fog/sky colors |
| Minimap overlay | ✅ | BLP tiles, zoom, click-to-teleport |

### Known Issues / Next Steps

1. **Runtime validation pending (critical handoff item)** — verify PRE2/RIBB-heavy models visually after parser expansion.
2. **Specular/env dome check pending** — confirm Dalaran dome-like materials now reflect outward after backface normal correction.
3. **Residual SEQS/material parity work** — continue porting edge-case behavior from `lib/wow-mdx-viewer` if specific models still diverge.
4. **WMO semi-transparent window materials** — Stormwind glass still maps to wrong geometry (root cause unknown).
5. **MDX cylindrical texture stretching** — barrels/tree trunks still show stretched planks on some assets.
6. **3.3.5 ADT loading freeze** — needs investigation.
7. **WMO culling too aggressive** — objects outside WMO not visible from inside.

---

## Key Architecture Decisions

### Coordinate System (Confirmed via Ghidra)
- WoW: right-handed, X=North, Y=West, Z=Up, Direct3D CW front faces
- OpenGL: CCW front faces
- Fix: Reverse winding at GPU upload + 180° Z rotation in placement
- Terrain: `rendererX = MapOrigin - wowY`, `rendererY = MapOrigin - wowX`
- WMO-only maps: raw WoW world coords (no MapOrigin conversion)

### Performance Constants

| Constant | Value | Location |
|----------|-------|----------|
| DoodadCullDistance (world) | 1500f | WorldScene.cs |
| DoodadSmallThreshold | 10f | WorldScene.cs |
| WmoCullDistance | 2000f | WorldScene.cs |
| NoCullRadius | 150f | WorldScene.cs |
| WMO DoodadCullDistance | 500f | WmoRenderer.cs |
| WMO DoodadMaxRenderCount | 64 | WmoRenderer.cs |
| AoiRadius | 4 (9×9) | TerrainManager.cs |
| AoiForwardExtra | 3 | TerrainManager.cs |
| MaxGpuUploadsPerFrame | 8 | TerrainManager.cs |
| MaxConcurrentMpqReads | 4 | TerrainManager.cs |

### Key Files

| File | Purpose |
|------|---------|
| `WorldScene.cs` | Placement transforms, instance management, culling |
| `WmoRenderer.cs` | WMO v14 GPU rendering, doodad culling, liquid |
| `ModelRenderer.cs` | MDX GPU rendering, alpha cutout, fog skip |
| `AlphaTerrainAdapter.cs` | Alpha 0.5.3 WDT terrain + AreaID + liquid type |
| `StandardTerrainAdapter.cs` | 0.6.0 / 3.3.5 split ADT terrain + MCLQ + WMO-only maps |
| `TerrainManager.cs` | AOI streaming, persistent cache, MPQ throttling |
| `LiquidRenderer.cs` | MCLQ/MLIQ liquid mesh rendering |
| `AreaTableService.cs` | AreaID → name with MapID filtering |
| `LightService.cs` | DBC Light/LightData zone-based lighting |
| `ReplaceableTextureResolver.cs` | DBC-based replaceable texture resolution |
| `MdxFile.cs` | MDX parser (GEOS, BONE, PIVT, HELP with KGTR/KGRT/KGSC tracks) |
| `MdxAnimator.cs` | Skeletal animation engine (hierarchy, interpolation, bone matrices) |
| `MdxViewer.csproj` | Project file with WoWDBDefs bundling |
| `.github/workflows/release-mdxviewer.yml` | CI/CD release workflow |
