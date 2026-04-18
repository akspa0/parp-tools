# Progress

### Apr 18, 2026 - wow-viewer now has shared classic MTLS payload ownership with focused reader and resolver coverage

- what changed:
	- added shared MDX material payload contracts in:
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxMaterialLayer.cs`
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxMaterial.cs`
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxMaterialFile.cs`
	- added `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxMaterialReader.cs` for full classic `MTLS` payload ownership over:
		- per-material priority plane parsing
		- fixed `TEXTURELAYER` fields
		- static emissive-gain ownership
		- `KMTE`, `KMTA`, and `KMTF` track parsing
	- `wow-viewer/src/viewer/WowViewer.App/MdxPreviewLoader.cs` now loads shared material payloads into the standalone preview result
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxRenderStateResolver.cs` now resolves material layers against runtime `MTLS` payloads when available and samples `KMTA` alpha tracks through the shared animation sampler
	- added focused coverage in:
		- `wow-viewer/tests/WowViewer.Core.Tests/MdxMaterialReaderTests.cs`
		- `wow-viewer/tests/WowViewer.Core.Tests/MdxRenderStateResolverTests.cs`
- validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "MdxMaterialReaderTests|MdxRenderStateResolverTests"` passed with `7` tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug` passed
- boundary:
	- this lands shared classic `MTLS` payload ownership plus animated layer-alpha consumption for the standalone preview path
	- texture-layer animation and full emissive routing are still future material-parity work even though their payload tracks are now owned in shared code

### Apr 18, 2026 - wow-viewer standalone MDX preview now ports the existing MdxViewer PRE2 loop and keeps EVTS or RIBB non-fabricated

- what changed:
	- added `wow-viewer/src/core/WowViewer.Core.Runtime/Mdx/MdxEffectRuntime.cs` with shared runtime-state contracts and evaluator logic for classic `MDX` effect seams:
		- `MdxEventRuntimeState`
		- `MdxParticleEmitter2RuntimeState`
		- `MdxRibbonRuntimeState`
		- `MdxEffectRuntimeEvaluator`
	- extended `wow-viewer/src/core/WowViewer.Core/Mdx/MdxAnimationSampler.cs` with shared integer-track sampling so classic ribbon texture-slot animation can be consumed from shared runtime code
	- `wow-viewer/src/viewer/WowViewer.App/MdxPreviewLoader.cs` now evaluates shared effect runtime state during standalone `MDX` preview loading and carries it in `MdxPreviewLoadResult`
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerDesktopApp.cs` now surfaces runtime effect counts plus event or particle or ribbon sample state in standalone `MDX` diagnostics and status text
	- `wow-viewer/src/viewer/WowViewer.App/MdxGpuPreviewRenderer.cs` now turns that shared runtime state into first visible preview output:
		- renderer-owned stepped effect time advances classic `MDX` runtime state over real frame time without reloading the preview
		- classic `PRE2` emitters now use a direct port of the existing `gillijimproject_refactor/src/MdxViewer/Rendering/ParticleSystem.cs` update loop, including random cone spawning, parent-bone-follow transforms, per-particle gravity integration, and fixed three-stage lifecycle interpolation
		- bone-followed particle emitter transforms now update at the stepped preview time instead of staying pinned to the original load frame
		- classic MDX geosets now rebuild on the stepped animation clock, so skinned vertices, geoset alpha, and texture-animation UV transforms no longer stay frozen at preview-load time
		- transparent MDX geosets now sort by material priority plane and camera distance during the transparent pass instead of staying in insertion order
		- classic MDX geosets now render one command per material layer instead of collapsing every material to layer `0`, which restores layered texture submission and per-layer UV selection in the standalone preview path
		- per-layer MDX lighting and depth behavior now resolves from material-layer flags inside `MdxRenderStateResolver`, so legacy `Unshaded` or `NoDepthTest` or `NoDepthSet` behavior is no longer dropped on the preview side
		- sphere-environment mapped MDX layers now drive view-space-normal UV generation in the preview shader instead of being treated like ordinary UV-mapped layers
		- classic `RIBB` payloads or runtime samples still surface in diagnostics, but ribbon rendering is no longer claimed as ported behavior in `WowViewer.App`
		- classic `EVTS` remains non-visual in the preview because the current file payload exposes trigger keys but not a native visual consumer mapping
	- added focused runtime coverage in `wow-viewer/tests/WowViewer.Core.Tests/MdxEffectRuntimeEvaluatorTests.cs`
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug` passed after the PRE2 renderer switched from app-local approximation logic to the MdxViewer loop
	- `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug` also passed after the later multi-layer material, layer-flag, and sphere-environment-map preview updates
- boundary:
	- this now lands shared classic `MDX` effect runtime evaluation plus renderer-local persistent preview playback for `PRE2`, stepped geoset-animation updates, multi-layer material submission, and layer-flag/env-map shaping, while keeping `EVTS` native-shaped and non-visual and dropping unsupported ribbon preview claims
	- current rendering is still bounded preview behavior rather than full native parity: persistent PRE2 simulation and geoset-command rebuilds are app-local, native `EVTS` consumers are not yet recovered, ribbon rendering is still unported, and there is still no active viewer runtime signoff

### Apr 18, 2026 - wow-viewer now has shared classic PRE2 payload ownership with focused reader coverage

- what changed:
	- added shared MDX runtime contracts in:
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxParticleEmitter2.cs`
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxParticleEmitter2File.cs`
	- added `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxParticleEmitter2Reader.cs` for full classic `PRE2` payload ownership over:
		- node transform tracks
		- deferred `PIVT` pivot assignment
		- static particle-emitter fields
		- spline-point and squirts payload parsing
		- `KVIS` / `KP2V`, `KP2S`, `KP2R`, `KP2L`, `KPLN`, `KP2G`, `KLIF`, `KP2E`, `KP2W`, `KP2N`, and `KP2Z` scalar tracks
	- `wow-viewer/src/viewer/WowViewer.App/MdxPreviewLoader.cs` now loads particle-emitter payloads into the standalone preview load result
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerDesktopApp.cs` now surfaces particle-emitter runtime counts and sample payload details in standalone `MDX` diagnostics
	- added focused coverage in `wow-viewer/tests/WowViewer.Core.Tests/MdxParticleEmitter2ReaderTests.cs`
- validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxParticleEmitter2ReaderTests -v minimal` passed with `1` test
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed
- boundary:
	- this lands shared classic `PRE2` payload ownership and app diagnostics exposure only
	- classic particle simulation or rendering playback is still future Plan 04 work

### Apr 18, 2026 - wow-viewer now has shared classic RIBB payload ownership with focused reader coverage

- what changed:
	- added shared MDX runtime contracts in:
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxRibbonEmitter.cs`
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxRibbonEmitterFile.cs`
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxIntKeyframe.cs`
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxIntTrack.cs`
	- extended `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxTrackReader.cs` with reusable color-track and int-track readers
	- added `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxRibbonEmitterReader.cs` for full classic `RIBB` payload ownership over:
		- node transform tracks
		- deferred `PIVT` pivot assignment
		- static ribbon fields
		- `KRHA` / `KRHB` / `KRAL` scalar tracks
		- `KRCO` color tracks
		- `KRTX` integer texture-slot tracks
		- `KVIS` / `KATV` visibility tracks
	- `wow-viewer/src/viewer/WowViewer.App/MdxPreviewLoader.cs` now loads ribbon payloads into the standalone preview load result
	- added focused coverage in `wow-viewer/tests/WowViewer.Core.Tests/MdxRibbonEmitterReaderTests.cs`
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --no-restore --filter "FullyQualifiedName=WowViewer.Core.Tests.MdxRibbonEmitterReaderTests.Read_SyntheticClassicRibbonPayload_AssignsPivotsAndTracks"` passed with `1` test
- boundary:
	- this lands shared classic ribbon-emitter payload ownership only
	- ribbon simulation/render playback and `PRE2` particle-emitter ownership remain future Plan 04 work

### Apr 18, 2026 - wow-viewer now has shared classic EVTS payload ownership with focused reader coverage

- what changed:
	- added shared MDX runtime contracts in:
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxEventTrack.cs`
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxEvent.cs`
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxEventFile.cs`
	- added `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxEventReader.cs` for full classic `EVTS` payload ownership over:
		- node transform tracks
		- deferred `PIVT` pivot assignment
		- raw `KEVT` key-time parsing with global-sequence id
	- `wow-viewer/src/viewer/WowViewer.App/MdxPreviewLoader.cs` now loads events into the standalone preview load result beside the other shared MDX payload seams
	- added focused coverage in `wow-viewer/tests/WowViewer.Core.Tests/MdxEventReaderTests.cs`
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --no-restore --filter "FullyQualifiedName~MdxEventReaderTests"` passed with `1` test
- boundary:
	- this lands shared classic `EVTS` payload ownership only
	- event-driven runtime behavior plus `PRE2` / `RIBB` emitter ownership are still future Plan 04 slices

### Apr 18, 2026 - wow-viewer now has shared helper and attachment MDX payload ownership with focused reader coverage

- what changed:
	- added shared MDX runtime contracts in:
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxHelper.cs`
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxHelperFile.cs`
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxAttachment.cs`
		- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxAttachmentFile.cs`
	- added shared readers in:
		- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxHelperReader.cs`
		- `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxAttachmentReader.cs`
	- both readers now own classic `v1300` / `v1400` payload parsing with full node-track parsing and deferred `PIVT` pivot assignment
	- attachment payload ownership now includes attachment id/path plus `KVIS` or `KATV` visibility-track parsing instead of only summary metadata
	- `wow-viewer/src/viewer/WowViewer.App/MdxPreviewLoader.cs` now loads helper and attachment payloads alongside the existing standalone preview MDX seams
	- added focused tests in:
		- `wow-viewer/tests/WowViewer.Core.Tests/MdxHelperReaderTests.cs`
		- `wow-viewer/tests/WowViewer.Core.Tests/MdxAttachmentReaderTests.cs`
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --no-restore --filter "MdxHelperReaderTests|MdxAttachmentReaderTests"` passed with `2` tests
- boundary:
	- this lands shared payload ownership and focused regression coverage for classic helper/attachment nodes
	- it does not yet add runtime consumption in the renderer for helper-driven transforms, attachment placement, or visibility playback, and `EVTS` / `PRE2` / `RIBB` remain future Plan 04 seams

### Apr 18, 2026 - wow-viewer now has a top-level numbered cut-away program for full old MdxViewer feature parity

- what changed:
	- added `gillijimproject_refactor/plans/wow_viewer_mdxviewer_feature_parity_cutaway_plan_2026-04-18.md`
	- that new plan stitches the already-existing M2/world/viewer/editor/shared-I/O plans into one numbered migration program aimed at replacing old `MdxViewer` ownership with `wow-viewer` ownership instead of treating the old app as the permanent architecture center
	- the new numbered lanes are:
		- Plan 01 - program control, parity matrix, and exit gates
		- Plan 02 - shared format ownership closure
		- Plan 03 - M2 runtime and renderer final closure
		- Plan 04 - MDX runtime and renderer closure
		- Plan 05 - WMO runtime and rendering closure
		- Plan 06 - world runtime and 3D world consumer cutover
		- Plan 07 - viewer shell/UX/workflow parity
		- Plan 08 - tool/inspect/converter/dataset cutover
		- Plan 09 - editor foundation and save-capable cutover
		- Plan 10 - compatibility retirement and final de-ownership
- validation:
	- planning/continuity slice only; no build or runtime proof was needed
- boundary:
	- this does not itself migrate a renderer/runtime feature
	- it gives future chats one canonical top-level parity/cut-away program so the next implementation slice can be chosen against the full migration target rather than only one narrow subsystem plan

### Apr 18, 2026 - wow-viewer standalone MDX preview now has repeatable real-data visual regression coverage

- what changed:
	- added `wow-viewer/src/viewer/WowViewer.App/MdxVisualRegressionRunner.cs` so the existing hidden-window `mdx-gpu-frame` path can be driven from a JSON manifest instead of ad hoc single captures
	- `wow-viewer/src/viewer/WowViewer.App/Program.cs` now exposes `mdx-visual-regression --manifest <cases.json> [--write-actual-root <dir>] [--write-diff-root <dir>] [--update-baselines]`
	- added `wow-viewer/testdata/visual/mdx-gpu-regression.manifest.json` with the first real-data MDX cases for:
		- `alpha053_wisp_default_frame`
		- `alpha053_banshee_default_frame`
	- checked in the first baseline PNGs under `wow-viewer/testdata/visual/mdx-gpu/`
	- added `wow-viewer/scripts/run_mdx_visual_regression.ps1` as a convenience wrapper over the built `WowViewer.App.dll`
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed
	- `dotnet i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/bin/Debug/net10.0/WowViewer.App.dll mdx-visual-regression --manifest i:/parp/parp-tools/wow-viewer/testdata/visual/mdx-gpu-regression.manifest.json --write-actual-root i:/parp/parp-tools/output/build-validation/mdx-gpu-visual-regression/actual --write-diff-root i:/parp/parp-tools/output/build-validation/mdx-gpu-visual-regression/diff --update-baselines` passed and wrote the initial baselines
	- `& 'I:\parp\parp-tools\wow-viewer\scripts\run_mdx_visual_regression.ps1'` passed with `2` regression cases
- boundary:
	- this lands a bounded baseline-image regression harness for standalone classic `MDX` preview only
	- broader case coverage, world/runtime visual proof, and CI-friendly GPU-hosted execution are still future work

### Apr 18, 2026 - wow-viewer standalone MDX preview now evaluates full classic CAMS payloads

- what changed:
	- added `wow-viewer/src/core/WowViewer.Core/Mdx/MdxCamera.cs`, `MdxCameraFile.cs`, and `MdxCameraResolver.cs` so classic `MDX` camera payloads and animated camera-state evaluation now live in shared core code
	- added `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxCameraReader.cs` for full classic `CAMS` payload ownership over fixed camera data plus `KCTR` or `KCRL` or `KVIS` or `KTTR` track payloads
	- `wow-viewer/src/viewer/WowViewer.App/MdxPreviewLoader.cs` now loads that camera payload beside the existing standalone preview inputs
	- `wow-viewer/src/viewer/WowViewer.App/PreviewCameraPlanner.cs` now uses the shared camera resolver when a model camera is used, so standalone `MDX` preview framing can follow animated camera translation or target or roll or visibility data instead of static summary pivots only
	- added focused tests in:
		- `wow-viewer/tests/WowViewer.Core.Tests/MdxCameraReaderTests.cs`
		- `wow-viewer/tests/WowViewer.Core.Tests/MdxCameraResolverTests.cs`
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter "MdxCameraReaderTests|MdxCameraResolverTests"` passed with `4` tests
- boundary:
	- this lands classic `CAMS` runtime payload ownership plus animated standalone preview camera playback only
	- helper/attachment/event runtime seams and broader world/runtime `MDX` consumer cutover remain future slices

### Apr 18, 2026 - wow-viewer MDX render-state semantics now have a shared core seam with focused regression tests

- what changed:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxRenderStateResolver.cs` now owns the bounded classic `MDX` preview render-state shaping that had been private inside `wow-viewer/src/viewer/WowViewer.App/MdxGpuPreviewRenderer.cs`
	- the new resolver centralizes:
		- first-layer material classification including replaceable-texture ids, transparent/additive handling, alpha-cutout classification, and depth-write behavior
		- per-geoset render-state shaping over geoset flags plus runtime or summary geoset-animation alpha/color signals
		- texture-animation translation/rotation/scale shaping over `KTAT` / `KTAR` / `KTAS` tracks
	- `wow-viewer/src/viewer/WowViewer.App/MdxGpuPreviewRenderer.cs` now consumes that shared resolver instead of app-private helpers
	- added `wow-viewer/tests/WowViewer.Core.Tests/MdxRenderStateResolverTests.cs` with focused regression coverage for:
		- transparent-key material handling
		- additive depth-write behavior
		- runtime geoset-animation alpha/color evaluation plus render-flag interaction
		- summary-only geoset-animation fallback behavior
		- texture-animation transform shaping
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter MdxRenderStateResolverTests` passed with `5` tests
- boundary:
	- this lands a renderer-facing semantics and automated-test seam for the bounded standalone classic `MDX` preview path
	- helper/attachment/event runtime behavior, particles/ribbons, and broader world/runtime `MDX` ownership remain future slices

### Apr 17, 2026 - wow-viewer desktop app close-loop crash no longer re-enters Silk window disposal from inside the render loop

- what changed:
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerDesktopApp.cs` no longer subscribes a `Closing` callback that immediately calls `Dispose()`
	- the app now relies on the existing outer `using WowViewerDesktopApp` lifetime in `wow-viewer/src/viewer/WowViewer.App/Program.cs`, so disposal happens after `_window.Run()` returns instead of from inside Silk.NET's close callback
	- `Dispose()` now detaches the window event handlers before disposing the window object, reducing teardown-time callback re-entry risk during app shutdown
- why:
	- the prior close path hit `System.InvalidOperationException: You cannot call Reset inside of the render loop!` because `_window.Dispose()` was being invoked while Silk.NET was already shutting the view down from its render loop
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed after the change
- boundary:
	- this is compile-validated lifecycle hardening for `WowViewer.App` shutdown only
	- runtime close confirmation is still pending from a real app run

### Apr 18, 2026 - wow-viewer app host now targets cross-platform `net10.0` with bounded Windows-only picker fallback

- what changed:
	- `wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj` now targets `net10.0` instead of `net10.0-windows`
	- removed `<UseWindowsForms>true</UseWindowsForms>` from the app project so target resolution no longer forces WinForms at compile level
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerDesktopApp.cs` replaced direct WinForms folder-dialog usage with `TryShowFolderDialog(...)`:
		- non-Windows hosts return `null` immediately
		- Windows hosts attempt reflective `System.Windows.Forms` dialog access and fail closed when unavailable
		- `HandleOpenGameFolderDialog()` now emits a clear status fallback that users can manually enter archive/client paths in existing input fields when picker support is unavailable
- validation:
	- `dotnet build .\wow-viewer\src\viewer\WowViewer.App\WowViewer.App.csproj -c Debug` passed (`WowViewer.App` built as `net10.0`)
	- `dotnet test .\wow-viewer\tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --filter "MdxSkinningHelperTests|MdxBonePoseBuilderTests"` passed with `7` tests
	- existing `CS1668` `LIB`-path environment warnings remained unchanged
- boundary:
	- this lands the project-target/platform correction and removes hard compile-time WinForms dependency from the app host
	- this does not yet provide native folder-picker parity on non-Windows desktops; manual path entry remains the bounded cross-platform fallback

### Apr 18, 2026 - standalone MDX GPU skinning path now has bounded CPU/GPU packing-parity coverage

- what changed:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSkinningHelper.cs` now owns `BuildSkinningVertexData(...)` so the standalone preview's skinning payload packing lives in shared core code instead of app-local inline loops
	- `wow-viewer/src/viewer/WowViewer.App/MdxGpuPreviewRenderer.cs` now delegates skinning-buffer packing to that shared helper while preserving the same 8-float interleaved attribute layout for `aBoneIndices` and `aBoneWeights`
	- added `wow-viewer/tests/WowViewer.Core.Tests/MdxSkinningHelperTests.cs` with bounded parity coverage for:
		- deterministic packed layout ordering
		- zero-fill behavior when requested vertex count exceeds provided index/weight rows
		- CPU-reference parity by unpacking helper-packed payloads and proving `ApplySkinning(...)` / `ApplySkinningNormal(...)` results match direct CPU inputs for the same vertices/bones/matrices
- validation:
	- `dotnet test .\wow-viewer\tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --filter "MdxSkinningHelperTests|MdxBonePoseBuilderTests"` passed with `7` passing tests
	- `dotnet build .\wow-viewer\src\viewer\WowViewer.App\WowViewer.App.csproj -c Debug` passed with the existing workspace `LIB` warnings and known unrelated nullable warnings
- boundary:
	- this closes bounded packing/parity coverage for standalone classic `MDX` GPU skinning input preparation only
	- this does not yet add shader-output pixel/image assertions, helper/attachment/event runtime behavior, particles/ribbons, or broader classic `MDX` world/runtime cutover

### Apr 18, 2026 - standalone MDX preview now uses GPU palette skinning instead of CPU-skinned vertex uploads

- what changed:
	- `wow-viewer/src/viewer/WowViewer.App/MdxGpuPreviewRenderer.cs` no longer uploads already-skinned classic `MDX` vertex positions/normals as the primary preview path
	- the renderer now uploads bind-pose positions/normals/UVs in one vertex buffer and uploads per-vertex bone indices/weights in a separate skinning buffer when the geoset actually uses classic matrix groups
	- the standalone MDX preview shader now accepts a bounded bone palette uniform array plus `uUseBoneSkinning` and applies weighted skinning for positions and normals in the vertex shader
	- the same preview path still rebuilds the classic bone palette from `BONE`/`PIVT` runtime data, including the prior billboard-bone pose rules, so correctness stays anchored on the existing pose solver while the deformation work moves to the GPU
	- posed/skinned bounds are still computed CPU-side for preview framing, but mesh deformation is no longer baked into uploaded vertex positions
- validation:
	- `dotnet build .\wow-viewer\src\viewer\WowViewer.App\WowViewer.App.csproj -c Debug` passed
- boundary:
	- this closes GPU palette skinning only for the bounded standalone classic `MDX` preview consumer
	- helper/attachment/event runtime behavior, particles, ribbons, and broader classic `MDX` runtime cutover remain future slices

### Apr 17, 2026 - standalone MDX preview now applies classic billboard bone rules during pose solving

- what changed:
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxBone.cs` now exposes explicit classic node-flag helpers for ignore-parent and billboard behaviors
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxBonePoseBuilder.cs` now accepts optional camera position and applies bounded billboard-facing rotation during pose solving for:
		- spherical billboards
		- cylindrical billboards using the documented axis-lock bits
	- the same pose builder still keeps the bounded inheritance rule handling local to the runtime seam instead of leaking those semantics back into app-local code
	- the standalone `MDX` preview in `wow-viewer/src/viewer/WowViewer.App/MdxGpuPreviewRenderer.cs` now benefits from that camera-aware pose solve in the same CPU-skinned path added in the prior slice
	- added focused regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/MdxBonePoseBuilderTests.cs` for spherical and cylindrical billboard cases
- validation:
	- `dotnet test .\wow-viewer\tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --filter "MdxBonePoseBuilderTests"` passed
	- `dotnet build .\wow-viewer\src\viewer\WowViewer.App\WowViewer.App.csproj -c Debug` passed
- boundary:
	- this closes billboard bone handling only for the bounded standalone classic `MDX` preview path
	- helper-node runtime usage, attachments/events, particles, ribbons, and broader world/runtime classic `MDX` cutover remain future slices

### Apr 17, 2026 - standalone MDX preview now applies classic BONE/PIVT pose solving and CPU skinning

- what changed:
	- added shared classic bone payload ownership in `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxBoneReader.cs`, including typed `KGTR`/`KGRT`/`KGSC` payload reads plus deferred `PIVT` pivot assignment into the new `wow-viewer/src/core/WowViewer.Core/Mdx/MdxBone.cs` and `MdxBoneFile.cs` contracts
	- added `wow-viewer/src/core/WowViewer.Core/Mdx/MdxBonePoseBuilder.cs` as the first reusable classic `MDX` hierarchy solver over sampled translation/rotation/scaling tracks and pivot-aware local transforms
	- added `wow-viewer/src/core/WowViewer.Core/Mdx/MdxSkinningHelper.cs` so classic `GEOS` matrix groups and matrix tables can be remapped to bone indices and used for weighted CPU skinning of vertices and normals
	- `wow-viewer/src/viewer/WowViewer.App/MdxPreviewLoader.cs` now loads classic bone payloads alongside the earlier summary/geometry/`GEOA`/`TXAN` readers for standalone preview requests
	- `wow-viewer/src/viewer/WowViewer.App/MdxGpuPreviewRenderer.cs` now builds classic bone matrices from the active sequence/time and skins geoset positions/normals before uploading preview buffers, with posed bounds preferred when skinned geometry is available
	- added focused regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/MdxBoneReaderTests.cs` and `MdxBonePoseBuilderTests.cs`
- validation:
	- `dotnet test .\wow-viewer\tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --filter "MdxBoneReaderTests|MdxBonePoseBuilderTests|MdxAnimationSamplerTests|MdxGeosetAnimationReaderTests"` passed
	- `dotnet build .\wow-viewer\src\viewer\WowViewer.App\WowViewer.App.csproj -c Debug` passed
- boundary:
	- this closes classic standalone `MDX` bone payload ownership plus bounded pose/deformation playback in the standalone preview only
	- helper/runtime nodes beyond bones, attachments/events, billboards, particles, ribbons, and world/runtime cutover remain open follow-up work

### Apr 17, 2026 - standalone MDX preview now evaluates animated GEOA alpha/color and TXAN UV transforms

- what changed:
	- added shared classic `GEOA` payload ownership in `wow-viewer/src/core/WowViewer.Core.IO/Mdx/MdxGeosetAnimationReader.cs` plus the new `wow-viewer/src/core/WowViewer.Core/Mdx/` payload contracts for animated geoset alpha/color keys instead of leaving that data at summary-only depth
	- added `wow-viewer/src/core/WowViewer.Core/Mdx/MdxAnimationSampler.cs` as the first reusable classic `MDX` track-evaluation seam for sequence-relative and global-sequence-relative scalar/color/vector/quaternion sampling
	- `wow-viewer/src/viewer/WowViewer.App/MdxPreviewLoader.cs` now loads shared `GEOA` and `TXAN` payload files alongside the existing summary/geometry readers for standalone preview requests
	- `wow-viewer/src/viewer/WowViewer.App/MdxGpuPreviewRenderer.cs` now uses the session `SequenceIndex`/`TimeMs` values for real preview behavior instead of only carrying them through request plumbing:
		- animated geoset alpha now modulates draw alpha through sampled `KGAO`
		- animated geoset color now modulates draw color through sampled `KGAC`
		- material UV animation now samples `KTAT`/`KTAR`/`KTAS` and applies the transform in shader space before texture fetch
		- material `CoordId` now selects the correct UV set instead of always sampling UV set `0`
	- added focused regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/MdxGeosetAnimationReaderTests.cs` and `wow-viewer/tests/WowViewer.Core.Tests/MdxAnimationSamplerTests.cs`
- validation:
	- `dotnet test .\wow-viewer\tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --filter "MdxAnimationSamplerTests|MdxGeosetAnimationReaderTests|MdxTextureAnimationReaderTests|MdxSummaryReaderTests"` passed
	- `dotnet build .\wow-viewer\src\viewer\WowViewer.App\WowViewer.App.csproj -c Debug` passed
- boundary:
	- this closes the first bounded animated standalone `MDX` preview slice only for `GEOA` alpha/color and `TXAN` UV transform playback
	- classic `MDX` skeletal transforms, pivot-aware node evaluation, skinned deformation, billboard behavior, particles, ribbons, and broader runtime parity remain open follow-up work

### Apr 17, 2026 - standalone MDX preview now uses the ported Frame Model path and desktop camera controls

- what changed:
	- `wow-viewer/src/viewer/WowViewer.App/PreviewCameraPlanner.cs` now owns a reusable MDX preview camera surface that defaults to the old `MdxViewer` `FrameBounds(...)` behavior instead of the earlier fit-to-corners camera path, while keeping explicit `orbit` and `model` modes available for bounded alternate capture paths
	- the same planner now uses a wider old-viewer-shaped default FOV for frame mode, and the standalone preview floor on legacy frame distance was reduced so small props no longer stay excessively far away
	- `wow-viewer/src/viewer/WowViewer.App/MdxGpuPreviewRenderer.cs` now resolves preview bounds from actual renderable geoset vertices before declared summary bounds, matching the old `MdxViewer` renderer's local bounds preference for normal MDX and fixing the main cause of the zoomed-out first-view regression
	- the same MDX preview renderer now also uses an explicit legacy-style global sun setup with brighter ambient defaults and softer wrap lighting, fixing the darker backlit shader behavior that was leaving non-emissive preview models too close to black
	- the same MDX preview renderer now also honors core per-geoset MDX render state that was already parsed but previously ignored in the standalone preview path, including `Unshaded`, `NoDepthTest`, `NoDepthSet`, and static geoset-animation alpha or color overrides
	- the same MDX preview renderer now also resolves bounded replaceable-texture cases that previously collapsed to the white fallback texture, first trying classic same-directory `_SkinNN.blp` companions and then the old hardcoded replaceable defaults when the MDX texture entry only exposes a `ReplaceableId`
	- `wow-viewer/src/viewer/WowViewer.App/PreviewCameraPlanner.cs` now keeps `frame` as the general default but auto-prefers the embedded portrait/model camera for MDX assets that expose both portrait cameras and replaceable textures, which fixes the back-facing first view on classic character-like assets without changing generic prop behavior
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerSession.cs` now persists bounded MDX camera settings for the desktop app, and `WowViewerDesktopApp.cs` now exposes frame or orbit or model selection plus orbit preset or custom azimuth or elevation and FOV or zoom controls in the MDX workspace pane
	- the MDX control surface now reloads the active preview with the current camera settings instead of leaving camera iteration as a CLI-only flow
- validation:
	- repeated `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug` passes succeeded through the camera-path port, bounds-source port, and desktop control wiring, with only the usual workspace `LIB` warnings plus existing nullable warnings in unrelated files
	- real-data capture proof via `WowViewer.App mdx-gpu-frame` on `wow-viewer/testdata/0.5.3/tree/Creature/Banshee/Banshee.mdx` confirmed that the final default frame path is no longer stuck in the tiny-speck regression state
	- real-data capture proof on `wow-viewer/testdata/0.6.0/World of Warcraft/Data/world/generic/activedoodads/chest01/chest01.mdx` confirmed the reduced standalone frame-distance floor improves small-prop framing without needing manual camera overrides
	- fresh real-data capture proofs on both `Banshee.mdx` and `chest01.mdx` after the shader update confirmed the preview now has a stable global sun or ambient rig instead of leaving non-emissive models dependent on self-lighting alone
	- a fresh `WowViewer.App mdx-gpu-frame` render on `wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` completed successfully after the per-geoset state patch and wrote `i:/parp/parp-tools/output/tmp/wisp-mdx-state-pass.png`, proving the standalone preview still executes on an effect-heavy asset while using the new depth-state and geoset-animation override path
	- focused inspect proof on `HumanMaleWarriorLight.mdx` confirmed the root texture case that had been missing in the standalone preview path: `replaceableTextures=1` with `TEXS[0]: replaceableId=11 ... path=n/a`
	- real-data capture proof on `HumanMaleWarriorLight.mdx` after the replaceable-texture patch wrote `i:/parp/parp-tools/output/tmp/humanmalewarriorlight-replaceable.png`, confirming the standalone preview no longer falls back to a white untextured render on that asset
	- real-data capture proof on the same asset with the unchanged default camera path then wrote `i:/parp/parp-tools/output/tmp/humanmalewarriorlight-default-fixed.png`, confirming the new automatic portrait-camera preference fixes the obvious back-facing first frame without requiring `--camera-mode model`
- boundary:
	- this closes bounded MDX first-frame camera parity plus app-side control plumbing only
	- the next camera-facing follow-up should be either live viewport interaction in the standalone MDX workspace or a later visual-bounds-aware framing mode for particle-heavy assets
	- broader MDX runtime work is still open, especially animated geoset tracks, particle or ribbon runtime ownership, and richer material or replaceable-texture semantics outside this bounded preview slice

### Apr 17, 2026 - LIT inspect now exposes parsed light entries and heuristic point sampling

- what changed:
	- `wow-viewer/src/core/WowViewer.Core/Lit` now includes typed `LitListEntrySummary` records and a bounded `LitSpatialSampler` helper over the existing `LitSummary` contract
	- `wow-viewer/src/core/WowViewer.Core.IO/Lit/LitSummaryReader.cs` now parses each 64-byte `LIT` list entry into chunk coordinates, world position, radius, dropoff, and name data instead of stopping at aggregate counts alone
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` `lit inspect` now prints a preview of parsed `LIT` entries and accepts `--sample-position <x,y,z>` to report the heuristic candidate entry or default fallback for a world-space point
	- `wow-viewer/tests/WowViewer.Core.Tests/LitSummaryReaderTests.cs` now covers both the parsed entry fields and the new spatial sampler behavior
- validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter LitSummaryReaderTests` passed
	- real-data inspect proof on `world/maps/azeroth/lights.lit` through the 0.6.0 test archive root now reports `57` named light entries and successfully samples `0,0,0` back to the default global light record
	- real-data inspect proof on `world/maps/azeroth/areatest.lit` confirms the older single-partial `LIT` shape still parses cleanly and does not fabricate list entries that are not present
- boundary:
	- this is still parser or inspect ownership plus heuristic spatial selection only
	- actual `LIT` color-band decode and runtime fog or light-color application remain the next implementation slice

### Apr 17, 2026 - Alpha rich-tile world-frame proof now carries all ready MDX placements through visibility and pass planning on the canonical 0.5.5 client

- what changed:
	- `wow-viewer/src/viewer/WowViewer.App/Program.cs` now exposes `world-placement-audit`, a bounded CLI proof command that scans occupied tiles for placement counts without forcing the full world-frame runtime path first
	- `wow-viewer/src/viewer/WowViewer.App/AlphaEmbeddedAdtReader.cs` now separates fast placement-only Alpha embedded-tile reads from the heavier full terrain or liquid fallback path and caches shared Alpha WDT state per map instead of re-reading the monolithic WDT for each tile
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerWorldRuntimeBridge.cs` now uses that placement-only Alpha path for tile selection and placement auditing, routes WMO readiness through the same Alpha-aware file resolver used for other Alpha assets instead of a retail-style archive existence probe, applies the legacy `MdxViewer` world MDX placement transform semantics, and runs the bounded frame through the `Quality` visibility profile so Alpha MDX uses an old-viewer-shaped culling contract instead of the stricter extracted balanced profile
	- `wow-viewer/src/core/WowViewer.Core/Mdx/MdxRenderCharacteristics.cs` now owns parser-derived MDX render traits (`HasOpaqueRenderContent`, `HasTransparentRenderContent`) from `MdxSummary`, and the shared `WorldObjectPassCoordinator` plus `WowViewerWorldRuntimeBridge` now consume those traits instead of keeping the decision logic inside the app layer
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldMdxRenderPlan.cs` now owns a runtime-first MDX GPU-plan contract, grouping opaque and transparent route lists into renderer-facing batches while preserving pass order and unbatched-vs-batched separation
	- `WowViewerWorldRuntimeBridge.cs` now threads that `WorldMdxRenderPlan` through the bounded frame result, and `Program.cs` now prints the derived GPU-plan batch counts in the `world-frame` proof path
	- focused tests now cover both the new MDX render-trait analyzer and the new opaque-route inclusion filter in `WorldObjectPassCoordinator`
- validation:
	- isolated build proof: `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -nologo -clp:Summary -p:OutDir=i:/parp/parp-tools/output/build-validation/wowviewer-alpha-bootstrap/` succeeded with the usual workspace `LIB` warnings plus existing nullable warnings in the touched Alpha/viewer helper files
	- real-data placement proof on `H:/CLIENTS/0.X_Pre-Release_OSX_enUS_0.5.5.3494/World of Warcraft`, `Kalimdor` via `WowViewer.App world-placement-audit --limit 12` reported `scannedTiles=972`, `tilesWithPlacements=564`, and multiple rich Alpha tiles such as `(37,37)` with `2689` total placements and `(39,40)` with `1548` total placements, including concrete sample WMO and MDX paths from the embedded Alpha tile path
	- focused library proof: `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "MdxRenderCharacteristicsAnalyzerTests|WorldObjectPassCoordinatorTests"` passed
	- focused library proof: `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldMdxRenderPlanBuilderTests|WorldObjectPassCoordinatorTests|MdxRenderCharacteristicsAnalyzerTests"` passed
	- real-data rich-tile runtime proof on tile `(39,40)` via `WowViewer.App world-frame` now reports `wmo=40 readyWmo=40 mdx=1508 readyMdx=1508 pending=0`, `visibleWmo=40`, `visibleMdx=1508`, `wmoOpaque=40`, `mdxOpaque=1339`, `mdxTransparent=341`, `opaqueRoutes=1339`, `transparentRoutes=341`, `gpu-plan: opaqueBatches=853 transparentBatches=273 opaqueInstances=1339 transparentInstances=341`, and `objectPhase=True` while still using `...Kalimdor.wdt.MPQ#alpha-tile(39,40)` as the placement source
- boundary:
	- this closes Alpha placement discovery, Alpha WMO asset-readiness proof, and bounded Alpha MDX visibility or pass-routing proof on the canonical rich tile; it does not yet close broader interactive viewer signoff or MDX performance work
	- the next local runtime slice should stay focused on turning the new runtime-owned MDX GPU plan into a real GPU consumer, because the same proof now shows we have a materially smaller batch surface to feed into renderer code without recreating `MdxViewer`-style parser or app coupling

### Apr 17, 2026 - migration priority corrected toward Alpha-first world-format closure and real viewer usability

- what changed:
	- recorded the user directive that the migration should stay anchored on full `wow-viewer` ownership of ADT-family parsing, Alpha-era WDT or ADT support, broader MDX support, and an actually usable viewer surface instead of drifting toward narrow bounded demos alone
	- clarified the current repo boundary: `wow-viewer` has early standard WDT or ADT shared seams plus a bounded world frame and classic `MDX` payload readers, but it still lacks a dedicated Alpha-WDT path, broad ADT-family closure, an implemented standalone MDX consumer, and real orbit or pan or zoom camera controls in the new app
	- set the next continuation bias toward Alpha-era world bring-up and shared-I/O extraction work that unlocks later viewer consumers cleanly
- validation:
	- repo-state confirmation in this chat via direct reads of `WdtSummaryReader`, `AdtSummaryReader`, `AdtPlacementReader`, `AdtV23SummaryReader`, `WowViewerDesktopApp`, `M2GpuPreviewRenderer`, `wow-viewer/README.md`, and the active cutover plans
	- no new build or runtime proof was needed because this update is a continuity and routing correction over already-validated code
- boundary:
	- this is a planning correction only; it does not yet land Alpha-WDT support, broader ADT-family readers, MDX consumer closure, or app-side camera controls

### Apr 17, 2026 - wow-viewer app slice 14 landed: the bounded world frame now has runtime-owned terrain heightmap and software preview data

- what changed:
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainChunkData.cs`, `WorldTerrainHeightmapData.cs`, `WorldTerrainTileData.cs`, and `WorldTerrainTileBuilder.cs` now extend the bounded terrain seam from MCNK header inventory to real MCVT-backed chunk heights plus a reconstructed 257x257 tile heightmap for the selected root ADT
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainVisualSnapshot.cs` now owns a deterministic software terrain preview contract and BMP writer over that runtime-owned terrain heightmap
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerWorldRuntimeBridge.cs` now carries the terrain preview in the bounded world-frame result, while `Program.cs` and `WowViewerDesktopApp.cs` now report terrain height ranges or corner or center samples and surface the terrain preview in the CLI proof path and desktop world-session workspace
	- `wow-viewer/tests/WowViewer.Core.Tests/WorldTerrainTileBuilderTests.cs` and `WorldTerrainVisualSnapshotBuilderTests.cs` now cover both the fixed development root ADT terrain-height seam and a bounded empty-preview fallback path for focused regression coverage
- validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldTerrainTileBuilderTests|WorldTerrainVisualSnapshotBuilderTests"` passed
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the existing workspace `LIB` warnings only
	- real-data runtime proof via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- world-frame --client-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --build-label 3.3.5.12340 --terrain-preview-output ".\output\build-validation\wow-viewer-world-terrain-preview\azeroth_39_32_terrain.bmp"` now reports terrain height-range and sample signals for the selected tile and writes a deterministic preview BMP from the runtime-owned terrain data
	- rerunning with `--hide-terrain` still drops the active terrain stage count to zero while preserving the same source-side terrain preview hash
- boundary:
	- this closes bounded terrain heightmap plus software terrain preview ownership only
	- textured terrain composition or true 3D terrain rendering remains the immediate next follow-up slice

### Apr 17, 2026 - wow-viewer app slice 13 landed: the bounded world frame now has runtime-owned WDL tile data

- what changed:
	- `wow-viewer/src/core/WowViewer.Core/Maps/WdlSummary.cs` plus `wow-viewer/src/core/WowViewer.Core.IO/Maps/WdlSummaryReader.cs` now own the shared WDL summary seam for MAOF/MARE tile data and tolerate both reversed and readable top-level chunk tags
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Wdl/WorldWdlTileData.cs` and `WorldWdlTileBuilder.cs` now own a bounded WDL tile service for selected-tile height ranges and corner or center samples
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerWorldRuntimeBridge.cs` now resolves the selected map WDL into that runtime-owned WDL service and uses actual tile presence instead of a hard-coded WDL source count in the bounded world frame
	- `wow-viewer/src/viewer/WowViewer.App/Program.cs` and `WowViewerDesktopApp.cs` now report WDL range and sample-height signals in the CLI proof path and desktop world diagnostics surfaces
	- `wow-viewer/tests/WowViewer.Core.Tests/WdlSummaryReaderTests.cs` and `WorldWdlTileBuilderTests.cs` now cover both the fixed development WDL and a synthetic readable-tag WDL fixture for focused regression coverage
- validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WdlSummaryReaderTests|WorldWdlTileBuilderTests"` passed
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the existing workspace `LIB` warnings only
	- real-data runtime proof via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- world-frame --client-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --build-label 3.3.5.12340` now reports WDL range and sample-height signals for the selected tile, and rerunning with `--hide-wdl` still drops the active WDL stage count to zero while preserving source-side WDL service data
- boundary:
	- this closes bounded WDL tile service ownership only
	- actual terrain rendering extraction remains the immediate next follow-up slice

### Apr 17, 2026 - wow-viewer app slice 12 landed: the bounded world frame now has runtime-owned terrain chunk inventory

- what changed:
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainChunkData.cs`, `WorldTerrainTileData.cs`, and `WorldTerrainTileBuilder.cs` now own a bounded terrain chunk service over root MCNK headers, exposing chunk coordinates, area ids, hole signals, liquid-flag signals, and current layer-count header values
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerWorldRuntimeBridge.cs` now resolves the selected root ADT into that terrain chunk service and uses it as the terrain-stage source for the bounded world frame instead of only the earlier aggregate terrain count
	- `wow-viewer/src/viewer/WowViewer.App/Program.cs` and `WowViewerDesktopApp.cs` now report terrain chunk samples in the CLI proof path and the desktop world diagnostics surfaces
	- `wow-viewer/tests/WowViewer.Core.Tests/WorldTerrainTileBuilderTests.cs` now covers both the fixed development root ADT and a synthetic root ADT with explicit MCNK header signals for focused terrain-service regression coverage
- validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WorldTerrainTileBuilderTests` passed
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the existing workspace `LIB` warnings only
	- real-data runtime proof via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- world-frame --client-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --build-label 3.3.5.12340` now reports terrain chunk samples for the selected tile, and rerunning with `--hide-terrain` still drops the active terrain stage count to zero while preserving source-side terrain service data
- boundary:
	- this closes bounded terrain chunk service ownership only
	- actual terrain rendering extraction remains the immediate next follow-up slice

### Apr 17, 2026 - wow-viewer app slice 11 landed: the bounded world frame now has runtime-owned liquid tile inventory

- what changed:
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Liquid/WorldLiquidLayerData.cs`, `WorldLiquidChunkData.cs`, `WorldLiquidTileData.cs`, and `WorldLiquidTileBuilder.cs` now own a bounded liquid tile service over shared MH2O decode, exposing chunk coordinates, layer metadata, visible liquid tile counts, and liquid-family grouping
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerWorldRuntimeBridge.cs` now resolves the selected root ADT into that liquid tile service and uses it as the liquid-stage source for the bounded world frame instead of only the earlier aggregate summary counts
	- `wow-viewer/src/viewer/WowViewer.App/Program.cs` and `WowViewerDesktopApp.cs` now report liquid type breakdowns plus bounded chunk samples in the CLI proof path and the desktop world diagnostics surfaces
	- `wow-viewer/tests/WowViewer.Core.Tests/WorldLiquidTileBuilderTests.cs` now covers both the fixed development root ADT and a synthetic MH2O-bearing root ADT for focused liquid-service regression coverage
- validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WorldLiquidTileBuilderTests` passed
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the existing workspace `LIB` warnings only
	- real-data runtime proof via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- world-frame --client-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --build-label 3.3.5.12340` now reports liquid chunk samples and type breakdowns for the selected tile, and rerunning with `--hide-liquid` still drops the active liquid stage count to zero while preserving source-side liquid service data
- boundary:
	- this closes bounded liquid service ownership only
	- terrain service or renderer extraction remains the immediate next follow-up slice

### Apr 17, 2026 - wow-viewer app slice 10 landed: the bounded world frame now has runtime-owned non-object tile-stage summary counts

- what changed:
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldTileStageSummary.cs` and `WorldTileStageSummaryBuilder.cs` now own the bounded root-ADT summary seam for WDL tile presence, terrain chunk counts, terrain hole counts, liquid chunk counts, liquid layer counts, and visible liquid tile counts over shared ADT readers
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerWorldRuntimeBridge.cs` now resolves the selected tile's root ADT through the same archive or loose-file path used by the bounded frame, carries the runtime-owned tile-stage summary in the result, and uses it to populate active WDL or terrain or liquid stage counts instead of placeholder zeros
	- `wow-viewer/src/viewer/WowViewer.App/Program.cs` and `WowViewerDesktopApp.cs` now report active-versus-source terrain-side counts in the `world-frame` CLI proof path and the desktop world-session status or diagnostics surfaces
	- `wow-viewer/tests/WowViewer.Core.Tests/WorldTileStageSummaryBuilderTests.cs` now covers both the fixed development root ADT and a synthetic MH2O-bearing root ADT for focused terrain and liquid summary regression coverage
- validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WorldTileStageSummaryBuilderTests` passed
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the existing workspace `LIB` warnings only
	- real-data runtime proof via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- world-frame --client-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --build-label 3.3.5.12340` now reports terrain-side source and active counts on the selected tile, and rerunning with `--hide-wdl --hide-terrain --hide-liquid` drops the active WDL or terrain or liquid counts while preserving the source counts
- boundary:
	- this closes bounded non-object stage-summary ownership only
	- actual terrain or WDL or liquid renderer extraction and overlay-stage ownership still remain separate follow-up work

### Apr 17, 2026 - wow-viewer app slice 09 landed: the bounded world frame now has runtime-owned pass options

- what changed:
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs` now owns richer pass options for WMO/MDX family gating plus sky/WDL/terrain/liquid/overlay stage gating, and `wow-viewer/tests/WowViewer.Core.Tests/WorldFramePassCoordinatorTests.cs` now proves the new disabled-layer behavior
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerSession.cs`, `Program.cs`, `WowViewerWorldRuntimeBridge.cs`, and `WowViewerDesktopApp.cs` now consume that runtime-owned options seam through persisted world-session state, `world-frame --hide-*` flags, and the bounded world-session controls or diagnostics
	- `wow-viewer/README.md` and the viewer-app cutover continuity files now describe this as the next landed runtime slice instead of leaving slice 08 as the active stop point
- validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WorldFramePassCoordinatorTests` passed
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the existing workspace `LIB` warnings only
	- real-data runtime option proof succeeded via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- world-frame --client-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --build-label 3.3.5.12340 --hide-doodads`, which changed the bounded frame result to zero visible or submitted MDX while keeping WMO counts active on the same auto-selected Azeroth tile `(39,32)`
- boundary:
	- this closes runtime-owned pass-option control only
	- terrain/WDL/liquid/overlay runtime-service extraction and broader renderer ownership still remain separate implementation work

### Apr 17, 2026 - wow-viewer app slice 05 landed: the desktop shell now has bounded world-session bootstrap over shared map readers

- what changed:
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerSession.cs` now carries a typed `WorldSession` workspace state for fixed client root, selected map input, and build label
	- added `wow-viewer/src/viewer/WowViewer.App/WowViewerWorldSessionBootstrapper.cs` as the app-owned world bootstrap service over `MapDirectoryLookup`, `ArchiveCatalogBootstrapper`, `MapFileSummaryReader`, `WdtSummaryReader`, and `WdtTileIndexReader`
	- `WowViewerDesktopApp.cs` now treats `World Session` as an implemented workspace with its own controls, summary surface, and WDT/tile diagnostics, while keeping the boundary explicit that no world renderer exists yet
	- `Program.cs` now supports `--workspace world` for desktop bootstrap and a direct `world-bootstrap` CLI proof command
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the existing workspace `LIB` warnings only
	- real-data bootstrap proof succeeded via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- world-bootstrap --client-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --build-label 3.3.5.12340`
	- that proof resolved `Azeroth` through `Map.dbc`, opened `World\Maps\Azeroth\Azeroth.wdt` from archive-backed data, and reported `687` occupied tiles with `MAIN` distinct flag summary `0x1:687`
- boundary:
	- this closes attach/open plus WDT/bootstrap proof only
	- a real world runtime consumer still remains the next separate slice

### Apr 17, 2026 - wow-viewer app slice 04 landed: the standalone M2 workspace now has a bounded GPU preview consumer

- what changed:
	- added `wow-viewer/src/viewer/WowViewer.App/M2GpuPreviewRenderer.cs` as an app-local GL consumer over `M2RenderFrame.DrawCommands`
	- added `wow-viewer/src/viewer/WowViewer.App/M2GpuPreviewCaptureRunner.cs` and a new `m2-gpu-frame` command in `Program.cs` so the same renderer can write hidden-window BMP proof artifacts
	- `WowViewerDesktopApp.cs` now uses that GPU renderer as the active standalone M2 preview path when loaded geometry exists, while keeping the software visual snapshot as an explicit fallback and diagnostic reference
	- `WowViewer.App.csproj` now references the vendored `SereniaBLPLib` BLP decoder, and `wow-viewer/libs/WoW-Tools/SereniaBLPLib/SereniaBLPLib/SereniaBLPLib.csproj` now explicitly disables central package management so the vendored project can restore cleanly under this workspace
	- `WowViewer.Core.Runtime/M2/M2RenderFrame.cs` now carries the per-command material/effect state the GPU consumer needs: diffuse or emissive color, alpha, blend mode, depth-write, alpha-test, transparency, additive state, and lighting flags
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the existing workspace `LIB` warnings only
	- real-data GPU proof succeeded via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- m2-gpu-frame --archive-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --virtual-path "Creature/Wolf/Wolf.m2" --build-label 3.3.5.12340 --sequence-index 0 --time-ms 0 --visual-size 512 --output "i:/parp/parp-tools/output/build-validation/wow-viewer-app-gpu-preview/wolf_335_gpu.bmp"`
	- the resulting proof artifact exists at `output/build-validation/wow-viewer-app-gpu-preview/wolf_335_gpu.bmp` (`1048630` bytes)
	- the older runtime proof still stayed stable through `m2-frame`, preserving runtime `9e9586068a443468ccec1abd62b3d717c0455e08999bd03beb21427a9df4ec30`, render-frame `177155d088dc8502be5b115b6b3d1a0fa67e75549cfe87c981bff6a8f8ac4122`, and visual `b2fabb6da814c393ea149fb7321cbd3e05d24db8852f59dca35c755c29bfb177`
- boundary:
	- this is a bounded standalone GPU preview slice only, not full native material parity or world ownership
	- camera-only overlay parity, WMO/MDX consumers, and world-session bootstrap remain separate follow-up slices

### Apr 17, 2026 - wow-viewer app slice 03 landed: the desktop shell now exposes explicit standalone workspaces

- what changed:
	- `wow-viewer/src/viewer/WowViewer.App/WowViewerSession.cs` now defines explicit standalone workspace modes for M2, WMO, and MDX
	- `WowViewerDesktopApp.cs` now has a dedicated `Workspaces` window and view toggle, and the control, preview, and diagnostics windows now reflect the active standalone workspace instead of always presenting as one generic M2 surface
	- only `StandaloneM2` is implemented in this slice; `StandaloneWmo` and `StandaloneMdx` are deliberate placeholder surfaces that state they are not implemented yet
	- `WowViewerAppSettings.cs` now persists workspace-window visibility, and `Program.cs` now supports `--workspace m2|wmo|mdx` when launching the desktop viewer
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the existing workspace `LIB` warnings only
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- --help` showed the new workspace bootstrap surface
	- real-data M2 proof still succeeded via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- m2-frame --archive-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --virtual-path "Creature/Wolf/Wolf.m2" --build-label 3.3.5.12340 --sequence-index 0 --time-ms 0`
	- that proof preserved the earlier Wolf hashes: runtime `9e9586068a443468ccec1abd62b3d717c0455e08999bd03beb21427a9df4ec30`, render-frame `177155d088dc8502be5b115b6b3d1a0fa67e75549cfe87c981bff6a8f8ac4122`, visual `b2fabb6da814c393ea149fb7321cbd3e05d24db8852f59dca35c755c29bfb177`
- boundary:
	- this closes the shell-side workspace split only
	- WMO and MDX are not implemented consumers yet, and the app still lacks a GPU preview renderer and world-session bootstrap

### Apr 17, 2026 - wow-viewer app slice 02 landed: the desktop host now runs through a typed viewer-session contract

- what changed:
	- added `wow-viewer/src/viewer/WowViewer.App/WowViewerSession.cs` as the app-local session seam for workspace mode, typed asset source selection, build label, and preview request state
	- `WowViewerDesktopApp.cs` now uses that session object instead of keeping raw source/build/preview fields directly on the host
	- `WowViewerAppSettings.cs` now persists the session object rather than the earlier flat source-setting blob
	- `Program.cs` now parses `viewer` bootstrap arguments into a typed session object, while `m2-frame` stays on the narrower request path used for direct runtime proof
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the existing workspace `LIB` warnings only
	- real-data app/runtime proof still succeeded via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- m2-frame --archive-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --virtual-path "Creature/Wolf/Wolf.m2" --build-label 3.3.5.12340 --sequence-index 0 --time-ms 0`
	- that proof preserved the earlier Wolf hashes: runtime `9e9586068a443468ccec1abd62b3d717c0455e08999bd03beb21427a9df4ec30`, render-frame `177155d088dc8502be5b115b6b3d1a0fa67e75549cfe87c981bff6a8f8ac4122`, visual `b2fabb6da814c393ea149fb7321cbd3e05d24db8852f59dca35c755c29bfb177`
- boundary:
	- this closes the host-side session seam only; the app still does not have a dedicated workspace split, GPU preview renderer, or world-session bootstrap yet

### Apr 17, 2026 - wow-viewer viewer-app cutover now has an explicit staged plan, and slice 01 app settings persistence is landed

- what changed:
	- added `gillijimproject_refactor/plans/wow_viewer_viewer_app_cutover_plan_2026-04-17.md` as the dedicated sequence for replacing old `ViewerApp` ownership with a real wow-viewer app over narrow slices instead of another monolithic migration note
	- the plan now stages the work as: app settings persistence, viewer session boundary, standalone asset workspaces, GPU M2 preview consumer, world session bootstrap, world runtime consumer bridge, shell surface expansion, and final legacy cutover review
	- implemented slice 01 in `wow-viewer/src/viewer/WowViewer.App/`:
		- `WowViewerAppSettings.cs` now owns wow-viewer-local persisted settings
		- `WowViewerDesktopApp.cs` now loads/saves archive-vs-local source mode, source paths, profile/sequence/time values, preview size, and core window toggles through `output/settings/wowviewer_app_settings.json`
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the existing workspace `LIB` warnings only
- boundary:
	- this makes the new app behave like a persistent wow-viewer-owned shell instead of a disposable probe window, but it is still not a typed viewer-session layer or a world-runtime consumer yet

### Apr 16, 2026 - wow-viewer now has a real desktop app shell instead of only the `m2-frame` console harness

- what changed:
	- `wow-viewer/src/viewer/WowViewer.App` now references Silk.NET windowing, OpenGL, input, and ImGui packages so the new repo can host its own viewer window
	- `WowViewer.App` now opens a docked desktop shell by default, while keeping `m2-frame` as a CLI command instead of deleting the earlier proof surface
	- the new shell uses a shared app-local `M2PreviewLoader`, so both the GUI preview and `m2-frame` go through the same runtime-owned `M2ModelReader` -> skin/runtime pipeline -> `M2RuntimeFramePipeline` path
	- the first desktop shell is intentionally narrow and explicit about scope: it supports archive-backed or local M2 requests, uploads the deterministic software visual snapshot as the preview image, and surfaces runtime hashes plus submission diagnostics and current runtime-boundary notes
	- the new app slice does not reference `gillijimproject_refactor/src/MdxViewer`; ownership of the shell and loader path now lives in `wow-viewer`
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` succeeded with the existing workspace `LIB` warnings only
	- real fixed-root proof through the shared loader path succeeded with `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- m2-frame --archive-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --virtual-path "Creature/Wolf/Wolf.m2" --sequence-index 0 --time-ms 0`
	- that proof reported `sequence=0->0`, `skinnedVertices=557`, `batches=2`, runtime hash `9e9586068a443468ccec1abd62b3d717c0455e08999bd03beb21427a9df4ec30`, render-frame hash `177155d088dc8502be5b115b6b3d1a0fa67e75549cfe87c981bff6a8f8ac4122`, and visual hash `b2fabb6da814c393ea149fb7321cbd3e05d24db8852f59dca35c755c29bfb177`
- boundary:
	- this is a real wow-viewer-owned app-shell slice, but it is still an M2 preview consumer over the software visual snapshot, not the final GPU renderer or a world-scene host
	- no interactive screenshot or world-load proof was captured yet for the new desktop window itself, so do not describe this as full viewer cutover

### Apr 17, 2026 - MdxViewer weak-signal terrain restore now covers full weak-signal ADT evidence again, with range-based per-cell masking instead of chunk or texture-bucket selection

- what changed:
	- `src/MdxViewer/ViewerApp.cs` now refreshes restore from camera movement again instead of using workbench-scope or loaded-tile precedence as the active path
	- `ShouldApplyTerrainWeakSignalRestoreToTile(...)` is still limited to the camera tile plus four direct neighbors, but it now accepts partial weak-signal evidence from `HasTerrainWeakSignalRestoreWholeTileEvidence(...)` instead of requiring the entire ADT to sit inside the weak-signal Z band
	- `TryBuildTerrainWeakSignalRestoredChunks(...)` still delegates to the whole-tile restore path, and the mixed-tile evidence path now uses only per-cell range checks instead of whole-chunk range checks or texture-bucket selection
	- `src/MdxViewer/ViewerApp_Sidebars.cs` now describes one active mode (`whole-tile factor, per-cell weak-signal clamp`) and no longer exposes the loaded-tiles or MCSH shadow-edge toggles in the active restore UI
	- persisted viewer settings now force the loaded-tiles and shadow flags off, and weak-signal restore itself now always loads disabled and saves back disabled so the viewer does not auto-enable the feature on startup anymore
- validation:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`
	- a fresh full `dotnet build` after this change is currently blocked by the already-running `ParpToolsWoWViewer` process locking `bin/Debug/net10.0-windows/ParpToolsWoWViewer.exe` and `.dll`
- boundary:
	- this is still not real-data validated; no fresh runtime proof was captured yet for the simplified camera-neighbor restore behavior with the broadened per-cell mask
	- if boundary seams still look wrong, the next implementation step should move the masked restore onto the shared `TileHeightmap257` grid before writing back into per-chunk data

### Apr 16, 2026 - the default wow-viewer M2 renderer in MdxViewer now advances skeletal animation and exposes runtime sequence controls instead of staying static-only

- what changed:
	- `src/MdxViewer/Rendering/IAnimationController.cs` now defines a renderer-agnostic animation-control contract so the viewer UI and keyboard controls no longer depend on `MdxAnimator` specifically
	- `src/MdxViewer/Rendering/MdxAnimator.cs` now implements that shared controller contract for the legacy MDX path without changing its existing bone-evaluation behavior
	- `src/MdxViewer/Rendering/M2RuntimeAnimator.cs` now owns pure-runtime sequence selection, frame advancement, and `%04d-%02d.anim` companion loading through the active `IDataSource`
	- `src/MdxViewer/Rendering/M2Renderer.cs` now uses wow-viewer runtime animation evaluation (`M2AnimatedRenderStateEvaluator`, `M2BonePoseEvaluator`, `M2SkinnedRenderModelBuilder`, `M2RenderConsumerFrameStateBuilder`) on each viewer update, uploads posed vertices back into the GL VBOs, and exposes runtime sequence playback through the shared animation-controller surface
	- `src/MdxViewer/ViewerApp_StartupAutomation.cs` and `src/MdxViewer/ViewerApp_CaptureAutomation.cs` now accept `--capture-after-frames` so startup captures can intentionally wait multiple settled frames before saving, which is useful for animation proof without manual UI interaction
	- `src/MdxViewer/ViewerApp.cs` runtime-notes text no longer calls the pure runtime path static-only; it now says skeletal sequence playback is active while full material/effect parity is still pending
- validation:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` succeeded with the same pre-existing workspace warnings only
	- fixed local Wrath client proof on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `Creature/Wolf/Wolf.m2`, using the default runtime renderer and startup capture delay support, wrote:
		- early frame: `output/build-validation/m2-runtime-animation/wolf/frame-001/standalone/3.3.5.12340/20260416_055536139_current_20260416_055536_no_ui.png`
		- later frame: `output/build-validation/m2-runtime-animation/wolf/frame-040/standalone/3.3.5.12340/20260416_055627078_current_20260416_055627_no_ui.png`
	- direct RGB comparison between those two captures reported `4996` changed pixels with bounding box `(514, 308, 586, 468)`, which sits tightly on the rendered wolf and confirms the runtime viewer path is changing rendered pose across frames instead of only compiling animation code
- boundary:
	- this is real active-viewer proof for skeletal pose playback in the pure runtime M2 path, not full animation parity; texture-transform animation, richer material/effect behavior, particles, and ribbons still remain separate follow-up work

### Apr 16, 2026 - the wow-viewer static M2 renderer is now the default viewer path, and the old env var now serves as a legacy opt-out instead of an opt-in

- what changed:
	- `Rendering/WowViewerM2RuntimeBridge.cs` now defaults successful runtime-backed M2 loads to the pure wow-viewer static renderer path even when `PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER` is unset
	- setting `PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER=0|false|no|off` now forces the old legacy compatibility draw backend when an adapted MDX fallback exists
	- standalone model info in `ViewerApp.cs` now reports that the wow-viewer static renderer is the default and that the env var is the legacy escape hatch, not the activation switch
- validation:
	- `get_errors` returned clean for `src/MdxViewer/Rendering/WowViewerM2RuntimeBridge.cs` and the touched `src/MdxViewer/ViewerApp.cs` text update
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` succeeded with the same existing workspace warnings only
	- fixed local Wrath client proof on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `Character/Human/Male/HumanMale.m2`, with no `PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER` variable set, wrote:
		- `output/build-validation/m2-character-humanmale/default-no-env/standalone/3.3.5.12340/20260416_053846167_current_20260416_053846_no_ui.png`
	- that no-env capture matches the already-fixed runtime character presentation rather than falling back to the old legacy-only default route
- boundary:
	- this changes the active viewer default, not the underlying runtime feature boundary; animation and full material parity are still pending

### Apr 16, 2026 - the pure runtime standalone M2 path now applies default character geoset selection and character variation overrides instead of drawing the full raw HumanMale geoset stack

- what changed:
	- `Rendering/M2Renderer.cs` now exposes the same character customization seam the legacy path already had: it can apply character geoset-selection groups against runtime `SkinSectionId` values and reload replaceable textures with selected hair or facial variation ids
	- `ViewerApp.cs` now refreshes standalone character customization state after `LoadM2RuntimeModel(...)` and routes `ApplyStandaloneCharacterCustomizationOverrides()` into both `MdxRenderer` and `M2Renderer`, instead of only the legacy MDX renderer path
- validation:
	- `get_errors` returned clean for `src/MdxViewer/Rendering/M2Renderer.cs` and `src/MdxViewer/ViewerApp.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` succeeded with the same existing workspace warnings only
	- fixed local Wrath client proof on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `Character/Human/Male/HumanMale.m2`, wrote:
		- pure runtime before fix: `output/build-validation/m2-character-humanmale/runtime/standalone/3.3.5.12340/20260416_052736076_current_20260416_052736_no_ui.png`
		- legacy control: `output/build-validation/m2-character-humanmale/legacy/standalone/3.3.5.12340/20260416_052750685_current_20260416_052750_no_ui.png`
		- pure runtime after fix: `output/build-validation/m2-character-humanmale/runtime-after-character-fix/standalone/3.3.5.12340/20260416_053313303_current_20260416_053313_no_ui.png`
	- the before image showed the pure runtime route still drawing the extra character geoset stack, while the after image collapses to the same default bare-body presentation as the legacy control
- boundary:
	- this closes one concrete standalone player-character repro around default geoset selection and variation handoff in `MdxViewer`
	- it does not yet mean full character-model parity for animation, material behavior, or every player or NPC family

### Apr 16, 2026 - the pure runtime no-cull fix appears to generalize across a small AhnQiraj passive-doodad object sweep, but character parity is still open

- what changed:
	- no new code landed in this slice; the goal was to validate whether the earlier `Rendering/M2Renderer.cs` no-cull change was a one-off `FoodHerbs_Level01.m2` repair or whether it closed the same failure shape across nearby doodad siblings
- validation:
	- fixed local Wrath client proof on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft` reran the same startup-capture comparison on three nearby AhnQiraj passive doodads
	- pure runtime captures with `PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER=1` wrote:
		- `output/build-validation/m2-object-sweep/runtime/Food_Level02/standalone/3.3.5.12340/20260416_052054291_current_20260416_052054_no_ui.png`
		- `output/build-validation/m2-object-sweep/runtime/Food_Level03/standalone/3.3.5.12340/20260416_052105099_current_20260416_052105_no_ui.png`
		- `output/build-validation/m2-object-sweep/runtime/Cloth_Level01/standalone/3.3.5.12340/20260416_052116025_current_20260416_052116_no_ui.png`
	- legacy control captures on the same assets wrote:
		- `output/build-validation/m2-object-sweep/legacy/Food_Level02/standalone/3.3.5.12340/20260416_052131779_current_20260416_052131_no_ui.png`
		- `output/build-validation/m2-object-sweep/legacy/Food_Level03/standalone/3.3.5.12340/20260416_052142675_current_20260416_052142_no_ui.png`
		- `output/build-validation/m2-object-sweep/legacy/Cloth_Level01/standalone/3.3.5.12340/20260416_052156383_current_20260416_052156_no_ui.png`
	- visual comparison says the earlier hollow or missing-face failure shape is no longer present on these three siblings; the two food variants look effectively aligned with the legacy controls, and the cloth sample is close enough that it does not show the previous projected-object collapse pattern
- boundary:
	- treat this as small real-data object-family evidence, not blanket M2 signoff
	- the user-reported character-model problems remain a separate unresolved track

### Apr 16, 2026 - projected object sections in the pure runtime M2 path were being culled away, which made `FoodHerbs_Level01.m2` crates look hollow

- what changed:
	- `Rendering/M2Renderer.cs` no longer enables backface culling for the pure runtime M2 renderer; it now matches the established legacy M2 path and keeps culling disabled while projected and mixed-winding object sections remain unproven
- validation:
	- fixed local Wrath client proof on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `World/Generic/PassiveDoodads/AHNQIRAJ/FoodHerbs_Level01.M2`, with `PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER=1`, wrote:
		- before no-cull fix: `output/build-validation/m2-foodherbs-runtime-correct/standalone/3.3.5.12340/20260416_051238388_current_20260416_051238_no_ui.png`
		- after no-cull fix: `output/build-validation/m2-foodherbs-runtime-after-cull-fix/standalone/3.3.5.12340/20260416_051511729_current_20260416_051511_no_ui.png`
	- legacy control on the same asset wrote:
		- `output/build-validation/m2-foodherbs-legacy-correct/standalone/3.3.5.12340/20260416_051302760_current_20260416_051302_no_ui.png`
	- `WowViewer.Tool.Inspect m2 inspect` on the same asset showed that the visibly broken crate and prop sections were largely `Diffuse_Projected:*` opaque or alpha-key passes, which fits the culling failure shape more than a texture-binding bug
- boundary:
	- this materially improves object-family runtime rendering for at least one real projected-heavy Wrath asset, but it is not a blanket closure for the separate character-model parity problems

### Apr 16, 2026 - runtime static M2 UV preservation moved `Band_DrumSet.m2` from black-failure proof to textured live capture, but projected or additive material parity is still open

- what changed:
	- `WowViewer.Core.Runtime/M2/M2StaticRenderModel.cs` and `M2StaticRenderModelBuilder.cs` now preserve both `TextureCoords0` and `TextureCoords1` into the runtime static vertex contract instead of flattening everything to UV0
	- `Rendering/M2Renderer.cs` now uploads both UV streams and lets each runtime texture binding choose UV0 or UV1
	- the pure runtime shader path also stopped multiplying sections by arbitrary debug tint, and it now treats `coordLookupValue=65535` as generated view-normal coordinates instead of blindly sampling UV0
- validation:
	- focused `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "M2FoundationTests|M2RuntimeTests"` passed `34/34`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` succeeded with the same existing invalid `LIB` path warnings only
	- fixed local Wrath client proof on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `Creature/band/Band_DrumSet.M2`, with `PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER=1`, wrote:
		- `output/build-validation/m2-native-static-texture-path/standalone/3.3.5.12340/standalone/3.3.5.12340/20260416_045415087_current_20260416_045415_no_ui.png`
	- legacy control capture on the same fixed local root and asset wrote:
		- `output/build-validation/m2-native-static-texture-path/legacy-control/standalone/3.3.5.12340/20260416_045517327_current_20260416_045517_no_ui.png`
- boundary:
	- this closes the earlier black or missing-texture live failure for one real geoset-heavy Wrath asset, but the pure runtime renderer still trails the legacy compatibility path on projected and additive material behavior
	- the `Band_DrumSet` control comparison still shows that `Diffuse_Projected:*` and additive overlay passes are not yet at parity even after the UV-contract repair; do not describe this as full geoset or material signoff

### Apr 16, 2026 - native static M2 texturing now has real viewer proof, and MPQ-backed `Scry_cam.m2` now renders through the camera-path route instead of failing on empty geometry

- what changed:
	- `Rendering/WowViewerM2RuntimeBridge.cs` now hands the pure runtime `M2Renderer` the data-source and replaceable-texture dependencies it was already shaped to use
	- `Rendering/M2Renderer.cs` now uploads UVs, resolves primary-stage texture bindings from the runtime material state, samples the resolved texture in the shader, and cleans up owned GL textures
	- `WowViewer.Core.Runtime/M2/M2CameraPathOverlayBuilder.cs` no longer rejects real camera-only assets just because they advertise one dummy view or carry a helper bone; the runtime now accepts camera-style canonical paths such as `*_cam.m2` or `Cameras\...` when the asset has cameras and no ribbon or particle families
	- added focused regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/M2RuntimeTests.cs` for the dummy-view-count camera asset shape
- validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "M2FoundationTests|M2RuntimeTests"` passed `34/34`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` succeeded with existing invalid `LIB` path warnings only
	- fixed local Wrath client proof on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft` via startup automation wrote:
		- textured Wolf capture: `output/build-validation/m2-native-static-texture-path/standalone/3.3.5.12340/20260416_040737484_current_20260416_040737_no_ui.png`
		- camera-path `Scry_cam` capture: `output/build-validation/m2-native-static-texture-path/standalone/3.3.5.12340/20260416_042648038_current_20260416_042648_no_ui.png`
- boundary:
	- this is real active-viewer proof for one textured static M2 and one camera-only `*_cam.m2` asset on the Wrath baseline, but the static path is still simplified runtime shading rather than full native render parity
	- the camera-only path remains sampled overlay visualization rather than mesh rendering

### Apr 16, 2026 - standalone `*_cam.m2` assets now bypass `.skin` loading and render as camera-path overlays

- what changed:
	- `WowViewer.Core.IO/M2/M2ModelReader.cs` strict MD20 ownership now includes first-class camera definitions on `M2ModelDocument`
	- `WowViewer.Core.Runtime/M2/M2CameraPathOverlayBuilder.cs` plus `M2CameraPathVisualization.cs` now own camera-only M2 classification, sampled path generation, and overlay bounds in the canonical repo
	- `MdxViewer` standalone M2 loading now probes the strict MD20 root, asks the wow-viewer runtime whether the asset is a camera-path candidate, and consumes the prebuilt overlay layout instead of owning that sampling logic locally
	- `Rendering/M2CameraPathRenderer.cs` is now only the GL line-draw consumer over wow-viewer-owned overlay data instead of being the design owner for camera-path interpretation
	- standalone model info and the model sidebar framing action now work for the camera-path renderer too
- validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "M2FoundationTests|M2RuntimeTests"` passed `33/33`
	- added focused strict-reader coverage for a synthetic camera-only MD20 root in `M2FoundationTests`
	- added focused runtime coverage for camera-path overlay generation in `M2RuntimeTests`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` still fails, but the remaining errors are from the earlier unfinished `M2Renderer` texture-path slice (`SelectBestReplaceableDisplayIndex`, `LoadSectionTextures`) rather than the new camera-path work
- boundary:
	- this closes the over-strict `.skin` assumption for standalone geometry-less flyby camera assets and moves the interpretation logic into `wow-viewer`, but it is still path visualization only and not a general fix for the separate unfinished textured static M2 renderer work

### Apr 15, 2026 - MdxViewer first gained an opt-in pure wow-viewer M2 renderer route for live testing


- what changed at that time:
	- `Rendering/WowViewerM2RuntimeBridge.cs` centralized M2 renderer-route selection for successful runtime-backed M2 loads
	- at that stage, setting `PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER=1` made standalone M2 loads, streamed world M2 loads, and WMO doodad M2 loads use the pure static `M2Renderer(_gl, runtimeModel, ...)` path inside `MdxViewer`
	- the default changed later on Apr 16, 2026; the same env var is now the legacy opt-out switch instead of the activation switch
	- standalone model info now tells the user whether the currently loaded M2 is using the pure wow-viewer static renderer or the legacy compatibility draw path
- validation:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` succeeded with existing workspace warnings only
	- live viewer testing with `PARP_M2_USE_WOW_VIEWER_RUNTIME_RENDERER=1` now indicates the route is stronger than a geometry-only curiosity: most non-character objects look plausibly correct, including M2 doodads inside WMOs
- boundary:
	- character-family assets remain the main visible problem area; texturing, geoset correctness, and some surface/material ordering still need focused work there
	- this is still not full textured-material parity; the exposed pure runtime renderer remains a simplified static shaded path

### Apr 15, 2026 - wow-viewer Wolf M2 geometry corruption was fixed in the static mesh path

- root cause:
	- `WowViewer.Core.Runtime/M2/M2StaticRenderModelBuilder.TryGetVertex` was treating strict skin header field `0x2C` as a blind vertex base offset before trying the direct skin lookup entry
	- on the real Cataclysm `Creature/Wolf/Wolf00.skin` proof, that field reported `53`, which matches documented `boneCountMax` values and produced a bad vertex shift over an already-complete `557`-entry local lookup table
	- the same session also proved the corruption was present in the static mesh output before skinning, so pose math was not the primary fault
- fix:
	- runtime vertex fetch now resolves the direct skin lookup entry first and only uses the extra header field as a fallback when the direct lookup is invalid
	- strict skin parsing now suppresses bogus optional shadow-batch metadata unless the advertised span is actually valid
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with existing invalid `LIB` path warnings only
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter "FullyQualifiedName~M2FoundationTests|FullyQualifiedName~M2RuntimeTests"` passed with `31` matching core tests
	- fixed local client proof on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `Creature/Wolf/Wolf.m2`, sequence `0`, time `0`, wrote:
		- `output/build-validation/wow-viewer-m2-wolf-idle-static-visual-fixed.bmp`
		- `output/build-validation/wow-viewer-m2-wolf-idle-skinned-visual-fixed.bmp`
	- those new proof outputs now read as a recognizable quadruped silhouette instead of the earlier wedge-like random geometry
	- the same proof now reports `shadowBatches=0`, render-frame hash `86048f9de460bb5e75a557d526609700f4292b61ccc0f8eae4b4bd6206f012bb`, and software visual hash `71aff63b3d0fba7e1eba03bcad894f2af0f2c87448fc9d706a976506b9f17ee5`
	- the same Wolf asset on `H:/CLIENTS/World of Warcraft Cata beta 11927` produced matching corrected counts and hashes, but that was a cross-build check rather than the active baseline
- remaining boundary:
	- this is a real first-party mesh-assembly fix in `wow-viewer`, but it is still software-proof validation rather than final GPU renderer signoff

### Apr 15, 2026 - wow-viewer M2 now has a shared app/inspect frame pipeline plus render-frame and software-visual proof outputs

- advanced the consumer slice beyond separate app and inspect orchestration:
	- extracted end-to-end M2 frame assembly into `WowViewer.Core.Runtime/M2/M2RuntimeFramePipeline`
	- the shared result now carries animated state, bone pose, skinned render model, render-consumer state, effect runtime state, submission plan, render frame, software visual snapshot, and golden frame
	- `WowViewer.App m2-frame` now consumes that shared runtime pipeline directly
	- `WowViewer.Tool.Inspect m2 inspect` now consumes the same pipeline and supports `--render-frame-output` plus `--visual-output` alongside `--golden-output`
	- added focused coverage in `M2RuntimeTests` for the shared pipeline result and deterministic render or visual hashes
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with existing invalid `LIB` path warnings only
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter "M2FoundationTests|M2RuntimeTests"` passed with `31` matching core tests
	- fixed local client proof on `H:/CLIENTS/World of Warcraft Cata beta 11927`, `Creature/Wolf/Wolf.m2`, sequence `20`, time `500`, produced matching app and inspect hashes for:
		- runtime golden state `113f55daaad3e996476eeff4c9e6fe37aa4c4d3cc364a48e38c6a86bc6fb980e`
		- render frame `a285c8ef68b0d3304a55d93a30a34f4722fea7c9ed9d429fd5bf1db903932988`
		- software visual snapshot `8880ba87d37662a59c8b07d040a7eeb40b1a1060585c9593b197712db6ccf5ec`
- remaining boundary:
	- this is still not active visual renderer signoff or GPU backend parity
	- the current visual output is a deterministic software proof harness over runtime draw data, which is useful for regression evidence but not final render closure

### Apr 15, 2026 - wow-viewer M2 runtime now has app-level frame consumption and golden snapshot proof

- advanced the staged M2 runtime work in `wow-viewer` beyond the previous inspect-only boundary:
	- added `M2EffectRegistry` / `M2ResolvedEffect` so runtime effect consumption now exposes native-style effect object keys, native family keys, blend/depth/alpha-test decisions, lighting flags, and state buckets
	- added family-aware particle/ribbon submission descriptors and explicit scene family policies with named handlers
	- moved core render-entry construction into `M2SceneSubmissionEntryBuilder` so inspect and app consumers share the same runtime contract
	- added `M2RuntimeGoldenFrameBuilder` for deterministic golden snapshots and runtime hashes
	- replaced the `WowViewer.App` console skeleton with an `m2-frame` command that loads `MD20`, exact skin, external anim, pose/skinning, render-consumer state, submission plan, and optional golden JSON
	- extended `m2 inspect` with `--golden-output` / `-g`, resolved effect-object output, and handler/state-scope submission output
- validation:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter "M2FoundationTests|M2RuntimeTests"` passed with `27` matching core tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug` passed with existing invalid `LIB` path warnings only
	- fixed local client proof on `H:/CLIENTS/World of Warcraft Cata beta 11927`, `Creature/Wolf/Wolf.m2`, sequence `20`, time `500`, produced app and inspect golden snapshots with matching hash `113f55daaad3e996476eeff4c9e6fe37aa4c4d3cc364a48e38c6a86bc6fb980e`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with existing invalid `LIB` path warnings only
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with `263` `WowViewer.Core.Tests` and `36` `WowViewer.Core.PM4.Tests`
- remaining boundary:
	- this is first app/runtime consumer proof and golden-state proof, not visual renderer signoff
	- particle/ribbon parser, simulation, generated geometry, and GPU submission remain future work; the landed slice closes the contract and handler-policy layer first

### Apr 15, 2026 - wow-viewer M2 slice 03 advanced past inspect-only animation, and slice 04 now has a first coordinator seam

- landed new `wow-viewer` M2 runtime ownership:
	- typed `M2BoneDefinition` parsing from `MD20`
	- shared `M2TrackSampler` with compressed M2 quaternion sampling
	- `M2BonePoseEvaluator` for sequence/time pose matrices
	- `M2SkinnedRenderModelBuilder` for CPU-side skinned render vertices over the structured render sections
	- `M2RenderConsumerFrameStateBuilder` so evaluated material/light state becomes renderer-facing pass state instead of remaining inspect-only data
	- `M2SceneSubmissionCoordinator` plus runtime option flags for first-pass family/state/capacity batching policy
	- inspect output now includes `ANIM.POSE`, `RENDER.CONSUMER`, and `SCENE.SUBMISSION`
- focused tests added in `M2RuntimeTests` for:
	- bone table parsing
	- parented pose solve plus skinning through skin bone lookup metadata
	- render-consumer light/material state
	- scene submission grouping, capacity splitting, and particle batching policy
- validation:
	- focused M2 test filter passed `24/24`
	- full `wow-viewer` build passed
	- full `wow-viewer` test suite passed: `260` core tests and `36` PM4 tests
	- real fixed-client proof used `H:/CLIENTS/World of Warcraft Cata beta 11927`, `Creature/Wolf/Wolf.m2`, sequence `20`, exact `Wolf0096-00.anim`, and printed the new runtime consumer/submission summaries
- remaining boundary:
	- this does not yet mean active app renderer cutover, final shader/effect parity, particle/ribbon submission implementation, or old `MdxViewer` runtime parity

### Apr 15, 2026 - wow-viewer M2 slices 01 and 02 are now real, and slice 03 is partially landed

- landed `wow-viewer` M2 ownership now includes:
	- strict `MD20` root parse, exact `%02d.skin` choose/load/init runtime, and first-party geometry/material tables
	- structured section/pass/material routing plus effect-recipe classification
	- external `%04d-%02d.anim` selection/load and alias ready-state ownership
	- first-party animated block parsing for colors, transparency weights, texture transforms, and lights
	- first-pass animated runtime evaluation and `m2 inspect --time-ms` output
- focused proof completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter M2FoundationTests` passed `19/19`
	- real asset probe on `Creature/Wolf/Wolf.m2` from `H:/CLIENTS/World of Warcraft Cata beta 11927` loaded external `Wolf0096-00.anim` and printed first-party `ANIM.RUNTIME` output
- remaining M2 work is no longer “foundation”:
	- animated bone pose and skinning application
	- render-consumer use of evaluated material/light state
	- family-specific runtime ownership and scene submission/batching
	- consumer cutover/parity harness beyond inspect
- proof boundary:
	- this is still not active-viewer runtime signoff or full renderer parity

### Apr 15, 2026 - landed uv-based training deployment scripts and removed implicit CPU fallback in train_v7

- implemented dedicated uv-managed training environment bootstrap scripts:
	- `gillijimproject_refactor/scripts/setup_training_env.ps1`
	- `gillijimproject_refactor/scripts/setup_training_env.sh`
	- `gillijimproject_refactor/scripts/requirements_train_v7.txt`
- bootstrap flow now creates a dedicated training venv (default `.venv-train`) on Python `3.11`, installs shared non-torch deps, installs backend-specific torch wheels (`cu128`, `rocm6.2.4`, `cpu`, or PyPI for `mps`), and validates requested accelerator capability before declaring success
- `src/WoWMapConverter/scripts/train_v7.py` no longer silently trains on CPU when CUDA is unavailable:
	- added explicit training-device resolver with hard failure by default when CUDA is missing
	- added `--allow-cpu` for intentional CPU-only debug runs
	- fail-fast diagnostic now prints Python executable plus torch CUDA/HIP build metadata and points users to the uv bootstrap scripts
- updated training docs:
	- `gillijimproject_refactor/docs/VLM_Training_Guide.md` now includes uv bootstrap commands and the new explicit CPU-override behavior
- proof boundary:
	- this slice is deployment/bootstrap and trainer safety behavior only
	- it does not yet implement a full pyproject/lockfile training packaging workflow or cross-repo training runner abstraction

### Apr 15, 2026 - recorded the real CPU training failure and redirected M2 continuation back to wow-viewer-owned parser/renderer work

- confirmed the immediate training failure was environment drift, not a mystery trainer choice:
	- active interpreter: `i:/parp/parp-tools/.venv/Scripts/python.exe`
	- active torch build: `2.11.0+cpu`
	- `torch.version.cuda = None`
	- `torch.cuda.is_available() = False`
	- host GPU remains visible through `nvidia-smi` (`RTX 4070 Ti SUPER`, driver `595.97`)
- confirmed the current trainer logic in `src/WoWMapConverter/scripts/train_v7.py` still silently falls back to CPU with `torch.device("cuda" if use_cuda else "cpu")` instead of failing fast when the environment is wrong
- captured the new workflow directive in continuity:
	- stop accepting haphazard training env selection; future training-env work should use a reproducible `uv`-managed bootstrap and deployment validation for target hardware
	- stop routing M2 fix work back into `MdxViewer` bandaids as the design owner
	- active corrective path is now a full first-party M2 parser/renderer cutover in `wow-viewer`
- added a new workflow asset for that correction:
	- `.github/prompts/wow-viewer-full-m2-parser-renderer-plan.prompt.md`
- refreshed the existing M2 prompt routing so future chats can distinguish:
	- full parser/renderer cutover planning
	- residual foundation ownership cleanup
	- narrower staged runtime follow-on slices

### Apr 15, 2026 - ML corpus export now has an explicit resume path instead of re-exporting completed map roots forever

- the user-reported rerun waste was real:
	- `scripts/export_ml_corpus.ps1` had been changed to always re-export every configured map so partial roots would not be silently treated as complete
	- that fixed stale partial datasets, but it also meant broad corpus reruns kept re-running already finished roots like `datasets/3_3_5_12340/*` with no durable notion of completion state
- active behavior now:
	- `gillijimproject_refactor/scripts/export_ml_corpus.ps1` accepts `-Resume`
	- `WoWMapConverter.Cli ml-corpus` accepts `--resume`
	- both entrypoints now persist per-map resume metadata in `.ml-corpus-resume-state.json` inside each dataset map root
	- resume considers a map complete when either:
		- matching resume state says export and harvest already finished for the same job settings, or
		- the existing `ml_dataset_manifest.json` is current against the actual tile JSON count and timestamps
	- when export is complete but harvest metadata is stale, resume skips `ml-export` and runs only `ml-harvest`
	- wrapper harvest gating now counts only real tile JSON files and ignores helper files like `texture_database.json`
- focused validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` succeeded with existing workspace warnings only
	- PowerShell parse validation for `scripts/export_ml_corpus.ps1` passed after the resume patches
	- `scripts/export_ml_corpus.ps1 -DryRun -Resume` now skips the already-complete `original_development/development` root without falling through into `ml-export` or `ml-harvest`
	- direct `ml-corpus --dry-run --resume` also reports the completed `original_development/development` root as a resume skip instead of scheduling fresh export work
- important boundary:
	- PowerShell wrapper dry-run still does not execute live `ml-list-maps`, so `all_maps` clients report no discovered maps in dry-run mode by design; use a real `-Resume` run for end-to-end map scheduling proof
	- existing completed roots do not need a pre-existing resume-state file because resume falls back to the current `ml_dataset_manifest.json` when it is fresh enough

### Apr 15, 2026 - World transparency now stops drawing WMO transparent shell and WMO doodad transparency during the earlier opaque world stage

- followed the live viewer regression where render order was visibly broken across both WMOs and M2-family objects, not just the adapted-M2 skinning path
- the concrete shared ordering bug was in world-pass composition:
	- `WorldScene` still treated WMOs as part of the earlier opaque world stage
	- but `WmoRenderer.RenderWithTransform(...)` internally rendered its full stack there: opaque shell, opaque doodads, liquids, transparent doodads, and transparent shell
	- this meant WMO transparent layers were never participating in the later global world transparent stage, so they could overpaint or underpaint free-standing MDX transparency out of order
- active behavior now:
	- `WmoRenderer` has an explicit world-pass split via `WmoRenderPass` so world rendering can request `Opaque` or `Transparent` instead of always running the full internal stack
	- `WorldScene` now calls visible WMO renderers with `WmoRenderPass.Opaque` during the earlier opaque stage only
	- the later world transparent stage now builds one back-to-front combined transparent sort over visible WMOs plus visible transparent MDX instances and renders:
		- WMO transparent/liquid/doodad-transparent work via `WmoRenderPass.Transparent`
		- MDX transparent layers via `RenderPass.Transparent`
	- adapted M2 skeletal animation is also back to opt-in only in `ModelRenderer`; `PARP_M2_ENABLE_ANIMATION=1` is now required before adapted M2 skinning uploads bone matrices again
- focused validation completed:
	- `get_errors` returned clean for `Rendering/WmoRenderer.cs`, `Terrain/WorldScene.cs`, and `Rendering/ModelRenderer.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` succeeded with existing workspace warnings only
- important boundary:
	- full-solution `MdxViewer.sln` build still hit an unrelated `AlphaWdtAnalyzer.Core` deps-file failure on `DBCD.dll`; the targeted viewer project build is the relevant proof for this slice
	- no live post-fix viewer rerender or screenshot proof has been captured yet in this chat, so this is compile-validated render-path correction rather than runtime visual signoff

### Apr 15, 2026 - MCCV terrain tint now gates through alpha instead of darkening transparent regions

- followed the live viewer regression report that MCCV was still rendering wrong in the active terrain path after the earlier BGRA and mid-gray cleanup work
- the concrete mismatch was still in runtime semantics, not parser ownership:
	- `StandardTerrainAdapter` still passes raw `MCCV` bytes through unchanged from ADT payloads
	- `TerrainMeshBuilder` / `TerrainTileMeshBuilder` still decode those raw bytes as BGRA into the vertex attribute
	- but `TerrainRenderer` was still applying tint as `clamp(vVertexColor.rgb * 2.0, 0.0, 2.0)` while ignoring MCCV alpha entirely
- active behavior now:
	- `MdxViewer.Terrain.TerrainRenderer` treats RGB as the tint color around mid-gray and uses alpha as the tint-strength gate via `mix(vec3(1.0), tintColor, tintStrength)`
	- alpha values at or below mid-gray now stay neutral instead of letting transparent MCCV regions darken terrain toward black
	- `WoWMapConverter.Core.VLM.VlmMinimapCleanupService.RemoveMccvTint(...)` was updated to invert that same alpha-gated shader model so dataset cleanup remains parity-correct with the viewer
	- `TerrainChunkData` docs now explicitly call the stored chunk payload raw BGRA bytes instead of RGBA
- focused validation completed:
	- `dotnet test i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/WoWMapConverter.Core.Tests.csproj -c Debug --filter VlmMinimapCleanupServiceTests` passed (`6/6`), including new cases for mid-gray-alpha neutrality and transparent MCCV tint not darkening the minimap cleanup output
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing workspace warnings only
- important boundary:
	- no real-data viewer rerender or screenshot proof was captured in this slice yet
	- this closes the code-path mismatch that kept alpha-neutral MCCV regions wrong, but runtime visual signoff is still deferred until the active dataset harvest finishes
	- the intended post-harvest viewer check should target `3.0.1+` roots only, because older clients do not carry MCCV payloads to validate here
	- current sequencing from the user is: finish the active `3.3.5` harvest first, process `4.0.0` next, then do bounded `MdxViewer` MCCV fix-up validation after those loads complete

### Apr 15, 2026 - `wow-viewer` now repairs dataset normalmaps from `heightmap_local` when erased terrain detail survives there

- `wow-viewer/tools/converter/WowViewer.Tool.Converter/MlRepairNormalmapsCommand.cs` now adds `ml-repair-normalmaps`, a dataset-side repair command that synthesizes `_normal.png` outputs from exported heightmaps instead of leaving missing or flattened normals to trainer-side fallback only:
	- prefers `terrain_data.heightmap_local`
	- falls back to `terrain_data.heightmap_global` when local data is unavailable
	- updates tile JSON with `normalmap_generated_from` and `normalmap_generated_reason`
	- can rewrite existing normalmaps with `--rewrite-existing` or `--rewrite-when-local-differs <mae>` when local/global surfaces materially disagree
	- supports `--only-liquid-tiles`, `--limit`, `--report`, and `--dry-run` so bounded probes stay traceable
- rationale for the slice:
	- the active erased-terrain case is not primarily missing MCNR alone; WoWEdit can flatten or squash chunk geometry while exported `heightmap_local` still preserves the more useful local relief
	- this makes `heightmap_local` the better reconstruction source for “developer” terrain shapes that no longer survive in the live flattened surface or its prior normalmap
- bounded proof succeeded with real data:
	- dry-run probe: `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-repair-normalmaps --dataset-root i:/parp/parp-tools/datasets/4_0_0_11927/Kalimdor --limit 5 --dry-run --report i:/parp/parp-tools/output/build-validation/kalimdor_normalmap_repair_dryrun.json`
	- dry-run outcome:
		- no missing-normal repairs in the first five tiles because those references and files already existed
		- `Kalimdor_0_1` still surfaced as a rewrite candidate with `local_global_mean_absolute_delta = 7.084023842588067` and `local_global_max_absolute_delta = 178.98297119140625`
	- isolated rewrite proof: `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-repair-normalmaps --dataset-root i:/parp/parp-tools/output/build-validation/kalimdor_0_1_normalmap_repair_probe --limit 1 --rewrite-when-local-differs 1.0 --report i:/parp/parp-tools/output/build-validation/kalimdor_0_1_normalmap_repair_probe/repair_report.json`
	- rewrite outcome:
		- regenerated `images/Kalimdor_0_1_normal.png` from `heightmap_local`
		- recorded `normalmap_generated_from = heightmap_local`
		- recorded `normalmap_generated_reason = rewrite_local_global_divergence`
- important boundary:
	- this slice proves converter-side repair and reporting only
	- it does not yet audit a full corpus for all erased-terrain candidates or prove active viewer runtime behavior from the regenerated normalmaps

### Apr 15, 2026 - `train_v7.py` now auto-wires synthetic controls into dataset loading, curation, and validation

- `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` now restores the missing runtime seams that the current trainer file was referencing but not defining:
	- `TileSample`
	- `normalize_token(...)`
	- `parse_tile_identity(...)`
	- dataset index construction and cache write/read helpers for the current JSON surface
	- brush mask resolution
	- WDL prior rendering
	- dataset length and map-index rebuilding
- the same trainer now automatically ensures a synthetic control dataset unless `--no-synthetic-controls` is set:
	- default root: `output/build-validation/training_synthetic_controls`
	- auto-generation command: `wow-viewer` `ml-generate-controls`
	- auto-harvest follow-up: `wow-viewer` `ml-harvest-brushes`
	- optional controls for the workflow surface:
		- `--synthetic-control-root`
		- `--regenerate-synthetic-controls`
- trainer-side admission rules now keep synthetic controls usable instead of discarding them as blank tiles:
	- missing synthetic metadata no longer stringifies to fake truthy values
	- synthetic controls are allowed through the low-height-range rejection path
	- synthetic tiles without a real exported normal map now receive a flat fallback normal prior instead of being rejected outright
	- curated training keeps synthetic controls instead of letting low-complexity sampling drop them
	- pinned validation now keeps `synthetic_controls_0_0` (`white_plate`) in the validation split as the stable non-interesting baseline control
- bounded proof succeeded with the live trainer code and real + synthetic roots:
	- import/parser smoke: `i:/parp/parp-tools/.venv/Scripts/python.exe gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py --help`
	- bounded loader proof used:
		- real probe root `output/build-validation/original_development_11927_overlay_probe`
		- auto-generated synthetic root `output/build-validation/training_synthetic_controls`
	- proof outcome:
		- trainer auto-generated and auto-harvested the synthetic root on first use
		- bounded dataset load produced `2` samples total with `1` real tile and `1` synthetic tile
		- `white_plate` (`synthetic_controls_0_0`) survived indexing and remained tagged as synthetic while the real tile stayed non-synthetic
		- split proof showed validation tiles = `synthetic_controls_0_0`, confirming the pinned baseline control path
- important boundary:
	- this slice wires deterministic synthetic controls into the existing trainer surface only
	- it does not yet add the next requested harvested-data hybrid control family inside `MlSyntheticControlGenerator`

### Apr 14, 2026 - `wow-viewer` now has deterministic synthetic control tiles, including a guaranteed blank `white_plate`

- `wow-viewer/tools/converter/WowViewer.Tool.Converter/MlSyntheticControlGenerator.cs` now adds `ml-generate-controls`, which writes dataset-shaped synthetic tiles under a target root instead of loose demo images only:
	- tile JSON in `dataset/`
	- source minimaps in `images/`
	- local/global heightmaps in `images/`
	- packed alpha atlas plus per-layer alpha masks and shadow under `stitched/`
	- `metadata.jsonl`, `dataset_info.json`, and `synthetic_control_manifest.json`
- current default synthetic control set includes:
	- `white_plate` as the explicit non-interesting flat control tile
	- `diagonal_ramp`
	- `ring_mound`
	- `terrace_steps`
- each synthetic tile now carries explicit control metadata in JSON, including:
	- `expected_interest_class`
	- `expected_brush_groups`
	- `expected_layer_stack_depth`
- bounded synthetic proof succeeded with:
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter -- ml-generate-controls --dataset-root i:/parp/parp-tools/output/build-validation/synthetic_controls_probe`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter -- ml-audit-signals --dataset-root i:/parp/parp-tools/output/build-validation/synthetic_controls_probe --output i:/parp/parp-tools/output/build-validation/synthetic_controls_probe/signal_audit.json`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter -- ml-harvest-brushes --dataset-root i:/parp/parp-tools/output/build-validation/synthetic_controls_probe --output-dir i:/parp/parp-tools/output/build-validation/synthetic_controls_probe/brush_imprints`
- proof outcome:
	- `white_plate` was generated with a base-only `chunk_layers` contract, zero alpha structure, and a packed alpha atlas path still present for contract stability
	- downstream brush harvest on the same synthetic root reported `patch_candidates: 0` and `groups_written: 0` for `synthetic_controls_0_0`, while patterned controls remained harvestable
- important boundary:
	- this first slice generates deterministic synthetic controls only; it does not yet synthesize hybrid controls by compositing harvested real tile payloads into fake tiles, which is still the next higher-value control-family follow-up

### Apr 14, 2026 - `ml-harvest-brushes` now emits stitched brush layers and first-pass fractal visuals in `wow-viewer`

- `wow-viewer/tools/converter/WowViewer.Tool.Converter/MlBrushImprintHarvester.cs` now writes additional deterministic brush-analysis outputs when harvesting succeeds:
	- tile-level `tile_masks/*_fractal_detail.png`
	- tile-level `tile_masks/*_fractal_candidate_mask.png`
	- tile-level `tile_masks/*_layer_stack_depth.png`
	- tile-level `tile_masks/*_fractal_stack_proxy.png`
	- tile-level `tile_masks/*_fractal_stack_candidate_mask.png`
	- stitched full-map layers under `brush_imprints/stitched/` when multiple tile masks exist, currently:
		- `<map>_full_brush_mask.png`
		- `<map>_full_fractal_detail.png`
		- `<map>_full_fractal_candidate_mask.png`
		- `<map>_full_layer_stack_depth.png`
		- `<map>_full_fractal_stack_proxy.png`
		- `<map>_full_fractal_stack_candidate_mask.png`
- tile summaries in `brush_imprint_manifest.json` now carry:
	- `fractal_detail_path`
	- `fractal_candidate_mask_path`
	- `layer_stack_depth_path`
	- `fractal_stack_proxy_path`
	- `fractal_stack_candidate_mask_path`
	- `fractal_mean_score`
	- `fractal_max_score`
	- `fractal_stack_mean_score`
	- `fractal_stack_max_score`
	- `layer_stack_max_depth`
- group JSONs now also carry per-group `fractal_detail_score` and `fractal_candidate` using the same multiscale residual family already used downstream in the prefab-library Python tooling
- bounded real-data proof succeeded with:
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug -- ml-harvest-brushes --dataset-root i:/parp/parp-tools/datasets/original_development/development --output-dir i:/parp/parp-tools/output/build-validation/brush-imprints/original_development_fractal_stitch_probe_20260414 --limit 6 --write-previews`
	- output wrote stitched files:
		- `stitched/development_full_brush_mask.png`
		- `stitched/development_full_fractal_detail.png`
		- `stitched/development_full_fractal_candidate_mask.png`
		- `stitched/development_full_layer_stack_depth.png`
		- `stitched/development_full_fractal_stack_proxy.png`
		- `stitched/development_full_fractal_stack_candidate_mask.png`
	- probe manifest recorded new tile-level fractal paths and scores for all six processed tiles, while group JSONs on real brush-bearing tiles such as `development_0_0_g0001.json` recorded nonzero `fractal_detail_score` values above the current candidate threshold
- current limitation:
	- the new stacked outputs are still a chunk-layer-count proxy fused with the existing multiscale height residual, not true layered-alpha reconstruction, because the inspected harvested corpora still show `alpha_bits: null`, `alpha_path: null`, `alpha_masks: []`, and `alpha_atlas: null`
- important boundary:
	- there is still no first-party automated test project covering `WowViewer.Tool.Converter`; this slice is currently proven by `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` plus the bounded real-data harvest above

### Apr 14, 2026 - `original_development` is now applied to a staged 4.0.0.11927 base client

- added a reusable overlay-staging helper at `gillijimproject_refactor/scripts/stage_original_development_overlay.ps1`
	- default base client: `H:\CLIENTS\World of Warcraft Cata beta 11927`
	- default staged root: `i:/parp/parp-tools/output/tmp/original_development_client_4_0_0_11927`
	- staged root contains a linked `Data` surface from the base client plus linked loose `World/Maps/development` and `World/Textures/Minimap` from the checked-in development data
- updated `scripts/ml_corpus_fixed_clients.json` so the `original_development` client now resolves to that staged root instead of the sparse raw `test_data/original_development` path
- bounded real-data export proof succeeded on the staged overlay root:
	- command used `ml-export --client i:/parp/parp-tools/output/tmp/original_development_client_4_0_0_11927 --map development --tile 31_36`
	- output root: `output/build-validation/original_development_11927_overlay_probe`
	- exporter found the loose development WDT/WDL and split ADT payloads from the overlay, resolved the minimap through the 11927-backed data surface, and wrote stitched/semantic outputs for the bounded tile
- important boundary:
	- the proof used a staged composite local root, not a later Cataclysm base client; if a real `4.0.1.12304` client becomes available, rebuild this overlay against that base and treat it as the preferred development-map host

### Apr 14, 2026 - Recovered the missing 4.0.0.11927 world roots with a fast core-export path

- validated the user's corpus-completeness concern on the fixed local Cataclysm beta client `H:\CLIENTS\World of Warcraft Cata beta 11927`:
	- `datasets/4_0_0_11927` initially contained only `Azeroth`, `EmeraldDream`, and `LostIsles`
	- `Azeroth/ml_dataset_manifest.json` initially showed only `1` processed tile
	- client discovery proved `Kalimdor` and other missing roots were present in the client, so the dataset state was genuinely incomplete
- fixed the wrapper/runtime seams that were preserving partial exports:
	- `scripts/export_ml_corpus.ps1` no longer uses the wrong client path during dry-run staged discovery, no longer applies the accidental second discovery override, and no longer skips maps just because `dataset/` already exists
	- `-Force` now clears an existing per-map output root before rerun
	- `WoWMapConverter.Cli` / `WoWMapConverter.Core.VLM.VlmDatasetExporter` now support `--skip-derived-assets` so missing world roots can be recovered without waiting for tilesets, stitched outputs, or semantic postprocess assets
- bounded real-data recovery outcomes under `datasets/4_0_0_11927`:
	- `Azeroth`: recovered to `839` tiles; a follow-up fast rerun backfilled the previously missing `839` global heightmaps after the first long run had been interrupted before that phase
	- `Kalimdor`: exported and harvested to `1011` tiles
	- `Deepholm`: exported and harvested to `100` tiles (export resolved archive-backed map directory `Deephome` automatically)
	- existing roots still present with manifests: `EmeraldDream` `256` tiles, `LostIsles` `149` tiles
- honest boundary after recovery:
	- the targeted 4.0.0.11927 world roots now exist with manifests and core heightmap coverage, but per-channel density is still not uniform everywhere; manifest coverage still shows partial source-minimap presence on some maps (`Azeroth 835/839`, `Kalimdor 1006/1011`, `EmeraldDream 91/256`), so future sessions should call this root-level recovery complete without claiming every harvested channel is uniformly complete

### Apr 14, 2026 - Terrain-only rebake now has bounded real-data export proof and semantic-raster audit proof

- completed the missing real-data validation step for the new chunk-rebaked `terrain_only_minimap` path:
	- ran `ml-export` on fixed local `H:\CLIENTS\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft`
	- bounded target was `EmeraldDream --tile 26_26`
	- output root: `output/build-validation/emeralddream_26_26_terrain_rebake_probe`
- real-data export outcome:
	- exporter loaded `4` real tileset textures from the client archives and wrote a fresh `terrain_only_minimap`
	- fresh tile JSON also now carries `holes_mask`, `area_id_map`, `chunk_flags_map`, `liquid_type_map`, and `dominant_effect_id_map`
	- the chosen tile already had a legacy corpus baseline with object masking, so the new output could be compared directly against the older `datasets/3_0_1_8303/EmeraldDream` export
- bounded image-analysis proof on the masked region:
	- saved cropped comparison artifacts under `output/build-validation/emeralddream_26_26_terrain_rebake_probe/analysis`
	- the object-mask footprint covered `861` pixels
	- `782` of those masked pixels changed by more than `8` intensity levels between legacy and rebaked `terrain_only_minimap`
	- masked-region RGB MAE between old and new terrain-only outputs was about `19.49`
	- visual crop comparison showed the rebaked output replacing the old smoother green fill with a darker terrain patch aligned to the surrounding ground/road family inside the masked footprint
- downstream audit proof:
	- `src/WoWMapConverter/scripts/audit_v7_signals.py --dataset-root output/build-validation/emeralddream_26_26_terrain_rebake_probe --image-sample-limit 1` reported `1/1` coverage for `terrain_only_minimap`, `holes_mask`, `area_id_map`, `chunk_flags_map`, `liquid_type_map`, and `dominant_effect_map`
- bounded real-data brush-archetype proof also completed:
	- ran `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-harvest-brushes --dataset-root i:/parp/parp-tools/datasets/original_development/development --output-dir i:/parp/parp-tools/output/build-validation/brush-imprints/original_development_archetype_probe_20260414 --limit 6 --write-previews`
	- processed `6` tiles with `0` missing-heightmap skips
	- wrote `117` group files covering `15029` patches and emitted `brush_archetype_manifest.json` with `115` deterministic archetypes
	- representative group JSONs now visibly carry `archetype_id`, `archetype_key`, `archetype_label`, and `shape_fingerprint`
- important boundary:
	- the checked-in `test_data/original_development` root is still useful for split-map exporter health, but not for proving texture rebake quality by itself because it lacks the BLP payloads; the same bounded probe there reached export but logged `Exported 0 textures`

### Apr 14, 2026 - Added a separate V7.6 doc set and a structured predicted-output dataset spec

- documented the checked-in V7.6 branch as its own model line at `gillijimproject_refactor/docs/v76-model-architecture-guide.md`
	- explains that V7.6 is a paired-output image-to-height+albedo branch, not the active V7.5.1 harvested-corpus terrain model
	- explains what the input image, target height, and synthesized target albedo are meant to teach the shared encoder and the two decoder heads
	- explains that the branch is meant to turn arbitrary image input into a structured predicted terrain dataset rather than a loose file dump
- added `gillijimproject_refactor/docs/v76-output-dataset-spec.md`
	- defines a structured output package with per-sample JSON, run-level manifest, `metadata.jsonl`, `dataset_info.json`, source-image copies, predicted height/albedo assets, optional mesh exports, and optional stitched quilt outputs
	- makes the provenance rule explicit so predicted outputs cannot be mistaken for harvested truth
- updated the surrounding docs to route readers correctly and keep the stories separated:
	- `README.md`
	- `docs/ML_DATASET_GROUNDING.md`
	- `docs/VLM_DATASET_EXPORTER.md`
	- `docs/VLM_Training_Guide.md`
	- `docs/V7_HEIGHT_REGRESSOR.md`
	- `docs/v75-model-architecture-guide.md`
- additional docs cleanup landed in the same pass:
	- `docs/VLM_DATASET_EXPORTER.md` no longer presents legacy per-tile `.bin` payloads as part of the canonical active export surface
- important boundary:
	- this is documentation and spec work only
	- the checked-in V7.6 inference scripts still write loose outputs today; the new spec is the intended replacement contract, not proof that the scripts already implement it

### Apr 14, 2026 - Dataset-grounding docs now explain the real harvest pipeline and defer prefab from the trusted supervision story

- added a dedicated provenance doc at `gillijimproject_refactor/docs/ML_DATASET_GROUNDING.md`
	- explains the real client roots and checked-in development root that seed `datasets/`
	- explains the staged harvest flow through `export_ml_corpus.ps1`, `ml-list-maps`, `ml-export`, `ml-harvest`, `ml-harvest-brushes`, and `ml-audit-signals`
	- enumerates the active V7.5.1 channels and states which ones are raw harvested assets versus deterministic derived channels
	- explicitly states that GAN is a training-time refinement objective, not the source of the dataset
- updated entry-point docs so readers can actually find that story:
	- `gillijimproject_refactor/README.md`
	- `gillijimproject_refactor/docs/VLM_DATASET_EXPORTER.md`
	- `gillijimproject_refactor/docs/VLM_Training_Guide.md`
	- `gillijimproject_refactor/docs/v75-model-architecture-guide.md`
- active policy correction captured in docs:
	- brush harvesting stays the trusted active auxiliary channel for the terrain model
	- prefab tooling remains available, but it is now documented as deferred or experimental and should not be presented as part of the current grounded supervision contract
- important boundary:
	- this is documentation and continuity correction only
	- it does not newly validate prefab outputs or change the underlying exporter or trainer behavior

### Apr 14, 2026 - Corpus export now stages archive-backed clients locally before heavy reads in both PowerShell and direct CLI paths

- implemented the first real archive-staging cutover instead of just documenting it:
	- added reusable helpers in `gillijimproject_refactor/scripts/wowarchive_client_staging.ps1`
	- added standalone helper `gillijimproject_refactor/scripts/stage_wowarchive_client.ps1` for staging one mounted client and pruning stale staged copies
	- updated `gillijimproject_refactor/scripts/export_ml_corpus.ps1` so archive-backed runs prefer staged local working roots before `ml-list-maps` or `ml-export`
	- updated direct `WoWMapConverter.Cli ml-corpus` so it resolves `local_client_path` versus `archive_client_path`, stages mounted roots, supports `all_maps`, and can prune stale staged copies itself
	- updated `gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json` with explicit `local_client_path` or `archive_client_path` entries plus `mount_root`, `mount_script`, `staging_root`, and `prune_staged_clients` defaults for the new workflow
- current resolution behavior in the PowerShell corpus runner:
	- prefer `local_client_path` or an already-local direct `client_path` when available
	- otherwise stage `archive_client_path` or mounted direct paths under the configured mount root into the configured staging root
	- optionally resolve explicit minimap roots through the same local-vs-archive policy
	- prune stale staged copies after the run while keeping the clients touched by the active job
- current resolution behavior in the direct CLI now matches that shape closely enough for the live config:
	- `ml-corpus` can resolve the same top-level mount and staging fields directly from JSON or command-line overrides
	- `all_maps` now works inside direct `ml-corpus` instead of only in the PowerShell wrapper
	- dry-run keeps using the mounted source for discovery when no staged copy exists yet, while non-dry runs copy first and then discover against the staged root
- validation completed in this chat:
	- script diagnostics reported no errors for the new helpers or the updated corpus runner
	- synthetic mounted-client smoke proved stage plus prune behavior end to end
	- synthetic `export_ml_corpus.ps1 -DryRun` proved the real corpus runner resolves an archive-backed config entry to the staged working root before invoking `ml-export`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` succeeded
	- direct CLI real-data dry-run against the mounted `3.0.1.8303` client printed the staged working root plus the mounted source path for `Azeroth`
	- direct CLI synthetic `--harvest-only` validation with `all_maps: true` discovered `SynthMap`, skipped harvest due missing dataset JSON as expected, and removed a stale staged client directory
- important boundary:
	- the live WoWArchive surface still appears to be `0.X-3.X`; keep `4_0_0_11927` local-only until a real archive-backed 4.x root is verified

### Apr 14, 2026 - WoWArchive client staging is now a first-class workflow rule

- added a dedicated WoWArchive staging workflow surface so future chats stop treating the mounted archive as the default heavy-read root:
	- new skills at `.github/skills/wowarchive-client-staging/SKILL.md` and `.codex/skills/wowarchive-client-staging/SKILL.md`
	- `.github/copilot-instructions.md` and `AGENTS.md` now name that skill, document the `MountAll.bat` workflow, and explicitly mention the `Explore` subagent as an available read-only discovery helper
	- shared-I/O and migration-continuation skills now point to staged local copies for broad archive-backed validation instead of direct mounted reads
	- `gillijimproject_refactor/memory-bank/data-paths.md`, `wow-viewer/README.md`, and the shared-I/O continuity plan now carry the same archive-source plus local-staging rule
- important boundary:
	- this is workflow and continuity enforcement only
	- no automated client-staging command or shared library implementation has been added yet

### Apr 14, 2026 - Workspace routing now treats dataset-builder ownership as canonical wow-viewer work

- updated the workspace workflow surface so future chats stop defaulting dataset-builder architecture into the legacy exporter path:
	- `.github/copilot-instructions.md` and `AGENTS.md` now route dataset-builder cutover, ML corpus export ownership, and terrain-supervision artifact generation into `wow-viewer`
	- new planning prompts were added at `.github/prompts/wow-viewer-dataset-builder-plan.prompt.md` and `.codex/prompts/wow-viewer-dataset-builder-plan.md`
	- migration-continuation skills and tool-suite plan-set prompts now route dataset-builder requests to that new prompt
	- `wow-viewer/README.md`, `plans/wow_viewer_shared_io_library_plan_2026-03-26.md`, and the new continuity file `plans/wow_viewer_dataset_builder_tool_plan_2026-04-14.md` now capture the same ownership rule
- the same workflow surface now also carries the next-level policy constraints:
	- target surfaces are shared library plus CLI plus viewer/editor plus dataset explorer plus supervised training tooling
	- the toolchain stays Bring Your Own Data and should not plan around distributing copyrighted corpora, model weights, or model outputs
	- long-range orchestration should keep backend seams open beyond CUDA-only assumptions
- important boundary:
	- this is workflow and continuity enforcement only
	- no shared dataset contract or `wow-viewer` dataset-builder tool implementation has been migrated yet

### Apr 14, 2026 - Trainer now hard-filters liquid-obscured junk tiles and known malformed EmeraldDream minimaps before train/val split

- added dataset-side curation in `src/WoWMapConverter/scripts/train_v7.py` so bad tiles are rejected during sample indexing instead of merely showing up later in previews:
	- liquid-obscured rejection is now on by default when `liquid_coverage >= 0.98` and combined minimap+normal signal is effectively absent (`combined variance <= 0.0010`, `combined gradient <= 0.0050`)
	- known malformed `EmeraldDream` minimaps are now rejected by default when they match the low-signal Blizzard-tooling corruption pattern (`variance <= 0.0022` and either `gradient <= 0.0040` or `extreme_fraction >= 0.90`)
	- both filters are exposed as CLI knobs so future runs can tighten or relax them without another code edit
- fixed a second curation bug in the same trainer:
	- complexity curation had been force-including every tile with any liquid mask at all
	- it now only auto-keeps liquid-bearing tiles when `liquid_coverage <= 0.85`, so water-heavy junk no longer dominates the curated train set
- recovered-corpus zero-epoch preflight after the change:
	- usable sample count dropped from the earlier `2358` to `2161`
	- per-root hard rejections included `111` liquid-obscured tiles on `3_0_1_8303/Northrend`, `21` on `0_5_3_3368/Azeroth`, `17` on `3_3_5_12340/Azeroth`, `5` on `original_development/development`, and `4` on `4_0_0_11927/LostIsles`
	- malformed `EmeraldDream` rejections were `19` on `3_0_1_8303/EmeraldDream`, `19` on `4_0_0_11927/EmeraldDream`, and `1` on `3_3_5_12340/EmeraldDream`
	- curated train count dropped from `1302` to `1247` after the liquid auto-inclusion fix, while brush-bearing curated tiles remained strong at `797 / 1247`
- important boundary:
	- this is dataset-culling proof and full-corpus preflight proof, not long-run model-quality proof yet
	- the next real CUDA run should use this curated loader path instead of the earlier pre-cull `2358`-sample state

### Apr 14, 2026 - V7 trainer now supports subset-manifest tile allowlists; interesting-tile smoke train passed end-to-end

- added exact subset training support in `src/WoWMapConverter/scripts/train_v7.py`:
	- new CLI option `--tile-manifest <path>` accepts the focused subset artifact (`interesting_tile_subset_manifest.json`)
	- loader now builds a root-scoped tile allowlist from `selected_tiles[*].dataset_map_root + tile_name`
	- dataset indexing keeps all existing checks (missing refs/files, map filters, blank skip), then applies tile allowlist filtering
	- checkpoint/training metadata now records `tile_manifest`
- real smoke proof completed with the focused subset roots (`0_5_3_3368/Azeroth`, `3_3_5_12340/EmeraldDream`, `4_0_0_11927/Azeroth`):
	- command used `--profile manual --tile-manifest ...interesting_tile_subset_manifest.json --epochs 1 --batch-size 2 --adversarial-scale 0`
	- loaded `46` samples (`48` selected tiles minus `2` skipped by default `--min-height-range 0.5` blank-tile guard)
	- split: train/val `41 / 5`
	- epoch 1 completed and wrote `best.pt`, `checkpoint.pt`, previews, and `training_log.json` under `output/ml-training/interesting_subset_smoke`
- important boundary:
	- this is confirmed trainability on the focused subset path, not a long-run quality signoff
	- run used CPU in this chat environment and disabled GAN for fast readiness proof

### Apr 13, 2026 - Focused Azeroth/EmeraldDream interesting-tile subset completed (48/48)

- user-directed pivot executed away from broad all-map corpus expansion to a focused subset harvest:
	- broad `export_ml_corpus.ps1 -Force` run was stopped once enough base output existed
	- interesting tile IDs were derived from historical validation previews under `output/ml-training/**/previews/val_epoch_*.json` for only `Azeroth` and `EmeraldDream`
- new helper script landed:
	- `scripts/harvest_interesting_tile_subset.py`
	- builds `interesting_tile_subset_manifest.json` and `interesting_tile_subset_missing_plan.json` from current `datasets/*/*/ml_dataset_manifest.json`
- targeted fill completed for missing tiles only:
	- `3_3_5_12340/EmeraldDream`: exported and harvested 11 specific interesting tiles via `ml-export --tile`
	- `4_0_0_11927/Azeroth`: exported and harvested `Azeroth_25_43` (missing in 3.3.5)
- final subset status:
	- `interesting_tile_count = 48`
	- `harvested_tile_rows = 48`
	- `missing_tile_count = 0`
	- composition: `Azeroth=37`, `EmeraldDream=11`; clients used `0_5_3_3368=36`, `3_3_5_12340=11`, `4_0_0_11927=1`
- output artifacts:
	- `output/build-validation/ml-audit/interesting_tile_subset_manifest.json`
	- `output/build-validation/ml-audit/interesting_tile_subset_missing_plan.json`

### Apr 14, 2026 - Full all-map corpus rerun is now wired and actively running from a clean datasets root

- landed workflow changes to support user-requested full-client map coverage rather than fixed map lists:
	- `WoWMapConverter.Cli` now exposes `ml-list-maps` (alias `vlm-list-maps`) for dynamic per-client map discovery
	- map discovery now recognizes both `.wdt` and legacy disk `.wdt.MPQ` layouts (critical for the `0_5_5_3494` client root)
	- `scripts/export_ml_corpus.ps1` now supports per-client `all_maps: true`, discovers map lists at runtime, and records per-map failures instead of silently passing
	- `scripts/ml_corpus_fixed_clients.json` now includes `0_5_3_3368`, `0_5_5_3494`, and `0_6_0_3592`, with `all_maps: true` enabled for all real client builds
- dataset reset and rerun status:
	- `i:/parp/parp-tools/datasets` was wiped and recreated before launch
	- full rerun launched with `scripts/export_ml_corpus.ps1 -Force` and is currently running through discovered maps (early status observed: `original_development/development` completed and next client jobs started)
- post-processing tooling added and smoke-validated:
	- new `scripts/build_minimal_ml_manifest.py` builds a deduplicated tile manifest plus a map-level minimal export plan from harvested manifests
	- partial-run smoke against live output succeeded (`1` manifest scanned, `352` tiles, `299` unique groups)
- boundary:
	- full all-client/all-map rerun is long-running and not yet complete in this chat; final dedupe/minimal artifacts must be regenerated after export+harvest completes for every map

### Apr 13, 2026 - Next chat should resume at the full corpus rerun and model-training stage

- the current user priority is no longer another incremental exporter probe; it is the end-to-end dataset and model workflow:
	- re-extract the configured corpus across the fixed client roots already captured in `scripts/ml_corpus_fixed_clients.json` and `datasets/`
	- re-harvest and re-audit the refreshed outputs
	- launch the actual training pass on the refreshed corpus and evaluate the resulting run artifacts
- the next continuation should treat the following as already-landed prerequisites that need to be folded into that rerun rather than re-litigated first:
	- MCSH shadow exclusion from `terrain_only_minimap`
	- MCCV cleanup parity
	- geometry-derived object masks
	- loose override precedence
	- `Deepholm` -> `Deephome` alias recovery
- known unresolved seam to keep in scope without derailing the rerun plan:
	- the Cataclysm `LostIsles_23_24` liquid-loss issue still needs a live MH2O parser fix, but that should only interrupt the broader rerun when it materially blocks the refreshed corpus pass

### Apr 13, 2026 - `terrain_only_minimap` no longer over-masks shadow-only tiles

- fixed the active V7.5 cleanup rule in `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs`:
	- `terrain_only_minimap` previously unioned stitched `shadowPath` with alpha/object/PM4/liquid masks
	- it now excludes stitched MCSH shadow output and only removes stitched alpha plus object, PM4, and liquid contamination
- focused regression coverage added:
	- `WoWMapConverter.Core.Tests/VLM/VlmDatasetExporterTests.cs` now asserts the terrain-only mask path selection ignores a provided shadow artifact path
- validation completed:
	- `dotnet test i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/WoWMapConverter.Core.Tests.csproj -c Debug --filter VlmDatasetExporterTests` passed (`12/12`)
	- bounded real-data proof on `H:\CLIENTS\3.X_Pre-Release_Windows_enUS_3.0.1.8303\World of Warcraft` for `EmeraldDream --tile 24_25` showed the prior bad state in the existing corpus (`shadow_maps` present, no alpha/liquid/object/PM4 masks, yet `terrain_only_minimap` existed) and the corrected state after re-export under `output/build-validation/emeralddream_tile_24_25_shadow_rule`, where `terrain_only_minimap` is now `null`
- important boundary:
	- shadow diagnostics are still exported; only the terrain-only cleanup union changed

### Apr 13, 2026 - Loose override support now reaches shared `md5translate` loading and exporter virtual asset reads

- landed the loose-file override slice across both shared I/O and the active exporter:
	- `wow-viewer/src/core/WowViewer.Core.IO/Files/Md5TranslateResolver.cs` now loads loose disk `md5translate` candidates before archive-backed candidates, including map-specific extra candidates such as `World/Maps/<Map>/md5translate.trs`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` now routes mapped minimaps, tileset BLP conversion, model bounds reads, model-footprint reads, split-WMO group reads, and LK tile scoring through a shared loose-first virtual asset helper before falling back to archive reads
- regression coverage added:
	- `wow-viewer/tests/WowViewer.Core.Tests/Md5TranslateResolverTests.cs` now proves a loose map-specific `md5translate.trs` overrides the archive-backed copy
	- `WoWMapConverter.Core.Tests/VLM/VlmDatasetExporterTests.cs` now proves loose virtual asset bytes win over an archive-backed file with the same virtual path
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter Md5TranslateResolverTests` passed (`3/3`)
	- `dotnet test i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/WoWMapConverter.Core.Tests.csproj -c Debug --filter VlmDatasetExporterTests` passed (`6/6`)
	- bounded real-data baseline on `H:\CLIENTS\World of Warcraft Cata beta 11927` exported `LostIsles_29_32` from archive-backed minimap path `textures/minimap/807183b22bf2ba9e1f0305a2d345c015.blp`
	- bounded overlay proof on `output/tmp/cata_loose_override_overlay_20260413` reused the same real client archives through a `Data` junction, added a loose `World/Maps/LostIsles/md5translate.trs` override plus `textures/minimap/override.png`, and the resulting exported tile center pixel changed from the archive-backed baseline `79,142,255,255` to the loose override `255,0,255,255`
- separate documentation follow-up completed in the active terrain guide:
	- `docs/v75-model-architecture-guide.md` now explicitly keeps `0.5.3`, `0.5.5`, and `0.6.0` as default early-build corpus anchors for V7.5 planning
- important boundary:
	- this closes the override precedence seam in code, focused tests, and a temporary overlay proof on top of the real 11927 archives
	- it still does not prove that the untouched stock 11927 client root already ships natural loose minimap overrides; the initial filesystem scan did not find any loose `md5translate` or minimap trees there

### Apr 13, 2026 - Broader cleaned-input retrain rerun completed and established the current non-adversarial baseline

- completed the broader rerun under `output/ml-training/v7_5_1_cleaned_inputs_20260413_rerun` after refreshing and re-auditing current dataset roots
- run outcome from `training_log.json`:
	- `12` epochs completed before stopping
	- best validation loss about `0.06906226947903633` at epoch `7`
	- final epoch train/val about `0.06429274735920545 / 0.07939258947968483`
	- metadata recorded `921` total loaded samples, `573` train, `98` val, `66` train groups, and `8` val groups
- dataset-root set recorded in metadata included the current cross-version cleaned-input corpus:
	- `0_7_0_3694/EmeraldDream`
	- `3_0_1_8303/EmeraldDream`, `Northrend`, `PVPZone01`, `PVPZone02`, `PVPZone03`, `PVPZone04`
	- `3_3_5_12340/Azeroth`, `3_3_5_12340/EmeraldDream`
	- `4_0_0_11927/Azeroth`, `Deepholm`, `EmeraldDream`, `LostIsles`
	- `original_development/development`
- paired audit output lives at `output/build-validation/ml-audit/v7_5_1_dataset_signal_audit_20260413_rerun.txt`
	- key refreshed roots now show `terrain_only_minimap` coverage of `198/352` for `original_development/development`, `25/100` for `Deepholm`, and `77/149` for `LostIsles`
	- the same audit reports nonzero object-mask files on `49/352`, `39/100`, and `12/149` of those roots respectively
- important boundary:
	- this rerun used cleaned inputs, but it did not exercise the intended GAN-on V7.5.1 schedule
	- saved metadata still shows `start_gan_epoch = 101`, `gan_enabled = false` for all epochs, and `gan_burst_after_best = 0`
	- treat it as the current cleaned-input non-adversarial baseline, not as final closure on the intended V7.5.1 adversarial training cadence

### Apr 13, 2026 - Exporter object-mask geometry path is live, refreshed roots were regenerated, and the new corpora pass a mixed-root training smoke

- landed the exporter-side object-mask overhaul in `WoWMapConverter.Core/VLM/VlmDatasetExporter.cs`:
	- `VlmObjectPlacement` now persists `model_path`
	- exporter masks now prefer projected footprint hulls from real `M2`, `MDX`, and `WMO` geometry instead of the earlier shadow-rectangle plus circle heuristic
	- fallback remains bounds-polygon or ellipse based when geometry is unavailable
- refreshed real-data dataset roots after the exporter change:
	- `dotnet run -- ... ml-export --client i:/parp/parp-tools/gillijimproject_refactor/test_data/original_development --minimap-root i:/parp/parp-tools/gillijimproject_refactor/test_data/development --map development --out i:/parp/parp-tools/datasets/original_development/development`
		- exported `352` tiles, skipped `0`
		- `ml-harvest` rebuilt the manifest with `352` processed tiles
	- `dotnet run -- ... ml-export --client H:/CLIENTS/World of Warcraft Cata beta 11927 --map Deepholm --out i:/parp/parp-tools/datasets/4_0_0_11927/Deepholm`
		- exported `100` tiles, skipped `0`
		- `ml-harvest` rebuilt the manifest with `100` processed tiles
	- `dotnet run -- ... ml-export --client H:/CLIENTS/World of Warcraft Cata beta 11927 --map LostIsles --out i:/parp/parp-tools/datasets/4_0_0_11927/LostIsles`
		- exported `149` tiles, skipped `0`
		- `ml-harvest` rebuilt the manifest with `149` processed tiles
- mixed-root training smoke on refreshed corpora completed:
	- dataset roots: refreshed `original_development/development`, `4_0_0_11927/Deepholm`, and `4_0_0_11927/LostIsles`
	- `466` valid samples loaded after blank-tile filtering
	- train/val split `418 / 48`
	- pinned validation refs still included `development:development_0_0`
	- epoch `1` finished with train `0.2071`, val `0.1754`, discriminator `0.4504`, GAN phase `steady`, and a saved best model under `output/tmp/v7_5_1_geometry_mask_refresh_smoke_20260413`
- exporter-output audit after refresh:
	- `original_development/development` object masks: `38` mask PNGs, average coverage about `3.58%`, worst tile about `17.09%`
	- `4_0_0_11927/Deepholm` object masks: `25` mask PNGs, average coverage about `33.49%`, worst tile about `88.69%`
	- `4_0_0_11927/LostIsles` object masks: `11` mask PNGs, average coverage about `20.72%`, worst tile about `71.17%`
- important boundary:
	- this is real refreshed-corpus proof plus trainability proof
	- future geometry-derived object-mask proof should pair at least one real `3_3_5_12340` root with one real `4_0_0_11927` root instead of treating the current 4.x runs as sufficient alone
	- it is not final exporter signoff for Cataclysm roots because some refreshed 4.x object masks are still pathologically large and are currently being kept in check mainly by trainer-side coverage rejection

### Apr 13, 2026 - MCCV parity fix now matches MdxViewer export and render behavior

- fixed the remaining MCCV correctness issue in the active VLM path by following `MdxViewer` exactly instead of only swapping channels:
	- `mccv_map` generation now matches `MdxViewer.Export.TerrainMccvIo.BuildTileImage(...)` and preserves raw MCCV bytes in PNG channel order
	- `no_mccv_minimap` cleanup now inverts the actual terrain shader tint model from `MdxViewer.Terrain.TerrainRenderer` by dividing out `clamp(vertexColor.rgb * 2.0, 0.0, 2.0)` after decoding the raw-view PNG back to BGRA tint
- focused validation completed:
	- `dotnet test i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/WoWMapConverter.Core.Tests.csproj -c Debug --filter VlmMinimapCleanupServiceTests` passed (`4/4`)
	- bounded real-data export probe on `H:\CLIENTS\World of Warcraft Cata beta 11927` with `--map Deepholm --limit 1` succeeded and regenerated corrected MCCV artifacts in `output/tmp/deepholm_mccv_inverse_probe_20260413`
- important boundary:
	- this is code/test/build proof plus bounded real-data artifact proof
	- existing harvested dataset roots still need re-export before training can fully benefit from the corrected MCCV cleanup

### Apr 13, 2026 - Recovered the failing 11927 `Deepholm` export path and made corpus harvest non-fatal on empty exports

- fixed the concrete corpus blocker reported by the user:
	- `ml-export --map Deepholm` under the 4.0.0.11927 client had been failing with `WDT not found`
	- listfile/archive inspection showed the internal directory is actually `Deephome`
- active exporter behavior now:
	- `VlmDatasetExporter` still prefers `Map.dbc` directory resolution first
	- when that does not resolve a requested map label, it now scans archive-known `World/Maps/*/*.wdt` entries and recovers exact normalized or small edit-distance aliases, which fixes `Deepholm -> Deephome`
- active corpus-runner behavior now:
	- `scripts/export_ml_corpus.ps1` no longer calls `ml-harvest` on a dataset root with zero tile JSON files
	- `WoWMapConverter.Cli ml-corpus` also skips harvest for empty dataset roots instead of turning the whole job into a false-negative failure
- validation completed:
	- added focused unit tests in `WoWMapConverter.Core.Tests/VLM/VlmDatasetExporterTests.cs` for normalized alias recovery and the concrete `Deepholm -> Deephome` case
	- `dotnet test i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/WoWMapConverter.Core.Tests.csproj -c Debug --filter VlmDatasetExporterTests` passed (`2/2`)
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` succeeded with existing warnings only
	- bounded real-data probe `ml-export --client H:\CLIENTS\World of Warcraft Cata beta 11927 --map Deepholm --limit 1` succeeded, resolved `Deephome`, and exported `1` tile
- important boundary:
	- this fixes the immediate dataset refresh blocker
	- the forced full V7.5.1 corpus rerun and retraining still need to run to completion before the cleaned model can be evaluated

### Apr 13, 2026 - Dataset workflow now targets `datasets/` and emits HF-style metadata surfaces

- workflow changes landed:
	- `ml-corpus` now honors per-client `label` and `minimap_root`
	- `scripts/export_ml_corpus.ps1` now defaults to `datasets/` and passes `--minimap-root` through when configured
	- `scripts/ml_corpus_fixed_clients.json` now points its configured jobs at `i:/parp/parp-tools/datasets` and includes the `original_development` split-root job alongside the fixed client builds/maps
	- `ml-harvest` now writes root-level `metadata.jsonl` and `dataset_info.json` for HF-style parsing in addition to `ml_dataset_manifest.json`
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` succeeded after the workflow changes
	- dry-run of `scripts/export_ml_corpus.ps1` showed all configured jobs writing under `datasets/<label>/<map>` and the `original_development` job carrying `--minimap-root i:/parp/parp-tools/gillijimproject_refactor/test_data/development`
- cleanup completed:
	- migrated the bounded proof root into `datasets/original_development/development_proof_20260413`
	- archived the old `output/ml-corpus` root to `output/archive/ml-corpus_legacy_20260413`

### Apr 13, 2026 - Bounded V7.5 export proof now succeeds on `original_development` with a separate minimap-only root

- completed a real bounded export proof for the V7.5 dataset path using:
	- terrain source: `gillijimproject_refactor/test_data/original_development/World/Maps/development`
	- minimap-only root: `gillijimproject_refactor/test_data/development`
	- output: `datasets/original_development/development_proof_20260413`
- exporter fix landed first because the approved terrain source is sparse:
	- WDT `MAIN` reported `1496` tiles
	- only `352` root ADTs were actually reachable in the loose `original_development` tree
	- `VlmDatasetExporter` now filters the LK tile list against reachable root ADTs before selection so bounded runs stop choosing nonexistent center tiles and silently skipping them
- bounded proof result:
	- `4` tiles exported, `0` skipped
	- exported tile set included `development_30_36`, `development_31_36`, `development_33_31`, and `development_34_34`
	- stitched outputs were produced for full minimap, full no-object minimap, full object visibility mask, full PM4 mask, and full heightmaps
- verified V7.5 payload examples:
	- `development_31_36.json` contains `mccv_map`, `no_mccv_minimap`, `object_visibility_mask`, `pm4_mask`, `no_object_minimap`, and `terrain_only_minimap`
	- `development_34_34.json` contains `mccv_map`, `no_liquid_minimap`, `no_mccv_minimap`, and `terrain_only_minimap` while object/PM4 fields remain null for that tile
- important boundary:
	- this proves the bounded export path, the explicit terrain-vs-minimap root separation, and the new root-level HF metadata emission on a real dataset root
	- it does not yet prove a full-map export, retraining quality, or that `WoWMuseum/335-dev` is a sufficient minimap source for this workflow

### Apr 13, 2026 - Fallback object masking now covers all object families instead of only `wmo`

- fixed a real masking bug in the active terrain path:
	- exporter fallback object masks in `VlmDatasetExporter.cs` had been limited to `wmo`
	- trainer and inference fallback object-context masks in `train_v7.py` / `infer_v7.py` also skipped non-`wmo` placements
- active behavior now includes all projected object placements in the fallback mask path, while still preferring exported precise PM4/seeded masks when available
- practical meaning:
	- maps without PM4 support are no longer forced into a WMO-only fallback assumption
	- M2/doodad occlusion can now contribute to the fallback mask instead of leaking straight into the minimap RGB/context path
- validation completed:
	- file-level diagnostics were clean for the touched exporter/masking files
	- `python -m py_compile` passed for `train_v7.py` and `infer_v7.py`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` succeeded again after also fixing the old `VlmMinimapCleanupService.cs` build blocker
- important boundary:
	- no real-data dataset re-export has been run yet, so this is code/build proof rather than corpus-output proof

### Apr 13, 2026 - Promoted the active terrain line to V7.5 with terrain-only minimap precedence

- landed the V7.5 contract bump across exporter, trainer, inference, audit, and docs
- active behavior now:
	- exporter writes `terrain_only_minimap` when it has enough auxiliary masks to build one
	- the cleaned minimap starts from `no_mccv_minimap` when available and then removes object, PM4, liquid, stitched alpha, and stitched shadow contamination before inpainting
	- `train_v7.py` and `infer_v7.py` now prefer `terrain_only_minimap` ahead of `no_object_minimap`, `no_mccv_minimap`, and raw `image`
	- dataset index cache version was bumped so stale root caches do not mask the new field
	- audit/docs now reflect the V7.5 semantics, and a new architecture guide lives at `docs/v75-model-architecture-guide.md`
- validation completed:
	- file-level diagnostics were clean for the touched exporter and script files
	- `python -m py_compile` passed for `train_v7.py`, `infer_v7.py`, and `audit_v7_signals.py`
- important boundary:
	- full `WoWMapConverter.Core` compile proof is still blocked by existing unrelated errors in `VlmMinimapCleanupService.cs`
	- no real-data re-export or bounded V7.5 training smoke has been run yet, so this is not trained-model proof

### Apr 13, 2026 - Relaunched the full improved V7.4 run with pinned `development_0_0` validation and safer object-mask precedence

- patched `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` and `infer_v7.py` so object-mask context now prefers:
	- `object_visibility_mask_cv2`, `pm4_mask`, `pm4_object_mask`, `collision_mask`
	- then `object_visibility_mask`
	- then coarse fallback WMO-box projection only if no exported mask exists
- added trainer-side validation pinning for trusted reference tile `development_0_0`
	- validation groups now always include the group containing that tile when it exists
	- static preview candidates now keep that tile in the fixed preview set instead of letting visual-score ranking push it out
- bounded real-data proof from `output/tmp/v7_4_validation_pin_smoke_20260413` on `output/ml-corpus/4_0_0_12304_original/development`:
	- static preview printed `development:development_0_0` first, followed by `development:development_0_1`
	- one-epoch smoke completed with train `0.1875`, val `0.2269`
- sampled representative corpus roots still showed no exported precise PM4/MPRL mask payloads yet:
	- checked roots included `4_0_0_12304_original/development` and `400_12304/development`
	- representative tiles `development_0_0` and `development_31_36` still had `object_visibility_mask_cv2`, `object_visibility_mask`, `pm4_mask`, `pm4_object_mask`, and `collision_mask` all `null`
	- practical meaning: the trainer/inference path is now ready for precise PM4-driven silhouettes, but current corpora still need the exporter seam to emit them
- restarted the full audited-corpus improved run into `output/ml-training/v7_4_wdl_trestle_reflect_brush_bestburst_pinval_20260413`
- launch debugging/result:
	- earlier audit-filter and PowerShell argument-shape issues were already fixed before this relaunch
	- live startup now confirms `26` audited roots, `6070` valid samples, raw train/val `5449 / 621`, curated train `3230`, and static previews beginning with `development:development_0_0`
- proof boundary:

### Apr 13, 2026 - Trainer now rejects pathological object-mask coverage and hard-forces pinned validation refs after the split

- patched `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` again after real-data audit showed some current seeded object masks still covered most of a tile
- active trainer behavior now:
	- precise object masks are accepted only when coverage is `<= 50%`
	- seeded exported masks are accepted only when coverage is `<= 25%`
	- trainer fallback masks are accepted only when coverage is `<= 20%`
	- fallback object footprints now rasterize as ellipses instead of axis-aligned rectangles
	- validation grouping now keys by full dataset-root path instead of short `dataset_name`
	- pinned validation refs such as `development_0_0` are explicitly moved into validation after split if they somehow miss the grouped selection path
	- trainer startup now prints `Pinned validation refs: ...` so that regression is visible immediately in logs
- bounded real-data proof:
	- audit on `datasets/4_0_0_11927/Deepholm` and `datasets/4_0_0_11927/LostIsles` reduced worst observed trainer-side object-mask coverage from near-full-tile seeded masks to about `0.1024`
	- one-epoch smoke on `datasets/original_development/development` + `datasets/4_0_0_11927/LostIsles` printed `Pinned validation refs: development:development_0_0`, listed that tile first in static previews, and produced non-zero GAN/discriminator metrics on epoch `1`
- boundary:
	- this is trainer-side mitigation, not exporter closure; current corpora can still contain oversized seeded masks, but the trainer no longer trusts them blindly
	- this records the corrected full-run relaunch and the validation/mask-path behavior change, not new convergence proof yet

### Apr 13, 2026 - Development-map V7.4 inference now exports anchored tile borders, and the trainer now penalizes both transition blur and border curl

- ran the epoch-51 checkpoint `output/ml-training/v7_4_brush_channel_bestburst_20260413/best.pt` against the exported `development` dataset as a real-data side-quest before retraining
- first mesh inspection showed two concrete issues:
	- hard terrain transitions were still too rampy
	- tile borders curled/sloped, which made adjacent tiles stitch poorly in a quilt export
- landed inference fixes in `gillijimproject_refactor/src/WoWMapConverter/scripts/infer_v7.py`:
	- loader now matches the active `13`-channel brush-conditioned checkpoint layout
	- new `--edge-anchor-width` path anchors the outer tile band to the WDL prior so border heights stop drifting freely
- landed retrain-targeted loss fixes in `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py`:
	- `transition` loss for stronger reconstruction pressure at sharp target terrain changes
	- `tile_edge` loss for stronger reconstruction pressure on the tile border band
- re-exported real-data development outputs to `output/tmp/v7_4_dev_infer_full_edgeanchored_20260413` with no output smoothing and `--edge-anchor-width 12`
- measured seam improvement on representative neighbors:
	- `development_31_36 south` vs `development_31_37 north`: about `118.8 -> 13.4`
	- `development_30_36 east` vs `development_31_36 west`: about `406.2 -> 29.2`
- proof boundary:
	- this proves the current inference/export path is materially better for quilt inspection and that the next retrain will include explicit edge/border supervision
	- it does not yet prove the retrained model has eliminated the curl or ramp issue without the inference-side anchor

### Apr 13, 2026 - Completed the first full V7.4 best-triggered run and captured the real stopping point

- finished `output/ml-training/v7_4_brush_channel_bestburst_20260413`
- exact run summary:
	- best val `0.05059416059936796` at epoch `51`
	- final epoch `112`
	- final train/val `0.04502828886709463 / 0.05751752530454428`
	- early stopping triggered after `12` non-improving patience steps
	- metadata confirms `13` input channels, `6070` valid samples, curated train `3237`, val `613`, `26` launched roots
- practical conclusion:
	- V7.4 is now clearly in the right regime
	- later epochs did not outperform the epoch-51 checkpoint, so it remains the best legacy-semantics reference checkpoint
	- that does not mean new WDL-trestle and reflect-padding runs should resume from it; fresh improved-variant training should start clean unless resuming a checkpoint from the same variant

### Apr 13, 2026 - Added mixed validation previews and explicit object-mask context previews

- `train_v7.py` preview generation now uses both static and random held-out validation tiles each epoch
- new preview artifacts now include:
	- `val_epoch_XXXX.png`
	- `val_epoch_XXXX_local.png`
	- `val_epoch_XXXX_context.png`
	- `val_epoch_XXXX.json`
- the new context preview explicitly visualizes object-mask overlay, masked minimap diagnostic, liquid mask, and brush mask so the user can inspect what occlusion/context signals the model was given
- real-data proof captured at:
	- `output/tmp/v7_objectmask_preview_smoke_20260413/previews/val_epoch_0001_context.png`
	- `output/tmp/v7_objectmask_preview_smoke_20260413/previews/val_epoch_0001.json`

### Apr 13, 2026 - Discriminator stabilization controls now have real multi-step proof, not just syntax proof

- added discriminator stabilization controls in `train_v7.py`:
	- `--disc-real-target`
	- `--disc-fake-target`
	- `--disc-label-noise`
	- `--disc-input-noise-std`
	- `--disc-grad-clip`
- validated them on a real `LostIsles` smoke with enough train steps and `--disc-every 1` so discriminator updates actually occurred
- observed epoch-2 GAN-on discriminator health:
	- `Disc: 0.9667`
	- real/fake mean `0.4209 / 0.3878`
- proof boundary:
	- this proves the stabilized discriminator path executes on real data
	- the full audited-corpus retrain still needs to be relaunched from the epoch-51 best checkpoint with those controls enabled

### Apr 13, 2026 - Fixed V7 trainer numerics so impossible negative validation loss can no longer overwrite `best.pt`

- investigated the `output/ml-training/v7_4_brush_channel_geomfirst_20260413` anomaly where epoch 28 reported `Val Loss: -0.0060`
- landed trainer-side repairs in `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py`:
	- cast structural loss inputs to float32 before SSIM, gradient, edge, frequency, laplacian, and bounds loss assembly
	- clamped SSIM variance and denominator terms to avoid invalid ratios under AMP
	- rejected non-finite or negative validation loss as invalid for LR scheduling and `best.pt` updates
	- fixed geometry-first telemetry so empty discriminator windows print `0.0000` instead of triggering NumPy warnings while GAN is off
- documentation sync:
	- updated `gillijimproject_refactor/docs/VLM_Training_Guide.md` to state that negative validation loss is an invalid numeric artifact, not a real improvement
- validation completed:
	- `C:\Users\akspa\anaconda3\python.exe -m py_compile i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` passed
	- tiny real-data smoke run on `output/ml-corpus/400_12304/development` completed one epoch with sane metrics (`Val Loss: 0.1466`) and no empty-discriminator warnings
- proof boundary:
	- this proves the active trainer no longer reproduces the negative-validation bug in the bounded smoke path
	- the old `v7_4_brush_channel_geomfirst_20260413/best.pt` remains untrusted because it was written before the validity guard existed

### Apr 13, 2026 - Added periodic GAN cadence controls and verified cooldown-driven GAN reactivation on real data

- extended `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` with:
	- `--gan-cycle-length`
	- `--gan-cycle-on-epochs`
	- `--gan-cooldown-after-best`
- trainer behavior now supports intermittent GAN detail passes instead of a single continuous adversarial phase
- checkpoint/history updates now persist GAN cadence state and cooldown state so resume continues the same schedule
- validation completed with a bounded real-data smoke on `output/ml-corpus/400_12304/development`:
	- command used `--start-gan-epoch 1 --gan-cycle-length 3 --gan-cycle-on-epochs 1 --gan-cooldown-after-best 2`
	- observed live cadence:
		- epoch 1 GAN on
		- epoch 2 GAN off via cooldown
		- epoch 3 GAN off via cooldown countdown
		- epoch 4 GAN on again after cooldown expired
- documentation sync:
	- updated `gillijimproject_refactor/docs/VLM_Training_Guide.md` with the new cadence flags and an audited-corpus launch example
- proof boundary:
	- this is real-data proof of the scheduling behavior, not yet evidence that a specific cadence is optimal for the full trusted brush-conditioned corpus

### Apr 13, 2026 - Switched the active GAN schedule strategy to best-triggered refinement bursts

- user rejected the arbitrary fixed warmup rule and requested GAN activation at any and every new best model instead
- extended `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` with `--gan-burst-after-best`
- when `--gan-burst-after-best > 0`, the trainer now:
	- waits with GAN off until a best checkpoint is achieved
	- arms GAN for the next `N` epochs after that best
	- rearms the burst again whenever a later best appears
	- overrides the older epoch-calendar GAN cadence path while active
- validation completed with a bounded real-data smoke on `output/ml-corpus/400_12304/development` using `--gan-burst-after-best 2`
	- observed live behavior:
		- epoch 1 GAN off while waiting for best
		- epoch 1 saved best and armed GAN
		- epoch 2 GAN on via `best-burst(2)`
		- epoch 2 saved best and rearmed GAN
		- epoch 3 GAN on again via rearmed burst
- documentation sync:
	- updated `gillijimproject_refactor/docs/VLM_Training_Guide.md` to make best-triggered GAN bursts the preferred launch recipe
- proof boundary:
	- this proves the best-trigger mechanism itself, not that `2` epochs is the final best burst length for the audited trusted corpus

### Apr 13, 2026 - Reduced the practical training horizon to `100` epochs and let early-stop count immediately

- the finished best-triggered run proved the current regime does not need `140` epochs:
	- best val `0.0506` occurred at epoch `51`
	- the run eventually stopped at epoch `112` after `12` non-improving patience steps
- updated `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` defaults:
	- `DEFAULT_NUM_EPOCHS = 100`
	- `DEFAULT_EARLY_STOP_START_EPOCH = 1`
- rationale:
	- with validation already driving best-checkpoint selection, LR scheduling, and best-triggered GAN bursts, there is no good reason to suppress early-stop counting until epoch `101`
	- this keeps the run bounded and avoids spending another `60+` epochs after the point where the curve has already told us enough

### Apr 13, 2026 - Retuned `train_v7.py` defaults for a long geometry-first warmup before GAN activation

- changed trainer defaults in `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` after the brush-conditioned run still showed the same wobble pattern rather than settling cleanly
- landed behavior changes:
	- `--adversarial-scale` default: `0.20`
	- `--start-gan-epoch` default: `101`
	- `--disc-every` default: `2`
	- `--early-stop-start-epoch` default: `101`
	- ReduceLROnPlateau now uses parser-controlled patience/factor with default patience `8`
	- early-stop patience no longer counts during the geometry-first warmup window
- rationale:
	- let the model learn terrain structure first
	- delay GAN until the base geometry path has a chance to settle
	- avoid scheduler/early-stop reactions that were previously too aggressive for the intended long-run regime
- proof boundary:
	- this is code-level schedule correction only; a new real run is still required to prove that the longer geometry-first phase improves convergence

### Apr 13, 2026 - Scaled brush-imprint harvest across the trusted corpus and added a first brush mask input channel to V7

- executed `ml-harvest-brushes` across the trusted corpus into `output/build-validation/brush-imprints/trusted/`
- all-corpus brush harvest summary:
	- `27` manifests
	- `10,541` processed tiles
	- `259,216` grouped candidates
	- `51,741,807` patch cells
	- only one zero-group root observed so far: `400_11927_Uldum`
- updated the harvester in `wow-viewer/tools/converter/WowViewer.Tool.Converter/MlBrushImprintHarvester.cs` so each tile now also emits a tile-level `brush_mask_path` under `brush_imprints/tile_masks/`
- integrated the first brush-conditioning seam into `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py`
	- raised `MODEL_INPUT_CHANNELS` from `12` to `13`
	- `TileSample` now carries `brush_mask_path`
	- dataset loader now reads `brush_imprints/brush_imprint_manifest.json` and resolves per-tile brush masks
	- training input tensor now appends the brush mask after the object mask channel
- validation completed:
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed
	- `C:\Users\akspa\anaconda3\python.exe -m py_compile i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py` passed
	- dry trainer smoke on `output/ml-corpus/400_12304/development` with `--epochs 0 --batch-size 1 --limit 4 --no-augment` loaded `3` usable samples and reached CUDA startup successfully with the new brush channel present
- proof boundary:
	- this is a first tile-level brush mask conditioning seam only
	- grouped brush candidates are harvested and stored, but not yet embedded or consumed directly as a separate brush model or retrieval system
	- next work should decide whether to keep iterating on tile-mask conditioning or split immediately into a dedicated brush-pattern model family over the harvested candidate dataset

### Apr 13, 2026 - Added first wow-viewer patch-scale brush-imprint harvester and validated it on rescued development tiles

- implemented a new wow-viewer command in `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs` backed by `wow-viewer/tools/converter/WowViewer.Tool.Converter/MlBrushImprintHarvester.cs`
	- command: `ml-harvest-brushes --dataset-root <path> [--output-dir <dir>] [--limit <count>] [--write-previews]`
- landed active behavior:
	- reads ML dataset JSON from `dataset/`
	- converts each tile into a `16x16` chunk grid and a `256x256` patch-cell grid
	- scores patch cells from terrain-shape change on the `257x257` global height lattice
	- flood-groups adjacent strong cells into candidate patch-group imprints
	- writes:
		- `brush_imprint_manifest.json`
		- one JSON file per grouped candidate under `groups/`
		- optional preview masks under `previews/`
- validation completed:
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed after the new command landed
	- real-data subset validation passed with:
		- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-harvest-brushes --dataset-root i:/parp/parp-tools/output/ml-corpus/400_12304/development --output-dir i:/parp/parp-tools/output/build-validation/brush-imprints/development_40012304 --limit 6 --write-previews`
	- subset results written under `output/build-validation/brush-imprints/development_40012304/`:
		- `6` tiles processed
		- `250` grouped candidates
		- `17,699` patch cells across those groups
		- previews written for inspection
	- representative output: `output/build-validation/brush-imprints/development_40012304/groups/development_34_34_g0001.json`
- proof boundary:
	- this is the first patch-scale archaeology dataset seam, not final brush identity recovery or final dedupe/classification
	- current grouping is terrain-shape-first and intended to seed the separate brush dataset the user asked for; deeper clustering/modeling is still next

### Apr 12, 2026 - Audited the trusted ML corpus at scale and launched the next V7 run from the audited root set

- executed the new wow-viewer audit command across the trusted corpus into `output/build-validation/ml-audit/trusted/`
- audit coverage/result summary:
	- `27` trusted audit reports generated
	- `10,541` tiles audited
	- `1,180` tiles missing source minimaps
	- `1,018` missing global-height tiles, isolated to `301_8303/Kalimdor`
	- `0` trusted audits currently expose stitched alpha-mask coverage
	- liquid review counts: `16` `below-terrain-likely`, `158` `uncertain`
- launch decision:
	- excluded quarantined roots as before
	- excluded `301_8303/Kalimdor` from the training launch because the audit proved the active trainer's required `heightmap_global` target is absent there
	- kept the rest of the audited roots for the next geometry-first/full-corpus run because the current trainer does not depend on alpha-mask coverage
- started a new training run with `C:\Users\akspa\anaconda3\python.exe`:
	- script: `gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py`
	- output: `output/ml-training/v7_4_audited_all_trusted_20260412`
	- args: `--profile manual --epochs 16 --learning-rate 8e-5 --disc-learning-rate 5e-5 --adversarial-scale 0.20 --start-gan-epoch 6 --disc-every 2 --patience 8 --amp-dtype auto --train-workers 4 --val-workers 2 --log-every 5`
	- live startup proof captured: preview-tile ranking printed, CUDA training initialized on `NVIDIA GeForce RTX 4070 Ti SUPER`, AMP `bfloat16`, TF32 on/on
- preserved next-slice requirement from user correction:
	- future prefab/brush dedupe must go below tile scale
	- treat tiles as `16x16` chunks and chunks as `16x16` patch candidates for later patch-level dedupe/brush harvesting work
- proof boundary:
	- this completes the first full trusted-corpus audit pass and starts the next run, but it does not yet prove convergence or patch-scale prefab recovery

### Apr 12, 2026 - Landed first wow-viewer `ml-audit-signals` command for V7.4 corpus truth auditing

- implemented a new headless audit command in `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs`:
	- `wowviewer-converter ml-audit-signals --dataset-root <path> [--output <report.json>] [--limit <count>]`
- landed active audit/report behavior:
	- reads legacy dataset tile JSONs from `dataset/`
	- computes per-tile signatures and grouped summaries for:
		- dedupe groups
		- concept clusters
		- retention recommendation (`canonical` / `review-duplicate`)
		- liquid semantic class (`visible-surface`, `below-terrain-likely`, `uncertain`, `none`)
		- signal coverage counts for minimap, heights, alpha, objects, liquids, and `no_liquid_minimap`
	- keeps the first slice inside wow-viewer command ownership without requiring immediate trainer changes or full schema migration first
- validation completed:
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-audit-signals --dataset-root i:/parp/parp-tools/output/ml-corpus/301_8303/Northrend --output i:/parp/parp-tools/output/build-validation/ml-audit/northrend_signal_audit.json --limit 32` passed
	- produced `output/build-validation/ml-audit/northrend_signal_audit.json` with first real sample metrics:
		- `32` tiles processed
		- `21` concept clusters
		- `24` dedupe groups
		- `8` tiles flagged as `review-duplicate`
		- liquid split: `26` visible-surface / `3` below-terrain-likely / `1` uncertain / `2` none
- proof boundary:
	- this is first-pass curation/audit proof on a bounded real corpus sample, not full corpus rerun coverage and not final semantic signoff for liquid supervision or concept identity

### Apr 12, 2026 - Added V7.3 fine-tune controls after epoch-6..10 validation drift and documented best-checkpoint continuation recipe

- observed continuation drift on the full trusted-corpus resume (`epochs 6..10`) while best stayed at `0.0493` from epoch 5:
	- val loss path: `0.1433`, `0.1589`, `0.1723`, `0.1682`, `0.1758`
	- throughput remained high (`4.32 steps/s`, `17.3 samples/s`) but quality did not recover
- landed trainer controls in `src/WoWMapConverter/scripts/train_v7.py` to support controlled GAN fine-tuning:
	- `--adversarial-scale`, `--start-gan-epoch`, `--disc-every`, `--disc-learning-rate`
	- resume state restoration for optimizer/discriminator/scheduler/scaler (opt-out with `--no-resume-optimizer`)
	- checkpoint payload now stores optimizer/discriminator/scheduler/scaler state plus patience counter
- documentation sync:
	- `docs/VLM_Training_Guide.md` now includes a dedicated "Fine-Tune Recipe" that resumes from `best.pt` into a new output folder with reduced GAN pressure
- proof boundary:
	- this improves control of late-epoch GAN pressure and resume continuity; improvement in best val requires the next continuation run to validate

### Apr 12, 2026 - Pivoted to geometry-first recovery run after epoch 7/8 drift persisted under reduced GAN pressure

- live continuation evidence showed no recovery after controls were introduced (`epoch 7 val 0.1813`, `epoch 8 val 0.1706`, best still `0.0493`)
- stopped the active run and launched a stricter recovery profile from `best.pt` into `output/ml-training/v7_3_all_trusted_maps_geom_recover_20260412`
- recovery profile settings:
	- `--learning-rate 1e-5`
	- `--disc-learning-rate 1e-5`
	- `--adversarial-scale 0.0`
	- `--start-gan-epoch 999`
	- `--disc-every 4`
	- `--no-augment`
	- `--no-resume-optimizer`
	- trust-filtered 31-root corpus (no `__UNTRUSTED_DO_NOT_USE`)
- current status:
	- run started successfully with CUDA + AMP bfloat16 + TF32 enabled; awaiting first post-pivot epoch summary for quality check

### Apr 12, 2026 - Restarted full trusted-corpus training from epoch 0 with current architecture/settings

- after repeated resume/fine-tune trajectories remained far above the epoch-5 best, switched to a clean restart to remove cross-run optimizer/schedule/history effects
- launched new run in `output/ml-training/v7_3_all_trusted_maps_fresh_20260412` with no `--resume`
- launch profile:
	- `--epochs 16 --learning-rate 8e-5 --disc-learning-rate 5e-5`
	- `--adversarial-scale 0.20 --start-gan-epoch 6 --disc-every 2 --patience 8`
	- `--amp-dtype auto --train-workers 4 --val-workers 2 --log-every 5`
	- 31 trust-filtered dataset roots (no quarantined lineage)
- current status:
	- run started cleanly on CUDA and is now the active canonical training trajectory to evaluate against the prior `0.0493` best

### Apr 12, 2026 - Landed live V7.3 CLI telemetry, explicit Tensor Core controls, and a measured +8.8% throughput gain on the benchmark slice

- implemented training-runtime visibility and performance controls in `src/WoWMapConverter/scripts/train_v7.py`:
	- tqdm live metrics now show rolling generator/discriminator loss, LR, and VRAM
	- per-epoch summary now prints throughput (`steps/s`, `samples/s`)
	- added `--amp-dtype auto|bfloat16|float16`, `--no-tf32`, `--no-amp`, `--no-cudnn-benchmark`, `--train-workers`, `--val-workers`, and `--log-every`
	- Tensor Core path now explicitly enables TF32 matmul/cuDNN by default when CUDA is active
	- stabilized AMP by forcing FFT frequency-loss inputs to float32 (fixed NaN path seen in the first mixed-precision benchmark)
	- updated default loader profile to the measured faster setting on this machine: train workers `4`, val workers `2`
- benchmark evidence (`Northrend`, one epoch, `limit=640`, batch `4`, RTX 4070 Ti SUPER):
	- baseline (`--no-amp --no-tf32 --no-cudnn-benchmark`, workers 4/2): `1.47 steps/s`, wall `72.15s`
	- Tensor Core profile (`--amp-dtype auto`, TF32 on, workers 4/2): `1.60 steps/s`, wall `69.10s`
	- measured throughput gain: `+8.8%`
- documentation sync:
	- `docs/VLM_Training_Guide.md` now includes the performance profile, benchmark values, and a ready-to-run trusted-corpus resume command
- continuation execution:
	- started full trusted-corpus continuation run to epoch `10` from `output/ml-training/v7_3_all_trusted_maps_20260411_235624/checkpoint.pt` with `--amp-dtype auto --train-workers 4 --val-workers 2 --log-every 5`
	- startup confirms trusted root filtering (`31` roots), cache hits, CUDA `AMP: bfloat16`, and TF32 matmul/cuDNN both enabled
- proof boundary:
	- this proves practical speed gain on the benchmark slice and stable mixed-precision execution; it is not by itself full-corpus quality/convergence signoff

### Apr 12, 2026 - Completed full trusted-corpus V7.3 resume through epoch 5 (best val loss 0.0493)

- completed the requested continuation run by resuming `output/ml-training/v7_3_all_trusted_maps_20260411_235624/checkpoint.pt` and training through epoch 5 on the same all-trusted root set
- trust gate held during execution: 31 non-quarantined roots only; no `__UNTRUSTED_DO_NOT_USE` paths included
- run outcome after resume (`output/ml-training/v7_3_all_trusted_maps_20260411_235624`):
	- resumed at epoch 1, completed epochs `2..5` on CUDA (`809` steps per epoch)
	- validation losses: epoch 2 `0.0807`, epoch 3 `0.0529`, epoch 4 `0.0678`, epoch 5 `0.0493`
	- best validation loss improved from prior `0.0979` baseline to `0.0493` (saved best model at epoch 5)
	- corpus shape remained stable: `6070` valid samples (`2708` blank skipped), train/val `5451/619`, curated train `3233`
- documentation/continuity sync completed:
	- `output/v73-model-architecture-guide.html` updated from "resume planned" to completed epoch-5 baseline state
	- `memory-bank/activeContext.md` updated with resumed-run evidence and current proof boundary
- proof boundary:
	- this is completed broad trusted-corpus baseline training with real run logs/artifacts, not final map-restoration quality signoff
	- no new external benchmark/eval pass was added in this step

### Apr 11, 2026 - Completed one-epoch full trusted-corpus V7.3 run across all collected non-quarantined ml-corpus roots

- executed full-scope training over every non-quarantined dataset root under `output/ml-corpus` (31 roots), including Development, Azeroth, EmeraldDream, Northrend, LostIsles, Kalimdor, Expansion01, and available PvP/Cata families
- quarantine guard held: no `__UNTRUSTED_DO_NOT_USE` roots were passed to training
- run outcome (`output/ml-training/v7_3_all_trusted_maps_20260411_235624`):
	- loaded `6070` valid samples (`2708` blank skipped)
	- train/val split: `5451 / 619`
	- curated train set: `3233`
	- epoch completed on CUDA (`809` train steps)
	- final metrics: `Train Loss 0.1762`, `Val Loss 0.0979`, `Best Val Loss 0.0979`
- startup behavior confirmed with new cache path:
	- frequent roots reported `index cache hit`
	- validation dataset startup reused preloaded sample index (`Reusing preloaded V7 sample index (6070 samples)`), avoiding a second parse pass
- proof boundary:
	- this is one-epoch smoke proof for full trusted-corpus coverage, not long-horizon convergence or final restoration-quality signoff

### Apr 11, 2026 - Added persistent V7 dataset index caching and stitched full-map object-mask outputs

- implemented startup acceleration in `src/WoWMapConverter/scripts/train_v7.py`:
	- added per-root persistent index cache (`.v7_dataset_index_cache.json`) keyed by json-count + latest mtime + total-size signature
	- switched sample collection to use cached index entries when signatures match instead of reparsing every tile JSON
	- removed duplicate startup scan for validation by reusing the first dataset's preloaded sample index (`preloaded_samples`) for `val_base_dataset`
- implemented stitched object-mask outputs in `src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs`:
	- now emits full-map stitched no-object minimap: `stitched/<map>_full_minimap_no_objects.png`
	- now emits full-map stitched object-visibility mask: `stitched/<map>_full_object_visibility_mask.png`
- validation completed:
	- startup cache probe on core restoration roots (Development/Azeroth/EmeraldDream/Northrend/LostIsles) with `--epochs 0` measured `first=142.05s`, `second=16.42s`
	- real-data export smoke (`Northrend`, 12 tiles, 3.0.1.8303 client) produced:
		- `output/tmp/vlm_stitch_object_mask_smoke/stitched/Northrend_full_minimap_no_objects.png`
		- `output/tmp/vlm_stitch_object_mask_smoke/stitched/Northrend_full_object_visibility_mask.png`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` succeeded (warnings only)
- proof boundary:
	- no new automated tests were added in `WoWMapConverter.Core.Tests`; validation here is build + real-data export smoke + measured startup timing

### Apr 11, 2026 - Hard-blocked quarantined dataset roots on the new CK24 OpenCV mask-refinement path

- honored the explicit do-not-use instruction for `output/ml-corpus/3_3_5_12340_devcopy__UNTRUSTED_DO_NOT_USE`
- `src/WoWMapConverter/scripts/refine_ck24_object_masks.py` now hard-fails any `--dataset-root` containing `__UNTRUSTED_DO_NOT_USE` (case-insensitive), matching the strict trust-gating behavior already used by active training discovery
- validation completed:
	- `python .../refine_ck24_object_masks.py --dataset-root "i:/parp/parp-tools/output/ml-corpus/3_3_5_12340_devcopy__UNTRUSTED_DO_NOT_USE" --dry-run` exits with `Refusing quarantined dataset root...`
- proof boundary:
	- this blocks the new CV2 refinement seam from consuming quarantined roots; it does not retroactively relabel historical artifacts that were generated before the guard

### Apr 11, 2026 - Stopped trusting old object-mask assumptions, proved fresh mask gating on small real exports, and resumed V7.3 training on that validated subset

- followed the explicit requirement to validate detection/masking on a few tiles before trusting another training attempt
- validation and run outcomes:
	- sampled legacy trusted roots under `output/ml-corpus/...` and confirmed stale object-mask state on checked WMO tiles (`object_visibility_mask` / `no_object_minimap` null) in `output/build-validation/mask-audit/few_tile_mask_check.json`
	- generated fresh real-data exports with current pipeline:
		- `output/build-validation/mask-audit/fresh-northrend-12`
		- `output/build-validation/mask-audit/fresh-lostisles-12`
	- fresh audit (`output/build-validation/mask-audit/fresh_mask_check.json`) reported:
		- `24` tiles total, `11` tiles with WMO objects
		- `8` tiles with non-empty object mask + no-object artifact files
		- `3` WMO tiles missing artifacts
	- projection diagnostics on those `3` tiles showed all had out-of-footprint WMO placements (`projectable_in_margin = 0`), so they are not in-tile mask misses
	- projection-aware pass report (`output/build-validation/mask-audit/fresh_mask_check_projectable.json`) showed `8/8` pass on projectable-WMO tiles
	- restarted V7.3 on only the fresh validated roots:
		- `python ... train_v7.py --profile manual --dataset-root fresh-northrend-12 --dataset-root fresh-lostisles-12 --include-map Northrend --include-map LostIsles --epochs 1`
		- `19` usable samples (`17/2` train/val), one-epoch CUDA smoke finished with best val loss `0.1949`
- proof boundary:
	- this is a focused smoke gate and smoke training restart, not full-corpus object-mask signoff across existing `output/ml-corpus/*`
	- no new automated tests were added; this is real-data audit + smoke training evidence

### Apr 11, 2026 - Repaired the dead MH2O liquid channel in the active LK exporter and landed the first shared wow-viewer MH2O reader

- followed the direct implementation request after the audit proved current corpora still had 0% effective liquid supervision: the active WotLK exporter branch was creating a liquid list and then returning `Liquids: null`
- landed active behavior:
	- `wow-viewer` now has a shared root-ADT MH2O payload seam via `src/core/WowViewer.Core.IO/Maps/AdtLiquidReader.cs` and `src/core/WowViewer.Core/Maps/AdtLiquidFile.cs`, with focused synthetic and development-path coverage in `wow-viewer/tests/WowViewer.Core.Tests/AdtLiquidReaderTests.cs`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` now reads root `MH2O`, materializes `terrain_data.liquids`, and stops dropping the liquid channel on LK exports
	- the active dataset contract now carries MH2O placement metadata (`x_offset`, `y_offset`, `width`, `height`, `exists_bitmap`), and both the stitched liquid outputs plus the `MdxViewer` dataset loader now use that metadata instead of painting every liquid chunk as full coverage
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/VLM/TileStitchingServiceLiquidTests.cs` now covers partial liquid-mask placement in the active converter tree
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter AdtLiquidReaderTests` passed
	- `dotnet test i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/WoWMapConverter.Core.Tests.csproj -c Debug --filter TileStitchingServiceLiquidTests` passed
	- one-tile real-data smoke export to `i:/parp/parp-tools/output/tmp/mh2o-smoke-335-azeroth` on the fixed `3.3.5.12340` client logged `Parsed 256 MH2O liquid layers for Azeroth_35_20`, wrote stitched `liquid_mask` and `liquid_height` outputs, and the tile JSON now contains a non-null `liquids` array
- proof boundary:
	- this proves the active exporter no longer leaves the liquid channel dead on the validated smoke tile, but it is not yet a broad corpus rerun or a cross-map signoff on partial-rect MH2O fidelity

### Apr 10, 2026 - Set the repo-shape direction for ML dataset and training workflow cutover toward wow-viewer ownership

- followed the architectural correction after the V7 signal audit proved the active corpora had silently lost effective liquid/object supervision: dataset gathering, corpus auditing, and training-workflow contracts should stop living across split legacy surfaces
- landed active planning guidance:
	- `gillijimproject_refactor/plans/wow_viewer_ml_tool_suite_cutover_plan_2026-04-10.md` now defines `wow-viewer` as the canonical owner for ML dataset contracts, shared signal extraction, headless corpus export or audit surfaces, and eventual training-workflow repo ownership
	- the same plan keeps `MdxViewer` only as a transitional GUI host for existing interactive validation or preview flows until equivalent `wow-viewer` app surfaces exist, instead of letting viewer-local ML business logic deepen again
	- the plan explicitly stages the next work as contract or audit first, then real liquid/object extraction parity, then thin-host GUI cutover, and only then trainer relocation
- proof boundary:
	- this is a direction or ownership reset only; it does not mean the wow-viewer ML export surface already has parity with the legacy exporter or that the dead liquid/object channels are already repaired

### Apr 10, 2026 - Switched VLM terrain heightmap or normalmap baking onto the MdxViewer-compatible 257x257 tile stitch path

- followed the direct correction to stop patching the old approximate exporter rasterizer in isolation and instead move the converter-side terrain bake onto the same coherent tile reconstruction logic already proven in `MdxViewer`
- landed active behavior:
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/TerrainTileBakeService.cs` now ports the viewer-style `257x257` tile bake into `WoWMapConverter.Core`, including chunk-edge overlap averaging, mixed-parity fill, fallback nearest fill, alpha-era raw `MCVT` normalization, and mesh-derived normal regeneration from the coherent tile height surface
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` now emits local or global heightmaps and tile normalmaps through that shared bake path for both alpha-style non-interleaved chunks and LK-style interleaved chunks instead of using the older chunk-local rasterizer or raw `MCNR` interpolation path
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/HeightmapBakeService.cs` now uses the same shared bake path, so `ml-bake-heightmap` and exporter outputs no longer drift on tile reconstruction rules
	- `src/WoWMapConverter/WoWMapConverter.Core.Tests/VLM/TerrainTileBakeServiceTests.cs` now adds focused regression coverage for alpha `MCVT` ordering and flat-surface normal generation
- validation completed:
	- `dotnet test i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/WoWMapConverter.Core.Tests.csproj -c Debug` passed with the new terrain bake tests plus the existing shadow-analysis tests
	- one-tile real-data smoke export on `H:/CLIENTS/3.X_Pre-Release_Windows_enUS_3.0.1.8303/World of Warcraft` via `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- vlm-export --client "H:/CLIENTS/3.X_Pre-Release_Windows_enUS_3.0.1.8303/World of Warcraft" --map Northrend --out "i:/parp/parp-tools/output/tmp/vlm_northrend_smoke_terrain_bake" --listfile "i:/parp/parp-tools/gillijimproject_refactor/test_data/community-listfile-withcapitals.csv" --limit 1` completed successfully and wrote `images/Northrend_15_12_heightmap.png`, `images/Northrend_15_12_heightmap_global.png`, and `images/Northrend_15_12_normal.png`
- proof boundary:
	- this proves the exporter and standalone baker now share the viewer-compatible terrain stitch path and that the `3.0.1.8303` `Northrend` smoke completes on real data, but it is still only a one-tile smoke and not a broad visual signoff across the broken Northrend corpus
	- the quick visual check in-chat showed generated outputs rather than a crash or missing-path failure, but it did not include a side-by-side comparison against a known-good viewer bake or the older broken exporter output

### Apr 10, 2026 - Fixed LK normalmap export and completed the first real-data V7 smoke on 3.3.5 Azeroth

- followed the reprioritized minimap-to-terrain path after auditing the exported corpora and confirming the immediate blocker: LK/modern `ml-export` was leaving `terrain_data.normalmap` null even on fresh real-data exports, so `train_v7.py` strict mode had no usable samples
- landed active behavior:
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` now carries LK `MCNR` normals and `MCCV` colors into `VlmChunkLayers` and emits `NormalmapPath` / `MccvMapPath` for LK tiles instead of hardcoding them to null
	- `src/WoWMapConverter/scripts/train_v7.py` now resizes 16-bit heightmaps with torch interpolation rather than importing `scipy.ndimage.zoom`, removing the runtime dependency that failed under the available NumPy 2.x environment during the first smoke attempt
- validation completed:
	- `dotnet test i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests/WoWMapConverter.Core.Tests.csproj -c Debug` passed with the existing two shadow-analysis tests still green
	- real-data smoke export to `i:/parp/parp-tools/output/tmp/v7-normal-smoke-335-azeroth` produced `images/Azeroth_35_20_normal.png`, and the tile JSON now points `terrain_data.normalmap` at that file
	- four-tile real dataset export to `i:/parp/parp-tools/output/tmp/v7-train-smoke-335-azeroth-4` produced four usable strict-mode samples with normalmaps
	- one-epoch V7 smoke with `C:\Users\akspa\anaconda3\python.exe ... train_v7.py --profile manual --dataset-root i:/parp/parp-tools/output/tmp/v7-train-smoke-335-azeroth-4 --epochs 1 --batch-size 1 --limit 4 --no-augment --spatial-group-size 1` completed on CUDA with `Train Loss: 0.3417`, `Val Loss: 0.3213`, and `Saved best model`
- proof boundary:
	- this is still only a tiny real-data smoke on `3.3.5.12340` `Azeroth`, not broad model-quality validation or a rerun of the larger harvested corpora
	- the workspace `.venv` remains unsuitable for V7 training as configured; the successful smoke used the machine-wide Conda Python that already had torch and torchvision installed

### Apr 10, 2026 - Added first-pass exporter fields for explained-vs-residual `MCSH` scar labeling

- followed the request to move from problem-definition into a concrete dataset seam without pretending that the full ML dataset or V7 training path is already proven
- landed active behavior:
	- `VlmDataModels.cs` now extends `shadow_analysis` with chunk-level explained/residual shadow counts and ratios plus region-level `explained_by_current_objects`, `explained_overlap_ratio`, `nearest_candidate_distance_px`, `scar_candidate_score`, and `scar_type`
	- `VlmShadowAssociationService.cs` now builds a projected current-object footprint mask in chunk-shadow space and uses it to classify connected `MCSH` regions as `explained_current`, `ambiguous_mixed`, `unexplained_scar`, or `non_object_shadow`
	- `src/WoWMapConverter/WoWMapConverter.Core.Tests/` now contains focused unit coverage for the pure shadow-analysis seam instead of leaving this slice entirely untested
- proof boundary:
	- this is still heuristic exporter-side labeling only; no real exported corpus audit, no end-to-end ML dataset validation run, and no successful V7 training run have been captured yet for this seam

### Apr 10, 2026 - Defined a third ML model family for `MCSH`-driven missing-object recovery

- followed the direction that `MCSH` shadow payloads can encode object-history evidence beyond generic shadow supervision
- landed active documentation guidance:
	- `docs/VLM_DATASET_EXPORTER.md`, `docs/VLM_Training_Guide.md`, and `docs/V7_HEIGHT_REGRESSOR.md` now describe a third model family alongside terrain and texture work
	- `docs/SHADOW_SCAR_OBJECT_RECOVERY.md` now captures the detailed rationale and workflow: explained-vs-unexplained shadow, orphan scar detection, retrieval-assisted attribution, and pseudo-label promotion
	- the new framing is `shadow scar` recovery: minimap + `MCSH` shadow evidence + surviving placements -> unexplained shadow regions, missing-object candidate masks, and restored placement hypotheses
	- the docs now call out the intended supervision path: compare rasterized current object footprints against `shadow_maps` / `shadow_bits`, then use the unexplained residual as the target surface for missing-object recovery
- proof boundary:
	- this is documentation and problem-definition work only; no extractor, labels, or trainer have been implemented yet

### Apr 10, 2026 - Repaired the fixed-client wrapper and made wow-viewer ml-corpus archive-aware plus split-ADT-aware

- followed the regression review on the first `wow-viewer` ML command port after the real fixed-client dry-run showed two concrete failures: the PowerShell wrapper no longer matched the checked-in config fields, and the converter required extracted map directories instead of scanning archive-backed clients
- landed active behavior:
	- `gillijimproject_refactor/scripts/export_ml_corpus.ps1` now resolves optional JSON properties safely under `Set-StrictMode`, honors `archive_root` plus `default_output_root`, resolves relative `client_path` values against `archive_root`, and keeps harvest enabled by default when `harvest_after_export` is absent
	- `wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs` `ml-corpus` now reads WDT `MAIN` through the new shared `WdtTileIndexReader` seam, builds tile reports directly from archive-backed client roots, and prefers `_tex0.adt` / `_obj0.adt` over root ADTs when those split companions exist
	- the same converter path now degrades malformed texture or placement reads into aggregated per-map warnings instead of aborting the entire run on the first bad tile
- validation completed:
	- `pwsh ./gillijimproject_refactor/scripts/export_ml_corpus.ps1 -DryRun` succeeded and printed the expected fixed-client export/harvest command list
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter WdtSummaryReaderTests` passed
	- `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` succeeded
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-corpus --config i:/parp/parp-tools/gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json --dry-run` completed with `maps=16 tiles=7409`
- proof boundary:
	- this closes the config mismatch and archive-awareness regressions only; it does not mean the full legacy ML export surface is ported into `wow-viewer`
	- the dry-run still reports aggregated `OverflowException` texture-read warnings on parts of the real `3.0.1.8303` corpus, which points at a remaining shared `AdtTextureReader` compatibility gap rather than another `ml-corpus` discovery failure

### Apr 10, 2026 - Added the fixed-client ML corpus wrapper, split texture decomposition into its own trainer, and removed alpha prediction from V7 terrain training

- followed the terrain-model boundary correction that alpha-mask or tileset decomposition should be a separate minimap-to-layer model, plus the request to operationalize the fixed `3.0.1.8303`, `3.3.5.12340`, and `4.0.0.11927` local clients into a reusable export corpus
- landed active behavior:
	- `gillijimproject_refactor/scripts/export_ml_corpus.ps1` now runs the checked-in `gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json` config, exporting the current narrow fixed-client subset into `output/ml-corpus/<client>/<map>/` and then running `ml-harvest` for each dataset root
	- `src/WoWMapConverter/scripts/train_texture_v1.py` now provides the first separate texture-decomposition training seam, using minimap RGB as input and exported alpha-mask / chunk-layer texture metadata as supervision for overlay alpha prediction plus chunk-slot texture classification
	- `src/WoWMapConverter/scripts/train_v7.py` and `infer_v7.py` now keep the terrain model geometry-only by removing the legacy alpha head from the active V7 contract while preserving the WDL + liquid + object-loss input design and bounds head
	- `src/MdxViewer/ViewerApp_MlTraining.cs` and the related ML docs now stop showing an `alpha` terrain-loss component and explicitly describe texture-layer decomposition as a separate model family
- validation completed:
	- `get_errors` reported no file-level errors on `train_v7.py`, `infer_v7.py`, `train_texture_v1.py`, and `ViewerApp_MlTraining.cs`
	- `pwsh ./gillijimproject_refactor/scripts/export_ml_corpus.ps1 -DryRun` printed the full fixed-client export/harvest command list without script errors
	- `i:/parp/parp-tools/.venv/Scripts/python.exe -m py_compile ... train_v7.py infer_v7.py train_texture_v1.py` completed without output
- proof boundary:
	- no real multi-client export run has been captured yet; the wrapper is command-surface validated only in this chat
	- no texture-model training run has been captured yet; this is implementation + syntax validation for the separate trainer seam, not model-quality signoff

### Apr 10, 2026 - MdxViewer validation captures now render with an orthographic top-down matrix during the batch

- followed the new viewer-validation report that generated minimaps were still offset from the source tile borders after the earlier settle-window and WL or doodad cleanup changes
- landed active behavior:
	- `src/MdxViewer/ViewerApp.cs` now uses a dedicated orthographic top-down view/projection when an active capture request belongs to an MdxViewer validation batch, instead of rendering those captures through the normal perspective scene camera
	- `src/MdxViewer/ViewerApp_CaptureAutomation.cs` now provides the validation-only matrices and keeps the validation shot itself centered on the requested tile while sampling tile-center terrain height for the top-down eye position
	- the same capture automation path now forces a deterministic validation-only terrain light direction for the batch and restores the previous override state afterward so minimap shading does not depend on the live world-light azimuth
	- the ML finalize flow now emits both a primary `viewer_validation_minimaps/` family and a matching `viewer_validation_minimaps/noliquids/` sub-folder keyed by the same tile basenames, then stitches both families into full-map composites under their respective `stitched/` folders
- validation completed:
	- `get_errors` reported no file-level errors on `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_CaptureAutomation.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` succeeded with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run
	- no new real-data validation capture comparison has been recorded yet for this orthographic framing fix

### Apr 09, 2026 - ML dataset finalize now skips baked 4k reference minimaps and uses doodad-free MdxViewer validation captures only

- followed the correction that the ML dataset workflow should stop generating baked `reference_minimaps` and should only queue live viewer validation captures for rendered minimap output
- landed active behavior:
	- `src/MdxViewer/ViewerApp.cs` finalize UI now runs the harvester in manifest-only mode, removes the baked-reference controls from the active ML surface, and updates the status text to describe manifest + viewer-validation only
	- `src/MdxViewer/ViewerApp_CaptureAutomation.cs` now forces `WorldScene.DoodadsVisible = false` for the duration of MdxViewer validation capture batches and restores the previous doodad visibility afterward
	- `src/WoWMapConverter/WoWMapConverter.Cli/Program.cs` now treats baked-reference harvest flags as ignored legacy inputs on the ML-facing help surface instead of continuing to advertise 4k reference output
- validation completed:
	- `get_errors` reported no file-level errors on the touched viewer and CLI files
- proof boundary:
	- no automated tests were added or run
	- no build or real-data validation has been captured yet for this slice in the current chat

### Apr 10, 2026 - Fixed the Wrath Silverpine/Tirisfall lamp M2 texture-collapse seam in fallback skin parsing

- followed the live M2 compatibility regression on `World\Generic\Human\Passive Doodads\Lamps\TirisfallStreetLamp01.m2` after narrowing the fault to fallback skin parsing rather than missing textures or renderer UV selection
- landed active behavior:
	- `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs` now honors the strict `SKIN` header layout during legacy fallback parsing, preserves `globalVertexOffset`, and stops inferring texture-unit stride from the end of the file when optional shadow-batch data is present
	- this restores correct batch/material decoding for the Wrath lamp repro, so the adapter now emits `textureComboIndex=0/1/2` and distinct materials for the post top, post body, and glow pass instead of mapping every geoset to texture `0`
- validation completed:
	- isolated build validation passed with `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir="i:/parp/parp-tools/output/build-validation/mdxviewer-m2-texture-fix-6/"`
	- real-data probe validation passed with `ParpToolsWoWViewer.exe --probe-m2-adapter "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" "World/Generic/Human/Passive Doodads/Lamps/TirisfallStreetLamp01.m2" --build 3.3.5.12340 --listfile "i:/parp/parp-tools/gillijimproject_refactor/test_data/community-listfile-withcapitals.csv"`; the probe now reports `[M2-BATCH]` entries for combo indices `0`, `1`, and `2` and `[M2-DIAG-MAT]` entries for `tex=0`, `tex=1`, and `tex=2`
- proof boundary:
	- no automated tests were added or run
	- no standalone viewer capture or interactive world-runtime retest was captured in this slice, so treat this as adapter/probe proof for the named Wrath lamp repro only

### Apr 09, 2026 - Chunk tool now supports invert-Z terrain edits and project-managed edited-heightmap export

- followed the request to let the existing chunk manipulator invert selected terrain vertically and save the result somewhere reusable without overstating terrain persistence support
- landed active behavior:
	- `src/MdxViewer/ViewerApp.cs` now adds an invert-Z chunk edit over the current chunk target or active selection, tracks dirty chunk-tool tiles across invert/paste edits, and exports those edited tiles as `257x257` L16 heightmaps with metadata plus a manifest under the editor project output folder
	- `src/MdxViewer/ViewerApp_Sidebars.cs` now exposes `Invert Z Chunk` / `Invert Z Selection`, `Save Edited Heightmaps`, dirty-count text, and last-output-folder status in the `Chunk Clipboard` window
	- `src/MdxViewer/ViewerApp_Workspaces.cs` now reflects the new chunk-tool heightmap-output path in the workspace save summary
- validation completed:
	- `get_errors` reported no file-level errors on the touched viewer files
	- isolated build validation passed with `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir="i:/parp/parp-tools/output/build-validation/mdxviewer-chunktool/"`; the default `bin/Debug` output remained locked by a live `ParpToolsWoWViewer` process, so the normal solution build could not be used as the proof target
- proof boundary:
	- no automated tests were added or run
	- no real-data viewer runtime retest has been captured yet for invert-Z edit feel or heightmap roundtrip on an actual loaded map session
	- this is still an output/export seam, not a general terrain ADT save pipeline

### Apr 09, 2026 - Added packed alpha atlas export for ML datasets and a viewer terrain analysis window for local-vs-global heightmap inspection

- followed the terrain-data complaint that alpha supervision was still spread across separate layer files even though the viewer already had a one-atlas export pattern, and the request to inspect per-tile vs map-global height scaling inside the viewer
- landed active behavior:
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/TileStitchingService.cs` now writes `*_alpha_atlas.png` with RGB=`alpha1..3` only, keeps stitched shadows separate, and `src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDataModels.cs` plus `VlmDatasetExporter.cs` now surface that file as `terrain_data.alpha_atlas`
	- the existing stitched `alpha_masks` outputs remain in place for compatibility, so the atlas is an additive packed view instead of a destructive format swap
	- `src/MdxViewer/ViewerApp_TerrainAnalysis.cs`, `ViewerApp.cs`, and `ViewerApp_Sidebars.cs` now add a floating `Terrain Analysis` window with per-tile-normalized heightmap preview, loaded-tile or whole-map normalized preview, and packed alpha-atlas preview for the current tile while stitched shadows remain separate outputs
- validation completed:
	- `get_errors` reported no file-level errors on the touched viewer and converter files
	- `docs/VLM_DATASET_EXPORTER.md` now documents the current ML dataset root layout, canonical tile and manifest lookup rules, and the new machine-readable schema files `docs/schemas/ml-dataset-tile.schema.json` and `docs/schemas/ml-dataset-manifest.schema.json`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Apr 09, 2026 with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run
	- no real-data viewer runtime retest has been captured yet for the new analysis window, and no real exported dataset root was re-opened to inspect the new `alpha_atlas` path on actual tile JSON

### Apr 09, 2026 - Merged the viewer dataset flow into one `Build ML Dataset` dialog and kept deterministic validation capture inline

- followed the complaint that `Harvest` should not be a separate viewer workflow and that the visible label should be `ML`, not `MK`
- landed active behavior:
	- `src/MdxViewer/ViewerApp.cs` now exposes `Build ML Dataset...` in the tools menu and uses `Build ML Dataset` as the dialog title instead of `Generate MK Dataset`
	- the same dialog now includes an inline `ML Dataset Manifest + Validation` section, so export, manifest generation, baked references, and MdxViewer validation capture configuration live in one place instead of a second modal
	- post-build manifest plus validation can auto-start after export in the same flow, while `src/MdxViewer/ViewerApp_CaptureAutomation.cs` still provides deterministic one-file-per-tile viewer-validation capture queueing with settle waits and batch state restore
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Apr 09, 2026 with existing workspace warnings only
	- file diagnostics stayed clean for `src/MdxViewer/ViewerApp.cs`
- proof boundary:
	- no real viewer capture batch was run yet, so this is still build proof only for the merged dialog flow and deterministic queue wiring
	- the active viewer and CLI surfaces now say `ML`, but internal type names still remain under `Mk*` and `VLM` for continuity
	- alpha-mask completeness and shared-reader ownership are still open

### Apr 09, 2026 - Added the first ML Dataset harvest manifest command and moved the active UI or CLI surface off the old VLM wording

- followed the new terrain-reconstruction direction by landing the first dataset-contract slice before any U-Net work: harvesting coverage, reference-minimap generation, and public naming cleanup
- landed active behavior:
	- `src/WoWMapConverter/WoWMapConverter.Core/VLM/MkDatasetHarvester.cs` now emits the default manifest file `ml_dataset_manifest.json` with per-tile coverage for source minimaps, local/global heightmaps, alpha masks, objects, chunk layers, and optional baked reference minimaps
	- `src/WoWMapConverter/WoWMapConverter.Cli/Program.cs` now routes `ml-harvest` plus `ml-export`, `ml-decode`, `ml-bake`, `ml-bake-heightmap`, `ml-synth`, and `ml-batch`, while preserving `mk-*` and `vlm-*` commands as compatibility aliases
	- `src/MdxViewer/ViewerApp.cs`, `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`, `src/MdxViewer/Terrain/VlmProjectLoader.cs`, `docs/VLM_DATASET_EXPORTER.md`, `docs/VLM_Training_Guide.md`, `plans/vlm_dataset_reconstruction_plan_2026-03-31.md`, and `src/MdxViewer/USERGUIDE.md` now present the surface as `ML Dataset`
	- `src/MdxViewer/ViewerApp.cs` now exposes one `Build ML Dataset` dialog with inline manifest/validation controls instead of a separate harvest modal
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` passed on Apr 09, 2026 with existing warnings only
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` passed on Apr 09, 2026 with existing warnings only
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- ml-harvest` printed the expected usage text
- proof boundary:
	- no real checked-in dataset export was available for `ml-harvest`, so this is still build and command-surface proof, not real-data signoff for harvest manifests or baked references
	- the new viewer harvest dialog is also build validated only; no interactive runtime capture or click-through was recorded yet
	- no ML or segmentation code landed yet; the next slice is still real-data harvesting and curation, not model training closure

### Apr 09, 2026 - Added raw-character replaceable candidate diagnostics and confirmed the tested 0.5.3 Human/Tauren variation overrides are still mostly geoset-only proofs

- followed the next likely gap after the variation-id override slice: determine whether missing non-default raw-character renders were really blocked on replaceable hair/facial textures or whether the tested assets simply did not expose those texture families on disk
- landed active viewer behavior:
	- `src/MdxViewer/Rendering/ReplaceableTextureResolver.cs` now exposes ordered replaceable-resolution candidates for probe/debug use, broadens the same-directory fallback for replaceable ids `6`, `7`, and `10`, and reports explicit diagnostic misses when no `CharSections` entry or same-directory texture match exists
	- `src/MdxViewer/AssetProbe.cs` now prints those candidate paths with existence state before decode so raw-character replaceable failures are observable instead of inferred only from `Decode: not found`
- validation completed:
	- isolated build validation passed with `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir="i:/parp/parp-tools/output/build-validation/mdxviewer-charprobe/"`; the normal debug output path remained locked by a live PowerShell process, so it was not used for the probe run
	- isolated real-data probe on `Character/Human/Male/HumanMale.mdx` with `--character-hair-variation 1` still showed the expected selection-group swap from `1` to `2`, but replaceable id `6` now explicitly reports `char-section-hair[var=1]/missing-section` plus missing same-directory hair-name candidates before ending in `Decode: not found`
	- isolated real-data probe on `Character/Tauren/Male/TaurenMale.mdx` with `--character-hair-variation 1` still showed the expected selection-group swap from `2` to `3`, but the raw model exposed only replaceable ids `1` and `8` in this case, so the proof remains geoset-only for Tauren male hair variation
	- isolated real-data probe on `Character/Human/Male/HumanMale.mdx` with `--character-facial-variation 1` likewise did not surface a new facial-hair replaceable slot in the tested raw model output
- proof boundary:
	- no automated tests were added or run
	- this slice improves diagnostics and fallback attempts, but it does not prove broad variation-specific hair/facial texture ownership for the tested 0.5.3 raw-character assets

### Apr 09, 2026 - Added scriptable standalone raw-character variation overrides for classic hair and facial-hair geosets

- followed the approved next step after default geoset repair: add a narrow override surface for raw classic character `VariationId` selection instead of leaving standalone validation locked to only variation `0`
- landed active viewer behavior:
	- `src/MdxViewer/Rendering/ReplaceableTextureResolver.cs` now exposes available hair and facial-hair variation ids per raw classic character model and can return explicit selection-group sets for requested variation ids
	- `src/MdxViewer/Rendering/ModelRenderer.cs` now exposes a targeted character-selection-group reapply path for the standalone MDX renderer
	- `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs` now surface raw `VariationId` combos for hair and facial-hair in standalone classic character model inspection, plus a reset-to-default action
	- `src/MdxViewer/ViewerApp_StartupAutomation.cs` and `src/MdxViewer/AssetProbe.cs` now accept `--character-hair-variation` and `--character-facial-variation`, so non-default raw-character cases can be validated by probe or startup capture without depending on manual UI interaction
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` passed on Apr 09, 2026 with existing workspace warnings only
	- `dotnet run --project "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj" -c Debug -- --probe-mdx "H:\053-client" "Character/Human/Male/HumanMale.mdx" --build 0.5.3.3368 --character-hair-variation 1` now reports a real selected-group swap (`SelectionGroup=1 DefaultVisible=True SelectedVisible=False`, `SelectionGroup=2 DefaultVisible=False SelectedVisible=True`)
	- `dotnet run --project "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj" -c Debug -- --probe-mdx "H:\053-client" "Character/Tauren/Male/TaurenMale.mdx" --build 0.5.3.3368 --character-hair-variation 1` also reports a real selected-group swap (`SelectionGroup=2 DefaultVisible=True SelectedVisible=False`, `SelectionGroup=3 DefaultVisible=False SelectedVisible=True`)
	- standalone runtime captures using the new startup option completed at `i:/parp/parp-tools/output/character_variation_validation/human_male_hair1/standalone/0.5.3.3368/20260409_003101393_current_20260409_003101_no_ui.png` and `i:/parp/parp-tools/output/character_variation_validation/tauren_male_hair1/standalone/0.5.3.3368/20260409_004126420_current_20260409_004126_no_ui.png`, showing the non-default override path runs through the real viewer and capture flow without breaking the render
- proof boundary:
	- no automated tests were added or run
	- this is still only a narrow raw-character variation-id surface for standalone inspection, not full closure on classic character customization breadth

### Apr 08, 2026 - Fixed the raw Alpha standalone character geoset-selection seam after the Tauren female texture repair

- followed the remaining `Character/Tauren/Female/TaurenFemale.mdx` failure after the replaceable-texture fix and confirmed the model was still rendering mutually exclusive classic character geosets together
- landed active viewer behavior:
	- `src/MdxViewer/Rendering/ReplaceableTextureResolver.cs` now loads `CharHairGeosets` and `CharacterFacialHairStyles` alongside `CharSections` and can return the default classic character geoset-selection set for raw `Character/...` models
	- `src/MdxViewer/Rendering/ModelRenderer.cs` now applies that selection set for raw standalone character MDX loads so the renderer no longer leaves every classic character geoset variant visible at once
	- `src/MdxViewer/AssetProbe.cs` now prints per-geoset `SelectionGroup` and `DefaultVisible` so live probes show the exact character visibility policy used for the render
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` passed on Apr 08, 2026 with existing workspace warnings only
	- `dotnet run --project .\MdxViewer.csproj -c Debug -- --probe-mdx "H:\053-client" "Character/Tauren/Female/TaurenFemale.mdx" --build 0.5.3.3368` still resolved `character\tauren\female\taurenfemaleSkin00_00.blp` and `...Skin00_00_Extra.blp`, and now also reported the filtered classic selection groups with the mutually exclusive alternates suppressed
	- standalone viewer runtime capture at `i:/parp/parp-tools/output/tauren_capture_geoset/standalone/0.5.3.3368/standalone/0.5.3.3368/20260408_235429888_current_20260408_235429_no_ui.png` shows a coherent textured Tauren female default body instead of the earlier broken geoset state
	- follow-up real-data probes on `Character/Human/Male/HumanMale.mdx`, `Character/SCOURGE/Female/ScourgeFemale.mdx`, `Character/Tauren/Male/TaurenMale.mdx`, and `Character/Troll/Female/TrollFemale.mdx` also loaded their expected default body textures and kept the same default-selection-group filtering instead of falling back to all visible geosets
	- follow-up standalone runtime captures at `i:/parp/parp-tools/output/character_validation/human_male/standalone/0.5.3.3368/20260409_001239225_current_20260409_001239_no_ui.png`, `i:/parp/parp-tools/output/character_validation/scourge_female/standalone/0.5.3.3368/20260409_001319208_current_20260409_001319_no_ui.png`, `i:/parp/parp-tools/output/character_validation/tauren_male/standalone/0.5.3.3368/20260409_001510190_current_20260409_001510_no_ui.png`, and `i:/parp/parp-tools/output/character_validation/troll_female/standalone/0.5.3.3368/20260409_001548840_current_20260409_001548_no_ui.png` all show coherent default raw-character renders
- proof boundary:
	- no automated tests were added or run
	- this is real-data standalone-character proof for several raw 0.5.3 default race or sex cases, not full closure on all character customization permutations

### Apr 08, 2026 - Fixed the raw Alpha dragon replaceable-texture lookup seam at the resolver layer with real client data

- followed the broken-classic-MDX-texturing report down to `Creature/Dragon/Dragon.mdx` on the real `H:\053-client` `0.5.3.3368` client instead of keeping the investigation at a generic material or shader level
- landed active viewer behavior:
	- `src/MdxViewer/Rendering/ReplaceableTextureResolver.cs` now adds broader creature-model lookup aliases and a `0.5.3`-only exact-model-path fallback sourced from alpha-core SQL creature display data
	- the resolver now prefers those fallback variants for `Resolve(...)`, `SelectBestDisplayIndex(...)`, `GetVariantCount(...)`, and `GetVariantDescription(...)` when the exact model path is known
	- `src/MdxViewer/AssetProbe.cs` now supports `--build` and uses the live replaceable resolver path so standalone MDX probes report the chosen display variant and resolved texture paths
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Apr 08, 2026 with existing workspace warnings only
	- `dotnet run --project .\MdxViewer.csproj -c Debug --no-build -- --probe-mdx "H:\053-client" "Creature/Dragon/Dragon.mdx" --build 0.5.3.3368` changed from bogus `ReplaceableDisplayIndex=0/0 Description=Helm, Rider` plus missing decodes to concrete resolved textures including `creature\\dragon\\DragonSkin1Green.blp` and `creature\\dragon\\DragonSkin2Green.blp`
	- standalone viewer runtime capture now works with `--capture-shot current`; a no-UI PNG was emitted at `i:/parp/parp-tools/output/dragon_capture/standalone/0.5.3.3368/20260408_232206684_current_20260408_232206_no_ui.png`, and the captured dragon render shows concrete green body textures instead of the earlier broken replaceable-texture state
- proof boundary:
	- no automated tests were added or run
	- this is now resolver + standalone-viewer proof for the dragon case, but it is still not broad MDX render closure and does not resolve the separate foliage or tree backface issue

### Apr 08, 2026 - Tightened taxi and POI interaction priority in the active viewer and reset taxi speed semantics around `0.10 = 100%`

- followed the request to make taxi riding and world-overlay picking usable in dense scenes instead of letting nearby scene-object bounds steal clicks
- landed active viewer behavior:
	- `src/MdxViewer/Terrain/WorldScene.cs` now treats taxi speed settings as `0.01..0.50`, where `0.10` is normal speed and values above that are slower or faster relative to that baseline instead of the old direct multiplier semantics
	- the same world-scene path now falls back to default taxi actor models (`Creature\Gryphon\Gryphon.mdx`, then `Creature\FelBat\BatTaxi.mdx`) when a route has no explicit override or mount-model resolution
	- taxi route lines and overlay pins for taxi nodes, taxi route handles, and area POIs are now visually larger and thicker in-world
	- `src/MdxViewer/ViewerApp.cs` now gives taxi nodes, taxi routes, and area POIs overlay-first pick priority before nearby WMO or MDX scene hits, adds viewport-picking for area POIs, and seeds the taxi actor override input from the resolved actor model path instead of leaving it blank when no explicit override exists
	- `src/MdxViewer/ViewerApp_Sidebars.cs` now keeps taxi controls on the right-side inspector only, clamps the taxi speed slider to the shared `0.01..0.50` range, explains the new semantics in the UI, and adds one-click Gryphon or FelBat default actor buttons
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Apr 08, 2026 with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run
	- no live runtime retest has been captured yet for the new click feel, default-actor fallback behavior, or slow-speed taxi pacing on a real route

### Apr 08, 2026 - Prepared the `v0.4.7.1` viewer release snapshot and locked the next migration seam back onto `WorldScene` to `wow-viewer`

- followed the request to package the current viewer fixes as `v0.4.7.1`
- landed release-alignment updates:
	- bumped `src/MdxViewer/MdxViewer.csproj` and `src/MdxViewer/MdxViewer.CrossPlatform.csproj` to `0.4.7.1`
	- added checked-in release notes at `src/MdxViewer/docs/releases/v0.4.7.1.md` and updated the GitHub Actions release workflow to ship that note as both the GitHub release body and `CHANGES-v0.4.7.1.md` in the archive
	- refreshed repo/viewer READMEs plus `src/MdxViewer/USERGUIDE.md` so the release snapshot now reflects the repaired taxi workflow, route capture hardening, sticky world-object selection, standalone WMO highlighted-group inspection, and the larger-range terrain follow-ups already in the active viewer host
	- refreshed continuity notes so future chats keep the next architecture step on the staged `WorldScene` to `wow-viewer` runtime split instead of treating `v0.4.7.1` as closure on that work
- proof boundary:
		- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore -nologo "-clp:ErrorsOnly;Summary"` passed on Apr 08, 2026 with existing workspace warnings only
	- this is release and continuity prep, not broad runtime signoff for the affected viewer systems

### Apr 08, 2026 - Added a dedicated minimap-generation continuity and prompt surface for the next wow-viewer migration slices

- followed the request to stop leaving the remaining minimap work implicit after the first path-filter slice landed
- recorded the integrated execution plan in `gillijimproject_refactor/plans/wow_viewer_minimap_generation_plan_2026-04-08.md`
- added `.github/prompts/wow-viewer-minimap-generation-plan-set.prompt.md` with ordered prompts for:
	- deterministic one-PNG-per-ADT capture queue
	- wow-viewer CLI minimap command surface
	- runtime-owned minimap-generation extraction
- updated `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`, `.github/prompts/wow-viewer-world-runtime-plan-set.prompt.md`, `.github/copilot-instructions.md`, `wow-viewer/README.md`, and the relevant continuity files so fresh chats can discover the new route automatically
- proof boundary:
	- this is workflow-surface and continuity work only
	- no additional minimap implementation or runtime validation landed in this slice

### Apr 07, 2026 - Standalone WMO inspection now keeps groups loaded on camera move and uses explicit highlighted labels

- followed the request to replace the bad standalone-WMO sidebar workflow with an in-scene group inspection surface
- landed behavior:
	- `src/MdxViewer/Rendering/WmoRenderer.cs` now exposes render-group bounds, names, colors, and visibility helpers for standalone WMOs
	- `src/MdxViewer/ViewerApp.cs` now calls a new standalone WMO overlay path immediately after the WMO render pass
	- `src/MdxViewer/Rendering/WmoRenderer.cs` now disables runtime group culling for standalone inspection WMOs, so camera movement no longer unloads visible groups
	- `src/MdxViewer/ViewerApp_WmoGroups.cs` now draws color-coded group boxes and mouse-driven select/toggle/isolate interactions for standalone WMOs while rendering large in-scene labels only for explicitly highlighted groups
	- `src/MdxViewer/ViewerApp_Sidebars.cs` now adds a compact standalone WMO group control block with overlay toggles plus `Hide/Show`, `Highlight Label` or `Remove Label`, `Isolate`, `Show All`, `Clear Labels`, `Clear Selection`, and `Frame` actions for the current group
	- `src/MdxViewer/ViewerApp.cs` WMO converter dialog now uses an explicit output-folder field with browse support and no longer exposes the dead `Extended` mode
	- `src/WoWMapConverter/WoWMapConverter.Cli/Program.cs` now removes the dead `--extended` / `--mode` path from `convert-wmo` and keeps the maintained converter as the only active CLI route
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore -nologo "-clp:ErrorsOnly;Summary"` passed on Apr 07, 2026 with existing workspace warnings only
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug --no-restore -nologo "-clp:ErrorsOnly;Summary"` passed on Apr 07, 2026 with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run
	- no live standalone-WMO runtime retest has been captured yet, so highlighted-label readability and interaction feel are still unproven in a real session

### Apr 07, 2026 - Fixed the active `MdxViewer` UI-to-scene input leak at the event/capture seam

- moved scene mouse-wheel handling out of the raw Silk input callback and into the per-frame update path after ImGui capture state is refreshed
- added a consistent scene keyboard gate so UI text/keyboard focus blocks chunk clipboard shortcuts, chrome hotkeys, minimap toggle, animation stepping, and free-fly movement
- tightened scene mouse blocking so hovered or active ImGui UI capture wins by default instead of only blocking non-viewport sidebars
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -nologo '-clp:ErrorsOnly;Summary'` passed on Apr 07, 2026 with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run
	- no live interaction retest has been captured yet for the original scroll/keyboard leakage bug

### Apr 07, 2026 - Landed the first pre-alpha UI slice as a persisted `MdxViewer` theme system

- followed the requested `3,2,1` order by doing theme infrastructure before paperdoll or shell rewrite work
- landed a new `src/MdxViewer/ViewerApp_Themes.cs` partial that centralizes ImGui theme application and adds a `Pre-Alpha Brass` option beside the current `Modern Slate` baseline
- wired theme persistence through `src/MdxViewer/ViewerApp.cs` viewer settings load/save so the selected theme survives restarts
- exposed the selector in `src/MdxViewer/ViewerApp_Sidebars.cs` under unified viewer settings
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Apr 07, 2026 with existing workspace warnings only
- proof boundary:
	- this is still chrome-only; paperdoll, old-shell layout work, and shared spell/character services remain open

### Apr 07, 2026 - Planned the next wow-viewer data/tool tranche around spells, paperdoll, `WorldSafeLocs`, and converter cutover

- followed the request to stop treating spell visualization, character composition, graveyard POIs, and version conversion as disconnected wishlist items
- recorded the concrete implementation plan in `gillijimproject_refactor/plans/wow_viewer_spell_paperdoll_poi_and_converter_plan_2026-04-07.md`
- locked the intended first three vertical slices:
	- shared `WorldSafeLocs` reader plus inspect/report surface
	- shared character-display resolver extraction from `MdxViewer` `ReplaceableTextureResolver`
	- shared spell inspect plus linked asset-bundle report surface
- also locked the converter direction:
	- expand `wow-viewer` `detect` into a single detect -> plan -> convert surface
	- merge overlapping WMO/model/terrain conversion behavior into `WowViewer.Tool.Converter` over shared services instead of preserving historical executable sprawl
- proof boundary:
	- this is continuity and planning only
	- no implementation, tests, or runtime validation landed in this slice yet

### Apr 07, 2026 - ADT investigation now exposes raw MCNK flags, with in-world chunk overlays and diagonal weak-corner markers

- followed the request to make impassable and related MCNK chunk flags visible in the active viewer instead of only inferable from raw file inspection
- landed behavior:
	- `src/MdxViewer/ViewerApp_Investigation.cs` now shows raw `MCNK` flag hex plus named flag labels in both the ADT investigation panel and the hovered-chunk tooltip
	- the same investigation surface now exposes a `Show MCNK Flag Overlay` toggle plus per-flag filters for impassable, river, ocean, magma, slime, shadow, MCCV, and baked-shadow chunks
	- `src/MdxViewer/ViewerApp.cs` now routes that overlay through the existing editor `BoundingBoxRenderer`, drawing chunk-top fills and outlines for loaded chunks whose raw flags match the selected overlay mask
	- diagonal impassable-only `2x2` chunk patterns now emit a visible weak-corner marker so the shared exposed corner is obvious in-world
	- `src/MdxViewer/Terrain/VlmTerrainManager.cs` now exposes loaded tile coordinates so the same overlay path can work against VLM terrain sessions too
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Apr 07, 2026 with existing workspace warnings only
- proof boundary:
	- this is compile validation only
	- no automated tests were added or run
	- no live viewer retest has been captured yet on the development map, so chunk coloring, tooltip readability, and weak-corner marker visibility are still unproven in a real session

### Apr 07, 2026 - Shared WMO liquid family resolution now drives `MdxViewer` baseline handling and the modern converter detect surface

- followed the request to stop treating 3.3.5 WMO `MLIQ` orientation as one hardcoded build-only rule when the parsed WMO already carries its own format version
- landed behavior:
	- added `wow-viewer/src/core/WowViewer.Core/Wmo/WmoLiquidLayoutResolver.cs` with asset-version-first WMO liquid family resolution and build-string fallback only when version is unknown
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoLiquidLayoutResolverTests.cs` to lock the new classification and neutral baseline behavior
	- `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs` now uses the shared resolver instead of the older `3.3.5.12340 => 270°` baseline rule
	- `wow-viewer/tools/converter/WowViewer.Tool.Converter` `detect` output now reports WMO liquid family and baseline rotation for `Wmo` and `WmoGroup` inputs
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --no-restore` passed with 270 tests succeeded and no failures
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug --no-restore` passed with existing warnings only
- proof boundary:
	- this is compile plus automated-library-test validation only
	- no focused real-data viewer retest has been captured yet for 0.5.3 versus 3.3.5 WMO liquid orientation
	- the broader legacy converter migration is still incomplete; this slice only moved the shared liquid-policy seam and the modern `detect` surface

### Apr 06, 2026 - Added taxi ride camera plus ffmpeg-backed direct mp4/mov capture

- followed the request to make taxi routes usable as an in-app teaser capture workflow
- landed behavior:
	- `WorldScene.cs` now publishes the live taxi actor pose that the animated route actor already uses internally
	- `ViewerApp_CaptureAutomation.cs` now supports direct video recording through `ffmpeg`, targeting `.mp4` or `.mov` from either the scene viewport or the full UI framebuffer
	- the same capture partial now owns a taxi ride camera with `Cockpit` and `Chase` modes plus configurable offsets
	- `ViewerApp_Sidebars.cs` now adds route-level ride-camera and route-video controls directly in the taxi section, so a selected route can be followed and recorded without leaving that workflow surface
	- `ViewerApp.cs` now updates the ride camera during the normal update loop, suppresses free-fly motion while attached, and persists the video capture settings payload in viewer settings
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed with existing warnings only
- proof boundary:
	- this is build validation only
	- no automated tests were added or run
	- no live runtime recording smoke test has been captured yet, so direct ffmpeg output and ride-camera feel are still unproven in a real viewer session

### Apr 06, 2026 - Fog distance now drives the detailed ADT footprint, and WDL far terrain can use minimap textures

- followed the request to make the detailed ADT loader react to fog distance and to make distant WDL terrain less obviously placeholder at low cost
- landed behavior:
	- `TerrainManager.cs` now computes detailed and retained ADT targets from the active terrain fog end distance instead of keeping one fixed `16`-tile near field in all cases
	- AOI refresh now also responds to fog-driven target changes, so changing fog does not wait for a tile-boundary crossing before the detailed footprint updates
	- `WdlTerrainRenderer.cs` now samples per-tile minimap textures via the existing `MinimapRenderer`, while preserving a height-color fallback for missing or not-yet-uploaded minimap tiles
	- `WorldScene.cs` and `ViewerApp.cs` now pass the viewer-owned minimap renderer into the WDL far-terrain path so the new textured fallback reuses the current cache and MD5/path-resolution logic
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed with existing warnings only
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started and reached configured game-folder loading before termination
- proof boundary:
	- this is compile plus startup-smoke validation only
	- no automated tests were added or run
	- no live visual or performance retest has been captured yet for textured WDL orientation, ADT/WDL handoff quality, or whether the fog-driven footprint actually improves the heavy-map frame time the user cares about

### Apr 06, 2026 - v0.4.7 release prep aligned version metadata, packaged docs, and GitHub Actions notes

- followed the release request to cut the current train as `v0.4.7`
- landed behavior/docs/workflow updates:
	- bumped `src/MdxViewer/MdxViewer.csproj` and `src/MdxViewer/MdxViewer.CrossPlatform.csproj` to `0.4.7`
	- added a checked-in `src/MdxViewer/docs/releases/v0.4.7.md` note and wired both release workflows to use it as the GitHub release body
	- changed packaged release docs so the archive now carries `README.md`, `MdxViewer.README.md`, `USERGUIDE.md`, and `CHANGES-v0.4.7.md`
	- refreshed both README snapshots and the shipped user guide so the release now talks about the real current shell, streaming/performance work, PM4 matching fix, and ongoing `wow-viewer` refactor direction instead of stale `v0.4.6.1` UI notes
- validation boundary:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 06, 2026 with existing workspace warnings only
	- no automated tests were added or run for this slice

### Apr 06, 2026 - Replaced the broken right-sidebar tabs with stacked sections

- followed live feedback that the new right-sidebar tabs were still not behaving correctly and should just be sequential panels in one sidebar instead
- landed behavior:
	- `ViewerApp_Sidebars.cs` now renders viewer tools as stacked collapsing sections instead of a tab bar
	- the right-sidebar section flow is now `Inspect`, `Terrain`, `PM4`, `World`, then `Diagnostics`
	- PM4 open/focus and editor task changes now use a one-shot section-open hint instead of tab selection state
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 06, 2026 with existing warnings only
- proof boundary:
	- this is compile validation only
	- no new live retest has been captured yet for the stacked-section UX

### Apr 06, 2026 - Removed the broken bottom drawer path and regrouped the shell around two static sidebars

- followed immediate live feedback after the fixed-frame cutover that the bottom drawer tabs were not functioning and that the shell should consolidate around left and right sidebars instead
- landed behavior:
	- `ViewerApp.cs` no longer reserves scene height for the bottom drawer and no longer draws that shell region
	- `ViewerApp_Sidebars.cs` now keeps the left side focused on world and asset navigation while the right sidebar owns the consolidated tool tabs for viewer workflows
	- editor mode no longer duplicates the workspace/task chooser in the left sidebar; the top toolbar still owns task routing and the right sidebar renders only the chosen editor-task surface
	- `ViewerApp_Pm4Utilities.cs` now focuses PM4 into the right sidebar instead of the removed drawer path
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 06, 2026 with existing warnings only
- proof boundary:
	- this is compile validation only
	- no new live usability retest has been captured yet after the two-sidebar regrouping

### Apr 06, 2026 - Cut the active shell back to a fixed-frame layout with a real bottom drawer and persisted static sizing

- followed the user's explicit rejection of the dockable-panel direction for the active viewer shell
- landed behavior:
	- `ViewerApp.cs` now defaults the viewer back to the fixed shell path and persists left sidebar width, right sidebar width, and bottom drawer height in viewer settings
	- `ViewerApp_Sidebars.cs` now treats the right sidebar as a selection/tool shelf and adds a resizable bottom drawer with grouped `Workspace`, `Terrain`, `PM4`, `World`, and `Diagnostics` tabs
	- `ViewerApp_Pm4Utilities.cs` now opens PM4 workflows into the fixed bottom drawer instead of forcing dockspace mode
	- viewport layout math now reserves bottom-drawer height so the scene frame scales inside the new static shell instead of rendering under it
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 06, 2026 with existing warnings only
- proof boundary:
	- this is compile validation only
	- no live runtime shell retest has been captured yet for actual usability or panel density on the development map

### Apr 06, 2026 - PM4 object match suggestions stopped ranking against the full loaded scene and now use the shared geometric comparator first

- followed direct runtime feedback that PM4 object suggestions had effectively collapsed to the same OilPlatform WMO for nearly every hovered/selected PM4 object
- root causes confirmed in the active viewer path:
	- `WorldScene.BuildPm4ObjectMatchObject(...)` was evaluating PM4 object matches against every loaded placement instead of the same local tile neighborhood used by the placement-correlation path
	- the object matcher was also ordering by the local `GetPm4ObjectMatchEvidenceRank(...)` heuristic before the shared footprint/overlap metrics, which let one nearby WMO family dominate even when the geometric fit was poor
- landed behavior:
	- PM4 object match candidates now stay within the local `±1` tile neighborhood first, with a whole-scene fallback only when that local neighborhood yields nothing
	- object-match ordering now starts from `WowViewer.Core.PM4.Services.Pm4CorrelationMath.CompareCandidateScores(...)`
	- linked-anchor gap and coarse evidence rank now act only as late tie-breaks instead of the main ranking policy
- validation completed:
	- editor diagnostics were clean for `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 06, 2026 with existing workspace warnings only
- proof boundary:
	- this is compile validation only
	- no fresh viewer retest has been captured yet to prove the top PM4 suggestions are now sane across the active development-map scene

### Apr 05, 2026 - Raised detailed ADT residency to a ranked 16-tile near field and loosened the terrain-world unique-asset load budget

- followed the user's direction to stop letting the active terrain path feel like only a handful of ADTs are detailed at once
- landed the streaming follow-up:
	- `TerrainManager.cs` now ranks a `5x5` candidate neighborhood and keeps the best `16` detailed terrain tiles instead of the older `8`-tile cross-plus-diagonal footprint
	- retention is slightly wider than the strict visible target so tile turnover is less abrupt around camera-tile boundary changes
	- terrain GPU upload throughput was raised modestly so the larger detailed footprint can populate faster
	- `WorldScene.cs` now gives terrain-world unique asset loads a less stingy per-frame budget for visible MDX/WMO requests and deferred drain when frame time is not already over budget
	- `WorldAssetManager` still keeps one loaded renderer per normalized asset path; this slice tuned the drain rate for those unique requests rather than landing new per-placement instanced rendering
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -v q -nologo`
- proof boundary:
	- this is build validation only
	- no automated tests were added or run
	- no live runtime retest has been captured yet for terrain pop-in or object streaming smoothness after the new `16`-tile policy

### Apr 05, 2026 - Converted the fixed options bar into a real workspace panel and added WoW-style `P` and `I` shell hotkeys

- followed the user's direct UI decision to stop pushing generic dock arrangements and instead lean into a WoW-like panel model
- landed the active-viewer shell follow-up:
	- the old fixed top options bar is no longer drawn in dockspace mode
	- `ViewerApp_Sidebars.cs` now exposes a real `Workspace Bars` panel containing workspace controls plus the former quick terrain/world display toggles
	- `ViewerApp.cs` now reclaims the old toolbar strip in dockspace mode by removing that top-height reservation from the scene/dock host layout path
	- `P` now toggles the new workspace-bars panel and focuses it when opened
	- `I` now toggles the existing right-side inspector/workflow set and focuses the selection panel when reopened in dockspace mode
	- the new workspace-bars panel is part of both the saved shell layout path and the grouped quadrant fallback layout
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -v q -nologo`
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started successfully and reached normal game-folder loading before shutdown
- proof boundary:
	- no automated tests were added or run
	- no live user signoff has been captured yet for the new hotkey flow or whether the workspace-bars panel fully replaces the old toolbar comfortably

### Apr 05, 2026 - Dockable shell panels now persist normalized positions and sizes, and the fallback shell arrangement is a quadrant stack with a reset action

- followed live shell feedback that the new dockable windows were still not usable enough in practice:
	- panel positions were not surviving restarts reliably
	- the default arrangement still felt messy and hard to recover from
- landed a narrow active-viewer shell improvement:
	- `ViewerApp.cs` now saves dockable shell panel rectangles into `output/settings/viewer_settings.json`
	- those saved rectangles are normalized to the dockspace host so they scale with later window sizes instead of restoring as one fixed pixel layout
	- dockable mode now has a real grouped fallback layout rather than only `FirstUseEver` placement:
		- top-left `Navigator` + `Selection`
		- top-right `Runtime Stats` + `Model Info`
		- bottom-left `PM4 Workbench` + `Minimap`
		- bottom-right `Terrain Controls` + `World Objects`
	- `View -> Reset Panel Layout` now clears saved shell panel rectangles, re-enables dockable mode, and reapplies the grouped fallback immediately
	- shell settings now also persist `Dockable Panels`, left sidebar visibility, and right sidebar visibility
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -v q -nologo`
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started successfully and loaded the configured game folder before shutdown
- proof boundary:
	- no automated tests were added or run
	- no live user retest has been captured yet for actual restart persistence or whether the grouped fallback feels meaningfully better on the current workload

### Apr 05, 2026 - Landed the first real shell slice: shared panel metadata/state and narrow-window layout clamping

- moved the active viewer one step past planning-only shell work:
	- `ViewerApp.cs` now owns a shared panel registry for the current core shell surfaces (`Navigator`, `Inspector`, `Minimap`)
	- dock-state capture, dock validation, scene viewport insets, and docked-window hit exclusion now reuse that panel model instead of handling each panel with a separate hardcoded code path
	- the fixed-sidebar width clamp was corrected so narrow windows no longer incorrectly allow `SidebarMaxWidth`; sidebars now clamp toward compact widths while preserving a hard minimum viewport width
	- the shell can now temporarily suppress side panels for layout when the window is too narrow to keep both visible at compact width, which is the first real stability bridge for non-maximized startup and later resizes
	- docked panel defaults and size constraints for navigator, inspector, and minimap now flow from one shared definition surface, which sets up the next slice for true lane/stack ownership without another ad hoc state spread
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -v q -nologo`
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started successfully and was then stopped after normal initialization output
- proof boundary:
	- no automated tests were added or run
	- no live runtime retest has been captured yet for actual resize behavior or panel usability under constrained window sizes

### Apr 05, 2026 - Dockable panel mode is now the default again, and the tabbed-sidebar detour was removed

- corrected the same-day shell mistake after user feedback:
	- dockable panels in ImGui dockspace are again the primary shell path
	- the tabbed fixed-sidebar host was removed from the fallback path instead of being treated as the new shell model
- concrete shell state after the correction:
	- the panel registry still owns explicit workflow panels for `Selection`, `PM4 Workbench`, `Terrain Controls`, `Runtime Stats`, `World Objects`, and `Model Info`
	- `_useDockspaceUi` now defaults on so the active viewer opens in the real dockable-panel mode by default
	- the dockspace path still renders those registered panels as separate windows and captures dock state for each of them
	- `OpenPm4Workbench(...)` now forces dockable mode and focuses the registered PM4 panel
	- the non-dock fallback is back to a plain legacy sidebar layout instead of tabbed lane hosts
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -v q -nologo`
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started normally and loaded the configured MPQ source plus loose overlay before shutdown
- proof boundary:
	- no automated tests were added or run
	- no live runtime retest has been captured yet for actual docked panel ergonomics or constrained-window behavior

### Apr 05, 2026 - Recorded the next viewer-shell direction: panel-based docks and drawers, no more UI-profile sprawl

- followed the user's explicit correction that the current UI is too confusing because too many controls are duplicated across sidebars, investigation surfaces, and other one-off windows
- recorded the next shell direction in continuity instead of treating it like another optional preset pass:
	- created `plans/mdxviewer_ui_panel_and_prefab_library_plan_2026-04-05.md` as the concrete implementation surface for the shell overhaul plus terrain brush or prefab harvesting pipeline
	- the active viewer should move toward individual dockable panels that can live in left, right, top, or bottom drawers/dock lanes
	- multiple panels should be able to stack in each dock area
	- panels should be pop-out capable when needed, but the default layout should stay understandable without profile switching
	- UI profiles/presets for shell organization are now treated as low-priority or non-useful compared with simply giving each workflow one canonical panel home
	- `Viewer` vs `Editor` is now treated as a workspace distinction over one editor-capable app, not as two separate shell identities
	- the same planning pass now captures the alpha-mask archaeology requirement: restore per-layer alpha inspection, build a brush or prefab detector, dedupe candidates using both 2D alpha and 3D terrain deformation data, and curate a known-good library
	- future shell work should focus on removing duplicated controls from `ViewerApp_Sidebars.cs`, `ViewerApp_Investigation.cs`, and related utility surfaces rather than adding more alternate UI modes
- proof boundary:
	- this is continuity-only and planning-only; no viewer-shell implementation changed in this step

### Apr 05, 2026 - Switched the active viewer back toward near-field ADT residency with WDL fallback and tighter object streaming

- followed the user's explicit requirement that only about `3-4` detailed ADTs should stay loaded while WDL covers distance terrain, because the prior retest was still around `18 FPS` with too many visible objects and too much terrain detail resident
- landed the streaming-policy follow-up:
	- `Camera.cs` now exposes a reusable forward vector and `ViewerApp.cs` passes it into `TerrainManager.UpdateAOI(...)`
	- `TerrainManager.cs` now uses a much smaller near-field AOI instead of the old broad square radius, and the latest tuning pass raises the fully detailed working set to 8 tiles by preferring the three most useful diagonals around the camera tile
	- `WorldScene.cs` no longer globally hides WDL for ADT-backed tiles at startup and now restores WDL visibility when an ADT tile unloads
	- object streaming now defaults to `0.50x` and can be lowered to `0.25x` in both the active viewer and the shared `wow-viewer` visibility collector
	- `ViewerApp_Investigation.cs` now exposes the lower object-stream floor in the live UI
	- `WdlTerrainRenderer.cs` now fades WDL tiles in and out over a short blend window instead of hard-popping fallback terrain on ADT load/unload
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj` smoke-started without immediate startup errors before the process was stopped
- proof boundary:
	- this proves the active viewer compiles and still starts after the AOI/WDL/object-range changes
	- it also proves the WDL fade transition compiles and starts cleanly
	- no automated tests were added or run for this slice
	- no live FPS or visible-count retest has been captured yet, so the performance outcome is still unproven

### Apr 05, 2026 - Added chunk-bucket broad-phase culling for streamed MDX/WMO objects and trimmed redundant WMO doodad transparent work

- followed a new live retest screenshot where the viewer had improved to around `14 FPS` but still showed object-heavy steady-state costs (`WMO vis/draw ~17 ms`, `MDX vis ~17 ms`, `MDX opaque ~18 ms`)
- landed the active viewer follow-up:
	- `Terrain/WorldScene.cs` now tracks aggregate bounds per streamed chunk bucket for MDX and WMO instances
	- per-frame visibility now checks those chunk buckets first and only runs the existing per-instance collectors inside buckets that survive the coarse frustum/cone/range gate
	- `Rendering/WmoRenderer.cs` now reuses a scratch list for visible doodads and skips the transparent doodad replay when a doodad renderer has no transparent world pass
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- this proves the new object broad-phase and WMO doodad trim compile in the active viewer
	- no automated tests were added or run for this slice
	- no live runtime retest has been captured yet, so no FPS claim is made yet

### Apr 05, 2026 - Restored tile-batched terrain submission in the active viewer after the user clarified the slowdown scales with loaded map tiles, not just object-heavy scenes

- followed the user's direct correction that all normal continent-sized maps are still around `5 FPS`, while only tiny maps with about `12` tiles normalize toward `60 FPS`
- landed the active terrain batching restore:
	- added `TerrainTileMesh` and `TerrainTileMeshBuilder`
	- switched `TerrainManager` from per-chunk terrain uploads to one batched terrain mesh per loaded tile
	- replaced the active `TerrainRenderer` with the tile-batching-capable path while preserving current public hooks such as MCCV toggles, runtime alpha/shadow replacement, and render-quality resampling
	- exposed terrain draw/uniform/texture-bind counters in the live renderer stats so the next retest has concrete evidence for whether terrain submission dropped
- terrain-alpha guardrail step completed:
	- compared the touched terrain batching file set against baseline commit `343dadfa27df08d384614737b6c5921efe6409c8`
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- this proves the active viewer compiles with the restored tile-batched terrain path
	- no automated tests were added or run for this slice
	- no live runtime or real-data terrain validation has been captured yet, so no FPS or alpha-blend safety claim is made yet

### Apr 05, 2026 - MDX object-pass route planning moved into wow-viewer runtime so WorldScene no longer decides batching and transparent order inline

- followed the user's post-fix retest that said the UI was less laggy but the scene was still only around `2-5 FPS`, then switched from host-side symptom work to the requested `WorldScene`-thinning path in `wow-viewer`
- landed the slice:
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassFrame.cs` now stores planned opaque and transparent MDX routes plus the first batched opaque visible index
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassCoordinator.cs` now plans opaque/transparent MDX routes and executes the planned route lists
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldVisibleMdxPassRoute.cs` is the new shared route contract
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now consumes that route-planning seam and only does renderer lookup plus actual draw submission for the planned entries
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldObjectPassCoordinatorTests|WorldFramePassCoordinatorTests|WorldObjectVisibilityCollectorTests"`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- this proves the extracted routing contract and the active compatibility build
	- it does not yet prove FPS recovery or full runtime closure on the live scene

### Apr 05, 2026 - UI multi-click lag traced to low-FPS mouse-event loss in the active ImGui backend path

- followed live feedback that the active UI itself was often taking `3+` clicks to register button presses, which was too specific to explain away as generic scene sluggishness alone
- confirmed the current root issue in `ViewerApp.cs`:
	- the active Silk.NET OpenGL ImGui backend samples mouse buttons once per frame with `CaptureState()`
	- at very low frame rates, short click down/up transitions can happen entirely between frames and never reach ImGui, making buttons appear randomly dead until one click happens to overlap a frame
- landed the fix:
	- `ViewerApp.cs` now queues raw Silk mouse down/up transitions and flushes them into ImGui as explicit mouse-button events right after `_imGui.Update(...)`
	- this is a host-side mitigation for low-FPS input loss; it does not claim the broader scene-performance collapse is solved
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- this is build validation only
	- no live runtime signoff has been captured yet for actual button responsiveness improvement

### Apr 05, 2026 - Liquid pass now frustum- and fog-culls loaded meshes instead of drawing every loaded chunk

- followed a fresh runtime screenshot showing `World CPU` still near `89 ms` with the terrain liquid stage alone around `16.69 ms`
- found a direct hot-path issue in the active viewer: `Terrain/LiquidRenderer.cs` was iterating and drawing all loaded liquid meshes with no frustum test and no fog-range distance cull
- landed the fix:
	- terrain and WL liquid meshes now carry bounds and are culled against the current frustum plus a fog-range distance threshold before draw submission
	- `ViewerApp_Sidebars.cs` now reports `Liquid visible: visible/total` for terrain and WL liquid meshes so the next live screenshot can confirm whether the pass is still oversubmitting
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- this is build validation only
	- no live runtime signoff has been captured yet for actual liquid-stage or FPS improvement

### Apr 05, 2026 - LIT base sampling bug fixed, and scene doodad visibility now suppresses WMO-internal doodads too

- followed new live evidence that the active LIT result still looked implausible and that scene-level doodad control might not actually be gating all doodad work
- landed two narrow fixes in the active `MdxViewer` path:
	- `Terrain/LitLoader.cs` now uses only an actual default light as the global/base LIT sample instead of accidentally treating the first light with any groups as the base, which was letting a local light tint the whole scene when file ordering was unfavorable
	- `Rendering/WmoRenderer.cs` plus `Terrain/WorldScene.cs` now apply the world scene's doodad visibility to WMO-internal doodad rendering, so `Show Doodads` can cut that hidden render path instead of leaving WMO doodads active behind the scene-level toggle
	- `ViewerApp_Investigation.cs` now says more honestly that LIT table selection is inspection-only while runtime sampling remains camera-driven and group-0-only
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- this is build validation only
	- no automated tests were added or run for this slice
	- no live viewer signoff has been captured yet for either corrected LIT output or WMO-related FPS improvement

### Apr 04, 2026 - FOV-aware object visibility profiles and viewer-side object-family controls landed on the active renderer path

- responded to new live feedback that the terrain-world viewer was still only reaching roughly `5 FPS` and needed explicit efficiency layers instead of only passive range throttles
- landed a new runtime-owned visibility policy slice in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility`:
	- `WorldObjectVisibilityProfile` with `Quality`, `Balanced`, and `Performance`
	- `WorldObjectVisibilityContext` now carries vertical FOV and the chosen visibility profile
	- `WorldObjectVisibilityCollector` now performs projected-size culling and skips queueing tiny low-value missing assets, while preserving near/front-view candidates
- active `MdxViewer` integration now consumes that policy:
	- `Terrain/WorldScene.cs` derives the live vertical FOV from the projection matrix and forwards it to the shared collector
	- the active viewer exposes `Show Scene Objects`, `Show WMOs`, `Show Doodads`, and `Object Detail` controls in the terrain/investigation surfaces
	- renderer stats now show the selected object-detail profile alongside the existing stream-range/readout data
- focused proof landed:
	- added new collector tests for performance-profile projected-size culling and missing-asset load gating in `wow-viewer/tests/WowViewer.Core.Tests/WorldObjectVisibilityCollectorTests.cs`
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldObjectVisibilityCollectorTests|WorldObjectPassCoordinatorTests|WorldFramePassCoordinatorTests"`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- no live same-scene retest has been captured yet, so no FPS claim is made from this slice alone
	- if standstill FPS is still unacceptable after retest, the next likely hotspot remains WMO draw/runtime decomposition rather than object-load admission alone

### Apr 04, 2026 - Integrated wow-viewer reference-renderer performance plan landed as the new parent renderer program

- added `gillijimproject_refactor/plans/wow_viewer_reference_renderer_performance_plan_2026-04-04.md` as the new high-level renderer program for `wow-viewer`
- this new plan:
	- treats `wow-viewer` as the canonical cross-version C# renderer target, not only a parser/tool repo
	- unifies world-runtime extraction, M2 runtime completion, real batching/submission work, spatial indexing/residency, shared lighting, and consumer cutover under one staged effort
	- keeps Alpha-era and 3.x-era ownership under one profile-driven engine instead of separate renderer designs
	- makes real-data proof harness work a first-class phase so later performance claims have fixed-scene evidence
- current recommended next implementation slices from the plan:
	- `wow-viewer` visible-set runtime extraction
	- `wow-viewer` M2 scene submission and batching design
	- fixed Alpha/3.3.5 performance proof harness
- proof boundary:
	- this is planning/continuity work only
	- no renderer code or runtime validation landed in this slice

### Apr 04, 2026 - World runtime slice 02 is now specified as a real build slice

- refined `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md` so the `visible-set extraction` item now names:
	- the exact first extraction seam
	- the exact runtime files to add under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility`
	- the exact `WorldScene` responsibilities that remain host-only afterward
	- the exact validation floor for the first PR-sized slice
- normalized the implementation queue item for world runtime slice 02 to match that narrower execution boundary instead of the older generic wording
- proof boundary:
	- this is planning/continuity refinement only
	- no runtime extraction code, build, or real-data validation landed in this documentation pass

### Apr 04, 2026 - First visible-set extraction bridge landed in wow-viewer runtime

- landed the first code slice of world runtime slice 02:
	- shared `WorldObjectInstance` plus runtime-owned visibility contracts and collector in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility`
	- active `MdxViewer` now consumes that seam for WMO/MDX/taxi visibility admission and visible-bucket ownership
- current host/runtime split after this slice:
	- `wow-viewer` owns pure visibility admission and visible-bucket scratch
	- `MdxViewer.WorldScene` still owns asset-ready lookup, pending-load queueing, animation advance, transparent sort, and submission
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WorldObjectVisibilityCollectorTests`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- no real-data runtime capture or performance proof was done in this slice
	- pass extraction, host thinning, and `WowViewer.App` consumer work remain open

### Apr 04, 2026 - First object-pass coordinator slice landed on top of visible-set extraction

- landed a first slice of world runtime slice 03:
	- runtime-owned object-pass scratch and pass helpers in `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes`
	- `WorldScene` now routes WMO opaque iteration, MDX animation dedup, MDX opaque iteration, transparent MDX sorting, and transparent MDX iteration through that runtime coordinator layer
- current host/runtime split after this slice:
	- `wow-viewer` owns object-pass sequencing scratch and iteration order for the visible object families
	- `MdxViewer.WorldScene` still owns GL state, renderer lookup, batch begin timing, and all non-object passes
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldObjectPassCoordinatorTests|WorldObjectVisibilityCollectorTests"`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- capture-proof boundary clarified during the same pass:
	- current `MdxViewer` capture automation can execute queued captures automatically, but queue creation is still UI-only and startup args do not yet provide a non-interactive capture path for the split development-map workflow
- proof boundary:
	- no real-data capture or performance signoff was completed in this slice
	- terrain/WDL/liquid/sky/overlay pass services and broader host thinning remain open

### Apr 04, 2026 - Startup capture hook landed, and slice 03 now owns the frame-order seam in wow-viewer

- landed a narrow non-interactive startup path in `gillijimproject_refactor/src/MdxViewer/ViewerApp_StartupAutomation.cs`:
	- direct base-client load with `--game-path` and `--build`
	- loose-overlay attach with `--loose-map-overlay`
	- world/asset load with `--world`
	- queued saved-shot capture with `--capture-shot`, optional `--capture-output`, optional `--capture-with-ui`, and optional `--exit-after-capture`
- widened world runtime slice 03 with a new runtime-owned frame-order seam:
	- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs`
	- `wow-viewer/tests/WowViewer.Core.Tests/WorldFramePassCoordinatorTests.cs`
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now routes the current lighting/sky/skybox/WDL/terrain and object-tail pass order through that coordinator while keeping host-side callbacks for the concrete renderer work
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldFramePassCoordinatorTests|WorldObjectPassCoordinatorTests|WorldObjectVisibilityCollectorTests"`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
	- no real-data startup capture was run in this session
	- no automated tests were added for the `MdxViewer` startup-automation path itself
	- no runtime/performance signoff is claimed yet

### Apr 04, 2026 - Scene picking now uses the same docked viewport projection as rendering, and WDL tile lookup is X-major again

- fixed the active `MdxViewer` selection-offset regression in `src/MdxViewer/ViewerApp.cs`:
	- the live picker and hover tooltip were already normalizing mouse coordinates against the docked scene viewport rect
	- the 3D scene itself was still being rendered with full-window viewport sizing and projection aspect, which let the mouse ray, tooltip, bounding boxes, and visible objects drift apart when docked side panels changed the usable scene width
	- the render path now applies the same docked scene viewport rectangle to OpenGL viewport setup and projection aspect, then restores the full framebuffer viewport before UI rendering
- fixed WDL tile lookup drift across active consumers:
	- `src/MdxViewer/Terrain/WdlTerrainRenderer.cs` now reads and hides WDL tiles with X-major indexing (`tileX * 64 + tileY`) instead of Y-major indexing
	- `WoWRollback/WoWRollback.PM4Module/Services/WdlService.cs` and `WoWRollback/WoWRollback.PM4Module/WdlToAdtProgram.cs` now use the same X-major MAOF lookup, which matches the rest of the map tile codepath and avoids swapped-tile WDL terrain generation
- validation completed:
	- file diagnostics were clean for the touched files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` succeeded with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run for this slice
	- no live runtime signoff was captured yet for the corrected click-selection alignment or the WDL terrain/generation output

### Apr 04, 2026 - Transparent MDX geosets now sort by camera depth within each priority plane

- fixed the active `MdxViewer` MDX material-order issue in `src/MdxViewer/Rendering/ModelRenderer.cs`:
	- transparent geosets were only being ordered by material priority plane and static geoset index during the transparent pass
	- that left some translucent MDX surfaces rendering as if they were behind their own model or nearby objects when multiple transparent geosets shared the same priority plane
	- the renderer now caches a model-space bounds center per geoset and sorts transparent geosets back-to-front by world-space camera distance within each priority plane instead of falling back to raw geoset index
- validation completed:
	- file diagnostics were clean for `src/MdxViewer/Rendering/ModelRenderer.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` succeeded with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run for this slice
	- no live runtime validation was captured yet for the affected MDX materials, so viewer signoff is still pending

### Apr 04, 2026 - Map GLB tile export now converts placement transforms into glTF Y-up space correctly

- fixed the active `MdxViewer` terrain-plus-objects GLB export mismatch in `src/MdxViewer/Export/MapGlbExporter.cs`:
	- the exporter was conjugating object placement transforms with the Z-up to Y-up basis in the wrong order for `System.Numerics` row-vector semantics
	- terrain vertices were already converted correctly, but placement matrices were landing in the wrong Y-up space, which made exported objects appear rotated or mirrored relative to the tile terrain
	- `ConvertTransformZupToYup(...)` now uses `C^{-1} * T_zup * C`, matching the direct `(X,Y,Z) -> (X,Z,-Y)` position conversion already used by the mesh builders
- validation completed:
	- file diagnostics were clean for `src/MdxViewer/Export/MapGlbExporter.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` succeeded with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run for this slice
	- no real-data viewer export re-run was performed in this session, so runtime/export signoff is still pending

### Apr 04, 2026 - MDX clicked-object selection now follows hovered identity, and terrain worlds prewarm streamed object assets

- fixed the remaining clicked-object mismatch in the live viewer path for dense MDX scenes:
	- `HoveredAssetInfo` now carries scene-object identity for MDX/WMO hits instead of only PM4 identity plus display text
	- `ViewerApp` now selects the hovered scene instance directly before falling back to the generic scene ray pick, so the clicked MDX should line up with the hovered tooltip target instead of a competing overlapping AABB
- replaced the old terrain-world object-load bottleneck with a scene load policy in `WorldScene`:
	- streamed terrain tiles now queue their tile-local MDX/WMO assets immediately on `OnTileLoaded(...)`
	- terrain maps now use a higher deferred-load throughput path with queue-pressure scaling instead of the old fixed `ProcessPendingLoads()` defaults and the old `6/3` visible-load promotion cap
	- WMO-only maps keep their eager-manifest path; terrain-streamed worlds share the same warmup behavior across early Alpha and later roots instead of adding new exact-version branches
- validation completed:
	- file diagnostics were clean for the touched files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` succeeded with warnings in this workspace
- proof boundary:
	- no automated tests were added or run for this slice
	- no live runtime validation was captured yet for the MDX selection fix or the terrain-world performance change

### Apr 04, 2026 - MdxViewer inspector build repaired; UniqueId range edits now activate hides directly

- repaired the current `ViewerApp_Sidebars.cs` compile break from the selected-object inspector regrouping slice:
	- `DrawModelInfoContent()` is restored to a valid standalone model-inspection block
	- the fixed right inspector now also exposes an `Inspector Width` slider so width can be changed without relying only on the custom splitter
- tightened the `UniqueId Archaeology` UI in `ViewerApp.cs`:
	- hide-range slider changes now set `UniqueIdFilterEnabled = true` immediately
	- the detected-layer table is compressed so the action buttons fit more reliably in the inspector
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing workspace warnings only
- proof boundary:
	- no automated tests were added or run for this slice
	- no live runtime validation was captured yet for the inspector recovery, UniqueId hide behavior, or fixed-sidebar width control

### Apr 04, 2026 - MdxViewer now keeps a grouped dirty-source queue for staged placement moves

- extended the first selected-placement save consumer into a multi-change dirty-map slice in `gillijimproject_refactor/src/MdxViewer`:
	- staged translation-only MDDF and MODF moves now persist across selection changes instead of being reset to the active selection only
	- pending moves are grouped by source ADT and can be written with `Save Current Source` or `Save All Pending`
	- the `Publish` workspace now exposes the same pending dirty-source queue so save packaging is visible outside the object inspector
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing workspace warnings only
- proof boundary:
	- this is still translation-only object persistence for existing ADT placements
	- no automated tests were added or run for this slice
	- no real-data interactive workflow validation was captured yet for the grouped save queue

### Apr 04, 2026 - MdxViewer now consumes the shared selected-placement save seam

- landed the first end-to-end UI wiring on top of the existing `wow-viewer` placement writer:
	- selected existing ADT MDDF/MODF placements can now be translated from the `Objects` workspace in `MdxViewer`
	- live preview updates propagate through `WorldScene`, tile-instance caches, and adapter placement lists instead of staying as status text only
	- save-target plumbing now supports either a resolved writable loose source path or an explicit user-chosen `.adt` output path
- supporting runtime/data-source work landed in the active viewer path:
	- writable loose-path resolution on `IDataSource`
	- placement-source and writable-path resolution on `ITerrainAdapter`
	- cached tile placement mutation in `TerrainManager`
	- tile-local placement entry tracking on `WorldScene.ObjectInstance`
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing workspace warnings only
- proof boundary:
	- this is not general map-save closure; it only covers translation-only saves for selected existing ADT object placements
	- no add/remove placement support, aggregated dirty-map model, terrain persistence, or runtime signoff was completed in this slice

### Apr 03, 2026 - wow-viewer first save-capable ADT object move transaction landed

- landed the first shared editor-save seam in `wow-viewer` instead of keeping object edits as viewer-only state:
	- new shared placement move contracts in `wow-viewer/src/core/WowViewer.Core/Maps/AdtPlacementEditTransaction.cs`
	- new shared in-place ADT placement writer in `wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtPlacementWriter.cs`
	- translation-only persistence for existing `MDDF` and `MODF` entries
	- `MODF` bounds are shifted with the moved placement so shared readers see a coherent translation result after save
- validation completed:
	- focused synthetic roundtrip coverage for `MDDF` and `MODF`
	- real-data roundtrip coverage against `gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_obj0.adt`
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter "AdtPlacementReaderTests|AdtPlacementWriterTests"` passed
- proof boundary:
	- this is a translation-only move seam for existing placements, not full editor save closure
	- no add/remove placement support, dirty-map pipeline, terrain write path, or save packaging landed in this slice

### Apr 03, 2026 - MdxViewer viewer/editor workspace shell landed in the live UI

- implemented the first actual editor-surface regrouping inside `gillijimproject_refactor/src/MdxViewer` instead of leaving the editor UI plan as prompt-only continuity:
	- new `Viewer` vs `Editor` workspace mode in the existing menu and toolbar
	- editor task routing for `Terrain`, `Objects`, `PM4 Evidence`, `Inspect`, and `Publish`
	- editor-mode navigator task rail on the left sidebar
	- editor-mode task inspector on the right sidebar
	- explicit status-bar affordances for workspace mode, active task, current target, and current save boundary
	- terrain task now hosts chunk clipboard inline in the inspector, while publish task makes export/capture-only status explicit
- proof boundary:
	- this is an MdxViewer UI-shell change only; it does not add map save, object persistence, or new format ownership
	- object task still reuses the existing mixed `DrawWorldObjectsContentCore()` surface as a first regrouping step, so follow-up extraction is still needed
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing workspace warnings only
- validation not completed:
	- no automated tests were added or run for this slice
	- no live runtime validation of the new workspace/task flow was performed yet

### Apr 03, 2026 - Plan-state audit and implementation queue normalization

- audited active and prompt-era planning docs under `gillijimproject_refactor/plans` to separate landed work from still-open slices
- added `gillijimproject_refactor/plans/plan_audit_2026-04-03.md` as the continuity snapshot for:
	- implemented-but-still-worded-as-pending items
	- truly open implementation gaps
	- stale/superseded prompt-era docs
- updated active plan statuses to reduce queue confusion:
	- `wow_viewer_m2_runtime_plan_2026-03-31.md` now explicitly treats slice 01 as landed and slices 02-05 as open
	- `wow_viewer_world_runtime_service_plan_2026-03-31.md` now marks slice 01 as partial and slices 02-05 as open
	- `mdxviewer_renderer_performance_plan_2026-03-31.md` now includes an Apr 03 status snapshot and an updated next-slice focus on phase 3
	- `wow_viewer_format_parity_matrix_2026-03-28.md` now reflects M2 foundation ownership as `partial` instead of `none`
- added `gillijimproject_refactor/plans/implementation_queue_2026-04-03.md` as the numbered chat-by-chat execution queue for upcoming implementation sessions
- proof boundary:
	- this slice is documentation/continuity maintenance only
	- no new runtime or library behavior was implemented in this audit pass

### Apr 03, 2026 - wow-viewer editor-transition prompts and continuity plan landed

- added a dedicated planning surface for the user’s stated shift from viewer-first tooling toward a real viewer-editor:
	- `.github/prompts/wow-viewer-editor-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-map-editing-foundation-plan.prompt.md`
	- `.github/prompts/wow-viewer-editor-ui-surface-plan.prompt.md`
	- matching `.codex/prompts/` mirrors
	- `gillijimproject_refactor/plans/wow_viewer_editor_plan_2026-04-03.md`
- wired the new prompt family into the existing workflow discovery surfaces:
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
	- `.codex/prompts/wow-viewer-tool-suite-plan-set.md`
	- `.github/copilot-instructions.md`
	- `AGENTS.md`
	- `wow-viewer/README.md`
- follow-up correction on Apr 03, 2026:
	- the editor plan set and map-editing foundation prompt are now explicitly worded to produce implementation-ready build plans with exact slice scope, validation commands, and immediate next actions rather than generic planning commentary
	- the editor UI surface prompt now follows the same rule, so workspace or panel planning should also return an implementation-ready UI slice with explicit dependencies and proof targets
	- the remaining companion prompts for CLI or GUI dual-surface planning and tool-migration sequencing now follow the same rule, so the full editor-transition prompt family behaves like a build queue rather than an architecture essay
- current proof boundary:
	- this slice is workflow-asset and continuity maintenance only
	- no editor runtime, map persistence, or UI-mode implementation has landed yet

### Apr 03, 2026 - MdxViewer adapted M2 material stacks and WDL far-terrain spacing were corrected in build-verified slices

- narrowed two live-runtime regressions in the active viewer path:
	- adapted M2 shiny or semi-transparent surfaces could collapse into incomplete translucent shells because `WarcraftNetM2Adapter.BuildMaterialsFromBatches(...)` still locked each skin section after its first texture unit
	- WDL far terrain was still being laid out on `WoWConstants.TileSize` instead of the viewer's 64x64 chunk grid, which stretched the low-detail mesh by `16x`
- landed focused fixes:
	- removed the per-section first-batch lock so the existing `MaterialLayer` grouping logic can build full adapted M2 material stacks again
	- switched `WdlTerrainRenderer` to `WoWConstants.ChunkSize` spacing for WDL cell placement
- current proof:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing workspace warnings only
- proof boundary:
	- no automated tests were added in this slice
	- no runtime viewer signoff has been captured yet for the affected shiny M2 models or the corrected WDL far-terrain path

### Apr 03, 2026 - wow-viewer now emits per-build ADT UniqueId manifests for later timeline diffs

- landed `map uniqueid-report` in `WowViewer.Tool.Inspect` as the first build-manifest workflow over the shared `AdtPlacementReader` seam
- the command now persists raw `MDDF` and `MODF` `UniqueId` evidence with model paths, placement metadata, duplicate-id summaries, per-source counts, and explicit failure rows instead of only printing placement counts
- current proof:
	- focused validation passed with `ArchiveVirtualFileReaderTests`, `ArchiveCatalogBootstrapperTests`, and `AdtPlacementReaderTests`
	- real development-map output now exists at `wow-viewer/output/reports/map-uniqueids/development.json`
	- that report currently records `64435` placements, `62490` distinct `UniqueId` values, `1701` reused IDs, and `114` explicit `Unknown`-kind file failures
- proof boundary:
	- this is a per-build report artifact, not yet a cross-build added/removed-object timeline engine

### Apr 03, 2026 - wow-viewer now caches trusted MPQ-era known-file universes per client/build

- landed the first shared archive listfile-cache seam in `wow-viewer`:
	- `ArchiveListfileCache` / `ArchiveListfileCacheManifest`
	- direct `IArchiveCatalog.LoadListfileEntries(...)` seeding
	- `ArchiveCatalogBootstrapOptions`-driven cache load/persist behavior
	- `archive build-listfile-cache` in `WowViewer.Tool.Inspect`
- trust model now matches the current MPQ-era rule:
	- internal MPQ listfiles from the client are primary
	- the vendored/community listfile is supplemental only
- current proof:
	- focused archive bootstrap tests passed after the compatibility fix that restored external-listfile path forwarding
	- real `0.6.0` archive data produced `wow-viewer/output/cache/archive-listfiles/0.6.0.3592.json` with `56742` trusted internal entries, `1291033` supplemental entries, and `1347773` merged known files
- consumer boundary:
	- archive-backed `mdx chunk-carriers` now benefits from the merged bootstrap file universe, but this is still discovery infrastructure rather than deeper parser/runtime closure

### Apr 03, 2026 - wow-viewer WMO flag typing now names exterior and exterior-lighting, but not `0x2`

- broadened the real-data WMO audit from Castle into Alpha Ironforge with `wmo inspect --flag-correlation`
- result:
	- the larger corpus kept the already-typed chunk-gating reads stable for BSP, lights, doodads, and liquid
	- `0x00000008` and `0x00000040` are no longer left anonymous in the shared layer; they are now typed as exterior and exterior-lighting based on the repo-local WMO notes plus the in-repo `Warcraft.NET` names
	- `0x00000002` remains intentionally unnamed because the current corpus still does not separate it into a clean shared behavior signal
- proof boundary:
	- this is still inspect/shared-summary progress in `wow-viewer`, not runtime culling or lighting signoff

### Apr 03, 2026 - wow-viewer WMO summary now carries root skybox presence and a real-data flag-correlation report

- extended the shared `wow-viewer` WMO seam one step past raw `MOSB` and `MOGP` readers:
	- `WmoSummary` now exposes root skybox presence directly as `HasSkybox`
	- `wmo inspect` now supports `--flag-correlation` to correlate `MOGP` bits against actual group chunk signals within a real root WMO
- real-data validation on `castle01.wmo.MPQ` now gives an explicit per-file evidence readout instead of only raw flag words:
	- `0x00000001` cleanly aligns with BSP presence in both groups
	- `0x00000800` aligns with doodad refs on the flagged group
	- `0x00000002` remains intentionally unknown
- proof boundary:
	- this is shared summary/reporting progress in `wow-viewer`, not runtime collision closure

### Apr 03, 2026 - LK to Alpha converter recovered from false-success and chunk-walker regressions

- fixed the active `WoWMapConverter.Core` LK→Alpha path so it no longer reports success when zero tiles convert and no longer dies behind the recent Alpha write regressions
- concrete root causes fixed in the active converter path:
	- `AlphaMcnkBuilder` had an impossible header contract: `McnkHeaderSize` was `0x88` while the writer immediately required a 128-byte Alpha header layout
	- `LkToAlphaConverter` rejected tiles too early before MCIN/top-level MCNK fallback completed
	- MCIN offsets were trusted without validating that they actually point at `MCNK` chunks
	- the top-level ADT chunk walker hard-coded odd-size padding and drifted one byte after real chunks like `MTEX` size `187`, which broke later chunk discovery on tiles such as `development_0_0.adt`
	- `MMDX` / `MWMO` extraction trusted chunk bounds too aggressively and could surface `startIndex` range failures on malformed scans
- real-data validation completed against the fixed museum path:
	- command: `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- convert-lk-to-alpha i:/parp/parp-tools/gillijimproject_refactor/test_data/WoWMuseum/335-dev/World/Maps/development/development.wdt --map-dir i:/parp/parp-tools/gillijimproject_refactor/test_data/WoWMuseum/335-dev/World/Maps/development -o i:/parp/parp-tools/output/tmp/lk-to-alpha-dev/development.wdt --verbose`
	- result after the first repair pass: `1358/2303` tiles
	- result after the chunk-walker/MCIN repair: `2303/2303` tiles
- proof boundary:
	- this is real-data CLI conversion proof for the old compatibility path, not active viewer runtime signoff and not yet a `wow-viewer` library migration

### Apr 02, 2026 - Canonical M2 documentation set landed under wow-viewer/docs/architecture/m2

- consolidated the active M2 implementation surface into one canonical doc set:
	- `wow-viewer/docs/architecture/m2/README.md`
	- `wow-viewer/docs/architecture/m2/implementation-contract.md`
	- `wow-viewer/docs/architecture/m2/native-build-matrix.md`
	- `wow-viewer/docs/architecture/m2/consumer-cutover.md`
- the new set intentionally separates:
	- implementation contract
	- per-build proof matrix
	- wow-viewer versus MdxViewer cutover rules
	- raw evidence and historical plan sources
- updated the main M2 entrypoints so future sessions land on the consolidated docs first:
	- `wow-viewer/README.md`
	- `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
	- `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`

### Apr 02, 2026 - Cataclysm next-build substitution narrowed to `4.0.0.11927`, with a hard binary-availability blocker

- advanced the cross-build investigation setup after the Wrath baseline by checking what later Win32 clients are actually reproducible in the current environment
- result:
	- the default ladder's next target `4.0.6a.13623` is not the build with existing repo-native evidence
	- the nearest documented Cataclysm-era native evidence in-repo is Win32 `4.0.0.11927`
	- a direct filesystem search under `I:\parp` found only local `WoW.exe` testdata for `0.5.5` and `0.6.0`, not Cataclysm/Mists/Warlords clients
- updated the canonical note and continuity to reflect the real proof boundary:
	- `4.0.0.11927` is the honest next Cataclysm substitution
	- current support for that slot in this session is static-only from older repo Ghidra notes
	- no fresh Cataclysm runtime attach or direct M2 choose/load/init/effect chain was possible from the currently visible files

### Apr 02, 2026 - Cataclysm `4.0.0.11927` first dedicated M2 static anchor map recovered

- used the live Ghidra-loaded `4.0.0.11927` binary to recover concrete Cataclysm M2 seams instead of relying only on older terrain/performance notes
- new confirmed static anchors recorded in the canonical note:
	- `FUN_007242d0` exact `%02d.skin` formatter
	- `FUN_00724270` exact `%04d-%02d.anim` formatter
	- `FUN_0072a740` choose-skin-profile seam
	- `FUN_0072a620` exact skin load + async callback setup
	- `FUN_0072a5f0` completion callback into `FUN_0072a4e0`
	- `FUN_0072a4e0` strict init + loaded-bit set + callback rebuild drain
	- `FUN_00725e00` active section/effect materialization from loaded skin data
	- `FUN_00724320` explicit `Diffuse_*` + `Combiners_*` effect builder with `Diffuse_T1Combiners_Opaque` fallback
	- `FUN_0072b3f0` external `%04d-%02d.anim` load path
	- `FUN_00402390` M2 runtime option registration with low bits matching Wrath but default mask `0x2008`
- runtime boundary after this slice:
	- x64dbg tools were available and attached, but the session dropped during the first rebasing attempt before a live Cataclysm breakpoint chain could be harvested
	- proof level for Cataclysm remains static-only until that runtime chain is recaptured

### Apr 02, 2026 - Win32 `0x20` flag now narrowed to a track-bearing shared-record class

- extended the canonical native note with a stronger conclusion for the long-running Wrath `0x20` question:
	- the repeated `0x20` checks in bootstrap relocation helpers now line up with exact wowdev record sizes for `M2Track<T>`, `M2Color`, `M2TextureTransform`, and `M2Light`
	- current best reading is that `0x20` marks a shared-record class with nested animated payloads that receives special relocation handling and is excluded from the compact runtime render list
- updated continuity to reflect the new proof boundary:
	- the remaining gap is the final user-facing label for that class, not whether `0x20` is real or whether it matters to bootstrap/runtime ownership

### Apr 02, 2026 - First Win32 world-path M2 choose-load capture recorded

- extended the Win32 `3.3.5.12340` native notes beyond UI-path traffic with a real in-world doodad load chain in:
	- `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
- newly confirmed world-path evidence:
	- model path `world\expansion02\doodads\generic\barbershop\barbershop_mirror_01.m2`
	- exact numbered skin output `world\expansion02\doodads\generic\barbershop\barbershop_mirror_0100.skin`
	- post-load success at `0x0083cd32` with `EAX=1`
	- downstream callback rebuild hits at `0x00832ea0`
- proof boundary:
	- this is world-path choose/load proof, not full world-path choose/load/init/effect closure yet
	- explicit `0x00838490` init capture and world-path `0x00836600` combiner capture are still pending

### Apr 02, 2026 - Second world-path skin sample and explicit init reachability captured before debugger drop

- added a second confirmed world-path sample to the canonical Win32 notes:
	- `world\expansion02\doodads\generic\barbershop\barbershop_shavecup.m2`
	- exact numbered skin `world\expansion02\doodads\generic\barbershop\barbershop_shavecup00.skin`
- proved downstream init-path reachability in the same in-world session after removing noisy front-half breakpoints:
	- `0x00838490`
	- `0x00838561`
	- `0x00836600`
- proof boundary:
	- this still does not close a world-attributed init or world-attributed combiner sample because the isolated downstream samples stayed UI-heavy
	- the x64dbg MCP session timed out and disconnected before further targeted sampling could continue

### Apr 02, 2026 - Fresh reattach pass captured world-attributed combiner and init completion

- after x64dbg restart and reattach, the narrowed downstream-only breakpoint set produced a clean world-attributed combiner sample in:
	- `world\generic\human\passive doodads\beds\duskwoodbed.m2`
- concrete world-path effect-routing result recorded in the canonical native note:
	- `Diffuse_T2`
	- `Combiners_Mod2x`
- the same world object also surfaced at `0x00838561`, the loaded-state write inside the skin-init routine
- proof level change:
	- Win32 Wrath now has direct world-path runtime evidence for choose/load, init completion, and at least one concrete combiner-family output

### Apr 02, 2026 - Static Wrath M2 runtime contract consolidated from decompilation

- expanded the canonical native note with direct decompilation-backed behavior for:
	- `M2_ChooseAndLoadSkinProfile`
	- `FUN_0083cb40`
	- `FUN_0083cb10`
	- `M2_InitializeSkinProfileAndRebuildInstances`
	- `FUN_00837a40`
	- `FUN_00836980`
	- `FUN_00837680`
	- `M2_BuildCombinerEffectName`
	- `FUN_00836c90`
	- `M2_RegisterRuntimeFlags`
	- `M2_NormalizeModelPathAndProbeSkins`
- concrete new facts now recorded include:
	- skin choose threshold ladder `0x100`, `0x40`, `0x35`, `0x15`
	- the normal and special-case combiner-family decision trees
	- the exact startup and callback-owned runtime flag bits
	- the callback-drain loop after successful skin init
	- the batching relevance of fallback bit `0x40`

### Apr 02, 2026 - Strict extension gate and external anim naming added to the Wrath contract

- extended the canonical note with direct Win32 decompilation of:
	- `FUN_0081c390` strict cache-open and extension normalization
	- `M2_FormatAnimFilename_04d_02d`
	- `FUN_00837ee0` animation-track relocation during root-model bootstrap
- new concrete facts recorded:
	- `.mdl` and `.mdx` are normalized to `.m2` in the real Win32 loader path
	- unsupported extensions still hard-fail through the `Model2: Invalid file extension` path
	- external animation filenames are formatted as `%04d-%02d.anim`
	- animation relocation is part of the strict bootstrap path, not post-init glue code

### Apr 02, 2026 - Prompt location correction: use .github, not .copilot, in this repo

- corrected continuity guidance for workflow asset placement:
	- workspace prompt and agent workflow assets for this repo stay in `.github/` (with `.codex/` mirrors)
	- `.copilot/` is not the canonical location for these repo-scoped workflow assets here
- cross-build M2 investigation prompt remains correctly located at:
	- `.github/prompts/m2-cross-build-native-investigation.prompt.md`
	- `.codex/prompts/m2-cross-build-native-investigation.md`
- boundary:
	- this is a continuity correction only
	- no runtime, parser, or renderer code behavior changed

### Apr 02, 2026 - Cross-build M2 native investigation prompt added for 3.3.5 through 6.x

- added a dedicated cross-build workflow asset for native M2 behavior recovery across expansion branches where current library support is partial:
	- `.github/prompts/m2-cross-build-native-investigation.prompt.md`
	- `.codex/prompts/m2-cross-build-native-investigation.md`
- routing and discoverability updates landed so this prompt is reachable from existing wow-viewer prompt sets and instruction registries:
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime-plan-set.prompt.md`
	- `.codex/prompts/wow-viewer-tool-suite-plan-set.md`
	- `.codex/prompts/wow-viewer-m2-runtime-plan-set.md`
	- `.github/copilot-instructions.md`
	- `AGENTS.md`
	- `.codex/prompts/README.md`
	- `wow-viewer/README.md`
- continuity boundary:
	- this slice adds workflow and investigation guidance only
	- no parser, renderer, or runtime behavior change was implemented in code in this slice

### Apr 02, 2026 - Restarted x64dbg live-open sampling added to native notes

- after restarting x64dbg and reattaching to Win32 WoW, a targeted live sample at `FUN_004609b0` captured active open paths including:
	- `sound\\emitters\\Emitter_Stormwind_BehindtheGate_03.wav`
	- `Shaders\\Pixel\\ps_3_0\\Desaturate.bls`
- canonical and Session A docs now include a tighter LIT-status boundary that combines:
	- strict `Model2` extension-gate logic from `FUN_0081c390` / `M2_NormalizeModelPathAndProbeSkins`
	- restarted live open-traffic samples
- current blocker captured in continuity:
	- repeated `DebugRun` pauses in system DLL frames are still disrupting efficient world-path M2 chain harvest, so debugger run-state stabilization remains the immediate prerequisite for full live world-path closure
- validation boundary:
	- documentation/reverse-engineering continuity only
	- no renderer code changes landed in this slice

### Apr 02, 2026 - Win32 subsystem deep-dive notes added for shaders, liquids, particles, lighting, and LIT status

- expanded native Win32 documentation packet and canonical handoff with subsystem-specific anchors:
	- `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/01-runtime-log.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/02-win32-m2-anchor-map.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/03-console-and-render-controls.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/04-next-steps.md`
- new evidence now explicitly maps:
	- rendering/effect ownership (`*.wfx` reload path, shader effect load/cache/bind seams)
	- liquid shader families and DBC-backed material/settings fallback behavior
	- particle dual-path submission (direct vs merged batch) and batch-compatibility constraints
	- world/map lighting seams (`Light*.dbc`, `WLIGHT`, `WCACHELIGHT`) plus debug script-light command path
- LIT status boundary in current Win32 pass:
	- no positive `.lit` or `.LIT` loader/path formatter anchor recovered
	- recovered `Unlit` labels are effect-mode names, not standalone file-family ownership
	- classification remains evidence-bounded and is recorded as unconfirmed/unsupported until positive anchors are found
- validation boundary:
	- documentation and reverse-engineering continuity updates only
	- no renderer code changes landed in this slice

### Apr 01, 2026 - Session A Deep-Capture and Hidden-Path Native Notes Landed

- expanded the Session A packet and canonical native research docs with second-pass Win32 evidence:
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/01-runtime-log.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/02-win32-m2-anchor-map.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/03-console-and-render-controls.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/04-next-steps.md`
	- `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
- new confirmed runtime chain evidence now includes profile select, exact `%02d.skin` path output, init-state transition, callback rebuild hits, and combiner return-handle capture (all currently in UI-model context)
- new hidden-path notes now include:
	- shared Win32 M2 runtime flag-word helpers (`DAT_00d3fcf4` + getter/setter/OR helper)
	- callback-owned bit toggles for doodad batching, particle batching, and additive particle sorting
	- `M2Faster`/`M2FasterDebug` high-bit mode routing and parser caveats
	- likely-dead startup fallback branch (`0x40`) under the normal `M2_RegisterRuntimeFlags` init flow
	- repeated `M2_NormalizeModelPathAndProbeSkins` prewarm chain callsites
- validation boundary:
	- this was documentation and reverse-engineering continuity work only
	- no renderer code changes landed in this slice
	- x64dbg control session timed out and ended (`is_debugging=false`), so world-path runtime captures remain pending until reattach

### Apr 01, 2026 - Adapted M2 Skeletal Animation Re-enabled With Material-Track Guardrails

- landed a renderer-side animation recovery in `src/MdxViewer/Rendering/ModelRenderer.cs`:
	- adapted M2 models now create/use `MdxAnimator` again by default (gateable with `PARP_M2_ENABLE_ANIMATION=0`)
	- GPU bone upload now runs for adapted M2 when enabled
	- vertex shader skinning now clamps bone indices to `0..127` and normalizes weight sums before matrix blend
- intentionally kept high-risk animation channels suppressed for adapted M2 while visibility recovery continues:
	- material alpha/color tracks remain static for M2 path
	- geoset animation alpha overrides remain disabled for M2 path
	- UV animation transforms remain disabled for M2 path
- this keeps skeleton motion online without reopening the known transparency-driven invisibility seam.
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 01, 2026 with existing warnings only
- validation boundary:
	- no runtime manual signoff captured yet in this slice for world doodad animation parity
	- this is a staged motion-recovery pass, not full M2 animation/material parity

### Apr 01, 2026 - M2 Visibility Hotfix Targets Shader Alpha Path With Animator Disabled

- landed a narrow renderer change in `src/MdxViewer/Rendering/ModelRenderer.cs` focused on the active M2 invisible-geometry seam:
	- when `_isM2AdapterModel` is true and animator is disabled, `EvaluateLayerAlpha(...)` now uses `StaticAlpha` only and does not multiply by `StaticColorAlpha`
	- added `PARP_M2_FORCE_SOLID=1` diagnostic mode to force adapted M2 geosets through an untextured solid-color shader path (opaque pass) for hard geometry-vs-material isolation
- rationale:
	- adapted M2 layers can carry static color-alpha metadata that evaluates to zero at frame 0
	- with animator suppressed, that value was still reaching shader uniform `uColor.a`, making all submitted geometry fully transparent even when vertices/indices were valid
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 01, 2026 with existing workspace warnings only
- validation boundary:
	- no manual world-scene runtime signoff captured in this slice yet
	- this is a targeted visibility hotfix and not full M2 material/animation parity closure

### Apr 01, 2026 - Added Headless `--probe-m2-adapter` Triage Mode For Phase 1 M2 Visibility Diagnostics

- landed a new non-UI probe entrypoint in `src/MdxViewer/AssetProbe.cs`:
	- `--probe-m2-adapter <gamePath> <modelVirtualPath> [--build <version>] [--skin <virtualPath>] [--listfile <path>]`
	- alias: `--probe-m2`
- probe behavior now explicitly targets Phase 1 investigation evidence without requiring interactive viewer rendering:
	- loads model bytes from MPQ/loose game roots
	- validates build/profile compatibility through `FormatProfileRegistry` and `WarcraftNetM2Adapter.ValidateModelProfile(...)`
	- tries companion skin candidates (or forced `--skin`) through `WarcraftNetM2Adapter.BuildRuntimeModel(...)`
	- prints renderer-equivalent geoset outcomes as `[M2-DIAG-CPU]`: total geosets, valid, index-rejected, empty-skipped
	- preserves adapter per-geoset logs already emitted as `[M2-ADAPT]`
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Apr 01, 2026 with existing workspace warnings only
	- command wiring was exercised through `dotnet run ... -- --probe-m2-adapter ...`
- validation boundary:
	- no successful real 3.3.5 M2+skin probe was captured in this slice because the attempted in-repo `0.6.0` testdata run did not contain the chosen 3.3.5 UI-model path
	- this slice proves command availability and compile integration, not M2 visibility closure

### Apr 01, 2026 - Fresh A/B Session A Investigation Packet Started (Stormwind Runtime)

- created a new clean-room documentation packet for A/B analysis under:
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/README.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/01-runtime-log.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/02-win32-m2-anchor-map.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/03-console-and-render-controls.md`
	- `wow-viewer/docs/architecture/ab-session-a-2026-04-01/04-next-steps.md`
- this packet is intentionally framed as a fresh-session baseline for later A/B comparison and does not rely on prior-session conclusions as the primary evidence source
- static/decompilation outputs captured in this slice:
	- confirmed Win32 M2 anchors for skin/profile choose-load-init and combiner family selection
	- recovered world/terrain render command registration and high-value console toggles (`showCull`, `showLowDetail`, `showSimpleDoodads`, `detailDoodadAlpha`, `terrainAlphaBitDepth`, plus M2 runtime flags)
- runtime status in this slice:
	- x64dbg breakpoints were set on the M2 anchors
	- confirmed live anchor hits captured after restart:
		- `0x0083cc80` (`M2_ChooseAndLoadSkinProfile`)
		- `0x00835a80` (`M2_FormatSkinFilename_02d`)
		- `0x00838490` (`M2_InitializeSkinProfileAndRebuildInstances`)
		- `0x00836600` (`M2_BuildCombinerEffectName`)
	- captured model path in those first hits is a UI model (`interface\\glues\\models\\ui_mainmenu_northrend\\ui_mainmenu_northrend.m2`), so world-path capture is still pending
- validation boundary:
	- this slice delivered documentation and anchor setup only
	- no renderer or adapter parity fix is claimed yet

### Apr 01, 2026 - M2 Investigation Tooling Boundary Updated To Offline Ghidra + x64dbg-mcp

- updated `.github/prompts/m2-rendering-investigation.prompt.md` to remove the old live-Ghidra requirement
- Phase 2 now explicitly uses:
	- offline static analysis in Ghidra against `WoW.exe` (3.3.5.12340)
	- live runtime debugging in x64dbg through `x64dbg-mcp`
- the prompt now requires Ghidra-mapped Win32 targets to be validated dynamically with x64dbg breakpoints/watchpoints before claiming parity conclusions
- continuity intent:
	- keep native reverse-engineering evidence grounded in an executable workflow that is actually available in this environment
	- continue recording both static and runtime findings in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
- validation boundary:
	- this slice updates workflow guidance only
	- no new renderer/adapter runtime fix was claimed from this change

### Apr 01, 2026 - Win32 3.3.5.12340 M2 Runtime Anchors Mapped For x64dbg

- mapped and renamed concrete Win32 M2 functions in the loaded `WoW.exe` for immediate `x64dbg-mcp` breakpoint usage:
	- `0x0083cc80` `M2_ChooseAndLoadSkinProfile`
	- `0x00838490` `M2_InitializeSkinProfileAndRebuildInstances`
	- `0x00836600` `M2_BuildCombinerEffectName`
	- `0x00835a80` `M2_FormatSkinFilename_02d`
	- `0x00835a20` `M2_FormatAnimFilename_04d_02d`
	- `0x00402760` `M2_RegisterRuntimeFlags`
	- `0x0053c430` `M2_NormalizeModelPathAndProbeSkins`
- source evidence came from direct string-xref anchors in offline Ghidra (`%02d.skin`, `%04d-%02d.anim`, `Combiners_Opaque`, `Diffuse_T1`, `M2UseZFill`, `CM2Model`)
- recorded these anchors in `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` under a dedicated Win32 breakpoint section
- validation boundary:
	- this slice establishes native-analysis anchors only
	- no runtime breakpoint-hit capture or renderer-fix claim yet

### Mar 31, 2026 - wow-viewer M2 Foundation Slice 01 Implemented

- implemented the first narrow `wow-viewer`-owned M2 seam rather than leaving slice 01 as planning-only work
- landed code:
	- `WowViewer.Core/M2` model identity, model document, skin document, submesh, batch, and profile-selection contracts
	- `WowViewer.Core.IO/M2` strict `MD20` and `SKIN` readers
	- `WowViewer.Core.Runtime/M2` choose/load/initialize skin-profile state
	- `WowViewer.Tool.Inspect` `m2 inspect` command for local-path or archive-backed model inspection
	- `WowViewer.Core.Tests/M2FoundationTests` coverage for identity normalization, strict root checks, strict skin parsing, and runtime-stage transitions
- validation:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
	- all `234` tests passed in the current wow-viewer solution run
- boundary:
	- this is not viewer-runtime parity or active `MdxViewer` signoff
	- no in-repo extracted real `.m2` / `.skin` asset was available, so this landing proves library/build/test behavior plus inspect ownership only

### Mar 31, 2026 - Ordered wow-viewer M2 Runtime Prompt Set Landed

- added the missing workflow surface for M2 runtime and renderer recovery so future chats stop mixing parser ownership, skin-state recovery, material routing, lighting, batching, and compatibility-only `MdxViewer` fixes in one prompt
- landed assets:
	- `.github/prompts/wow-viewer-m2-runtime-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/01-md20-and-skin-runtime-foundation.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/02-section-classification-and-material-routing.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/03-animation-lighting-and-effect-runtime.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/04-scene-submission-and-batching.prompt.md`
	- `.github/prompts/wow-viewer-m2-runtime/05-consumer-cutover-and-parity-harness.prompt.md`
	- matching Codex prompt mirrors plus `gillijimproject_refactor/plans/wow_viewer_m2_runtime_plan_2026-03-31.md`
- routing effect:
	- M2 runtime ownership, exact `%02d.skin` behavior, section/material routing, animation/lighting state, and scene batching now have a dedicated staged prompt set in the same style as PM4/shared-I/O/world-runtime work
	- broader `WorldScene` extraction and repeated asset-miss suppression still belong to the separate world-runtime prompt family
- validation boundary:
	- this entry was workflow/continuity work only when it landed
	- slice 01 has since landed as a separate implementation step; keep using this entry for prompt-routing history, not current implementation status

### Mar 31, 2026 - Conservative Adapted-M2 Material Rollback Restored A Sane Giant-Root Payload

- followed stronger runtime evidence from the standalone viewer: `AzjolRoofGiant.m2` still loaded as a selectable adapted/runtime model but rendered no visible geometry even outside the world-scene path
- found a concrete regression seam in `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`:
	- the current adapted M2 path was applying `ApplyLayerAnimationMetadata(...)` to each generated material layer
	- that is newer behavior than the old conservative path and can zero final layer alpha through transparency/color animation tracks even for static doodads
- landed a narrow rollback:
	- adapted M2 materials now stay on the conservative path again and do not graft raw layer transparency/color/UV animation metadata into runtime layers
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed on Mar 31, 2026 with existing warnings only
	- direct probe validation using the built viewer binary now reports `AzjolRoofGiant.m2` as a sane opaque adapted model with `574` vertices, `1063` triangles, `AZULBLANKROCK.BLP`, and `StaticAlpha=1.000`
- validation boundary:
	- no automated tests were added or run
	- live viewer runtime signoff is still pending until the rebuilt app is reopened on the standalone giant-root asset and the development world

### Mar 31, 2026 - Remaining Invisible World M2 Set Is Now Scoped To The Shaded Pass, Not Placement

- followed the first live viewer report after the build-mismatch correction: many MPQ-backed M2s are still missing, especially the giant root structures that should cover the development terrain
- the newest runtime evidence changed the problem statement again:
	- object tooltip selection still resolves those missing models
	- world bounding-box overlays still show their placements and extents in the expected locations
	- this means scene registration and bounds are alive, but does not yet prove the shaded geoset draw path is succeeding for those instances
- current best reading:
	- the stale-build fix remains valid because it explained and reproduced one real failure mode
	- the remaining blocker is now a narrower render-path problem in the active viewer for adapted M2 shading/submission/material state
- next investigation target is explicit:
	- verify whether those root models reach `ModelRenderer.RenderGeosets(...)` in opaque and transparent passes
	- if they do, compare their layer family / blend routing against visible adapted M2s
	- if they do not, add temporary runtime diagnostics or forced solid-color rendering so the viewer can separate geometry submission failure from texture/material invisibility
- validation boundary:
	- no new fix landed for this remaining seam yet
	- no runtime signoff should be claimed from the build-mismatch correction alone

### Mar 31, 2026 - M2 Loads Now Override Stale Build Selection With The Real Client Build

- followed the next concrete blocker after the shared renderer fixes: `AzjolRoofGiant.m2` still showed as invisible in the viewer, but a new headless probe proved the real issue was build mismatch rather than necessarily bad adapter extraction
- captured hard evidence with the same asset on the same 3.3.5 client root:
	- `--build 3.3.5.12340` produced sane adapted output (`574` verts / `1063` tris, valid bounds, resolved skin/texture)
	- `--build 3.0.1.8303` produced degenerate output (`1` vert / `1` tri, broken bounds)
- landed a narrow runtime correction in `src/MdxViewer/Terrain/BuildVersionCatalog.cs`, `src/MdxViewer/ViewerApp.cs`, `src/MdxViewer/Terrain/WorldAssetManager.cs`, and `src/MdxViewer/Rendering/WmoRenderer.cs`:
	- M2-family loads now prefer the build inferred from the actual game/client path over a stale selected build when those disagree
	- standalone M2 open, world M2 loading, and WMO doodad M2 loading all use that effective build for adapter and converter fallback paths
- validation completed:
	- `get_errors` returned clean for the edited files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed with existing warnings only
- validation boundary:
	- no real viewer runtime confirmation has been captured yet
	- the next check is to reopen the failing standalone asset and the development map under the saved 3.3.5 client and confirm the previously invisible MPQ-backed M2s now render

### Mar 31, 2026 - World M2 Doodads Now Use Per-Instance Rendering In WorldScene Again

- followed the next active runtime blocker after slice 01: user reported fewer repeated asset hiccups, but many world M2s were still invisible even though hover/picking showed the objects existed
- landed a narrow world render-path correction in `src/MdxViewer/Terrain/WorldScene.cs`:
	- M2-adapted world doodads now use `RenderWithTransform(...)` again in both opaque and transparent passes instead of the shared batched `RenderInstance(...)` path
	- classic MDX doodads still stay batched
	- batch initialization now comes from the first actually batched renderer rather than the first visible renderer overall
- why this matters:
	- the active batch/unbatch gate only covered particle/ribbon cases, so adapted world M2s were still falling onto the generic instanced path despite earlier continuity pointing at that path as the likely invisible-model seam
	- this keeps the fix narrow in the active viewer without reopening the broader renderer split yet
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime confirmation has been captured yet, so remaining WMO/M2 hiccups are still open until the viewer is exercised on the development map

### Mar 31, 2026 - Missing Base Textures No Longer Make Adapted M2s Fully Invisible

- followed the next user-visible M2 blocker after the world-scene submission fix: more objects now appear, but another set of MPQ-backed M2s still stays invisible in both world view and standalone model view
- landed a shared renderer-path correction in `src/MdxViewer/Rendering/ModelRenderer.cs`:
	- adapted M2s now treat a missing base-layer texture as a neutral fallback-texture case instead of letting the whole geoset render path disappear
	- the renderer also no longer suppresses the normal fallback-geoset draw merely because an adapted/pre-release layer missed texture resolution
- why this matters:
	- the shared `ModelRenderer` path is used by both world placements and standalone model viewing, so this fix targets the common “loaded but fully invisible” symptom directly
	- it keeps the next investigation honest: if some MPQ M2s are still malformed after this, the likely remaining bug is in adapter skin/submesh/material extraction rather than another world-scene visibility seam
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Rendering/ModelRenderer.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug --no-restore` passed with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime confirmation has been captured yet, so any remaining partial or malformed MPQ M2s are still open

### Mar 31, 2026 - First Negative Asset Lookup Suppression Slice Landed In MdxViewer

- implemented the ordered world-runtime slice 01 in the active viewer path instead of starting the broader service extraction early
- landed behavior in `src/MdxViewer/Terrain/WorldAssetManager.cs`, `src/MdxViewer/Rendering/WmoRenderer.cs`, `src/MdxViewer/ViewerApp.cs`, and `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- cached failed MDX world loads now stay failed for the current session rather than being retried through lazy load, queue, and deferred drain paths
	- known-missing external `.skin` results now suppress repeated world-path prefetch fanout and repeated companion-skin miss logs
	- standalone M2 open and WMO doodad M2 load paths now also log the same missing `.skin` problem once per resolved model path instead of flooding repeats
	- the terrain sidebar now reports suppressed failed-MDX retries plus known/suppressed skin-miss telemetry so the miss path is visible without reading raw logs
- why this matters:
	- this removes one concrete source of repeated asset-miss noise before later visibility/pass extraction slices
	- it keeps the slice narrow: no broad asset-service rewrite, no pass ownership move, and no new renderer abstraction was introduced here
- validation completed:
	- `get_errors` returned clean for the edited files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime capture or fixed-shot smoke was run in this slice, so measured log/frame improvement on the development map is still pending

### Mar 31, 2026 - Ordered wow-viewer World Runtime Prompt Set Landed

- user selected the staged world-runtime decomposition path instead of a one-off next extraction only
- added a dedicated Copilot workflow surface for fresh implementation chats:
	- `.github/prompts/wow-viewer-world-runtime-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-world-runtime/01-negative-asset-lookup-suppression.prompt.md`
	- `.github/prompts/wow-viewer-world-runtime/02-visible-set-runtime-extraction.prompt.md`
	- `.github/prompts/wow-viewer-world-runtime/03-world-pass-service-extraction.prompt.md`
	- `.github/prompts/wow-viewer-world-runtime/04-world-scene-host-thinning.prompt.md`
	- `.github/prompts/wow-viewer-world-runtime/05-wow-viewer-app-runtime-consumer.prompt.md`
	- `gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md`
- why this matters:
	- the next chats now have one ordered path for the `WorldScene` split instead of rediscovering the sequence every time
	- the first slice explicitly targets repeated `.skin` miss churn and failed MDX retry noise before deeper pass extraction, which should improve measurement quality and reduce obvious hidden runtime waste
- validation boundary:
	- this step only created workflow and continuity assets
	- no code fix for the `.skin` retry issue landed yet in this step
### Mar 31, 2026 - First WorldScene Seam Extracted Into wow-viewer Core.Runtime

- followed the new architectural direction to split `WorldScene.cs` by moving the first stable slice into `wow-viewer` instead of performing another app-local refactor only
- landed the first shared runtime seam across `wow-viewer` and `MdxViewer`:
	- added `WowViewer.Core.Runtime.WorldRenderStageStats`, `WorldRenderFrameStats`, and `WorldRenderOptimizationAdvisor`
	- moved `WorldScene` to consume those runtime-owned contracts instead of keeping the public telemetry surface embedded in the app project
	- added xUnit coverage for empty-frame, MDX-dominant, and terrain-dominant optimization hints in `wow-viewer/tests/WowViewer.Core.Tests/WorldRenderOptimizationAdvisorTests.cs`
	- added `WowViewer.Core.Runtime` project references to the active `MdxViewer` consumers so the extracted seam is compile-proven in the legacy app
- why this matters:
	- this establishes `wow-viewer` as the canonical owner of the first reusable world-render contract rather than leaving the seam trapped in `WorldScene`
	- it creates a concrete next extraction path for visibility, pass ownership, and scene composition without overstating that the renderer itself has already moved
- validation completed:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with 226 tests succeeding
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings only
- validation boundary:
	- no runtime viewer signoff was performed
	- no render behavior changed intentionally in this slice beyond where the telemetry contract is owned

### Mar 31, 2026 - First Renderer-Stats Slice Landed In WorldScene And The Sidebar

- followed the renderer-first roadmap by implementing the first measurement slice instead of jumping straight into another broad renderer rewrite
- landed the change in `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- added a reusable per-frame render contract in `WorldScene` that now owns visible WMO/MDX scratch lists, transparent-sort scratch, stage timings, and MDX batched-vs-unbatched counts
	- the active world frame now records timings for deferred asset drain, taxi actor update, lighting, sky, skybox backdrop, WDL, terrain, WMO visibility, WMO submission, MDX animation, MDX visibility, MDX opaque submission, liquids, MDX transparent sort, MDX transparent submission, and the late overlay/debug block
	- the terrain sidebar now exposes a `Renderer Stats` tree showing the last captured world-frame CPU timings and a heuristic `next win` hint based on those numbers
- why this matters:
	- this is the first active `WorldScene` seam that turns the renderer-performance roadmap into runtime data instead of only continuity notes
	- it also establishes the smallest viable world render-frame contract needed for the next batching/culling slice without pretending the full render-layer refactor is already done
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime capture was performed yet in this chat, so the new stats panel and its next-win hint still need manual confirmation on the development map

### Mar 31, 2026 - Renderer-First Performance Roadmap Recorded For The Active MdxViewer World Path

- followed the direct reprioritization that camera-movement performance is now the biggest blocker, ahead of more shell tweaks or more isolated feature additions
- recorded the active renderer roadmap in `gillijimproject_refactor/plans/mdxviewer_renderer_performance_plan_2026-03-31.md`
- plan decisions locked for future slices:
	- work the active `src/MdxViewer/Terrain/WorldScene.cs` path first instead of treating dormant `RenderQueue.cs` as if it already owned the frame
	- start with per-frame instrumentation plus an explicit world render-frame contract
	- then reduce MDX submission churn and batching waste
	- then pull WMO shell/liquid/transparent ownership outward from renderer-local sequencing into clearer scene-level layers
	- keep PM4/debug/editor overlays as explicit late layers instead of letting them stay mixed into the main world submission cost
	- finish DBC lighting integration after render-layer ownership is explicit
	- add graveyards from `WorldSafeLocs.dbc` only after the renderer frame is stabilized, reusing the Area POI / taxi lazy-load overlay model
- validation boundary:
	- this slice is planning only
	- no automated tests were added or run
	- no runtime performance measurements were captured yet from the new plan itself

### Mar 31, 2026 - Fixed Sidebar Shell Now Uses Draggable Split Panels

- followed direct viewer-shell feedback that the current fixed sidebars were still not meaningfully resizable and felt like a broken layout mode
- landed shell changes in `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- fixed mode now renders explicit draggable left/right splitter bars instead of relying on hidden ImGui window-border resize behavior
	- left and right panels stay edge-anchored while splitter drag updates the stored sidebar widths directly
	- fixed panels now opt into `NoResize` because the supported resize path is the splitter itself, not window-border grabbing
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Sidebars.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime signoff has been completed yet for the new fixed-panel splitter behavior

### Mar 31, 2026 - Mouse Camera Input Restored After Narrowing The Fixed-Sidebar Splitter Windows

- followed direct runtime feedback that mouse camera control stopped working after the fixed-sidebar splitter shell landed
- root cause was the splitter host itself in `src/MdxViewer/ViewerApp_Sidebars.cs`:
	- the first splitter implementation used one transparent full-width window over the whole viewport height
	- that window could still cause ImGui mouse capture outside the actual splitter bars, which interfered with scene camera input
- landed fix:
	- replaced the full-width splitter host with narrow splitter-only windows for the left/right drag handles
	- only the actual splitter strips now capture mouse input, leaving the rest of the viewport available to the normal camera path again
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp_Sidebars.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime signoff has been completed yet for restored mouse-look behavior

### Mar 31, 2026 - Hover Tooltip Toggle And UniqueId Archaeology Filter Landed In MdxViewer

- followed the latest direct viewer workflow request to make scene exploration less noisy and more controllable:
	- hover cards needed an explicit disable path
	- object layers needed a `UniqueId`-based scrubber for digital archaeology across either the whole map or the current camera tile
- landed behavior in `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/Terrain/WorldScene.cs`:
	- added a `Hover Tooltips` checkbox that suppresses scene hover overlay rendering without removing the hover metadata pipeline itself
	- replaced the first cutoff-only archaeology filter with explicit min/max range semantics so the viewer can hide placements within a chosen `UniqueId` span
	- propagated tile coordinates onto flattened scene object instances so camera-tile filtering works for terrain-loaded and external spawn objects
	- applied the hide filter to render submission, hover hit testing, scene picking, and bounding-box debug drawing so hidden ranges behave consistently
	- added gap-based archaeology layer detection for the active scope plus a viewer table that lists detected layers with range, count, WMO/M2 breakdown, and one-click hide actions
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime signoff has been completed yet for the new archaeology controls on the development map

### Mar 31, 2026 - User Fog Range Control Restored Over Zone Lighting

- followed direct runtime feedback that fog could no longer be effectively removed and that the older farther-view behavior had regressed
- root cause was the Mar 30 shared-lighting change letting `LightService` overwrite `TerrainLighting.FogStart` and `FogEnd` every frame while zone lighting was active
- landed fix in `src/MdxViewer/Terrain/TerrainLighting.cs` and `src/MdxViewer/Terrain/WorldScene.cs`:
	- external lighting override now applies only directional light, ambient light, and fog color
	- live fog distance remains on the user-controlled `TerrainLighting` values, so terrain sidebar fog sliders and far visibility budget work again
- also fixed compile blockers still present in the current `WorldScene` hover helpers so the viewer solution could be revalidated cleanly
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/TerrainLighting.cs` and `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime signoff has been completed yet for the restored no-fog / farther-view behavior

### Mar 31, 2026 - VLM Dataset Reconstruction Planning Reset Landed

- followed a direct request to stop treating VLM work as a vague training exercise and define how a v7-like missing-layer reconstruction model should be grounded in real map data
- confirmed the current exporter boundary before planning:
	- `WoWMapConverter.Core.VLM.VlmDatasetExporter` already exports chunk heights, local/global heightmaps, normals, MCCV, raw shadow bits, derived shadow analysis, alpha masks, liquids, objects, WDL data, and binary tile output
	- the older docs under `docs/VLM_Training_Guide.md` and `docs/VLM_DATASET_EXPORTER.md` do not fully describe that active schema
- landed new planning surfaces:
	- `gillijimproject_refactor/plans/vlm_dataset_reconstruction_plan_2026-03-31.md`
	- `.github/prompts/vlm-dataset-reconstruction-plan.prompt.md`
	- updated `.github/prompts/development-repair-implementation-plan.prompt.md` so future chats route VLM dataset/model asks away from the repair-pipeline prompt
- planning decision recorded in continuity:
	- `development` is now explicitly the reconstruction target/evaluation corpus, not the only teacher corpus
	- the next dataset slice should be manifest/provenance/completeness classification and curation across real exported maps before additional model work
- validation boundary:
	- no exporter code behavior changed in this slice
	- no automated tests were added or run
	- no real-data export, curation, or training command was executed in this slice

### Mar 31, 2026 - Terrain WDT Global WMO Parsing Fixed For ADT Maps; M2 UV Regression First Mitigation Landed

- followed direct runtime bug reports after the corrupted-chat continuity recovery:
	- terrain maps were missing WDT-level global WMO placements that should render roof or shell geometry over ADT terrain
	- M2s still had active material regressions, including oversized leaf-like detail doodads and inconsistent transparency behavior
- landed terrain fix in `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`:
	- WDT global `MWMO` or `MODF` parsing now triggers for terrain maps when the file carries the historical `MPHD` global-map-object flag or plainly contains both chunks, not only for `IsWmoBased` maps
	- terrain-map WDT placements now convert into renderer coordinates like ADT `MODF` placements instead of staying in raw WMO-map coordinates
- landed first M2 mitigation in `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`:
	- negative resolved texture coord ids now fall back to UV0
	- negative coord ids no longer imply `SphereEnvMap`
	- this intentionally moves current behavior back toward the older known-good adapter path while preserving explicit positive UV-set selection
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run for these viewer regressions
	- no development-map runtime signoff has been completed yet for either the restored WDT global WMO path or the current M2 adapter mitigation

### Mar 31, 2026 - Active Tree-Trunk M2 Regression Trimmed Back By Restoring The Conservative Per-Section Material Path

- followed direct runtime feedback that the remaining M2 regression still made some trees appear to be made of leaves with no trunks
- landed a narrower compatibility fix in `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`:
	- restored the old stable runtime behavior of taking the first batch/material per section
	- forced the active runtime material path back to `UV0` for that conservative section material path
	- this intentionally backs away from the newer richer batch/layer interpretation until it can be proven against real viewer output
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 31, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime signoff has been completed yet for restored tree-trunk rendering on the development map

### Mar 30, 2026 - WorldScene WMO Submission Now Uses A Visible-Instance Bucket

- followed the renderer-performance continuity work after the shared LightService or TerrainLighting lighting fix
- landed a narrow structural slice in `src/MdxViewer/Terrain/WorldScene.cs`:
	- added a reusable visible-WMO scratch contract so world-scene WMO instances are culled once per frame before submission instead of recomputing cull decisions inline inside the WMO draw loop
	- this brings the WMO path closer to the existing visible-MDX path and creates a cleaner seam for a future split between opaque shell, liquids, doodad transparent, and transparent shell passes
	- current user-visible behavior should stay the same because `WmoRenderer.RenderWithTransform(...)` still owns the actual WMO-local pass order
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run for this renderer-structure slice
	- no development-map runtime signoff has been completed yet for the visible-WMO submission path

### Mar 30, 2026 - PM4 UI Glossary Clarified Viewer-Derived `part` Labels

- followed direct user feedback that the PM4 inspector had become too opaque to trust for day-to-day use
- landed clarification across `src/MdxViewer/ViewerApp_Pm4Utilities.cs`, `src/MdxViewer/ViewerApp.cs`, `src/MdxViewer/Terrain/WorldScene.cs`, and `src/MdxViewer/README.md`:
	- the PM4 workbench now includes a glossary/evidence block explaining which labels are raw PM4 chunk names, viewer aliases, or viewer-generated structure
	- `part` / `ObjectPartId` is now explicitly described as a viewer-generated split id from the current overlay build, not a raw PM4 field on disk
	- selected-object and graph text now repeat that distinction where the user actually sees `part`
- validation boundary:
	- no automated tests were added or run
	- no real-data runtime validation was performed for this terminology-only clarification slice

### Mar 30, 2026 - DBC LightService Now Drives One Shared World Lighting State

- followed user direction to stop deferring renderer correctness work while the world path is still slow and visually inconsistent
- landed a first lighting-correctness slice in `src/MdxViewer/Terrain/TerrainLighting.cs` and `Terrain/WorldScene.cs`:
	- `TerrainLighting` can now take an external per-frame lighting override for direct light, ambient light, fog color, and fog range
	- when `LightService` has an active zone, `WorldScene.Render(...)` now maps `LightService.TimeOfDay` into the shared terrain lighting clock, applies the DBC-driven colors/fog override, and updates that shared state before rendering skybackdrops, WDL, terrain, liquids, WMOs, or MDXs
	- when no zone is active, rendering falls back to the old procedural `TerrainLighting` path cleanly
- why this matters:
	- before this slice, one frame could mix `Light.dbc` sky/fog with fallback terrain/object light colors, so lighting parity was already broken even before the larger render-layer work
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/TerrainLighting.cs`, `src/MdxViewer/Terrain/WorldScene.cs`, and the associated capture-fix file `src/MdxViewer/ViewerApp_CaptureAutomation.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- validation boundary:
	- no automated tests were added or run for this viewer-side lighting correction
	- no development-map runtime signoff has been completed yet for the shared LightService or TerrainLighting path

### Mar 30, 2026 - v0.4.6.1 Release Prep Added PM4 Tooltip-Focused Notes And Beginner UI Guidance

- followed release request for `v0.4.6.1` with emphasis on clearer PM4 WoW-styled tooltip display messaging and reduced first-run confusion
- landed behavior/docs/workflow updates:
	- bumped `src/MdxViewer/MdxViewer.csproj` version metadata to `0.4.6.1`
	- updated welcome/status UI wording so users are directed to open a base game path first (`File > Open Game Folder (MPQ)`) instead of defaulting to standalone file usage
	- refreshed `src/MdxViewer/README.md` and `gillijimproject_refactor/README.md` with `0.4.6.1` snapshot, explicit beginner flow, and conservative support-range language
	- added `src/MdxViewer/docs/ui-screenshot-guide.md` and `src/MdxViewer/docs/screenshots/README.md` to standardize screenshot capture/drop workflow for README/release image selection
	- updated release-note body and quick-start text in both release workflows so GitHub release output matches the new onboarding and PM4 tooltip emphasis
- validation boundary:
	- no automated tests were added or run
	- this slice is build/docs/workflow prep and still needs runtime screenshot curation for final README hero-image selection

### Mar 30, 2026 - PM4 Workbench Tab Flicker/Snapback Removed

- followed user runtime feedback that `Selection` and `Correlation` in the PM4 workbench were only visible for a split second and then dropped out
- landed behavior:
	- added `_pendingPm4WorkbenchTab` one-shot tab focus state in `src/MdxViewer/ViewerApp.cs`
	- `OpenPm4Workbench(...)` now sets one-shot pending tab focus instead of continuously forcing the live tab state
	- `src/MdxViewer/ViewerApp_Pm4Utilities.cs` now uses non-closable tab items and applies `SetSelected` only when there is a pending tab request
	- pending tab request is cleared after tab-bar draw, so manual tab clicks persist across frames
- documentation update landed in the same slice:
	- `src/MdxViewer/README.md` now explicitly documents fixed sidebars as startup default and records the missing screenshot-guide follow-up for key workflows
- runtime boundary:
	- no automated tests were added or run for this UI-state fix
	- no live viewer/runtime signoff has been completed yet for tab persistence in the PM4 workbench

### Mar 30, 2026 - PM4 Sidebar Tabs Restored And Hover Info Hit Testing Narrowed

- followed direct user runtime feedback that the PM4 sidebar workflow had regressed badly after the inspector-first shell change:
	- `Overlay`, `Selection`, and `Correlation` felt like dead tabs because the workbench section was effectively missing
	- sidebar match-save flows were blocked because PM4 selection could still yield to normal scene picks
	- `WL*` hover/info was too hard to reach near PM4 content and the hover radius felt too broad
- landed behavior:
	- `ViewerApp_Pm4Utilities.OpenPm4Workbench(...)` now forces the right inspector open for PM4 workbench requests
	- `ViewerApp_Sidebars.DrawRightSidebar()` now renders `PM4 Workbench` whenever a world scene exists instead of hiding it until overlay or selection state already exists
	- `ViewerApp.PickObjectAtMouse(...)` now prefers the hovered PM4 object before regular scene-hit arbitration and uses the same preference for `Shift + Left Click` collection adds
	- `WorldScene` now separates hover-info hit testing from the larger wireframe-reveal brush, using a tighter screen-space brush for `WMO`, `MDX`, `WL*`, and PM4 hover cards while preserving the broader reveal brush for overlay visibility
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/ViewerApp.cs`, `src/MdxViewer/ViewerApp_Pm4Utilities.cs`, `src/MdxViewer/ViewerApp_Sidebars.cs`, and `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- runtime boundary:
	- no automated tests were added or run for this viewer-side interaction fix
	- no live viewer/runtime signoff has been completed yet for restored PM4 sidebar behavior or WL hover reachability on the development map

### Mar 30, 2026 - PM4 Camera-Window Load Regression Reduced By Removing A Redundant Zero-CK24 Link Rescan

- followed the user runtime report that PM4 overlay loads were stalling around `1/12` or `1/15` camera-window files and effectively never finishing
- root cause found in `src/MdxViewer/Terrain/WorldScene.cs`:
	- the Mar 30 zero-`CK24` regrouping follow-up added `SplitZeroCk24SeedGroup(...)`
	- that path did an extra `MSLK` grouping pass and then re-scanned each returned group again to decide whether to preserve it or connectivity-split it
	- on large zero-`CK24` seed groups this added avoidable whole-link rescans to an already expensive PM4 object assembly path
- landed fix:
	- added shared `TryPartitionSurfaceGroupByMslk(...)` so both `SplitSurfaceGroupByMslk(...)` and `SplitZeroCk24SeedGroup(...)` reuse one partition result
	- zero-`CK24` handling now keeps linked `MSLK` families intact and only connectivity-splits the true unlinked remainder, without rebuilding the same grouping state again
- validation completed:
	- `get_errors` returned clean for `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- runtime boundary:
	- no automated tests were added or run for this viewer-side PM4 load path
	- no live viewer/runtime signoff has been completed yet on the fixed development map workflow

### Mar 30, 2026 - PM4 Unknowns Family Expansion Landed For MSLK And MSUR Attribution

- extended `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4ResearchUnknownsAnalyzer.cs` and the shared unknowns report contracts so `pm4 unknowns` now emits family-level attribution summaries instead of only field distributions and edge-fit counts
- landed new report sections for:
	- dominant `MSLK` families grouped by `TypeFlags` or `Subtype` or `SystemFlag`
	- dominant `MSUR` families grouped by `AttributeMask` or `GroupKey` or `IndexCount`
	- per-family linkage signals against direct `MSUR` fits, `MPRL` fits, `LinkId` patterns, `GroupObjectId -> MPRL.Unk04`, `CK24`, `MDOS`, and incoming-link fanout
- fixed-corpus result on `gillijimproject_refactor/test_data/development/World/Maps/development` now gives sharper evidence for where to dig next:
	- dominant `MSLK` families are sentinel-tile-link heavy and concentrated in a small repeated set of `TypeFlags` or `Subtype` combinations
	- dominant `MSUR` families split between large zero-`CK24` umbrella families and non-zero-`CK24` object-facing families with much broader `CK24` and `MDOS` fanout
	- this strengthens the current interpretation that some `group=3` zero-`CK24` families are umbrella/root-style surfaces, while several `group=18` families are better candidates for object-facing attribution analysis
- validation completed:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 30, 2026
	- `dotnet i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/bin/Debug/net10.0/WowViewer.Tool.Inspect.dll pm4 unknowns --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/wow-viewer/output/pm4_unknowns_development_report.json` passed on Mar 30, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --filter UnknownsDirectory_DevelopmentCorpus_ProducesExpectedHighLevelSignals` passed on Mar 30, 2026
- interpretation boundary:
	- this still does not close the actual names or bit-level semantics of `MSLK.TypeFlags` or `Subtype` or `MSUR.AttributeMask` or `GroupKey`
	- it does give a much better corpus-scale target list for the next outlier or family-specific investigation

### Mar 30, 2026 - PM4 MSHD Corpus Analyzer And Inspect Verb Landed

- extended `wow-viewer/src/core/WowViewer.Core.PM4` with a dedicated `Pm4ResearchMshdAnalyzer` plus reusable MSHD report models so `MSHD` can be measured against actual PM4 chunk-family and grouping metrics instead of guessed from one-off tiles
- added a new Tool.Inspect verb:
	- `pm4 mshd --input <directory> [--output <report.json>]`
- fixed-corpus result on `gillijimproject_refactor/test_data/development/World/Maps/development` currently weakens the specific theory that `MSHD` directly stores active root-group or type-bucket counts:
	- `616` files scanned, `502` with `MSHD`
	- `Field0C..Field1C` are zero in all `502` sampled headers
	- `Field00 == Field08` in only `233/502` files
	- no measured field produced the kind of strong exact-match or high-correlation signal that would directly tie it to current `MSLK` or `MSUR` or `MPRL` grouping counts in this corpus slice
- validation completed:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 30, 2026
	- `dotnet i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/bin/Debug/net10.0/WowViewer.Tool.Inspect.dll pm4 mshd --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/wow-viewer/output/pm4_mshd_development_report.json` passed on Mar 30, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --filter MshdDirectory_DevelopmentCorpus_PreservesCurrentTrailingZeroAndWeakBucketSignals` passed on Mar 30, 2026
- interpretation boundary:
	- `MSHD` is still unresolved semantically
	- this slice rules against one specific bucket-count interpretation for the current development corpus; it does not prove the surviving fields are inert or final placeholders in every PM4 family

### Mar 30, 2026 - PM4 Workbench Moved Into The Inspector And Startup PM4 Noise Reduced

- followed the user's shell-overhaul request instead of adding another isolated PM4 panel:
	- PM4 bounds now start disabled
	- PM4 x-ray now starts disabled
	- fixed sidebars are now the default startup shell mode so the inspector no longer drifts by default
	- `World Objects` now keeps only a light PM4 summary plus a `PM4 Workbench` entry point
	- the right sidebar now owns the main PM4 workflow through one consolidated workbench surface
	- the hover card now stays shorter and more selection-oriented, pushing the heavy detail into click-time inspection instead of mouse-over spam
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing project warnings only
- runtime boundary:
	- no automated tests were added or run
	- no live viewer/runtime signoff has been completed yet on the development map for the new inspector-first PM4 shell

### Mar 30, 2026 - PM4 Hover Overlay Now Uses A Tooltip-Style Card And Hover-Time Match Preview

- followed the user's request to make the new hover overlay look more like a WoW item tooltip and to expose PM4 potential matches directly from mouse-over
- landed behavior:
	- `WorldScene.UpdateHoveredAssetInfo(...)` now recognizes PM4 overlay objects and returns a PM4 object key for the hovered part when the PM4 overlay is active
	- `ViewerApp.DrawSceneHoverAssetOverlay()` now renders a darker gold-bordered tooltip-style card with brighter title text and stronger path or detail styling
	- hovered PM4 parts now show a compact top-candidate list sourced from a separate hovered-object PM4 match cache, so the overlay can preview likely `WMO` or `M2` matches without changing selection
	- PM4 derived-report invalidation now also clears the hovered-object match cache so tooltip suggestions do not go stale after PM4 reloads or regrouping changes
- continuity updates landed alongside the code change:
	- `wow-viewer/README.md` now records the current `CK24` low-16 versus `CK24=0x000000` research framing
	- `plans/wow_viewer_pm4_library_plan_2026-03-25.md` now records the same note and keeps the hover or graph ranking surfaces labeled as research instrumentation only
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- runtime boundary:
	- no automated tests were added or run
	- no live viewer/runtime signoff has been completed yet for PM4 hover-time tooltip behavior or candidate quality

### Mar 30, 2026 - PM4 Shift-Click Collection Now Treats PM4 As The Primary Target

- followed direct runtime feedback that the first PM4 multi-select slice was still failing in practice:
	- `Shift + Left Click` could silently do nothing because normal scene-object hit priority still beat PM4 picking
	- per-item collection removal could leave stale PM4 highlight state behind
- landed behavior:
	- `ViewerApp.PickObjectAtMouse(...)` now handles `addPm4ToCollection` as a PM4-first branch, directly selecting and toggling the PM4 part when any PM4 hit exists under the cursor instead of comparing against regular scene hits first
	- failed additive clicks now report a clear status message telling the user no PM4 hit was found and to use graph `Collect` buttons for dense overlaps
	- per-item `Remove` in `ViewerApp_Pm4Utilities.DrawPm4ObjectCollectionSummary(...)` now resyncs PM4 collection highlighting immediately
- validation completed:
	- file diagnostics were clean for `src/MdxViewer/ViewerApp.cs` and `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- runtime boundary:
	- no automated tests were added or run
	- no live viewer/runtime signoff has been completed yet for the corrected Shift-click PM4 collection path

### Mar 30, 2026 - PM4 Collection JSON Export Added For Multi-Part Comparison

- added a viewer-side PM4 collection workflow to help inspect whether several PM4 parts are one object family or repeated overlapping copies
- landed behavior:
	- graph-driven collection now provides direct `Collect` buttons on MSLK link groups, MDOS groups, and individual parts, so the workflow does not depend on unreliable viewport PM4 picking
	- collected PM4 parts now show a distinct in-scene highlight color in the PM4 overlay and bounds path
	- export JSON now includes per-part debug info, merged-group ownership, signature buckets, same-signature center-overlap clusters, and `likelyDuplicateScore` metrics for quick duplicate inspection
- validation completed:
	- file diagnostics were clean for `src/MdxViewer/Terrain/WorldScene.cs`, `src/MdxViewer/ViewerApp.cs`, and `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- runtime boundary:
	- no automated tests were added or run
	- no live viewer/runtime signoff has been completed yet for the new PM4 collection workflow

### Mar 30, 2026 - PM4 Selection Family Split Rolled Back To Stop Half-Object Selection

- user runtime feedback after the Mar 30 selection-family experiments showed a hard regression:
	- the separate family-selection path could first pull in unrelated nearby PM4 pieces
	- after the same-tile helper rollback, it could also under-select and visibly split one object into smaller fragments
- landed rollback in `src/MdxViewer/Terrain/WorldScene.cs`:
	- removed `_pm4SelectedObjectFamilyGroupKeys`
	- removed `_selectedPm4ObjectFamilyGroupKey`
	- returned PM4 selection, highlight, and selected-object graph grouping to `_pm4MergedObjectGroupKeys`
	- kept the selected-only PM4 match cache path and selected-object match builder introduced in `ViewerApp` and `ViewerApp_Pm4Utilities`
- validation completed:
	- file diagnostics for `src/MdxViewer/Terrain/WorldScene.cs` were clean after the rollback
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing solution warnings only
- runtime boundary:
	- no automated tests were added or run
	- no live viewer/runtime signoff has been completed yet after this rollback

### Mar 30, 2026 - Selected PM4 Match Lookup Stops Rebuilding The Global Report And Zero-Root Ranking Stops Forcing WMO First

- followed new user feedback that clicking one zero-`CK24` part stalled before the sidebar reacted and then surfaced obviously wrong `WMO` suggestions for what looked like `M2`-family data
- root causes confirmed in the active viewer path:
	- `ViewerApp_Pm4Utilities.TryGetSelectedPm4ObjectMatch(...)` was forcing the full global `BuildPm4ObjectMatchReport(...)` path even when the UI only needed the currently selected PM4 object
	- `WorldScene.BuildPm4ObjectMatchReport(...)` hard-ranked all candidates with `WMO` mesh evidence first, which skewed zero/root-family selections toward `WMO` even before local anchor or overlap evidence could compete
- landed behavior:
	- the selected-object sidebar/window path now uses a lightweight selected-object PM4 match builder plus a small cache keyed by the selected PM4 object and match-count setting instead of rebuilding the global object-match report on click
	- `WorldScene` now reuses a shared object-match evaluation helper for both the full report and the selected-object path
	- zero/root PM4 objects (`CK24 == 0` or root-like link ownership) no longer get blanket `WMO`-first ranking; when linked refs exist they now prefer `M2` candidates before the normal same-tile, anchor-gap, planar, overlap, and footprint checks
	- non-zero PM4 families keep the old `WMO`-mesh-first ranking path
- validation completed:
	- editor diagnostics were clean for `src/MdxViewer/Terrain/WorldScene.cs`, `src/MdxViewer/ViewerApp.cs`, and `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026
- important boundary:
	- this is still build validation only in this session
	- no live viewer/runtime signoff has been completed yet to prove the click latency is now acceptable or that the zero/root top matches are materially better on the development map

### Mar 30, 2026 - Zero-CK24 Same-Tile Merge Gap Closed In WorldScene

- traced the remaining zero/root PM4 selection fragmentation in `src/MdxViewer/Terrain/WorldScene.cs` past the seed split and into the later merged-group map that drives selected-family ownership
- root cause confirmed in this slice:
	- zero-`CK24` parts already use synthetic per-part keys by default
	- `_pm4MergedObjectGroupKeys` was the only later regrouping seam
	- shared `Core.PM4` merged-group math intentionally skips same-tile merges, so same-tile zero-`CK24` families could never be recombined there
- landed behavior:
	- preserved shared cross-tile connector merging
	- added a local same-tile merge pass for synthetic zero-`CK24` keys using connector overlap plus local frame evidence from bounds, placement anchors, linked `MPRL` floors, and linked-heading summaries
- supporting real-data evidence recorded during the slice:
	- zero-`CK24` forensic export on `development_23_18.pm4` reported `1150` surfaces across `204` distinct link groups, with `203` non-zero `MSLK.GroupObjectId` values
	- this supported the conclusion that the missing family was not simply one blob with no link ownership
- validation completed:
	- editor diagnostics for `src/MdxViewer/Terrain/WorldScene.cs` were clean after the final fix
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/tmp/mdxviewer-bin/` passed on Mar 30, 2026 with existing warnings only
- runtime boundary:
	- no live viewer/manual signoff yet for the new same-tile zero-`CK24` regrouping behavior on the development map

### Mar 30, 2026 - Zero-CK24 Regrouping Stops Skipping MSLK Ownership In WorldScene

- narrowed the remaining zero/root PM4 regrouping problem in `src/MdxViewer/Terrain/WorldScene.cs` to the first split step after `(GroupKey, AttributeMask)` seed buckets were formed
- previous behavior:
	- zero-`CK24` seed buckets always skipped `SplitSurfaceGroupByMslk(...)`
	- they went directly into connectivity splitting, so disconnected but still `MSLK.GroupObjectId`-related pieces were split apart before placement or matching logic could treat them as one family
- landed behavior:
	- zero/root seed buckets now run through `SplitZeroCk24SeedGroup(...)`
	- groups with a non-zero dominant `MSLK.GroupObjectId` stay intact
	- only the remaining groups with no `MSLK` ownership evidence fall back to connectivity splitting
	- linked `MPRL` collection now also follows `MSLK.GroupObjectId` families first when one of those groups is already established for the current PM4 subgroup
- important semantic boundary:
	- this does not restore the old fake `MSLK.MsurIndex` path; current repo evidence still supports the 20-byte active `MSLK` layout and keeps `RefIndex` semantics partially open
- validation completed:
	- editor/file diagnostics for `src/MdxViewer/Terrain/WorldScene.cs` were clean after the change
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` was attempted on Mar 30, 2026 but did not complete because `ParpToolsWoWViewer (16096)` held output DLL locks (`MSB3021` / `MSB3027` copy failures)
	- no live viewer/runtime signoff was completed yet on the development map

### Mar 29, 2026 - v0.4.6 Release Prep Aligns PM4 Wins With The Next Renderer Seam

- user runtime feedback after the latest PM4 runtime fixes is now strongly positive: PM4 objects are described as almost `100%` correct on the development map
- release target is now being moved from `0.4.5` to `0.4.6`
- release-facing notes that need to stay grouped for this build:
	- PM4 overlay decoding and placement improved through the recent camera-window, tile-remap, empty-carrier, and linked-group placement fixes
	- first rendering-performance slices landed by removing repeated MDX visibility work and deferring WMO doodad expansion
- next renderer priority recorded for continuity:
	- add real render layers / submission buckets instead of keeping all world-scene submission embedded directly in `WorldScene.Render(...)`
	- focus the next performance slice on draw-call/state churn and layer ownership, not only on another isolated culling micro-fix
- validation/build boundary for this continuity update:
	- versioning and release-note surfaces were aligned in this pass
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Release` passed on Mar 29, 2026 with existing warnings only
	- local self-contained publish for `0.4.6` completed successfully after updating the release workflow publish step to tolerate duplicate dependency-side publish outputs from `WoWRollback.PM4Module`
	- local release archive `parp-tools-wow-viewer-v0.4.6-win-x64.zip` was produced on Mar 29, 2026 and the publish output still bundled `1315` WoWDBDefs `.dbd` files
	- the first cloud `v0.4.6` Actions run failed for two real release-workflow reasons, not because of local environment dirt:
		- the root release workflow, not the `gillijimproject_refactor` copy, was the workflow GitHub actually executed
		- cross-platform publish was still assuming a bundled `WowViewer.Core.IO` `area_crosswalk.csv` resource that is not tracked and should not be shipped
	- follow-up fix landed on Mar 29, 2026:
		- `WowViewer.Core.IO` now treats the embedded area crosswalk as optional instead of required, keeping runtime mapping on archive-backed or explicit user data paths
		- `MdxViewer.CrossPlatform.csproj` now carries the same `WowViewer.Core.IO` and `WowViewer.Core.PM4` references as the Windows project so Linux publish no longer fails on missing PM4 namespaces
		- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter AreaIdMapperTests` passed on Mar 29, 2026
		- `dotnet publish i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.CrossPlatform.csproj -c Release -f net10.0 -p:TargetFramework=net10.0 -r linux-x64 --self-contained true -p:PublishSingleFile=false -p:IncludeNativeLibrariesForSelfExtract=true -p:ErrorOnDuplicatePublishOutputFiles=false -p:TreatWarningsAsErrors=false -o i:/parp/parp-tools/output/MdxViewer-linux-x64-smoke` passed on Mar 29, 2026

### Mar 29, 2026 - WMO Doodads Stop Eagerly Expanding On The Render Thread And Object Fog Defaults Off

- followed the first `WorldScene` render-pass optimization with a second narrower slice aimed at the remaining reported symptoms:
	- multi-second hitches while tiles or data stream in
	- world objects appearing inside unwanted fog
- root cause for the new hitch slice:
	- `src/MdxViewer/Rendering/WmoRenderer.cs` constructor work still eagerly loaded the active doodad set, which recursively constructs doodad `MdxRenderer`s and can stall badly when new WMOs enter view
- landed behavior:
	- added deferred initial doodad loading state to `WmoRenderer` so world-scene WMO shells can appear first and doodad model loads are then drained incrementally under a small per-frame budget
	- `src/MdxViewer/Terrain/WorldAssetManager.cs` now creates world-scene `WmoRenderer` instances with `deferInitialDoodadLoads: true`
	- `src/MdxViewer/Terrain/WorldScene.cs` now reduces main-thread deferred asset processing to `6` loads with a `4 ms` budget per frame
	- `WorldScene` now uses a dedicated object-fog policy that defaults off, while WMO cull distance still uses terrain fog end instead of the disabled object-fog range
	- `src/MdxViewer/ViewerApp.cs` now exposes `Fog Objects` in the world-objects UI so the fogged-object path can be toggled back on when needed
- validation completed:
	- `get_errors` returned clean for the touched viewer files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings only
- runtime boundary:
	- this is compile-validated only in this session
	- no manual viewer traversal or measured hitch reduction was captured yet after the deferred doodad change

### Mar 29, 2026 - WorldScene MDX Render Passes Stop Repeating The Same Visibility Work

- started the user-requested performance pivot away from PM4-first work by targeting the hottest obvious CPU path in `src/MdxViewer/Terrain/WorldScene.cs`
- root cause for this first slice:
	- the scene was re-walking `_mdxInstances` and `_taxiActorInstances` across separate animation/update, opaque, and transparent passes
	- opaque and transparent passes were repeating frustum checks, AABB distance checks, and `TryGetQueuedMdx(...)` lookups for the same instances in the same frame
- landed behavior:
	- added a reusable visible-instance scratch list for loaded MDX/taxi instances
	- `CollectVisibleMdxInstances(...)` now performs the cull and renderer-resolution pass once, computing reusable opaque/transparent fade factors at the same time
	- opaque rendering now iterates the preclassified visible list
	- transparent rendering now sorts only the already-visible list instead of re-culling the world again
- validation completed:
	- `get_errors` on `WorldScene.cs` returned clean
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings only
- runtime boundary:
	- this is compile-validated only in this session
	- no measured frame-time comparison or manual viewer FPS confirmation has been captured yet

### Mar 29, 2026 - PM4 Terminology Guardrail Synced Across wow-viewer Continuity Files

- reconciled current PM4 field names against wowdev `PM4` and `PD4` documentation plus the standalone corpus-backed confidence reports
- recorded the hard boundary that several active names are local research aliases rather than documentation-backed field names:
	- `MSUR.GroupKey`
	- `MSUR.AttributeMask`
	- `MSUR.MdosIndex`
	- `MSUR.PackedParams`
	- derived `CK24`, `Ck24Type`, `Ck24ObjectId`
	- `MSLK.GroupObjectId`
- also recorded the stronger current corrections that should survive even when names change:
	- `MSUR` bytes `0x04..0x0f` are geometry-validated normals
	- the current `MSUR.Height` name is misleading because the float behaves like a signed plane-distance term
	- `MSLK.RefIndex` is not closed as a universal `MSUR` index across the corpus
- continuity surfaces updated:
	- `gillijimproject_refactor/src/Pm4Research.Core/README.md`
	- `gillijimproject_refactor/memory-bank/activeContext.md`
	- `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md`
	- `wow-viewer/README.md`
	- `.github/prompts/wow-viewer-pm4-library-implementation.prompt.md`
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
- validation boundary:
	- this was a terminology and continuity correction only; no new PM4 code or runtime behavior changed in this pass

### Mar 29, 2026 - Shared PM4 Hierarchy Research Slice Landed And `MdxViewer` Cut Over

- moved the active viewer PM4 research path off legacy `src/Pm4Research.Core` and into shared `wow-viewer/src/core/WowViewer.Core.PM4`
- landed `Research/Pm4ResearchHierarchyAnalyzer.cs` in `Core.PM4`, porting the old split-family object-hypothesis research and extending each candidate with:
	- dominant `MSLK.GroupObjectId`
	- shared placement comparison (`CoordinateMode`, planar transform, world pivot, frame yaw, heading delta)
	- the existing MPRL footprint evidence
- added `WowViewer.Tool.Inspect pm4 hierarchy --input <file.pm4> [--output <report.json>]`
- rewired `src/MdxViewer/Terrain/WorldScene.cs` so selected-object PM4 research now uses shared snapshot, shared decode audit, and shared hierarchy analysis from `Core.PM4`
- updated the `PM4 Research` viewer panel to show the new shared hierarchy and placement signals for top hypothesis matches
- validation completed:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 29, 2026 with existing environment warnings
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --filter "Hierarchy_DevelopmentTile_ExposesSharedPlacementAndLinkGroupEvidence|Pm4ResearchIntegrationTests"` passed on Mar 29, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 hierarchy --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` produced a real hierarchy report on Mar 29, 2026
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this lands more grounded PM4 scene-graph evidence in shared code and in the viewer UI, but it does not claim the CK24 placement regression is solved yet

### Mar 29, 2026 - PM4 Incremental Loads Stop Clearing Prior Residency

- fixed a real PM4 runtime residency bug in `src/MdxViewer/Terrain/WorldScene.cs` that could make PM4 objects disappear permanently while moving around the viewer, especially when crossing into a new PM4 camera-window load
- root cause:
	- `BeginPm4OverlayLoad(...)` was resetting `_pm4LoadedCameraWindow` to `null` before the async load finished
	- `TryFinalizePm4OverlayLoad()` uses `_pm4LoadedCameraWindow.HasValue` to decide whether a load should merge into existing PM4 state or replace it
	- because the window had already been nulled, every incremental load finalized as a full replacement and cleared earlier PM4 tiles instead of preserving them
- landed behavior:
	- normal background PM4 loads now keep the previously loaded camera window intact until finalize decides whether to merge
	- manual `Reload PM4` now explicitly clears PM4 runtime state first and then starts a fresh cache-bypassing load, so reload behaves like a real full reload instead of a partial merge with stale bookkeeping
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is build-validated only; the exact user repro of “PM4 disappears as I approach and reload does not bring it back” still needs manual viewer confirmation after this fix

### Mar 29, 2026 - PM4 Same-Tile Candidate Collisions Now Keep One Canonical File

- narrowed the PM4 file-selection path in `src/MdxViewer/Terrain/WorldScene.cs` so both runtime loading and offline PM4 OBJ export now keep only one preferred `.pm4` candidate per effective tile instead of blindly merging every file that parses to the same tile coordinate
- reason: the latest non-zero `CK24` graph exports still showed exact paired duplicate parts like `part=0` and `part=495` even with `Split CK24 by MdosIndex` and `Split CK24 by Connectivity` disabled, which strongly fits same-tile candidate collisions rather than a pure transform-math bug
- current selection policy prefers the most canonical `.../World/Maps/<map>/<map>_<x>_<y>.pm4` style path and logs dropped same-tile candidates for follow-up diagnosis
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is still build-validated only; manual viewer confirmation is still required to prove the paired duplicate-part pattern and opposite-corner PM4 placements actually disappear on the development map

### Mar 29, 2026 - Global PM4 Y Mirror No Longer Applies To Zero-CK24 Root Buckets

- narrowed the PM4 object transform path in `src/MdxViewer/Terrain/WorldScene.cs` so the global `Mirror PM4 N/S` flip no longer applies to `CK24=0x000000` objects
- reason: current live viewer evidence showed the bad zero/root PM4 overlays snapping back into the correct placed-object location only after pressing `Wind Obj Y`, which is effectively cancelling the global Y mirror on that selected object
- preserved the global Y mirror for non-zero `CK24` groups, since those were the original reason the default north/south mirror was enabled in the first place
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is still build-validated only; manual viewer confirmation is still required to prove the zero/root PM4 objects now overlap the placed objects again without using `Wind Obj Y`

### Mar 29, 2026 - Zero-CK24 Seed Groups No Longer Re-Split After Seed Connectivity Split

- narrowed the PM4 zero/root-bucket runtime path in `src/MdxViewer/Terrain/WorldScene.cs` so seed groups that already require the mandatory connectivity split no longer also honor the later viewer toggle stages for `Split CK24 by MdosIndex` or `Split CK24 by Connectivity`
- reason: the current viewer evidence and graph exports point to zero/root `CK24` groups being fragmented twice, which can manufacture paired sub-parts after the first seed-level split and then make later frame or winding experiments look like rotation regressions instead of grouping regressions
- preserved the later MDOS/connectivity toggle behavior for non-zero `CK24` groups; this change is only for the zero/root seed path that already split once at the seed stage
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is build-validated only; no live viewer signoff was completed after the regrouping change, so the development map still needs manual confirmation that the artificial paired-part split is actually gone

### Mar 29, 2026 - Viewer Shell Resize/Input Sync Hardened Again

- fixed the recurring viewer-shell regression in `src/MdxViewer/ViewerApp.cs` where resize and mouse-hit behavior could break again even outside PM4-specific windows
- the old bridge only partially synchronized Silk window metrics into ImGui: it used the private `ImGuiController.WindowResized(...)` hook against logical size, but did not explicitly keep `ImGuiIO.DisplaySize` and `DisplayFramebufferScale` synchronized from both logical and framebuffer sizes
- `ViewerApp` now subscribes to both logical `Resize` and `FramebufferResize`, and `SyncImGuiWindowMetrics(...)` updates the private Silk hook plus explicit `ImGui` size/framebuffer-scale values before each `_imGui.Update(...)`
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is build-validated only; no live shell resize or hit-testing signoff was completed after the patch

### Mar 29, 2026 - Zero-CK24 PM4 Mixed-Bucket Placement Fix Landed In MdxViewer

- fixed a PM4 runtime-consumer regression in `src/MdxViewer/Terrain/WorldScene.cs` where zero-`CK24` / root-style seed buckets were connectivity-split only after one shared placement basis had already been resolved for the whole mixed bucket
- root cause fit the latest manual symptom: some `M2`-aligned PM4 data drifted while nearby WMO-aligned PM4 remained mostly stable, because non-zero `CK24` WMO-style groups still had coherent per-`CK24` placement while zero/root buckets could mix unrelated parts before placement resolution
- the zero/root-style path now resolves coordinate mode or placement solution or connector keys per linked connectivity group instead of reusing one mixed-bucket planar transform or pivot or frame yaw across all zero-`CK24` parts in the seed bucket
- preserved the existing non-zero `CK24` behavior: shared per-`CK24` frame basis is still reused across split parts for the WMO-style path
- also recorded the user clarification that `CK24=0x000000` should not be treated as "just M2 data"; it is better treated as an unresolved root or umbrella bucket for now
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is compile-validated only; the development map still needs manual runtime confirmation to prove the M2/root-bucket drift is actually corrected

### Mar 29, 2026 - PM4 CK24 Alignment Controls Narrowed To Tile-Local Buckets

- replaced the earlier raw-`CK24` alignment state in `src/MdxViewer/Terrain/WorldScene.cs` so those transforms are now keyed by `(tileX, tileY, ck24)` instead of only `ck24`
- this stops exploratory fixes for `CK24=0x000000` from rotating or mirroring every matching raw bucket across the loaded PM4 overlay when the actual issue appears tile-local
- updated `src/MdxViewer/ViewerApp_Pm4Utilities.cs` so the PM4 alignment window now describes and edits tile-local `CK24` transforms, and added direct tile/object winding toggle buttons for faster handedness checks
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this is build-validated only; no live viewer signoff was completed after the tile-local control change

### Mar 29, 2026 - Shared CK24 PM4 Forensic Export Slice Landed

- added `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4ForensicsModels.cs` and `Research/Pm4Ck24ForensicsAnalyzer.cs` so `Core.PM4` now owns a research-only CK24 forensic report instead of relying on viewer-only PM4 graph JSON
- the new report exposes component-level link groups, raw MSLK rows, raw linked MPRL rows, footprint counts, and placement-vs-heading comparison for a target `CK24`
- extended `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` so `pm4 export-json --ck24 <decimal|0xHEX>` emits the targeted CK24 forensic report while the original no-`--ck24` path still emits the coarse single-file PM4 analysis report
- fixed PM4 inspect JSON serialization for vector payloads by enabling field serialization, so the new shared forensic JSON shows real `Vector3` coordinates instead of empty objects
- added real-data PM4 regression coverage in `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4ResearchIntegrationTests.cs` for a dense linked CK24 case (`0x412CDC`) and a sparse no-linked-MPRL case (`0x41C0F5`) on `development_00_00.pm4`
- validation completed:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 29, 2026 after the slice landed
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter Pm4ResearchIntegrationTests` passed on Mar 29, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect -- pm4 export-json --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4 --ck24 0x412CDC --output i:/parp/parp-tools/wow-viewer/output/pm4_ck24_412CDC_forensics.json` wrote real-data shared forensic JSON
- boundary:
	- this slice proves shared export and regression behavior, not final PM4 runtime semantics or viewer signoff

### Mar 29, 2026 - PM4 CK24 Frame Yaw No Longer Rotates Visible Mesh Geometry

- changed the PM4 CK24 object-generation path in `src/MdxViewer/Terrain/WorldScene.cs` so `worldYawCorrection` stays on the object frame or anchor path instead of being baked directly into the generated mesh lines and triangles
- root cause was viewer evidence that CK24 objects were being visually rotated and displaced as though frame correction had been applied to the mesh itself, which inverted the intended ownership between visible geometry and the object frame basis
- `BuildCk24ObjectLines(...)` and `BuildCk24ObjectTriangles(...)` now convert PM4 mesh vertices without the CK24 frame yaw correction, while placement-anchor computation still retains the frame-yaw path
- validation completed:
	- editor/language-service checks passed for `src/MdxViewer/Terrain/WorldScene.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- this fix has not yet been re-checked in the live viewer during this session, so the exact effect on the opposite-side `CK24` placements still needs manual confirmation

### Mar 29, 2026 - PM4 Raw CK24 Layer Alignment Added In MdxViewer

- added a parallel PM4 raw-`CK24` layer transform path in `src/MdxViewer/Terrain/WorldScene.cs` so any selected PM4 object can now drive whole-layer translation or rotation or scale keyed by the original `ck24` value instead of only the resolved object-group key
- this specifically unblocks exploratory work on `CK24=0x000000`, which had been structurally split into synthetic per-part groups for object transforms and therefore could not previously be rotated as one layer
- `BuildPm4ObjectTransform(...)` now applies raw-`CK24` layer transform before the existing object-group transform, and the scene now tracks raw-layer pivots from combined bounds across all loaded tiles
- extended `src/MdxViewer/ViewerApp_Pm4Utilities.cs` with `CK24 Layer` move or rotate or scale controls, reset actions, and a print action while preserving the existing per-object-group controls beneath them
- extended PM4 interchange JSON reporting so each exported object also includes the raw-layer transform state currently affecting its `ck24`
- validation completed:
	- editor/language-service checks passed for `WorldScene.cs` and `ViewerApp_Pm4Utilities.cs`
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 29, 2026 with existing warnings
- runtime boundary:
	- the new `CK24 Layer` controls were not yet exercised in the live viewer during this session, so the `0x000000` per-tile or quadrant-orientation hypothesis remains unverified

### Mar 28, 2026 - PM4 Graph JSON Export No Longer Fails On Non-Finite Heading Values

- fixed the selected-object `PM4 Graph` JSON export in `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
- root cause was raw `System.Text.Json` serialization of `Pm4LinkedPositionRefSummary`, whose heading fields can be non-finite when a graph link group has no normal heading evidence
- replaced raw struct serialization with a JSON-safe projected payload and finite-or-null handling for linked-position-ref heading values so the export stays valid standard JSON instead of throwing in the status bar
- also repaired the remaining `Pm4OverlayCacheService` type reference so the earlier shared `Pm4PlanarTransform` cleanup still builds cleanly
- validation completed:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 28, 2026 after the fix
- runtime boundary:
	- the export button itself was not re-clicked in this session after the patch, so UI-level runtime confirmation is still pending

### Mar 28, 2026 - PM4 Overlay North/South Mirror Defaulted In MdxViewer

- removed the duplicate viewer-local `Pm4PlanarTransform` contract from `src/MdxViewer/Terrain/WorldScene.cs` so the active PM4 consumer now uses the shared `WowViewer.Core.PM4` placement contract directly
- updated the CK24 coordinate-mode consumer path to keep the shared typed `Pm4CoordinateModeResolution` result instead of collapsing it to a bool immediately
- switched viewer-side default world-space planar-transform fallback usage onto shared `Pm4PlacementContract`
- enabled the existing PM4 object-group mirror path by default through `_pm4FlipAllObjectsY = true`, because current manual runtime evidence showed PM4 overlay geometry landing on the wrong north/south side of nearby `M2` placements with handedness-related rotation mismatch
- kept the PM4 correlation-input builder viewer-local for now; the current correction is about overlay alignment, not correlation ownership
- updated the PM4 UI toggle label in `src/MdxViewer/ViewerApp.cs` from `Flip All Obj Y` to `Mirror PM4 N/S` so the control matches the behavior it actually applies
- validation so far is editor/language-service only; no build or new real-data runtime signoff was completed in this pass because the requested `MdxViewer` build was cancelled before completion

### Mar 28, 2026 - Post-MDX Default Continuation Reset

- recorded that the default next implementation area after pausing `MDX` is `wow-viewer` `Core.PM4` library completion
- preserved non-`MDX` shared-I/O as a secondary path only for narrow ADT/WDT/WMO slices with concrete proof targets

### Mar 28, 2026 - MDX Audit Reclassified Recent Shared Readers

- audited recent `wow-viewer` classic `MDX` work against active `MdxViewer` implementation instead of continuity claims alone
- confirmed that `GEOS` is the clearest real classic-parser parity seam among the recent payload slices
- reclassified `TXAN` payload ownership as a new shared-reader seam informed by legacy runtime concepts and M2-adapter usage, not a direct classic `MdxViewer` parser port
- reclassified `HTST` payload ownership as a new shared-reader seam with no current active classic `MdxViewer` parser/runtime parity
- reclassified `CLID` payload ownership as beyond active classic parser parity; active viewer currently uses only shared collision summary metadata in probe/model-info surfaces
- identified the hotter missed legacy parity seam: classic `ATSQ` geoset-animation and related material animation behavior already used by the active renderer

### Mar 28, 2026 - MDX Chunk Expansion Explicitly Paused

- recorded user direction that further speculative `MDX` chunk implementation should stop
- future continuation should not treat unresolved `MDX` families as the automatic next slice just because carrier discovery or inspect support exists
- only resume new `MDX` chunk work if the user explicitly asks for a named seam or if a concrete active consumer requirement makes it necessary

### Mar 29, 2026 - Shared Classic `MDX` `TXAN` Payload Slice Landed

- advanced the shared classic `MDX` migration from unresolved `TXAN` discovery into first typed texture-animation payload ownership so actual `KTAT` or `KTAR` or `KTAS` keyframes no longer remain only as a top-level chunk id in inspect output
- added shared `MdxTextureAnimationFile` and `MdxTextureAnimation` contracts for indexed classic texture-animation entries
- added `WowViewer.Core.IO.Mdx.MdxTextureAnimationReader` for classic `TXAN` payload reads in `v1300` and `v1400`, including counted section framing and actual translation or rotation or scaling keyframe payload parsing
- extracted the reusable vector3 and compressed-quaternion keyframe parsing into `WowViewer.Core.IO.Mdx.MdxTrackReader` and switched `MdxHitTestReader` to reuse it, so `HTST` and `TXAN` now share one track interpretation instead of drifting apart
- extended `WowViewer.Tool.Inspect mdx export-json` with `--include-texture-animations` so the new library-owned payload seam is immediately reusable on filesystem or archive-backed inputs without adding a tool-local parser path
- added `wow-viewer/tests/WowViewer.Core.Tests/MdxTextureAnimationReaderTests.cs` with a synthetic tracked texture-animation fixture, a real Alpha no-TXAN regression on `Wisp.mdx`, and a fixed real standard-era archive-backed AirElemental case
- this landing proves payload ownership and JSON export only; it does not claim runtime texture-transform evaluation or viewer cutover
- this should not be used as justification to continue automatic `MDX` chunk expansion work

### Mar 29, 2026 - Shared Classic `MDX` `HTST` Payload Slice Landed

- advanced the shared classic `MDX` migration from `HTST` summary-only ownership into first typed hit-test payload ownership so fixed shape payloads and actual `KGTR` or `KGRT` or `KGSC` keyframes no longer have to stay trapped behind summary-only metadata
- added shared `MdxHitTestFile` and `MdxHitTestShape` contracts plus reusable typed node-track payload contracts for vector3 and compressed-quaternion keyframes with interpolation metadata
- added `WowViewer.Core.IO.Mdx.MdxHitTestReader` for classic `HTST` payload reads in `v1300` and `v1400`, including fixed box or cylinder or sphere or plane payloads and actual transform keyframe payload parsing
- extended `WowViewer.Tool.Inspect mdx export-json` with `--include-hit-test` so the new library-owned payload seam is immediately reusable on filesystem or archive-backed inputs without adding a tool-local parser path
- added `wow-viewer/tests/WowViewer.Core.Tests/MdxHitTestReaderTests.cs` with a synthetic tracked hit-test fixture, a fixed real Alpha `Wisp.mdx` case, and a fixed real standard-era archive-backed `anubisath.mdx` case
- this landing proves payload ownership and JSON export only; it does not claim runtime hit detection or viewer cutover

### Mar 28, 2026 - Shared Classic `MDX` `CLID` Payload Slice Landed

- advanced the shared classic `MDX` migration from `CLID` summary-only ownership into first typed collision-mesh payload ownership so full `VRTX` or `TRI ` or `NRMS` data no longer has to stay trapped behind inspect-only summaries
- added shared `MdxCollisionFile` and `MdxCollisionMesh` contracts plus `WowViewer.Core.IO.Mdx.MdxCollisionReader` for classic `CLID` payload reads in `v1300` and `v1400`
- extracted the chunk-level payload logic into a shared `MdxCollisionChunkReader` helper and switched `MdxSummaryReader` to reuse it, so summary and payload coverage now share one `CLID` interpretation instead of drifting apart
- extended `WowViewer.Tool.Inspect mdx export-json` with `--include-collision` so the new library-owned payload seam is immediately reusable on filesystem or archive-backed inputs without adding a new tool-local parser path
- added `wow-viewer/tests/WowViewer.Core.Tests/MdxCollisionReaderTests.cs` with a synthetic collision fixture, a fixed real Alpha `Wisp.mdx` case, and a fixed real standard-era archive-backed dwarf-character case
- this landing proves payload ownership and JSON export only; it does not claim runtime collision behavior or viewer cutover

### Mar 28, 2026 - `mdx export-json` Inspect Slice Landed

- added `WowViewer.Tool.Inspect mdx export-json` as a thin JSON export surface over the shared `MdxSummaryReader`, with optional `--include-geometry` over the current shared `MdxGeometryReader` seam
- kept the slice library-first: the new command reuses the shared readers for both filesystem and archive-backed inputs instead of adding any second `MDX` parser in the inspect tool
- fixed the initial JSON serialization bug by enabling field serialization for `System.Numerics` payloads, so vectors and UV data now serialize as real coordinates instead of empty objects
- validated the slice with real data on both the fixed Alpha `Wisp.mdx` summary surface and the fixed standard-era `chest01.mdx` summary-plus-geometry surface
- this slice is still export of already-owned seams, not new chunk-family ownership or runtime `MDX` parity

### Mar 28, 2026 - `mdx chunk-carriers` Inspect Workflow Landed

- added `WowViewer.Tool.Inspect mdx chunk-carriers --chunks <FOURCC[,FOURCC...]>` as a thin carrier-discovery surface over shared `MdxSummaryReader`
- kept the slice tool-thin and library-first: the new command scans either a filesystem `MDX` file or directory or an archive-backed dataset via `MpqArchiveCatalog`, but it still uses the shared summary reader for chunk ownership instead of adding tool-local chunk parsing
- added practical scan controls with `--path-filter <text>` and `--limit <n>` so standard archive scans can stay targeted when future sessions are looking for the next fixed positive carrier
- validated the new workflow with a real positive archive-backed `LITE` scan over `braziers`, which found `dwarvenbrazier01.mdx` plus `3` more positive standard-era carriers, and with a real negative alpha-corpus scan proving the current unpacked `0.5.3` sample set still has no `TXAN`, `PREM`, or `CORN` carriers across `229` files
- this slice does not add new `MDX` chunk-summary ownership by itself; it adds the discovery workflow needed to choose the next real-data-backed seam without guessing

### Mar 28, 2026 - Viewer UI Resize And Input Regression Fixed

- fixed a real `MdxViewer` shell regression where panels drew at the wrong size and toolbar or sidebar buttons stopped responding after window resize or maximize
- updated `src/MdxViewer/ViewerApp.cs` to explicitly resync the packaged Silk `ImGuiController` logical window size through its private `WindowResized(Vector2D<int>)` hook, while keeping the OpenGL viewport bound to framebuffer resize
- validated the patch with `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` plus a short viewer startup smoke
- the user manually retested the resized UI on Mar 28, 2026 and reported that it now seems to be working
- this is still manual runtime validation only; there is no automated UI regression coverage for the resize or hit-testing path

### Mar 28, 2026 - ViewerApp Shared `MDX` Runtime Metadata Consumer Landed

- started the first non-probe runtime `MDX` consumer cutover in `MdxViewer` without changing the renderer ownership boundary
- updated `src/MdxViewer/ViewerApp.cs` so the real `MDLX` disk or data-source route now reads shared `MdxSummaryReader` plus `MdxGeometryReader` metadata before the legacy `MdxFile.Load(...)` render load
- switched standalone runtime model-info and load-status counts to prefer shared version or model-name or geoset or vertex or triangle or pivot or collision metadata, while keeping explicit legacy fallback when shared reads fail
- validated the slice with `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` plus a real startup smoke on `wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx` that printed `[SharedMDX] Runtime metadata consumer: summary=yes geometry=yes file=Wisp.mdx`
- this is still a metadata-only runtime consumer cutover, not a renderer cutover or world-scene asset-loader cutover

### Mar 28, 2026 - AssetProbe Shared `PIVT` And `CLID` Signals Landed

- expanded the existing `AssetProbe` shared `MDX` compatibility surface past summary or `GEOS` counts into visible shared pivot-table and collision-mesh reporting
- updated `src/MdxViewer/AssetProbe.cs` so `SharedMDX` probes now emit `SharedPIVT` and `SharedCLID` lines when the shared summary exposes those chunks
- validated the output on archive-backed real assets: `chest01.mdx` now reports `SharedPIVT: count=6`, and `Creature/AncientOfWar/AncientofWar.mdx` now reports both `SharedPIVT: count=72` and `SharedCLID: vertices=8 triangles=12`
- this is still probe-only validation; it does not move collision or pivot handling into the runtime renderer path by itself

### Mar 28, 2026 - AssetProbe Shared `GEOS` Consumer Cutover Landed

- advanced the active `MdxViewer` compatibility surface from shared `MDX` summary-only probe validation into first shared `GEOS` payload consumer validation
- updated `src/MdxViewer/AssetProbe.cs` so probe-side geoset reporting now comes from `WowViewer.Core.IO.Mdx.MdxGeometryReader` instead of depending on `MdxFile.Load(...)` geoset objects
- kept the cutover narrow: the probe still uses legacy `MdxFile.Load(...)` for the rest of the model parse and texture or material reporting
- validated the change by building `MdxViewer` and running `--probe-mdx` on both archive-backed `chest01.mdx` and `Creature/AncientOfWar/AncientofWar.mdx`, with the latter confirming full shared reporting across `5` geosets
- this is still non-UI compatibility validation only, not a runtime model-loading cutover

### Mar 28, 2026 - Shared Classic `MDX` `GEOS` Payload Slice Landed

- advanced the shared classic `MDX` migration from `GEOS` summary-only ownership into first typed geoset payload ownership so render-facing mesh data no longer has to stay trapped behind `MdxFile.Load(...)`
- added shared `MdxGeometryFile` and `MdxGeosetGeometry` contracts for vertices, normals, UV sets, primitive types, face groups, indices, vertex groups, matrix tables, bone tables, and footer metadata
- added `WowViewer.Core.IO.Mdx.MdxGeometryReader` with classic counted `GEOS` payload support for `v1300` and `v1400`, including Alpha-style direct `UVAS` reads and optional explicit `UVBS` support
- added `wow-viewer/tests/WowViewer.Core.Tests/MdxGeometryReaderTests.cs` with a synthetic classic-`GEOS` payload fixture, a fixed real standard-era positive carrier, and a real on-disk alpha-era positive carrier from the existing `0.5.3` corpus
- validated the slice with focused shared-reader tests against both standard-era and alpha-era real data
- this landing now has the first shared classic `GEOS` payload seam in `wow-viewer`; it is still not runtime buffer assembly, skinning evaluation, or viewer render cutover

### Mar 28, 2026 - Shared Classic `MDX` `LITE` Summary Slice Landed

- advanced the shared classic `MDX` migration from `GLBS` into `LITE` so classic light metadata no longer remains only as a known-but-unparsed top-level chunk id in `wow-viewer`
- added shared `MdxLightType` and `MdxLightSummary` and extended `MdxSummary` with `Lights` plus `LightCount`
- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `LITE` summary support for `v1300` and `v1400`, including inherited node metadata plus static attenuation or color or intensity fields and optional `KLAS`, `KLAE`, `KLAC`, `KLAI`, `KLBC`, `KLBI`, and `KVIS` metadata
- updated `WowViewer.Tool.Inspect mdx inspect` to report `lights=` and print `LITE[n]` lines
- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`LITE` fixture and a fixed real archive-backed `0.6.0` `dwarvenbrazier01.mdx` light case
- added a real unpacked `0.5.3` alpha-corpus smoke over `229` MDX files to prove the new `LITE` path does not regress current alpha-era parsing even though the bundled `0.5.3` sample set contains no `LITE` chunks today
- validated the seam with focused shared-reader tests plus real inspect output on `world/generic/dwarf/passive doodads/braziers/dwarvenbrazier01.mdx`
- this landing now has strong synthetic coverage plus fixed real standard `0.6.0` `MDX` validation for classic `LITE`

### Mar 28, 2026 - Shared Classic `MDX` `GLBS` Summary Slice Landed

- advanced the shared classic `MDX` migration from `CLID` into `GLBS` so global sequence duration tables no longer remain only as known-but-unparsed top-level chunk ids in `wow-viewer`
- added shared `MdxGlobalSequenceSummary` and extended `MdxSummary` with `GlobalSequences` plus `GlobalSequenceCount`
- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with strict `GLBS` summary support for counted `uint32` durations and invalid payload-size rejection
- updated `WowViewer.Tool.Inspect mdx inspect` to report `globalSequences=` and print `GLBS[n]` lines
- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic `GLBS` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` global-sequence case
- validated the seam with focused shared-reader tests plus real inspect output on `wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx`
- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `GLBS`

### Mar 28, 2026 - Shared Classic `MDX` `CLID` Summary Slice Landed

- advanced the shared classic `MDX` migration from `HTST` into `CLID` so collision meshes no longer remain only as known-but-unparsed top-level chunk ids in `wow-viewer`
- added shared `MdxCollisionSummary` and extended `MdxSummary` with nullable collision ownership
- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic `CLID` summary support for `v1300` and `v1400`, including ordered `VRTX` or `TRI ` or `NRMS` subchunk parsing, derived bounds, and max-index coverage
- updated `WowViewer.Tool.Inspect mdx inspect` to report `collisionVertices=` or `collisionTriangles=` and print a `CLID:` line
- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`CLID` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` collision case
- validated the seam with focused shared-reader tests plus real inspect output on `wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx`
- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `CLID`

### Mar 28, 2026 - Shared Classic `MDX` `HTST` Summary Slice Landed

- advanced the shared classic `MDX` migration from `EVTS` into `HTST` so hit-test shapes no longer remain only as known-but-unparsed top-level chunk ids in `wow-viewer`
- added shared `MdxGeometryShapeType` and `MdxHitTestShapeSummary` contracts and extended `MdxSummary` with `HitTestShapes` plus `HitTestShapeCount`
- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `HTST` summary support for `v1300` and `v1400`, including inherited node metadata plus fixed box or cylinder or sphere or plane payload fields
- updated `WowViewer.Tool.Inspect mdx inspect` to report `hitTestShapes=` and print `HTST[n]` lines
- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`HTST` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` hit-test-shape case
- validated the seam with focused shared-reader tests plus real inspect output on `wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx`
- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `HTST`

### Mar 28, 2026 - Shared Classic `MDX` `EVTS` Summary Slice Landed

- advanced the shared classic `MDX` migration from `CAMS` into `EVTS` so event nodes no longer remain only as known-but-unparsed top-level chunk ids in `wow-viewer`
- added shared `MdxEventSummary` and `MdxEventTrackSummary` contracts and extended `MdxSummary` with `Events` plus `EventCount`
- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `EVTS` summary support for `v1300` and `v1400`, including inherited node metadata and optional `KEVT` key-time metadata
- updated `WowViewer.Tool.Inspect mdx inspect` to report `events=` and print `EVTS[n]` lines
- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`EVTS` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` event case
- validated the seam with focused shared-reader tests plus real inspect output on `wow-viewer/testdata/0.5.3/tree/Creature/Wisp/Wisp.mdx`
- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `EVTS`

### Mar 28, 2026 - Shared Classic `MDX` `CAMS` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past classic `RIBB` summary ownership so `MdxSummaryReader` now also exposes classic camera summary coverage for fixed camera metadata and summary-only camera-track metadata.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxCameraSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `Cameras` and `CameraCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `CAMS` summary support for `v1300` and `v1400`, including per-camera section sizing, fixed payload fields, and optional `KCTR`, `KCRL`, `KVIS`, and `KTTR` metadata
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `CAMS[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`CAMS` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` camera case
- Validation limits:
	- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `CAMS`
	- this is still summary-only classic camera ownership, not camera playback, interpolation evaluation, viewer camera selection, or runtime portrait parity

### Mar 28, 2026 - Shared Classic `MDX` `PRE2` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past classic `RIBB` summary ownership so `MdxSummaryReader` now also exposes classic particle-emitter-v2 summary coverage for `MDLGENOBJECT`-derived effect metadata.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxParticleEmitter2Summary`
	- reused `WowViewer.Core.Mdx.MdxTrackSummary` for classic `PRE2` scalar track metadata
	- extended `WowViewer.Core.Mdx.MdxSummary` with `ParticleEmitters2` and `ParticleEmitter2Count`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `PRE2` summary support for `v1300` and `v1400`, including outer emitter sizing, inner node sizing, classic scalar payload fields, spline-count handling, and optional `KVIS` or `KP2V` plus `KP2S`, `KP2R`, `KP2L`, `KPLN`, `KP2G`, `KLIF`, `KP2E`, `KP2W`, `KP2N`, and `KP2Z` metadata
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `PRE2[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`PRE2` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` particle-emitter case
- Validation limits:
	- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `PRE2`
	- this is still summary-only classic particle-emitter ownership, not particle simulation, UV animation playback, spline playback, or runtime render parity

### Mar 28, 2026 - Shared Classic `MDX` `ATCH` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past classic `HELP` summary ownership so `MdxSummaryReader` now also exposes classic attachment summary coverage for `MDLGENOBJECT`-derived attachment metadata.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxAttachmentSummary`
	- added `WowViewer.Core.Mdx.MdxVisibilityTrackSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `Attachments` and `AttachmentCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `ATCH` summary support for `v1300` and `v1400`, including outer section sizing, inner node sizing, `KGTR` or `KGRT` or `KGSC` transform metadata, and optional `KVIS` or `KATV` visibility metadata
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `ATCH[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`ATCH` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` attachment case
- Validation limits:
	- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `ATCH`
	- this is still summary-only classic attachment ownership, not visibility evaluation, asset resolution, or runtime attachment/render parity

### Mar 28, 2026 - Shared Classic `MDX` `HELP` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past classic `BONE` summary ownership so `MdxSummaryReader` now also exposes classic helper-node summary coverage for `MDLGENOBJECT` metadata.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxHelperSummary`
	- added `WowViewer.Core.Mdx.MdxNodeTrackSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `Helpers` and `HelperCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `HELP` summary support for `v1300` and `v1400`, including `KGTR` or `KGRT` or `KGSC` track metadata
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `HELP[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`HELP` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` helper case
- Validation limits:
	- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `HELP`
	- this is still summary-only classic helper ownership, not transform evaluation, attachment behavior, billboard handling, or viewer playback parity

### Mar 28, 2026 - Shared Classic `MDX` `BONE` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past classic `GEOA` summary ownership so `MdxSummaryReader` now also exposes classic bone summary coverage for render-facing skeleton metadata.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxBoneSummary`
	- added `WowViewer.Core.Mdx.MdxNodeTrackSummary` as the shared classic node-track contract reused by `BONE` and `HELP`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `Bones` and `BoneCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `BONE` summary support for `v1300` and `v1400`, including `KGTR` or `KGRT` or `KGSC` track metadata plus geoset-link fields
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `BONE[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`BONE` fixture and a fixed real Alpha `0.5.3` `Wisp.mdx` bone case
- Validation limits:
	- this landing now has strong synthetic coverage plus fixed real Alpha `0.5.3` `MDX` validation for classic `BONE`
	- this is still summary-only classic bone ownership, not transform evaluation, runtime skeleton assembly, or viewer playback parity

### Mar 28, 2026 - Shared Classic `MDX` `GEOA` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past classic `GEOS` structure ownership so `MdxSummaryReader` now also exposes classic geoset-animation summary coverage for render-facing metadata.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxGeosetAnimationSummary`
	- added `WowViewer.Core.Mdx.MdxGeosetAnimationTrackSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `GeosetAnimations` and `GeosetAnimationCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted `GEOA` summary support for `v1300` and `v1400`, including `KGAO` or `KGAC` track metadata
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `GEOA[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with a synthetic classic-`GEOA` fixture and an optional real archive-backed `GEOA` probe case
- Validation limits:
	- this landing now has strong synthetic coverage plus real Alpha `0.5.3` `MDX` validation for classic `GEOA`; the fixed `0.6.0` archive corpus still has no guaranteed positive `GEOA` asset identified
	- this is still summary-only classic geoset-animation ownership, not animation evaluation, viewer playback parity, or runtime geoset-state cutover

### Mar 28, 2026 - Shared Classic `MDX` `GEOS` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past `SEQS` and `PIVT` so `MdxSummaryReader` now also exposes classic geoset summary coverage for render-facing mesh structure.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxGeosetSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `Geosets` and `GeosetCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with classic counted tagged `GEOS` summary support for `v1300` and `v1400`
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `GEOS[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with synthetic classic-geoset coverage and a real archive-backed `chest01.mdx` geoset case
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `8` passing tests
	- real archive-backed `mdx inspect` on `world/generic/activedoodads/chest01/chest01.mdx` passed and reported `geosets=2`, `CHUNK[5]: id=GEOS`, and real `GEOS[0]` plus `GEOS[1]` lines
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `174` passing tests
	- this is still summary-only classic geoset ownership, not full mesh decode, geoset-animation ownership, skinning parity, or runtime render cutover

### Mar 28, 2026 - Shared `MDX` `PIVT` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` past the new `SEQS` summary layer so `MdxSummaryReader` now also exposes `PIVT` pivot-table coverage.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxPivotPointSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `PivotPoints` and `PivotPointCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with strict `PIVT` `12`-byte-entry summary support and legacy-matching invalid-size failure behavior
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `PIVT[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with synthetic pivot fixtures and optional real pivot-positive archive coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `6` passing tests
	- real archive-backed `mdx inspect` on `world/generic/activedoodads/chest01/chest01.mdx` passed and reported `pivotPoints=6`, `CHUNK[8]: id=PIVT`, and real `PIVT[n]` lines
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `172` passing tests
	- this is still summary-only pivot ownership, not bone binding, helper or emitter placement parity, or runtime node transform ownership

### Mar 28, 2026 - Shared `MDX` `SEQS` Summary Slice Landed

- Extended the shared `MDX` seam in `wow-viewer` beyond `TEXS` and `MTLS` so `MdxSummaryReader` now also exposes first `SEQS` summary coverage.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxSequenceSummary`
	- extended `WowViewer.Core.Mdx.MdxSummary` with `Sequences` and `SequenceCount`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with counted legacy named `128/132/136/140`-byte `SEQS` summary support, counted named `0x8C` support, and the numeric-heavy `0x8C` `0.9.0` path as summary-only metadata
	- updated `WowViewer.Tool.Inspect mdx inspect` to print `SEQS[n]` lines
	- extended `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs` with synthetic sequence fixtures and optional real animated-archive coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter MdxSummaryReaderTests` passed on Mar 28, 2026 with `4` passing tests
	- real archive-backed `mdx inspect` on `world/generic/passivedoodads/particleemitters/greengroundfog.mdx` passed and reported `sequences=1`, `CHUNK[2]: id=SEQS`, and `SEQS[0]: name=Stand ... blendTime=150`
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `170` passing tests
	- this is still summary-only animation ownership, not animation-track parsing, bone/geoset ownership, or runtime viewer playback parity

### Mar 28, 2026 - Shared Root-ADT Plus `_tex0` Texture Reader And Broadened JSON Export Landed

- Generalized the earlier `_tex0`-only terrain-detail seam into a shared ADT texture reader for root `ADT` and `_tex0.adt` files.
- Landed pieces:
	- replaced `_tex0`-specific `AdtTex*` contracts with `AdtTextureChunkLayer`, `AdtTextureChunk`, and `AdtTextureFile`
	- added `WowViewer.Core.IO.Maps.AdtTextureReader` for shared root or `_tex0` per-chunk layer-table and decoded-alpha reads
	- updated `AdtMcalSummaryReader` to aggregate through the generalized shared reader
	- broadened `WowViewer.Tool.Converter export-tex-json` so it now accepts `file.adt` and `file_tex0.adt`
	- updated inspect `--dump-tex-chunks` to consume the generalized shared reader
	- replaced `_tex0`-only regression coverage with `AdtTextureReaderTests`, including synthetic root plus real development-root coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AdtTextureReaderTests|AdtMcalSummaryReaderTests|AdtMcalDecoderTests|AdtSummaryReaderTests|AdtMcnkSummaryReaderTests|MapFileSummaryReaderTests|WowFileDetectorTests"` passed on Mar 28, 2026 with `37` passing tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `168` passing tests
	- real-data converter export now passes on both `development_0_0.adt` and `development_0_0_tex0.adt`, and root `--output` file-write also passed
	- the fixed development root dataset still does not positively prove real root-layer payload decode because its texture layering lives in `_tex0.adt`; that proof remains synthetic in this slice

### Mar 28, 2026 - Thin `_tex0` JSON Export Surface Landed In `WowViewer.Tool.Converter`

- Added the first real converter/export consumer for the new shared `_tex0` terrain seam.
- Landed pieces:
	- updated `WowViewer.Tool.Converter` with `export-tex-json --input <file_tex0.adt> [--output <report.json>]`
	- validated `_tex0` inputs through shared `WowFileDetector`
	- serialized shared `AdtTexReader` output directly to stdout or an output file instead of adding tool-local parsing
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `166` passing tests
	- real-data stdout export on `development_0_0_tex0.adt` passed and printed JSON with shared `SourcePath`, `TextureNames`, and `Chunks`
	- real-data file export on `development_0_0_tex0.adt` passed and wrote `wowviewer-development_0_0_tex0.json` under `%TEMP%`
	- this is still a thin JSON export over the shared read seam, not a broader terrain conversion workflow or a WoW terrain write path

### Mar 28, 2026 - Shared `_tex0` Per-Chunk Layer And Decoded Alpha Reader Landed

- Added the next terrain shared-I/O slice in `wow-viewer` after split-family routing plus aggregate `MCAL` summary.
- Landed pieces:
	- added `WowViewer.Core.Maps.AdtTexChunkLayer`
	- added `WowViewer.Core.Maps.AdtTexChunk`
	- added `WowViewer.Core.Maps.AdtTexFile`
	- added `WowViewer.Core.IO.Maps.AdtTexReader`
	- extended `MapSummaryReaderCommon` with shared `ReadStringEntries(...)`
	- updated `AdtMcalSummaryReader` so `_tex0` summary aggregation now consumes the shared reader instead of duplicating `_tex0` parsing logic
	- updated `WowViewer.Tool.Inspect map inspect` with `--dump-tex-chunks` so `_tex0` reports can print shared per-chunk `MCNK(tex)` and `LAYER` detail lines on demand
	- added synthetic plus real-data coverage in `AdtTexReaderTests`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AdtTexReaderTests|AdtMcalSummaryReaderTests|AdtMcalDecoderTests|AdtSummaryReaderTests|AdtMcnkSummaryReaderTests|MapFileSummaryReaderTests|WowFileDetectorTests"` passed on Mar 28, 2026 with `35` passing tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `166` passing tests
	- real-data `map inspect --dump-tex-chunks` on `test_data/development/World/Maps/development/development_0_0_tex0.adt` passed and reported `ADT TEX detail: textures=5 chunks=256`, preserved aggregate `decodedLayers=519`, and printed real per-layer `Compressed` plus `BigAlpha` outputs
	- this is still not runtime `MdxViewer` terrain signoff and not a shared port of Cataclysm residual-alpha synthesis or chunk-edge stitching

### Mar 28, 2026 - Shared `ADT` Split-Family Routing And Direct `MCAL` Decode Summary Seams Landed

- Added the first terrain-focused shared ownership slice in `wow-viewer` under the broader full-format-ownership reset.
- Landed pieces:
	- added `WowViewer.Core.Maps.AdtTileFamily`
	- added `WowViewer.Core.Maps.AdtTextureLayerDescriptor`
	- added `WowViewer.Core.Maps.AdtMcalDecodeProfile`
	- added `WowViewer.Core.Maps.AdtMcalAlphaEncoding`
	- added `WowViewer.Core.Maps.AdtMcalDecodedLayer`
	- added `WowViewer.Core.Maps.AdtMcalSummary`
	- added `WowViewer.Core.IO.Maps.AdtTileFamilyResolver`
	- added `WowViewer.Core.IO.Maps.AdtMcalDecoder`
	- added `WowViewer.Core.IO.Maps.AdtMcalSummaryReader`
	- updated `WowViewer.Tool.Inspect map inspect` to print shared ADT family routing and `MCAL` decode summary lines
	- updated `MapFileKind` plus `MapFileSummaryReader` so `_lod.adt` is preserved as `AdtLod`
	- added focused synthetic and real-data coverage in:
		- `AdtTileFamilyResolverTests`
		- `AdtMcalDecoderTests`
		- `AdtMcalSummaryReaderTests`
		- plus adjacent map-summary and detector assertions
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AdtMcalDecoderTests|AdtMcalSummaryReaderTests|AdtTileFamilyResolverTests|AdtSummaryReaderTests|AdtMcnkSummaryReaderTests|MapFileSummaryReaderTests|WowFileDetectorTests"` passed on Mar 28, 2026 with `35` passing tests
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 28, 2026 with `164` passing tests
	- real-data `map inspect` on `test_data/development/World/Maps/development/development_0_0_tex0.adt` passed and reported `overlayLayers=519`, `decodedLayers=519`, `missingPayloadLayers=0`, `compressed=515`, and `bigAlpha=4`
	- this is still not runtime `MdxViewer` terrain signoff and not a full shared port of Cataclysm residual-alpha synthesis or chunk-edge stitching

### Mar 28, 2026 - Full Format Ownership Program Reset Captured

- The migration target for `wow-viewer` was clarified beyond the earlier narrow summary-seam framing.
- New explicit rule:
	- `wow-viewer` must fully re-own every active `MdxViewer` format family as first-party library and tooling behavior.
	- current detector and summary slices are progress, but not closure.
- Added `gillijimproject_refactor/plans/wow_viewer_full_format_ownership_plan_2026-03-28.md` to lock the broader program target, ownership standard, format-family scope, workstreams, and execution order.
- Added `gillijimproject_refactor/plans/wow_viewer_format_parity_matrix_2026-03-28.md` to track the family-by-family gap between active `MdxViewer` behavior and `wow-viewer` ownership.
- Updated the shared-I/O plan and continuity docs so future sessions do not mistake current `BLP`, `MDX`, `WMO`, `ADT`, or `WDT` summary ownership for the final migration target.
- This was planning and continuity work only. No new implementation or validation was performed in this reset itself.

### Mar 28, 2026 - Shared `MDX` Top-Level Plus `TEXS` And `MTLS` Summary Seams And Consumer Validation Landed

- Added the first shared `MDX` model-family seam in `wow-viewer` and immediately validated it through the existing non-UI `MdxViewer` probe path.
- Landed pieces:
	- added `WowViewer.Core.Mdx.MdxChunkIds`
	- added `WowViewer.Core.Mdx.MdxChunkSummary`
	- added `WowViewer.Core.Mdx.MdxTextureSummary`
	- added `WowViewer.Core.Mdx.MdxMaterialLayerSummary`
	- added `WowViewer.Core.Mdx.MdxMaterialSummary`
	- added `WowViewer.Core.Mdx.MdxSummary`
	- extended `WowViewer.Core.IO.Mdx.MdxSummaryReader` with shared `TEXS` texture-table summary support and narrow `MTLS` material-layer summary support
	- updated `WowViewer.Tool.Inspect` with `mdx inspect --input <file.mdx>` and `mdx inspect --archive-root <dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>]`
	- added synthetic plus real standard-archive coverage in `wow-viewer/tests/WowViewer.Core.Tests/MdxSummaryReaderTests.cs`
	- extended `gillijimproject_refactor/src/MdxViewer/AssetProbe.cs` so the consumer probe now prints shared `MDX` summary signals for the probed model bytes, including `TEXS` texture-count and first-path signals plus compact `MTLS` material-layer summary signals
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "MdxSummaryReaderTests|WowFileDetectorTests"` passed on Mar 27, 2026 with `11` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- mdx inspect --archive-root "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" --virtual-path world/generic/activedoodads/chest01/chest01.mdx --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed and reported `version=1300`, `model=Chest01`, `textures=2`, `materials=2`, and real `TEXS` plus `MTLS` layer lines on the archive-backed asset
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --probe-mdx "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" "world/generic/activedoodads/chest01/chest01.mdx" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed and now prints `SharedMDX` plus real first-texture `TEXS` paths and first-layer `MTLS` signals alongside the earlier `SharedBLP` signals
	- this is still top-level `MDX` plus narrow `TEXS` and `MTLS` summary ownership and consumer validation only, not runtime viewer model-path signoff, deep material semantics, animation-track parity, or `M2` parity

### Mar 27, 2026 - `MdxViewer` Compatibility Validation Now Exercises Shared `BLP` Summary Reads

- Updated `gillijimproject_refactor/src/MdxViewer/AssetProbe.cs` so the active viewer consumer now uses the latest shared `wow-viewer` `BLP` seam during non-UI asset probing.
- Landed pieces:
	- added shared `WowFileDetector` output for probed model and texture bytes
	- added shared `BlpSummaryReader` output for resolved texture files classified as `Blp`
	- kept `SereniaBLPLib` decode in place for width and alpha inspection, so the probe now shows both shared-header signals and legacy decode signals together
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 27, 2026 with existing warnings
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -- --probe-mdx "i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data" "world/generic/activedoodads/chest01/chest01.mdx" --listfile "i:/parp/parp-tools/wow-viewer/libs/wowdev/wow-listfile/listfile.txt"` passed on Mar 27, 2026 and printed real `SharedBLP` summary lines for both chest textures
	- this is not automated test coverage and not runtime viewer signoff

### Mar 27, 2026 - Shared `BLP` Header Summary Seam And Inspect Surface Landed

- Added the first real texture-family shared-I/O seam in `wow-viewer` instead of stopping at cross-family file detection.
- Landed pieces:
	- added `WowViewer.Core.Blp.BlpFormat`
	- added `WowViewer.Core.Blp.BlpCompressionType`
	- added `WowViewer.Core.Blp.BlpPixelFormat`
	- added `WowViewer.Core.Blp.BlpMipMapEntry`
	- added `WowViewer.Core.Blp.BlpSummary`
	- added `WowViewer.Core.IO.Blp.BlpSummaryReader`
	- updated `WowViewer.Tool.Inspect` with `blp inspect --input <file.blp>` and `blp inspect --archive-root <dir> --virtual-path <path/to/file.blp> [--listfile <listfile.txt>]`
	- added synthetic and archive-backed real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/BlpSummaryReaderTests.cs`
	- extended `wow-viewer/tests/WowViewer.Core.Tests/WowFileDetectorTests.cs` with a synthetic `BLP2` detector case
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "BlpSummaryReaderTests|WowFileDetectorTests"` passed on Mar 27, 2026 with `11` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- blp inspect --archive-root i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data --virtual-path interface/minimap/minimaparrow.blp` now reports a real archive-backed `BLP2` summary and per-mip coverage through the shared reader
	- this is still header-summary ownership only, not full pixel decode, write support, or model-family parity

### Mar 27, 2026 - Shared `MOLT` Per-Light Detail Seam And Opt-In Inspect Dump Landed

- Added the next narrow WMO follow-up after the root-light summary fix: shared per-entry `MOLT` detail ownership plus an inspect flag that exposes those raw fields on demand.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoLightDetail`
	- added `WowViewer.Core.IO.Wmo.WmoLightReaderCommon`
	- added `WowViewer.Core.IO.Wmo.WmoLightDetailReader`
	- updated `WowViewer.Core.IO.Wmo.WmoLightSummaryReader` so summary aggregation reuses the shared detail decode path instead of duplicating `MOLT` layout logic
	- updated `WowViewer.Tool.Inspect wmo inspect` with `--dump-lights` so root WMO reports can print `MOLT[n]` lines for Alpha `32`-byte and standard `48`-byte entries without changing the default summary output
	- added synthetic regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoLightDetailReaderTests.cs`
	- extended `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs` with real Alpha and standard per-light detail assertions on Ironforge
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoLightSummaryReaderTests|WmoLightDetailReaderTests|Read_IronforgeAlphaPerAssetMpq_ProducesExpectedRootLightSummary|Read_IronforgeAlphaPerAssetMpq_RootLightDetails_UseLegacyLayout|Read_IronforgeStandard060_RootLightSummary_UsesStandardTailAttenuationOffsets|Read_IronforgeStandard060_RootLightDetails_ExposeRawStandardLayoutFields"` passed on Mar 27, 2026 with `8` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --archive-root i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data --virtual-path world/wmo/khazmodan/cities/ironforge/ironforge.wmo --dump-lights` now prints real standard per-light `MOLT[n]` lines with raw `headerFlagsWord` and quaternion rotation values
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/KhazModan/Cities/Ironforge/ironforge.wmo.MPQ --dump-lights` now prints real Alpha per-light `MOLT[n]` lines with legacy `32`-byte entry sizing and no later-layout fields
	- this is still a shared detail-read and inspect-surface slice, not broader light-behavior interpretation or a write path

### Mar 27, 2026 - WMO Group Optional `MOLR`, `MOBN`, `MOBR`, And `MOBN->MOBR` Summary Slice Landed

- Added the next narrow shared WMO group slice in `wow-viewer` for the remaining low-risk optional group chunks plus one first linkage seam between BSP nodes and BSP face refs.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupLightRefSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupLightRefSummaryReader`
	- added `WowViewer.Core.Wmo.WmoGroupBspNodeSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupBspNodeSummaryReader`
	- added `WowViewer.Core.Wmo.WmoGroupBspFaceSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupBspFaceSummaryReader`
	- added `WowViewer.Core.Wmo.WmoGroupBspFaceRangeSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupBspFaceRangeSummaryReader`
	- extended `WmoGroupSummary` and `WmoEmbeddedGroupSummary` so shared consumers can see `lightRefs`, `bspNodes`, and `bspFaceRefs` without their own chunk scans
	- updated `WowViewer.Tool.Inspect wmo inspect` so group files now print `MOLR`, `MOBN`, `MOBR`, and `MOBN->MOBR`, and Alpha monolithic root aggregate output now includes optional-chunk totals
	- added synthetic regression coverage for all four new seams and real-data `castle01.wmo.MPQ` coverage for embedded BSP totals and embedded-group reader replay
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoRealDataTests|WmoEmbeddedGroupSummaryReaderTests|WmoGroupSummaryReaderTests|WmoGroupLightRefSummaryReaderTests|WmoGroupBspNodeSummaryReaderTests|WmoGroupBspFaceSummaryReaderTests|WmoGroupBspFaceRangeSummaryReaderTests"` passed on Mar 27, 2026 with `9` passing tests
	- real `castle01.wmo.MPQ` inspect now reports `lightRefs=0`, `bspNodes=583`, and `bspFaceRefs=6716` across its embedded Alpha root groups
	- this is still summary and range coverage only, not full BSP traversal or consumer cutover

### Mar 27, 2026 - Alpha Root Per-Embedded-Group Inspect Routing Landed For `MOBN`, `MOBR`, And `MOBN->MOBR`

- Added the next narrow follow-up after the embedded-group aggregate: real per-group inspect routing for Alpha monolithic roots so the existing shared BSP summaries are visible on each embedded `MOGP` instead of only in totals.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoEmbeddedGroupDetail`
	- added `WowViewer.Core.IO.Wmo.WmoEmbeddedGroupDetailReader`
	- extended the group optional readers with internal `ReadMogpPayload(...)` entry points for embedded-root reuse
	- updated `WowViewer.Tool.Inspect wmo inspect` so Alpha roots now print `MOGP(root)[n]`, `MOBN(root)[n]`, `MOBR(root)[n]`, and `MOBN->MOBR(root)[n]`
	- added synthetic regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupDetailReaderTests.cs`
	- extended real-data `castle01.wmo.MPQ` coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupDetailReaderTests|WmoRealDataTests|WmoEmbeddedGroupSummaryReaderTests|WmoGroupBspNodeSummaryReaderTests|WmoGroupBspFaceSummaryReaderTests|WmoGroupBspFaceRangeSummaryReaderTests"` passed on Mar 27, 2026 with `8` passing tests
	- real `castle01.wmo.MPQ` inspect now prints per-group BSP lines for both embedded groups, including `127` or `456` `MOBN` nodes and `1145` or `5571` `MOBR` refs respectively
	- this is still shared inspect routing for the current BSP summaries, not full per-group routing for every embedded subchunk family

### Mar 27, 2026 - Alpha Root Per-Embedded-Group Inspect Routing Expanded To Existing Shared Group Summaries

- Expanded the earlier BSP-only root detail seam so Alpha monolithic roots now reuse the already-owned shared group readers for additional geometry and metadata lines instead of only printing BSP summaries per embedded group.
- Landed pieces:
	- extended `WowViewer.Core.Wmo.WmoEmbeddedGroupDetail` to carry `MLIQ`, `MOBA`, `MOPY`, `MOTV`, `MOCV`, `MODR`, `MOVI` or `MOIN`, `MOVT`, and `MONR` summaries
	- added internal `ReadMogpPayload(...)` entry points to the matching shared group readers
	- updated `WowViewer.Core.IO.Wmo.WmoEmbeddedGroupDetailReader` to populate those additional summaries from root-embedded `MOGP` payloads
	- updated `WowViewer.Tool.Inspect wmo inspect` so Alpha roots now print `MONR(root)[n]`, `MOVT(root)[n]`, `MOVI(root)[n]` or `MOIN(root)[n]`, `MODR(root)[n]`, `MOCV(root)[n]`, `MOTV(root)[n]`, `MOPY(root)[n]`, and `MOBA(root)[n]`, with `MLIQ(root)[n]` ready when present
	- extended synthetic and real-data regression coverage in `WmoEmbeddedGroupDetailReaderTests` and `WmoRealDataTests`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupDetailReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `4` passing tests
	- real `castle01.wmo.MPQ` inspect now prints positive per-group lines for `MONR`, `MOVT`, `MOIN`, `MOCV`, `MOTV`, `MOPY`, and `MOBA`, plus `MODR` on the embedded group that actually has doodad refs
	- `castle01.wmo.MPQ` still does not positively prove per-group `MOLR` or `MLIQ`, because those embedded groups remain zero-ref or liquid-free on this asset

### Mar 27, 2026 - `ironforge.wmo.MPQ` Added Positive Real Coverage For Per-Group `MOLR` And `MLIQ`

- Used real `ironforge.wmo.MPQ` as the next Alpha validation asset because it exercises the remaining per-group light-ref and liquid paths that `castle01.wmo.MPQ` does not.
- Landed pieces:
	- added `WmoRealDataTests.Read_IronforgeAlphaPerAssetMpq_EmbeddedGroupDetailsExposePositiveLightAndLiquidSignals`
	- updated `WowViewer.Tool.Inspect wmo inspect` so an invalid optional `MOLT` summary no longer aborts the whole report before later root or embedded-group lines print
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoRealDataTests"` passed on Mar 27, 2026 with `4` passing tests
	- real `ironforge.wmo.MPQ` inspect now reaches positive per-group `MOLR(root)[n]` lines such as groups `120`, `121`, `123`, `124`, and `125`, plus a positive `MLIQ(root)[127]` line with `liquidType=Magma`
	- this validates the per-group shared detail seam on a second real Alpha monolithic root, but it does not claim Ironforge `MOLT` root-light parsing is fully understood yet

### Mar 27, 2026 - Shared `MOLT` Root-Light Summary Now Reads Real Alpha `ironforge.wmo.MPQ`

- Fixed the narrow real-data gap exposed by Ironforge: the shared `MOLT` reader now handles legacy 32-byte Alpha root-light entries instead of assuming only the later 48-byte layout.
- Landed pieces:
	- updated `WowViewer.Core.IO.Wmo.WmoLightSummaryReader` with version-aware `MOLT` entry-size inference
	- extended `WmoLightSummary` and inspect output with `attenStartRange`, a raw later-layout `headerFlagsWord` summary from bytes `2..3`, and later-layout rotation metrics (`rotationEntries`, `nonIdentityRotations`, `rotationLenRange`)
	- extended `WmoLightSummaryReaderTests` with explicit synthetic `v14` and `v17` coverage
	- extended `WmoRealDataTests` with a direct Ironforge root-light assertion, including positive attenuation-start coverage
	- corrected the standard 48-byte layout after real `0.6.0` archive proof: bytes `2..3` now land as a raw `headerFlagsWord`, quaternion rotation reads from offsets `24..39`, and attenuation reads from offsets `40` and `44`
	- added shared `ArchiveVirtualFileReader` and updated `WowViewer.Tool.Inspect wmo inspect` with `--archive-root` plus `--virtual-path` so standard-archive root WMOs can be inspected without extracting them first
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoLightSummaryReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `7` passing tests
	- real Ironforge inspect now prints `MOLT: payloadBytes=6976 entries=218 distinctTypes=1 attenuated=218 intensityRange=[0.120, 1.000] attenStartRange=[1.306, 8.333] maxAttenEnd=29.611 ...`
	- a real `0.6.0` standard-archive Ironforge regression now also proves `48`-byte `MOLT` uses a non-zero `headerFlagsWord` of `0x0101` at bytes `2..3`, quaternion rotation at offsets `24..39`, and attenuation at offsets `40` and `44`
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --archive-root i:/parp/parp-tools/wow-viewer/testdata/0.6.0/World of Warcraft/Data --virtual-path world/wmo/khazmodan/cities/ironforge/ironforge.wmo` now also reports the same standard root-light summary through the CLI, including `headerFlagsWordRange=[0x0101, 0x0101]`, `headerFlagsWordDistinct=1`, `headerFlagsWordNonZero=218`, `rotationEntries=218`, `nonIdentityRotations=218`, and `rotationLenRange=[1.118, 1.118]`
	- this is still a shared semantic-summary slice, not deeper light behavior or a write path
	- the per-light inspect dump has now landed, so the clean next step is to test more real standard roots for `headerFlagsWord` variability instead of re-opening the already-settled Ironforge attenuation and rotation offsets

### Mar 27, 2026 - Alpha `MOGI -> MOGP(root)` Linkage Summary Landed

- Added the next narrow Alpha follow-up by linking root `MOGI` entries to embedded top-level `MOGP` blocks by ordinal pairing.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoEmbeddedGroupLinkageSummary`
	- added `WowViewer.Core.IO.Wmo.WmoEmbeddedGroupLinkageSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so Alpha monolithic roots now print an `MOGI->MOGP(root)` linkage line
	- added synthetic regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupLinkageSummaryReaderTests.cs`
	- extended real-data coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `130` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupLinkageSummaryReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `2` targeted passing tests
	- real `castle01.wmo.MPQ` inspect now reports `flagMatches=0` and `boundsMatches=2` for the paired Alpha group-info vs embedded-group linkage surface
	- this is still linkage-summary ownership, not detailed per-group route selection or remediation logic

### Mar 27, 2026 - Alpha Monolithic Root Embedded-Group Aggregate Summary Landed

- Added the next narrow Alpha follow-up after `MOMO` root support by summarizing the embedded top-level `MOGP` group blocks that still live in monolithic 0.5.3 root files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoEmbeddedGroupSummary`
	- added `WowViewer.Core.IO.Wmo.WmoEmbeddedGroupSummaryReader`
	- reused `WmoGroupSummaryReader` logic through a shared internal `MOGP` payload helper instead of duplicating group-header interpretation
	- updated `WowViewer.Tool.Inspect wmo inspect` so Alpha monolithic roots now print an `MOGP(root)` aggregate line when embedded groups are present
	- added synthetic regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoEmbeddedGroupSummaryReaderTests.cs`
	- extended real-data coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `129` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WmoEmbeddedGroupSummaryReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `2` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports embedded-group aggregate metrics
	- this is still aggregate ownership, not per-embedded-group detailed Alpha consumer routing

### Mar 27, 2026 - Alpha MOMO Root WMO Support And Real 0.5.3 `.wmo.MPQ` Validation Landed

- Added shared Alpha root-WMO support for the `MOMO` container so the existing root-summary stack can read real 0.5.3 monolithic WMO roots.
- Landed pieces:
	- added shared `MOMO` chunk id in `WowViewer.Core.Wmo.WmoChunkIds`
	- updated `WowViewer.Core.IO.Files.WowFileDetector` so `MVER` + `MOMO` is classified as `Wmo`
	- expanded `WowViewer.Core.IO.Wmo.WmoRootReaderCommon` to flatten Alpha `MOMO` subchunks into a root-chunk view reusable by later shared readers
	- moved the main root-summary readers onto `WmoRootReaderCommon`, including the semantic summary reader, group-info reader, material reader, texture-table reader, doodad-name reader, doodad-set reader, doodad-placement reader, group-name table reader, skybox reader, and the shared portal-root helper
	- loosened `WowViewer.Core.Wmo.WmoGroupInfoSummary` so negative `MOGI` name offsets from real Alpha data are treated as valid summary signals instead of rejected input
	- improved `WowViewer.Core.IO.Files.AlphaArchiveReader` internal-name candidate generation for non-map `World\...` paths and direct `.MPQ` inputs
	- updated `WowViewer.Tool.Inspect wmo inspect` to load `.wmo.MPQ` inputs through the shared Alpha archive fallback and then run the shared stream-based readers
	- added real-data regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/WmoRealDataTests.cs`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `128` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "AlphaArchiveReaderTests|WmoRealDataTests"` passed on Mar 27, 2026 with `7` targeted passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/wow-viewer/testdata/0.5.3/tree/World/wmo/Azeroth/Buildings/Castle/castle01.wmo.MPQ` passed on Mar 27, 2026 and now reports real Alpha-era root-WMO semantic lines directly from the per-asset MPQ
	- this is still root-summary ownership; it is not yet full Alpha monolithic group-consumer ownership

### Mar 27, 2026 - Batched Root WMO Portal Linkage Summary Slices For MOPT->MOPV, MOPR->MOPT, And MOPR->MOGI Landed

- Added a portal-linkage focused batched root-WMO landing in `wow-viewer` after the earlier raw portal summary slice.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoPortalVertexRangeSummary`
	- added `WowViewer.Core.IO.Wmo.WmoPortalVertexRangeSummaryReader`
	- added `WowViewer.Core.Wmo.WmoPortalRefRangeSummary`
	- added `WowViewer.Core.IO.Wmo.WmoPortalRefRangeSummaryReader`
	- added `WowViewer.Core.Wmo.WmoPortalGroupRangeSummary`
	- added `WowViewer.Core.IO.Wmo.WmoPortalGroupRangeSummaryReader`
	- expanded `WmoRootReaderCommon` with optional chunk reads to avoid false-positive optional root-chunk lookups
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes dedicated portal-linkage lines for `MOPT->MOPV`, `MOPR->MOPT`, and `MOPR->MOGI`
	- added synthetic regression coverage for all three portal-linkage seams plus a missing-`MOVV` guard regression
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `125` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `94` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-portal-linkage-batch-test.wmo` passed on Mar 27, 2026 for a synthetic root-portal-linkage smoke case
	- this is still summary work, not full portal topology validation or runtime culling ownership

### Mar 27, 2026 - Batched Root WMO Visibility Summary Slices For MOVV, MOVB, And MOVB->MOVV Landed

- Added another batched root-WMO landing in `wow-viewer` for visibility-owner chunks plus their first narrow linkage seam.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoVisibleVertexSummary`
	- added `WowViewer.Core.IO.Wmo.WmoVisibleVertexSummaryReader`
	- added `WowViewer.Core.Wmo.WmoVisibleBlockSummary`
	- added `WowViewer.Core.IO.Wmo.WmoVisibleBlockSummaryReader`
	- added `WowViewer.Core.Wmo.WmoVisibleBlockReferenceSummary`
	- added `WowViewer.Core.IO.Wmo.WmoVisibleBlockReferenceSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes dedicated `MOVV`, `MOVB`, and `MOVB->MOVV` semantic lines when those chunks are present
	- added synthetic regression coverage for all three seams
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `121` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `90` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-visibility-batch-test.wmo` passed on Mar 27, 2026 for a synthetic root-visibility smoke case
	- this is still summary work, not runtime visibility-volume ownership or write support

### Mar 27, 2026 - Batched Root WMO Linkage Summary Slices For MODD->MODN, MOGI->MOGN, And MODS->MODD Landed

- Added a linkage-focused batched root-WMO landing in `wow-viewer` instead of another raw-payload-only step.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoDoodadNameReferenceSummary`
	- added `WowViewer.Core.IO.Wmo.WmoDoodadNameReferenceSummaryReader`
	- added `WowViewer.Core.Wmo.WmoGroupNameReferenceSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupNameReferenceSummaryReader`
	- added `WowViewer.Core.Wmo.WmoDoodadSetRangeSummary`
	- added `WowViewer.Core.IO.Wmo.WmoDoodadSetRangeSummaryReader`
	- added shared `WowViewer.Core.IO.Wmo.WmoRootReaderCommon`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes dedicated linkage lines for `MODD->MODN`, `MOGI->MOGN`, and `MODS->MODD`
	- added synthetic regression coverage for all three linkage seams
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `118` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `87` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-linkage-batch-test.wmo` passed on Mar 27, 2026 for a synthetic root-linkage smoke case
	- this is still summary work, not full consumer cutover or write support

### Mar 27, 2026 - Batched Root WMO Metadata Slices For MOLT, MFOG, And MCVP Landed

- Added another batched root-WMO metadata landing in `wow-viewer` for lights, fog, and an opaque trailing chunk.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoLightSummary`
	- added `WowViewer.Core.IO.Wmo.WmoLightSummaryReader`
	- added `WowViewer.Core.Wmo.WmoFogSummary`
	- added `WowViewer.Core.IO.Wmo.WmoFogSummaryReader`
	- added `WowViewer.Core.Wmo.WmoOpaqueChunkSummary`
	- added `WowViewer.Core.IO.Wmo.WmoOpaqueChunkSummaryReader`
	- expanded shared `WmoChunkIds` with `MOLT`, `MFOG`, `MCVP`, `MOVV`, and `MOVB`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes dedicated `MOLT`, `MFOG`, and `MCVP` semantic lines when present
	- added synthetic regression coverage for all three seams
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `115` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `84` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-meta-batch-test.wmo` passed on Mar 27, 2026 for a synthetic root-metadata smoke case
	- this is still summary work, not deeper light/fog rendering semantics or opaque `MCVP` ownership

### Mar 27, 2026 - Batched Root WMO Portal Summary Slices For MOPV, MOPT, And MOPR Landed

- Added a second batched root-WMO landing in `wow-viewer` for portal-owner chunks.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoPortalVertexSummary`
	- added `WowViewer.Core.IO.Wmo.WmoPortalVertexSummaryReader`
	- added `WowViewer.Core.Wmo.WmoPortalInfoSummary`
	- added `WowViewer.Core.IO.Wmo.WmoPortalInfoSummaryReader`
	- added `WowViewer.Core.Wmo.WmoPortalRefSummary`
	- added `WowViewer.Core.IO.Wmo.WmoPortalRefSummaryReader`
	- expanded shared `WmoChunkIds` with `MOPV`, `MOPT`, and `MOPR`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes dedicated `MOPV`, `MOPT`, and `MOPR` semantic lines when portal data is present
	- added synthetic regression coverage for all three portal seams
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `112` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `81` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-portals-test.wmo` passed on Mar 27, 2026 for a synthetic root-portal smoke case
	- this is still summary work, not root-to-group portal routing ownership or write support

### Mar 27, 2026 - Batched Root WMO Summary Slices For MODD, MOGN, And MOSB Landed

- Added a batched three-slice root-WMO landing in `wow-viewer` instead of another single-slice step.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoDoodadPlacementSummary`
	- added `WowViewer.Core.IO.Wmo.WmoDoodadPlacementSummaryReader`
	- added `WowViewer.Core.Wmo.WmoGroupNameTableSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupNameTableSummaryReader`
	- added `WowViewer.Core.Wmo.WmoSkyboxSummary`
	- added `WowViewer.Core.IO.Wmo.WmoSkyboxSummaryReader`
	- expanded shared `WmoChunkIds` with `MOGN` and `MOSB`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes dedicated `MODD`, `MOGN`, and `MOSB` semantic lines when present
	- added synthetic regression coverage for all three seams

- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `109` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `78` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-batch-test.wmo` passed on Mar 27, 2026 for a synthetic batched root-WMO smoke case
	- this is still summary work, not root-table linkage or write support

### Mar 27, 2026 - Shared WMO Root Doodad-Set Semantic Summary Slice Landed

- Added the next narrow WMO root seam in `wow-viewer`: shared `MODS` doodad-set semantic summary.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoDoodadSetSummary`
	- added `WowViewer.Core.IO.Wmo.WmoDoodadSetSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes a dedicated `MODS` semantic line when doodad sets are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadSetSummaryReaderTests.cs` for synthetic empty and non-empty `MODS` coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `106` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `75` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-mods-test.wmo` passed on Mar 27, 2026 for a synthetic root doodad-set smoke case
	- this is still semantic summary work, not `MODD` linkage or write support

### Mar 27, 2026 - Shared WMO Root Doodad-Name Table Semantic Summary Slice Landed

- Added the next narrow WMO root seam in `wow-viewer`: shared `MODN` doodad-name-table semantic summary.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoDoodadNameTableSummary`
	- added `WowViewer.Core.IO.Wmo.WmoDoodadNameTableSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes a dedicated `MODN` semantic line when doodad-name tables are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoDoodadNameTableSummaryReaderTests.cs` for synthetic mixed `.mdx` or `.m2` `MODN` coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `105` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `74` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-modn-test.wmo` passed on Mar 27, 2026 for a synthetic root doodad-name smoke case
	- this is still semantic summary work, not `MODD` linkage or write support

### Mar 27, 2026 - Shared WMO Root Texture-Table Semantic Summary Slice Landed

- Added the next narrow WMO root seam in `wow-viewer`: shared `MOTX` texture-table semantic summary.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoTextureTableSummary`
	- added `WowViewer.Core.IO.Wmo.WmoTextureTableSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes a dedicated `MOTX` semantic line when texture tables are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoTextureTableSummaryReaderTests.cs` for synthetic mixed-extension `MOTX` coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `104` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `73` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-motx-test.wmo` passed on Mar 27, 2026 for a synthetic root texture-table smoke case
	- this is still semantic summary work, not `MOMT` offset resolution or write support

### Mar 27, 2026 - Shared WMO Root Material Semantic Summary Slice Landed

- Added the next narrow WMO root seam in `wow-viewer`: shared `MOMT` material semantic summary.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoMaterialSummary`
	- added `WowViewer.Core.IO.Wmo.WmoMaterialSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes a dedicated `MOMT` semantic line when material entries are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoMaterialSummaryReaderTests.cs` for synthetic standard and legacy `MOMT` coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `103` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `72` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-momt-test.wmo` passed on Mar 27, 2026 for a synthetic root-material smoke case
	- this is still semantic summary work, not `MOTX` resolution or write support

### Mar 27, 2026 - Shared WMO Root Group-Info Semantic Summary Slice Landed

- Added the next narrow WMO root seam in `wow-viewer`: shared `MOGI` group-info semantic summary.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupInfoSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupInfoSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so root-WMO output now includes a dedicated `MOGI` semantic line when group info is present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupInfoSummaryReaderTests.cs` for synthetic standard and legacy `MOGI` coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `101` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `70` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-root-mogi-test.wmo` passed on Mar 27, 2026 for a synthetic root-group-info smoke case
	- this is still semantic summary work, not `MOGN` name resolution or write support

### Mar 27, 2026 - Shared WMO Group Normal Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MONR` normal semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupNormalSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupNormalSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MONR` semantic line when normal payloads are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupNormalSummaryReaderTests.cs` for synthetic component-range and near-unit coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `99` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `68` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-normal-test.wmo` passed on Mar 27, 2026 for a synthetic normal smoke case
	- this is still semantic summary work, not tangent-space ownership or write support

### Mar 27, 2026 - Shared WMO Group Vertex Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MOVT` vertex semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupVertexSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupVertexSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MOVT` semantic line when vertex payloads are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupVertexSummaryReaderTests.cs` for synthetic vertex-bound coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `98` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `67` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-vertex-test.wmo` passed on Mar 27, 2026 for a synthetic vertex smoke case
	- this is still semantic summary work, not topology linkage or write support

### Mar 27, 2026 - Shared WMO Group Index Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MOVI` or `MOIN` index semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupIndexSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupIndexSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MOVI` or `MOIN` semantic line when index payloads are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupIndexSummaryReaderTests.cs` for synthetic `MOVI` and `MOIN` coverage including a degenerate-triangle case
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `97` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `66` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-index-test.wmo` passed on Mar 27, 2026 for a synthetic index smoke case
	- this is still semantic summary work, not topology ownership or write support

### Mar 27, 2026 - Shared WMO Group Doodad-Ref Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MODR` doodad-ref semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupDoodadRefSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupDoodadRefSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MODR` semantic line when doodad refs are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupDoodadRefSummaryReaderTests.cs` for synthetic duplicate-ref coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `95` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `64` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-doodadref-test.wmo` passed on Mar 27, 2026 for a synthetic doodad-ref smoke case
	- this is still semantic summary work, not root-linkage ownership or write support

### Mar 27, 2026 - Shared WMO Group Vertex-Color Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MOCV` vertex-color semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupVertexColorSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupVertexColorSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MOCV` semantic line when vertex colors are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupVertexColorSummaryReaderTests.cs` for synthetic primary plus extra-set color coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `94` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `63` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-color-test.wmo` passed on Mar 27, 2026 for a synthetic vertex-color smoke case
	- this is still semantic summary work, not runtime lighting interpretation or write support

### Mar 27, 2026 - Shared WMO Group UV Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MOTV` UV semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupUvSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupUvSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MOTV` semantic line when UV data is present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupUvSummaryReaderTests.cs` for synthetic primary plus extra-set UV coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `93` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `62` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-uv-test.wmo` passed on Mar 27, 2026 for a synthetic UV smoke case
	- this is still semantic summary work, not runtime UV selection or write support

### Mar 27, 2026 - Shared WMO Group Face-Material Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MOPY` face-material semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupFaceMaterialSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupFaceMaterialSummaryReader`
	- extended shared `WmoGroupReaderCommon` with shared `MOPY` entry-size inference
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MOPY` semantic line when face-material entries are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupFaceMaterialSummaryReaderTests.cs` for synthetic v17-style and v16-style `MOPY` coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `92` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `61` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-face-v17-test.wmo` passed on Mar 27, 2026 for a synthetic face-material smoke case
	- this is still semantic summary work, not face-to-batch reconstruction or write support

### Mar 27, 2026 - Shared WMO Group Batch Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MOBA` batch semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupBatchSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupBatchSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MOBA` semantic line when batches are present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupBatchSummaryReaderTests.cs` for synthetic v17-style and v16-style batch coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `90` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `59` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-batch-test.wmo` passed on Mar 27, 2026 for a synthetic group-batch smoke case
	- this is still semantic summary work, not full batch reconstruction or write support

### Mar 27, 2026 - Shared WMO Group Liquid Semantic Summary Slice Landed

- Added the next deeper WMO seam in `wow-viewer`: shared `MLIQ` semantic summary for WMO group files.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoLiquidBasicType`
	- added `WowViewer.Core.Wmo.WmoGroupLiquidSummary`
	- added shared `WowViewer.Core.IO.Wmo.WmoGroupReaderCommon` so WMO group readers share one `MOGP` payload and subchunk scan surface
	- added `WowViewer.Core.IO.Wmo.WmoGroupLiquidSummaryReader`
	- updated `WowViewer.Tool.Inspect wmo inspect` so group-file output now includes a dedicated `MLIQ` semantic line when liquid is present
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupLiquidSummaryReaderTests.cs` for synthetic `MLIQ` height-range and ocean-inference coverage
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `88` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `57` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-liquid-test.wmo` passed on Mar 27, 2026 for a synthetic group-liquid smoke case
	- this is still semantic summary work, not full WMO liquid mesh generation or write support

### Mar 27, 2026 - Shared WMO Group Semantic Summary Slice Landed

- Added the next narrow WMO follow-up seam in `wow-viewer`: shared `MOGP` group semantic summary.
- Landed pieces:
	- added `WowViewer.Core.Wmo.WmoGroupSummary`
	- added `WowViewer.Core.IO.Wmo.WmoGroupSummaryReader`
	- expanded shared `WmoChunkIds` to cover group subchunk ids used by the summary seam
	- updated shared `WowFileDetector` so `MOGP`-first files classify as `WmoGroup`
	- updated `WowViewer.Tool.Inspect wmo inspect` so it prints either a root-WMO or group-WMO report through shared detection
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoGroupSummaryReaderTests.cs` and an additional `WowFileDetectorTests` case for `MOGP`-first detection
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `87` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `56` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-group-summary-test.wmo` passed on Mar 27, 2026 for a synthetic group-file smoke case
	- this is still summary work, not deep WMO group parsing or write support

### Mar 27, 2026 - Shared ADT MCNK Semantic Summary And First WMO Root Summary Slices Landed

- Added the next narrow ADT chunk-internal semantic-summary layer in `wow-viewer` and the first shared WMO root semantic-summary seam.
- Landed ADT pieces:
	- added `WowViewer.Core.Maps.AdtChunkIds`
	- added `WowViewer.Core.Maps.AdtMcnkSummary`
	- added `WowViewer.Core.IO.Maps.AdtMcnkSummaryReader`
	- updated `WowViewer.Tool.Inspect map inspect` to print a shared `MCNK` semantic-summary line for ADT-family files
	- added `wow-viewer/tests/WowViewer.Core.Tests/AdtMcnkSummaryReaderTests.cs` for synthetic root, `_tex0.adt`, and `_obj0.adt` buffers plus real-data `development_0_0.adt`, `development_0_0_tex0.adt`, and `development_0_0_obj0.adt`
- Landed WMO pieces:
	- added `WowViewer.Core.Wmo.WmoChunkIds`
	- added `WowViewer.Core.Wmo.WmoSummary`
	- added `WowViewer.Core.IO.Wmo.WmoSummaryReader`
	- added `wmo inspect --input <file.wmo>` to `WowViewer.Tool.Inspect`
	- added `wow-viewer/tests/WowViewer.Core.Tests/WmoSummaryReaderTests.cs` for a synthetic WMO root summary case
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `84` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `53` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 27, 2026 and now prints the shared ADT `MCNK` semantic summary on real split-texture data
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- wmo inspect --input i:/parp/parp-tools/output/synthetic-wmo-summary-test.wmo` passed on Mar 27, 2026 for a synthetic root-WMO smoke case because no checked-in fixed real WMO file was available in this workspace snapshot
	- this is still semantic summary work, not deep MCNK parsing, group-file WMO parsing, or write support

### Mar 27, 2026 - Shared ADT Semantic Summary Slice Landed

- Added the first shared ADT semantic-summary layer in `wow-viewer` beyond raw chunk inventory.
- Landed pieces:
	- added `WowViewer.Core.Maps.AdtSummary`
	- added `WowViewer.Core.IO.Maps.AdtSummaryReader`
	- added shared `MapSummaryReaderCommon` helper coverage for top-level payload and string-block reads used by both WDT and ADT summary readers
	- expanded `MapChunkIds` with top-level `MAMP`
	- updated `WowViewer.Tool.Inspect map inspect` to print the shared ADT semantic summary for root, `_tex0.adt`, and `_obj0.adt` files
	- added `wow-viewer/tests/WowViewer.Core.Tests/AdtSummaryReaderTests.cs` for synthetic root, `_tex0.adt`, and `_obj0.adt` buffers plus real-data `development_0_0.adt`, `development_0_0_tex0.adt`, and `development_0_0_obj0.adt`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `77` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `46` passing tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_tex0.adt` passed on Mar 27, 2026 and now prints the shared ADT semantic summary on real texture-split data
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_0_0_obj0.adt` passed on Mar 27, 2026 and now prints the shared ADT semantic summary on real object-split data
	- this is still top-level semantic summary work, not deep ADT parsing or write support

### Mar 27, 2026 - Shared WDT Semantic Summary Slice Landed

- Added the first shared WDT semantic-summary layer in `wow-viewer` beyond raw chunk inventory.
- Landed pieces:
	- added `WowViewer.Core.Maps.WdtSummary`
	- added `WowViewer.Core.IO.Maps.WdtSummaryReader`
	- extended the shared WDT seam with standard `MAIN` flag summary metadata instead of flattening every non-zero standard entry to occupancy only
	- added `WowViewer.Core.Maps.WdtMainFlagsSummary` and `WdtMainFlagValueSummary` so standard `MAIN` readers can expose `hasAdt`, `allWater`, `loaded`, unknown-bit, async-id, and distinct-flag distribution signals without taking over tile discovery ownership
	- expanded `MapChunkIds` with Alpha-only `MDNM` and `MONM`
	- updated `WowViewer.Tool.Inspect map inspect` to print the shared WDT semantic summary plus a standard `MAIN` flag-distribution line when available
	- added `wow-viewer/tests/WowViewer.Core.Tests/WdtSummaryReaderTests.cs` coverage for synthetic standard WDT flags, synthetic Alpha WDT boundary behavior, and real-data `development.wdt`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `71` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WdtSummaryReaderTests` passed on Mar 31, 2026 with `3` passing focused WDT-summary tests
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- map inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development.wdt` passed on Mar 31, 2026 and now prints `WDT MAIN flags: any=1496 hasAdt=1496 allWater=0 loaded=0 unknown=0 asyncIds=0 distinct=0x1:1496`
	- this is still top-level semantic summary work, not deep WDT parsing, per-tile contract ownership, or write support

### Mar 27, 2026 - Shared AreaIdMapper Archive-Backed Loading Replaced Constructor-Time Extracted-Tree Probing

- Reworked the shared `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs` seam so it can load `AreaTable` and `Map` directly from shared archive readers instead of assuming extracted `DBFilesClient` trees.
- Landed pieces:
	- added archive-backed `TryLoadFromArchives(...)` using `IArchiveReader`, `DbClientFileReader`, and an in-memory DBCD provider
	- normalized shorthand build strings like `0.5.3` and `3.3.5` to the full WoWDBDefs-compatible builds expected by DBCD
	- changed `WoWMapConverter.Core.Converters.AlphaToLkConverter` to lazy mapper initialization from explicit DBC paths or explicit `AlphaClientPath` and `LkClientPath` archive roots
	- added CLI options `--alpha-client` and `--lk-client` in `WoWMapConverter.Cli`
	- expanded `wow-viewer/tests/WowViewer.Core.Tests/AreaIdMapperTests.cs` with synthetic archive-backed coverage and explicit archive-missing diagnostics
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug` passed on Mar 27, 2026 with `37` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` passed on Mar 27, 2026 with the existing warning floor
	- no real MPQ-root conversion smoke test was run in this slice, so the active proof level is shared-library regression plus converter buildability, not end-to-end runtime signoff

### Mar 27, 2026 - Shared AreaIdMapper DBCD Wiring And Explicit Fallback Warning Landed

- Upgraded the shared `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs` seam from raw-file-only loading to a real DBCD + WoWDBDefs-backed path for the active `AreaTable` and `Map` use case.
- Landed pieces:
	- `WowViewer.Core.IO.csproj` now uses the viewer-aligned vendored DBCD project from `gillijimproject_refactor/lib/wow.tools.local/DBCD` and bundles `gillijimproject_refactor/lib/WoWDBDefs/definitions` into output
	- `AreaIdMapper` now discovers bundled or vendored `WoWDBDefs` definitions and uses DBCD for known `0.5.3` and `3.3.5` extracted table trees when present, preferring `gillijimproject_refactor/test_data` roots first
	- `AreaIdMapper` still falls back to the existing narrow raw `DbcReader` when definitions or build inference are unavailable, preserving the old tests and compatibility path
	- `TryAutoLoadFromTestData()` now reports an explicit missing-tree diagnostic instead of silently returning false
	- `AlphaToLkConverter` now forwards that diagnostic as a visible runtime warning before falling back to crosswalk-only behavior
	- added shared-library tests for explicit missing-tree reporting and a synthetic DBCD-backed `AreaTable`/`Map` auto-load case
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 27, 2026 with `66` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 27, 2026 with the existing warning floor
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- convert i:/parp/parp-tools/gillijimproject_refactor/test_data/0.5.3/alphawdt/World/Maps/PVPZone01/PVPZone01.wdt -o i:/parp/parp-tools/output/pvpzone01-alpha-to-lk-smoke-dbcd-check3 -v` passed on Mar 27, 2026 and confirmed the new explicit warning path now names the preferred `gillijimproject_refactor/test_data/*/tree/DBFilesClient` roots first when extracted DBC trees are missing
	- no runtime proof was added yet for the schema-backed path against real extracted `test_data/*/tree/DBFilesClient` tables because those files are still absent in this workspace

### Mar 26, 2026 - Shared AreaIdMapper And Crosswalk Ownership Landed

- Finished the remaining live old-repo DBC-backed area-mapping cutover onto shared `wow-viewer/src/core/WowViewer.Core.IO/Dbc/AreaIdMapper.cs`.
- Moved the embedded `area_crosswalk.csv` resource into `wow-viewer` and wired `WowViewer.Core.IO.csproj` to embed it there.
- Retargeted `WoWMapConverter.Core.Converters.AlphaToLkConverter` to the shared mapper.
- Deleted the old-repo `Dbc/AreaIdMapper.cs`, the dead `Services/AreaIdCrosswalk.cs`, and the old `Resources/area_crosswalk.csv` copy from `WoWMapConverter.Core`.
- Added focused shared-library regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/AreaIdMapperTests.cs` for:
	- constructor-loaded embedded crosswalk defaults
	- matching-report CSV parsing through `LoadCrosswalkCsv(...)`
	- continent-hinted exact-name matching through `LoadDbcs(...)`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `64` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with `53` warnings and no new build break
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -c Debug` passed on Mar 26, 2026 with `3` warnings after the import cleanup
	- no real-data runtime validation was run

### Mar 26, 2026 - Shared Alpha MPQ Old-Repo Caller Cutover Landed

- Finished the remaining active old-repo per-asset MPQ caller cutover onto shared `wow-viewer/src/core/WowViewer.Core.IO/Files/AlphaArchiveReader.cs`:
	- `WoWMapConverter.Core.VLM.VlmDatasetExporter`
	- `WoWMapConverter.Core.Converters.WmoV14ToV17Converter`
	- `WoWMapConverter.Core.Converters.WmoV14ToV17ExtendedConverter`
- Deleted the now-dead duplicate `WoWMapConverter.Core/Services/AlphaMpqReader.cs` implementation.
- Added focused shared-library regression coverage in `wow-viewer/tests/WowViewer.Core.Tests/AlphaArchiveReaderTests.cs` for:
	- default block selection in per-asset MPQs without explicit internal names
	- companion `.MPQ` fallback using internal-name candidates from the requested path
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `61` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with `53` warnings and no new build break
	- no `MdxViewer` build was required because the active viewer already consumed the shared reader and only a comment changed there
	- no runtime validation was run

### Mar 26, 2026 - Dead Old DBC Helper Cleanup Landed

- Tightened the old-repo boundary after the shared `Core.IO` cutovers by deleting the now-dead helper layer from `WoWMapConverter.Core`:
	- `Dbc/DbcReader.cs`
	- `Services/NativeMpqService.cs`
	- `Services/Md5TranslateResolver.cs`
	- `Services/MapDbcService.cs`
	- `Services/GroundEffectService.cs`
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Dbc/AreaIdMapper.cs` now uses shared `WowViewer.Core.IO.Dbc.DbcReader`
	- the remaining active DBC-backed seam in `WoWMapConverter.Core` is now explicit instead of being hidden behind dead duplicate helpers
	- `AreaIdMapper` remains live through `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Converters/AlphaToLkConverter.cs`
- Validation limits:
	- workspace diagnostics for `WoWMapConverter.Core` reported no errors after the cleanup
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with `54` warnings and no new build break
	- no new `wow-viewer` tests were run because the shared library code did not change in this pass
	- this proves cleanup and dependency-boundary tightening, not completion of the remaining area-crosswalk migration seam

### Mar 26, 2026 - Shared DBC Lookup And VLM Archive Cutover Landed

- Extended `wow-viewer/src/core/WowViewer.Core.IO` with the next shared non-PM4 table-backed helper slice:
	- `DbcReader`
	- `DbcHeader`
	- `MapDirectoryLookup`
	- `GroundEffectLookup`
- Scope:
	- re-homes the tiny DBC parser plus the active map-directory and ground-effect lookup helpers out of `WoWMapConverter.Core`
	- expands shared `DbClientFileReader` probing to cover `DBFilesClient`, `DBC`, and root `.dbc` or `.db2` candidates
	- keeps active VLM archive and lookup behavior on shared `Core.IO` seams instead of `NativeMpqService` and old helper ownership
- Added regression coverage for:
	- `Map.dbc` archive-backed directory resolution
	- archive-backed ground-effect doodad lookup resolution
	- expanded shared DBC or DB2 probe ordering
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj` now references `wow-viewer/src/core/WowViewer.Core.IO`
	- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` now uses shared `IArchiveCatalog` or `IArchiveReader`
	- `VlmDatasetExporter` now resolves map directories through shared `MapDirectoryLookup`
	- `VlmDatasetExporter` now resolves ground-effect doodads through shared `GroundEffectLookup`
	- `VlmDatasetExporter` now loads MD5 minimap translation through shared callback-based `Md5TranslateResolver`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `59` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug` passed on Mar 26, 2026 with the existing warning floor
	- no viewer runtime validation was run
	- `MdxViewer` was not rebuilt in this slice because the active consumer change targeted VLM in `WoWMapConverter.Core`

### Mar 26, 2026 - Concrete Shared MPQ Catalog Port Landed

- Extended `wow-viewer/src/core/WowViewer.Core.IO` with the concrete standard MPQ implementation used by the active viewer consumer path:
	- `MpqArchiveCatalog`
	- `MpqArchiveCatalogFactory`
	- internal `MpqDiagnostics`
- Scope:
	- re-homes the actual archive loading, hash-table lookup, block-table parsing, decompression, and patch-priority behavior out of `WoWMapConverter.Core.Services.NativeMpqService`
	- keeps the active `MdxViewer` MPQ consumer path on a library-owned implementation instead of a compatibility adapter over the old repo
	- preserves the utility surface that still matters for future shared use, including internal listfile extraction and direct file-0 reads
- Added regression coverage for:
	- higher-priority patch reads winning over base archives
	- patched-delete fallback to base archive content
	- internal listfile extraction
	- direct file-0 reads from a standalone MPQ
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs` now defaults to shared `MpqArchiveCatalogFactory`
	- deleted `gillijimproject_refactor/src/MdxViewer/DataSources/NativeMpqArchiveCatalog.cs`
	- active `MdxViewer` `.cs` source no longer references `WoWMapConverter.Core.Services.NativeMpqService` in its standard MPQ path
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `57` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 with the existing warning floor
	- no viewer runtime validation was run
	- older `NativeMpqService` code still exists for other non-migrated old-repo consumers, so this slice proves active-path ownership cutover rather than full old-repo deletion

### Mar 26, 2026 - Shared Archive Bootstrap And Alpha Wrapper Cutovers Landed

- Extended `wow-viewer/src/core/WowViewer.Core.IO` with two new shared non-PM4 archive seams:
	- `ArchiveCatalogBootstrapper`
	- `ArchiveCatalogBootstrapResult`
	- `AlphaArchiveReader`
	- `PkwareExplode`
- Scope:
	- re-homes the standard archive bootstrap and external listfile parsing path out of `MpqDataSource`
	- re-homes the Alpha per-asset MPQ wrapper reader out of direct `WoWMapConverter.Core.Services.AlphaMpqReader` consumer usage
	- keeps `MpqDataSource` as a consumer of shared `Core.IO` archive helpers instead of an owner of those seams
- Added regression coverage for:
	- external listfile row parsing and bootstrap aggregation
	- Alpha internal-name candidate generation
	- Alpha direct-file fallback behavior
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs` now uses shared `ArchiveCatalogBootstrapper`
	- `MpqDataSource` now uses shared `AlphaArchiveReader`
	- active `MdxViewer` source no longer directly references `WoWMapConverter.Core.Services.AlphaMpqReader`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `53` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 with the existing warning floor
	- no viewer runtime validation was run
	- `NativeMpqService` still remains behind the compatibility adapter; this slice does not prove a full MPQ implementation port

### Mar 26, 2026 - Shared Archive-Reader MPQ Cutover Landed

- Extended `wow-viewer/src/core/WowViewer.Core.IO` with a new shared non-PM4 archive access seam:
	- `IArchiveReader`
	- `IArchiveCatalog`
	- `IArchiveCatalogFactory`
	- `DbClientFileReader`
- Scope:
	- re-homes the standard MPQ reader boundary out of direct `MdxViewer` ownership and onto shared `Core.IO` contracts
	- keeps the current `NativeMpqService` implementation behind a compatibility adapter instead of treating it as the active consumer contract
	- re-homes `DBFilesClient` DBC or DB2 candidate probing into shared `Core.IO`
- Added regression coverage for:
	- DBC or DB2 path candidate ordering
	- first-match table reads through a shared archive reader
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDataSource.cs` now uses shared archive interfaces for standard MPQ access and prefetch worker creation
	- `gillijimproject_refactor/src/MdxViewer/DataSources/MpqDBCProvider.cs` now uses shared `DbClientFileReader`
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now consumes `ArchiveReader` instead of `MpqService`
	- direct `NativeMpqService` coupling is isolated to `DataSources/NativeMpqArchiveCatalog.cs`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `49` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 with the existing warning floor
	- no viewer runtime validation was run
	- Alpha wrapper reads remain on `WoWMapConverter.Core.Services.AlphaMpqReader`; that seam was not ported in this slice

### Mar 26, 2026 - Shared MD5 Minimap Translation Cutover Landed

- Extended `wow-viewer/src/core/WowViewer.Core.IO` with a new shared non-PM4 path-translation seam:
	- `Md5TranslateIndex`
	- `Md5TranslateResolver.TryLoad(...)`
	- `MinimapService.GetMinimapTilePath(...)`
	- `MinimapService.MinimapTileExists(...)`
- Scope:
	- re-homes MD5 minimap translation loading and minimap tile path helpers out of `WoWMapConverter.Core.Services`
	- keeps archive reads abstracted behind callbacks instead of hard-coding `NativeMpqService` into the shared seam
	- retargets the active `MdxViewer` minimap and GLB-export consumers to shared `WowViewer.Core.IO` types
- Added regression coverage for:
	- map-specific archive TRS loading
	- `dir:` context parsing for disk-backed TRS files
- Consumer follow-up now also landed:
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now loads MD5 minimap translation through shared `Core.IO`
	- `Rendering/MinimapRenderer.cs` and `Export/MapGlbExporter.cs` now consume shared `Md5TranslateIndex` and `MinimapService`
	- `MdxViewer.csproj` now references `wow-viewer/src/core/WowViewer.Core.IO`
	- `ViewerApp` no longer uses `WoWMapConverter.Core.Services.DevelopmentMapAnalyzer.DefaultDevelopmentMapDirectory`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `47` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 with the existing warning floor
	- no viewer runtime validation was run

### Mar 26, 2026 - Direct MdxViewer PM4 Import Cutover Landed

- Removed the remaining direct `WoWMapConverter.Core.Formats.PM4` dependency from the active `MdxViewer` PM4 consumer path:
	- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` now reads PM4 through `WowViewer.Core.PM4.Services.Pm4ResearchReader`
	- `WorldScene` now aliases PM4 decode or model usage to shared `WowViewer.Core.PM4` document and chunk types instead of `WoWMapConverter` PM4 wrapper types
	- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs` now uses the shared PM4 reader for loose-overlay build-hint detection
	- removed the stale PM4 import from `gillijimproject_refactor/src/MdxViewer/ViewerApp_Sidebars.cs`
- Boundary outcome:
	- no direct `WoWMapConverter.Core.Formats.PM4` import remains under `gillijimproject_refactor/src/MdxViewer`
	- `MdxViewer` still keeps a broader `WoWMapConverter.Core` project reference for non-PM4 subsystems; that wider cutover is not part of this slice
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed on Mar 26, 2026 after the cutover
	- no new automated tests were added or run for this viewer-side refactor
	- no viewer runtime validation was run

### Mar 26, 2026 - PM4 Linked-Position-Ref Summary Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-owned PM4 placement or research-adjacent seam:
	- `Pm4LinkedPositionRefSummary`
	- `Pm4PlacementMath.SummarizeLinkedPositionRefs(...)`
- Scope:
	- re-homes linked MPRL position-ref summary aggregation from `WorldScene`
	- keeps floor-range, heading-range, and circular-mean summary logic on shared PM4 contracts instead of viewer-local summary code
	- does not change PM4 inspect or report payload shape in this slice
- Added regression coverage for:
	- synthetic linked position refs with mixed normal and terminator entries
	- terminator-only linked position refs preserving NaN heading fallback
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `31` passing PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `45` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-linked-position-ref-summary-hookup/` passed on Mar 26, 2026
	- no PM4 inspect command or viewer runtime validation was run because the slice did not change analyzer or report output

### Mar 26, 2026 - PM4 Placement-Solution Consumer Hookup Landed

- Updated `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` so the active viewer now consumes the existing shared PM4 placement-solution seam:
	- the CK24 overlay path now calls `Pm4PlacementMath.ResolvePlacementSolution(...)`
	- planar transform, world pivot, and world yaw correction now come from one shared typed placement result instead of three separate consumer-owned steps
	- removed the redundant per-piece consumer wrappers for that PM4 placement path
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-placement-solution-hookup/` passed on Mar 26, 2026
	- no new `wow-viewer` tests were added or rerun because the shared library seam was unchanged in this slice
	- no viewer runtime validation was run

### Mar 26, 2026 - PM4 Connector-Key Consumer Hookup Landed

- Updated `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` so the active viewer now consumes the existing shared PM4 connector-key seam:
	- `BuildCk24ConnectorKeys()` now builds a shared `Pm4PlacementSolution`
	- connector-key derivation now comes from `Pm4PlacementMath.BuildConnectorKeys(...)`
	- removed the redundant viewer-local connector-point conversion and quantization implementation for that PM4 grouping input path
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-connector-key-hookup/` passed on Mar 26, 2026
	- no new `wow-viewer` tests were added or rerun because the shared library seam was unchanged in this slice
	- no viewer runtime validation was run

### Mar 26, 2026 - PM4 Merge-Map Consumer Hookup Landed

- Updated `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` so the active viewer now consumes the existing shared PM4 merge-map seam:
	- `RebuildPm4MergedObjectGroups()` now maps local overlay groups to shared `Pm4ConnectorMergeCandidate` inputs
	- canonical merged-group resolution now comes from `Pm4PlacementMath.BuildMergedGroupMap(...)`
	- removed the redundant viewer-local union-find and merge-heuristic implementation for that PM4 grouping path
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-merge-map-hookup/` passed on Mar 26, 2026
	- no new `wow-viewer` tests were added or rerun because the shared library seam was unchanged in this slice
	- no viewer runtime validation was run

### Mar 26, 2026 - PM4 Correlation Geometry-Input Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-owned PM4 correlation seam:
	- `Pm4GeometryLineSegment`
	- `Pm4GeometryTriangle`
	- `Pm4CorrelationGeometryInput`
	- `Pm4CorrelationMath.BuildObjectStatesFromGeometry(...)`
- Scope:
	- re-homes PM4 correlation geometry-input assembly from `WorldScene` into `Core.PM4`
	- keeps PM4 line or triangle transform application and sampled world-geometry point derivation on shared PM4 contracts instead of viewer-local flattening helpers
	- explicitly keeps WMO-facing correlation report payload ownership outside `Core.PM4`
- Added regression coverage for:
	- synthetic PM4 geometry-input object-state construction without viewer-specific world-point assembly
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `29` passing PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `45` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-correlation-geometry-hookup/` passed on Mar 26, 2026
	- no viewer runtime validation was run

### Mar 26, 2026 - PM4 Correlation Object-State Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-owned PM4 correlation-state seam:
	- `Pm4CorrelationObjectDescriptor`
	- `Pm4CorrelationObjectInput`
	- `Pm4CorrelationObjectState`
	- `Pm4CorrelationMath.BuildObjectStates(...)`
	- public footprint-hull and footprint-area helpers for transformed or precomputed world geometry
- Scope:
	- re-homes PM4 correlation object summarization, bounds derivation, sampled footprint hull construction, and empty-geometry fallback out of `WorldScene`
	- keeps WMO correlation report consumption on shared state and shared hull or metric helpers instead of viewer-local state records and duplicated polygon code
	- does not yet move the full correlation-report payload contract itself into `Core.PM4`
- Added regression coverage for:
	- synthetic object-state bounds and footprint derivation
	- empty-geometry fallback center behavior
	- transformed footprint-hull construction
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `28` passing PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `42` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-correlation-state-hookup/` passed on Mar 26, 2026
	- no viewer runtime validation was run

### Mar 26, 2026 - PM4 Correlation-Math Library Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-owned PM4 correlation seam:
	- `Pm4CorrelationMetrics`
	- `Pm4CorrelationCandidateScore`
	- `Pm4CorrelationMath.EvaluateMetrics(...)`
	- `Pm4CorrelationMath.CompareCandidateScores(...)`
- Scope:
	- re-homes planar-gap, vertical-gap, overlap-ratio, footprint-distance, polygon-clipping, and correlation-candidate ranking helpers from `WorldScene`
	- keeps future PM4 correlation reports or placement matching on shared library contracts instead of viewer-owned anonymous metric tuples
	- does not yet move the active viewer's correlation-report assembly or object-state construction into `Core.PM4`
- Added regression coverage for:
	- synthetic overlap and footprint-distance metric calculation
	- same-tile ranking precedence over stronger cross-tile overlap
	- footprint-overlap precedence when tile parity matches
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `25` passing PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `39` passing tests
	- no active-viewer compile or runtime validation was run because consumer compatibility did not change

### Mar 26, 2026 - PM4 Connector-Group Merge Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-owned PM4 grouping seam:
	- `Pm4ObjectGroupKey`
	- `Pm4ConnectorMergeCandidate`
	- `Pm4PlacementMath.BuildMergedGroupMap(...)`
- Scope:
	- re-homes connector-overlap, bounds-padding, and center-distance merge heuristics from `WorldScene`
	- keeps canonical merged-group selection in the shared library instead of leaving it viewer-owned
	- does not yet move active-viewer object-group rebuild wiring into `Core.PM4`
- Added regression coverage for:
	- neighbor-tile merge resolution with shared connector keys
	- same-tile non-merge protection even with shared connector keys
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `22` passing PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `36` passing tests
	- no active-viewer compile or runtime validation was run because consumer compatibility did not change

### Mar 26, 2026 - PM4 Connector-Key Library Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-owned PM4 grouping or correlation helper seam:
	- `Pm4ConnectorKey`
	- `Pm4PlacementMath.BuildConnectorKeys(...)`
- Scope:
	- converts `MSUR.MdosIndex` exterior vertices into quantized world-space connector keys through typed `Pm4PlacementSolution`
	- keeps connector dedupe and deterministic ordering in the shared library
	- does not pull renderer-space conversion or group-merge heuristics into this slice
- Added regression coverage for:
	- distinct sorted connector-key extraction
	- yaw-corrected connector placement in world space
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed on Mar 26, 2026 with `22` passing PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `36` passing tests
	- no active-viewer compile or runtime validation was run because consumer compatibility did not change

### Mar 26, 2026 - wow-viewer Source-Of-Truth Reset

- Updated the working rule for future `wow-viewer` chats:
	- `WowViewer.Core.PM4`, `WowViewer.Core`, and `WowViewer.Core.IO` are now the canonical implementation targets for new `wow-viewer` work
	- `MdxViewer` is now a reference or compatibility consumer, not the default PM4 source of truth
	- default validation for `wow-viewer` work is `WowViewer.slnx` build or test plus the relevant tool command on the fixed development dataset
	- `MdxViewer` compile validation is now optional and should be run only when a slice changes consumer compatibility or the user explicitly asks for it
- This is a workflow and continuity reset, not runtime proof by itself.

### Mar 26, 2026 - PM4 Handoff State Prepared For Fresh Chat

- Refreshed the PM4 continuity state so the next session can start from the actual current boundary instead of re-deriving it.
- Current PM4 state to carry forward:
	- `wow-viewer` now has the research reader, inspect or audit or report verbs, and the current extracted placement-math stack in `Core.PM4`
	- active `MdxViewer` now consumes shared `Core.PM4` only for the narrow planar-transform, world-yaw, and world-space centroid seams
	- the typed coordinate-mode resolver already exists in `Core.PM4`, but its active-viewer consumer hookup is still the clean next seam rather than a solved problem
- Fresh validation re-run on Mar 26, 2026:
	- `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug` passed with `22` PM4 tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed with `11` placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with `36` total tests
- Explicit non-claim preserved for future sessions:
	- PM4 is not "finished"
	- library and compile validation do not equal runtime viewer PM4 signoff
	- renderer-space composition, broader object placement flow, and remaining research semantics are still open
- Recommended next PM4 slice for the next chat:
	- wire `ResolveCoordinateMode(...)` into the active viewer through a narrow adapter follow-up, then keep continuity files synchronized with whatever that slice proves

### Mar 25, 2026 - wow-viewer Tool Inventory And Cutover Plan

- Added planning document:
	- `plans/wow_viewer_tool_inventory_and_cutover_plan_2026-03-25.md`
- Purpose:
	- inventory the old repo tool sprawl and make explicit keep, merge, kill, and archaeology calls for the future `wow-viewer` repo.
- Main decisions captured:
	- keep one interactive app shell, one converter CLI, one inspect CLI, one optional catalog CLI, and a real PM4 library from day one instead of porting every legacy executable.
	- merge `WoWMapConverter.Cli`, `AlphaLkToAlphaStandalone`, and the still-useful conversion seams from `WoWRollback` into one future converter surface.
	- merge `AlphaWdtAnalyzer.Cli` and `AlphaWdtInspector`; keep `DBCTool.V2` behavior only.
	- PM4 correction: current `MdxViewer` behavior is the runtime reference, and `Pm4Research` should be ported into the new repo as the future PM4 library family rather than left behind as a pure archaeology seam.
	- treat `parpToolbox` and `PM4Tool` as supporting PM4 evidence rather than as production app identities.
	- keep poorly scoped or obsolete executables such as `ADTPrefabTool`, `DBCTool`, old WoWRollback GUI or viewer surfaces, and archived WMOv14 tools in `parp-tools` only.
	- follow-up planning docs now exist for the bootstrap layout, the CLI or GUI dual-surface design, and the PM4 library direction.
	- migration emphasis is now `1, 3, 2`: bootstrap layout and skeleton first, dual-surface plan second, deeper PM4 consolidation third.
- Validation limits:
	- planning and documentation only
	- no builds or runtime validation were run because no code changed

### Mar 25, 2026 - wow-viewer Initial Skeleton Scaffolded

- Created a new `wow-viewer/` folder at the workspace root with an initial solution and project graph:
	- `WowViewer.slnx`
	- `src/viewer/WowViewer.App`
	- `src/core/WowViewer.Core`
	- `src/core/WowViewer.Core.IO`
	- `src/core/WowViewer.Core.Runtime`
	- `src/core/WowViewer.Core.PM4`
	- `src/tools-shared/WowViewer.Tools.Shared`
	- `tools/converter/WowViewer.Tool.Converter`
	- `tools/inspect/WowViewer.Tool.Inspect`
- Added first-pass root files and bootstrap placeholders:
	- `Directory.Build.props`
	- `Directory.Packages.props`
	- `eng/Version.props`
	- `README.md`
	- `scripts/bootstrap.ps1`
	- `scripts/bootstrap.sh`
	- `scripts/validate-real-data.ps1`
- The scaffold encodes the current PM4 planning decision directly:
	- `Core.PM4` exists immediately
	- placeholder code names `MdxViewer` as the PM4 runtime reference and `Pm4Research` as the library seed
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- this is still only a skeleton and placeholder-code build, not a real implementation or runtime-validated migration

### Mar 25, 2026 - First PM4 Reader Slice Ported Into Core.PM4

- Ported the first real PM4 code from `gillijimproject_refactor/src/Pm4Research.Core` into `wow-viewer/src/core/WowViewer.Core.PM4`.
- Added:
	- typed PM4 chunk models
	- research document container
	- binary PM4 reader
	- exploration snapshot builder
- Scope boundary:
	- this is a raw research-facing PM4 layer only
	- it does not yet move current `MdxViewer` reconstruction, grouping, transform, or correlation behavior into `Core.PM4`
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 after the port
	- no runtime validation or viewer integration was performed in this slice

### Mar 25, 2026 - Single-File PM4 Analyzer And Inspect Verbs Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the first single-file PM4 analyzer or report layer.
- Added working Tool.Inspect PM4 verbs:
	- `pm4 inspect`
	- `pm4 export-json`
- Smoke-tested against the fixed development reference tile `development_00_00.pm4`.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 inspect --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed on Mar 25, 2026
	- this is still single-file research analysis only, not viewer integration or broad PM4 signoff

### Mar 25, 2026 - PM4 Audit Path And Placement Contracts Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with:
	- decode audit and corpus-audit report models plus analyzer
	- first extracted MdxViewer-facing placement contracts: `Pm4AxisConvention`, `Pm4CoordinateMode`, `Pm4PlanarTransform`, `Pm4CoordinateService`, and `Pm4PlacementContract`
- Added new working Tool.Inspect PM4 verbs:
	- `pm4 audit`
	- `pm4 audit-directory`
- Captured the current research note that CK24 low-16 object values may align with expected `UniqueID` ranges on the development map, but this remains a hypothesis until correlated against real placement data.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 audit-directory --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026 and scanned `616` PM4 files with no unknown chunks or file-level diagnostics
	- this is still not the full viewer reconstruction or solver migration

### Mar 25, 2026 - First PM4 Tests Added To wow-viewer

- Added `tests/WowViewer.Core.PM4.Tests` as the first test project in the new repo.
- Locked current behavior with real-data assertions against:
	- `development_00_00.pm4`
	- the fixed development PM4 corpus directory
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 with `6` passing tests
	- this is still narrow fixed-dataset regression coverage, not broad PM4 correctness signoff

### Mar 25, 2026 - PM4 Linkage Report And Placement-Math Helper Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with:
	- linkage report types and corpus analyzer
	- first extracted placement-math helper layer from current `WorldScene`
- Added a new working Tool.Inspect PM4 verb:
	- `pm4 linkage --input <directory> [--output <report.json>]`
- Validated fixed-corpus linkage findings:
	- `616` files scanned
	- `150` files with ref-index mismatches
	- `58` files with bad `MDOS` refs
	- `4553` total ref-index mismatches
- Interpretation boundary preserved:
	- low16 CK24 object values may still sit in plausible `UniqueID` ranges, but the linkage report does not support treating them as globally unique identifiers by range alignment alone.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 linkage --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 with `7` passing tests

### Mar 25, 2026 - PM4 MSCN Report Family Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with MSCN relationship report types and corpus analyzer.
- Added a new working Tool.Inspect PM4 verb:
	- `pm4 mscn --input <directory> [--output <report.json>]`
- Validated fixed-corpus MSCN findings:
	- `616` files scanned
	- `309` files with MSCN
	- `1,342,410` MSCN points
	- `MSUR.MdosIndex -> MSCN`: `511,891` fits and `6,201` misses
	- raw bounds overlap outperformed swapped-XY overlap by a wide margin in this corpus slice
- Interpretation boundary preserved:
	- MSCN still looks relevant as a companion layer, but this slice does not support a simple swapped-XY explanation as the dominant corpus-wide answer.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 mscn --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 25, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 25, 2026 with `7` passing tests

### Mar 26, 2026 - PM4 Unknowns Report And Normal-Axis Solver Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with:
	- unknowns report types and corpus analyzer
	- the next extracted `WorldScene` solver seam: normal-based axis detection and scoring in `Pm4PlacementMath`
- Added a new working Tool.Inspect PM4 verb:
	- `pm4 unknowns --input <directory> [--output <report.json>]`
- Validated fixed-corpus unknowns findings:
	- `616` files scanned
	- `309` non-empty geometry or link files
	- `1,273,335` sentinel-pattern `MSLK.LinkId` values and no non-sentinel values in this corpus slice
	- `598,882` active `MSLK` path windows: `399,183` indices-only fits and `199,699` dual-fit windows
	- `MSLK.RefIndex -> MSUR` remains partial with `4,553` misses
- Solver-seam consequence:
	- `Core.PM4` now owns both the current range-based axis fallback and the next normal-based axis scoring helpers, reducing how much of the placement heuristic remains marooned inside `WorldScene`.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026
	- `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/inspect/WowViewer.Tool.Inspect/WowViewer.Tool.Inspect.csproj -- pm4 unknowns --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed on Mar 26, 2026
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `8` passing tests

### Mar 26, 2026 - PM4 Planar-Transform Resolver Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next extracted PM4 solver seam from `WorldScene`:
	- `Pm4PlacementMath.ResolvePlanarTransform`
	- MPRL centroid-distance scoring
	- MPRL footprint scoring
	- MPRL yaw comparison with quarter-turn fallback
- Added regression coverage for:
	- current whole-tile development PM4 resolver behavior
	- a synthetic world-space quarter-turn planar candidate case
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `9` passing tests
	- measured whole-tile development-tile result currently resolves to tile-local planar transform `(swap=false, invertU=false, invertV=false)` for the fixed test slice
	- this is still a solver-seam extraction, not full viewer-runtime PM4 integration or final placement correctness signoff

### Mar 26, 2026 - PM4 World-Yaw Correction And First Viewer Consumer Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next extracted PM4 solver seam from `WorldScene`:
	- `Pm4PlacementMath.TryComputeWorldYawCorrectionRadians`
	- signed basis fallback against MPRL heading evidence
- Added regression coverage for:
	- a synthetic non-zero world-yaw correction case
- Started active viewer consumption of shared PM4 solver logic:
	- `MdxViewer.csproj` now references `wow-viewer` `Core.PM4`
	- `WorldScene` now delegates planar-transform resolution and world-yaw correction into shared `Core.PM4` through explicit type adapters
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `10` passing tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-corepm4-hookup/` passed on Mar 26, 2026
	- no automated viewer integration tests were added or run
	- no real-data runtime signoff yet on viewer-visible PM4 behavior after the shared-library hookup

### Mar 26, 2026 - PM4 World-Space Centroid Slice And Second Viewer Consumer Hookup Landed

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next extracted PM4 solver seam from `WorldScene`:
	- `Pm4PlacementMath.ComputeSurfaceWorldCentroid`
	- shared surface-derived pivot computation using the chosen PM4 axis convention, coordinate mode, and planar transform
- Added regression coverage for:
	- a synthetic tile-local world-space centroid case using the real PM4 tile-size mapping
- Extended active viewer consumption of shared PM4 solver logic:
	- `WorldScene.ComputeSurfaceWorldCentroid(...)` now delegates into shared `Core.PM4` through the existing explicit adapters
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `11` passing tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `4` passing placement-focused tests
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug -p:OutDir=i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-pm4-centroid-hookup/` passed on Mar 26, 2026
	- no automated viewer integration tests were added or run
	- no real-data runtime signoff yet on viewer-visible PM4 behavior after the added shared centroid hook-up

### Mar 26, 2026 - PM4 World-Space Yaw Helper Slice Landed In wow-viewer

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-only PM4 math seam:
	- `Pm4PlacementMath.RotateWorldAroundPivot`
	- `Pm4PlacementMath.ConvertPm4VertexToWorld(...)` overload for corrected world-space conversion around a pivot
- Added regression coverage for:
	- a synthetic world-space pivot rotation case
	- a synthetic tile-local corrected world-position case using the real PM4 tile-size mapping
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `6` passing placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `13` passing tests
	- no new active-viewer code changed in this slice
	- no real-data viewer runtime signoff was performed in this slice

### Mar 26, 2026 - PM4 Placement-Solution Contract Slice Landed In wow-viewer

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the first typed placement-result contract:
	- `Pm4PlacementSolution`
	- `Pm4PlacementMath.ResolvePlacementSolution(...)`
	- `Pm4PlacementMath.ConvertPm4VertexToWorld(Vector3, Pm4PlacementSolution)`
- Added regression coverage for:
	- a synthetic world-space placement-solution case with resolved transform and pivot but no yaw correction
	- a synthetic world-space placement-solution case with resolved transform, pivot, and meaningful yaw correction
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `8` passing placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `15` passing tests
	- no new active-viewer code changed in this slice
	- no real-data viewer runtime signoff was performed in this slice

### Mar 26, 2026 - wow-viewer Copilot Workflow Surface Updated

- Updated `.github/copilot-instructions.md` so `wow-viewer` is now treated as an active primary path, with explicit PM4 library-first guardrails and validation rules.
- Added new reusable project skills:
	- `.github/skills/wow-viewer-pm4-library/SKILL.md`
	- `.github/skills/wow-viewer-migration-continuation/SKILL.md`
- Added a dedicated PM4 continuation prompt:

### Mar 26, 2026 - First Non-PM4 Shared Map Inspect Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core` with the first non-PM4 map-format contracts:
	- `MapChunkIds`
	- `MapFileKind`
	- `MapChunkLocation`
	- `MapFileSummary`
- Extended `wow-viewer/src/core/WowViewer.Core.IO` with the first shared WDT or ADT top-level reader slice:
	- `ChunkedFileReader`
	- `MapFileSummaryReader`
- Added a real non-PM4 inspect consumer on top of shared IO:
	- `wow-viewer/tools/inspect/WowViewer.Tool.Inspect` verb `map inspect --input <file.wdt|file.adt>`
- Added regression coverage for:
	- synthetic WDT summary detection
	- synthetic ADT summary detection
	- fixed-dataset `development.wdt`
	- fixed-dataset `development_0_0.adt`
- Validation limits:
	- this is still only top-level chunk summary behavior, not full ADT or WDT parsing or writing

### Mar 26, 2026 - First Shared Cross-Family Detection Slice Landed

- Extended `wow-viewer/src/core/WowViewer.Core` with the first broader file-detection contracts:
	- `WowFileKind`
	- `WowFileDetection`
- Extended `wow-viewer/src/core/WowViewer.Core.IO` with the first shared cross-family detector:
	- `WowFileDetector`
- Refactored `MapFileSummaryReader` to consume that shared detector instead of its own file-kind heuristics.
- Added the first non-placeholder converter command on top of the shared detector:
	- `wow-viewer/tools/converter/WowViewer.Tool.Converter` verb `detect --input <file>`
- Added regression coverage for:
	- synthetic WDT detection
	- fixed-dataset WDT, root ADT, split texture ADT, split object ADT, and PM4 detection
- Validation limits:
	- this is still only shared classification and version detection, not conversion or payload parsing

### Mar 26, 2026 - wow-viewer Shared I/O Workflow Surface Tightened

- Added a dedicated non-PM4 shared-I/O skill:
	- `.github/skills/wow-viewer-shared-io-library/SKILL.md`
- Added a dedicated non-PM4 shared-I/O implementation prompt:
	- `.github/prompts/wow-viewer-shared-io-implementation.prompt.md`
- Added a dedicated shared-I/O continuity plan:
	- `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md`
- Updated routing surfaces so future sessions can distinguish:
	- PM4 implementation work
	- shared `Core` or `Core.IO` implementation work
	- broader tool-suite or migration planning
- Updated `.github/copilot-instructions.md` with shared-I/O first reads and guardrails.
- Added a forward-maintenance rule across instructions and continuity surfaces:
	- new `wow-viewer` skills or implementation prompts must also update `.github/copilot-instructions.md`, `wow-viewer/README.md`, the relevant continuity plan, and the memory bank in the same slice
- Validation limits:
	- workflow and continuity updates only
	- no new runtime claim beyond the already-validated shared detector and summary slices
	- `.github/prompts/wow-viewer-pm4-library-implementation.prompt.md`
- Updated `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md` so implementation-sized `Core.PM4` work now routes to the dedicated PM4 library prompt instead of only the broader migration-planning prompts.
- Updated `gillijimproject_refactor/plans/wow_viewer_pm4_library_plan_2026-03-25.md` so the new `.github` skills or prompts are recorded as the canonical shared workflow surface for the active PM4 migration slice.
- Validation limits:
	- documentation or workflow updates only
	- no code build or runtime validation was needed for this customization slice

### Mar 26, 2026 - PM4 Coordinate-Mode Resolver Slice Landed In wow-viewer

- Extended `wow-viewer/src/core/WowViewer.Core.PM4` with the next library-only PM4 solver seam:
	- `Pm4CoordinateModeResolution`
	- `Pm4PlacementMath.ResolveCoordinateMode(...)`
	- shared coordinate-mode score evaluation for tile-local versus world-space interpretation using the already-extracted planar-transform, footprint, and centroid scoring helpers
- Added regression coverage for:
	- current development-tile coordinate-mode behavior
	- a synthetic world-space coordinate-mode case
	- the missing-evidence fallback path
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter PlacementMath` passed on Mar 26, 2026 with `11` passing placement-focused tests
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `18` passing tests
	- no new active-viewer code changed in this slice
	- no real-data viewer runtime signoff was performed in this slice

### Mar 26, 2026 - wow-viewer Bootstrap And Non-PM4 Core Follow-Up

- Corrected two clear plan-adherence gaps in `wow-viewer`:
	- bootstrap scripts are no longer placeholders and now clone the baseline upstream repos from the migration draft into `libs/`
	- the repo now has its first non-PM4 shared-core slice in `WowViewer.Core` and `WowViewer.Core.IO`
- Added core foundation files:
	- `src/core/WowViewer.Core/Chunks/FourCC.cs`
	- `src/core/WowViewer.Core/Chunks/ChunkHeader.cs`
	- `src/core/WowViewer.Core.IO/Chunked/ChunkHeaderReader.cs`
- Added new test project:
	- `tests/WowViewer.Core.Tests`
- Validation limits:
	- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed on Mar 26, 2026 with `22` total passing tests
	- bootstrap scripts were implemented but not executed here because cloning external repos would require network access and would materially change the workspace contents
	- this still does not mean the broader shared I/O and runtime migration phases are complete

### Mar 25, 2026 - Post-v0.4.5 Roadmap Prompt Bundle + Isolated Branch

- Detailed Copilot prompt assets for the larger `wow-viewer` tool-suite/library refactor now live under workspace `.github/prompts/` instead of `gillijimproject_refactor/plans`, because this work is prompt-surface/workflow material rather than just another local markdown note set.
- Added dedicated workspace prompt files:
	- `.github/prompts/wow-viewer-tool-suite-plan-set.prompt.md`
	- `.github/prompts/wow-viewer-bootstrap-layout-plan.prompt.md`
	- `.github/prompts/wow-viewer-shared-io-library-plan.prompt.md`
	- `.github/prompts/wow-viewer-tool-inventory-cutover-plan.prompt.md`
	- `.github/prompts/wow-viewer-cli-gui-surface-plan.prompt.md`
	- `.github/prompts/wow-viewer-tool-migration-sequence-plan.prompt.md`
- Created a dedicated follow-on branch for next-version work:
	- `feature/v0.4.6-v0.5.0-roadmap`
- Added new planning prompt files under `plans/`:
	- `post_v0_4_5_plan_set_2026-03-25.md`
	- `v0_4_6_v0_5_0_roadmap_prompt_2026-03-25.md`
	- `wowrollback_uniqueid_timeline_prompt_2026-03-25.md`
	- `alpha_core_sql_scene_liveness_prompt_2026-03-25.md`
	- `viewer_performance_recovery_prompt_2026-03-25.md`
	- `v0_5_0_new_repo_library_migration_prompt_2026-03-25.md`
	- `v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md`
- Updated existing planning files:
	- `v0_5_0_goal_stack_prompt_2026-03-25.md`
	- `enhanced_terrain_shader_lighting_prompt_2026-03-25.md`
- Planning direction captured:
	- `v0.4.6` is now framed as the first WoWRollback / `UniqueID` timeline integration slice inside the active viewer, plus Alpha-Core SQL caching/fidelity work and a first performance recovery pass.
	- `v0.5.0` is now reframed as the migration into `https://github.com/akspa0/wow-viewer`, with a canonical shared library plus split viewer/tool consumers, instead of just a larger in-place renderer/performance milestone inside `parp-tools`.
	- latest constraint on that migration: fully re-own the first-party read/parse/write/convert stack, including current base libraries such as `gillijimproject-csharp`, while keeping upstream externals like `Warcraft.NET`, `DBCD`, `WoWDBDefs`, `Alpha-Core`, `WoWTools.Minimaps`, and `SereniaBLPLib` under `libs/` and tracking original repos where practical.
	- repository bootstrap should also automate support-material pulls such as `wow-listfile`.
	- possible alpha-era support contributions upstream to `Noggit` / `noggit-red` remain an explicit stretch/outreach track, not the core delivery target for `v0.5.0`.
	- possible secondary integration/evaluation seams include `MapUpconverter`, `ADTMeta`, `wow.export`, and `wow.tools.local`.
	- a first concrete `wow-viewer` repo tree plus migration order draft now exists so future planning can refine a named proposal instead of reopening the repo-shape argument each session.
- Documentation follow-up:
	- root `README.md` now states the documented support range more plainly (`0.5.3` through `4.0.0.11927`) and does a better job surfacing the built-in converters, WMO `v14/v16/v17` support, SQL-driven spawns, PM4 tooling, and screenshot automation reality.
- Validation limits:
	- planning/docs only
	- no automated tests or builds were run for this slice because no code changed

### Mar 25, 2026 - Fullscreen Minimap Transpose Repair + Runtime User Signoff

- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- reverted the over-corrected world-axis swap from the earlier Designer Island follow-up
	- camera tile readout and minimap teleport now stay on the direct world `X/Y` mapping used by the active viewer instead of reinterpreting the entire minimap world orientation
- `src/MdxViewer/MinimapHelpers.cs`
	- reverted the broad POI/taxi overlay world-axis swap
	- kept the narrower camera-marker screen-placement transpose so marker placement matches the already-correct drawn tile grid
- `src/MdxViewer/ViewerApp.cs`
	- restored the legacy `DrawMinimap_OLD()` fallback path to the same final orientation logic so old code does not preserve the over-corrected behavior
- Root cause:
	- the minimap tile grid itself was already oriented correctly after the `ChunkSize` regression repair
	- the first axis patch then over-corrected the world/camera layer; the real remaining seam was the marker/grid screen transposition
- Release outcome:
	- runtime user confirmation after this final patch says the fullscreen minimap is fixed
	- the fullscreen minimap should no longer be treated as an open `v0.4.5` blocker
- Validation limits:
	- build plus targeted runtime user signoff: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-transpose-repair/"` passed on Mar 25, 2026
	- runtime user feedback then confirmed the repaired Designer Island/top-right minimap behavior on the fixed development dataset
	- no automated tests were added or run

### Mar 25, 2026 - World Object Culling And Object-Fog Tuning

- `src/MdxViewer/Terrain/WorldScene.cs`
	- world WMO visibility now uses point-to-AABB distance instead of AABB-center distance for frustum grace and distance culling, which keeps large nearby objects from disappearing when the camera is close to their edge.
	- near-camera frustum-cull grace is now larger and scales with object bounds instead of relying only on a small fixed radius.
	- WMO cull range now expands relative to fog end instead of staying pinned to a short fixed distance.
	- object render passes now use a delayed object-fog start so distant objects are not pushed into fog color as aggressively while still remaining rendered.
	- MDX/taxi-object frustum gating now uses AABB distance for the near-camera exemption path as well.
- `src/MdxViewer/Rendering/WmoRenderer.cs`
	- the separate internal WMO doodad cull distance was raised substantially and now also expands with fog range at runtime.
	- WMO doodad render cap was increased to reduce disappearing interior/attached doodads in dense sets.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-object-culling-fog/"` passed on Mar 25, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on object pop-in reduction or far-fog feel.

### Mar 25, 2026 - Taxi Override Workflow, World Return, And Override Persistence

- `src/MdxViewer/ViewerApp.cs`
	- added world-return capture/restore helpers so opening a standalone model can preserve the current world session path and camera for later restoration.
	- added browser-selection helpers plus taxi override application helpers so a selected browser asset can be applied directly to the active taxi-route override target.
	- added persisted taxi override storage in viewer settings keyed by map name and route ID, with replay when the current world loads.
	- fixed a follow-up compile break in the same slice where helper methods were accidentally inserted inside `LoadFileFromDisk()`; final solution build is the only validation that counts for this slice.
- `src/MdxViewer/ViewerApp_Sidebars.cs`
	- file browser now exposes `Open Selected`, `Copy Path`, `Use For Taxi Override`, and `Return To Last World`.
	- taxi inspector now exposes `Use Selected Browser Asset`, `Copy Override Path`, `Open Override Asset`, and `Return To Last World` around the existing override controls.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-taxi-workflow/"` passed on Mar 25, 2026.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on map/object swapping, saved taxi overrides, or browser-selected override application.

### Mar 25, 2026 - Enhanced Terrain Shader / Lighting Planning Prompt Captured

- Added planning prompt file:
	- `plans/enhanced_terrain_shader_lighting_prompt_2026-03-25.md`
- Purpose:
	- capture the current direction for an enhanced-quality terrain renderer path, shader-family reconstruction strategy, and lighting-model expansion without collapsing the active historical renderer into a speculative rewrite.
- Prompt requirements captured:
	- explicit `Historical` vs `Enhanced` render-mode architecture
	- terrain-only first vertical slice
	- render-quality UI/settings expansion
	- `LightService` expansion as a separate required seam from shader work
	- shader-family translation strategy for terrain, WMO/map-object, Model2, liquid, and particles
	- terrain decode/shading guardrails and real-data validation requirements
- Validation limits:
	- documentation/planning only
	- no code changes to the active renderer from this prompt file itself
	- no automated tests or builds were run for this planning-only slice

### Mar 25, 2026 - Enhanced Renderer Prompt Set Added

- Added focused companion planning prompts:
	- `plans/enhanced_renderer_plan_set_2026-03-25.md`
	- `plans/enhanced_renderer_architecture_prompt_2026-03-25.md`
	- `plans/enhanced_terrain_first_slice_prompt_2026-03-25.md`
	- `plans/shader_family_and_lighting_roadmap_prompt_2026-03-25.md`
- Purpose:
	- give Copilot narrower planning entry points instead of forcing every session through one umbrella renderer prompt.
- Split of responsibilities:
	- plan-set index selects the right prompt
	- architecture prompt covers runtime boundaries and mode ownership
	- first-slice prompt covers the first landable enhanced terrain implementation slice
	- roadmap prompt covers post-slice lighting and shader-family rollout
- Validation limits:
	- planning/documentation only
	- no renderer behavior changed by this prompt set

### Mar 25, 2026 - Minimap Tile-Scale Regression Repair

- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- reverted the Mar 24 `TileSize` swap for minimap camera position, pan clamping, and teleport math.
	- docked and fullscreen minimap now both use `WoWConstants.ChunkSize` again for the active viewer's `64x64` world-tile grid.
- `src/MdxViewer/MinimapHelpers.cs`
	- POI markers, taxi route polylines, taxi node markers, and shared minimap click-to-world conversion were restored to the same `ChunkSize`-based grid spacing.
- `src/MdxViewer/ViewerApp.cs`
	- the legacy `DrawMinimap_OLD()` fallback path was restored to the same minimap scale so the bad `TileSize` assumption does not survive outside the shared helper path.
- Root cause:
	- `WoWConstants.TileSize` was the wrong spacing for the active minimap path even though the name looked plausible; the live viewer still uses `WoWConstants.ChunkSize` for the `64x64` world-tile grid, so the prior swap pushed marker placement, overlays, pan bounds, and click-to-teleport out of sync.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-regression-repair/"` passed on Mar 25, 2026.
	- no automated tests were added or run.
	- no real-data runtime signoff yet on docked/fullscreen minimap behavior, marker placement, pan feel, or minimap teleport correctness.

### Mar 25, 2026 - Minimap Axis-Mapping Repair

- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- camera tile readout, pan clamping, and minimap teleport now treat the minimap click result as `(row, column)` and write it into renderer space with the correct axis order.
- `src/MdxViewer/MinimapHelpers.cs`
	- camera marker placement now uses the same row/column orientation as the tile grid instead of drawing the marker with transposed axes.
	- POI and taxi overlays now project with the same minimap axis order as the base tiles.
- `src/MdxViewer/ViewerApp.cs`
	- the legacy `DrawMinimap_OLD()` fallback path was updated to the same axis mapping so old code does not preserve the Designer Island/top-right teleport bug.
- Root cause:
	- the minimap grid is drawn with horizontal screen position from tile column and vertical screen position from tile row, but the teleport path and camera marker were still mixing row/column into renderer `X/Y` in the opposite order.
	- concrete runtime symptom: clicking the top-right Designer Island teleported the marker to the lower-left even though the minimap status text reported the intended tile.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-minimap-axis-repair/"` passed on Mar 25, 2026.
	- no automated tests were added or run.
	- no real-data runtime signoff yet on fullscreen/docked minimap marker alignment or top-right teleport correctness.

### Mar 25, 2026 - Fullscreen Minimap Release Blocker Closed

- The earlier fullscreen-minimap blocker status is now historical, not current.
- Final state for `v0.4.5`:
	- the bad `TileSize` minimap hypothesis was reverted
	- the later broad world-axis swap was also reverted
	- the landed fix is the narrower transpose-only repair recorded above, followed by runtime user confirmation that the previously broken top-right Designer Island scenario now behaves correctly
- Planning prompts remain useful as archaeology if the bug regresses again, but they should no longer be read as describing an active unresolved blocker.

### Mar 25, 2026 - Taxi Route Actor Prototype + Node Inspector Controls

- `src/MdxViewer/Terrain/TaxiPathLoader.cs`
	- taxi node loading now resolves mount metadata through the historical DBC chain:
		- `TaxiNodes.MountCreatureID[2]`
		- `Creature.DisplayID[4]`
		- `CreatureDisplayInfo.ModelID` + `CreatureModelScale`
		- `CreatureModelData.ModelName`
	- `TaxiNode` now exposes resolved mount creature IDs, display ID, scale, and model path for viewer-side taxi actor rendering.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- added animated taxi actor runtime support for selected taxi nodes/routes.
	- taxi actors now sample route waypoints, advance over time, and render through the existing MDX world-render path.
	- added viewer controls/state for `ShowTaxiActors` and `TaxiActorSpeedMultiplier`.
- `src/MdxViewer/ViewerApp.cs`
	- taxi list selection now routes through shared selection helpers instead of setting raw IDs directly.
	- viewport clicking can now pick visible taxi node indicators and sync them into the selected-object inspector state.
	- taxi selection now clears conflicting world/PM4 selections and populates selected-object info for node/route inspection.
- `src/MdxViewer/ViewerApp_Sidebars.cs`
	- the inspector now shows taxi-route controls when a taxi node or route is selected.
	- added the requested `Taxi Speed` slider plus a `Show Animated Taxi Actor` toggle.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer-taxi/"` passed on Mar 25, 2026.
	- a normal build to the default output path was blocked by the running `ParpToolsWoWViewer` process holding file locks.
	- no automated tests were added or run.
	- no real-data runtime signoff yet on taxi mount resolution, taxi node viewport picking, or in-scene route animation.

### Mar 24, 2026 - WMO Vertex-Light Prototype

- `src/MdxViewer/Rendering/WmoRenderer.cs`
	- WMO vertex buffers now include a baked vertex-light attribute alongside position, normal, and diffuse UV.
	- the renderer now consumes parsed `MOCV` colors when present and usable.
	- if parsed `MOCV` is missing but raw v14 lightmap payloads are present, it now samples preserved `MOLV` / `MOLD` / `MOLM` data into per-vertex baked-light colors during buffer build.
	- the WMO shader now multiplies the existing textured/diffuse lighting path by that baked-light color, which gives the active viewer a first object-light prototype without inventing a fake per-batch lightmap texture system.
- Scope limits:
	- not full client-faithful object-lightmap parity yet.
	- no dedicated batch-local lightmap texture binding path yet.
	- no runtime real-data signoff yet.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed.
	- no automated tests were added or run.
	- no real-data runtime validation was performed in this slice.

### Mar 24, 2026 - 0.5.3 Terrain/Object Render Fast-Path And Viewer Perf Gap

- Reverse-engineering only against the symbolized `0.5.3` client plus viewer code audit; no repo code changes landed in this slice.
- durable report extended at `documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
- confirmed `0.5.3` terrain-side render behavior relevant to performance/parity:
	- `CreateRenderLists` (`0x00698230`) is a real precompute/batch-build step for terrain texcoords/render lists
	- `RenderLayers` (`0x006a5d00`) and `RenderLayersDyn` (`0x006a64b0`) use locked GX buffers plus prepared chunk batches instead of a generic frame-time rebuild path
	- `0.5.3` terrain already has shader-assisted paths via `CMap::psTerrain` / `CMap::psSpecTerrain` plus `shaderGxTexture`
	- terrain draw cost is reduced by distance through runtime layer-count collapse (`textureLodDist`)
	- moving terrain layer behavior is now directly supported in the terrain path: runtime layer flag `0x40` triggers an extra texture transform indexed into the time-varying world transform tables
	- terrain shadows are drawn as a separate modulation pass
- confirmed `0.5.3` object/light behavior relevant to parity:
	- `RenderMapObjDefGroups` (`0x0066e030`) walks visible `CMapObjDefGroup` lists and dispatches group renders rather than using one generic world-object loop
	- `CreateLightmaps` (`0x006adba0`) allocates per-group lightmap textures and registers update callbacks
	- `RenderGroupLightmap(...)` and `RenderGroupLightmapTex(...)` tighten the lightmap conclusion further: the client has a dedicated group-lightmap render path with its own vertex stream and combine pass structure
	- `UpdateLightmapTex(...)` exposes CPU lightmap memory plus stride on `GxTex_Latch`, which supports a longer-lived lightmap texture path rather than ad hoc per-draw shading
	- `CalcLightColors` (`0x006c4da0`) computes substantially richer lighting state than the active viewer currently models (direct, ambient, multiple sky/cloud/water channels, fog, storm blending)
- viewer-side gap captured from the same slice:
	- `StandardTerrainAdapter` still actively uses `MPHD` only for big-alpha/profile handling and still flattens `MAIN` entries to tile presence
	- `TerrainRenderer` is still a generic base+overlay loop with only `MCLY 0x100` interpretation
	- `LightService` remains a simplified DBC interpolator
	- `WmoRenderer` / `MdxRenderer` still flatten renderer specialization heavily
	- `WorldScene` hot-path render work plus uncapped PM4 forensic budgets remain a practical perf risk when enabled, and the existing `RenderQueue` abstraction is not yet the active submission path for world rendering
- validation limits:
	- no automated tests were added or run
	- no viewer build or runtime real-data signoff was performed in this RE-only slice

### Mar 24, 2026 - WoW 2.0.0 Beta Ghidra Recon For M2 / Light / Particle Risk

- Reverse-engineering only against a loaded beta `2.0.0` client binary in Ghidra; no repo code changes landed in this slice.
- durable report added at `documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
- Confirmed engine-side anchors relevant to safe `2.x` support planning:
	- `FUN_00717b00` loads `shaders\vertex\Model2.bls` and `shaders\pixel\Model2.bls` for `Model2`.
	- `FUN_006b3b20` preloads map-object pixel BLS variants including translucent diffuse/specular programs.
	- terrain follow-up clarified:
		- the terrain shader split is now tighter than earlier notes:
			- `FUN_006a2360` loads `terrain1..4` into `DAT_00caf304..310` and `terrain1_s..4_s` into `DAT_00caf548..554`
			- `FUN_006cee30` uses those two contiguous tables as one-pass cached programs indexed by chunk layer count
			- `terrainp` / `terrainp_s` are the slower manual terrain fallback path inside `FUN_006cee30`
			- `terrainp_u` / `terrainp_us` are currently only confirmed in startup/shutdown, not yet in an active draw path
		- `XTextures\slime\slime.%d.blp` now traces into an animated `WCHUNKLIQUID` texture-family path through `FUN_0069b310` and its caller cluster
		- `WCHUNKLIQUID` rendering is not one single effect path: `FUN_006c65b0` dispatches modes `0/4/8` to animated texture-family renderers and modes `2/3/6/7` to a direct-coordinate path; `FUN_0069e200` builds cell strips for mode values `1/4/6`
		- `FUN_006c65b0` passes the raw mode nibble into `FUN_0069b310`, so liquid mode doubles as animated family index
		- currently recovered family table entries are `0=lake_a`, `1=ocean_h`, `2=lava`, `3=slime`, `4=lake_a` again; higher traced slots remain unresolved in this pass
		- novelty/dead-code candidates now include unresolved family slot `6`, unused `XTextures\river\fast_a.%d.blp`, and terrain-side `terrainp_u` / `terrainp_us` that still only show up in startup/shutdown
	- `FUN_0072d1a0` plus `FUN_0072cc60` / `FUN_0072cc90` / `FUN_0072cdc0` show `M2Light` objects being spatially bucketed / relinked at runtime instead of handled as a static flat light list.
	- `FUN_007c26c0`, `FUN_007ca9d0`, `FUN_007c3180`, and `FUN_007c79d0` show `ParticleSystem2` bootstrapping and runtime `CParticle2` / `CParticle2_Model` object storage, which keeps the smoke issue open on the renderer/runtime side.
	- `LightFloatBand.dbc`, `LightIntBand.dbc`, `LightParams.dbc`, `Light.dbc`, and `LightSkybox.dbc` all use strict `WDBC` loaders with schema checks and ID->row pointer tables.
- Current conclusion:
	- later `2.x` profile routing is a reasonable structural start, but real parity risk sits in shader/material selection and light/particle runtime interpretation, not in raw table loading.
- Validation limits:
	- no automated tests were added or run.
	- no viewer build or runtime signoff was performed as part of this RE-only slice.

### Mar 24, 2026 - Later 2.x M2-Family Profile Routing Enablement

- `src/MdxViewer/Terrain/FormatProfileRegistry.cs`
	- added `M2Profile_20x_Unknown` for later `2.x` / TBC-era model routing.
	- active `2.x` window is now `MD20` with versions `0x104..0x107` and the existing parser split threshold remains `0x108`.
- `src/MdxViewer/ViewerApp.cs`
	- fallback build options now include `2.4.3.8606`, so the viewer can select a later `2.x` model profile even without `Map.dbd` build metadata.
- `src/MdxViewer/Rendering/ReplaceableTextureResolver.cs`
	- added short-build alias support for `2.4.3 -> 2.4.3.8606`.
- `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs`
	- neutralized the profiled legacy `MD20` trace wording so TBC routing no longer logs as if it were only the pre-release `3.0.1` path.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"` passed.
	- no automated tests were added or run.
	- no runtime real-data signoff yet on an actual later `2.x` client dataset.

### Mar 24, 2026 - 0.12 Standalone Model Browser Recovery

- `src/MdxViewer/DataSources/MpqDataSource.cs`
	- loose-file indexing now includes `.mdl`
	- Alpha nested wrapper scan now includes `.mdx.MPQ`, `.mdl.MPQ`, and `.m2.MPQ`
	- nested model wrappers now register alternate model-extension aliases into the file set / Alpha wrapper cache so the same wrapped asset can resolve through `.mdx`, `.mdl`, or `.m2`
- `src/MdxViewer/ViewerApp.cs`
	- the browser-side `.mdx` file bucket now includes early `.mdl` assets as part of the same standalone model family
	- standalone disk loads now accept `.mdl`
	- unsupported standalone M2-family loads now fail early with a clear error when the active build has no resolved `M2Profile`
- `src/MdxViewer/ViewerApp_Sidebars.cs`
	- the file-browser type selector now labels the early-model bucket as `.mdx/.mdl`
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed with existing warnings
	- no automated tests were added or run
	- no runtime real-data signoff yet on a real `0.12` client dataset

### Mar 24, 2026 - 0.6.0 Through 2.x Terrain Alpha Grid Regression Fix

- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
	- legacy standard-ADT terrain alpha decode no longer uses the naive sequential nibble expansion for the entire `0.6.0` through `2.x` band.
	- `TerrainAlphaDecodeMode.LegacySequential` now prefers relaxed MCAL per-layer decode with inferred layer spans and preserved `DoNotFixAlphaMap` handling.
	- fallback legacy 4-bit decode now goes through the existing row-aware unpack + legacy edge-fix helpers, which is the actual seam tied to the chunk-grid artifact.
- Build follow-up required by the same slice:
	- the earlier in-progress minimap candidate-path patch still had compile errors in `src/MdxViewer/Rendering/MinimapRenderer.cs`; those were corrected so the terrain change could be validated with a real solution build.
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug` passed.
	- no automated tests were added or run.
	- no runtime real-data validation yet on the affected legacy terrain clients.

### Mar 24, 2026 - v0.4.5 Branding + MH2O LiquidType Classification Fix

- `src/MdxViewer/ViewerApp.cs`
	- viewer window titles now use `parp-tools WoW Viewer`
	- Help -> About now opens a modal with version, author, and credits instead of only setting a status line
	- standard terrain world loads now pass the active DBC provider/build metadata into `StandardTerrainAdapter`
- `src/MdxViewer/MdxViewer.csproj` and `src/MdxViewer/MdxViewer.CrossPlatform.csproj`
	- version metadata now targets `0.4.5`
	- the emitted assembly/executable name is now `ParpToolsWoWViewer`
- `.github/workflows/release-mdxviewer.yml`
	- release workflow is now branded for `parp-tools WoW Viewer`
	- workflow dispatch example now points at `v0.4.5`
	- build environment now uses .NET 10 and publishes `parp-tools-wow-viewer-<version>-win-x64.zip`
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
	- `MH2O` liquid family selection now prefers `LiquidType.dbc.Type` through DBCD instead of treating `LiquidTypeId` as a direct render class
	- fallback behavior now still handles later 3.3.5 / 4.0 liquid IDs when DBC metadata is unavailable
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/Liquids/LiquidConverter.cs`
	- shared fallback `LiquidTypeId -> MCLQ family` mapping now recognizes `13`, `14`, `17`, `19`, and `20`
- Validation limits:
	- build only: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"` passed
	- no automated tests were added or run
	- no runtime real-data signoff yet on the corrected 3.3.5 / 4.0 liquid-family rendering

### Mar 24, 2026 - PM4 Color Legend + Selected-Object Graph UI

- `src/MdxViewer/Terrain/WorldScene.cs`
	- now exposes viewer-derived PM4 legend data for the active color mode instead of forcing users to reverse-map swatches by eye
	- now exposes a selected-object PM4 hierarchy summary built from the current overlay assembly using:
		- CK24 root group
		- MSLK-linked subgroup
		- optional MDOS subgroup
		- connectivity/object-part leaf nodes
- `src/MdxViewer/ViewerApp.cs`
	- world-objects PM4 controls now show a `PM4 Color Legend` block under the color-mode selector so `MSUR Attr Mask` and other categorical modes are directly identifiable by value/count
- `src/MdxViewer/ViewerApp_Sidebars.cs`
	- selected PM4 objects now show a `PM4 Graph` tree in the inspector so users can inspect link-group / MDOS / part structure for the clicked object
- Mar 24 follow-up on the same PM4 UI slice:
	- PM4 graph leaf rows are now actionable: clicking a part reselects that exact PM4 part and `Frame` moves the camera to it
	- the selected PM4 graph can now be exported as JSON for later PM4 research/planning work
- Planning follow-up:
	- `plans/unified_format_io_overhaul_prompt_2026-03-23.md` now records the current pragmatic PM4 hierarchy contract and the rule that centroids are derived anchors, not raw PM4 graph nodes
- Validation limits:
	- no automated tests were added or run
	- runtime real-data signoff is still required before claiming the new graph view fully matches raw PM4 ownership semantics

### Mar 23, 2026 - Viewer Tool Dialogs Now Reuse Active Client / Loose Overlay Paths

- The viewer already retained enough session state to stop forcing repeated path browsing across several tools:
	- active MPQ base client path via `MpqDataSource.GamePath`
	- attached loose overlay roots via `MpqDataSource.OverlayRoots`
	- current loaded map name via `TerrainManager.MapName` / `VlmTerrainManager.MapName`
- `src/MdxViewer/ViewerApp.cs`
	- tool menu actions now prepare dialog inputs before opening:
		- `Generate VLM Dataset`
		- `Terrain Texture Transfer`
		- `Map Converter`
		- `WMO Converter`
	- added helper methods that resolve the current session’s base client, loose overlay, map directory, and WDT path from the already loaded viewer state instead of making the user browse for them again.
- Important current behavior:
	- VLM export prefers the active MPQ base path and current map name.
	- terrain transfer prefers loose overlay map dir as source and base-client map dir as target when both exist.
	- map converter seeds from the current map WDT/map dir when a usable on-disk path exists.
	- standalone WMO conversion still auto-seeds from the currently loaded WMO file.
- Validation limits:
	- file diagnostics on `src/MdxViewer/ViewerApp.cs` were clean after the change.
	- no automated tests were added or run.
	- no new full viewer build or runtime real-data signoff was recorded for this exact slice.

### Mar 23, 2026 - Unified Format I/O Overhaul Proposal Captured

- The user wants the current scattered terrain/model/WMO knowledge moved into one shared read/write library used by all tooling.
- Explicit proposal direction now captured for follow-up planning:
	- one shared format I/O library for Alpha, LK 3.3.5, and relevant 4.x read/write paths
	- terrain + placement + model + WMO conversion under one orchestration surface
	- retire the split where `MdxViewer` has newer runtime-read knowledge while `WoWMapConverter.Core` still carries older write/conversion assumptions
	- do not over-claim Alpha placement downconversion until MODF/MDDF write support is actually implemented and validated
- Planning prompt added at `plans/unified_format_io_overhaul_prompt_2026-03-23.md`.

### Mar 23, 2026 - Viewer Docs Refresh + Render Quality Follow-Up

- Initial doc refresh was followed by a user rewrite of `src/MdxViewer/README.md` to remove bad assumptions and make the support/workflow description more grounded.
- Preserve that correction for future handoff:
	- do not overstate platform restrictions
	- do not overstate supported versions beyond the user-corrected README
	- do not write branch-local language into docs intended for eventual `main`
	- keep the render-quality statement narrow: texture filtering is the landed win; MSAA availability is context-dependent and not required for this slice
- Validation limits:
	- the active viewer solution had already built successfully on Mar 23 before the doc refresh
	- no automated tests were added or run for the documentation update

### Mar 22, 2026 - Viewer Debug Workflow Follow-Up: PM4 OBJ Export + Minimap Guardrails + Terrain Hole Override

- Added a viewer-side offline PM4 OBJ export path so PM4 inspection no longer depends only on the live overlay's currently loaded subset.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- now exports per-tile OBJ, per-object OBJ, and `pm4_obj_manifest.json` from direct PM4 file scans against the active data source
- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- now exposes `Export PM4 OBJ Set` in the PM4 utilities UI
- `src/MdxViewer/ViewerApp.cs`, `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`, `src/MdxViewer/Rendering/MinimapRenderer.cs`
	- minimap teleport now requires triple-clicking the same tile within the confirmation window
	- minimap drag-vs-click discrimination now uses full drag-origin distance
	- minimap zoom/pan/window visibility now persist in viewer settings
	- decoded minimap tiles now cache on disk under `output/cache/minimap/<cache-segment>`
- `src/MdxViewer/Terrain/TerrainMeshBuilder.cs`, `src/MdxViewer/Terrain/TerrainManager.cs`, `src/MdxViewer/Terrain/VlmTerrainManager.cs`, `src/MdxViewer/ViewerApp_Sidebars.cs`
	- added viewer-side terrain hole override controls
	- loaded terrain tiles can now be rebuilt with `HoleMask` ignored either globally or for the current camera tile
	- this does not edit ADT data on disk; it is a mesh rebuild / debug visibility feature only
- Validation limits:
	- file diagnostics were clean on the edited viewer files
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
	- no automated tests were added or run
	- no runtime real-data signoff yet on PM4 OBJ parity, minimap UX/cache behavior, or terrain-hole rebuild behavior

### Mar 21, 2026 - Standalone PM4 Research Library

- Added `src/Pm4Research.Core` as a fresh-start PM4 reading library independent from the active viewer reconstruction path.
- Current implementation:
	- walks PM4 files chunk-by-chunk and preserves raw payloads with offsets/sizes
	- independently decodes the currently understood chunk layouts: `MVER`, `MSHD`, `MSLK`, `MSPV`, `MSPI`, `MSVT`, `MSVI`, `MSUR`, `MSCN`, `MPRL`, `MPRR`, `MDBH`, `MDBI`, `MDBF`, `MDOS`, `MDSF`
	- exposes `Pm4ResearchFile` and `Pm4ExplorationSnapshot` so future rediscovery work can compare raw chunk evidence without going through `WorldScene`
	- exposes decode-audit reports so future rediscovery work can measure chunk-size consistency and cross-chunk reference validity before inferring object semantics
- Why this was added:
	- current PM4 viewer work has hit repeated transform-contract ambiguity
	- the user requested a fresh perspective instead of continuing to layer fixes onto the existing PM4 read/reconstruct path
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Core/Pm4Research.Core.csproj -c Debug` passed.
	- no automated tests added or run.
	- no runtime real-data validation yet; this is groundwork for fresh PM4 format exploration.

### Mar 21, 2026 - PM4 Decode-Confidence Audit Pass

- Added raw decode-audit commands to `src/Pm4Research.Cli`:
	- `inspect-audit`
	- `scan-audit`
- The audit path measures:
	- chunk presence versus populated-data presence
	- stride/size consistency for the typed chunk layouts
	- cross-chunk reference validity for `MSVI -> MSVT`, `MSPI -> MSPV`, `MSUR -> MSVI`, `MSLK -> MSUR`, `MSLK -> MSPI`, `MDSF -> MSUR`, `MDSF -> MDOS`, and `MDOS -> MDBH`
- Real-data findings from the full development corpus (`616` PM4 files):
	- zero file-walk overrun or trailing-byte diagnostics
	- zero unknown chunk signatures after adding typed support for the documented destructible-building chunks
	- recurring decode structure is carried by the `MS*` / `MPR*` families, not by Wintergrasp destructible chunks
	- `MDBI` and `MDBF` are one-tile only in this corpus, and `MDBH` / `MDOS` / `MDSF` only carry populated destructible-building payload on the trusted `development_00_00.pm4` reference tile; their wider corpus presence is mostly placeholder/empty chunk stubs
	- meaningful open seam surfaced by the audit: aggregate `MSLK.RefIndex -> MSUR` mismatches still exist in the corpus, so the current standalone `MSLK.RefIndex == MSUR index` assumption is not fully closed
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- inspect-audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-audit --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed.
	- no automated tests were added or run.

### Mar 21, 2026 - Targeted `MSLK.RefIndex` Mismatch Audit

- Added dedicated commands to `src/Pm4Research.Cli`:
	- `inspect-mslk-refindex`
	- `scan-mslk-refindex`
- The new audit reports invalid `MSLK.RefIndex >= MSUR.Count` cases by tile and records which other PM4 index domains those values still fit on the same file.
- Real-data findings from the full development corpus:
	- `150` files contain `4553` total `MSLK.RefIndex -> MSUR` mismatches
	- the trusted `development_00_00.pm4` tile has zero such mismatches, so it is not the main linkage-problem reference tile
	- bad `RefIndex` values almost never fit `MPRL` counts, which weakens the idea that the unresolved `RefIndex` population is simply pointing into `MPRL`
	- many bad values still fit within `MSLK`, `MSPI`, `MSVI`, and `MSCN` counts on the affected files, making those domains stronger next-step candidates
	- some mismatch-heavy tiles show repeated `LinkId` clusters, which may help isolate families of alternate `RefIndex` semantics in a follow-up pass
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- inspect-mslk-refindex --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development/development_00_00.pm4` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed.
	- no automated tests were added or run.

### Mar 21, 2026 - Comprehensive PM4 Unknowns Map

- Added `scan-unknowns` to `src/Pm4Research.Cli` and `Pm4ResearchUnknownsAnalyzer` to `src/Pm4Research.Core`.
- Purpose:
	- stop spreading PM4 unknowns across isolated ad-hoc commands and stale notes
	- produce one corpus-scale report that states which raw PM4 relationships are verified, which are partial, and which fields remain open
- The report now covers:
	- chunk population vs populated payload counts
	- verified raw edges such as `MSUR -> MSVI`, `MSVI -> MSVT`, `MSLK -> MSPI`, `MSPI -> MSPV`, and `MDSF -> {MSUR, MDOS}`
	- partial/open edges such as `MSLK.RefIndex`, `MSLK.GroupObjectId -> MPRL.Unk04`, `MPRR.Value1`, and `MDOS.buildingIndex`
	- field distributions for `MSHD`, `MSLK`, `MSUR`, `MPRL`, and `MPRR`
	- `LinkId` pattern summary and `MSLK.MspiIndexCount` ambiguity buckets
	- a generated unknown list with evidence and next proof tasks
- Key real-data findings from the current corpus run:
	- `LinkId` is uniformly `0xFFFFYYXX` in the current dataset
	- `MPRL.Unk02` is always `-1` and `Unk06` is always `0x8000`
	- `MPRL.Unk14` ranges `-1..15` and still looks floor-like; `Unk16` collapses to two values (`0x0000`, `0x3FFF`)
	- `MSLK.MspiIndexCount` has no triangles-only evidence in the current corpus, but still has a large overlap bucket where both interpretations fit
	- `MPRR` remains a mixed/open field family; current counts do not justify naming it as purely `MPRL` or purely geometry-facing
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-unknowns --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed.
	- no automated tests were added or run.

### Mar 21, 2026 - MSCN Relationship Report

- Added `scan-mscn` to `src/Pm4Research.Cli` and `Pm4ResearchMscnAnalyzer` to `src/Pm4Research.Core`.
- Purpose:
	- stop treating MSCN as a vague side channel or re-importing old rollback claims without revalidating them in the standalone raw path
	- measure MSCN directly against CK24, `MSUR.MdosIndex`, mesh-side geometry, and `MSLK.GroupObjectId`
- The report now covers:
	- `MSUR.MdosIndex -> MSCN` validity across the full corpus
	- CK24-group MSCN coverage and mesh+MSCN coexistence
	- raw-vs-swapped MSCN bounds overlap against CK24 mesh bounds
	- low16 / low24 `MSLK.GroupObjectId` fits against CK24 identity layers
	- MSCN coordinate-space buckets against file tile coordinates
- Key real-data findings from the current corpus run:
	- `MSUR.MdosIndex -> MSCN` is strong but not closed (`511891` fits, `6201` misses)
	- `1886 / 1895` CK24 groups carry valid MSCN-backed node coverage
	- raw MSCN bounds overlap CK24 mesh bounds far more often than swapped-XY MSCN bounds (`1162` vs `10` fits)
	- current standalone corpus evidence does not support the older blanket claim that MSCN is simply world-space plus XY swap
	- `MSLK.GroupObjectId` is not a direct full CK24 key (`0 / 1272796` low24 fits) and only weakly overlaps CK24 low16 object ids (`399 / 1272397`)
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mscn --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed.
	- no automated tests were added or run.

### Mar 21, 2026 - PM4 Linkage / CK24 ObjectId Report

- Added `scan-linkage` to `src/Pm4Research.Cli` and `Pm4ResearchLinkageAnalyzer` to `src/Pm4Research.Core`.
- Purpose:
	- stop treating the UI `Ck24ObjectId` label as if it were a separately proven PM4 identity field
	- join `MSLK.RefIndex` mismatches, `MSLK.GroupObjectId`, CK24 identity layers, and bad `MSUR.MdosIndex` clusters in one corpus report
- The report now covers:
	- low16 and low24 `MSLK.GroupObjectId` fits against CK24 identity layers on mismatch entries
	- file-local reuse of non-zero `Ck24ObjectId` across multiple full CK24 values and type bytes
	- top mismatch families grouped by `LinkId + TypeFlags + Subtype`
	- top bad-`MdosIndex` CK24 clusters
- Key real-data findings from the current corpus run:
	- the UI `Ck24ObjectId` is just the low 16 bits of `MSUR.PackedParams -> CK24`
	- that low16 layer is usually a near one-to-one slice of full CK24 in-file, not a broadly reused hierarchy id (`2` reuse cases out of `1601` analyzed non-zero object-id groups)
	- both reuse cases occur on tile `36_24`, where one low16 object id survives across two full CK24 values and two type bytes
	- `MSLK.GroupObjectId` remains weak as the missing identity/hierarchy answer for unresolved `RefIndex` mismatches (`16` low16 matches, `15` low24 matches across `4553` mismatches)
	- `58` files carry bad `MSUR.MdosIndex` references, including several large non-zero CK24 families, not only `CK24=0` aggregates
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-linkage --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development` passed.
	- no automated tests were added or run.

### Mar 21, 2026 - PM4 Structure-Confidence / Decode-Trust Report

- Added `scan-structure-confidence` to `src/Pm4Research.Cli` and `Pm4ResearchStructureConfidenceAnalyzer` to `src/Pm4Research.Core`.
- Purpose:
	- stop collapsing two separate questions into one:
		- "is this chunk layout byte-level real?"
		- "are these field names and meanings actually proven?"
	- give the standalone PM4 path an explicit guardrail against inherited GPT-era structure lore and stale rollback assumptions
- The report now covers:
	- chunk-level layout confidence vs semantic confidence vs hallucination risk
	- field-level classification (`verified-reference`, `derived-bit-slice`, `named-guess`, `conflicted-reference`, `sparse-reference`, etc.)
	- explicit source-conflict inventory where older notes or field names overstate certainty
	- one summary that counts how much of the current standalone decoder is truly byte-closed versus only semantically guessed
- Key real-data findings from the current corpus run:
	- `13` tracked chunk families currently land in `high` layout confidence on the fixed development corpus
	- semantic confidence is much weaker:
		- `1` field `high`
		- `4` fields `medium`
		- `10` fields `low`
		- `4` fields `very-low`
	- strongest current byte+semantic anchors are `MSPV`, `MSPI`, `MSVT`, `MSVI`, `MSUR -> MSVI`, and `MDSF -> {MSUR, MDOS}`
	- highest current hallucination-risk zones are `MSLK.RefIndex`, `MSUR` bytes `4..19`, `MPRR.Value1`, `MPRL.Unk04/14/16`, and sparse destructible fields such as `MDOS.buildingIndex`
	- the new conflict inventory now records concrete overstated legacy claims around `MSLK.LinkId`, `MSLK.RefIndex`, `MSUR.MdosIndex`, `MSUR.Normal + Height`, MSCN coordinate frame, and `MPRR.Value1`
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-structure-confidence --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_structure_confidence_report.json` passed.
	- no automated tests were added or run.

### Mar 21, 2026 - MSUR Geometry Audit + RefIndex Family Classifier + Placement-Truth Refresh

- Added two new standalone PM4 commands in `src/Pm4Research.Cli`:
	- `scan-msur-geometry`
	- `scan-mslk-refindex-classifier`
- Added two new standalone analyzers in `src/Pm4Research.Core`:
	- `Pm4ResearchMsurGeometryAnalyzer`
	- `Pm4ResearchMslkRefIndexClassifier`
- Purpose:
	- close the specific decoder-trust seam around whether `MSUR` bytes `4..19` are real geometric fields or inherited naming fiction
	- replace the undifferentiated `4553`-row `MSLK.RefIndex` mismatch blob with likely target-domain family buckets
	- advance step 3 pragmatically by reusing the existing active `pm4-validate-coords` path for real `_obj0.adt` placement truth instead of pretending standalone PM4 can prove that by itself
- Key real-data findings from `scan-msur-geometry`:
	- analyzed surfaces: `518092`
	- degenerate surfaces: `0`
	- unit-length stored normals: `518092 / 518092`
	- strong positive stored-vs-geometry normal alignment: `518092 / 518092`
	- the trailing float currently named `Height` behaves like the negative plane-distance term along the stored normal, with best candidate `storedPlane.-` mean absolute error `0.00367829`
	- practical correction: `MSUR` bytes `4..19` are no longer a top decoder hallucination-risk seam, but the final float is semantically better described as a signed plane term than as generic height
- Key real-data findings from `scan-mslk-refindex-classifier`:
	- files with mismatches: `150`
	- total mismatch rows: `4553`
	- resolved/classified families: `505`
	- ambiguous families still remaining: `344`
	- resolved rows covered by classified families: `2651`
	- the classifier uses lift above corpus baseline so domains like `MPRR` do not win just by size (`98.4%` raw fit baseline)
	- largest current resolved family population: `probable-MSVT` (`293` families), with smaller but real `MSPI` / `MSPV` / `MSVI` / `MSCN` / `MPRL` slices
- Key real-data findings from the placement-truth refresh using existing active `pm4-validate-coords`:
	- tiles scanned: `616`
	- tiles validated against `_obj0.adt` placements: `206`
	- `MPRL` refs inside expected tile bounds: `114301 / 114301` (`100.0%`)
	- `MPRL` refs within `32` units of a nearest placement: `107907 / 114301` (`94.4%`)
	- average nearest placement distance: `10.98`
	- this materially strengthens `MPRL.Position` against real object-placement truth; it does **not** close `MPRR`
- Follow-up correction to decode-trust state:
	- refreshed `scan-structure-confidence` after these audits
	- semantic confidence counts moved from `1/4/10/4` (`high/medium/low/very-low`) to `2/4/9/4`
	- highest current hallucination-risk zones are now `MSLK.RefIndex`, `MPRR.Value1`, `MPRL.Unk04/14/16`, and sparse destructible fields such as `MDOS.buildingIndex`
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-msur-geometry --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_msur_geometry_report.json` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex-classifier --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --output i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_mslk_refindex_classifier_report.json` passed.
	- `dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- pm4-validate-coords --input i:/parp/parp-tools/gillijimproject_refactor/test_data/development/World/Maps/development --json i:/parp/parp-tools/gillijimproject_refactor/output/pm4_reports/development_pm4_coordinate_validation_report.json` passed, but only after a broader active-tree build that emitted many unrelated existing warnings.
	- no automated tests were added or run.

### Mar 21, 2026 - PM4 Research Workflow Validation On Trusted Tile `development_00_00.pm4`

- Validated the standalone PM4 research workflow on `test_data/development/World/Maps/development/development_00_00.pm4`, which is now the preferred reference tile for PM4 rediscovery work.
- Real-data findings from the standalone CLI on that tile:
	- `54` chunks total
	- `MSPV=8778`, `MSVT=6318`, `MSCN=9990`, `MPRL=2493`
	- top CK24 groups include large multi-surface families such as `0x40AA0A`, `0x418D9F`, and `0x421809`
	- `MPRL.Unk04` still spans only about `0.01°..22.30°` on this tile in the standalone read path, consistent with earlier viewer forensics that it is not a simple absolute building-yaw field here
- Dataset note:
	- the matching original `00_00` ADT triplet is not present anywhere in the workspace, so repository-side validation is currently limited to PM4 analysis and not full in-repo PM4-vs-ADT visual signoff
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Core/Pm4Research.Core.csproj -c Debug` passed.
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/Pm4Research.Cli/Pm4Research.Cli.csproj -c Debug` passed.
	- no automated tests were added or run.
	- real-data PM4 analysis was performed on the file above, but no in-repo ADT-backed runtime signoff was possible because the companion ADTs are not in the workspace.

### Mar 21, 2026 - MPRL Semantic Correction From User Domain Knowledge

- User clarified the intended semantics of `MPRL` explicitly:
	- `MPRL` points are literal terrain/object collision-footprint intersections.
	- they mark the `XYZ` positions where ADT terrain is pierced by the object for collision stitching.
	- this makes terrain and object part of the same collision mesh at those points.
- Consequence for active PM4 work:
	- reject the old whole-object `MPRL` center/bounds translation idea, but do not reduce `MPRL` to vague anchor noise.
	- treat `MPRL` as collision-footprint reference data when scoring or comparing PM4 object hypotheses.
- Research-tooling follow-up:
	- `Pm4Research.Core` object hypotheses now include `MPRL` footprint counts against raw PM4 bounds so corpus analysis can see which candidate objects actually capture linked/tile-level `MPRL` seam points.

### Mar 21, 2026 - PM4 Link-Decode Sanity Fix + Linked-MPRL Summary Instrumentation

### Mar 21, 2026 - PM4 Object-Local Base Frame Layer

### Mar 21, 2026 - PM4 MPRL Axis Contract Correction

- Follow-up after direct comparison with older PM4 R&D exports and `WoWRollback/Pm4Reader` forensic notes.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- viewer-side `MPRL` position handling no longer assumes ADT-style planar `X/Z`, vertical `Y`.
	- the common `XY+Zup` mesh path now restores the older fixed `MSVT` viewer/world basis `(Y, X, Z)` instead of treating raw `(X, Y, Z)` as already canonical.
	- PM4 axis convention is now detected once per file and reused across CK24 groups instead of being redetected per CK24.
	- `BuildMprlPlanarPoints(...)`, `NearestPositionRefDistanceSquared(...)`, and `BuildPm4PositionRefMarkers(...)` now all convert `MPRL.Position` to world as `(PositionX, PositionZ, PositionY)` to match that restored `MSVT` basis.
- Why this was needed:
	- older PM4 forensics matched `MPRL` fields against raw `MSVT` axes, but the active viewer also needs to fold in the older successful `MSVT -> (Y, X, Z)` world basis from the R&D exporter.
	- without that fixed `MSVT` basis, the viewer was trying to approximate the right layout with per-object swap/invert heuristics, which can push PM4 into mirrored or polar-opposite fits against real WMO/M2 placements.
	- keeping axis convention per CK24 could still let neighboring wall/object fragments choose different mesh bases, which matched the remaining “random offset / mirrored” runtime symptom.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings.
	- no automated tests added or run.
	- no runtime real-data signoff yet that this restores PM4 placements on the affected tiles.

- Follow-up after the user chose the structural PM4 path instead of another heuristic yaw tweak.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- PM4 overlay objects now retain placed-space bounds/center, but their line and triangle geometry is localized around a preserved linked-group placement anchor instead of each fragment center.
	- each PM4 overlay object now carries a baked base placement transform that restores that anchored local geometry into the solved placed frame during rendering/export.
	- PM4 batched overlay rendering now applies `base placement -> overlay/object transforms` instead of assuming PM4 geometry is already final placed geometry.
	- PM4 JSON export re-applies the baked base transform so exported geometry remains in placed space.
- Why this was needed:
	- the viewer previously flattened PM4 geometry directly into placed space too early, which made “object inside container” reasoning and future placement-frame work harder.
	- the earlier experiment to move placement ownership down to linked subgroups regressed coherence and was reverted; this local-frame layer is structural groundwork without changing the CK24 solve boundary.
	- follow-up runtime diagnosis showed that rebasing split objects to per-fragment centers also discarded their original linked-group placement offsets, so split parts now preserve the pre-split anchor.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings.
	- no automated tests added or run.
	- no runtime signoff yet that this closes the remaining PM4 natural-rotation mismatch.

- Runtime forensics on `development_00_00.pm4` found that the active viewer path was consulting legacy `MSLK` fields that were never actually populated by `WoWMapConverter.Core`:
	- `MslkEntry.MsurIndex` defaulted to `0`
	- `WorldScene` was still checking that field when splitting surface groups and collecting linked `MPRL` refs
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/PM4/Pm4File.cs`
	- legacy `MSLK` entries now initialize unsupported fields to sentinels instead of zero (`MsurIndex = uint.MaxValue`, `MsviFirstIndex = -1`, `MsviIndexCount = 0`)
	- this prevents fake `surface 0` associations from leaking into PM4 viewer grouping/link logic
- `src/MdxViewer/Terrain/WorldScene.cs`
	- PM4 overlay objects now carry a linked-`MPRL` summary payload: total refs, normal refs, terminators, floor min/max, heading min/max/mean
	- PM4 JSON interchange export now includes the same linked-`MPRL` summary per object
- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- selected-object PM4 debug UI now shows linked-`MPRL` heading/floor summary directly in the alignment panel
- Tile-specific forensic result recorded from the raw dump of `development_00_00.pm4`:
	- `MPRL.Unk04` only spans about `0.01° .. 22.3°` across the tile, so it is not behaving like a simple absolute building-yaw field here
	- `Unk06` is constant `0x8000`
	- `Unk16` splits normal entries from terminators
	- `Unk14` still looks floor-like
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
	- no automated tests added or run
	- no runtime signoff yet on final PM4 object orientation

### Mar 21, 2026 - PM4 Documentation Refresh

- Refreshed `documentation/pm4-current-decoding-logic-2026-03-20.md` into a current viewer-side PM4 reconstruction contract instead of leaving PM4 behavior spread across many memory-bank entries and stale experiments.
- The updated document now records:
	- the three distinct PM4 reading layers: raw file data, linked object assembly, and final viewer render derivation
	- the active CK24 reconstruction pipeline in `WorldScene.BuildPm4TileObjects(...)`
	- the current `MPRL` contract as anchor/scoring input, not linked-center translation ownership
	- the stronger negative result from runtime evidence: current PM4 viewer behavior does not support an `MPRL` bounding-box/container paradigm
	- the split planar-candidate policy for tile-local versus world-space PM4
	- the `12°` coarse-only yaw correction guardrail
	- the list of rejected experiments that should not be reintroduced casually
- Updated memory-bank active-context files to point future PM4 work at that document first.
- Validation limits:
	- documentation and memory-bank update only; no code changes were made in this slice.
	- no automated tests were added or run.

### Mar 21, 2026 - Dockspace UI Recovery + PM4 Translation Rollback

### Mar 21, 2026 - Viewer PM4/WMO Correlation Export

- User asked to stop treating PM4, ADT placements, and WMO mesh data as separate lanes and to wire the correlation path into `MdxViewer` itself.
- `src/MdxViewer/Terrain/WorldAssetManager.cs`
	- added `WmoMeshSummary` and a new `TryGetWmoMeshSummary(...)` path.
	- factored WMO parsing so the existing v14/v17 read path can be reused for correlation output without depending on a live `WmoRenderer`.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- added `BuildPm4WmoPlacementCorrelationJson(...)`.
	- the export walks currently loaded tile WMO placements, derives WMO mesh summaries/local bounds, and ranks nearby PM4 overlay objects by tile-neighborhood plus bounds-gap / overlap heuristics.
	- output includes ADT placement identity, WMO mesh counts/bounds, and PM4 object metadata such as `CK24`, object part, linked-ref counts, and dominant `MSUR` fields.
- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- added `Dump PM4/WMO Correlation JSON` to the existing `PM4 Alignment` window.
	- save flow matches the existing PM4 object JSON export path.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings.
	- no automated tests were added or run.
	- no runtime real-data validation was performed for the new export workflow.

### Mar 21, 2026 - Viewer PM4/WMO Correlation Panel + Footprint Scoring Follow-Up

- User chose the next PM4/WMO steps explicitly:
	- add a live in-viewer correlation panel instead of staying export-only
	- strengthen ranking with actual transformed geometry / footprint comparison instead of only AABB heuristics
- `src/MdxViewer/Terrain/WorldAssetManager.cs`
	- extended `WmoMeshSummary` with sampled WMO geometry points so footprint comparison can reuse cached parse output instead of reopening meshes during report generation.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- added `BuildPm4WmoPlacementCorrelationReport(...)` as a typed viewer-side report model behind the JSON export.
	- correlation ranking now incorporates footprint overlap, footprint area similarity, and symmetric hull-distance metrics derived from transformed WMO sample geometry and PM4 object footprint hulls.
	- added `SelectPm4Object(...)` so a reported candidate can be promoted directly into the live PM4 selection state.
- `src/MdxViewer/ViewerApp.cs`
	- added persistent window/filter state for a new `PM4/WMO Correlation` tool window.
	- added a `View` menu toggle and a `PM4/WMO` launch button beside the existing PM4 controls.
- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- added a real `PM4/WMO Correlation` window with refresh, near-only filtering, model/path filtering, placement browsing, candidate drill-down, PM4 selection, and camera framing actions.
	- `PM4 Alignment` now links directly into that panel instead of only offering JSON export.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings after the follow-up.
	- no automated tests were added or run.
	- no runtime real-data validation has been performed yet for the new panel interactions or the footprint-based ranking quality.

- Latest user report after the ViewerApp partial split and earlier PM4 MPRL-frame experiment:
	- `World Maps` starting collapsed was wrong.
	- PM4 alignment had gotten worse.
	- the viewer still needed a real dock-panel UI.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- removed the linked-`MPRL` bounds-center translation path from `BuildPm4TileObjects(...)`.
	- removed the viewer-only `TryResolveMprlAuthoritativeAdjustment(...)` translation stage and the extra `worldTranslation` plumbing through PM4 line/triangle conversion.
	- kept the earlier geometry-pivot path plus coarse yaw-correction logic instead of forcing linked CK24 groups into one translated `MPRL` center.
- `src/MdxViewer/ViewerApp.cs`
	- enabled ImGui docking in source.
	- added a dockspace host between the menu/toolbar region and the status bar.
	- added a `View -> Dock Panels` toggle.
	- scene viewport math no longer subtracts fixed sidebar widths.
- `src/MdxViewer/ViewerApp_Sidebars.cs`
	- restored `World Maps` to default-open on first draw.
	- left/right shell panels can now run as normal titled dockable windows (`Navigator`, `Inspector`) when dock panels are enabled, while preserving the older fixed-sidebar mode as fallback.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings.
	- no automated tests added or run.
	- no runtime real-data signoff yet on PM4 alignment recovery or on the dock-panel workflow.

### Mar 21, 2026 - ViewerApp Partial-Class Refactor

- User asked for the oversized viewer shell file to be broken up instead of continuing to grow `src/MdxViewer/ViewerApp.cs` as one 6000+ line class.
- `src/MdxViewer/ViewerApp.cs`
	- removed the moved client-dialog, PM4 utility, minimap/status, and sidebar-heavy UI method bodies from the main file.
	- kept the remaining world-objects implementation in-place as `DrawWorldObjectsContentCore()` so the split stayed low-risk and behavior-preserving.
- Added new partials:
	- `src/MdxViewer/ViewerApp_ClientDialogs.cs`
	- `src/MdxViewer/ViewerApp_Pm4Utilities.cs`
	- `src/MdxViewer/ViewerApp_MinimapAndStatus.cs`
	- `src/MdxViewer/ViewerApp_Sidebars.cs`
- Why this shape:
	- the repo already used partial-class decomposition for `ViewerApp`, so continuing that pattern was safer than a broad UI architecture rewrite.
	- this is a maintainability slice, not a user-facing viewer redesign.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings after fixing missing imports in the new partial files.
	- no automated tests added or run.

### Mar 21, 2026 - PM4 MPRL-Authoritative CK24 Frame Follow-Up

- This earlier viewer-side linked-`MPRL` translation experiment is no longer active.
- Runtime user validation reported that it made PM4 alignment worse, and the translation path was later removed in the `Dockspace UI Recovery + PM4 Translation Rollback` follow-up above.
- Keep this in mind when continuing PM4 work:
	- `MPRL` ownership as a semantic hypothesis is still open.
	- the specific implementation that translated CK24 groups into linked `MPRL` bounds centers should be treated as a regressed experiment, not current behavior.

### Mar 21, 2026 - PM4 Per-Object Bounds Overlay

### Mar 21, 2026 - PM4 Small-Yaw Correction Clamp

### Mar 21, 2026 - Viewer UI / Perf Slice: Hideable Chrome + Clipped Lists

- User priority shifted to viewer-shell usability and UI render cost because the fixed sidebar layout was getting in the way of PM4 debugging itself.
- `src/MdxViewer/ViewerApp.cs`
	- added `Tab`-driven hide-chrome mode so the menu bar, toolbar, sidebars, status bar, and floating utility windows can be suppressed quickly during scene inspection.
	- reduced default shell noise by no longer forcing every major sidebar section open on first draw.
	- clipped the obvious large per-frame UI lists instead of rendering every row:
		- file browser
		- discovered world maps
		- renderer subobject visibility toggles
		- WMO / MDX placement lists
		- POI / taxi node / taxi route lists
- Why this is scoped this way:
	- it attacks the two immediately visible pain points without doing a high-risk full UI rewrite: constant shell clutter and unbounded list drawing.
	- it does not attempt to restore the old dockspace/panel architecture yet.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing solution warnings
	- no automated tests added or run
	- no runtime signoff yet on actual frame-time improvement or on whether the new defaults are the right interaction model for PM4-heavy sessions

- Follow-up after user runtime report that PM4 objects were now almost correct but still carried a coherent `5..10` degree vertical-axis offset.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- `TryComputeWorldYawCorrectionRadians(...)` no longer applies tiny geometry-derived residual yaw corrections.
	- the CK24 world-yaw correction threshold moved from `2°` to `12°` so principal-axis noise does not override near-correct MPRL rotation.
- Why this is narrower than a raw-angle constant rewrite:
	- the PM4 repo tooling and format notes still treat MPRL `Unk04` / low-16 rotation as a standard `360 * value / 65536` angle.
	- the likely over-correction seam was the viewer-only continuous principal-axis fit, not the packed-angle scale itself.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
	- no automated tests added or run
	- no runtime signoff yet on the affected PM4 objects after this clamp

### Mar 21, 2026 - Archive I/O Performance Slice

- Reduced confirmed archive/path-resolution waste on the active MPQ-era viewer path without changing renderer or terrain behavior.
- `src/MdxViewer/DataSources/MpqDataSource.cs`
	- added `MpqDataSourceStats` instrumentation for `FileExists`, `ReadFile`, raw-byte cache behavior, and prefetch queue / worker timing
	- preserved the separate read-only prefetch MPQ workers, but now measures queue wait and worker read time explicitly
	- removed redundant normalized-vs-original duplicate MPQ existence probes in the MPQ-backed path
- `src/MdxViewer/Terrain/WorldAssetManager.cs`
	- added `WorldAssetReadStats` plus a resolved-read-path cache for world asset loads
	- `ReadFileData(...)` now caches the winning fallback path and no longer retries duplicate lowercase or `.mpq` forms that `MpqDataSource` already resolves internally
	- prefetch now warms the canonical model path and strongest `.skin` candidate first instead of broadly spraying alias permutations on every queued asset
- `src/MdxViewer/ViewerApp.cs`
	- world stats panel now shows asset-read probe counters plus MPQ read/prefetch counters so runtime profiling has exact signal instead of guesswork
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing solution warnings
	- no automated tests added or run
	- no runtime real-data validation yet on fixed MPQ-era datasets, so streaming-latency benefit is still instrumentation-backed but unproven in a live scene

- Added a PM4-specific bounds overlay in `src/MdxViewer/Terrain/WorldScene.cs` so PM4 object-part AABBs are visible in the main scene instead of only existing implicitly for picking/culling/debug text.
- Added a matching `PM4 Bounds` toggle in `src/MdxViewer/ViewerApp.cs` beside the existing PM4 MPRL and centroid toggles.
- Current behavior:
	- PM4 bounds draw through the existing `BoundingBoxRenderer` pass.
	- selected PM4 groups are highlighted, and the exact selected PM4 object is drawn white.
	- PM4 bounds rendering respects existing PM4 tile/object visibility checks and per-object transforms.
- Important limit:
	- the current PM4 object bounds still come from the rendered PM4 object geometry path, not from `MSCN` directly.
	- this is a debugging/visibility slice to validate extent mismatch hypotheses, not a solved MSCN container correction.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing solution warnings only
	- no automated tests added or run
	- no runtime signoff yet on the reported PM4 extent mismatch case

### Mar 21, 2026 - PM4 World-Space Orientation Solver Fix

### Mar 21, 2026 - PM4 Tile-Local Orientation Guardrail

- Follow-up after runtime report that PM4 tiles other than `0_0` and `0_1` were coherently rotated about `90°` counter-clockwise.
- `src/MdxViewer/Terrain/WorldScene.cs`
	- `EnumeratePlanarTransforms(...)` now keeps tile-local PM4 on the established non-swapped south-west tile basis and only tests non-swapped mirror variants there.
	- `ConvertPm4VertexToWorld(...)` now assembles tile-local viewer-world positions with the correct WoW tile convention: file `tileY` advances world `X`, and file `tileX` advances world `Y`.
	- quarter-turn `swap` candidates remain available only for world-space PM4, where the earlier handedness fix was actually needed.
- Why this is narrower than reverting the whole solver expansion:
	- the quarter-turn solve is still needed for world-space PM4 cases.
	- the regression came from applying that same basis search to tile-local PM4, which already has a stable tile-frame mapping.
	- origin tiles masked a second seam: unswapped file tile indices can still place tile-local PM4 onto the wrong non-origin grid cell even when the planar basis is otherwise correct.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing warnings
	- no automated tests added or run
	- no runtime signoff yet on the reported non-origin tile placement/orientation case

- Fixed one concrete PM4 handedness bug in `src/MdxViewer/Terrain/WorldScene.cs` after runtime evidence showed mirrored solutions like `swap=True` / `windingFlip=True` on structures that should only need a quarter-turn basis correction.
- Root cause:
	- `ResolvePlanarTransform(...)` only tested `identity` and `swap` for world-space PM4 data
	- this forced some world-space objects into mirrored fits because the rigid `+/-90` degree candidates were never evaluated
- Current behavior:
	- world-space PM4 now evaluates the rigid planar transforms first (`identity`, `180`, `+90`, `-90`)
	- mirrored candidates are now removed from the active PM4 planar solver so winding parity stays rigid-only instead of drifting into reversed/opposite-facing fits
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests added or run
	- no runtime signoff yet on the guardtower staircase case

### Mar 21, 2026 - PM4 Picking Arbitration Fix

- Fixed a viewer interaction bug where visible PM4 overlay objects could not be selected because `ViewerApp.PickObjectAtMouse(...)` returned on WMO/MDX selection before PM4 picking ran.
- Current behavior:
	- `WorldScene` now provides nearest-hit helpers for regular scene objects and PM4 overlay objects.
	- `ViewerApp` compares both hit distances from the same mouse ray and selects the closer target instead of hard-prioritizing WMO/MDX.
- Why this matters:
	- PM4 alignment tooling depends on left-click selection, and PM4 geometry commonly overlaps the same world objects whose WMO/MDX AABBs were previously swallowing the click.
- Validation limits:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests added or run
	- no runtime click-validation yet on a live PM4 overlay session

### Mar 21, 2026 - 4.0.0.11927 M2 Wrap + Blend Correction

- Follow-up after the first M2 parity slice focused on the remaining Cataclysm-era runtime symptoms the user reported: texture clamping/stretching and incorrect blend family selection on `4.0.0.11927` assets.
- Root gaps corrected in the active viewer path:
	- `src/MdxViewer/Rendering/ModelRenderer.cs` now treats `WrapWidth` / `WrapHeight` as repeat flags for all M2-adapted models, while classic MDX keeps the legacy clamp-flag interpretation.
	- `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs` no longer shifts M2 blend ids after mode `2`; ids `3`..`7` now map deliberately into the closest local renderer families.
- Current mapping details:
	- `0=Load`, `1=Transparent`, `2=Blend`, `3=Add` (`NoAlphaAdd`), `4=Add`, `5=Modulate`, `6=Modulate2X`, `7=AddAlpha` (`BlendAdd`)
	- `NoAlphaAdd` and `BlendAdd` are still approximations because the local MDX renderer has no separate states for them yet
- Validation limits for this checkpoint:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed with existing solution warnings only
	- no automated tests added or run
	- no runtime real-data validation yet on the affected Cataclysm-era M2 assets

### Mar 21, 2026 - M2 Material Parity Slice: Explicit Env-Map + UV Selector Recovery

- Landed the first non-heuristic M2 material-parity implementation slice in `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs` and `src/MdxViewer/Rendering/ModelRenderer.cs`.
- Confirmed root gap before editing:
	- `ModelRenderer` already had separate alpha-cutout / blended / additive / env-map shader-state paths
	- `WarcraftNetM2Adapter` was still flattening M2 batch intent by hardcoding `CoordId = 0`, dropping raw `.skin` texture-coordinate lookup metadata, and only preserving the first UV set from vertex data
- Current code change:
	- merges raw `.skin` `textureCoordComboIndex` metadata back into the Warcraft.NET skin path
	- preserves both M2 UV sets from raw `MD20` vertex data
	- reads raw `textureCoordCombos` so `-1` now drives reflective `SphereEnvMap` and `1` can route to UV1
	- adds focused renderer trace output showing pass + resolved material family for M2 batches under debug focus
- Current scope/limits:
	- improved: reflective / env-mapped family selection and UV-set fidelity where metadata exists
	- still open: texture transform animation, transparency/color track parity, broader shader-combo parity, and real-data visible validation on heavy reflection/transparency assets
- Validation limits at this checkpoint:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
	- no automated tests added or run
	- no runtime validation yet on real assets

### Mar 21, 2026 - PM4 Decode Triage Framed + Renderer Parity Planned

- PM4 overlay debugging is now in a more precise phase than the earlier loose-overlay indexing/precedence work:
	- user runtime symptom: `PM4: 2674 files found, none decoded into overlay data`
	- status interpretation: files are being discovered, but none survive into renderable overlay objects
	- `WorldScene` now has failure-bucket instrumentation for tile parse, tile range, read, decode, and parsed-but-zero-object cases
- Current root-cause direction for the `4.0` base-client failure versus `3.3.5`:
	- PM4 parser/object builder does not appear to key directly on `_dbcBuild`
	- map discovery / WDT resolution / active candidate set still does
	- the `2674` PM4 candidate count should be investigated against the fixed development dataset expectation of `616` PM4 files
- Formalized the rendering recovery program needed for PM4 object-variant matching:
	1. M2 material, transparency, and reflective parity
	2. lighting DBC expansion
	3. skybox / environment parity
- Added dedicated prompt plans for each implementation slice:
	- `.github/prompts/m2-material-parity-implementation-plan.prompt.md`
	- `.github/prompts/lighting-dbc-expansion-implementation-plan.prompt.md`
	- `.github/prompts/sky-environment-parity-implementation-plan.prompt.md`
- Validation limits for this update:
	- no new implementation code landed for the three rendering tracks yet
	- no automated tests were added or run in this planning pass
	- no runtime signoff yet for PM4 failure-bucket output or the rendering program

### Mar 21, 2026 - WMO Blend-Mode Correction + Loose PM4 Overlay Precedence

- Corrected one concrete WMO material/rendering mismatch in `src/MdxViewer/Rendering/WmoRenderer.cs`:
	- raw WMO material `BlendMode` is now mapped to `EGxBlend`
	- opaque pass handles `Opaque` and `AlphaKey`
	- transparent pass now handles only `Blend` and `Add` with matching blend funcs
- Fixed loose overlay precedence in `src/MdxViewer/DataSources/MpqDataSource.cs`:
	- loose-file resolution now searches `_looseRoots` newest-first so the most recently attached overlay overrides earlier roots
	- PM4 loose-path failures now emit the same trace help that previously existed only for WMO failures
- Build validation:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
- Validation limits:
	- no automated tests were added or run
	- no runtime real-data validation yet for the WMO sheen symptom
	- no runtime real-data validation yet for base+loose-overlay PM4 loading

### Mar 21, 2026 - Explicit Base-Build Selection Restored For Viewer MPQ Loads

- Restored explicit client build selection in `MdxViewer` instead of relying only on path-based build inference:
	- added `src/MdxViewer/Terrain/BuildVersionCatalog.cs`
	- `Open Game Folder (MPQ)...` now routes through a build-selection dialog before calling `LoadMpqDataSource(...)`
	- build options are loaded from `WoWDBDefs/definitions/Map.dbd` when available, with a fallback list that includes `4.0.0.11927` and `4.0.1.12304`
- Persisted base-build identity for saved clients:
	- `KnownGoodClientPath` now stores `BuildVersion`
	- viewer settings now also store `LastSelectedBuildVersion`
	- reopening a saved base or loading a loose map folder against a saved base now reuses the saved explicit build when present
- Added a runtime hint for PM4-era dataset mismatches:
	- loose overlay attach inspects the first PM4 version marker it finds
	- known markers currently map `11927 -> 4.0.0.11927` and `12304 -> 4.0.1.12304`
	- viewer logs a warning when PM4 overlay hint and active base-client build disagree
- Build validation:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
- Validation limits:
	- no automated tests were added or run
	- no runtime real-data validation yet with the development PM4 overlay against a verified `4.0.1.12304` base client

### Mar 21, 2026 - 4.0.0.11927 Terrain Blend Documentation + First Runtime Recovery Slice

- Closed the stale documentation gap around 4.0 terrain texturing by recording the wow.exe-backed runtime model instead of repeating the older "same as 3.3.5" shorthand.
- Reverse-engineered/runtime-documented behavior now preserved in repo docs and prompts:
	- chunk alpha assembly is neighbor-aware, not chunk-local only
	- neighbor layers are matched by texture id
	- 8-bit layers without direct alpha payload can be synthesized as residual coverage
	- runtime blend textures are created through the `TerrainBlend` path
- Documentation and prompt updates landed in:
	- `documentation/wow-400-terrain-blend-wow-exe-guide.md`
	- `docs/archive/WoW_400_ADT_Analysis.md`
	- `docs/archive/WoW_400_DeepDive_Analysis.md`
	- `docs/archive/WoW_301_DeepDive_Analysis.md`
	- `docs/ADT_WDT_Format_Specification.md`
	- `specifications/ghidra/prompt-400.md`
	- `.github/prompts/wow-400-terrain-blend-recovery.prompt.md`
- Active viewer implementation now includes the first 4.0 recovery slice in `StandardTerrainAdapter` / `TerrainChunkData`:
	- dedicated `Cataclysm400` alpha-decode mode stays separate from `LichKingStrict`
	- preserves per-layer `AlphaSourceFlags`
	- synthesizes missing residual 8-bit alpha when a layer lacks direct payload
	- stitches same-tile chunk-edge alpha texels by matching neighbor layer texture ids
- Build validation:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
- Validation limits:
	- no new automated tests were added or run for this slice
	- no real-data runtime verification yet on `test_data/development/World/Maps/development`
	- this is the first runtime-backed recovery slice, not full `TerrainBlend` parity closure

### Mar 20, 2026 - PM4 Tile Mapping Normalization + Reboot Handoff

- Applied PM4 viewer tile mapping guardrail in `WorldScene`:
	- map PM4 filename `x_y` into terrain tile keys as `(tileX=x, tileY=y)`
	- remove MPRL-centroid tile reassignment from PM4 overlay load path
	- merge duplicate PM4 tile payloads instead of overwriting (objects/stats/position refs)
- Build validation:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
- Runtime handoff:
	- user will restart machine before runtime checks
	- next required validation is the reported tile adjacency case (`00_00`, `01_00`, `01_01`) to confirm no PM4 tile drift/data loss

### Mar 20, 2026 - WDL Spawn Chooser Cross-Version Regression Handoff

- User runtime report: WDL heightmap spawn chooser currently does not function on tested versions.
- Status correction:
	- treat prior notes that implied chooser readiness/fallback behavior was sufficient as unverified for the active branch state
	- keep this issue open until runtime behavior is re-proven end-to-end
- Investigation target area for the next pass:
	- map-row spawn action gating (`WdlPreviewWarmState` and readiness transitions)
	- preview cache warmup readiness state propagation into chooser UI
	- chooser commit path versus fallback load path when preview prep fails
- Validation requirement for closure:
	- real-data runtime proof on at least one Alpha-era map and one 3.x map
	- explicit evidence that user can pick a spawn point and spawn is applied
- Validation limits in this handoff-only update:
	- no code changes made
	- no automated tests added or run

### Mar 19, 2026 - Terrain Texture Transfer Command (Backend Slice)

- Added first backend/library + CLI slice for mapped terrain texture transfer:
	- command: `terrain-texture-transfer`
	- payload scope: `MTEX`, `MCLY`, `MCAL`, `MCSH`, and MCNK holes
	- mapping modes: explicit `--pair` and auto `--global-delta`
	- supports `dry-run` manifests and `apply` output ADT writing
- Added split-ADT resilience for the active development dataset:
	- if `SplitAdtMerger` serialization fails, command now composes transferable texture payload from root + `_tex0.adt`
	- MCNK subchunk parsing now tolerates headerless tex0 MCNK payloads
	- top-level chunk walk/rebuild now handles odd-size boundary variance seen in split files
	- merge path now skips `obj0`-only sidecars (without `_tex0`) and uses root bytes directly for terrain-texture transfer
- Real-data validation performed (fixed path):
	- source/target: `test_data/development/World/Maps/development`
	- dry-run sample: `development_0_0 -> development_0_0` (chunk pairs=256, copied flags true for MTEX/MCLY/MCAL/MCSH/holes)
	- apply sample: same pair wrote output ADT + summary/tile manifests
	- non-identity sample: `development_0_0 -> development_1_0` succeeded in both dry-run and apply with full payload transfer and no manual-review flags
	- small global-delta batch (`--global-delta 1,0 --tile-limit 3`) completed; 2 tiles clean, 1 tile (`development_0_1 -> development_1_1`) still flagged manual-review due one target MCNK with no parseable subchunks
- Validation limits:
	- no viewer runtime visual signoff yet for transferred outputs in this pass
	- no new automated tests added in this pass

### Mar 19, 2026 - MdxViewer Thin UI Hook For Terrain Texture Transfer

- Added a thin UI entry in `MdxViewer` (`ViewerApp`) for the backend terrain texture transfer flow:
	- File menu item: `Terrain Texture Transfer...`
	- dialog supports source/target/output folders, dry-run/apply toggle, explicit-pair or global-delta mapping, chunk offsets, payload toggles, and optional manifest path
	- execution runs asynchronously via the existing app-thread pattern and surfaces summary + warnings in an in-dialog log panel
- Build validation passed for the viewer after wiring:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Validation limits:
	- no new runtime visual validation in the viewer yet for this dialog path
	- this UI slice does not resolve the known `development_0_1 -> development_1_1` target MCNK parse edge case from backend validation

### Mar 19, 2026 - Canonical Fresh-Output Pass For 3.3.5 Development Map

- Executed a full-map identity transfer pass to materialize a fresh canonical output folder for viewer use:
	- command: `terrain-texture-transfer --source-dir ...335-dev... --target-dir ...335-dev... --global-delta 0,0 --mode apply`
	- output root: `output/development-335-canonical-texture-transfer`
- Real-data result summary:
	- tiles planned/processed/written: 2303 / 2303 / 2303
	- manual review: 0
	- chunk pairs applied: 589,568
	- missing source/out-of-range chunk remaps: 0 / 0
	- summary manifest: `output/development-335-canonical-texture-transfer/manifests/summary.json`
	- companion `development.wdt` and `development.wdl` copied into output root
- Operational guidance:
	- this is now a viable "open the generated folder in MdxViewer" workflow for the tested 3.3.5 development dataset
	- this does not replace targeted non-identity remap validation when using non-zero global deltas or explicit cross-tile mappings

### Mar 19, 2026 - Development Repair WL Attribution + Texture Payload Manifests

- Reworked `DevelopmentRepairService` WL ingestion so repair no longer assumes tile-named `*.wl*` files.
	- new behavior pre-indexes all map-level WL files (`.wlw/.wlm/.wlq/.wll`) once, converts to MH2O by world position, and applies per-tile liquids from that coordinate-attributed index
	- tile manifests now record the actual WL source file paths used (for example `Clayton Test.wlw`) instead of synthetic `tileName.wlw` expectations
- Expanded per-tile JSON payload (`TextureData`) with terrain texturing data modeled after the VLM chunk-layer shape:
	- includes MTEX texture list
	- includes per-chunk layers with texture id/path, flags, alpha offset, effect id, plus optional base64 alpha bytes and byte count
	- extractor now chooses the richest source among output ADT, `_tex0.adt`, and root ADT so split-source tiles can still emit texture payload data
- Real-data validation performed on fixed paths:
	- command: `development-repair --mode repair --input-dir test_data/development/World/Maps/development --tile-limit 50`
	- observed manifests with `WlLiquidsConverted=true` and map-level WL source filenames attached to those tiles
	- reference check only: `development-repair --mode repair --input-dir test_data/WoWMuseum/335-dev/World/Maps/development --tile-limit 1` (used only to inspect payload shape, not as canonical pipeline input)
	- policy now enforced in code: `development-repair` rejects WoWMuseum `335-dev` input and requires building clean outputs from `test_data/development/World/Maps/development` constituent parts
- Validation limits:
	- this pass did not include viewer-side visual validation of generated MH2O/texturing results
	- no new automated regression tests were added in this pass

## Mar 17, 2026 - Recovery Branch Checkpoint (v0.4.0 base)

- Active branch reset in main tree: recovery/v0.4.0-surgical-main-tree (base 343dadf).
- Restored .github customization stack from main and committed as 845748b.
- Build from this branch passes in primary tree environment.
- Terrain alpha decode profile routing is now staged in code:
	- TerrainAlphaDecodeMode in AdtProfile
	- LichKingStrict for 3.x profiles
	- LegacySequential for 0.x profiles
	- StandardTerrainAdapter alpha extraction routes by profile mode

### Critical Pending Validation

- Runtime terrain checks still required on both families:
	- Alpha-era terrain
	- LK 3.3.5 terrain
- Do not mark terrain safety complete until these real-data checks are done.

### Immediate Next Work

1. Finalize commit state for the profile/decode changes (if still local).
2. Run manual runtime spot-checks for alpha decode output.
3. Resume surgical commit intake from v0.4.0..main in SAFE-first order.

### Mar 17, 2026 - Intake Triage Update

- Reviewed queued commits `177f961`, `d50cfe7`, `326e6f8`, `4e2f681`, `37f669c`, `39799bf`, and `62ecf64` against the recovery branch and terrain-alpha guardrails.
- Marked `177f961` and `37f669c` as RISKY and out of scope for safe-first intake.
- Marked `d50cfe7`, `326e6f8`, `4e2f681`, `39799bf`, and `62ecf64` as MIXED; only isolated helper/tooling slices are candidates.
- Selected first SAFE extraction: corrected `TerrainImageIo` alpha-atlas helper from `62ecf64` only.
- Explicitly rejected the earlier `d50cfe7` `TerrainImageIo` version because it hardcoded atlas edge remapping that the recovery notes already identified as changing shipped data.
- No claim of terrain safety from this triage alone; runtime real-data validation is still required.

### Mar 17, 2026 - First SAFE Batch Applied

- Added `src/MdxViewer/Export/TerrainImageIo.cs` from the corrected `62ecf64` implementation only.
- Kept ViewerApp, TerrainRenderer, WorldScene, test-project, and terrain decode heuristic changes out of this batch.
- Build gate passed: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- Runtime terrain validation remains pending; build-only status is not sufficient for terrain signoff.

### Mar 18, 2026 - Rendering Recovery Batch

- Applied the `WorldAssetManager` renderer-residency fix from main so placed MDX/WMO renderers are no longer evicted out from under live world instances.
- `GetMdx` / `GetWmo` now lazy-load missing models and cached failed loads can be retried.
- Added the minimal skybox backdrop path from main:
	- route skybox-like MDX/M2 placements into a dedicated list
	- render the nearest skybox as a camera-anchored backdrop before terrain
	- added `ModelRenderer.RenderBackdrop(...)` with forced no-depth state for all layers
- Verified that the recovery branch already contained the reflective M2 depth-flag fix and env-map backface guard, so those regressions were not reintroduced here.
- Build gate passed again: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- Runtime validation still required; build success does not prove:
	- doodad/WMO reload correctness after moving away and back
	- correct skybox classification on real map data
	- MH2O liquid correctness on LK 3.3.5 tiles

### Mar 18, 2026 - MCCV + MPQ Recovery Batch

- Restored MCCV terrain color support on the active chunk-based terrain path.
- `TerrainChunkData` now carries MCCV bytes, `StandardTerrainAdapter` populates them, `TerrainMeshBuilder` uploads them, and `TerrainRenderer` applies them in shader.
- Initial MCCV fix improved output but did not fully match runtime behavior.
- Applied the isolated `NativeMpqService` recovery slice from the mixed MPQ commits:
	- expanded patch archive ordering for locale/custom patch names
	- full normalized path encrypted-key derivation with basename fallback
	- compression bitmask handling for MPQ sectors
	- BZip2 support via SharpZipLib
- Build gates passed:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime validation still required; build success does not prove:
	- patched 1.x+ MPQ read correctness on real patch chains
	- encrypted later-version MPQ entry reads on real data
	- MCCV highlight/tint correctness on real 3.x terrain

### Mar 18, 2026 - MCCV + Patch-Letter Follow-up

- Reworked MCCV semantics after user runtime feedback showed the first shader heuristic was still wrong.
- Current interpretation now matches the repo's own MCCV writer comments:
	- bytes are treated as BGRA, not RGBA
	- neutral/no-tint values are mid-gray (`127`) rather than white
	- terrain tint uses RGB remapped around mid-gray, not MCCV alpha strength
- Extended `NativeMpqService.LoadArchives(...)` to discover MPQs recursively so nested/custom `patch-[A-Z].mpq` archives are included in the patch chain.
- Kept Alpha single-asset wrapper archives (`.wmo.mpq`, `.wdt.mpq`, `.wdl.mpq`) out of the generic recursive scan because they are handled separately by the viewer data source.
- Build gates passed again after this follow-up:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- Runtime validation is still the blocker:
	- confirm 3.x MCCV transparent/neutral regions no longer darken to black
	- confirm maps stored inside `patch-[A-Z].mpq` are now discovered and load through normal WDT/ADT lookup paths

### Mar 18, 2026 - 3.x Alpha Offset-0 Experiment Reverted

- The recent LK offset-0 fallback change in `StandardTerrainAdapter.ExtractAlphaMaps(...)` was reverted after runtime validation showed it was wrong.
- Updated conclusion:
	- treating `AlphaMapOffset == 0` as a valid relaxed fallback for the active 3.x terrain path is not the correct fix
	- keep the revert and continue investigating the real 3.x alpha decode/sourcing failure separately
- Validation status:
	- normal `dotnet build .../MdxViewer.sln -c Debug` still conflicts with the running viewer process locking `bin/Debug`
	- use the alternate-output build for compile validation while the viewer stays open

### Mar 18, 2026 - 3.x Profile-Driven Alpha Recovery

- Investigated the remaining 3.x terrain failure after the offset-0 revert.
- Confirmed the active recovery branch was still missing rollback-era handling for:
	- MPHD/WDT big-alpha mask `0x4 | 0x80`
	- split `*_tex0.adt` sourcing for textures/layers/alpha/shadow data
	- stronger MCAL decode semantics for compressed alpha, big alpha, and do-not-fix chunks
- Applied the recovery batch:
	- `FormatProfileRegistry`: added `BigAlphaFlagsMask` and `PreferTex0ForTextureData`; 3.0.1 and 3.3.5 profiles now use `0x4 | 0x80` and prefer `*_tex0.adt`
	- `StandardTerrainAdapter`: can read MTEX + MCNK data from `*_tex0.adt`, route layer/alpha/shadow sourcing through that file, pass the MCNK `0x8000` do-not-fix flag into alpha decode, and infer big-alpha per chunk
	- `WoWMapConverter.Core/Formats/LichKing/Mcal.cs`: replaced the broken/simple decoder with the stronger compressed / big-alpha / 4-bit implementation
- Build validation passed:
	- `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/WoWMapConverter.Core.csproj -c Debug`
	- `dotnet build "I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="I:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`
- Runtime validation is still the blocker:
	- no claim yet that 3.x alpha blending is correct on real user data
	- next check is whether the failing 3.x sample now uses more than one alpha layer and stops looking like 4-bit Alpha-era decode

### Mar 18, 2026 - Terrain Runtime Validation Update

- User runtime validation now confirms the current terrain alpha recovery on two real data families:
	- Alpha 0.5.3 terrain renders correctly again after restoring the alpha-era edge fix in `AlphaTerrainAdapter`
	- 3.0.1 alpha-build terrain renders correctly on the profile-driven strict 3.x path
- Earlier runtime feedback also reported the 3.3.5 sample looked correct before the 0.5.3 regression was fixed.
- Status change:
	- terrain validation is no longer build-only for the tested 0.5.3 and 3.0.1 samples
	- broader signoff across more 3.x maps is still pending, so do not generalize this to all LK-era terrain yet

### Mar 18, 2026 - Remaining ModelRenderer Slice From 39799bf

- Applied the last model-side hunk from `39799bf` after the MPQ reader work was already in place.
- `ModelRenderer` now skips particle rendering on the world-scene batched render path only.
- Standalone model viewing still renders particles as before.
- Reason: per-instance transforms are not yet propagated into particle simulation for placed models, and leaving them enabled there can produce visibly wrong camera-locked effects.
- Build gate passed: `dotnet build I:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.

### Mar 18, 2026 - WDL Preview Warmup + Texture Reuse Batch

- Ported the missing `main` WDL preview cache support into the recovery branch:
	- added `WdlPreviewCacheService`
	- `ViewerApp` now warms discovered WDL previews in the background and opens the preview dialog through the cache-aware path
	- `ViewerApp_WdlPreview` now shows warmup/error state instead of only a synchronous failure dialog
- Added a targeted model-load performance slice in `ModelRenderer`:
	- per-model texture diagnostic logs are now opt-in via `PARP_MDX_TEXTURE_DIAG`
	- BLP/PNG textures now use a shared refcounted GL texture cache so repeated world doodads do not decode/upload the same texture once per instance
- Build validation passed: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- Runtime validation is still required before claiming:
	- WDL preview warmup/cache behavior is correct on the user's real map set
	- M2 load time is materially improved in the real world scene

### Mar 18, 2026 - WDL Parser Recovery + Transparency Heuristic Follow-up

- Addressed the newly reported WDL read failure after the preview-cache port:
	- `WoWMapConverter.Core/VLM/WdlParser.cs` no longer rejects all non-`0x12` WDL versions up front
	- parser now scans the WDL chunk stream for `MAOF` and accepts MAOF offsets that reference either `MARE` headers or direct height payloads
- Unified active viewer WDL reads through `src/MdxViewer/Terrain/WdlDataSourceResolver.cs` so both preview warmup and `WdlTerrainRenderer` use the same `.wdl` / `.wdl.mpq` + file-set lookup path.
- Closed a remaining 3.x model-path gap in `WmoRenderer` by extending doodad extension fallback from only `.mdx`/`.mdl` to also include `.m2`.
- Adjusted `ModelRenderer` transparency routing:
	- shared texture cache entries now retain simple alpha-shape metadata
	- classic non-M2 `Transparent` layer-0 materials only use hard cutout when the texture alpha is binary
	- textures with intermediate alpha now stay on the blended path
- Build validation passed: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- No automated tests were added or run.
- No runtime real-data validation has been performed yet for this batch.

### Mar 18, 2026 - Standalone 3.x Model Load Freeze Follow-up

- Addressed the reported freeze / non-load behavior when opening individual 3.x `.mdx` files in the viewer.
- Root cause on the active standalone path was different from the world/WMO loaders:
	- standalone container probing only recognized `MD20`, not `MD21`
	- standalone M2 adaptation eagerly scanned the full `.skin` file list on the UI thread before trying the obvious same-basename candidates
	- standalone file loads also lacked the world path's canonical model-path recovery and MD20 converter fallback
- Current fix in `src/MdxViewer/ViewerApp.cs`:
	- standalone probe now routes both `MD20` and `MD21` through the M2-family path even when the file extension is `.mdx`
	- standalone M2 loads now resolve a canonical model path through MPQ file-set indexes before skin lookup
	- predictable `.skin` candidates are tried first, and the broader `.skin` file-list search is only used as a fallback with a per-session cache
	- standalone MD20 loads now also have the same M2->MDX converter fallback used elsewhere when direct adaptation cannot complete
	- standalone skin-path cache is cleared when a new MPQ data source is loaded
- Build validation passed: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`.
- No automated tests were added or run.
- No runtime real-data validation has been performed yet for this batch.

### Mar 18, 2026 - M2 Empty-Fallback Guardrail

- Follow-up after runtime feedback that some M2-family models still "load" into an empty viewport with `0` geosets / vertices.
- Current conclusion:
	- at least some failures are not clean adapter failures; the raw `MD20` converter fallback can produce an `MDX` shell that parses but has no renderable geometry
	- that state is misleading in the UI because it looks like a loaded model rather than an unsupported / failed conversion
- Current fix:
	- `WarcraftNetM2Adapter` now exposes shared renderable-geometry checks
	- standalone `ViewerApp`, world `WorldAssetManager`, and WMO doodad `WmoRenderer` now reject converted fallback models unless they contain at least one renderable geoset
	- rejected fallback loads now preserve/log the underlying failure instead of silently treating an empty converted model as success
- Build validation passed: `dotnet build "i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln" -c Debug -p:OutDir="i:/parp/parp-tools/gillijimproject_refactor/output/build-validation/mdxviewer/"`.
- No automated tests were added or run.
- No runtime real-data validation has been performed yet for this batch.
- This is a diagnostics/correctness guardrail, not proof that pre-release `3.0.1` M2 layouts are fully supported.

### Mar 18, 2026 - Pre-release 3.0.1 M2 Scope Clarified

- User runtime verification after the guardrail patch indicates most remaining M2 problems are specific to the pre-release `3.0.1` model family rather than the later `3.3.5` family.
- Current working assumption:
	- pre-release `3.0.1` model files may be a transitional or hybrid `MDX` + `M2` variant
	- later-WotLK assumptions should not be silently reused for that path
- Separate runtime issue remains open across both model families:
	- neon-pink transparent surfaces still appear on both `MDX` and M2-family assets
	- treat that as a shared renderer/material/shader problem, not proof of a model-parser-only defect
- Resulting investigation split for the next pass:
	1. add true version/profile-aware handling for pre-release `3.0.1` model structure
	2. audit shared transparent-surface handling, texture resolution, and blend/shader parity independently of format parsing
- No new code changes were made in this note-only follow-up.
- Runtime evidence came from the user's real data, not fixtures.

### Mar 19, 2026 - Pre-release 3.0.1 Model Profile Guardrail

- Live `wow.exe` decompilation for build `3.0.1.8303` confirmed the client-side model gate is stricter than the active generic adapter path:
	- required root magic is `MD20`
	- accepted version range is `0x104..0x108`
	- parser behavior splits structurally at `0x108`
- Active viewer code now routes that profile knowledge into all three shared M2-family entry points:
	- standalone `ViewerApp.LoadM2FromBytes(...)`
	- world `WorldAssetManager.LoadMdxModel(...)`
	- WMO doodad `WmoRenderer.LoadM2DoodadRenderer(...)`
- `WorldScene` / `WorldAssetManager` now receive the build string at construction time so constructor-time manifest loads use the same profile guard instead of waiting for later `SetDbcCredentials(...)`.
- `WarcraftNetM2Adapter` now fails fast on build/profile mismatches before `.skin` search or fallback conversion:
	- `3.0.1.8303` and unknown `3.0.x` profiles reject `MD21` roots and out-of-range MD20 versions
	- `3.3.5.12340` currently keeps `MD21` container allowance to avoid broad later-branch regression while the parser path remains shared
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no new runtime real-data validation yet for the guarded `3.0.1` model path
	- do not claim this as a full pre-release `3.0.1` render fix; it is a profile-routing/compatibility guardrail
- Separate shared renderer issue is still open:
	- neon-pink transparent surfaces remain a Track B problem across classic `MDX` and M2-family assets

### Mar 19, 2026 - Standalone Data-Source M2 Read-Path Fix

- The new user-visible `Failed to read: ...` symptom on standalone/browser-loaded M2-family assets was not a parser error.
- Root cause:
	- `ViewerApp.LoadFileFromDataSource(...)` still did an exact `_dataSource.ReadFile(virtualPath)` and returned early
	- M2-family assets in the file browser can appear under alias paths that need the same canonical resolution logic already used later in the standalone M2 path
- Current fix:
	- data-source loads for `.mdx` / `.mdl` / `.m2` now resolve through `ResolveStandaloneCanonicalModelPath(...)`
	- browser-side model reads now use `ReadStandaloneFileData(...)` before giving up
	- successful reads now carry the resolved virtual path into the later container-probe path
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- runtime retry on the user's actual 3.0.1 data is still required to confirm the failure moved from read-time to the next real blocker

### Mar 19, 2026 - Pre-release 3.0.1 wow.exe Documentation Pass

- Shifted from speculative code changes to binary-backed documentation after the user reported that models still do not load.
- New documented `wow.exe` facts for build `3.0.1.8303`:
	- common loader chain is `FUN_0077e2c0` -> `FUN_0077d3c0` -> `FUN_0079bc70` -> `FUN_0079bc50` -> `FUN_0079bb30` -> `FUN_0079a8c0`
	- accepted model-family extensions are normalized to `.m2` before parse/bootstrap continues
	- high-level failure falls back to `Spells\\ErrorCube.mdx`
	- root parser is `MD20`-only with version range `0x104..0x108`
	- parser layout splits at `0x108`
	- confirmed validator families now include shared span strides `1`, `2`, `4`, `8`, `0x0C`, `0x30`, `0x44` and nested record families `0x70`, `0x2C`, `0x38`, `0xD4`, `0x7C`
	- version split families are legacy `0xDC` + `0x1F8` versus later `0xE0` + `0x234`
- New artifacts created for fresh chats:
	- `documentation/pre-release-3.0.1-m2-wow-exe-guide.md`
	- `.github/prompts/pre-release-3-0-1-m2-implementation-plan.prompt.md`
	- `.github/prompts/pre-release-3-0-1-m2-ghidra-followup.prompt.md`
	- `.github/prompts/pre-release-3-0-1-m2-runtime-triage.prompt.md`
- Validation status for this pass:
	- no automated tests were added or run
	- no new build was needed because this pass only added documentation and prompts
	- no runtime real-data validation was performed

### Mar 19, 2026 - 3.0.1 Pre-release Profile Routing Broadening

- Follow-up after the wow.exe-backed profile guardrail: the active registry no longer binds the pre-release `3.0.1` profile only to exact build `3.0.1.8303`.
- Current behavior:
	- any parsed `3.0.1.x` build now resolves to the same pre-release `3.0.1` ADT, WMO, and M2 profiles
	- other `3.0.x` builds still fall back to the generic unknown `3.0.x` profile until there is binary evidence for a narrower mapping
- Why this matters:
	- standalone model loads, world doodads, WMO doodads, and terrain/WMO profile routing now stay on the pre-release path for the whole `3.0.1` family instead of silently downgrading non-`8303` builds to the generic `3.0.x` profile
- Validation status:
	- build validation pending for this specific routing change
	- no automated tests were added or run
	- no runtime real-data validation was performed

### Mar 19, 2026 - 3.0.1 Pre-release M2 Parser + Fallback Alignment

- Follow-up after the routing-only fix was not enough: active model loading now includes a dedicated pre-release `MD20` parse path in `WarcraftNetM2Adapter` instead of sending raw `3.0.1` files through Warcraft.NET's later-layout `MD21` assumptions.
- Current viewer-side behavior:
	- standalone, world, and WMO doodad adapter loads normalize pre-release `MD20` data through a local parsed-model abstraction
	- the old forced profile-specific `.skin` parser path was disabled because the wow.exe-derived `0x70` / `0x2C` family sizes were not proven `.skin` submesh / batch strides
	- converter fallback now receives the active build version and avoids hard-parsing later-layout animation / bone tables for pre-release `3.0.1`
	- converter skin fallback keeps only the index / triangle tables required for geometry conversion instead of forcing nonessential fixed-stride submesh / texture-unit tables
- Why this matters:
	- the primary runtime path and the fallback conversion path no longer disagree about pre-release `3.0.1` model-family assumptions
	- non-`8303` `3.0.1.x` builds now reach both the right profile and a compatible loader path
- Validation status for this pass:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests were added or run
	- no runtime real-data validation was performed

### Mar 19, 2026 - Standalone Alias Recovery + Unsuffixed Skin Candidates

- Follow-up after fresh runtime errors still showed two model-load gaps:
	- standalone/browser `DataSourceRead` failures could still stop at the unresolved `.mdx` alias path even when the world-model path already had broader file-set heuristics
	- companion skin discovery only tried `00`-`03` suffixed names, not the unsuffixed `.skin` form some transitional assets may use
- Current fix:
	- `ViewerApp` standalone canonical resolution and data-source reads now reuse the broader candidate set already proven useful on the world path: exact path, extension aliases, bare filename aliases, and `Creature\Name\Name.{mdx|m2|mdl}` guesses
	- standalone resolution now also probes guessed candidates through `FileExists` / `ReadFile` instead of depending only on the prebuilt file index
	- shared `WarcraftNetM2Adapter.BuildSkinCandidates(...)` now includes unsuffixed `.skin` candidates before the numbered `00`-`03` forms
- Why this matters:
	- user-visible `Failed to read requested='...mdx'` errors can now recover through the same alias breadth the world loader already had
	- `Missing companion .skin for M2` can now recover when the sidecar is present under the base `.skin` name instead of only numbered variants
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests were added or run
	- runtime validation on the specific failing assets is still pending

### Mar 19, 2026 - Cocoon Optional-Span Parser Follow-up

- Fresh runtime log from `Creature\Cocoon\Cocoon.mdx` showed the profiled pre-release parser was now reached, but it still failed before geometry extraction because an unresolved optional table span (`colors`, stride `0x2C`) was treated as fatal.
- Current fix:
	- `WarcraftNetM2Adapter.ParseProfiledMd20Model(...)` now hard-validates only the spans the runtime model builder actually dereferences for viewer geometry

### Mar 19, 2026 - MCNK Index Repair Hook For Development ADT Export

- Added a rollback-CLI `repair-mcnk-indices` command that audits or rewrites root ADT `MCNK` header `IndexX` / `IndexY` values.
- `development-repair` now runs the same fixup in-memory on exported root ADTs by default; disable with `--repair-mcnk-indices false` if raw output is needed.
- Repair logic prefers `MCIN` order when present and otherwise falls back to top-level `MCNK` scan order.
- Real-data audit on the loose source folder `test_data/development/World/Maps/development` found:
	- 466 root ADT filenames
	- 114 zero-byte placeholders
	- 352 non-empty roots with chunk data
	- 0 detected `MCNK` index mismatches under scan-order validation on those raw loose roots
- Validation limits:
	- this does not prove generated WDL-derived / repaired export sets are clean because the referenced `PM4ADTs/*` outputs are not present in this workspace
	- `dotnet run/build` for `WoWRollback.Cli` is still blocked here by pre-existing missing `WoWFormatLib` / `CascLib` references under `WoWRollback.AnalysisModule`, so end-to-end CLI execution was not revalidated in this environment
	- optional / unresolved table families now use a nonfatal validator that logs and skips invalid spans instead of rejecting the entire model
	- per-texture filename spans are also treated as optional so a bad embedded name table does not abort the whole model
- Why this matters:
	- `Cocoon.mdx` was failing in the parser before any real geometry read was attempted
	- this keeps the wow.exe-backed strictness for required geometry tables while avoiding false rejects from still-unmapped optional families on `0x104..0x107` models
- Validation status:
	- `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed
	- no automated tests were added or run
	- no runtime real-data validation was performed after the fix

### Mar 19, 2026 - Classic 0.5.3 MDX Regression Closed; 3.0.1 Still Open

- User runtime validation now confirms the classic Alpha `0.5.3` MDX rendering regression is fixed.
- Confirmed repair stack in `src/MdxViewer/Rendering/ModelRenderer.cs`:
	- direct-path replaceable fallback is restricted to `_isM2AdapterModel`
	- wrap/clamp interpretation is split between classic MDX and M2-adapted models
	- classic `Layer 0 + Transparent` once again always uses alpha-cutout
- A new direct-asset diagnostic path was added in `src/MdxViewer/AssetProbe.cs` and wired through `src/MdxViewer/Program.cs`:
	- `--probe-mdx` loads an asset from a real client path, prints parsed materials, and reports decoded BLP alpha statistics
	- this was used on `DuskwoodTree07.mdx` to prove the remaining canopy failure was in renderer behavior after decode, not in TEXS parsing or BLP decode
- Current status change:
	- classic `0.5.3` MDX should be treated as restored for the tested runtime sample
	- pre-release `3.0.1` rendering is still buggy and remains the active unresolved model-family track

### Mar 19, 2026 - PM4 Coordinate Validation Command

- Added `WoWMapConverter.Core/Formats/PM4/Pm4CoordinateService.cs` as the first authoritative PM4 placement helper set in active core code.
- Added `WoWMapConverter.Core/Formats/PM4/Pm4CoordinateValidator.cs` to validate transformed `MPRL` refs against real `_obj0.adt` placements from the fixed development dataset.
- Added CLI command: `wowmapconverter pm4-validate-coords [--input-dir <dir>] [--tile-limit <n>] [--threshold <units>] [--json <path>]`.
- Important scope limit:
	- this is a real-data validation path for `MPRL` only
	- it does not yet validate MSCN semantics
	- it does not yet build the cross-tile CK24 registry
- Validation status at this note:
	- initial real-data slice showed `MPRL` is already in ADT placement order, not tile-local
	- broadened sample run on 100 validated tiles reported 38,133 refs in expected tile bounds (100.0%) and 36,070 refs within 32 units of a nearest `_obj0.adt` placement (94.6%)
	- average nearest-placement distance on that sample was 10.86 units
	- broader work is still pending for CK24 aggregation and MSCN semantics

### Mar 20, 2026 - PM4 Viewer Overlay Diagnostics/Grouping/Winding Pass

- Added active PM4 overlay rendering + diagnostics in `src/MdxViewer/Terrain/WorldScene.cs` and `src/MdxViewer/ViewerApp.cs`.
- Added PM4 color modes for structural inspection (`CK24` type/object/key, tile, dominant group/attribute, height).
- Added optional PM4 3D markers (`MPRL` refs and object centroids).
- Added CK24 decomposition controls for disjoint geometry:
	- split by shared vertex connectivity
	- optional split by dominant `MSUR.MdosIndex` before connectivity
- Added per-object planar transform solve and winding parity correction:
	- candidate swap/invert U/V planar transforms scored against nearest `MPRL` anchors
	- mirrored parity now flips triangle winding order to avoid backward-wound faces
- Validation status:
	- repeated `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` passed (warnings only)
	- no automated tests were added or run
	- runtime real-data signoff still pending for merged/disjoint PM4 object cases
- Scope boundary:
	- this does not replace the pending map-level CK24 registry or finalize MSCN semantics
	- current PM4 reconstruction should be treated as viewer debug instrumentation + heuristics, not final export-grade identity mapping

## ✅ Working

### Mar 19, 2026 - 4.x Split ADT No-MCIN Fallback

- Real-data audit of the fixed `test_data/development/World/Maps/development` loose roots confirmed the current 4.x load failure is primarily a no-`MCIN` issue, not an `MCNK.IndexX/IndexY` issue:
	- 466 root ADT filenames
	- 114 zero-byte placeholders
	- 352 non-empty roots
	- 0 non-empty roots with `MCIN`
- `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` now falls back to top-level `MCNK` scan order when a root ADT omits `MCIN`.
- `src/WoWMapConverter/WoWMapConverter.Core/Converters/LkToAlphaConverter.cs` now uses the same root fallback so later split roots can flow into the existing Alpha conversion path instead of throwing immediately on missing `MCIN`.
- Scope limit:
	- this is a geometry/chunk-order recovery step first
	- full 4.x `_tex0.adt` texture-layer parity is still not claimed
	- the converter only consumes split texture companions when they expose LK-style `MCNK` payloads large enough for the current Alpha builder
- Validation status at this note:
	- code edits landed
	- build/runtime validation still pending after this patch

### MdxViewer (3D World Viewer) — Primary Project
- **Alpha 0.5.3 WDT terrain**: ✅ Monolithic format, 256 MCNK per tile, async streaming
- **0.6.0 split ADT terrain**: ✅ StandardTerrainAdapter, MCNK with header offsets (Feb 11)
- **0.6.0 WMO-only maps**: ✅ MWMO+MODF parsed from WDT (Feb 11)
- **Terrain liquid (MCLQ)**: ✅ Per-vertex sloped heights, absolute world Z, waterfall support (Feb 11)
- **WMO v14 rendering**: ✅ 4-pass: opaque → doodads → liquids → transparent
- **WMO liquid (MLIQ)**: ✅ matId-based type detection, correct positioning (Feb 11)
- **WMO doodad culling**: ✅ Distance (500u) + cap (64) + nearest-first sort + fog passthrough
- **WMO doodad loading**: ✅ FindInFileSet case-insensitive + mdx/mdl swap → 100% load rate
- **MDX rendering**: ✅ Two-pass opaque/transparent, alpha cutout, specular highlights, sphere env map
- **MDX GEOS version compatibility**: ✅ Ported version-routed GEOS parser behavior from `wow-mdx-viewer` (v1300/v1400 strict path + v1500 strict path + guarded fallback)
- **MDX SEQS name compatibility**: ✅ Counted 0x8C named-record detection broadened to reduce fallback `Seq_{animId}` names on playable models
- **MDX PRE2/RIBB parsing parity**: ✅ Expanded parser coverage for PRE2 and RIBB payload/tail animation chunks (runtime visual verification pending)
- **MDX animation engine**: ✅ BONE/PIVT/HELP parsing, keyframe interpolation, bone hierarchy (Feb 12)
- **Full-load mode**: ✅ `--full-load` (default) loads all tiles at startup with progress (Feb 11)
- **MCSH shadow maps**: ✅ 64×64 bitmask applied to all terrain layers
- **AOI streaming**: ✅ 9×9 tiles, directional lookahead, persistent tile cache, MPQ throttling (Feb 11)
- **Frustum culling**: ✅ View-frustum + distance + fade
- **AreaID lookup**: ✅ Low 16-bit extraction + low byte fallback for MapID mismatch
- **DBC Lighting**: ✅ LightService loads Light.dbc + LightData.dbc, zone-based ambient/fog/sky colors
- **Replaceable Textures**: ✅ DBC CDI variant validation against MPQ + model dir scan fallback
- **Minimap overlay**: ✅ From minimap tile images
- **PM4 debug overlay (viewer-side)**: 🔧 In progress — color modes, 3D markers, CK24 split modes, and parity-aware winding fixes landed; runtime signoff still pending

### Model Parsers & Tools
- **MDX-L_Tool**: ✅ Core parsing and Archaeology logic complete.
- **GEOS Chunk (Alpha)**: ✅ Robust scanner for Version 1300 validated.
- **Texture Export**: ✅ DBC-driven `ReplaceableId` resolution working.
- **OBJ Splitter**: ✅ Geoset-keyed export verified on complex creatures.
- **0.5.3 Alpha WDT/ADT**: ✅ Monolithic format, sequential MCNK.
- **WMO v14/v17 converter**: ✅ Both directions implemented.
- **BLP**: ✅ BlpResizer complete.

### Data Generation
- **VLM Datasets (Alpha)**: ✅ Azeroth v10 (685 tiles).

## ⚠️ Partial / In Progress

### MdxViewer — Rendering Quality & Performance
- **3.3.5 ADT loading freeze**: Needs investigation
- **WMO culling too aggressive**: Objects outside WMO not visible from inside
- **MDX GPU skinning**: Bone matrices computed per-frame but not yet applied in vertex shader (needs BIDX/BWGT vertex attributes)
- **MDX animation UI**: Sequence selection combo box in ImGui panel not yet wired
- **MDX per-geoset color/alpha**: Only static alpha used; animated GeosetAnims not wired
- **MDX particles/ribbons**: Parser coverage expanded; runtime behavior verification still pending on effect-heavy assets
- **MDX texture UV animation**: Not implemented
- **MDX billboard bones**: Not implemented
- **WMO lighting**: v14-16 grayscale lightmap + v17 MOCV vertex colors not implemented
- **Vulkan RenderManager**: Research phase — `IRenderBackend` abstraction for Silk.NET Vulkan

### Build & Release Infrastructure
- **GitHub Actions**: ✅ `.github/workflows/release-mdxviewer.yml` — tag push or manual dispatch
- **WoWDBDefs bundling**: ✅ 1315 `.dbd` files copied to output via csproj Content items
- **Self-contained publish**: ✅ `dotnet publish -c Release -r win-x64 --self-contained` verified

### MDX-L_Tool Enhancements
- **M2 Export (v264)**: 🔧 Implementing binary writer.

## ❌ Known Issues

### MdxViewer Rendering Bugs (Feb 12, 2026)

#### MDX Sphere Env / Specular Orientation (Feb 14, 2026)
- **Symptom**: Reflective/specular surfaces (e.g., dome-like geometry) appeared inward-facing on some two-sided materials.
- **Fix Applied**: Fragment shader now flips normals/view-space normals on backfaces before env UV generation and lighting/specular.
- **Status**: 🔧 Patched in code, pending visual confirmation on Dalaran dome repro.

#### WMO Semi-Transparent Window Materials
- **Symptom**: Stormwind WMO maps blue/gold stained glass textures to white marble columns instead of window frames
- **Hypothesis 1**: Secondary MOTV chunk not skipped → MOBA batch parsing misalignment
- **Fix Attempt 1**: Added `reader.BaseStream.Position += chunkSize;` when secondary MOTV encountered in `WmoV14ToV17Converter.ParseMogp` (line 922)
- **Result**: ❌ FAILED — window materials still map to wrong geometry
- **Status**: Root cause still unknown. May not be MOTV-related. Need to check console logs to verify if secondary MOTV is even present in Stormwind groups.

#### MDX Cylindrical Texture Stretching
- **Symptom**: Barrels, tree trunks show single wood plank stretched around entire circumference instead of tiled texture
- **Hypothesis 1**: Texture wrap mode incorrectly clamping both S and T axes when only one should clamp
- **Fix Attempt 1**: Changed `ModelRenderer.LoadTextures` to use per-axis clamp flags (clampS/clampT) based on `tex.Flags & 0x1` and `tex.Flags & 0x2` (lines 778-779)
- **Result**: ❌ FAILED — textures still stretched on cylindrical objects
- **Status**: Root cause still unknown. May not be wrap mode related. Need to check console logs to verify texture flags and investigate UV coordinates.

### AdtModfInjector
- **Problem**: Appends MWMO/MODF chunks to end of file; result is Noggit-incompatible.

## Key Technical Insights

### MCLQ Liquid Heights (Feb 11, 2026)
- MCLQ per-vertex heights (81 entries × 8 bytes) are absolute world Z values
- Heights can slope for waterfalls — adjacent water planes at different Z levels
- MH2O (3.3.5) was overwriting valid MCLQ data with garbage on 0.6.0 ADTs
- Fix: Skip MH2O when MCLQ liquid already found; never overwrite existing MCLQ
- WMO MLIQ liquid type: use `matId & 0x03` from MLIQ header, NOT tile flag bits

### Performance Tuning (Feb 11, 2026)
- AOI: 9×9 tiles (radius 4), forward lookahead 3, GPU uploads 8/frame
- MPQ read throttling: `SemaphoreSlim(4)` prevents I/O saturation
- Persistent tile cache: `TileLoadResult` stays in memory, re-entry is instant
- Dedup sets removed: objects always reload correctly after tile unload/reload

### WMO/MDX Coordinate System (Feb 9, 2026)
- WoW: right-handed (X=North, Y=West, Z=Up), Direct3D CW winding
- OpenGL: CCW winding for front faces
- **Fix**: Reverse winding at GPU upload + 180° Z rotation in placement
- MDX rotations: `rx = Rotation.X`, `ry = Rotation.Y` — NO axis swap
- WMO-only maps: raw WoW world coords (no MapOrigin conversion)

### WMO MLIQ Liquid Positioning (Feb 9, 2026)
- MLIQ data has inherent 90° CW misrotation (wowdev wiki)
- Fix: `axis0 = cornerX - j * tileSize`, `axis1 = cornerY + i * tileSize`
- Tile visibility: bit 3 (0x08) = hidden
- GroupLiquid=15 always → magma (old WMO "green lava" type)

### Replaceable Texture Resolution (Feb 10, 2026)
- Try ALL CDI variants, validate each resolved texture exists in MPQ
- If no DBC variant validates, fall through to model directory scan
