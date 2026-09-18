# Active Context — wow-viewer

Last updated: 2026-09-18

## Fresh-chat route

1. [Spec status](../specs/STATUS.md) — select exactly one active owner.
2. This compact handoff.
3. That owner's `spec.md`, `plan.md`, and `tasks.md`, and linked receipt only.

The [documentation router](../docs/README.md), [spec routing registry](../specs/registry.md),
[progress ledger](progress.md) (dated session history), and [archives](archive/README.md) are
on-demand context, never default reading. Landed-work narrative lives in `progress.md`, not here —
this file states only the current lane and what's still open.

## Current state — v0.6.0-alpha released, push pending (2026-09-18)

`eng/Version.props` is `0.6.0` / `InformationalVersion 0.6.0-alpha`; release notes, CHANGELOG, README,
era matrix, USERGUIDE §12–13 and CLI-TOOLS are updated. Release commit `c247f383`, tag `v0.6.0-alpha`
created **locally** — the operator still has to push (it fires `wowviewer-release.yml` and the GitHub
prerelease). Viewer build clean; title bar reads `v0.6.0-alpha` (operator screenshot).

## Implement next — five v0.6 lanes (none implemented)

Pick exactly one; load its `spec.md`, then `plan.md`/`tasks.md`, then only its linked receipt.
Ordered by dependency, not by number.

| Order | Spec | One-line scope | Next bounded action | Proof owner |
|---|---|---|---|---|
| 1 | [246 modern M2 camera paths + benchmarking](../specs/246-modern-m2-camera-paths-and-benchmarking/spec.md) · Epic 3 | `MD21` CASC M2 camera tracks as playable paths; path-driven modern renderer benchmark with legacy-shaped receipt | `speckit-plan`, FR-002 diagnosis first: locate where modern cameras drop (era dispatch vs `MD21` conversion — `M2ModelReader` already has a `0x74` modern camera branch) | operator run for FPS + path witness |
| 2 | [242 WMO instancing](../specs/242-wmo-instancing-performance/spec.md) · Epic 3 | Per-placement instancing under scene lights; modern data currently one draw call per placement (~5.5 FPS, 16,431 WMO draw calls) | `speckit-plan`; no new god-class members; 246 is its measurement vehicle | operator before/after run |
| 3 | [243 modern→legacy conversion](../specs/243-modern-to-legacy-map-conversion/spec.md) · Epic 4 | One-way modern → **LK v18** + **Alpha 0.5.3**, multi-layer alpha/texture-id merge, batch maps, low-touch UI, optional assets | `speckit-plan`; owned service + UI inventory row | operator client-load witness |
| 4 | [244 modern liquid flow](../specs/244-modern-liquid-flow/spec.md) · Epic 4 | WDT `MAI2` `liquidFlowTexture` (R = +Y west, G = −X south, 128 = zero) → viewer liquid context + one shared flow datum + legacy disposition | `speckit-plan`; confirm magnitude scaling against a real flow texture | operator visual check |
| 5 | [245 modern chunk survey](../specs/245-modern-chunk-completeness-survey/spec.md) · Epic 4 | Inventory every discarded modern chunk + per-chunk legacy build-in feasibility via alpha-mask/texture-id re-expression, loss stated | `speckit-plan`; reproducible corpus-walk receipt | self (counts) |

Handy facts for these lanes: legacy MCLQ already carries a flow vector
([`MclqChunk`](../src/core/WowViewer.Core.IO/Liquids/MclqChunk.cs)) — 244's Alpha-side home;
`inspect casc bench` measures **CASC reads, not renderer frames**, so 246's benchmark is genuinely new;
the v0.6.0-alpha notes already list `MAI2` as uninterpreted (244 closes that).

## Spec 224 governance — audit closed, gate open (last audit 2026-09-11)

224-T201's receipt/symbol audit is complete across all specs that carried checked tasks. 224-T202
(archival) and Gate 2 remain open; next monthly cleanup run is due 2026-10-01. Reports:
[2026-09-11](../specs/224-speckit-governance/evidence/cleanup-2026-09-11.md) ·
[2026-09-10](../specs/224-speckit-governance/evidence/cleanup-2026-09-10.md).

## Open operator-reported defects (2026-09-10, filed not fixed)

Spec 232 Phase 7: T067 (–) "Bake MCSH shadows" option, T068 broken "Include WMO geometry" checkbox,
T069 no-water minimap shading, T070 cell fine-tune granularity, T071 offset counter sizing. Spec 231
Phase 8: T080 hovered-WMO doodad-set combo. Save-pipeline gap is Spec 234's whole purpose.

## Active lane — Spec 236 Scene Lighting, Doodad Performance & Phase Map Tooling — Phase 1 & Minimap Defect Fixes Implemented (2026-09-15)

[236-scene-lighting-doodad-performance/spec.md](../specs/236-scene-lighting-doodad-performance/spec.md) — Implementation active on branch `v0.5.4-dev`.
- **Phase Map Liquid & Doodad Elevation**:
  - Added `LiquidChunkData.WithRehoming` method.
  - Synchronized liquid chunks, doodads (`mddf`), and WMOs (`modf`) with `ZOffset` and `ZScale` in `StandardTerrainAdapter` and `AlphaTerrainAdapter`.
  - Prevented double-transformation on rotated alpha phase layers (`PlacementsPreTransformed = true`).
  - Elevated preserved base doodads and WMOs when overlay changes heightmap elevation.
- **Minimap Live Dragging & Footprint Rendering**:
  - Fixed `WorldScene.GetLayerFootprints()`: removed unrotated guard that stripped `TileOffsetX`/`TileOffsetY`.
  - Expanded `MinimapHelpers.RenderMinimapContent` to include all active phase layer footprints outside base map bounds.
  - Enabled auto-selection of any clicked phase layer footprint and allowed dragging of rotated/mirrored layers.
- **Interactive Fullscreen Minimap Donor Tile Tool**:
  - Implemented `MinimapDonorToolService.cs` (Spec 228 God-Class freeze compliant).
  - Mode switcher (`Navigate` vs `DonorTileTool`, toggle via key `T` or top toolbar).
  - Left-click tile: sets donor source `(sx, sy)` with green border and `[SRC]` badge.
  - Hover target: cyan preview outline and connector line.
  - Left-click target: places donor tile, activates `UsePlacedTilesOnly`, and auto-refreshes terrain.
  - Right-click: cancels source or removes target placement.
- **MDX & M2 Shading Corrections**:
  - Removed `!gl_FrontFacing` normal inversion in `ModelRenderer.cs` and `M2Renderer.cs`.
  - Implemented Half-Lambert diffuse wrapping (`(N·L * 0.5 + 0.5)^2`) matching `WmoRenderer.cs`.
- **Automatic Daytime Disabled by Default**:
  - Changed `TerrainLighting.AutomaticTimeOfDayEnabled` default to `false` so scenes remain stable at midday.
- **Wireframe Defect on Textured WMO Geometry**:
  - Bound `gb.Ebo` explicitly before `GL.DrawElements` in `WmoRenderer.cs`, restoring wireframes on textured geometry.
- **WMO Group Area Names & Hierarchies**:
  - Preserved `uint WmoGroupId` at MOGP offset 0x38 in `WmoV14ToV17Converter` and `WmoV17ToV14Converter`.
  - Added `WMOAreaTable.dbc` loading and hierarchical name resolution in `AreaTableService.cs`.
  - Wired hit-testing in `WmoRenderer.cs`, `WorldScene.cs`, and `ViewerApp.cs` (Spec 228 God-Class Freeze compliant).
- **Zone & WMO Audio Playback + External Emitter Support**:
  - Added zone music/ambience playback, UI controls in `ViewerApp_Audio.cs`, and support for WMO/doodad emitters.
- **Map Generator WDL Output & Continuous Fractal Relief**:
  - Added `WdlWriter.ExtractTileHeightsFromLk(LkAdtData)` and emitted `{mapName}.wdl` in `NewMapCreatorService.cs`.
  - Integrated continuous multi-octave harmonic fractal noise in `TemplatedTerrainGenerator.cs`.
- **Verification**:
  - Full solution build: 0 errors.
  - WDL Writer tests: 3 passed, 0 failed.
  - Templated Terrain Generator tests: 6 passed, 0 failed.
  - Phase unit tests: 76 passed, 0 failed.
  - Receipt: `specs/236-scene-lighting-doodad-performance/evidence/phase1-shading-fix.md`.

## Prior lane — Spec 235 Legacy MDX/M2 Rendering (1.0.0-3.0.1) — Phases 0–4 Implemented & Receipted (2026-09-11)

[235-legacy-mdx-m2-rendering/spec.md](../specs/235-legacy-mdx-m2-rendering/spec.md) — Implementation complete for Phases 0 through 4 on branch `235-legacy-mdx-m2-rendering`.
- **Core Reader Unification (`M2ModelReaderDispatcher.cs`, `M2Era100ModelReader.cs`)**:
  - Dropped the `0x102`–`0x107` `NotSupportedException` refusal wall.
  - Supports MD20 versions `<= 0x107` with classic division layout in `M2Era100ModelReader`.
  - Implemented 108-byte (`0x6C`, version `0x100`) and 112-byte (`0x70`, version `0x104`–`0x107` with `boneNameCrc` at `+0x0C`) bone parsing with track normalization.
  - Populates `EmbeddedSkinDocuments` from embedded division records; `M2SkinProfileRuntime` initializes directly without looking for non-existent external `.skin` files.
- **World Placement & Fallback (`WorldAssetManager.cs`, `WowViewerM2RuntimeBridge.cs`)**:
  - Added explicit handling for `M2Era1121EraTag.Md20_1X_V100_Era100` routing directly to the runtime bridge.
  - Implemented FR-005 Bounding-Box Fallback (`BuildBoundingBoxFallbackModel`, 8-vertex, 12-triangle unit box with `usesCompatibilityFallback: true`) so models with no drawable geometry render bounds rather than remaining invisible.
- **Tooling (`WowViewer.Tool.Inspect`)**:
  - `RunM2Inspect` generates `M2GeometryDocument` for `Md20_1X_V100_Era100` using `GlobalVertices`.
- **Verification Across Staged Clients (All Pass Exit 0)**:
  - 1.0.0.3980 (`xyz.m2`): `ERA: 1.0.0 (MD20 v0x100)`, bones=1, geometry available=true, vertices=72.
  - 2.0.0.5610 (`BloodElfMale.m2`): `ERA: 1.0.0 (MD20 v0x100)`, bones=138, geometry available=true, vertices=4864.
  - 2.4.3.8606 (`BloodElfMale.m2`): `ERA: 1.0.0 (MD20 v0x100)`, bones=143, geometry available=true, vertices=5712.
  - 3.0.1.8303 (`BloodElfMale.m2`): `ERA: 1.0.0 (MD20 v0x100)`, bones=143, geometry available=true, vertices=5712.
  - 3.3.0.10958 (`BloodElfMale.m2`): `ERA: 3.3.5 (MD20 v0x108)`, bones=151, geometry available=true, vertices=6778.
- **Unit Tests & Build**:
  - `M2Era100ModelReaderTests`: 11 passed, 0 failed.
  - `ModelRouteClassifierTests`: 7 passed, 0 failed.
  - Full solution build: 0 errors.
- **Receipt**: [phase1-reader-unification.md](../specs/235-legacy-mdx-m2-rendering/evidence/phase1-reader-unification.md).
- **2.0.0 .mdx -> .m2 Resolution & UV/Vertex Layout Fix (2026-09-12)**:
  - Fixed `FormatProfileRegistry.M2Profile20xUnknown` by lowering `MinSupportedVersion` from `0x104` to `0x100` so 2.0.0 MD20 version `0x100` models pass profile validation without throwing `InvalidDataException`.
  - Added `_dataSource.FileExists` fallback to `WorldAssetManager.ResolveCanonicalModelPath` and `WmoRenderer.ResolveCanonicalDoodadPath` for alternate extensions (`.m2`, `.mdl`) when listfile resolution misses.
  - Updated `resolvedModelPath` from `_resolvedReadPathCache` in `WorldAssetManager.LoadMdxModel` so skin resolution operates on the true `.m2` asset.
  - Fixed `M2Era100Constants.cs` vertex layout offsets: corrected inverted offsets so Normal (`0x14`), UV0 (`0x20`), UV1 (`0x28`), BoneWeights (`0x0C`), and BoneIndices (`0x10`) match the standard WoW 48-byte M2Vertex binary layout, fixing flat/stretched distorted textures and broken UVs. Added unit test `Era100Reader_ReadsVertexAttributes_WithStandardM2LayoutOffsets` (12/12 tests pass).
  - Evidence receipt: [2.0.0-mdx-m2-rendering-fix.md](../specs/235-legacy-mdx-m2-rendering/evidence/2.0.0-mdx-m2-rendering-fix.md).
- **FX Emitter Bounds Fallback & Texture UV Clamping Corrections (2026-09-12)**:
  - Fixed giant solid opaque white bounding-box cubes covering scenes: particle/sound/emitter doodads (`stratholmefloatingembers.m2`, `hellfire_fireparticle.m2`) have `rawVertexCount == 0` on disk by design. In `WowViewerM2RuntimeBridge.cs`, early-return an empty `M2StaticRenderModel` with 0 sections rather than generating an FR-005 bounding box cube.
  - Fixed flat untextured surfaces and single-color streaks on complex 3D meshes (`ballistaruined.m2`, wood, arrows, wheels): in `M2Renderer.cs`, corrected texture clamping flags from `(flags & 0x1u) == 0` to `!= 0`, ensuring textures repeat by default (`TextureWrapMode.Repeat`) and only clamp when bit 0x1 (S) or 0x2 (T) is set. In `ModelRenderer.cs`, corrected adapter clamp deriving and disabled aggressive `ClampToEdge` override on non-opaque textures in `NormalizeAdaptedM2TextureSampling`.
- **Skybox Blending & Transparent Cutout Materials Corrections (2026-09-12)**:
  - Fixed skybox rendering as low-res "N64 polygonal blobs": in `M2Renderer.cs`, line 731 had `if (!backdrop && transparent)`, which disabled blending for all transparent backdrop passes, turning atmospheric domes, cloud layers, star fields, and horizon glows into solid opaque polygons. Changed to `if (transparent)` so blending is enabled for transparent sections regardless of `backdrop`.
  - Fixed plant foliage (`zangarplantgroup05.m2`) and spiderwebs rendering as solid opaque planes: in `M2Era100Constants.cs` / `M2Era100ModelReader.cs`, dynamically compute embedded division section stride (supporting 48-byte records with sort center/radius in 0x104+ models rather than hardcoding 32-byte stride). In `WarcraftNetM2Adapter.cs`, prevented `ShouldPreferProfiledRenderFlags` from overwriting valid render flags when `current` has plausible blend modes, and capped candidate scoring to prevent large tables from overriding genuine materials with 0s (`Opaque`). In `M2StaticRenderModelBuilder.cs`, removed invalid `((batch.GeosetIndex & 0x2) != 0)` projection flag check. In `WowViewerM2RuntimeBridge.cs`, preserved full combiner family mapping including `AlphaKey`.
  - Added unit test `ReadDetailed_ZangarPlantGroup05_ParsesMaterialsAndEmbeddedSectionsCorrectly` (14/14 tests pass).
  - Evidence receipt: [skybox-blending-and-transparent-materials-fix.md](../specs/235-legacy-mdx-m2-rendering/evidence/skybox-blending-and-transparent-materials-fix.md).
- **Bone Quaternion Normalization & AlphaKey Cutout Material Transfer Corrections (2026-09-13)**:
  - Fixed 0x100 character models crumpling into mangled limbs (`TrollFemale.m2`): classic 0x100 models store bone rotation tracks on disk as 16-byte uncompressed IEEE 754 float quaternions (`C4Quaternion`: X, Y, Z, W), NOT 8-byte `M2CompQuaternion`. When read as 8-byte compressed shorts, identity rotation `(0,0,0,1)` decoded to `(-0.5,-0.5,-0.5,-0.5)`, distorting bones by 120° and collapsing the skeleton. In `M2Era100ModelReader.cs`, added `IsUncompressedQuaternionTrack` detector and converted 16-byte float quaternions into canonical 8-byte `M2CompQuaternion` structures in `_extension`, allowing standard runtime sampling (`M2TrackSampler`) to evaluate identity and animation rotations without distortion.
  - Fixed 0x100 foliage and canopies rendering with solid black/white margins (`terokkartreelarge.mdx`, `razorfen_canopy01_hole.mdx`): in `WarcraftNetM2Adapter.ParseEra100Model`, replaced placeholder loop that hardcoded all render flags to `Opaque` (0), propagating `geometry.Materials[i].Flags` and `geometry.Materials[i].BlendMode`. In `M2Renderer.cs`, initialized `buffers.AlphaCutout = section.Material.BlendMode == M2BlendMode.AlphaKey`.
  - Added unit tests `Era100_Synthetic_UncompressedQuaternion_NormalizedAndSampled`, `Era100_Synthetic_MaterialsWithAlphaKey_ParsedCorrectly`, and `BuildEmbeddedStaticRenderModel_SyntheticEra100_TransfersAlphaKeyBlendMode` (17/17 Era100 tests pass, 3/3 embedded profile tests pass).
  - Evidence receipt: [quaternion-normalization-and-alphakey-transfer-fix.md](../specs/235-legacy-mdx-m2-rendering/evidence/quaternion-normalization-and-alphakey-transfer-fix.md).
- **Creature Variant & Replaceable Texture Resolution Corrections (2026-09-13)**:
  - Fixed 0x100 creature variants (`DragonSpawnArmored.mdx`) rendering completely untextured / flat white: in `WowViewerM2RuntimeBridge.cs:BuildEra100Material`, enforced `stageCount = Math.Max(1, (int)batch.TextureCount)`, added direct texture indexing fallback (`geometry.Textures[lookupIndex]`) when `lookupIndex >= geometry.TextureLookup.Count`, and defaulted empty-filename passes to creature skin slot 11.
  - In `M2Renderer.cs`, `ReplaceableTextureResolver.cs`, and `ModelRenderer.cs`: added creature variant suffix stripping (`StripVariantSuffix`), candidate base searching (`[ modelBase, folderBase, strippedBase ]`), scored directory scan for `.blp` textures in `_modelDir`, and safety-net fallback in `TryLoadMaterialTexture` so untextured passes attempt creature skin resolution before rendering.
  - In `ReplaceableTextureResolver.cs`: added `baseSection: 5` (Underwear) prioritization and real client underwear filenames (`NakedPelvisSkin`, `NakedTorsoSkin`); pruned robe skirts (`1201`), cloaks (`1101`), tabards (`1301`), and shoulders (`1401`/`1501`) from `DefaultCharacterSelectionGroups`.
  - Added unit test `BuildEra100StaticRenderModel_TransfersTextureBindingsAndFallbackSlots` (18/18 Era100 tests pass).
  - Evidence receipt: [creature-variant-and-replaceable-texture-resolution-fix.md](../specs/235-legacy-mdx-m2-rendering/evidence/creature-variant-and-replaceable-texture-resolution-fix.md).
- **2.0.0 M2 Animation Playback, Sequence Resolution, and Character Geoset/Texture Fix (2026-09-13)**:
  - Fixed frozen animations where only eyes blinked: in 0x100 models, keyframes in `OldTrack` use global timeline timestamps `[startTimestamp, endTimestamp]`. Normalized sampling in `M2TrackSampler.cs` with `sampleTime = checked((int)start) + ResolveSampleTime(timeMs, duration)`, maintaining 100% regression freedom on 3.x/4.x models (`StartTimestamp = 0`).
  - Fixed 0x44 sequence stride reading offsets in `M2Era100ModelReader.cs` and `M2Era1121ModelReader.cs` (`startTimestamp` at `+0x04`, `endTimestamp` at `+0x08`, derived duration `end - start`).
  - Added complete AnimationData.dbc mapping (IDs 0..248) in `M2AnimationNameResolver.cs`, resolving names like 50 (`Loot`) and 107 (`AttackThrown`).
  - Fixed character models missing midsections and loading with all 51 geosets enabled: added basic body submeshes `0..8` to `DefaultCharacterSelectionGroups`, corrected `_sectionVisibility[i] = visible` loop indexing in `M2Renderer.cs`, and unconditionally applied default customization on model load in `ViewerApp.cs`.
  - Added regression test `Inspect200OrcFemaleSequencesAndBones` verifying continuous bone rotation across time.
  - Evidence receipt: [2.0.0-animation-and-character-geoset-fix.md](../specs/235-legacy-mdx-m2-rendering/evidence/2.0.0-animation-and-character-geoset-fix.md).

## Current lane — Spec 236 Unified Scene Lighting, Doodad Performance & Client-Constrained World Pipeline (v0.5.4-dev)

[236-scene-lighting-doodad-performance/spec.md](../specs/236-scene-lighting-doodad-performance/spec.md) — Authored on branch `v0.5.4-dev`.
Overhaul dark MDX shading bug, add Half-Lambert model diffuse, implement multi-surface scene light casting (torches/WMO MOLT lights onto terrain and WMO surfaces), optimize doodad rendering performance via unified GPU instancing, constrain custom map generator to loaded client listfile assets, and fix GLB export / map merge save pipeline (incorporating Spec 234). **Current next step: continue Phase 2 after the WMO shell slice by adding terrain and doodad/model scene-light consumers.**

- **Phase 2 WMO emitted-light slice (2026-09-16)**:
  - Added `SceneLight`, `SceneLightManager`, and `ISceneLightEmitter` in the viewer rendering layer.
  - `WorldScene` rebuilds active scene lights from visible WMO placements plus MDX/M2/WMO-internal doodad emitters.
  - `WmoRenderer` emits WMO `MOLT` and internal doodad lights, uploads up to eight nearby lights, and accumulates local point-light diffuse in the WMO shell shader.
  - WMO shell instancing is disabled while scene lights are active so each placement receives its own nearby-light selection rather than an approximated shared light set.
  - Receipt: `specs/236-scene-lighting-doodad-performance/evidence/phase2-wmo-light-casting-slice.md`; T011 checked, T010/T012/T013/Gate 2 remain open.


## Other open lanes (all implemented-with-operator-gates; see each spec's tasks.md for detail)

- **Spec 232** (cartography composition) — T056/T057/T058/T059/T064/T066 source-landed; operator
  visual witnesses owed for all. T051 (lock badge), T053 (WL inspector), T054 (default page), T015d
  (rotated-seam screenshot) also await witnesses. Next after witnesses: T055 (UniqueId color-coding),
  T065 (chunk off-by-one re-audit).
- **Spec 233** (marketing capture automation) — P1 source landed; **T015 operator tour witness**
  (`FlybyUndead`, Warm Path, Feature Tour + Video) is the next concrete action, then Phase 4 receipts.
- **Spec 231** (Editor/Archaeology UI overhaul) — Phases 0–4, 6-batch-1, and 7 landed; Phase 6
  batch 2 (T061–T064) and Phase 5 (navigation smoke, inventory v3 final) open.
- **Spec 227** (UI re-audit) — only T001/T002 done; **T004 gate genuinely still open** (confirmed
  by 2026-09-10 audit), which is why Spec 228 stays blocked.
- **Spec 228** (source decomposition) — blocked on 227 T004, 0 tasks landed.
- **Spec 223** (UI consolidation) — operator retest owed (fog, WMO-only global WMO, playback/
  capture, ffmpeg, spatial UI) with a real client; video-capture release hardening also needs a
  real with/without-UI recording + playback + Play+Video witness before shipping.
- **Spec 226** (renderer polish) — wireframe root causes fixed; captures still owed.
- v0.5.3-rc1 — tag push + GitHub Actions release are operator-owned; visual pass on wireframe/
  selection/toolbar changes still owed.

## Non-negotiable constraints

- Preserve MPQ/ADT/WMO/M2/MDX readers and `AlphaWdtWriter`; no feature scope rides on a refactor.
- Runtime visual, input, FPS, audio, video, and client-data proof are operator-owned.
- Preserve unrelated dirty work; stage named files only.
- New UI work follows Spec 228 (owned service classes, no new `ViewerApp_*`/`WorldScene` members)
  and AGENTS.md §9 receipts.

## Handoff

**Immediate:** continue Spec 236 Phase 2 from the WMO shell slice: add terrain local-light upload/evaluation, then doodad/model external scene-light consumers, then request operator-owned runtime visual proof for torch/brazier spill onto WMO geometry and ground. Operator decisions still owed: the 16 un-checked Spec 223 checks (accept or supply retroactive receipts) and the Spec 231 T074 receipt gap.

**Do not claim:** any Spec 232/233/223/226 visual/runtime acceptance; the Spec 223 Phase 1–5 source
tasks are now un-checked pending receipts; and the Spec 235 plan rewrite stays blocked on a real-file
inspection (read `tools/inspect` usage + the `--client`/`--game-path` invocations in specs 104/154/205
first).
