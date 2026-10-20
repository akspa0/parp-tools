# Progress — wow-viewer

Last updated: 2026-10-20

## 2026-10-20 — PM4 M2 per-object partition is negative

- Extended `Pm4M2PartitionProbe` to test topology, MSVT vertex sets, MSUR index ranges, Padding,
  +0x18, Normal, Height and MSLK GroupObjectId against authored M2 origins.
- 170 tiles, 112,818 M2 surfaces, 22,286 components. 2,000 components mix different models; the
  largest covers 133 origins across 7 model paths. 6,576 placements share vertices; 9,570 have
  overlapping MSUR ranges; Padding is 0 for every M2 surface; 7,315 placements carry multiple
  GroupObjectIds. The merge is genuine; per-object M2 ownership remains U.
- [Receipt](../specs/279-decode-pm4-format/evidence/phase-3-m2-partition-2026-10-20.md).

## 2026-10-20 — PM4 M2/WMO partition from authored placements

- New read-only `Pm4PlacementGroundTruthProbe` reads companion `_obj0.adt` MDDF/MODF placements
  (the earlier ASCII `MDDF` scan was a false negative; on-disk signatures are reversed) and tests
  each placement origin against every surface's projected polygon.
- Whole map: 616 files, 411 with placements, 308,892 surfaces. M2 covered 12,147/59,544 (20.4%);
  WMO covered 883/1,116 (79.1%).
- **Partition:** `Ck24Type == 0` / `GroupKey == 3` selects 14,489 of 16,338 M2-only surfaces (88.7%)
  vs 57 of 1,961 WMO-only (2.9%). WMO is `Ck24Type` 64–67 / `GroupKey` 16–19. Link types 17/18/20/26/28
  are WMO-exclusive; 1/2/4/12 are M2-leaning. The zero bucket is the M2 class, not a per-object ID.
- 4 new probe tests pass. [Receipt](../specs/279-decode-pm4-format/evidence/phase-3-placement-ground-truth-2026-10-20.md).

## 2026-10-20 — Spec 279 PM4 reference and continuing research

- [Current reference](../docs/research/pm4-format-reference.md): 16 parsed signatures, raw layouts,
  evidence levels and 17 unknown groups; old wiki draft explicitly superseded.
- Spec Kit reference gate passed; all-map +0x04 test rejects a general reverse pointer, meaning
  still unknown. **2A:** raw +0x01 selects source edges (674,013 matches vs shifted control 137);
  point-index alternatives unconfirmed. **2B:** +0x04 is not a unique edge ID (889,594 reused
  values); type-1 reciprocal records share it 98.92%. **MSHD:** five of eight words always zero;
  Field00/08 correlate with Mdbh ~0.76–0.79. [2A receipt](../specs/279-decode-pm4-format/evidence/phase-2a-edge-ordinal-2026-10-20.md),
  [2B receipt](../specs/279-decode-pm4-format/evidence/phase-2b-edge-identity-2026-10-20.md),
  [MSHD receipt](../specs/279-decode-pm4-format/evidence/phase-3-mshd-2026-10-20.md).

## 2026-10-20 — PM4 target-tile references validated across the whole map

- All 616 PM4s read in short summary passes (4–7 s). Exact surface/link windows and packed target
  tile X=low byte/Y=next byte validated by geometry, back-links and wrong-order/local controls.
- 15,773 cross-tile records; 14,659 targets available, none out of range there. Paired 10,222 links:
  addressed/local alternatives give 9,748/15 back-links and 5,026/28 boundaries. File-local blanket
  assumption is refuted. Raw 0x01/0x11 direct adjacency differs from path-geometry kinds.
- 32 focused tests pass; addressed hypotheses added only to research audit. No matching gain,
  complete format decode or M2 ownership claimed. Authored editor path-node layer remains separate.
  [Receipt / measured-refuted-unknown ledger](../specs/270-pm4-object-association-ledger-and-placement/evidence/structural-foundations-2026-10-20.md).

## 2026-10-20 — Whole-map PM4 partition instrument, tile pools and WL hints

- Operator directed map-wide CK24-zero/MSUR/MSLK research: scaled copies can be merged; raw
  group/type/attrMask meanings are unknown. Museum placements are independent authored reference.
- Prepared an owned core audit preserving all raw fields/links/provenance, ten partition hypotheses,
  scale-family diagnostics and authored transforms; whole-map runtime remains operator-owned/not run.
- Added persistent tile/range object pools and explicit candidate filtering, with no score changes.
  Real CLI output: 2→1 synthetic candidates; 70 WL filenames/11 recognized aliases; no guessed tiles.
- 25 focused tests pass and both runners build. Existing matchers/readers/viewer unchanged; UI and
  automatic WL footprint association remain open. [Receipt and operator usage](../specs/270-pm4-object-association-ledger-and-placement/evidence/map-context-instrument-2026-10-20.md).

## 2026-10-20 — PM4 sparse-M2 investigation (no production changes)

- Operator correction: older ChatGPT-era PM4 claims may be coincidences. Preserve working WMO
  behavior; investigate sparse M2 geometry rather than treating old field names as established facts.
- A read-only five-pair probe records input hashes, raw groups, component fingerprints and controls.
  On M2-only 16_33, 170/234 components have ≤4 triangles; default bins yield 171 distinct fingerprints.
  Near-origin proximity is 233/234 within 24 yd, but the reflected control still scores 192/234;
  neither proximity nor finer-bin separation proves asset identity.
- The 50_38 test's “zero WMOs” premise conflicts with two authored wall posts and one M2 in its
  companion ADT. Existing tests pass **4/4**, not the earlier handoff's 7/7; no real asset match proven.
- No production matcher/reader, original data or restored map changed. Operator's follow-up corrects
  the next direction: all PM4s as one map object, retained cross-file/chunk context before isolated
  M2 fragment matching. WMO baseline ~70% accurate (operator-reported); parsed is not fully decoded.
  [Receipt](../specs/270-pm4-object-association-ledger-and-placement/evidence/investigation-2026-10-20.md).

Older entries are preserved in [the indexed archive](archive/2026-10-20-progress-pre-2026-10-14.md).

## 2026-10-20 — MCNK Subchunk Desync Fix, 3.3.5 Format Profile Auto-Detection, Terrain Blending Restoration & PM4 Surface Correlation

- **MCNK Subchunk Walking Desynchronization Fix (`Mcnk.cs`)**:
  - **Issue 1 (MCSH overrun)**: In `Mcnk.cs`, `ComputeConsumedSize` for MCSH computed `Header.SizeMcsh - 8` = `512 - 8 = 504`. But `Header.SizeMcsh` in WoW MCNK is the raw payload size (512 bytes), not payload + 8. Advancing by 504 bytes landed 8 bytes early, directly into shadow data, corrupting the subsequent MCAL subchunk token on all 71 chunks with shadows.
    - *Fix*: `fromHeader = _parseOptions.UseHeaderShadowSize ? Header.SizeMcsh : 0;` (without `- 8`).
  - **Issue 2 (MCAL declared size mismatch)**: In WotLK ADTs, Blizzard tools wrote the subchunk MCAL size as 0, 4, or 512, while the true payload size was stored in `Header.SizeMcal - 8`. In chunks 11 and 34, `fromHeader` was 472 and 470, while declared subchunk size was 512. Previously `Math.Max(size, fromHeader)` took 512 and overshot the end of the chunk by 32 bytes, causing an overrun break and leaving `McalRawData = null`.
    - *Fix*: Prioritized `fromHeader` (`Header.SizeMcal - 8`) whenever valid (`fromHeader > 0 && fromHeader <= dataLength - dataStart`).
  - **Issue 3 (Subchunk loop boundary)**: In `Mcnk.cs`, `while (remaining > 8)` skipped the final 8-byte subchunk (such as `MCSE` with declared size 0), leaving `remaining = 8` and `SubchunkWalkConsumedExactly = false`.
    - *Fix*: Changed to `while (remaining >= 8)` and checked `consumedSize > (uint)remaining - 8`.
  - **Issue 4 (DeclaredSize fallback)**: Added safety check so that even under `DeclaredSize`, if MCAL declares size 0 or 4 and `Header.SizeMcal >= 8`, it consumes `fromHeader`.
  - Added unit tests in `McnkSubchunkWalkTests.cs` (`Scan_McalWithZeroDeclaredSize_ConsumesFromHeaderWhenConfigured`, `Scan_McalWithOverDeclaredSize_ConsumesHeaderSizeWithoutOverrunning`). All 6/6 tests pass.
- **Format Profile Resolution & Auto-Detection**:
  - In `FormatProfileRegistry.cs` and `AdtFormatProfile.cs`, mapped all 3.x clients (`major == 3`) to `AdtProfile33512340` (and `WmoProfile33512340` / `M2Profile33512340`) instead of incorrectly falling through to `AdtProfile060070Baseline`.
  - Relaxed `TryParseBuild` to accept 2, 3, or 4 component version strings.
  - Added `TryDetectLichKingAdt` to `StandardTerrainAdapter.cs`: even when client version is null/unresolved, tiles with `MH2O`, `ofsMcnr == 0x520`, or `sizeMcal > 0` are automatically promoted to `AdtProfile33512340`.
- **Terrain Shader Blending Restoration (`TerrainRenderer.cs`)**:
  - Restored authentic sequential `mix()` blending for layers 0..7 with accumulated alpha tracking.
  - Preserved base layer visibility toggle and strict ASCII encoding.
- **Verification on Real Client Data**:
  - Verified on real Wrath 3.3.5 client ADT `Northrend_23_18.adt` (`H:/CLIENTS/WoW335/modernwow`): 256 of 256 chunks now parse 100% of their alpha payload with 0 failed chunks (previously failed on dozens of chunks, causing checkerboarding and black slopes).
- **PM4 M2 Surface Correlation & Unauthored Placement**:
  - Implemented `Pm4SurfaceCorrelationContracts.cs`, `Pm4SurfaceCorrelationExtractor.cs`, and `Pm4SurfaceCorrelationMatcher.cs`.
  - Extracted collision mesh and attachment geometry in `M2ModelReader.cs` and `M2ModelDocument.cs`.
  - Integrated unauthored placement resolver (`Pm4UnauthoredPlacementResolver.cs`), WMO fall-through filtering (`WmoContainerFallThroughFilter.cs`), and spawn streaming services.
  - Added test suite `Pm4M2SurfaceCorrelationTests.cs` (7/7 tests pass).

## 2026-10-18 — Spec 274 Shipped, Performance Features Committed, WMO Portal Occlusion Quarantined

- **Spec 274 Committed (`e667ec6a`)**:
  - Implemented headless clean dataset export across 37 Zarr v3 map revisions (`v60_batch_export_clean_pairs.py`), clean pair dataset loader, lightweight domain translation model (`domain_translation_model.py`), and DXT1 parity validation.
  - All 228 pytest tests pass in `data-harvester`.
- **4.3.4 Format Flags Inspection Committed (`00667224`)**:
  - Added `FormatFlagsDescriber.cs` and integrated into Terrain, Object, Model, and Standalone inspection panels.
  - Unit tests pass 7/7.
- **Renderer Performance Committed (`0ab72a43`)**:
  - Optimized spatial M2 lighting collection, terrain streaming MCNK FourCC fast-path, and asset path resolver negative caching.
- **WMO Visibility Status & Quarantine**:
  - Aggressive portal occlusion culling in `WmoRenderer.UpdateRuntimeVisibility` caused interior rooms and floors to disappear (e.g. `duskwoodabandoned_humantwostory.wmo` missing ground floor through doorways).
  - Restored fail-safe frustum-visible admission loop in `WmoRenderer.cs` to ensure geometry never disappears during navigation.
  - Quarantined WMO portal culling for dedicated audit against 3.3.5 `MapObj.cpp` in a fresh chat.

## 2026-10-17 — 3.3.5 Stormwind WMO Group Premature Unloading & False Interior Culling Fix

- **Root Cause Diagnosed**:
  - `FindContainingInteriorGroups` in `WmoPortalVisibilityDecision.cs` used `BoundsPadding = 2.0f` (6 feet in every direction) on interior room AABBs and ignored exterior portals. In tight city environments like Stormwind where buildings have angled walls, archways, and narrow streets, walking down the street placed the camera inside the expanded AABBs of adjacent interior shops/rooms.
  - This falsely switched the WMO into `WmoPortalVisibilityMode.Interior`. Because the camera was actually outdoors on the street, depth-0 portal checks (`!IsOnPortalSide`) failed to traverse from the inside of the room outwards.
  - In `WmoRenderer.cs`, `Mode == Interior` with `reachedExterior == false` culled ALL exterior groups (`flags & 0x8`), causing exterior facades, streets, archway walls, and roofs to vanish while leaving isolated second-floor interior planks floating in mid-air and exposing bare terrain hills.
- **Client Ground-Truth Alignment (`H:\wowdecomps\3.3.5a\wowdecomp-main`)**:
  - `CWorldScene::LocateViewer3()` (`World.cpp:2901-2950`) & `CMap::LocateViewerMapObjs()` (`Map.cpp:4033-4130`): native client casts a vertical downward ray; if the hit triangle belongs to an exterior group (`flags & 0x8`) or terrain, `exterior = 1` and `s_viewerMapObjDef = nullptr`, declaring the camera OUTDOORS.
  - When outdoors, `CMapObjDef::CullMapObjDefGroupFromExterior()` (`MapObj.cpp:2598`) renders all exterior groups in the camera frustum and traverses exterior portals inward.
- **Implemented Fix**:
  - In `WmoPortalVisibilityDecision.cs`:
    - Reduced `BoundsPadding` from `2.0f` to `0.1f` to prevent interior AABB bleeding through exterior walls into streets.
    - Updated `FindContainingInteriorGroups` to test candidate interior groups against their exterior-connecting portals. If the camera is on the exterior (street) side of an exterior portal (`!IsOnPortalSide(portal, cameraLocalPosition, edge.SourceSide)`), the candidate is rejected as a false positive, keeping `Mode = Exterior`.
  - In `WmoRenderer.cs`:
    - Added `ContainsStrict` helper and fail-safe exterior group admission (`admitExterior = reachedExterior || !isStrictlyInsideSource`). If the camera is not strictly enclosed inside the candidate interior group's tight bounds, exterior groups (`flags & 0x8`) in the frustum are never culled.
  - Added 3 regression unit tests in `WmoPortalVisibilityDecisionTests.cs`: `CameraOutsideExteriorPortalDoesNotTriggerInteriorMode`, `CameraInsideAabbOfInteriorRoomFailsToExteriorWhenOnExteriorSideOfPortal`, and `CameraInsideInteriorRoomStillCorrectlyTriggersInteriorMode`.
  - All 16 `WmoPortalVisibilityDecisionTests` pass (100%); all 181 WMO tests pass (100%); full solution builds with 0 errors.

## 2026-10-17 — 4.3.4 Client Ground-Truth Format Flags & Viewer Inspection Surfaces

- Audited WoW 4.3.4 (build 15595) decompilation (`H:\wowdecomps\4.3.4\wowdecomp15595-w15595`) against client engine sources (`MapLoad.cpp`, `WorldScene.cpp`, `MapChunk.cpp`, `WorldView.cpp`, `MapObjRead.cpp`, `MapObjGroup.cpp`, `M2Model.cpp`):
  - **WDT MPHD Flags**: Clarified bit `0x4` (gates `shaderType = 2` specular/envmap modern shader mode; not merely uncompressed alpha), `0x1` (Global WMO map), `0x2` (Vertex format 2), `0x8` (Sorted doodads/cull exterior), `0x10` (Vertex normals format 16), `0x20` (Flip ground winding).
  - **WDT MAIN Flags**: Audited `0x1` (HasAdt) and `0x2` (Infinite Ocean generation plane `0x20`).
  - **WMO Root & Group Flags**: Confirmed group `0x4000000` + `0x80` (`0x4000080`) antiportal occluder semantics (bsp cleared, zero render batches, excluded from rendering and collision, feeds occluder quads).
  - **M2 Flags**: Confirmed `0x1` (Tilt normal), `0x3` (Surface basis), `0x80` (Cast shadows), and M2 material flags (`0x1` Unlit, `0x2` Unfogged, `0x4` TwoSided, `0x10` DepthWriteDisabled, `0x20` HitTestTransparent, `0x40` DisableShadowCasting, etc.).
  - **MCNK Flags**: Decoded `0x40` (HasMccv) and `0x8000` (FixedAlpha uncompressed).
- Created durable documentation: [`flags-ground-truth.md`](../test_data/native-research/4.3.4-decomp/flags-ground-truth.md).
- Implemented `FormatFlagsDescriber.cs` in `WowViewer.Core.Runtime.World` providing clean bitmask decoders for all formats.
- Surfaced decoded ground truth flags throughout the viewer UI:
  - **Terrain Inspector**: Added WDT MPHD Flags, WDT MAIN tile flags, and MCNK chunk flags.
  - **Object Inspector (`InspectorPayloadsService.cs`)**: Added WMO Root Flags and M2 Model Flags.
  - **Model Inspector (`ModelInspectorPanelService.WmoGroups.cs` & `Sidebars.cs`)**: Added Group Flags to standalone WMO group controls, Root Flags to Selected WMO controls.
  - **Standalone Model Loader (`StandaloneModelLoaderService.cs`)**: Added Root Flags and Model Flags to loaded model info strings.
- Added comprehensive unit tests in `FormatFlagsDescriberTests.cs` (7/7 passed). Solution builds with 0 errors.

## 2026-10-15 UTC — Classic WC3 Model Alignment (4eb0da/war3-model) & Replaceable Texture Support

- Based on classic `4eb0da/war3-model` specifications (non-Reforged):
  - Fixed variable-length stream parsing for dynamic chunks in version 800 MDX (`LITE`, `BONE`, `HELP`, `PRE2`, `RIBB`) via `Wc3MdxAdapter.cs` so chunks without leading counts are correctly framed without corrupting stream offsets.
  - Added safe per-entry error recovery to `MdxFile.cs` for lights, bones, and helpers to avoid single malformed entries aborting whole model loads.
  - Added WC3 replaceable texture resolution in `ModelRenderer.cs` for IDs 1 (TeamColor), 2 (TeamGlow), 11 (Cliff), 21/32 (Water), and 31 (Lumber).
  - Added synthetic fallback generation in `Wc3DataSource.cs` for unauthored replaceable textures (TeamColor00, TeamGlow00, Cliff0, Lumber) and relaxed archive candidate matching for models with textures in adjacent or root directories.
  - All 194 Wc3 and MDX tests pass (100%).

## 2026-10-14 local / 2026-10-15 UTC — WC3 Beta BLP0 and MDX Fixes

- Operator screenshot proved the BLP0 placeholder was the actual blocker. Added WC3-only shared-header/external-mip JPEG/palette decoding and raw four-channel JPEG handling. Real Crown1 decoded at 128×32 and its mip 3 at 16×4, with input/output hashes recorded.
- MDX files now use an isolated existing-renderer preview; real v800 material framing required an opt-in adapter above the unchanged reader. Adjacent model parses at 6 geosets / 90 vertices / 3 sequences. Viewer build and 86 focused tests pass; GPU appearance remains operator-owned. [Receipt](../specs/272-wc2-wc3-viewer-support/evidence/receipt-beta-blp0-and-mdx-2026-10-15.md).

## 2026-10-14 — WC3 File Browser Code Landed

- Implemented owned WC3 loose/MPQ file session, image/build-text preview, safe byte-preserving extraction and cancellation; wired **File Browser → WC3 files (loose / MPQ)** without changing shared readers or world rendering.
- Solution build passed (0 errors); 77 focused WC3/MPQ/bootstrap/BLP tests passed. [Receipt](../specs/272-wc2-wc3-viewer-support/evidence/receipt-wc3-implementation.md). Real beta/UI proof remains operator-owned; BLP0/external-mip preview is unsupported, and shared 4-bit alpha limitations are visible.

## 2026-10-14 — Operator Corrected Spec 272 to WC3 File Browser Support

- [Current spec](../specs/272-wc2-wc3-viewer-support/spec.md) supersedes the initial WC2-first interpretation: WC3 tilesets and build text in the existing File Browser, select loose/installed data, and unpack selected files; no world-renderer work. Existing MPQ/listfile readers are the base, not a rewrite.
- Authored [plan](../specs/272-wc2-wc3-viewer-support/plan.md), research, data model, local contracts, and quickstart. The extracted beta root is operator supplied but uninspected (inventory declined). Zezula's public listfile link was confirmed; both HTTPS ZIP retrievals failed certificate-name validation, with no bypass or verified package.
- No runtime code/client decode/extraction/UI proof occurred. Next: receipt-gated tasks, permitted beta sample evidence, and local listfile import/acquisition. Prior GDS/WC1 and restoration inputs remain deferred history.

## 2026-10-14 — WC2-First Warcraft Support Specified (No Runtime Implementation)

- Created [Spec 272](../specs/272-wc2-wc3-viewer-support/spec.md) and its [operator addendum](../specs/272-wc2-wc3-viewer-support/operator-addendum.md): original DOS WC2 terrain/sprites first; classic WC3 viewing, texture comparison/synthesis, coordinate helpers, and terrain bridging later. Battle.net is not required; PlayStation source is reference-only. The operator identifies retained WC1 behavior in the shared GDS source: plan WC1 as a reuse-driven follow-on, with specific data variants validated separately.
- [Bounded discovery](../specs/272-wc2-wc3-viewer-support/evidence/discovery-2026-10-14.md) read two archive headers/hashes, eight loose map chunk envelopes/hashes, and focused local source references. Real archive fields contradict the draft's constant-type assertion; coordinate scale claims also require reconciliation.
- No viewer, format reader, source data, or runtime test changed. No terrain/sprite output or client proof exists. Next: bounded DOS WC2 Spec Kit plan, then receipt-gated tasks; future implementation is not scheduled into the release.

## 2026-10-14 — Ground-Effect Regression Reverted; v0.6.0.1 Reissued

- Operator reported the tint changes made the viewer unable to load anything. Removed the per-instance tint field, interface overload, MDX material multiplication, scene tint plumbing, and its new test assertions. The affected source/test files compare byte-for-byte with the pre-tint `b6f54a0b` baseline (`git diff --exit-code` passed).
- Focused DetailDoodads tests pass 8/8; Debug solution build passes with 0 errors. These do not prove real-client/map loading; no crash log or stack trace was available.
- Reissued `v0.6.0.1` at `5ba5eb2a` with the prior ground-effect render path and split/deduplicated CLI packages. Four-RID build `38512662293` and release workflow `38513490254` passed; eight assets are attached.
- Operator still must verify the replacement viewer loads the previously failing client/map (Spec 271 C4). Ground-effect lighting remains deferred; do not reintroduce the tint change without root-cause diagnosis and isolated validation.

## 2026-10-14 — v0.6.0.1 Reissued After Ground-Effect Load Regression

- Reverted the six ground-effect tint/API source/test changes; `git diff --exit-code b6f54a0b -- <six files>` returned 0, proving those files match the pre-tint baseline. No crash log/reproduction was available, so root cause beyond that code delta remains unconfirmed.
- Focused DetailDoodads tests passed 8/8; Debug solution build passed with 0 errors.
- Four-RID baseline build run `38512662293` and replacement release run `38513490254` succeeded. Reissued `v0.6.0.1` still contains eight separate/deduplicated viewer/tools ZIPs.
- **Operator proof still owed:** download/run the replacement viewer against the previously failing client/map and confirm loading succeeds (Spec 271 C4). No visual/runtime claim is made from build/test output.
- `Version.props` is restored to `0.6.1-dev` on active branch `v0.6.1-dev`; release tag commit keeps `0.6.0.1`.

## 2026-10-14 — Ground-Effect Tint Reverted After Viewer Load Regression

- **Operator report:** after the ground-effect doodad changes in v0.6.0.1, the viewer cannot load anything.
- **Action:** removed the per-instance tint parameter/property, renderer API overload, shader material tint multiplication, and associated tests. Existing ground-effect placement/rendering code is restored to its pre-tint path; shared CLI packaging is untouched.
- **Evidence limits:** no crash log, stack trace, or client reproduction was available. The code delta is the suspect, but the root cause is not independently established. Do not reintroduce the tint implementation.
- **Verification:** focused DetailDoodads tests pass 8/8; full Debug solution build passes with 0 errors. These do not prove real-client load behavior.
- **Next:** rebuild/reissue v0.6.0.1 with the tint code absent, then operator verifies that a real client/map loads. Track lighting separately under Spec 259 T014/T015.

## 2026-10-14 — v0.6.0.1 Tool Build/Release Correction + Ground-Effect Tint

- **Root cause:** the release workflow published only the viewer project; none of the nine executable
  projects under `wow-viewer/tools/` were in platform release outputs. Two projects were also absent from
  `WowViewer.slnx`.
- **Build correction:** all nine CLI projects now publish into one shared `publish/<rid>/tools/` folder
  under the platform build tree. Shared DBD definitions, listfiles, runtime DLLs and libraries are emitted
  once; the workflow verifies each apphost, project DLL, deps file, and runtimeconfig. The viewer and
  tools are separate ZIP artifacts per platform.
- **Ground effects (subsequently reverted):** the attempted terrain MCCV/MCSH → MDX per-instance tint
  path was removed after the operator reported that the viewer could not load content. Its earlier tint
  unit tests passed but did not detect the runtime regression; do not treat that attempt as delivered.
- **Verification:** `dotnet build` passed (0 errors); focused DetailDoodads tests 8/8; the Inspector ran
  `--help`; four-RID CI run `38506799386` passed. Release run `38508425730` passed; replacement tag
  `v0.6.0.1` publishes eight assets. Downloaded Windows tools ZIP verified 9 apphosts, 1,342 DBD files,
  one common `WowViewer.Core.dll`, and Inspector `--help` exit 0. Receipts:
  [Phase C release](../specs/271-v060-release-and-integrated-help/evidence/receipt-phase-c-v0601.md) ·
  [ground-effect tint](../specs/259-ground-effects-and-wmo-detail-doodads/evidence/receipt-lighting-propagation-2026-10-14.md).
- Restored `eng/Version.props` to `0.6.1-dev` after tagging. Existing dirty `imgui.ini` files were not staged.
