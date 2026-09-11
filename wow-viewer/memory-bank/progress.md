# Progress — wow-viewer

Last updated: 2026-09-10

## 2026-09-10 — Spec 235 Phase 0 reconciliation: real findings, and a process failure worth recording

- **Findings (all code-evidenced, written up in
  [235/research.md](../specs/235-legacy-mdx-m2-rendering/research.md))**: `ResolveMdxProfile` is dead
  code (explains "MDX is fine"); `ResolveModelProfile`'s `major==1` null disables both
  embedded-profile routes for all of Vanilla; **`WorldAssetManager`'s no-skin chain has no
  `Md20_1X_V100_Era100` branch at all**, so the 1.0.0 reader is never reached from world placement —
  fixing its bone bug alone would render nothing; **no bounding-box fallback exists anywhere in that
  chain** (every failure returns `null`), which is precisely the "no bounding boxes either" symptom;
  the `0x102`-`0x107` refusal is a hardcoded `NotSupportedException` citing a stale spec number;
  **Warcraft.NET's `MD21` already reads the whole era generically and is already wrapped by our own
  adapter**, missing only the embedded skin/view walk — the same single gap Spec 104 diagnosed;
  and `M2Era100Constants`' Ghidra offsets are +8 shifted vs Warcraft.NET's layout with tests that are
  self-admittedly synthetic and cannot detect a wrong offset.
- **Consequence**: plan.md's six-phase per-era structure is the wrong shape and predates these
  findings. The reframe (one generic reader + the embedded-skin walk + drop the version wall + a real
  bbox fallback, bespoke readers only where a real file proves they're needed) is recorded at the end
  of research.md but **not yet applied to plan.md**.
- **Process failure, recorded deliberately**: no real client file was ever read this session. After
  `m2 inspect --archive-root` failed, the response was to guess asset paths (three of them), then
  guess flags, then propose writing new throwaway tooling — in a repo that already has complete
  inspection tooling and a vendored listfile. The operator's correction was blunt and correct: *"we
  have tooling to inspect anything, how the hell are we in unknown waters."* The failure was not
  reading the existing tool's usage or the working invocations in specs 104/154/205 before
  improvising. Next session starts there.
- Also: `wowdev.wiki` 403s WebFetch (both `/M2` and `?action=raw`); use a browser or local copy.

## 2026-09-10 — Spec 235 authored (legacy MDX/M2 1.0.0-3.0.1), found and reconciled duplicate specs

- Operator directive: MDX/M2 rendering for the 1.0.0-3.0.1 client range is "very much
  non-functional" (no objects render, no bounding boxes most of the time); fuckported
  (non-standard chunk-rewritten) assets must render wherever Warcraft.NET can read them; MDX
  torch/light-emitter effects are missing in some cases. Used `speckit-specify` to author
  [235-legacy-mdx-m2-rendering](../specs/235-legacy-mdx-m2-rendering/spec.md) on new branch
  `235-legacy-mdx-m2-rendering` (created off `234-map-save-new-map`'s tip).
- **Grounded in a real code bug found this session**: `FormatProfileRegistry.ResolveMdxProfile`
  has no case for any build with major version >= 1 and silently falls back to the 0.6.0/0.7.0 MDX
  profile; `ResolveModelProfile` (M2) has no case for `major == 1` at all and returns `null` for
  the entire Vanilla retail era.
- **Caught and corrected a near-duplication mid-authoring**: before finalizing, checking
  `specs/epics/active-epics.md` surfaced Spec 104 (legacy-m2-rendering, 1.0.0-2.4.3, full plan/
  tasks/research kit, 7/27 tasks checked, header claims "Status: Active") and Spec 154
  (m2-era-reader-parity, 1.x-3.0.1, planned but never task-broken) — both nearly identical in
  scope, both invisible to default routing because neither was listed in any epic. 154's own
  measured survey (2026-08-15, real staged clients) is far more precise than this session's
  initial framing: the broken range is **exactly** `0x100` through `0x107`; `3.0.1` declares
  `0x107` while `3.3.0.10958` declares `0x108` and reads cleanly (151 bones, 155 sequences) — the
  real reference point is `0x108`/3.3.0, not "3.3.5 through 4.0.0" as casually assumed; a
  `4.0.0.11927` beta also crashes (unhandled, reading camera records), contradicting "4.x already
  works." Rewrote 235 to explicitly supersede 104 and 154's unimplemented residue, incorporate
  their measured findings, and add dated amendment notes atop both (content preserved, not
  archived). Spec 105 (1.0.0 texture/animation/lighting, one model) and Spec 193 (Benilla 1.12.1
  reference client) remain valid, non-duplicated prior art. Updated `epics/active-epics.md` Epic 4
  and `specs/STATUS.md` row 10 accordingly.
- **No implementation. Not planned yet.** Next: `speckit-plan`, starting with FR-014 — reconcile
  `FormatProfileRegistry` against whatever era-resolution mechanism 104/154 already built
  (`M2ModelReader100` per Spec 193) before writing any new code.

## 2026-09-10 — First real Spec 224 governance audit (224-T201) + operator bug batch

- **Ran the actual `speckit-cleanup` audit for the first time** (Spec 224 Phase 2 had been open
  since its own authoring — the 2026-09-06 pass only reorganized files, never checked receipts).
  Audited via parallel read-only agents: **227 clean** (both checked tasks have real receipts,
  T004 gate genuinely still open, correctly blocking Spec 228). **232 found real staleness: 8 of
  12 checked tasks (T050, T052, T056, T057, T058, T059, T064, T066) were checked against
  acceptance criteria their own linked receipts admit are still open** — several literally
  contained "Unchecked pending operator witness" text while the checkbox read `[x]`. All 8
  corrected to `[ ]` with an inline audit note; 4 (T015a/b/c/e) confirmed as genuine passes.
  **233, 231, 223 audits failed on a session rate limit** (resets 6pm America/New_York) — not yet
  re-run; do not treat those specs' checked tasks as verified.
- Compacted `memory-bank/activeContext.md` (188 → ~65 lines; had re-grown from a 2026-09-06
  cleanup's 48 lines back to 188 in 4 days by accumulating session narrative that already
  duplicated this file). Full prior version archived at
  [archive/2026-09-10-pre-cleanup-active-context.md](archive/2026-09-10-pre-cleanup-active-context.md).
- **Operator bug/gap report, recorded verbatim-intent, not yet investigated**: (1) no-water
  synthesized minimaps show shading glitches that normal ones don't (terrain shading suspected,
  unconfirmed); (2) "Include WMO geometry" minimap export checkbox does not work; (3) the "Bake
  MCSH shadows" minimap export checkbox must be removed permanently — MCSH is not encoded in real
  minimaps, so the option is misleading by construction, not just unused; (4) the toolbar's
  hovered-WMO doodad-set combo (Spec 231 T032) disappears as soon as the mouse leaves the render
  window, making it unusable; (5) cell-level fine-tune needs true 1-cell (not ~8-cell) X/Y
  granularity — current tooling could not align Hellfire Ramparts to its Expansion01 tiles closer
  than 1–3 cells off in one axis and 1–2 in the other; (6) numeric offset/transform counters are
  too small to show a signed two-digit value at 100% zoom and must scale with UI text size. Items
  1–3 and 5–6 filed as new Phase 7 tasks (T067–T071) in
  [232's tasks.md](../specs/232-cartography-composition-project/tasks.md); item 4 filed as Phase 8
  T080 in [231's tasks.md](../specs/231-editor-archaeology-ui-overhaul/tasks.md). **Operator also
  re-stated the top-priority gap directly**: "we still have no saving of the ADTs that we layer up
  in the Archeology option, which is the whole point of this whole toolkit" — this is exactly
  Spec 234 (current branch), still Draft/not-planned; speckit-plan is the next action once this
  cleanup pass closes.
- Cleanup report: [specs/224-speckit-governance/evidence/cleanup-2026-09-10.md](../specs/224-speckit-governance/evidence/cleanup-2026-09-10.md).

## 2026-09-09 — Spec 232 T064: magnetic WDL edge-snap for composed layers

- Operator directive: use the existing WDL magnetization mechanism to blend composed tile borders
  into the surrounding terrain (Teldrassil-on-Kalimdor case). Implemented as a per-layer
  `EdgeBlendWdl` strength (0 = off): the tile-boundary outer vertices of contributed heightmap
  chunks blend toward the base map's WDL 17×17 macro lattice via the new pure-math
  PhaseEdgeBlender (Core.Runtime, beside WdlLatticeMagnetizer). Edges shared with a neighbor
  target tile that also receives the layer's heights stay untouched — only footprint-boundary
  edges blend. Applied in both adapters' MergePhaseTile after the Z transform; host wiring feeds
  the parsed base WDL to the adapters (shared parse with the stratigraphy path); layer-card
  slider + project persistence. PhaseEdgeBlenderTests 4/4; Maps 181/181; build 0 errors.
  Receipt: [t064](../specs/232-cartography-composition-project/evidence/t064-wdl-edge-snap-receipt.md).
  Visual witness operator-owned.

## 2026-09-09 — Spec 232 T066: layer-rigid cell fine-tune

- Operator report: cell fine-tune "shifts the cells around instead of just moving the TILE", and
  with heightmap + rotation + cell offset composed checkerboard garbage (white plates, wrong
  elevations). Root cause: per-chunk supply-tile resolution (ResolveCellShiftedChunk +
  ResolveTileSource per chunk) composed inconsistently under rotation and pulled border chunks
  from unrelated donor tiles. Both adapters' `BuildCellShiftedTile` are now LAYER-RIGID: each
  target tile collects its own donor tile's content plus the 3×3-neighborhood spill (each
  contributor resolves its own donor through the shared tile map, rotation included; chunk
  (sx, sy) lands at (sx + cellDx + 16i, sy + cellDy + 16j); per-axis ranges disjoint at
  |offset| ≤ 15). Amended after the first tile-rigid pass dropped slid-out content ("missing
  stuff in between"). Maps: 177/177; build 0 errors. Receipt:
  [t066](../specs/232-cartography-composition-project/evidence/t066-tile-rigid-cell-shift-receipt.md).
  Visual witness operator-owned.

## 2026-09-09 — Spec 232 T056–T059: placement coordinates, layer Z, minimap interaction

- Operator directives landed: (T056) donor-tile picker now reads ADT-name order xx_yy — the old
  (row, col) interpretation landed every placement on the diagonal mirror of the requested tile;
  (T057) per-layer `ZOffset`/`ZScale` world-Z transform (terrain heights, liquids, placements)
  with UI + project persistence via the new owned service `PhaseLayerZ`; (T058) minimap surfaces
  now own their `MinimapInteractionState` — the shared instance made other surfaces consume click
  sequences so triple-click teleport never fired; (T059) placed-only layers drag on the minimap,
  moving each placement's target with the pointer and re-streaming on release. Maps: 177/177;
  solution build 0 errors; full-suite has 10 pre-existing failures in unrelated subsystems.
  Receipt: [t056-t059](../specs/232-cartography-composition-project/evidence/t056-t059-placement-z-minimap-receipt.md).
  Operator witnesses owed for all four; T064 (magnetic WDL snapping) and T065 (chunk off-by-one
  re-audit) recorded as follow-ups.

## 2026-09-09 — Spec 232 T015e MCAL alpha repair (MCLY regression on overlapped maps)

- Root-caused the operator's report that texture layers broke on overlapped maps after the T015c
  full-tile route: `AlphaTileData.ToTileLoadResult` sliced per-chunk MCAL alpha from the 256×256
  downsampled pack with a 64-px-per-chunk stride, so chunks past (3,3) decoded silent zero alpha
  and every transformed overlapped tile collapsed to a single flat texture. Repair: the reader now
  also carries the full-resolution 1024×1024 pack (`McalAlphaPackFull`), `ToTileLoadResult` slices
  from it with a 256-pack nearest-upsample fallback, and `RotateQuarterTurn` moves it with the same
  index map. `McalAlphaPack` (256²) semantics unchanged for the dataset contract. Focused Maps:
  174/174 passed; `dotnet build WowViewer.slnx -c Debug --no-restore`: 0 errors. **Visual MCLY
  witness remains operator-owned** (folds into the T015d seam screenshot). Receipt:
  [t015e-mcal-alpha-repair-receipt.md](../specs/232-cartography-composition-project/evidence/t015e-mcal-alpha-repair-receipt.md).
- Flagged (not fixed, no live caller): `TerrainTileTensorPack.ToTileLoadResult` has the same
  256-pack/64-stride mismatch.

## 2026-09-08 — Spec 233 renderer marketing-capture automation P1

- Created the full SpecKit contract (spec, plan, research, data model, task pack, JSON contracts,
  quickstart) and completed the first two source phases. The shared Runtime marketing model now
  validates versioned recipes/timed beats, rejects unsafe output traversal, produces a safe
  relative-path authoring descriptor, and keeps tour advancement allocation-free on steady frames.
  Viewer composition adds **Feature Tour + Video** beside **Play + Video**: it reuses Warm Path and
  raw framebuffer/ffmpeg capture, hides ordinary chrome only while recording, renders timed
  callouts in the with-UI capture tap, and restores prior chrome state afterward. Focused marketing
  tests: 12/12; Debug solution build: 0 errors.
- **Operator gate remains open**: this is not a recorded video, visual/UI timing, encoder playback,
  FPS, hitch, receipt, ComfyUI, MCP, or README-media witness. The next real action is T015 with
  `FlybyUndead` after the existing ffmpeg release-hardening gate. Receipts:
  [design](../specs/233-marketing-capture-automation/evidence/t001-design-receipt.md),
  [foundation](../specs/233-marketing-capture-automation/evidence/t003-t008-foundation-receipt.md),
  [P1 source](../specs/233-marketing-capture-automation/evidence/t009-t014-us1-source-receipt.md).

## 2026-09-08 — Spec 223 T609 video-capture release hardening

- Repaired the hidden developer-PATH dependency: Capture Automation now resolves an optional,
  operator-supplied `ffmpeg.exe` beside the viewer before configured/PATH fallback, validates
  `libx264` through **Verify ffmpeg**, normalizes quoted paths, and reports missing encoder or
  output-path errors safely. Build/publish copies `Capture/ffmpeg/win-x64/ffmpeg.exe` to the
  viewer root when the release operator supplies it; the repository neither provides that binary
  nor selects its licence. Focused resolver tests: 6/6 passed; full Debug solution build: 0 errors.
- **T609 remains unchecked**: actual with-UI/no-UI and camera-path video recording, output playback,
  binary provenance/licence notices, and a published-build witness are still operator-owned.
  Receipt: [t609-video-capture-release-hardening-2026-09-08.md](../specs/223-ui-consolidation-audit/evidence/t609-video-capture-release-hardening-2026-09-08.md).

## 2026-09-08 — Spec 232 T054 Archaeology Map Layers default

- No-page Archaeology entry now opens Cartography page 5, whose existing default sub-tab is Map
  Layers; explicit UniqueId/Range routes and remembered pages retain their existing authority.
  `dotnet build WowViewer.slnx -c Debug --no-restore`: 0 errors.
- **T054 remains unchecked** pending an operator UI witness for default entry and remembered
  explicit selection. Receipt:
  [t054-archaeology-map-layers-default-receipt.md](../specs/232-cartography-composition-project/evidence/t054-archaeology-map-layers-default-receipt.md).

## 2026-09-08 — Spec 232 T053 WL inspector fall-through repair

- Audited the suspected minimap route and repaired the actual viewport failure: minimap footprint
  hit-testing does not invoke WL selection, but the terrain-occlusion guard cleared the WL
  source-data hover as soon as a composed terrain layer lay in front of its bounds. WL hover is
  now retained for the click-inspector while ordinary placed-scene-object occlusion stays active.
  `dotnet build WowViewer.slnx -c Debug --no-restore`: 0 errors.
- **T053 remains unchecked** pending an operator witness that opens the same WL inspector with
  and without a placed phase layer. Receipt:
  [t053-wl-inspector-fallthrough-receipt.md](../specs/232-cartography-composition-project/evidence/t053-wl-inspector-fallthrough-receipt.md).

## 2026-09-08 — Spec 232 T050 placed-tiles-only composition

- Completed and receipted **T050 / FR-13**: the donor tile picker now creates an explicit
  donor-to-target placement in persisted `UsePlacedTilesOnly` mode, so a one-tile request cannot
  compose the donor map at all other offset targets. The per-layer `Compose placed tiles only`
  toggle restores the legacy whole-map offset route; minimap footprints follow the placed targets
  and placed-only layers cannot be offset-dragged misleadingly. Focused Maps: 37/37 passed;
  `dotnet build WowViewer.slnx -c Debug --no-restore`: 0 errors.
- **Next implementation**: T051 per-tile locks. **Separate operator gate**: T015d seam-free
  rotated DeadminesInstance + cell-nudge visual witness. Receipt:
  [t050-placed-tiles-only-receipt.md](../specs/232-cartography-composition-project/evidence/t050-placed-tiles-only-receipt.md).

## 2026-09-08 — Spec 232 T051 per-tile lock implementation

- Implemented T051's structural path: `Locked` now belongs to an explicit donor-to-target
  placement; both adapters prevent subsequent layers from composing that target, and the minimap
  resolves texture/footprints in the same order and draws an `L` owner badge. Layer cards expose
  a per-placement lock toggle, and project JSON round-trips the lock bit. Final focused Maps:
  39/39 passed; `dotnet build WowViewer.slnx -c Debug --no-restore`: 0 errors.
- **T051 remains unchecked** until the operator supplies a minimap capture that shows the badge
  and a later layer being excluded. Then audit T053's WL* click-inspector regression. Receipt:
  [t051-per-tile-lock-implementation-receipt.md](../specs/232-cartography-composition-project/evidence/t051-per-tile-lock-implementation-receipt.md).

## 2026-09-08 — Spec 232 T015 full-tile seam-repair route

- Restored the missing Spec Kit design set for [Spec 232](../specs/232-cartography-composition-project/spec.md):
  plan, research, data model, internal composition contract, and real-data operator quickstart.
- Completed and receipted **T015a–T015c**: `AlphaTileData.RotateQuarterTurn` moves full-tile
  channel lattices before MCNK slicing; the synthetic 257×257 test proves the content reaching
  target chunk (0,0) is exactly the source chunk selected by the established policy slot map.
  The Alpha adapter's direct and cell-shift transformed routes now consume that full-tile result,
  with an explicit typed skip if a donor cannot be read rather than a known-bad per-MCNK fallback.
  Focused Maps suite: 33/33 passed; `dotnet build WowViewer.slnx -c Debug`: 0 errors.
- **Next**: T015d's DeadminesInstance seam screenshot and cell-nudge witness remain operator-owned;
  no runtime or visual repair is claimed. Receipts:
  [t015a-t015b-core-lattice-receipt.md](../specs/232-cartography-composition-project/evidence/t015a-t015b-core-lattice-receipt.md),
  [t015c-alpha-adapter-full-tile-receipt.md](../specs/232-cartography-composition-project/evidence/t015c-alpha-adapter-full-tile-receipt.md).

## 2026-09-07 — Wireframe/selection fixes, export freeze fix, Spec 231 UI overhaul planned

- Wireframe root cause fixed across terrain and models: all wireframe passes drew textured
  lines identical to the fill beneath them (invisible on terrain; alpha-cutout orange
  fragments + silhouette-only lines on objects). Flat-color passes now in
  [TerrainRenderer.cs](../src/viewer/WoWViewer/Terrain/TerrainRenderer.cs) (semi-transparent
  white), [M2Renderer.cs](../src/viewer/WoWViewer/Rendering/M2Renderer.cs),
  [ModelRenderer.cs](../src/viewer/WoWViewer/Rendering/ModelRenderer.cs) via a color-override
  parameter on [IModelRenderer](../src/viewer/WoWViewer/Rendering/IModelRenderer.cs).
- Selection highlight is now the model's red wireframe instead of a bounding box
  ([WorldScene.cs](../src/viewer/Terrain/WorldScene.cs) selection block; box fallback only
  when the model is not streamed; WMO-doodad placement markers keep pins/axes).
- Editor toolbar: `Anim` toggle (world doodad animations default ON now) + hovered-WMO
  doodad-set combo; left sidebar layer/overlay wall collapsed by default; Imports & Exports
  groups collapsed by default (first de-congestion pass; structural fix is Spec 231).
- PM4 OBJ export freeze fixed: moved off the render thread with status + re-entrancy guard
  ([ExportPm4ObjectsObjSet](../src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs)).
- **Spec 231 authored per operator directive** (speckit; implementation deferred to a fresh
  session): [231-editor-archaeology-ui-overhaul/](../specs/231-editor-archaeology-ui-overhaul/spec.md)
  with plan (4-page Editor IA, Archaeology de-hosting, dedupe D1–D6, Spec 228 page-class
  pattern, phases P0–P5) and gated tasks. Registered in STATUS.md + UI epic.
- Receipts: `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj` 0 errors at each
  step. Visual/interactive acceptance operator-owned. See also the 2026-09-06 v0.5.3-rc1
  entry below for the version-plumbing + changelog work committed together.

## 2026-09-06 — Version plumbing fix + v0.5.3-rc1 bump

- Fixed the About-box/version drift: `eng/Version.props` was never imported (bare relative
  `Exists()` in the repo-level `Directory.Build.props` resolves against the *project* directory),
  and a shadowing `Directory.Build.props` in `src/viewer/WoWViewer/` blocked the repo-level file.
  Both now chain correctly; the viewer csproj's hardcoded `0.5.2.2` block was removed.
- Viewer title/About now derive from the assembly `InformationalVersion` (SDK `+<git-commit>`
  metadata trimmed for display) instead of a hardcoded const
  ([ViewerApp.cs](../src/viewer/WoWViewer/ViewerApp.cs)).
- Version bumped to `0.5.3-rc1` (assembly `0.5.3.0`) per operator direction; release notes
  rewritten as a real changelog of the ~160 commits since `v0.5.2.1` (5.0.1 era support, PM4
  semantics campaign, Rosetta, editor platform, converter fixes) at
  [docs/releases/v0.5.3-rc1.md](../docs/releases/v0.5.3-rc1.md). Receipt: `dotnet build` of both
  `WoWViewer.csproj` and `WoWViewer.CrossPlatform.csproj` 0 errors; both targets'
  `ParpToolsWoWViewer.dll` report `ProductVersion 0.5.3-rc1+<sha>`, `FileVersion 0.5.3.0`.
  The CrossPlatform csproj also hardcoded `0.5.2.2` (the binary the operator was running) and was
  de-hardcoded in the same pass. Tag push (`v0.5.3-rc1`) and GitHub Actions release run are
  operator-owned.

## 2026-09-06 — Context and documentation reduction pass

- Created [docs/README.md](../docs/README.md) as the canonical documentation router; legacy
  `DOCUMENTATION-STATUS.md` and `PLANS-OVERVIEW.md` are now redirects.
- Preserved, rather than deleted, high-confidence historical material under `docs/archive/`:
  the 49-file 2026 game-viewer plan pack, the consumed M2 investigation packet, the stale
  2026-08-01 spec audit, and the intact legacy MdxViewer tarball.
- Moved only clearly superseded specs 080, 145, and 195 to
  `specs/archived/superseded/` with successor pointers. They are not asserted complete.
- Added [Spec 228](../specs/228-source-decomposition/plan.md) planning artifacts. The first source
  extraction remains blocked by the Spec 227 T004 UI-authority gate.
- The full Spec 224 receipt audit is still open; this pass recorded its archive and routing work
  without checking its cleanup tasks or silently closing Gate 1.

## 2026-09-06 — Current implementation handoff

- Spec 227 T001/T002 source documentation is receipted. The next task is the operator-owned T003
  screenshot/input matrix, then T004 gate.
- Spec 223's fog/WMO/capture acceptance retest remains separately operator-owned.

Earlier same-day narrative was preserved in
[memory-bank/archive](archive/README.md), not discarded.
