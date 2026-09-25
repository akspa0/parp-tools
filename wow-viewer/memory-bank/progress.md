# Progress — wow-viewer

Last updated: 2026-09-25

## 2026-09-25 — U-01 ViewerApp campaign: 42,973 → 2,512 lines (E3 + T010–T012, code only)

Operator direction: "make ViewerApp as concise as possible". 54 behaviour-preserving steps moved every
cohesive feature cluster out of the `ViewerApp` partial class into 51 owned services + 3 static helpers under
`src/viewer/WoWViewer/Workbench/Services/` (menu bar, converter/export dialogs, loaders, data-source session,
settings, shell layout, hover/pick, PM4 workbench, archaeology, editor/workbench/inspector panels, capture,
camera paths, …). One `IViewerAppHost` (414 members, `ref` properties for ImGui `ref` fields) implemented in
`ViewerApp_Host.cs`; services built in the `ViewerApp()` constructor. `ViewerApp.cs` 16,746 → 1,725; 24
partial files removed. Every step: line-multiset audit, full-solution build, commit only on 0 errors;
end-to-end audit shows only boilerplate lines changed. Tests: same 26 environmental failures. Warnings:
+24 CS0618 (obsolete `ShellPanelId` uses now reported outside its declaring type), −4 CS0169 (reporting
change bisected to `4f3f722`, source identical). Found: terrain-analysis UI is unreferenced (dead).
Runtime not claimed — smoke is U01-T007/T013. Receipt:
`specs/251-epic-viewer-ux-and-code-health/evidence/u01-viewerapp-extraction-2026-09-25.md`.

## 2026-09-25 — U-01 E1: PM4 overlay extracted from `WorldScene` (code only)

Epic 251 U01-T002. 387 members + 42 top-level PM4 types moved verbatim (Roslyn member map +
reference closure) into `src/viewer/WoWViewer/Terrain/Pm4/`: `Pm4OverlayScene` (3 partial files:
state/load/cache, selection, reports), five static helper classes, model files, and
`IPm4OverlayHost` (13 members, implemented explicitly by `WorldScene`). `WorldScene.cs`
17,175 → 8,326 lines; every new file < 2,000. Callers use `WorldScene.Pm4Overlay.X` (278 receiver-only
lines in 7 `ViewerApp*` files). Solution build 0 errors, no new warnings in touched files; test
failure set identical to `HEAD` (26 environmental). Runtime not claimed — U01-T003 smoke is operator-owned.
Receipt: `specs/251-epic-viewer-ux-and-code-health/evidence/u01-e1-pm4-extraction-2026-09-25.md`.

## 2026-09-23 — Spec reconciliation: 137 specs audited against code, archived, replaced by 7 epics

Operator-directed, branch `v0.6.0-dev`. Twelve read-only audit batches checked every open spec against
source, tests, CLI verbs and receipts (checkboxes were not evidence). Result: 16 complete, 104 folded,
13 superseded, 4 cold. All 137 spec directories and the three `epic-*` folders moved with `git mv` to
`specs/archived/`, each stamped with its disposition and successor. Open residue now lives in Epics
248–254 (formats · renderer · reconstruction/editor · UX/code health · world/audio · PM4 · datasets/ML).
Nothing is scheduled until the operator triages `specs/TRIAGE.md` (Want / Drop / Later).

Measured gaps between belief and code (now in TRIAGE §1): no map save pipeline exists; editor undo
reverses only 2 of 5+ operation kinds; zone music is disabled by policy and misreads `ZoneMusic`;
fog does not bound streaming; `WorldAssetManager` still falls back to M2→MDX; `WorldScene.cs` 17,153
and `ViewerApp.cs` 16,746 lines; data-paths env-var overrides were never implemented.

Memory bank corrected against a fact check (techContext, systemPatterns, projectbrief, coding_standards,
data-paths); workstream notes now name their owning epic; entries before 2026-09-15 moved to
`archive/2026-09-23-progress-pre-2026-09-15.md`.
Receipt: [reconciliation README](../specs/archived/reconciliation-2026-09-23/README.md).

## 2026-09-21 — v0.6.0-alpha2 released

- `InformationalVersion` → `0.6.0-alpha2`; notes at `wow-viewer/docs/releases/v0.6.0-alpha2.md`.
- Commit `d4228e87` on `v0.5.4-dev`, tag `v0.6.0-alpha2` pushed. Release workflow run **35566694458
  succeeded** (5/5 jobs, 4 binaries attached, prerelease published).
- Only `Version.props` + the notes were staged; `imgui.ini` and the untracked `SereniaBLPLib` working
  copy were left alone (§5 worktree safety). Operator had already committed the day's code as
  `3fdd5719` + `cbb36ec0`.
- `git exit 128` annotations on the build jobs are non-blocking (they appear on passing jobs); worth a
  look sometime, likely submodule/`git describe` during build.
- **Ships unverified**: M2 wrap fix (all eras), LkAdtWriter chunk fixes (17 call sites), DAT export
  never loaded in a client, DAT layer never composed on screen, new buttons never clicked.

## 2026-09-20 — One export button, format by checkbox; DAT items moved to a File submenu

- **Operator**: one sidebar button with checkbox-settable LK/Alpha output, and put the DAT items in a
  File submenu.
- **`MapExportFormats`** (new owned settings service, static — no ViewerApp fields, §10) holds the LK v18
  / Alpha 0.5.3 selection and draws the checkboxes. The File menu and the sidebar call the same
  `DrawCheckboxes`, so the two surfaces cannot drift.
- **File menu**: four peer DAT entries collapsed into one **"DAT Terrain (v22/23/26)"** submenu — open,
  format checkboxes, export, DAT v26 export, height scale.
- **Sidebar**: "Export as ‹formats›..." under World Overview, shown only for DAT terrain, disabled with
  a reason when no format is ticked.
- **Alpha target implemented by reuse**, not a second converter: DAT → `LkAdtData` →
  `LkToAlphaConverter.ConvertTile` → `AlphaWdtWriter`. Tiles convert **lazily through the writer's
  provider** because each `AlphaTileData` allocates ~20 MB (16 MB alpha pack + 4 MB shadow); holding all
  699 v26 tiles would be ~14 GB.
- **Verified**: `--format lk+alpha` on the v22 corpus writes 4 ADTs + LK WDT + a 1.3 MB Alpha WDT.
  Alpha walk: `MVER MPHD MAIN MDNM MONM` + 4×(`MHDR MCIN MTEX MDDF MODF`) + 1024 `MCNK`, closes exactly
  on EOF. CLI gained `--format lk|alpha|lk+alpha`.
- **NOT clicked** — the menu and sidebar are unexercised; verification is via the CLI path they share.

## 2026-09-20 — LkAdtWriter emitted incomplete ADTs (affects ALL LK output, 17 call sites)

- **Operator**: the written LK ADTs are missing chunks the format requires, subchunks too. Correct, and
  **not** DAT-specific — `LkAdtWriter` is shared by `AlphaToLkConverter`, `SplitAdtToLkCommand`,
  `RosettaTilesetGenerator`, `NewMapCreatorService`, viewer map save and more.
- **Six defects**: (1) `MCSE` never written; (2) **`ofsMCCV` (+0x74) never set — MCCV was written into
  the file where nothing could find it**, 179,200 orphaned chunks in the v26 export; (3) MCNK flag
  `0x40` never set; (4) `ofsMCLV` (+0x78) never set; (5) `MHDR.ofsMFBO` hardcoded 0 while MFBO was
  written, and flag `0x1` never set; (6) `MTXF` never written, `MHDR.ofsMTXF` hardcoded 0.
  `mccvOffset`/`mclvOffset` were computed into locals and discarded.
- **Nearly broke a working thing**: reader and writer looked 8 bytes apart on MCNK sub-chunk offsets.
  Measured a real ADT first — `ofsMCVT = 136`, counted from the MCNK tag — and **both were already
  correct**. No change made there.
- **Verified**: re-exported v22 + v23 and walked the output checking each header offset resolves to a
  chunk whose tag matches the field (`MCVT/MCNR/MCLY/MCRF/MCAL/MCSH/MCSE/MCCV`). **Zero misaddressed
  offsets**, walk closes on EOF, `MTXF` and `MCSE`×256 now present. LK regression tests 83 passed /
  1 failed, the failure being the pre-existing alpha-quantisation drift baselined earlier today.
- **NOT loaded in a client or Noggit** — structural validity is not loading. Operator proof outstanding.
- Receipt: `specs/archived/247-dat-capture-and-adt-export/evidence/lk-writer-missing-chunks-2026-09-20.md`.

## 2026-09-20 — v22 objects loaded a random wrong model each run: a data race

- **Operator**: "it loads a random different model every time, implying that an index is not being
  used right." Right in substance, wrong index.
- **Decode ruled out**: `ACDO.ModelIndex` is a **per-tile** index into that tile's `ADOO` list; all
  four v22 files are in range, and the lists genuinely differ per tile (`ADOO[1]` =
  `TerokkarTreeStump` in one, `TerokkarBush01` in another). All three call sites resolve per-tile
  correctly. `adt-ahdr objects --list` is correct and deterministic.
- **Root cause**: `AhdrTerrainAdapter` has no WDT, so it builds the shared model-name table **lazily
  during tile loads**, and exposed it as a live `List<string>`. Tiles load on the **ThreadPool, 4
  concurrently**; `WorldScene.OnTileLoaded` indexes that list from the main thread. A reader can see
  the new `Count` against the old backing array mid-`Add` → wrong name, different every run. WDT-backed
  adapters do not show it because their tables are populated before streaming.
- **Fix**: publish immutable `volatile string[]` snapshots, swapped in under `_placementLock`
  **before** the new index is returned, so no index can outrun the array a reader holds.
- **NOT confirmed fixed visually** — needs an operator run; a race is never proven absent by one clean
  run. Next suspect if it persists: `WorldAssetManager.NormalizeKey` collisions / MDX cache.
- **Latent, untouched**: `_placedUniqueIds` is never cleared, so a tile re-entering after eviction
  does not re-add its placements to the adapter-wide list `BuildInstances` iterates.
- Receipt: `specs/archived/237-adt-v26-terrain/evidence/v22-random-model-race-2026-09-20.md`.

## 2026-09-20 — DAT->LK export gets a UI entry point (it had none)

- **Operator**: "how does the exporter work? there's nowhere to push to export!" — correct, US3 shipped
  CLI-only. **File > Export Loaded DAT as LK ADT...** added, enabled only when a DAT folder is loaded,
  asks for an output directory, reports on the status line. Picker opened inline so **no new ViewerApp
  field** (§10).
- **Shared, not duplicated**: the folder walk + WDT write + manifest moved into
  `DatToLkAdtFolderExporter` (core). The CLI now delegates and **lost 4,615 chars** of duplicate code;
  the viewer did not gain a second copy. v22 re-export is byte-for-byte identical in counts
  (999 chunks, 289 MCSH, 997 area ids, 849 objects), so the refactor is behaviour-preserving.
- **Two manifest defects fixed**: `ADST`/`AOCH` were being dropped silently (violating 247
  FR-013/SC-004), and count-bearing notes de-duplicated badly, producing ~15 near-identical lines per
  manifest. Counts are now report counters emitted once.

## 2026-09-20 — Modern write support: measured state + v26 completeness

- **Operator**: "we should be writing modern ADT's but we have no support for any modern chunks or
  writers". **Confirmed**, precisely: `MapConversionTargetFormat.MopSplitAdt` is declared with a
  display name, command value and parser, but `HasWriter => false`, `LkAdtWriter.EnsureTargetFormat`
  throws for it, and it has **zero consumers**. A prepared socket, nothing plugged in — honest (no
  mislabelled output) but empty. `MopAdtChunkParser` has exactly **one** method, `ParseMtxpChunk`.
- **`AdtRawChunkBlobCollector` already captures every unparsed chunk verbatim**, but its 5 consumers
  are all ML/dataset; **no writer takes raw blobs**. So unknown chunks survive into training data and
  are discarded from files. A passthrough writer is the cheap half-step to a modern writer.
- **Counterpoint, MEASURED**: LK v18 is **not** lossy for DAT v26. `ALYR` per `ACNK` over all 179,200
  v26 chunks maxes at **4** — exactly LK's limit, 0 chunks with 5+. Heights/normals/MCCV/layers/alpha/
  placements all map. Only `ADST` (321 rows, positionless by construction) and `AOCH` (2048 B,
  all-zero) have no LK equivalent. The modern-writer gap is real but does not bite the 247 export lane.
- **Defect fixed in the same pass**: the export manifest silently dropped `ADST` and `AOCH`, violating
  247 FR-013/SC-004 ("no silent drops"). Now named, with counts. Also fixed a note-dedup bug that was
  emitting 15 near-duplicate running-count lines per manifest.
- **Asymmetry is the real finding**: the project reads modern and writes only legacy, so no
  modern→modern workflow is possible. Sequencing proposed (not started): 245's chunk inventory → raw
  passthrough → native split writer behind the existing socket.
- Receipt: `specs/archived/245-modern-chunk-completeness-survey/evidence/modern-write-support-state-2026-09-20.md`.

## 2026-09-20 — ACDO negative uniqueIds MEASURED: v26 runs two allocators

- **Operator question**: "v26 has objects with negative uniqueID's, not sure if that's right?" — it is.
- **MEASURED** over all 5,309 v26 `ACDO`: **two dense sequential allocators**, 4,096 positive
  (63,418,942..63,423,379, 92.3% dense) and **1,213 negative** (-1,233..-2, 98.5% dense), **0
  duplicates** across all 5,309. Four tiles contain both; no split by file, record size or model kind.
  Counting down from -2 is the shape of a locally allocated, never-committed id.
- **v22 has none** (849 ids, 681,015..728,199); the v23 sample has no `ACDO` at all. v26-only so far.
- Retires the unmeasured 2026-09-19 suspicion, and corrects its prediction that a negative would show
  as a large unsigned value — `AhdrTerrainAdapter` casts to `int`, so the Inspector's `-210` is right.
- **Export consequence**: LK `MDDF` uniqueId is unsigned, so negatives become ~4.29 billion. Manifest
  now reports the count (1,213 on the v26 run). Pass-through vs remap vs drop is an operator call.
- **Also**: v26 corpus exported end to end — 699 tiles, 179,200 chunks, 8,457 MCAL, 0 area ids, 0
  shadows, matching 237's independent measurements. All three revisions now export.
- Receipt: `specs/archived/237-adt-v26-terrain/evidence/acdo-negative-uniqueids-2026-09-20.md`.

## 2026-09-20 (later still) — Spec 247 US5: DAT Folders as Cartography Layers

- **Finding**: a Cartography layer identified its donor only by `PhaseLayerSettings.MapName`, resolved
  **inside the base adapter's own data source** (`LoadMapTile`/`OverlayTileExists`/`GetOccupiedTiles`
  on Standard; a sibling typed adapter on Alpha). A DAT folder has no map name and no WDT, so no seam
  existed. The data shape already matched — `AhdrTerrainAdapter.LoadTileWithPlacements` returns the
  same `TileLoadResult` composition consumes.
- **Design**: `MapName` also accepts `dat:<absolute folder>`; new `DatLayerSource` owns parsing plus a
  per-folder adapter cache. No core model change, so Cartography project persistence and layer cloning
  keep working. Every branch is a guard at the top of a method, so non-DAT layers are untouched.
- **Also**: `DatLayerHeightDivisor` (36) puts donor inches in the base map's yards; "Add DAT folder..."
  button opens the picker inline, adding **no ViewerApp members** (§10).
- **Why it matters beyond convenience**: Cartography's offset/rotation/mirror controls are the way to
  settle the DAT axis question that forced `--transpose` on the US3 exporter — align once by eye and
  the alignment is the answer.
- **Verification**: slnx 0 errors; Ahdr+DatToLk+DatLayer tests 31/31 (10 new). **NOT witnessed on
  screen** — no DAT layer has been composed over a base map; alignment, textures and project-file
  round-trip all unverified. `StandardTerrainAdapter` only; Alpha needs its own change. Receipt:
  `specs/archived/247-dat-capture-and-adt-export/evidence/us5-dat-as-cartography-layer-2026-09-20.md`.

## 2026-09-20 (later) — Spec 247: DAT → LK v18 ADT Export DELIVERED

- **New**: `DatToLkAdtConverter` (core) + `adt-ahdr export-lk` CLI. Converts AHDR-family DAT tiles to
  LK v18 ADT + WDT with a CARRIED/DROPPED manifest. Reuses `LkAdtWriter`/`LkWdtWriter`/
  `AdtAhdrTileSlicer`/`AdtAhdrAlpha` — **no new format writer**.
- **Real runs**: v22 Expansion01 → 4 tiles, 999/999 chunks, 289 MCSH, 997 area ids, 849 MDDF/MODF.
  v23 IcecrownCitadel → 3 tiles, **48 MCAL** (196,608 B = 48×4096, alpha path proven), 768 MCCV.
  Independent chunk walk: 256 MCNK per file, 0 unaccounted bytes.
- **Chunks addressed by ACNK index, never ordinal** — v22 omits empties (25 empty MCNK synthesized).
- **v22 alpha still dropped** (1386 layers): the `AMAP` codec is unidentified. Upper layers are
  deliberately omitted rather than emitted opaque, which would hide layer 0.
- **AMAP codec attempt 1 failed but is now scoreable**: eliminated MCAL RLE (137/1373), a 32-variant
  grid, pure `(count,value)` pairs (589 odd-length payloads), zlib. **Oracle confirmed**: `ACNK` +0x12
  2-bit 8×8 predominant-layer map, 0 violations across 997 v22 chunks.
- **v22/v23 carry area ids and v22 carries live shadows** — corrects spec 241's v26-only table.
- **Verification**: slnx build 0 errors; Ahdr+DatToLk tests 21/21. **Not loaded in a client/viewer** —
  operator proof outstanding; `--transpose` exists if axes come out wrong. Receipt:
  `specs/archived/247-dat-capture-and-adt-export/evidence/us3-dat-to-lk-adt-2026-09-20.md`.

## 2026-09-20 — First DAT v22 Ever Loaded + Path-Picker Load Blocker

- **Load blocker (all pickers)**: `ImGuiPathPicker.ResolveSelection` returned `_currentDirectory` and
  ignored `_pathInputBuffer`, so a pasted path confirmed without Enter/Go was silently discarded and
  the process working directory was used instead. New `TryCommitPathBar()` resolves it on confirm or
  errors. Affected every picker (CASC, exports, overlays), not just DAT.
- **First v22 in project history**: `E:\WC2\wrat2\world\maps\Expansion01` — 4 files, MVER/AHDR 22,
  Terokkar / Bone Wastes tileset art. They **render** (operator screenshot: 4 tiles, 999 chunks,
  119 FPS). Also confirmed real **v23** at `.../IcecrownCitadel` (3 files).
- **v22 is structurally distinct**: no `ACVT`, no `AFBO`, **omits empty `ACNK`** (243-255, not 256),
  carries `ASHD` + `ACDO`. `ACNK` header is 0x40 on all three revisions. `ALYR` flag `0x100` gates
  `AMAP` on v22 too (2382/2383).
- **Open defect**: v22 `AMAP` is **encoded** (128-3474 B, never 4096) and the encoding is
  **unidentified** — v18 MCAL RLE refuted (137/1385 = chance). `AhdrTerrainAdapter:178` requires every
  layer to carry a 4096-byte map, so v22 drops all alpha and **renders layer 0 only**. Not fixed.
- **UI**: reader-side labels renamed `DAT v26` → `DAT (v22/23/26)`; export labels keep v26 (the writer
  emits only v26). New `TryReadVersion` reports per-file revision in the info panel.
- **Verification**: slnx build 0 errors; AHDR tests 15/15; full suite failures all pre-existing
  (verified against a stashed clean tree). Receipts:
  `specs/archived/237-adt-v26-terrain/evidence/first-v22-dat-render-2026-09-20.md` and
  `.../real-v23-dat-icecrown-2026-09-20.md`.

## 2026-09-19 — DAT GLB Export + Historical Data Record

- **GLB export for DAT folders**: `MapGlbExporter` now takes a nullable `IDataSource?` (placements/
  textures skipped when null; terrain mesh still exports), and the GLB menu is enabled for
  `_renderer != null || _terrainManager != null` (was gated on a standalone-model flag + a data
  source, so it was disabled for DAT folders).
- **Historical record**: new harvest command `dump-dat` writes a JSON record of every AHDR-family DAT
  file (header, tile location, textures, models, per-chunk layer/object counts, ADST refs,
  diagnostics). Produced `output/dat-records/kalimdor-lost-isles.json` (4 records).
- **Finding**: the Kalimdor files are **Lost Isles (`expansion03`)** terrain — `area_51_31` (310
  layers) and `area_51_32` (614 layers), textures `expansion03\lostisles\li_*` + `Tileset\Generic\*`.
- **Verification**: viewer + harvest builds 0 errors; dump ran on the real files. Receipt:
  `specs/archived/237-adt-v26-terrain/evidence/dat-glb-export-and-record-2026-09-19.md`.
- **Still open**: synthesized minimap for DAT files (needs an AHDR→`TerrainTileTensorPack` builder +
  a DAT-folder input mode in the harvest).

## 2026-09-19 — DAT v22/v23 Loading via Filename Tile Location

- **Operator request**: load any version of the DAT files; Wrath v23 files at
  `E:\WC2\wrath\World\Maps\Kalimdor` (a deleted Lost Isles pre-alpha region of Kalimdor).
- **Analysis**: the files are **DAT v23** (`MVER` 23), AHDR-family with the **same vocabulary as v26**
  (AHDR 129×129/16×16, AVTX 132100, ANRM 99075, ATEX, ACNK×256 with one ALYR + AMAP 4096) but **no
  ALOC chunk**.
- **Root cause**: `AhdrTerrainAdapter` required ALOC to place a file, so every v22/v23 file was
  skipped. The reader itself is version-agnostic.
- **Fix**: new `AdtAhdrReader.TryParseTileLocationFromName` (trailing `XX_YY` → X, Y) and an adapter
  fallback when ALOC is absent; v26 keeps its ALOC.
- **Verification**: viewer build 0 errors; `AdtAhdr` tests 13/13 (new `AdtAhdrV23Tests`). Receipt:
  `specs/archived/237-adt-v26-terrain/evidence/v22-v23-filename-tile-location-2026-09-19.md`.
- **Recorded (not a task)**: v26 objects can carry a negative uniqueID (operator suspects
  non-shipping/untracked objects); `UniqueId` is read as `uint`.

## 2026-09-18 — Spec 243 Modern-to-Legacy Map Conversion: Plan Authored

- Ran `speckit-plan` for [Spec 243](../specs/archived/243-modern-to-legacy-map-conversion/spec.md) (the map
  converter's modern→legacy lane, previously spec-only). Pointed `.specify/feature.json` at 243 and
  ran `setup-plan.ps1`; authored the full design set:
  - `plan.md` — technical context, constitution check (all PASS), real source layout, 6-phase
    breakdown (research → design → core service → assets → surfaces → validation).
  - `research.md` — 8 decisions: modern layer stack from the existing readers; **coverage-ranked
    layer merge** (keep base + top capacity-1, fold dropped into nearest kept, deterministic
    tie-break); area-average alpha downsample; FileDataID→path resolution with unresolved reported;
    reuse `LkAdtWriter`/`AlphaWdtWriter`; **one owned `ModernToLegacyMapConversionService`** surfaced
    in CLI + Editor; determinism; route validation via `MapConversionFormats`.
  - `data-model.md` — SourceMap, LayerStack, MergePolicy, MergeRecord, ConversionRun, AssetManifest.
  - `contracts/` — service API, CLI `convert-map` contract, merge-report JSON schema.
  - `quickstart.md` — operator commands for both targets + batch + asset inclusion.
- Registered in `STATUS.md` row 17 and the activeContext "Implement next" table (row 3). Next:
  `speckit-tasks`, then implement Phase 2 (core service).

## 2026-09-18 — Synthesized-Minimap DXT1 + MCCV Options

- **DXT1 option**: exposed the harvest tool's existing `--no-dxt1` in the viewer's synthesized-minimap
  export dialog ("Apply DXT1 compression", default on) so later-era outputs can skip the codec floor.
- **MCCV option**: added `ApplyMccv` to `TerrainMinimapLighting` + a `ResolveMccvTint` helper in
  `TerrainMinimapCompositor` (multiplies albedo by `clamp(mccv*2,0,2)`, matching the terrain shader),
  a harvest `--mccv` flag, and an "Apply MCCV vertex colors" dialog checkbox (default off).
- **Verification**: viewer + harvest builds both exit 0, 0 errors. Receipt:
  `specs/archived/111-minimap-lighting-calibration/evidence/synthesized-minimap-dxt1-mccv-options-2026-09-18.md`.
- **Still open**: liquids render with grid lines/omissions in synthesized minimaps — root cause not
  established; needs a zoomed capture + exact map/era.

## 2026-09-18 — Generated-Map MCNR Byte-Order Fix (synthesized-minimap shadow side)

- **Operator report**: synthesized minimaps from New Map Creator output show the terrain shadow on
  the north-west flank instead of the south-east.
- **Root cause**: `TemplatedTerrainGenerator` wrote MCNR components in `(X, Y, Z)` byte order, but the
  disk order is signed `(X, Z, Y)` (`BlankAdtFactory.CreateUpNormals`, `AlphaTerrainAdapter.DecodeNormal`,
  `WorldTerrainTileBuilder.TryReadMcnrNormals`). The up component landed in the horizontal Y axis and
  the Y slope in Z, so every generated slope shaded on the wrong side. Both `RecalculateChunkNormals`
  and `GenerateFlatNormals` were wrong.
- **Fix**: write `(X, Z, Y)` in both functions.
- **Verification**: `dotnet test WowViewer.Core.Tests --filter "FullyQualifiedName~TemplatedTerrainGenerator"`:
  7 passed, 0 failed, including new regression test
  `GenerateMap_WritesMcnrInDiskXzyOrderSoTerrainNormalsPointUp`. Receipt:
  `specs/archived/192-terrain-template-brush-generator/evidence/mcnr-byte-order-fix-2026-09-18.md`.
- **Still open (same operator report)**: (1) liquids render with grid lines/omissions in synthesized
  minimaps; (2) option to skip DXT1 compression on outputs; (3) option to include MCCV vertex colors.

## 2026-09-18 — Spec 236 Phase 2: Terrain Light Casting + Ambient/Sun Contract

- Landed the terrain half of Spec 236 Phase 2 multi-surface light casting:
  - **T010**: `SceneLightManager` gained `SceneAmbientLight` (direction/light color/ambient color),
    `Ambient`, `SetAmbient(...)` (finite-guarded), and `Clear()` now resets ambient; `WorldScene`
    publishes it from the active `TerrainLighting` profile in `RebuildSceneLights`.
  - **T012**: both `TerrainRenderer` shader programs (legacy chunk + batched tile) now declare
    `uLocalLightCount`/`uLocalLightPos[8]`/`uLocalLightColor[8]`/`uLocalLightIntensity[8]`/
    `uLocalLightStart[8]`/`uLocalLightEnd[8]`, carry `vWorldNormal`, and accumulate a bounded
    per-fragment point-light diffuse term. New `UploadLocalLights` + `LocalLightUniforms` select the
    nearest lights via `SceneLightManager.QueryAffecting(chunk/tile bounds)` per draw.
  - **T013 (terrain half)**: `TerrainManager.Render` accepts an optional `SceneLightManager` and
    `WorldScene` forwards `_sceneLightManager` into the terrain pass.
  - **FR-007 doodad/model external-light consumer** (same day): optional `SceneLightManager?` added to
    `IModelRenderer.BeginBatch`/`RenderWithTransform`; `MdxRenderer.UploadMdxLights(matrix, sceneLights)`
    selects nearest manager lights (omni) by transformed model AABB (manager already contains the
    model's own omni lights, so no double-counting); `WorldScene` passes `_sceneLightManager` into the
    unbatched/state-hoisted/transparent doodad passes and the WMO doodad-batch fallback; `WmoRenderer`
    threads `sceneLights` into WMO-internal doodad draws; `M2Renderer` forwards to legacy. Deliberate
    boundary: GPU-**instanced** opaque doodad batches (one light set for many placements) and native
    non-legacy M2 stay base-lit — per-placement routing is the next step, shared with Spec 242.
- **Still open**: T013 (instanced-opaque + native-M2 boundary) and operator visual Gate 2.
- Verification: `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug`: exit 0,
  0 errors; focused `dotnet test WowViewer.Core.Tests --filter "TerrainLighting|WorldObjectPassCoordinator|M2Runtime"`:
  47 passed, 0 failed. Receipts: `specs/archived/236-scene-lighting-doodad-performance/evidence/phase2-light-casting.md`,
  `specs/archived/236-scene-lighting-doodad-performance/evidence/phase2-doodad-light-consumer.md`.

## 2026-09-18 — v0.6.0-alpha Release Prep + Modern-Data WMO Performance Finding

- **Release prep (docs + version)**:
  - `eng/Version.props`: `0.6.0` / `0.6.0.0` / `InformationalVersion 0.6.0-alpha` (was `0.5.4` / `0.5.4-dev`). The viewer title bar and About box read this value, so the tag `v0.6.0-alpha` and the UI now agree.
  - New release notes `docs/releases/v0.6.0-alpha.md`; `CHANGELOG.md` entry; `README.md` (root + wow-viewer) version refs, feature bullets and era matrix; `docs/WoWViewer/README.md`, `USERGUIDE.md` §12–13, `docs/CLI-TOOLS.md` version attributions.
  - Corrected stale UI text in `ViewerApp_CascAhdr.cs` (DAT v26 folder status claimed objects/vertex colours were missing; both decode now).
  - Verification: `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug` exit 0, 0 errors; operator screenshot confirms the title bar reads `WoWViewer v0.6.0-alpha`. Release commit `c247f383` and tag `v0.6.0-alpha` created locally; push left to the operator.
- **Performance finding (operator-reported, `wow_classic_beta` 1.60.1 `Azeroth`)**:
  - Measured on the running v0.6.0-alpha build: ~5.5 FPS, ~230 ms uncapped frame, `WMO draw calls 16,431`, WMO render pass ≈5,493 ms of 7,716 ms.
  - Root cause candidate: `WorldScene.cs:11410` gates WMO shell GPU instancing on the **global** `_sceneLightManager.Count == 0`. Modern data always has emitted lights, so every WMO placement takes the per-instance fallback path and instancing is effectively always off.
  - The **per-placement** test already exists: `WmoRenderer.cs:1982-1983` transforms the placement AABB and calls `SceneLightManager.QueryAffecting(worldMin, worldMax, …)`, which returns 0 when no light's attenuation sphere touches the placement.
  - Bounded fix proposal (not yet implemented): gate batching on "no light affecting **this** placement" instead of "no lights in the scene".
  - Operator decision (2026-09-18): ship v0.6.0-alpha as-is; tracked as [Spec 242](../specs/archived/242-wmo-instancing-performance/spec.md) (registered in `STATUS.md` row 16, v0.6 scope table, and Epic 3).
- **New spec — 243 Modern-to-Legacy Map Conversion (operator-directed, prioritised)**:
  - One-way **modern → legacy**: outputs **LK v18 ADT/WDT** and **Alpha 0.5.3 monolithic WDT**; no old→modern writers ("too early", no real engine to consume them).
  - Core requirement: merge the multi-layer modern chunk (up to 8 layers + `AMAP` weights) into the target's layer model, combining texture ids and alpha masks, with a per-tile report of what was merged/dropped/unresolved.
  - Batch many maps in one run with per-map failure isolation; near-zero-touch UI (direction + target + input only); optional referenced-asset inclusion with a manifest.
  - New dir `specs/archived/243-modern-to-legacy-map-conversion/` (`spec.md` + `checklists/requirements.md`); registered in `STATUS.md` row 17 / v0.6 scope table and Epic 4.
- **Specs 244 + 245 opened (operator-directed, modern-data gaps)**:
  - Grounded the "new water directional flow" claim against [`wowdev.wiki/WDT`](https://wowdev.wiki/WDT) rather than guessing: the modern WDT **`MAI2`** chunk (≥ `12.0.5.66330`) is `MapFileDataIDs2[64*64]` at 32 bytes/entry, and its first field is `liquidFlowTexture` — documented as "R channel = +Y flows west, +G = −X flows south, 128 is 0 flow"; the other seven fields are `unknown1..unknown7`. This is the same `MAI2` the v0.6.0-alpha notes list as uninterpreted.
  - **Spec 244** (`specs/archived/244-modern-liquid-flow/`): decode `MAI2` + resolve the flow texture through the FileDataID/CASC path, decode to a normalized vector with 128 = zero, surface it as liquid **context in the viewer UI**, publish one shared flow datum, and report per-target legacy disposition (Alpha MCLQ already has a flow vector — `MclqChunk.MclqFlowVector`; LK `MH2O` has none → dropped). Flow-aware *rendering* out of scope.
  - **Spec 245** (`specs/archived/245-modern-chunk-completeness-survey/`): inventory every chunk in the modern WDT/`_occ`/`_lgt`, root ADT, `_tex0`, `_obj0`, `_lod` families with counts, current handling, code reference, documented meaning + confidence, and a disposition; then, per candidate, state **legacy build-in feasibility** for LK v18 and Alpha 0.5.3 including alpha-mask and texture-id re-expression, with the loss stated. Research deliverable, no runtime change; feeds 243/244.
  - Registered both in `STATUS.md` (rows 18/19, v0.6 scope table) and Epic 4 (members + Next).
- **Spec 246 opened (operator-directed): modern M2 camera paths + modern-data benchmarking**:
  - Operator: "we gotta fix it so we can load the m2's from the modern wow assets as camera paths, too. I'd like to be able to benchmark the renderer with the new client data like we do for older client data."
  - Evidence gathered: `M2ModelReader.ReadCameras` already has a **modern** branch (`CameraStrideModern = 0x74`, field-of-view track) gated on `version >= CataVersionThreshold`; but the standalone camera-path load (`ViewerApp.TryLoadStandaloneCameraPathM2`) reads via that reader and gates on `M2CameraPathOverlayBuilder.CanBuild` (`CameraCount > 0`), while modern WoW: Forever models arrive as chunked **`MD21`** and go through the chunked/conversion path. The loss point is therefore *unestablished* — spec FR-002 forbids fixing an assumed cause.
  - Benchmark evidence: frame-time mean/p99 + hitches live in `WorldRenderFrameHistory`; the marketing-tour flow already produces a path-driven video + receipt for legacy data; `inspect casc bench` only benchmarks **CASC data reads**, not renderer frames. So modern client data has no renderer benchmark today.
  - Spec 246 (`specs/archived/246-modern-m2-camera-paths-and-benchmarking/`): US1 modern camera-path import (same document/overlay as legacy, coordinate space resolved or reported); US2 path-driven modern benchmark with a receipt of build/map/path/warmup/frames/mean+p99/hitches/counters; US3 same-shape receipts across eras. Registered in `STATUS.md` row 20 / v0.6 scope table (order 8) and Epic 3 (members + Next); it is the measurement vehicle for 242.
- **Git**: `c247f383` (release docs + version bump, tag `v0.6.0-alpha`), `8bce10fe` (spec 243 + register 242/243), `f595531c` (specs 244/245 + register), `c4eb3c2f` (spec 246 + register). Push left to the operator — it triggers `wowviewer-release.yml` and the GitHub prerelease.
- **Fresh-chat readiness**: `activeContext.md` now carries a single **"Implement next — five v0.6 lanes"** table (order, scope, next bounded action, proof owner) for 246 → 242 → 243 → 244 → 245, and the older governance/defect sections were compressed. None of the five is implemented; all are spec-only, so a fresh chat can start any one with `speckit-plan`.

## 2026-09-16 — Spec 236 Phase 2 WMO Emitted-Light Casting Slice

- **Implemented**:
  - Added viewer-layer `SceneLight`, `SceneLightManager`, and `ISceneLightEmitter` for source-side emitted-light collection without changing the base model renderer interface.
  - Exposed MDX omni `LITE`, legacy-backed M2 `LITE`, native M2 animated omni lights, WMO `MOLT`, and WMO-internal doodad lights as scene lights.
  - Wired `WorldScene` to rebuild scene lights from visible placements and feed WMO shell draws.
  - Updated `WmoRenderer` shell shader to upload/evaluate up to eight nearby point lights with bounded attenuation and per-fragment diffuse.
  - Disabled WMO shell instancing while active scene lights exist so lit WMO placements do not share an approximate light set.
- **Verification**:
  - `dotnet build wow-viewer/WowViewer.slnx -c Debug`: exit 0, 0 errors.
  - `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~WorldObjectPassCoordinator"`: 11 passed, 0 failed.
  - Full solution `dotnet test --no-build` still reports existing non-lighting failures (`LkToAlphaRoundTripTests` drift and old `WorldFramePassCoordinatorTests` WDL-default expectation when targeted with object-pass tests); no runtime visual proof claimed.
- **Receipt**: `specs/archived/236-scene-lighting-doodad-performance/evidence/phase2-wmo-light-casting-slice.md`.

## 2026-09-15 — v0.5.3 Remediation: Phase Tile Alignment, Authentic Map Creator Assets, GLB Exporters, and Startup Stability

- **Defects Remediated**:
  - **Phase Maps & Donor Tiles Misalignment / Floating Doodads**:
    - In `StandardTerrainAdapter.cs` and `AlphaTerrainAdapter.cs`, raw donor tiles loaded via `LoadMapTile` retained donor coordinates (e.g. tile 45, 30). In `PhaseChunkMerger.cs`, when `TakeChannel(PhaseDataChannel.Heightmap)` was true, `PhaseChunkMerger` copied that raw world position, tearing the terrain 8 tiles away into the distance while doodads remained at base ground level. Added `RehomeChunksForTarget` to re-home donor chunks directly onto the target tile slot `(tileX, tileY)`.
    - In `BuildCellShiftedTile`, calculated exact donor-to-target tile delta `tileDx = -(tileX - source.SourceTileX) * tileSpan` and adjusted chunk coordinates accordingly.
    - Updated `TranslatePhasePlacements` to support `(phase, sourceTileX, sourceTileY, tileX, tileY, layer)` using `tileDx = -(tileX - sourceTileX) * tileSpan` + `cellDx, cellDy`, marking `PlacementsPreTransformed = true`.
    - Wrapped `LoadMapTile` with `try/finally` tracking `_currentLoadingMapName`, gating `MddfPlacements.Add` and `ModfPlacements.Add` so donor map placements do not pollute the base adapter's placement collection.
  - **Authentic Assets in New Map Creator**:
    - Replaced all fictitious hardcoded asset paths in `BiomePalette.ForTheme` (`whitemarble`, `elwynngrass.blp`, etc.) with authentic paths verified against cached listfiles: Elwynn grass/cobble/dirt, Stormwind cobblestone, Dragonblight snow, Alterac dirt/grass, and Barrens/Ashenvale terrain tilesets.
    - Replaced fictitious doodad model paths in `TemplatedTerrainGenerator.cs` (`HumanFountain.mdx`) with verified listfile models (`StormwindFountain01.m2`, `StormwindStreetlamp01.m2`, `StormWindBench01.m2`, `ElwynnFirTree01.m2`).
    - Added comprehensive unit tests in `TemplatedTerrainGeneratorTests.cs` validating all themes and model paths.
  - **Broken GLB Scene & Collision Mesh Exports in Editor Data I/O**:
    - In `ViewerApp.cs`, `_wantExportGlb` and `_wantExportGlbCollision` checked `if (_loadedFilePath != null)`, which was null when a map was loaded. Added branches for `else if (_terrainManager != null && _dataSource != null)` to export the active camera tile GLB via `MapGlbExporter.ExportTile` with clear status reporting.
    - In `MapGlbExporter.cs`, updated `TryLoadMdxMesh` to detect M2 files (`WarcraftNetM2Adapter.IsMd20` / `IsMd21`), load skin candidates, and convert via `M2ToMdxConverter` before passing to `MdxFile.Load`, enabling doodads to export into GLB scenes without throwing.
    - In `ViewerApp.cs`, added M2 model conversion in `_wantExportGlb` when loose models are loaded.
  - **Console Window Behind GUI**:
    - In `WoWViewer.csproj`, switched `<OutputType>Exe</OutputType>` to `<OutputType>WinExe</OutputType>` to suppress the unwanted background console window on Windows.
  - **Fresh Folder Crash on Game Client Load**:
    - In `Program.cs`, registered global `AppDomain.CurrentDomain.UnhandledException` and `TaskScheduler.UnobservedTaskException` handlers writing to `crash.log` in both the current directory and `%LOCALAPPDATA%\WoWViewer\crash.log`.
    - In `ListfileDownloader.cs`, updated `GetListfilePath` to return cached file immediately, fetch in background without blocking UI thread, or bound wait to 2.5 seconds max on fresh install before gracefully falling back.
- **Verification**:
  - `dotnet build WowViewer.slnx -c Debug`: 0 errors.
  - `dotnet test` (`TemplatedTerrainGeneratorTests`): 5 passed, 0 failed.
  - `dotnet test` (Core terrain & phase tests): 246 passed.
