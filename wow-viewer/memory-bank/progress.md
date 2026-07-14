# Progress — wow-viewer

Last updated: 2026-07-14 (WoWViewer GitHub Actions CI added; cross-platform audit complete)

## 2026-07-14 (evening) — WoWViewer GitHub Actions CI + cross-platform audit

- **Added `.github/workflows/wowviewer-build.yml`** at the true repo root (this repo is
  `akspa0/parp-tools`; `wow-viewer/` is a plain subdirectory, confirmed not a submodule via
  `.gitmodules` and `git rev-parse --show-toplevel`). Jobs: Windows build+test (real viewer,
  always runs on push/PR touching `wow-viewer/`), Linux compile-only check (advisory,
  `continue-on-error: true`), win-x64 release publish + GitHub Release (gated on `v*` tag or
  manual dispatch with `publish_release: true` — never auto-triggered). All three validated
  locally before commit (dotnet 10.0.301 present locally): full solution build 0 errors/365
  warnings; `WoWViewer.CrossPlatform.csproj` alone 0 errors/435 warnings; 4 portable tool
  projects build clean; full `dotnet test WowViewer.slnx` run to confirm nothing regressed.
- **Cross-platform audit (Explore agent, full findings), key results:**
  - `WoWViewer.CrossPlatform.csproj` (plain `net10.0`) already existed prior to this session,
    dependency graph is portable (no `-windows` TFM, no `UseWindowsForms`/`UseWPF` anywhere in
    the graph), and the three WinForms file-dialog calls in `ViewerApp.cs` were already
    correctly `#if WINDOWS`-guarded with `return null` fallback (SDK auto-defines `WINDOWS`
    only for `-windows`-suffixed TFMs) — someone had already done real groundwork here.
  - **The actual blocker: `BlpFile.GetBitmap()`** (SereniaBLPLib, `System.Drawing`/GDI+) is
    called at real rendering/export sites and throws `PlatformNotSupportedException` at
    runtime off-Windows since .NET 7 (compiles fine, crashes on first texture load — confirmed
    empirically, 435 CA1416 warnings on local build, concentrated in `VlmDatasetExporter.cs`).
    Cross-platform-safe `BlpFile.GetImage()` (ImageSharp) already exists in the same library
    and is already used correctly in `src/core/WowViewer.Core.IO/Blp/BlpRgbReader.cs:32` and
    `AlphaBlpCompatibilityService.cs:36`. Call sites still needing the swap (not done this
    pass — sizeable, core-rendering-code change, needs user scoping first):
    `AssetProbe.cs:961`, `Export/GlbExporter.cs:592`, `Export/MapGlbExporter.cs:279`,
    `Rendering/LoadingScreen.cs:274`, `Rendering/M2Renderer.cs:1206`,
    `Rendering/MinimapRenderer.cs:184`, `Rendering/ModelRenderer.cs:2730`,
    `Rendering/WmoRenderer.cs:1730`, `Terrain/TerrainRenderer.cs:1045,1212`,
    `Terrain/Vlm/VlmDatasetExporter.cs:4880`, `src/core/WowViewer.Core.Renderer/Texture/
    TextureCache.cs:102,169`, `libs/WoW-Tools/MDX-L_Tool/Services/TextureService.cs:131`,
    `tools/harvest/.../Program.cs:1706,1731`, `tools/converter/.../Program.cs:2930,2941`,
    `tools/mask-validate/.../Program.cs:202`.
  - `WowViewer.Tool.ValidationCapture` is deliberately, permanently Windows-only by design
    (`ValidationWorldSceneAdapter.cs` throws `PlatformNotSupportedException` itself for GPU
    capture via a hidden-window render host) — not a bug, documented constraint, never a
    Linux CI target.
  - Portable today (tool level): `inspect` (`map generate-blank`), `wdl-read`, `enrich`, and
    `converter`'s `terrain-patch-adt` subcommand specifically. `capture`, `harvest`,
    `mask-validate` still hit `GetBitmap()`.
  - No `Microsoft.Win32` usage anywhere in source. One other Windows-only P/Invoke
    (`WindowsNativeFileDialogs.cs:251`, `shell32.dll`) already correctly `OperatingSystem.
    IsWindows()`-guarded, only pulled in by `mask-validate`.
  - `WoWViewer.CrossPlatform.csproj`, `WmoMinimap`, `V22Enrich`, `App.Defunct` are not listed
    in `WowViewer.slnx` — CI builds them by direct csproj path; left the `.slnx` untouched to
    avoid changing the user's local `dotnet build WowViewer.slnx` behavior unasked.
  - No existing CI anywhere for this project before this pass (only vendored upstream libs
    under `libs/` have their own irrelevant `.github/`/`appveyor.yml`).
- **Fixed (small, verified compiling):** two hardcoded-backslash path bugs —
  `tools/harvest/WowViewer.Tool.Harvest/Program.cs:398` and `tools/converter/WowViewer.Tool.
  Converter/LkToAlphaCommand.cs:1885` — both real on-disk filesystem paths (not the MPQ
  virtual-path `\` convention used correctly elsewhere), now `Path.Combine`.
- **First real CI push found two more pre-existing repo bugs, both fixed (2026-07-14):**
  1. **`.gitignore` `maps/` (unanchored) was shadowing real C# source, not just data dirs.**
     Matched any directory named `maps` anywhere in the tree, not just at repo root — silently
     hid `wow-viewer/src/core/WowViewer.Core/Maps/` and `.../WowViewer.Core.IO/Maps/`. 9 source
     files were never committed (invisible to any fresh clone, always present locally since the
     files exist on disk regardless of git tracking). Anchored `runs/`, `datasets/`, `publish/`,
     `maps/` with a leading `/`; recovered all 9 files; verified no other `.cs` anywhere in
     `src/`/`tools/`/`tests/`/`libs/` is similarly shadowed.
  2. **6 vendored libs under `wow-viewer/libs/` were orphaned git submodule gitlinks with no
     `.gitmodules` entry** (`Marlamin/WoWTools.Minimaps`, `ModernWoWTools/Warcraft.NET`,
     `WoW-Tools/SereniaBLPLib`, `wowdev/DBCD`, `wowdev/WoWDBDefs`, `wowdev/wow-listfile`) —
     each had a real nested `.git` clone locally (never lost), but no upstream URL was recorded
     anywhere, so every fresh clone (every CI run) got a completely empty folder for all 6.
     **USER decision: convert to real submodules (option they explicitly chose over flattening
     to plain files), updated to each upstream's latest commit** — "should not cause a rift."
     Before updating, checked each repo for local un-pushed commits first (a naive
     force-reset to origin/master would have silently destroyed them): `WoW-Tools/SereniaBLPLib`
     and `wowdev/DBCD` both carry a local, user-authored "Disable central package version
     management" commit (works around the `Directory.Packages.props` central-versioning
     conflict with SereniaBLPLib's own per-TFM ImageSharp pin) — preserved via rebase for DBCD
     (21 commits behind → rebased clean), left as-is for SereniaBLPLib (its `master` was
     *behind* the locally-patched commit, not ahead — nothing to gain from resetting).
     `WoWDBDefs` (+116 commits) and `wow-listfile` (+158 commits) fast-forwarded cleanly, no
     local divergence. `Marlamin/WoWTools.Minimaps` and `ModernWoWTools/Warcraft.NET` were
     already at their upstream tip. Full solution rebuild after updating: 0 errors (confirmed
     "no rift" empirically, not just assumed). CI workflow updated: all 3 jobs now run
     `git submodule update --init --depth 1 -- <the 6 paths>` after checkout — deliberately
     NOT `submodules: true`, which would also pull unrelated, much larger submodules elsewhere
     in the repo (`gillijimproject_refactor`'s Depth-Anything-3, `PM4Tool/lib/*`, `dirac`,
     `headroom`).
  Both bugs were invisible from `git status` inspection alone and had persisted for a long
  time — proving the exact value of standing up real CI, first-run-ever, at the top of this
  same session.
- **Third CI attempt failed differently: `fatal: remote error: upload-pack: not our ref
  0bb9dac...`** — SereniaBLPLib and DBCD both carried a user-authored "Disable central package
  version management" commit that was **never pushed to the actual GitHub remote**, discovered
  when preserving it during the submodule fix above. A submodule can only ever pin a commit
  that exists on its own remote; CI (or any fresh clone) can never fetch a local-only commit.
  **USER, on hearing this: confirmed vendored libs are never supposed to be directly patched**
  (same policy as `gillijimproject_refactor`'s read-only boundary) and approved discarding both
  commits. Root cause of why the patch existed: 5 `ProjectReference`s to `SereniaBLPLib.csproj`/
  `DBCD.csproj`/`DBCD.IO.csproj` were missing `GlobalPropertiesToRemove="ManagePackageVersionsCentrally"`
  (2 genuinely missing on SereniaBLPLib refs in `WowViewer.Tool.Harvest.csproj` and
  `WowViewer.Tool.WmoMinimap.csproj`; DBCD had zero refs using it) — added to all 5, but proved
  **insufficient alone**: it only affects MSBuild's build-time ProjectReference graph walk, not
  solution-wide `dotnet restore`/`dotnet build WowViewer.slnx`'s separate restore-graph
  evaluation, which still hit NU1008 (central package management forbids explicit `Version` on
  a `PackageReference`, which both vendored csprojs declare). Tried and abandoned: a
  `wow-viewer/libs/Directory.Build.props`/`.targets` ancestor override (DBCD ships its own
  nearer `Directory.Build.props`/`.targets` from upstream, which wins and stops the auto-import
  walk before reaching an ancestor file; even for SereniaBLPLib, which has no such nearer file
  and *did* show the property correctly overridden via a standalone `-getProperty` check,
  solution-wide restore still ignored it — likely global-property propagation from the
  solution-level evaluation, not fully root-caused). **What actually worked**: a
  path-conditioned `PropertyGroup` inside `wow-viewer/Directory.Packages.props` itself (the
  file NuGet's CPM detection is keyed on) — `Condition="$(MSBuildProjectFullPath.Contains(...))"`
  matching `libs/WoW-Tools/SereniaBLPLib` and `libs/wowdev/DBCD`, setting
  `ManagePackageVersionsCentrally=false` for just those paths. Verified: full clean-build
  (all `obj/` cleared first) 0 errors; `WoWViewer.CrossPlatform.csproj` 0 errors; both edited
  tool csprojs 0 errors. Both submodules reset to genuinely fetchable pristine upstream commits
  (`SereniaBLPLib` → `origin/master` 2323219; `DBCD` → rebased-tip-minus-patch 9ca6553) —
  vendored libs are pristine again, matching policy.
- **Same `wow-viewer/libs/*` gitignore rule was ALSO hiding a second, much bigger problem:**
  `libs/WoW-Tools/{Warcraft.NET, MDX-L_Tool, WoWMapConverter.Core, WoWRollback,
  GillijimProject}` were **completely untracked** (0 files each, not a partial gap like the
  `maps/` bug). `WoW-Tools/Warcraft.NET` was a *second*, entirely separate nested-git clone of
  the exact same upstream (`ModernWoWTools/Warcraft.NET.git`) already wired up at a different
  path — and it was the one `WoWViewer.csproj`/`CrossPlatform.csproj` actually referenced (the
  submodule fixed earlier was only used by that library's own tests/docs). **USER decision:
  point everything at one copy; MDX-L_Tool and WoWRollback are obsolete now that wow-viewer is
  self-contained.** Verified before deleting: zero `MdxLTool` namespace usage anywhere in
  `wow-viewer/src` (vestigial — `WoWViewer.csproj` referenced it but nothing used it;
  functionality already natively ported to `Terrain/Transfer/M2ToMdxConverter.cs`); the only
  `WoWRollback` mention in `wow-viewer/src` is a dead `throw new NotSupportedException(...)`
  string; `WoWMapConverter.Core` (287MB, of which 271MB was `bin/` build-output bloat under
  `WoWRollback.PM4Module`, not source) is referenced by nothing in the real solution and itself
  depends on WoWRollback. Repointed `WoWViewer.csproj`/`CrossPlatform.csproj`'s Warcraft.NET
  `ProjectReference` to the `ModernWoWTools` copy, dropped the `MDX-L_Tool` reference entirely,
  deleted all 4 untracked dirs (zero git history lost — none were ever tracked), rebuilt clean
  (0 errors). Properly tracked `GillijimProject` (895K, plain source, genuinely needed, no
  nested `.git`) by narrowing the gitignore rule: `wow-viewer/libs/*` now has an explicit
  `!wow-viewer/libs/WoW-Tools` / `wow-viewer/libs/WoW-Tools/*` /
  `!wow-viewer/libs/WoW-Tools/GillijimProject` allow-list instead of blanket-hiding everything.
  `ManagePackageVersionsCentrally` opt-out condition in `Directory.Packages.props` extended to
  cover the now-active `ModernWoWTools/Warcraft.NET` path too (same NU1008 pattern). Full clean
  rebuild (all `obj/` cleared) + `WoWViewer.CrossPlatform.csproj` standalone: both 0 errors.
- **Local `dotnet test WowViewer.slnx` run surfaced ~20 pre-existing failures, unrelated to
  this session's edits** (confirmed: the two touched files aren't referenced by the failing
  test projects). All failures are in `*RealData*`/`*Corpus*`-named tests
  (`AdtMcrfRealDataTests`, `M2Era1121HeaderDumpTests`, `Pm4RegionObjectGrouperTests`, and more —
  full list not yet catalogued) that read a staged WoW client via `AdtRealDataTestCatalog.
  GetStagedClients()`, which returns `[]` cleanly when `output/tmp/wowarchive-clients` is
  absent (the expected CI state — staged clients are local-only per AGENTS.md). This machine
  has a *partial* corpus (some clients staged, specific files missing), which is a different
  failure mode than *total absence* — whether these same tests skip cleanly on a truly clean
  CI checkout is unknown until a real run happens. **Kept `dotnet test` as a real, ungated gate
  in CI** (no `continue-on-error`) rather than guessing at a filter: a permission-system check
  correctly caught that `continue-on-error` on this step would let `publish-release` (which
  `needs` this job) ship a build even with real test failures — reverted. If CI goes red here
  on real-data tests, the correct fix is a proper skip-if-corpus-absent guard in those ~20
  tests, not a CI bypass.

## 2026-07-14 (later) — Banding investigation + RunPod deployment (T022)

- **Banding investigation:** verified live against V18 zarr that no precise data (height, WDL
  prior, normals) is routed through 8-bit image encoding — only `minimap_rgb` is uint8,
  correctly. Found and fixed two real causes instead: `output_head_mode` was never exposed to
  the trainer (every run silently hard-clamped a tanh-scaled residual every step — a plausible
  v7 banding/terracing mechanism); now `--output-head-mode {legacy_clamped,
  linear_unclamped_train}`, recorded in checkpoints, auto-resolved on inference. v8's
  PixelShuffle decoder lacked ICNR init (Aitken 2017) — a checkerboard-artifact class v7 never
  had; fixed + regression-tested. Left as-is (shared v7/v8, documented trestle design, not a
  bug): the 17×17 WDL prior is only C0-continuous when upsampled to 256×256 — visible ~16px
  facets. 15/15 tests green. Full writeup: `research-v8-optimization.md` §6.
- **RunPod deployment (T022):** local GPU overheated mid-run — training moves to RunPod.
  `scripts/package_spec103_runpod.py` subsets the V18 store to the 6 fields
  `train_spec103_v7.py` actually reads AND to curation-kept rows only: **measured 3.2 GB ->
  127 MB bundle** (138 MB tar), 2253/5134 tiles. Verified end-to-end (not just "should work"):
  opened the bundled zarr + manifest and ran the real `V7TileDataset` against it, confirming
  finite (13,256,256) inputs. `runpod/spec103/{install_deps,verify_bundle,smoke,train}.sh`
  follow the existing V24 bundle pattern; no HF downloads needed (v8/v7 train from scratch,
  unlike V24's DA-V2+LoRA). Added `--limit` to the trainer for the smoke stage. `train.sh`
  always passes `--resume` for spot-preemption safety. Command: quickstart.md §5.

## 2026-07-13 (late) — v8 lean architecture implemented; primary lane by USER decision

## 2026-07-14 — Procedural synthetic dropped as a gate; real data is the proving ground

- **USER decision:** procedural patterns (flat/ramp/ridge/crater/plateau) don't replicate real
  terrain and the WDL prior trivially solves them (v8 smoke run: l1_global ≈ 0.0006 at init and
  at best — the metric is prior-dominated, not learning). The intended synthetic lane =
  **synthesize signals from real terrain** (deterministic shadow/hillshade of real height, T018),
  not invented terrain. Real-data v8 run (quickstart §3) is now the soundness test; ready to run
  (curation manifest 2253 kept, Azeroth 332-tile holdout).
- **Trainer hardening from the smoke run:** batch clamped to train-set size; `drop_last` only
  when ≥2 full batches (tiny sets no longer silently produce 0 train batches); hard exit on an
  empty train loader; loud warning when planned steps are too few for `--ema-decay` (the
  validated EMA model would otherwise stay ~= its initial weights). 13/13 tests green.

## 2026-07-13 (late) — v8 lean architecture implemented; primary lane by USER decision

- **Why:** v7's 117.06M-param U-Net (73% of params at 8×8–16×16; 119.9 GFLOPs @256) meant ~26 h
  before a training run proved sound or not. USER decision: modern lean arch is primary, no
  baseline-first gatekeeping; v7 kept for ablation only.
- **What:** [`v8_model.py`](wow-viewer/data-harvester/src/harvester/spec103/v8_model.py)
  `V8LeanUNet` (`v8-lean-convnextv2-v1`): ConvNeXt-V2 blocks (7×7 reflect DW + GRN), widths
  32-64-128-256-384, pixel-shuffle decoder, pooled global-context mixer + bounds head.
  **Measured 6,204,198 params (25 MB) / 16.4 GFLOPs @256** — 18.9× / 7.3× less than v7. Head,
  trestle residual, clamp modes copied verbatim; the 13-ch contract, `combined_loss`, trainer,
  inference, previews, mesh export, and label-free harness run unchanged.
- **Wiring:** trainer `--arch v8|v7` (v8 default), arch recorded in checkpoints + run identity;
  `infer_spec103_v7.py` auto-resolves arch (pre-v8 checkpoints default to v7). Tests: 6 new v8
  CPU sanity tests incl. a <10M-param budget guard; 13/13 spec103 suite green. Docs synced
  (plan, tasks T021, quickstart, research-v8-optimization.md = survey + decision record).

## 2026-07-14 — Curation default tightened (drop ANY object tile)

- **Curation default tightened:** `--max-object-coverage` default is now `0.0` (drop ANY object) in both
  [`spec103_curate_dataset.py`](wow-viewer/data-harvester/scripts/spec103_curate_dataset.py:59) and
  [`train_spec103_v7.py`](wow-viewer/data-harvester/scripts/train_spec103_v7.py:198). Was 0.02.
  The model architecture is **unchanged** (13 channels) — this is a tile *selection* change only, not an
  architecture change. Object tiles are impossible height targets (spec Principle #5: height under an
  object is occluded in the minimap), so they are dropped, not learned.
- **Tests:** 7/7 CPU sanity green. Docs synced (research-v7-contract, plan, quickstart, spec FR-013, tasks).

## 2026-07-13 (evening) — Spec 103 Phases 0–4 agent work implemented

- **Contract pinned** (`specs/103-image-only-reconstruction/research-v7-contract.md`): real v7 aux
  channels 7-12 are height-min/max hints, liquid mask, liquid height, object mask, brush — the plan's
  alpha/holes guess was wrong and is corrected in plan.md. Missing/dropped WDL prior = 0.5 fill (v7's own
  fallback). Resolution decision: 256, `output_size` parameterized (the port's only deviation).
- **Lane ported + tested:** `src/harvester/spec103/{v7_model,v7_losses,v7_inputs}.py`; 7/7 CPU sanity
  tests (`tests/spec103/test_v7_sanity.py`): channel order, trestle residual, prior dropout, targets/bounds,
  forward/loss/backward, world-unit round trip.
- **Scripts prepared (USER runs the GPU/dotnet steps — quickstart.md):** synthetic known-height author
  (flat/ramp/ridge/crater/plateau, non-adjacent tiles; prints exact `map generate-blank` +
  `terrain-patch-adt` + `Capture render` commands) → 13-ch store builder (captured PNGs or labeled
  hillshade fallback) → lean trainer (holdout by any index column, AMP/EMA/warmup+cosine/early-stop/resume,
  `--wdl-prior-dropout` with per-epoch `val_no_prior`, `--height-hints gt|wdl|none`, `--loss v7|l1`,
  `--max-object-coverage` clean-tile selection, FR-011 run identity + peak VRAM) → batch inference
  (predicted height_257 npy + paired WDL lattice npz, `terrain-patch-adt`-compatible) → OBJ export →
  label-free harness (border agreement, plausibility, checkerboard/blockiness; `--gt-store` dev-only baselines).
- **Speckit synced same pass:** plan.md (pinned channel table, loss/object decisions, Phase 5 scoped
  deferred lanes T016/T019, implementation state), tasks.md (T001-T010, T012-T017, T019 checked;
  T011/T018 + training runs USER-blocked), quickstart.md new.

## 2026-07-13 — Pivot to Spec 103 (revive v7); image-only law established

- **New governing law** in Spec 103: input is one image; every signal is generated from it; validation is
  label-free. **V24 / Spec 094 dropped** as non-functional. `wdl_height_33` prohibited; the WDL prior is the
  verified `height257[::16]` / `[8::16]` transform. **Spec 102 M0 paused/superseded** but preserved
  (simple trainer + 42/42-green strict tests).

## Key facts for the next session

- Next step is entirely USER runs: quickstart §1 (synthetic authoring → dotnet generate/patch/capture →
  store → training), then T011 caveat catalog in research-v7-contract.md §8, then real-data run (§3).
- v7 reference (read-only): `gillijimproject_refactor/src/WoWMapConverter/scripts/{v7_model,train_v7,v7_losses,infer_v7}.py`.
- Real store = existing V18 `output/datasets/v18/3_3_5_12340.zarr` (5134 tiles; has minimap_rgb, height_257,
  normal_xyz, liquid_mask/height, object_precise_mask — FR-012 satisfied, no copy needed;
  `spec103_build_real_store.py` verifies and pins it).

## Durable boundaries

- `gillijimproject_refactor` read-only (port from, never edit). C# WDL reader + AlphaWdtWriter frozen.
- The USER runs all training/capture/heavy jobs (AGENTS RULE 0). Staged clients only; never `H:\CLIENTS`.
- Older M0 strict-target detail: `memory-bank/archive/2026-07-13-spec102-strict-target-detail.md`.
