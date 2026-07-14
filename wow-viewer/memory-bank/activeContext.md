# Active Context — wow-viewer

Last updated: 2026-07-14 (RunPod deployment ready; WoWViewer cross-platform CI investigation started)

## For tomorrow (pick up here)

1. **Spec 103 training**: everything is built and verified, nothing left to code. Run
   `specs/103-image-only-reconstruction/quickstart.md` **"Start here"** section (top of file):
   build the RunPod bundle → transfer → `install_deps.sh` → `verify_bundle.sh` → `smoke.sh` →
   `train.sh` on the pod. If output shows banding, try `OUTPUT_HEAD_MODE=linear_unclamped_train`.
2. **WoWViewer CI**: built and locally validated (see entry below) — `.github/workflows/
   wowviewer-build.yml` exists at the true repo root (`i:\parp\parp-tools\.github\`, NOT under
   `wow-viewer/` — this repo is `akspa0/parp-tools`, not a submodule, confirmed via `git
   rev-parse --show-toplevel` and `.gitmodules`). Push it and watch the first real run; nothing
   else to build. To cut a release: push a `v*` tag, or run the workflow manually with
   `publish_release: true`.
3. **WoWViewer real cross-platform viewer** (separate, larger, NOT started): swap `BlpFile.
   GetBitmap()` → `GetImage()` at ~19 call sites (listed in progress.md) so the existing
   `WoWViewer.CrossPlatform.csproj` becomes an actually-functional Linux/macOS viewer, not just
   a compiling one. Sizeable, touches core rendering files — scope with the user before starting.

## Current target — Spec 103: revive the v7 terrain regressor on clean signals

- **Governing law (image-only):** the only deployment input is one image tile. Every other signal is generated from it; no model reads a ground-truth signal at inference; downstream trains on generated (not ground-truth) upstream; a target the image cannot support is invalid. Validation is **label-free** (self-consistency), never label-comparison. See `specs/103-image-only-reconstruction/spec.md`.
- **Implemented (agent side, 2026-07-13):** v7 contract pinned in `specs/103-.../research-v7-contract.md`;
  lane ported to `data-harvester/src/harvester/spec103/` (`v7_model.py` — only deviation: `output_size`
  parameterized, 256 default; `v7_losses.py` verbatim; `v7_inputs.py` 13-ch assembler). 7/7 CPU sanity
  tests green (`tests/spec103/`). All Phase 2–4 scripts written: `spec103_make_synthetic_adts.py`,
  `spec103_build_synthetic_store.py`, `train_spec103_v7.py`, `infer_spec103_v7.py`,
  `spec103_build_real_store.py`, `spec103_export_mesh.py`, `validate_spec103_labelfree.py`.
  Commands: `specs/103-.../quickstart.md`. **Blocked on USER runs**: capture, training, T011 caveat catalog, T018 shadow capture.
- **Pinned 13-ch truth (plan's old aux guess was wrong):** 0-2 minimap, 3-5 normals (both recovery-attenuated
  ×0.85/×0.70 then ImageNet-normalized), 6 WDL prior (outer 17×17 only, align_corners=True, **0.5 fill when
  missing — dropout reuses this**), 7-8 tile height min/max hint planes (`--height-hints gt|wdl|none`),
  9 liquid mask, 10 liquid height, 11 object mask, 12 brush (zeros). Loss reads 9/11/12 — order is load-bearing.
  **The model architecture is unchanged (13 channels).**
- **WDL prior = verified transform:** `outer = height257[::16,::16]`, `inner = height257[8::16,8::16]`.
  Derived at batch time from `height_257` — no reharvest, nothing stored. **Never** `wdl_height_33`.
- **Procedural-synthetic PoC DROPPED as a gate (USER decision 2026-07-14):** flat/ramp/ridge/crater
  patterns don't replicate real terrain, and the WDL prior trivially solves them (v8 run: init
  l1_g ≈ 0.0006 — nothing to learn on the global channel). The intended "synthetic" lane was always
  **signals synthesized FROM real terrain** (deterministic shadow/hillshade renders of real height —
  T018's reinterpretation), never invented terrain. The 10-tile procedural store survives only as a
  pipeline smoke test. **Soundness test = the real-data v8 run** (quickstart §3; everything ready:
  V18 store + curation manifest, 2253 kept, Azeroth holdout 332/1921).
- **Synthetic chain (kept for smoke tests; all existing C# used as-is):** `map generate-blank` (Inspect tool) → known-height .npy →
  `terrain-patch-adt` (Converter) → `Capture render` (perspective-camera caveat recorded) or
  `--synthesize-minimaps` hillshade fallback. Synthetic tiles are placed non-adjacent so the patcher's seam
  stitching never mutates a known pattern.
- **Curation is mandatory (FR-013 / Principle #5), clean-by-default:** object tiles are impossible height
  targets (terrain under an object is occluded in the minimap), so they are DROPPED, not learned — the user
  was right and I initially defaulted keep-all in violation of the spec; fixed. `spec103_curate_dataset.py`
  buckets every tile and drops object_contaminated / blank_minimap / height_normal_mismatch, writes an
  auditable `curation_manifest.parquet` (+ map/height-regime buckets) the trainer consumes via
  `--curation-manifest`. **Default `--max-object-coverage 0.0`** (drop ANY object; was 0.02). V18 at 0.0:
  5134 → 2650 kept. `1.0` is v7-faithful keep-all ablation only. Trainer reports `val_no_prior` every epoch (prior-dropout robustness).
- **Banding investigation (2026-07-14):** verified live against V18 zarr — height_257/normal_xyz/
  liquid_height/object_precise_mask are all float32; only minimap_rgb is uint8 (correctly, the
  deployment image). No precise data is routed through 8-bit image encoding. Real causes found:
  (1) `output_head_mode` was never exposed to the trainer — every run silently hard-clamped a
  tanh-scaled residual every step (tanh saturation → residual clusters near ±scale = plausible
  v7 banding/terracing source); now `--output-head-mode {legacy_clamped, linear_unclamped_train}`,
  recorded in checkpoints, auto-resolved by inference. (2) v8's PixelShuffle upsampling lacked
  ICNR init (Aitken 2017) — a checkerboard-artifact class v7 never had (bilinear+conv instead);
  fixed + regression-tested. Left as-is (shared v7/v8, not a bug): the 17×17 WDL prior is only
  C0-continuous when bilinear-upsampled to 256×256 — visible ~16px facets the ±0.20 residual
  can't fully correct; watch for it in `val_previews/`. 15/15 tests green.
  Full writeup: `specs/103-image-only-reconstruction/research-v8-optimization.md` §6.
- **Local GPU training is OFF (2026-07-14):** USER's GPU overheated mid-run; **no more local
  training runs** — the path forward is RunPod deployment (see [[project_v24_runpod_migration]]
  for prior RunPod lessons: US datacenters only, runpodctl.exe location, verify before killing).
- **RunPod deployment built (2026-07-14, T022):** `scripts/package_spec103_runpod.py` +
  `runpod/spec103/{install_deps,verify_bundle,smoke,train}.sh`. Bundle subsets BOTH fields
  (only the 6 arrays `train_spec103_v7.py` reads, not the V18 store's other 18) AND rows
  (curation-kept only) — measured **3.2 GB -> 127 MB** (2253/5134 tiles), verified end-to-end
  through the real `V7TileDataset`. No HF downloads (v8/v7 train from scratch). Added `--limit`
  to the trainer for the smoke stage; `train.sh` always passes `--resume` (spot-preemption
  safe). Command: quickstart.md §5.
- **v8 is the PRIMARY architecture (USER decision 2026-07-13; implemented + tested):**
  `V8LeanUNet` (`src/harvester/spec103/v8_model.py`, ConvNeXt-V2 blocks, pixel-shuffle decoder,
  global-context mixer) — measured **6.2M params / 16.4 GFLOPs @256** vs v7's 117.06M / 119.9
  (73% of v7's params sat at 8×8–16×16). Identical 13-ch/trestle/bounds contract → loss, trainer,
  inference, previews, harness all unchanged. Trainer default `--arch v8` (`--arch v7` = 117M
  ablation only, NOT a gate); checkpoints record arch, inference auto-resolves. 13/13 CPU tests.
  Driver: v7's ~26 h time-to-signal was unacceptable; v8 targets minutes on synthetic. Survey +
  rationale: `specs/103-image-only-reconstruction/research-v8-optimization.md` (T021).
  Excluded: DA-family (blacklist), diffusion predictors, 100M+ depth foundations.
- **The USER runs all training/capture/heavy jobs.** The agent prepares scripts + commands only (AGENTS RULE 0).

## WoWViewer CI + cross-platform build (2026-07-14, new lane)

- **GitHub Actions added:** `.github/workflows/wowviewer-build.yml` (repo root — the actual
  GitHub repo is `akspa0/parp-tools`; `wow-viewer/` is a plain subdirectory, not a submodule).
  Three jobs: (1) build+test on `windows-latest` via `WowViewer.slnx` (the real, functional
  viewer — always runs); (2) compile-only check of `WoWViewer.CrossPlatform.csproj` +
  4 confirmed-portable tool projects on `ubuntu-latest` (`continue-on-error: true` — advisory,
  keeps the port from bit-rotting without gating on non-functional-yet code); (3) publish a
  self-contained win-x64 build + GitHub Release, gated on a `v*` tag push or manual
  `workflow_dispatch` with `publish_release: true` (never auto-triggered — matches
  [[feedback_no_auto_deploy]]). All three validated **locally** before commit: full solution
  build (0 errors), CrossPlatform target build (0 errors, 435 warnings — mostly the predicted
  CA1416 GDI+ hits), portable tools build clean.
- **Audit finding (Explore agent, full results in progress.md): the cross-platform port is
  further along than expected but NOT functional yet.** `WoWViewer.CrossPlatform.csproj`
  (plain `net10.0`, no WinForms) already existed, compiles cross-platform-clean at the TFM/
  dependency-graph level, and the three WinForms file-dialog calls in `ViewerApp.cs` were
  already correctly `#if WINDOWS`-guarded (SDK auto-defines `WINDOWS` only for `-windows`
  TFMs) with graceful `return null` fallback. **The real blocker: `BlpFile.GetBitmap()`**
  (`SereniaBLPLib`, System.Drawing/GDI+) is called at ~19 actual rendering/export sites
  (M2Renderer, WmoRenderer, TerrainRenderer, MinimapRenderer, GlbExporter, MapGlbExporter,
  LoadingScreen, AssetProbe, Core.Renderer's TextureCache, MDX-L_Tool's TextureService, plus
  harvest/converter/mask-validate tool code) and **throws `PlatformNotSupportedException` at
  runtime off-Windows since .NET 7** — compiles fine, crashes on first texture load. The fix
  (`BlpFile.GetImage()`, ImageSharp-based, already exists and is already used correctly in
  `BlpRgbReader.cs`/`AlphaBlpCompatibilityService.cs`) is scoped but NOT done — a real,
  sizeable follow-up task (~19 call sites in core rendering code), not started without user
  sign-off given the blast radius.
- **Fixed in this pass (small, unambiguous, verified compiling):** two hardcoded-backslash
  filesystem-path bugs that would break on Linux — `tools/harvest/.../Program.cs:398` and
  `tools/converter/.../LkToAlphaCommand.cs:1885`, both now `Path.Combine`. (Distinct from MPQ
  virtual-path strings elsewhere, which correctly and intentionally use `\` as the game-data
  convention — those were not touched.)
- **`WowViewer.Tool.ValidationCapture` is deliberately, permanently Windows-only by design**
  (throws `PlatformNotSupportedException` itself for its GPU hidden-window capture path) — not
  a portability bug, never expected to run on Linux.
- **Confirmed portable today** (tool-project level): `inspect` (`map generate-blank`),
  `wdl-read`, `enrich`, and `converter`'s `terrain-patch-adt` subcommand specifically (its
  other, minimap-related subcommands still hit `GetBitmap()`). `capture`, `harvest`, and
  `mask-validate` still have the GetBitmap runtime landmine.
- **`WowViewer.CrossPlatform.csproj`, `WmoMinimap`, `V22Enrich`, and `App.Defunct` are not in
  `WowViewer.slnx`** — deliberately left out of the solution file (CI builds them by direct
  csproj path instead) to avoid changing the user's local `dotnet build WowViewer.slnx` behavior
  without being asked.

## Dropped / paused

- **V24 / Spec 094 is NOT functional — dropped.** Do not revive it.
- **Spec 102 M0 object-mask lane is paused/superseded** by Spec 103. Preserved: simple M0 trainer
  (`train_spec102_m0_simple.py`) + inference; strict fragment-trace target + 42/42-green tests remain inactive.

## Boundaries

- New work in `wow-viewer/`; `gillijimproject_refactor` is read-only reference (port from, never edit).
- Staged clients only: `output/tmp/wowarchive-clients/`. Never `H:\CLIENTS`.
- Spec 080 owns the UI lane.
