# Quickstart: Skybox Rendering Validation

**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Date**: 2026-08-18

**Rendering proof here is automated.** The viewer has startup capture automation and there is a
headless production-scene profiler; both are driven directly. Nothing below blocks on someone
looking at a screen.

`$Root` is the configured client root — supplied at runtime, never hardcoded into source or
committed config (Constitution VI). Record it with every result.

---

## Build and unit tests

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

Focused first, per AGENTS.md:

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug `
  --filter "FullyQualifiedName~Sky"
```

---

## The two automation gaps (Phase 1, T001–T002)

These are closed before anything else, because without them the day-cycle and frame-cost tests
cannot be produced at all.

**T001 — `--time-of-day` pin.** No flag currently exists to fix the world clock, so US1's time-scrub
test and US2's day sweep have no reproducible input. Added in
`src/viewer/WoWViewer/ViewerApp_StartupAutomation.cs` alongside the existing `--capture-*` flags.

**T002 — camera motion in `profile-render`.** `ProductionWorldSceneProfiler.cs:92` resolves the
camera position **once** and reuses it for every sampled frame (lines 101, 110). A static window's
p99 and max are not valid evidence — this is the known blind spot that has produced false null
results in this project before. Add per-frame motion and emit `CameraMovedDuringWindow` in the JSON.

---

## Frame-cost baseline (Phase 1, T005–T008)

**Blocks all sky code changes** — unrecoverable afterwards.

```powershell
$Root = "<configured client root>"
$Out  = "I:/parp/parp-tools/wow-viewer/output/spec160"
New-Item -ItemType Directory -Force $Out | Out-Null

dotnet run --project I:/parp/parp-tools/wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/WowViewer.Tool.ValidationCapture.csproj -c Debug -- `
  profile-render `
  --client-root $Root `
  --map-input "World\Maps\Azeroth\Azeroth.wdt" `
  --output "$Out/baseline-dense.json" `
  --frames 120 `
  --warmup-frames 16
```

Repeat with a sparse map into `baseline-sparse.json` so the budget is not fitted to one scene.

**Gate**: both reports show `CameraMovedDuringWindow: true` and carry `Sky` / `SkyboxBackdrop`
p50/p99/max. Then fill [contracts/frame-budget.md](./contracts/frame-budget.md) — it has explicit
`_(to fill)_` fields so an unfilled budget is visible rather than assumed.

---

## Capture a sky frame

The general form every visual test below uses:

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug -- `
  --game-path $Root `
  --world "World\Maps\Azeroth\Azeroth.wdt" `
  --time-of-day 0.5 `
  --capture-shot current `
  --capture-no-ui `
  --capture-after-frames 60 `
  --capture-output "$Out/shots" `
  --exit-after-capture
```

`--capture-no-ui` excludes ImGui so the diff is scene-only. `--capture-after-frames` lets terrain and
assets stream in before the shot; raise it if a capture shows unloaded scenery.

---

## US1 — Authored colours reach the screen (T024–T026)

The differential test. "The sky looks right" proves nothing; a pixel delta does.

1. Capture the sky. Modify an authored sky colour in client data. Capture again.
2. **Pass**: non-zero delta in the sky region.
3. **Detector proof**: run the same pair against the **pre-change** build and assert the delta is
   **zero**. If it is not zero, the test is measuring something other than what it claims and no
   later result in this spec is trustworthy.
4. Time-scrub: capture at several `--time-of-day` values, assert the sky differs between them.
5. No-profile map: capture, assert a sky renders and the `profile-render` JSON reports provenance
   `HardcodedFallback` with `IsAuthored=false`.

---

## US2 — Model visible across the whole day (T034–T036)

1. **Day sweep**: capture at N values of `--time-of-day` across the full cycle; assert the model is
   present in every frame with no discontinuity at the former night threshold.
2. **Clock-driven proof**: two captures at *different* pinned times must differ; two captures at the
   *same* pinned time must be byte-identical. The second half is what proves animation follows the
   world clock rather than `DateTime.UtcNow`.
3. **Broken model**: point the reference at an invalid path, capture, assert a gradient sky still
   renders and the failure is reported once.

---

## US3 — Five-band gradient (T045–T046)

1. **Band isolation**: modify one mid-sky band, capture, assert the delta is confined to that band's
   height range while the zenith and horizon rows are unchanged.
2. **Inversion check**: assert the rendered zenith row matches the *authored zenith* colour. Band
   order is the reverse of LIT track index — track 2 is zenith, track 6 is horizon — so a direct
   index-to-order copy renders the sky upside down. See
   [contracts/sky-gradient.md](./contracts/sky-gradient.md) G2.
3. **Seam scan**: walk a vertical pixel column, assert no discontinuity in the colour derivative at
   band boundaries.
4. **Regression guard**: a two-band source must match the pre-change gradient.

---

## US4 — WMO interior skyboxes (T056)

1. Capture from an inside-WMO camera position and an outside one; assert the skies differ.
2. Capture the outside position twice across a simulated re-entry; assert identical output. That is
   the no-strobe proof.
3. Capture inside a WMO whose declared name is unresolvable; assert the outdoor sky persists and the
   reference is reported.

---

## US5 — Classification (T063)

Assert from `profile-render` JSON, not from images:

1. On a declaration-bearing build, classification counts and reported rules match expectation.
2. On a LIT-era build — which per research R1 has **no** outdoor model declaration — the filename
   fallback engages **and says so**. This is the expected normal path on this branch's target era.

---

## Phase 8 — Non-regression close-out (T064–T068)

1. Re-run `profile-render` on the **same two maps** as the baseline, camera motion on, compare
   against the recorded budget.
2. Assert no new hitches attributed to `Sky` or `SkyboxBackdrop`.
3. Run with sky disabled; assert both stages measure **zero**, not merely small.
4. Failure matrix by capture — no profile, missing asset, still-loading asset, unresolvable WMO
   reference — assert every case renders a non-black sky.
5. **Terrain fog guard**: capture terrain on both Alpha-era and LK 3.3.5, assert no pixel delta
   versus a pre-change capture. Sky already shares fog colour, which puts this next to the
   constitution's terrain risk area.

---

## Evidence to record

Per Constitution III, for every result: command, configured root (reported, never committed), build
identity and fingerprint, map, and outcome — **including negative results**, which per AGENTS.md
continuity rules are kept deliberately to prevent repeated dead ends.
