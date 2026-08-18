# Quickstart: Skybox Rendering Validation

**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Date**: 2026-08-18

Commands are PowerShell-ready. Per AGENTS.md, **real-client visual, FPS, and rendering proof is
user-run** — an agent prepares these and does not launch them. Build and test success is explicitly
**not** rendering proof.

---

## Build and focused tests

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

Focused tests before solution-wide, per AGENTS.md:

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug `
  --filter "FullyQualifiedName~Sky"
```

Solution-wide, once focused tests pass:

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

## Launch the viewer

```powershell
dotnet run --project I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

Client root is configured at runtime — never passed as a hardcoded path in source or committed
config (Constitution VI). **Record the configured root and build identity with every result below.**

---

## Phase 0 — Baseline capture (blocks all other work)

Fill [contracts/frame-budget.md](./contracts/frame-budget.md) from this. Once sky code changes, this
baseline cannot be recovered.

1. Launch the viewer on a **dense** map.
2. **Move the camera continuously** through the capture window. A static camera invalidates the
   measurement — the project has a recorded case of a static-camera profiler producing false null
   results, and p99/max are the whole point here.
3. Open the frame-history diagnostics panel and confirm the window reports camera movement.
4. Record `Sky` and `SkyboxBackdrop` **p50 / p99 / max** and any hitches attributed to either.
5. Repeat on a **sparse** map, so the budget is not fitted to one scene.
6. Record build identity, configured root, map, and frame count alongside each.

**Gate**: baseline recorded for both maps, both windows movement-valid, budget written into
`contracts/frame-budget.md` before any Phase 1 code lands.

---

## Phase 1 — US1: authored colours reach the screen

The differential test, because "the sky looks right" proves nothing:

1. Load a map with an authored sky profile. Note the rendered sky colour.
2. Change an authored sky colour in the client data. Reload.
3. **The rendered sky must change correspondingly.** Today it does not change at all — that is
   SC-001's "0% → 100%".
4. Scrub time of day; confirm the sky follows the authored timed samples.
5. Load a map with **no** resolvable profile; confirm a fallback sky renders **and reports itself as
   the fallback** in diagnostics.
6. Confirm every sky value in diagnostics names its source and record (FR-003).

---

## Phase 2 — US2: skybox model across the whole day

1. Load a map with a resolvable skybox model. Set time to **midday**.
2. Confirm the model renders. (Before this phase, it renders only at night.)
3. Sweep the full day cycle; confirm it is **continuously** present with no pop at the day/night
   threshold (SC-002).
4. Scrub time of day; confirm the model's appearance advances **with the clock**, not on wall-clock
   time. Freeze time and confirm the sky stops rather than continuing to animate.
5. Confirm terrain and objects always draw in front of the sky.
6. Point the model reference at a deliberately broken path; confirm the gradient still renders and
   the failure is reported **once**, not per frame.

---

## Phase 3 — US3: five-band gradient

1. Load a map whose source authors the full band set.
2. Change **one mid-sky band** in isolation. Reload.
3. Confirm the change appears **in that band's region** while zenith and horizon stay put (SC-003).
4. Check band boundaries for visible seams (FR-009).
5. **Inversion check**: confirm the zenith is the zenith. Band order is the reverse of LIT track
   index — track 2 is zenith, track 6 is horizon — so a direct index-to-order copy renders the sky
   upside down. See [contracts/sky-gradient.md](./contracts/sky-gradient.md) G2.
6. **Regression guard**: a two-band source must render identically to the pre-change gradient (G6).

---

## Phase 4 — US4: WMO interior skyboxes

1. Enter a WMO known to declare a skybox; confirm the visible sky changes.
2. Leave; confirm the outdoor sky is restored.
3. Cross the boundary **repeatedly**; confirm no flicker or strobing (SC-005).
4. Enter a WMO whose declared name cannot be resolved; confirm the outdoor sky persists and the
   unresolved reference is reported.
5. Enter a WMO declaring **no** skybox; confirm the outdoor sky is unchanged.
6. If available, test nested or overlapping WMOs with different declared skyboxes; confirm selection
   is stable frame to frame.

---

## Phase 5 — US5: data-driven classification

1. Find an asset the client data **declares** as a skybox whose filename contains none of `skybox`,
   `skybowl`, `environments/stars/`. Confirm it is treated as sky.
2. Find a non-sky asset whose filename **does** contain one of those keywords. Confirm it is not.
3. Confirm diagnostics report which declaration classified each (FR-021).
4. On a LIT-era build — which has **no** outdoor model declaration — confirm the filename fallback
   engages **and reports itself** (FR-020). This is the expected normal path on this branch's target
   era, not an edge case.

---

## Phase 6 — Non-regression close-out

1. Re-capture `Sky` and `SkyboxBackdrop` on the **same two maps** as Phase 0, moving camera, and
   compare against the recorded budget (SC-008).
2. Confirm no new hitches attributed to either stage.
3. Disable sky rendering; confirm both stages measure **zero**, not merely small (SC-009).
4. Walk the full failure matrix — no profile, missing asset, still-loading asset, unresolvable WMO
   reference — and confirm **every** case renders a sky (SC-007).
5. **Terrain fog guard**: confirm terrain fog is unchanged on both Alpha-era and LK 3.3.5 terrain.
   Sky already shares fog colour, which puts this adjacent to the constitution's terrain risk area.

---

## Evidence to record

For every result above, per Constitution III:

- Command run
- Configured client root (reported, never committed)
- Build identity and fingerprint
- Map
- Outcome, including negative results

Negative results are kept deliberately — per AGENTS.md continuity rules they prevent repeated dead
ends.
