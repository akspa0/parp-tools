# WoW 1.0.0 M2 Model Cameras ("camera tracks") — Viewer Implementation Guide (2026-07-15)

Goal: play back **M2/MDX model cameras** in the viewer — a camera object baked into a
model that glides along an authored spline over time (the mechanism behind cinematic
flythroughs and the `Cameras\*.m2` files). This is the camera analogue of the taxi-route
follower you already have: sample a spline over a timeline and drive the view from it.

Traced from **WoW.exe 1.0.0.3980**. Evidence + decompilations:
[`evidence/1.0.0-ghidra/m2_camera.c`](wow-viewer/specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/m2_camera.c).
Confidence: **[V]** verified in decomp, **[I]** inferred/standard.

---

## 1. Where the cameras live in the M2

**[V]** M2 header (format `0x100`):

| Header ofs | Field | Element |
|-----------|-------|---------|
| 0x124 | `cameras` (count + offset) | M2Camera, **0x7c (124) B** each |
| 0x12C | `cameraLookup` (count + offset) | `int16[]` — maps a camera *id* → record *index* |

- `GetCameraCount()` = `cameras.count`.
- `cameraLookup[id]` → index into `cameras` (`0xFFFF` = none). IDs are semantic
  (portrait / char-select / flythrough); indices are physical.
- **[V]** In 1.0.0 the cameras (and their animation) are **fully embedded** in the `.m2` —
  no external `.anim`/`.skin`. You already have the model bytes.

---

## 2. M2Camera record — 0x7c (124 bytes) **[V]**

```c
struct M2Camera {              // 0x7c
  uint32   type;               // 0x00  -1 = portrait; >=0 = camera kind/id
  float    fov;                // 0x04  DIAGONAL fov, RADIANS (see §5)
  float    farClip;            // 0x08
  float    nearClip;           // 0x0c
  M2Track  positions;          // 0x10  C3Vector spline — the EYE path
  C3Vector positionBase;       // 0x2c  eye fallback/pivot (used iff positions empty)
  M2Track  targetPosition;     // 0x38  C3Vector spline — the LOOK-AT path
  C3Vector targetBase;         // 0x54  target fallback (used iff targetPosition empty)
  M2Track  roll;               // 0x60  float spline — camera roll (radians)
};
```

Recovered from the block relocator `FUN_00720450` (0x7c stride) — it relocates exactly
three tracks at base offsets 0x10 / 0x38 / 0x60, with C3Vector spline values on the first
two and a float spline on the third, plus the two `C3Vector` bases. This is the canonical
classic-era M2Camera; the byte offsets above are confirmed for `0x100`.

### 2.1 THE version gotcha — old M2Track format **[V]**

1.0.0 uses the **old (v256) M2Track**, which still has the **interpolation-ranges array**.
Your per-version reader must parse this or every camera (and every animated track) offset
is wrong:

```c
struct M2Track_v256 {          // 0x1c (28 bytes)
  uint16  interpolationType;   // 0x00  0=none 1=linear 2=hermite/bezier
  uint16  globalSequence;      // 0x02  0xFFFF = none
  M2Array interpRanges;        // 0x04  <-- PRESENT in 0x100; elem = {u32 start,u32 end}
  M2Array timestamps;          // 0x0c  elem = uint32 ms
  M2Array values;              // 0x14  elem = M2SplineKey<T>
};
// Wrath+ (rev >= 264) DROPS interpRanges -> track becomes 0x14 bytes. Do NOT reuse a
// modern (rangeless) track reader for 0x100.
```

`M2SplineKey<T>` stores tangents inline: `{ T value, T inTan, T outTan }`.
- float roll key = 3 floats = **0xC bytes** (confirmed: roll values relocated with 0xC stride).
- C3Vector position/target key = 3×C3Vector = **0x24 bytes**.
- For `interpolationType==1 (linear)` only `value` is meaningful; for `2` use the
  Hermite tangents.

---

## 3. Runtime model (how the client stores/animates it) **[V]**

- Each `CM2Model` instance has a **camera-instance array at instance `+0x398`**, one entry
  per record, **0x84 (132) bytes** each. Field `+0x80` of each instance is a pointer back
  to the source `M2Camera` record; the preceding bytes hold the currently-evaluated camera
  state (eye/target/roll/matrices).
- Accessors: `HasCamera(id)` (`FUN_0070edc0`), `GetCameraById(id)` (`FUN_0070ee30`),
  `GetCameraByIndex(i)` (`FUN_0070eeb0`), all via `cameraLookup`.
- The camera is consumed by: character **portraits** (`FUN_0053b6d0` → renders the model
  through `GetCameraById(0)` into `Portrait1`), **model-view UI widgets**
  (`FUN_00743630`/`FUN_007435f0`, camera held as a smart-ptr at widget `+0x2d8`), and the
  **cinematic system** (`InCinematic`/`OpeningCinematic`/`StopCinematic`) which drives the
  main `CGCamera` from a model camera — i.e. exactly the flythrough use-case.

---

## 4. Evaluation algorithm (what to implement) **[V layout / I math]**

Per frame, at time `t` (ms) within the camera's chosen animation **sequence** (loop it):

```
eye    = sampleTrack(cam.positions,      t)  ?? cam.positionBase   // fallback iff no keys
target = sampleTrack(cam.targetPosition, t)  ?? cam.targetBase
roll   = sampleTrack(cam.roll,           t)  ?? 0                  // radians
fov    = cam.fov                                                  // static in 1.0.0

// eye/target are in the camera-model's LOCAL space. Place the model in the world with
// matrix M (identity for a standalone previewed camera; the WMO/world placement for an
// in-world cinematic camera), then:
eyeW    = M * eye
targetW = M * target
fwd     = normalize(targetW - eyeW)
up      = rotateAroundAxis(worldUp, fwd, roll)     // roll spins 'up' about the view dir
view    = lookAt(eyeW, targetW, up)
proj    = perspective(verticalFov(fov, aspect), aspect, cam.nearClip, cam.farClip)
```

- **[V]** `eye` and `target` come from **separate splines** — the camera can look around
  independently of where it flies. `positionBase`/`targetBase` are the values used when the
  respective track has **zero keys** (do not sum base + track).
- You already have `sampleTrack` (taxi/anim). Reuse it — camera tracks are ordinary
  M2Tracks (2× C3Vector, 1× float), just remember the **0x1c old-format layout** (§2.1).
- `globalSequence != 0xFFFF` → drive that track from the global-sequence timer instead of
  the sequence clock (same rule as any other M2 track).

---

## 5. FOV — the one thing to calibrate **[I]**

`cam.fov` is the **diagonal FOV in radians**. The client feeds it into its projection
directly; the exact aspect remap wasn't traced. Practical recipe:

1. Start with `verticalFov = 2*atan( tan(fov/2) / sqrt(1 + aspect²) ) * sqrt(1+aspect²)`
   — or simpler, treat `fov` as the vertical fov and multiply by a constant `k≈0.6`.
2. Calibrate `k` once against a known camera M2 (e.g. a login-screen / cinematic camera)
   so the framing matches a reference screenshot, then lock it.

Everything else (near/far/eye/target/roll) is exact from the record.

---

## 6. Driving it like taxi routes

- Pick the camera's **animation sequence** (usually sequence 0, or the one whose id matches
  the camera use). Advance `t` from `seq.start`→`seq.end` at real time; loop or one-shot.
- Expose the same controls you have for taxi playback: play/pause, scrub `t`, loop, speed.
- For a standalone "watch the camera" mode: load the camera M2, set model matrix = identity,
  and render your existing scene through the evaluated `view`/`proj`. For an in-world
  cinematic: place the camera M2 at its world position and use that as `M`.
- **Great debugging aid** (matches the user's intent): because eye and target are separate
  splines, you can draw both polylines + the roll to *see* the authored path — the camera
  equivalent of visualizing a taxi route.

---

## 7. Implementation checklist for the viewer

1. **Reader**: in the `0x100` M2 branch, parse `cameras` (0x124) + `cameraLookup` (0x12C);
   parse each 0x7c record with the field map in §2; parse embedded tracks with the **old
   0x1c layout** (§2.1). ← the only genuinely new parsing.
2. **Evaluator**: reuse your M2 track sampler for the 3 camera tracks + static fov (§4).
3. **View builder**: `lookAt(eye,target,up-with-roll)` + `perspective(fov,near,far)` (§4–5).
4. **Player UI**: sequence timeline + play/scrub/loop, plus optional path visualization (§6).
5. **Verify**: pick a `Cameras\*.m2` (or a creature with a camera) and confirm the framing
   against the in-game portrait/cinematic; calibrate the fov constant once (§5).

---

## 8. Open / to verify

- Exact FOV→projection aspect handling (calibrate empirically, §5).
- Whether any 1.0.0 camera uses `globalSequence` (handle it generically regardless).
- The in-world cinematic **placement + sequencing** table (which camera plays when) — the
  `m_cinematic` manager; not needed to play a single camera M2, only to reproduce scripted
  multi-shot cinematics.
