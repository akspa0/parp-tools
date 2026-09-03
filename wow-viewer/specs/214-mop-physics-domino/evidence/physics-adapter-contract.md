# T004 / T005 — Physics Adapter Boundary and Model-Sidecar Discovery

**Method**: read-only Ghidra session, program `/Wow.exe`, project `Mists of Pandaria 5.0.1.15464`,
PE x86, image base `0x00400000`. Nothing in the program was modified.

**Scope**: WoW's own adapter layer (`Physics.cpp`, `PhysicsInt.cpp`, `PhysData.cpp`,
`Engine\Source\Physics/PhysData.h`) and the file format it reads. **No Domino algorithm is decoded
here.** Domino appears only where the adapter calls into it, and only as a boundary.

Every claim below is marked **MEASURED** (read directly from decompiled code at a cited address) or
**INFERRED** (a reading of measured bytes that has not itself been proven).

---

## 1. Sidecar discovery — MEASURED, and T005 is answered

`FUN_005a29a0`, anchored by `Physics.cpp` at `0x00d7592c`.

1. The model filename is copied into a 260-byte stack buffer (Storm `SStr` copy, 0x103 cap; the
   copy helper asserts `source` at `SStr.inl:111`).
2. The extension is located and the string truncated at the `.`.
3. **Assert `"(fileName-ext)+6 <= 260"` at `Physics.cpp:47`** — the replacement extension is 6 bytes
   including the terminator.
4. Two immediate stores append it:
   - `*(u32*)p = 0x7968702e` produces bytes `2E 68 70 79` = `.` `p` `h` `y`
   - `*(u16*)(p+4) = 0x0073` produces bytes `73 00` = `s` `NUL`

   **The sidecar extension is `.phys`.**

**This is why searching the binary for a `.phys` string returns nothing** — the extension is never a
string literal, it is two immediates. A search-driven pass would have concluded the extension is
absent and gone looking for a chunk inside the M2 instead. Same shape as
[[feedback_verify_detector_power_before_null_results]]: the null result came from a detector that
could not see the thing.

5. `0x50` (80) bytes are allocated at `Physics.cpp:50` for the `PhysData` object, then the parser runs.
6. **On any parse failure the object is destructed and deleted and the function returns 0.**

**Model association**: the sidecar path is the model path with its extension replaced. There is no
index, no lookup table, and no id — association is purely by filename. That resolves T005's "actual
model association".

## 2. Instantiation and the live registry — MEASURED

`FUN_005a2d10`, `Physics.cpp`.

- **Assert `"model && model->IsLoaded()"` at `Physics.cpp:70`** — physics attaches only to a
  *fully loaded* model. Sidecar work must be sequenced after model load completes, not alongside it.
- `0x78` (120) bytes are allocated at `Physics.cpp:75` for the runtime instance.
- `FUN_005a4ce0` builds it from the model; **on failure the instance is destructed and 0 returned.**
- On success the instance pointer is appended to a global growable array:
  `0x00f59338` data, `0x00f5933c` count, `0x00f59340` capacity, growing by **1.5x** (`n + n/2`,
  floored at the required size). This is the set of live physicalised objects.

## 3. Physics world and gravity — MEASURED

`FUN_005a3010`, `PhysicsInt.cpp`.

- **Assert `"s_world == 0"` at `PhysicsInt.cpp:16`**; the world singleton is `0x00f59350`.
- The world object is `0x180` (384) bytes.
- Immediately after construction, four dwords are written at world `+0x80`:
  `{0, 0, 0xC1200000, 0}`. **`0xC1200000` is IEEE-754 `-10.0f`.**

  **Gravity is `(0, 0, -10.0)` — Z-up, magnitude 10.0 units/s².** Not 9.81. A port that uses
  real-world gravity will not match the client.
- A second 4-float global at `0x00f59360` is zeroed at world creation. Purpose unknown (INFERRED
  candidate: the wind or global-force input that spec 215 would drive; **not verified**).

## 4. The `.phys` container — MEASURED

`FUN_005a5080`, `PhysData.cpp`. Opened with the debug tag `"PhysData.cpp(261)"`.

**Header**: `u32` tag compared against `0x50485953`. In memory that is the byte sequence
`53 59 48 50` = `S Y H P`, i.e. the tag is stored **reversed on disk**, the same convention as every
other Blizzard chunked format (`MVER` stored as `REVM`). The logical tag is **`PHYS`**.

The `u16` at the start of the `PHYS` payload **must be 0**. That is a version field and 5.0.1 accepts
only version 0 — an era-gating fact, and exactly the kind of check our reader must carry rather than
assume.

**Iteration**: chunk walking begins at `header + 8 + PHYS.size` and advances by `8 + size` per chunk
until `m_data + fileSize`. Each chunk is `{u32 tag, u32 size, payload}`. Counts are derived by
**dividing the chunk size by a fixed record stride**, which is what makes every record size below a
measured fact rather than a guess.

| Compared `u32` | On-disk bytes | Logical tag | Stride | `PhysData` count / ptr |
|---:|---|---|---:|---|
| `0x424F5853` | `SXOB` | **BOXS** | 60 (`0x3c`) | `+0x00` / `+0x04` |
| `0x43415053` | `SPAC` | **CAPS** | 28 (`0x1c`) | `+0x08` / `+0x0C` |
| `0x53504853` | `SHPS` | **SPHS** | 16 (`>>4`) | `+0x10` / `+0x14` |
| `0x53484150` | `PAHS` | **SHAP** | 20 (`0x14`) | `+0x18` / `+0x1C` |
| `0x424F4459` | `YDOB` | **BODY** | 28 (`0x1c`) | `+0x20` / `+0x24` |
| `0x5350484A` | `JHPS` | **SPHJ** | 28 (`0x1c`) | `+0x28` / `+0x2C` |
| `0x53484F4A` | `JOHS` | **SHOJ** | 108 (`0x6c`) | `+0x30` / `+0x34` |
| `0x57454C4A` | `JLEW` | **WELJ** | 104 (`0x68`) | `+0x38` / `+0x3C` |
| `0x4A4F494E` | `NIOJ` | **JOIN** | 16 (`>>4`) | `+0x40` / `+0x44` |

`+0x48` is the file buffer (`m_data`), `+0x4C` the file size. The struct ends at `0x50` — **exactly
the 80 bytes allocated at `Physics.cpp:50`**, which independently confirms the field map is complete
with no gaps.

### The field names are measured, not guessed

`PhysData.h` bounds-check asserts name every array. This is the strongest evidence in the pack —
Blizzard's own identifiers:

| `PhysData.h` line | Assert text | Names |
|---:|---|---|
| 157 (`0x9d`) | `i<m_boxShapeCount` | BOXS |
| 159 (`0x9f`) | `i<m_capsuleShapeCount` | CAPS |
| 161 (`0xa1`) | `i<m_sphereShapeCount` | SPHS |
| 163 (`0xa3`) | `i<m_shapeCount` | SHAP |
| 166 (`0xa6`) | `i<m_bodyCount` | BODY |
| 169 (`0xa9`) | `i<m_sphericalJointCount` | SPHJ |
| 171 (`0xab`) | `i<m_shoulderJointCount` | SHOJ |
| 173 (`0xad`) | `i<m_weldJointCount` | WELJ |
| 175 (`0xaf`) | `i<m_jointCount` | JOIN |

**`SHOJ` is a *shoulder* joint and `SPHJ` a *spherical* joint** — named by the client, so neither is
an assumption. Recording this now is precisely the discipline in
[[feedback_a_name_stops_the_looking]]: these names are earned, and the fields still marked unknown
below are honestly unknown.

## 5. Record layouts

From the shape builder `FUN_005a3630` and the body builder `FUN_005a4ce0` (both `PhysData.h`).

**SPHS — 16 bytes. MEASURED.**
`+0` vec3 centre, `+12` f32 radius.

**CAPS — 28 bytes. MEASURED reads; naming INFERRED.**
`+0` vec3, `+12` vec3, `+24` f32. INFERRED: two segment endpoints and a radius.

**BOXS — 60 bytes. Partially measured.**
The shape builder reads **only** `+0x30`, `+0x34`, `+0x38` (a vec3 at offset 48). Bytes `0..47` are
not touched on this path. INFERRED: a 4x3 (12-float) transform followed by half-extents. **The first
48 bytes are unverified** and must not be written into a reader as a transform without a second
measurement.

**SHAP — 20 bytes. Partially measured.**
`+0` u16 shape type — **0 = box, 1 = capsule, 2 = sphere** (MEASURED: each branch bounds-checks the
matching array). `+2` u16 index into that type's array (MEASURED). `+4` not read here. `+8`, `+12`,
`+16` are three dwords passed into the Domino shape descriptor (MEASURED as passed; **semantics
unknown** — the community `.phys` layout calls these friction / restitution / density, which is a
plausible INFERENCE but is not measured here).

**BODY — 28 bytes. MEASURED.**
`+0` u16 body type, remapped to a Domino enum (type 0 maps to 1, type 1 maps to 0, anything else to
2); `+4` vec3 position; `+16` u16 **bone index**; `+20` u32 first shape index; `+24` u32 shape count.
The last two are measured from the shape builder's use of `body+0x14` and `body+0x18` as the SHAP
range.

**JOIN — 16 bytes. MEASURED.**
`+0` u32 body A index, `+4` u32 body B index, `+8` not read, `+12` u16 joint type
(**0 = spherical, 1 = shoulder, 2 = weld**, each bounds-checked against its own array),
`+14` u16 index into that array.

**SPHJ (28) / SHOJ (108) / WELJ (104)** — consumed as runs of vec3s (SHOJ and WELJ each read about 9
vec3s plus trailing scalars). Read extents are measured; per-field semantics are **not** decoded.

**Cross-check**: this chunk set and these strides match the publicly documented `.phys` format on
wowdev.wiki. That agreement is a useful confirmation, but the layouts above were recovered
independently from this binary, and where the two differ **this file is the evidence**.

## 6. Runtime binding — MEASURED

`FUN_005a4ce0` builds bodies, then shapes, then joints, in that order.

- The sidecar hangs off the model at `model->+0x2C` then `+0x180`. **If that pointer is null the
  function returns 0 and no physics is created** — a model without a `.phys` is simply not
  physicalised.
- Bodies are walked with stride 28; each body's runtime entry goes into an array at `instance+0x48`
  with **stride 8**: `[+0]` the bone index, `[+4]` the Domino body pointer. The joint builder resolves
  `JOIN`'s body indices through exactly this array (`instance+0x48 + 4 + index*8`), which
  cross-confirms both layouts.
- The body of type 0 is additionally stored at `instance+0x44` as the instance's root body.
- Body world placement is the bone transform composed with the body's `+4` vec3 offset.

## 7. Observable behaviour of the update — MEASURED

`FUN_005a40d0`, the per-instance update, taking the timestep as its argument.

- **A model flag selects the path**: bit 3 of `model+0x10`. Clear means idle, set means driven.
- **Distance culling is real and it is here.** On the idle path, if the physics-culling query returns
  0, the model's distance (`model+0x98`) is compared against the culling distance and the function
  **returns early**, doing no work. This is the client behaviour our `PhysicsBudget` policy already
  models — and it validates that cull distance belongs at the per-instance update, not in the solver.
- **First update teleports, it does not push.** The instance's `+0x70` flag (set to 1 at creation) is
  cleared on the first update, and on that pass every body is *placed* at its bone position rather
  than velocity-driven. Without this a newly spawned ragdoll would be launched from the origin.
- **Bodies are driven kinematically from the animation.** On the driven path, linear and angular
  velocity are computed as the delta between the previous and current bone transform multiplied by
  `1/dt`, and written to the body at `+0x180` (linear) and `+0x190` (angular), each guarded by
  *body type != 2*. Physics follows the animation; it does not replace it.
- **`0xFFFF` is the "no bone" sentinel.** Body entries whose bone index is `0xFFFF` are skipped in
  the driving loop. A reader must preserve this value rather than clamping it.
- **A large-motion guard exists.** Per-step linear and angular motion are tested against thresholds
  derived from `0.5/dt` and `0.25*pi/dt`; exceeding them takes an interpolation path instead of
  imparting the full velocity. INFERRED intent: prevent teleports and animation pops from exploding
  the simulation. The exact blend arithmetic is not cleanly recoverable from the decompiler output
  and is **not** reproduced here.
- **Floating-point mode is pinned around the build.** `__clearfp()`, then
  `_control87_2(0x9001F, 0x8001F)`, then `MXCSR = (MXCSR & ~0x3F) | 0x1F80`, with both the x87
  control word and MXCSR **restored afterwards**. The client deliberately fixes FP exception masking
  and rounding across physics work. This is directly relevant to SC-004/SC-005 determinism: our
  solver adapter should establish and restore an equivalent deterministic FP state rather than
  inheriting whatever the renderer left.
- Every vector handed across the boundary is validated by the `dmMath.h:127` finite check described
  in [domino-caller-map.md](domino-caller-map.md).

## 8. Runtime knobs — MEASURED

| Global | Meaning | Set by |
|---|---|---|
| `0x00EB6350` | physics processing enabled | `FUN_005a2e60`; **no argument toggles it** |
| `0x00EB6358` | physics culling distance (f32, via `atof`) | `FUN_005a2f20` |
| `0x00EB635C` | next physics instance id, post-incremented per instance | `FUN_005a4ce0` |

## 9. Malformed and absent data — MEASURED, and it fails closed everywhere

This answers T005's "malformed-data behaviour" requirement.

| Condition | Behaviour |
|---|---|
| No `.phys` file | Open fails, parser returns 0, `PhysData` deleted; no physics, model renders normally |
| Wrong `PHYS` magic | Parser returns 0, same path |
| Version `u16` not 0 | Parser returns 0, same path |
| Chunk absent | Count and pointer remain 0; consumer loops do not execute. No error, no diagnostic |
| **Unknown chunk tag** | **Skipped via its size field; iteration continues.** The format is forward-compatible by construction |
| Index out of range | `PhysData.h` bounds assert (`FUN_00684120`), tagged with the array name |
| Body with bone `0xFFFF` | Skipped by the driving loop |
| Double load | Assert `m_data == 0` at `PhysData.cpp:267` |
| Model not fully loaded | Assert `model && model->IsLoaded()` at `Physics.cpp:70` |

**Design consequence for our port.** The client's own posture is *silent, total fail-closed*: an
unparseable or missing sidecar produces a normally-rendered, non-physicalised model and no user-facing
error. Our reader must match the fallback but **must not** match the silence — spec 214's contract
requires an explicit diagnostic for every skip, which is a deliberate divergence, not an oversight.
The unknown-chunk rule is the one place the client is explicitly permissive, and a reader that
rejects unknown tags would be *stricter than the client* and would fail on later-era files.

---

## Gate status

T004 and T005 are **answered and unblocked**:

- The adapter boundary is traced end to end: discovery, parse, body/shape/joint build, update.
- Model association is by **filename with the extension replaced by `.phys`** — no id, no table.
- Malformed-data behaviour is measured at every stage and is uniformly fail-closed.

**Still open before Phase 2 parsing may begin**: T002, a real-client asset manifest with build
fingerprint and hashes for actual `.phys` files, so the layouts above are validated against real
bytes rather than against decompiled arithmetic alone.

**Still open before any solver work**: T007. Nothing in this file selects or licenses a solver.

## Explicit unknowns

- `BOXS` bytes `0..47` (INFERRED as a 4x3 transform; unverified).
- `SHAP` `+4`, and the semantics of `+8` / `+12` / `+16`.
- `JOIN` `+8`.
- Per-field semantics of `SPHJ`, `SHOJ`, `WELJ` beyond their read extents.
- The zeroed 4-float world global at `0x00F59360`.
- The exact large-motion blend arithmetic in `FUN_005a40d0`.
- Whether `PHYS` version 0 is the only version this client accepts in *all* paths, or only this one.
