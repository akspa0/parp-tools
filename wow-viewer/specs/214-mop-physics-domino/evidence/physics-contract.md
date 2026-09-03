# 5.0.1 Physics Contract

**Status**: Phase 0 deliverable for US1. **Normative.** This is the document a reviewer checks; the
investigation behind it is in [domino-caller-map.md](domino-caller-map.md) and
[physics-adapter-contract.md](physics-adapter-contract.md), and the library decision is in
[solver-selection.md](solver-selection.md).

**Program**: `/Wow.exe`, project `Mists of Pandaria 5.0.1.15464`, PE x86, image base `0x00400000`,
38,405 functions. **The session was read-only** — no bytes, symbols, labels, comments, data types,
functions, or analysis settings were modified, and no exception to that was taken (FR-004).

**Every claim carries an address and a label.** **M** = measured at the cited address. **I** =
inferred from measured bytes and *not itself proven*. Nothing here transcribes a Domino algorithm
(FR-005); the contract is layouts, data flow, and observable behaviour only.

**Era gate.** All of this is 5.0.1. **Domino does not exist in 0.5.3.** Nothing below may be applied
to another build without its own evidence.

---

## 1. Layer boundary (FR-003)

| Layer | Owns | Anchors |
|---|---|---|
| **Domino** | The solver. `Engine\Source\Domino/` | 15 headers, `0x00e0a690`–`0x00e0d428`; assertion handler `FUN_00c29680`; string `0x00e0ac8c` |
| **WoW adapter** | Sidecar load, model binding, culling, kinematic driving. `Engine\Source\Physics/` | `Physics.cpp` `0x00d7592c`, `PhysicsInt.cpp` `0x00d75a40`, `PhysData.cpp` `0x00d75dac`, `PhysData.h` `0x00d75b60` |

**M** — These are sibling directories, not nested. `dmMath.h` is the one Domino header the adapter
includes directly: its string sits at `0x00d75a88`, inside the *adapter's* string region rather than
with the other Domino headers, and two adapter functions assert against it.

**We reimplement the adapter. We replace Domino with a licensed library.** The boundary above is
exactly where that substitution happens.

## 2. Sidecar discovery

**M** `FUN_005a29a0` — the sidecar path is the model path **with its extension replaced by `.phys`**.
Association is by filename alone: no id, no index, no lookup table.

**M** The extension is written as two immediates, `*(u32*)p = 0x7968702e` and `*(u16*)(p+4) = 0x0073`
(`.phy` + `s\0`). **There is therefore no `.phys` string in the binary**, and a string search for one
returns a false negative.

**M** `Physics.cpp:47` asserts `(fileName-ext)+6 <= 260`; the buffer is 260 bytes.

## 3. Container format

**M** `FUN_005a5080`. Chunked, with tags stored **reversed on disk** (the `MVER`/`REVM` convention).

- Magic: `PHYS` (compared as `0x50485953`; on-disk bytes `SYHP`).
- **Version: the `u16` at the start of the `PHYS` payload MUST be 0.** 5.0.1 accepts only version 0.
- Chunks are `{u32 tag, u32 size, payload}`; walking starts at `header + 8 + PHYS.size` and advances
  by `8 + size` until `m_data + fileSize`.
- **Record counts are derived by dividing chunk size by a fixed stride**, which is what makes every
  stride below measured rather than assumed.

| Tag | On-disk | Stride | `PhysData` count / ptr |
|---|---|---:|---|
| `BOXS` | `SXOB` | 60 | `+0x00` / `+0x04` |
| `CAPS` | `SPAC` | 28 | `+0x08` / `+0x0C` |
| `SPHS` | `SHPS` | 16 | `+0x10` / `+0x14` |
| `SHAP` | `PAHS` | 20 | `+0x18` / `+0x1C` |
| `BODY` | `YDOB` | 28 | `+0x20` / `+0x24` |
| `SPHJ` | `JHPS` | 28 | `+0x28` / `+0x2C` |
| `SHOJ` | `JOHS` | 108 | `+0x30` / `+0x34` |
| `WELJ` | `JLEW` | 104 | `+0x38` / `+0x3C` |
| `JOIN` | `NIOJ` | 16 | `+0x40` / `+0x44` |

`+0x48` `m_data`, `+0x4C` file size. **The struct ends at `0x50`, matching the 80-byte allocation at
`Physics.cpp:50`** — an independent check that the field map has no gaps.

## 4. Field naming (FR-002)

**M** Every array name comes from Blizzard's own bounds asserts in `PhysData.h`, not from an
inherited or community name:

`m_boxShapeCount` (157) · `m_capsuleShapeCount` (159) · `m_sphereShapeCount` (161) ·
`m_shapeCount` (163) · `m_bodyCount` (166) · `m_sphericalJointCount` (169) ·
`m_shoulderJointCount` (171) · `m_weldJointCount` (173) · `m_jointCount` (175).

**`SHOJ` is a *shoulder* joint; `SPHJ` is a *spherical* joint.** Both are measured, not guessed.
Fields that remain unnamed below are genuinely unknown and are marked so rather than filled in from
the community layout — the discipline in [[feedback_a_name_stops_the_looking]].

## 5. Record layouts

| Record | Size | Layout | Label |
|---|---:|---|---|
| `SPHS` | 16 | `+0` vec3 centre, `+12` f32 radius | **M** |
| `CAPS` | 28 | `+0` vec3, `+12` vec3, `+24` f32 | **M** reads; endpoint/radius naming **I** |
| `BOXS` | 60 | `+48` vec3 read by the shape builder; **bytes `0..47` never read on this path** | `+48` **M**; 4×3 transform **I, unverified** |
| `SHAP` | 20 | `+0` u16 type (**0 box, 1 capsule, 2 sphere**), `+2` u16 index into that type's array, `+4` unread, `+8`/`+12`/`+16` three dwords passed to the shape descriptor | type/index **M**; the three dwords **M as passed, semantics unknown** |
| `BODY` | 28 | `+0` u16 type (0→1, 1→0, else 2), `+4` vec3 position, `+16` u16 bone index, `+20` u32 first shape, `+24` u32 shape count | **M** |
| `JOIN` | 16 | `+0` u32 body A, `+4` u32 body B, `+8` unread, `+12` u16 type (**0 spherical, 1 shoulder, 2 weld**), `+14` u16 index | **M** |
| `SPHJ` / `SHOJ` / `WELJ` | 28 / 108 / 104 | consumed as runs of vec3s plus trailing scalars | read extents **M**; per-field semantics **not decoded** |

## 6. Build order and binding

**M** `FUN_005a4ce0` → bodies, then shapes (`FUN_005a3630`), then joints (`FUN_005a3ab0`). That order
is a constraint on any reimplementation: joints resolve body pointers that must already exist.

**M** The runtime body array at `instance+0x48` has **stride 8**: `[+0]` bone index, `[+4]` solver
body pointer. `JOIN`'s body indices resolve through this array
(`instance+0x48 + 4 + index*8`) — which cross-confirms both the `JOIN` and `BODY` layouts.

**M** The sidecar hangs off the model at `model->+0x2C` → `+0x180`; **null means no physics**, and the
model is unaffected.

**M** `Physics.cpp:70` asserts `model && model->IsLoaded()` — physics attaches only after the model is
fully loaded.

**M** Live instances are held in a global array at `0x00f59338` (count `0x00f5933c`, capacity
`0x00f59340`) growing **1.5×**.

## 7. Observable behaviour

| Behaviour | Detail | Label |
|---|---|---|
| **Gravity** | `(0, 0, -10.0)` written at world `+0x80` (`0xC1200000` = `-10.0f`), `FUN_005a3010`. **Not 9.81** | **M** |
| **World** | 384-byte singleton at `0x00f59350`; `PhysicsInt.cpp:16` asserts `s_world == 0` | **M** |
| **Distance culling** | `FUN_005a40d0` compares model distance (`model+0x98`) against the cull distance and **returns early**, doing no work. Culling is in the per-instance update, *not* in the solver | **M** |
| **First update teleports** | The `instance+0x70` flag (set at creation) makes the first update *place* bodies at their bones rather than velocity-drive them | **M** |
| **Kinematic driving** | Linear and angular velocity = (current − previous bone transform) × `1/dt`, written to body `+0x180` / `+0x190`, guarded by body type ≠ 2. **Physics follows animation; it does not replace it** | **M** |
| **No-bone sentinel** | Bone index `0xFFFF` is skipped by the driving loop. Preserve the value; do not clamp it | **M** |
| **Large-motion guard** | Per-step motion tested against thresholds from `0.5/dt` (linear) and `0.25π/dt` (angular); exceeding them takes an interpolation path | thresholds **M**; intent (teleport protection) **I**; blend arithmetic **not reproduced** |
| **FP determinism** | `__clearfp()`, `_control87_2(0x9001F, 0x8001F)`, `MXCSR = (MXCSR & ~0x3F) \| 0x1F80`, **both restored afterwards**. The client deliberately pins FP state across physics work | **M** |
| **Input validation** | Every vector crossing into the solver is checked by `dmMath.h:127` (`F32Check`) | **M** |

**Runtime knobs (M)**: `0x00EB6350` enabled (no-argument toggles), `0x00EB6358` cull distance (f32),
`0x00EB635C` next instance id.

## 8. Failure behaviour

**M, uniformly fail-closed**: missing file, wrong magic, version ≠ 0, or a failed parse all produce a
deleted `PhysData` and a **normally rendered, non-physicalised model**. Absent chunks leave count and
pointer at 0 so consumer loops simply do not run. Out-of-range indices hit named `PhysData.h` asserts.
Double load asserts `m_data == 0` at `PhysData.cpp:267`.

**M, the one permissive rule: unknown chunk tags are skipped via their size field.** The format is
forward-compatible by construction, so **a reader that rejects unknown tags would be stricter than
the client** and would break on later-era files.

**Required divergence.** The client is silent about all of this. Our reader must match the *fallback*
but not the *silence*: FR-010 and FR-012 require an explicit diagnostic for every skipped or
malformed construct. This is deliberate, not an oversight.

## 9. Solver contract

**Selected: Jitter2 2.8.10, MIT** ([solver-selection.md](solver-selection.md)). Shape coverage
(box/capsule/sphere) is satisfied. Determinism is CI-enforced upstream. Cloth is reachable by adapting
~115 lines of first-party MIT sample onto shipped library primitives.

**The adapter obligations above are ours regardless of solver**: gravity `-10.0`, pinned FP state,
cull-at-update, teleport-on-first-update, `0xFFFF` preservation, and build order.

**Open risk**: **shoulder** and **weld** joints have no obvious Jitter2 counterpart; only spherical
maps cleanly. Deferred to Phase 6 rather than assumed.

## 10. Explicit unknowns

`BOXS` bytes `0..47` · `SHAP` `+4` and the semantics of `+8`/`+12`/`+16` · `JOIN` `+8` ·
per-field semantics of `SPHJ`/`SHOJ`/`WELJ` · the zeroed 4-float world global at `0x00F59360` ·
the exact large-motion blend arithmetic · whether `PHYS` version 0 is the only version accepted on
*every* path · **line-level attribution for Domino-internal functions** (see the T008 review).

**None of this has been validated against real bytes.** T002 remains open, and every layout above is
recovered from decompiled arithmetic rather than from an observed `.phys` file.
