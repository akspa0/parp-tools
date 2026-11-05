# 5.0.1 Atmosphere & Physics — Ghidra Evidence

> **Owner since 2026-09-23:** Epic 249 (sky/fog/lighting) and 252 (physics W-20, weather W-21) — [249-epic-renderer-performance-and-correctness](../specs/249-epic-renderer-performance-and-correctness/spec.md). Spec numbers below refer to archived specs under `specs/archived/` (reconciliation-2026-09-23).

Last verified: 2026-09-02 against the Ghidra project `Mists of Pandaria 5.0.1.15464`,
program `/Wow.exe`, PE x86, image base `0x00400000`, **38,405 functions**
(matches the spec 197 baseline). Project path
`C:\WoW4-data\MoPBeta\ghidra\Mists of Pandaria 5.0.1.15464`, bridge on
`http://127.0.0.1:8089`.

This is a reverse-engineering reconnaissance note, not a claim of runtime proof. The session was
**read-only**: no bytes, symbols, labels, comments, data types, functions, or analysis settings were
changed. The program was opened (it was closed at session start) with `auto_analyze: false`.

## Why this file exists

Sky, fog and lighting already have owning specs — **160** (skybox rendering), **147** (fog coverage),
**143** (world context and lighting parity). What those specs lack is *native evidence from a client
that implements the complete system*. This note is that evidence, published once and consumed by
them, rather than duplicated into competing specs. Spec **214** (physics) and spec **215** (weather)
own the two systems that have no existing spec.

The method is the one spec 197 established: Blizzard's build preserved `.cpp` source-path strings in
the binary, so the source file names are recoverable and each one anchors a cluster of functions.
Addresses below are Ghidra static addresses in the loaded image.

## Source anchors — measured

| System | Anchor string | Address |
|---|---|---:|
| Physics (WoW side) | `Physics.cpp` | `0x00d7592c` |
| Physics (WoW side) | `PhysicsInt.cpp` | `0x00d75a40` |
| Physics data | `PhysData.cpp` | `0x00d75dac` |
| Physics data | `PhysData.cpp(261)` (assert) | `0x00d75dbc` |
| Physics data | `Engine\Source\Physics/PhysData.h` | `0x00d75b60` |
| Sky | `DNSky.cpp` | `0x00e0a44c` |
| Sky | `DNOverrideSky.cpp` | `0x00e0a2eb` |
| Clouds | `DNClouds.cpp` | `0x00e0a0ca` |
| Zone light | `DNZoneLight.cpp` | `0x00e0a458` |
| Model light | `M2Light.cpp` | `0x00d70923` |
| Fog | `MaterialFog.cpp` | `0x00d98234` |
| Fog | `PassFogCombine.cpp` | `0x00e101f8` |
| Fog effect | `EffectSwirlingFog.cpp` | `0x00e0ff54` |
| Weather | `MapWeather.cpp` | `0x00debaae` |
| Weather | `Lightning.cpp` | `0x00e26da8` |
| Particles | `ParticleSystem2.cpp` | `0x00d74f84` |
| Ribbons | `RibbonEmitter.cpp` | `0x00d76f13` |

`DN` is **Day/Night**. RTTI confirms it is a namespace, not a prefix convention:
`.?AVCZoneLight@DayNight@@` at `0x00eecb04` demangles to `DayNight::CZoneLight`, and
`ZoneLightPointRec` appears at `0x00eecae4`. RTTI is present throughout this binary, so the class
names of this subsystem are recoverable rather than guessable.

## The physics engine is Domino — measured

The solver is not written inline in the WoW source tree. Full build-server paths survive:

```text
D:\BuildServer\WoW\3\work\WoW-code\trunk\Engine\Source\Domino/...
```

with these headers named directly in assert strings:

| Domino area | Headers seen |
|---|---|
| `Common/` | `dmMath.h`, `dmBuffer.h`, `dmGrowableStack.h`, `dmTable.h`, `dmPool.h`, `dmArray.h` |
| `Dynamics/` | `dmIsland.h`, `Contacts/dmContact.h` |
| `Collision/BroadPhase/` | `dmDynamicTree.h` |
| `Collision/Primitives/` | `dmPolytope.h`, `dmPrimitives.h`, `dmMeshData.h`, `dmTreeMesh.h` |
| `Collision/Functions/` | `dmDistance.h` |
| `Collision/ConvexHull/` | `dmHullBuilder.h` |

Note the sibling directory: `Engine\Source\Physics/PhysData.h` sits **beside** `Engine\Source\Domino`,
not inside it. Domino is the general-purpose solver; `Physics*.cpp` / `PhysData` is WoW's own
adapter over it. Those are two different layers and must not be conflated when naming things.

### The assertion pivot

```text
"Domino Assertion: %s, in file %s, line %d\n"   @ 0x00e0ac8c
```

One data xref, from `FUN_00c29680` — the assertion handler. **Callers of `FUN_00c29680` are the
Domino internals**, and each call site passes its own file and line, so decompiling a caller yields
the originating header *and line number* for free. This is the highest-value entry point in the
subsystem and should be the first pass.

Measured: **~270 call sites across ~90 distinct functions**, occupying a contiguous region
approximately **`0x00c26e50` – `0x00c4e012`** (~160 KB of code). Two outliers at `0x005a3010` and
`0x005a40d0` sit far outside that range and are the likely WoW-side callers.

This count is a **floor, not the subsystem size**: only functions that assert appear here. Treat
~90 as the lower bound on Domino functions and the contiguous range as the search space.

### Runtime knobs

Console/CVar strings prove the system is switchable at runtime and give named behaviour to look for:

| String | Address |
|---|---:|
| `Enabling physics processing.` | `0x00d75990` |
| `Disabling physics processing.` | `0x00d75970` |
| `Enabling physics culling.` | `0x00d759cc` |
| `Disabling physics culling.` | `0x00d759b0` |
| `Physics culling dist set to %f.` | `0x00d759e8` |

A **culling distance** exists, which means the client does not simulate every physicalised object in
the world. Any reimplementation that simulates unconditionally will not match the client and will
not perform like it.

`TransportPhysics.dbc` (`0x00e6befc`, with strings `TransportPhysics` / `transportPhysics` at
`0x00da2314` / `0x00da2824`) is a separate data-driven path for transports.

### Followed up 2026-09-03 — the adapter and the sidecar format are solved

Spec 214 worked this section to conclusion. Full evidence lives in that spec; the results that belong
in this shared note:

- **The two outliers at `0x005a3010` / `0x005a40d0` are identified.** `FUN_005a3010` is world creation
  (`PhysicsInt.cpp:16`, asserts `s_world == 0`); `FUN_005a40d0` is the per-instance update. Both
  assert the Domino finite-float check `dmMath.h:127`, which is why `dmMath.h`'s string is pooled with
  the `Physics.cpp` strings rather than with the other Domino headers.
- **The model sidecar is `.phys`**, located by replacing the model filename's extension — no id and no
  lookup table. **There is no `.phys` string in the binary**; the extension is written as two
  immediates (`0x7968702e` then `0x73`), so a string search returns a false negative.
- **The container is fully decoded**: reversed-tag chunks, magic `PHYS`, version `u16` must be 0, nine
  record chunks with strides measured from the parser's own size divisions. Field names come from
  Blizzard's `PhysData.h` bounds asserts.
- **Gravity is `(0, 0, -10.0)`** written at world `+0x80` — Z-up, magnitude 10.0, not 9.81.
- **The culling distance CVar is `0x00EB6358`**, and culling is enforced by an early return in the
  per-instance update, not inside the solver.
- **The client pins x87 control word and MXCSR around physics work and restores them afterwards.**
  Anything reimplementing this needs an equivalent deterministic FP discipline.

Method note for anyone repeating this: the caller map was recovered from **header-string
cross-references**, not from ~90 decompilations. That is far cheaper, but it sees 58 named functions
where direct xrefs to `FUN_00c29680` see ~90 — the difference is call sites inside regions Ghidra has
not resolved into functions, plus hoisted string operands. Neither number is the subsystem size.

See [`214-mop-physics-domino/evidence/physics-adapter-contract.md`](../specs/archived/214-mop-physics-domino/evidence/physics-adapter-contract.md)
and [`domino-caller-map.md`](../specs/archived/214-mop-physics-domino/evidence/domino-caller-map.md).

## Atmosphere data chain — measured

DBC filenames present in the binary:

| Table | Address |
|---|---:|
| `DBFilesClient\Light.dbc` | `0x00e05f78` |
| `DBFilesClient\LightParams.dbc` | `0x00e0628c` |
| `DBFilesClient\LightData.dbc` | `0x00e06344` |
| `DBFilesClient\LightSkybox.dbc` | `0x00e05f14` |
| `DBFilesClient\ZoneLight.dbc` | `0x00e05890` |
| `DBFilesClient\ZoneLightPoint.dbc` | `0x00e0581c` |
| `DBFilesClient\Weather.dbc` | `0x00e059f0` |

`ZoneLight` + `ZoneLightPoint` are the pair that makes 5.0.1 different from the alpha-era model:
lighting is selected by **zone polygon**, not only by the radial `Light.dbc` position/falloff. That
is a structural change, not a tuning change, and it is the thing specs 143/160 most need to know.

## Era warning — read before implementing

**None of the above is evidence about 0.5.3.** This binary is the *complete* implementation; the
alpha client is not. Domino does not exist in 0.5.3, `LightData`/`ZoneLight` may not, and the fog
model differs. This project has already paid for era-blind decoding twice — see
[[project_mcnr_axis_order_wrong]] (MCNR component order is era-split, `(x,z,y)` in 0.5.3 vs
`(x,y,z)` in Cata+) and [[feedback_era_gate_minimap_generation]].

The agreed policy for anything built on this note: **decode from 5.0.1 because it is complete, then
era-gate the behaviour**, carrying provenance and flagging unknown builds — never assume a 5.0.1
mechanism was present earlier. Establishing what 0.5.3 actually did is separate work against the
0.5.3 binary, and the difference between the two is the finding, not an inconvenience.

## Current ownership

Since the 2026-09-23 Spec Kit reconciliation, this shared evidence feeds **Epic 249** (sky/fog/lighting)
and **Epic 252** (physics W-20, weather W-21). Specs 143, 147, 160, 214, and 215 are archived
historical owners; follow the current epic/spec registry for scheduled implementation work. This note
supplies evidence and does not authorize implementation by itself.

## Evidence boundary — subsystem-specific

- The physics adapter and `.phys` sidecar format have follow-up decode results recorded in the earlier
  physics section and archived Spec 214 evidence. It is incorrect to characterize this entire note as
  “not decompiled.”
- For the atmosphere/weather portion, the listed source anchors, strings, RTTI names, and DBC filenames
  are reconnaissance, not a complete decode. Record layouts, most field offsets, full update order,
  and viewer integration still require evidence in the owning spec.
- **None of the 5.0.1 findings establish 0.5.3 behavior.** The Alpha client needs its own binary
  evidence; carry build provenance and era-gate any implementation.
