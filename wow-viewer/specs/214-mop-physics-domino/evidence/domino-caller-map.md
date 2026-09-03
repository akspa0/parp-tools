# T003 — Domino Assertion Caller Map

**Method**: read-only Ghidra session against program `/Wow.exe` in project
`Mists of Pandaria 5.0.1.15464`, PE x86, image base `0x00400000`, 38,405 functions. No bytes,
symbols, labels, comments, types or analysis settings were changed.

**Scope rule for this file**: attribution only — which functions assert, and from which Domino header
and line. **No Domino algorithm is decoded, described, or transcribed here.** The WoW-side adapter is
recorded separately in [physics-adapter-contract.md](physics-adapter-contract.md).

## The pivot, and a correction to the method

The workstream note proposed decompiling callers of the assertion handler `FUN_00c29680` to recover
each call site's file and line. That works but costs ~90 decompilations.

A cheaper and equally sound pivot was used instead: **each asserting function references its own
header path string as a call operand**, so `get_xrefs_to <header string>` yields the
function-to-header mapping directly, with no decompilation.

**The two methods do not return the same population, and the difference is a finding, not noise.**

| Method | Call sites | Distinct named functions |
|---|---:|---:|
| Direct xrefs to `FUN_00c29680` (workstream, 2026-09-02) | ~270 | ~90 |
| Header-string xrefs (this pass) | 199 | **58 named + 2 WoW-side** |

The header-string pivot under-counts because it only sees sites where the path string is a *direct*
operand. It misses (a) call sites in regions Ghidra has not resolved into a function — 25 such sites
were returned with an address but no containing function, clustered at `0x00c35005`–`0x00c3653a`,
`0x00c41145`–`0x00c41251` and `0x00c445d5`–`0x00c446e1` — and (b) any site whose string operand is
hoisted into a register by the compiler.

**Neither number is the subsystem size.** Only functions that assert are visible to either pivot.
~90 remains the floor and the contiguous range `0x00c26e50`–`0x00c4e012` remains the search space.

## Header inventory — measured

All 15 Domino headers named in assert strings, with the address of each path string.

| Domino area | Header | String address | Distinct asserting functions |
|---|---|---:|---:|
| `Common/` | `dmMath.h` | `0x00d75a88` | 19 (+2 WoW-side) |
| `Common/` | `dmBuffer.h` | `0x00e0a878` | 14 |
| `Common/` | `dmArray.h` | `0x00e0b6a0` | 6 |
| `Common/` | `dmTable.h` | `0x00e0af88` | 1 |
| `Common/` | `dmPool.h` | `0x00e0b048` | 1 |
| `Common/` | `dmGrowableStack.h` | `0x00e0a8c8` | 1 |
| `Dynamics/` | `dmIsland.h` | `0x00e0a690` | 2 |
| `Dynamics/Contacts/` | `dmContact.h` | `0x00e0bb68` | 2 |
| `Collision/BroadPhase/` | `dmDynamicTree.h` | `0x00e0ac28` | 1 |
| `Collision/Primitives/` | `dmPolytope.h` | `0x00e0a758` | 4 |
| `Collision/Primitives/` | `dmPrimitives.h` | `0x00e0ad88` | 2 |
| `Collision/Primitives/` | `dmMeshData.h` | `0x00e0d388` | 2 |
| `Collision/Primitives/` | `dmTreeMesh.h` | `0x00e0d428` | 3 |
| `Collision/Functions/` | `dmDistance.h` | `0x00e0c088` | 1 |
| `Collision/ConvexHull/` | `dmHullBuilder.h` | `0x00e0c280` | 2 |

Assertion format string: `"Domino Assertion: %s, in file %s, line %d\n"` at `0x00e0ac8c`, one data
xref, from the handler `FUN_00c29680`.

**Note the string-region split.** `dmMath.h` sits at `0x00d75a88`, inside the *`Physics.cpp` /
`PhysData.cpp` string region* (`0x00d7592c`–`0x00d75dbc`), while every other Domino header sits in
`0x00e0a*`–`0x00e0d*`. That is consistent with `dmMath.h` being the one Domino header included by
WoW's own adapter translation units — and it is confirmed independently below, because two WoW-side
functions assert against it.

## Function-to-header attribution — measured

Addresses are containing functions of the string reference.

**`dmMath.h`** — `FUN_00c284e0`, `FUN_00c29a10`, `FUN_00c29aa0`, `FUN_00c29b70`, `FUN_00c29c90`,
`FUN_00c29e90`, `FUN_00c2b630`, `FUN_00c2b7a0`, `FUN_00c2ba60`, `FUN_00c2cda0`, `FUN_00c306c0`,
`FUN_00c312b0`, `FUN_00c31b50`, `FUN_00c346b0`, `FUN_00c40140`, `FUN_00c41c90`, `FUN_00c42fb0`,
`FUN_00c45c80`, `FUN_00c46500`, **`FUN_005a3010`**, **`FUN_005a40d0`**

**`dmBuffer.h`** — `FUN_00c2a5b0`, `FUN_00c3ce90`, `FUN_00c3cf00`, `FUN_00c3cf90`, `FUN_00c3d020`,
`FUN_00c3d0a0`, `FUN_00c3d2b0`, `FUN_00c3d360`, `FUN_00c3d710`, `FUN_00c3e9b0`, `FUN_00c4b4f0`,
`FUN_00c4d710`, `FUN_00c4d780`, `FUN_00c4ddd0`

**`dmArray.h`** — `FUN_00c2cf20`, `FUN_00c2cfd0`, `FUN_00c2d030`, `FUN_00c2e800`, `FUN_00c3da60`,
`FUN_00c4da60`

**`dmTable.h`** — `FUN_00c2a500`  ·  **`dmPool.h`** — `FUN_00c2aac0`  ·
**`dmGrowableStack.h`** — `FUN_00c274f0`  ·  **`dmDynamicTree.h`** — `FUN_00c2df60`  ·
**`dmDistance.h`** — `FUN_00c374f0`

**`dmIsland.h`** — `FUN_00c27db0`, `FUN_00c284e0`

**`dmContact.h`** — `FUN_00c306c0`, `FUN_00c47e60`

**`dmPolytope.h`** — `FUN_00c26e50`, `FUN_00c4fc10`, `FUN_00c50490`, `FUN_00c529e0`

**`dmPrimitives.h`** — `FUN_00c29e90`, `FUN_00c30210`

**`dmMeshData.h`** — `FUN_00c4c300`, `FUN_00c4c330`

**`dmTreeMesh.h`** — `FUN_00c4c3d0`, `FUN_00c4c4f0`, `FUN_00c4c880`

**`dmHullBuilder.h`** — `FUN_00c3bbc0`, `FUN_00c3ca20`

Three functions assert against two headers each — `FUN_00c284e0` (`dmMath` + `dmIsland`),
`FUN_00c29e90` (`dmMath` + `dmPrimitives`), `FUN_00c306c0` (`dmMath` + `dmContact`) — which is why
the per-header counts sum to 61 while the distinct total is 58.

## The two WoW-side callers — measured, and identified

The workstream flagged `0x005a3010` and `0x005a40d0` as "outliers … the likely WoW-side callers".
**Confirmed, and both are now identified**; they are the reason `dmMath.h`'s string is pooled with
the `Physics.cpp` strings.

| Function | Role | Evidence |
|---|---|---|
| `FUN_005a3010` | Physics world creation | asserts `s_world == 0` at `PhysicsInt.cpp:16` |
| `FUN_005a40d0` | Per-instance update / model-driven step | asserts twice against `dmMath.h:127` mid-update |

Both assert the same Domino predicate, `v.F32Check() == teMATH_F32CHECK_RESULT_OK` at
**`dmMath.h` line 0x7f (127)**. That is a **finite-float validation on every vector the adapter hands
to Domino** — an observable input-validation contract at the layer boundary, and the single most
directly reusable behaviour in this file. Our port should validate vectors at the same boundary
rather than trusting model-derived transforms.

Full adapter detail is in [physics-adapter-contract.md](physics-adapter-contract.md).

## What this file deliberately does not contain

No Domino function was decompiled for this map. The mapping is address-and-attribution metadata
recovered from string cross-references. Reproducing the algorithms behind these asserts is out of
scope for spec 214 by constitution and by operator direction: the solver will be an independently
licensed C# library, and Domino is decoded only as a data and behaviour contract.
