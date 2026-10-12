# Contract: M2 Era Profile

**Feature**: 105-format-version-profiles | **Date**: 2026-07-15

This is a **library contract**, not a network API — the "consumers" are core readers and the runtime
sampler. It defines the single canonical answer to *"what layout does this M2 use, and how do I know?"*

Related: spec 104's [`contracts/m2-format-profile.md`](../../104-legacy-m2-rendering/contracts/m2-format-profile.md)
documents the 1.0.0 on-disk format itself. **That file and this one must be updated together** when a
layout fact changes (Constitution: Spec Docs Are Source of Truth).

---

## The era table

| Era | Version field | Sequence stride | Time base | Track addressing | Track stride | Evidence |
|---|---|---|---|---|---|---|
| `Era100` | `0x100` | `0x44` | `StartEnd` | `FlatWithRanges` | `0x1C` | Ghidra `FUN_0070f6d0`, `FUN_0070f960`, `FUN_0071f440` (WoW.exe 1.0.0.3980) |
| `Era1121` | `0x100` | `0x6C` | `StartEnd` **(PROVISIONAL)** | `FlatWithRanges` **(PROVISIONAL)** | `0x1C` **(PROVISIONAL)** | No 1.12.1 client traced — inferred from era-100 + wowdev `≤ BC` |
| `ThreeX` | `0x108` | `0x40` | `Duration` | `Nested` | `0x14` | Working reader + wowdev |
| `FourX` | `0x109` | `0x40` | `Duration` | `Nested` | `0x14` | Working reader + wowdev |
| `Mdlx` | n/a (MDLX) | n/a | n/a | n/a | n/a | Chunked path, unaffected |

**`Era100` and `Era1121` share version field `0x100`.** This is the entire reason era resolution is
non-trivial and why the current code trial-parses. See Resolution below.

---

## Evidence citation rule (FR-011)

Every fact in the table above carries one of:

- a **Ghidra address** + the binary it came from,
- a **wiki/reference** citation, or
- `PROVISIONAL: <reason>`.

A provisional fact MUST be distinguishable **at the point of use**, not only in a comment. Rationale:
every bug fixed in this area to date was caused by a guessed layout that read like a known one.

**Currently provisional and why**: the entire `Era1121` row. Its sequence field offsets are suspected
wrong for the same reason era-100's were (Wrath offsets under a non-Wrath stride — research R1), but
**no 1.12.1 client has been traced**. It is **not** corrected by analogy: era-100 and era-1121 already
proved they diverge despite sharing a version field, so analogy is precisely the reasoning that
produced the bug.

> **Open question for the user**: 1.12.1 models may be animating wrongly today for this exact reason,
> and nobody has reported it. Worth an explicit check.

---

## Resolution contract

**Input**: model bytes + optional build hint. **Output**: exactly one `M2EraProfile`, or a loud failure.

```
1. magic != MD20 and != MDLX          -> FAIL, naming the magic
2. magic == MDLX                      -> Mdlx
3. version == 0x108                   -> ThreeX
4. version >= 0x109                   -> FourX
5. version == 0x100:
     a. build hint supplied           -> Era100 or Era1121   [deterministic]
     b. structural discriminator      -> Era100 or Era1121   [deterministic, PROVISIONAL]
     c. neither resolves              -> FAIL, naming version + ambiguity
6. otherwise                          -> FAIL, naming the version   (e.g. 2.x, tracked under spec 049)
```

**Guarantees**

- **No trial-parse.** Resolution never attempts a parse and infers from failure (FR-012).
- **Fails closed.** An unresolvable model raises, naming what was unrecognized. It never falls through
  to another era's layout (FR-013).
- **Core never references the viewer.** The build hint is supplied *by the caller* (the viewer already
  holds `_dbcBuild`). Dependency inversion — research R4; Constitution I.

**Why the discriminator is not the trial-parse renamed**: a trial-parse attempts a full parse and
infers from failure — it **fails open**, misrouting whenever a wrong-layout parse happens to produce
plausible offsets. A structural discriminator reads one declared field and decides — it **fails
closed**. Step 5c is the difference.

**The discriminator is PROVISIONAL** (research R4 step 2): the 1.0.0 header carries one more array
across `0x74–0xAC` than v256, so the two headers' sizes should differ by 8 bytes, observable as a
different first-data offset. **This is reasoned, not measured.** It must be confirmed against a real
1.0.0 and a real 1.12.1 model (plan Phase 4 step 3) before being relied on. Until then the build hint
is the only proven path.

---

## Consumer contract

| Consumer | May rely on | Must not |
|---|---|---|
| Era readers | profile layout facts | hardcode a stride the profile owns |
| `M2TrackSampler` | `Addressing`, `Start`, `Duration` | sniff bytes to infer era (FR-002) |
| Viewer | supplying a build hint | reach into the era table to re-decide |

**Backward compatibility**: the build hint is **optional**. Existing core callers and focused tests
compile and behave unchanged (they fall to 5b). Making it mandatory was rejected — it breaks every
caller for a case the discriminator usually handles (research R4).

---

## Out of scope

- ADT, WMO, MDX profile surfaces. `FormatProfileRegistry` remains their canonical owner (FR-018/020).
  Its **inert** `M2Profile` records are deleted; its Adt/Wmo/Mdx halves carry real varying values
  inside the Constitution's Terrain Alpha Risk Area and are not touched.
- `WarcraftNetM2Adapter` / the legacy MDX fallback (FR-017), except removing the validation call left
  dangling by the `M2Profile` deletion — pending user confirmation (research R6).
