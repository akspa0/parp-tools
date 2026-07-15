# Contract: Per-version M2 format profile

This is the durable, cumulative research deliverable (spec FR-006). Each in-scope M2 format version gets
one profile entry recording its **confirmed** header offsets and embedded-skin structure plus the
**evidence** that established them (a documentation reference for P1/P2, or an x64dbg trace for P3). The
reader consumes this; the phases populate it. An entry is only "confirmed" when a real model of that
version renders correctly (P1/P2) or the layout is trace-verified (P3).

## Profile entry schema

Each entry MUST record:

- **Version**: the `uint32` at header `0x04` (and the client builds that emit it).
- **Skin storage**: `embedded` (≤ 263) or `external .skin` (≥ 264).
- **Header offsets used**: view count offset, view offset offset, and any field whose position differs
  from the WotLK baseline in the current reader.
- **Skin-profile struct**: the field order/sizes of the embedded view header (index/tri/prop/submesh/
  texunit count+offset pairs).
- **Submesh struct** and **texture-unit struct**: field order/sizes for this version.
- **Evidence**: `wowdev.wiki:<section>` / `reference-impl:<name>` / `x64dbg-trace:<notes-ref>` — and the
  specific staged-client model(s) that validated it (path under `output/tmp/wowarchive-clients/`).
- **Status**: `confirmed` | `assumed-from-sibling` | `blocked (no staged client)` | `open`.

## Reader output contract (version-independent)

Regardless of version, `M2ModelReader` MUST emit, for a successfully-parsed model, geometry in the shape
the existing render path already consumes:

- a set of submeshes, each with a resolved triangle index range over the global vertex array, and
- a texture/material binding per submesh.

For ≤ 263 this comes from the embedded skin profile (LOD 0); for ≥ 264 from the external `.skin`
(unchanged). Consumers (`M2Renderer`, exporters) MUST NOT need to know which source was used — the
version branch lives entirely inside the reader. This preserves the single-canonical-owner rule
(constitution II) and the no-regression constraint (FR-009): the ≥ 264 path emits the same contract it
does today.

## Profile entries

> Populated by Phases 1–3. Template rows below; fill as each version is confirmed.

### Version 256 — Classic 1.x (incl. 1.12.1) — Status: open (P1 target)

- Skin storage: embedded
- Header offsets used: view count `0x44` (existing) / view offset `0x48` **(confirm P1)**
- Skin-profile / submesh / texunit structs: **(confirm P1)**
- Evidence: _pending_ — target model(s): _pending staged 1.12.1 client_

### Versions ~257–263 — TBC 2.0–2.4.3 — Status: open (P1 target: 2.4.3; P2: 2.0.0α/2.1/2.2/2.3)

- Skin storage: embedded
- Header offsets used: **(confirm P1 for 2.4.3, P2 for the rest)**
- Deltas vs 256: **(record per version)**
- Evidence: _pending_ — target model(s): _pending staged 2.4.3 client_

### Version 256 (0x100) — WoW 1.0.0 (Vanilla release) — Status: confirmed (static, Ghidra)

- Skin storage: **embedded** — no external `.skin` or `.anim` files exist on 1.0.0
  (string-sweep: zero `.skin`/`%02d.skin`/`.anim` hits).
- Version field: header `0x04` == `0x100` (parser `FUN_0071e190` hard-rejects `!= 0x100`
  → `Corrupt model data`). **Corrects `research.md` line 28**: 1.0.0 is 0x100, NOT
  pre-256. This is the 0.12→1.x boundary: 0.12 MDX is pre-0x100 and is rejected by 1.0.0.
- Header offsets used (full map in `research-1.0.0-ghidra-trace.md` §4): bones `0x34`
  (0x6c), vertices `0x44` (0x30), **divisions `0x4C` (0x2c)** = embedded skin profiles,
  textures `0x5C` (0x10), attachments `0x104` (0x30), events `0x114` (0x2c),
  lights `0x11C` (0xd4), cameras `0x124` (0x7c), ribbons `0x134` (0xdc),
  particles `0x13C` (0x1f8). Sequences `0x1C` (0x44), sequenceLookup `0x24` (int16[]).
- Skin-profile struct (M2Division 0x2c): vertexLookup (int16[]) @0x04, indices
  (int16[]) @0x0C, uint32[] @0x14, sections (0x20 B) @0x1C, batches (0x18 B) @0x24
  (`division->batches.count == 1`). Materializer `FUN_006b7720` builds 0x20-B render
  vertices by remapping through vertexLookup into the 0x30-B global vertex table.
- Evidence: `ghidra-static-trace:research-1.0.0-ghidra-trace.md` + raw decompilations
  `output/ghidra_1.0.0/*.c`. Target model(s): _pending staged 1.0.0 client render check_.
- Status: **confirmed (static layout)**; render validation pending a staged 1.0.0 client.

### Versions pre-256 — Alphas 0.12 / 0.11 — Status: open (P3 target)

- Skin storage: embedded (layout undocumented; distinct from 0x100).
- Header offsets used: **(recover via Ghidra/x64dbg trace of the 0.12 client)**
- Note: 0.11/0.12 MDX files use a pre-`0x100` version. The wow-viewer reader already
  handles these (they render fine); the viewer gap is **1.x+ (`0x100` → 3.0.1)**, not
  pre-`0x100`. The 1.0.0 game-client parser rejects `!= 0x100` as `Corrupt model data`
  — a fact about the native client, not the viewer. Design: accept any version, dispatch
  to per-version codepaths.
- Evidence: _pending_ — needs 0.12 `WoW.exe` loaded in Ghidra.

### Version ≥ 264 — WotLK+ (baseline, already working) — Status: confirmed (do not regress)

- Skin storage: external `.skin`
- The current reader path. Recorded here so regressions are visible against a spread of 264+ models.
