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

### Versions pre-256 — Alphas 1.0.0 / 0.12 / 0.11 — Status: open (P3 target, x64dbg)

- Skin storage: embedded (layout partly undocumented)
- Header offsets used: **(recover via x64dbg trace per version)**
- Skin-profile struct deltas: **(recover P3)**
- Evidence: _pending_ — `x64dbg-trace:<ref>` + target model(s) per staged alpha client

### Version ≥ 264 — WotLK+ (baseline, already working) — Status: confirmed (do not regress)

- Skin storage: external `.skin`
- The current reader path. Recorded here so regressions are visible against a spread of 264+ models.
