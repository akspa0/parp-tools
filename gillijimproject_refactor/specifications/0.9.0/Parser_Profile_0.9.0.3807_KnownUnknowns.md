# Parser Profile 0.9.0.3807 — Known Unknowns (Deep Dive)

## Purpose
Track unresolved behaviors that still block deterministic viewer compatibility for `0.9.0.3807`.

---

## 1) ADT Known Unknowns

## ADT-3807-U1: MH2O secondary path existence
- Current evidence:
  - Primary ADT chain (`FUN_006e6220` -> `FUN_006e72e0` -> `FUN_006d7130` -> `FUN_006d7590`) is strict `MCLQ`-centric.
  - No direct `MH2O` string/token evidence surfaced in this pass.
- Unresolved:
  - Whether a dormant or map-conditional MH2O parser exists elsewhere.
- Proof targets:
  - constant search/xref for `MH2O` token value in code/data,
  - any root offset slot consumption beyond known MHDR fields,
  - liquid fallback functions outside `FUN_006d7380` + `FUN_006b1200` chain.
- Impact severity: **visual artifact** (missing/wrong liquids).
- Confidence now: **Medium**.

## ADT-3807-U2: MCNK optional-subchunk tolerance
- Current evidence:
  - `FUN_006d7590` hard-asserts the strict required set.
- Unresolved:
  - Whether optional chunks are consumed in post-CreatePtrs routines for this build.
- Proof targets:
  - callers of `FUN_006d7130` and downstream post-load functions touching MCNK payload beyond required offsets.
- Impact severity: **hard parse fail** (if strictness mismatched).
- Confidence now: **Medium-High**.

---

## 2) WMO Known Unknowns

## WMO-3807-U1: `MLIQ` semantic naming of header words
- Current evidence:
  - `FUN_006e8960` copies `MLIQ` words into `group+0xF0..0x10C`, sample base at `+0x26`, stride `0x08`.
- Unresolved:
  - exact semantic labels (width/height/material/type/min/max/flags naming map).
- Proof targets:
  - xrefs from group fields `+0xF0..+0x114` into render/query consumers.
- Impact severity: **visual artifact**.
- Confidence now: **Medium-High**.

## WMO-3807-U2: `MPB*` chain runtime semantics
- Current evidence:
  - strict optional sequence present when `flags & 0x400`.
- Unresolved:
  - whether decoded data affects portal visibility only, collision, or draw-culling.
- Proof targets:
  - xrefs from structures filled in `FUN_006e8960` to traversal/culling paths.
- Impact severity: **perf only** to **visual artifact**.
- Confidence now: **Medium**.

---

## 3) MDX Known Unknowns

## MDX-3807-U1: Required-vs-optional matrix per dispatcher path
- Current evidence:
  - dispatcher `FUN_0042a6a0` and two section-read paths are recovered.
  - many section seekers (`TEXS`, `GEOS`, `MTLS`, `ATCH`, `PRE2`, `SEQS`, `PIVT`, `RIBB`, `LITE`, `CAMS`, `CLID`, `GEOA`).
- Unresolved:
  - authoritative required/optional matrix for each path and load-flag mode.
- Proof targets:
  - branch conditions around each call in `FUN_0042a6a0` and `FUN_0042b4d0` with fail behavior map.
- Impact severity: **hard parse fail**.
- Confidence now: **Medium**.

## MDX-3807-U2: Keyframe interpolation/compression semantics
- Current evidence:
  - sequence and shape sections are parsed (`SEQS`, `HTST`), with variant decode switches in `FUN_0042a910`.
- Unresolved:
  - final interpolation/compression policy needed for profile fields.
- Proof targets:
  - runtime animation application consumers of sequence/key arrays,
  - any decode-mode flags controlling quaternion/rotation compression.
- Impact severity: **visual artifact** (animation corruption).
- Confidence now: **Low-Medium**.

## MDX-3807-U3: TEXS replaceable/UV/wrap behavior
- Current evidence:
  - strict `TEXS` divisibility by `0x10C` proven in multiple loaders.
- Unresolved:
  - replaceable texture and UV/wrap policy by build mode.
- Proof targets:
  - material-layer consumer path linking `TEXS` records to shader/material setup.
- Impact severity: **visual artifact**.
- Confidence now: **Medium**.

---

## 4) Priority closure plan (implementation-focused)

1. Resolve ADT MH2O existence (token/consumer proof).
2. Resolve WMO MLIQ field naming via consumer xrefs.
3. Build MDX per-path required/optional section matrix from dispatcher branches.
4. Close MDX interpolation/compression behavior from animator-side consumption.
