# Parser Profile 0.9.1.3810 — Known Unknowns

## Purpose
Track unresolved fields/behaviors with proof targets and impact severity, per playbook Template C.

---

## 1) ADT Known Unknowns

## ADT-U1: MCNK optional subchunk tolerance beyond strict set
- Current evidence:
  - `0x002973c4` `CMapChunk::CreatePtrs` hard-asserts expected subchunks (`MCVT/MCNR/MCLY/MCRF/MCSH/MCAL/MCLQ/MCSE`).
- Unresolved:
  - Whether alternate/extra optional subchunks are consumed in other code paths for this exact build.
- Candidate interpretations:
  - A) strict-only set is complete
  - B) additional optional chunks handled post-pointer-setup in other routines
- Required proof functions:
  - `CMapChunk::Create`, `CreateLayer`, any additional `MCNK` post-processors.
- Impact severity: **parser break** (if assumptions are wrong).
- Confidence now: **Medium**.

## ADT-U2: MH2O secondary path existence
- Current evidence:
  - No strong active MH2O parse path observed in primary ADT terrain flow.
- Unresolved:
  - Whether any secondary/unused path in this binary can parse MH2O for some maps.
- Required proof:
  - Xref-scan for MH2O token constants and any function with MHDR offset slot that maps to MH2O.
- Impact severity: **visual artifact** (liquid missing/wrong), possibly parser fallback noise.
- Confidence now: **Medium**.

---

## 2) WMO Known Unknowns

## WMO-U1: `MLIQ` header field semantics
- Current evidence:
  - `0x002a1c4c` parses `MLIQ`, decodes vectors/fields, uses mode bit from mask block.
- Unresolved:
  - Exact semantic names for fields at group offsets `+0xF0..+0x10C` (material/type/depth/flags map).
- Candidate interpretations:
  - A) dimensions + origin + material/type id
  - B) dimensions + min/max level + mode/flags
- Required proof:
  - downstream liquid render/query functions consuming `CMapObjGroup` fields `+0xF0..+0x114`.
- Impact severity: **visual artifact**.
- Confidence now: **Medium-High**.

## WMO-U2: `MPB*` optional block semantics
- Current evidence:
  - `0x002a1c4c` gate `flags & 0x400` requires `MPBV/MPBP/MPBI/MPBG` sequence.
- Unresolved:
  - Exact runtime use of parsed block in culling/portal traversal.
- Required proof:
  - xrefs from stored pointers to portal visibility/culling routines.
- Impact severity: **perf only** to **visual artifact**.
- Confidence now: **Medium**.

---

## 3) MDX Known Unknowns

## MDX-U1: Full top-level section dispatcher order
- Current evidence:
  - Individual readers confirmed (`TEXS`, `GEOS`, `LITE`, `CAMS`, ribbon readers).
- Unresolved:
  - Canonical required-vs-optional load order for all major sections in 0.9.1 binary path.
- Required proof:
  - dispatcher routine that calls `ReadBin*` family in sequence.
- Impact severity: **parser break**.
- Confidence now: **Low-Medium**.

## MDX-U2: Sequence/keyframe compression contract
- Current evidence:
  - sequence/global sequence reader functions and checks exist.
- Unresolved:
  - exact interpolation/compression/rotation handling policy required for profile fields.
- Required proof:
  - read/decode functions for keyframe blocks + runtime animation application calls.
- Impact severity: **visual artifact** (animation corruption).
- Confidence now: **Low-Medium**.

## MDX-U3: Texture replaceable/UV-wrap policy by build
- Current evidence:
  - `TEXS` record-size and decode path confirmed.
- Unresolved:
  - replaceable texture rules and UV wrap semantics for this exact pre-release build.
- Required proof:
  - texture-layer decode and render material application functions.
- Impact severity: **visual artifact**.
- Confidence now: **Low-Medium**.

---

## 4) Cross-Build (0.9.0.x) Unknowns

## X-U1: 0.9.0.x parity with 0.9.1.3810
- Current evidence:
  - none from this session (only 0.9.1 binary currently loaded).
- Unresolved:
  - whether WMO group gates/MLIQ and MDX contracts drift between 0.9.0 and 0.9.1.
- Required proof:
  - repeat same function-level extraction on a specific 0.9.0.x binary.
- Impact severity: **parser break** (if incorrectly assumed identical).
- Confidence now: **Low**.

---

## 5) Immediate Next Proof Targets (priority order)

1. Locate and decompile top-level MDX `ReadBin*` dispatcher (resolve MDX-U1).
2. Trace WMO `MLIQ` consumer path (resolve WMO-U1).
3. Acquire and run exact same pass on one `0.9.0.x` binary (resolve X-U1).
