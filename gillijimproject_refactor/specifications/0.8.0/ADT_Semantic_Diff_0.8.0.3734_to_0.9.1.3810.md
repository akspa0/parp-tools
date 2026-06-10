# ADT Semantic Diff — 0.8.0.3734 → 0.9.1.3810

## Scope
Field/behavior differences for ADT-focused parsing between builds `0.8.0.3734` and `0.9.1.3810` based on Ghidra extraction notes.

## High-Level Result
Core ADT architecture is stable across both builds (same MHDR offset family, same MCNK subchunk ecosystem), but `MCLQ` grows materially in 0.9.1 and flow data handling becomes more explicit.

---

## 1) Stable between both builds

### MHDR root offsets
Both builds use the same practical MHDR offset chain for:
- `MCIN`, `MTEX`, `MMDX`, `MMID`, `MWMO`, `MWID`, `MDDF`, `MODF`.

`MHDR + 0x00` remains effectively opaque/not consumed directly in the analyzed create path in both notes.

### MCNK count semantics
Both builds clearly treat these as counters:
- `+0x10` doodad refs
- `+0x38` mapobject refs
- `+0x5C` sound emitter count (`MCSE` loop bound)

### MCNK liquid slot gating
Both builds gate up to four liquid slots with flags:
- `0x04`, `0x08`, `0x10`, `0x20`.

---

## 2) Meaningful changes

### MCLQ per-layer stride
- **0.8.0**: `0x2D4` bytes per liquid layer.
- **0.9.1**: `0x324` bytes per liquid layer.

Delta: `+0x50` bytes in 0.9.1.

### Flow data representation
- **0.8.0**: flow/special behavior is present but inferred through mode-driven consumers; parser stores a mode scalar and pointer chain with less explicit typed conversion.
- **0.9.1**: parser explicitly handles `SWFlowv[2]` (typed conversion call) and a clearer post-`0x2D0` flow block interpretation.

Interpretation: 0.9.1 formalizes/extends flow payload while preserving older tile+height fundamentals.

### MCLQ scalar naming confidence
- **0.8.0**: early layer scalars (`+0x000/+0x004`) remain semantically unresolved.
- **0.9.1**: stronger evidence supports min/max liquid Z interpretation tied to liquid AABox behavior.

---

## 3) Practical parser impact

### If writing a multi-build ADT reader
- Keep a **versioned MCLQ profile**:
  - 0.8.0 profile: `stride=0x2D4`
  - 0.9.1 profile: `stride=0x324`
- Keep common logic reusable:
  - 9x9 sample lattice (height at sample `+4`)
  - 8x8 tile map semantics
  - same slot gating bits
- Treat unresolved scalar lanes and select mode bits as build-specific optional metadata.

---

## 4) Confidence
- **High**: stride change, gating parity, count-field parity, tile/height fundamentals.
- **Medium**: exact one-to-one semantic mapping for every scalar in the early 0.8.0 MCLQ header.
