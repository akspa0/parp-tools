# Task 4 - GEOS subchunk length semantics

## Evidence summary
- GEOS handler: `FUN_007a2a00` (dispatch from `FUN_00784200`).
- Vertex/normal subparser pattern: `FUN_0079f3e0`.
  - Reads tag via `FUN_007f4160`.
  - Reads length/count via `FUN_007f40a0`.
  - Uses count directly for allocation and read loops.
- Triangle index pattern in `FUN_00789d10`:
  - Reads `TRI ` tag, then count `uVar2`.
  - Reads exactly `uVar2` indices (`FUN_007f4320` + byte advance `uVar2*2`).

## Table
| Version | Subchunk | DeclaredLengthInterpretation | ElementStride | DerivedCount | EvidenceAddress |
|---|---|---|---:|---|---|
| 1300 | VRTX | elements (not bytes) | 12 | `count = len` | `FUN_0079f3e0` (`*param_5 += uVar2*0xC`) |
| 1400 | VRTX | elements (same parser path) | 12 | `count = len` | same |
| 1500 | VRTX | elements (same parser path) | 12 | `count = len` | same |
| 1300 | NRMS | elements | 12 | `count = len` | `FUN_0079f3e0` |
| 1400 | NRMS | elements | 12 | `count = len` | same |
| 1500 | NRMS | elements | 12 | `count = len` | same |
| 1300 | PVTX / TRI | elements | 2 | `count = len` | `FUN_00789d10` (`local_8 += uVar2*2`) |
| 1400 | PVTX / TRI | elements | 2 | `count = len` | same |
| 1500 | PVTX / TRI | elements | 2 | `count = len` | same |
| 1300 | GNDX / MTGC / MATS / UV* | not proven in current pass | - | - | additional disassembly needed |
| 1400 | GNDX / MTGC / MATS / UV* | not proven in current pass | - | - | additional disassembly needed |
| 1500 | GNDX / MTGC / MATS / UV* | not proven in current pass | - | - | additional disassembly needed |

## Key conclusion
- For proven core paths (VRTX/NRMS/PVTX), this binary interprets declared lengths as **element counts**, not byte lengths, under the same handler code for 1300/1400/1500.
