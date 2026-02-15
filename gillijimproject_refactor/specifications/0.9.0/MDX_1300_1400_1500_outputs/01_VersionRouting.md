# Task 1 - Version routing and parse entry

## Version switch map

### A) Top-level section dispatch (tag -> handler)
- Function: `FUN_00784200`
- Role: dispatches MDX chunk tags to chunk handlers.
- Key routes:
  - `VERS` (`0x53524556`) -> `FUN_007abf40` at call site `0x00784447`
  - `SEQS` (`0x53514553`) -> `FUN_007a9c70` at call site `0x00784419`
  - `GEOS` (`0x534f4547`) -> `FUN_007a2a00` at call site `0x00784430`

### B) Parser profile dispatch
- Function: `FUN_00783f20`
- Role: dispatches parser profile IDs (`0x103..0x119`) to profile-specific parse routines.
- `0x103` path calls `FUN_007abdd0` (site `0x00783f53`).

### C) Version read/store/compare
- `VERS` chunk handler: `FUN_007abf40`
  - Reads version dword via `FUN_007f40a0()` and stores at `model + 0x3a8`.
  - Store site in disasm: `0x007abf88` (`MOV [param_3+0x3A8], EAX`).
- Profile parser: `FUN_007abdd0`
  - Reads `FormatVersion` property key `0x14c` and stores at `model + 0x3a8`.
  - Compare site: `0x007abeab` (`CMP EAX, 0x5DC`).
  - Predicate: reject if `version > 0x5DC` (`1500`).
  - Reject branch target: call error sink at `0x007abec2`, message `"File version ... newer than newest version ..."`.

## Findings relevant to 1300/1400/1500
- No direct split on `1300 (0x514)` vs `1400 (0x578)` vs `1500 (0x5DC)` was found in this parser path.
- Effective version policy in this binary: **allow <= 1500, reject > 1500**.
- Therefore observed 1400/1500 regressions are likely from fixed record-layout assumptions in chunk handlers, not an explicit version switch branch.
