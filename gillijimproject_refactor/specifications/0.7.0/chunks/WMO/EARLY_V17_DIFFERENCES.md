# Early v17 WMO (0.7.0.3694) vs Later v17 (1.12+) — Confirmed Differences

## Scope
This is a **behavior-first** delta sheet for the first known WMO v17-era implementation in build 0.7.0.3694.

The goal is to capture what is materially different from the commonly documented later-era v17 behavior.

## Confirmed differences in 0.7.0.3694

### 1) Version check still enforces `0x10`
- Root and group loaders both check version `0x10` (`FUN_006c11a0`, `FUN_006c17f0`).
- Practical implication: this implementation is an early transitional branch, not the later fully stabilized v17 reader used in much newer clients.

### 2) Split root/group file model is already active
- Groups are loaded from `_<index:03d>.wmo` filenames in `FUN_006c1570`.
- Group file format is validated as `MVER -> MOGP`, then required subchunks.
- This split behavior is central to streaming/visibility logic in this build.

### 3) MOGP behavior is portal-recursive, not just flat visibility
- Group graph expansion uses recursion (`FUN_006ab560`, `FUN_006ab730`) tied to portal arrays and per-group portal start/count-like fields.
- This demonstrates a strong portal-driven indoor visibility model in the early implementation.

### 4) Optional `MPB*` sequence is hard-gated in parser
- When `flags & 0x400`, parser enforces strict ordered sequence:
  `MPBV -> MPBP -> MPBI -> MPBG` (`FUN_006c1c60`).
- This sequence is often absent or treated differently in later public docs/toolchains.

### 5) Runtime setup is chunk-gated by explicit state bits
- Group state bits (`+0x0C`) track completion and routing:
  - `0x40` for `MOLR` light-link pass,
  - `0x20` for `MODR` doodad-link pass,
  - `0x08/0x10` classification from `MOGI.flags & 0x48`.
- Indicates early engine strongly couples parse flags to frame-time pass scheduling.

### 6) Liquid path is tightly integrated and sampled via group-local grids
- `MLIQ` fields are copied into `group+0xE0..0x104` and queried by `FUN_006a3d90`/`FUN_006a3e60`.
- Sampling uses grid dims/origin + per-tile flags + interpolated heights.
- This is operational runtime behavior, not merely passive data storage.

## Structural observations (confirmed in this build)

- `MOGP` has a 68-byte header before required subchunks (`FUN_006c17f0` jumps to `piVar1+0x16`).
- Required group sequence remains:
  `MOPY, MOVI, MOVT, MONR, MOTV, MOBA`.
- Optional gates remain strongly flag-controlled in parser and in runtime use.

## What this means for reverse-engineering/spec work

- Do not treat this build as equivalent to “classic 1.12+ v17” docs.
- Tooling should model this as **early-v17 transitional behavior** with:
  - v10 version check,
  - split-group streaming,
  - portal-recursive visibility,
  - strict optional chunk gates,
  - and chunk-driven runtime pass bits.

## Confidence

- All differences listed above are grounded in direct 0.7.0.3694 decompilation.
- Unknowns are limited to naming of some non-critical metadata fields, not to the behavior-level conclusions.