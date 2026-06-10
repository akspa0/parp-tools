# MDX / M2 Format Delta — 0.9.0 vs Earlier (0.5.3 / 0.8.0)

## Purpose
Reference for engineers working on 0.9.0-era clients/tools: what changed in MDX (MDLX container) and early MD20 (M2) handling compared to older 0.5.3/0.8.0 builds. Use this to gate loaders/converters and renderer expectations.

## High-level shifts
- MDL/MDX is now the dominant binary container (`MDLX` magic) with stricter parsing; text/looser MDL paths are still present but separated.
- Early MD20 (version 0x100) exists concurrently; conversion to MDLX is expected for unified rendering.
- Section landscape is richer (ribbons, helper nodes, particle v2, VANS/PERP) and size accounting is strict.
- Error handling flows through callbacks, not global logging.

## MDLX container changes
- **Magic enforcement:** `MDLX` required; non-MDLX rejected with fatal callback.
- **Strict size accounting:** Each section’s declared size must fit in remaining bytes; total consumption must not exceed file size. Overrun triggers fatal error (`"MDLFile overran total file size."`). Older builds often streamed until EOF without explicit bounds.
- **Dispatcher coverage:** Section tags are enumerated and dispatched; unknown tags are warned and skipped (payload is consumed). This is new vs older silent-accept behavior.

### Section set (0.9.0 vs earlier)
Present and enforced in 0.9.0 binary path:
- `MODL` (root)
- `SEQS` (animation sequences)
- `MTLS` (materials)
- `TEXS` (textures)
- `GEOS` (geosets)
- `GEOA` (geoset anim)
- `BONE` (bones)
- `PIVT` (pivots)
- `HTST` (hit/shape)
- `CLID` (collision)
- `CAM3` (cameras)
- `LITE` (lights)
- `ATCH` (attachments)
- `RIBB` (ribbon emitters)
- `PRE2` (particle emitters v2)
- `HELP` (helpers)
- `VANS` (version/anim state block; handler `FUN_007abf40`)
- `PERP` (particle/geometry-related; handler `FUN_007990e0`)

Notes vs 0.5.3/0.8.0:
- RIBB, HELP, PRE2, VANS, PERP are enforced and parsed; some were absent or stubbed in earlier builds.
- TEXS/GEOS/MTLS still resemble Warcraft III-era MDX but are validated more strictly (size divisibility expected per handler; see below).

### Parser/allocator behavior
- **Allocator:** Files are slurped into `(size + 1)` buffers via `FUN_007839c0`/`FUN_00783be0`; freed with `FUN_00783c60` (pointer-4). Keep this convention if rehosting.
- **Callbacks:** All errors/warnings route through a vtable at `callbacks+0xC` with severity codes; unknown sections produce warnings but continue.
- **Section finder:** `FUN_00783c80` walks tag/size pairs to find payloads in-memory (useful for targeted extraction/tests).

### Size/stride expectations seen in MDX-side handlers (carryover but now enforced)
- TEXS: size divisible by 0x10C in related MDX parsers (mirrors later-era client checks).
- SEQS: array of 0x28 records in MDX parsers (sequence structs).
- PIVT: 12-byte records (float3 pivots).
- HTST: switch on type 0..3; multiple shapes.
- GEOS: geometry block consumed by multiple parsers; ensure aligned to expected vertex/index/tangent arrays.

## Early M2 (MD20 v0x100) interplay in 0.9.0
- MD20 still appears but is expected to be converted to MDLX for rendering (see 0.12 doc for converter pseudocode).
- Profile-gated parsing is recommended: version 0x100 should map to an early-pre-modern profile with typed offset/count tables and strict span checks.
- Skin-like and effect-like stride assumptions (from later profiles) hold: Skin stride ~0x2C, Effect stride ~0xD4/0x7C for particles.
- Magic enforcement: MD20 must match `0x3032444D` before attempting conversion.

## Behavioral deltas vs 0.5.3 / 0.8.0
- **Strictness:** Bounds checks are hard errors; older loaders frequently allowed trailing/oversized sections.
- **Richer effects:** Ribbon emitters and helper nodes are first-class; particles use PRE2 handler with MDL flags (UNSHADED, USE_MODEL_SPACE, etc.).
- **Error surfacing:** Structured callbacks instead of ad-hoc logging; makes embedding in tools safer.
- **Dispatcher warning path:** Unknown sections are explicitly warned then skipped; earlier builds could silently ignore without consuming bytes correctly.

## Implementation guidance
- Split containers first (`MDLX` vs `MD20`); do not auto-fallback.
- Enforce size bounds and divisibility where handlers expect fixed strides (start with TEXS 0x10C stride, SEQS 0x28, PIVT 12-byte).
- Keep callback plumbing: surface fatal vs warning; allow unknown tags to skip but log.
- Add a 0.9.0 MDX profile in your registry reflecting the section set above; route MD20 v0x100 through a conversion step that outputs MDLX and then reuse the MDLX loader.
- Add tests: malformed section size → fatal; unknown tag with valid size → warning and continue.

## Quick reference: key functions (0.9.0 binary)
- Loader entry: `FUN_00783470`
- Mode switch + parse: `FUN_00783550`
- Binary walker: `FUN_007837b0`
- Section dispatcher: `FUN_00784200`
- Allocate/free: `FUN_007839c0` / `FUN_00783c60`
- Magic+payload fetch: `FUN_00783a80` / `FUN_00783b30`
- Section finder: `FUN_00783c80`

## WMO interplay (doodads in transitional builds)
- Root token order stays strict (`MVER(0x11)` then MOHD, MOTX, MOMT, MOGN, MOGI, MOSB, MOPV, MOPT, MOPR, MOVV, MOVB, MOLT, MODS, MODN, MODD, MFOG, optional MCVP) — see 0.9.0 WMO contract.
- Model references remain classic: `MMDX` holds MDX paths, `MMID` is the offset table; `MWMO`/`MWID` mirror this for WMO placements. Client still expects MDLX-era MDX models for doodads even when MD20 appears elsewhere.
- MODF records (size 0x40) unchanged: `nameId` indexes `MWMO`, position/rotation floats, scale (uint16), flags. No new fields observed for MD20 support; the loader assumes referenced names resolve to MDLX/MDX.
- Doodad sets: `MODD` records sized `/0x28`; flags and color fields behave as in 0.8.0. Placement still indexes `MMDX` strings; no alternate container tag is accepted.
- Renderer implication: World scene must route doodad loads through the MDLX loader (or MD20→MDLX converter) before binding. If you add MD20 files to WMO doodad tables, you must intercept on load and convert; the WMO parser will not do this translation.
- Validation: Keep `MMID` bounds checks when resolving `MMDX` offsets; malformed offsets should raise fatal via the WMO loader callback path. This prevents overrun when mis-sized string blocks reference MDLX names.

## Open items
- Decompile per-section handlers (`GEOS`, `MTLS`, `TEXS`, `BONE`, `SEQS`, `PRE2`) to lock record layouts for 0.9.0.
- Validate exact stride/divisibility rules per handler and mirror them in parser asserts.
- Confirm whether any optional legacy text-MDL path is still reachable in shipped assets; default to binary-only.
