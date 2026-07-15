# WoW 1.0.0 (Beta-3) Deep-Dive — M2 Record Layouts, WMO Format, Dev/Dead Code (Ghidra, 2026-07-15)

Companion to the M2 format trace (`specs/104-legacy-m2-rendering/research-1.0.0-ghidra-trace.md`)
and the renderer-features trace (`docs/architecture/wow-1.0.0-renderer-features-ghidra-trace-2026-07-15.md`).
This doc goes deeper: **per-field M2 record layouts**, the **WMO/WDT format** as 1.0.0 sees it,
and **dead code / development tools still present** in this beta-3 binary. Recovered via the
GhidraMCP HTTP API; raw decompilations in `output/ghidra_1.0.0/sub_*.c` and `deep_*.c`
(copied to `specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/`).

This build is **beta 3** (string `BETA_BUILD` at `0x007fded0`, referenced by `FUN_004751b0` and
`FUN_004a4ad0`) — not a final release — and is intended to set up the whole 1.x era of objects.

---

## 1. M2 relocator legend (used by every record below)

The header parser `FUN_0071e190` delegates to per-block relocators. Each `M2Array<T>` is
`{count, offset}` and is bounds-checked then fixed up to `{count, base+offset}`.

| Function | Kind | Element stride |
| --- | --- | --- |
| `FUN_0071f2c0` | `M2Array<int16>` | 2 |
| `FUN_0071fe10` | `M2Array<uint32>` | 4 |
| `FUN_0071f3c0` | `M2Array<T>` | 0xc (12) |
| `FUN_0071f5f0` | M2Track **timestamps** | 4 |
| `FUN_007203d0` | M2Track **vector3** spline | 8-byte keys |
| `FUN_00720eb0` | M2Track (variant) | 8-byte keys |
| `FUN_00720d30` | M2Track | 0x10 keys |
| `FUN_00720e30` | M2Track (particle tail) | — |
| `FUN_00720f30` | M2Track (camera) | — |
| inline | raw spline-key array | 8-byte keys (`count*8`) |

An `M2Track<T>` is `{count, ofs, tsCount, tsOfs, …}`: a spline-key array + a parallel
timestamp array. A `count*8` inline array is a spline-key block; `count*4` is a
timestamp/uint32 block.

---

## 2. M2 record layouts (version 0x100)

All offsets are within the record. Sizes confirmed from the sub-parser bounds multipliers.

### M2Bone — 0x6c (108 B) — `FUN_0071f440`
- `0x10/0x14`: translation track (8-byte keys), `0x18` ts, `0x20` 0xc-arr
- `0x2C/0x30`: rotation track (8-byte keys), `0x34` ts, `0x3C` 0x10-arr
- `0x48/0x4C`: scale track (8-byte keys), `0x50/0x54` uint32 arr
- `0x58`: 0xc-arr
- `parentIndex` (ushort, `0xFFFF` = root) — asserted `((0xffff)!=bones[i].parentIndex)||(i>0)`

### M2Vertex — 0x30 (48 B) — `FUN_0071f670`
8 fields copied into the 0x20-B render vertex by the division materializer (`FUN_006b7720`):
position, normal, uv, bone weights/indices (packed). (Per-field split = next RE step.)

### M2Division (embedded skin profile) — 0x2c (44 B) — `FUN_0071f6f0` + `FUN_0071e0b0`
- `0x00/0x04`: vertexLookup (`int16[]`) — div-local vtx → global M2Vertex
- `0x08/0x0C`: indices (`int16[]`) — triangle index buffer
- `0x10/0x14`: `uint32[]` — bone-palette / property table
- `0x18/0x1C`: sections (0x20 B each) — render sections / geometry groups
- `0x20/0x24`: batches (0x18 B each, `FUN_0071f340`) — `division->batches.count == 1` asserted

### M2Sequence — 0x44 (68 B) — header 0x1C
Animation sequence (start/end time, flags, …). `sequenceFallbackById` (alias table, `int16[]`) at 0x24.

### M2Texture — 0x10 (16 B) — `FUN_0071f930`
`{type, flags, nameLen, nameOfs}`; `nameOfs` → texture filename string.

### M2Color — 0x38 (56 B) — `FUN_0071f7a0` — animated color block.
### M2TextureWeight — 0x1c (28 B) — `FUN_0071fa00`.
### M2TextureTransform — 0x1c (28 B) — `FUN_0071fb20`.
### M2ReplaceableTexLookup — 0x54 (84 B) — `FUN_0071fc40`.

### M2Attachment — 0x30 (48 B) — `FUN_0071fe90` + `FUN_0070e500`
- `0x04`: `boneIndex` (ushort)
- `0x08/0x0C/0x10`: position offset (3 floats, bone-local)
- (rotation/flags in remaining bytes — confirm in focused pass)
- World transform = `modelWorldMatrix * boneMatrix[boneIndex] * offset`

### M2Event — 0x2c (44 B) — `FUN_0071ffc0`.

### M2Light — 0xd4 (212 B) — `FUN_007200b0`
**Larger than 3.3.5 (0x9c) — more animated fields.** Tracks at `0x14, 0x30, 0x4C` (vector3
tracks) each with ts + 0xc-arr; spline-key arrays at `0x68, 0x84, 0xA0, 0xBC` (count*8) +
timestamp tracks; uint32 arrays at `0xA8, 0xC4` (count*4); a variable array at `0xCC/0xD0`.
Maps to ambient color, diffuse color, specular color, attenuation start/end, intensity
(per-field semantic mapping = next RE step). Falloff computed in `FUN_0071c9d0`.

### M2Camera — 0x7c (124 B) — `FUN_00720450`
- `0x14`: source-position track, `0x1C` ts, `0x24` track
- `0x3C/0x40`: spline keys (count*8), `0x44` ts, `0x4C` track
- `0x64/0x68`: spline keys (count*8), `0x6C/0x70`: uint32 arr, `0x74/0x78`: 0xc-arr
- Semantics: source pos, target pos, near/far/fov (per-field = next RE step).

### M2Ribbon — 0xdc (220 B) — `FUN_00720600`
Ribbon emitter (trail effect); track-bearing record (per-field = next RE step).

### M2Particle — 0x1f8 (504 B) — `FUN_007208e0` — the largest, most-animated block
- `0x18/0x1C`, `0x20/0x24`: two inline buffers (likely texture filename + extra string)
- **~16 animated tracks** at `0x38, 0x54, 0x70, 0x8C, 0xA8, 0xC4, 0xE0, 0xFC, 0x118, 0x134, …`
  each = vector3 track + ts + variant (emission rate, speed, lifespan, color, alpha, scale,
  rotation, spread, gravity, drag, …)
- `0x1D4/0x1D8`: 0xc-arr, `0x1E0/0x1E4`: 8-byte spline, `0x1E8` ts, `0x1F0` track
- Runtime: `GetEmitter` (`FUN_0070ef60`) → emitter array at `CM2Model+0x3a8 + index*4`.

> **wow-viewer implication**: these layouts are enough to parse every M2 block for 1.0.0
> (`0x100`). The remaining work is per-field *semantic* naming (which track is color vs
> alpha vs scale), which is straightforward follow-up RE on the same sub-parsers.

---

## 3. WMO / WDT format (1.0.0)

### WDT — `FUN_006976f0` (root world-map definition)
Chunk order, strict-token-validated:
1. `MVER` (version)
2. `MPHD` — **0x20 (32) bytes** map header (flags etc.)
3. `MAIN` — **0x8000 (32768) bytes** tile grid (64×64 × 8 B)
4. **if `MPHD.flags & 1`** (global-WMO / single-object world, e.g. Stormwind):
   - `MWMO` — global WMO filename string
   - `MODF` — map object placement definitions
- Fallback WMO when missing: `World\wmo\Dungeon\test\missingwmo.wmo` (`0x0083699c`).

### WMO group — `FUN_006c5380` (MOGP reader)
- `MVER` version must be **0x11 (17)** — asserted `version == 0x0011` (`0x00838e48`).
  **1.0.0 WMO version = 0x11.**
- `MOGP` group header (FourCC `0x4d4f4750`), then sub-chunks read by `FUN_006c55a0`
  starting at MOGP+0x58. Confirmed group sub-chunks (string-validated):
  `MOPY` (materials/flags), `MOVT` (vertices), `MOLR` (light refs), `MOBA` (batches),
  `MOCV` (vertex colors), `MLIQ` (WMO liquid).
- Batches are **0x18 (24) B** each (`FUN_006c5080`, walked at +0xd8 stride 0x18).
- Internal assert name (dev humor, still in binary): **`lameAssLink_IsLinked`** (`0x00838ed0`)
  + `parent` (`0x00838ec8`) — the group-link/parent linkage check.

### WMO ancillary
- **Portals**: `portal->count <= 12` (`0x00837cc4`) — **max 12 portals per group**. `PortalExt`
  (`.?AUSPortalExt@@`). Dev toggles: `TogglePortals`, `Portal display enabled/disabled`,
  `Portal vis enabled/disabled`.
- **Doodads**: `SetDoodadAnim`/`GetDoodadAnim`/`doodadAnim` (`0x00805468/78`), `GroundEffectDoodad.dbc`,
  `detailDoodadAlpha`, `showSimpleDoodads`, `showDetailDoodads`.
- **WMOAreaTable.dbc** (`0x0081f6b0`, `WMOAreaTableRec`) — area names/flags inside WMOs.
- **CWModelFadeout** (`.?AVCWModelFadeout@@`, `0x00832c5c`) — WMO distance fadeout.
- `s_unitShowMode` (`0x008397f8`).

> **wow-viewer implication**: 1.0.0 WMO is the classic MWMO-root + MOGP-group layout at
> **version 0x11**, with MOPY/MOVT/MOLR/MOBA/MOCV/MLIQ sub-chunks and 0x18-B batches —
> close to the existing `gillijimproject_refactor` WMO reader, but verify the version gate
> and the 0x11-specific MOGP header field offsets. Max 12 portals/group is a hard limit.

---

## 4. Dead code / development tools still in the beta-3 binary

These are **live, referenced code paths** (confirmed via xrefs), not stripped — typical of a
beta build. Useful to know what the client *can* do that a retail build later removed.

| String | Address | Referenced by | What it is |
| --- | --- | --- | --- |
| `BETA_BUILD` | `0x007fded0` | `FUN_004751b0`, `FUN_004a4ad0` | beta-build marker / version gate |
| `Godmode enabled` / `Godmode disabled` | `0x00827714/28` | `0x005ea9a1` | **godmode cheat** toggle (live) |
| `ConsoleExec`, `SetConsoleKey`, `console`, `closeconsole`, `consolelines` | `0x0081738c…` | `0x00816d00` | **developer console** still present |
| `GetDebugStats`, `debugTargetInfo`, `debugobjectpathing`, `FrameXML_Debug`, `debug` | `0x00805530…` | `0x00804720…` | debug cvars/toggles |
| `TogglePortals`, `Portal display/vis enabled` | `0x008053fc…` | — | portal debug viz |
| `ProfileInternal` (`STRINGBLOCK/PROFILE/SECTION/KEYVALUE`) | `0x007fa58c…` | — | **profiler** subsystem still in binary |
| `FIXME: Not yet implemented` | `0x00843af8` | `FUN_0075a320` | unimplemented-feature leftover |
| `lameAssLink_IsLinked` | `0x00838ed0` | WMO group reader | dev-humor assert name |
| `Total unused texture in Mbytes` | `0x007fae34` | — | debug texture-memory stat |
| `World\wmo\Dungeon\test\missingwmo.wmo` | `0x0083699c` | — | test/placeholder WMO asset |
| `movie`, `movieSubtitle`, `Show movie on startup` | `0x007f565c…` | — | intro movie subsystem |
| `MOUNTDISPLAYIDNOMOUNTATTACHMENT` | `0x008291cc` | — | mount-attachment debug log |

> **wow-viewer implication**: none of these are needed for rendering, but they confirm the
> build is beta-3 and reveal the dev console + godmode + profiler are reachable. If you ever
> want to drive the live client for dynamic validation, the **developer console**
> (`ConsoleExec`/`SetConsoleKey`) is the entry point — it can toggle `TogglePortals`,
> `Godmode`, `SetDoodadAnim`, `SetWaterDetail`, etc. directly, which is far more convenient
> than x64dbg for quick visual checks.

---

## 5. Open follow-ups (deep)

- Per-field **semantic** naming for M2Light / M2Camera / M2Particle / M2Ribbon tracks
  (which track = which animated value) — decompile the runtime samplers (`FUN_0070f6d0`,
  `FUN_00716330`) against the track offsets.
- M2Vertex per-field split (position/normal/uv/bone packing) — decompile `FUN_006b7720`'s
  8-field copy loop.
- WMO **root .wmo** file reader (MOHD/MOGN/MOGI/MOTX/MOMT/MOPV/MOPT/MOPR/MODS/MODD) — the
  group reader `FUN_006c5380` is found; the root reader is the next target.
- Liquid type index→name map (read 12 pointers at `0x00834d4c`).
- Enumerate the full **developer console command table** (the `ConsoleExec` dispatch) —
  likely a goldmine of dev tools for dynamic validation.