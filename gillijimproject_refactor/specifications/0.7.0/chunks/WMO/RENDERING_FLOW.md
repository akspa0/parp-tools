# WMO Rendering/Data Flow — Build 0.7.0.3694

## Scope
This document explains **how parsed WMO chunk data is used at runtime** in the 0.7.0.3694 client, not just field layout.

Primary evidence functions:
- Root/group parse: `FUN_006c11a0`, `FUN_006c17f0`, `FUN_006c1a10`, `FUN_006c1c60`
- World visibility loop: `FUN_0069cca0`
- Group material setup: `FUN_006c14c0`
- Group list/build: `FUN_00699c10`
- Doodad/light pass linking: `FUN_00699de0`, `FUN_00699fa0`
- Ray/hit queries: `FUN_006a1860`, `FUN_006a9b00`, `FUN_006a9f90`
- Portal/group recursion: `FUN_006ab560`, `FUN_006ab730`
- Liquid queries: `FUN_006a3d90`, `FUN_006a3e60`

## End-to-end pipeline

1. Root WMO is parsed (`FUN_006c11a0`) and group metadata is stored from `MOGI`.
2. Group files are loaded on demand (`FUN_006c1570`), validated as `MVER(0x10)->MOGP` (`FUN_006c17f0`).
3. Group geometry chunks are wired (`FUN_006c1a10`), optional blocks are wired by flags (`FUN_006c1c60`).
4. Per-batch materials/textures are resolved from `MOBA` material IDs (`FUN_006c17f0` -> `FUN_006c14c0`).
5. Visibility loop (`FUN_0069cca0`) frustum-tests map object and each group (`MOGI` bounds), then:
   - queues group draw data (`FUN_00699c10`),
   - links doodads from `MODR` (`FUN_00699de0`),
   - links lights from `MOLR` (`FUN_00699fa0`).

## Required group chunks and runtime role

### `MOPY`
- Stored at `group + 0xB4`, count `group + 0x10C`.
- Used in geometric query/raycast path (`FUN_006a9b00`) as `polyList`.
- Selected polygon index is used to index `MOPY` records (`polyList + polyIndex*4`), meaning MOPY data participates in per-poly material/flag behavior for interaction tests.

### `MOVI`
- Stored at `group + 0xB8`, count `group + 0x110`.
- Acts as triangle index stream for group geometry; consumed through group render/query paths downstream of `FUN_006a2fb0` and associated geometry tests.

### `MOVT`
- Stored at `group + 0xBC`, count `group + 0x114`.
- Base vertex positions for render and intersection calculations.

### `MONR`
- Stored at `group + 0xC0`, count `group + 0x118`.
- Normal vectors paired with vertex data for lighting/interaction math.

### `MOTV`
- Stored at `group + 0xC4`, count `group + 0x11C`.
- UV stream used during textured rendering.

### `MOBA`
- Stored at `group + 0xC8`, count `group + 0x120`.
- Each batch references a material ID byte (`entry + 0x17`) used in `FUN_006c17f0` to call `FUN_006c14c0`.
- This drives texture handle setup from root material/texture tables before rendering.

## Optional group chunks and runtime role

### `MOLR` (`flags & 0x200`)
- Stored at `group + 0xCC`, count `group + 0x124`.
- `FUN_00699fa0` iterates this list and instantiates/links light objects into group runtime lists.

### `MODR` (`flags & 0x800`)
- Stored at `group + 0xD0`, count `group + 0x128`.
- `FUN_00699de0` iterates doodad refs, resolves doodad instances, and links them into group/object render lists.

### `MOBN` + `MOBR` (`flags & 0x1`)
- Combined via `FUN_006c0050(...)` into group BSP/runtime query data.
- Used by intersection/query paths (e.g., in `FUN_006a2fb0` callers) for efficient spatial filtering.

### `MOCV` (`flags & 0x4`)
- Stored at `group + 0xDC`, count `group + 0x12C`.
- Optional per-vertex color stream; presence toggles additional colorized render behavior in pipelines that read this stream.

### `MLIQ` (`flags & 0x1000`)
- Data copied into `group + 0xE0..0x104` pointer/parameter region.
- Enables group-local liquid geometry/logic branch for rendering and interaction.
- `FUN_006a3d90` performs liquid presence/type checks from this region.
- `FUN_006a3e60` interpolates liquid surface height and returns liquid-type/control values.

## Root chunk runtime role

### `MOGI` (group infos)
- Core per-group control records (flags + bounds) used every frame.
- `FUN_00699c10` reads each group info to build runtime group-def entries.
- Group flag bits (from `FUN_006aaac0`) choose render path classification (`0x48`-based branch) and influence state bits in group-def setup.

Usage-derived `MOGI` record map in this build:
- `+0x00`: flags
- `+0x04..+0x18`: local group AABB min/max (6 floats)
- `+0x1C`: metadata index (not consumed in traced render/cull path)

### `MOHD`
- `MOHD + 0x1C` is copied to runtime (`mapObj + 0x184`) and used in group/light setup paths.
- `MOHD + 0x24..0x38` (6 dwords) is copied to `mapObj + 0x18C..0x1A0` and treated as root AABB.
- `FUN_006aa6f0` computes center/radius from that AABB; `FUN_006aa850` bulk-copies it for downstream cull/query operations.

### `MOPV` / `MOPT` / `MOPR`
- Portal adjacency/visibility recursion path (`FUN_006ab560`) traverses group links and portal geometry.
- Runtime uses map-object portal arrays (fed from root portal chunks) together with per-group portal start/count-like header fields.
- This is a defining early-v17 behavior: recursive group propagation through portal graph, not only flat frustum tests.

### `MOMT` + `MOTX`
- Material table + texture string blob.
- Consumed by `FUN_006c14c0` while processing each `MOBA` batch material ID.
- Texture handles are created/cached into material runtime slots (+0x38/+0x3C in material entries).

## Render list behavior (what actually gets drawn)

- Group drawables are queued in `FUN_00699c10` after visibility checks.
- If group has `MODR`, doodad instances are linked/activated in `FUN_00699de0`.
- If group has `MOLR`, light instances are linked/activated in `FUN_00699fa0`.
- Group/object state flags (`+0x0C` bits like `0x20`, `0x40`, `0x08`) are set as side effects and used to avoid duplicate setup and to route pass behavior.

Observed state-bit behavior (group/runtime `+0x0C`):
- `0x40`: light-ref (`MOLR`) setup completed for that group (`FUN_00699fa0`).
- `0x20`: doodad-ref (`MODR`) setup completed for that group (`FUN_00699de0`).
- `0x08` / `0x10`: classification path set during group-def build from `MOGI.flags & 0x48` (`FUN_00699c10`).

## Interaction / collision behavior

- Query paths (`FUN_006a1860`, `FUN_006a9b00`, `FUN_006a9f90`) iterate visible groups through `FUN_006aaea0`.
- Per-group ray/intersection tests call `FUN_006a2fb0` and use group data wired from `MOGP` payload.
- `MOPY` is explicitly used as polygon-side lookup material/flag data in hit results.
- Portal recursion (`FUN_006ab560`) expands candidate groups via portal adjacency while applying bounding tests.
- Liquid checks run as a separate optional branch (`flags & 0x1000`) during world interaction queries.

## Practical implication for tooling

- Correct parsing alone is insufficient: tools that want engine-equivalent behavior must preserve:
  - `MOGI` flags/bounds semantics,
  - `MOBA` material-id mapping into `MOMT/MOTX`,
  - optional chunk gates (`MOLR/MODR/MOBN/MOBR/MOCV/MLIQ`),
  - and group runtime-state transitions triggered in visibility passes.