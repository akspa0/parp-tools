# MDL / MDX / M2 Deep Dive — 0.9.0 (Reverse-engineered)

## Scope
- Document the 0.9.0 MDL/MDLX container loader, section dispatcher, and helper utilities recovered from the binary.
- Capture actionable implementation guidance for the viewer/converter pipeline (MDLX and MD20-to-MDX path), highlighting differences vs earlier 0.5.3/0.8.0-era assumptions.
- Provide a section tag map to the per-section handlers to accelerate targeted decompilation.

## Evidence anchors (functions)
- `FUN_00783470`: High-level MDLX load entry (path/null guards, allocator setup, parser call, error callback on failure).
- `FUN_00783550`: Mode switch (text vs binary), buffer allocation/free, drives binary parse via `FUN_007837b0`.
- `FUN_007837b0`: Core MDLX binary walker (validates magic `0x584C444D`, iterates section headers, bounds checks, dispatches sections, finalizes).
- `FUN_00784200`: Section tag dispatcher (maps 4CC tags to per-section handlers; warns and skips unknown tags).
- `FUN_007839c0` / `FUN_00783be0`: Allocate + slurp file into buffer (`size+1`), return pointer; `FUN_00783c60` frees (`ptr-4`).
- `FUN_00783a80` / `FUN_00783b30`: Thin wrappers for loading/validating MDLX magic (`MDLX`), return payload pointer.
- `FUN_00783c80`: Section finder over in-memory MDLX buffer (walks tag/size blocks, returns pointer to payload for a given tag).
- `FUN_00783680`: Non-binary path iterator (loops `FUN_00783f20` across sections, then finalizes with `FUN_00785ea0`).

## Loader flow (binary MDLX)
```text
LoadMdl(path, length, callbacks?):
  ensure path != null; ensure output callbacks not null (default DAT_01009198)
  if !open(path): error("path") and return 0
  allocate buffer via FUN_007839c0 (reads whole file -> buf)
  if buf == null: cleanup and return 0
  if buf[0] != 0x584C444D ("MDLX"): free buf; callback(error "not binary model"); return 0
  sectionBytesRemaining = fileSize - 4
  rc = ParseSections(buf+4, sectionBytesRemaining, callbacks)
  free buf
  return rc

ParseSections(base, length, callbacks):
  cursor = base; used = 4 (initial MDLX magic already consumed)
  lastTag = 0; lastPayloadStart = 0
  while used < length:
    tag = read_u32(cursor)
    size = read_u32(cursor+4)
    used += 8
    if size > (fileEnd - cursor - 8): callback(error "section too large", lastTag); return 0
    if size > 0:
      handled = DispatchSection(tag, cursor+8, callbacks)
      if handled == 0: return 0
      used += size
    lastTag = tag; lastPayloadStart = cursor+8
    cursor += 8 + size
  if used > length: callback(error "MDLFile overran total file size"); return 0
  return Finalize()

DispatchSection(tag, payload, callbacks):
  switch(tag):
    HELP -> FUN_0079bb60
    LITE -> FUN_0079d470
    RIBB -> FUN_007903f0
    PRE2 -> FUN_00795700
    GEOA -> FUN_007a2280
    CLID -> FUN_00789d10
    BONE -> FUN_0079eb30
    PERP -> FUN_007990e0
    ATCH -> FUN_0079ac10
    MODL -> FUN_007ab640
    VANS -> FUN_007abf40
    CAM3 -> FUN_0078dd20
    SBLG -> FUN_007aa3c0
    MTLS -> FUN_007a7440
    GEOS -> FUN_007a2a00
    SEQS -> FUN_007a9c70
    HTST -> FUN_0078aa10
    VEST -> FUN_0078bd40
    TEXS -> FUN_007a8df0
    PIVT -> FUN_0079a340
    default -> warn(Unknown tag); skip payload via FUN_007f4260; return 1
  return handler(payload, callbacks)
```

## Behavioral notes and strictness
- Magic and size gates: `FUN_007837b0` enforces MDLX magic first, then length accounting (no section may exceed remaining bytes; total used must not exceed file size). This is stricter than earlier 0.5.3/0.8.0-era parsers that often streamed until EOF without explicit overrun checks.
- Error surface: all fatal/soft errors are routed through the callback vtable at `callbacks + 0xC` with level codes (e.g., fatal when magic mismatch or section length exceeds file bounds). Keep these hooks intact when porting.
- Allocation strategy: both binary and non-binary paths allocate `(fileSize + 1)` via `FUN_00667c40`, then free via `FUN_00667e50(ptr, filePath, line, 0)`. Preserve this "size stored in front" convention if rehosting the allocator.
- Section search: `FUN_00783c80` walks `[tag, size, payload...]` blocks and returns the payload pointer when `tag` matches, adjusting by the input buffer base and supplied search extent. Use this for targeted extraction without reparsing.

## Section map (tag → handler) with likely semantics
- `MODL` (`0x4D4F444C`): model header/root.
- `SEQS` (`0x53514553`): animation sequences.
- `MTLS` (`0x4D544C53`): materials.
- `TEXS` (`0x54455853`): textures (size divisible by 0x10C in modern MDX path; verify sizes here).
- `GEOS` (`0x47454F53`): geosets / geometry blocks.
- `GEOA` (`0x47454F41`): geoset animations (color/alpha tracks).
- `BONE` (`0x424F4E45`): bones.
- `PIVT` (`0x50495654`): pivots (12-byte records typical).
- `HTST` (`0x48545354`): hit/shape data (variants by type).
- `CLID` (`0x434C4944`): collision data.
- `CAM3` (`0x43414D33`): cameras.
- `LITE` (`0x4C495445`): lights.
- `ATCH` (`0x41544348`): attachments.
- `RIBB` (`0x52494242`): ribbon emitters.
- `PRE2` (`0x50524532`): particle emitters (v2).
- `HELP` (`0x48454C50`): helper nodes.
- `VANS` (`0x56414E53`): version/anim state block (seen as `VANS` token; handler `FUN_007abf40`).
- `PERP` (`0x50455250`): likely particle/geometry-related (handler `FUN_007990e0`).

## Section-level loader expectations (0.9.0 binary path)
- `TEXS`: observed downstream stride guard of 0x10C per texture entry in modern MDX paths; keep divisibility check `size % 0x10C == 0`. Expect filename offsets and flags similar to WC3/WoW MDX.
- `SEQS`: arrays of 0x8C-sized records in this build (see `FUN_007a9c70`); parser asserts `(sectionSize - 4) == count * 0x8C` where `count = first u32`. Rejects non-integral sizes with "Invalid SEQX" fatal.
- `SEQS` field map (0x8C per record, `FUN_007a9c70`):
  - +0x00: uint32 animId (loop index)
  - +0x04: uint32 subId (FUN_007f40a0)
  - +0x08..0x0F: zeroed
  - +0x50: uint32 startTime
  - +0x54: uint32 endTime
  - +0x58: float moveSpeed (FUN_007f41a0)
  - +0x5C: uint32 flags/unk
  - +0x60..0x6B: 3 * float (bbox min?) via FUN_007f4470
  - +0x6C..0x77: 3 * float (bbox max?) via FUN_007f4470
  - +0x78: float blendTime (?) via FUN_007f41a0
  - +0x7C: uint32 playbackSpeed (?) via FUN_007f4060
  - +0x80: uint32 frequency (?) via FUN_007f4060
  - +0x84: uint32 pad/unk (FUN_007f40a0)
  - +0x88: uint32 pad/unk
  - Gaps 0x10..0x4F unused here.
- `PIVT`: 12-byte float3 pivots; enforce `size % 0x0C == 0`.
- `GEOS`: geoset records are 0x104 in-memory nodes (`FUN_007a4bb0` ctor). Section begins with a u32 count; each geoset is filled in two passes (`FUN_007a2f10` headers/counts → `FUN_007a2bc0` payload arrays). Overrun fatal string: "Geoset section overran read buffer." Keep `size >= 4 + count * (payload bytes)`, and abort if the running cursor ever overtakes section end.
- `GEOS` field map (0x104 per record, recovered):
  - +0x08: vertexCount (u32); +0x0C: ptr to vertex float3 array (3 * f32). Expands/zeros to count.
  - +0x1C: normalCount (u32); +0x20: ptr to normal float3 array. Expands/zeros to count (mirrors vertexCount).
  - +0x28 sub-vector: UV set — ptr at [+0x28+0x0C], each entry 2 * f32; count must match vertexCount (errors if zero-sized).
  - +0x3C sub-vector: per-vertex u32 channel A (4 bytes per vertex). Acts like the classic `VertexGroup` palette index; streamed via `FUN_007f42b0` with bounds checks.
  - +0x50 sub-vector: per-vertex u32 channel B (4 bytes per vertex). Acts like the classic `MatrixIndex` palette slot; also read with `FUN_007f42b0`.
  - +0x64 sub-vector: normals vector wrapper reused for capacity checks while filling (pointer + counts shadow +0x1C/+0x20).
  - +0x70/+0x84 blocks: matrix/face group table — count at +0x80; entries are 0x10 bytes {u32 start, u32 count, u16 a, u16 b, u16 c}. After streaming these entries, the parser aligns to 8 bytes (`FUN_007f4320` + padding bytes). Treat as matrix group → matrix index spans until renderer naming is confirmed.
  - +0x30: groupCount forced to 1; +0x34 points to a group header whose +8 field is resized to vertexCount (zeros index ranges).
  - Bounding data: +0xCC..0xD4 bbox min, +0xD8..0xE0 bbox max, +0xE4 radius (radius subtracted from mins after read for extents-as-minus-radius convention).
  - +0xC8: geoset id/selection token (first header read).
  - +0xFC: material/token id set by the text parser (`FUN_0079f4d0` case 0x1b5); used downstream to bind MTLS. Keep bounds-checked against material count.
  - +0x100: geoset flags bitfield. Text parser sets bits: 0x10 (case 0x1ab), 0x20 (case 0x1b6), 0x01 (case 0x1d5). Treat this as render/selectability flags; keep logged until renderer consumers are mapped.
  - Two-pass fill: header pass reads counts/bounds, resizes vertex/normal/UV/group arrays; payload pass streams vertex float3s, per-vertex u32 A/B, normals, UVs, and matrix group entries, with hard bounds checks each step.
- `GEOA`: color/alpha anim tracks for geosets; carries MDLKEYFRAME style tracks; ensure track counts * stride do not exceed chunk end.
- `MTLS`: materials parsed via `FUN_007a7440`; count from first u32; each material node is 0x18 bytes in the owning array and then populated by `FUN_007a7590`. Fatal on overrun with "MaterialSection" string.
- `MTLS` field hints (0x18 per node): zeroed, vtable at +0x00 set to `PTR_FUN_0080d944`; slots +0x04,+0x08,+0x0C,+0x10,+0x14 cleared; detailed layer/flag fields live inside `FUN_007a7590` (not yet mapped).
- `Material layer` field map (0x4C per layer inside `FUN_007a7590` → `FUN_007a7740`):
  - +0x00: uint32 textureId
  - +0x04: uint32 renderFlagIndex
  - +0x08: uint32 shading/opacity flags (MDX MTLS-style):
    * 0x01: Unlit/Unshaded
    * 0x02: Unfogged
    * 0x04: TwoSided
    * 0x08: DepthWriteDisable (requires depth test elsewhere)
    * 0x10: Billboarded layer (rare in MTLS; usually set via GENOBJECT)
    * 0x20: DepthTestDisable
    * 0x40: Unk render mod (treat as alpha-keyed)
    * 0x80: Unk; keep masked and logged if set
  - +0x0C: uint32 textureUnit
  - +0x10: uint32 layerPriority
  - +0x14: uint32 uvAnimId
  - +0x18..0x1C: two int16 indices (bone/transform refs) set via `FUN_0079a090`
  - +0x20: float color mult (default 1.0)
  - +0x24..0x4B: populated via sub-chunks in `FUN_007a7740`:
    * Tag `0x41544d4b` (“KMTA”): alpha track — count then entries {time, 1–3 floats depending on flags}. Uses count at +0x10/+0x11, data pointer at +0x0E.
    * Tag `0x46544d4b` (“KMTF”): texture frame/transform track — count then entries {time, 1–3 floats}. Uses count at +0x05, data pointer at +0x06.
    * Each subchunk bounds-checks against remaining bytes; failure → fatal "TexLayer".
  - Defaults: textureId/renderFlag/flags/unit/priority/uvAnim from header u32s; tracks default 0; color mult default 1.0. Render/shading flags live in the third u32 (interpret per MDX MTLS: unlit, two-sided, unfogged, depth-write, etc.).
- `BONE`: bones parsed in 0xB8-byte records (`FUN_0079eb30`); count from first u32. Uses vector growth at +0x458 (base) with capacity at +0x450 and allocator at +0x45c; overrun fatal string DAT_008a1538 ("Bone section overran read buffer.").
- `BONE` field map (0xB8 per record):
  - GENOBJECT preamble via `FUN_00787a70` consumes the bulk of the struct.
  - +0xB0, +0xB4: two u32s read immediately after GENOBJECT (likely parent/index and flags/id).
  - After each record, `FUN_007870c0(recordIndex, 0x30000000)` applies a composite flag mask.
  - On allocation failure for a record, fatal via DAT_008a1538; on GENOBJECT read failure, emits "Error reading gen object portion of Bone section".
- `HTST`: type-switched shapes (0..3) similar to WC3 hitboxes; enforce per-variant record sizes.
- `CLID`: collision uses shared VRTX/NRMS layout; ensure indices do not exceed vertex count.
- `PRE2`: particle emitter v2; count from first u32; each emitter occupies 0x50C bytes; parsed via `FUN_00795870`; fatal on overrun/null via "ParticleEmitter2 Section".
- `PRE2` field hints (0x50C per emitter): requires `FUN_00797590()` success before parsing; GENOBJECT-like base likely; `FUN_00795870` fills emitter parameters/tracks; treat payload opaque until further mapping.
- `PRE2` field map highlights (0x50C, `FUN_00795870`):
  - +0x00: GENOBJECT preamble via `FUN_00787a70`
  - +0xB0: uint32 flags/id
  - +0xB4: float speed/scale
  - +0xD4, +0xF4, +0x114, +0x134, +0x154, +0x174, +0x194, +0x1B4, +0x1D4: multiple float params (spawn, spread, gravity-like fields)
  - +0x1F8..0x214: assorted uint32/float trio parameters (texture rows/cols, slowdown, drag)
  - +0x218..0x22C: three float3 blocks (likely colors or vectors)
  - +0x230..0x232: 3 bytes (color/alpha channels)
  - +0x234..0x23C: three floats (color/alpha scales)
  - +0x240..0x26C: eleven uint32s (IDs to tracks/lookup tables)
  - +0x28C: uint32 emissionSpeed(?)
  - +0x290..0x298: uint32 + two uint32s (blending/render flags)
  - +0x29C..0x3A0: two blocks of 0x104 bytes zeroed (track tables)
  - +0x4A4..0x4F4: 0x38 bytes of floats (curve constants)
  - +0x500: uint32 secondary count; +0x504 pointer to 3-float entries grown if count>0 (e.g., plane normals)
  - Sub-chunks inside emitter parsed by tags:
    * `0x4B504C4E` (“KPLN”): longitude keys — count, then entries of {uint32 time; float3 values} (stride 0x10 or 0x18 depending on mode)
    * `0x4732504B` (“KP2G”): gravity keys — same pattern {time; float3}
    * `0x4532504B` (“KP2E”): emission rate keys — same pattern {time; float3}
    * All subchunks validate available bytes; on failure emit specific error strings (`Error reading ... portion of ParticleEmitter2 section`).
  - +0x4F8: allocator/vtable for track arrays; +0x4FC capacity; +0x500 size; +0x504 data.
  - +0x140/0x180/0x120 blocks hold per-track arrays for gravity/emission/longitude (uint32 count, pointer to entries of time+float3).
- `RIBB`: ribbon emitters present; expect track bundles (color/alpha/height) and bone indices; enforce record-size divisibility.
- `HELP`: helpers are GENOBJECT-derived with transforms only; use for attach/FX parents.
- `VANS` / `PERP`: present; treat as required to parse when encountered; warn-only skip allowed by dispatcher, but profiles should expect them in 0.9.0 MDLX.

## Evidence tables (0.9.0 loader/gates)
- `FUN_00783470` (build 0.9.0): Role=entry/gate; Magic checks=MDLX enforced indirectly via `FUN_007837b0`; Extension coercion=no; Bounds checks=delegated; Failure=fatal callback on open/NULL/magic fail; Caller chain=external load entry (disk path); Sample=result: non-MDLX file → error callback then 0.
- `FUN_00783550` (build 0.9.0): Role=mode switch/normalize; Magic checks=via downstream; Version checks=none; Extension coercion=none; Bounds checks=delegated; Failure=fatal callback when `FUN_00783070` returns 2 → `FUN_00784890` (“Unrecognized file extension: %s”); Caller chain=from `FUN_00783470`; Sample: bad extension → hard fail before allocation.
- `FUN_007837b0` (build 0.9.0): Role=MDLX walker/validator; Magic checks=MDLX; Version checks=none; Bounds checks=section overrun, total overrun; Failure=fatal on size overruns; Unknown tags=warning skip via dispatcher.
- `FUN_00784200` (build 0.9.0): Role=dispatcher; Magic checks=none; Version checks=none; Bounds checks=none; Failure mode=warning skip on unknown tags (string 008a16ec).
- `FUN_00783c80` (build 0.9.0): Role=section finder; Magic checks=none; Bounds checks=walks tag/size; Failure=returns null on missing tag, no callback.
- `FUN_00784890` (build 0.9.0): Role=extension validator; Magic checks=none; Version checks=none; Extension coercion=produces fatal callback using string 008a1a54 (“Unrecognized file extension: %s”); Caller chain=only from `FUN_00783550`.
- Overrun strings (0.9.0): 008a0aa8 “MDLFile overran total file size”, 008a0a6c “Section length was greater than bytes remaining in file.” consumed by `FUN_007837b0`.
- `FUN_00784810` (build 0.9.0): Role=section overrun reporter; Magic checks=none; Bounds checks: reports when a section overruns remaining buffer; Failure mode=fatal callback with/without line number (strings 008a196c / 008a199c); Caller chain=from section readers (xref: PRE2/others via dispatcher tree).
- Section-specific overrun guards (0.9.0):
  - `FUN_00789d10` (CLID) uses 008abc98 “Collision section overran read buffer.”
  - `FUN_0078aa10` (HTST) uses 008abd38 “Hit test section overran read buffer.”
  - `FUN_0078bd40` (VEST) uses 008abe00 “EventObject section overran read buffer.”
  - `FUN_0079bb60` (HELP) uses 008ac838 “Helper section overran read buffer.”
  - `FUN_007a2a00` (GEOS) uses 008acc70 “Geoset section overran read buffer.”

## Call graph snapshots (0.9.0)
- Disk entry: `FUN_00783470` → `FUN_00783550` → (if extension invalid) `FUN_00784890` fatal; else alloc → `FUN_007837b0` → `FUN_00784200` dispatch per section; frees via `FUN_00667e50`/`FUN_00783c60`.
- In-memory section lookup: `FUN_00783c80` walks tag/size blocks, no callbacks.
- Unknown tag path: `FUN_00784200` warns and skips via `FUN_007f4260` then continues.

## Cross-build routing notes (for loader implementation)
- 0.9.0: container-first; MDLX magic required. No MD20 branch observed in current chain. Extension fatal guard present (`FUN_00784890`).
- 0.12 baseline: MD20 validator requires magic `0x3032444D` and version 0x100 before conversion (see 0.12 transitional contract references provided). Extension normalization happens before parser selection.
- 0.9.1 target behavior: container-first policy (route by root magic; extension advisory only), per parser profile doc.

## WMO interplay (0.9.0 doodads referencing MDX/MD20)
- Root order stays strict (MVER 0x11, MOHD, MOTX, MOMT, MOGN, MOGI, MOSB, MOPV, MOPT, MOPR, MOVV, MOVB, MOLT, MODS, MODN, MODD, MFOG, optional MCVP). Loader asserts this sequence; divergent order should be treated as fatal.
- Doodad/model reference blocks remain classic: `MMDX` (null-terminated filenames) + `MMID` (uint32 offsets). `MWMO`/`MWID` mirror this for world models. Bounds checks must ensure every MMID offset lands within MMDX and is null-terminated before chunk end.
- Placement records:
  - `MODF` size 0x40: nameId → MWMO, position (3*float), rotation (3*float), extents (2*float3), flags (uint32), doodad set id (uint16), name set, padding. No new fields for MD20; parser assumes MDLX-targeted names.
  - `MODD` size 0x28: nameId → MMDX, position/rotation (floats), scale (uint16), color (rgba8), flags (uint8), light references; behaves same as 0.8.0.
- No alternate container tag exists for MD20 here: if any `MMDX` entry points to an MD20 file, the loader must detect and convert MD20→MDLX before render; WMO parser will not do translation.
- Renderer impact: world scene should hand off doodad loads to the MDLX loader (or conversion step) based on magic. Keep MMID/MWID bounds checks to avoid overrun if malformed string tables appear in proto assets.

## Differences vs 0.5.3 / 0.8.0-era expectations
- Stricter size accounting: 0.9.0 explicitly aborts on section overruns and total file overrun; earlier builds tolerated trailing bytes.
- Richer section set: ribbons (`RIBB`), helper (`HELP`), particle v2 (`PRE2`), and `VANS`/`PERP` dispatch targets are present and enforced; these did not all exist or were stubbed in 0.5.3.
- Required magic: binary path refuses non-MDLX and emits a hard error; earlier builds sometimes autodetected or accepted text/legacy streams.
- Uniform callback plumbing: errors routed through a vtable make the loader reusable across tool modes; older loaders often used global logging.

## Guidance for implementation (viewer / converter)
- Container detection: treat `MDLX` as binary MDL; do not auto-fallback to text. Keep a strict `MDLX` vs `MD20` split before applying model profiles.
- Section guards: enforce `size` <= remaining bytes and reject negative/overflow. Optional sections should still respect divisibility (e.g., `TEXS` stride checks from MDX path).
- Profile registry: introduce `MdxProfile_090_3807` that encodes the section set above and ties into `ResolveModelProfile` for both MDLX and MD20-to-MDX conversion.
- Conversion flow (MD20 → MDX): reuse the guard logic from the 0.12 profile doc — validate magic/version, choose profile, then feed converted bytes into the MDLX loader path.
- Rendering: the recovered dispatcher matches the modern MDX section layout; rendering code should already understand geoset/material/sequence structures. Add explicit handling for ribbons (`RIBB`) and helper nodes if missing in 0.5.3-era renderers.
- Tooling hooks: expose callbacks from the loader to surface fatal vs warning conditions (unknown tag warnings keep parsing after skipping payload).

## Targeted next steps
1. Confirm renderer usage for GEOS matrix/face group entries (0x10 stride) and the +0xFC/+0x100 spans to lock names; keep logging unknowns until validated.
2. Backport strict length checks into the runtime parser to match `FUN_007837b0` behavior.
3. Add unit tests that feed malformed section sizes to confirm abort-on-overrun.
4. Validate renderer support for `RIBB` and `HELP` records; add stub geometry if necessary to prevent crashes.
5. Align the MD20-to-MDX converter with the profile registry so early MD20 (`0x100`) routes through the MDLX path with matching stride assumptions.
