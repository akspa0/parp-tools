# Phase 0 Research: Legacy M2 format landscape (0.11 – 2.4.3)

Purpose: establish what is known per M2 format version before changing the reader, and enumerate the
exact unknowns each phase must resolve. This is a living document — Phases 1–3 promote "to confirm"
entries into the durable per-version profile in [contracts/m2-format-profile.md](contracts/m2-format-profile.md).

Sources: wowdev.wiki (M2, M2/.skin), the existing reader
[M2ModelReader.cs](../../src/core/WowViewer.Core.IO/M2/M2ModelReader.cs), the read-only reference
`gillijimproject_refactor`, and (Phase 3) x64dbg traces of the actual clients. Where a byte offset is
not yet independently confirmed for a given version it is marked **(confirm P1/P3)** — the reader must
verify against a real hex dump + a reference implementation before trusting it.

## Decision 1 — The discriminator is the M2 format version, not the client build

**Decision**: Branch reader behavior on the `uint32` format version at header offset `0x04`, not on the
client build string. **Rationale**: multiple builds share one format version, and the format changed at
version boundaries; the version field is the only in-file, authoritative discriminator. **Alternatives
rejected**: build-string branching (not available from the file alone, and wrong granularity).

Approximate version → era map (confirm exact values per staged client in P1):

| M2 version (hdr 0x04) | Era / builds | Skin profiles |
| --- | --- | --- |
| 256 (0x100) | Classic 1.x incl. 1.12.1 | **embedded** — layout classification required |
| ~257–263 | TBC 2.0–2.4.3 | **embedded** |
| 264 (0x108) | WotLK 3.x | **external `.skin`** |
| ≥ 265 (0x109+) | Cata+ (current code's `CataVersionThreshold`) | external + chunked |
| 256 (0x100) | **WoW 1.0.0** (Vanilla release) | **embedded classic layout** — **CONFIRMED via Ghidra** (see `research-1.0.0-ghidra-trace.md`) |
| pre-256 | Alphas 0.11 / 0.12 only | **embedded**, layout undocumented — **(open P3)** |

The current reader assumes the ≥ 264 world (external skins) and so reads **no** embedded geometry for
everything ≤ 263. That is the empty-box bug.

## Decision 2 — Embedded skin profiles (`nViews` / `ofsViews`) are the missing geometry

**1.x routing correction (2026-07-15)**: 1.0.0 assets are `MD20` M2 files. The 1.0.0 client
normalizes `.mdx`/`.mdl` aliases to `.m2`; those aliases are not a fallback data format. A detected
`0x100` classic layout must be parsed only by the era-100 M2 reader and fail loudly if that reader
cannot consume it.

**Decision**: For version ≤ 263, read the embedded skin-profile table the header points at and extract
geometry + material bindings from it. **Rationale**: pre-WotLK `.m2` files carry their "views" (skin
profiles / LODs) inline; WotLK externalized them to `ModelNN.skin`. The reader currently reads
`viewCount` (at `ViewCountOffset = 0x44` in the existing code) but passes
`embeddedSkinProfileOffset: 0`, so it never walks them. **Alternatives rejected**: looking for external
`.skin` files (they don't exist for these builds).

**Known structural shape of an embedded skin profile / "view"** (old M2, pre-WotLK — confirm exact field
order/offsets P1):
- index list (`nIndex` / `ofsIndex`) — indices into the model's global vertex array
- triangle list (`nTris` / `ofsTris`) — indices into the index list (3 per face)
- vertex-property list (`nProps` / `ofsProps`) — bone influence bytes
- **submeshes / geosets** (`nSub` / `ofsSub`) — each: submesh id, vertexStart, vertexCount, triangleStart,
  triangleCount, plus bone/center/bounds fields. **This is what maps to a drawable mesh section.**
- **texture units / batches** (`nTex` / `ofsTex`) — each binds a submesh to a texture/material +
  render flags. **This is the material binding.**
- lod / header scalar fields

**Bones/vertices** live in the main M2 header (already read: `BoneCountOffset = 0x2C`, and the vertex
block referenced elsewhere). The submesh vertexStart/Count index into that global vertex array; the
skin's index list provides the draw order. The renderer needs: global vertices (have), per-submesh
index ranges (missing), and per-submesh texture bindings (missing).

## Decision 3 — Header offset deltas: verify, don't assume, per version

**Decision**: Treat the existing fixed header offsets (`ViewCountOffset = 0x44`, `BoundsOffset = 0xA0`,
`LightCountOffset = 0x108`, …) as the **WotLK layout** and verify each in-scope version's header against a
hex dump before reusing them. **Rationale**: bounds parse correctly today (so `0xA0` is right, or close,
for the tested files), which proves the *front* of the header aligns — but geometry-related fields
further in may shift in older/alpha layouts; the alphas especially are known to move fields.
**Alternatives rejected**: assuming one layout for all ≤ 263 (the P2/P3 versions are exactly where that
breaks).

Immediate P1 verification targets: the offset of `nViews`/`ofsViews` for 256 and 263 (the code reads a
count at `0x44` — confirm the paired offset field is at `0x48` and that both are correct for these
versions).

## Decision 4 — Investigation order by documentation availability

**Decision**: 2.4.3 + 1.12.1 first (documented, reference implementations exist), then 2.0.0-alpha/2.1/
2.2/2.3, then 1.0.0/0.12/0.11. **Rationale**: build and prove the embedded-skin reader where an
independent oracle exists, so that when we reach the undocumented alphas we only need x64dbg for the
genuine unknowns, not the whole structure. **Alternatives rejected**: chronological (oldest-first) order
— it front-loads the hardest, least-documented work with no validated baseline.

## Decision 5 — Tooling: Ghidra (static) + x64dbg (dynamic) both available

**Update (2026-07-15)**: Ghidra IS now installed (`H:\ghidra_11.3.2_PUBLIC`) with the
GhidraMCP plugin + `bridge_mcp_ghidra.py` wired into `.mcp.json` as the `ghidra` server.
The 1.0.0 client was fully traced statically (see `research-1.0.0-ghidra-trace.md`) — the
MD20 parser decompilation gave the complete header field map and the version-`0x100`
rejection root cause without any dynamic tracing. **Decision**: prefer Ghidra static RE
for format-reading surfaces (it yields exact offsets/sizes directly); keep x64dbg for
runtime/draw-path questions the static view can't answer. The original "Ghidra not
installed" rationale is obsolete.

## Open unknowns (resolved during implementation)

- **U1 (P1)**: Exact byte offset + field order of `nViews`/`ofsViews` and the embedded skin-profile struct
  for version 256 and 263. Resolve via hex dump of a known 1.12.1 and 2.4.3 M2 cross-checked with a
  reference implementation.
- **U2 (P1)**: Whether the old submesh struct and texture-unit struct field sizes match the WotLK
  `.skin` structs or differ (they are known to differ in some fields). Resolve by parsing a known model
  and confirming triangle/submesh counts render correctly.
- **U3 (P2)**: Any header/skin offset deltas between 2.0.0-alpha and 2.1–2.4.3. Resolve by diffing parses
  across the mid-range staged clients.
- **U4 (P3)**: The header layout of the pre-256 alpha M2s (0.11, 0.12, 1.0.0) — field positions and
  whether the skin-profile struct differs further. Resolve via x64dbg trace of each alpha client's M2
  load routine.
- **U5 (all)**: Vertex struct stability across eras (bone weight/index packing). Out of scope for the
  empty-box fix, but flag if geometry renders *deformed* (distinct failure) so it's tracked separately.

## Non-goals (reaffirmed from spec)

Animation, particles, ribbons, attachments, and bone-driven deformation correctness are out of scope.
The target is static mesh + material rendering. WotLK+ (≥ 264) must not regress.
