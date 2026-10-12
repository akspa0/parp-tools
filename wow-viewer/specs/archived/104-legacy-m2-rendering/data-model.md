# Phase 1 Data Model: Legacy M2 embedded geometry

The entities the reader must recover from a ≤ 263 `.m2` to render mesh + materials. Field offsets
marked **(confirm)** are verified per version during P1/P3 against a hex dump + reference impl and then
recorded in [contracts/m2-format-profile.md](contracts/m2-format-profile.md). This model describes the
data *shape*, not the C# types.

## M2 Header (front of the .m2)

Already partly parsed by the current reader (magic, version, bounds). Relevant fields for this feature:

| Field | Current reader offset | Notes |
| --- | --- | --- |
| magic `"MD20"` | 0x00 | present |
| version | 0x04 | **the format discriminator** (Decision 1) |
| global vertex block | (read elsewhere) | the global vertex array submeshes index into |
| bone count / offset | 0x2C / 0x30 | already read; bones referenced by submeshes/props |
| **view (skin profile) count** | 0x44 (`ViewCountOffset`) | already read as `viewCount` |
| **view (skin profile) offset** | 0x48 **(confirm)** | the field the current reader ignores — points at the embedded skin-profile table |
| bounds min/max/radius | 0xA0 / 0xB8 | already parse correctly (front-of-header aligned) |

The reader currently emits `embeddedSkinProfileCount: 0, embeddedSkinProfileOffset: 0`. The fix: for
version ≤ 263, populate these from the header's view count/offset.

## Embedded Skin Profile (a "view" / LOD)

For ≤ 263 there are `viewCount` of these, inline in the file at the view offset. LOD 0 (the first) is
the full-detail mesh — the one to render. Each skin profile is a small header of count/offset pairs:

| Sub-array | Element | Purpose |
| --- | --- | --- |
| **Index list** (`nIndex`/`ofsIndex`) | uint16 | indices into the global vertex array |
| **Triangle list** (`nTris`/`ofsTris`) | uint16 | 3 per face; index into the *index list* → draw order |
| Vertex properties (`nProps`/`ofsProps`) | 4 bytes | per-vertex bone influence (not needed for static render) |
| **Submeshes** (`nSub`/`ofsSub`) | Submesh struct (below) | the drawable sections |
| **Texture units** (`nTex`/`ofsTex`) | TextureUnit struct (below) | material/texture bindings |

Exact struct field order/sizes for the old (pre-WotLK) layout are **(confirm P1)** — they differ from
the WotLK `.skin` structs in some fields (Research U2).

## Submesh / Geoset

The unit of rendering and material binding. Minimum fields needed:

| Field | Purpose |
| --- | --- |
| submesh id | grouping / body-part identity (not required to render, useful for filtering) |
| vertexStart, vertexCount | slice of the global vertex array this submesh uses |
| triangleStart, triangleCount | slice of the skin's triangle list this submesh draws |
| (bounds/center fields) | present; not required for the empty-box fix |

A submesh renders as: `triangles[triangleStart .. triangleStart+triangleCount]` →
`indexList[…]` → global vertices, drawn with the material bound to this submesh.

## Texture Unit / Batch (material binding)

Associates a submesh with the texture(s) and render state to draw it:

| Field | Purpose |
| --- | --- |
| submesh index | which submesh this batch draws |
| texture / material reference | index into the model's texture and material/render-flag tables |
| render flags / blend | how to draw (opaque/alpha/etc.) |

The model-level texture and material tables are in the main header and are already read (or read via the
existing WotLK path); the missing link is the **per-submesh binding** these texture units provide.

## Relationships (what feeds the renderer)

```text
M2 Header
├── global vertices ............................ (already read)
├── texture table, material/render-flag table .. (already read)
└── view offset/count ─▶ Embedded Skin Profile[LOD 0]
                          ├── index list ─┐
                          ├── triangle list ─▶ per-submesh draw ranges
                          ├── Submeshes ──────▶ vertex/triangle ranges  ┐
                          └── Texture Units ──▶ submesh → texture/material ┘
                                                        │
                                                        ▼
                                          existing M2 render path (M2Renderer)
```

The renderer already knows how to draw indexed triangles with bound textures (it does so for WotLK via
`.skin`). The feature's job is to hand it the same shape of data sourced from the *embedded* profile.

## Validation rules

- View count/offset must be within file bounds; offset 0 with count 0 is only valid for ≥ 264 (external
  skins). For ≤ 263, count 0 or offset 0 is a parse failure → fail safe (spec FR-005).
- Every submesh's vertex/triangle ranges must lie within the index/triangle/vertex arrays; out-of-range →
  reject that submesh, keep the rest, never crash.
- Triangle count must be a multiple of 3.
- A skin profile whose parsed submeshes cover zero triangles → treat as "no geometry", fall back to
  bounding box for that model (do not claim success).
