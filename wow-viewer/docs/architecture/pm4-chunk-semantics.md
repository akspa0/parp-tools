# PM4 Chunk Semantics

**Status**: Authoritative (2026-06-09) | **Owner**: WoWViewer PM4 lane

## Why this doc exists

For ~2 years the PM4 streams were rendered as tiny wireframe pins in the viewer and described as "navmesh graph nodes" — both wrong. The 2026-06-09 visual analysis combined with code inspection of the actual mesh walk and TypeFlags classification produced a much more accurate reading. This doc is the single source of truth for what each PM4 stream is, how the streams connect, and what the saturated cyan/magenta cubes actually represent.

Any new PM4 work in `wow-viewer` MUST be consistent with this doc. The older "navmesh graph nodes" interpretation is dead and must not resurface in code, comments, glossary text, or AI session context.

## The streams, in dependency order

| Stream | What it is | One-line semantic | Reach path |
|---|---|---|---|
| **MSUR** | Surface record (one per polygon fan) | A walkable or structural polygon the client knows about | `MSUR[i].IndexCount` (fan size), `MSUR[i].MsviFirstIndex` (into `MSVI`) |
| **MSVI** | Mesh-index stream | Indices into `MSVT`; one per vertex of every polygon | `MSVI[MSUR[i].MsviFirstIndex..MsviFirstIndex+IndexCount]` → `MSVT[indices]` |
| **MSVT** | Mesh vertex positions | The actual 3D positions of every vertex of every polygon | Reached via `MSVI` |
| **MSCN** | Scene-graph connector anchors | A 3D point per `MSUR` entry, used by the client as the placement/connector anchor for that surface | `MSCN[MSUR[i].MscnRefIndex]` (post-2026-06-09 canonical name; the legacy `MdosIndex` alias was removed) |
| **MSLK** | Link record | Says: "surface at `RefIndex` connects to a path-vertex chain `MSPI[link.MspiFirstIndex..MspiFirstIndex+link.MspiIndexCount]`, and the connection has `TypeFlags`." | `link.RefIndex` (MSUR or MPRL), `link.MspiFirstIndex`+`MspiIndexCount` (MSPI window) |
| **MSPI** | Path-index stream | Indices into `MSPV`; one per entry the link's path chain uses | `MSPI[link.MspiFirstIndex..MspiFirstIndex+link.MspiIndexCount]` → `MSPV[indices]` |
| **MSPV** | Path-vertex positions | The 3D positions the client uses to draw the connection between two surfaces (wall-floor corner, roof ridge) | Reached via `MSPI` |
| **MPRL** | Per-tile position reference | A 3D position + heading; used by the client for spawn anchors | Linked from `MSLK.RefIndex` when the link is an MPRL reference (not all are) |

## MSLK.TypeFlags — the walkable/structural classifier

From `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4AssetMatchScorer.cs:16-18`:

```csharp
private const byte TypeFlag_M2Top = 0x03;          // M2 top surface (walkable)
private const byte TypeFlag_InteriorFloor = 0x10;  // WMO interior floor (walkable)
private const byte TypeFlag_ExteriorSolid = 0x12;  // WMO exterior solid (wall/structural)
```

For every connected surface (every `MSUR` reached by an `MSLK` link), the `MSLK` record carries one of these TypeFlags plus a `Subtype` (open semantics) and a `SystemFlag` (likely a constant flag, dominated by `0x8000`).

Practical reading:

- `TypeFlags ∈ {0x03, 0x10}` — the surface is **walkable** (M2 top or WMO interior floor). The polygon is a place the client lets the player stand.
- `TypeFlags == 0x12` — the surface is **structural** (WMO exterior wall). The polygon is a barrier the client blocks the player from crossing without the appropriate edge link.
- A surface with no incoming `MSLK` link is an **orphan**. Some orphans are intentional (e.g., decoration surfaces with no nav relevance); some indicate a real decode bug.

## What the cyan cubes (MSCN) and magenta cubes (MSPV) actually mean

- **Cyan cube (MSCN entry)** — A connector anchor for one `MSUR` surface. The client uses this position to attach the surface to the scene graph and to compute its placement. Every surface has a connector (when the index is in range) so every surface contributes one cyan cube.
- **Magenta cube (MSPV entry)** — A path-vertex in a connection chain. The client uses this position to draw the actual connection between two surfaces (wall-floor corner, roof ridge, buttress line). Only present when the WMO has `MSLK` records that bridge adjacent surfaces.

A connected WMO with walkable tops + structural walls + interior floors will have:
- One cyan cube per surface (every surface has a connector).
- One magenta cube per `MSPI` index reached from a bridging `MSLK` record. Disjoint sub-parts of the same composite WMO contribute zero magenta.

A spike trap (many small disjoint WMO triangles, no `MSLK` between them) will have:
- One cyan cube per spike triangle.
- Zero magenta cubes, because the spikes are not connected to each other via `MSLK` links.

A single contiguous M2 (e.g., a totem) will have:
- One cyan cube per surface.
- Many magenta cubes — every adjacent surface pair is connected by an `MSLK` link, and the link's `MSPI` window references a chain of `MSPV` vertices.

## Why this matters

This reading unlocks four new uses for the data:

1. **Walkable surface extraction (ground truth)** — The set of `MSUR` records whose `MSLK.TypeFlags ∈ {0x03, 0x10}` is the actual walkable navmesh, classified by the client itself. This is a clean ground-truth source for **liquid masks, walkable tiles, and terrain stitch artifacts** in the V18 dataset work. Out of scope for spec 051; flagged here for a future V18 spec.

2. **Wall/structure detection for roof + object masks** — The set of `MSUR` records with `MSLK.TypeFlags == 0x12` (exterior solid), plus their `MSPV` corner points, describe building footprints with structural fidelity. Useful for the V18 roof-mask/object-mask lane.

3. **Object classification (spec 051 Phase 6)** — The `cyan : magenta` ratio is a per-object topology fingerprint:
   - High ratio (cyan dominant, ~0 magenta) → disjoint decoration, batch-spawned, doesn't benefit from WMO file matching.
   - Balanced ratio (~1:1) → connected WMO, complex mesh, the matcher should run.
   - Magenta-heavy (ratio < 1) → contiguous M2 with a dense connection graph.

4. **Matcher pre-filter (spec 050/052)** — Spec 050/052 should pre-filter the candidate set with this ratio so it only runs the expensive WMO corpus search on objects whose structure suggests a single-WMO answer is plausible.

## Common pitfalls

- **`MSUR._0x18` is `MscnRefIndex`, NOT `MdosIndex`.** The 2026-06-09 rename landed: `Pm4MsurEntry.MdosIndex` is gone, the public viewer property is `Pm4OverlayObject.DominantMscnRefIndex`, the PM4 overlay cache version is `7`, and the cache signature key is `splitByMscnRef=...`. The `MDOS` chunk (destructible object state) is a completely separate, real PM4 chunk — do not confuse it with this rename.
- **`MSUR._0x10` is a signed plane-distance term, NOT a Y-up height.** Don't use it for terrain stitch; the `Pm4MsurEntry.PlaneDistance` alias makes this clearer.
- **MSPV ≠ "shared vertices" between polygons.** MSPV is the *client's* representation of the path along a connection chain. Whether two polygons actually share a vertex in the underlying WMO is irrelevant; what matters is whether the client has an `MSLK` link between them.
- **MSCN ≠ "polygon centroid".** MSCN is the placement connector for the surface. It can be a centroid, a corner, a side midpoint — whatever the client decided the connector should be. Treat it as opaque per-surface placement evidence.
- **The earlier "navmesh graph nodes" reading is wrong and dead.** Don't reintroduce it in code, comments, or AI context. The streams are a per-surface connector + a per-connection-chain path-vertex set, not a graph of nav nodes.

## PM4 overlay cache layout (spec 054)

The viewer keeps two PM4 overlay cache layers, both keyed on content not on the camera window:

1. **Per-window cache** (`Pm4OverlayCacheService`, magic `PM4C`, version `8`): one big gzip blob per `(mapName, candidateSignature)`. The candidate signature hashes the file set + the two split flags, so a different camera window produces a different signature and misses the cache. This layer is now a fast path only when the user re-visits the exact same window; the per-file layer is the new source of truth.
2. **Per-file cache** (`Pm4PerFileCacheService`, magic `PM4F`, version `8`): one tiny gzip blob per PM4 file, located at `output/cache/pm4-overlay/{dataSourceSegment}/{mapName}/files/{normalizedPath}.pm4cache`. The cache key is `(dataSourceSegment, mapName, normalizedPath)`, so two different data sources or two different maps never collide. The stamp is `(fileLength, looseFileWriteTicks)`; a length mismatch deletes the entry on read. This layer is the new source of truth for the camera-window load path.

In-memory there is also a `Pm4PerFileCache` (LRU cap 256, soft eviction) in `WorldScene` that mirrors the on-disk per-file content for the current session. The read order in `WorldScene.LoadPm4OverlayAsync` is: in-memory cache hit → on-disk per-file cache hit → fresh `BuildPm4TileObjects` decode → write to both caches. A cache hit short-circuits the line/triangle budget enforcement, which is the expensive part.

The version bump from 7 to 8 is intentional: the old per-window cache blobs are invalidated on next read because their `CacheVersion != 8`. The per-file cache uses a separate magic (`PM4F`) so the two layers do not collide on disk.

## Cross-references

- **Spec 051** (`wow-viewer/specs/051-pm4-mscn-mspv-visualization/`) — owns the per-object signature and the per-cube visualization (the work that surfaced this reading).
- **Spec 046** (`wow-viewer/specs/046-pm4-asset-matching/`) — owns the future PM4 → WMO/M2 matching pipeline. Will use TypeFlags + per-object signature as matcher pre-filters.
- **Spec 050 / 052** — owns the WMO group matching and signature matcher. Will consume the signature fields exported by spec 051.
- **Spec 054** (`wow-viewer/specs/054-pm4-camera-window-cache/`) — owns the per-file + per-window two-layer cache. The version bump 7 → 8 in the per-window cache magic and the new per-file cache layer (`PM4F`, version 8) are the artifacts of this work.
- **PM4 Color Palette** (`wow-viewer/docs/architecture/pm4-color-palette.md`) — `Pm4ColorMscn` and `Pm4ColorMspv` are the saturated cyan/magenta tokens for the cubes. Future work may add a `TypeFlags`-tinted sub-family (0x03 cyan-tinted, 0x10 white-cyan, 0x12 darker cyan) so the cyan blob reads as "walkable structure" vs "wall" at a glance.
- **`Pm4MsurEntry` / `Pm4MslkEntry` / `Pm4KnownChunkSet`** in `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4ResearchChunkModels.cs` — the canonical type definitions.
- **`Pm4RegionObjectGrouper.cs:368-381`** — the actual mesh-walking code (`MSUR.MsviFirstIndex` → `MSVI[first..first+IndexCount]` → `MSVT[vertexIndex]`).
- **`Pm4AssetMatchScorer.cs:16-18`** — the TypeFlags constants.
- **`Pm4PlacementMath.cs:677`** — the connector-key construction that uses `MSUR.MscnRefIndex` as a placement anchor.
