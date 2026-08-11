# Data Model: World Scene Graph and Synthetic Workload Foundation

## WorldSceneNode

| Field | Type | Rule |
|---|---|---|
| `id` | string | Stable within a workload and unique within one graph; non-empty |
| `kind` | enum | Map, tile, chunk, WMO placement/group, M2 placement/attachment, PM4 structure, overlay, or synthetic proxy |
| `parent` | node or null | Exactly one parent except the graph root; cycles forbidden |
| `children` | ordered node list | No duplicate child; detach removes the parent edge |
| `local_transform` | `Matrix4x4` | Relative to parent; identity is allowed |
| `world_transform` | `Matrix4x4` | Derived from the parent chain and refreshed on reparent/transform change |
| `local_bounds_min/max` | `Vector3` | Ordered min/max coordinates; finite values required for rejectable nodes |
| `world_bounds_min/max` | `Vector3` | Conservative transformed bounds; must contain descendants when rejectable |
| `can_reject_subtree` | bool | False when bounds are unknown, incomplete, or containment is unproven |
| `is_renderable` | bool | True only for nodes contributing render content |
| `is_queryable` | bool | True only for nodes eligible for spatial queries |
| `requires_update` | bool | True for culled content that still needs time-based updates |
| `asset_key` | string or null | Shared asset identity; placement identity remains in `id` |
| `render_pass_mask` | flags | Opaque, alpha-tested, transparent, liquid, overlay, or none |
| `portal_group` | integer or null | Optional WMO group identifier for later portal traversal |

## WorldSceneGraph

- Owns one root node and all reachable descendants.
- Supports attach, detach, lookup by stable ID, depth-first enumeration, and invariant validation.
- Attachment rejects duplicate IDs, second parents, cycles, invalid local bounds, and duplicate
  child edges.
- Detach makes the complete subtree unreachable from the graph and clears parent ownership.
- Graph snapshots report total nodes, renderable/queryable/update-required counts, depth, node-kind
  counts, and non-rejectable counts.

## SyntheticWorldWorkload

| Field | Type | Rule |
|---|---|---|
| `schema` | string | `v1-synthetic-world-workload` |
| `workload_class` | enum | `synthetic_world_scene` only in this contract |
| `fixture_name` | string | Stable human-readable fixture identity |
| `seed` | unsigned integer | Required for deterministic replay |
| `resident_region_count` | positive integer | Sparse-region count, not assumed to be a dense grid |
| `chunks_per_region` | positive integer | Controls terrain subtree width |
| `wmo_placements`, `wmo_groups_per_placement` | non-negative integers | Controls nested WMO structure |
| `m2_placements`, `repeated_asset_count` | non-negative integers | Controls placement and shared-update load |
| `pm4_overlay_count` | non-negative integer | Controls overlay descendants |
| `portal_link_count` | non-negative integer | Metadata only until portal traversal phase |
| `render_pass_mix` | flags/count descriptor | Must include the declared material/pass mix |
| `camera` | position/orientation/frustum descriptor | Fixed in replay manifest |
| `graph` | serialized node inventory | Expected IDs, parent IDs, bounds, transforms, and pass descriptors |

## WorldSceneGraphSnapshot

The snapshot is a diagnostic value, not a render result. It contains total and per-kind counts,
maximum depth, root bounds, non-rejectable count, render-pass counts, update-required count, and a
stable ordered list of node IDs. It is sufficient to prove fixture identity before visibility or
GPU measurements exist.

## Invariants

1. Stable IDs are unique within a graph and workload.
2. Every non-root node has exactly one reachable parent.
3. No node is its own ancestor.
4. A rejectable parent contains every descendant world bound.
5. Transforms and bounds are finite; unknown/incomplete bounds use `can_reject_subtree=false`.
6. Detaching a subtree removes every descendant from lookup and snapshot enumeration.
7. A synthetic minimap/image record cannot be loaded as a `SyntheticWorldWorkload` without an
   explicit world-runtime adapter; this phase has no such adapter.

## Full-Map Runtime Entities

| Entity | States / fields | Rule |
|---|---|---|
| `TileResidencyRecord` | tile ID, bounds, `indexed`, `cpu_decoded`, `gpu_ready`, `retained`, priority, last-used frame | Map discovery creates only `indexed`; the camera/admission policy alone promotes later states. |
| `FrameWorkBudget` | CPU decode ms, GPU-upload count/ms, object/overlay preparation ms | Expensive work is admitted incrementally and reported when deferred; a frame may not synchronously drain an unbounded queue. |
| `OverlayWorkRecord` | owner, invalidation key, requested/started/completed frame, CPU ms, output count, deferred reason | Every overlay operation must have a named owner and cache key; a broad `overlay` duration is invalid evidence. |
| `RendererCapabilityRecord` | GL feature support, instance-buffer path, indirect/multi-draw support, fallback reason | Modern submission is selected only through this record and preserves a tested legacy fallback. |

Additional invariants:

8. Full-map index discovery does not decode ADT payloads, create terrain meshes, or instantiate world
   objects.
9. A tile becomes GPU-ready only after its CPU data and object graph are complete; eviction removes
   all higher-residency state but retains the lightweight index.
10. Overlay work may be reused only when its declared invalidation key matches; otherwise it must be
    budgeted and attributed rather than recomputed invisibly.
