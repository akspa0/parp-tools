# Synthetic World Workload Contract

**Schema**: `v1-synthetic-world-workload`
**Workload class**: `synthetic_world_scene`

This contract describes a deterministic generated 3-D scene workload. It is not a synthetic
minimap/image contract and does not authorize GPU execution by itself.

## Required top-level fields

```json
{
  "schema": "v1-synthetic-world-workload",
  "workload_class": "synthetic_world_scene",
  "fixture_name": "dense-sparse-world-v1",
  "seed": 142,
  "resident_region_count": 4,
  "chunks_per_region": 16,
  "wmo_placements": 8,
  "wmo_groups_per_placement": 4,
  "m2_placements": 128,
  "repeated_asset_count": 32,
  "pm4_overlay_count": 4,
  "portal_link_count": 12,
  "render_pass_mix": {
    "opaque": 1,
    "alpha_tested": 1,
    "transparent": 1,
    "liquid": 1,
    "overlay": 1
  },
  "camera": {
    "position": [0, 0, 64],
    "yaw_degrees": 0,
    "pitch_degrees": -20,
    "vertical_fov_degrees": 60,
    "near_plane": 1,
    "far_plane": 8192
  },
  "nodes": []
}
```

## Validation rules

- `schema` and `workload_class` are exact string matches.
- All counts are non-negative, with region/chunk counts positive.
- `seed`, camera, and all count fields participate in deterministic generation.
- Node IDs are unique and parent IDs must refer to an earlier or root node.
- Node bounds must be ordered and finite when `can_reject_subtree=true`.
- The serialized node inventory must round-trip to the same ordered snapshot.
- A report must preserve the manifest hash and repository commit; it must not infer provenance
  from a directory name.
- An image-only minimap manifest is invalid for this schema and must be reported as
  `not_renderer_benchmark` by later tooling.

## Non-goals

- This contract does not define OpenGL buffers, shaders, GPU timestamp queries, client file paths,
  WMO portal frustum mathematics, or final render submission.
- This contract does not claim that generated proxy geometry has real-client visual fidelity.
