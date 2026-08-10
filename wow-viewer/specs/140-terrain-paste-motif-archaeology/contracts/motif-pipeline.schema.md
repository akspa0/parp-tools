# Motif Pipeline Contract v1

This is the boundary between evidence extraction/retrieval and downstream terrain reconstruction.

## Required envelope

```json
{
  "schema": "terrain-motif-guidance-v1",
  "window_id": "string",
  "source_group_id": "string",
  "source_kind": "synthetic_control | real_client | derived_review",
  "signals": {
    "observation": "available | unavailable | invalid",
    "height_reference": "available | unavailable | validation_only",
    "alpha": "available | unavailable | invalid",
    "texture_layers": "available | unavailable | invalid",
    "tileset_auxiliary": "available | unavailable | invalid",
    "object_slots": "available | unavailable | invalid"
  },
  "tileset_profile": null,
  "motif_hypotheses": [],
  "paint_order_hypothesis": null,
  "fractal_descriptor": null,
  "object_evidence": null,
  "provenance": {},
  "content_hash": "sha256:..."
}
```

## Motif hypothesis

Each item in `motif_hypotheses` MUST contain:

- `family_id` or `null` when unconfirmed;
- `status`: `unconfirmed`, `recurring`, or `rejected`;
- `matched_signals`;
- `transform`;
- `alignment_error`;
- `cross_boundary_support`;
- `confidence`;
- `source_window_ids`;
- `evidence_hash`.

## Paint-order hypothesis

`paint_order_hypothesis` is optional and MUST contain:

- `base_layer` identifying opaque layer 0 and its base texture;
- `first_paint_layer` identifying layer 1, its texture, MCAL offset, and availability;
- `ordered_layers` for layer 1+ with layer IDs, texture IDs, and MCAL offset provenance;
- `incremental_occupancy` and `cumulative_occupancy` availability;
- `paste_family_refs`;
- `relationship_status`;
- `paint_relief_score`;
- `confidence` and `evidence_hash`.

## Fail-closed rules

- A missing signal is never replaced with a target array.
- An unconfirmed hypothesis cannot be emitted as a hard scaffold.
- Opaque layer 0 cannot be serialized as a fabricated `alpha_0` array; layer 1 is the first paste/paint candidate.
- Paint/relief correlation cannot be labeled causal authoring history without independent evidence.
- A guidance bundle without source provenance, split ownership, or content hash is invalid.
- Real height may be used to score validation, but it MUST be marked `validation_only` when the bundle is built for inference.
- Object evidence is optional and cannot make a terrain bundle valid or invalid.
