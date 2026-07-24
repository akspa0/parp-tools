# CLOSED — Spec 120 (minimap OBB object detector / DINOv2 placement retrieval)

**Closed**: 2026-07-24 by Spec 121 (`specs/121-v7-wdl-height`).

The curated-embeddings cleanup worked (1,473 junk assets pruned, 4,368 verified world-asset
vectors in `curated_embeddings.parquet`), but the DINOv2 retrieval pivot inherits Spec 119's
measured scale failure: minimap object instances are 5–29px blobs and carry no discriminable
identity signal at any embedding scale. Retrieval, detection, and placement of objects **from
minimaps** is abandoned.

Do not retry with a different backbone — the blocker is input resolution physics, not embedding
quality. The only noted viable variant (re-render the library through the minimap compositor at
8–32px and train at that scale) is explicitly out of scope per the user's 2026-07-24 redirect.

Kept as reference: `curated_embeddings.parquet`, `scripts/spec120_dinov2_retrieval.py`, and the
curation audit trail.
