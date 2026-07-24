# CLOSED — Spec 119 (object-library classifier/segmenter + quality lens)

**Closed**: 2026-07-24 by Spec 121 (`specs/121-v7-wdl-height`).

The library-side work trained and passed its own gates (classifier 0.9137 held-out, segmenter IoU
0.9921, quality lens SC-004) — but its purpose died at the deployment boundary. The minimap
retrieval PoC measured real object instances at **p50 = 10px, max = 29px** and showed the 128px
library embedding cannot discriminate blobs at that scale (every crop matched unrelated round
textures at ~0.99 cosine). Object identity does not survive minimap scale. Segmenting and
classifying objects **on minimaps** does not work and is abandoned as a task.

The precise object masks (Spec 118 `object_geometry_visible_*` arrays) are repurposed as a
**loss-side signal only** in Spec 121. Do not resurrect minimap object identity without new
evidence that the scale physics has changed (e.g. higher-resolution deployment imagery).

Kept as reference: trained checkpoints (`classifier.pt`, `segmenter.pt`), `embeddings.parquet`,
the retrieval PoC sheet, and the `harvester/spec119/` package (untouched, read-only).
