# MOBA — Group Render Batches

## Summary
Group render-batch records used for draw segmentation.

## Parent Chunk
`MOGP` (group file).

## Build 0.7.0.3694 evidence
- Required sixth subchunk in `FUN_006c1a10`.
- Count formula: `count = chunkSize >> 5` (32-byte stride).

## Confidence
- Presence/order/stride: **High**

## Runtime behavior (0.7.0.3694)

- `FUN_006c17f0` iterates `MOBA` entries after parse (`count = group+0x120`).
- For each batch, it reads material ID byte at `batchEntry + 0x17` and calls `FUN_006c14c0(materialId)`.
- `FUN_006c14c0` resolves textures from root `MOMT/MOTX` data and caches runtime texture handles in material slots.
- Practical effect: `MOBA` drives draw segmentation **and** material/texture binding setup.
