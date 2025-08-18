# Data Notes

- Pools: `scene.Vertices`, `scene.MscnVertices` (MSCN appended after regular).
- Surfaces (MSUR): contain N = (Nx,Ny,Nz), Height, MsviFirstIndex, IndexCount, CompositeKey, GroupKey.
- Potential patching: high-detail regions might be represented by sub-divided surfaces with varying Height steps.
