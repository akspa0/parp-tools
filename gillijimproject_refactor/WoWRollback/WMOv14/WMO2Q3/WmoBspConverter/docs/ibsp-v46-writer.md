# IBSP v46 Writer Notes

Lumps (17) in order:
0 Entities (null-terminated text)
1 Textures
2 Planes
3 Nodes
4 Leaves
5 LeafFaces
6 LeafBrushes
7 Models
8 Brushes
9 BrushSides
10 Vertices (44 bytes each)
11 MeshVertices (int32 indices)
12 Effects
13 Faces (104 bytes each)
14 Lightmaps
15 LightGrid
16 VisData

Struct highlights:
- Face: texture,effect,type,firstVertex,numVertices,firstMeshVertex,numMeshVertices,lightmap, (lightmap start/size), origin, vecS, vecT, normal, patchSize.
- Node/Leaf AABB: int32. Node size 36, Leaf size 48.
- Mesh workflow: set Type=3; fill MeshVertices with 0,1,2 per triangle and point FirstMeshVertex/NumMeshVertices appropriately.

Gotchas:
- Entities must be null-terminated. Many loaders assume trailing \0.
- 4-byte alignment after each lump.
- Version must be 46 with magic IBSP.
