using System.Numerics;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;

namespace MdxViewer.Terrain;

/// <summary>
/// Builds a single GPU mesh per tile (256 chunks) to drastically reduce draw calls.
/// Geometry is concatenated; per-vertex attributes carry chunk-slice (0..255) and 4 diffuse texture indices.
/// Also uploads a per-tile 64x64x256 RGBA8 texture array storing alpha1/alpha2/alpha3/shadow.
/// </summary>
public sealed class TerrainTileMeshBuilder
{
    private readonly GL _gl;
    private bool _ignoreHoleMask;
    internal const int AlphaShadowSliceSize = 64;
    internal const int AlphaShadowSliceCount = 256;

    public bool IgnoreHoleMask
    {
        get => _ignoreHoleMask;
        set => _ignoreHoleMask = value;
    }

    public TerrainTileMeshBuilder(GL gl)
    {
        _gl = gl;
    }

    public unsafe (TerrainTileMesh? tileMesh, List<TerrainChunkInfo> chunkInfos) BuildTileMesh(int tileX, int tileY, IReadOnlyList<TerrainChunkData> chunks)
    {
        var chunkInfos = new List<TerrainChunkInfo>(chunks.Count);
        if (chunks.Count == 0) return (null, chunkInfos);

        const int vertsPerChunk = 145;
        const int floatsPerVert = 12; // pos3, normal3, uv2, color4

        int vertexCount = chunks.Count * vertsPerChunk;
        var vertices = new float[vertexCount * floatsPerVert];
        var chunkSlice = new byte[vertexCount];
        var texIndices = new ushort[vertexCount * 4];

        // Conservative upper bound: 256 triangles * 3 indices per chunk
        var indices = new List<ushort>(chunks.Count * 256 * 3);

        // Tile bounds
        var tileMin = new Vector3(float.MaxValue);
        var tileMax = new Vector3(float.MinValue);

        var alphaShadow = BuildAlphaShadowArray(chunks);

        for (int chunkIndex = 0; chunkIndex < chunks.Count; chunkIndex++)
        {
            var chunk = chunks[chunkIndex];
            int slice = (chunk.ChunkY * 16) + chunk.ChunkX;
            if ((uint)slice >= 256u) slice = chunkIndex & 255;

            // Build vertices
            var bMin = new Vector3(float.MaxValue);
            var bMax = new Vector3(float.MinValue);

            for (int i = 0; i < vertsPerChunk; i++)
            {
                GetVertexPosition(i, out int row, out int col, out bool isInner);

                float cellSize = WoWConstants.ChunkSize / 16f;
                float subCellSize = cellSize / 8f;

                float x, y;
                if (!isInner)
                {
                    x = col * subCellSize;
                    y = row / 2 * subCellSize;
                }
                else
                {
                    x = (col + 0.5f) * subCellSize;
                    y = (row / 2 + 0.5f) * subCellSize;
                }

                float z = (i < chunk.Heights.Length) ? chunk.Heights[i] : 0f;

                float wx = chunk.WorldPosition.X - y;
                float wy = chunk.WorldPosition.Y - x;

                int vBase = (chunkIndex * vertsPerChunk + i) * floatsPerVert;
                vertices[vBase + 0] = wx;
                vertices[vBase + 1] = wy;
                vertices[vBase + 2] = z;

                var n = (i < chunk.Normals.Length) ? chunk.Normals[i] : Vector3.UnitZ;
                vertices[vBase + 3] = n.X;
                vertices[vBase + 4] = n.Y;
                vertices[vBase + 5] = n.Z;

                if (!isInner)
                {
                    vertices[vBase + 6] = col / 8f;
                    vertices[vBase + 7] = (row / 2) / 8f;
                }
                else
                {
                    vertices[vBase + 6] = (col + 0.5f) / 8f;
                    vertices[vBase + 7] = (row / 2 + 0.5f) / 8f;
                }

                var mccv = MccvColorDecoder.DecodeModulation(chunk.MccvColors, i);
                vertices[vBase + 8] = mccv.X;
                vertices[vBase + 9] = mccv.Y;
                vertices[vBase + 10] = mccv.Z;
                vertices[vBase + 11] = mccv.W;

                int vIdx = chunkIndex * vertsPerChunk + i;
                chunkSlice[vIdx] = (byte)slice;

                ushort t0 = 0xFFFF;
                ushort t1 = 0xFFFF;
                ushort t2 = 0xFFFF;
                ushort t3 = 0xFFFF;
                if (chunk.Layers.Length > 0) t0 = (ushort)Math.Clamp(chunk.Layers[0].TextureIndex, 0, 0xFFFE);
                if (chunk.Layers.Length > 1) t1 = (ushort)Math.Clamp(chunk.Layers[1].TextureIndex, 0, 0xFFFE);
                if (chunk.Layers.Length > 2) t2 = (ushort)Math.Clamp(chunk.Layers[2].TextureIndex, 0, 0xFFFE);
                if (chunk.Layers.Length > 3) t3 = (ushort)Math.Clamp(chunk.Layers[3].TextureIndex, 0, 0xFFFE);

                int tBase = vIdx * 4;
                texIndices[tBase + 0] = t0;
                texIndices[tBase + 1] = t1;
                texIndices[tBase + 2] = t2;
                texIndices[tBase + 3] = t3;

                bMin = Vector3.Min(bMin, new Vector3(wx, wy, z));
                bMax = Vector3.Max(bMax, new Vector3(wx, wy, z));
            }

            tileMin = Vector3.Min(tileMin, bMin);
            tileMax = Vector3.Max(tileMax, bMax);
            chunkInfos.Add(new TerrainChunkInfo(chunk.TileX, chunk.TileY, chunk.ChunkX, chunk.ChunkY, bMin, bMax, chunk.AreaId));

            // Indices for this chunk
            var chunkIndices = BuildIndices(chunk.HoleMask, _ignoreHoleMask);
            int baseVertex = chunkIndex * vertsPerChunk;
            for (int i = 0; i < chunkIndices.Length; i++)
                indices.Add((ushort)(chunkIndices[i] + baseVertex));

        }

        if (indices.Count == 0) return (null, chunkInfos);

        var tileMesh = Upload(tileX, tileY, vertices, chunkSlice, texIndices, indices.ToArray(), tileMin, tileMax, chunks.Count);
        UploadAlphaShadowArray(tileMesh, alphaShadow);

        return (tileMesh, chunkInfos);
    }

    private unsafe TerrainTileMesh Upload(int tileX, int tileY, float[] vertices, byte[] chunkSlice, ushort[] texIndices, ushort[] indices, Vector3 boundsMin, Vector3 boundsMax, int chunkCount)
    {
        uint vao = _gl.GenVertexArray();
        _gl.BindVertexArray(vao);

        // VBO0: vertex floats
        uint vbo0 = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo0);
        fixed (float* ptr = vertices)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(vertices.Length * sizeof(float)), ptr, BufferUsageARB.StaticDraw);

        uint stride = 12 * sizeof(float);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
        _gl.EnableVertexAttribArray(1);
        _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(2);
        _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, stride, (void*)(6 * sizeof(float)));
        _gl.EnableVertexAttribArray(5);
        _gl.VertexAttribPointer(5, 4, VertexAttribPointerType.Float, false, stride, (void*)(8 * sizeof(float)));

        // VBO1: chunk slice (uint8)
        uint vbo1 = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo1);
        fixed (byte* ptr = chunkSlice)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)chunkSlice.Length, ptr, BufferUsageARB.StaticDraw);
        _gl.EnableVertexAttribArray(3);
        _gl.VertexAttribIPointer(3, 1, VertexAttribIType.UnsignedByte, 1, (void*)0);

        // VBO2: tex indices u16x4
        uint vbo2 = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo2);
        fixed (ushort* ptr = texIndices)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(texIndices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);
        _gl.EnableVertexAttribArray(4);
        _gl.VertexAttribIPointer(4, 4, VertexAttribIType.UnsignedShort, (uint)(4 * sizeof(ushort)), (void*)0);

        // EBO
        uint ebo = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
        fixed (ushort* ptr = indices)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

        _gl.BindVertexArray(0);

        return new TerrainTileMesh
        {
            Gl = _gl,
            TileX = tileX,
            TileY = tileY,
            Vao = vao,
            VboVertices = vbo0,
            VboChunkSlice = vbo1,
            VboTexIndices = vbo2,
            Ebo = ebo,
            IndexCount = (uint)indices.Length,
            TexIndices = texIndices,
            BoundsMin = boundsMin,
            BoundsMax = boundsMax,
            ChunkCount = chunkCount,
        };
    }

    private unsafe void UploadAlphaShadowArray(TerrainTileMesh tileMesh, byte[] alphaShadow)
    {
        const int size = 64;
        const int depth = 256;

        uint tex = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2DArray, tex);

        fixed (byte* ptr = alphaShadow)
        {
            _gl.TexImage3D(TextureTarget.Texture2DArray, 0, InternalFormat.Rgba8, size, size, depth, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
        }

        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

        _gl.BindTexture(TextureTarget.Texture2DArray, 0);
        tileMesh.AlphaShadowArrayTexture = tex;
    }

    internal static byte[] BuildAlphaShadowArray(IReadOnlyList<TerrainChunkData> chunks)
    {
        var alphaShadow = new byte[AlphaShadowSliceSize * AlphaShadowSliceSize * 4 * AlphaShadowSliceCount];

        for (int chunkIndex = 0; chunkIndex < chunks.Count; chunkIndex++)
        {
            var chunk = chunks[chunkIndex];
            int slice = (chunk.ChunkY * 16) + chunk.ChunkX;
            if ((uint)slice >= AlphaShadowSliceCount)
            {
                slice = chunkIndex & 255;
            }

            FillAlphaShadowSlice(alphaShadow, slice, chunk);
        }

        return alphaShadow;
    }

    private static void FillAlphaShadowSlice(byte[] alphaShadow, int slice, TerrainChunkData chunk)
    {
        const int size = AlphaShadowSliceSize;
        int sliceBase = slice * size * size * 4;

        // Alpha maps for layers 1..3
        for (int layer = 1; layer <= 3; layer++)
        {
            int channel = layer - 1; // 0=R,1=G,2=B
            bool hasLayer = layer < chunk.Layers.Length;
            bool usesAlphaMap = hasLayer && (chunk.Layers[layer].Flags & 0x100u) != 0;

            if (chunk.AlphaMaps.TryGetValue(layer, out var alpha) && alpha != null && alpha.Length >= size * size)
            {
                for (int y = 0; y < size; y++)
                {
                    for (int x = 0; x < size; x++)
                    {
                        int dst = y * size + x;
                        alphaShadow[sliceBase + dst * 4 + channel] = alpha[dst];
                    }
                }
                continue;
            }

            // Layer exists but has no explicit alpha map: treat as full-opacity overlay.
            if (hasLayer && !usesAlphaMap)
            {
                for (int y = 0; y < size; y++)
                {
                    for (int x = 0; x < size; x++)
                    {
                        int dst = y * size + x;
                        alphaShadow[sliceBase + dst * 4 + channel] = 255;
                    }
                }
            }
        }

        // Shadow
        if (chunk.ShadowMap != null && chunk.ShadowMap.Length >= size * size)
        {
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    int dst = y * size + x;
                    alphaShadow[sliceBase + dst * 4 + 3] = chunk.ShadowMap[dst];
                }
            }
        }
    }

    // ---- Chunk mesh topology helpers (copied from TerrainMeshBuilder) ----

    private static void GetVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;

        for (int r = 0; r < 17; r++)
        {
            int rowSize = (r % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = r;
                col = remaining;
                isInner = (r % 2 != 0);
                return;
            }
            remaining -= rowSize;
        }
    }

    private static int OuterIndex(int outerRow, int outerCol) => outerRow * 17 + outerCol;
    private static int InnerIndex(int innerRow, int innerCol) => innerRow * 17 + 9 + innerCol;

    internal static ushort[] BuildIndices(int holeMask, bool ignoreHoleMask = false)
    {
        var indices = new List<ushort>(256 * 3);

        for (int cellY = 0; cellY < 8; cellY++)
        {
            for (int cellX = 0; cellX < 8; cellX++)
            {
                if (!ignoreHoleMask && holeMask != 0)
                {
                    int holeX = cellX / 2;
                    int holeY = cellY / 2;
                    int holeBit = 1 << (holeY * 4 + holeX);
                    if ((holeMask & holeBit) != 0)
                        continue;
                }

                ushort tl = (ushort)OuterIndex(cellY, cellX);
                ushort tr = (ushort)OuterIndex(cellY, cellX + 1);
                ushort bl = (ushort)OuterIndex(cellY + 1, cellX);
                ushort br = (ushort)OuterIndex(cellY + 1, cellX + 1);
                ushort center = (ushort)InnerIndex(cellY, cellX);

                indices.Add(center);
                indices.Add(tr);
                indices.Add(tl);

                indices.Add(center);
                indices.Add(br);
                indices.Add(tr);

                indices.Add(center);
                indices.Add(bl);
                indices.Add(br);

                indices.Add(center);
                indices.Add(tl);
                indices.Add(bl);
            }
        }

        return indices.ToArray();
    }
}
