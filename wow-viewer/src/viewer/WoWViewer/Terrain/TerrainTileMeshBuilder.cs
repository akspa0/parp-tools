using System.Numerics;
using WoWViewer.Rendering;
using Silk.NET.OpenGL;

namespace WoWViewer.Terrain;

/// <summary>
/// Builds a single GPU mesh per tile (256 chunks) to drastically reduce draw calls.
/// Geometry is concatenated; per-vertex attributes carry chunk-slice (0..255) and up to 8 diffuse texture indices.
/// Also uploads a per-tile 64x64x256 RGBA8 texture array storing alpha1/alpha2/alpha3/shadow, plus a second
/// array for alpha4..alpha7 when any chunk in the tile has more than 4 layers.
/// </summary>
public sealed class TerrainTileMeshBuilder
{
    /// <summary>
    /// Layers rendered per chunk. Measured 2026-09-16 on wow_classic_beta 1.60.1 Azeroth (188,216 chunks):
    /// 5 layers 2,176, 6 layers 295, 7 layers 14, 8 layers 1; none above 8.
    /// </summary>
    public const int MaxLayers = 8;

    private readonly GL _gl;

    public TerrainTileMeshBuilder(GL gl)
    {
        _gl = gl;
    }

    public unsafe (TerrainTileMesh? tileMesh, List<TerrainChunkInfo> chunkInfos) BuildTileMesh(int tileX, int tileY, IReadOnlyList<TerrainChunkData> chunks)
    {
        var chunkInfos = new List<TerrainChunkInfo>(chunks.Count);
        if (chunks.Count == 0)
            return (null, chunkInfos);

        const int vertsPerChunk = 145;
        const int floatsPerVert = 12;

        int vertexCount = chunks.Count * vertsPerChunk;
        var vertices = new float[vertexCount * floatsPerVert];
        var chunkSlice = new byte[vertexCount];
        var texIndices = new ushort[vertexCount * MaxLayers];
        var indices = new List<ushort>(chunks.Count * 256 * 3);

        var tileMin = new Vector3(float.MaxValue);
        var tileMax = new Vector3(float.MinValue);

        const int alphaSize = 64;
        const int sliceCount = 256;
        var alphaShadow = new byte[alphaSize * alphaSize * 4 * sliceCount];
        int maxChunkLayers = 0;
        foreach (TerrainChunkData chunk in chunks)
            maxChunkLayers = Math.Max(maxChunkLayers, Math.Min(MaxLayers, chunk.Layers.Length));
        byte[]? alphaExt = maxChunkLayers > 4 ? new byte[alphaSize * alphaSize * 4 * sliceCount] : null;

        for (int chunkIndex = 0; chunkIndex < chunks.Count; chunkIndex++)
        {
            var chunk = chunks[chunkIndex];
            int slice = (chunk.ChunkY * 16) + chunk.ChunkX;
            if ((uint)slice >= 256u)
                slice = chunkIndex & 255;

            var boundsMin = new Vector3(float.MaxValue);
            var boundsMax = new Vector3(float.MinValue);

            for (int i = 0; i < vertsPerChunk; i++)
            {
                GetVertexPosition(i, out int row, out int col, out bool isInner);

                float cellSize = WoWConstants.ChunkSize / 16f;
                float subCellSize = cellSize / 8f;

                float x;
                float y;
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

                float z = i < chunk.Heights.Length ? chunk.Heights[i] : 0f;

                float wx = chunk.WorldPosition.X - y;
                float wy = chunk.WorldPosition.Y - x;

                int vertexBase = (chunkIndex * vertsPerChunk + i) * floatsPerVert;
                vertices[vertexBase + 0] = wx;
                vertices[vertexBase + 1] = wy;
                vertices[vertexBase + 2] = z;

                var normal = i < chunk.Normals.Length ? chunk.Normals[i] : Vector3.UnitZ;
                vertices[vertexBase + 3] = normal.X;
                vertices[vertexBase + 4] = normal.Y;
                vertices[vertexBase + 5] = normal.Z;

                if (!isInner)
                {
                    vertices[vertexBase + 6] = col / 8f;
                    vertices[vertexBase + 7] = (row / 2) / 8f;
                }
                else
                {
                    vertices[vertexBase + 6] = (col + 0.5f) / 8f;
                    vertices[vertexBase + 7] = (row / 2 + 0.5f) / 8f;
                }

                float red = 127f / 255f;
                float green = 127f / 255f;
                float blue = 127f / 255f;
                float alpha = 127f / 255f;
                if (chunk.MccvColors != null)
                {
                    int colorBase = i * 4;
                    if (colorBase + 3 < chunk.MccvColors.Length)
                    {
                        // MCCV is stored as BGRA. Mid-gray (~127) is the neutral/no-tint value.
                        blue = chunk.MccvColors[colorBase + 0] / 255f;
                        green = chunk.MccvColors[colorBase + 1] / 255f;
                        red = chunk.MccvColors[colorBase + 2] / 255f;
                        alpha = chunk.MccvColors[colorBase + 3] / 255f;
                    }
                }

                vertices[vertexBase + 8] = red;
                vertices[vertexBase + 9] = green;
                vertices[vertexBase + 10] = blue;
                vertices[vertexBase + 11] = alpha;

                int vertexIndex = chunkIndex * vertsPerChunk + i;
                chunkSlice[vertexIndex] = (byte)slice;

                int texBase = vertexIndex * MaxLayers;
                for (int layer = 0; layer < MaxLayers; layer++)
                {
                    texIndices[texBase + layer] = layer < chunk.Layers.Length
                        ? (ushort)Math.Clamp(chunk.Layers[layer].TextureIndex, 0, 0xFFFE)
                        : (ushort)0xFFFF;
                }

                boundsMin = Vector3.Min(boundsMin, new Vector3(wx, wy, z));
                boundsMax = Vector3.Max(boundsMax, new Vector3(wx, wy, z));
            }

            tileMin = Vector3.Min(tileMin, boundsMin);
            tileMax = Vector3.Max(tileMax, boundsMax);
            chunkInfos.Add(new TerrainChunkInfo(chunk.TileX, chunk.TileY, chunk.ChunkX, chunk.ChunkY, boundsMin, boundsMax, chunk.AreaId));

            var chunkIndices = BuildIndices(chunk.HoleMask);
            int baseVertex = chunkIndex * vertsPerChunk;
            for (int i = 0; i < chunkIndices.Length; i++)
                indices.Add((ushort)(chunkIndices[i] + baseVertex));

            FillAlphaShadowSlice(alphaShadow, alphaExt, slice, chunk);
        }

        if (indices.Count == 0)
            return (null, chunkInfos);

        var tileMesh = Upload(tileX, tileY, vertices, chunkSlice, texIndices, indices.ToArray(), tileMin, tileMax, chunks.Count, maxChunkLayers);
        tileMesh.AlphaShadowArrayTexture = UploadRgbaArray(alphaShadow);
        if (alphaExt != null)
            tileMesh.AlphaExtArrayTexture = UploadRgbaArray(alphaExt);

        return (tileMesh, chunkInfos);
    }

    private unsafe TerrainTileMesh Upload(int tileX, int tileY, float[] vertices, byte[] chunkSlice, ushort[] texIndices, ushort[] indices, Vector3 boundsMin, Vector3 boundsMax, int chunkCount, int maxChunkLayers)
    {
        uint vao = _gl.GenVertexArray();
        _gl.BindVertexArray(vao);

        uint vertexBuffer = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vertexBuffer);
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

        uint chunkSliceBuffer = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, chunkSliceBuffer);
        fixed (byte* ptr = chunkSlice)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)chunkSlice.Length, ptr, BufferUsageARB.StaticDraw);
        _gl.EnableVertexAttribArray(3);
        _gl.VertexAttribIPointer(3, 1, VertexAttribIType.UnsignedByte, 1, (void*)0);

        uint texIndexBuffer = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, texIndexBuffer);
        fixed (ushort* ptr = texIndices)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(texIndices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);
        uint texIndexStride = (uint)(MaxLayers * sizeof(ushort));
        _gl.EnableVertexAttribArray(4);
        _gl.VertexAttribIPointer(4, 4, VertexAttribIType.UnsignedShort, texIndexStride, (void*)0);
        _gl.EnableVertexAttribArray(6);
        _gl.VertexAttribIPointer(6, 4, VertexAttribIType.UnsignedShort, texIndexStride, (void*)(4 * sizeof(ushort)));

        uint elementBuffer = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, elementBuffer);
        fixed (ushort* ptr = indices)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

        _gl.BindVertexArray(0);

        return new TerrainTileMesh
        {
            Gl = _gl,
            TileX = tileX,
            TileY = tileY,
            Vao = vao,
            VboVertices = vertexBuffer,
            VboChunkSlice = chunkSliceBuffer,
            VboTexIndices = texIndexBuffer,
            Ebo = elementBuffer,
            IndexCount = (uint)indices.Length,
            BoundsMin = boundsMin,
            BoundsMax = boundsMax,
            ChunkCount = chunkCount,
            MaxChunkLayerCount = maxChunkLayers,
        };
    }

    private unsafe uint UploadRgbaArray(byte[] alphaShadow)
    {
        const int size = 64;
        const int depth = 256;

        uint texture = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2DArray, texture);

        fixed (byte* ptr = alphaShadow)
        {
            _gl.TexImage3D(TextureTarget.Texture2DArray, 0, InternalFormat.Rgba8, size, size, depth, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
        }

        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

        _gl.BindTexture(TextureTarget.Texture2DArray, 0);
        return texture;
    }

    private static void FillAlphaShadowSlice(byte[] alphaShadow, byte[]? alphaExt, int slice, TerrainChunkData chunk)
    {
        const int size = 64;
        int sliceBase = slice * size * size * 4;

        static int EdgeFixedIndex(int x, int y)
        {
            if (x >= 63) x = 62;
            if (y >= 63) y = 62;
            return y * size + x;
        }

        for (int layer = 1; layer < MaxLayers; layer++)
        {
            // Layers 1..3 share the shadow array (RGB); layers 4..7 use the extension array (RGBA).
            byte[]? target = layer <= 3 ? alphaShadow : alphaExt;
            if (target == null)
                break;

            int channel = layer <= 3 ? layer - 1 : layer - 4;
            bool hasLayer = layer < chunk.Layers.Length;
            bool usesAlphaMap = hasLayer && (chunk.Layers[layer].Flags & 0x100u) != 0;

            if (chunk.AlphaMaps.TryGetValue(layer, out var alpha) && alpha != null && alpha.Length >= size * size)
            {
                for (int y = 0; y < size; y++)
                {
                    for (int x = 0; x < size; x++)
                    {
                        int dst = y * size + x;
                        int src = EdgeFixedIndex(x, y);
                        target[sliceBase + dst * 4 + channel] = alpha[src];
                    }
                }
                continue;
            }

            if (hasLayer && !usesAlphaMap)
            {
                for (int y = 0; y < size; y++)
                {
                    for (int x = 0; x < size; x++)
                    {
                        int dst = y * size + x;
                        target[sliceBase + dst * 4 + channel] = 255;
                    }
                }
            }
        }

        if (chunk.ShadowMap != null && chunk.ShadowMap.Length >= size * size)
        {
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    int dst = y * size + x;
                    // MCSH is a native 64x64 mask. The MCAL edge-fix does not apply to it.
                    alphaShadow[sliceBase + dst * 4 + 3] = chunk.ShadowMap[dst];
                }
            }
        }
    }

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
                isInner = r % 2 != 0;
                return;
            }

            remaining -= rowSize;
        }
    }

    private static int OuterIndex(int outerRow, int outerCol) => outerRow * 17 + outerCol;

    private static int InnerIndex(int innerRow, int innerCol) => innerRow * 17 + 9 + innerCol;

    private static ushort[] BuildIndices(int holeMask)
    {
        var indices = new List<ushort>(256 * 3);

        for (int cellY = 0; cellY < 8; cellY++)
        {
            for (int cellX = 0; cellX < 8; cellX++)
            {
                if (holeMask != 0)
                {
                    int holeX = cellX / 2;
                    int holeY = cellY / 2;
                    int holeBit = 1 << (holeY * 4 + holeX);
                    if ((holeMask & holeBit) != 0)
                        continue;
                }

                ushort topLeft = (ushort)OuterIndex(cellY, cellX);
                ushort topRight = (ushort)OuterIndex(cellY, cellX + 1);
                ushort bottomLeft = (ushort)OuterIndex(cellY + 1, cellX);
                ushort bottomRight = (ushort)OuterIndex(cellY + 1, cellX + 1);
                ushort center = (ushort)InnerIndex(cellY, cellX);

                indices.Add(center);
                indices.Add(topRight);
                indices.Add(topLeft);

                indices.Add(center);
                indices.Add(bottomRight);
                indices.Add(topRight);

                indices.Add(center);
                indices.Add(bottomLeft);
                indices.Add(bottomRight);

                indices.Add(center);
                indices.Add(topLeft);
                indices.Add(bottomLeft);
            }
        }

        return indices.ToArray();
    }
}
