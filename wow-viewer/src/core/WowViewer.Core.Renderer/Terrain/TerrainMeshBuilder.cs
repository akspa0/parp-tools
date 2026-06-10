using System.Numerics;
using Silk.NET.OpenGL;
using WowViewer.Core.Maps;
using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.Core.Renderer.Terrain;

public sealed class TerrainMeshBuilder
{
    private readonly GL _gl;

    public TerrainMeshBuilder(GL gl)
    {
        _gl = gl ?? throw new ArgumentNullException(nameof(gl));
    }

    public unsafe TerrainMesh Build(int tileX, int tileY, WorldTerrainTileData tileData, int diffuseLayerCount)
    {
        int chunkCount = tileData.Chunks.Count;
        if (chunkCount == 0)
            throw new ArgumentException("Tile data has no chunks");

        const int vertsPerChunk = 145;
        const int floatsPerVert = 12;

        int vertexCount = chunkCount * vertsPerChunk;
        var vertices = new float[vertexCount * floatsPerVert];
        var chunkSlices = new byte[vertexCount];
        var texIndices = new ushort[vertexCount * 4];
        var indices = new List<ushort>(chunkCount * 768);

        var tileMin = new Vector3(float.MaxValue);
        var tileMax = new Vector3(float.MinValue);

        const int alphaSize = 64;
        var alphaShadow = new byte[alphaSize * alphaSize * 4 * 256];

        // Collect all unique texture paths from all chunks
        var texturePaths = new List<string>();
        var texturePathToLayerIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < chunkCount; i++)
        {
            var chunk = tileData.Chunks[i];
            foreach (var layer in chunk.TextureLayers)
            {
                if (!string.IsNullOrEmpty(layer.TexturePath) && !texturePathToLayerIndex.ContainsKey(layer.TexturePath))
                {
                    texturePathToLayerIndex[layer.TexturePath] = texturePaths.Count;
                    texturePaths.Add(layer.TexturePath);
                }
            }
        }

        for (int chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++)
        {
            var chunk = tileData.Chunks[chunkIndex];

            if (chunk.Heights == null || chunk.Heights.Length < vertsPerChunk)
                continue;

            int slice = (chunk.IndexY * 16) + chunk.IndexX;
            if ((uint)slice >= 256u)
                slice = chunkIndex & 255;

            var boundsMin = new Vector3(float.MaxValue);
            var boundsMax = new Vector3(float.MinValue);

            for (int i = 0; i < vertsPerChunk; i++)
            {
                GetVertexPosition(i, out int row, out int col, out bool isInner);

                float x = isInner
                    ? (col + 0.5f) * TerrainConstants.HalfCellSize
                    : col * TerrainConstants.HalfCellSize;
                float y = isInner
                    ? (row / 2 + 0.5f) * TerrainConstants.HalfCellSize
                    : row / 2 * TerrainConstants.HalfCellSize;

                float z = i < chunk.Heights.Length ? chunk.Heights[i] : 0f;

                float wx = TerrainConstants.MapOrigin - (tileY * TerrainConstants.TileSize) - (chunk.IndexY * TerrainConstants.ChunkSize) - y;
                float wy = TerrainConstants.MapOrigin - (tileX * TerrainConstants.TileSize) - (chunk.IndexX * TerrainConstants.ChunkSize) - x;

                int vb = (chunkIndex * vertsPerChunk + i) * floatsPerVert;
                vertices[vb + 0] = wx;
                vertices[vb + 1] = wy;
                vertices[vb + 2] = z;

                var normal = ComputeDefaultNormal();
                vertices[vb + 3] = normal.X;
                vertices[vb + 4] = normal.Y;
                vertices[vb + 5] = normal.Z;

                vertices[vb + 6] = isInner ? (col + 0.5f) / 8f : col / 8f;
                vertices[vb + 7] = isInner ? (row / 2 + 0.5f) / 8f : (row / 2) / 8f;

                vertices[vb + 8] = 127f / 255f;
                vertices[vb + 9] = 127f / 255f;
                vertices[vb + 10] = 127f / 255f;
                vertices[vb + 11] = 127f / 255f;

                int vi = chunkIndex * vertsPerChunk + i;
                chunkSlices[vi] = (byte)slice;

                const ushort invalidTex = 0xFFFF;
                ushort tex0 = invalidTex;
                ushort tex1 = invalidTex;
                ushort tex2 = invalidTex;
                ushort tex3 = invalidTex;

                for (int li = 0; li < chunk.TextureLayers.Count && li < 4; li++)
                {
                    var layer = chunk.TextureLayers[li];
                    if (!string.IsNullOrEmpty(layer.TexturePath) && texturePathToLayerIndex.TryGetValue(layer.TexturePath, out int layerIdx))
                    {
                        ushort idx = (ushort)layerIdx;
                        if (li == 0) tex0 = idx;
                        else if (li == 1) tex1 = idx;
                        else if (li == 2) tex2 = idx;
                        else if (li == 3) tex3 = idx;
                    }
                }

                int ti = vi * 4;
                texIndices[ti + 0] = tex0;
                texIndices[ti + 1] = tex1;
                texIndices[ti + 2] = tex2;
                texIndices[ti + 3] = tex3;

                boundsMin = Vector3.Min(boundsMin, new Vector3(wx, wy, z));
                boundsMax = Vector3.Max(boundsMax, new Vector3(wx, wy, z));
            }

            tileMin = Vector3.Min(tileMin, boundsMin);
            tileMax = Vector3.Max(tileMax, boundsMax);

            int baseVertex = chunkIndex * vertsPerChunk;
            var chunkIndices = BuildIndices(chunk.HoleMask);
            for (int i = 0; i < chunkIndices.Length; i++)
                indices.Add((ushort)(chunkIndices[i] + baseVertex));

            FillAlphaShadowSlice(alphaShadow, slice, chunk);
        }

        if (indices.Count == 0)
            throw new InvalidOperationException("No indices generated for tile");

        return Upload(tileX, tileY, vertices, chunkSlices, texIndices, indices.ToArray(), tileMin, tileMax, chunkCount, texturePaths, diffuseLayerCount, alphaShadow);
    }

    private unsafe TerrainMesh Upload(
        int tileX, int tileY, float[] vertices, byte[] chunkSlices, ushort[] texIndices,
        ushort[] indices, Vector3 boundsMin, Vector3 boundsMax, int chunkCount,
        List<string> texturePaths, int diffuseLayerCount, byte[] alphaShadow)
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
        fixed (byte* ptr = chunkSlices)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)chunkSlices.Length, ptr, BufferUsageARB.StaticDraw);
        _gl.EnableVertexAttribArray(3);
        _gl.VertexAttribIPointer(3, 1, VertexAttribIType.UnsignedByte, 1, (void*)0);

        uint texIndexBuffer = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, texIndexBuffer);
        fixed (ushort* ptr = texIndices)
            _gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(texIndices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);
        _gl.EnableVertexAttribArray(4);
        _gl.VertexAttribIPointer(4, 4, VertexAttribIType.UnsignedShort, (uint)(4 * sizeof(ushort)), (void*)0);

        uint elementBuffer = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, elementBuffer);
        fixed (ushort* ptr = indices)
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(indices.Length * sizeof(ushort)), ptr, BufferUsageARB.StaticDraw);

        _gl.BindVertexArray(0);

        int layerCount = Math.Max(1, texturePaths.Count);
        uint diffuseArray = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2DArray, diffuseArray);
        _gl.TexImage3D(TextureTarget.Texture2DArray, 0, InternalFormat.Rgba8, 256, 256, (uint)layerCount, 0, PixelFormat.Rgba, PixelType.UnsignedByte, null);
        for (int layer = 0; layer < layerCount; layer++)
        {
            var defaultPixels = CreateCheckerPixels(256, 256, 96, 64, 48, 48, 48, 64);
            fixed (byte* ptr = defaultPixels)
                _gl.TexSubImage3D(TextureTarget.Texture2DArray, 0, 0, 0, layer, 256, 256, 1, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
        }
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
        _gl.GenerateMipmap(TextureTarget.Texture2DArray);
        _gl.BindTexture(TextureTarget.Texture2DArray, 0);

        uint alphaShadowArray = _gl.GenTexture();
        _gl.BindTexture(TextureTarget.Texture2DArray, alphaShadowArray);
        fixed (byte* ptr = alphaShadow)
            _gl.TexImage3D(TextureTarget.Texture2DArray, 0, InternalFormat.Rgba8, 64, 64, 256, 0, PixelFormat.Rgba, PixelType.UnsignedByte, ptr);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
        _gl.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);
        _gl.BindTexture(TextureTarget.Texture2DArray, 0);

        var mesh = new TerrainMesh
        {
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
            TexturePaths = texturePaths,
            DiffuseLayerCount = texturePaths.Count,
            DiffuseArrayTexture = diffuseArray,
            AlphaShadowArrayTexture = alphaShadowArray,
        };
        mesh.SetGl(_gl);
        return mesh;
    }

    private static void FillAlphaShadowSlice(byte[] alphaShadow, int slice, WorldTerrainChunkData chunk)
    {
        const int size = 64;
        int sliceBase = slice * size * size * 4;

        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int efx = x < 63 ? x : 62;
                int efy = y < 63 ? y : 62;

                int idx = sliceBase + (y * size + x) * 4;

                float a1 = 0f, a2 = 0f, a3 = 0f;

                foreach (var layer in chunk.TextureLayers)
                {
                    if (layer.DecodedAlpha?.AlphaMap is { Length: > 0 } alpha)
                    {
                        int ai = efy * 64 + efx;
                        byte val = ai < alpha.Length ? alpha[ai] : byte.MinValue;

                        if (layer.Index == 1) a1 = val / 255f;
                        else if (layer.Index == 2) a2 = val / 255f;
                        else if (layer.Index == 3) a3 = val / 255f;
                    }
                    else if (layer.Index == 1) a1 = 1f;
                    else if (layer.Index == 2) a2 = 1f;
                    else if (layer.Index == 3) a3 = 1f;
                }

                alphaShadow[idx + 0] = (byte)(a1 * 255);
                alphaShadow[idx + 1] = (byte)(a2 * 255);
                alphaShadow[idx + 2] = (byte)(a3 * 255);
                alphaShadow[idx + 3] = 0;
            }
        }
    }

    private static void GetVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;

        for (int currentRow = 0; currentRow < 17; currentRow++)
        {
            int rowSize = (currentRow % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = currentRow;
                col = remaining;
                isInner = (currentRow % 2) != 0;
                return;
            }

            remaining -= rowSize;
        }
    }

    private static int OuterIndex(int outerRow, int outerCol)
    {
        return outerRow * 17 + outerCol;
    }

    private static int InnerIndex(int innerRow, int innerCol)
    {
        return innerRow * 17 + 9 + innerCol;
    }

    private static Vector3 ComputeDefaultNormal()
    {
        return Vector3.UnitZ;
    }

    private static int[] BuildIndices(ushort holeMask)
    {
        var result = new List<int>(768);
        for (int cellY = 0; cellY < 8; cellY++)
        {
            for (int cellX = 0; cellX < 8; cellX++)
            {
                if (holeMask != 0)
                {
                    int holeX = cellX / 2;
                    int holeY = cellY / 2;
                    int holeBit = 1 << ((holeY * 4) + holeX);
                    if ((holeMask & holeBit) != 0)
                        continue;
                }

                int tl = OuterIndex(cellY, cellX);
                int tr = OuterIndex(cellY, cellX + 1);
                int bl = OuterIndex(cellY + 1, cellX);
                int br = OuterIndex(cellY + 1, cellX + 1);
                int center = InnerIndex(cellY, cellX);

                result.Add(center); result.Add(tr); result.Add(tl);
                result.Add(center); result.Add(br); result.Add(tr);
                result.Add(center); result.Add(bl); result.Add(br);
                result.Add(center); result.Add(tl); result.Add(bl);
            }
        }

        return result.ToArray();
    }

    private static byte[] CreateCheckerPixels(int w, int h, byte r1, byte g1, byte b1, byte r2, byte g2, byte b2)
    {
        var pixels = new byte[w * h * 4];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                bool cx = (x / 16) % 2 == 0;
                bool cy = (y / 16) % 2 == 0;
                int idx = (y * w + x) * 4;
                if (cx == cy)
                {
                    pixels[idx + 0] = r1;
                    pixels[idx + 1] = g1;
                    pixels[idx + 2] = b1;
                }
                else
                {
                    pixels[idx + 0] = r2;
                    pixels[idx + 1] = g2;
                    pixels[idx + 2] = b2;
                }
                pixels[idx + 3] = 255;
            }
        }
        return pixels;
    }
}
