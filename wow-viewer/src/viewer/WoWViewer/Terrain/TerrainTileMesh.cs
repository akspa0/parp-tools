using System.Numerics;
using Silk.NET.OpenGL;

namespace WoWViewer.Terrain;

public sealed class TerrainTileMesh : IDisposable
{
    public int TileX { get; init; }
    public int TileY { get; init; }

    public uint Vao { get; init; }
    public uint VboVertices { get; init; }
    public uint VboChunkSlice { get; init; }
    public uint VboTexIndices { get; init; }
    public uint Ebo { get; init; }
    public uint IndexCount { get; init; }

    public int ChunkCount { get; init; }

    public Vector3 BoundsMin { get; init; }
    public Vector3 BoundsMax { get; init; }

    public uint AlphaShadowArrayTexture { get; set; }

    /// <summary>
    /// Alpha for layers 4..7 (RGBA = layer 4,5,6,7), one slice per chunk. 0 when no chunk in the tile has
    /// more than 4 layers, which keeps the 4MB array off tiles that do not need it.
    /// </summary>
    public uint AlphaExtArrayTexture { get; set; }

    /// <summary>Highest MCLY layer count among this tile's chunks, capped at <see cref="TerrainTileMeshBuilder.MaxLayers"/>.</summary>
    public int MaxChunkLayerCount { get; init; }

    public uint DiffuseArrayTexture { get; set; }

    public int DiffuseLayerCount { get; set; }

    internal GL? Gl { get; set; }

    public void Dispose()
    {
        if (Gl == null)
            return;

        if (Vao != 0) Gl.DeleteVertexArray(Vao);
        if (VboVertices != 0) Gl.DeleteBuffer(VboVertices);
        if (VboChunkSlice != 0) Gl.DeleteBuffer(VboChunkSlice);
        if (VboTexIndices != 0) Gl.DeleteBuffer(VboTexIndices);
        if (Ebo != 0) Gl.DeleteBuffer(Ebo);
        if (AlphaShadowArrayTexture != 0) Gl.DeleteTexture(AlphaShadowArrayTexture);
        if (AlphaExtArrayTexture != 0) Gl.DeleteTexture(AlphaExtArrayTexture);
        if (DiffuseArrayTexture != 0) Gl.DeleteTexture(DiffuseArrayTexture);
    }
}

public readonly struct TerrainChunkInfo
{
    public readonly int TileX;
    public readonly int TileY;
    public readonly int ChunkX;
    public readonly int ChunkY;
    public readonly Vector3 BoundsMin;
    public readonly Vector3 BoundsMax;
    public readonly int AreaId;

    public TerrainChunkInfo(int tileX, int tileY, int chunkX, int chunkY, Vector3 boundsMin, Vector3 boundsMax, int areaId)
    {
        TileX = tileX;
        TileY = tileY;
        ChunkX = chunkX;
        ChunkY = chunkY;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        AreaId = areaId;
    }
}
