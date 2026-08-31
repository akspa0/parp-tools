using System.Numerics;

namespace WowViewer.Core.Editor.Operations;

/// <summary>
/// Continuous global coordinate indexing an individual 33.334m MCNK chunk across the entire 64x64 world map.
/// Global coordinates range from 0 to 1023 along both axes (64 tiles * 16 chunks = 1024 chunks).
/// </summary>
public readonly record struct GlobalChunkCoordinate(int Gx, int Gy)
{
    public const float MapOrigin = 17066.666f;
    public const float TileSize = 533.33333f;
    public const float ChunkSize = 33.333333f;
    public const int ChunksPerTile = 16;
    public const int TotalGlobalChunksPerAxis = 1024;

    public int TileX => (int)MathF.Floor((float)Gx / ChunksPerTile);
    public int TileY => (int)MathF.Floor((float)Gy / ChunksPerTile);
    public int ChunkX => ((Gx % ChunksPerTile) + ChunksPerTile) % ChunksPerTile;
    public int ChunkY => ((Gy % ChunksPerTile) + ChunksPerTile) % ChunksPerTile;

    public bool IsValid => Gx >= 0 && Gx < TotalGlobalChunksPerAxis && Gy >= 0 && Gy < TotalGlobalChunksPerAxis;

    /// <summary>
    /// World center X coordinate (+X = North).
    /// </summary>
    public float WorldCenterX => MapOrigin - ((float)Gx + 0.5f) * ChunkSize;

    /// <summary>
    /// World center Y coordinate (+Y = West).
    /// </summary>
    public float WorldCenterY => MapOrigin - ((float)Gy + 0.5f) * ChunkSize;

    public static GlobalChunkCoordinate FromTileAndChunk(int tileX, int tileY, int chunkX, int chunkY)
    {
        return new GlobalChunkCoordinate(tileX * ChunksPerTile + chunkX, tileY * ChunksPerTile + chunkY);
    }

    public static GlobalChunkCoordinate FromWorldPosition(float worldX, float worldY)
    {
        int gx = (int)MathF.Floor((MapOrigin - worldX) / ChunkSize);
        int gy = (int)MathF.Floor((MapOrigin - worldY) / ChunkSize);
        return new GlobalChunkCoordinate(Math.Clamp(gx, 0, TotalGlobalChunksPerAxis - 1), Math.Clamp(gy, 0, TotalGlobalChunksPerAxis - 1));
    }

    public GlobalChunkCoordinate Offset(int deltaGx, int deltaGy)
    {
        return new GlobalChunkCoordinate(Gx + deltaGx, Gy + deltaGy);
    }

    public override string ToString() => $"GlobalChunk({Gx},{Gy}) [Tile({TileX},{TileY}) Chunk({ChunkX},{ChunkY})]";
}
