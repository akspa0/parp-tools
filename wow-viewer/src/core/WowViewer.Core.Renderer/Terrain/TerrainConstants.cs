namespace WowViewer.Core.Renderer.Terrain;

public static class TerrainConstants
{
    public const float TileSize = 533.33333f;
    public const float ChunkSize = TileSize / 16f;
    public const float CellSize = ChunkSize / 8f;
    public const float HalfCellSize = CellSize / 2f;
    public const int ChunksPerTile = 256;
    public const int RowsPerChunk = 17;
    public const int ColsPerChunk = 17;
    public const int VertsPerChunk = 145;
    public const int AlphaSize = 64;
    public const float TextureWorldScale = 8f / ChunkSize;
    public const float MapOrigin = 32f * TileSize;
    public const int MaxTextureLayers = 4;
    public const ushort InvalidTextureIndex = 0xFFFF;

    public static float TileCenterX(int tileX) => (tileX - 32f) * TileSize + TileSize / 2f;
    public static float TileCenterY(int tileY) => (tileY - 32f) * TileSize + TileSize / 2f;
}
