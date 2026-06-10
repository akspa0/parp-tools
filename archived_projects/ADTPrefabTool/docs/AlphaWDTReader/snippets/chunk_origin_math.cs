// docs/AlphaWDTReader/snippets/chunk_origin_math.cs
// Purpose: Canonical 3.x world/tile/chunk origin math to align Alpha-derived data.
// Reconcile constants with wow.tools.local ADT reader logic.

namespace Snippets
{
    public static class OriginMath
    {
        // Constants mirror 3.x (WotLK) grid sizing
        // Tile size: 533.33333f meters; chunk grid: 16x16 chunks per tile.
        // Vertex layout uses 9x9 + 8x8; terrain spacing is derived below.
        public const int ChunksPerTile = 16;
        public const float TileSize = 533.33333f;
        public const float ChunkSize = TileSize / ChunksPerTile; // ~33.333333f

        // Compute the world-space origin (lower-left corner) of a chunk
        public static (float x, float y) ComputeChunkOrigin(int tileX, int tileY, int chunkX, int chunkY)
        {
            var worldX = tileX * TileSize + chunkX * ChunkSize;
            var worldY = tileY * TileSize + chunkY * ChunkSize;
            return (worldX, worldY);
        }
    }
}
