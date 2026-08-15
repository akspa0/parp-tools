using System.Numerics;

namespace WowViewer.Core.Maps;

/// <summary>
/// Shared terrain-coordinate operations used by geometry and spatial audio.
/// Terrain chunk world positions are the renderer-space corner at the high X/Y
/// side of the chunk; local file coordinates extend toward decreasing renderer X/Y.
/// </summary>
public static class TerrainCoordinateTransform
{
    /// <summary>
    /// Computes the renderer-space corner used by the terrain adapters for one
    /// resident chunk. Tile spacing is supplied explicitly because the viewer's
    /// historical map contract uses that value for its loaded-tile grid.
    /// </summary>
    public static Vector3 ChunkCorner(
        int tileX,
        int tileY,
        int chunkX,
        int chunkY,
        float mapOrigin,
        float tileSpacing,
        int chunksPerTileEdge)
    {
        if (chunksPerTileEdge <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunksPerTileEdge));

        float chunkSpacing = tileSpacing / chunksPerTileEdge;
        return new Vector3(
            mapOrigin - tileX * tileSpacing - chunkY * chunkSpacing,
            mapOrigin - tileY * tileSpacing - chunkX * chunkSpacing,
            0f);
    }

    /// <summary>
    /// Converts a chunk-local game-space X/Y/Z position into renderer space.
    /// The decoded MCSE vector is stored in game X/Y/Z order, while the renderer
    /// uses the existing axis swap and corner-minus-local convention.
    /// </summary>
    public static Vector3 FromChunkLocal(Vector3 localPosition, Vector3 chunkWorldPosition)
    {
        return new Vector3(
            chunkWorldPosition.X - localPosition.Y,
            chunkWorldPosition.Y - localPosition.X,
            localPosition.Z);
    }

    /// <summary>Returns the center of a chunk from its renderer-space corner.</summary>
    public static Vector3 ChunkCenter(Vector3 chunkWorldPosition, float chunkWidth, float surfaceZ)
    {
        float halfWidth = chunkWidth * 0.5f;
        return new Vector3(
            chunkWorldPosition.X - halfWidth,
            chunkWorldPosition.Y - halfWidth,
            surfaceZ);
    }
}
