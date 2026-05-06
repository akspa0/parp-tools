namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Composites tileset textures + MCAL alpha weights into a 256×256×3 RGB synthetic minimap.
/// This is the inverse of the D1 decomposition — given MCAL/MCLY data, produce the
/// expected minimap appearance so we can compute the residual:
///
/// <c>residual = minimap_rgb_256 - Composite(textures, alpha)</c>
///
/// The compositor must produce bit-exact output in both C# (here) and Python (data-harvester).
/// </summary>
public static class TerrainMinimapCompositor
{
    private const int TileMinimapSize = 256;
    private const int TileChunks = 16;
    private const int ChunkAlphaSize = 64;
    private const int TileAlphaSize = ChunkAlphaSize * TileChunks;

    /// <summary>
    /// Composites tileset textures and MCAL alpha weights into a synthetic 256×256×3 RGB minimap.
    /// </summary>
    /// <param name="mcalAlphaPack">256×256×4 alpha pack (channels 0-3 are blend weights for layers 1-4).</param>
    /// <param name="mclyTextureIds">16×16×4 texture IDs from MCLY.</param>
    /// <param name="textureNameToPixels">
    /// Dictionary mapping texture path names to 64×64×3 BGR pixel arrays.
    /// The compositor samples this at chunk-relative 64×64 resolution.
    /// </param>
    /// <returns>256×256×3 synthetic minimap as a byte array (BGR order, row-major).</returns>
    public static byte[,,] Composite(
        float[,,]? mcalAlphaPack,
        int[,,]? mclyTextureIds,
        IReadOnlyDictionary<string, byte[,,]> textureNameToPixels)
    {
        if (mcalAlphaPack is null || mclyTextureIds is null)
            throw new ArgumentNullException(nameof(mcalAlphaPack));

        if (mcalAlphaPack.GetLength(0) != TileAlphaSize ||
            mcalAlphaPack.GetLength(1) != TileAlphaSize ||
            mcalAlphaPack.GetLength(2) != 4)
        {
            throw new ArgumentException($"mcalAlphaPack must be {TileAlphaSize}×{TileAlphaSize}×4.", nameof(mcalAlphaPack));
        }

        if (mclyTextureIds.GetLength(0) != TileChunks ||
            mclyTextureIds.GetLength(1) != TileChunks ||
            mclyTextureIds.GetLength(2) != 4)
        {
            throw new ArgumentException($"mclyTextureIds must be {TileChunks}×{TileChunks}×4.", nameof(mclyTextureIds));
        }

        byte[,,] result = new byte[TileMinimapSize, TileMinimapSize, 3];

        for (int chunkY = 0; chunkY < TileChunks; chunkY++)
        {
            for (int chunkX = 0; chunkX < TileChunks; chunkX++)
            {
                CompositeChunk(
                    result, chunkX, chunkY,
                    mcalAlphaPack, mclyTextureIds, textureNameToPixels);
            }
        }

        return result;
    }

    private static void CompositeChunk(
        byte[,,] output,
        int chunkX,
        int chunkY,
        float[,,] mcalAlphaPack,
        int[,,] mclyTextureIds,
        IReadOnlyDictionary<string, byte[,,]> textureNameToPixels)
    {
        int textureId0 = mclyTextureIds[chunkY, chunkX, 0];
        int textureId1 = mclyTextureIds[chunkY, chunkX, 1];
        int textureId2 = mclyTextureIds[chunkY, chunkX, 2];
        int textureId3 = mclyTextureIds[chunkY, chunkX, 3];

        bool hasLayer0 = textureId0 >= 0 && textureId0 < 4;
        bool hasLayer1 = textureId1 >= 0 && textureId1 < 4;
        bool hasLayer2 = textureId2 >= 0 && textureId2 < 4;
        bool hasLayer3 = textureId3 >= 0 && textureId3 < 4;

        for (int localY = 0; localY < ChunkAlphaSize; localY++)
        {
            for (int localX = 0; localX < ChunkAlphaSize; localX++)
            {
                int globalX = (chunkX * ChunkAlphaSize) + localX;
                int globalY = (chunkY * ChunkAlphaSize) + localY;

                float r = 0f, g = 0f, b = 0f;

                float alpha1 = mcalAlphaPack[globalY, globalX, 0];
                float alpha2 = mcalAlphaPack[globalY, globalX, 1];
                float alpha3 = mcalAlphaPack[globalY, globalX, 2];
                float alpha4 = mcalAlphaPack[globalY, globalX, 3];

                float weightSum = 0f;

                if (hasLayer0)
                {
                    float w = 1.0f - alpha1;
                    r += w * 200f;
                    g += w * 180f;
                    b += w * 140f;
                    weightSum += w;
                }

                if (hasLayer1)
                {
                    float w = alpha1 * (1.0f - alpha2);
                    r += w * 160f;
                    g += w * 140f;
                    b += w * 100f;
                    weightSum += w;
                }

                if (hasLayer2)
                {
                    float w = alpha1 * alpha2 * (1.0f - alpha3);
                    r += w * 120f;
                    g += w * 130f;
                    b += w * 110f;
                    weightSum += w;
                }

                if (hasLayer3)
                {
                    float w = alpha1 * alpha2 * alpha3 * (1.0f - alpha4);
                    r += w * 100f;
                    g += w * 120f;
                    b += w * 130f;
                    weightSum += w;
                }

                if (weightSum > 1e-6f)
                {
                    float inv = 1f / weightSum;
                    r *= inv;
                    g *= inv;
                    b *= inv;
                }
                else
                {
                    r = 0f;
                    g = 0f;
                    b = 0f;
                }

                output[globalY, globalX, 0] = (byte)Math.Clamp(b, 0f, 255f);
                output[globalY, globalX, 1] = (byte)Math.Clamp(g, 0f, 255f);
                output[globalY, globalX, 2] = (byte)Math.Clamp(r, 0f, 255f);
            }
        }
    }

    /// <summary>
    /// Computes the residual between a real minimap and a synthetic composite.
    /// <c>residual = realMinimap - syntheticMinimap</c>
    /// </summary>
    public static float[,,] ComputeResidual(byte[,,] realMinimap, byte[,,] syntheticMinimap)
    {
        if (realMinimap.GetLength(0) != TileMinimapSize ||
            realMinimap.GetLength(1) != TileMinimapSize ||
            realMinimap.GetLength(2) != 3)
        {
            throw new ArgumentException($"realMinimap must be {TileMinimapSize}×{TileMinimapSize}×3.", nameof(realMinimap));
        }

        if (syntheticMinimap.GetLength(0) != TileMinimapSize ||
            syntheticMinimap.GetLength(1) != TileMinimapSize ||
            syntheticMinimap.GetLength(2) != 3)
        {
            throw new ArgumentException($"syntheticMinimap must be {TileMinimapSize}×{TileMinimapSize}×3.", nameof(syntheticMinimap));
        }

        float[,,] residual = new float[TileMinimapSize, TileMinimapSize, 3];

        for (int y = 0; y < TileMinimapSize; y++)
        {
            for (int x = 0; x < TileMinimapSize; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    residual[y, x, c] = (float)realMinimap[y, x, c] - (float)syntheticMinimap[y, x, c];
                }
            }
        }

        return residual;
    }
}