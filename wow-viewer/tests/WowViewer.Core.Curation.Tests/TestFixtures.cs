using WowViewer.Core.Maps;

namespace WowViewer.Core.Curation.Tests;

/// <summary>Small, hand-built <see cref="TerrainTileTensorPack"/> fixtures shared across this test
/// project -- a genuinely flat/blank tile and a genuinely high-relief, well-painted tile.</summary>
internal static class TestFixtures
{
    public static TerrainTileTensorPack FlatBlankPack()
    {
        // No decoded normal data at all -- matches a genuinely blank/unpopulated ADT tile more
        // realistically than "fully-covered, flat normals": a truly blank tile is one nothing
        // decoded meaningful terrain data for, not one with full valid-but-flat MCNR coverage.
        var height = new float[257, 257]; // all zero -> zero variance

        return new TerrainTileTensorPack
        {
            MapName = "Kalimdor",
            BuildKey = "alpha",
            TileX = 19,
            TileY = 12,
            Height257 = height,
            McnrNormalXyz = null,
            McnrMask257 = null,
            McalAlphaPack256 = new float[256, 256, 4],
            MclyLayerMask = new bool[16, 16, 4],
            MinimapRgb256 = new byte[256, 256, 3],
            AvailableSignals = new HashSet<string>(),
        };
    }

    public static TerrainTileTensorPack HighReliefWellPaintedPack()
    {
        var rand = new Random(42);
        var height = new float[257, 257];
        for (int y = 0; y < 257; y++)
            for (int x = 0; x < 257; x++)
                height[y, x] = MathF.Sin(x * 0.15f) * MathF.Cos(y * 0.15f) * 80f; // strong relief

        var normals = new float[257, 257, 3];
        for (int y = 0; y < 257; y++)
        {
            for (int x = 0; x < 257; x++)
            {
                // Alternate steep-tilted normals to produce high normal_relief/edge signal.
                float sign = ((x + y) % 2 == 0) ? 1f : -1f;
                normals[y, x, 0] = 0.6f * sign;
                normals[y, x, 1] = 0.5f * -sign;
                normals[y, x, 2] = 0.4f;
            }
        }

        var normalMask = new bool[257, 257];
        for (int y = 0; y < 257; y++)
            for (int x = 0; x < 257; x++)
                normalMask[y, x] = true;

        var alpha = new float[256, 256, 4];
        for (int y = 0; y < 256; y++)
        {
            for (int x = 0; x < 256; x++)
            {
                alpha[y, x, 0] = 0.4f;
                alpha[y, x, 1] = 0.6f; // additional layer heavily painted everywhere
            }
        }

        var mclyMask = new bool[16, 16, 4];
        var mclyTextureIds = new int[16, 16, 4];
        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                mclyMask[cx, cy, 0] = true;
                mclyMask[cx, cy, 1] = true;
                mclyTextureIds[cx, cy, 0] = 0;
                mclyTextureIds[cx, cy, 1] = 1;
                mclyTextureIds[cx, cy, 2] = -1;
                mclyTextureIds[cx, cy, 3] = -1;
            }
        }

        var minimap = new byte[256, 256, 3];
        for (int y = 0; y < 256; y++)
        {
            for (int x = 0; x < 256; x++)
            {
                byte v = (byte)rand.Next(0, 256);
                minimap[y, x, 0] = v;
                minimap[y, x, 1] = (byte)(255 - v);
                minimap[y, x, 2] = (byte)((v * 7) % 256);
            }
        }

        var objectMask = new float[257, 257];
        for (int y = 0; y < 257; y++)
            for (int x = 0; x < 257; x++)
                objectMask[y, x] = ((x + y) % 5 == 0) ? 1f : 0f; // moderate object contamination

        return new TerrainTileTensorPack
        {
            MapName = "Kalimdor",
            BuildKey = "alpha",
            TileX = 20,
            TileY = 12,
            Height257 = height,
            McnrNormalXyz = normals,
            McnrMask257 = normalMask,
            McalAlphaPack256 = alpha,
            MclyLayerMask = mclyMask,
            MclyTextureIds = mclyTextureIds,
            MinimapRgb256 = minimap,
            ObjectMask257 = objectMask,
            AvailableSignals = new HashSet<string> { "has_normal_xyz", "has_alpha_256", "has_mcly_texture_ids" },
        };
    }

    /// <summary>A tile whose normals encode real relief but whose height is suspiciously flat --
    /// the exact "poisoned supervision" case <c>mismatch_detector.py</c> was built to catch.</summary>
    public static TerrainTileTensorPack HeightNormalMismatchPack()
    {
        var height = new float[257, 257]; // flat: height range ~0
        var normals = new float[257, 257, 3];
        for (int y = 0; y < 257; y++)
        {
            for (int x = 0; x < 257; x++)
            {
                float sign = ((x + y) % 2 == 0) ? 1f : -1f;
                normals[y, x, 0] = 0.7f * sign;
                normals[y, x, 1] = 0.6f * -sign;
                normals[y, x, 2] = 0.3f;
            }
        }
        var normalMask = new bool[257, 257];
        for (int y = 0; y < 257; y++)
            for (int x = 0; x < 257; x++)
                normalMask[y, x] = true;

        return new TerrainTileTensorPack
        {
            MapName = "Kalimdor",
            BuildKey = "alpha",
            TileX = 21,
            TileY = 12,
            Height257 = height,
            McnrNormalXyz = normals,
            McnrMask257 = normalMask,
        };
    }
}
