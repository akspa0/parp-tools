using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 113 US1 (T004): detail mode must sample real texels (position-dependent color from a
/// high-frequency texture) while the default material-average mode stays position-independent —
/// and the detail render of a smoothly tiling texture must not collapse into the flat average.
/// </summary>
public sealed class TerrainMinimapDetailRenderTests
{
    private static TerrainTileTensorPack BuildPack()
    {
        var textureIds = new int[16, 16, 4];
        for (int cy = 0; cy < 16; cy++)
            for (int cx = 0; cx < 16; cx++)
            {
                textureIds[cy, cx, 1] = -1;
                textureIds[cy, cx, 2] = -1;
                textureIds[cy, cx, 3] = -1;
            }

        return new TerrainTileTensorPack
        {
            TileName = "detail_0_0",
            MapName = "detail",
            BuildKey = "alpha",
            SourceAdtPath = "detail.wdt#alpha-tile(0,0)",
            TileX = 0,
            TileY = 0,
            Height257 = new float[257, 257],
            MclyTextureIds = textureIds,
            MclyTextureNames = new[] { "tileset/high_freq.blp" },
            AvailableSignals = new HashSet<string> { "mcly_texture_ids" },
        };
    }

    /// <summary>A 64x64 checkerboard whose 16px blocks survive the mip needed at 1024/tile.</summary>
    private static byte[,,] CheckerboardTexture()
    {
        var texture = new byte[64, 64, 3];
        for (int y = 0; y < 64; y++)
            for (int x = 0; x < 64; x++)
            {
                byte value = (byte)(((x / 16 + y / 16) % 2 == 0) ? 255 : 0);
                texture[y, x, 0] = value;
                texture[y, x, 1] = value;
                texture[y, x, 2] = value;
            }
        return texture;
    }

    private static byte[,,] SmoothPeriodicTexture()
    {
        var texture = new byte[64, 64, 3];
        for (int y = 0; y < 64; y++)
            for (int x = 0; x < 64; x++)
            {
                double wave = 127.5
                    + 62.0 * Math.Sin(2.0 * Math.PI * x / 64.0)
                    + 48.0 * Math.Cos(2.0 * Math.PI * y / 64.0);
                byte value = (byte)Math.Clamp(Math.Round(wave), 0, 255);
                texture[y, x, 0] = value;
                texture[y, x, 1] = value;
                texture[y, x, 2] = value;
            }
        return texture;
    }

    private static double Downsample2xMae(Image<Rgba32> high, Image<Rgba32> low)
    {
        Assert.Equal(low.Width * 2, high.Width);
        Assert.Equal(low.Height * 2, high.Height);
        double error = 0;
        long samples = 0;
        for (int y = 0; y < low.Height; y++)
            for (int x = 0; x < low.Width; x++)
            {
                double average = (
                    high[x * 2, y * 2].R
                    + high[x * 2 + 1, y * 2].R
                    + high[x * 2, y * 2 + 1].R
                    + high[x * 2 + 1, y * 2 + 1].R) / 4.0;
                error += Math.Abs(average - low[x, y].R);
                samples++;
            }
        return error / samples;
    }

    private static double PixelStd(Image<Rgba32> image)
    {
        double sum = 0, sumSq = 0;
        long n = 0;
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    double luma = (row[x].R + row[x].G + row[x].B) / 3.0;
                    sum += luma;
                    sumSq += luma * luma;
                    n++;
                }
            }
        });
        double mean = sum / n;
        return Math.Sqrt(Math.Max(0, sumSq / n - mean * mean));
    }

    [Fact]
    public void DetailMode_SamplesRealTexels_MaterialAverageStaysFlat()
    {
        var pack = BuildPack();
        var textures = new Dictionary<int, byte[,,]> { [0] = CheckerboardTexture() };
        var lighting = TerrainMinimapLighting.Neutral;

        using Image<Rgba32> averageRender = TerrainMinimapCompositor.Compose(
            pack, textures, new TerrainMinimapCompositionOptions(1024, lighting, DetailTexels: false));
        using Image<Rgba32> detailRender = TerrainMinimapCompositor.Compose(
            pack, textures, new TerrainMinimapCompositionOptions(1024, lighting, DetailTexels: true));

        double averageStd = PixelStd(averageRender);
        double detailStd = PixelStd(detailRender);

        // Material-average: one flat color per texture -> near-zero variance across the tile.
        Assert.True(averageStd < 2.0, $"material-average render should be flat, std={averageStd:F2}");
        // Detail: real checkerboard texels survive at 1024 -> strong high-frequency content.
        Assert.True(detailStd > 40.0, $"detail render should carry real texel contrast, std={detailStd:F2}");
    }

    [Fact]
    public void DetailMode_UndecodableTextureFallsThroughHonestly()
    {
        var pack = BuildPack();
        // Texture id 0 declared by MCLY but NOT decodable (absent from the dictionary): the detail
        // path must fall through the same honest fallback as material-average, never fabricate.
        var textures = new Dictionary<int, byte[,,]>();

        using Image<Rgba32> detailRender = TerrainMinimapCompositor.Compose(
            pack, textures, new TerrainMinimapCompositionOptions(256, TerrainMinimapLighting.Neutral, DetailTexels: true));

        // With no decodable texture at all the blend yields black (Vector3.Zero path), identical
        // to the material-average behavior for the same inputs.
        using Image<Rgba32> averageRender = TerrainMinimapCompositor.Compose(
            pack, textures, new TerrainMinimapCompositionOptions(256, TerrainMinimapLighting.Neutral, DetailTexels: false));
        Assert.Equal(averageRender[128, 128], detailRender[128, 128]);
    }

    [Fact]
    public void DetailMode_MipFilteringKeepsTwoToOneRendersStableWithoutMoire()
    {
        var pack = BuildPack();
        var textures = new Dictionary<int, byte[,,]> { [0] = SmoothPeriodicTexture() };

        using Image<Rgba32> render1024 = TerrainMinimapCompositor.Compose(
            pack, textures, new TerrainMinimapCompositionOptions(1024, TerrainMinimapLighting.Neutral, DetailTexels: true));
        using Image<Rgba32> render512 = TerrainMinimapCompositor.Compose(
            pack, textures, new TerrainMinimapCompositionOptions(512, TerrainMinimapLighting.Neutral, DetailTexels: true));

        double mae = Downsample2xMae(render1024, render512);
        Assert.True(mae < 4.0, $"mip-correct detail renders should remain stable across 2x sampling, MAE={mae:F3}");
        Assert.Equal(8f, TerrainMinimapCompositionOptions.TextureRepeatsPerChunk);
    }

    [Fact]
    public void DefaultMode_IsUnchanged_MaterialAverage()
    {
        var pack = BuildPack();
        var textures = new Dictionary<int, byte[,,]> { [0] = CheckerboardTexture() };

        using Image<Rgba32> render = TerrainMinimapCompositor.Compose(
            pack, textures, new TerrainMinimapCompositionOptions(256, TerrainMinimapLighting.Neutral));

        // Default (no DetailTexels) must remain the flat material view -- the 256 minimap and every
        // existing consumer are unchanged by the new mode.
        Assert.True(PixelStd(render) < 2.0);
    }
}
