using WowViewer.Core.Runtime.World.Terrain.Stratigraphy;
using Xunit;

namespace WowViewer.Core.Tests;

public class WdlLatticeMagnetizerTests
{
    [Fact]
    public void SampleWdlHeight_BilinearInterpolation_InterpolatesCorrectly()
    {
        var height17 = new short[17, 17];
        height17[0, 0] = 100;
        height17[0, 1] = 200;
        height17[1, 0] = 100;
        height17[1, 1] = 200;

        // Sample at midpoint of first cell: normX = 0.5 / 16 = 0.03125, normY = 0.5 / 16 = 0.03125
        float sampled = WdlLatticeMagnetizer.SampleWdlHeight(height17, 0.5f / 16f, 0.5f / 16f);
        Assert.Equal(150f, sampled, 1);
    }

    [Fact]
    public void MagnetizeChunkHeights_FullStrength_LocksToWdlWithRelief()
    {
        var height17 = new short[17, 17];
        for (int y = 0; y < 17; y++)
            for (int x = 0; x < 17; x++)
                height17[y, x] = 300;

        var src145 = new float[145];
        src145[0] = 10f; // Relief delta = 10 - 0 = 10

        var result = WdlLatticeMagnetizer.MagnetizeChunkHeights(
            src145,
            height17,
            chunkX: 0,
            chunkY: 0,
            reliefFactor: 2.0f,
            polarityInverted: false,
            magnetizationStrength: 1.0f,
            anchorHeight: 0f);

        // Expect: 300 (WDL height) + 10 * 2 (Relief) = 320
        Assert.Equal(320f, result[0], 1);
    }

    [Fact]
    public void WdlFileWriter_WriteAndVerifyChunkHeaders()
    {
        var dict = new Dictionary<(int tileX, int tileY), WdlTileData>();
        var tile = new WdlTileData { HasData = true };
        tile.Height17[0, 0] = 42;
        dict[(32, 48)] = tile;

        byte[] bytes = WdlFileWriter.Write(dict);

        Assert.NotNull(bytes);
        Assert.True(bytes.Length > 16384 + 1090); // Must contain MAOF and at least 1 MARE chunk
    }
}
