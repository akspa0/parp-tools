using WowViewer.Core.Runtime.World.Terrain.Stratigraphy;
using Xunit;

namespace WowViewer.Core.Tests;

public sealed class StratigraphyLevelAnalyzerTests
{
    [Fact]
    public void CountSurvivingLevels_BitExactFlat_ReturnsOne()
    {
        float[] flat = new float[257 * 257];
        Array.Fill(flat, 10.5f);

        int levels = StratigraphyLevelAnalyzer.CountSurvivingLevels(flat);
        Assert.Equal(1, levels);
    }

    [Fact]
    public void CountSurvivingLevels_QuantizedTerrain_ReturnsExactCount()
    {
        float[] values = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f];
        float[] data = new float[100];
        for (int i = 0; i < data.Length; i++)
            data[i] = values[i % values.Length];

        int levels = StratigraphyLevelAnalyzer.CountSurvivingLevels(data);
        Assert.Equal(5, levels);
    }

    [Fact]
    public void AnalyzeTile_ClassicErasureHeightmap_ClassifiesClassicErasureStratum()
    {
        float[,] lattice = new float[257, 257];
        // Squeezed mountain with 0.15m height range
        for (int y = 0; y < 257; y++)
            for (int x = 0; x < 257; x++)
                lattice[y, x] = MathF.Sin(x * 0.05f) * MathF.Cos(y * 0.05f) * 0.15f;

        StratigraphyTileAnalysis analysis = StratigraphyLevelAnalyzer.AnalyzeTile(lattice, tileX: 30, tileY: 15, tileName: "Kalimdor_30_15");

        Assert.True(analysis.IsWeakSignalCandidate);
        Assert.Equal(TemporalStratum.ClassicErasure_33x, analysis.DominantStratum);
        Assert.True(analysis.TotalSurvivingLevels > 50);
        Assert.True(analysis.SuggestedAmplificationFactor >= 30f);
    }

    [Fact]
    public void ClassifyChunkStratum_HoledChunk_ReturnsHoledDevMesh()
    {
        ushort holeMask = 0x000F;
        TemporalStratum stratum = StratigraphyLevelAnalyzer.ClassifyChunkStratum(minH: 0f, maxH: 50f, survivingLevels: 64, holeMask: holeMask);
        Assert.Equal(TemporalStratum.Holed_DevMesh_1x, stratum);
    }

    [Fact]
    public void ClassifyChunkStratum_DeepProtoTrace_ReturnsDeepProto()
    {
        TemporalStratum stratum = StratigraphyLevelAnalyzer.ClassifyChunkStratum(minH: 0f, maxH: 0.005f, survivingLevels: 12, holeMask: 0);
        Assert.Equal(TemporalStratum.DeepProto_64x_512x, stratum);
    }
}
