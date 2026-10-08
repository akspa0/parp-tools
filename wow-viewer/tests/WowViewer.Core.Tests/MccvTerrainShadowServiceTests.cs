using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class MccvTerrainShadowServiceTests
{
    [Fact]
    public void GetChunkVertexCoordinates_Has145VerticesWithCorrectOuterAndInnerLattice()
    {
        (float U, float V)[] coords = MccvTerrainShadowService.GetChunkVertexCoordinates();

        Assert.Equal(145, coords.Length);

        // First vertex is top-left outer: (0, 0)
        Assert.Equal(0f, coords[0].U, precision: 4);
        Assert.Equal(0f, coords[0].V, precision: 4);

        // 81st vertex (index 80) is bottom-right outer: (1, 1)
        Assert.Equal(1f, coords[80].U, precision: 4);
        Assert.Equal(1f, coords[80].V, precision: 4);

        // 82nd vertex (index 81) is first inner vertex: (0.5/8, 0.5/8) = (0.0625, 0.0625)
        Assert.Equal(0.0625f, coords[81].U, precision: 4);
        Assert.Equal(0.0625f, coords[81].V, precision: 4);

        // 145th vertex (index 144) is last inner vertex: (7.5/8, 7.5/8) = (0.9375, 0.9375)
        Assert.Equal(0.9375f, coords[144].U, precision: 4);
        Assert.Equal(0.9375f, coords[144].V, precision: 4);
    }

    [Fact]
    public void ExtractMccvTileLuminance_NeutralChunkYieldsNeutralLuminance()
    {
        var emptyChunks = new Dictionary<int, byte[]>();
        float[,] luma = MccvTerrainShadowService.ExtractMccvTileLuminance(emptyChunks, 256);

        Assert.Equal(256, luma.GetLength(0));
        Assert.Equal(256, luma.GetLength(1));

        // Unpopulated chunks default to neutral (127 / 255 ~= 0.498f)
        float sample = luma[128, 128];
        Assert.InRange(sample, 0.49f, 0.51f);
    }

    [Fact]
    public void SynthesizeMccvChunks_PreservesGradientsAndPacks580BytesBgra()
    {
        // Create 256x256 linear gradient from 0.1 to 0.9
        var gradient = new float[256, 256];
        for (int y = 0; y < 256; y++)
        {
            for (int x = 0; x < 256; x++)
            {
                gradient[y, x] = 0.1f + (0.8f * (x / 255.0f));
            }
        }

        Dictionary<int, byte[]> chunks = MccvTerrainShadowService.SynthesizeMccvChunks(gradient);

        Assert.Equal(256, chunks.Count);
        foreach ((int index, byte[] bgra) in chunks)
        {
            Assert.Equal(580, bgra.Length);

            // Verify BGRA structure: B == G == R, Alpha == 255
            for (int v = 0; v < 145; v++)
            {
                int off = v * 4;
                Assert.Equal(bgra[off + 0], bgra[off + 1]); // B == G
                Assert.Equal(bgra[off + 1], bgra[off + 2]); // G == R
                Assert.Equal(255, bgra[off + 3]);          // Alpha == 255
            }
        }

        // Roundtrip rasterization
        float[,] raster = MccvTerrainShadowService.ExtractMccvTileLuminance(chunks, 256);
        float ncc = MccvTerrainShadowService.ComputeNormalizedCrossCorrelation(gradient, raster);

        // High fidelity roundtrip correlation
        Assert.True(ncc >= 0.95f, $"Expected NCC >= 0.95, got {ncc:F4}");
    }

    [Fact]
    public void InjectMccvIntoChunks_PopulatesMccvColorsAndSetsMcnkFlag()
    {
        var chunks = new List<LkMcnkData>();
        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                chunks.Add(new LkMcnkData
                {
                    IndexX = cx,
                    IndexY = cy,
                    Flags = 0,
                });
            }
        }

        var residual = new float[256, 256];
        for (int y = 0; y < 256; y++)
        {
            for (int x = 0; x < 256; x++)
            {
                residual[y, x] = 0.5f;
            }
        }

        IReadOnlyList<LkMcnkData> injected = MccvTerrainShadowService.InjectMccvIntoChunks(chunks, residual);

        Assert.Equal(256, injected.Count);
        foreach (LkMcnkData chunk in injected)
        {
            Assert.True((chunk.Flags & MccvTerrainShadowService.McnkHasMccvFlag) != 0);
            Assert.NotNull(chunk.MccvColors);
            Assert.Equal(580, chunk.MccvColors.Length);
        }
    }

    [Fact]
    public void ComputeNormalizedCrossCorrelation_EvaluatesSignalsAccurately()
    {
        var s1 = new float[32, 32];
        var s2 = new float[32, 32];
        var s3 = new float[32, 32];

        for (int y = 0; y < 32; y++)
        {
            for (int x = 0; x < 32; x++)
            {
                float v = (x + y) / 64.0f;
                s1[y, x] = v;
                s2[y, x] = v;
                s3[y, x] = 1.0f - v; // Inverted
            }
        }

        float nccIdentical = MccvTerrainShadowService.ComputeNormalizedCrossCorrelation(s1, s2);
        float nccInverted = MccvTerrainShadowService.ComputeNormalizedCrossCorrelation(s1, s3);

        Assert.Equal(1.0f, nccIdentical, precision: 4);
        Assert.Equal(-1.0f, nccInverted, precision: 4);
    }

    [Fact]
    public void ExtractMccvTileLuminance_RealClassicBetaTile_ExtractsAndRasterizes()
    {
        const string installDir = @"I:\wow12\World of Warcraft";
        if (!Directory.Exists(installDir)) return;

        var storage = WowViewer.Core.IO.Casc.CascStorage.OpenLocal(installDir, "wow_classic_beta", @"output\cache\casc", allowCdnFill: false);
        byte[]? Read(uint id) => id != 0 && storage.TryReadFile(id, out byte[]? b) == WowViewer.Core.IO.Casc.CascReadStatus.Ok ? b : null;

        // Azeroth WDT FileDataID: 775971
        byte[]? wdt = Read(775971);
        if (wdt == null) return;

        int maidOffset = -1;
        for (int i = 0; i + 8 <= wdt.Length; i++)
        {
            if ((wdt[i] == 'D' && wdt[i + 1] == 'I' && wdt[i + 2] == 'A' && wdt[i + 3] == 'M') ||
                (wdt[i] == 'M' && wdt[i + 1] == 'A' && wdt[i + 2] == 'I' && wdt[i + 3] == 'D'))
            {
                maidOffset = i;
                break;
            }
        }
        if (maidOffset < 0) return;

        int maidPayload = maidOffset + 8;
        // Test tile (32, 48) -> slot = 32 * 64 + 48 = 2096
        int slot = (32 * 64) + 48;
        uint rootAdtId = BitConverter.ToUInt32(wdt, maidPayload + (slot * 32) + 0);
        if (rootAdtId == 0) return;

        byte[]? adtBytes = Read(rootAdtId);
        if (adtBytes == null) return;

        IReadOnlyDictionary<int, byte[]> chunkColors = AdtMccvTileImageBuilder.ReadChunkColors(adtBytes, $"root_{rootAdtId}.adt");
        Assert.NotEmpty(chunkColors);

        float[,] raster = MccvTerrainShadowService.ExtractMccvTileLuminance(chunkColors, 256);
        Assert.Equal(256, raster.GetLength(0));
        Assert.Equal(256, raster.GetLength(1));

        // Mean luminance should be in reasonable terrain lighting range [0.1, 0.9]
        double sum = 0.0;
        for (int y = 0; y < 256; y++)
            for (int x = 0; x < 256; x++)
                sum += raster[y, x];

        float mean = (float)(sum / (256 * 256));
        Assert.InRange(mean, 0.1f, 0.9f);
    }
}
