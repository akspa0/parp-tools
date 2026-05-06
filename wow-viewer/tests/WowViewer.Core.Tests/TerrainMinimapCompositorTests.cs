using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

public class TerrainMinimapCompositorTests
{
    [Fact]
    public void ComputeResidual_SameInputs_ProducesZeroResidual()
    {
        byte[,,] minimap = new byte[256, 256, 3];
        byte[,,] synthetic = new byte[256, 256, 3];

        for (int y = 0; y < 256; y++)
            for (int x = 0; x < 256; x++)
                for (int c = 0; c < 3; c++)
                {
                    minimap[y, x, c] = 100;
                    synthetic[y, x, c] = 100;
                }

        float[,,] residual = TerrainMinimapCompositor.ComputeResidual(minimap, synthetic);

        for (int y = 0; y < 256; y++)
            for (int x = 0; x < 256; x++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(0f, residual[y, x, c], 0.0001f);
    }

    [Fact]
    public void ComputeResidual_DifferentInputs_ProducesCorrectResidual()
    {
        byte[,,] minimap = new byte[256, 256, 3];
        byte[,,] synthetic = new byte[256, 256, 3];

        for (int y = 0; y < 256; y++)
            for (int x = 0; x < 256; x++)
            {
                minimap[y, x, 0] = 200;
                minimap[y, x, 1] = 200;
                minimap[y, x, 2] = 200;
                synthetic[y, x, 0] = 50;
                synthetic[y, x, 1] = 50;
                synthetic[y, x, 2] = 50;
            }

        float[,,] residual = TerrainMinimapCompositor.ComputeResidual(minimap, synthetic);

        for (int y = 0; y < 256; y++)
            for (int x = 0; x < 256; x++)
            {
                Assert.Equal(150f, residual[y, x, 0], 0.0001f);
                Assert.Equal(150f, residual[y, x, 1], 0.0001f);
                Assert.Equal(150f, residual[y, x, 2], 0.0001f);
            }
    }

    [Fact]
    public void ComputeResidual_InvalidMinimapShape_Throws()
    {
        byte[,,] badMinimap = new byte[128, 128, 3];
        byte[,,] synthetic = new byte[256, 256, 3];

        Assert.Throws<ArgumentException>(() =>
            TerrainMinimapCompositor.ComputeResidual(badMinimap, synthetic));
    }

    [Fact]
    public void ComputeResidual_InvalidSyntheticShape_Throws()
    {
        byte[,,] minimap = new byte[256, 256, 3];
        byte[,,] badSynthetic = new byte[128, 128, 3];

        Assert.Throws<ArgumentException>(() =>
            TerrainMinimapCompositor.ComputeResidual(minimap, badSynthetic));
    }

    [Fact]
    public void Composite_InvalidAlphaShape_Throws()
    {
        float[,,] badAlpha = new float[128, 128, 4];
        int[,,] mclyIds = new int[16, 16, 4];
        var textures = new Dictionary<string, byte[,,]>();

        Assert.Throws<ArgumentException>(() =>
            TerrainMinimapCompositor.Composite(badAlpha, mclyIds, textures));
    }

    [Fact]
    public void Composite_InvalidMclyShape_Throws()
    {
        float[,,] alpha = new float[1024, 1024, 4];
        int[,,] badMcly = new int[8, 8, 4];
        var textures = new Dictionary<string, byte[,,]>();

        Assert.Throws<ArgumentException>(() =>
            TerrainMinimapCompositor.Composite(alpha, badMcly, textures));
    }
}