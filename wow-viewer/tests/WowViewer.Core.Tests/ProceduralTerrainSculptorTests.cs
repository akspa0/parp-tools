using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Procedural;
using WowViewer.Core.Maps;
using Xunit;

namespace WowViewer.Core.Tests;

public class ProceduralTerrainSculptorTests
{
    private static RosettaAssetEntry TestAsset(string path, float size = 2f) => new(
        path,
        RosettaAssetKind.Model,
        new Vector3(-size / 2f, -size / 2f, -size / 2f),
        new Vector3(size / 2f, size / 2f, size / 2f));

    [Fact]
    public void GenerateChunkHeights_ProducesValid145VertexArray()
    {
        var assets = new List<RosettaAssetEntry>
        {
            TestAsset("character/human/male/humanmale.mdx", 2f)
        };

        var layoutOptions = new AdaptiveLayoutOptions(Density: DensityPreset.Balanced);
        List<AdaptiveExhibitPlacement> placements = AdaptiveLayoutPacker.PackExhibits(assets, layoutOptions);

        var terrainOptions = new TerrainSculptorOptions(Roughness: 0.3f, MaxSlopeDegrees: 25.0f);
        float[] heights = ProceduralTerrainSculptor.GenerateChunkHeights(0, 0, placements, terrainOptions);

        Assert.Equal(145, heights.Length);
        Assert.All(heights, h => Assert.True(float.IsFinite(h)));
    }

    [Fact]
    public void GenerateChunkHeights_EnforcesMaximumWalkableSlope()
    {
        var assets = new List<RosettaAssetEntry>
        {
            TestAsset("creature/dragon/blackdragon.mdx", 20f)
        };

        var layoutOptions = new AdaptiveLayoutOptions(Density: DensityPreset.Balanced);
        List<AdaptiveExhibitPlacement> placements = AdaptiveLayoutPacker.PackExhibits(assets, layoutOptions);

        // High roughness and amplitude test to verify slope clamp
        var terrainOptions = new TerrainSculptorOptions(
            Roughness: 1.0f,
            HillAmplitudeMeters: 30.0f,
            MaxSlopeDegrees: 25.0f);

        float[] heights = ProceduralTerrainSculptor.GenerateChunkHeights(0, 0, placements, terrainOptions);

        float maxStepAllowed = ProceduralTerrainSculptor.OuterVertexSpacing * MathF.Tan(25.0f * MathF.PI / 180f) + 0.05f;

        // Check horizontal step between adjacent outer vertices
        for (int row = 0; row < 9; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                int i0 = (row * 9) + col;
                int i1 = i0 + 1;
                float diff = MathF.Abs(heights[i1] - heights[i0]);
                Assert.True(diff <= maxStepAllowed, $"Step {diff:F2}m exceeded max allowed {maxStepAllowed:F2}m at ({row},{col})");
            }
        }
    }

    [Fact]
    public void SampleBlendedHeight_FormsSmoothRampToPodiumCenter()
    {
        var assets = new List<RosettaAssetEntry>
        {
            TestAsset("item/objectcomponents/weapon/sword_1h_01.mdx", 1.5f)
        };

        var layoutOptions = new AdaptiveLayoutOptions(Density: DensityPreset.Compact);
        List<AdaptiveExhibitPlacement> placements = AdaptiveLayoutPacker.PackExhibits(assets, layoutOptions);
        var p = placements[0];

        float centerU = p.CellU + (p.CellSize * 0.5f);
        float centerV = p.CellV + (p.CellSize * 0.5f);

        var terrainOptions = new TerrainSculptorOptions(Roughness: 0f, BaseElevation: 10f, PodiumHeightMeters: 3f);

        float centerZ = ProceduralTerrainSculptor.SampleBlendedHeight(centerU, centerV, placements, terrainOptions);
        float midRampZ = ProceduralTerrainSculptor.SampleBlendedHeight(centerU + (p.CellSize * 0.41f), centerV, placements, terrainOptions);
        float outsideZ = ProceduralTerrainSculptor.SampleBlendedHeight(centerU + (p.CellSize * 0.8f), centerV, placements, terrainOptions);

        // Center should be raised podium height
        Assert.Equal(13f, centerZ, 2);

        // Outside should be base terrain height
        Assert.Equal(10f, outsideZ, 2);

        // Mid ramp should be smoothly interpolated between base and center
        Assert.True(midRampZ > outsideZ && midRampZ < centerZ, $"Mid ramp Z {midRampZ} was not between {outsideZ} and {centerZ}");
    }
}
