using WoWMapConverter.Core.VLM;
using Xunit;

namespace WoWMapConverter.Core.Tests.VLM;

public sealed class VlmShadowAssociationServiceTests
{
    private const float TileSize = 533.33333f;
    private const float ChunkSize = TileSize / 16f;

    [Fact]
    public void AnalyzeTile_ObjectExplainsMostOfShadow_RegionIsMarkedExplained()
    {
        VlmChunkShadowBits shadowBits = CreateShadowBits(
            0,
            static shadow => FillRect(shadow, 28, 28, 8, 8));

        VlmObjectPlacement explainedObject = new(
            Name: "lamp_post",
            NameId: 1,
            UniqueId: 100,
            X: -(ChunkSize * 0.5f),
            Y: -(ChunkSize * 0.5f),
            Z: 0f,
            RotX: 0f,
            RotY: 0f,
            RotZ: 0f,
            Scale: 1f,
            Category: "m2",
            BoundsMin: [-2f, -2f, 0f],
            BoundsMax: [2f, 2f, 4f]);

        VlmChunkShadowAnalysis[] analysis = VlmShadowAssociationService.AnalyzeTile(
            [shadowBits],
            [0f, 0f, 0f],
            [explainedObject]);

        VlmChunkShadowAnalysis chunk = Assert.Single(analysis);
        VlmShadowRegion region = Assert.Single(chunk.Regions);

        Assert.True(region.ExplainedByCurrentObjects);
        Assert.Equal("explained_current", region.ScarType);
        Assert.True(region.ExplainedOverlapRatio >= 0.60f);
        Assert.True(chunk.ExplainedShadowPixelCount > 0);
        Assert.True(chunk.ExplainedShadowRatio > chunk.ResidualShadowRatio);
    }

    [Fact]
    public void AnalyzeTile_OrphanShadowRegion_IsMarkedAsScarCandidate()
    {
        VlmChunkShadowBits shadowBits = CreateShadowBits(
            0,
            static shadow => FillRect(shadow, 22, 22, 10, 10));

        VlmChunkShadowAnalysis[] analysis = VlmShadowAssociationService.AnalyzeTile(
            [shadowBits],
            [0f, 0f, 0f],
            Array.Empty<VlmObjectPlacement>());

        VlmChunkShadowAnalysis chunk = Assert.Single(analysis);
        VlmShadowRegion region = Assert.Single(chunk.Regions);

        Assert.False(region.ExplainedByCurrentObjects);
        Assert.Equal("unexplained_scar", region.ScarType);
        Assert.Equal(region.PixelCount, region.ResidualShadowPixelCount);
        Assert.Equal(0f, region.ExplainedOverlapRatio);
        Assert.Equal(1, chunk.ScarCandidateRegionCount);
        Assert.True(chunk.ScarCandidateScore >= 0.55f);
        Assert.Equal(chunk.ShadowedPixelCount, chunk.ResidualShadowPixelCount);
    }

    private static VlmChunkShadowBits CreateShadowBits(int chunkIndex, Action<byte[]> fillShadow)
    {
        byte[] shadow = Enumerable.Repeat((byte)255, 64 * 64).ToArray();
        fillShadow(shadow);
        byte[] raw = ShadowMapService.WriteShadow(shadow);
        return new VlmChunkShadowBits(chunkIndex, Convert.ToBase64String(raw));
    }

    private static void FillRect(byte[] shadow, int startX, int startY, int width, int height)
    {
        for (int y = startY; y < startY + height; y++)
        {
            for (int x = startX; x < startX + width; x++)
                shadow[(y * 64) + x] = 0;
        }
    }
}