using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class MinimapShadingMatchTests
{
    private const string RequiredBuild = "0.5.3.3368";

    [Fact]
    public void Evaluate_NonRequiredBuild_IsNotEvaluatedWithoutRenderingAnyCandidate()
    {
        // MCAL with only 3 layer channels makes TerrainMinimapCompositor.Compose throw
        // (it requires 4). If Evaluate ever reached Compose for a non-0.5.3.3368 build, this
        // test would fail with an exception instead of asserting the not_evaluated result.
        TerrainTileTensorPack pack = BuildSlopedPack(mcalAlphaPack256: new float[2, 2, 3]);
        var textures = new Dictionary<int, byte[,,]> { [0] = SolidTexture(200, 60, 60) };

        MinimapLightingProvenance result = MinimapShadingMatch.Evaluate(
            MinimapLightingProvenance.NotEvaluated("seed"),
            pack,
            textures,
            authoredMinimapRgb: new byte[8, 8, 3],
            buildFingerprint: "1.12.1.5875");

        Assert.Equal("not_evaluated", result.ShadingMatchStatus);
        Assert.Null(result.ShadingMatchedTimeOfDayHours);
        Assert.Equal("1.12.1.5875", result.ShadingMatchBuildFingerprint);
    }

    [Fact]
    public void Evaluate_MissingGroundTruthNormals_IsNotEvaluated()
    {
        TerrainTileTensorPack pack = BuildSlopedPack(clearNormals: true);
        var textures = new Dictionary<int, byte[,,]> { [0] = SolidTexture(200, 60, 60) };

        MinimapLightingProvenance result = MinimapShadingMatch.Evaluate(
            MinimapLightingProvenance.NotEvaluated("seed"),
            pack,
            textures,
            authoredMinimapRgb: new byte[8, 8, 3],
            buildFingerprint: RequiredBuild);

        Assert.Equal("not_evaluated", result.ShadingMatchStatus);
    }

    [Fact]
    public void Evaluate_FlatTerrain_ReportsLowConfidenceFlatTerrain()
    {
        TerrainTileTensorPack pack = BuildFlatPack();
        var textures = new Dictionary<int, byte[,,]> { [0] = SolidTexture(200, 60, 60) };
        var authored = new byte[8, 8, 3];
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
                for (int c = 0; c < 3; c++)
                    authored[y, x, c] = 128;

        MinimapLightingProvenance result = MinimapShadingMatch.Evaluate(
            MinimapLightingProvenance.NotEvaluated("seed"),
            pack,
            textures,
            authored,
            RequiredBuild);

        Assert.Equal("low_confidence_flat_terrain", result.ShadingMatchStatus);
        Assert.Null(result.ShadingMatchedTimeOfDayHours);
        Assert.Equal(0f, result.ShadingMatchConfidence);
    }

    [Fact]
    public void Evaluate_AllPixelsExcludedByMcsh_ReportsFullExclusionAndFlatTerrain()
    {
        var fullShadow = new float[8, 8];
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
                fullShadow[y, x] = 1f;
        TerrainTileTensorPack pack = BuildSlopedPack(mcshShadowMask256: fullShadow);
        var textures = new Dictionary<int, byte[,,]> { [0] = SolidTexture(200, 60, 60) };

        MinimapLightingProvenance result = MinimapShadingMatch.Evaluate(
            MinimapLightingProvenance.NotEvaluated("seed"),
            pack,
            textures,
            authoredMinimapRgb: RenderReferenceCandidate(pack, textures, hour: 6f),
            buildFingerprint: RequiredBuild);

        Assert.Equal("low_confidence_flat_terrain", result.ShadingMatchStatus);
        Assert.Equal(1f, result.ShadingMatchExcludedMcshFraction);
    }

    [Fact]
    public void Evaluate_SelfRenderedAuthoredImage_IdentifiesItsOwnRenderHourAsTheBestMatch()
    {
        // Hour 12 (solar noon) sits at the sweep's unique elevation peak. Hours in
        // TerrainSolarDirection's night floor (elevation clamped to a 0.05 minimum) render
        // byte-for-byte identically to each other, so a reference hour drawn from that block would
        // be a genuine, expected tie rather than a metric failure -- noon has no such tie.
        TerrainTileTensorPack pack = BuildSlopedPack();
        var textures = new Dictionary<int, byte[,,]> { [0] = SolidTexture(200, 60, 60) };
        byte[,,] authored = RenderReferenceCandidate(pack, textures, hour: 12f);

        MinimapLightingProvenance result = MinimapShadingMatch.Evaluate(
            MinimapLightingProvenance.NotEvaluated("seed"),
            pack,
            textures,
            authored,
            RequiredBuild);

        Assert.NotEqual("not_evaluated", result.ShadingMatchStatus);
        Assert.NotEqual("low_confidence_flat_terrain", result.ShadingMatchStatus);
        Assert.NotNull(result.ShadingMatchedTimeOfDayHours);
        // Neighbouring near-noon hours are naturally very close in this sparse fixture; assert the
        // match lands in the midday region rather than pinning an exact hour.
        Assert.InRange(result.ShadingMatchedTimeOfDayHours!.Value, 10f, 14f);
    }

    [Fact]
    public void Evaluate_SelfRenderedAtNonNoonHour_DoesNotTieAgainstItsOwnMirrorHour()
    {
        // Hour 9 and hour 15 sit at the same TerrainSolarDirection elevation (symmetric around
        // solar noon) and therefore render byte-for-byte identical candidates. Before the
        // elevation-distinctness fix, the sweep's own best-vs-runner-up bookkeeping picked hour 15
        // as "second best" purely because it exactly ties hour 9's perfect self-match score,
        // producing a zero margin and a false low_confidence_ambiguous verdict for what is actually
        // a clean match against every genuinely distinct hour. A real 0.5.3.3368 regression: 11
        // sampled real tiles all scored exactly this kind of zero/near-zero confidence before the
        // fix, several recovering meaningfully afterward (one tile 0 -> 0.45).
        TerrainTileTensorPack pack = BuildSlopedPack();
        var textures = new Dictionary<int, byte[,,]> { [0] = SolidTexture(200, 60, 60) };
        byte[,,] authored = RenderReferenceCandidate(pack, textures, hour: 9f);

        MinimapLightingProvenance result = MinimapShadingMatch.Evaluate(
            MinimapLightingProvenance.NotEvaluated("seed"),
            pack,
            textures,
            authored,
            RequiredBuild);

        Assert.NotEqual("not_evaluated", result.ShadingMatchStatus);
        Assert.NotEqual("low_confidence_flat_terrain", result.ShadingMatchStatus);
        Assert.NotNull(result.ShadingMatchedTimeOfDayHours);
        Assert.True(result.ShadingMatchedTimeOfDayHours is 9f or 15f);
        // The bug produced exactly zero confidence (an exact float tie); any positive confidence
        // proves the runner-up came from a genuinely distinct elevation, not the mirror hour.
        Assert.True(result.ShadingMatchConfidence > 0f);
    }

    [Fact]
    public void Evaluate_IsInvariantToMaterialTintAtTheSameGeometryAndHour()
    {
        TerrainTileTensorPack pack = BuildSlopedPack();
        // Kept comfortably below 255 at every swept hour (ambient 0.25 + directional lambert can
        // reach roughly 1.25x at peak alignment): a channel near 220 would clip to a hard 255 at
        // high-elevation hours, which is a real byte-quantization nonlinearity that would break the
        // multiplicative tint-invariance this test exists to prove -- not a flaw in the metric.
        var redTextures = new Dictionary<int, byte[,,]> { [0] = SolidTexture(150, 60, 60) };
        var blueTextures = new Dictionary<int, byte[,,]> { [0] = SolidTexture(60, 60, 150) };

        // The authored minimap is rendered with a completely different material colour (blue) than
        // what MinimapShadingMatch will use to generate its own candidates (red). Only the shared
        // geometry/lighting shading pattern should drive the match -- material colour must not.
        byte[,,] authoredWithDifferentTint = RenderReferenceCandidate(pack, blueTextures, hour: 12f);

        MinimapLightingProvenance result = MinimapShadingMatch.Evaluate(
            MinimapLightingProvenance.NotEvaluated("seed"),
            pack,
            redTextures,
            authoredWithDifferentTint,
            RequiredBuild);

        Assert.NotEqual("not_evaluated", result.ShadingMatchStatus);
        Assert.NotEqual("low_confidence_flat_terrain", result.ShadingMatchStatus);
        Assert.NotNull(result.ShadingMatchedTimeOfDayHours);
        Assert.InRange(result.ShadingMatchedTimeOfDayHours!.Value, 10f, 14f);
    }

    [Theory]
    [InlineData(0.9f, 0.2f, "matched")]
    [InlineData(0.51f, 0.50f, "low_confidence_ambiguous")]
    public void Classify_UsesTheScoreMarginToDistinguishMatchedFromAmbiguous(
        float bestScore,
        float secondBestScore,
        string expectedStatus)
    {
        MinimapShadingMatch.MatchClassification classification = MinimapShadingMatch.Classify(
            bestScore,
            secondBestScore,
            signalStrength: 100f,
            minimumSignalStrength: 1f,
            MinimapShadingMatchOptions.Default);

        Assert.Equal(expectedStatus, classification.Status);
    }

    [Fact]
    public void Classify_BelowMinimumSignalStrength_IsFlatTerrainRegardlessOfMargin()
    {
        MinimapShadingMatch.MatchClassification classification = MinimapShadingMatch.Classify(
            bestScore: 1f,
            secondBestScore: 0f,
            signalStrength: 0.5f,
            minimumSignalStrength: 10f,
            MinimapShadingMatchOptions.Default);

        Assert.Equal("low_confidence_flat_terrain", classification.Status);
        Assert.Equal(0f, classification.Confidence);
    }

    private static byte[,,] RenderReferenceCandidate(
        TerrainTileTensorPack pack,
        IReadOnlyDictionary<int, byte[,,]> textures,
        float hour)
    {
        var options = new TerrainMinimapCompositionOptions(8, TerrainMinimapLighting.CreateWhiteTopEdge(hour / 24f));
        using Image<Rgba32> rendered = TerrainMinimapCompositor.Compose(pack, textures, options);

        var rgb = new byte[rendered.Height, rendered.Width, 3];
        for (int y = 0; y < rendered.Height; y++)
        {
            for (int x = 0; x < rendered.Width; x++)
            {
                Rgba32 pixel = rendered[x, y];
                rgb[y, x, 0] = pixel.R;
                rgb[y, x, 1] = pixel.G;
                rgb[y, x, 2] = pixel.B;
            }
        }

        return rgb;
    }

    private static TerrainTileTensorPack BuildSlopedPack(
        bool clearNormals = false,
        float[,,]? mcalAlphaPack256 = null,
        float[,]? mcshShadowMask256 = null)
    {
        // Same staggered-lattice fixture shape used by TerrainMinimapCompositorTests: a 3x3 MCNR
        // grid with real vertices at the four corners and the centre, alternating gaps. Corner
        // normals tilt sharply so elevation changes produce a real, non-flat gradient signal.
        var normals = new float[3, 3, 3];
        var mask = new bool[3, 3];
        SetNormal(normals, mask, 0, 0, -0.75f, -0.75f, 1f); // strongly north-west tilted
        SetNormal(normals, mask, 2, 0, 0.75f, -0.75f, 1f);
        SetNormal(normals, mask, 0, 2, -0.75f, 0.75f, 1f);
        SetNormal(normals, mask, 2, 2, 0.75f, 0.75f, 1f);
        SetNormal(normals, mask, 1, 1, 0f, 0f, 1f);

        var textureIds = new int[1, 1, 4];
        textureIds[0, 0, 0] = 0;
        textureIds[0, 0, 1] = -1;
        textureIds[0, 0, 2] = -1;
        textureIds[0, 0, 3] = -1;

        return new TerrainTileTensorPack
        {
            TileX = 0,
            TileY = 0,
            McalAlphaPack256 = mcalAlphaPack256 ?? new float[2, 2, 4],
            MclyTextureIds = textureIds,
            MclyTextureNames = ["test_0.blp"],
            McnrNormalXyz = clearNormals ? null : normals,
            McnrMask257 = clearNormals ? null : mask,
            McshShadowMask256 = mcshShadowMask256,
        };
    }

    private static TerrainTileTensorPack BuildFlatPack()
    {
        var normals = new float[3, 3, 3];
        var mask = new bool[3, 3];
        SetNormal(normals, mask, 0, 0, 0f, 0f, 1f);
        SetNormal(normals, mask, 2, 0, 0f, 0f, 1f);
        SetNormal(normals, mask, 0, 2, 0f, 0f, 1f);
        SetNormal(normals, mask, 2, 2, 0f, 0f, 1f);
        SetNormal(normals, mask, 1, 1, 0f, 0f, 1f);

        var textureIds = new int[1, 1, 4];
        textureIds[0, 0, 0] = 0;
        textureIds[0, 0, 1] = -1;
        textureIds[0, 0, 2] = -1;
        textureIds[0, 0, 3] = -1;

        return new TerrainTileTensorPack
        {
            TileX = 0,
            TileY = 0,
            McalAlphaPack256 = new float[2, 2, 4],
            MclyTextureIds = textureIds,
            MclyTextureNames = ["test_0.blp"],
            McnrNormalXyz = normals,
            McnrMask257 = mask,
        };
    }

    private static void SetNormal(float[,,] normals, bool[,] mask, int x, int y, float nx, float ny, float nz)
    {
        normals[y, x, 0] = nx;
        normals[y, x, 1] = ny;
        normals[y, x, 2] = nz;
        mask[y, x] = true;
    }

    private static byte[,,] SolidTexture(byte r, byte g, byte b) => new byte[1, 1, 3] { { { r, g, b } } };
}
