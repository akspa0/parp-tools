using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class TerrainTextureFallbackPolicyTests
{
    [Fact]
    public void GetSpecularCompanionPath_ReturnsSameStemBlpCompanion()
    {
        string? fallback = TerrainTextureFallbackPolicy.GetSpecularCompanionPath(
            "Tileset\\Durotar\\DurotarIGrass.blp");

        Assert.Equal("Tileset\\Durotar\\DurotarIGrass_s.blp", fallback);
    }

    [Theory]
    [InlineData("Tileset\\Durotar\\DurotarIGrass_s.blp")]
    [InlineData("Tileset\\Durotar\\DurotarIGrass.png")]
    [InlineData("")]
    public void GetSpecularCompanionPath_RejectsUnsafeOrInapplicablePaths(string path)
    {
        Assert.Null(TerrainTextureFallbackPolicy.GetSpecularCompanionPath(path));
    }

    [Fact]
    public void GetRgbProxyCandidates_PrefersSameStemCompanionThenMovedExactNameBeforeLocalRelatedDiffuse()
    {
        IReadOnlyList<TerrainTextureFallbackCandidate> candidates = TerrainTextureFallbackPolicy.GetRgbProxyCandidates(
            "Tileset\\Durotar\\DurotarIGrass.blp",
            [
                "Tileset\\Durotar\\DurotarIGrass_s.blp",
                "World\\Textures\\DurotarIGrass.blp",
                "Tileset\\Durotar\\DurotarRock.blp",
                "Tileset\\Durotar\\DurotarDryGrass.blp",
                "Tileset\\Durotar\\DurotarGrass.blp",
                "Tileset\\Durotar\\DurotarGrass_s.blp",
                "Tileset\\Mulgore\\DurotarGrass.blp",
            ]);

        Assert.Collection(
            candidates,
            candidate =>
            {
                Assert.Equal("Tileset\\Durotar\\DurotarIGrass_s.blp", candidate.ResolvedPath);
                Assert.Equal(TerrainTextureFallbackPolicy.SpecularCompanionRgbProxy, candidate.ResolutionKind);
            },
            candidate =>
            {
                Assert.Equal("World\\Textures\\DurotarIGrass.blp", candidate.ResolvedPath);
                Assert.Equal(TerrainTextureFallbackPolicy.RelatedDiffuseRgbProxy, candidate.ResolutionKind);
            },
            candidate =>
            {
                Assert.Equal("Tileset\\Durotar\\DurotarGrass.blp", candidate.ResolvedPath);
                Assert.Equal(TerrainTextureFallbackPolicy.RelatedDiffuseRgbProxy, candidate.ResolutionKind);
            },
            candidate => Assert.Equal("Tileset\\Durotar\\DurotarDryGrass.blp", candidate.ResolvedPath),
            candidate => Assert.Equal("Tileset\\Mulgore\\DurotarGrass.blp", candidate.ResolvedPath));
    }

    [Fact]
    public void GetRgbProxyCandidates_RejectsUnrelatedNamesAndMaterialOnlyCandidates()
    {
        IReadOnlyList<TerrainTextureFallbackCandidate> candidates = TerrainTextureFallbackPolicy.GetRgbProxyCandidates(
            "Tileset\\Durotar\\DurotarIGrass.blp",
            [
                "Tileset\\Durotar\\DurotarRock.blp",
                "Tileset\\Durotar\\DurotarRock_s.blp",
                "Tileset\\Mulgore\\MulgoreRock.blp",
            ]);

        TerrainTextureFallbackCandidate candidate = Assert.Single(candidates);
        Assert.Equal("Tileset\\Durotar\\DurotarIGrass_s.blp", candidate.ResolvedPath);
        Assert.Equal(TerrainTextureFallbackPolicy.SpecularCompanionRgbProxy, candidate.ResolutionKind);
    }

    [Fact]
    public void GetCatalogRgbLastResortCandidates_PrefersTheSameDirectoryBeforeTerrainFamilyOrGenericBlps()
    {
        IReadOnlyList<TerrainTextureFallbackCandidate> candidates = TerrainTextureFallbackPolicy.GetCatalogRgbLastResortCandidates(
            "Tileset\\Durotar\\DurotarMissing.blp",
            [
                "World\\Generic\\Gray.blp",
                "Tileset\\Barrens\\BarrensDirt.blp",
                "Tileset\\Durotar\\DurotarRock.blp",
                "Tileset\\Durotar\\DurotarRock_s.blp",
            ]);

        Assert.Collection(
            candidates,
            candidate =>
            {
                Assert.Equal("Tileset\\Durotar\\DurotarRock.blp", candidate.ResolvedPath);
                Assert.Equal(TerrainTextureFallbackPolicy.CatalogRgbLastResortProxy, candidate.ResolutionKind);
            },
            candidate => Assert.Equal("Tileset\\Barrens\\BarrensDirt.blp", candidate.ResolvedPath),
            candidate => Assert.Equal("World\\Generic\\Gray.blp", candidate.ResolvedPath));
    }
}
