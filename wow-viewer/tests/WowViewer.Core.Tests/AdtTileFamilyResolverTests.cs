using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtTileFamilyResolverTests
{
    [Fact]
    public void Resolve_TemporarySplitTile_ReturnsExpectedCompanionPaths()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer-adt-family-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string rootPath = Path.Combine(tempDir, "testmap_3_7.adt");
            string texPath = Path.Combine(tempDir, "testmap_3_7_tex0.adt");
            string objPath = Path.Combine(tempDir, "testmap_3_7_obj0.adt");
            File.WriteAllBytes(rootPath, [1]);
            File.WriteAllBytes(texPath, [2]);
            File.WriteAllBytes(objPath, [3]);

            AdtTileFamily family = AdtTileFamilyResolver.Resolve(objPath);

            Assert.Equal(Path.GetFullPath(objPath), family.SourcePath);
            Assert.True(family.HasRoot);
            Assert.True(family.HasTex0);
            Assert.True(family.HasObj0);
            Assert.False(family.HasLod);
            Assert.Equal(Path.GetFullPath(rootPath), family.RootPath);
            Assert.Equal(Path.GetFullPath(texPath), family.Tex0Path);
            Assert.Equal(Path.GetFullPath(objPath), family.Obj0Path);
            Assert.Equal(MapFileKind.AdtTex, family.TextureSourceKind);
            Assert.Equal(MapFileKind.AdtObj, family.PlacementSourceKind);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Resolve_DevelopmentTile_PrefersSplitTextureAndPlacementSources()
    {
        AdtTileFamily family = AdtTileFamilyResolver.Resolve(MapTestPaths.DevelopmentRootAdtPath);

        Assert.True(family.HasRoot);
        Assert.True(family.HasTex0);
        Assert.True(family.HasObj0);
        Assert.False(family.HasLod);
        Assert.Equal(MapFileKind.AdtTex, family.TextureSourceKind);
        Assert.Equal(MapFileKind.AdtObj, family.PlacementSourceKind);
		Assert.Equal(Path.GetFullPath(MapTestPaths.DevelopmentTexAdtPath), family.Tex0Path);
		Assert.Equal(Path.GetFullPath(MapTestPaths.DevelopmentObjAdtPath), family.Obj0Path);
    }
}