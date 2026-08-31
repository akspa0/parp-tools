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
    public void Resolve_BandOneCompanion_NormalizesToSameFamilyAndSelectsBandOne()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer-adt-family-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string rootPath = Path.Combine(tempDir, "testmap_3_7.adt");
            string tex1Path = Path.Combine(tempDir, "testmap_3_7_tex1.adt");
            string obj1Path = Path.Combine(tempDir, "testmap_3_7_obj1.adt");
            File.WriteAllBytes(rootPath, [1]);
            File.WriteAllBytes(tex1Path, [2]);
            File.WriteAllBytes(obj1Path, [3]);

            AdtTileFamily family = AdtTileFamilyResolver.Resolve(tex1Path);

            Assert.Equal(Path.GetFullPath(rootPath), family.RootPath);
            Assert.Equal(Path.GetFullPath(tex1Path), family.Tex1Path);
            Assert.Equal(Path.GetFullPath(obj1Path), family.Obj1Path);
            Assert.True(family.HasRoot);
            Assert.True(family.HasTex1);
            Assert.True(family.HasObj1);
            Assert.False(family.HasTex0);
            Assert.False(family.HasObj0);
            Assert.Equal(AdtLodBand.Band1, family.SelectCompanionBand());
            Assert.Equal(Path.GetFullPath(tex1Path), family.GetTextureSourcePath(AdtLodBand.Band1));
            Assert.Equal(MapFileKind.AdtTex1, family.GetTextureSourceKind(AdtLodBand.Band1));
            Assert.Equal(Path.GetFullPath(obj1Path), family.GetPlacementSourcePath(AdtLodBand.Band1));
            Assert.Equal(MapFileKind.AdtObj1, family.GetPlacementSourceKind(AdtLodBand.Band1));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Resolve_BothBands_PrefersCompleteRequestedBandAndFallsBackToCompleteAlternate()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer-adt-family-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string rootPath = Path.Combine(tempDir, "testmap_3_7.adt");
            string tex0Path = Path.Combine(tempDir, "testmap_3_7_tex0.adt");
            string obj0Path = Path.Combine(tempDir, "testmap_3_7_obj0.adt");
            string tex1Path = Path.Combine(tempDir, "testmap_3_7_tex1.adt");
            File.WriteAllBytes(rootPath, [1]);
            File.WriteAllBytes(tex0Path, [2]);
            File.WriteAllBytes(obj0Path, [3]);
            File.WriteAllBytes(tex1Path, [4]);

            AdtTileFamily family = AdtTileFamilyResolver.Resolve(rootPath);

            Assert.Equal(AdtLodBand.Band0, family.SelectCompanionBand());
            Assert.Equal(AdtLodBand.Band0, family.SelectCompanionBand(AdtLodBand.Band1));
            Assert.Equal(Path.GetFullPath(tex0Path), family.TextureSourcePath);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

}
