using WoWViewer.Rendering;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

/// <summary>
/// The 1.60.1 (BUILD 1.60.1.69876) CreatureModelData/CreatureDisplayInfo layouts identify assets by
/// FileDataID instead of naming them. These tests pin the row-shape interpretation that keeps
/// replaceable textures resolving on modern clients; without it every section binds the error texture.
/// </summary>
public sealed class ReplaceableTextureFileDataIdRowTests
{
    private static Func<string, object?> Row(params (string Key, object Value)[] fields)
    {
        Dictionary<string, object?> values = new(StringComparer.Ordinal);
        foreach ((string key, object value) in fields)
            values[key] = value;
        return name => values.TryGetValue(name, out object? value) ? value : null;
    }

    [Fact]
    public void ModernCreatureModelDataRowResolvesModelPathFromFileDataId()
    {
        Func<uint, string?> previous = FileDataIdPaths.Resolver!;
        FileDataIdPaths.Resolver = id => id == 116912 ? "Creature\\Gryphon\\Gryphon.m2" : null;
        try
        {
            Assert.Equal(
                "Creature\\Gryphon\\Gryphon.m2",
                ReplaceableTextureResolver.ReadModelName(Row(("FileDataID", 116912u))));
        }
        finally
        {
            FileDataIdPaths.Resolver = previous;
        }
    }

    [Fact]
    public void LegacyCreatureModelDataRowStillReadsItsModelName()
    {
        Assert.Equal(
            "Creature\\Troll\\Troll.mdx",
            ReplaceableTextureResolver.ReadModelName(Row(("ModelName", "Creature\\Troll\\Troll.mdx"))));
    }

    [Fact]
    public void ModernCreatureDisplayInfoRowResolvesVariationsFromFileDataIds()
    {
        Func<uint, string?> previous = FileDataIdPaths.Resolver!;
        FileDataIdPaths.Resolver = id => id switch
        {
            1001u => "Creature\\Gryphon\\GryphonWhite.blp",
            1002u => "Creature\\Gryphon\\GryphonBrown.blp",
            _ => null,
        };
        try
        {
            string[] variations = ReplaceableTextureResolver.ReadTextureVariations(Row(
                ("TextureVariationFileDataID[0]", 1001u),
                ("TextureVariationFileDataID[1]", 1002u),
                ("TextureVariationFileDataID[2]", 0u),
                ("TextureVariationFileDataID[3]", 0u)));

            Assert.Equal("Creature\\Gryphon\\GryphonWhite.blp", variations[0]);
            Assert.Equal("Creature\\Gryphon\\GryphonBrown.blp", variations[1]);
        }
        finally
        {
            FileDataIdPaths.Resolver = previous;
        }
    }

    [Fact]
    public void LegacyCreatureDisplayInfoRowStillReadsItsVariationStrings()
    {
        string[] variations = ReplaceableTextureResolver.ReadTextureVariations(Row(
            ("TextureVariation", new[] { "TrollWhite", "TrollBrown", string.Empty })));

        Assert.Equal("TrollWhite", variations[0]);
        Assert.Equal("TrollBrown", variations[1]);
    }

    [Fact]
    public void UnresolvedFileDataIdFallsBackToTheVirtualPathRatherThanAnErrorTexture()
    {
        Func<uint, string?> previous = FileDataIdPaths.Resolver!;
        FileDataIdPaths.Resolver = _ => null;
        try
        {
            // fdid:<id> is read directly by FileDataID-aware data sources, so the model still has a
            // real texture request instead of the missing-texture placeholder.
            Assert.Equal(
                "fdid:999999",
                ReplaceableTextureResolver.ReadModelName(Row(("FileDataID", 999999u))));
        }
        finally
        {
            FileDataIdPaths.Resolver = previous;
        }
    }
}
