using WowViewer.Core.Files;

namespace WowViewer.Core.Tests;

public sealed class AssetPathTaxonomyTests
{
    [Theory]
    [InlineData("World/Generic/PassiveDoodads/Fruits/Fruit_Apple.m2", "World\\Generic\\PassiveDoodads\\Fruits\\Fruit_Apple.m2")]
    [InlineData("\\World\\wmo\\Azeroth\\Buildings\\Stormwind\\Stormwind.wmo.MPQ\\", "World\\wmo\\Azeroth\\Buildings\\Stormwind\\Stormwind.wmo.MPQ")]
    public void Normalize_StandardizesSeparatorsAndOuterSlashes(string rawPath, string expected)
    {
        string normalized = AssetPathTaxonomy.Normalize(rawPath);

        Assert.Equal(expected, normalized);
    }

    [Fact]
    public void Describe_ExtractsHierarchyAndCategoryForAppleAsset()
    {
        AssetPathDescriptor descriptor = AssetPathTaxonomy.Describe("WORLD\\GENERIC\\PASSIVEDOODADS\\FRUITS\\FRUIT_APPLE.M2");

        Assert.Equal("M2", descriptor.AssetKind);
        Assert.Equal("other", descriptor.ObjectType);
        Assert.Equal("WORLD", descriptor.RootSegment);
        Assert.Equal("FRUITS", descriptor.LeafCategory);
        Assert.Equal("WORLD\\GENERIC\\PASSIVEDOODADS\\FRUITS", descriptor.CategoryKey);
        Assert.Equal("WORLD > GENERIC > PASSIVEDOODADS > FRUITS", descriptor.HierarchyLabel);
    }

    [Fact]
    public void Describe_RecognizesLegacy053WmoArchivePath()
    {
        AssetPathDescriptor descriptor = AssetPathTaxonomy.Describe("World\\wmo\\Azeroth\\Buildings\\Stormwind\\Stormwind.wmo.MPQ");

        Assert.Equal("WMO", descriptor.AssetKind);
        Assert.Equal("building", descriptor.ObjectType);
        Assert.Equal("World\\wmo\\Azeroth\\Buildings\\Stormwind", descriptor.CategoryKey);
        Assert.Equal("Stormwind.wmo.MPQ", descriptor.FileName);
    }

    [Theory]
    [InlineData("World\\Trees\\Oak01.m2", "tree")]
    [InlineData("World\\Rocks\\Cliff_A.m2", "rock")]
    [InlineData("World\\Buildings\\Inn\\HumanInn.wmo", "building")]
    [InlineData("World\\Generic\\Fence\\Fence01.m2", "structure")]
    [InlineData("World\\Generic\\Flowers\\Flower01.m2", "detail")]
    public void ClassifyObjectType_UsesSharedKeywordBuckets(string assetPath, string expected)
    {
        string objectType = AssetPathTaxonomy.ClassifyObjectType(assetPath);

        Assert.Equal(expected, objectType);
    }
}