using WowViewer.Core.Editor.Eras;

namespace WowViewer.Core.Editor.Tests.Eras;

public class EditorBuildEraRangeTests
{
    [Fact]
    public void All_contains_everything()
    {
        var range = EditorBuildEraRange.All;

        Assert.True(range.Contains(EditorBuildVersion.Parse("0.5.3.3368")));
        Assert.True(range.Contains(EditorBuildVersion.Parse("99.0.0")));
        Assert.Equal(0, range.Specificity);
    }

    [Fact]
    public void Between_is_inclusive_exclusive()
    {
        var range = EditorBuildEraRange.Between(
            EditorBuildVersion.Parse("1.0"),
            EditorBuildVersion.Parse("3.0"),
            "1.x-2.x");

        Assert.True(range.Contains(EditorBuildVersion.Parse("1.0")));
        Assert.True(range.Contains(EditorBuildVersion.Parse("2.9.9")));
        Assert.False(range.Contains(EditorBuildVersion.Parse("3.0")));
        Assert.Equal(2, range.Specificity);
        Assert.Equal("1.x-2.x", range.ToString());
    }

    [Fact]
    public void Between_rejects_inverted_bounds()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            EditorBuildEraRange.Between(EditorBuildVersion.Parse("3.0"), EditorBuildVersion.Parse("1.0")));
    }

    [Fact]
    public void Half_open_ranges_have_one_endpoint()
    {
        var atLeast = EditorBuildEraRange.AtLeast(EditorBuildVersion.Parse("2.0"));
        var upTo = EditorBuildEraRange.UpTo(EditorBuildVersion.Parse("2.0"));

        Assert.Equal(1, atLeast.Specificity);
        Assert.Equal(1, upTo.Specificity);
        Assert.True(atLeast.Contains(EditorBuildVersion.Parse("2.1")));
        Assert.False(upTo.Contains(EditorBuildVersion.Parse("2.1")));
    }
}