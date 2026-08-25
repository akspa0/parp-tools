using WowViewer.Core.Editor.Eras;

namespace WowViewer.Core.Editor.Tests.Eras;

public class EraHandlerResolverTests
{
    [Fact]
    public void Resolve_picks_most_specific_matching_handler()
    {
        var defaultHandler = new EraHandler<string>(EditorBuildEraRange.All, "default", -1);
        var wide = new EraHandler<string>(EditorBuildEraRange.AtLeast(EditorBuildVersion.Parse("1.0")), "1.x+", 0);
        var narrow = new EraHandler<string>(
            EditorBuildEraRange.Between(EditorBuildVersion.Parse("1.0"), EditorBuildVersion.Parse("2.0")),
            "1.x",
            1);

        var result = EraHandlerResolver.Resolve(
            EditorBuildVersion.Parse("1.12.1.5875"),
            [wide, narrow],
            defaultHandler);

        Assert.NotNull(result);
        Assert.Equal("1.x", result.Value.Handler);
    }

    [Fact]
    public void Resolve_falls_back_when_no_specific_match()
    {
        var defaultHandler = new EraHandler<string>(EditorBuildEraRange.All, "default", -1);
        var onlyLate = new EraHandler<string>(EditorBuildEraRange.AtLeast(EditorBuildVersion.Parse("4.0")), "4.x+", 0);

        var result = EraHandlerResolver.Resolve(
            EditorBuildVersion.Parse("3.3.5.12340"),
            [onlyLate],
            defaultHandler);

        Assert.NotNull(result);
        Assert.Equal("default", result.Value.Handler);
    }

    [Fact]
    public void Resolve_is_deterministic_on_equal_specificity()
    {
        var first = new EraHandler<string>(EditorBuildEraRange.AtLeast(EditorBuildVersion.Parse("1.0")), "first", 0);
        var second = new EraHandler<string>(EditorBuildEraRange.AtLeast(EditorBuildVersion.Parse("1.0")), "second", 1);

        var result = EraHandlerResolver.Resolve(EditorBuildVersion.Parse("1.5"), [first, second]);

        Assert.NotNull(result);
        Assert.Equal("first", result.Value.Handler);
    }

    [Fact]
    public void Resolve_returns_null_when_nothing_matches()
    {
        var onlyLate = new EraHandler<string>(EditorBuildEraRange.AtLeast(EditorBuildVersion.Parse("4.0")), "4.x+", 0);

        var result = EraHandlerResolver.Resolve(EditorBuildVersion.Parse("1.0"), [onlyLate]);

        Assert.Null(result);
    }
}