using WowViewer.Core.Editor.Eras;

namespace WowViewer.Core.Editor.Tests.Eras;

public class EditorBuildVersionTests
{
    [Theory]
    [InlineData("0.5.3.3368", "0.5.3.3368", 0)]
    [InlineData("1.12.1.5875", "2.4.3.8606", -1)]
    [InlineData("3.3.5.12340", "3.3.5.11723", 1)]
    [InlineData("1.12", "1.11.2", 1)]
    [InlineData("1.12", "1.12.1", -1)]
    public void Compare_orders_numeric_segments(string left, string right, int expectedSign)
    {
        var a = EditorBuildVersion.Parse(left);
        var b = EditorBuildVersion.Parse(right);

        int actualSign = a.CompareTo(b);
        Assert.Equal(Math.Sign(expectedSign), Math.Sign(actualSign));
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("not-a-version")]
    public void TryParse_rejects_non_numeric(string value)
    {
        Assert.False(EditorBuildVersion.TryParse(value, out _));
    }

    [Fact]
    public void Parse_round_trips_original_string()
    {
        var version = EditorBuildVersion.Parse("10.2.7.54510");

        Assert.Equal("10.2.7.54510", version.Original);
        Assert.Equal("10.2.7.54510", version.ToString());
    }

    [Fact]
    public void Equality_uses_segment_comparison_not_string()
    {
        var a = EditorBuildVersion.Parse("1.2.3");
        var b = EditorBuildVersion.Parse("1.2.3");

        Assert.True(a == b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }
}