using WowViewer.Core.IO.Blp;

namespace WowViewer.Core.Tests;

public class BlpRgbReaderTests
{
    [Fact]
    public void ReadRgb_CorruptEmptyBytes_ReturnsLoadError()
    {
        var result = BlpRgbReader.ReadRgb(Array.Empty<byte>(), "Test/Corrupt.blp");
        Assert.Equal(1, result.LoadError);
        Assert.Equal(0, result.Width);
        Assert.Equal(0, result.Height);
        Assert.Null(result.Rgb);
    }

    [Fact]
    public void ReadRgb_GarbageBytes_ReturnsLoadError()
    {
        var garbage = new byte[] { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 };
        var result = BlpRgbReader.ReadRgb(garbage, "Test/Garbage.blp");
        Assert.Equal(1, result.LoadError);
        Assert.Equal(0, result.Width);
        Assert.Equal(0, result.Height);
        Assert.Null(result.Rgb);
    }

    [Fact]
    public void ReadRgb_TooSmallBytes_ReturnsLoadError()
    {
        // Only 3 bytes — not even the BLP signature
        var tiny = new byte[] { 0x42, 0x4C, 0x50 }; // partial "BLP"
        var result = BlpRgbReader.ReadRgb(tiny, "Test/Tiny.blp");
        Assert.Equal(1, result.LoadError);
    }

    [Fact]
    public void ReadRgb_NullSource_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => BlpRgbReader.ReadRgb(null!, "Test/Null.blp"));
    }

    [Fact]
    public void ReadRgb_EmptyPath_Throws()
    {
        Assert.Throws<ArgumentException>(() => BlpRgbReader.ReadRgb(new byte[] { 0x42 }, ""));
    }

    /// <summary>
    /// Real-data test: decode a known BLP from a staged client.
    /// This test requires a staged client at the known path.
    /// It is a manual/optional test — skip via [Fact(Skip = "requires staged client")].
    /// </summary>
    [Fact(Skip = "requires staged client with BLP files")]
    public void ReadRgb_RealBlp_ReturnsCorrectDimensions()
    {
        // Replace path with actual staged BLP from output/tmp/wowarchive-clients/
        string blpPath = Path.Combine(
            TestContext.WorkingDirectory,
            "..", "..", "..", "..", "..",
            "output", "tmp", "wowarchive-clients", "3_3_5_12340",
            "World of Warcraft", "Data", "patch-2.MPQ",
            "Tileset", "Generic", "Black.blp");

        // BLP is inside MPQ — this test is a placeholder for manual validation.
        // To validate: extract a BLP from the staged client, read it, assert result.LoadError == 0.
        Assert.True(false, "Manual validation required. See comment above.");
    }
}

// Stub to satisfy TestContext.WorkingDirectory reference.
// The real TestContext is in xunit.runner.visualstudio.
internal static class TestContext
{
    public static string WorkingDirectory => Directory.GetCurrentDirectory();
}