using MdxViewer.Terrain;
using Xunit;

namespace MdxViewer.Tests.Terrain;

public sealed class MccvColorDecoderTests
{
    [Fact]
    public void DecodeModulation_ReturnsNeutralWhite_WhenDataIsMissing()
    {
        var decoded = MccvColorDecoder.DecodeModulation(null, 0);

        Assert.Equal(1.0f, decoded.X);
        Assert.Equal(1.0f, decoded.Y);
        Assert.Equal(1.0f, decoded.Z);
        Assert.Equal(1.0f, decoded.W);
    }

    [Fact]
    public void DecodeModulation_SwizzlesBgra_AndIgnoresAlpha()
    {
        byte[] raw =
        {
            0x10, // blue
            0x20, // green
            0x30, // red
            0xFF, // alpha, ignored
        };

        var decoded = MccvColorDecoder.DecodeModulation(raw, 0);

        Assert.Equal(0x30 * (2.0f / 255.0f), decoded.X, 4);
        Assert.Equal(0x20 * (2.0f / 255.0f), decoded.Y, 4);
        Assert.Equal(0x10 * (2.0f / 255.0f), decoded.Z, 4);
        Assert.Equal(1.0f, decoded.W);
    }

    [Fact]
    public void DecodeModulation_MapsNeutral7F_CloseToUnity()
    {
        byte[] raw =
        {
            0x7F,
            0x7F,
            0x7F,
            0x00,
        };

        var decoded = MccvColorDecoder.DecodeModulation(raw, 0);

        Assert.InRange(decoded.X, 0.99f, 1.01f);
        Assert.InRange(decoded.Y, 0.99f, 1.01f);
        Assert.InRange(decoded.Z, 0.99f, 1.01f);
    }
}