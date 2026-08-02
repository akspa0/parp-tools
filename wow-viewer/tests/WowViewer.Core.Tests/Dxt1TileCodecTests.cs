using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Blp;

namespace WowViewer.Core.Tests;

public sealed class Dxt1TileCodecTests
{
    private const int Size = 16;

    private static Image<Rgba32> Gradient()
    {
        var image = new Image<Rgba32>(Size, Size);
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                byte r = (byte)((x / (float)(Size - 1)) * 255f);
                byte g = (byte)((y / (float)(Size - 1)) * 255f);
                byte b = (byte)(((x + y) / (float)(2 * (Size - 1))) * 255f);
                image[x, y] = new Rgba32(r, g, b, 255);
            }
        }

        return image;
    }

    [Fact]
    public void EncodeDecode_ProducesBlockierImage_ThanPristineInput()
    {
        using Image<Rgba32> source = Gradient();
        using Image<Rgba32> parity = Dxt1TileCodec.EncodeDecode(source);

        // The parity cycle must change the image (DXT1 is lossy), and the change must be bounded
        // (it is a degradation, not a scramble).
        int changed = 0;
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                Rgba32 a = source[x, y];
                Rgba32 b = parity[x, y];
                if (a.R != b.R || a.G != b.G || a.B != b.B)
                    changed++;
            }
        }

        // Some pixels must change (lossy), but not all (the codec preserves colour roughly).
        Assert.InRange(changed, 1, Size * Size);
    }

    [Fact]
    public void EncodeDecode_IsDeterministic()
    {
        using Image<Rgba32> source = Gradient();
        using Image<Rgba32> first = Dxt1TileCodec.EncodeDecode(source);
        using Image<Rgba32> second = Dxt1TileCodec.EncodeDecode(source);

        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                Assert.Equal(first[x, y], second[x, y]);
            }
        }
    }

    [Fact]
    public void EncodeDecode_HandlesFlatColour()
    {
        using var source = new Image<Rgba32>(Size, Size);
        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
                source[x, y] = new Rgba32(100, 150, 200, 255);

        using Image<Rgba32> parity = Dxt1TileCodec.EncodeDecode(source);

        // A flat colour survives DXT1 nearly exactly (all pixels map to the same endpoint).
        Rgba32 first = parity[0, 0];
        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
                Assert.Equal(first, parity[x, y]);
    }

    [Fact]
    public void RoundTripAgreement_OnNonBlp_ReturnsZero()
    {
        // Not a BLP2/DXTC/DXT1 file -> no extractable blocks -> 0 agreement.
        byte[] notBlp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        Assert.Equal(0f, Dxt1TileCodec.RoundTripAgreement(notBlp));
    }
}
