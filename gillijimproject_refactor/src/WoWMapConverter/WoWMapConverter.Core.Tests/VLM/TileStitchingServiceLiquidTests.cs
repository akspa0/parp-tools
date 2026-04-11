using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WoWMapConverter.Core.VLM;
using Xunit;

namespace WoWMapConverter.Core.Tests.VLM;

public sealed class TileStitchingServiceLiquidTests
{
    [Fact]
    public void StitchLiquidMask_PartialMh2oLayer_OnlyPaintsVisibleSubRect()
    {
        VlmLiquidData liquid = new(
            ChunkIndex: 0,
            LiquidType: 0,
            MinHeight: 10f,
            MaxHeight: 12f,
            MaskPath: null,
            Heights: [10f, 10f, 10f, 11f, 11f, 11f, 12f, 12f, 12f],
            XOffset: 3,
            YOffset: 4,
            Width: 2,
            Height: 2,
            ExistsBitmapBase64: Convert.ToBase64String([0b_0000_0111]));

        byte[] png = TileStitchingService.StitchLiquidMask([liquid], "test_0_0");

        using Image<L8> image = Image.Load<L8>(png);

        Assert.Equal((byte)255, image[3 * 8, 4 * 8].PackedValue);
        Assert.Equal((byte)255, image[4 * 8, 4 * 8].PackedValue);
        Assert.Equal((byte)255, image[3 * 8, 5 * 8].PackedValue);
        Assert.Equal((byte)0, image[4 * 8, 5 * 8].PackedValue);
        Assert.Equal((byte)0, image[0, 0].PackedValue);
    }
}