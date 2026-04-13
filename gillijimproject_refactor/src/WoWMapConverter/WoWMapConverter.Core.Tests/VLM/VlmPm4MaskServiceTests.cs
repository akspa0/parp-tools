using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WoWMapConverter.Core.Formats.PM4;
using WoWMapConverter.Core.VLM;
using Xunit;

namespace WoWMapConverter.Core.Tests.VLM;

public sealed class VlmPm4MaskServiceTests
{
    [Fact]
    public void BuildPm4Mask_InTilePositionRefs_ReturnsMaskCoverage()
    {
        MprlEntry[] refs =
        [
            new MprlEntry { PositionX = 12f, PositionY = 0f, PositionZ = 18f },
            new MprlEntry { PositionX = 24f, PositionY = 0f, PositionZ = 30f }
        ];

        byte[] maskBytes = VlmPm4MaskService.BuildPm4Mask("development_0_0", refs, 256, 256);

        Assert.NotEmpty(maskBytes);

        using Image<L8> mask = Image.Load<L8>(maskBytes);
        Assert.True(CountCoveredPixels(mask) > 0);
    }

    [Fact]
    public void BuildPm4Mask_OutOfTilePositionRefs_ReturnsEmpty()
    {
        MprlEntry[] refs =
        [
            new MprlEntry { PositionX = 640f, PositionY = 0f, PositionZ = 640f }
        ];

        byte[] maskBytes = VlmPm4MaskService.BuildPm4Mask("development_0_0", refs, 256, 256);

        Assert.Empty(maskBytes);
    }

    private static int CountCoveredPixels(Image<L8> image)
    {
        int count = 0;
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<L8> row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    if (row[x].PackedValue > 0)
                        count++;
                }
            }
        });
        return count;
    }
}