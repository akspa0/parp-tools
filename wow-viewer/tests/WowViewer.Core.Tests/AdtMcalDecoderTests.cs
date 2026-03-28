using System.Buffers.Binary;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtMcalDecoderTests
{
    [Fact]
    public void DecodeLayer_CompressedPayload_ReturnsExpandedAlpha()
    {
        byte[] mcal =
        [
            0x83, 0x11,
            0x02, 0x22, 0x33,
            0xFF, 0x44,
        ];

        AdtMcalDecodedLayer? decoded = AdtMcalDecoder.DecodeLayer(
            mcal,
            new AdtTextureLayerDescriptor(1, 7, 0x200, 0, 0),
            nextLayer: null,
            useBigAlpha: false,
            doNotFixAlphaMap: false,
            AdtMcalDecodeProfile.LichKingStrict);

        Assert.NotNull(decoded);
        Assert.Equal(AdtMcalAlphaEncoding.Compressed, decoded.Encoding);
        Assert.Equal(7, decoded.SourceBytesConsumed);
        Assert.Equal(0x11, decoded.AlphaMap[0]);
        Assert.Equal(0x11, decoded.AlphaMap[2]);
        Assert.Equal(0x22, decoded.AlphaMap[3]);
        Assert.Equal(0x33, decoded.AlphaMap[4]);
        Assert.Equal(0x44, decoded.AlphaMap[5]);
    }

    [Fact]
    public void DecodeLayer_PackedPayload_AppliesLegacyEdgeFix()
    {
        byte[] mcal = Enumerable.Repeat((byte)0x21, 2048).ToArray();

        AdtMcalDecodedLayer? decoded = AdtMcalDecoder.DecodeLayer(
            mcal,
            new AdtTextureLayerDescriptor(1, 1, 0x100, 0, 0),
            nextLayer: null,
            useBigAlpha: false,
            doNotFixAlphaMap: false,
            AdtMcalDecodeProfile.LichKingStrict);

        Assert.NotNull(decoded);
        Assert.Equal(AdtMcalAlphaEncoding.Packed4Bit, decoded.Encoding);
        Assert.True(decoded.AppliedFixup);
        Assert.Equal(2048, decoded.SourceBytesConsumed);
        Assert.Equal(decoded.AlphaMap[62], decoded.AlphaMap[63]);
        Assert.Equal(decoded.AlphaMap[(62 * 64) + 5], decoded.AlphaMap[(63 * 64) + 5]);
    }

    [Fact]
    public void DecodeLayer_CataclysmFixedBigPayload_ExpandsToFullGrid()
    {
        byte[] mcal = Enumerable.Repeat((byte)0x7F, 63 * 63).ToArray();

        AdtMcalDecodedLayer? decoded = AdtMcalDecoder.DecodeLayer(
            mcal,
            new AdtTextureLayerDescriptor(1, 2, 0x100, 0, 0),
            nextLayer: null,
            useBigAlpha: true,
            doNotFixAlphaMap: false,
            AdtMcalDecodeProfile.Cataclysm400);

        Assert.NotNull(decoded);
        Assert.Equal(AdtMcalAlphaEncoding.BigAlphaFixed, decoded.Encoding);
        Assert.True(decoded.AppliedFixup);
        Assert.Equal(63 * 63, decoded.SourceBytesConsumed);
        Assert.Equal(0x7F, decoded.AlphaMap[0]);
        Assert.Equal(0x7F, decoded.AlphaMap[63]);
        Assert.Equal(0x7F, decoded.AlphaMap[(63 * 64) + 63]);
    }
}