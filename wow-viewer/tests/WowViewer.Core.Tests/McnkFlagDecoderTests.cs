using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class McnkFlagDecoderTests
{
    [Theory]
    [InlineData(0x00u, AdtLiquidBasicType.Water)]
    [InlineData(0x01u, AdtLiquidBasicType.Water)]
    [InlineData(0x02u, AdtLiquidBasicType.Water)]
    [InlineData(0x04u, AdtLiquidBasicType.Water)]
    [InlineData(0x08u, AdtLiquidBasicType.Ocean)]
    [InlineData(0x0Cu, AdtLiquidBasicType.Ocean)]
    [InlineData(0x10u, AdtLiquidBasicType.Magma)]
    [InlineData(0x14u, AdtLiquidBasicType.Magma)]
    [InlineData(0x18u, AdtLiquidBasicType.Magma)]
    [InlineData(0x1Cu, AdtLiquidBasicType.Magma)]
    [InlineData(0x20u, AdtLiquidBasicType.Slime)]
    [InlineData(0x24u, AdtLiquidBasicType.Slime)]
    [InlineData(0x28u, AdtLiquidBasicType.Slime)]
    [InlineData(0x2Cu, AdtLiquidBasicType.Slime)]
    [InlineData(0x30u, AdtLiquidBasicType.Magma)]
    [InlineData(0x3Cu, AdtLiquidBasicType.Magma)]
    [InlineData(0x40u, AdtLiquidBasicType.Water)]
    [InlineData(0x80u, AdtLiquidBasicType.Water)]
    [InlineData(0xFFu, AdtLiquidBasicType.Magma)]
    public void Decode_ReturnsExpectedBasicType_ForAllMcnkFlagCombinations(uint mcnkFlags, AdtLiquidBasicType expected)
    {
        AdtLiquidBasicType actual = McnkFlagDecoder.Decode(mcnkFlags);

        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(0x00u, (byte)0x00, AdtLiquidBasicType.Water)]
    [InlineData(0x00u, (byte)0x01, AdtLiquidBasicType.Water)]
    [InlineData(0x00u, (byte)0x02, AdtLiquidBasicType.Ocean)]
    [InlineData(0x00u, (byte)0x03, AdtLiquidBasicType.Magma)]
    [InlineData(0x00u, (byte)0x04, AdtLiquidBasicType.Slime)]
    [InlineData(0x00u, (byte)0x0F, AdtLiquidBasicType.Water)]
    [InlineData(0x20u, (byte)0x00, AdtLiquidBasicType.Slime)]
    [InlineData(0x20u, (byte)0x04, AdtLiquidBasicType.Slime)]
    [InlineData(0x10u, (byte)0x00, AdtLiquidBasicType.Magma)]
    [InlineData(0x10u, (byte)0x02, AdtLiquidBasicType.Ocean)]
    [InlineData(0x10u, (byte)0x03, AdtLiquidBasicType.Magma)]
    [InlineData(0x08u, (byte)0x00, AdtLiquidBasicType.Ocean)]
    [InlineData(0x08u, (byte)0x03, AdtLiquidBasicType.Magma)]
    [InlineData(0x30u, (byte)0x00, AdtLiquidBasicType.Magma)]
    [InlineData(0xFFu, (byte)0x0F, AdtLiquidBasicType.Magma)]
    public void DecodeWithMclqTileNibble_NibbleWinsWhenNonZero(uint mcnkFlags, byte nibble, AdtLiquidBasicType expected)
    {
        AdtLiquidBasicType actual = McnkFlagDecoder.DecodeWithMclqTileNibble(mcnkFlags, nibble);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void Decode_ZeroFlags_ReturnsWater()
    {
        Assert.Equal(AdtLiquidBasicType.Water, McnkFlagDecoder.Decode(0u));
    }

    [Fact]
    public void Decode_MagmaBitSet_ReturnsMagma()
    {
        Assert.Equal(AdtLiquidBasicType.Magma, McnkFlagDecoder.Decode(0x10u));
    }

    [Fact]
    public void Decode_SlimeBitSet_ReturnsSlime()
    {
        Assert.Equal(AdtLiquidBasicType.Slime, McnkFlagDecoder.Decode(0x20u));
    }

    [Fact]
    public void Decode_OceanBitSet_ReturnsOcean()
    {
        Assert.Equal(AdtLiquidBasicType.Ocean, McnkFlagDecoder.Decode(0x08u));
    }

    [Fact]
    public void Decode_RiverBitSet_ReturnsWater()
    {
        Assert.Equal(AdtLiquidBasicType.Water, McnkFlagDecoder.Decode(0x04u));
    }

    [Fact]
    public void Decode_MagmaAndSlimeBothSet_MagmaWins()
    {
        Assert.Equal(AdtLiquidBasicType.Magma, McnkFlagDecoder.Decode(0x30u));
    }

    [Fact]
    public void Decode_AllBitsSet_MagmaWins()
    {
        Assert.Equal(AdtLiquidBasicType.Magma, McnkFlagDecoder.Decode(0xFFu));
    }

    [Fact]
    public void Decode_SlimeAndOceanBothSet_SlimeWins()
    {
        Assert.Equal(AdtLiquidBasicType.Slime, McnkFlagDecoder.Decode(0x28u));
    }
}
