using System.Numerics;
using WowViewer.Core.PM4.Models;
using Xunit;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4MsurCk24ByteDecompositionTests
{
    // A PackedParams value chosen so that the 24-bit CK24 is
    // 0xAABBCC. 0xAABBCC_00 in the 32-bit word means the low byte of
    // the 32-bit word is the always-zero padding trailer.
    private const uint PackedParams_AABBCC00 = 0xAABBCC00u;

    [Fact]
    public void Ck24_Returns_TwentyFourBitValue_StrippingLowPaddingByte()
    {
        Pm4MsurEntry entry = MakeEntry(PackedParams_AABBCC00);

        Assert.Equal(0x00AABBCCu, entry.Ck24);
    }

    [Fact]
    public void Ck24Type_Returns_TopByte_Of_PackedParams()
    {
        Pm4MsurEntry entry = MakeEntry(PackedParams_AABBCC00);

        Assert.Equal(0xAA, entry.Ck24Type);
    }

    [Fact]
    public void Ck24HighByte_Returns_MiddleByte_Of_Ck24()
    {
        Pm4MsurEntry entry = MakeEntry(PackedParams_AABBCC00);

        Assert.Equal(0xBB, entry.Ck24HighByte);
    }

    [Fact]
    public void Ck24LowByte_Returns_LowByte_Of_Ck24()
    {
        Pm4MsurEntry entry = MakeEntry(PackedParams_AABBCC00);

        Assert.Equal(0xCC, entry.Ck24LowByte);
    }

    [Fact]
    public void Ck24ObjectId_Is_LossyFlattening_Of_HighAndLowBytes()
    {
        Pm4MsurEntry entry = MakeEntry(PackedParams_AABBCC00);

        Assert.Equal(0xBBCC, entry.Ck24ObjectId);
    }

    [Fact]
    public void HighAndLowBytes_Are_Independent_Of_Ck24ObjectId_Flattening()
    {
        // The bond hypothesis is that 0xBB and 0xCC carry information
        // independently. After flattening into Ck24ObjectId, that
        // information is lost. Verify that the byte-level fields
        // preserve what the 16-bit flattening discards: the pair
        // (0xBB, 0xCC) is recoverable from Ck24HighByte and
        // Ck24LowByte, but not from Ck24ObjectId alone.
        Pm4MsurEntry entry = MakeEntry(PackedParams_AABBCC00);

        byte high = entry.Ck24HighByte;
        byte low = entry.Ck24LowByte;

        Assert.Equal(0xBB, high);
        Assert.Equal(0xCC, low);

        // Distinct test for the same byte pair with a different
        // (high, low) value to make sure the getters aren't
        // accidentally constant.
        Pm4MsurEntry other = MakeEntry(0x42DEAD00u);
        Assert.Equal(0xDE, other.Ck24HighByte);
        Assert.Equal(0xAD, other.Ck24LowByte);
    }

    [Fact]
    public void Ck24Type_Is_Independent_Of_HighAndLowBytes()
    {
        // 0x42 is the WMO-interior type bucket (per prior CK24
        // research notes in pm4-region-aware-object-grouping). The
        // bond hypothesis concerns 0xBB / 0xCC, not 0xAA. Verify
        // that the type byte is independent: a single type with
        // many distinct (high, low) pairs would be the smoking gun.
        Pm4MsurEntry entry = MakeEntry(0x42DEAD00u);

        Assert.Equal(0x42, entry.Ck24Type);
        Assert.Equal(0xDE, entry.Ck24HighByte);
        Assert.Equal(0xAD, entry.Ck24LowByte);
    }

    private static Pm4MsurEntry MakeEntry(uint packedParams)
    {
        return new Pm4MsurEntry(
            GroupKey: 0x03,
            IndexCount: 4,
            AttributeMask: 0x12,
            Padding: 0,
            Normal: new Vector3(0f, 1f, 0f),
            Height: 100f,
            MsviFirstIndex: 0,
            _0x18: 0,
            PackedParams: packedParams);
    }
}
