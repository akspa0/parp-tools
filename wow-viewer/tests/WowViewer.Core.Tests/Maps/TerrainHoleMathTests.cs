using System.Buffers.Binary;
using WowViewer.Core.Maps;
using WowViewer.Core.Runtime.World.Terrain;
using Xunit;

namespace WowViewer.Core.Tests.Maps;

public sealed class TerrainHoleMathTests
{
    [Fact]
    public void IsCellHoled16_CorrectlyGroups2x2Cells()
    {
        // Low-res bit 0 is (hx=0, hy=0), covering cells (0,0), (1,0), (0,1), (1,1)
        ushort mask = 0x0001;

        Assert.True(TerrainHoleMath.IsCellHoled16(mask, 0, 0));
        Assert.True(TerrainHoleMath.IsCellHoled16(mask, 1, 0));
        Assert.True(TerrainHoleMath.IsCellHoled16(mask, 0, 1));
        Assert.True(TerrainHoleMath.IsCellHoled16(mask, 1, 1));

        Assert.False(TerrainHoleMath.IsCellHoled16(mask, 2, 0));
        Assert.False(TerrainHoleMath.IsCellHoled16(mask, 0, 2));
    }

    [Fact]
    public void IsCellHoled64_TestsIndividualCellPrecision()
    {
        // In 64-bit mask: cell at (cx=3, cy=2) is bit 2*8 + 3 = 19
        ulong mask = 1UL << (2 * 8 + 3);

        Assert.True(TerrainHoleMath.IsCellHoled64(mask, 3, 2));
        Assert.False(TerrainHoleMath.IsCellHoled64(mask, 2, 2));
        Assert.False(TerrainHoleMath.IsCellHoled64(mask, 3, 3));
        Assert.False(TerrainHoleMath.IsCellHoled64(mask, 0, 0));
    }

    [Fact]
    public void UpsampleLowResToHighRes_ExpandsEachLowResBitToFourHighResBits()
    {
        ushort lowRes = 0x0005; // Bits 0 and 2 (hx=0,hy=0 and hx=2,hy=0)
        ulong highRes = TerrainHoleMath.UpsampleLowResToHighRes(lowRes);

        // Group (0,0) -> cells (0,0),(1,0),(0,1),(1,1)
        Assert.True(TerrainHoleMath.IsCellHoled64(highRes, 0, 0));
        Assert.True(TerrainHoleMath.IsCellHoled64(highRes, 1, 0));
        Assert.True(TerrainHoleMath.IsCellHoled64(highRes, 0, 1));
        Assert.True(TerrainHoleMath.IsCellHoled64(highRes, 1, 1));

        // Group (1,0) should NOT be holed
        Assert.False(TerrainHoleMath.IsCellHoled64(highRes, 2, 0));
        Assert.False(TerrainHoleMath.IsCellHoled64(highRes, 3, 0));

        // Group (2,0) -> cells (4,0),(5,0),(4,1),(5,1)
        Assert.True(TerrainHoleMath.IsCellHoled64(highRes, 4, 0));
        Assert.True(TerrainHoleMath.IsCellHoled64(highRes, 5, 0));
        Assert.True(TerrainHoleMath.IsCellHoled64(highRes, 4, 1));
        Assert.True(TerrainHoleMath.IsCellHoled64(highRes, 5, 1));
    }

    [Fact]
    public void DownsampleHighResToLowRes_MarksGroupHoledIfAnySubcellHoled()
    {
        // Single cell holed at (cx=5, cy=3) -> group (hx=2, hy=1)
        ulong highRes = 1UL << (3 * 8 + 5);
        ushort lowRes = TerrainHoleMath.DownsampleHighResToLowRes(highRes);

        ushort expectedBit = (ushort)(1 << (1 * 4 + 2));
        Assert.Equal(expectedBit, lowRes);
    }

    [Fact]
    public void ReadHoleMasks_Legacy_ReadsFromOffset0x3C()
    {
        byte[] payload = new byte[128];
        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(0x3C, 2), 0x000F);

        var (mask64, mask16) = TerrainHoleMath.ReadHoleMasks(payload, mcnkFlags: 0);

        Assert.Equal((ushort)0x000F, mask16);
        Assert.True(TerrainHoleMath.IsCellHoled64(mask64, 0, 0));
        Assert.True(TerrainHoleMath.IsCellHoled64(mask64, 7, 1));
        Assert.False(TerrainHoleMath.IsCellHoled64(mask64, 0, 2));
    }

    [Fact]
    public void ReadHoleMasks_ModernHighRes_ReadsFromOffset0x14()
    {
        byte[] payload = new byte[128];
        // HighResHoles bit is 0x10000
        const uint highResFlag = 0x10000u;
        ulong expected64 = (1UL << 0) | (1UL << 19) | (1UL << 63);
        BinaryPrimitives.WriteUInt64LittleEndian(payload.AsSpan(0x14, 8), expected64);

        var (mask64, mask16) = TerrainHoleMath.ReadHoleMasks(payload, mcnkFlags: highResFlag);

        Assert.Equal(expected64, mask64);
        Assert.True(TerrainHoleMath.IsCellHoled64(mask64, 0, 0));
        Assert.True(TerrainHoleMath.IsCellHoled64(mask64, 3, 2));
        Assert.True(TerrainHoleMath.IsCellHoled64(mask64, 7, 7));
        Assert.False(TerrainHoleMath.IsCellHoled64(mask64, 1, 0));

        // And mask16 downsamples correctly
        Assert.True(TerrainHoleMath.IsCellHoled16(mask16, 0, 0));
        Assert.True(TerrainHoleMath.IsCellHoled16(mask16, 3, 2));
        Assert.True(TerrainHoleMath.IsCellHoled16(mask16, 7, 7));
    }

    [Fact]
    public void WorldTerrainHoleMask_WithHighRes_WorksSeamlessly()
    {
        ulong mask64 = 1UL << (4 * 8 + 3);
        WorldTerrainHoleMask mask = new(mask64);

        Assert.True(mask.HasHoles);
        Assert.True(mask.IsCellHoled(3, 4));
        Assert.False(mask.IsCellHoled(2, 4));
        Assert.False(mask.IsCellHoled(3, 3));
    }
}
