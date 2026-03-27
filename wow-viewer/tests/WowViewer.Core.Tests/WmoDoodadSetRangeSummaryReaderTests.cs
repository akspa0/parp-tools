using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoDoodadSetRangeSummaryReaderTests
{
    [Fact]
    public void Read_ModsAndModdBuffers_ProducesRangeSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODS", CreateModsPayload((0, 4), (10, 0), (12, 6))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODD", new byte[12 * 40]),
        ];

        using MemoryStream stream = new(bytes);
        WmoDoodadSetRangeSummary summary = WmoDoodadSetRangeSummaryReader.Read(stream, "synthetic_mods_modd_root.wmo");

        Assert.Equal(3, summary.EntryCount);
        Assert.Equal(12, summary.PlacementCount);
        Assert.Equal(1, summary.EmptySetCount);
        Assert.Equal(1, summary.FullyCoveredSetCount);
        Assert.Equal(1, summary.OutOfRangeSetCount);
        Assert.Equal(18, summary.MaxRangeEnd);
    }

    private static byte[] CreateModsPayload(params (uint Start, uint Count)[] entries)
    {
        byte[] bytes = new byte[entries.Length * 32];
        for (int i = 0; i < entries.Length; i++)
        {
            int offset = i * 32;
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset + 20, 4), entries[i].Start);
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset + 24, 4), entries[i].Count);
        }

        return bytes;
    }
}
