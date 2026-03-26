using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;

namespace WowViewer.Core.Tests;

public sealed class FourCcAndChunkHeaderTests
{
    [Fact]
    public void FourCc_FromString_RoundTripsThroughFileBoundary()
    {
        FourCC fourCc = FourCC.FromString("MVER");

        Assert.Equal("MVER", fourCc.ToString());
        Assert.Equal(new byte[] { (byte)'R', (byte)'E', (byte)'V', (byte)'M' }, fourCc.ToFileBytes());
        Assert.Equal(fourCc, FourCC.FromFileUInt32(fourCc.ToFileUInt32()));
    }

    [Fact]
    public void FourCc_FromFileBytes_ProducesReadableIdentifier()
    {
        FourCC fourCc = FourCC.FromFileBytes([(byte)'R', (byte)'E', (byte)'V', (byte)'M']);

        Assert.Equal("MVER", fourCc.ToString());
    }

    [Fact]
    public void ChunkHeaderReader_TryRead_ParsesReadableChunkHeader()
    {
        byte[] bytes =
        [
            (byte)'R', (byte)'E', (byte)'V', (byte)'M',
            0x10, 0x00, 0x00, 0x00
        ];

        bool success = ChunkHeaderReader.TryRead(bytes, out ChunkHeader header);

        Assert.True(success);
        Assert.Equal(FourCC.FromString("MVER"), header.Id);
        Assert.Equal(16u, header.Size);
    }

    [Fact]
    public void ChunkHeaderReader_TryRead_RejectsShortBuffer()
    {
        bool success = ChunkHeaderReader.TryRead([1, 2, 3], out ChunkHeader header);

        Assert.False(success);
        Assert.Equal(default, header);
    }
}