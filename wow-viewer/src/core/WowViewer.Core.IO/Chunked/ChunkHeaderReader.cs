using System.Buffers.Binary;
using WowViewer.Core.Chunks;

namespace WowViewer.Core.IO.Chunked;

public static class ChunkHeaderReader
{
    public static bool TryRead(ReadOnlySpan<byte> data, out ChunkHeader header)
    {
        if (data.Length < ChunkHeader.SizeInBytes)
        {
            header = default;
            return false;
        }

        uint rawId = BinaryPrimitives.ReadUInt32LittleEndian(data);
        uint size = BinaryPrimitives.ReadUInt32LittleEndian(data[4..]);
        header = new ChunkHeader(FourCC.FromFileUInt32(rawId), size);
        return true;
    }
}