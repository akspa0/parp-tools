using GillijimProject.Utilities;

namespace GillijimProject.WowFiles;

public static class Chunk
{
    public static ChunkHeader ReadHeader(ReadOnlySpan<byte> data, long offset)
    {
        Util.Assert(offset >= 0 && offset + 8 <= data.Length, $"Chunk header OOB @0x{offset:X}");
        uint tag = Util.ReadUInt32LE(data, (int)offset);
        uint size = Util.ReadUInt32LE(data, (int)offset + 4);
        long payload = offset + 8;
        Util.Assert(payload + size <= data.Length, $"Chunk payload OOB tag={Util.FourCcToDisplay(tag)} size={size}");
        return new ChunkHeader(tag, size, payload);
    }

    public static ChunkHeader ReadHeader(Stream s, long offset)
    {
        Util.Assert(offset >= 0 && offset + 8 <= s.Length, $"Chunk header OOB @0x{offset:X}");
        s.Seek(offset, SeekOrigin.Begin);
        Span<byte> buf = stackalloc byte[8];
        int read = s.Read(buf);
        Util.Assert(read == 8, "Failed to read chunk header");
        uint tag = Util.ReadUInt32LE(buf, 0);
        uint size = Util.ReadUInt32LE(buf, 4);
        long payload = offset + 8;
        Util.Assert(payload + size <= s.Length, $"Chunk payload OOB tag={Util.FourCcToDisplay(tag)} size={size}");
        return new ChunkHeader(tag, size, payload);
    }
}
