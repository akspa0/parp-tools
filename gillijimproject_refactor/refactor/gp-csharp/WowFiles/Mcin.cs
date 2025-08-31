using GillijimProject.Utilities;

namespace GillijimProject.WowFiles;

public sealed class McinAlpha
{
    public sealed record Entry(uint OfsRel, uint SizeUnused, long AbsoluteOffset)
    {
        public bool HasChunk => OfsRel != 0;
    }

    public IReadOnlyList<Entry> Entries { get; }

    private McinAlpha(List<Entry> entries) => Entries = entries;

    public static McinAlpha Parse(ReadOnlySpan<byte> data, long payloadOffset, long tileStart)
    {
        var list = new List<Entry>(256);
        for (int i = 0; i < 256; i++)
        {
            uint ofs = Util.ReadUInt32LE(data, (int)payloadOffset + i * 16); 
            uint sizeUnused = Util.ReadUInt32LE(data, (int)payloadOffset + i * 16 + 4);
            long abs = ofs; 
            list.Add(new Entry(ofs, sizeUnused, abs));
        }
        return new McinAlpha(list);
    }

    public static McinAlpha Parse(Stream s, long payloadOffset, long tileStart)
    {
        var list = new List<Entry>(256);
        Span<byte> row = stackalloc byte[16]; 
        for (int i = 0; i < 256; i++)
        {
            s.Seek(payloadOffset + i * 16, SeekOrigin.Begin);
            int read = s.Read(row);
            Util.Assert(read == 16, "Failed to read MCIN entry");
            uint ofs = Util.ReadUInt32LE(row, 0);
            uint sizeUnused = Util.ReadUInt32LE(row, 4);
            long abs = ofs; 
            list.Add(new Entry(ofs, sizeUnused, abs));
        }
        return new McinAlpha(list);
    }
}
