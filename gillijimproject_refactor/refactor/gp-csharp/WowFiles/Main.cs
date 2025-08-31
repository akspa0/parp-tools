using GillijimProject.Utilities;

namespace GillijimProject.WowFiles;

public sealed class MainAlpha
{
    public sealed record Entry(uint MhdrOffset, bool HasMhdr);
    public IReadOnlyList<Entry> Entries { get; }

    private MainAlpha(List<Entry> entries) => Entries = entries;

    public static MainAlpha Parse(ReadOnlySpan<byte> data, ChunkHeader main)
    {
        Util.Assert(main.Tag == Tags.MAIN, "Not MAIN chunk");
        var span = data[(int)main.PayloadOffset..(int)(main.PayloadOffset + main.Size)];
        Util.Assert(main.Size % 16 == 0, $"MAIN size {main.Size} not multiple of 16");
        var list = new List<Entry>((int)(main.Size / 16));
        for (int i = 0; i < main.Size; i += 16)
        {
            uint mhdrOfs = Util.ReadUInt32LE(span, i);
            uint size = Util.ReadUInt32LE(span, i + 4); // may be 0
            bool has = mhdrOfs != 0; // presence indicated by offset; size may be 0 per ref
            list.Add(new Entry(mhdrOfs, has));
        }
        return new MainAlpha(list);
    }

    public static MainAlpha Parse(Stream s, ChunkHeader main)
    {
        Util.Assert(main.Tag == Tags.MAIN, "Not MAIN chunk");
        Util.Assert(main.Size % 16 == 0, $"MAIN size {main.Size} not multiple of 16");
        var list = new List<Entry>((int)(main.Size / 16));
        Span<byte> row = stackalloc byte[16];
        long pos = main.PayloadOffset;
        for (long i = 0; i < main.Size; i += 16)
        {
            s.Seek(pos + i, SeekOrigin.Begin);
            int read = s.Read(row);
            Util.Assert(read == 16, "Failed to read MAIN entry");
            uint mhdrOfs = Util.ReadUInt32LE(row, 0);
            // uint size = Util.ReadUInt32LE(row, 4); // may be 0; not needed for presence
            bool has = mhdrOfs != 0;
            list.Add(new Entry(mhdrOfs, has));
        }
        return new MainAlpha(list);
    }
}
