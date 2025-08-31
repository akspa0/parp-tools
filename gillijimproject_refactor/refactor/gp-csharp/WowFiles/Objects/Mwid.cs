using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Objects;

public sealed class MwidAlpha
{
    public IReadOnlyList<uint> WmoIndices { get; }
    
    private MwidAlpha(List<uint> wmoIndices) => WmoIndices = wmoIndices;
    
    public static MwidAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MWID, "Expected MWID tag");
        Util.Assert(ch.Size % 4 == 0, $"MWID size {ch.Size} not multiple of 4");
        
        int count = (int)(ch.Size / 4);
        var indices = new List<uint>(count);
        Span<byte> buffer = stackalloc byte[4];
        
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        for (int i = 0; i < count; i++)
        {
            int read = s.Read(buffer);
            Util.Assert(read == 4, $"Failed to read WMO index {i}");
            indices.Add(Util.ReadUInt32LE(buffer, 0));
        }
        
        return new MwidAlpha(indices);
    }
    
    public static MwidAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MWID, "Expected MWID tag");
        Util.Assert(ch.Size % 4 == 0, $"MWID size {ch.Size} not multiple of 4");
        
        int count = (int)(ch.Size / 4);
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var indices = new List<uint>(count);
        
        for (int i = 0; i < count; i++)
        {
            indices.Add(Util.ReadUInt32LE(span, i * 4));
        }
        
        return new MwidAlpha(indices);
    }
}
