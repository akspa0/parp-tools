using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Objects;

public sealed class MmidAlpha
{
    public IReadOnlyList<uint> ModelIndices { get; }
    
    private MmidAlpha(List<uint> modelIndices) => ModelIndices = modelIndices;
    
    public static MmidAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MMID, "Expected MMID tag");
        Util.Assert(ch.Size % 4 == 0, $"MMID size {ch.Size} not multiple of 4");
        
        int count = (int)(ch.Size / 4);
        var indices = new List<uint>(count);
        Span<byte> buffer = stackalloc byte[4];
        
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        for (int i = 0; i < count; i++)
        {
            int read = s.Read(buffer);
            Util.Assert(read == 4, $"Failed to read model index {i}");
            indices.Add(Util.ReadUInt32LE(buffer, 0));
        }
        
        return new MmidAlpha(indices);
    }
    
    public static MmidAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MMID, "Expected MMID tag");
        Util.Assert(ch.Size % 4 == 0, $"MMID size {ch.Size} not multiple of 4");
        
        int count = (int)(ch.Size / 4);
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var indices = new List<uint>(count);
        
        for (int i = 0; i < count; i++)
        {
            indices.Add(Util.ReadUInt32LE(span, i * 4));
        }
        
        return new MmidAlpha(indices);
    }
}
