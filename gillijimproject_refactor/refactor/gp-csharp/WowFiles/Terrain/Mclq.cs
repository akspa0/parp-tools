using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Terrain;

public sealed class MclqAlpha
{
    public ReadOnlyMemory<byte> RawData { get; }
    public uint Size { get; }
    
    private MclqAlpha(ReadOnlyMemory<byte> rawData, uint size)
    {
        RawData = rawData;
        Size = size;
    }
    
    public static MclqAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCLQ, "Expected MCLQ tag");
        
        // MCLQ has variable size - store raw data for Alpha-specific fallback handling
        var buffer = new byte[ch.Size];
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        int read = s.Read(buffer);
        Util.Assert(read == ch.Size, $"Failed to read MCLQ data, expected {ch.Size} bytes");
        
        return new MclqAlpha(buffer, ch.Size);
    }
    
    public static MclqAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCLQ, "Expected MCLQ tag");
        
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var buffer = new byte[ch.Size];
        span.CopyTo(buffer);
        
        return new MclqAlpha(buffer, ch.Size);
    }
}
