using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Terrain;

public sealed class McvtAlpha
{
    public const int ExpectedSize = 580; // 145 * 4 bytes
    public const int VertexCount = 145;
    
    public IReadOnlyList<float> Heights { get; }
    
    private McvtAlpha(List<float> heights) => Heights = heights;
    
    public static McvtAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCVT, "Expected MCVT tag");
        Util.Assert(ch.Size == ExpectedSize, $"MCVT size {ch.Size} != expected {ExpectedSize}");
        
        var heights = new List<float>(VertexCount);
        Span<byte> buffer = stackalloc byte[4];
        
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        for (int i = 0; i < VertexCount; i++)
        {
            int read = s.Read(buffer);
            Util.Assert(read == 4, $"Failed to read height vertex {i}");
            float height = BitConverter.ToSingle(buffer);
            heights.Add(height);
        }
        
        return new McvtAlpha(heights);
    }
    
    public static McvtAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCVT, "Expected MCVT tag");
        Util.Assert(ch.Size == ExpectedSize, $"MCVT size {ch.Size} != expected {ExpectedSize}");
        
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var heights = new List<float>(VertexCount);
        
        for (int i = 0; i < VertexCount; i++)
        {
            float height = BitConverter.ToSingle(span[(i * 4)..((i + 1) * 4)]);
            heights.Add(height);
        }
        
        return new McvtAlpha(heights);
    }
}
