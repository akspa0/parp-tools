using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Terrain;

public readonly record struct VertexColor(byte R, byte G, byte B, byte A);

public sealed class MccvAlpha
{
    public const int VertexCount = 145;
    public const int ExpectedSize = VertexCount * 4; // 145 * 4 = 580 bytes
    
    public IReadOnlyList<VertexColor> Colors { get; }
    
    private MccvAlpha(List<VertexColor> colors) => Colors = colors;
    
    public static MccvAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCCV, "Expected MCCV tag");
        Util.Assert(ch.Size == ExpectedSize, $"MCCV size {ch.Size} != expected {ExpectedSize}");
        
        var colors = new List<VertexColor>(VertexCount);
        Span<byte> buffer = stackalloc byte[4];
        
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        for (int i = 0; i < VertexCount; i++)
        {
            int read = s.Read(buffer);
            Util.Assert(read == 4, $"Failed to read vertex color {i}");
            colors.Add(new VertexColor(buffer[0], buffer[1], buffer[2], buffer[3]));
        }
        
        return new MccvAlpha(colors);
    }
    
    public static MccvAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCCV, "Expected MCCV tag");
        Util.Assert(ch.Size == ExpectedSize, $"MCCV size {ch.Size} != expected {ExpectedSize}");
        
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var colors = new List<VertexColor>(VertexCount);
        
        for (int i = 0; i < VertexCount; i++)
        {
            int offset = i * 4;
            colors.Add(new VertexColor(span[offset], span[offset + 1], span[offset + 2], span[offset + 3]));
        }
        
        return new MccvAlpha(colors);
    }
}
