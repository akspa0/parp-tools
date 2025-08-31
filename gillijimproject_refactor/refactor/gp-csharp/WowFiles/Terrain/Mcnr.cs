using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Terrain;

public readonly record struct Normal(sbyte X, sbyte Y, sbyte Z);

public sealed class McnrAlpha
{
    public const int ExpectedSize = 435; // 145 * 3 bytes
    public const int NormalCount = 145;
    
    public IReadOnlyList<Normal> Normals { get; }
    
    private McnrAlpha(List<Normal> normals) => Normals = normals;
    
    public static McnrAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCNR, "Expected MCNR tag");
        Util.Assert(ch.Size == ExpectedSize, $"MCNR size {ch.Size} != expected {ExpectedSize}");
        
        var normals = new List<Normal>(NormalCount);
        Span<byte> buffer = stackalloc byte[3];
        
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        for (int i = 0; i < NormalCount; i++)
        {
            int read = s.Read(buffer);
            Util.Assert(read == 3, $"Failed to read normal {i}");
            normals.Add(new Normal((sbyte)buffer[0], (sbyte)buffer[1], (sbyte)buffer[2]));
        }
        
        return new McnrAlpha(normals);
    }
    
    public static McnrAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCNR, "Expected MCNR tag");
        Util.Assert(ch.Size == ExpectedSize, $"MCNR size {ch.Size} != expected {ExpectedSize}");
        
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var normals = new List<Normal>(NormalCount);
        
        for (int i = 0; i < NormalCount; i++)
        {
            int offset = i * 3;
            normals.Add(new Normal((sbyte)span[offset], (sbyte)span[offset + 1], (sbyte)span[offset + 2]));
        }
        
        return new McnrAlpha(normals);
    }
}
