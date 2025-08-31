using GillijimProject.Utilities;
using System.Text;

namespace GillijimProject.WowFiles.Objects;

public sealed class MmdxAlpha
{
    public IReadOnlyList<string> ModelFilenames { get; }
    
    private MmdxAlpha(List<string> modelFilenames) => ModelFilenames = modelFilenames;
    
    public static MmdxAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MMDX, "Expected MMDX tag");
        
        var buffer = new byte[ch.Size];
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        int read = s.Read(buffer);
        Util.Assert(read == ch.Size, $"Failed to read MMDX data, expected {ch.Size} bytes");
        
        return ParseFilenames(buffer);
    }
    
    public static MmdxAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MMDX, "Expected MMDX tag");
        
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var buffer = new byte[ch.Size];
        span.CopyTo(buffer);
        
        return ParseFilenames(buffer);
    }
    
    private static MmdxAlpha ParseFilenames(ReadOnlySpan<byte> data)
    {
        var filenames = new List<string>();
        int start = 0;
        
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] == 0) // null terminator
            {
                if (i > start)
                {
                    var filename = Encoding.UTF8.GetString(data[start..i]);
                    filenames.Add(filename);
                }
                start = i + 1;
            }
        }
        
        return new MmdxAlpha(filenames);
    }
}
