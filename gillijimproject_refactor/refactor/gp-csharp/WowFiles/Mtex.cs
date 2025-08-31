using GillijimProject.Utilities;
using System.Text;

namespace GillijimProject.WowFiles;

public sealed class MtexAlpha
{
    public IReadOnlyList<string> TextureFilenames { get; }
    
    private MtexAlpha(List<string> textureFilenames) => TextureFilenames = textureFilenames;
    
    public static MtexAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MTEX, "Expected MTEX tag");
        
        var buffer = new byte[ch.Size];
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        int read = s.Read(buffer, 0, (int)ch.Size);
        Util.Assert(read == ch.Size, $"Failed to read MTEX data");
        
        return ParseTextureFilenames(buffer);
    }
    
    public static MtexAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MTEX, "Expected MTEX tag");
        
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        return ParseTextureFilenames(span);
    }
    
    private static MtexAlpha ParseTextureFilenames(ReadOnlySpan<byte> data)
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
        
        // Handle case where last string doesn't end with null terminator
        if (start < data.Length)
        {
            var filename = Encoding.UTF8.GetString(data[start..]);
            filenames.Add(filename);
        }
        
        return new MtexAlpha(filenames);
    }
}
