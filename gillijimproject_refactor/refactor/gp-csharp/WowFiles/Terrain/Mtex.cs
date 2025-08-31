using GillijimProject.Utilities;
using System.Text;

namespace GillijimProject.WowFiles.Terrain;

public sealed class MtexAlpha : IChunkData
{
    public IReadOnlyList<string> TextureFilenames { get; }
    public uint Tag => Tags.MTEX;
    public ReadOnlyMemory<byte> RawData { get; }
    public long SourceOffset { get; }
    
    private MtexAlpha(List<string> textureFilenames, ReadOnlyMemory<byte> rawData, long sourceOffset)
    {
        TextureFilenames = textureFilenames;
        RawData = rawData;
        SourceOffset = sourceOffset;
    }
    
    public static MtexAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MTEX, "Expected MTEX tag");
        
        var buffer = new byte[ch.Size];
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        int read = s.Read(buffer, 0, (int)ch.Size);
        Util.Assert(read == ch.Size, $"Failed to read MTEX data");
        
        var filenames = ParseTextureFilenames(buffer);
        return new MtexAlpha(filenames, buffer, absoluteOffset);
    }
    
    public static MtexAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MTEX, "Expected MTEX tag");
        
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var buffer = span.ToArray();
        var filenames = ParseTextureFilenames(span);
        return new MtexAlpha(filenames, buffer, absoluteOffset);
    }
    
    private static List<string> ParseTextureFilenames(ReadOnlySpan<byte> data)
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
        
        return filenames;
    }
    
    public byte[] ToBytes()
    {
        var result = new byte[8 + RawData.Length];
        BitConverter.GetBytes(Tag).CopyTo(result, 0);
        BitConverter.GetBytes((uint)RawData.Length).CopyTo(result, 4);
        RawData.Span.CopyTo(result.AsSpan(8));
        return result;
    }
}
