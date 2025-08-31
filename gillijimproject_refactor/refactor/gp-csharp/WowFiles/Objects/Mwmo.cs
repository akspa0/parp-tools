using GillijimProject.Utilities;
using System.Text;

namespace GillijimProject.WowFiles.Objects;

public sealed class MwmoAlpha : IChunkData
{
    public IReadOnlyList<string> WmoFilenames { get; }
    public uint Tag => Tags.MWMO;
    public ReadOnlyMemory<byte> RawData { get; }
    public long SourceOffset { get; }
    
    private MwmoAlpha(List<string> wmoFilenames, ReadOnlyMemory<byte> rawData, long sourceOffset)
    {
        WmoFilenames = wmoFilenames;
        RawData = rawData;
        SourceOffset = sourceOffset;
    }
    
    public static MwmoAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MWMO, "Expected MWMO tag");
        
        var buffer = new byte[ch.Size];
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        int read = s.Read(buffer, 0, (int)ch.Size);
        Util.Assert(read == ch.Size, $"Failed to read MWMO data");
        
        var filenames = ParseWmoFilenames(buffer);
        return new MwmoAlpha(filenames, buffer, absoluteOffset);
    }
    
    public static MwmoAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MWMO, "Expected MWMO tag");
        
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var buffer = span.ToArray();
        var filenames = ParseWmoFilenames(span);
        return new MwmoAlpha(filenames, buffer, absoluteOffset);
    }
    
    private static List<string> ParseWmoFilenames(ReadOnlySpan<byte> data)
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
