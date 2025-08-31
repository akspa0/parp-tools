using GillijimProject.Utilities;
using System.Text;

namespace GillijimProject.WowFiles.Objects;

public sealed class MonmAlpha : IChunkData
{
    public IReadOnlyList<string> WmoFilenames { get; }
    public uint Tag => Tags.MONM;
    public ReadOnlyMemory<byte> RawData { get; }
    public long SourceOffset { get; }
    
    private MonmAlpha(List<string> wmoFilenames, ReadOnlyMemory<byte> rawData, long sourceOffset)
    {
        WmoFilenames = wmoFilenames;
        RawData = rawData;
        SourceOffset = sourceOffset;
    }
    
    public static MonmAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MONM, "Expected MONM tag");
        
        var buffer = new byte[ch.Size];
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        int read = s.Read(buffer, 0, (int)ch.Size);
        Util.Assert(read == ch.Size, $"Failed to read MONM data");
        
        var filenames = ParseWmoFilenames(buffer);
        return new MonmAlpha(filenames, buffer, absoluteOffset);
    }
    
    public static MonmAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MONM, "Expected MONM tag");
        
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var buffer = span.ToArray();
        var filenames = ParseWmoFilenames(span);
        return new MonmAlpha(filenames, buffer, absoluteOffset);
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
