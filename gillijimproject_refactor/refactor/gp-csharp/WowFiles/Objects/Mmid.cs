using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Objects;

public sealed class MmidAlpha : IChunkData
{
    public IReadOnlyList<uint> ModelIndices { get; }
    public uint Tag => Tags.MMID;
    public ReadOnlyMemory<byte> RawData { get; }
    public long SourceOffset { get; }
    
    private MmidAlpha(List<uint> modelIndices, ReadOnlyMemory<byte> rawData, long sourceOffset)
    {
        ModelIndices = modelIndices;
        RawData = rawData;
        SourceOffset = sourceOffset;
    }
    
    public static MmidAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MMID, "Expected MMID tag");
        Util.Assert(ch.Size % 4 == 0, $"MMID size {ch.Size} not multiple of 4");
        
        var buffer = new byte[ch.Size];
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        int read = s.Read(buffer, 0, (int)ch.Size);
        Util.Assert(read == ch.Size, $"Failed to read MMID data");
        
        int count = (int)(ch.Size / 4);
        var indices = new List<uint>(count);
        
        for (int i = 0; i < count; i++)
        {
            uint index = Util.ReadUInt32LE(buffer, i * 4);
            indices.Add(index);
        }
        
        return new MmidAlpha(indices, buffer, absoluteOffset);
    }
    
    public static MmidAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MMID, "Expected MMID tag");
        Util.Assert(ch.Size % 4 == 0, $"MMID size {ch.Size} not multiple of 4");
        
        int count = (int)(ch.Size / 4);
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var buffer = span.ToArray();
        var indices = new List<uint>(count);
        
        for (int i = 0; i < count; i++)
        {
            uint index = Util.ReadUInt32LE(span, i * 4);
            indices.Add(index);
        }
        
        return new MmidAlpha(indices, buffer, absoluteOffset);
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
