using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Terrain;

public sealed class McrfAlpha : IChunkData
{
    public IReadOnlyList<uint> Indices { get; }
    public uint Tag => Tags.MCRF;
    public ReadOnlyMemory<byte> RawData { get; }
    public long SourceOffset { get; }
    
    private McrfAlpha(List<uint> indices, ReadOnlyMemory<byte> rawData, long sourceOffset)
    {
        Indices = indices;
        RawData = rawData;
        SourceOffset = sourceOffset;
    }
    
    public static McrfAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCRF, "Expected MCRF tag");
        Util.Assert(ch.Size % 4 == 0, $"MCRF size {ch.Size} not multiple of 4");
        
        var buffer = new byte[ch.Size];
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        int read = s.Read(buffer, 0, (int)ch.Size);
        Util.Assert(read == ch.Size, $"Failed to read MCRF data");
        
        int count = (int)(ch.Size / 4);
        var indices = new List<uint>(count);
        
        for (int i = 0; i < count; i++)
        {
            uint index = Util.ReadUInt32LE(buffer, i * 4);
            indices.Add(index);
        }
        
        return new McrfAlpha(indices, buffer, absoluteOffset);
    }
    
    public static McrfAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCRF, "Expected MCRF tag");
        Util.Assert(ch.Size % 4 == 0, $"MCRF size {ch.Size} not multiple of 4");
        
        int count = (int)(ch.Size / 4);
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var buffer = span.ToArray();
        var indices = new List<uint>(count);
        
        for (int i = 0; i < count; i++)
        {
            uint index = Util.ReadUInt32LE(span, i * 4);
            indices.Add(index);
        }
        
        return new McrfAlpha(indices, buffer, absoluteOffset);
    }
    
    /// <summary>
    /// Gets doodad (M2) indices from the beginning of the indices array
    /// </summary>
    public IReadOnlyList<uint> GetDoodadIndices(int doodadCount)
    {
        Util.Assert(doodadCount <= Indices.Count, $"Requested {doodadCount} doodads but only {Indices.Count} indices available");
        return Indices.Take(doodadCount).ToList();
    }
    
    /// <summary>
    /// Gets WMO indices from the end of the indices array
    /// </summary>
    public IReadOnlyList<uint> GetWmoIndices(int wmoCount)
    {
        Util.Assert(wmoCount <= Indices.Count, $"Requested {wmoCount} WMOs but only {Indices.Count} indices available");
        return Indices.Skip(Indices.Count - wmoCount).ToList();
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
