using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Terrain;

public sealed class McrfAlpha
{
    public IReadOnlyList<uint> Indices { get; }
    
    private McrfAlpha(List<uint> indices) => Indices = indices;
    
    public static McrfAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCRF, "Expected MCRF tag");
        Util.Assert(ch.Size % 4 == 0, $"MCRF size {ch.Size} not multiple of 4");
        
        int count = (int)(ch.Size / 4);
        var indices = new List<uint>(count);
        Span<byte> buffer = stackalloc byte[4];
        
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        for (int i = 0; i < count; i++)
        {
            int read = s.Read(buffer);
            Util.Assert(read == 4, $"Failed to read index {i}");
            indices.Add(Util.ReadUInt32LE(buffer, 0));
        }
        
        return new McrfAlpha(indices);
    }
    
    public static McrfAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCRF, "Expected MCRF tag");
        Util.Assert(ch.Size % 4 == 0, $"MCRF size {ch.Size} not multiple of 4");
        
        int count = (int)(ch.Size / 4);
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var indices = new List<uint>(count);
        
        for (int i = 0; i < count; i++)
        {
            uint index = Util.ReadUInt32LE(span, i * 4);
            indices.Add(index);
        }
        
        return new McrfAlpha(indices);
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
}
