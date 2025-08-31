using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Objects;

public readonly record struct M2Placement(
    uint M2Index,
    uint UniqueId,
    float X, float Y, float Z,
    float RotX, float RotY, float RotZ,
    ushort Scale,
    ushort Flags
);

public sealed class MddfAlpha : IChunkData
{
    public const int EntrySize = 36; // bytes per M2 placement entry
    public IReadOnlyList<M2Placement> Placements { get; }
    public uint Tag => Tags.MDDF;
    public ReadOnlyMemory<byte> RawData { get; }
    public long SourceOffset { get; }
    
    private MddfAlpha(List<M2Placement> placements, ReadOnlyMemory<byte> rawData, long sourceOffset)
    {
        Placements = placements;
        RawData = rawData;
        SourceOffset = sourceOffset;
    }
    
    public static MddfAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MDDF, "Expected MDDF tag");
        Util.Assert(ch.Size % EntrySize == 0, $"MDDF size {ch.Size} not multiple of {EntrySize}");
        
        var buffer = new byte[ch.Size];
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        int read = s.Read(buffer, 0, (int)ch.Size);
        Util.Assert(read == ch.Size, $"Failed to read MDDF data");
        
        int count = (int)(ch.Size / EntrySize);
        var placements = new List<M2Placement>(count);
        
        for (int i = 0; i < count; i++)
        {
            var entrySpan = buffer.AsSpan(i * EntrySize, EntrySize);
            placements.Add(ParsePlacement(entrySpan));
        }
        
        return new MddfAlpha(placements, buffer, absoluteOffset);
    }
    
    public static MddfAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MDDF, "Expected MDDF tag");
        Util.Assert(ch.Size % EntrySize == 0, $"MDDF size {ch.Size} not multiple of {EntrySize}");
        
        int count = (int)(ch.Size / EntrySize);
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var buffer = span.ToArray();
        var placements = new List<M2Placement>(count);
        
        for (int i = 0; i < count; i++)
        {
            var entrySpan = span[(i * EntrySize)..((i + 1) * EntrySize)];
            placements.Add(ParsePlacement(entrySpan));
        }
        
        return new MddfAlpha(placements, buffer, absoluteOffset);
    }
    
    private static M2Placement ParsePlacement(ReadOnlySpan<byte> data)
    {
        uint m2Index = Util.ReadUInt32LE(data, 0);
        uint uniqueId = Util.ReadUInt32LE(data, 4);
        float x = BitConverter.ToSingle(data[8..12]);
        float y = BitConverter.ToSingle(data[12..16]);
        float z = BitConverter.ToSingle(data[16..20]);
        float rotX = BitConverter.ToSingle(data[20..24]);
        float rotY = BitConverter.ToSingle(data[24..28]);
        float rotZ = BitConverter.ToSingle(data[28..32]);
        ushort scale = Util.ReadUInt16LE(data, 32);
        ushort flags = Util.ReadUInt16LE(data, 34);
        
        return new M2Placement(m2Index, uniqueId, x, y, z, rotX, rotY, rotZ, scale, flags);
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
