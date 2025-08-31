using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Objects;

public readonly record struct M2Placement(
    uint ModelIndex,
    uint UniqueId,
    float X, float Y, float Z,
    float RotX, float RotY, float RotZ,
    ushort Scale,
    ushort Flags
);

public sealed class MddfAlpha
{
    public const int EntrySize = 36; // bytes per M2 placement entry
    public IReadOnlyList<M2Placement> Placements { get; }
    
    private MddfAlpha(List<M2Placement> placements) => Placements = placements;
    
    public static MddfAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MDDF, "Expected MDDF tag");
        Util.Assert(ch.Size % EntrySize == 0, $"MDDF size {ch.Size} not multiple of {EntrySize}");
        
        int count = (int)(ch.Size / EntrySize);
        var placements = new List<M2Placement>(count);
        Span<byte> buffer = stackalloc byte[EntrySize];
        
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        for (int i = 0; i < count; i++)
        {
            int read = s.Read(buffer);
            Util.Assert(read == EntrySize, $"Failed to read M2 placement {i}");
            placements.Add(ParsePlacement(buffer));
        }
        
        return new MddfAlpha(placements);
    }
    
    public static MddfAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MDDF, "Expected MDDF tag");
        Util.Assert(ch.Size % EntrySize == 0, $"MDDF size {ch.Size} not multiple of {EntrySize}");
        
        int count = (int)(ch.Size / EntrySize);
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var placements = new List<M2Placement>(count);
        
        for (int i = 0; i < count; i++)
        {
            var entrySpan = span[(i * EntrySize)..((i + 1) * EntrySize)];
            placements.Add(ParsePlacement(entrySpan));
        }
        
        return new MddfAlpha(placements);
    }
    
    private static M2Placement ParsePlacement(ReadOnlySpan<byte> data)
    {
        uint modelIndex = Util.ReadUInt32LE(data, 0);
        uint uniqueId = Util.ReadUInt32LE(data, 4);
        float x = BitConverter.ToSingle(data[8..12]);
        float y = BitConverter.ToSingle(data[12..16]);
        float z = BitConverter.ToSingle(data[16..20]);
        float rotX = BitConverter.ToSingle(data[20..24]);
        float rotY = BitConverter.ToSingle(data[24..28]);
        float rotZ = BitConverter.ToSingle(data[28..32]);
        ushort scale = Util.ReadUInt16LE(data, 32);
        ushort flags = Util.ReadUInt16LE(data, 34);
        
        return new M2Placement(modelIndex, uniqueId, x, y, z, rotX, rotY, rotZ, scale, flags);
    }
}
