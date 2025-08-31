using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Objects;

public readonly record struct WmoPlacement(
    uint WmoIndex,
    uint UniqueId,
    float X, float Y, float Z,
    float RotX, float RotY, float RotZ,
    float BoundingBoxMinX, float BoundingBoxMinY, float BoundingBoxMinZ,
    float BoundingBoxMaxX, float BoundingBoxMaxY, float BoundingBoxMaxZ,
    ushort Flags,
    ushort DoodadSet,
    ushort NameSet,
    ushort Scale
);

public sealed class ModfAlpha
{
    public const int EntrySize = 64; // bytes per WMO placement entry
    public IReadOnlyList<WmoPlacement> Placements { get; }
    
    private ModfAlpha(List<WmoPlacement> placements) => Placements = placements;
    
    public static ModfAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MODF, "Expected MODF tag");
        Util.Assert(ch.Size % EntrySize == 0, $"MODF size {ch.Size} not multiple of {EntrySize}");
        
        int count = (int)(ch.Size / EntrySize);
        var placements = new List<WmoPlacement>(count);
        Span<byte> buffer = stackalloc byte[EntrySize];
        
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        for (int i = 0; i < count; i++)
        {
            int read = s.Read(buffer);
            Util.Assert(read == EntrySize, $"Failed to read WMO placement {i}");
            placements.Add(ParsePlacement(buffer));
        }
        
        return new ModfAlpha(placements);
    }
    
    public static ModfAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MODF, "Expected MODF tag");
        Util.Assert(ch.Size % EntrySize == 0, $"MODF size {ch.Size} not multiple of {EntrySize}");
        
        int count = (int)(ch.Size / EntrySize);
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var placements = new List<WmoPlacement>(count);
        
        for (int i = 0; i < count; i++)
        {
            var entrySpan = span[(i * EntrySize)..((i + 1) * EntrySize)];
            placements.Add(ParsePlacement(entrySpan));
        }
        
        return new ModfAlpha(placements);
    }
    
    private static WmoPlacement ParsePlacement(ReadOnlySpan<byte> data)
    {
        uint wmoIndex = Util.ReadUInt32LE(data, 0);
        uint uniqueId = Util.ReadUInt32LE(data, 4);
        float x = BitConverter.ToSingle(data[8..12]);
        float y = BitConverter.ToSingle(data[12..16]);
        float z = BitConverter.ToSingle(data[16..20]);
        float rotX = BitConverter.ToSingle(data[20..24]);
        float rotY = BitConverter.ToSingle(data[24..28]);
        float rotZ = BitConverter.ToSingle(data[28..32]);
        float bbMinX = BitConverter.ToSingle(data[32..36]);
        float bbMinY = BitConverter.ToSingle(data[36..40]);
        float bbMinZ = BitConverter.ToSingle(data[40..44]);
        float bbMaxX = BitConverter.ToSingle(data[44..48]);
        float bbMaxY = BitConverter.ToSingle(data[48..52]);
        float bbMaxZ = BitConverter.ToSingle(data[52..56]);
        ushort flags = Util.ReadUInt16LE(data, 56);
        ushort doodadSet = Util.ReadUInt16LE(data, 58);
        ushort nameSet = Util.ReadUInt16LE(data, 60);
        ushort scale = Util.ReadUInt16LE(data, 62);
        
        return new WmoPlacement(wmoIndex, uniqueId, x, y, z, rotX, rotY, rotZ,
            bbMinX, bbMinY, bbMinZ, bbMaxX, bbMaxY, bbMaxZ,
            flags, doodadSet, nameSet, scale);
    }
}
