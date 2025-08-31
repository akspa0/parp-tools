using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Terrain;

public readonly record struct BoundingBox(float MinX, float MinY, float MinZ, float MaxX, float MaxY, float MaxZ);

public sealed class McbbAlpha
{
    public IReadOnlyList<BoundingBox> BoundingBoxes { get; }
    
    private McbbAlpha(List<BoundingBox> boundingBoxes) => BoundingBoxes = boundingBoxes;
    
    public static McbbAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCBB, "Expected MCBB tag");
        Util.Assert(ch.Size % 24 == 0, $"MCBB size {ch.Size} not multiple of 24");
        
        int count = (int)(ch.Size / 24);
        var boxes = new List<BoundingBox>(count);
        Span<byte> buffer = stackalloc byte[24];
        
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        for (int i = 0; i < count; i++)
        {
            int read = s.Read(buffer);
            Util.Assert(read == 24, $"Failed to read bounding box {i}");
            
            float minX = BitConverter.ToSingle(buffer[0..4]);
            float minY = BitConverter.ToSingle(buffer[4..8]);
            float minZ = BitConverter.ToSingle(buffer[8..12]);
            float maxX = BitConverter.ToSingle(buffer[12..16]);
            float maxY = BitConverter.ToSingle(buffer[16..20]);
            float maxZ = BitConverter.ToSingle(buffer[20..24]);
            
            boxes.Add(new BoundingBox(minX, minY, minZ, maxX, maxY, maxZ));
        }
        
        return new McbbAlpha(boxes);
    }
    
    public static McbbAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCBB, "Expected MCBB tag");
        Util.Assert(ch.Size % 24 == 0, $"MCBB size {ch.Size} not multiple of 24");
        
        int count = (int)(ch.Size / 24);
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var boxes = new List<BoundingBox>(count);
        
        for (int i = 0; i < count; i++)
        {
            int offset = i * 24;
            float minX = BitConverter.ToSingle(span[(offset + 0)..(offset + 4)]);
            float minY = BitConverter.ToSingle(span[(offset + 4)..(offset + 8)]);
            float minZ = BitConverter.ToSingle(span[(offset + 8)..(offset + 12)]);
            float maxX = BitConverter.ToSingle(span[(offset + 12)..(offset + 16)]);
            float maxY = BitConverter.ToSingle(span[(offset + 16)..(offset + 20)]);
            float maxZ = BitConverter.ToSingle(span[(offset + 20)..(offset + 24)]);
            
            boxes.Add(new BoundingBox(minX, minY, minZ, maxX, maxY, maxZ));
        }
        
        return new McbbAlpha(boxes);
    }
}
