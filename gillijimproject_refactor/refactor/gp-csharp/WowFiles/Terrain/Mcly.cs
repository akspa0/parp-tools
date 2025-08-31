using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Terrain;

public readonly record struct AlphaLayer(
    uint TextureId,
    uint Flags,
    uint OffsetInMcal,
    uint EffectId
);

public sealed class MclyAlpha
{
    public const int EntrySize = 16; // bytes per layer entry
    public IReadOnlyList<AlphaLayer> Layers { get; }
    
    private MclyAlpha(List<AlphaLayer> layers) => Layers = layers;
    
    public static MclyAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCLY, "Expected MCLY tag");
        Util.Assert(ch.Size % EntrySize == 0, $"MCLY size {ch.Size} not multiple of {EntrySize}");
        
        int count = (int)(ch.Size / EntrySize);
        var layers = new List<AlphaLayer>(count);
        Span<byte> buffer = stackalloc byte[EntrySize];
        
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        for (int i = 0; i < count; i++)
        {
            int read = s.Read(buffer);
            Util.Assert(read == EntrySize, $"Failed to read layer {i}");
            layers.Add(ParseLayer(buffer));
        }
        
        return new MclyAlpha(layers);
    }
    
    public static MclyAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCLY, "Expected MCLY tag");
        Util.Assert(ch.Size % EntrySize == 0, $"MCLY size {ch.Size} not multiple of {EntrySize}");
        
        int count = (int)(ch.Size / EntrySize);
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var layers = new List<AlphaLayer>(count);
        
        for (int i = 0; i < count; i++)
        {
            var entrySpan = span[(i * EntrySize)..((i + 1) * EntrySize)];
            layers.Add(ParseLayer(entrySpan));
        }
        
        return new MclyAlpha(layers);
    }
    
    private static AlphaLayer ParseLayer(ReadOnlySpan<byte> data)
    {
        uint textureId = Util.ReadUInt32LE(data, 0);
        uint flags = Util.ReadUInt32LE(data, 4);
        uint offsetInMcal = Util.ReadUInt32LE(data, 8);
        uint effectId = Util.ReadUInt32LE(data, 12);
        
        return new AlphaLayer(textureId, flags, offsetInMcal, effectId);
    }
}
