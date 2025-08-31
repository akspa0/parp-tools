using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Terrain;

public readonly record struct AlphaLayer(
    uint TextureId,
    uint Flags,
    uint OffsetInMcal,
    uint EffectId
);

public sealed class MclyAlpha : IChunkData
{
    public const int EntrySize = 16; // bytes per layer entry
    public IReadOnlyList<AlphaLayer> Layers { get; }
    public uint Tag => Tags.MCLY;
    public ReadOnlyMemory<byte> RawData { get; }
    public long SourceOffset { get; }
    
    private MclyAlpha(List<AlphaLayer> layers, ReadOnlyMemory<byte> rawData, long sourceOffset)
    {
        Layers = layers;
        RawData = rawData;
        SourceOffset = sourceOffset;
    }
    
    public static MclyAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCLY, "Expected MCLY tag");
        Util.Assert(ch.Size % EntrySize == 0, $"MCLY size {ch.Size} not multiple of {EntrySize}");
        
        var buffer = new byte[ch.Size];
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        int read = s.Read(buffer, 0, (int)ch.Size);
        Util.Assert(read == ch.Size, $"Failed to read MCLY data");
        
        int count = (int)(ch.Size / EntrySize);
        var layers = new List<AlphaLayer>(count);
        
        for (int i = 0; i < count; i++)
        {
            var entrySpan = buffer.AsSpan(i * EntrySize, EntrySize);
            layers.Add(ParseLayer(entrySpan));
        }
        
        return new MclyAlpha(layers, buffer, absoluteOffset);
    }
    
    public static MclyAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCLY, "Expected MCLY tag");
        Util.Assert(ch.Size % EntrySize == 0, $"MCLY size {ch.Size} not multiple of {EntrySize}");
        
        int count = (int)(ch.Size / EntrySize);
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var buffer = span.ToArray();
        var layers = new List<AlphaLayer>(count);
        
        for (int i = 0; i < count; i++)
        {
            var entrySpan = span[(i * EntrySize)..((i + 1) * EntrySize)];
            layers.Add(ParseLayer(entrySpan));
        }
        
        return new MclyAlpha(layers, buffer, absoluteOffset);
    }
    
    private static AlphaLayer ParseLayer(ReadOnlySpan<byte> data)
    {
        uint textureId = Util.ReadUInt32LE(data, 0);
        uint flags = Util.ReadUInt32LE(data, 4);
        uint offsetInMcal = Util.ReadUInt32LE(data, 8);
        uint effectId = Util.ReadUInt32LE(data, 12);
        
        return new AlphaLayer(textureId, flags, offsetInMcal, effectId);
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
