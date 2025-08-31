using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Terrain;

public readonly record struct SoundEmitter(uint SoundId, float X, float Y, float Z);

public sealed class McseAlpha
{
    public IReadOnlyList<SoundEmitter> SoundEmitters { get; }
    
    private McseAlpha(List<SoundEmitter> soundEmitters) => SoundEmitters = soundEmitters;
    
    public static McseAlpha Parse(Stream s, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(s, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCSE, "Expected MCSE tag");
        Util.Assert(ch.Size % 16 == 0, $"MCSE size {ch.Size} not multiple of 16");
        
        int count = (int)(ch.Size / 16);
        var emitters = new List<SoundEmitter>(count);
        Span<byte> buffer = stackalloc byte[16];
        
        s.Seek(ch.PayloadOffset, SeekOrigin.Begin);
        for (int i = 0; i < count; i++)
        {
            int read = s.Read(buffer);
            Util.Assert(read == 16, $"Failed to read sound emitter {i}");
            
            uint soundId = Util.ReadUInt32LE(buffer, 0);
            float x = BitConverter.ToSingle(buffer[4..8]);
            float y = BitConverter.ToSingle(buffer[8..12]);
            float z = BitConverter.ToSingle(buffer[12..16]);
            
            emitters.Add(new SoundEmitter(soundId, x, y, z));
        }
        
        return new McseAlpha(emitters);
    }
    
    public static McseAlpha Parse(ReadOnlySpan<byte> data, long absoluteOffset)
    {
        var ch = Chunk.ReadHeader(data, absoluteOffset);
        Util.Assert(ch.Tag == Tags.MCSE, "Expected MCSE tag");
        Util.Assert(ch.Size % 16 == 0, $"MCSE size {ch.Size} not multiple of 16");
        
        int count = (int)(ch.Size / 16);
        var span = data[(int)ch.PayloadOffset..(int)(ch.PayloadOffset + ch.Size)];
        var emitters = new List<SoundEmitter>(count);
        
        for (int i = 0; i < count; i++)
        {
            int offset = i * 16;
            uint soundId = Util.ReadUInt32LE(span, offset);
            float x = BitConverter.ToSingle(span[(offset + 4)..(offset + 8)]);
            float y = BitConverter.ToSingle(span[(offset + 8)..(offset + 12)]);
            float z = BitConverter.ToSingle(span[(offset + 12)..(offset + 16)]);
            
            emitters.Add(new SoundEmitter(soundId, x, y, z));
        }
        
        return new McseAlpha(emitters);
    }
}
