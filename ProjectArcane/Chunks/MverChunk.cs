using System.IO;

public class MverChunk : ChunkBase
{
    public override string ChunkId => "MVER";
    
    public uint Version { get; private set; }

    public override void Parse(BinaryReader reader, uint size)
    {
        if (size != 4)
        {
            throw new InvalidDataException($"MVER chunk size must be 4 bytes, found {size} bytes");
        }

        Version = reader.ReadUInt32();
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        writer.Write(Version);
    }

    public override string ToHumanReadable()
    {
        return $"Version: {Version}";
    }
} 