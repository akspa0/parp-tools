using System.IO;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Version information chunk for ADT files.
/// </summary>
public class MverChunk : ChunkBase
{
    public override string ChunkId => "MVER";
    
    /// <summary>
    /// Gets the version of the ADT file.
    /// Possible values:
    /// - 18: Retail format (Classic through current)
    /// - 22: Experimental format (Cataclysm beta)
    /// - 23: Experimental format (Cataclysm beta)
    /// </summary>
    public uint Version { get; private set; }

    public override void Parse(BinaryReader reader, uint size)
    {
        if (size != 4)
        {
            throw new InvalidDataException($"MVER chunk size must be 4 bytes, found {size} bytes");
        }

        Version = reader.ReadUInt32();
        
        // Validate version for ADT files
        if (Version != 18 && Version != 22 && Version != 23)
        {
            throw new InvalidDataException($"Unexpected ADT version: {Version}. Only versions 18 (retail) and 22/23 (experimental) are supported.");
        }
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