using System;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// Version information for WMO files.
    /// v14 - Alpha client
    /// v17 - Retail client
    /// </summary>
    public class MVER : IChunk
    {
        public uint Version { get; set; }

        public void Read(BinaryReader reader)
        {
            Version = reader.ReadUInt32();
            
            if (Version != 14 && Version != 17)
                throw new InvalidDataException($"Unsupported WMO version: {Version}");
        }

        public void Write(BinaryWriter writer)
        {
            writer.Write(Version);
        }
    }
} 