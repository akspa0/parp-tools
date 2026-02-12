using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Interfaces;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOLM chunk - Lightmap information for v14 (Alpha) WMO
    /// Contains information for blitting the MOLD color palette
    /// </summary>
    public class MOLM : ChunkBase
    {
        public override string ChunkId => "MOLM";

        public class LightmapEntry
        {
            public byte X { get; set; }
            public byte Y { get; set; }
            public byte Width { get; set; }
            public byte Height { get; set; }
        }

        public List<LightmapEntry> Lightmaps { get; private set; }

        public MOLM()
        {
            Lightmaps = new List<LightmapEntry>();
        }

        public override void Read(BinaryReader reader, uint size)
        {
            int numEntries = (int)(size / 4); // Each entry is 4 bytes
            Lightmaps.Clear();

            for (int i = 0; i < numEntries; i++)
            {
                var entry = new LightmapEntry
                {
                    X = reader.ReadByte(),
                    Y = reader.ReadByte(),
                    Width = reader.ReadByte(),
                    Height = reader.ReadByte()
                };
                Lightmaps.Add(entry);
            }
        }
    }
} 