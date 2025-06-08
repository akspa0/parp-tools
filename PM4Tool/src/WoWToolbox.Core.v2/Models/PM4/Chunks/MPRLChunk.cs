using System.Collections.Generic;
using System.IO;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    public struct MprlEntry
    {
        public ushort Unknown_0x00 { get; set; }
        public short Unknown_0x02 { get; set; }
        public ushort Unknown_0x04 { get; set; }
        public ushort Unknown_0x06 { get; set; }
        public float PositionX { get; set; }
        public float PositionY { get; set; }
        public float PositionZ { get; set; }
        public short Unknown_0x14 { get; set; }
        public ushort Unknown_0x16 { get; set; }

        // Add Position property for transform logic
        public C3Vector Position => new C3Vector(PositionX, PositionY, PositionZ);

        public const int StructSize = 24;
    }

    public class MPRLChunk
    {
        public List<MprlEntry> Entries { get; private set; } = new();

        public void Read(BinaryReader br, long size)
        {
            if (size == 0) return;

            if (size % MprlEntry.StructSize != 0)
            {
                throw new InvalidDataException($"MPRL chunk size {size} is not a multiple of {MprlEntry.StructSize} bytes.");
            }

            var numEntries = (int)(size / MprlEntry.StructSize);
            Entries = new List<MprlEntry>(numEntries);

            for (var i = 0; i < numEntries; i++)
            {
                Entries.Add(new MprlEntry
                {
                    Unknown_0x00 = br.ReadUInt16(),
                    Unknown_0x02 = br.ReadInt16(),
                    Unknown_0x04 = br.ReadUInt16(),
                    Unknown_0x06 = br.ReadUInt16(),
                    PositionX = br.ReadSingle(),
                    PositionY = br.ReadSingle(),
                    PositionZ = br.ReadSingle(),
                    Unknown_0x14 = br.ReadInt16(),
                    Unknown_0x16 = br.ReadUInt16()
                });
            }
        }
    }
} 