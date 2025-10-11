using System.IO;

namespace ArcaneFileParser.Core.Formats.WMO.Structures
{
    /// <summary>
    /// SMOGxBatch - Batch structure for v14 WMO groups
    /// Used in MOGP for both interior and exterior batches
    /// </summary>
    public class SMOGxBatch
    {
        public uint StartVertex { get; set; }
        public uint Count { get; set; }
        public uint MinIndex { get; set; }
        public uint MaxIndex { get; set; }
        public uint MaterialId { get; set; }

        public void Read(BinaryReader reader)
        {
            StartVertex = reader.ReadUInt32();
            Count = reader.ReadUInt32();
            MinIndex = reader.ReadUInt32();
            MaxIndex = reader.ReadUInt32();
            MaterialId = reader.ReadUInt32();
        }

        public void Write(BinaryWriter writer)
        {
            writer.Write(StartVertex);
            writer.Write(Count);
            writer.Write(MinIndex);
            writer.Write(MaxIndex);
            writer.Write(MaterialId);
        }
    }
} 