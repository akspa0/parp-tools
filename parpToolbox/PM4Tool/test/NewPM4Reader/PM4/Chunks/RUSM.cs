using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// Structure for RUSM (MSUR reversed) entries as defined in documentation.
    /// </summary>
    public class RUSMEntry
    {
        /// <summary>
        /// First byte - earlier documentation has this as bitmask32 flags
        /// </summary>
        public byte Flags { get; set; }
        
        /// <summary>
        /// Second byte - count of indices in MSVI
        /// </summary>
        public byte IndexCount { get; set; }
        
        /// <summary>
        /// Third byte
        /// </summary>
        public byte Value1 { get; set; }
        
        /// <summary>
        /// Fourth byte - likely padding
        /// </summary>
        public byte Padding { get; set; }
        
        /// <summary>
        /// Float at offset 0x04
        /// </summary>
        public float Value2 { get; set; }
        
        /// <summary>
        /// Float at offset 0x08
        /// </summary>
        public float Value3 { get; set; }
        
        /// <summary>
        /// Float at offset 0x0C
        /// </summary>
        public float Value4 { get; set; }
        
        /// <summary>
        /// Float at offset 0x10
        /// </summary>
        public float Value5 { get; set; }
        
        /// <summary>
        /// MSVI first index at offset 0x14
        /// </summary>
        public uint MSVIFirstIndex { get; set; }
        
        /// <summary>
        /// UInt32 at offset 0x18
        /// </summary>
        public uint Value6 { get; set; }
        
        /// <summary>
        /// UInt32 at offset 0x1C
        /// </summary>
        public uint Value7 { get; set; }
    }
    
    /// <summary>
    /// RUSM chunk (MSUR reversed)
    /// According to documentation, this contains information about n-gons defined in MSVI (IVSM reversed).
    /// </summary>
    public class RUSM : BaseMeshChunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public override string Signature => "RUSM";
        
        /// <summary>
        /// Entries in the RUSM chunk.
        /// Each entry contains information about n-gons defined in MSVI (IVSM reversed).
        /// </summary>
        public List<RUSMEntry> Entries { get; } = new List<RUSMEntry>();
        
        /// <summary>
        /// Initializes a new instance of the <see cref="RUSM"/> class.
        /// </summary>
        public RUSM()
        {
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="RUSM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public RUSM(BinaryReader reader)
        {
            ReadBinary(reader);
        }
        
        /// <summary>
        /// Parses the RUSM data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        protected override void ParseData(BinaryReader reader)
        {
            Entries.Clear();
            
            // Each entry is 32 bytes according to documentation
            int bytesAvailable = (int)reader.BaseStream.Length;
            int entryCount = bytesAvailable / 32;
            
            for (int i = 0; i < entryCount; i++)
            {
                var entry = new RUSMEntry
                {
                    Flags = reader.ReadByte(),
                    IndexCount = reader.ReadByte(),
                    Value1 = reader.ReadByte(),
                    Padding = reader.ReadByte(),
                    Value2 = reader.ReadSingle(),
                    Value3 = reader.ReadSingle(),
                    Value4 = reader.ReadSingle(),
                    Value5 = reader.ReadSingle(),
                    MSVIFirstIndex = reader.ReadUInt32(),
                    Value6 = reader.ReadUInt32(),
                    Value7 = reader.ReadUInt32()
                };
                
                Entries.Add(entry);
            }
        }
        
        /// <summary>
        /// Gets a detailed description of the chunk contents.
        /// </summary>
        /// <returns>A detailed description string.</returns>
        public override string GetDetailedDescription()
        {
            if (!IsParsed)
            {
                return $"RUSM chunk: {DataSize} bytes (unparsed)";
            }
            
            var sb = new StringBuilder();
            sb.AppendLine($"RUSM entry count: {Entries.Count}");
            sb.AppendLine("These entries define properties of n-gons indexed in MSVI (IVSM reversed)");
            
            // Display the first few entries
            int displayCount = Math.Min(Entries.Count, 10);
            for (int i = 0; i < displayCount; i++)
            {
                var entry = Entries[i];
                sb.AppendLine($"  Entry {i}:");
                sb.AppendLine($"    Flags: {entry.Flags}, IndexCount: {entry.IndexCount}, Value1: {entry.Value1}");
                sb.AppendLine($"    Values: {entry.Value2:F3}, {entry.Value3:F3}, {entry.Value4:F3}, {entry.Value5:F3}");
                sb.AppendLine($"    MSVI FirstIndex: {entry.MSVIFirstIndex}, Value6: {entry.Value6}, Value7: {entry.Value7}");
            }
            
            if (Entries.Count > displayCount)
            {
                sb.AppendLine($"  ... and {Entries.Count - displayCount} more entries");
            }
            
            return sb.ToString();
        }
    }
} 