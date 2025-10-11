using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// Structure for KLSM (MSLK reversed) entries as defined in documentation.
    /// </summary>
    public class KLSMEntry
    {
        /// <summary>
        /// First byte - earlier documentation has this as bitmask32 flags
        /// </summary>
        public byte Flags { get; set; }
        
        /// <summary>
        /// Second byte
        /// </summary>
        public byte Value1 { get; set; }
        
        /// <summary>
        /// UInt16 at offset 0x02 - likely padding according to documentation
        /// </summary>
        public ushort Padding { get; set; }
        
        /// <summary>
        /// UInt32 at offset 0x04 - an index somewhere
        /// </summary>
        public uint Index { get; set; }
        
        /// <summary>
        /// MSPI first index - 3 bytes at offset 0x08 (-1 if last byte is 0)
        /// </summary>
        public int MSPIFirstIndex { get; set; }
        
        /// <summary>
        /// MSPI index count - byte at offset 0x0b
        /// </summary>
        public byte MSPIIndexCount { get; set; }
        
        /// <summary>
        /// UInt32 at offset 0x0c - always 0xffffffff in version_48
        /// </summary>
        public uint Value2 { get; set; }
        
        /// <summary>
        /// UInt16 at offset 0x10
        /// </summary>
        public ushort Value3 { get; set; }
        
        /// <summary>
        /// UInt16 at offset 0x12 - always 0x8000 in version_48
        /// </summary>
        public ushort Value4 { get; set; }
    }
    
    /// <summary>
    /// KLSM chunk (MSLK reversed)
    /// </summary>
    public class KLSM : BaseMeshChunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public override string Signature => "KLSM";
        
        /// <summary>
        /// Entries in the KLSM chunk.
        /// </summary>
        public List<KLSMEntry> Entries { get; } = new List<KLSMEntry>();
        
        /// <summary>
        /// Initializes a new instance of the <see cref="KLSM"/> class.
        /// </summary>
        public KLSM()
        {
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="KLSM"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public KLSM(BinaryReader reader)
        {
            ReadBinary(reader);
        }
        
        /// <summary>
        /// Parses the KLSM data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        protected override void ParseData(BinaryReader reader)
        {
            Entries.Clear();
            
            // Each entry is 20 bytes according to documentation
            int bytesAvailable = (int)reader.BaseStream.Length;
            int entryCount = bytesAvailable / 20;
            
            for (int i = 0; i < entryCount; i++)
            {
                var entry = new KLSMEntry
                {
                    Flags = reader.ReadByte(),
                    Value1 = reader.ReadByte(),
                    Padding = reader.ReadUInt16(),
                    Index = reader.ReadUInt32(),
                    
                    // Read 3 bytes as int24 for MSPI_first_index
                    MSPIFirstIndex = reader.ReadByte() | (reader.ReadByte() << 8) | (reader.ReadByte() << 16),
                    MSPIIndexCount = reader.ReadByte(),
                    
                    Value2 = reader.ReadUInt32(),
                    Value3 = reader.ReadUInt16(),
                    Value4 = reader.ReadUInt16()
                };
                
                // Sign-extend the 24-bit value to 32-bit if highest bit is set
                if ((entry.MSPIFirstIndex & 0x800000) != 0)
                {
                    entry.MSPIFirstIndex |= unchecked((int)0xFF000000);
                }
                
                // Documentation says MSPIFirstIndex is -1 if MSPIIndexCount (at 0x0b) is 0
                if (entry.MSPIIndexCount == 0)
                {
                    entry.MSPIFirstIndex = -1;
                }
                
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
                return $"KLSM chunk: {DataSize} bytes (unparsed)";
            }
            
            var sb = new StringBuilder();
            sb.AppendLine($"KLSM entry count: {Entries.Count}");
            
            // Display the first few entries
            int displayCount = Math.Min(Entries.Count, 10);
            for (int i = 0; i < displayCount; i++)
            {
                var entry = Entries[i];
                sb.AppendLine($"  Entry {i}:");
                sb.AppendLine($"    Flags: {entry.Flags}, Value1: {entry.Value1}, Padding: 0x{entry.Padding:X4}");
                sb.AppendLine($"    Index: {entry.Index}");
                sb.AppendLine($"    MSPI FirstIndex: {entry.MSPIFirstIndex}, IndexCount: {entry.MSPIIndexCount}");
                sb.AppendLine($"    Value2: 0x{entry.Value2:X8}, Value3: 0x{entry.Value3:X4}, Value4: 0x{entry.Value4:X4}");
            }
            
            if (Entries.Count > displayCount)
            {
                sb.AppendLine($"  ... and {Entries.Count - displayCount} more entries");
            }
            
            return sb.ToString();
        }
    }
} 