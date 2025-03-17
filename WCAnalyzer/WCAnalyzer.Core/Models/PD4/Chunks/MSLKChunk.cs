using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using WCAnalyzer.Core.Utilities;
using Warcraft.NET.Files.Interfaces;

namespace WCAnalyzer.Core.Models.PD4.Chunks
{
    /// <summary>
    /// MSLK chunk - Contains link data.
    /// <para>
    /// According to documentation: 
    /// struct { 
    ///     uint8_t _0x00; // earlier documentation has this as bitmask32 flagsu 
    ///     uint8_t _0x01; 
    ///     uint16_t _0x02; // Always 0 in version_48, likely padding.u 
    ///     uint32_t _0x04; // An index somewhere.u 
    ///     int24_t MSPI_first_index; // -1 if _0x0b is 0 
    ///     uint8_t MSPI_index_count; 
    ///     uint32_t _0x0c; // Always 0xffffffff in version_48.u 
    ///     uint16_t _0x10; 
    ///     uint16_t _0x12; // Always 0x8000 in version_48.u 
    /// } mslk[];
    /// </para>
    /// </summary>
    /// <remarks>
    /// The MSLK chunk is used to store links to other parts of the file, typically related to MSPI indices.
    /// Each entry in this chunk is 20 bytes (0x14) in size.
    /// </remarks>
    public class MSLKChunk : PD4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSLK";

        /// <summary>
        /// Gets the link entries.
        /// </summary>
        public List<LinkEntry> Entries { get; private set; } = new List<LinkEntry>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSLKChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
        public MSLKChunk(byte[] data) : base(data)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            // Create the list of entries if there are any in the chunk
            if (data.Length > 0)
            {
                // Each entry is 20 bytes (0x14)
                int entrySize = 20;
                
                // Check if data length is at least 4 bytes for the entry count
                if (data.Length < 4)
                {
                    throw new InvalidDataException($"MSLK chunk data is too small: {data.Length} bytes (minimum 4 bytes needed for entry count)");
                }
                
                // Read the entry count from the first 4 bytes
                byte[] lengthBytes = new byte[4];
                Array.Copy(data, 0, lengthBytes, 0, 4);
                uint entriesCount = BitConverter.ToUInt32(lengthBytes, 0);
                
                // Check for overflow and convert to int safely
                int entryCount = SafeConversions.UIntToInt(entriesCount);
                
                // Validate that the data is large enough for the number of entries
                if (data.Length < 4 + (entryCount * entrySize))
                {
                    throw new InvalidDataException($"MSLK chunk data is too small: {data.Length} bytes (need {4 + (entryCount * entrySize)} bytes for {entryCount} entries)");
                }
                
                Entries = new List<LinkEntry>(entryCount);
            }
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        /// <exception cref="InvalidDataException">Thrown when the chunk data is invalid or corrupted.</exception>
        protected override void ReadData()
        {
            Entries.Clear();

            if (Data == null || Data.Length == 0)
            {
                return; // Nothing to read
            }

            try
            {
                using (var ms = new MemoryStream(Data))
                using (var reader = new BinaryReader(ms))
                {
                    // Each entry is 20 bytes (0x14)
                    int entrySize = 20;
                    uint totalBytes = (uint)Data.Length;
                    
                    // Calculate number of entries safely
                    int entryCount = Data.Length / entrySize;
                    
                    for (int i = 0; i < entryCount; i++)
                    {
                        // Check if we have enough data remaining to read an entry
                        if (ms.Position + entrySize > ms.Length)
                        {
                            throw new InvalidDataException($"Unexpected end of data while reading MSLK entry {i}");
                        }
                        
                        var entry = new LinkEntry
                        {
                            Value0x00 = reader.ReadByte(),
                            Value0x01 = reader.ReadByte(),
                            Value0x02 = reader.ReadUInt16(),
                            Value0x04 = reader.ReadUInt32(),
                            
                            // Read and interpret the MSPI_first_index (int24) and MSPI_index_count (uint8)
                            // int24 is a 3-byte integer, we'll read 3 bytes and reconstruct it
                            MSPIFirstIndex = SafeConversions.ReadInt24(reader),
                            MSPIIndexCount = reader.ReadByte(),
                            
                            Value0x0c = reader.ReadUInt32(),
                            Value0x10 = reader.ReadUInt16(),
                            Value0x12 = reader.ReadUInt16()
                        };
                        
                        Entries.Add(entry);
                    }
                }
            }
            catch (Exception ex) when (ex is IOException || ex is ArgumentException)
            {
                throw new InvalidDataException("Error reading MSLK chunk data", ex);
            }
        }
        
        /// <summary>
        /// Represents a link entry in the MSLK chunk.
        /// </summary>
        public class LinkEntry
        {
            /// <summary>
            /// Gets or sets the value at offset 0x00 (uint8_t).
            /// Earlier documentation has this as bitmask32 flags.
            /// </summary>
            public byte Value0x00 { get; set; }
            
            /// <summary>
            /// Gets or sets the value at offset 0x01 (uint8_t).
            /// </summary>
            public byte Value0x01 { get; set; }
            
            /// <summary>
            /// Gets or sets the value at offset 0x02 (uint16_t).
            /// Always 0 in version_48, likely padding.
            /// </summary>
            public ushort Value0x02 { get; set; }
            
            /// <summary>
            /// Gets or sets the value at offset 0x04 (uint32_t).
            /// An index somewhere.
            /// </summary>
            public uint Value0x04 { get; set; }
            
            /// <summary>
            /// Gets or sets the MSPI first index (int24_t).
            /// -1 if MSPIIndexCount is 0.
            /// </summary>
            public int MSPIFirstIndex { get; set; }
            
            /// <summary>
            /// Gets or sets the MSPI index count (uint8_t).
            /// </summary>
            public byte MSPIIndexCount { get; set; }
            
            /// <summary>
            /// Gets or sets the value at offset 0x0c (uint32_t).
            /// Always 0xffffffff in version_48.
            /// </summary>
            public uint Value0x0c { get; set; }
            
            /// <summary>
            /// Gets or sets the value at offset 0x10 (uint16_t).
            /// </summary>
            public ushort Value0x10 { get; set; }
            
            /// <summary>
            /// Gets or sets the value at offset 0x12 (uint16_t).
            /// Always 0x8000 in version_48.
            /// </summary>
            public ushort Value0x12 { get; set; }
            
            /// <summary>
            /// Returns a string that represents the current object.
            /// </summary>
            /// <returns>A string that represents the current object.</returns>
            public override string ToString()
            {
                return $"MSPI First Index: {MSPIFirstIndex}, Count: {MSPIIndexCount}, ID: {Value0x04:X8}";
            }
        }
    }
} 