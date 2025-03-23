using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MCMT chunk - Contains material information for terrain textures
    /// </summary>
    public class McmtChunk : ADTChunk
    {
        /// <summary>
        /// The MCMT chunk signature
        /// </summary>
        public const string SIGNATURE = "MCMT";

        /// <summary>
        /// Represents a terrain material entry
        /// </summary>
        public class MaterialEntry
        {
            /// <summary>
            /// Gets or sets the material ID
            /// </summary>
            public uint MaterialId { get; set; }

            /// <summary>
            /// Gets or sets the material flags
            /// </summary>
            public uint Flags { get; set; }
        }

        /// <summary>
        /// Gets the list of material entries
        /// </summary>
        public List<MaterialEntry> Materials { get; private set; } = new List<MaterialEntry>();

        /// <summary>
        /// Initializes a new instance of the <see cref="McmtChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public McmtChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Parse();
        }

        /// <summary>
        /// Parses the MCMT chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MCMT chunk has no data");
                    return;
                }
                
                using var ms = new MemoryStream(Data);
                using var reader = new BinaryReader(ms);
                
                // Each material entry is 8 bytes (4 for ID, 4 for flags)
                const int EntrySize = 8;
                
                // Validate that the data size is a multiple of entry size
                if (Data.Length % EntrySize != 0)
                {
                    Logger?.LogWarning($"MCMT chunk data size {Data.Length} is not a multiple of {EntrySize}");
                }
                
                // Calculate the number of entries
                int count = Data.Length / EntrySize;
                
                // Read material entries
                for (int i = 0; i < count; i++)
                {
                    try
                    {
                        var entry = new MaterialEntry
                        {
                            MaterialId = reader.ReadUInt32(),
                            Flags = reader.ReadUInt32()
                        };
                        
                        Materials.Add(entry);
                    }
                    catch (EndOfStreamException)
                    {
                        Logger?.LogWarning($"MCMT: Unexpected end of stream while reading material entry {i}");
                        break;
                    }
                }
                
                Logger?.LogDebug($"MCMT: Read {Materials.Count} material entries");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MCMT chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Writes the chunk data to a binary writer
        /// </summary>
        /// <param name="writer">The binary writer to write to</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                Logger?.LogError("Cannot write MCMT chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                
                // Calculate the data size (8 bytes per material entry)
                int dataSize = Materials.Count * 8;
                writer.Write(dataSize);
                
                // Write each material entry
                foreach (var material in Materials)
                {
                    writer.Write(material.MaterialId);
                    writer.Write(material.Flags);
                }
                
                Logger?.LogDebug($"MCMT: Wrote {Materials.Count} material entries");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MCMT chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Gets the material entry at the specified index
        /// </summary>
        /// <param name="index">The material index</param>
        /// <returns>The material entry, or null if the index is invalid</returns>
        public MaterialEntry GetMaterial(int index)
        {
            if (index < 0 || index >= Materials.Count)
            {
                return null;
            }
            
            return Materials[index];
        }
    }
} 