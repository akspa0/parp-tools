using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MDID chunk - Contains file data IDs for doodad objects
    /// </summary>
    public class MdidChunk : ADTChunk
    {
        /// <summary>
        /// The MDID chunk signature
        /// </summary>
        public const string SIGNATURE = "MDID";

        /// <summary>
        /// Gets the file data IDs for doodad objects
        /// </summary>
        public List<uint> DoodadFileDataIds { get; private set; } = new List<uint>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MdidChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MdidChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Parse();
        }

        /// <summary>
        /// Parses the MDID chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MDID chunk has no data");
                    return;
                }
                
                using var ms = new MemoryStream(Data);
                using var reader = new BinaryReader(ms);
                
                // Validate that the data size is a multiple of 4 bytes (uint)
                if (Data.Length % 4 != 0)
                {
                    Logger?.LogWarning($"MDID chunk data size {Data.Length} is not a multiple of 4");
                }
                
                // Calculate the number of IDs
                int count = Data.Length / 4;
                
                // Read doodad file data IDs
                for (int i = 0; i < count; i++)
                {
                    try
                    {
                        uint fileDataId = reader.ReadUInt32();
                        DoodadFileDataIds.Add(fileDataId);
                    }
                    catch (EndOfStreamException)
                    {
                        Logger?.LogWarning($"MDID: Unexpected end of stream while reading doodad file data ID {i}");
                        break;
                    }
                }
                
                Logger?.LogDebug($"MDID: Read {DoodadFileDataIds.Count} doodad file data IDs");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MDID chunk: {ex.Message}");
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
                Logger?.LogError("Cannot write MDID chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                
                // Calculate the data size (4 bytes per file data ID)
                int dataSize = DoodadFileDataIds.Count * 4;
                writer.Write(dataSize);
                
                // Write each doodad file data ID
                foreach (var fileDataId in DoodadFileDataIds)
                {
                    writer.Write(fileDataId);
                }
                
                Logger?.LogDebug($"MDID: Wrote {DoodadFileDataIds.Count} doodad file data IDs");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MDID chunk: {ex.Message}");
                throw;
            }
        }
    }
} 