using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MHID chunk - Contains height texture file data IDs
    /// </summary>
    public class MhidChunk : ADTChunk
    {
        /// <summary>
        /// The MHID chunk signature
        /// </summary>
        public const string SIGNATURE = "MHID";

        /// <summary>
        /// Gets the list of height texture file data IDs
        /// </summary>
        public List<uint> HeightTextureFileDataIds { get; } = new List<uint>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MhidChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MhidChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Parse();
        }

        /// <summary>
        /// Parses the MHID chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MHID chunk has no data");
                    return;
                }
                
                using var ms = new MemoryStream(Data);
                using var reader = new BinaryReader(ms);
                
                // Validate that the data size is a multiple of 4 bytes (uint)
                if (Data.Length % 4 != 0)
                {
                    Logger?.LogWarning($"MHID chunk data size {Data.Length} is not a multiple of 4");
                }
                
                // Calculate how many file data IDs we should have
                int count = Data.Length / 4;
                
                // Read file data IDs (4 bytes each)
                for (int i = 0; i < count; i++)
                {
                    try
                    {
                        uint fileDataId = reader.ReadUInt32();
                        HeightTextureFileDataIds.Add(fileDataId);
                    }
                    catch (EndOfStreamException)
                    {
                        break;
                    }
                }
                
                Logger?.LogDebug($"MHID: Read {HeightTextureFileDataIds.Count} height texture file data IDs");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MHID chunk: {ex.Message}");
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
                Logger?.LogError("Cannot write MHID chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                writer.Write(HeightTextureFileDataIds.Count * 4); // Size in bytes (4 bytes per file data ID)
                
                // Write each file data ID
                foreach (var fileDataId in HeightTextureFileDataIds)
                {
                    writer.Write(fileDataId);
                }
                
                Logger?.LogDebug($"MHID: Wrote {HeightTextureFileDataIds.Count} height texture file data IDs");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MHID chunk: {ex.Message}");
                throw;
            }
        }
    }
} 