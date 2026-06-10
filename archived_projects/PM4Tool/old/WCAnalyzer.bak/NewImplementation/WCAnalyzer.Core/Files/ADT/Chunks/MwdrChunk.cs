using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MWDR chunk - Contains references to WMO doodads
    /// </summary>
    public class MwdrChunk : ADTChunk
    {
        /// <summary>
        /// The MWDR chunk signature
        /// </summary>
        public const string SIGNATURE = "MWDR";

        /// <summary>
        /// Gets the doodad references (typically indices into a doodad set)
        /// </summary>
        public List<uint> DoodadReferences { get; private set; } = new List<uint>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MwdrChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MwdrChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Parse();
        }

        /// <summary>
        /// Parses the MWDR chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MWDR chunk has no data");
                    return;
                }
                
                using var ms = new MemoryStream(Data);
                using var reader = new BinaryReader(ms);
                
                // Validate that the data size is a multiple of 4 bytes (uint)
                if (Data.Length % 4 != 0)
                {
                    Logger?.LogWarning($"MWDR chunk data size {Data.Length} is not a multiple of 4");
                }
                
                // Calculate the number of references
                int count = Data.Length / 4;
                
                // Read doodad references
                for (int i = 0; i < count; i++)
                {
                    try
                    {
                        uint reference = reader.ReadUInt32();
                        DoodadReferences.Add(reference);
                    }
                    catch (EndOfStreamException)
                    {
                        Logger?.LogWarning($"MWDR: Unexpected end of stream while reading doodad reference {i}");
                        break;
                    }
                }
                
                Logger?.LogDebug($"MWDR: Read {DoodadReferences.Count} WMO doodad references");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MWDR chunk: {ex.Message}");
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
                Logger?.LogError("Cannot write MWDR chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                
                // Calculate the data size (4 bytes per reference)
                int dataSize = DoodadReferences.Count * 4;
                writer.Write(dataSize);
                
                // Write each doodad reference
                foreach (var reference in DoodadReferences)
                {
                    writer.Write(reference);
                }
                
                Logger?.LogDebug($"MWDR: Wrote {DoodadReferences.Count} WMO doodad references");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MWDR chunk: {ex.Message}");
                throw;
            }
        }
    }
} 