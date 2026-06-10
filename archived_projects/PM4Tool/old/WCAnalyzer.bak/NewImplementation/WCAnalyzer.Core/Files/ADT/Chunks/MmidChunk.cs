using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MMID chunk - Contains offsets to model filenames in the MMDX chunk
    /// </summary>
    public class MmidChunk : ADTChunk
    {
        /// <summary>
        /// The MMID chunk signature
        /// </summary>
        public const string SIGNATURE = "MMID";

        /// <summary>
        /// Gets the list of offsets
        /// </summary>
        public List<uint> Offsets { get; } = new List<uint>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MmidChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MmidChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Parse();
        }

        /// <summary>
        /// Parses the MMID chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MMID chunk has no data");
                    return;
                }
                
                using var ms = new MemoryStream(Data);
                using var reader = new BinaryReader(ms);
                
                // Validate that the data size is a multiple of 4 bytes (uint)
                if (Data.Length % 4 != 0)
                {
                    Logger?.LogWarning($"MMID chunk data size {Data.Length} is not a multiple of 4");
                }
                
                // Read offsets (multiples of 4 bytes)
                while (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    try
                    {
                        uint offset = reader.ReadUInt32();
                        Offsets.Add(offset);
                    }
                    catch (EndOfStreamException)
                    {
                        break;
                    }
                }
                
                Logger?.LogDebug($"MMID: Read {Offsets.Count} model offsets");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MMID chunk: {ex.Message}");
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
                Logger?.LogError("Cannot write MMID chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                writer.Write(Offsets.Count * 4); // Size in bytes (4 bytes per offset)
                
                // Write each offset
                foreach (var offset in Offsets)
                {
                    writer.Write(offset);
                }
                
                Logger?.LogDebug($"MMID: Wrote {Offsets.Count} model offsets");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MMID chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Gets a model filename from the MMDX chunk using the offset at the specified index
        /// </summary>
        /// <param name="index">The index into the offsets list</param>
        /// <param name="mmdxChunk">The MMDX chunk containing the model filenames</param>
        /// <returns>The model filename, or null if the index or offset is invalid</returns>
        public string GetModelFilename(int index, MmdxChunk mmdxChunk)
        {
            if (index < 0 || index >= Offsets.Count || mmdxChunk == null)
            {
                return null;
            }
            
            uint offset = Offsets[index];
            return mmdxChunk.GetFilenameAtOffset((int)offset);
        }
    }
} 