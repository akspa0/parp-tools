using System;
using System.IO;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MCRW chunk - Contains render water information for a map chunk
    /// </summary>
    public class McrwChunk : ADTChunk
    {
        /// <summary>
        /// The MCRW chunk signature
        /// </summary>
        public const string SIGNATURE = "MCRW";

        /// <summary>
        /// Gets or sets the water render flags
        /// </summary>
        public uint Flags { get; set; }

        /// <summary>
        /// Gets the raw data for the chunk
        /// </summary>
        public byte[] RawData { get; private set; } = Array.Empty<byte>();

        /// <summary>
        /// Initializes a new instance of the <see cref="McrwChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public McrwChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Parse();
        }

        /// <summary>
        /// Parses the MCRW chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MCRW chunk has no data");
                    return;
                }

                // Store the raw data for later analysis
                RawData = new byte[Data.Length];
                Array.Copy(Data, RawData, Data.Length);
                
                using var ms = new MemoryStream(Data);
                using var reader = new BinaryReader(ms);
                
                // Read flags
                if (Data.Length >= 4)
                {
                    Flags = reader.ReadUInt32();
                    Logger?.LogDebug($"MCRW: Read flags: 0x{Flags:X8}");
                }
                else
                {
                    Logger?.LogWarning($"MCRW: Chunk data size {Data.Length} is too small for flags (expected at least 4 bytes)");
                }
                
                // Log extra data if present
                if (Data.Length > 4)
                {
                    Logger?.LogDebug($"MCRW: Chunk contains {Data.Length - 4} bytes of additional data");
                }
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MCRW chunk: {ex.Message}");
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
                Logger?.LogError("Cannot write MCRW chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                
                // If we have raw data, write it
                if (RawData.Length > 0)
                {
                    writer.Write(RawData.Length);
                    writer.Write(RawData);
                    Logger?.LogDebug($"MCRW: Wrote {RawData.Length} bytes from raw data");
                }
                else
                {
                    // Otherwise write the flags
                    writer.Write(4); // Size in bytes
                    writer.Write(Flags);
                    Logger?.LogDebug($"MCRW: Wrote flags: 0x{Flags:X8}");
                }
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MCRW chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Gets a hexadecimal string representation of the raw data
        /// </summary>
        /// <returns>A string containing the hexadecimal representation</returns>
        public string ToHexString()
        {
            if (RawData == null || RawData.Length == 0)
            {
                return string.Empty;
            }
            
            return BitConverter.ToString(RawData).Replace("-", " ");
        }
    }
} 