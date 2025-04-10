using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents a MtfxChunk (MTFX) chunk that contains texture file extensions data, similar to how we implemented other chunks with unknown formats
    /// This chunk complements the MtexChunk by providing additional information about texture files.
    /// </summary>
    public class MtfxChunk : ADTChunk
    {
        /// <summary>
        /// The signature for this chunk type.
        /// </summary>
        public const string SIGNATURE = "MTFX";

        /// <summary>
        /// Gets the raw texture extension data.
        /// </summary>
        public byte[] RawData { get; private set; }

        /// <summary>
        /// Gets the number of texture extensions in this chunk (estimated).
        /// </summary>
        public uint Count { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MtfxChunk"/> class.
        /// </summary>
        /// <param name="data">The raw chunk data.</param>
        /// <param name="logger">The logger instance.</param>
        public MtfxChunk(byte[] data, ILogger logger) 
            : base(SIGNATURE, data, logger)
        {
            RawData = Array.Empty<byte>();
            Parse(data);
        }

        /// <summary>
        /// Parses the raw chunk data to extract texture file extension information.
        /// </summary>
        /// <param name="data">The raw chunk data.</param>
        protected override void Parse(byte[] data)
        {
            if (data == null || data.Length == 0)
            {
                Logger.LogWarning("MtfxChunk: Empty data provided to Parse method");
                return;
            }

            try
            {
                // Store the raw data for now
                // In a future implementation, we could parse the specific texture extension format
                // when more documentation becomes available
                RawData = new byte[data.Length];
                Array.Copy(data, RawData, data.Length);

                // Calculate the count of texture extensions if the data is valid
                if (data.Length > 0 && data.Length % 4 == 0)
                {
                    // Assuming each texture extension is 4 bytes (UInt32)
                    Count = (uint)(data.Length / 4);
                    Logger.LogDebug($"MtfxChunk: Estimated {Count} texture extensions based on data size");
                    
                    // Log the first few entries as UInt32 values for analysis
                    LogEntryValues();
                }

                Logger.LogDebug($"MtfxChunk: Stored raw data for {Count} texture extensions ({data.Length} bytes)");
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, $"MtfxChunk: Error parsing chunk data: {ex.Message}");
            }
        }

        /// <summary>
        /// Logs the first few entries as UInt32 values for analysis.
        /// </summary>
        private void LogEntryValues()
        {
            if (RawData.Length < 4 || Count == 0 || Logger == null)
                return;

            try
            {
                using var ms = new MemoryStream(RawData);
                using var reader = new BinaryReader(ms);

                // Log a subset of values
                int entriesToLog = Math.Min(5, (int)Count);
                Logger.LogDebug($"MtfxChunk: First {entriesToLog} extension values:");

                for (int i = 0; i < entriesToLog; i++)
                {
                    uint value = reader.ReadUInt32();
                    Logger.LogDebug($"  Extension[{i}] = 0x{value:X8}");
                }
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, $"MtfxChunk: Error logging entry values: {ex.Message}");
            }
        }

        /// <summary>
        /// Writes the chunk data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                Logger.LogWarning("MtfxChunk: Null writer provided to Write method");
                return;
            }

            if (RawData == null || RawData.Length == 0)
            {
                Logger.LogWarning("MtfxChunk: No texture extension data to write");
                return;
            }

            try
            {
                // Write the chunk signature
                writer.Write(SignatureBytes);

                // Write the data size
                writer.Write(RawData.Length);

                // Write the raw texture extension data
                writer.Write(RawData);

                Logger.LogDebug($"MtfxChunk: Successfully wrote {Count} texture extensions ({RawData.Length} bytes)");
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, $"MtfxChunk: Error writing chunk data: {ex.Message}");
            }
        }

        /// <summary>
        /// Gets the extension value at the specified index.
        /// </summary>
        /// <param name="index">The index of the extension to retrieve.</param>
        /// <returns>The UInt32 extension value at the specified index, or 0 if the index is invalid.</returns>
        public uint GetExtensionValue(int index)
        {
            if (RawData == null || RawData.Length == 0)
            {
                Logger.LogWarning("MtfxChunk: No texture extension data available");
                return 0;
            }

            if (index < 0 || index >= Count)
            {
                Logger.LogWarning($"MtfxChunk: Invalid index {index} (valid range: 0-{Count - 1})");
                return 0;
            }

            try
            {
                using var ms = new MemoryStream(RawData);
                using var reader = new BinaryReader(ms);

                // Seek to the specific extension entry
                ms.Position = index * 4;
                return reader.ReadUInt32();
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, $"MtfxChunk: Error retrieving extension value at index {index}: {ex.Message}");
                return 0;
            }
        }

        /// <summary>
        /// Gets the raw data at the specified offset.
        /// </summary>
        /// <param name="offset">The offset into the data.</param>
        /// <param name="length">The length of data to retrieve.</param>
        /// <returns>The requested section of raw data or an empty array if parameters are invalid.</returns>
        public byte[] GetRawData(int offset, int length)
        {
            if (RawData == null || RawData.Length == 0)
            {
                Logger.LogWarning("MtfxChunk: No texture extension data available");
                return Array.Empty<byte>();
            }

            if (offset < 0 || offset >= RawData.Length)
            {
                Logger.LogWarning($"MtfxChunk: Invalid offset {offset} (valid range: 0-{RawData.Length - 1})");
                return Array.Empty<byte>();
            }

            if (length <= 0 || offset + length > RawData.Length)
            {
                Logger.LogWarning($"MtfxChunk: Invalid length {length} at offset {offset} (max available: {RawData.Length - offset})");
                return Array.Empty<byte>();
            }

            try
            {
                byte[] result = new byte[length];
                Array.Copy(RawData, offset, result, 0, length);
                return result;
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, $"MtfxChunk: Error retrieving raw data: {ex.Message}");
                return Array.Empty<byte>();
            }
        }

        /// <summary>
        /// Gets all extension values as an array of UInt32.
        /// </summary>
        /// <returns>An array of extension values, or an empty array if no data is available.</returns>
        public uint[] GetAllExtensionValues()
        {
            if (RawData == null || RawData.Length == 0 || Count == 0)
            {
                return Array.Empty<uint>();
            }

            try
            {
                using var ms = new MemoryStream(RawData);
                using var reader = new BinaryReader(ms);

                uint[] values = new uint[Count];
                for (int i = 0; i < Count; i++)
                {
                    values[i] = reader.ReadUInt32();
                }

                return values;
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, $"MtfxChunk: Error retrieving all extension values: {ex.Message}");
                return Array.Empty<uint>();
            }
        }

        /// <summary>
        /// Gets a hexadecimal string representation of the texture extension data for debugging.
        /// </summary>
        /// <param name="maxLength">The maximum number of bytes to include in the output.</param>
        /// <returns>A hexadecimal string representation of the data.</returns>
        public string GetHexDump(int maxLength = 128)
        {
            if (RawData == null || RawData.Length == 0)
            {
                return "[Empty]";
            }

            int bytesToShow = Math.Min(maxLength, RawData.Length);
            System.Text.StringBuilder sb = new System.Text.StringBuilder(bytesToShow * 3);

            for (int i = 0; i < bytesToShow; i++)
            {
                sb.Append(RawData[i].ToString("X2"));
                sb.Append(' ');
                
                if ((i + 1) % 16 == 0)
                {
                    sb.Append(Environment.NewLine);
                }
            }

            if (bytesToShow < RawData.Length)
            {
                sb.Append($"... ({RawData.Length - bytesToShow} more bytes)");
            }

            return sb.ToString();
        }
    }
} 