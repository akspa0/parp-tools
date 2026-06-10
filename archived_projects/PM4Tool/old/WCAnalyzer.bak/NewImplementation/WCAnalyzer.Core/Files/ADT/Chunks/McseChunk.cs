using System;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MCSE chunk - Contains sound emitter data for a map chunk
    /// </summary>
    public class McseChunk : BaseChunk
    {
        /// <summary>
        /// The signature of the MCSE chunk
        /// </summary>
        public const string SIGNATURE = "MCSE";

        /// <summary>
        /// Gets the raw sound emitter data
        /// </summary>
        public byte[] EmittersData { get; private set; }

        /// <summary>
        /// Gets the number of sound emitters
        /// </summary>
        public uint Count { get; private set; }

        /// <summary>
        /// Gets the estimated number of emitters based on data size
        /// </summary>
        public int EstimatedEmitterCount => EmittersData.Length > 0 ? EmittersData.Length / 16 : 0; // assuming 16 bytes per emitter

        /// <summary>
        /// Initializes a new instance of the <see cref="McseChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public McseChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            EmittersData = Array.Empty<byte>();
            Parse();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="McseChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="count">Number of sound emitters expected</param>
        /// <param name="logger">Optional logger</param>
        public McseChunk(byte[] data, uint count, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Count = count;
            EmittersData = Array.Empty<byte>();
            Parse();
        }

        /// <summary>
        /// Parses the sound emitters data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MCSE chunk has no data");
                    return;
                }
                
                // Store the raw data for now
                // In a future implementation, we could parse the specific sound emitter structs
                // when more documentation becomes available
                EmittersData = new byte[Data.Length];
                Array.Copy(Data, EmittersData, Data.Length);
                
                // If Count wasn't provided in constructor, estimate based on data size
                if (Count == 0)
                {
                    // Estimate count based on data size (assuming 16 bytes per emitter)
                    Count = (uint)EstimatedEmitterCount;
                }
                
                Logger?.LogDebug($"MCSE: Stored raw data for {Count} sound emitters ({Data.Length} bytes)");
                LogDataSummary();
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MCSE chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Logs a summary of the data for debugging purposes
        /// </summary>
        private void LogDataSummary()
        {
            if (Logger == null || EmittersData.Length == 0)
                return;
                
            try
            {
                using var ms = new MemoryStream(EmittersData);
                using var reader = new BinaryReader(ms);
                
                // Log the first few bytes as different numeric types to help with analysis
                if (EmittersData.Length >= 4)
                {
                    ms.Position = 0;
                    var uint32Value = reader.ReadUInt32();
                    Logger?.LogDebug($"MCSE: First 4 bytes as UInt32: {uint32Value}");
                }
                
                // Log data size in a structured format for pattern analysis
                Logger?.LogDebug($"MCSE: Data size: {EmittersData.Length} bytes");
                
                // Check if data size might represent an array of common data structures
                if (EmittersData.Length % 4 == 0)
                    Logger?.LogDebug($"MCSE: Data size is divisible by 4 ({EmittersData.Length / 4} elements if 4-byte values)");
                if (EmittersData.Length % 8 == 0)
                    Logger?.LogDebug($"MCSE: Data size is divisible by 8 ({EmittersData.Length / 8} elements if 8-byte values)");
                if (EmittersData.Length % 16 == 0)
                    Logger?.LogDebug($"MCSE: Data size is divisible by 16 ({EmittersData.Length / 16} elements if 16-byte values)");
                
                // Check if this might contain DWORD values for the first emitter
                if (EmittersData.Length >= 16 && EmittersData.Length % 4 == 0)
                {
                    ms.Position = 0;
                    Logger?.LogDebug("MCSE: First emitter DWORD values:");
                    for (int i = 0; i < Math.Min(4, EmittersData.Length / 4); i++)
                    {
                        uint value = reader.ReadUInt32();
                        Logger?.LogDebug($"  [{i}] = {value} (0x{value:X8})");
                    }
                }
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error logging MCSE data summary: {ex.Message}");
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
                Logger?.LogError("Cannot write MCSE chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                writer.Write(EmittersData.Length);
                writer.Write(EmittersData);
                
                Logger?.LogDebug($"MCSE: Wrote {EmittersData.Length} bytes of sound emitter data");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MCSE chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Gets the raw data at the specified offset
        /// </summary>
        /// <param name="offset">The offset into the data</param>
        /// <param name="count">The number of bytes to read</param>
        /// <returns>The raw data</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if the offset or count is out of range</exception>
        public byte[] GetRawData(int offset, int count)
        {
            if (offset < 0 || offset >= EmittersData.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(offset), $"Offset must be between 0 and {EmittersData.Length - 1}");
            }
            
            if (count < 0 || offset + count > EmittersData.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(count), $"Count must be between 0 and {EmittersData.Length - offset}");
            }
            
            byte[] result = new byte[count];
            Array.Copy(EmittersData, offset, result, 0, count);
            return result;
        }
        
        /// <summary>
        /// Gets a hexadecimal representation of the data for debugging
        /// </summary>
        /// <param name="maxLength">The maximum number of bytes to include</param>
        /// <returns>A string containing the hexadecimal representation</returns>
        public string GetHexDump(int maxLength = 128)
        {
            if (EmittersData.Length == 0)
                return "[Empty]";
                
            int length = Math.Min(EmittersData.Length, maxLength);
            StringBuilder sb = new StringBuilder(length * 3);
            
            for (int i = 0; i < length; i++)
            {
                sb.Append(EmittersData[i].ToString("X2"));
                sb.Append(' ');
                
                if ((i + 1) % 16 == 0)
                    sb.Append('\n');
            }
            
            if (length < EmittersData.Length)
                sb.Append($"... ({EmittersData.Length - length} more bytes)");
                
            return sb.ToString();
        }
    }
} 