using System;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MCRD chunk - Contains terrain hole data for a map chunk (Cataclysm+)
    /// </summary>
    public class McrdChunk : BaseChunk
    {
        /// <summary>
        /// The signature of the MCRD chunk
        /// </summary>
        public const string SIGNATURE = "MCRD";

        /// <summary>
        /// Gets the raw hole data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="McrdChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public McrdChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            RawData = Array.Empty<byte>();
            Parse();
        }

        /// <summary>
        /// Parses the MCRD chunk data to extract terrain hole information
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MCRD chunk has no data");
                    return;
                }
                
                // Store the raw data for later analysis
                RawData = new byte[Data.Length];
                Array.Copy(Data, RawData, Data.Length);
                
                Logger?.LogDebug($"MCRD: Read {Data.Length} bytes of terrain hole data");
                LogDataSummary();
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MCRD chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Logs a summary of the data for debugging purposes
        /// </summary>
        private void LogDataSummary()
        {
            if (Logger == null || RawData.Length == 0)
                return;
                
            try
            {
                using var ms = new MemoryStream(RawData);
                using var reader = new BinaryReader(ms);
                
                // Log the first few bytes as different numeric types to help with analysis
                if (RawData.Length >= 4)
                {
                    ms.Position = 0;
                    var uint32Value = reader.ReadUInt32();
                    Logger?.LogDebug($"MCRD: First 4 bytes as UInt32: {uint32Value}");
                }
                
                // Log data size in a structured format for pattern analysis
                Logger?.LogDebug($"MCRD: Data size: {RawData.Length} bytes");
                
                // Check if data size might represent an array of common data structures
                if (RawData.Length % 4 == 0)
                    Logger?.LogDebug($"MCRD: Data size is divisible by 4 ({RawData.Length / 4} elements if 4-byte values)");
                if (RawData.Length % 8 == 0)
                    Logger?.LogDebug($"MCRD: Data size is divisible by 8 ({RawData.Length / 8} elements if 8-byte values)");
                
                // Check if this might contain DWORD values
                if (RawData.Length >= 8 && RawData.Length % 4 == 0)
                {
                    ms.Position = 0;
                    Logger?.LogDebug("MCRD: First few DWORD values:");
                    for (int i = 0; i < Math.Min(8, RawData.Length / 4); i++)
                    {
                        uint value = reader.ReadUInt32();
                        Logger?.LogDebug($"  [{i}] = {value} (0x{value:X8})");
                    }
                }
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error logging MCRD data summary: {ex.Message}");
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
                Logger?.LogError("Cannot write MCRD chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                writer.Write(RawData.Length);
                writer.Write(RawData);
                
                Logger?.LogDebug($"MCRD: Wrote {RawData.Length} bytes of terrain hole data");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MCRD chunk: {ex.Message}");
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
            if (offset < 0 || offset >= RawData.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(offset), $"Offset must be between 0 and {RawData.Length - 1}");
            }
            
            if (count < 0 || offset + count > RawData.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(count), $"Count must be between 0 and {RawData.Length - offset}");
            }
            
            byte[] result = new byte[count];
            Array.Copy(RawData, offset, result, 0, count);
            return result;
        }
        
        /// <summary>
        /// Attempts to read 32-bit values if the data appears to be an array of DWORD values
        /// </summary>
        /// <returns>An array of DWORD values, or null if the data doesn't appear to be DWORD values</returns>
        public uint[] TryGetAsDwordArray()
        {
            if (RawData.Length % 4 != 0 || RawData.Length == 0)
                return null;
                
            try
            {
                using var ms = new MemoryStream(RawData);
                using var reader = new BinaryReader(ms);
                
                uint[] values = new uint[RawData.Length / 4];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = reader.ReadUInt32();
                }
                
                return values;
            }
            catch
            {
                return null;
            }
        }
        
        /// <summary>
        /// Gets a hexadecimal representation of the data for debugging
        /// </summary>
        /// <param name="maxLength">The maximum number of bytes to include</param>
        /// <returns>A string containing the hexadecimal representation</returns>
        public string GetHexDump(int maxLength = 128)
        {
            if (RawData.Length == 0)
                return "[Empty]";
                
            int length = Math.Min(RawData.Length, maxLength);
            StringBuilder sb = new StringBuilder(length * 3);
            
            for (int i = 0; i < length; i++)
            {
                sb.Append(RawData[i].ToString("X2"));
                sb.Append(' ');
                
                if ((i + 1) % 16 == 0)
                    sb.Append('\n');
            }
            
            if (length < RawData.Length)
                sb.Append($"... ({RawData.Length - length} more bytes)");
                
            return sb.ToString();
        }
    }
} 