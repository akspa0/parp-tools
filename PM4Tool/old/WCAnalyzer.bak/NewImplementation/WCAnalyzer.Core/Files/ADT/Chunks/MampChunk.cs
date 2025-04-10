using System;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MAMP chunk - Contains map objects data for ADT files (Cataclysm+)
    /// </summary>
    public class MampChunk : BaseChunk
    {
        /// <summary>
        /// The signature of the MAMP chunk
        /// </summary>
        public const string SIGNATURE = "MAMP";

        /// <summary>
        /// Gets the raw map objects data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MampChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MampChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            RawData = Array.Empty<byte>();
            Parse();
        }

        /// <summary>
        /// Parses the MAMP chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MAMP chunk has no data");
                    return;
                }
                
                // Store the raw data for later analysis
                RawData = new byte[Data.Length];
                Array.Copy(Data, RawData, Data.Length);
                
                Logger?.LogDebug($"MAMP: Read {Data.Length} bytes of map objects data");
                LogDataSummary();
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MAMP chunk: {ex.Message}");
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
                
                // Log data size
                Logger?.LogDebug($"MAMP: Data size: {RawData.Length} bytes");
                
                // Check for common data pattern
                if (RawData.Length == 1)
                {
                    Logger?.LogDebug($"MAMP: Contains a single byte value: {RawData[0]} (0x{RawData[0]:X2})");
                }
                else if (RawData.Length == 4)
                {
                    ms.Position = 0;
                    uint uint32Value = reader.ReadUInt32();
                    Logger?.LogDebug($"MAMP: Contains a UInt32 value: {uint32Value} (0x{uint32Value:X8})");
                    
                    ms.Position = 0;
                    int int32Value = reader.ReadInt32();
                    Logger?.LogDebug($"MAMP: As Int32: {int32Value}");
                    
                    ms.Position = 0;
                    float floatValue = reader.ReadSingle();
                    Logger?.LogDebug($"MAMP: As Float: {floatValue}");
                }
                
                // Check if data size might represent an array of common data structures
                if (RawData.Length % 4 == 0 && RawData.Length > 4)
                {
                    Logger?.LogDebug($"MAMP: Data size is divisible by 4 ({RawData.Length / 4} elements if 4-byte values)");
                    
                    // Print first few elements as various types for analysis
                    ms.Position = 0;
                    Logger?.LogDebug("MAMP: First few DWORD values:");
                    for (int i = 0; i < Math.Min(4, RawData.Length / 4); i++)
                    {
                        uint value = reader.ReadUInt32();
                        Logger?.LogDebug($"  [{i}] = {value} (0x{value:X8})");
                    }
                    
                    ms.Position = 0;
                    Logger?.LogDebug("MAMP: First few float values:");
                    for (int i = 0; i < Math.Min(4, RawData.Length / 4); i++)
                    {
                        float value = reader.ReadSingle();
                        Logger?.LogDebug($"  [{i}] = {value}");
                    }
                }
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error logging MAMP data summary: {ex.Message}");
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
                Logger?.LogError("Cannot write MAMP chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                writer.Write(RawData.Length);
                writer.Write(RawData);
                
                Logger?.LogDebug($"MAMP: Wrote {RawData.Length} bytes of map objects data");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MAMP chunk: {ex.Message}");
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
        /// Attempts to get the MAMP value as a single byte
        /// </summary>
        /// <returns>The byte value, or null if the data is not a single byte</returns>
        public byte? TryGetAsByte()
        {
            if (RawData.Length != 1)
                return null;
                
            return RawData[0];
        }
        
        /// <summary>
        /// Attempts to read 32-bit value if the data is exactly 4 bytes
        /// </summary>
        /// <returns>The uint32 value, or null if the data is not 4 bytes</returns>
        public uint? TryGetAsUInt32()
        {
            if (RawData.Length != 4)
                return null;
                
            try
            {
                using var ms = new MemoryStream(RawData);
                using var reader = new BinaryReader(ms);
                
                return reader.ReadUInt32();
            }
            catch
            {
                return null;
            }
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
        /// Attempts to read single-precision floating point values if the data appears to be an array of floats
        /// </summary>
        /// <returns>An array of float values, or null if the data doesn't appear to be float values</returns>
        public float[] TryGetAsFloatArray()
        {
            if (RawData.Length % 4 != 0 || RawData.Length == 0)
                return null;
                
            try
            {
                using var ms = new MemoryStream(RawData);
                using var reader = new BinaryReader(ms);
                
                float[] values = new float[RawData.Length / 4];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = reader.ReadSingle();
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