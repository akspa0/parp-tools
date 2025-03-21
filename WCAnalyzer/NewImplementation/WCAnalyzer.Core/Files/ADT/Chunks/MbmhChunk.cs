using System;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MBMH chunk - Contains blend map header data for ADT files (Cataclysm+)
    /// </summary>
    public class MbmhChunk : BaseChunk
    {
        /// <summary>
        /// The signature of the MBMH chunk
        /// </summary>
        public const string SIGNATURE = "MBMH";

        /// <summary>
        /// Gets the raw blend map header data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MbmhChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MbmhChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            RawData = Array.Empty<byte>();
            Parse();
        }

        /// <summary>
        /// Parses the MBMH chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MBMH chunk has no data");
                    return;
                }
                
                // Store the raw data for later analysis
                RawData = new byte[Data.Length];
                Array.Copy(Data, RawData, Data.Length);
                
                Logger?.LogDebug($"MBMH: Read {Data.Length} bytes of blend map header data");
                LogDataSummary();
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MBMH chunk: {ex.Message}");
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
                Logger?.LogDebug($"MBMH: Data size: {RawData.Length} bytes");
                
                // Check if data size is large enough to be a header
                if (RawData.Length >= 20) // Approximate size based on common header structures
                {
                    ms.Position = 0;
                    uint flags = reader.ReadUInt32();
                    Logger?.LogDebug($"MBMH: Possible flags: 0x{flags:X8}");
                    
                    uint width = reader.ReadUInt32();
                    uint height = reader.ReadUInt32();
                    Logger?.LogDebug($"MBMH: Possible dimensions: {width}x{height}");
                    
                    uint layers = reader.ReadUInt32();
                    uint unk1 = reader.ReadUInt32();
                    Logger?.LogDebug($"MBMH: Possible layers: {layers}, Unknown: 0x{unk1:X8}");
                }
                
                // Check if data size might represent an array of common data structures
                if (RawData.Length % 4 == 0 && RawData.Length > 4)
                {
                    Logger?.LogDebug($"MBMH: Data size is divisible by 4 ({RawData.Length / 4} elements if 4-byte values)");
                    
                    // Print first few elements as various types for analysis
                    ms.Position = 0;
                    Logger?.LogDebug("MBMH: First few DWORD values:");
                    for (int i = 0; i < Math.Min(8, RawData.Length / 4); i++)
                    {
                        uint value = reader.ReadUInt32();
                        Logger?.LogDebug($"  [{i}] = {value} (0x{value:X8})");
                    }
                    
                    ms.Position = 0;
                    Logger?.LogDebug("MBMH: First few float values:");
                    for (int i = 0; i < Math.Min(8, RawData.Length / 4); i++)
                    {
                        float value = reader.ReadSingle();
                        Logger?.LogDebug($"  [{i}] = {value}");
                    }
                }
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error logging MBMH data summary: {ex.Message}");
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
                Logger?.LogError("Cannot write MBMH chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                writer.Write(RawData.Length);
                writer.Write(RawData);
                
                Logger?.LogDebug($"MBMH: Wrote {RawData.Length} bytes of blend map header data");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MBMH chunk: {ex.Message}");
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
        /// Attempts to read the blend map header structure, if the data matches expected format
        /// </summary>
        /// <returns>An array of DWORD values representing the header, or null if format doesn't match</returns>
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