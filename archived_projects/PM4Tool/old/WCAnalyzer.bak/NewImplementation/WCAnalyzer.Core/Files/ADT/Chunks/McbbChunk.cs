using System;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MCBB chunk - Collision bounding box data (Mists of Pandaria+)
    /// Based on the name, it likely defines collision boundaries for map chunks
    /// </summary>
    public class McbbChunk : BaseChunk
    {
        /// <summary>
        /// The signature of the MCBB chunk
        /// </summary>
        public const string SIGNATURE = "MCBB";

        /// <summary>
        /// Gets the raw bounding box data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="McbbChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public McbbChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            RawData = Array.Empty<byte>();
            Parse();
        }

        /// <summary>
        /// Parses the MCBB chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MCBB chunk has no data");
                    return;
                }
                
                // Store the raw data for later analysis
                RawData = new byte[Data.Length];
                Array.Copy(Data, RawData, Data.Length);
                
                Logger?.LogDebug($"MCBB: Read {Data.Length} bytes of collision bounding box data");
                LogDataSummary();
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MCBB chunk: {ex.Message}");
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
                Logger?.LogDebug($"MCBB: Data size: {RawData.Length} bytes");
                
                // If this is a bounding box, common sizes would be 24 bytes (6 floats) or 32 bytes (8 floats)
                if (RawData.Length == 24)
                {
                    Logger?.LogDebug("MCBB: Data size is 24 bytes, potentially a bounding box (min/max points as 3 floats each)");
                    
                    ms.Position = 0;
                    float minX = reader.ReadSingle();
                    float minY = reader.ReadSingle();
                    float minZ = reader.ReadSingle();
                    float maxX = reader.ReadSingle();
                    float maxY = reader.ReadSingle();
                    float maxZ = reader.ReadSingle();
                    
                    Logger?.LogDebug($"MCBB: Potential bounding box: Min({minX}, {minY}, {minZ}), Max({maxX}, {maxY}, {maxZ})");
                }
                else if (RawData.Length == 32)
                {
                    Logger?.LogDebug("MCBB: Data size is 32 bytes, potentially a bounding box with additional data");
                }
                
                // Check if data size might represent an array of common data structures
                if (RawData.Length % 4 == 0)
                    Logger?.LogDebug($"MCBB: Data size is divisible by 4 ({RawData.Length / 4} elements if 4-byte values)");
                if (RawData.Length % 8 == 0)
                    Logger?.LogDebug($"MCBB: Data size is divisible by 8 ({RawData.Length / 8} elements if 8-byte values)");
                if (RawData.Length % 12 == 0)
                    Logger?.LogDebug($"MCBB: Data size is divisible by 12 ({RawData.Length / 12} elements if Vector3)");
                if (RawData.Length % 16 == 0)
                    Logger?.LogDebug($"MCBB: Data size is divisible by 16 ({RawData.Length / 16} elements if Vector4)");
                
                // Log the first few bytes as different numeric types to help with analysis
                if (RawData.Length >= 16)
                {
                    ms.Position = 0;
                    Logger?.LogDebug("MCBB: First 4 floats:");
                    for (int i = 0; i < 4 && ms.Position + 4 <= RawData.Length; i++)
                    {
                        float value = reader.ReadSingle();
                        Logger?.LogDebug($"  [{i}] = {value}");
                    }
                }
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error logging MCBB data summary: {ex.Message}");
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
                Logger?.LogError("Cannot write MCBB chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                writer.Write(RawData.Length);
                writer.Write(RawData);
                
                Logger?.LogDebug($"MCBB: Wrote {RawData.Length} bytes of collision bounding box data");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MCBB chunk: {ex.Message}");
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
        /// Attempts to interpret data as a bounding box if it's 24 bytes (6 floats)
        /// </summary>
        /// <returns>A tuple with min/max points if the data matches the expected format, null otherwise</returns>
        public (float MinX, float MinY, float MinZ, float MaxX, float MaxY, float MaxZ)? TryGetAsBoundingBox()
        {
            if (RawData.Length != 24)
                return null;
                
            try
            {
                using var ms = new MemoryStream(RawData);
                using var reader = new BinaryReader(ms);
                
                float minX = reader.ReadSingle();
                float minY = reader.ReadSingle();
                float minZ = reader.ReadSingle();
                float maxX = reader.ReadSingle();
                float maxY = reader.ReadSingle();
                float maxZ = reader.ReadSingle();
                
                return (minX, minY, minZ, maxX, maxY, maxZ);
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