using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MBBB chunk - Bounding box data for ADT files (Mists of Pandaria+)
    /// This may be related to bounding boxes for specific map objects or regions
    /// </summary>
    public class MbbbChunk : BaseChunk
    {
        /// <summary>
        /// The signature of the MBBB chunk
        /// </summary>
        public const string SIGNATURE = "MBBB";

        /// <summary>
        /// Gets the raw bounding box data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MbbbChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MbbbChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            RawData = Array.Empty<byte>();
            Parse();
        }

        /// <summary>
        /// Parses the MBBB chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MBBB chunk has no data");
                    return;
                }
                
                // Store the raw data for later analysis
                RawData = new byte[Data.Length];
                Array.Copy(Data, RawData, Data.Length);
                
                Logger?.LogDebug($"MBBB: Read {Data.Length} bytes of bounding box data");
                LogDataSummary();
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MBBB chunk: {ex.Message}");
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
                Logger?.LogDebug($"MBBB: Data size: {RawData.Length} bytes");
                
                // Check if data size is divisible by common sizes to detect arrays
                if (RawData.Length % 4 == 0)
                {
                    Logger?.LogDebug($"MBBB: Data size is divisible by 4 (possible array of DWORDs or floats)");
                    
                    // If divisible by 4, try to interpret as an array of 32-bit values
                    float[] floatValues = TryGetAsFloatArray();
                    if (floatValues != null)
                    {
                        int valuesToLog = Math.Min(8, floatValues.Length);
                        StringBuilder sb = new StringBuilder("MBBB: First few float values: ");
                        for (int i = 0; i < valuesToLog; i++)
                        {
                            sb.Append(floatValues[i].ToString("F4"));
                            if (i < valuesToLog - 1)
                                sb.Append(", ");
                        }
                        Logger?.LogDebug(sb.ToString());
                        
                        // Check if this might be a bounding box (6 floats: min x,y,z, max x,y,z)
                        if (floatValues.Length == 6)
                        {
                            Logger?.LogDebug($"MBBB: Data matches pattern for bounding box (6 floats): " +
                                $"Min({floatValues[0]:F2}, {floatValues[1]:F2}, {floatValues[2]:F2}), " +
                                $"Max({floatValues[3]:F2}, {floatValues[4]:F2}, {floatValues[5]:F2})");
                        }
                        
                        // Check if this might be a collection of Vector3 coordinates
                        if (floatValues.Length % 3 == 0)
                        {
                            Logger?.LogDebug($"MBBB: Data size suggests it may contain {floatValues.Length / 3} Vector3 values");
                            
                            // Log the first few Vector3 values
                            int vectorsToLog = Math.Min(3, floatValues.Length / 3);
                            StringBuilder vectorSb = new StringBuilder("MBBB: First few Vector3 values: ");
                            for (int i = 0; i < vectorsToLog; i++)
                            {
                                int baseIndex = i * 3;
                                vectorSb.Append($"({floatValues[baseIndex]:F2}, {floatValues[baseIndex+1]:F2}, {floatValues[baseIndex+2]:F2})");
                                if (i < vectorsToLog - 1)
                                    vectorSb.Append(", ");
                            }
                            Logger?.LogDebug(vectorSb.ToString());
                        }
                    }
                }
                
                // Check for other common data patterns
                if (RawData.Length % 12 == 0)
                {
                    Logger?.LogDebug($"MBBB: Data size is divisible by 12 (possible array of Vector3s or 3 floats per entry)");
                }
                
                if (RawData.Length % 24 == 0)
                {
                    Logger?.LogDebug($"MBBB: Data size is divisible by 24 (possible array of Quads or 6 floats per entry)");
                }
                
                // Log the first few bytes of the data for debugging
                if (RawData.Length >= 16)
                {
                    StringBuilder hexSb = new StringBuilder("MBBB: First 16 bytes: ");
                    for (int i = 0; i < Math.Min(16, RawData.Length); i++)
                    {
                        hexSb.Append(RawData[i].ToString("X2"));
                        if (i < Math.Min(16, RawData.Length) - 1)
                            hexSb.Append(" ");
                    }
                    Logger?.LogDebug(hexSb.ToString());
                }
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error logging MBBB data summary: {ex.Message}");
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
                Logger?.LogError("Cannot write MBBB chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                writer.Write(RawData.Length);
                writer.Write(RawData);
                
                Logger?.LogDebug($"MBBB: Wrote {RawData.Length} bytes of bounding box data");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MBBB chunk: {ex.Message}");
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
        /// Attempts to interpret the data as an array of 32-bit floating point values
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
        /// Attempts to get a bounding box from the data if it's in the expected format
        /// </summary>
        /// <param name="min">The minimum point of the bounding box</param>
        /// <param name="max">The maximum point of the bounding box</param>
        /// <returns>True if the data represents a bounding box, false otherwise</returns>
        public bool TryGetAsBoundingBox(out Vector3 min, out Vector3 max)
        {
            min = Vector3.Zero;
            max = Vector3.Zero;
            
            // A bounding box should be 6 floats (min x,y,z, max x,y,z)
            if (RawData.Length != 24)
            {
                return false;
            }
            
            float[] values = TryGetAsFloatArray();
            if (values != null && values.Length == 6)
            {
                min = new Vector3(values[0], values[1], values[2]);
                max = new Vector3(values[3], values[4], values[5]);
                return true;
            }
            
            return false;
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