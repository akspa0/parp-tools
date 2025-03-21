using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Interfaces;

namespace NewImplementation.WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MLLL chunk - Legion Layer Data
    /// This chunk appears to contain layer data for Legion+ terrain rendering
    /// (MLLL could stand for Map Legion Layer List)
    /// </summary>
    public class MlllChunk : ADTChunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MLLL";

        /// <summary>
        /// Gets the raw chunk data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MlllChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MlllChunk(byte[] data, ILogger? logger = null) : base(SIGNATURE, data, logger)
        {
            RawData = Array.Empty<byte>();
            Parse();
        }

        /// <summary>
        /// Parses the chunk data
        /// </summary>
        protected virtual void Parse()
        {
            try
            {
                if (Data.Length == 0)
                {
                    Logger?.LogWarning($"{SIGNATURE} chunk has no data");
                    return;
                }
                
                // Store the raw data for later analysis
                RawData = new byte[Data.Length];
                Array.Copy(Data, RawData, Data.Length);
                
                Logger?.LogDebug($"{SIGNATURE}: Read {Data.Length} bytes of data");
                LogDataSummary();
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing {SIGNATURE} chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Logs a summary of the data for debugging purposes
        /// </summary>
        protected virtual void LogDataSummary()
        {
            if (RawData.Length == 0)
            {
                Logger?.LogWarning($"{SIGNATURE} chunk data is empty");
                return;
            }

            Logger?.LogDebug($"{SIGNATURE} chunk data size: {RawData.Length} bytes");

            // Check for structured data patterns
            if (RawData.Length % 4 == 0)
            {
                Logger?.LogDebug($"{SIGNATURE} data size is divisible by 4 (possible array of ints, floats, or small structs)");
                int elementCount = RawData.Length / 4;
                Logger?.LogDebug($"This would represent {elementCount} 4-byte elements");
                
                // Sample as integers
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int samplesToRead = Math.Min(8, elementCount);
                    StringBuilder sampleValues = new StringBuilder();
                    
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        if (i > 0) sampleValues.Append(", ");
                        sampleValues.Append(br.ReadInt32());
                    }
                    
                    Logger?.LogDebug($"First {samplesToRead} values as int32: {sampleValues}");
                }
                
                // Also sample as floats
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int samplesToRead = Math.Min(8, elementCount);
                    StringBuilder sampleValues = new StringBuilder();
                    
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        if (i > 0) sampleValues.Append(", ");
                        sampleValues.Append(br.ReadSingle());
                    }
                    
                    Logger?.LogDebug($"First {samplesToRead} values as float: {sampleValues}");
                }
            }
            
            // Check for common structure sizes
            for (int structSize = 8; structSize <= 32; structSize += 4)
            {
                if (RawData.Length % structSize == 0 && RawData.Length >= structSize)
                {
                    int structCount = RawData.Length / structSize;
                    Logger?.LogDebug($"{SIGNATURE} data could represent {structCount} structs of size {structSize} bytes");
                    
                    // For Legion layers, common values might be:
                    // - detail level (float or int)
                    // - distance threshold (float)
                    // - flags/properties (int)
                    
                    // Sample first struct
                    if (structSize <= 16) // Only show first struct for reasonable sizes
                    {
                        using (var ms = new MemoryStream(RawData))
                        using (var br = new BinaryReader(ms))
                        {
                            StringBuilder structValues = new StringBuilder();
                            structValues.Append("First struct data: [");
                            
                            for (int i = 0; i < structSize / 4; i++)
                            {
                                if (i > 0) structValues.Append(", ");
                                uint val = br.ReadUInt32();
                                float fVal = BitConverter.ToSingle(BitConverter.GetBytes(val), 0);
                                structValues.Append($"{val}/0x{val:X}/{fVal:F3}");
                            }
                            
                            structValues.Append("]");
                            Logger?.LogDebug(structValues.ToString());
                        }
                    }
                }
            }
            
            // Log the first few bytes for debugging
            StringBuilder hexDump = new StringBuilder();
            for (int i = 0; i < Math.Min(RawData.Length, 64); i++)
            {
                hexDump.Append(RawData[i].ToString("X2"));
                if ((i + 1) % 16 == 0)
                    hexDump.Append("\n");
                else if ((i + 1) % 4 == 0)
                    hexDump.Append(" ");
            }
            
            Logger?.LogDebug($"First bytes as hex:\n{hexDump}");
        }
        
        /// <summary>
        /// Gets the data as an array of 32-bit integer values
        /// </summary>
        /// <returns>Array of int values if the data size is divisible by 4, otherwise null</returns>
        public int[]? GetAsInt32Array()
        {
            if (RawData.Length == 0 || RawData.Length % 4 != 0)
                return null;
            
            int count = RawData.Length / 4;
            int[] values = new int[count];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < count; i++)
                {
                    values[i] = br.ReadInt32();
                }
            }
            
            return values;
        }
        
        /// <summary>
        /// Gets the data as an array of 32-bit floating point values
        /// </summary>
        /// <returns>Array of float values if the data size is divisible by 4, otherwise null</returns>
        public float[]? GetAsFloatArray()
        {
            if (RawData.Length == 0 || RawData.Length % 4 != 0)
                return null;
            
            int count = RawData.Length / 4;
            float[] values = new float[count];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < count; i++)
                {
                    values[i] = br.ReadSingle();
                }
            }
            
            return values;
        }
        
        /// <summary>
        /// Gets the data interpreted as structured records of the specified size (in bytes)
        /// </summary>
        /// <param name="recordSize">Size of each record in bytes (must be divisible by 4)</param>
        /// <returns>A list of byte arrays, each representing one record</returns>
        public List<byte[]>? GetAsStructuredRecords(int recordSize)
        {
            if (RawData.Length == 0 || recordSize <= 0 || recordSize % 4 != 0 || RawData.Length % recordSize != 0)
                return null;
            
            int recordCount = RawData.Length / recordSize;
            var records = new List<byte[]>(recordCount);
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < recordCount; i++)
                {
                    byte[] record = br.ReadBytes(recordSize);
                    records.Add(record);
                }
            }
            
            return records;
        }
        
        /// <summary>
        /// Gets the raw data for manual inspection
        /// </summary>
        /// <returns>The raw data</returns>
        public byte[] GetRawData()
        {
            return RawData;
        }
    }
} 