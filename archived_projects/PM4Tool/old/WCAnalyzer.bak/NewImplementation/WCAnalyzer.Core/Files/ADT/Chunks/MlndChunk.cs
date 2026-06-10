using System;
using System.IO;
using System.Numerics;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Interfaces;

namespace NewImplementation.WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MLND chunk - Legion Node Data
    /// This chunk appears to contain node data for Legion+ terrain LOD structure
    /// (MLND could stand for Map Legion Node Data)
    /// </summary>
    public class MlndChunk : ADTChunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MLND";

        /// <summary>
        /// Gets the raw chunk data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MlndChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MlndChunk(byte[] data, ILogger? logger = null) : base(SIGNATURE, data, logger)
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

            // Check for possible struct sizes
            int[] likelyStructSizes = new[] { 12, 16, 20, 24, 32, 36, 48, 64 };
            foreach (int structSize in likelyStructSizes)
            {
                if (RawData.Length % structSize == 0)
                {
                    int count = RawData.Length / structSize;
                    Logger?.LogDebug($"Data could represent {count} structs of size {structSize} bytes");
                    
                    // Node data often contains:
                    // - Position (Vector3 - 12 bytes)
                    // - Bounding box info (min/max - 24 bytes)
                    // - Index references (4 bytes each)
                    // - Flags/status values (4 bytes)
                }
            }
            
            // Check if the data can be interpreted as Vector3 positions
            if (RawData.Length % 12 == 0)
            {
                int vectorCount = RawData.Length / 12;
                Logger?.LogDebug($"Data could represent {vectorCount} Vector3 values (positions)");
                
                // Sample a few vectors
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int samplesToRead = Math.Min(5, vectorCount);
                    StringBuilder vectorSamples = new StringBuilder();
                    
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        if (i > 0) vectorSamples.Append(", ");
                        float x = br.ReadSingle();
                        float y = br.ReadSingle();
                        float z = br.ReadSingle();
                        vectorSamples.Append($"({x:F2}, {y:F2}, {z:F2})");
                    }
                    
                    Logger?.LogDebug($"First {samplesToRead} values as Vector3: {vectorSamples}");
                }
            }
            
            // Check for possible int32 arrays (indices, references)
            if (RawData.Length % 4 == 0)
            {
                int intCount = RawData.Length / 4;
                Logger?.LogDebug($"Data could represent {intCount} Int32 values");
                
                // Sample a few values
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int samplesToRead = Math.Min(10, intCount);
                    StringBuilder intSamples = new StringBuilder();
                    
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        if (i > 0) intSamples.Append(", ");
                        int val = br.ReadInt32();
                        intSamples.Append($"{val} (0x{val:X8})");
                    }
                    
                    Logger?.LogDebug($"First {samplesToRead} values as Int32: {intSamples}");
                }
            }
            
            // Log the first few bytes as hex dump
            StringBuilder hexDump = new StringBuilder();
            for (int i = 0; i < Math.Min(RawData.Length, 64); i++)
            {
                hexDump.Append(RawData[i].ToString("X2"));
                if ((i + 1) % 16 == 0)
                    hexDump.Append('\n');
                else if ((i + 1) % 4 == 0)
                    hexDump.Append(' ');
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
        /// Gets the data as an array of Vector3 values (each 12 bytes: X, Y, Z as floats)
        /// </summary>
        /// <returns>Array of Vector3 values if the data size is divisible by 12, otherwise null</returns>
        public Vector3[]? GetAsVector3Array()
        {
            if (RawData.Length == 0 || RawData.Length % 12 != 0)
                return null;
            
            int count = RawData.Length / 12;
            Vector3[] vectors = new Vector3[count];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < count; i++)
                {
                    float x = br.ReadSingle();
                    float y = br.ReadSingle();
                    float z = br.ReadSingle();
                    vectors[i] = new Vector3(x, y, z);
                }
            }
            
            return vectors;
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