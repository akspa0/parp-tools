using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Interfaces;

namespace NewImplementation.WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MLVH chunk - Legion Vertex Height Data
    /// This chunk appears to contain vertex height data for Legion+ terrain rendering
    /// (MLVH could stand for Map Legion Vertex Heights)
    /// </summary>
    public class MlvhChunk : ADTChunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MLVH";

        /// <summary>
        /// Gets the raw chunk data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MlvhChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MlvhChunk(byte[] data, ILogger? logger = null) : base(SIGNATURE, data, logger)
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

            // Check for standard sizes that might indicate height data
            if (RawData.Length % 2 == 0)
            {
                Logger?.LogDebug($"{SIGNATURE} data size is divisible by 2 (possible array of shorts/uint16)");
                int count = RawData.Length / 2;
                
                // If it's height data, it might follow the pattern of heightmap grids (ex: 9x9, 17x17, 33x33)
                if (Math.Sqrt(count) % 1 == 0)
                {
                    int gridSize = (int)Math.Sqrt(count);
                    Logger?.LogDebug($"{SIGNATURE} data could represent a {gridSize}x{gridSize} grid of height values");
                }
                
                // Sample a few values
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int sampleCount = Math.Min(10, count);
                    StringBuilder sampleValues = new StringBuilder();
                    
                    for (int i = 0; i < sampleCount; i++)
                    {
                        if (i > 0) sampleValues.Append(", ");
                        sampleValues.Append(br.ReadInt16());
                    }
                    
                    Logger?.LogDebug($"First {sampleCount} values as int16: {sampleValues}");
                }
            }
            
            if (RawData.Length % 4 == 0)
            {
                Logger?.LogDebug($"{SIGNATURE} data size is divisible by 4 (possible array of ints/floats)");
                int count = RawData.Length / 4;
                
                // If it's height data, it might follow the pattern of heightmap grids (ex: 9x9, 17x17, 33x33)
                if (Math.Sqrt(count) % 1 == 0)
                {
                    int gridSize = (int)Math.Sqrt(count);
                    Logger?.LogDebug($"{SIGNATURE} data could represent a {gridSize}x{gridSize} grid of height values (as floats)");
                }
                
                // Sample a few values as floats (heightmaps are often stored as float values)
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int sampleCount = Math.Min(10, count);
                    StringBuilder sampleValues = new StringBuilder();
                    
                    for (int i = 0; i < sampleCount; i++)
                    {
                        if (i > 0) sampleValues.Append(", ");
                        sampleValues.Append(br.ReadSingle());
                    }
                    
                    Logger?.LogDebug($"First {sampleCount} values as float: {sampleValues}");
                }
            }
            
            // Log the first few bytes for debugging
            StringBuilder hexDump = new StringBuilder();
            for (int i = 0; i < Math.Min(RawData.Length, 48); i++)
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
        /// Gets the data as an array of 16-bit height values
        /// </summary>
        /// <returns>Array of short values if the data size is divisible by 2, otherwise null</returns>
        public short[]? GetAsInt16Array()
        {
            if (RawData.Length == 0 || RawData.Length % 2 != 0)
                return null;
            
            int count = RawData.Length / 2;
            short[] heights = new short[count];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < count; i++)
                {
                    heights[i] = br.ReadInt16();
                }
            }
            
            return heights;
        }
        
        /// <summary>
        /// Gets the data as an array of 32-bit floating point height values
        /// </summary>
        /// <returns>Array of float values if the data size is divisible by 4, otherwise null</returns>
        public float[]? GetAsFloatArray()
        {
            if (RawData.Length == 0 || RawData.Length % 4 != 0)
                return null;
            
            int count = RawData.Length / 4;
            float[] heights = new float[count];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < count; i++)
                {
                    heights[i] = br.ReadSingle();
                }
            }
            
            return heights;
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