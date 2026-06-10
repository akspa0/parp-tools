using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Interfaces;

namespace NewImplementation.WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MLVI chunk - Legion Vertex Index Data
    /// This chunk appears to contain vertex indices for Legion+ terrain rendering
    /// (MLVI could stand for Map Legion Vertex Indices)
    /// </summary>
    public class MlviChunk : ADTChunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MLVI";

        /// <summary>
        /// Gets the raw chunk data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MlviChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MlviChunk(byte[] data, ILogger? logger = null) : base(SIGNATURE, data, logger)
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

            // Check if data is divisible by common index sizes
            if (RawData.Length % 2 == 0)
            {
                Logger?.LogDebug($"{SIGNATURE} data size is divisible by 2 (possible array of shorts/uint16/indices)");
                int indexCount = RawData.Length / 2;
                Logger?.LogDebug($"This would represent {indexCount} 16-bit indices");
                
                // Sample the first few values as uint16
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int samplesToRead = Math.Min(8, indexCount);
                    StringBuilder sampleValues = new StringBuilder();
                    
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        if (i > 0) sampleValues.Append(", ");
                        sampleValues.Append(br.ReadUInt16());
                    }
                    
                    Logger?.LogDebug($"First {samplesToRead} values as uint16: {sampleValues}");
                }
            }
            
            if (RawData.Length % 4 == 0)
            {
                Logger?.LogDebug($"{SIGNATURE} data size is divisible by 4 (possible array of ints/uint32/indices)");
                int indexCount = RawData.Length / 4;
                Logger?.LogDebug($"This would represent {indexCount} 32-bit indices");
                
                // Sample the first few values as uint32
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int samplesToRead = Math.Min(8, indexCount);
                    StringBuilder sampleValues = new StringBuilder();
                    
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        if (i > 0) sampleValues.Append(", ");
                        sampleValues.Append(br.ReadUInt32());
                    }
                    
                    Logger?.LogDebug($"First {samplesToRead} values as uint32: {sampleValues}");
                }
            }
            
            // Check for triangle patterns (indices typically come in groups of 3)
            if (RawData.Length % 6 == 0 && RawData.Length >= 6)
            {
                int triangleCount = RawData.Length / 6;
                Logger?.LogDebug($"{SIGNATURE} data could represent {triangleCount} triangles (if using 16-bit indices)");
                
                // Sample triangles
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int samplesToRead = Math.Min(3, triangleCount);
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        ushort i1 = br.ReadUInt16();
                        ushort i2 = br.ReadUInt16();
                        ushort i3 = br.ReadUInt16();
                        Logger?.LogDebug($"Triangle {i+1}: [{i1}, {i2}, {i3}]");
                    }
                }
            }
            
            if (RawData.Length % 12 == 0 && RawData.Length >= 12)
            {
                int triangleCount = RawData.Length / 12;
                Logger?.LogDebug($"{SIGNATURE} data could represent {triangleCount} triangles (if using 32-bit indices)");
                
                // Sample triangles
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int samplesToRead = Math.Min(3, triangleCount);
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        uint i1 = br.ReadUInt32();
                        uint i2 = br.ReadUInt32();
                        uint i3 = br.ReadUInt32();
                        Logger?.LogDebug($"Triangle {i+1}: [{i1}, {i2}, {i3}]");
                    }
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
        /// Gets the data as an array of 16-bit indices
        /// </summary>
        /// <returns>Array of ushort values if the data size is divisible by 2, otherwise null</returns>
        public ushort[]? GetAsUInt16Array()
        {
            if (RawData.Length == 0 || RawData.Length % 2 != 0)
                return null;
            
            int count = RawData.Length / 2;
            ushort[] indices = new ushort[count];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < count; i++)
                {
                    indices[i] = br.ReadUInt16();
                }
            }
            
            return indices;
        }
        
        /// <summary>
        /// Gets the data as an array of 32-bit indices
        /// </summary>
        /// <returns>Array of uint values if the data size is divisible by 4, otherwise null</returns>
        public uint[]? GetAsUInt32Array()
        {
            if (RawData.Length == 0 || RawData.Length % 4 != 0)
                return null;
            
            int count = RawData.Length / 4;
            uint[] indices = new uint[count];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < count; i++)
                {
                    indices[i] = br.ReadUInt32();
                }
            }
            
            return indices;
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