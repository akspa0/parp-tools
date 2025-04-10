using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Interfaces;

namespace NewImplementation.WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MLHD chunk - Legion Header Data
    /// This chunk is used in Legion+ ADT files and contains information related to the Legion expansion features
    /// </summary>
    public class MlhdChunk : ADTChunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MLHD";

        /// <summary>
        /// Gets the raw chunk data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MlhdChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MlhdChunk(byte[] data, ILogger? logger = null) : base(SIGNATURE, data, logger)
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
            
            // Look for common structures or patterns
            if (RawData.Length == 32)
            {
                Logger?.LogDebug($"{SIGNATURE} data is 32 bytes - common size for a header structure");
            }
            
            // Check for standard sizes
            if (RawData.Length % 4 == 0)
            {
                Logger?.LogDebug($"{SIGNATURE} data size is divisible by 4 (possible array of ints or floats)");
                
                // Try to interpret as ints/flags and floats
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    if (RawData.Length >= 4)
                    {
                        ms.Position = 0;
                        uint value1 = br.ReadUInt32();
                        Logger?.LogDebug($"First 4 bytes as uint: {value1} (0x{value1:X8})");
                        
                        // Check if it could be flags
                        Logger?.LogDebug($"Possible flags: {Convert.ToString(value1, 2).PadLeft(32, '0')}");
                        
                        ms.Position = 0;
                        float floatValue = br.ReadSingle();
                        Logger?.LogDebug($"First 4 bytes as float: {floatValue}");
                    }
                    
                    // If enough data, check values at common offsets
                    if (RawData.Length >= 8)
                    {
                        ms.Position = 4;
                        uint value2 = br.ReadUInt32();
                        Logger?.LogDebug($"Second 4 bytes as uint: {value2} (0x{value2:X8})");
                        
                        ms.Position = 4;
                        float floatValue = br.ReadSingle();
                        Logger?.LogDebug($"Second 4 bytes as float: {floatValue}");
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
        /// Gets the data as an array of 32-bit values
        /// </summary>
        /// <returns>Array of uint values if the data size is divisible by 4, otherwise null</returns>
        public uint[]? GetAsUInt32Array()
        {
            if (RawData.Length == 0 || RawData.Length % 4 != 0)
                return null;
            
            int count = RawData.Length / 4;
            uint[] values = new uint[count];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < count; i++)
                {
                    values[i] = br.ReadUInt32();
                }
            }
            
            return values;
        }
        
        /// <summary>
        /// Gets the data as an array of floats
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
        /// Gets the raw data for manual inspection
        /// </summary>
        /// <returns>The raw data</returns>
        public byte[] GetRawData()
        {
            return RawData;
        }
    }
} 