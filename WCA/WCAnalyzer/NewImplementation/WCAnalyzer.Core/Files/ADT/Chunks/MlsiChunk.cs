using System;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Interfaces;

namespace NewImplementation.WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MLSI chunk - Legion Surface Info
    /// This chunk appears to contain surface information data for Legion+ terrain structures
    /// (MLSI could stand for Map Legion Surface Information)
    /// </summary>
    public class MlsiChunk : ADTChunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MLSI";

        /// <summary>
        /// Gets the raw chunk data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MlsiChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MlsiChunk(byte[] data, ILogger? logger = null) : base(SIGNATURE, data, logger)
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
            
            // Check for byte pattern distributions
            byte[] distribution = new byte[256];
            for (int i = 0; i < RawData.Length; i++)
            {
                distribution[RawData[i]]++;
            }
            
            // Find the most common byte values
            int firstMax = 0, secondMax = 0, thirdMax = 0;
            byte firstMaxByte = 0, secondMaxByte = 0, thirdMaxByte = 0;
            
            for (int i = 0; i < distribution.Length; i++)
            {
                if (distribution[i] > firstMax)
                {
                    thirdMax = secondMax;
                    thirdMaxByte = secondMaxByte;
                    secondMax = firstMax;
                    secondMaxByte = firstMaxByte;
                    firstMax = distribution[i];
                    firstMaxByte = (byte)i;
                }
                else if (distribution[i] > secondMax)
                {
                    thirdMax = secondMax;
                    thirdMaxByte = secondMaxByte;
                    secondMax = distribution[i];
                    secondMaxByte = (byte)i;
                }
                else if (distribution[i] > thirdMax)
                {
                    thirdMax = distribution[i];
                    thirdMaxByte = (byte)i;
                }
            }
            
            // Calculate zero and non-zero bytes for alpha channel detection
            int zeroCount = distribution[0];
            int nonZeroCount = RawData.Length - zeroCount;
            
            Logger?.LogDebug($"Byte distribution: Most common: 0x{firstMaxByte:X2} ({firstMax} occurrences, {(float)firstMax / RawData.Length:P1})");
            Logger?.LogDebug($"Second most common: 0x{secondMaxByte:X2} ({secondMax} occurrences, {(float)secondMax / RawData.Length:P1})");
            Logger?.LogDebug($"Third most common: 0x{thirdMaxByte:X2} ({thirdMax} occurrences, {(float)thirdMax / RawData.Length:P1})");
            Logger?.LogDebug($"Zero bytes: {zeroCount} ({(float)zeroCount / RawData.Length:P1}), Non-zero bytes: {nonZeroCount} ({(float)nonZeroCount / RawData.Length:P1})");
            
            // Check common structure sizes
            int[] commonSizes = new[] { 2, 4, 8, 12, 16, 20, 24, 32, 36, 40, 48 };
            foreach (int size in commonSizes)
            {
                if (RawData.Length % size == 0 && RawData.Length / size > 1)
                {
                    Logger?.LogDebug($"Data could contain {RawData.Length / size} elements of size {size} bytes");
                }
            }
            
            // Check for possible word arrays (indices, flag bits)
            if (RawData.Length % 2 == 0)
            {
                int wordCount = RawData.Length / 2;
                Logger?.LogDebug($"Data could represent {wordCount} 16-bit values");
                
                // Sample some values
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int samplesToRead = Math.Min(10, wordCount);
                    StringBuilder sampleValues = new StringBuilder();
                    
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        if (i > 0) sampleValues.Append(", ");
                        ushort val = br.ReadUInt16();
                        sampleValues.Append($"0x{val:X4}");
                    }
                    
                    Logger?.LogDebug($"First {samplesToRead} 16-bit values: {sampleValues}");
                }
            }
            
            // Check for possible 32-bit arrays (indices, flags, counters)
            if (RawData.Length % 4 == 0)
            {
                int dwordCount = RawData.Length / 4;
                Logger?.LogDebug($"Data could represent {dwordCount} 32-bit values");
                
                // Sample some values
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int samplesToRead = Math.Min(8, dwordCount);
                    StringBuilder sampleValues = new StringBuilder();
                    
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        if (i > 0) sampleValues.Append(", ");
                        uint val = br.ReadUInt32();
                        sampleValues.Append($"0x{val:X8}");
                    }
                    
                    Logger?.LogDebug($"First {samplesToRead} 32-bit values: {sampleValues}");
                    
                    // Reset stream to beginning and sample as float
                    ms.Position = 0;
                    StringBuilder floatSamples = new StringBuilder();
                    
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        if (i > 0) floatSamples.Append(", ");
                        float val = br.ReadSingle();
                        floatSamples.Append($"{val:F3}");
                    }
                    
                    Logger?.LogDebug($"First {samplesToRead} float values: {floatSamples}");
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
        /// Gets the data as an array of bytes
        /// </summary>
        /// <returns>Array of byte values</returns>
        public byte[] GetAsByteArray()
        {
            return RawData;
        }
        
        /// <summary>
        /// Gets the data as an array of 16-bit unsigned integer values
        /// </summary>
        /// <returns>Array of ushort values if the data size is divisible by 2, otherwise null</returns>
        public ushort[]? GetAsUInt16Array()
        {
            if (RawData.Length == 0 || RawData.Length % 2 != 0)
                return null;
            
            int count = RawData.Length / 2;
            ushort[] values = new ushort[count];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < count; i++)
                {
                    values[i] = br.ReadUInt16();
                }
            }
            
            return values;
        }
        
        /// <summary>
        /// Gets the data as an array of 32-bit unsigned integer values
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
        /// Gets the raw data for manual inspection
        /// </summary>
        /// <returns>The raw data</returns>
        public byte[] GetRawData()
        {
            return RawData;
        }
    }
} 