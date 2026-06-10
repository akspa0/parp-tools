using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Interfaces;

namespace NewImplementation.WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MLDX chunk - Legion Doodad References
    /// This chunk appears to contain doodad reference data for Legion+ terrain
    /// (MLDX could stand for Map Legion DooDad indeX)
    /// </summary>
    public class MldxChunk : ADTChunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MLDX";

        /// <summary>
        /// Gets the raw chunk data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MldxChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MldxChunk(byte[] data, ILogger? logger = null) : base(SIGNATURE, data, logger)
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
            
            // Check for various possible structures related to doodad references
            
            // Check for null-terminated strings (possible filenames)
            List<string> possibleStrings = new List<string>();
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                while (ms.Position < ms.Length)
                {
                    long startPos = ms.Position;
                    
                    // Try to read until null or end of stream
                    StringBuilder sb = new StringBuilder();
                    byte b;
                    bool isValidString = true;
                    
                    while (ms.Position < ms.Length && (b = br.ReadByte()) != 0)
                    {
                        // Check if character is in valid ASCII range
                        if (b < 32 || b > 126)
                        {
                            isValidString = false;
                            break;
                        }
                        sb.Append((char)b);
                    }
                    
                    // If we found a valid string of reasonable length
                    if (isValidString && sb.Length > 3 && sb.Length < 260)
                    {
                        string str = sb.ToString();
                        if (str.EndsWith(".mdx") || str.EndsWith(".m2") || str.EndsWith(".wmo"))
                        {
                            possibleStrings.Add(str);
                            Logger?.LogDebug($"Possible filename at offset {startPos}: {str}");
                        }
                    }
                    
                    // If not a valid string, go back to start position + 1
                    if (!isValidString)
                    {
                        ms.Position = startPos + 1;
                    }
                }
            }
            
            if (possibleStrings.Count > 0)
            {
                Logger?.LogDebug($"Found {possibleStrings.Count} possible doodad filenames");
            }
            
            // Check for uint32 arrays (possible indices or offsets)
            if (RawData.Length % 4 == 0)
            {
                int valueCount = RawData.Length / 4;
                Logger?.LogDebug($"Data could represent {valueCount} uint32 values (possible indices or offsets)");
                
                // Sample some values
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int samplesToRead = Math.Min(10, valueCount);
                    StringBuilder sampleValues = new StringBuilder();
                    
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        if (i > 0) sampleValues.Append(", ");
                        uint val = br.ReadUInt32();
                        sampleValues.Append($"0x{val:X8}");
                    }
                    
                    Logger?.LogDebug($"First {samplesToRead} uint32 values: {sampleValues}");
                }
            }
            
            // Check for struct arrays (e.g., position, rotation, scale)
            int[] commonStructSizes = new[] { 12, 16, 20, 24, 28, 32, 36 };
            foreach (int structSize in commonStructSizes)
            {
                if (RawData.Length % structSize == 0 && RawData.Length > 0)
                {
                    int structCount = RawData.Length / structSize;
                    if (structCount > 1)
                    {
                        Logger?.LogDebug($"Data could represent {structCount} structs of size {structSize} bytes");
                        
                        // Sample first struct
                        if (structCount >= 1 && structSize <= 36)
                        {
                            StringBuilder structSample = new StringBuilder();
                            structSample.Append($"First struct ({structSize} bytes): ");
                            
                            using (var ms = new MemoryStream(RawData))
                            using (var br = new BinaryReader(ms))
                            {
                                // Read first struct as bytes (for hex display)
                                byte[] firstStruct = br.ReadBytes(structSize);
                                for (int i = 0; i < firstStruct.Length; i++)
                                {
                                    if (i > 0 && i % 4 == 0) structSample.Append(' ');
                                    structSample.Append(firstStruct[i].ToString("X2"));
                                }
                            }
                            
                            Logger?.LogDebug(structSample.ToString());
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
        /// Gets the data as an array of 32-bit unsigned integers
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
        /// Tries to extract string data from the chunk, assuming it contains null-terminated strings
        /// </summary>
        /// <returns>List of extracted strings, may be empty if no valid strings found</returns>
        public List<string> ExtractStrings()
        {
            List<string> strings = new List<string>();
            if (RawData.Length == 0)
                return strings;
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                while (ms.Position < ms.Length)
                {
                    long startPos = ms.Position;
                    
                    // Try to read a null-terminated string
                    StringBuilder sb = new StringBuilder();
                    byte b;
                    bool isValidString = true;
                    
                    while (ms.Position < ms.Length && (b = br.ReadByte()) != 0)
                    {
                        // Check if character is in valid ASCII range
                        if (b < 32 || b > 126)
                        {
                            isValidString = false;
                            break;
                        }
                        sb.Append((char)b);
                    }
                    
                    // If we found a valid string
                    if (isValidString && sb.Length > 0 && ms.Position < ms.Length)
                    {
                        strings.Add(sb.ToString());
                    }
                    
                    // If not a valid string or we didn't reach a null terminator, move forward
                    if (!isValidString || ms.Position >= ms.Length)
                    {
                        ms.Position = startPos + 1;
                    }
                    
                    // If we've read a string and reached a null terminator, continue from next byte
                }
            }
            
            return strings;
        }
        
        /// <summary>
        /// Gets structured data from the chunk, assuming a fixed record size
        /// </summary>
        /// <param name="recordSize">Size of each record in bytes</param>
        /// <returns>Array of byte arrays, each containing one record's data</returns>
        public byte[][]? GetStructuredData(int recordSize)
        {
            if (RawData.Length == 0 || recordSize <= 0 || RawData.Length % recordSize != 0)
                return null;
            
            int recordCount = RawData.Length / recordSize;
            byte[][] records = new byte[recordCount][];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < recordCount; i++)
                {
                    records[i] = br.ReadBytes(recordSize);
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