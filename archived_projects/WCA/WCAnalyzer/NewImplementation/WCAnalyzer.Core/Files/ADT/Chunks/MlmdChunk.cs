using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Interfaces;

namespace NewImplementation.WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MLMD chunk - Legion Metadata
    /// This chunk appears to contain metadata information for Legion+ terrain features
    /// (MLMD could stand for Map Legion MetaData)
    /// </summary>
    public class MlmdChunk : ADTChunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MLMD";

        /// <summary>
        /// Gets the raw chunk data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MlmdChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MlmdChunk(byte[] data, ILogger? logger = null) : base(SIGNATURE, data, logger)
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
            
            // Check for header structure (often fixed size at the beginning)
            if (RawData.Length >= 16)
            {
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    try
                    {
                        uint value1 = br.ReadUInt32();
                        uint value2 = br.ReadUInt32();
                        uint value3 = br.ReadUInt32();
                        uint value4 = br.ReadUInt32();
                        
                        Logger?.LogDebug($"Possible header values: {value1} (0x{value1:X8}), {value2} (0x{value2:X8}), {value3} (0x{value3:X8}), {value4} (0x{value4:X8})");
                        
                        // Also interpret as floats in case they represent coordinates or dimensions
                        ms.Position = 0;
                        float fValue1 = br.ReadSingle();
                        float fValue2 = br.ReadSingle();
                        float fValue3 = br.ReadSingle();
                        float fValue4 = br.ReadSingle();
                        
                        Logger?.LogDebug($"As float values: {fValue1:F3}, {fValue2:F3}, {fValue3:F3}, {fValue4:F3}");
                    }
                    catch (Exception ex)
                    {
                        Logger?.LogWarning($"Error reading potential header values: {ex.Message}");
                    }
                }
            }
            
            // Check for common record/entry structure sizes
            int[] possibleSizes = new[] { 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 64 };
            foreach (int size in possibleSizes)
            {
                // Skip sizes smaller than 4 bytes and check if data length is divisible by the size
                if (size >= 4 && RawData.Length % size == 0 && RawData.Length / size > 1)
                {
                    int entryCount = RawData.Length / size;
                    Logger?.LogDebug($"Data could contain {entryCount} entries of {size} bytes each");
                    
                    // If there's a reasonable number of fixed-sized entries, sample the first one
                    if (entryCount > 1 && entryCount < 1000 && size <= 32)
                    {
                        using (var ms = new MemoryStream(RawData))
                        using (var br = new BinaryReader(ms))
                        {
                            byte[] firstEntry = br.ReadBytes(size);
                            
                            // Format the entry as hex and try to interpret as various types
                            StringBuilder entryHex = new StringBuilder();
                            for (int i = 0; i < firstEntry.Length; i++)
                            {
                                if (i > 0 && i % 4 == 0) entryHex.Append(' ');
                                entryHex.Append(firstEntry[i].ToString("X2"));
                            }
                            
                            Logger?.LogDebug($"First entry ({size} bytes): {entryHex}");
                            
                            // Try to interpret the first entry in different ways
                            if (size % 4 == 0)
                            {
                                int ints = size / 4;
                                StringBuilder asInts = new StringBuilder();
                                asInts.Append("As Int32 values: ");
                                
                                ms.Position = 0;
                                for (int i = 0; i < ints; i++)
                                {
                                    if (i > 0) asInts.Append(", ");
                                    asInts.Append(br.ReadInt32());
                                }
                                
                                Logger?.LogDebug(asInts.ToString());
                                
                                ms.Position = 0;
                                StringBuilder asFloats = new StringBuilder();
                                asFloats.Append("As float values: ");
                                
                                for (int i = 0; i < ints; i++)
                                {
                                    if (i > 0) asFloats.Append(", ");
                                    asFloats.Append($"{br.ReadSingle():F3}");
                                }
                                
                                Logger?.LogDebug(asFloats.ToString());
                            }
                        }
                    }
                }
            }
            
            // Check for string data in larger chunks
            if (RawData.Length > 16)
            {
                StringBuilder textContent = new StringBuilder();
                int consecutiveText = 0;
                int textStart = -1;
                
                for (int i = 0; i < RawData.Length; i++)
                {
                    byte b = RawData[i];
                    
                    // Check for printable ASCII characters (32-126) or common control chars
                    if ((b >= 32 && b <= 126) || b == 9 || b == 10 || b == 13)
                    {
                        if (consecutiveText == 0)
                            textStart = i;
                        
                        consecutiveText++;
                        
                        if (b >= 32 && b <= 126)
                            textContent.Append((char)b);
                        else if (b == 9)
                            textContent.Append("\\t");
                        else if (b == 10)
                            textContent.Append("\\n");
                        else if (b == 13)
                            textContent.Append("\\r");
                    }
                    else
                    {
                        // If we found a reasonable length string, log it
                        if (consecutiveText >= 4)
                        {
                            Logger?.LogDebug($"Possible text at offset {textStart}: \"{textContent}\"");
                        }
                        
                        textContent.Clear();
                        consecutiveText = 0;
                    }
                }
                
                // Check if we ended with text content
                if (consecutiveText >= 4)
                {
                    Logger?.LogDebug($"Possible text at offset {textStart}: \"{textContent}\"");
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
        /// Attempts to extract potential metadata fields, assuming a fixed header size
        /// </summary>
        /// <param name="headerSize">The assumed size of the header in bytes</param>
        /// <returns>Dictionary of key/value pairs, where the key is the field offset</returns>
        public Dictionary<int, uint>? ExtractMetadataFields(int headerSize = 16)
        {
            if (RawData.Length < headerSize)
                return null;
            
            var fields = new Dictionary<int, uint>();
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int offset = 0; offset < headerSize; offset += 4)
                {
                    if (offset + 4 <= RawData.Length)
                    {
                        fields[offset] = br.ReadUInt32();
                    }
                }
            }
            
            return fields;
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
        /// Gets structured data from the chunk, assuming a fixed record size
        /// </summary>
        /// <param name="recordSize">Size of each record in bytes</param>
        /// <param name="headerSize">Size of header to skip before records begin (optional)</param>
        /// <returns>Array of byte arrays, each containing one record's data</returns>
        public byte[][]? GetStructuredData(int recordSize, int headerSize = 0)
        {
            if (RawData.Length <= headerSize || recordSize <= 0 || (RawData.Length - headerSize) % recordSize != 0)
                return null;
            
            int recordCount = (RawData.Length - headerSize) / recordSize;
            byte[][] records = new byte[recordCount][];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                // Skip header if specified
                if (headerSize > 0)
                    ms.Position = headerSize;
                
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