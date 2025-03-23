using System;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MBMI chunk - Contains blend map information for ADT files (Cataclysm+)
    /// </summary>
    public class MbmiChunk : BaseChunk
    {
        /// <summary>
        /// The signature of the MBMI chunk
        /// </summary>
        public const string SIGNATURE = "MBMI";

        /// <summary>
        /// Gets the raw blend map information data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MbmiChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MbmiChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            RawData = Array.Empty<byte>();
            Parse();
        }

        /// <summary>
        /// Parses the MBMI chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MBMI chunk has no data");
                    return;
                }
                
                // Store the raw data for later analysis
                RawData = new byte[Data.Length];
                Array.Copy(Data, RawData, Data.Length);
                
                Logger?.LogDebug($"MBMI: Read {Data.Length} bytes of blend map information data");
                LogDataSummary();
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MBMI chunk: {ex.Message}");
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
                Logger?.LogDebug($"MBMI: Data size: {RawData.Length} bytes");
                
                // Check if data size appears to be a collection of entries
                int entrySize = 0;
                
                // Try to determine if data consists of fixed-size entries
                foreach (int size in new[] { 4, 8, 12, 16, 20, 24, 32 })
                {
                    if (RawData.Length % size == 0 && RawData.Length > 0)
                    {
                        entrySize = size;
                        int count = RawData.Length / size;
                        Logger?.LogDebug($"MBMI: Data may contain {count} entries of {size} bytes each");
                        
                        // Only log the details for the first possibility we find
                        if (entrySize > 0)
                            break;
                    }
                }
                
                // If we found a potential entry size, log some sample entries
                if (entrySize > 0)
                {
                    int count = RawData.Length / entrySize;
                    int samplesToLog = Math.Min(count, 4); // Log at most 4 sample entries
                    
                    for (int i = 0; i < samplesToLog; i++)
                    {
                        ms.Position = i * entrySize;
                        Logger?.LogDebug($"MBMI: Entry {i} raw data:");
                        
                        StringBuilder sb = new StringBuilder();
                        for (int j = 0; j < entrySize; j += 4)
                        {
                            if (j + 4 <= entrySize)
                            {
                                uint value = reader.ReadUInt32();
                                sb.Append($"0x{value:X8} ");
                            }
                        }
                        
                        Logger?.LogDebug($"  {sb}");
                    }
                }
                else
                {
                    // If no entry size was determined, just log the first few DWORDs
                    if (RawData.Length >= 4)
                    {
                        ms.Position = 0;
                        int dwordsToLog = Math.Min(RawData.Length / 4, 8);
                        Logger?.LogDebug("MBMI: First few DWORD values:");
                        
                        for (int i = 0; i < dwordsToLog; i++)
                        {
                            uint value = reader.ReadUInt32();
                            Logger?.LogDebug($"  [{i}] = {value} (0x{value:X8})");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error logging MBMI data summary: {ex.Message}");
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
                Logger?.LogError("Cannot write MBMI chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                writer.Write(RawData.Length);
                writer.Write(RawData);
                
                Logger?.LogDebug($"MBMI: Wrote {RawData.Length} bytes of blend map information data");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MBMI chunk: {ex.Message}");
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
        /// Attempts to interpret the data as an array of DWORD values
        /// </summary>
        /// <returns>An array of DWORD values, or null if not applicable</returns>
        public uint[] TryGetAsDwordArray()
        {
            if (RawData.Length % 4 != 0 || RawData.Length == 0)
                return null;
                
            try
            {
                using var ms = new MemoryStream(RawData);
                using var reader = new BinaryReader(ms);
                
                uint[] values = new uint[RawData.Length / 4];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = reader.ReadUInt32();
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
        
        /// <summary>
        /// Estimates the number of entries based on common entry sizes
        /// </summary>
        /// <returns>The estimated number of entries, or 0 if unknown</returns>
        public int EstimateEntryCount()
        {
            // Check common entry sizes (4, 8, 12, 16, 20, 24, 32 bytes)
            foreach (int size in new[] { 4, 8, 12, 16, 20, 24, 32 })
            {
                if (RawData.Length % size == 0 && RawData.Length > 0)
                {
                    return RawData.Length / size;
                }
            }
            
            return 0; // Unknown entry size
        }
    }
} 