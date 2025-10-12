using System;
using Microsoft.Extensions.Logging;
using System.IO;
using WCAnalyzer.Core.Common.Extensions;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.PM4.Chunks
{
    /// <summary>
    /// MPRR chunk - Contains a list of 4-byte structures
    /// Each record consists of two uint16 values (4 bytes total)
    /// </summary>
    public class MprrChunk : PM4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MPRR")
        /// </summary>
        public const string SIGNATURE = "MPRR";
        
        /// <summary>
        /// Size of a single record in bytes
        /// </summary>
        private const int RECORD_SIZE = 4; // 2 uint16 values = 4 bytes
        
        /// <summary>
        /// Structure representing a single record in the MPRR chunk
        /// </summary>
        public struct MprrRecord
        {
            /// <summary>
            /// First value in the record
            /// </summary>
            public ushort Value1 { get; set; }
            
            /// <summary>
            /// Second value in the record
            /// </summary>
            public ushort Value2 { get; set; }
        }
        
        /// <summary>
        /// Creates a new MPRR chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MprrChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
        }
        
        /// <summary>
        /// Creates a new empty MPRR chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MprrChunk(ILogger? logger = null)
            : base(SIGNATURE, new byte[0], logger)
        {
        }
        
        /// <summary>
        /// Gets the number of records in the chunk
        /// </summary>
        /// <returns>Number of complete records</returns>
        public int GetRecordCount()
        {
            return _data.Length / RECORD_SIZE;
        }
        
        /// <summary>
        /// Gets a record at the specified index
        /// </summary>
        /// <param name="index">Record index (0-based)</param>
        /// <returns>Record structure</returns>
        /// <exception cref="ArgumentOutOfRangeException">If index is out of range</exception>
        public MprrRecord GetRecord(int index)
        {
            int recordCount = GetRecordCount();
            if (index < 0 || index >= recordCount)
            {
                throw new ArgumentOutOfRangeException(nameof(index), $"Record index must be between 0 and {recordCount - 1}");
            }
            
            int offset = index * RECORD_SIZE;
            ushort value1 = BitConverter.ToUInt16(_data, offset);
            ushort value2 = BitConverter.ToUInt16(_data, offset + 2);
            
            return new MprrRecord
            {
                Value1 = value1,
                Value2 = value2
            };
        }
        
        /// <summary>
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            int recordCount = GetRecordCount();
            return $"{SIGNATURE} Chunk: {recordCount} record(s)";
        }
        
        /// <summary>
        /// Parses the chunk data from the binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to parse from.</param>
        /// <param name="size">The size of the chunk data.</param>
        public override void Parse(BinaryReader reader, uint size)
        {
            _data = reader.ReadBytes((int)size);
            
            // Validate that the data size is divisible by 4 (size of an entry = 2 * 2 bytes)
            if (_data.Length % 4 != 0)
            {
                LogParsingError($"Invalid MPRR chunk data size: {_data.Length}. Should be divisible by 4.");
            }
        }
        
        /// <summary>
        /// Writes the chunk data to the specified binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        public override void Write(BinaryWriter writer)
        {
            WriteChunkHeader(writer);
            writer.Write(_data);
        }
        
        /// <summary>
        /// Gets a hexadecimal representation of the raw data for debugging.
        /// </summary>
        /// <returns>A hexadecimal string representation of the data.</returns>
        public string GetHexDump()
        {
            return BitConverter.ToString(_data);
        }
    }
    
    /// <summary>
    /// Represents an entry in the MPRR chunk.
    /// </summary>
    public struct MprrEntry
    {
        /// <summary>
        /// Gets or sets the first value of the entry.
        /// </summary>
        public ushort Value1 { get; set; }
        
        /// <summary>
        /// Gets or sets the second value of the entry.
        /// </summary>
        public ushort Value2 { get; set; }
        
        /// <summary>
        /// Returns a string representation of the entry.
        /// </summary>
        /// <returns>A string representation of the entry.</returns>
        public override string ToString()
        {
            return $"[{Value1}, {Value2}]";
        }
    }
} 