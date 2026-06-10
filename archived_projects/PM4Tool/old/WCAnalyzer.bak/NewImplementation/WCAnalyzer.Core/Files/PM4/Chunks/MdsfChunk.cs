using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.Logging;
using System.IO;
using WCAnalyzer.Core.Common.Extensions;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.PM4.Chunks
{
    /// <summary>
    /// MDSF chunk - Array of 8-byte structures
    /// Each structure contains two float values
    /// </summary>
    public class MdsfChunk : PM4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MDSF")
        /// </summary>
        public const string SIGNATURE = "MDSF";
        
        /// <summary>
        /// Size of a single structure in bytes
        /// </summary>
        private const int STRUCTURE_SIZE = 8;
        
        /// <summary>
        /// Structure containing two float values
        /// </summary>
        public struct MdsfStructure
        {
            /// <summary>
            /// First float value
            /// </summary>
            public float Value1 { get; set; }
            
            /// <summary>
            /// Second float value
            /// </summary>
            public float Value2 { get; set; }
            
            /// <summary>
            /// Creates a string representation of the structure
            /// </summary>
            /// <returns>String representation</returns>
            public override string ToString()
            {
                return $"({Value1}, {Value2})";
            }
        }
        
        /// <summary>
        /// Creates a new MDSF chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MdsfChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
        }
        
        /// <summary>
        /// Creates a new empty MDSF chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MdsfChunk(ILogger? logger = null)
            : base(SIGNATURE, new byte[0], logger)
        {
        }
        
        /// <summary>
        /// Gets the number of structures in the chunk
        /// </summary>
        /// <returns>Number of structures</returns>
        public int GetStructureCount()
        {
            return _data.Length / STRUCTURE_SIZE;
        }
        
        /// <summary>
        /// Gets a structure at the specified index
        /// </summary>
        /// <param name="index">Structure index (0-based)</param>
        /// <returns>MdsfStructure containing two float values</returns>
        /// <exception cref="ArgumentOutOfRangeException">If index is out of range</exception>
        public MdsfStructure GetStructure(int index)
        {
            int structureCount = GetStructureCount();
            
            if (index < 0 || index >= structureCount)
            {
                throw new ArgumentOutOfRangeException(nameof(index), 
                    $"Structure index must be between 0 and {structureCount - 1}");
            }
            
            int offset = index * STRUCTURE_SIZE;
            
            float value1 = BitConverter.ToSingle(_data, offset);
            float value2 = BitConverter.ToSingle(_data, offset + 4);
            
            return new MdsfStructure
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
            return $"{SIGNATURE} Chunk: {GetStructureCount()} structures";
        }
        
        /// <summary>
        /// Writes this chunk to a byte array
        /// </summary>
        /// <returns>Byte array containing chunk data</returns>
        public override byte[] Write()
        {
            return _data;
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
    /// Represents an entry in the MDSF chunk.
    /// </summary>
    public struct MdsfEntry
    {
        /// <summary>
        /// Gets or sets the first value of the entry.
        /// </summary>
        public uint Value1 { get; set; }
        
        /// <summary>
        /// Gets or sets the second value of the entry.
        /// </summary>
        public uint Value2 { get; set; }
        
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