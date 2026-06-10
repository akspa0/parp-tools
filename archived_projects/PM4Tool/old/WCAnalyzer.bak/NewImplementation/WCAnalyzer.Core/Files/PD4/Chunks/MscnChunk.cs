using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using Microsoft.Extensions.Logging;
using System.IO;
using WCAnalyzer.Core.Common.Extensions;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.PD4.Chunks
{
    /// <summary>
    /// MSCN chunk - Contains vector data (x, y, z, w) for PD4 format
    /// </summary>
    public class MscnChunk : PD4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MSCN")
        /// </summary>
        public const string SIGNATURE = "MSCN";
        
        /// <summary>
        /// Size of a vector in bytes (4 floats * 4 bytes)
        /// </summary>
        private const int VECTOR_SIZE = 16;
        
        /// <summary>
        /// Creates a new MSCN chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MscnChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Validate size
            if (_data.Length % VECTOR_SIZE != 0)
            {
                Logger?.LogWarning($"MSCN chunk has invalid size: {_data.Length} bytes. Should be divisible by {VECTOR_SIZE}.");
            }
        }
        
        /// <summary>
        /// Creates a new empty MSCN chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MscnChunk(ILogger? logger = null)
            : base(SIGNATURE, new byte[0], logger)
        {
        }
        
        /// <summary>
        /// Gets the number of vectors in the chunk
        /// </summary>
        /// <returns>Number of vectors</returns>
        public int GetVectorCount()
        {
            return _data.Length / VECTOR_SIZE;
        }
        
        /// <summary>
        /// Gets a vector at the specified index
        /// </summary>
        /// <param name="index">Vector index (0-based)</param>
        /// <returns>Vector4 containing x, y, z, w values</returns>
        /// <exception cref="ArgumentOutOfRangeException">If index is out of range</exception>
        public Vector4 GetVector(int index)
        {
            int vectorCount = GetVectorCount();
            
            if (index < 0 || index >= vectorCount)
            {
                throw new ArgumentOutOfRangeException(nameof(index), 
                    $"Vector index must be between 0 and {vectorCount - 1}");
            }
            
            int offset = index * VECTOR_SIZE;
            
            float x = BitConverter.ToSingle(_data, offset);
            float y = BitConverter.ToSingle(_data, offset + 4);
            float z = BitConverter.ToSingle(_data, offset + 8);
            float w = BitConverter.ToSingle(_data, offset + 12);
            
            return new Vector4(x, y, z, w);
        }
        
        /// <summary>
        /// Parses the chunk data from the binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to parse from.</param>
        /// <param name="size">The size of the chunk data.</param>
        public override void Parse(BinaryReader reader, uint size)
        {
            _data = reader.ReadBytes((int)size);
            
            // Validate that the data size is divisible by 12 (size of a vector: 3 * 4 bytes)
            if (_data.Length % VECTOR_SIZE != 0)
            {
                LogParsingError($"Invalid MSCN chunk data size: {_data.Length}. Should be divisible by {VECTOR_SIZE}.");
            }
        }
        
        /// <summary>
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            return $"{SIGNATURE} Chunk: {GetVectorCount()} vectors";
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
} 