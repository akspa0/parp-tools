using System;
using System.Numerics;
using Microsoft.Extensions.Logging;
using System.IO;
using WCAnalyzer.Core.Common.Extensions;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.PM4.Chunks
{
    /// <summary>
    /// MSCN chunk - Contains vector data not related to MSPV and MSLK
    /// Each vector consists of 4 float values (16 bytes)
    /// </summary>
    public class MscnChunk : PM4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MSCN")
        /// </summary>
        public const string SIGNATURE = "MSCN";
        
        /// <summary>
        /// Size of a single vector in bytes
        /// </summary>
        private const int VECTOR_SIZE = 16; // 4 floats = 16 bytes
        
        /// <summary>
        /// The raw chunk data.
        /// </summary>
        private byte[] _data;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MscnChunk"/> class.
        /// </summary>
        public MscnChunk() : base(SIGNATURE)
        {
            _data = new byte[0];
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MscnChunk"/> class with the specified data.
        /// </summary>
        /// <param name="data">The raw chunk data.</param>
        public MscnChunk(byte[] data) : base(SIGNATURE)
        {
            _data = data;
        }
        
        /// <summary>
        /// Parses the chunk data from the binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to parse from.</param>
        /// <param name="size">The size of the chunk data.</param>
        public override void Parse(BinaryReader reader, uint size)
        {
            _data = reader.ReadBytes((int)size);
            
            // Validate that the data size is divisible by 16 (size of a vector: 4 * 4 bytes)
            if (_data.Length % 16 != 0)
            {
                LogParsingError($"Invalid MSCN chunk data size: {_data.Length}. Should be divisible by 16.");
            }
        }
        
        /// <summary>
        /// Gets the number of vectors in the chunk
        /// </summary>
        /// <returns>Number of complete vectors</returns>
        public int GetVectorCount()
        {
            if (_data.Length < VECTOR_SIZE)
            {
                return 0;
            }
            
            return _data.Length / VECTOR_SIZE;
        }
        
        /// <summary>
        /// Gets a vector at the specified index
        /// </summary>
        /// <param name="index">Vector index (0-based)</param>
        /// <returns>Vector representation</returns>
        /// <exception cref="ArgumentOutOfRangeException">If index is out of range</exception>
        public Vector4 GetVector(int index)
        {
            int vectorCount = GetVectorCount();
            if (index < 0 || index >= vectorCount)
            {
                throw new ArgumentOutOfRangeException(nameof(index), $"Vector index must be between 0 and {vectorCount - 1}");
            }
            
            int offset = index * VECTOR_SIZE;
            float x = BitConverter.ToSingle(_data, offset);
            float y = BitConverter.ToSingle(_data, offset + 4);
            float z = BitConverter.ToSingle(_data, offset + 8);
            float w = BitConverter.ToSingle(_data, offset + 12);
            
            return new Vector4(x, y, z, w);
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
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            int vectorCount = GetVectorCount();
            return $"{SIGNATURE} Chunk: {vectorCount} vector(s)";
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