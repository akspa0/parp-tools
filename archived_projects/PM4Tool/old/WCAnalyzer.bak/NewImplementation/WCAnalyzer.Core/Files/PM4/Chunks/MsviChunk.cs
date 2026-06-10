using System;
using Microsoft.Extensions.Logging;
using System.IO;
using WCAnalyzer.Core.Common.Extensions;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.PM4.Chunks
{
    /// <summary>
    /// MSVI chunk for PM4 files. Contains indices into the MSVT chunk.
    /// </summary>
    public class MsviChunk : PM4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MSVI")
        /// </summary>
        public const string SIGNATURE = "MSVI";
        
        /// <summary>
        /// Size of a single index in bytes
        /// </summary>
        private const int INDEX_SIZE = 4; // uint32 = 4 bytes
        
        /// <summary>
        /// The raw chunk data.
        /// </summary>
        private byte[] _data;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MsviChunk"/> class.
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MsviChunk(ILogger? logger = null)
            : base(SIGNATURE, new byte[0], logger)
        {
            _data = new byte[0];
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MsviChunk"/> class with the specified data.
        /// </summary>
        /// <param name="data">The raw chunk data.</param>
        /// <param name="logger">Optional logger</param>
        public MsviChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
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
            
            // Validate that the data size is divisible by 4 (size of an index)
            if (_data.Length % 4 != 0)
            {
                LogParsingError($"Invalid MSVI chunk data size: {_data.Length}. Should be divisible by 4.");
            }
        }
        
        /// <summary>
        /// Gets the number of indices in the chunk
        /// </summary>
        /// <returns>Number of complete indices</returns>
        public int GetIndexCount()
        {
            return _data.Length / INDEX_SIZE;
        }
        
        /// <summary>
        /// Gets an index at the specified position
        /// </summary>
        /// <param name="index">Index position (0-based)</param>
        /// <returns>Index value</returns>
        /// <exception cref="ArgumentOutOfRangeException">If index is out of range</exception>
        public uint GetIndex(int index)
        {
            int indexCount = GetIndexCount();
            if (index < 0 || index >= indexCount)
            {
                throw new ArgumentOutOfRangeException(nameof(index), $"Index must be between 0 and {indexCount - 1}");
            }
            
            int offset = index * INDEX_SIZE;
            return BitConverter.ToUInt32(_data, offset);
        }
        
        /// <summary>
        /// Gets all indices as an array.
        /// </summary>
        /// <returns>An array of all indices.</returns>
        public uint[] GetAllIndices()
        {
            uint[] indices = new uint[GetIndexCount()];
            for (int i = 0; i < GetIndexCount(); i++)
            {
                indices[i] = GetIndex(i);
            }
            return indices;
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
            int indexCount = GetIndexCount();
            return $"{SIGNATURE} Chunk: {indexCount} index/indices";
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