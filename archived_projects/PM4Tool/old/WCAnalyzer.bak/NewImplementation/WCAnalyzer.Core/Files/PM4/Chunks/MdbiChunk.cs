using System;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.PM4.Chunks
{
    /// <summary>
    /// MDBI chunk - Destructible Building Index, a sub-chunk of MDBH
    /// Contains index data for destructible buildings
    /// </summary>
    public class MdbiChunk : PM4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MDBI")
        /// </summary>
        public const string SIGNATURE = "MDBI";
        
        /// <summary>
        /// Size of a single index entry in bytes (4 bytes per uint32)
        /// </summary>
        private const int INDEX_SIZE = 4;
        
        /// <summary>
        /// Creates a new MDBI chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MdbiChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Check if data size is multiple of INDEX_SIZE
            if (_data.Length % INDEX_SIZE != 0)
            {
                Logger?.LogWarning($"MDBI chunk has irregular size: {_data.Length} bytes. Not a multiple of {INDEX_SIZE} bytes per index.");
            }
        }
        
        /// <summary>
        /// Creates a new empty MDBI chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MdbiChunk(ILogger? logger = null)
            : base(SIGNATURE, Array.Empty<byte>(), logger)
        {
        }
        
        /// <summary>
        /// Gets the number of indices in this chunk
        /// </summary>
        /// <returns>Number of indices</returns>
        public int GetIndexCount()
        {
            return _data.Length / INDEX_SIZE;
        }
        
        /// <summary>
        /// Gets the index value at the specified position
        /// </summary>
        /// <param name="position">Position of the index</param>
        /// <returns>The index value as a uint32</returns>
        public uint GetIndex(int position)
        {
            int indexCount = GetIndexCount();
            
            if (position < 0 || position >= indexCount)
            {
                Logger?.LogWarning($"Index position {position} out of range [0-{indexCount - 1}]. Returning 0.");
                return 0;
            }
            
            int offset = position * INDEX_SIZE;
            return BitConverter.ToUInt32(_data, offset);
        }
        
        /// <summary>
        /// Adds an index to the chunk
        /// </summary>
        /// <param name="index">The index value to add</param>
        public void AddIndex(uint index)
        {
            byte[] newData = new byte[_data.Length + INDEX_SIZE];
            
            // Copy existing data
            Array.Copy(_data, 0, newData, 0, _data.Length);
            
            // Add new index data
            Array.Copy(BitConverter.GetBytes(index), 0, newData, _data.Length, INDEX_SIZE);
            
            // Update data
            _data = newData;
        }
        
        /// <summary>
        /// Sets the index value at the specified position
        /// </summary>
        /// <param name="position">Position of the index to set</param>
        /// <param name="index">The new index value</param>
        /// <returns>True if the index was set, false if position was out of range</returns>
        public bool SetIndex(int position, uint index)
        {
            int indexCount = GetIndexCount();
            
            if (position < 0 || position >= indexCount)
            {
                Logger?.LogWarning($"Cannot set index at position {position}. Valid range is [0-{indexCount - 1}].");
                return false;
            }
            
            int offset = position * INDEX_SIZE;
            Array.Copy(BitConverter.GetBytes(index), 0, _data, offset, INDEX_SIZE);
            return true;
        }
        
        /// <summary>
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            return $"{SIGNATURE} Chunk: {GetIndexCount()} indices";
        }
        
        /// <summary>
        /// Writes this chunk to a byte array
        /// </summary>
        /// <returns>Byte array containing chunk data</returns>
        public override byte[] Write()
        {
            return _data;
        }
    }
} 