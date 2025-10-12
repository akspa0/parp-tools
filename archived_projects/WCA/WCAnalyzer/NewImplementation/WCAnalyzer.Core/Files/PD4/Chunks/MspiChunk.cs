using System;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.PD4.Chunks
{
    /// <summary>
    /// MSPI chunk - Contains polygon indices for the PD4 format
    /// Indices are typically 16-bit unsigned values that reference vertices
    /// </summary>
    public class MspiChunk : PD4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MSPI")
        /// </summary>
        public const string SIGNATURE = "MSPI";
        
        /// <summary>
        /// Size of each index in bytes (typically 2 bytes for a uint16)
        /// </summary>
        private const int INDEX_SIZE = 2;
        
        /// <summary>
        /// Creates a new MSPI chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MspiChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Check if data size is a multiple of INDEX_SIZE
            if (_data.Length % INDEX_SIZE != 0)
            {
                Logger?.LogWarning($"MSPI chunk has irregular size: {_data.Length} bytes. Not a multiple of {INDEX_SIZE} bytes per index.");
            }
        }
        
        /// <summary>
        /// Creates a new empty MSPI chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MspiChunk(ILogger? logger = null)
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
        /// Gets an index at the specified position
        /// </summary>
        /// <param name="position">Position of the index to retrieve</param>
        /// <returns>The index value at the specified position</returns>
        public ushort GetIndex(int position)
        {
            int indexCount = GetIndexCount();
            
            if (position < 0 || position >= indexCount)
            {
                Logger?.LogWarning($"Index position {position} out of range [0-{indexCount - 1}]. Returning 0.");
                return 0;
            }
            
            int offset = position * INDEX_SIZE;
            return BitConverter.ToUInt16(_data, offset);
        }
        
        /// <summary>
        /// Adds an index to the chunk
        /// </summary>
        /// <param name="index">Index value to add</param>
        public void AddIndex(ushort index)
        {
            byte[] newData = new byte[_data.Length + INDEX_SIZE];
            
            // Copy existing data
            Array.Copy(_data, 0, newData, 0, _data.Length);
            
            // Add new index
            byte[] indexBytes = BitConverter.GetBytes(index);
            Array.Copy(indexBytes, 0, newData, _data.Length, INDEX_SIZE);
            
            // Update data
            _data = newData;
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