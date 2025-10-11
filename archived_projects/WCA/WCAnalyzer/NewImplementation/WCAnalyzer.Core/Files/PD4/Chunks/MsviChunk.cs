using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.Logging;
using System.IO;
using WCAnalyzer.Core.Common.Extensions;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.PD4.Chunks
{
    /// <summary>
    /// MSVI chunk - Contains indices into the MSVT chunk for PD4 format
    /// </summary>
    public class MsviChunk : PD4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MSVI")
        /// </summary>
        public const string SIGNATURE = "MSVI";
        
        /// <summary>
        /// Size of a single index in bytes
        /// </summary>
        private const int INDEX_SIZE = 4;
        
        /// <summary>
        /// Creates a new MSVI chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MsviChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Validate size
            if (_data.Length % INDEX_SIZE != 0)
            {
                Logger?.LogWarning($"MSVI chunk has invalid size: {_data.Length} bytes. Should be divisible by {INDEX_SIZE}.");
            }
        }
        
        /// <summary>
        /// Creates a new empty MSVI chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MsviChunk(ILogger? logger = null)
            : base(SIGNATURE, new byte[0], logger)
        {
        }
        
        /// <summary>
        /// Gets the number of indices in the chunk
        /// </summary>
        /// <returns>Number of indices</returns>
        public int GetIndexCount()
        {
            return _data.Length / INDEX_SIZE;
        }
        
        /// <summary>
        /// Gets an index at the specified position
        /// </summary>
        /// <param name="position">Position (0-based)</param>
        /// <returns>Index value as uint</returns>
        /// <exception cref="ArgumentOutOfRangeException">If position is out of range</exception>
        public uint GetIndex(int position)
        {
            int indexCount = GetIndexCount();
            
            if (position < 0 || position >= indexCount)
            {
                throw new ArgumentOutOfRangeException(nameof(position), 
                    $"Index position must be between 0 and {indexCount - 1}");
            }
            
            int offset = position * INDEX_SIZE;
            return BitConverter.ToUInt32(_data, offset);
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
} 