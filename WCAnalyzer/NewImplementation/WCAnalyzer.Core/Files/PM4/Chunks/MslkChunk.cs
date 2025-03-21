using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.PM4.Chunks
{
    /// <summary>
    /// MSLK chunk - Contains link data between objects in the PM4 format
    /// These links typically connect objects or define relationships between components
    /// </summary>
    public class MslkChunk : PM4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MSLK")
        /// </summary>
        public const string SIGNATURE = "MSLK";
        
        /// <summary>
        /// Size of a basic link record in bytes
        /// </summary>
        private const int LINK_RECORD_SIZE = 8; // Typically 2 uint32 identifiers
        
        /// <summary>
        /// Creates a new MSLK chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MslkChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Check if data size is a multiple of LINK_RECORD_SIZE
            if (_data.Length % LINK_RECORD_SIZE != 0)
            {
                Logger?.LogWarning($"MSLK chunk has irregular size: {_data.Length} bytes. Not a multiple of {LINK_RECORD_SIZE} bytes per link record.");
            }
        }
        
        /// <summary>
        /// Creates a new empty MSLK chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MslkChunk(ILogger? logger = null)
            : base(SIGNATURE, Array.Empty<byte>(), logger)
        {
        }
        
        /// <summary>
        /// Gets the number of link records in this chunk
        /// </summary>
        /// <returns>Number of link records</returns>
        public int GetLinkCount()
        {
            return _data.Length / LINK_RECORD_SIZE;
        }
        
        /// <summary>
        /// Gets source and target IDs for a link at the specified index
        /// </summary>
        /// <param name="index">Index of the link record</param>
        /// <returns>Tuple with source and target IDs</returns>
        public (uint SourceId, uint TargetId) GetLink(int index)
        {
            int linkCount = GetLinkCount();
            
            if (index < 0 || index >= linkCount)
            {
                Logger?.LogWarning($"Link index {index} out of range [0-{linkCount - 1}]. Returning zeros.");
                return (0, 0);
            }
            
            int offset = index * LINK_RECORD_SIZE;
            
            uint sourceId = BitConverter.ToUInt32(_data, offset);
            uint targetId = BitConverter.ToUInt32(_data, offset + 4);
            
            return (sourceId, targetId);
        }
        
        /// <summary>
        /// Adds a link to the chunk
        /// </summary>
        /// <param name="sourceId">Source identifier</param>
        /// <param name="targetId">Target identifier</param>
        public void AddLink(uint sourceId, uint targetId)
        {
            byte[] newData = new byte[_data.Length + LINK_RECORD_SIZE];
            
            // Copy existing data
            Array.Copy(_data, 0, newData, 0, _data.Length);
            
            // Add new link data
            int offset = _data.Length;
            Array.Copy(BitConverter.GetBytes(sourceId), 0, newData, offset, 4);
            Array.Copy(BitConverter.GetBytes(targetId), 0, newData, offset + 4, 4);
            
            // Update data
            _data = newData;
        }
        
        /// <summary>
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            return $"{SIGNATURE} Chunk: {GetLinkCount()} links";
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