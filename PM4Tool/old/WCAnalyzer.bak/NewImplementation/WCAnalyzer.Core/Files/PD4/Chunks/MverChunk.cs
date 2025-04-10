using System;
using System.IO;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.PD4.Chunks
{
    /// <summary>
    /// MVER chunk - Contains version information for PD4 format
    /// </summary>
    public class MverChunk : PD4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MVER")
        /// </summary>
        public const string SIGNATURE = "MVER";
        
        /// <summary>
        /// Expected size of version data in bytes
        /// </summary>
        private const int VERSION_SIZE = 4;
        
        /// <summary>
        /// Creates a new MVER chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MverChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Validate size
            if (_data.Length != VERSION_SIZE)
            {
                Logger?.LogWarning($"MVER chunk has invalid size: {_data.Length} bytes. Expected {VERSION_SIZE} bytes.");
            }
        }
        
        /// <summary>
        /// Creates a new MVER chunk with the specified version
        /// </summary>
        /// <param name="version">Version number</param>
        /// <param name="logger">Optional logger</param>
        public MverChunk(uint version, ILogger? logger = null)
            : base(SIGNATURE, BitConverter.GetBytes(version), logger)
        {
        }
        
        /// <summary>
        /// Creates a new empty MVER chunk with version 0
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MverChunk(ILogger? logger = null)
            : this(0, logger)
        {
        }
        
        /// <summary>
        /// Gets the version number from the chunk
        /// </summary>
        /// <returns>Version number</returns>
        public uint GetVersion()
        {
            if (_data.Length < VERSION_SIZE)
            {
                Logger?.LogWarning("MVER chunk data is too small to contain a valid version. Returning 0.");
                return 0;
            }
            
            return BitConverter.ToUInt32(_data, 0);
        }
        
        /// <summary>
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            return $"{SIGNATURE} Chunk: Version {GetVersion()}";
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