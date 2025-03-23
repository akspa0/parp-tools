using System;
using System.IO;
using System.Numerics;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.PM4.Chunks
{
    /// <summary>
    /// MSHD chunk - Contains mesh header information for PM4 format
    /// Structure:
    /// - materialId (uint32)
    /// - [unknown field] (uint32)
    /// - flags (uint32)
    /// - [unknown field] (uint32)
    /// </summary>
    public class MshdChunk : PM4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MSHD")
        /// </summary>
        public const string SIGNATURE = "MSHD";
        
        /// <summary>
        /// Expected size of mesh header data in bytes
        /// </summary>
        private const int HEADER_SIZE = 16; // 4 uint32 fields
        
        /// <summary>
        /// Creates a new MSHD chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MshdChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Validate size
            if (_data.Length != HEADER_SIZE)
            {
                Logger?.LogWarning($"MSHD chunk has invalid size: {_data.Length} bytes. Expected {HEADER_SIZE} bytes.");
            }
        }
        
        /// <summary>
        /// Creates a new empty MSHD chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MshdChunk(ILogger? logger = null)
            : base(SIGNATURE, new byte[HEADER_SIZE], logger)
        {
        }
        
        /// <summary>
        /// Gets the material ID from the mesh header
        /// </summary>
        /// <returns>Material ID</returns>
        public uint GetMaterialId()
        {
            if (_data.Length < 4)
            {
                Logger?.LogWarning("MSHD chunk data is too small to contain a valid material ID. Returning 0.");
                return 0;
            }
            
            return BitConverter.ToUInt32(_data, 0);
        }
        
        /// <summary>
        /// Gets the flags field from the mesh header
        /// </summary>
        /// <returns>Flags value</returns>
        public uint GetFlags()
        {
            if (_data.Length < 12)
            {
                Logger?.LogWarning("MSHD chunk data is too small to contain valid flags. Returning 0.");
                return 0;
            }
            
            return BitConverter.ToUInt32(_data, 8);
        }
        
        /// <summary>
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            return $"{SIGNATURE} Chunk: MaterialId={GetMaterialId()}, Flags=0x{GetFlags():X8}";
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