using System;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.PD4.Chunks
{
    /// <summary>
    /// MSUR chunk - Contains surface/material data for the PD4 format
    /// This chunk typically stores material properties, textures, and rendering parameters
    /// </summary>
    public class MsurChunk : PD4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MSUR")
        /// </summary>
        public const string SIGNATURE = "MSUR";
        
        /// <summary>
        /// Creates a new MSUR chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MsurChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
        }
        
        /// <summary>
        /// Creates a new empty MSUR chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MsurChunk(ILogger? logger = null)
            : base(SIGNATURE, Array.Empty<byte>(), logger)
        {
        }
        
        /// <summary>
        /// Gets the size of the surface data
        /// </summary>
        /// <returns>Size in bytes</returns>
        public int GetDataSize()
        {
            return _data.Length;
        }
        
        /// <summary>
        /// Attempts to get the material name if present
        /// The name is typically stored as a null-terminated string in the data
        /// </summary>
        /// <returns>Material name or empty string if not found</returns>
        public string GetMaterialName()
        {
            // Try to find a null-terminated string at the beginning of the data
            int nullTerminator = Array.IndexOf(_data, (byte)0);
            if (nullTerminator > 0)
            {
                try
                {
                    return System.Text.Encoding.ASCII.GetString(_data, 0, nullTerminator);
                }
                catch (Exception ex)
                {
                    Logger?.LogWarning($"Failed to extract material name: {ex.Message}");
                }
            }
            
            return string.Empty;
        }
        
        /// <summary>
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            string materialName = GetMaterialName();
            if (!string.IsNullOrEmpty(materialName))
            {
                return $"{SIGNATURE} Chunk: Material '{materialName}', {GetDataSize()} bytes";
            }
            
            return $"{SIGNATURE} Chunk: {GetDataSize()} bytes";
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