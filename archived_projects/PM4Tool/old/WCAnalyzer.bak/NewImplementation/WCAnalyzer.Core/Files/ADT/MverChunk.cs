using System;
using System.IO;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.ADT
{
    /// <summary>
    /// MVER chunk - contains version information
    /// </summary>
    public class MverChunk : ADTChunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MVER";
        
        /// <summary>
        /// Version number
        /// </summary>
        public uint Version { get; private set; }
        
        /// <summary>
        /// Creates a new MVER chunk
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MverChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
        }
        
        /// <summary>
        /// Creates a new MVER chunk with specified version
        /// </summary>
        /// <param name="version">The version number (usually 18)</param>
        /// <param name="logger">Optional logger</param>
        public MverChunk(uint version, ILogger? logger = null)
            : base(SIGNATURE, new byte[4], logger)
        {
            Version = version;
            Data = BitConverter.GetBytes(version);
            IsParsed = true;
        }
        
        /// <summary>
        /// Parse the chunk data
        /// </summary>
        /// <returns>True if parsing succeeded, false otherwise</returns>
        public override bool Parse()
        {
            try
            {
                if (Data.Length < 4)
                {
                    LogError($"MVER chunk data is too small: {Data.Length} bytes");
                    return false;
                }
                
                Version = BitConverter.ToUInt32(Data, 0);
                
                if (Version != 18)
                {
                    LogWarning($"Unexpected ADT version: {Version} (expected 18)");
                }
                
                IsParsed = true;
                return true;
            }
            catch (Exception ex)
            {
                LogError($"Error parsing MVER chunk: {ex.Message}");
                return false;
            }
        }
        
        /// <summary>
        /// Write the chunk data
        /// </summary>
        /// <returns>Binary data for this chunk</returns>
        public override byte[] Write()
        {
            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter writer = new BinaryWriter(ms))
            {
                writer.Write(Version);
                return ms.ToArray();
            }
        }
        
        /// <summary>
        /// Returns a string representation of this chunk
        /// </summary>
        public override string ToString()
        {
            return $"{SIGNATURE} (Version: {Version})";
        }
    }
} 