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
    /// MCRC chunk for PD4 files. Contains a CRC value or placeholder.
    /// </summary>
    public class McrcChunk : PD4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MCRC")
        /// </summary>
        public const string SIGNATURE = "MCRC";
        
        /// <summary>
        /// Size of CRC value in bytes
        /// </summary>
        private const int CRC_SIZE = 4;
        
        /// <summary>
        /// Gets or sets the CRC value.
        /// </summary>
        public uint Value { get; set; }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="McrcChunk"/> class.
        /// </summary>
        public McrcChunk() : base(SIGNATURE)
        {
            Value = 0;
        }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="McrcChunk"/> class with the specified value.
        /// </summary>
        /// <param name="value">The CRC value.</param>
        public McrcChunk(uint value) : base(SIGNATURE)
        {
            Value = value;
        }
        
        /// <summary>
        /// Creates a new MCRC chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public McrcChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Validate size
            if (_data.Length != CRC_SIZE)
            {
                Logger?.LogWarning($"MCRC chunk has invalid size: {_data.Length} bytes. Expected {CRC_SIZE} bytes.");
            }
        }
        
        /// <summary>
        /// Creates a new MCRC chunk with a specified CRC value
        /// </summary>
        /// <param name="crcValue">The CRC value</param>
        /// <param name="logger">Optional logger</param>
        public McrcChunk(uint crcValue, ILogger? logger = null)
            : base(SIGNATURE, BitConverter.GetBytes(crcValue), logger)
        {
        }
        
        /// <summary>
        /// Parses the chunk data from the binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to parse from.</param>
        /// <param name="size">The size of the chunk data.</param>
        public override void Parse(BinaryReader reader, uint size)
        {
            if (size != 4)
            {
                LogParsingError($"Invalid MCRC chunk size: {size}. Expected 4 bytes.");
            }
            
            Value = reader.ReadUInt32();
        }
        
        /// <summary>
        /// Writes the chunk data to the specified binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        public override void Write(BinaryWriter writer)
        {
            WriteChunkHeader(writer);
            writer.Write(Value);
        }
        
        /// <summary>
        /// Gets the CRC value from the chunk
        /// </summary>
        /// <returns>CRC value as uint</returns>
        public uint GetCrcValue()
        {
            if (_data.Length < CRC_SIZE)
            {
                Logger?.LogWarning("MCRC chunk data is too small to contain a valid CRC value. Returning 0.");
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
            return $"{SIGNATURE} Chunk: CRC=0x{GetCrcValue():X8}";
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