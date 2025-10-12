using System;
using System.IO;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.ADT
{
    /// <summary>
    /// MHDR chunk - ADT header information
    /// </summary>
    public class MhdrChunk : ADTChunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MHDR";
        
        /// <summary>
        /// Flags for this ADT
        /// </summary>
        public uint Flags { get; private set; }
        
        /// <summary>
        /// Offset to MCIN chunk
        /// </summary>
        public uint McInOffset { get; private set; }
        
        /// <summary>
        /// Offset to MTEX chunk
        /// </summary>
        public uint MTexOffset { get; private set; }
        
        /// <summary>
        /// Offset to MMDX chunk
        /// </summary>
        public uint MMdxOffset { get; private set; }
        
        /// <summary>
        /// Offset to MMID chunk
        /// </summary>
        public uint MMidOffset { get; private set; }
        
        /// <summary>
        /// Offset to MWMO chunk
        /// </summary>
        public uint MWmoOffset { get; private set; }
        
        /// <summary>
        /// Offset to MWID chunk
        /// </summary>
        public uint MWidOffset { get; private set; }
        
        /// <summary>
        /// Offset to MDDF chunk
        /// </summary>
        public uint MddfOffset { get; private set; }
        
        /// <summary>
        /// Offset to MODF chunk
        /// </summary>
        public uint ModfOffset { get; private set; }
        
        /// <summary>
        /// Offset to MH2O chunk
        /// </summary>
        public uint MH2OOffset { get; private set; }
        
        /// <summary>
        /// Offset to MTXF chunk
        /// </summary>
        public uint MtxfOffset { get; private set; }
        
        /// <summary>
        /// Creates a new MHDR chunk
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MhdrChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
        }
        
        /// <summary>
        /// Parse the chunk data
        /// </summary>
        /// <returns>True if parsing succeeded, false otherwise</returns>
        public override bool Parse()
        {
            try
            {
                if (Data.Length < 64)  // MHDR is 64 bytes in size
                {
                    LogError($"MHDR chunk data is too small: {Data.Length} bytes (expected 64)");
                    return false;
                }
                
                using (MemoryStream ms = new MemoryStream(Data))
                using (BinaryReader reader = new BinaryReader(ms))
                {
                    Flags = reader.ReadUInt32();
                    McInOffset = reader.ReadUInt32();
                    MTexOffset = reader.ReadUInt32();
                    MMdxOffset = reader.ReadUInt32();
                    MMidOffset = reader.ReadUInt32();
                    MWmoOffset = reader.ReadUInt32();
                    MWidOffset = reader.ReadUInt32();
                    MddfOffset = reader.ReadUInt32();
                    ModfOffset = reader.ReadUInt32();
                    MH2OOffset = reader.ReadUInt32();
                    MtxfOffset = reader.ReadUInt32();
                    
                    // Skip unused data (4 uint values)
                    reader.BaseStream.Position += 16;
                }
                
                IsParsed = true;
                return true;
            }
            catch (Exception ex)
            {
                LogError($"Error parsing MHDR chunk: {ex.Message}");
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
                writer.Write(Flags);
                writer.Write(McInOffset);
                writer.Write(MTexOffset);
                writer.Write(MMdxOffset);
                writer.Write(MMidOffset);
                writer.Write(MWmoOffset);
                writer.Write(MWidOffset);
                writer.Write(MddfOffset);
                writer.Write(ModfOffset);
                writer.Write(MH2OOffset);
                writer.Write(MtxfOffset);
                
                // Write unused data (4 uint values set to 0)
                writer.Write(0);
                writer.Write(0);
                writer.Write(0);
                writer.Write(0);
                
                return ms.ToArray();
            }
        }
        
        /// <summary>
        /// Returns a string representation of this chunk
        /// </summary>
        public override string ToString()
        {
            return $"{SIGNATURE} (Flags: 0x{Flags:X})";
        }
    }
} 