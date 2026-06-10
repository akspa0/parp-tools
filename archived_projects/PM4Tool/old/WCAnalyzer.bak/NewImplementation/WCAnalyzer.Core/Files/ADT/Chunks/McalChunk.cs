using Microsoft.Extensions.Logging;
using System;
using System.IO;
using WCAnalyzer.Core.Common.Interfaces;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents an MCAL chunk in an ADT file, containing alpha maps for texture blending.
    /// </summary>
    public class McalChunk : ADTChunk
    {
        /// <summary>
        /// The MCAL chunk signature
        /// </summary>
        public const string SIGNATURE = "MCAL";
        
        /// <summary>
        /// Flag indicating if the alpha map uses the "big alpha" format.
        /// </summary>
        private bool _useBigAlpha = false;
        
        /// <summary>
        /// Gets the raw alpha map data.
        /// </summary>
        public byte[] AlphaMapData { get; private set; } = Array.Empty<byte>();

        /// <summary>
        /// Initializes a new instance of the <see cref="McalChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <param name="logger">Optional logger.</param>
        public McalChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
        }

        /// <summary>
        /// Parses the chunk data
        /// </summary>
        public override void Parse()
        {
            if (Data == null || Data.Length == 0)
            {
                AddError("No data to parse for MCAL chunk");
                return;
            }

            try
            {
                // The MCAL chunk simply contains raw alpha map data
                AlphaMapData = Data;
                
                // We'll check the size to determine if it's using the big alpha format (4096 bytes)
                // or the standard format (2048 bytes)
                if (AlphaMapData.Length >= 4096)
                {
                    _useBigAlpha = true;
                    Logger?.LogDebug($"MCAL: Using big alpha format (4096 bytes)");
                }
                else
                {
                    Logger?.LogDebug($"MCAL: Using standard alpha format (2048 bytes)");
                }
                
                Logger?.LogDebug($"MCAL: Parsed {AlphaMapData.Length} bytes of alpha map data");
            }
            catch (Exception ex)
            {
                AddError($"Error parsing MCAL chunk: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Gets the alpha map data for a specific texture layer.
        /// </summary>
        /// <param name="offsetInMcal">The offset in the MCAL chunk where the layer's alpha map data begins.</param>
        /// <param name="isCompressed">Whether the alpha map is compressed.</param>
        /// <returns>A byte array containing the alpha map values (0-255) for the specified layer.</returns>
        public byte[] GetAlphaMap(uint offsetInMcal, bool isCompressed)
        {
            // Alpha maps are used for blending textures
            // Each alpha map is 64x64 pixels, with each pixel having a value from 0-255
            byte[] alphaMap = new byte[64 * 64];
            
            // If we don't have any alpha map data or the offset is invalid, return an empty alpha map
            if (AlphaMapData.Length == 0 || offsetInMcal >= AlphaMapData.Length)
            {
                // Fill with zeros (fully transparent)
                Logger?.LogWarning($"MCAL: Invalid offset {offsetInMcal} for alpha map");
                return alphaMap;
            }
            
            // Get the alpha map data starting at the specified offset
            int offset = (int)offsetInMcal;
            byte[] alphaBuffer = new byte[AlphaMapData.Length - offset];
            Array.Copy(AlphaMapData, offset, alphaBuffer, 0, alphaBuffer.Length);
            
            // Check if the alpha map is compressed
            if (isCompressed)
            {
                return ReadCompressedAlpha(alphaBuffer);
            }
            else if (_useBigAlpha)
            {
                return ReadBigAlpha(alphaBuffer);
            }
            else
            {
                return ReadUncompressedAlpha(alphaBuffer);
            }
        }
        
        /// <summary>
        /// Reads a compressed alpha map.
        /// </summary>
        /// <param name="alphaBuffer">The compressed alpha map data.</param>
        /// <returns>The decompressed alpha map as a 64x64 byte array.</returns>
        private byte[] ReadCompressedAlpha(byte[] alphaBuffer)
        {
            byte[] alphaMap = new byte[64 * 64];
            
            int offInner = 0;
            int offOuter = 0;
            
            // Decompress the alpha map using run-length encoding
            while (offOuter < 4096 && offInner < alphaBuffer.Length)
            {
                bool fill = (alphaBuffer[offInner] & 0x80) != 0; // Check if the high bit is set
                int count = (alphaBuffer[offInner] & 0x7F);     // Get the count (low 7 bits)
                offInner++;
                
                if (offInner >= alphaBuffer.Length)
                    break;
                
                for (int k = 0; k < count && offOuter < 4096; k++)
                {
                    if (offOuter >= alphaMap.Length)
                        break;
                    
                    alphaMap[offOuter] = alphaBuffer[offInner];
                    offOuter++;
                    
                    if (!fill && offInner < alphaBuffer.Length - 1)
                    {
                        offInner++;
                    }
                }
                
                if (fill && offInner < alphaBuffer.Length - 1)
                {
                    offInner++;
                }
            }
            
            Logger?.LogDebug($"MCAL: Decompressed alpha map (compressed format)");
            return alphaMap;
        }
        
        /// <summary>
        /// Reads a "big alpha" format alpha map (4096 bytes, one byte per pixel).
        /// </summary>
        /// <param name="alphaBuffer">The big alpha map data.</param>
        /// <returns>The alpha map as a 64x64 byte array.</returns>
        private byte[] ReadBigAlpha(byte[] alphaBuffer)
        {
            byte[] alphaMap = new byte[64 * 64];
            
            // Copy the data directly (one byte per pixel)
            int count = Math.Min(alphaBuffer.Length, alphaMap.Length);
            Array.Copy(alphaBuffer, alphaMap, count);
            
            // Copy the last row to fix potential issues
            if (count >= 63 * 64)
            {
                Array.Copy(alphaMap, 62 * 64, alphaMap, 63 * 64, 64);
            }
            
            Logger?.LogDebug($"MCAL: Read alpha map (big alpha format)");
            return alphaMap;
        }
        
        /// <summary>
        /// Reads an uncompressed alpha map (2048 bytes, 4 bits per pixel).
        /// </summary>
        /// <param name="alphaBuffer">The uncompressed alpha map data.</param>
        /// <returns>The alpha map as a 64x64 byte array.</returns>
        private byte[] ReadUncompressedAlpha(byte[] alphaBuffer)
        {
            byte[] alphaMap = new byte[64 * 64];
            
            // Each byte in the buffer contains two 4-bit alpha values
            int inner = 0;
            int outer = 0;
            
            for (int j = 0; j < 64 && outer < alphaBuffer.Length; j++)
            {
                for (int i = 0; i < 32 && outer < alphaBuffer.Length; i++)
                {
                    // First 4 bits (low nibble)
                    if (inner < alphaMap.Length)
                    {
                        alphaMap[inner] = (byte)((255 * (alphaBuffer[outer] & 0x0F)) / 0x0F);
                        inner++;
                    }
                    
                    // Second 4 bits (high nibble)
                    if (inner < alphaMap.Length && (i != 31 || j == 63))
                    {
                        alphaMap[inner] = (byte)((255 * ((alphaBuffer[outer] & 0xF0) >> 4)) / 0x0F);
                        inner++;
                    }
                    
                    outer++;
                }
            }
            
            // Copy the last row to fix potential issues
            if (alphaMap.Length >= 64 * 64)
            {
                Array.Copy(alphaMap, 62 * 64, alphaMap, 63 * 64, 64);
            }
            
            Logger?.LogDebug($"MCAL: Read alpha map (standard format)");
            return alphaMap;
        }

        /// <summary>
        /// Writes the chunk data to the specified writer
        /// </summary>
        /// <param name="writer">The binary writer to write to</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                AddError("Cannot write to null writer");
                return;
            }

            try
            {
                // Just write the raw alpha map data
                writer.Write(AlphaMapData);
            }
            catch (Exception ex)
            {
                AddError($"Error writing MCAL chunk: {ex.Message}");
            }
        }
    }
} 