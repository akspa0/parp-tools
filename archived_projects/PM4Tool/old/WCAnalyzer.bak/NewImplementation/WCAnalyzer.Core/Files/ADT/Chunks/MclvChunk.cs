using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MCLV chunk - Contains vertex colors for terrain
    /// </summary>
    public class MclvChunk : ADTChunk
    {
        /// <summary>
        /// The MCLV chunk signature
        /// </summary>
        public const string SIGNATURE = "MCLV";

        /// <summary>
        /// Gets the vertex colors (ARGB format)
        /// </summary>
        public uint[] VertexColors { get; private set; } = Array.Empty<uint>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MclvChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MclvChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Parse();
        }

        /// <summary>
        /// Parses the MCLV chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MCLV chunk has no data");
                    return;
                }
                
                using var ms = new MemoryStream(Data);
                using var reader = new BinaryReader(ms);
                
                // Validate that the data size is a multiple of 4 bytes (uint)
                if (Data.Length % 4 != 0)
                {
                    Logger?.LogWarning($"MCLV chunk data size {Data.Length} is not a multiple of 4");
                }
                
                // Calculate how many color values we should have
                int count = Data.Length / 4;
                VertexColors = new uint[count];
                
                // Read color values (4 bytes each, ARGB format)
                for (int i = 0; i < count; i++)
                {
                    try
                    {
                        VertexColors[i] = reader.ReadUInt32();
                    }
                    catch (EndOfStreamException)
                    {
                        Logger?.LogWarning($"MCLV: Unexpected end of stream while reading color {i}");
                        break;
                    }
                }
                
                Logger?.LogDebug($"MCLV: Read {VertexColors.Length} vertex colors");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MCLV chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Writes the chunk data to a binary writer
        /// </summary>
        /// <param name="writer">The binary writer to write to</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                Logger?.LogError("Cannot write MCLV chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                writer.Write(VertexColors.Length * 4); // Size in bytes (4 bytes per color)
                
                // Write each color
                foreach (var color in VertexColors)
                {
                    writer.Write(color);
                }
                
                Logger?.LogDebug($"MCLV: Wrote {VertexColors.Length} vertex colors");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MCLV chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Gets the R, G, B, A components of a vertex color
        /// </summary>
        /// <param name="index">The vertex index</param>
        /// <returns>A tuple containing (A, R, G, B) values from 0-255</returns>
        public (byte A, byte R, byte G, byte B) GetColorComponents(int index)
        {
            if (index < 0 || index >= VertexColors.Length)
            {
                return (0, 0, 0, 0);
            }
            
            uint color = VertexColors[index];
            
            byte a = (byte)((color >> 24) & 0xFF);
            byte r = (byte)((color >> 16) & 0xFF);
            byte g = (byte)((color >> 8) & 0xFF);
            byte b = (byte)(color & 0xFF);
            
            return (a, r, g, b);
        }
    }
} 