using Microsoft.Extensions.Logging;
using System;
using System.Drawing;
using System.IO;
using System.Numerics;
using WCAnalyzer.Core.Common.Interfaces;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents an MCCV chunk in an ADT file, containing per-vertex colors for terrain.
    /// Available in Wrath of the Lich King and later versions.
    /// </summary>
    public class MccvChunk : ADTChunk
    {
        /// <summary>
        /// The MCCV chunk signature
        /// </summary>
        public const string SIGNATURE = "MCCV";
        
        /// <summary>
        /// The number of vertices in the grid (9x9 + 8x8 grid).
        /// </summary>
        public const int VertexCount = 145;
        
        /// <summary>
        /// Gets the per-vertex colors. Each color is an RGBA value (red, green, blue, alpha).
        /// </summary>
        public Color[] VertexColors { get; private set; } = new Color[VertexCount];
        
        /// <summary>
        /// Gets the raw vertex color data as bytes.
        /// </summary>
        public byte[] RawColorData { get; private set; } = Array.Empty<byte>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MccvChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <param name="logger">Optional logger.</param>
        public MccvChunk(byte[] data, ILogger? logger = null)
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
                AddError("No data to parse for MCCV chunk");
                return;
            }
            
            try
            {
                using (var ms = new MemoryStream(Data))
                using (var reader = new BinaryReader(ms))
                {
                    // Each vertex color is stored as 4 bytes (R, G, B, A)
                    RawColorData = new byte[VertexCount * 4];
                    
                    for (int i = 0; i < VertexCount; i++)
                    {
                        // Read the RGBA values
                        byte r = reader.ReadByte();
                        byte g = reader.ReadByte();
                        byte b = reader.ReadByte();
                        byte a = reader.ReadByte();
                        
                        // Store the raw data
                        int offset = i * 4;
                        RawColorData[offset] = r;
                        RawColorData[offset + 1] = g;
                        RawColorData[offset + 2] = b;
                        RawColorData[offset + 3] = a;
                        
                        // Create a Color object
                        VertexColors[i] = Color.FromArgb(a, r, g, b);
                    }
                    
                    Logger?.LogDebug($"MCCV: Parsed {VertexCount} vertex colors");
                }
            }
            catch (Exception ex)
            {
                AddError($"Error parsing MCCV chunk: {ex.Message}");
            }
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
                // Write the RawColorData if available, otherwise rebuild from VertexColors
                if (RawColorData.Length == VertexCount * 4)
                {
                    writer.Write(RawColorData);
                }
                else
                {
                    // Rebuild the raw data from the vertex colors
                    for (int i = 0; i < VertexCount; i++)
                    {
                        var color = VertexColors[i];
                        writer.Write(color.R);
                        writer.Write(color.G);
                        writer.Write(color.B);
                        writer.Write(color.A);
                    }
                }
                
                Logger?.LogDebug($"MCCV: Wrote {VertexCount} vertex colors");
            }
            catch (Exception ex)
            {
                AddError($"Error writing MCCV chunk: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Gets the vertex color at the specified grid coordinates.
        /// </summary>
        /// <param name="x">The x coordinate (0-8 for the 9x9 grid, 0-7 for the 8x8 grid).</param>
        /// <param name="y">The y coordinate (0-8 for the 9x9 grid, 0-7 for the 8x8 grid).</param>
        /// <param name="isCenter">True if accessing the 9x9 center grid, false for the 8x8 corner grid.</param>
        /// <returns>The color at the specified coordinates or default if coordinates are invalid.</returns>
        public Color GetColor(int x, int y, bool isCenter)
        {
            try
            {
                if (isCenter)
                {
                    if (x < 0 || x > 8 || y < 0 || y > 8)
                    {
                        AddError($"Center grid coordinates must be between 0 and 8, got ({x}, {y})");
                        return Color.Black;
                    }
                    
                    // The 9x9 grid is the first 81 values
                    return VertexColors[y * 9 + x];
                }
                else
                {
                    if (x < 0 || x > 7 || y < 0 || y > 7)
                    {
                        AddError($"Corner grid coordinates must be between 0 and 7, got ({x}, {y})");
                        return Color.Black;
                    }
                    
                    // The 8x8 grid is the last 64 values, offset by the 9x9 grid size
                    return VertexColors[81 + (y * 8 + x)];
                }
            }
            catch (Exception ex)
            {
                AddError($"Error retrieving color at ({x}, {y}, isCenter={isCenter}): {ex.Message}");
                return Color.Black;
            }
        }
        
        /// <summary>
        /// Gets the vertex color at the specified grid coordinates as a Vector4 (RGBA, values from 0 to 1).
        /// </summary>
        /// <param name="x">The x coordinate (0-8 for the 9x9 grid, 0-7 for the 8x8 grid).</param>
        /// <param name="y">The y coordinate (0-8 for the 9x9 grid, 0-7 for the 8x8 grid).</param>
        /// <param name="isCenter">True if accessing the 9x9 center grid, false for the 8x8 corner grid.</param>
        /// <returns>The color as a Vector4 (R, G, B, A).</returns>
        public Vector4 GetColorVector(int x, int y, bool isCenter)
        {
            Color color = GetColor(x, y, isCenter);
            
            return new Vector4(
                color.R / 255f,
                color.G / 255f,
                color.B / 255f,
                color.A / 255f
            );
        }
    }
} 