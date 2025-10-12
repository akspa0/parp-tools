using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Numerics;
using WCAnalyzer.Core.Common.Interfaces;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents an MCNR chunk in an ADT file, containing vertex normal data for a map chunk.
    /// </summary>
    public class McnrChunk : ADTChunk
    {
        /// <summary>
        /// The MCNR chunk signature
        /// </summary>
        public const string SIGNATURE = "MCNR";

        /// <summary>
        /// The number of vertices in the normal grid (9x9 + 8x8 grid).
        /// </summary>
        public const int VertexCount = 145;
        
        /// <summary>
        /// The padding length for the MCNR chunk in WotLK format.
        /// </summary>
        private const int PaddingLength = 13;
        
        /// <summary>
        /// Gets the vertex normals. These are stored as unit vectors in a 9x9 + 8x8 grid.
        /// </summary>
        public Vector3[] VertexNormals { get; } = new Vector3[VertexCount];

        /// <summary>
        /// Gets the raw, compressed normal data. 
        /// Each normal is stored as 3 bytes, where each byte represents a signed fixed-point value in [-1,1].
        /// </summary>
        public byte[] RawNormalData { get; private set; } = Array.Empty<byte>();
        
        /// <summary>
        /// Gets the padding data for the MCNR chunk (13 bytes in WotLK format).
        /// </summary>
        public byte[] Padding { get; private set; } = Array.Empty<byte>();

        /// <summary>
        /// Initializes a new instance of the <see cref="McnrChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <param name="logger">Optional logger.</param>
        public McnrChunk(byte[] data, ILogger? logger = null)
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
                AddError("No data to parse for MCNR chunk");
                return;
            }

            try
            {
                using (var ms = new MemoryStream(Data))
                using (var reader = new BinaryReader(ms))
                {
                    // Each vertex normal is stored as 3 bytes (x, y, z), where each byte is a signed fixed-point value in [-1,1]
                    RawNormalData = new byte[VertexCount * 3];
                    
                    for (int i = 0; i < VertexCount; i++)
                    {
                        // Read the raw bytes
                        byte x = reader.ReadByte();
                        byte y = reader.ReadByte();
                        byte z = reader.ReadByte();
                        
                        // Store the raw data
                        RawNormalData[i * 3] = x;
                        RawNormalData[i * 3 + 1] = y;
                        RawNormalData[i * 3 + 2] = z;
                        
                        // Convert to Vector3 (each byte is converted to a float in the range [-1, 1])
                        VertexNormals[i] = new Vector3(
                            (x - 127f) / 127f,
                            (y - 127f) / 127f,
                            (z - 127f) / 127f
                        );
                        
                        // Normalize if necessary (should already be normalized, but just in case)
                        if (VertexNormals[i].Length() != 0)
                        {
                            VertexNormals[i] = Vector3.Normalize(VertexNormals[i]);
                        }
                    }
                    
                    // Read padding bytes (13 bytes in WotLK format)
                    if (reader.BaseStream.Position + PaddingLength <= reader.BaseStream.Length)
                    {
                        Padding = reader.ReadBytes(PaddingLength);
                    }
                    
                    Logger?.LogDebug($"MCNR: Parsed {VertexCount} vertex normals");
                }
            }
            catch (Exception ex)
            {
                AddError($"Error parsing MCNR chunk: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Gets the vertex normal at the specified grid coordinates.
        /// </summary>
        /// <param name="x">The x coordinate (0-8 for the 9x9 grid, 0-7 for the 8x8 grid).</param>
        /// <param name="y">The y coordinate (0-8 for the 9x9 grid, 0-7 for the 8x8 grid).</param>
        /// <param name="isCenter">True if accessing the 9x9 center grid, false for the 8x8 corner grid.</param>
        /// <returns>The normal vector at the specified coordinates, or a default vector if out of range.</returns>
        public Vector3 GetNormal(int x, int y, bool isCenter)
        {
            if (isCenter)
            {
                if (x < 0 || x > 8 || y < 0 || y > 8)
                {
                    AddError($"Center grid coordinates must be between 0 and 8, got ({x}, {y})");
                    return Vector3.UnitZ; // Default to upward normal
                }
                
                // The 9x9 grid is the first 81 values
                return VertexNormals[y * 9 + x];
            }
            else
            {
                if (x < 0 || x > 7 || y < 0 || y > 7)
                {
                    AddError($"Corner grid coordinates must be between 0 and 7, got ({x}, {y})");
                    return Vector3.UnitZ; // Default to upward normal
                }
                
                // The 8x8 grid is the last 64 values, offset by the 9x9 grid size
                return VertexNormals[81 + (y * 8 + x)];
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
                // Write the raw normal data
                for (int i = 0; i < VertexCount; i++)
                {
                    if (i * 3 + 2 < RawNormalData.Length)
                    {
                        writer.Write(RawNormalData[i * 3]);     // X
                        writer.Write(RawNormalData[i * 3 + 1]); // Y
                        writer.Write(RawNormalData[i * 3 + 2]); // Z
                    }
                    else
                    {
                        // If we don't have raw data, convert from the Vector3
                        Vector3 normal = VertexNormals[i];
                        
                        // Convert back to byte format
                        byte x = (byte)Math.Clamp((int)(normal.X * 127f + 127f), 0, 255);
                        byte y = (byte)Math.Clamp((int)(normal.Y * 127f + 127f), 0, 255);
                        byte z = (byte)Math.Clamp((int)(normal.Z * 127f + 127f), 0, 255);
                        
                        writer.Write(x);
                        writer.Write(y);
                        writer.Write(z);
                    }
                }
                
                // Write padding bytes
                if (Padding.Length > 0)
                {
                    writer.Write(Padding);
                }
                else
                {
                    // Write zeros for padding if we don't have original padding
                    for (int i = 0; i < PaddingLength; i++)
                    {
                        writer.Write((byte)0);
                    }
                }
            }
            catch (Exception ex)
            {
                AddError($"Error writing MCNR chunk: {ex.Message}");
            }
        }
    }
} 