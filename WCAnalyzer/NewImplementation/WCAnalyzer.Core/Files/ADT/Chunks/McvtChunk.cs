using Microsoft.Extensions.Logging;
using System;
using System.IO;
using WCAnalyzer.Core.Common.Interfaces;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents an MCVT chunk in an ADT file, containing height map data for a map chunk.
    /// </summary>
    public class McvtChunk : ADTChunk
    {
        /// <summary>
        /// The MCVT chunk signature
        /// </summary>
        public const string SIGNATURE = "MCVT";

        /// <summary>
        /// Gets the height map data. This is a 9x9 grid (center of each cell) plus an 8x8 grid (cell corners),
        /// giving a total of 145 height values.
        /// </summary>
        public float[] HeightMap { get; } = new float[145];

        /// <summary>
        /// Initializes a new instance of the <see cref="McvtChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        /// <param name="logger">Optional logger.</param>
        public McvtChunk(byte[] data, ILogger? logger = null)
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
                AddError("No data to parse for MCVT chunk");
                return;
            }

            try
            {
                using (var ms = new MemoryStream(Data))
                using (var reader = new BinaryReader(ms))
                {
                    // An MCVT chunk contains 145 float values representing the height map for the chunk
                    // The values are arranged in a 9x9 + 8x8 grid pattern
                    for (int i = 0; i < 145; i++)
                    {
                        HeightMap[i] = reader.ReadSingle();
                    }

                    Logger?.LogDebug($"MCVT: Parsed 145 height values");
                }
            }
            catch (Exception ex)
            {
                AddError($"Error parsing MCVT chunk: {ex.Message}");
            }
        }

        /// <summary>
        /// Gets the height value at the specified grid coordinates.
        /// </summary>
        /// <param name="x">The x coordinate (0-8 for the 9x9 grid, 0-7 for the 8x8 grid).</param>
        /// <param name="y">The y coordinate (0-8 for the 9x9 grid, 0-7 for the 8x8 grid).</param>
        /// <param name="isCenter">True if accessing the 9x9 center grid, false for the 8x8 corner grid.</param>
        /// <returns>The height value at the specified coordinates, or 0 if out of range.</returns>
        public float GetHeight(int x, int y, bool isCenter)
        {
            if (isCenter)
            {
                if (x < 0 || x > 8 || y < 0 || y > 8)
                {
                    AddError($"Center grid coordinates must be between 0 and 8, got ({x}, {y})");
                    return 0;
                }
                
                // The 9x9 grid is the first 81 values
                return HeightMap[y * 9 + x];
            }
            else
            {
                if (x < 0 || x > 7 || y < 0 || y > 7)
                {
                    AddError($"Corner grid coordinates must be between 0 and 7, got ({x}, {y})");
                    return 0;
                }
                
                // The 8x8 grid is the last 64 values, offset by the 9x9 grid size
                return HeightMap[81 + (y * 8 + x)];
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
                // Write each height value
                foreach (float height in HeightMap)
                {
                    writer.Write(height);
                }
            }
            catch (Exception ex)
            {
                AddError($"Error writing MCVT chunk: {ex.Message}");
            }
        }
    }
} 