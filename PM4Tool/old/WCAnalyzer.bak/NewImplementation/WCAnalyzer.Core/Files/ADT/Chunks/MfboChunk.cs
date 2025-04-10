using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Interfaces;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// Represents an MfboChunk (MFBO) chunk that contains flight boundaries and terrain height limits.
    /// This chunk defines a bounding box for the terrain with minimum and maximum height planes.
    /// Introduced in Burning Crusade expansion.
    /// </summary>
    public class MfboChunk : ADTChunk
    {
        /// <summary>
        /// The signature for this chunk type.
        /// </summary>
        public const string SIGNATURE = "MFBO";

        /// <summary>
        /// Gets the maximum bounding plane - a 3x3 grid of height values.
        /// </summary>
        public List<List<short>> Maximum { get; private set; }

        /// <summary>
        /// Gets the minimum bounding plane - a 3x3 grid of height values.
        /// </summary>
        public List<List<short>> Minimum { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MfboChunk"/> class.
        /// </summary>
        /// <param name="data">The raw chunk data.</param>
        /// <param name="logger">The logger instance.</param>
        public MfboChunk(byte[] data, ILogger logger) 
            : base(SIGNATURE, data, logger)
        {
            Maximum = new List<List<short>>();
            Minimum = new List<List<short>>();
            Parse(data);
        }

        /// <summary>
        /// Parses the raw chunk data to extract maximum and minimum height planes.
        /// </summary>
        /// <param name="data">The raw chunk data.</param>
        protected override void Parse(byte[] data)
        {
            if (data == null || data.Length == 0)
            {
                Logger.LogWarning("MfboChunk: Empty data provided to Parse method");
                return;
            }

            if (data.Length < 36) // 18 shorts (2 bytes each) for both planes
            {
                Logger.LogWarning($"MfboChunk: Insufficient data length. Expected at least 36 bytes, got {data.Length}");
                return;
            }

            try
            {
                using (MemoryStream ms = new MemoryStream(data))
                using (BinaryReader reader = new BinaryReader(ms))
                {
                    // Clear existing data
                    Maximum.Clear();
                    Minimum.Clear();

                    // Read the maximum height plane (3x3 grid of shorts)
                    for (int y = 0; y < 3; y++)
                    {
                        var maxRow = new List<short>();
                        for (int x = 0; x < 3; x++)
                        {
                            maxRow.Add(reader.ReadInt16());
                        }
                        Maximum.Add(maxRow);
                    }

                    // Read the minimum height plane (3x3 grid of shorts)
                    for (int y = 0; y < 3; y++)
                    {
                        var minRow = new List<short>();
                        for (int x = 0; x < 3; x++)
                        {
                            minRow.Add(reader.ReadInt16());
                        }
                        Minimum.Add(minRow);
                    }

                    Logger.LogDebug($"MfboChunk: Successfully parsed maximum and minimum height planes (3x3 grids)");
                }
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, $"MfboChunk: Error parsing chunk data: {ex.Message}");
            }
        }

        /// <summary>
        /// Writes the chunk data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                Logger.LogWarning("MfboChunk: Null writer provided to Write method");
                return;
            }

            // Validate grid dimensions
            if (Maximum.Count != 3 || Minimum.Count != 3 || 
                Maximum.Any(row => row.Count != 3) || Minimum.Any(row => row.Count != 3))
            {
                Logger.LogError("MfboChunk: Invalid grid dimensions for Write method. Both Maximum and Minimum must be 3x3 grids");
                return;
            }

            try
            {
                // Write the chunk signature
                writer.Write(SignatureBytes);

                // Calculate the size of the data (18 shorts = 36 bytes)
                writer.Write(36);

                // Write the maximum height plane (3x3 grid of shorts)
                foreach (var row in Maximum)
                {
                    foreach (var height in row)
                    {
                        writer.Write(height);
                    }
                }

                // Write the minimum height plane (3x3 grid of shorts)
                foreach (var row in Minimum)
                {
                    foreach (var height in row)
                    {
                        writer.Write(height);
                    }
                }

                Logger.LogDebug("MfboChunk: Successfully wrote maximum and minimum height planes");
            }
            catch (Exception ex)
            {
                Logger.LogError(ex, $"MfboChunk: Error writing chunk data: {ex.Message}");
            }
        }

        /// <summary>
        /// Gets the maximum height value from the maximum plane.
        /// </summary>
        /// <returns>The maximum height value.</returns>
        public short GetMaximumHeight()
        {
            if (Maximum == null || Maximum.Count == 0 || Maximum.Any(row => row.Count == 0))
            {
                Logger.LogWarning("MfboChunk: Maximum plane is empty or invalid when calling GetMaximumHeight");
                return 0;
            }

            short max = short.MinValue;
            foreach (var row in Maximum)
            {
                foreach (var height in row)
                {
                    if (height > max)
                    {
                        max = height;
                    }
                }
            }
            return max;
        }

        /// <summary>
        /// Gets the minimum height value from the minimum plane.
        /// </summary>
        /// <returns>The minimum height value.</returns>
        public short GetMinimumHeight()
        {
            if (Minimum == null || Minimum.Count == 0 || Minimum.Any(row => row.Count == 0))
            {
                Logger.LogWarning("MfboChunk: Minimum plane is empty or invalid when calling GetMinimumHeight");
                return 0;
            }

            short min = short.MaxValue;
            foreach (var row in Minimum)
            {
                foreach (var height in row)
                {
                    if (height < min)
                    {
                        min = height;
                    }
                }
            }
            return min;
        }

        /// <summary>
        /// Gets a height value from the maximum plane at the specified coordinates.
        /// </summary>
        /// <param name="x">The x-coordinate (0-2).</param>
        /// <param name="y">The y-coordinate (0-2).</param>
        /// <returns>The height value at the specified coordinates, or 0 if coordinates are invalid.</returns>
        public short GetMaximumHeight(int x, int y)
        {
            if (x < 0 || x > 2 || y < 0 || y > 2 || 
                Maximum == null || Maximum.Count <= y || Maximum[y].Count <= x)
            {
                Logger.LogWarning($"MfboChunk: Invalid coordinates ({x},{y}) for GetMaximumHeight");
                return 0;
            }

            return Maximum[y][x];
        }

        /// <summary>
        /// Gets a height value from the minimum plane at the specified coordinates.
        /// </summary>
        /// <param name="x">The x-coordinate (0-2).</param>
        /// <param name="y">The y-coordinate (0-2).</param>
        /// <returns>The height value at the specified coordinates, or 0 if coordinates are invalid.</returns>
        public short GetMinimumHeight(int x, int y)
        {
            if (x < 0 || x > 2 || y < 0 || y > 2 || 
                Minimum == null || Minimum.Count <= y || Minimum[y].Count <= x)
            {
                Logger.LogWarning($"MfboChunk: Invalid coordinates ({x},{y}) for GetMinimumHeight");
                return 0;
            }

            return Minimum[y][x];
        }

        /// <summary>
        /// Gets the vertical range of the chunk (difference between maximum and minimum heights).
        /// </summary>
        /// <returns>The vertical range in height units.</returns>
        public short GetVerticalRange()
        {
            short max = GetMaximumHeight();
            short min = GetMinimumHeight();
            return (short)(max - min);
        }
    }
} 