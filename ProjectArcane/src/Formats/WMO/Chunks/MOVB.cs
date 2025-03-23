using System;
using System.Collections.Generic;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// Visible blocks chunk. Contains indices into MOVV vertices to define convex volumes for visibility testing.
    /// Each block consists of a count followed by vertex indices that form a convex volume.
    /// </summary>
    public class MOVB : IChunk
    {
        /// <summary>
        /// Gets the list of visible blocks.
        /// </summary>
        public List<VisibleBlock> VisibleBlocks { get; } = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;
            var endPosition = startPos + size;

            // Clear existing data
            VisibleBlocks.Clear();

            // Read blocks until we reach the end of the chunk
            while (reader.BaseStream.Position < endPosition)
            {
                try
                {
                    // Read vertex count for this block
                    var vertexCount = reader.ReadUInt16();

                    // Validate reasonable vertex count (convex volumes typically use 4-8 vertices)
                    if (vertexCount < 3 || vertexCount > 32)
                        throw new InvalidDataException($"Invalid vertex count in visible block: {vertexCount}");

                    // Create new block
                    var block = new VisibleBlock
                    {
                        FirstVertex = reader.ReadUInt16(),
                        Count = vertexCount
                    };

                    VisibleBlocks.Add(block);
                }
                catch (EndOfStreamException)
                {
                    throw new InvalidDataException("Unexpected end of MOVB chunk data");
                }
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var block in VisibleBlocks)
            {
                writer.Write(block.Count);
                writer.Write(block.FirstVertex);
            }
        }

        /// <summary>
        /// Validates all visible blocks against a MOVV chunk.
        /// </summary>
        /// <param name="movv">The MOVV chunk containing the vertices.</param>
        /// <returns>True if all blocks are valid, false otherwise.</returns>
        public bool ValidateBlocks(MOVV movv)
        {
            if (movv == null)
                return false;

            foreach (var block in VisibleBlocks)
            {
                // Check if indices are within bounds
                if (block.FirstVertex >= movv.VisibleVertices.Count ||
                    block.FirstVertex + block.Count > movv.VisibleVertices.Count)
                    return false;

                // Validate block geometry
                if (!movv.ValidateGeometry(block.FirstVertex, block.Count))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Gets a validation report for all visible blocks.
        /// </summary>
        /// <param name="movv">Optional MOVV chunk for detailed validation.</param>
        public string GetValidationReport(MOVV movv = null)
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOVB Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Blocks: {VisibleBlocks.Count}");
            report.AppendLine();

            // Basic block statistics
            var minVertices = ushort.MaxValue;
            var maxVertices = ushort.MinValue;
            var totalVertices = 0;

            foreach (var block in VisibleBlocks)
            {
                minVertices = Math.Min(minVertices, block.Count);
                maxVertices = Math.Max(maxVertices, block.Count);
                totalVertices += block.Count;
            }

            report.AppendLine("Block Statistics:");
            report.AppendLine($"  Min Vertices per Block: {minVertices}");
            report.AppendLine($"  Max Vertices per Block: {maxVertices}");
            report.AppendLine($"  Average Vertices per Block: {(float)totalVertices / VisibleBlocks.Count:F2}");
            report.AppendLine($"  Total Vertex References: {totalVertices}");
            report.AppendLine();

            // Detailed validation if MOVV is provided
            if (movv != null)
            {
                report.AppendLine("Block Validation:");
                for (int i = 0; i < VisibleBlocks.Count; i++)
                {
                    var block = VisibleBlocks[i];
                    var vertices = movv.GetBlockVertices(block.FirstVertex, block.Count);
                    var isValid = vertices != null && movv.ValidateGeometry(block.FirstVertex, block.Count);

                    report.AppendLine($"  Block {i}:");
                    report.AppendLine($"    Vertices: {block.Count} (Start: {block.FirstVertex})");
                    report.AppendLine($"    Valid: {isValid}");

                    if (!isValid)
                    {
                        if (block.FirstVertex >= movv.VisibleVertices.Count)
                            report.AppendLine($"    Error: Start index {block.FirstVertex} exceeds vertex count {movv.VisibleVertices.Count}");
                        else if (block.FirstVertex + block.Count > movv.VisibleVertices.Count)
                            report.AppendLine($"    Error: Block extends beyond vertex array ({block.FirstVertex + block.Count} > {movv.VisibleVertices.Count})");
                        else
                            report.AppendLine("    Error: Invalid geometry configuration");
                    }
                }
            }

            return report.ToString();
        }

        /// <summary>
        /// Represents a single visible block in the WMO.
        /// </summary>
        public class VisibleBlock
        {
            /// <summary>
            /// Gets or sets the index of the first vertex in the MOVV chunk.
            /// </summary>
            public ushort FirstVertex { get; set; }

            /// <summary>
            /// Gets or sets the number of vertices in this block.
            /// </summary>
            public ushort Count { get; set; }
        }
    }
} 