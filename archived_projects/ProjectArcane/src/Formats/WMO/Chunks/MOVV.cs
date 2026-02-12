using System;
using System.Collections.Generic;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// Visible vertices chunk. Contains vertices used for visibility testing and optimization.
    /// Each vertex is 12 bytes (3 floats) and defines points in 3D space for visibility bounding geometry.
    /// </summary>
    public class MOVV : IChunk
    {
        /// <summary>
        /// Gets the list of visible vertices.
        /// </summary>
        public List<C3Vector> VisibleVertices { get; } = new();

        /// <summary>
        /// Gets the raw vertex data for direct access.
        /// </summary>
        private byte[] RawData { get; set; }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Validate chunk size
            if (size % 12 != 0)
                throw new InvalidDataException($"MOVV chunk size {size} is not divisible by 12 (vertex size)");

            // Store raw data for potential direct access
            RawData = reader.ReadBytes((int)size);

            // Clear existing data
            VisibleVertices.Clear();

            // Read vertices (12 bytes each - 3 floats)
            int vertexCount = (int)size / 12;
            for (int i = 0; i < vertexCount; i++)
            {
                VisibleVertices.Add(new C3Vector
                {
                    X = BitConverter.ToSingle(RawData, i * 12),
                    Y = BitConverter.ToSingle(RawData, i * 12 + 4),
                    Z = BitConverter.ToSingle(RawData, i * 12 + 8)
                });
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var vertex in VisibleVertices)
            {
                writer.Write(vertex.X);
                writer.Write(vertex.Y);
                writer.Write(vertex.Z);
            }
        }

        /// <summary>
        /// Gets a subset of vertices for a specific visible block.
        /// </summary>
        /// <param name="startVertex">Starting vertex index.</param>
        /// <param name="count">Number of vertices in the block.</param>
        /// <returns>Array of vertices for the block, or null if indices are invalid.</returns>
        public C3Vector[] GetBlockVertices(ushort startVertex, ushort count)
        {
            // Validate indices
            if (startVertex >= VisibleVertices.Count || startVertex + count > VisibleVertices.Count)
                return null;

            var blockVerts = new C3Vector[count];
            for (int i = 0; i < count; i++)
            {
                blockVerts[i] = VisibleVertices[startVertex + i];
            }
            return blockVerts;
        }

        /// <summary>
        /// Validates that vertices form valid bounding geometry.
        /// </summary>
        public bool ValidateGeometry(ushort startVertex, ushort count)
        {
            var vertices = GetBlockVertices(startVertex, count);
            if (vertices == null || count < 3)
                return false;

            // Check for invalid vertex values
            foreach (var vertex in vertices)
            {
                if (float.IsNaN(vertex.X) || float.IsInfinity(vertex.X) ||
                    float.IsNaN(vertex.Y) || float.IsInfinity(vertex.Y) ||
                    float.IsNaN(vertex.Z) || float.IsInfinity(vertex.Z))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Gets a validation report for visible vertices.
        /// </summary>
        public string GetValidationReport(MOVB movb = null)
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOVV Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Vertices: {VisibleVertices.Count}");
            report.AppendLine();

            // Calculate bounding box
            if (VisibleVertices.Count > 0)
            {
                var minX = float.MaxValue;
                var minY = float.MaxValue;
                var minZ = float.MaxValue;
                var maxX = float.MinValue;
                var maxY = float.MinValue;
                var maxZ = float.MinValue;

                foreach (var vertex in VisibleVertices)
                {
                    minX = Math.Min(minX, vertex.X);
                    minY = Math.Min(minY, vertex.Y);
                    minZ = Math.Min(minZ, vertex.Z);
                    maxX = Math.Max(maxX, vertex.X);
                    maxY = Math.Max(maxY, vertex.Y);
                    maxZ = Math.Max(maxZ, vertex.Z);
                }

                report.AppendLine("Bounding Box:");
                report.AppendLine($"  Min: ({minX:F2}, {minY:F2}, {minZ:F2})");
                report.AppendLine($"  Max: ({maxX:F2}, {maxY:F2}, {maxZ:F2})");
                report.AppendLine($"  Size: ({maxX - minX:F2}, {maxY - minY:F2}, {maxZ - minZ:F2})");
                report.AppendLine();
            }

            if (movb != null)
            {
                report.AppendLine("Block Validation:");
                for (int i = 0; i < movb.VisibleBlocks.Count; i++)
                {
                    var block = movb.VisibleBlocks[i];
                    var isValid = ValidateGeometry(block.FirstVertex, block.Count);
                    report.AppendLine($"  Block {i}:");
                    report.AppendLine($"    Vertices: {block.Count} (Start: {block.FirstVertex})");
                    report.AppendLine($"    Valid Geometry: {isValid}");

                    if (!isValid)
                    {
                        var vertices = GetBlockVertices(block.FirstVertex, block.Count);
                        if (vertices != null)
                        {
                            report.AppendLine("    Vertex Positions:");
                            for (int j = 0; j < vertices.Length; j++)
                            {
                                report.AppendLine($"      [{j}]: {vertices[j]}");
                            }
                        }
                    }
                }
            }

            return report.ToString();
        }
    }
} 