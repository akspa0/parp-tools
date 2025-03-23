using System;
using System.Collections.Generic;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// Portal vertices chunk. Contains vertex positions for portal planes used in visibility calculations.
    /// Usually 4 vertices per portal (quads) but can have more complex shapes.
    /// </summary>
    public class MOPV : IChunk
    {
        /// <summary>
        /// Gets the list of portal vertices.
        /// </summary>
        public List<C3Vector> Vertices { get; } = new();

        /// <summary>
        /// Gets the number of portals based on vertex count and portal definitions.
        /// This is calculated after MOPT is loaded and provides vertex counts.
        /// </summary>
        public int EstimatedPortalCount => Vertices.Count / 4; // Most portals are quads

        /// <summary>
        /// Gets the raw vertex data for direct access.
        /// </summary>
        private byte[] RawData { get; set; }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Store raw data for potential direct access
            RawData = reader.ReadBytes((int)size);

            // Clear existing data
            Vertices.Clear();

            // Read vertices (12 bytes each - 3 floats)
            int vertexCount = (int)size / 12;
            for (int i = 0; i < vertexCount; i++)
            {
                Vertices.Add(new C3Vector
                {
                    X = BitConverter.ToSingle(RawData, i * 12),
                    Y = BitConverter.ToSingle(RawData, i * 12 + 4),
                    Z = BitConverter.ToSingle(RawData, i * 12 + 8)
                });
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var vertex in Vertices)
            {
                writer.Write(vertex.X);
                writer.Write(vertex.Y);
                writer.Write(vertex.Z);
            }
        }

        /// <summary>
        /// Gets the vertices for a specific portal.
        /// </summary>
        /// <param name="startVertex">Starting vertex index from MOPT.</param>
        /// <param name="count">Number of vertices in the portal from MOPT.</param>
        /// <returns>Array of vertices forming the portal, or null if invalid indices.</returns>
        public C3Vector[] GetPortalVertices(ushort startVertex, ushort count)
        {
            // Validate indices
            if (startVertex >= Vertices.Count || startVertex + count > Vertices.Count)
                return null;

            var portalVerts = new C3Vector[count];
            for (int i = 0; i < count; i++)
            {
                portalVerts[i] = Vertices[startVertex + i];
            }
            return portalVerts;
        }

        /// <summary>
        /// Validates that a portal's vertices form a valid convex polygon.
        /// </summary>
        public bool ValidatePortalGeometry(ushort startVertex, ushort count)
        {
            var vertices = GetPortalVertices(startVertex, count);
            if (vertices == null || count < 3)
                return false;

            // Check if vertices form a planar polygon
            if (!IsCoplanar(vertices))
                return false;

            // Check if polygon is convex
            return IsConvexPolygon(vertices);
        }

        /// <summary>
        /// Checks if vertices are coplanar (lie on the same plane).
        /// </summary>
        private bool IsCoplanar(C3Vector[] vertices)
        {
            if (vertices.Length < 3)
                return true;

            // Get plane normal using first three vertices
            var v1 = vertices[1] - vertices[0];
            var v2 = vertices[2] - vertices[0];
            var normal = v1.Cross(v2).Normalize();

            // Check if all other vertices lie on the same plane
            var tolerance = 0.001f; // Small tolerance for floating point errors
            for (int i = 3; i < vertices.Length; i++)
            {
                var v = vertices[i] - vertices[0];
                var distance = Math.Abs(v.Dot(normal));
                if (distance > tolerance)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Checks if a polygon is convex by verifying all interior angles are less than 180 degrees.
        /// </summary>
        private bool IsConvexPolygon(C3Vector[] vertices)
        {
            if (vertices.Length < 3)
                return false;

            for (int i = 0; i < vertices.Length; i++)
            {
                var current = vertices[i];
                var next = vertices[(i + 1) % vertices.Length];
                var nextNext = vertices[(i + 2) % vertices.Length];

                var edge1 = next - current;
                var edge2 = nextNext - next;

                // Cross product should point in same direction as normal for convex polygon
                var cross = edge1.Cross(edge2);
                if (cross.Z < 0)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Gets a validation report for portal vertices.
        /// </summary>
        public string GetValidationReport(MOPT mopt)
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOPV Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Vertices: {Vertices.Count}");
            report.AppendLine($"Estimated Portal Count: {EstimatedPortalCount}");
            report.AppendLine();

            if (mopt != null)
            {
                report.AppendLine("Portal Geometry Validation:");
                for (int i = 0; i < mopt.Portals.Count; i++)
                {
                    var portal = mopt.Portals[i];
                    var isValid = ValidatePortalGeometry(portal.StartVertex, portal.Count);
                    report.AppendLine($"  Portal {i}:");
                    report.AppendLine($"    Vertices: {portal.Count} (Start: {portal.StartVertex})");
                    report.AppendLine($"    Valid Geometry: {isValid}");
                    
                    if (!isValid)
                    {
                        var verts = GetPortalVertices(portal.StartVertex, portal.Count);
                        if (verts != null)
                        {
                            report.AppendLine("    Vertex Positions:");
                            for (int j = 0; j < verts.Length; j++)
                            {
                                report.AppendLine($"      [{j}]: {verts[j]}");
                            }
                        }
                    }
                }
            }

            return report.ToString();
        }

        /// <summary>
        /// Calculates the area of a portal.
        /// </summary>
        public float CalculatePortalArea(ushort startVertex, ushort count)
        {
            var vertices = GetPortalVertices(startVertex, count);
            if (vertices == null || count < 3)
                return 0;

            // Use triangulation for area calculation
            float area = 0;
            var v0 = vertices[0];
            for (int i = 1; i < count - 1; i++)
            {
                var v1 = vertices[i];
                var v2 = vertices[i + 1];
                var cross = (v1 - v0).Cross(v2 - v0);
                area += cross.Length() * 0.5f;
            }

            return area;
        }
    }
} 