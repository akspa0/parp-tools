using System;
using System.Collections.Generic;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// Portal information chunk. 20 bytes per portal.
    /// Maximum of 128 portals per WMO.
    /// </summary>
    public class MOPT : IChunk
    {
        public const int PORTAL_SIZE = 20;
        public const int MAX_PORTALS = 128;

        public class Portal
        {
            public ushort StartVertex { get; set; }  // Index into MOPV vertex list
            public ushort Count { get; set; }        // Number of vertices in this portal
            public C4Plane Plane { get; set; }       // Portal plane equation

            public Portal()
            {
                Plane = new C4Plane();
            }

            /// <summary>
            /// Validates that the portal has valid vertex indices.
            /// </summary>
            public bool ValidateIndices(int totalVertices)
            {
                return StartVertex + Count <= totalVertices && Count >= 3;
            }

            /// <summary>
            /// Recalculates the portal plane equation from vertices.
            /// </summary>
            public void RecalculatePlane(C3Vector[] vertices)
            {
                if (vertices == null || vertices.Length < 3)
                    return;

                // Calculate normal using first three vertices
                var v1 = vertices[1] - vertices[0];
                var v2 = vertices[2] - vertices[0];
                var normal = v1.Cross(v2).Normalize();

                // Calculate D term using a point on the plane (first vertex)
                var d = -(normal.X * vertices[0].X + normal.Y * vertices[0].Y + normal.Z * vertices[0].Z);

                Plane = new C4Plane
                {
                    A = normal.X,
                    B = normal.Y,
                    C = normal.Z,
                    D = d
                };
            }

            /// <summary>
            /// Validates that the stored plane equation matches the actual geometry.
            /// </summary>
            public bool ValidatePlane(C3Vector[] vertices)
            {
                if (vertices == null || vertices.Length < 3)
                    return false;

                // Create a temporary plane from vertices
                var tempPortal = new Portal();
                tempPortal.RecalculatePlane(vertices);

                // Compare planes with tolerance
                const float tolerance = 0.001f;
                return Math.Abs(Plane.A - tempPortal.Plane.A) < tolerance &&
                       Math.Abs(Plane.B - tempPortal.Plane.B) < tolerance &&
                       Math.Abs(Plane.C - tempPortal.Plane.C) < tolerance &&
                       Math.Abs(Plane.D - tempPortal.Plane.D) < tolerance;
            }
        }

        /// <summary>
        /// Gets the list of portals.
        /// </summary>
        public List<Portal> Portals { get; } = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Validate portal count
            var portalCount = (int)size / PORTAL_SIZE;
            if (portalCount > MAX_PORTALS)
                throw new InvalidDataException($"WMO contains {portalCount} portals, exceeding maximum of {MAX_PORTALS}");

            // Clear existing data
            Portals.Clear();

            // Read portals
            for (int i = 0; i < portalCount; i++)
            {
                var portal = new Portal
                {
                    StartVertex = reader.ReadUInt16(),
                    Count = reader.ReadUInt16(),
                    Plane = reader.ReadC4Plane()
                };

                Portals.Add(portal);
            }
        }

        public void Write(BinaryWriter writer)
        {
            if (Portals.Count > MAX_PORTALS)
                throw new InvalidOperationException($"Cannot write more than {MAX_PORTALS} portals");

            foreach (var portal in Portals)
            {
                writer.Write(portal.StartVertex);
                writer.Write(portal.Count);
                writer.Write(portal.Plane);
            }
        }

        /// <summary>
        /// Validates all portals against vertex data.
        /// </summary>
        public bool ValidatePortals(MOPV mopv)
        {
            if (mopv == null)
                return false;

            foreach (var portal in Portals)
            {
                // Check vertex indices
                if (!portal.ValidateIndices(mopv.Vertices.Count))
                    return false;

                // Get portal vertices
                var vertices = mopv.GetPortalVertices(portal.StartVertex, portal.Count);
                if (vertices == null)
                    return false;

                // Validate portal plane
                if (!portal.ValidatePlane(vertices))
                    return false;

                // Validate portal geometry
                if (!mopv.ValidatePortalGeometry(portal.StartVertex, portal.Count))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Gets a validation report for all portals.
        /// </summary>
        public string GetValidationReport(MOPV mopv)
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOPT Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Portal Count: {Portals.Count} (Max: {MAX_PORTALS})");
            report.AppendLine();

            if (mopv != null)
            {
                report.AppendLine("Portal Validation:");
                for (int i = 0; i < Portals.Count; i++)
                {
                    var portal = Portals[i];
                    var vertices = mopv.GetPortalVertices(portal.StartVertex, portal.Count);
                    var indicesValid = portal.ValidateIndices(mopv.Vertices.Count);
                    var planeValid = vertices != null && portal.ValidatePlane(vertices);
                    var geometryValid = vertices != null && mopv.ValidatePortalGeometry(portal.StartVertex, portal.Count);

                    report.AppendLine($"  Portal {i}:");
                    report.AppendLine($"    Vertices: {portal.Count} (Start: {portal.StartVertex})");
                    report.AppendLine($"    Valid Indices: {indicesValid}");
                    report.AppendLine($"    Valid Plane: {planeValid}");
                    report.AppendLine($"    Valid Geometry: {geometryValid}");
                    report.AppendLine($"    Plane: {portal.Plane}");

                    if (vertices != null)
                    {
                        var area = mopv.CalculatePortalArea(portal.StartVertex, portal.Count);
                        report.AppendLine($"    Area: {area:F2}");
                    }

                    report.AppendLine();
                }
            }

            return report.ToString();
        }

        /// <summary>
        /// Recalculates all portal planes from vertex data.
        /// </summary>
        public void RecalculatePortalPlanes(MOPV mopv)
        {
            if (mopv == null)
                return;

            foreach (var portal in Portals)
            {
                var vertices = mopv.GetPortalVertices(portal.StartVertex, portal.Count);
                if (vertices != null)
                {
                    portal.RecalculatePlane(vertices);
                }
            }
        }
    }
} 