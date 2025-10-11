using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Foundation.Transforms;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// A service for handling coordinate transformations for PM4 file data.
    /// This class was reconstructed based on usage in test files, as the original source could not be located.
    /// The transformations assume a conversion from a Z-up coordinate system (WoW default) to a Y-up system (standard for rendering).
    /// </summary>
    public class CoordinateService : ICoordinateService
    {
        // This standard transformation converts from WoW's Z-up coordinate system to a standard Y-up system.
        // (X, Y, Z) in-game -> (X, Z, -Y) in render space.
        

        public Vector3 FromMsvtVertex(MSVT_Vertex vertex) => CoordinateTransforms.FromMsvtVertex(vertex);

        public Vector3 FromMsvtVertexSimple(MSVT_Vertex vertex) => CoordinateTransforms.FromMsvtVertex(vertex);

        // Overload accepting legacy/chunk MsvtVertex
        public Vector3 FromMsvtVertexSimple(WoWToolbox.Core.v2.Foundation.PM4.Chunks.MsvtVertex vertex)
        {
            var converted = new MSVT_Vertex { Position = new Vector3(vertex.X, vertex.Y, vertex.Z) };
            return CoordinateTransforms.FromMsvtVertex(converted);
        }

        public Vector3 FromMscnVertex(Vector3 vertex)
        {
            // MSCN chunks store vertices already as XYZ; apply Z-up to Y-up.
            return CoordinateTransforms.FromMscnVertex(vertex);
        }

        public Vector3 FromMscnVertex(MSCN_Vertex vertex)
        {
            return CoordinateTransforms.FromMscnVertex(vertex.Position);
        }

        public Vector3 FromMspvVertex(WoWToolbox.Core.v2.Models.PM4.Chunks.C3Vector vertex)
        {
            // Assuming identical orientation as MSPV_Vertex placeholder
            var converted = new MSPV_Vertex { Position = new Vector3(vertex.X, vertex.Y, vertex.Z) };
            return CoordinateTransforms.FromMspvVertex(converted.Position);
        }

        public Vector3 FromMspvVertex(MSPV_Vertex vertex)
        {
            return CoordinateTransforms.FromMspvVertex(vertex.Position);
        }

        // Overload for Warcraft.NET.Files.Structures.C3Vector used by PM4 parsing layer
        public Vector3 FromMspvVertex(Warcraft.NET.Files.Structures.C3Vector vertex)
        {
            var converted = new MSPV_Vertex { Position = new Vector3(vertex.X, vertex.Y, vertex.Z) };
            return CoordinateTransforms.FromMspvVertex(converted.Position);
        }

        public Vector3 FromMprlEntry(MPRL_Entry entry)
        {
            // MPRL_Entry uses Vector3 directly.
            return CoordinateTransforms.FromMprlEntry(entry);
        }

        // Overload accepting full MprlEntry from Foundation chunk
        public Vector3 FromMprlEntry(WoWToolbox.Core.v2.Foundation.PM4.Chunks.MprlEntry entry)
        {
            return CoordinateTransforms.FromMprlEntry(entry);
        }

        /// <summary>
        /// Computes smooth vertex normals for a given set of vertices and triangle indices.
        /// </summary>
        /// <param name="vertices">The list of vertices.</param>
        /// <param name="indices">The list of triangle indices.</param>
        /// <returns>A list of computed normals, one for each vertex.</returns>
        public List<Vector3> ComputeVertexNormals(IList<Vector3> vertices, IList<int> indices)
        {
            var normals = new Vector3[vertices.Count];

            for (int i = 0; i < indices.Count; i += 3)
            {
                var i1 = indices[i];
                var i2 = indices[i + 1];
                var i3 = indices[i + 2];

                // Ensure indices are within bounds
                if (i1 >= vertices.Count || i2 >= vertices.Count || i3 >= vertices.Count)
                {
                    continue; 
                }

                var v1 = vertices[i1];
                var v2 = vertices[i2];
                var v3 = vertices[i3];

                // Calculate the face normal using a cross product.
                var faceNormal = Vector3.Cross(v2 - v1, v3 - v1);

                // Add the face normal to the normal of each vertex that makes up the face.
                normals[i1] += faceNormal;
                normals[i2] += faceNormal;
                normals[i3] += faceNormal;
            }

            // Normalize all the vertex normals.
            for (int i = 0; i < normals.Length; i++)
            {
                normals[i] = Vector3.Normalize(normals[i]);
            }

            return normals.ToList();
        }
    }
}
