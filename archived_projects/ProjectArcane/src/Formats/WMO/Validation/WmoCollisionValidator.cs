using System;
using System.Collections.Generic;
using System.Linq;
using ArcaneFileParser.Core.Formats.WMO.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Validation
{
    /// <summary>
    /// Validates collision integrity in WMO files.
    /// </summary>
    public class WmoCollisionValidator
    {
        private readonly WMOFile wmoFile;
        private readonly List<string> validationErrors;

        public WmoCollisionValidator(WMOFile file)
        {
            wmoFile = file;
            validationErrors = new List<string>();
        }

        public bool Validate()
        {
            validationErrors.Clear();

            // Validate each group
            var mogi = wmoFile.GetChunk<MOGI>("MOGI");
            if (mogi == null || mogi.Groups == null)
            {
                validationErrors.Add("Missing MOGI chunk or group data");
                return false;
            }

            for (int i = 0; i < mogi.Groups.Length; i++)
            {
                ValidateGroup(i);
            }

            return validationErrors.Count == 0;
        }

        private void ValidateGroup(int groupIndex)
        {
            var group = wmoFile.Groups[groupIndex];
            if (group == null)
            {
                validationErrors.Add($"Missing group data for index {groupIndex}");
                return;
            }

            // Get required chunks
            var mogp = group.GetChunk<MOGP>("MOGP");
            var mobn = group.GetChunk<MOBN>("MOBN");
            var mobr = group.GetChunk<MOBR>("MOBR");
            var mopy = group.GetChunk<MOPY>("MOPY");
            var movi = group.GetChunk<MOVI>("MOVI");
            var movt = group.GetChunk<MOVT>("MOVT");

            if (!ValidateRequiredChunks(groupIndex, mogp, mobn, mobr, mopy, movi, movt))
                return;

            // Validate BSP tree integrity
            ValidateBspTree(groupIndex, mobn, mobr, movi);

            // Validate collision faces
            ValidateCollisionFaces(groupIndex, mopy, movi, movt);

            // Validate vertex and index integrity
            ValidateVertexData(groupIndex, movi, movt);

            // If this is an indoor group, validate portal relationships
            if (mogp.Flags.HasFlag(GroupFlags.Indoor))
            {
                ValidatePortalRelationships(groupIndex, group);
            }
        }

        private bool ValidateRequiredChunks(int groupIndex, params object[] chunks)
        {
            for (int i = 0; i < chunks.Length; i++)
            {
                if (chunks[i] == null)
                {
                    validationErrors.Add($"Group {groupIndex} is missing required chunk {GetChunkName(i)}");
                    return false;
                }
            }
            return true;
        }

        private string GetChunkName(int index) => index switch
        {
            0 => "MOGP",
            1 => "MOBN",
            2 => "MOBR",
            3 => "MOPY",
            4 => "MOVI",
            5 => "MOVT",
            _ => "Unknown"
        };

        private void ValidateBspTree(int groupIndex, MOBN mobn, MOBR mobr, MOVI movi)
        {
            // Check BSP node count
            if (mobn.Nodes.Count == 0)
            {
                validationErrors.Add($"Group {groupIndex} has empty BSP tree");
                return;
            }

            // Validate each BSP node
            var visitedNodes = new HashSet<int>();
            ValidateBspNode(groupIndex, 0, mobn, mobr, movi, visitedNodes);

            // Check for orphaned nodes
            for (int i = 0; i < mobn.Nodes.Count; i++)
            {
                if (!visitedNodes.Contains(i))
                {
                    validationErrors.Add($"Group {groupIndex} has orphaned BSP node at index {i}");
                }
            }
        }

        private void ValidateBspNode(int groupIndex, int nodeIndex, MOBN mobn, MOBR mobr, MOVI movi, HashSet<int> visitedNodes)
        {
            if (nodeIndex < 0 || nodeIndex >= mobn.Nodes.Count || visitedNodes.Contains(nodeIndex))
                return;

            visitedNodes.Add(nodeIndex);
            var node = mobn.Nodes[nodeIndex];

            // Validate leaf node data
            if (node.IsLeaf)
            {
                if (node.NumFaces > 0)
                {
                    // Validate face indices
                    uint endIndex = node.FaceStartIndex + node.NumFaces;
                    if (endIndex > mobr.FaceIndices.Count)
                    {
                        validationErrors.Add($"Group {groupIndex} BSP node {nodeIndex} references invalid MOBR indices");
                        return;
                    }

                    // Validate MOVI indices referenced by MOBR
                    for (uint i = node.FaceStartIndex; i < endIndex; i++)
                    {
                        var moviIndex = mobr.FaceIndices[(int)i];
                        if (moviIndex >= movi.Indices.Count)
                        {
                            validationErrors.Add($"Group {groupIndex} BSP node {nodeIndex} references invalid MOVI index");
                            return;
                        }
                    }
                }
            }
            else
            {
                // Validate child nodes
                if (node.NegativeChildIndex != -1)
                    ValidateBspNode(groupIndex, node.NegativeChildIndex, mobn, mobr, movi, visitedNodes);
                if (node.PositiveChildIndex != -1)
                    ValidateBspNode(groupIndex, node.PositiveChildIndex, mobn, mobr, movi, visitedNodes);
            }
        }

        private void ValidateCollisionFaces(int groupIndex, MOPY mopy, MOVI movi, MOVT movt)
        {
            var collisionFaces = new List<int>();
            
            // Find all collision faces
            for (int i = 0; i < mopy.PolyMaterials.Count; i++)
            {
                var material = mopy.PolyMaterials[i];
                if (material.MaterialId == 0xFF || material.IsCollidable)
                {
                    collisionFaces.Add(i);
                }
            }

            // Validate each collision face
            foreach (var faceIndex in collisionFaces)
            {
                // Get vertex indices for this face
                var v1 = movi.Indices[faceIndex * 3];
                var v2 = movi.Indices[faceIndex * 3 + 1];
                var v3 = movi.Indices[faceIndex * 3 + 2];

                // Validate vertex indices
                if (v1 >= movt.Vertices.Count || v2 >= movt.Vertices.Count || v3 >= movt.Vertices.Count)
                {
                    validationErrors.Add($"Group {groupIndex} collision face {faceIndex} references invalid vertices");
                    continue;
                }

                // Validate face geometry
                var vert1 = movt.Vertices[v1];
                var vert2 = movt.Vertices[v2];
                var vert3 = movt.Vertices[v3];

                // Check for degenerate triangles
                if (IsDegenerate(vert1, vert2, vert3))
                {
                    validationErrors.Add($"Group {groupIndex} collision face {faceIndex} is degenerate");
                }
            }
        }

        private bool IsDegenerate(C3Vector v1, C3Vector v2, C3Vector v3)
        {
            // Check if any two vertices are the same
            if (v1 == v2 || v2 == v3 || v3 == v1)
                return true;

            // Check if vertices are collinear
            var edge1 = v2 - v1;
            var edge2 = v3 - v1;
            var cross = edge1.Cross(edge2);
            
            // If cross product is zero (or very close), vertices are collinear
            return cross.LengthSquared() < 1e-6f;
        }

        private void ValidateVertexData(int groupIndex, MOVI movi, MOVT movt)
        {
            // Check for invalid vertex positions
            for (int i = 0; i < movt.Vertices.Count; i++)
            {
                var vertex = movt.Vertices[i];
                if (float.IsNaN(vertex.X) || float.IsNaN(vertex.Y) || float.IsNaN(vertex.Z) ||
                    float.IsInfinity(vertex.X) || float.IsInfinity(vertex.Y) || float.IsInfinity(vertex.Z))
                {
                    validationErrors.Add($"Group {groupIndex} vertex {i} has invalid coordinates");
                }
            }

            // Check for invalid indices
            for (int i = 0; i < movi.Indices.Count; i++)
            {
                if (movi.Indices[i] >= movt.Vertices.Count)
                {
                    validationErrors.Add($"Group {groupIndex} index {i} references invalid vertex");
                }
            }
        }

        private void ValidatePortalRelationships(int groupIndex, WMOGroup group)
        {
            var mopt = wmoFile.GetChunk<MOPT>("MOPT");
            var mopv = wmoFile.GetChunk<MOPV>("MOPV");
            var mopr = wmoFile.GetChunk<MOPR>("MOPR");

            if (mopt == null || mopv == null || mopr == null)
                return; // Portals are optional

            // Get portal references for this group
            var groupPortals = mopr.References
                .Where(r => r.GroupIndex == groupIndex)
                .ToList();

            foreach (var portalRef in groupPortals)
            {
                if (portalRef.PortalIndex >= mopt.Portals.Count)
                {
                    validationErrors.Add($"Group {groupIndex} references invalid portal index {portalRef.PortalIndex}");
                    continue;
                }

                var portal = mopt.Portals[portalRef.PortalIndex];
                
                // Validate portal vertices
                var vertices = mopv.GetPortalVertices(portal.StartVertex, portal.Count);
                if (vertices == null)
                {
                    validationErrors.Add($"Group {groupIndex} portal {portalRef.PortalIndex} has invalid vertices");
                    continue;
                }

                // Validate portal geometry
                if (!mopv.ValidatePortalGeometry(portal.StartVertex, portal.Count))
                {
                    validationErrors.Add($"Group {groupIndex} portal {portalRef.PortalIndex} has invalid geometry");
                }
            }
        }

        public IReadOnlyList<string> GetValidationErrors() => validationErrors;
    }
} 