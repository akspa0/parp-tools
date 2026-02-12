using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using WmoBspConverter.Wmo;

namespace WmoBspConverter.Quake3
{
    /// <summary>
    /// Converts WMO v14 data to Quake 3 BSP format.
    /// Uses proper Q3 structures and coordinate transforms.
    /// </summary>
    public class WmoToQ3Converter
    {
        private const float SCALE = 100.0f; // WMO to Q3 scale
        
        public Q3Bsp Convert(WmoV14Parser.WmoV14Data wmoData)
        {
            Console.WriteLine($"[Q3] Converting WMO v14 to Q3 BSP...");
            
            var bsp = new Q3Bsp();
            
            // Convert textures
            ConvertTextures(wmoData, bsp);
            
            // Convert geometry
            ConvertGeometry(wmoData, bsp);
            
            // Build BSP structures
            BuildBspStructures(wmoData, bsp);
            
            // Generate entities
            GenerateEntities(wmoData, bsp);
            
            Console.WriteLine($"[Q3] Converted: {bsp.Vertices.Count} verts, {bsp.Faces.Count} faces, {bsp.Textures.Count} textures");
            
            return bsp;
        }
        
        private void ConvertTextures(WmoV14Parser.WmoV14Data wmoData, Q3Bsp bsp)
        {
            foreach (var textureName in wmoData.Textures)
            {
                var cleanName = System.IO.Path.GetFileNameWithoutExtension(textureName).ToLowerInvariant();
                bsp.Textures.Add(new Q3Texture
                {
                    Name = $"wmo/{cleanName}",
                    Flags = 0,
                    Contents = 1 // CONTENTS_SOLID
                });
            }
            
            // Add default texture if none
            if (bsp.Textures.Count == 0)
            {
                bsp.Textures.Add(new Q3Texture
                {
                    Name = "common/caulk",
                    Flags = 0,
                    Contents = 1
                });
            }
        }
        
        private void ConvertGeometry(WmoV14Parser.WmoV14Data wmoData, Q3Bsp bsp)
        {
            foreach (var group in wmoData.Groups)
            {
                ConvertGroup(group, bsp);
            }
        }
        
        private void ConvertGroup(WmoV14Parser.WmoGroupData group, Q3Bsp bsp)
        {
            if (group.Indices.Count < 3 || group.Vertices.Count == 0)
                return;
            
            // Process triangles
            for (int i = 0; i + 2 < group.Indices.Count; i += 3)
            {
                var i0 = group.Indices[i];
                var i1 = group.Indices[i + 1];
                var i2 = group.Indices[i + 2];
                
                if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
                    continue;
                
                var v0 = group.Vertices[i0];
                var v1 = group.Vertices[i1];
                var v2 = group.Vertices[i2];
                
                // Check for degenerate triangle
                var e1 = v1 - v0;
                var e2 = v2 - v0;
                var normal = Vector3.Normalize(Vector3.Cross(e1, e2));
                if (float.IsNaN(normal.X) || normal.Length() < 0.001f)
                    continue;
                
                // Get material for this triangle
                int triIndex = i / 3;
                int materialId = triIndex < group.FaceMaterials.Count ? group.FaceMaterials[triIndex] : 0;
                materialId = Math.Max(0, Math.Min(materialId, bsp.Textures.Count - 1));
                
                // Get UVs if available
                var uv0 = i0 < group.UVs.Count ? group.UVs[i0] : Vector2.Zero;
                var uv1 = i1 < group.UVs.Count ? group.UVs[i1] : Vector2.Zero;
                var uv2 = i2 < group.UVs.Count ? group.UVs[i2] : Vector2.Zero;
                
                // Transform to Q3 coordinates and add vertices
                var firstVert = bsp.Vertices.Count;
                bsp.Vertices.Add(CreateQ3Vertex(v0, uv0, normal));
                bsp.Vertices.Add(CreateQ3Vertex(v1, uv1, normal));
                bsp.Vertices.Add(CreateQ3Vertex(v2, uv2, normal));
                
                // Add mesh indices (relative to first vertex)
                var firstMeshVert = bsp.MeshVerts.Count;
                bsp.MeshVerts.Add(0);
                bsp.MeshVerts.Add(1);
                bsp.MeshVerts.Add(2);
                
                // Create face
                bsp.Faces.Add(new Q3Face
                {
                    TextureIndex = materialId,
                    Effect = -1,
                    Type = 3, // Mesh face
                    FirstVertex = firstVert,
                    NumVertices = 3,
                    FirstMeshVert = firstMeshVert,
                    NumMeshVerts = 3,
                    Normal = TransformNormal(normal)
                });
            }
        }
        
        private Q3Vertex CreateQ3Vertex(Vector3 wmoPos, Vector2 uv, Vector3 normal)
        {
            return new Q3Vertex
            {
                Position = TransformPosition(wmoPos),
                TexCoord = uv,
                LightmapCoord = Vector2.Zero,
                Normal = TransformNormal(normal),
                Color = 0xFFFFFFFF
            };
        }
        
        private Vector3 TransformPosition(Vector3 wmoPos)
        {
            // WMO: X=left/right, Y=height, Z=forward/back
            // Q3:  X=left/right, Y=forward/back, Z=height
            // Transform: (X, Y, Z) → (X, Z, -Y) × scale
            return new Vector3(
                wmoPos.X * SCALE,
                wmoPos.Z * SCALE,
                -wmoPos.Y * SCALE
            );
        }
        
        private Vector3 TransformNormal(Vector3 wmoNormal)
        {
            // Same transform as position but no scaling
            return Vector3.Normalize(new Vector3(
                wmoNormal.X,
                wmoNormal.Z,
                -wmoNormal.Y
            ));
        }
        
        private void BuildBspStructures(WmoV14Parser.WmoV14Data wmoData, Q3Bsp bsp)
        {
            // Build planes from faces
            foreach (var face in bsp.Faces)
            {
                bsp.Planes.Add(new Q3Plane
                {
                    Normal = face.Normal,
                    Distance = 0 // Simplified - would need proper calculation
                });
            }
            
            // Create minimal BSP tree (single node/leaf)
            var bounds = ComputeBounds(bsp.Vertices);
            
            bsp.Nodes.Add(new Q3Node
            {
                PlaneIndex = 0,
                Children = new[] { -1, -1 }, // Both children are leaves
                Mins = new[] { (int)bounds.min.X, (int)bounds.min.Y, (int)bounds.min.Z },
                Maxs = new[] { (int)bounds.max.X, (int)bounds.max.Y, (int)bounds.max.Z }
            });
            
            bsp.Leaves.Add(new Q3Leaf
            {
                Cluster = 0,
                Area = 0,
                Mins = new[] { (int)bounds.min.X, (int)bounds.min.Y, (int)bounds.min.Z },
                Maxs = new[] { (int)bounds.max.X, (int)bounds.max.Y, (int)bounds.max.Z },
                FirstLeafFace = 0,
                NumLeafFaces = bsp.Faces.Count,
                FirstLeafBrush = 0,
                NumLeafBrushes = 0
            });
            
            // Add all faces to leaf
            for (int i = 0; i < bsp.Faces.Count; i++)
            {
                bsp.LeafFaces.Add(i);
            }
            
            // Create model
            bsp.Models.Add(new Q3Model
            {
                Mins = bounds.min,
                Maxs = bounds.max,
                FirstFace = 0,
                NumFaces = bsp.Faces.Count,
                FirstBrush = 0,
                NumBrushes = 0
            });
        }
        
        private (Vector3 min, Vector3 max) ComputeBounds(List<Q3Vertex> vertices)
        {
            if (vertices.Count == 0)
                return (Vector3.Zero, Vector3.Zero);
            
            var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            
            foreach (var v in vertices)
            {
                min = Vector3.Min(min, v.Position);
                max = Vector3.Max(max, v.Position);
            }
            
            return (min, max);
        }
        
        private void GenerateEntities(WmoV14Parser.WmoV14Data wmoData, Q3Bsp bsp)
        {
            var entities = new StringBuilder();
            
            // Worldspawn
            entities.AppendLine("{");
            entities.AppendLine("\"classname\" \"worldspawn\"");
            entities.AppendLine($"\"message\" \"WMO v{wmoData.Version} - {wmoData.Groups.Count} groups\"");
            entities.AppendLine("}");
            
            // Player spawn
            if (bsp.Vertices.Count > 0)
            {
                var bounds = ComputeBounds(bsp.Vertices);
                var center = (bounds.min + bounds.max) * 0.5f;
                var spawnZ = bounds.min.Z + 64f;
                
                entities.AppendLine("{");
                entities.AppendLine("\"classname\" \"info_player_deathmatch\"");
                entities.AppendLine($"\"origin\" \"{center.X:F1} {center.Y:F1} {spawnZ:F1}\"");
                entities.AppendLine("\"angle\" \"0\"");
                entities.AppendLine("}");
            }
            
            bsp.Entities = entities.ToString();
        }
    }
}
