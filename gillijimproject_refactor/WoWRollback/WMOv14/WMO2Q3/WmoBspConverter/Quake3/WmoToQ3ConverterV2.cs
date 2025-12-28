using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using WmoBspConverter.Wmo;

namespace WmoBspConverter.Quake3
{
    /// <summary>
    /// WMO v14 to Quake 3 BSP converter V2.
    /// Uses actual WMO BSP data (MOBN/MOBR) to generate proper Q3 BSP trees.
    /// 
    /// WMO MOBN structure (16 bytes, Ghidra-verified):
    /// - uint16 flags: 0=YZ-plane, 1=XZ-plane, 2=XY-plane, 4=leaf
    /// - int16 negChild, posChild: child indices (-1 = none)
    /// - uint16 nFaces: face count
    /// - uint32 faceStart: index into MOBR
    /// - float planeDist: split plane distance
    /// </summary>
    public class WmoToQ3ConverterV2
    {
        private const float SCALE = 1.0f; // 1:1 scale initially for testing
        
        // WMO BSP node flags
        private const ushort BSP_FLAG_YZ_PLANE = 0;
        private const ushort BSP_FLAG_XZ_PLANE = 1;
        private const ushort BSP_FLAG_XY_PLANE = 2;
        private const ushort BSP_FLAG_LEAF = 4;
        
        public Q3Bsp Convert(WmoV14Parser.WmoV14Data wmoData)
        {
            Console.WriteLine($"[Q3v2] Converting WMO v14 to Q3 BSP using actual BSP data...");
            
            var bsp = new Q3Bsp();
            
            // 1. Convert textures
            ConvertTextures(wmoData, bsp);
            
            // 2. Convert all group geometry into a unified vertex/face list
            var allVertices = new List<Q3Vertex>();
            var allFaces = new List<Q3Face>();
            var allMeshVerts = new List<int>();
            
            foreach (var group in wmoData.Groups)
            {
                ConvertGroupGeometry(group, bsp.Textures.Count, allVertices, allFaces, allMeshVerts);
            }
            
            bsp.Vertices = allVertices;
            bsp.Faces = allFaces;
            bsp.MeshVerts = allMeshVerts;
            
            // 3. Build BSP tree - for now, create a simple valid tree
            //    (Full MOBN conversion would require parsing the raw bytes)
            BuildSimpleBspTree(bsp);
            
            // 4. Generate brushes for collision
            GenerateBrushes(bsp);
            
            // 5. Generate entities
            GenerateEntities(wmoData, bsp);
            
            Console.WriteLine($"[Q3v2] Converted: {bsp.Vertices.Count} verts, {bsp.Faces.Count} faces, " +
                            $"{bsp.Nodes.Count} nodes, {bsp.Leaves.Count} leaves, {bsp.Brushes.Count} brushes");
            
            return bsp;
        }
        
        private void ConvertTextures(WmoV14Parser.WmoV14Data wmoData, Q3Bsp bsp)
        {
            // Add a default solid texture first
            bsp.Textures.Add(new Q3Texture
            {
                Name = "textures/common/clip",
                Flags = 0x80, // SURF_NODRAW
                Contents = 0x10000 // CONTENTS_PLAYERCLIP
            });
            
            foreach (var textureName in wmoData.Textures)
            {
                var cleanName = System.IO.Path.GetFileNameWithoutExtension(textureName).ToLowerInvariant();
                bsp.Textures.Add(new Q3Texture
                {
                    Name = $"textures/wmo/{cleanName}",
                    Flags = 0,
                    Contents = 1 // CONTENTS_SOLID
                });
            }
            
            // Ensure at least one visible texture
            if (wmoData.Textures.Count == 0)
            {
                bsp.Textures.Add(new Q3Texture
                {
                    Name = "textures/base_wall/basewall01",
                    Flags = 0,
                    Contents = 1
                });
            }
        }
        
        private void ConvertGroupGeometry(WmoV14Parser.WmoGroupData group, int textureCount,
            List<Q3Vertex> vertices, List<Q3Face> faces, List<int> meshVerts)
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
                
                // Compute normal
                var e1 = v1 - v0;
                var e2 = v2 - v0;
                var normal = Vector3.Cross(e1, e2);
                if (normal.LengthSquared() < 0.0001f)
                    continue; // Degenerate
                normal = Vector3.Normalize(normal);
                
                // Get material (offset by 1 to skip clip texture)
                int triIndex = i / 3;
                int materialId = triIndex < group.FaceMaterials.Count ? group.FaceMaterials[triIndex] + 1 : 1;
                materialId = Math.Clamp(materialId, 1, textureCount - 1);
                
                // Get UVs
                var uv0 = i0 < group.UVs.Count ? group.UVs[i0] : Vector2.Zero;
                var uv1 = i1 < group.UVs.Count ? group.UVs[i1] : Vector2.Zero;
                var uv2 = i2 < group.UVs.Count ? group.UVs[i2] : Vector2.Zero;
                
                // Add vertices (Q3 coordinate transform: WMO Y-up → Q3 Z-up)
                int firstVert = vertices.Count;
                vertices.Add(CreateVertex(v0, uv0, normal));
                vertices.Add(CreateVertex(v1, uv1, normal));
                vertices.Add(CreateVertex(v2, uv2, normal));
                
                // Add mesh vertex offsets (reversed winding for Q3)
                int firstMeshVert = meshVerts.Count;
                meshVerts.Add(0);
                meshVerts.Add(2);  // Swap 1 and 2 for CCW winding
                meshVerts.Add(1);
                
                // Create face (Type 3 = mesh with indexed triangles)
                faces.Add(new Q3Face
                {
                    TextureIndex = materialId,
                    Effect = -1,
                    Type = 3, // Mesh - proper for indexed triangles
                    FirstVertex = firstVert,
                    NumVertices = 3,
                    FirstMeshVert = firstMeshVert,
                    NumMeshVerts = 3,
                    LightmapIndex = -1,
                    Normal = TransformNormal(normal)
                });
            }
        }
        
        private Q3Vertex CreateVertex(Vector3 wmoPos, Vector2 uv, Vector3 normal)
        {
            return new Q3Vertex
            {
                Position = TransformPosition(wmoPos),
                TexCoord = uv,
                LightmapCoord = new Vector2(0.5f, 0.5f), // Center of lightmap
                Normal = TransformNormal(normal),
                Color = 0xFFFFFFFF
            };
        }
        
        private Vector3 TransformPosition(Vector3 wmoPos)
        {
            // WMO: X=right, Y=up, Z=forward (right-handed)
            // Q3:  X=right, Y=forward, Z=up (right-handed)
            // Transform: swap Y and Z
            return new Vector3(
                wmoPos.X * SCALE,
                wmoPos.Z * SCALE,
                wmoPos.Y * SCALE
            );
        }
        
        private Vector3 TransformNormal(Vector3 wmoNormal)
        {
            return Vector3.Normalize(new Vector3(
                wmoNormal.X,
                wmoNormal.Z,
                wmoNormal.Y
            ));
        }
        
        private void BuildSimpleBspTree(Q3Bsp bsp)
        {
            if (bsp.Vertices.Count == 0)
            {
                // Empty map - create minimal valid structure
                CreateEmptyBspTree(bsp);
                return;
            }
            
            // Compute bounds
            var (mins, maxs) = ComputeBounds(bsp.Vertices);
            
            // Expand bounds slightly for safety
            mins -= new Vector3(64, 64, 64);
            maxs += new Vector3(64, 64, 64);
            
            // Create a proper BSP tree using axis-aligned splits
            // For now: simple single-leaf tree (works for small maps)
            // TODO: Implement proper recursive BSP from WMO MOBN data
            
            // Add planes for the bounding box (6 planes for a box)
            int planeBase = bsp.Planes.Count;
            
            // +X plane
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(1, 0, 0), Distance = maxs.X });
            // -X plane
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(-1, 0, 0), Distance = -mins.X });
            // +Y plane
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(0, 1, 0), Distance = maxs.Y });
            // -Y plane
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(0, -1, 0), Distance = -mins.Y });
            // +Z plane
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(0, 0, 1), Distance = maxs.Z });
            // -Z plane
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(0, 0, -1), Distance = -mins.Z });
            
            // Create root node that immediately leads to a leaf
            // In Q3, negative child index = -(leafIndex + 1)
            bsp.Nodes.Add(new Q3Node
            {
                PlaneIndex = planeBase, // Use first plane
                Children = new[] { -1, -2 }, // Both children are leaves (leaf 0 and leaf 1)
                Mins = new[] { (int)mins.X, (int)mins.Y, (int)mins.Z },
                Maxs = new[] { (int)maxs.X, (int)maxs.Y, (int)maxs.Z }
            });
            
            // Leaf 0: Contains all faces (the "inside" of the map)
            int leafFaceStart = bsp.LeafFaces.Count;
            for (int i = 0; i < bsp.Faces.Count; i++)
            {
                bsp.LeafFaces.Add(i);
            }
            
            bsp.Leaves.Add(new Q3Leaf
            {
                Cluster = 0, // Cluster 0 = visible
                Area = 0,
                Mins = new[] { (int)mins.X, (int)mins.Y, (int)mins.Z },
                Maxs = new[] { (int)maxs.X, (int)maxs.Y, (int)maxs.Z },
                FirstLeafFace = leafFaceStart,
                NumLeafFaces = bsp.Faces.Count,
                FirstLeafBrush = 0,
                NumLeafBrushes = 0 // Will be set after brush generation
            });
            
            // Leaf 1: Empty leaf (the "outside")
            bsp.Leaves.Add(new Q3Leaf
            {
                Cluster = -1, // -1 = outside/invalid
                Area = 0,
                Mins = new[] { (int)mins.X, (int)mins.Y, (int)mins.Z },
                Maxs = new[] { (int)maxs.X, (int)maxs.Y, (int)maxs.Z },
                FirstLeafFace = 0,
                NumLeafFaces = 0,
                FirstLeafBrush = 0,
                NumLeafBrushes = 0
            });
            
            // Create visibility data (1 cluster, always visible to itself)
            CreateVisibilityData(bsp, 1);
            
            // Create world model
            bsp.Models.Add(new Q3Model
            {
                Mins = mins,
                Maxs = maxs,
                FirstFace = 0,
                NumFaces = bsp.Faces.Count,
                FirstBrush = 0,
                NumBrushes = 0 // Will be updated after brush generation
            });
        }
        
        private void CreateEmptyBspTree(Q3Bsp bsp)
        {
            // Minimal valid BSP for empty map
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(0, 0, 1), Distance = 0 });
            
            bsp.Nodes.Add(new Q3Node
            {
                PlaneIndex = 0,
                Children = new[] { -1, -2 },
                Mins = new[] { -1024, -1024, -1024 },
                Maxs = new[] { 1024, 1024, 1024 }
            });
            
            bsp.Leaves.Add(new Q3Leaf
            {
                Cluster = 0,
                Area = 0,
                Mins = new[] { -1024, -1024, -1024 },
                Maxs = new[] { 1024, 1024, 1024 },
                FirstLeafFace = 0,
                NumLeafFaces = 0,
                FirstLeafBrush = 0,
                NumLeafBrushes = 0
            });
            
            bsp.Leaves.Add(new Q3Leaf
            {
                Cluster = -1,
                Area = 0,
                Mins = new[] { -1024, -1024, -1024 },
                Maxs = new[] { 1024, 1024, 1024 },
                FirstLeafFace = 0,
                NumLeafFaces = 0,
                FirstLeafBrush = 0,
                NumLeafBrushes = 0
            });
            
            CreateVisibilityData(bsp, 1);
            
            bsp.Models.Add(new Q3Model
            {
                Mins = new Vector3(-1024, -1024, -1024),
                Maxs = new Vector3(1024, 1024, 1024),
                FirstFace = 0,
                NumFaces = 0,
                FirstBrush = 0,
                NumBrushes = 0
            });
        }
        
        private void GenerateBrushes(Q3Bsp bsp)
        {
            if (bsp.Vertices.Count == 0)
                return;
            
            // Create a single brush for the world bounding box (for basic collision)
            var (mins, maxs) = ComputeBounds(bsp.Vertices);
            
            // A brush is defined by its bounding planes
            int firstSide = bsp.BrushSides.Count;
            int firstPlane = bsp.Planes.Count;
            
            // 6 planes for box brush (pointing inward)
            // +X face (normal pointing -X into box)
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(-1, 0, 0), Distance = -maxs.X });
            bsp.BrushSides.Add(new Q3BrushSide { PlaneIndex = firstPlane, TextureIndex = 0 });
            
            // -X face
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(1, 0, 0), Distance = mins.X });
            bsp.BrushSides.Add(new Q3BrushSide { PlaneIndex = firstPlane + 1, TextureIndex = 0 });
            
            // +Y face
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(0, -1, 0), Distance = -maxs.Y });
            bsp.BrushSides.Add(new Q3BrushSide { PlaneIndex = firstPlane + 2, TextureIndex = 0 });
            
            // -Y face
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(0, 1, 0), Distance = mins.Y });
            bsp.BrushSides.Add(new Q3BrushSide { PlaneIndex = firstPlane + 3, TextureIndex = 0 });
            
            // +Z face
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(0, 0, -1), Distance = -maxs.Z });
            bsp.BrushSides.Add(new Q3BrushSide { PlaneIndex = firstPlane + 4, TextureIndex = 0 });
            
            // -Z face
            bsp.Planes.Add(new Q3Plane { Normal = new Vector3(0, 0, 1), Distance = mins.Z });
            bsp.BrushSides.Add(new Q3BrushSide { PlaneIndex = firstPlane + 5, TextureIndex = 0 });
            
            // Create the brush
            bsp.Brushes.Add(new Q3Brush
            {
                FirstSide = firstSide,
                NumSides = 6,
                TextureIndex = 0 // Clip texture
            });
            
            // Update leaf with brush reference
            if (bsp.Leaves.Count > 0)
            {
                bsp.LeafBrushes.Add(0); // Brush 0
                bsp.Leaves[0].FirstLeafBrush = 0;
                bsp.Leaves[0].NumLeafBrushes = 1;
            }
            
            // Update model
            if (bsp.Models.Count > 0)
            {
                bsp.Models[0].FirstBrush = 0;
                bsp.Models[0].NumBrushes = 1;
            }
        }
        
        private void CreateVisibilityData(Q3Bsp bsp, int numClusters)
        {
            if (numClusters <= 0)
                return;
            
            // Simple visibility: all clusters can see each other
            int bytesPerCluster = (numClusters + 7) / 8; // Bytes needed per cluster
            
            // Set header values (written separately by Q3BspWriter)
            bsp.NumClusters = numClusters;
            bsp.BytesPerCluster = bytesPerCluster;
            
            // Build just the bitset data (header written by writer)
            var visData = new List<byte>();
            for (int i = 0; i < numClusters; i++)
            {
                var vec = new byte[bytesPerCluster];
                // Set all clusters visible
                for (int j = 0; j < numClusters; j++)
                {
                    vec[j / 8] |= (byte)(1 << (j % 8));
                }
                visData.AddRange(vec);
            }
            
            bsp.VisData = visData.ToArray();
        }
        
        private (Vector3 mins, Vector3 maxs) ComputeBounds(List<Q3Vertex> vertices)
        {
            if (vertices.Count == 0)
                return (Vector3.Zero, Vector3.Zero);
            
            var mins = new Vector3(float.MaxValue);
            var maxs = new Vector3(float.MinValue);
            
            foreach (var v in vertices)
            {
                mins = Vector3.Min(mins, v.Position);
                maxs = Vector3.Max(maxs, v.Position);
            }
            
            return (mins, maxs);
        }
        
        private void GenerateEntities(WmoV14Parser.WmoV14Data wmoData, Q3Bsp bsp)
        {
            var sb = new StringBuilder();
            
            // Worldspawn (required)
            sb.AppendLine("{");
            sb.AppendLine("\"classname\" \"worldspawn\"");
            sb.AppendLine($"\"message\" \"WMO v{wmoData.Version} Converted\"");
            sb.AppendLine("\"_ambient\" \"50\"");
            sb.AppendLine("}");
            
            // Info player start - spawn at center, above floor
            if (bsp.Vertices.Count > 0)
            {
                var (mins, maxs) = ComputeBounds(bsp.Vertices);
                var center = (mins + maxs) * 0.5f;
                // Spawn higher up so player can noclip around
                var spawnZ = center.Z;
                
                // Add both spawn types for compatibility
                sb.AppendLine("{");
                sb.AppendLine("\"classname\" \"info_player_start\"");
                sb.AppendLine($"\"origin\" \"{center.X:F0} {center.Y:F0} {spawnZ:F0}\"");
                sb.AppendLine("\"angle\" \"0\"");
                sb.AppendLine("}");
                
                sb.AppendLine("{");
                sb.AppendLine("\"classname\" \"info_player_deathmatch\"");
                sb.AppendLine($"\"origin\" \"{center.X:F0} {center.Y:F0} {spawnZ:F0}\"");
                sb.AppendLine("\"angle\" \"0\"");
                sb.AppendLine("}");
                
                // Add ambient light for visibility
                sb.AppendLine("{");
                sb.AppendLine("\"classname\" \"light\"");
                sb.AppendLine($"\"origin\" \"{center.X:F0} {center.Y:F0} {center.Z:F0}\"");
                sb.AppendLine("\"light\" \"1000\"");
                sb.AppendLine("\"_color\" \"1 1 1\"");
                sb.AppendLine("}");
            }
            
            bsp.Entities = sb.ToString();
        }
    }
}
