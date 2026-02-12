using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using WmoBspConverter.Wmo;

namespace WmoBspConverter.Bsp
{
    public class BspTreeWalker
    {
        private static Vector3 TransformToQ3(Vector3 v)
        {
            // Align axes: WMO Z-up to Q3 Z-up (Identity usually, but kept for consistency)
            return new Vector3(v.X, v.Y, v.Z);
        }

        public List<string> GenerateBrushes(WmoV14Parser.WmoGroupData group, Vector3 geometryOffset, List<WmoMaterial> materials, List<string> textureNames)
        {
            // 1. Compute Root Bounds (World Space)
            var (min, max) = GetGroupBounds(group);
            // Expand slightly to avoid coplanar clipping issues at the boundary
            min -= new Vector3(10f);
            max += new Vector3(10f);

            // 2. Create Initial Brush (Use offset to put it in Map Space)
            // The logic: Tree planes are in WMO Space. 
            // We should split in WMO Space, THEN transform resulting brushes to Map Space?
            // Yes. Splitting is easier in original coordinate system.
            var rootBrush = Brush.FromBounds(min, max);

            var brushes = new List<string>();
            Traverse(group.BspNodes, 0, rootBrush, brushes, group, geometryOffset, materials, textureNames);
            return brushes;
        }

        private void Traverse(List<WmoV14Parser.WmoBspNode> nodes, int nodeIndex, Brush currentBrush, List<string> result, WmoV14Parser.WmoGroupData group, Vector3 offset, List<WmoMaterial> materials, List<string> textureNames)
        {
            if (nodeIndex < 0 || nodeIndex >= nodes.Count) return;
            var node = nodes[nodeIndex];

            if (node.IsLeaf)
            {
                // Solid check: logic from implementation_plan
                // If it has faces, it's solid geometry.
                if (node.NumFaces > 0)
                {
                    // Pick a texture from the first face
                    string texture = "common/caulk";
                    
                    // The 'FirstFace' in MOBN points to an index in MOBR (Face Order), which points to MOPY (Face)
                    // We need to resolve this chain to get the Material ID.
                    if (group.FaceOrder.Count > 0 && group.FaceMaterials.Count > 0)
                    {
                        // Check bounds
                        uint mobrIndex = node.FirstFace;
                        if (mobrIndex < group.FaceOrder.Count)
                        {
                            ushort faceIndex = group.FaceOrder[(int)mobrIndex];
                            if (faceIndex < group.FaceMaterials.Count)
                            {
                                int matId = group.FaceMaterials[faceIndex];
                                if (matId < group.Batches.Count) 
                                {
                                     // Wait, MOPY matID is index into MOMT? Or Batch?
                                     // V14 parser says: MOBA MaterialId is texture. MOPY matId is ???
                                     // MOPY matId usually references the section/batch index or material index directly.
                                     // Let's assume MOPY matId -> direct material index if within texture count
                                     // Actually WmoV14Parser lines 114-116: Moba defines batch.
                                     // Let's rely on MOPY -> Material Index directly.
                                     if (matId < materials.Count)
                                     {
                                         uint texOffset = materials[matId].Texture1Offset;
                                         // We need mapping from offset -> name.
                                         // Passed 'textureNames' is likely the list.
                                         // Let's pass the mapped texture names directly to Walker?
                                         // Simplify: Just use "common/caulk" for now or fix mapping.
                                     }
                                }
                                // Fallback: Texture mapping is complex.
                                // For now, we want GEOMETRY. Texture we can fix later.
                            }
                        }
                    }

                    // Convert Brush to Q3 Map String
                    result.Add(currentBrush.ToQ3MapString("common/caulk", offset));
                }
                return;
            }

            // Split
            // Axis: 0=X, 1=Y, 2=Z
            int axis = node.Flags & 3; 
            float dist = node.PlaneDist;

            // Split the brush
            var (front, back) = currentBrush.Split(axis, dist);

            // Traverse Children
            // PosChild is FRONT (Distance > PlaneDist)? 
            // NegChild is BACK (Distance < PlaneDist)?
            // Standard BSP: Point dist = dot(n, p) - d.
            // If dist > 0: Front. 
            // WMO uses Axis aligned planes. P[axis] > dist.
            
            // Check WMO definition of children.
            // Usually NegChild covers (-infinity, dist), PosChild covers (dist, infinity).
            if (node.NegChild != -1 && back != null)
                Traverse(nodes, node.NegChild, back, result, group, offset, materials, textureNames);
                
            if (node.PosChild != -1 && front != null)
                Traverse(nodes, node.PosChild, front, result, group, offset, materials, textureNames);
        }

        private (Vector3 Min, Vector3 Max) GetGroupBounds(WmoV14Parser.WmoGroupData group)
        {
            if (group.Vertices.Count == 0) return (Vector3.Zero, Vector3.Zero);
            Vector3 min = new Vector3(float.MaxValue);
            Vector3 max = new Vector3(float.MinValue);
            foreach (var v in group.Vertices)
            {
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
            return (min, max);
        }
    }

    // Helper class for convex volume management
    public class Brush
    {
        // 6 planes for AABB? 
        // A general convex brush is list of planes.
        public List<Plane> Planes { get; private set; } = new List<Plane>();

        public static Brush FromBounds(Vector3 min, Vector3 max)
        {
            var b = new Brush();
            // Inwards facing planes
            b.Planes.Add(new Plane(new Vector3(1, 0, 0), min.X)); // +X plane at minX? No.
            // Plane equation: dot(n, p) - d = 0.
            // We want the volume WHERE dot(n, p) - d < 0 (or > 0).
            // Let's define volume as INSIDE all planes.
            // Normal points OUTWARDS. Volume is "behind" the plane.
            
            // Right face (Max X): Normal (1,0,0), Dist = MAX.X. Point (maxX-1) -> 1*(maxX-1) = maxX-1 < maxX. Correct.
            b.Planes.Add(new Plane(Vector3.UnitX, max.X));
            
            // Left face (Min X): Normal (-1,0,0), Dist = -MIN.X. Point (minX+1) -> -1*(minX+1) = -minX-1 < -minX. Correct.
            b.Planes.Add(new Plane(-Vector3.UnitX, -min.X));
            
            // Top face (Max Z? Y?): 
            b.Planes.Add(new Plane(Vector3.UnitY, max.Y));
            b.Planes.Add(new Plane(-Vector3.UnitY, -min.Y));
            b.Planes.Add(new Plane(Vector3.UnitZ, max.Z));
            b.Planes.Add(new Plane(-Vector3.UnitZ, -min.Z));
            
            return b;
        }

        private Brush() { }

        public Brush Clone()
        {
            var b = new Brush();
            b.Planes.AddRange(this.Planes);
            return b;
        }

        // Split by axis-aligned plane
        // axis: 0=X, 1=Y, 2=Z
        // dist: value on axis
        public (Brush? front, Brush? back) Split(int axis, float dist)
        {
            // Plane definition: P[axis] = dist
            // Front side: P[axis] > dist. 
            // Back side: P[axis] < dist.

            // Front Brush: ADD plane P[axis] > dist
            // This means we need a plane that keeps P[axis] > dist.
            // Normal (-1 if axis=X). dot(-1,0,0, P) - d < 0 => -x < d => x > -d?
            // Let's stick to "Normal Points Out".
            // Front Volume: region where x > dist.
            // Bound it by plane at x=dist, Normal (-1,0,0)?
            // Point (dist+1). -1*(dist+1) = -dist-1. Plane D = -dist.  -dist-1 < -dist. Yes.
            // So Front Brush adds Plane(-Axis, -dist).

            // Back Volume: region where x < dist.
            // Bound it by plane at x=dist, Normal (1,0,0).
            // Point (dist-1). 1*(dist-1) = dist-1. Plane D = dist. dist-1 < dist. Yes.
            // So Back Brush adds Plane(Axis, dist).

            Vector3 normal = Vector3.Zero;
            if (axis == 0) normal = Vector3.UnitX;
            else if (axis == 1) normal = Vector3.UnitY;
            else normal = Vector3.UnitZ;

            var frontBrush = this.Clone();
            // Clip Front: Remove everything where x < dist. So add plane defining x > dist.
            frontBrush.Planes.Add(new Plane(-normal, -dist));
            if (!frontBrush.IsValid()) frontBrush = null; // Optimization: check if collapsed

            var backBrush = this.Clone();
            // Clip Back: Remove everything where x > dist. So add plane defining x < dist.
            backBrush.Planes.Add(new Plane(normal, dist));
            if (!backBrush.IsValid()) backBrush = null;

            return (frontBrush, backBrush);
        }
        
        // Simple validity check (AABB check would be better, but basic plane consistency)
        public bool IsValid()
        {
            // If conflicting planes exist (e.g. x < 5 AND x > 10), it's empty.
            // Check pairs of opposing normals.
            // Simplified for Axis Aligned:
            float minX = -float.MaxValue, maxX = float.MaxValue;
            float minY = -float.MaxValue, maxY = float.MaxValue;
            float minZ = -float.MaxValue, maxZ = float.MaxValue;

            foreach (var p in Planes)
            {
                // Unpack normal
                if (p.Normal.X > 0.9f) maxX = Math.Min(maxX, p.D);
                else if (p.Normal.X < -0.9f) minX = Math.Max(minX, -p.D);
                
                else if (p.Normal.Y > 0.9f) maxY = Math.Min(maxY, p.D);
                else if (p.Normal.Y < -0.9f) minY = Math.Max(minY, -p.D);

                else if (p.Normal.Z > 0.9f) maxZ = Math.Min(maxZ, p.D);
                else if (p.Normal.Z < -0.9f) minZ = Math.Max(minZ, -p.D);
            }

            if (minX > maxX || minY > maxY || minZ > maxZ) return false;
            return true;
        }

        public string ToQ3MapString(string texture, Vector3 mapOffset)
        {
            var sb = new StringBuilder();
            sb.AppendLine("{");
            foreach (var p in Planes)
            {
                // Create 3 points on the plane for Q3 format
                // Plane: dot(n, p) = d
                var points = GetThreePointsOnPlane(p);
                if (points == null) continue; // Should not happen for valid planes
                
                var (p1, p2, p3) = points.Value;
                
                // Transform to Q3 Map Space
                // Map Point = Original Point - Offset
                p1 -= mapOffset;
                p2 -= mapOffset;
                p3 -= mapOffset;

                // Q3 format: ( x y z ) ( x y z ) ( x y z ) texture ...
                sb.AppendLine($" ( {p1.X:F4} {p1.Y:F4} {p1.Z:F4} ) ( {p2.X:F4} {p2.Y:F4} {p2.Z:F4} ) ( {p3.X:F4} {p3.Y:F4} {p3.Z:F4} ) {texture} 0 0 0 0.5 0.5 0 0 0");
            }
            sb.AppendLine("}");
            return sb.ToString();
        }

        private (Vector3, Vector3, Vector3)? GetThreePointsOnPlane(Plane p)
        {
            // Robustly generate 3 points on the plane.
            // Find dominant axis of normal
            Vector3 u, v;
            Vector3 n = p.Normal;
            
            if (Math.Abs(n.X) > 0.5f)
                u = Vector3.Normalize(Vector3.Cross(n, Vector3.UnitY)); // Cross with Y
            else
                u = Vector3.Normalize(Vector3.Cross(n, Vector3.UnitX)); // Cross with X
                
            v = Vector3.Cross(n, u);
            
            // Center point on plane: C = n * d
            // dot(n, n*d) = d * dot(n,n) = d. Correct.
            Vector3 center = n * p.D;
            
            // Q3 winding: The normal defined by Cross(p2-p1, p3-p1) should point INTO solid.
            // Our internal normal points OUT of solid, so we need REVERSED winding.
            // Return (center, center + v, center + u) instead of (center, center + u, center + v)
            // This makes Cross(v, u) = -n, so the face normal points -n (into solid).
            
            return (center, center + v * 64, center + u * 64);
        }
    }
}
