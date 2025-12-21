using System.Numerics;
using WoWRollback.PM4Module.Pipeline;

namespace WoWRollback.PM4Module.Decoding;

public class Pm4ObjectBuilder
{
    /// <summary>
    /// Reconstructs WMO candidates from PM4 data.
    /// Groups surfaces by CK24, then separates instances by vertex connectivity.
    /// </summary>
    public static List<Pm4WmoCandidate> BuildCandidates(Pm4FileStructure pm4, int tileX, int tileY)
    {
        var candidates = new List<Pm4WmoCandidate>();
        
        // Group surfaces by CK24 (excluding 0 which is nav mesh)
        var surfacesByCk24 = pm4.Surfaces
            .Where(s => s.CK24 != 0)
            .GroupBy(s => s.CK24);

        foreach (var group in surfacesByCk24)
        {
            uint ck24 = group.Key;
            var surfaces = group.ToList();

            // === INSTANCE SEPARATION ===
            // Use Union-Find to group surfaces that share vertices (Mesh Indices refer to same Mesh Vertex)
            // Note: We group by Vertex INDEX, not Vertex POSITION (topology based)
            
            // Map: VertexIndex -> List of SurfaceIndices that use it
            var vertexToSurfaces = new Dictionary<uint, List<int>>();
            for (int si = 0; si < surfaces.Count; si++)
            {
                var surf = surfaces[si];
                for (int i = 0; i < surf.IndexCount; i++)
                {
                    uint msviIdx = surf.MsviFirstIndex + (uint)i;
                    if (msviIdx < pm4.MeshIndices.Count)
                    {
                        uint msvtIdx = pm4.MeshIndices[(int)msviIdx];
                        if (!vertexToSurfaces.ContainsKey(msvtIdx))
                            vertexToSurfaces[msvtIdx] = new List<int>();
                        vertexToSurfaces[msvtIdx].Add(si);
                    }
                }
            }

            // Union-Find setup
            int n = surfaces.Count;
            int[] parent = new int[n];
            for (int i = 0; i < n; i++) parent[i] = i;

            int Find(int x)
            {
                if (parent[x] != x) parent[x] = Find(parent[x]);
                return parent[x];
            }

            void Union(int a, int b)
            {
                int rootA = Find(a);
                int rootB = Find(b);
                if (rootA != rootB) parent[rootA] = rootB;
            }

            // Union all surfaces sharing a vertex
            foreach (var list in vertexToSurfaces.Values)
            {
                for (int i = 1; i < list.Count; i++)
                {
                    Union(list[0], list[i]);
                }
            }

            // Group surfaces by root (Instance)
            var instances = new Dictionary<int, List<MsurChunk>>();
            for (int i = 0; i < n; i++)
            {
                int root = Find(i);
                if (!instances.ContainsKey(root))
                    instances[root] = new List<MsurChunk>();
                instances[root].Add(surfaces[i]);
            }

            // Create a candidate for each instance
            foreach (var instanceSurfs in instances.Values)
            {
                var geometry = ExtractGeometry(instanceSurfs, pm4);
                
                // Calculate Dominant Angle
                float domAngle = CalculateDominantAngle(instanceSurfs);

                // Find MPRL Data (Rotation/Position)
                var (mprlRot, mprlPos) = FindMprlData(geometry.BoundsMin, geometry.BoundsMax, pm4.PositionRefs);

                // Type Flags (Byte 2)
                byte typeFlags = (byte)((ck24 >> 16) & 0xFF);

                // Create Candidate
                var candidate = new Pm4WmoCandidate(
                    CK24: ck24,
                    InstanceId: instances.Values.ToList().IndexOf(instanceSurfs),
                    TileX: tileX,
                    TileY: tileY,
                    BoundsMin: geometry.BoundsMin,
                    BoundsMax: geometry.BoundsMax,
                    DominantAngle: domAngle,
                    SurfaceCount: instanceSurfs.Count,
                    VertexCount: geometry.Vertices.Count, // Vertex count of the main mesh
                    TypeFlags: typeFlags,
                    MprlRotationDegrees: mprlRot,
                    MprlPosition: mprlPos,
                    DebugGeometry: geometry.Vertices,
                    DebugFaces: geometry.Faces,
                    DebugMscnVertices: geometry.MscnVertices
                );

                candidates.Add(candidate);
            }
        }

        return candidates;
    }

    private static (float? Rotation, Vector3? Position) FindMprlData(Vector3 min, Vector3 max, List<MprlChunk> mprlList)
    {
        if (mprlList == null || mprlList.Count == 0) return (null, null);

        var center = (min + max) / 2f;
        
        // Search for MPRL entry near the object
        // MPRL stored as Y, Z, X -> Need to swap to X, Y, Z to match our World Space
        
        var nearby = mprlList
            .Where(p => p.EntryType == 0) // Non-terminator
            .Select(p => new {
                Entry = p,
                // Apply Swap: X=Z_disk, Y=X_disk, Z=Y_disk
                // Wait. Spec: Stored X, Z, Y -> Loaded X, Y, Z -> Swap.
                // Pm4Decoder Raw: v.X, v.Y, v.Z.
                // Legacy logic: X=RawZ, Y=RawX, Z=RawY.
                Pos = new Vector3(p.Position.Z, p.Position.X, p.Position.Y)
            })
            .Where(x => 
                x.Pos.X >= min.X - 50 && x.Pos.X <= max.X + 50 &&
                x.Pos.Y >= min.Y - 50 && x.Pos.Y <= max.Y + 50)
            .OrderBy(x => Vector3.Distance(x.Pos, center))
            .FirstOrDefault();

        if (nearby != null)
        {
            float rot = 360f * nearby.Entry.Rotation / 65536f;
            return (rot, nearby.Pos);
        }

        return (null, null);
    }

    private record ExtractedGeometry(List<Vector3> Vertices, List<int[]> Faces, List<Vector3> MscnVertices, Vector3 BoundsMin, Vector3 BoundsMax);

    private static ExtractedGeometry ExtractGeometry(List<MsurChunk> surfaces, Pm4FileStructure pm4)
    {
        var vertices = new List<Vector3>();
        var faces = new List<int[]>();
        var mscnVertices = new List<Vector3>();
        var msvtToLocalMap = new Dictionary<uint, int>(); // Global MSVT Index -> Local Vertex Index

        // 1. Collect Mesh Vertices (MSVT)
        foreach (var surf in surfaces)
        {
            var faceIndices = new List<int>();
            for (int i = 0; i < surf.IndexCount; i++)
            {
                uint msviIdx = surf.MsviFirstIndex + (uint)i;
                if (msviIdx < pm4.MeshIndices.Count)
                {
                    uint msvtIdx = pm4.MeshIndices[(int)msviIdx];
                    
                    if (!msvtToLocalMap.TryGetValue(msvtIdx, out int localIdx))
                    {
                        if (msvtIdx < pm4.MeshVertices.Count)
                        {
                            localIdx = vertices.Count;
                            // SWAP X/Y to match World Space (Spec: Stored Y, X, Z -> World X, Y, Z)
                            var rawV = pm4.MeshVertices[(int)msvtIdx];
                            vertices.Add(new Vector3(rawV.Y, rawV.X, rawV.Z));
                            msvtToLocalMap[msvtIdx] = localIdx;
                        }
                    }
                    
                    if (msvtToLocalMap.ContainsKey(msvtIdx))
                        faceIndices.Add(msvtToLocalMap[msvtIdx]);
                }
            }
            if (faceIndices.Count >= 3)
                faces.Add(faceIndices.ToArray());
        }

        // 2. Collect MSCN Vertices (Vertical Walls) linked via MdosIndex
        foreach (var surf in surfaces)
        {
            if (surf.MdosIndex < pm4.SceneNodes.Count)
            {
                var mscnNode = pm4.SceneNodes[(int)surf.MdosIndex];
                // SWAP X/Y for MSCN as well to match MSVT frame
                mscnVertices.Add(new Vector3(mscnNode.Y, mscnNode.X, mscnNode.Z));
            }
        }

        // Calculate Bounds (Combine both for overall bounds)
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        
        var allPoints = vertices.Concat(mscnVertices);
        if (allPoints.Any())
        {
            foreach (var v in allPoints)
            {
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
        }
        else
        {
            min = Vector3.Zero;
            max = Vector3.Zero;
        }

        return new ExtractedGeometry(vertices, faces, mscnVertices, min, max);
    }

    private static float CalculateDominantAngle(List<MsurChunk> surfaces)
    {
        // Histogram of Wall Angles (Normal.Z near 0)
        // 72 bins of 5 degrees
        float[] histogram = new float[72];
        float maxArea = 0;
        int maxBin = -1;

        foreach (var surf in surfaces)
        {
            // Only consider walls (Z component of normal is small)
            if (Math.Abs(surf.Normal.Z) < 0.7f) // < ~45 degrees tilt
            {
                // Calculate yaw angle from Normal (X, Y)
                // Normal points OUT from wall. Wall angle is perpendicular.
                // Atan2(Y, X) is normal angle.
                double angle = Math.Atan2(surf.Normal.Y, surf.Normal.X) * 180.0 / Math.PI;
                if (angle < 0) angle += 360.0;
                
                int bin = (int)(angle / 5.0) % 72;
                
                // Weight by surface area approximation (Height^2 or similar?)
                // Since we don't have true area, we can use 1 or IndexCount
                float weight = surf.IndexCount; 
                
                histogram[bin] += weight;

                if (histogram[bin] > maxArea)
                {
                    maxArea = histogram[bin];
                    maxBin = bin;
                }
            }
        }

        if (maxBin != -1)
        {
            return maxBin * 5.0f;
        }
        return 0f;
    }
}
