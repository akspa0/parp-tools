using System.Numerics;
using WoWRollback.PM4Module.Pipeline;

namespace WoWRollback.PM4Module.Decoding;

public class Pm4ObjectBuilder
{
    /// <summary>
    /// Reconstructs WMO candidates from PM4 data.
    /// Groups surfaces by CK24, then splits by MSVI gaps to separate instances.
    /// All CK24 groups matched against both WMO and M2 libraries.
    /// </summary>
    public static List<Pm4WmoCandidate> BuildCandidates(Pm4FileStructure pm4, int tileX, int tileY)
    {
        var candidates = new List<Pm4WmoCandidate>();
        
        // Group ALL surfaces by CK24 (no filtering - PM4 doesn't distinguish WMO vs M2)
        var surfacesByCk24 = pm4.Surfaces
            .GroupBy(s => s.CK24);

        foreach (var group in surfacesByCk24)
        {
            uint ck24 = group.Key;
            var surfaces = group.ToList();
            
            // Split by MSVI gaps to separate individual object instances
            var instances = MsViGapSplitter.SplitByMsviGaps(surfaces, gapThreshold: 50);
            
            // Create one candidate per instance
            for (int instanceId = 0; instanceId < instances.Count; instanceId++)
            {
                var instanceSurfaces = instances[instanceId];
                
                // Extract geometry for this instance
                var geometry = ExtractGeometry(instanceSurfaces, pm4);
                
                if (geometry.Vertices.Count < 3)
                    continue;
                
                // Calculate Dominant Angle
                float domAngle = CalculateDominantAngle(instanceSurfaces);

                // Type Flags (Byte 2 of CK24)
                byte typeFlags = (byte)((ck24 >> 16) & 0xFF);
                
                // Find CLOSEST MPRL entry to use for position/rotation
                var centroid = (geometry.BoundsMin + geometry.BoundsMax) / 2f;
                var (mprlRot, mprlPos) = FindClosestMprl(centroid, pm4.PositionRefs);
                
                // Create candidate for this instance
                var candidate = new Pm4WmoCandidate(
                    CK24: ck24,
                    InstanceId: instanceId,
                    TileX: tileX,
                    TileY: tileY,
                    BoundsMin: geometry.BoundsMin,
                    BoundsMax: geometry.BoundsMax,
                    DominantAngle: domAngle,
                    SurfaceCount: instanceSurfaces.Count,
                    VertexCount: geometry.Vertices.Count,
                    TypeFlags: typeFlags,
                    MprlRotationDegrees: mprlRot,
                    MprlPosition: mprlPos ?? centroid, // Use centroid if no MPRL
                    DebugGeometry: geometry.Vertices,
                    DebugFaces: geometry.Faces,
                    DebugMscnVertices: geometry.MscnVertices
                );
                candidates.Add(candidate);
            }
        }

        return candidates;
    }
    
    /// <summary>
    /// Find all MPRL entries near the object bounds - each is a placement instance.
    /// </summary>
    /// <summary>
    /// Find the closest MPRL entry to a centroid point.
    /// </summary>
    private static (float? Rotation, Vector3? Position) FindClosestMprl(Vector3 centroid, List<MprlChunk> mprlList)
    {
        if (mprlList == null || mprlList.Count == 0) 
            return (null, null);

        var closest = mprlList
            .Where(p => p.EntryType == 0) // Skip terminators
            .Select(p => new {
                Entry = p,
                // Use raw coordinates to match Pm4Reader
                Pos = p.Position
            })
            .OrderBy(x => Vector3.DistanceSquared(x.Pos, centroid))
            .FirstOrDefault();

        if (closest != null && Vector3.Distance(closest.Pos, centroid) < 100) // Within 100 yards
        {
            float rot = 360f * closest.Entry.Rotation / 65536f;
            return (rot, closest.Pos);
        }

        return (null, null);
    }
    
    private static List<(float Rotation, Vector3 Position)> FindAllMprlPlacements(
        Vector3 min, Vector3 max, List<MprlChunk> mprlList)
    {
        var placements = new List<(float, Vector3)>();
        
        if (mprlList == null || mprlList.Count == 0) 
            return placements;

        foreach (var p in mprlList)
        {
            if (p.EntryType != 0) continue; // Skip terminators
            
            // Use raw coordinates to match Pm4Reader
            var pos = p.Position;
            
            // Check if within generous bounds
            if (pos.X >= min.X - 50 && pos.X <= max.X + 50 &&
                pos.Y >= min.Y - 50 && pos.Y <= max.Y + 50)
            {
                float rot = 360f * p.Rotation / 65536f;
                placements.Add((rot, pos));
            }
        }

        return placements;
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
                // Use raw coordinates to match Pm4Reader
                Pos = p.Position
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
                            // Use raw coordinates to match Pm4Reader and WMO coordinate system
                            vertices.Add(pm4.MeshVertices[(int)msvtIdx]);
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
                // Use raw coordinates to match Pm4Reader and WMO coordinate system
                mscnVertices.Add(pm4.SceneNodes[(int)surf.MdosIndex]);
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
