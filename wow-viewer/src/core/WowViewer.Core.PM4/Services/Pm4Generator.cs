using System.Numerics;

namespace WowViewer.Core.PM4.Services;

public static class Pm4Generator
{
    private const float MapOrigin = 17066.666f;
    private const float TileSize = 533.333f;

    public static Pm4GenerationData GenerateFromCollisionMesh(
        IReadOnlyList<Vector3> localVertices,
        IReadOnlyList<ushort> indices,
        Vector3 placementPosition,
        Vector3 placementRotationDegrees,
        float scale,
        uint ck24Type = 0x43,
        ushort ck24ObjectId = 1,
        uint regionId = 0)
    {
        if (localVertices.Count == 0 || indices.Count < 3)
            return CreateEmptyPm4(regionId);

        Matrix4x4 worldTransform = BuildPlacementTransform(placementPosition, placementRotationDegrees, scale);
        List<Vector3> worldVertices = new(localVertices.Count);
        for (int i = 0; i < localVertices.Count; i++)
            worldVertices.Add(Vector3.Transform(localVertices[i], worldTransform));

        List<Vector3> pm4Vertices = new(worldVertices.Count);
        for (int i = 0; i < worldVertices.Count; i++)
            pm4Vertices.Add(WorldToPm4Raw(worldVertices[i]));

        var (simplifiedVerts, simplifiedIndices, polyCounts) = SimplifyByPlaneClustering(pm4Vertices, indices);
        var (uniqueVertices, remappedIndices32) = DeduplicateVertices(simplifiedVerts, simplifiedIndices);

        Vector3 centroid = ComputeCentroid(uniqueVertices);

        uint ck24 = ((uint)ck24Type << 16) | ck24ObjectId;
        uint packedParams = ((uint)ck24Type << 24) | ((uint)ck24ObjectId << 8);

        List<Pm4GenerationMsur> msurEntries = new();
        int indexCursor = 0;
        foreach (int polyVertexCount in polyCounts)
        {
            if (polyVertexCount < 3 || indexCursor + polyVertexCount > remappedIndices32.Count)
            {
                indexCursor += polyVertexCount;
                continue;
            }

            uint i0 = remappedIndices32[indexCursor];
            uint i1 = remappedIndices32[indexCursor + 1];
            uint i2 = remappedIndices32[indexCursor + 2];

            if (i0 >= (uint)uniqueVertices.Count ||
                i1 >= (uint)uniqueVertices.Count ||
                i2 >= (uint)uniqueVertices.Count)
            {
                indexCursor += polyVertexCount;
                continue;
            }

            Vector3 a = uniqueVertices[(int)i0];
            Vector3 b = uniqueVertices[(int)i1];
            Vector3 c = uniqueVertices[(int)i2];
            Vector3 normal = Vector3.Normalize(Vector3.Cross(b - a, c - a));

            float height = ComputePlaneD(a, normal);

            msurEntries.Add(new Pm4GenerationMsur(
                GroupKey: 0,
                IndexCount: (byte)polyVertexCount,
                AttributeMask: 0,
                Padding: 0,
                Normal: normal,
                Height: height,
                MsviFirstIndex: (uint)(indexCursor * 4),
                MscnRefIndex: 0,
                PackedParams: packedParams));

            indexCursor += polyVertexCount;
        }

        if (msurEntries.Count == 0)
            return CreateEmptyPm4(regionId);

        Vector3 mscnPoint = centroid;
        Vector3 mprlPos = WorldToMprlPosition(centroid);

        return new Pm4GenerationData(
            Version: 0x3010,
            Mshd: new Pm4GenerationMshd(
                Field00: 0x216,
                Field04: regionId,
                Field08: 0,
                Field0C: 0,
                Field10: 0,
                Field14: 0,
                Field18: 0,
                Field1C: 0),
            Mspv: Array.Empty<Vector3>(),
            Mspi: Array.Empty<uint>(),
            Msvt: uniqueVertices,
            Msvi: remappedIndices32,
            Msur: msurEntries,
            Mscn: new[] { mscnPoint },
            Mprl: new[]
            {
                new Pm4GenerationMprl(
                    Unk00: 0,
                    Unk02: -1,
                    Unk04: 0,
                    Unk06: 0x8000,
                    Position: mprlPos,
                    Unk14: -1,
                    Unk16: 0)
            },
            Mprr: new[] { new Pm4GenerationMprr(Value1: 0xFFFF, Value2: 0) },
            Mslk: new[]
            {
                new Pm4GenerationMslk(
                    TypeFlags: 0x12,
                    Subtype: 0,
                    Padding: 0,
                    GroupObjectId: ck24ObjectId,
                    MspiFirstIndex: -1,
                    MspiIndexCount: 0,
                    LinkId: 0,
                    RefIndex: 0,
                    SystemFlag: 0x8000)
            },
            Mdbh: null,
            Mdbi: Array.Empty<uint>(),
            Mdbf: Array.Empty<Pm4GenerationMdbf>(),
            Mdos: Array.Empty<Pm4GenerationMdos>(),
            Mdsf: Array.Empty<Pm4GenerationMdsf>());
    }

    private static Vector3 ComputeFaceNormal(Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 e1 = b - a;
        Vector3 e2 = c - a;
        Vector3 cross = Vector3.Cross(e1, e2);
        float len = cross.Length();
        return len > 0.0001f ? cross / len : Vector3.UnitZ;
    }

    private sealed class FacePlane
    {
        public Vector3 Normal;
        public float D;
        public List<int> Indices = [];
    }

    private static List<FacePlane> ClusterFacesByPlane(
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<ushort> indices,
        float normalDotThreshold = 0.99f,
        float planeDistThreshold = 0.3f)
    {
        List<FacePlane> clusters = [];

        for (int i = 0; i + 2 < indices.Count; i += 3)
        {
            ushort i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
            if (i0 >= vertices.Count || i1 >= vertices.Count || i2 >= vertices.Count)
                continue;

            Vector3 a = vertices[i0], b = vertices[i1], c = vertices[i2];
            Vector3 normal = ComputeFaceNormal(a, b, c);
            float d = ComputePlaneD(a, normal);

            bool added = false;
            foreach (FacePlane cluster in clusters)
            {
                float dot = Math.Abs(Vector3.Dot(normal, cluster.Normal));
                if (dot >= normalDotThreshold)
                {
                    Vector3 center = (a + b + c) / 3f;
                    float distToClusterPlane = Math.Abs(Vector3.Dot(center, cluster.Normal) + cluster.D);
                    if (distToClusterPlane <= planeDistThreshold)
                    {
                        cluster.Indices.Add(i0);
                        cluster.Indices.Add(i1);
                        cluster.Indices.Add(i2);
                        added = true;
                        break;
                    }
                }
            }

            if (!added)
            {
                FacePlane cluster = new()
                {
                    Normal = normal,
                    D = d,
                    Indices = [i0, i1, i2]
                };
                clusters.Add(cluster);
            }
        }

        return clusters;
    }

    private static List<FacePlane> SplitByConnectivity(FacePlane cluster, IReadOnlyList<ushort> allIndices)
    {
        List<ushort> faceList = cluster.Indices.Select(static i => (ushort)i).ToList();

        Dictionary<ushort, List<int>> vertToTri = new();
        for (int i = 0; i + 2 < faceList.Count; i += 3)
        {
            ushort v0 = faceList[i], v1 = faceList[i + 1], v2 = faceList[i + 2];
            if (!vertToTri.ContainsKey(v0)) vertToTri[v0] = [];
            if (!vertToTri.ContainsKey(v1)) vertToTri[v1] = [];
            if (!vertToTri.ContainsKey(v2)) vertToTri[v2] = [];
            int triIdx = i / 3;
            vertToTri[v0].Add(triIdx);
            vertToTri[v1].Add(triIdx);
            vertToTri[v2].Add(triIdx);
        }

        int triCount = faceList.Count / 3;
        bool[] visited = new bool[triCount];
        List<FacePlane> components = [];

        for (int t = 0; t < triCount; t++)
        {
            if (visited[t]) continue;

            List<int> componentTris = [];
            Queue<int> queue = new();
            queue.Enqueue(t);
            visited[t] = true;

            while (queue.Count > 0)
            {
                int cur = queue.Dequeue();
                componentTris.Add(cur);
                int baseIdx = cur * 3;
                ushort cv0 = faceList[baseIdx], cv1 = faceList[baseIdx + 1], cv2 = faceList[baseIdx + 2];

                void TryAddNeighbor(ushort v, int skipTri)
                {
                    if (!vertToTri.TryGetValue(v, out List<int>? neighbors)) return;
                    foreach (int nt in neighbors)
                    {
                        if (nt != skipTri && !visited[nt])
                        {
                            visited[nt] = true;
                            queue.Enqueue(nt);
                        }
                    }
                }

                TryAddNeighbor(cv0, cur);
                TryAddNeighbor(cv1, cur);
                TryAddNeighbor(cv2, cur);
            }

            FacePlane comp = new()
            {
                Normal = cluster.Normal,
                D = cluster.D,
                Indices = []
            };
            foreach (int tri in componentTris)
            {
                int baseIdx = tri * 3;
                comp.Indices.Add(faceList[baseIdx]);
                comp.Indices.Add(faceList[baseIdx + 1]);
                comp.Indices.Add(faceList[baseIdx + 2]);
            }
            components.Add(comp);
        }

        return components;
    }

    private static List<Vector3> ComputeConvexHull2D(List<Vector2> points)
    {
        if (points.Count <= 3)
        {
            var unique = new HashSet<Vector2>(points).ToList();
            return unique.Select(p => new Vector3(p.X, p.Y, 0)).ToList();
        }

        points = [.. new HashSet<Vector2>(points)];
        points.Sort((a, b) => a.X != b.X ? a.X.CompareTo(b.X) : a.Y.CompareTo(b.Y));

        if (points.Count <= 1)
            return points.Select(p => new Vector3(p.X, p.Y, 0)).ToList();

        List<Vector2> lower = new(points.Count);
        foreach (Vector2 p in points)
        {
            while (lower.Count >= 2 && Cross2D(lower[^2], lower[^1], p) <= 0)
                lower.RemoveAt(lower.Count - 1);
            lower.Add(p);
        }

        List<Vector2> upper = new(points.Count);
        for (int i = points.Count - 1; i >= 0; i--)
        {
            Vector2 p = points[i];
            while (upper.Count >= 2 && Cross2D(upper[^2], upper[^1], p) <= 0)
                upper.RemoveAt(upper.Count - 1);
            upper.Add(p);
        }

        lower.RemoveAt(lower.Count - 1);
        upper.RemoveAt(upper.Count - 1);
        lower.AddRange(upper);

        return lower.Select(p => new Vector3(p.X, p.Y, 0)).ToList();
    }

    private static float Cross2D(Vector2 o, Vector2 a, Vector2 b)
    {
        return (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);
    }

    private static PlanarSimplifyResult SimplifyByPlaneClustering(
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<ushort> indices)
    {
        List<int> globalToHull = new(vertices.Count);
        for (int i = 0; i < vertices.Count; i++)
            globalToHull.Add(-1);

        List<Vector3> outputVerts = [];
        List<ushort> outputIndices = [];
        List<int> polyVertexCounts = [];

        List<FacePlane> clusters = ClusterFacesByPlane(vertices, indices);

        foreach (FacePlane cluster in clusters)
        {
            HashSet<int> vertexSet = [];
            for (int i = 0; i < cluster.Indices.Count; i++)
                vertexSet.Add(cluster.Indices[i]);

            if (vertexSet.Count < 3)
                continue;

            Vector3 normal = cluster.Normal;
            int dominantAxis = Math.Abs(normal.X) >= Math.Abs(normal.Y)
                ? (Math.Abs(normal.X) >= Math.Abs(normal.Z) ? 0 : 2)
                : (Math.Abs(normal.Y) >= Math.Abs(normal.Z) ? 1 : 2);

            int ax0 = dominantAxis == 0 ? 1 : 0;
            int ax1 = dominantAxis == 2 ? 1 : 2;

            List<Vector2> projected = new(vertexSet.Count);
            List<int> projectedVertIndices = new(vertexSet.Count);
            foreach (int vi in vertexSet)
            {
                Vector3 v = vertices[vi];
                projected.Add(new Vector2(
                    ax0 == 0 ? v.X : ax0 == 1 ? v.Y : v.Z,
                    ax1 == 0 ? v.X : ax1 == 1 ? v.Y : v.Z));
                projectedVertIndices.Add(vi);
            }

            List<Vector2> hull2d = ComputeConvexHull2D(projected).Select(p => new Vector2(p.X, p.Y)).ToList();
            if (hull2d.Count < 3)
                continue;

            // Remove collinear hull vertices
            hull2d = SimplifyCollinear2D(hull2d, 0.01f);

            if (hull2d.Count < 3)
                continue;

            foreach (Vector2 hp in hull2d)
            {
                float bestDist = float.MaxValue;
                int bestVi = -1;
                Vector3 bestVert = Vector3.Zero;
                for (int j = 0; j < projected.Count; j++)
                {
                    float dx = projected[j].X - hp.X;
                    float dy = projected[j].Y - hp.Y;
                    float d = dx * dx + dy * dy;
                    if (d < bestDist)
                    {
                        bestDist = d;
                        bestVi = projectedVertIndices[j];
                        bestVert = vertices[bestVi];
                    }
                }

                if (globalToHull[bestVi] >= 0)
                {
                    outputIndices.Add((ushort)globalToHull[bestVi]);
                }
                else
                {
                    globalToHull[bestVi] = outputVerts.Count;
                    outputIndices.Add((ushort)outputVerts.Count);
                    outputVerts.Add(bestVert);
                }
            }

            polyVertexCounts.Add(hull2d.Count);
        }

        return new PlanarSimplifyResult(outputVerts, outputIndices, polyVertexCounts);
    }

    private static List<Vector2> SimplifyCollinear2D(List<Vector2> polygon, float angleTolerance)
    {
        if (polygon.Count <= 3) return polygon;

        List<Vector2> result = new(polygon.Count);
        int n = polygon.Count;

        for (int i = 0; i < n; i++)
        {
            Vector2 prev = polygon[(i - 1 + n) % n];
            Vector2 curr = polygon[i];
            Vector2 next = polygon[(i + 1) % n];

            Vector2 e1 = curr - prev;
            Vector2 e2 = next - curr;
            float len1 = e1.Length();
            float len2 = e2.Length();

            if (len1 < 0.0001f || len2 < 0.0001f)
                continue;

            float cross = Math.Abs(e1.X * e2.Y - e1.Y * e2.X);
            float dot = e1.X * e2.X + e1.Y * e2.Y;
            float area = cross / (len1 * len2);

            if (area > angleTolerance)
                result.Add(curr);
        }

        return result.Count >= 3 ? result : polygon;
    }

    private sealed record PlanarSimplifyResult(
        List<Vector3> Vertices,
        List<ushort> Indices,
        List<int> PolyVertexCounts);

    private static Pm4GenerationData CreateEmptyPm4(uint regionId)
    {
        return new Pm4GenerationData(
            Version: 0x3010,
            Mshd: new Pm4GenerationMshd(
                Field00: 0x216,
                Field04: regionId,
                Field08: 0, Field0C: 0, Field10: 0, Field14: 0, Field18: 0, Field1C: 0),
            Mspv: Array.Empty<Vector3>(),
            Mspi: Array.Empty<uint>(),
            Msvt: Array.Empty<Vector3>(),
            Msvi: Array.Empty<uint>(),
            Msur: Array.Empty<Pm4GenerationMsur>(),
            Mscn: Array.Empty<Vector3>(),
            Mprl: Array.Empty<Pm4GenerationMprl>(),
            Mprr: Array.Empty<Pm4GenerationMprr>(),
            Mslk: Array.Empty<Pm4GenerationMslk>(),
            Mdbh: null,
            Mdbi: Array.Empty<uint>(),
            Mdbf: Array.Empty<Pm4GenerationMdbf>(),
            Mdos: Array.Empty<Pm4GenerationMdos>(),
            Mdsf: Array.Empty<Pm4GenerationMdsf>());
    }

    private static Matrix4x4 BuildPlacementTransform(Vector3 position, Vector3 rotationDegrees, float scale)
    {
        float rx = rotationDegrees.X * MathF.PI / 180f;
        float ry = rotationDegrees.Y * MathF.PI / 180f;
        float rz = rotationDegrees.Z * MathF.PI / 180f;

        return Matrix4x4.CreateRotationZ(MathF.PI)
            * Matrix4x4.CreateScale(scale)
            * Matrix4x4.CreateRotationX(rx)
            * Matrix4x4.CreateRotationY(ry)
            * Matrix4x4.CreateRotationZ(rz)
            * Matrix4x4.CreateTranslation(position);
    }

    private static Vector3 WorldToPm4Raw(Vector3 worldPos)
    {
        return new Vector3(
            MapOrigin - worldPos.Y,
            MapOrigin - worldPos.X,
            worldPos.Z);
    }

    private static Vector3 WorldToMprlPosition(Vector3 worldPos)
    {
        return new Vector3(worldPos.X, worldPos.Z, worldPos.Y);
    }

    private static (List<Vector3> UniqueVertices, List<uint> RemappedIndices) DeduplicateVertices(
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<ushort> originalIndices)
    {
        Dictionary<Vector3, uint> dedupMap = new(vertices.Count);
        List<Vector3> unique = new(vertices.Count);
        List<uint> remapped = new(originalIndices.Count);

        for (int i = 0; i < originalIndices.Count; i++)
        {
            int srcIndex = originalIndices[i];
            if (srcIndex < 0 || srcIndex >= vertices.Count)
                continue;

            Vector3 v = vertices[srcIndex];
            if (!dedupMap.TryGetValue(v, out uint newIndex))
            {
                newIndex = (uint)unique.Count;
                dedupMap[v] = newIndex;
                unique.Add(v);
            }

            remapped.Add(newIndex);
        }

        return (unique, remapped);
    }

    private static Vector3 ComputeCentroid(IReadOnlyList<Vector3> vertices)
    {
        Vector3 sum = Vector3.Zero;
        for (int i = 0; i < vertices.Count; i++)
            sum += vertices[i];
        return vertices.Count > 0 ? sum / vertices.Count : Vector3.Zero;
    }

    private static float ComputePlaneD(Vector3 pointOnPlane, Vector3 normal)
    {
        return -Vector3.Dot(normal, pointOnPlane);
    }
}
