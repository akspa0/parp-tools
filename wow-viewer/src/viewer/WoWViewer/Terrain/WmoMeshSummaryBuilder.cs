using System.Numerics;
using WowViewer.Core.IO.Converters;

namespace WoWViewer.Terrain;

/// <summary>
/// Static helper for building WMO geometry summaries, sample footprints, bounds, and version detection.
/// </summary>
internal static class WmoMeshSummaryBuilder
{
    public static WmoMeshSummary BuildWmoMeshSummary(WmoV14ToV17Converter.WmoV14Data wmo)
    {
        int vertexCount = 0;
        int indexCount = 0;
        int batchCount = 0;
        Vector3[] footprintSampleVertices = BuildWmoFootprintSamples(wmo.Groups);
        WmoGroupMeshSummary[] groupSummaries = BuildWmoGroupMeshSummaries(wmo.Groups);
        ComputeWmoGeometryBounds(wmo, out Vector3 boundsMin, out Vector3 boundsMax);

        foreach (WmoV14ToV17Converter.WmoGroupData group in wmo.Groups)
        {
            vertexCount += group.Vertices.Count;
            indexCount += group.Indices.Count;
            batchCount += group.Batches.Count;
        }

        return new WmoMeshSummary(
            Version: (int)wmo.Version,
            GroupCount: wmo.Groups.Count,
            VertexCount: vertexCount,
            IndexCount: indexCount,
            TriangleCount: indexCount / 3,
            BatchCount: batchCount,
            BoundsMin: boundsMin,
            BoundsMax: boundsMax,
            FootprintSampleVertices: footprintSampleVertices,
            GroupSummaries: groupSummaries);
    }

    public static WmoGroupMeshSummary[] BuildWmoGroupMeshSummaries(IReadOnlyList<WmoV14ToV17Converter.WmoGroupData> groups)
    {
        if (groups.Count == 0)
            return Array.Empty<WmoGroupMeshSummary>();

        var summaries = new WmoGroupMeshSummary[groups.Count];
        for (int groupIndex = 0; groupIndex < groups.Count; groupIndex++)
        {
            WmoV14ToV17Converter.WmoGroupData group = groups[groupIndex];
            ComputeVectorBounds(group.Vertices, out Vector3 boundsMin, out Vector3 boundsMax);
            summaries[groupIndex] = new WmoGroupMeshSummary(
                GroupIndex: groupIndex,
                VertexCount: group.Vertices.Count,
                IndexCount: group.Indices.Count,
                TriangleCount: group.Indices.Count / 3,
                BoundsMin: boundsMin,
                BoundsMax: boundsMax,
                FootprintSampleVertices: BuildSampleVertices(group.Vertices, 128));
        }

        return summaries;
    }

    public static void ComputeWmoGeometryBounds(WmoV14ToV17Converter.WmoV14Data wmo, out Vector3 boundsMin, out Vector3 boundsMax)
    {
        bool hasBounds = false;
        Vector3 min = new(float.MaxValue);
        Vector3 max = new(float.MinValue);

        foreach (WmoV14ToV17Converter.WmoGroupData group in wmo.Groups)
        {
            List<Vector3> vertices = group.Vertices;
            for (int vertexIndex = 0; vertexIndex < vertices.Count; vertexIndex++)
            {
                Vector3 vertex = vertices[vertexIndex];
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
                hasBounds = true;
            }
        }

        if (hasBounds)
        {
            boundsMin = min;
            boundsMax = max;
            return;
        }

        boundsMin = wmo.BoundsMin;
        boundsMax = wmo.BoundsMax;
    }

    public static Vector3[] BuildWmoFootprintSamples(IReadOnlyList<WmoV14ToV17Converter.WmoGroupData> groups)
    {
        int totalVertexCount = 0;
        for (int groupIndex = 0; groupIndex < groups.Count; groupIndex++)
            totalVertexCount += groups[groupIndex].Vertices.Count;

        if (totalVertexCount <= 0)
            return Array.Empty<Vector3>();

        const int maxSamples = 256;
        int stride = Math.Max(1, totalVertexCount / maxSamples);
        var samples = new List<Vector3>(Math.Min(totalVertexCount, maxSamples));
        int globalVertexIndex = 0;

        for (int groupIndex = 0; groupIndex < groups.Count && samples.Count < maxSamples; groupIndex++)
        {
            List<Vector3> vertices = groups[groupIndex].Vertices;
            for (int vertexIndex = 0; vertexIndex < vertices.Count && samples.Count < maxSamples; vertexIndex++, globalVertexIndex++)
            {
                if (globalVertexIndex % stride == 0)
                    samples.Add(vertices[vertexIndex]);
            }
        }

        if (samples.Count == 0)
        {
            for (int groupIndex = 0; groupIndex < groups.Count; groupIndex++)
            {
                List<Vector3> vertices = groups[groupIndex].Vertices;
                if (vertices.Count > 0)
                {
                    samples.Add(vertices[0]);
                    break;
                }
            }
        }

        return samples.ToArray();
    }

    public static Vector3[] BuildSampleVertices(IReadOnlyList<Vector3> vertices, int maxSamples)
    {
        if (vertices.Count == 0 || maxSamples <= 0)
            return Array.Empty<Vector3>();

        int stride = Math.Max(1, vertices.Count / maxSamples);
        var samples = new List<Vector3>(Math.Min(vertices.Count, maxSamples));
        for (int index = 0; index < vertices.Count && samples.Count < maxSamples; index += stride)
            samples.Add(vertices[index]);

        return samples.ToArray();
    }

    public static void ComputeVectorBounds(IReadOnlyList<Vector3> vertices, out Vector3 boundsMin, out Vector3 boundsMax)
    {
        if (vertices.Count == 0)
        {
            boundsMin = Vector3.Zero;
            boundsMax = Vector3.Zero;
            return;
        }

        Vector3 min = new(float.MaxValue);
        Vector3 max = new(float.MinValue);
        for (int index = 0; index < vertices.Count; index++)
        {
            Vector3 vertex = vertices[index];
            min = Vector3.Min(min, vertex);
            max = Vector3.Max(max, vertex);
        }

        boundsMin = min;
        boundsMax = max;
    }

    /// <summary>
    /// Detect WMO version from raw bytes. Returns 14 for Alpha (MOMO container), version number for v17+, or 0.
    /// </summary>
    public static int DetectWmoVersion(byte[] data)
    {
        if (data.Length < 12) return 0;
        string magic = System.Text.Encoding.ASCII.GetString(data, 0, 4);
        string reversed = new string(magic.Reverse().ToArray());

        // v14 Alpha: starts with MOMO container
        if (magic == "MOMO" || reversed == "MOMO") return 14;

        // v17+: starts with MVER chunk
        if (magic == "MVER" || reversed == "MVER")
        {
            uint size = BitConverter.ToUInt32(data, 4);
            if (size >= 4 && data.Length >= 12)
                return (int)BitConverter.ToUInt32(data, 8);
        }
        return 0;
    }
}
