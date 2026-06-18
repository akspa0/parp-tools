using System.Numerics;
using System.Text.Json;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Maps;
using WowViewer.Core.Mdx;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Wmo;

internal sealed record Pm4Ck24GeometryExport(
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int SurfaceCount,
    int VertexCount,
    Vector3 Pm4BoundsMin,
    Vector3 Pm4BoundsMax,
    IReadOnlyList<Vector3> Pm4Vertices,
    IReadOnlyList<uint> Pm4CornerIndices,
    IReadOnlyList<byte> Pm4IndexCounts);

internal sealed record WmoMeshInPm4Space(
    int UniqueId,
    string ModelPath,
    Vector3 Position,
    Vector3 Rotation,
    int GroupCount,
    IReadOnlyList<Vector3> WmoLocalVerts,
    IReadOnlyList<Vector3> WmoWorldVerts,
    IReadOnlyList<Vector3> WmoPm4Verts,
    IReadOnlyList<ushort> Indices,
    IReadOnlyList<WmoGroupFaceMaterialDetail> FaceMaterials);

internal sealed record Pm4CorrelateModelsResult(
    string Pm4Path,
    string PlacementsPath,
    int TileX,
    int TileY,
    IReadOnlyList<Pm4Ck24GroupSummary> Ck24Groups,
    IReadOnlyList<Pm4PlacementCollisionSummary> PlacementSummaries,
    IReadOnlyList<Pm4CorrelationEntry> Correlations,
    IReadOnlyList<string> Warnings);

internal sealed record Pm4Ck24GroupSummary(
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int SurfaceCount,
    int TotalIndexCount,
    int VertexCount,
    Vector3 Pm4BoundsMin,
    Vector3 Pm4BoundsMax,
    Vector3 WowBoundsMin,
    Vector3 WowBoundsMax);

internal sealed record Pm4PlacementCollisionSummary(
    int UniqueId,
    string ModelPath,
    string AssetKind,
    Vector3 Position,
    Vector3 Rotation,
    float Scale,
    int GroupCount,
    int TotalCollisionVertices,
    int TotalCollisionFaces,
    Vector3 LocalBoundsMin,
    Vector3 LocalBoundsMax,
    Vector3 WorldBoundsMin,
    Vector3 WorldBoundsMax);

internal sealed record Pm4CorrelationEntry(
    int UniqueId,
    string ModelPath,
    string AssetKind,
    uint Ck24,
    byte Ck24Type,
    double WowBoundsOverlap,
    double Pm4BoundsOverlap,
    double WowCenterDistance,
    double Pm4CenterDistance);

internal sealed record WmoLocalBoundsEntry(
    string WmoPath,
    int GroupCount,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    float SortedDim0,
    float SortedDim1,
    float SortedDim2);

internal sealed class Pm4FingerprintGroup
{
    public string Fingerprint { get; }
    public int Surfaces { get; }
    public int Indices { get; }
    public int Vertices { get; }
    public byte Ck24Type { get; }
    public List<uint> Ck24Values { get; } = [];
    public List<int> ObjectIds { get; } = [];
    public float MergedSortedDim0 { get; set; }
    public float MergedSortedDim1 { get; set; }
    public float MergedSortedDim2 { get; set; }

    public Pm4FingerprintGroup(string fingerprint, int surfaces, int indices, int vertices, byte ck24Type)
    {
        Fingerprint = fingerprint;
        Surfaces = surfaces;
        Indices = indices;
        Vertices = vertices;
        Ck24Type = ck24Type;
    }
}

internal sealed record Pm4IdentityMatch(
    int Surfaces,
    int Indices,
    int Vertices,
    byte Ck24Type,
    string Fingerprint,
    float Pm4SortedDim0,
    float Pm4SortedDim1,
    float Pm4SortedDim2,
    string WmoPath,
    float WmoSortedDim0,
    float WmoSortedDim1,
    float WmoSortedDim2,
    double DimensionRatio,
    double Score);

internal static class Pm4CorrelateModelsSupport
{
    private const float MapOrigin = 17066.666f;
    private const float TileSize = 533.333f;

    public static Pm4CorrelateModelsResult Correlate(
        string pm4Path,
        string placementsPath,
        string archiveRoot,
        ArchiveCatalogBootstrapOptions bootstrapOptions,
        string? pm4ArchiveVirtualPath = null,
        string? adtArchiveVirtualPath = null)
    {
        string coordPath = pm4ArchiveVirtualPath ?? pm4Path;
        if (!Pm4CoordinateService.TryParseTileCoordinates(coordPath, out int tileX, out int tileY))
            throw new InvalidOperationException("Could not parse tile coordinates from PM4 path.");

        Pm4ResearchDocument document;
        if (File.Exists(pm4Path))
        {
            document = Pm4ResearchReader.ReadFile(pm4Path);
        }
        else if (pm4ArchiveVirtualPath is not null)
        {
            byte[] pm4Bytes = ArchiveVirtualFileReader.ReadVirtualFile(
                pm4ArchiveVirtualPath, [archiveRoot], bootstrapOptions);
            document = Pm4ResearchReader.Read(pm4Bytes, pm4ArchiveVirtualPath);
        }
        else
        {
            throw new FileNotFoundException($"PM4 file not found: {pm4Path}");
        }

        IReadOnlyList<Vector3> meshVertices = document.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = document.KnownChunks.Msvi;
        List<string> warnings = [];

        bool isTileLocal = Pm4PlacementMath.IsLikelyTileLocal(meshVertices);
        List<Pm4Ck24GroupSummary> ck24Groups = BuildCk24GroupSummaries(document, meshVertices, meshIndices, tileX, tileY, isTileLocal);

        AdtPlacementCatalog placements;
        if (File.Exists(placementsPath))
        {
            placements = AdtPlacementReader.Read(placementsPath);
        }
        else if (adtArchiveVirtualPath is not null)
        {
            byte[] adtBytes = ArchiveVirtualFileReader.ReadVirtualFile(
                adtArchiveVirtualPath, [archiveRoot], bootstrapOptions);
            using MemoryStream adtStream = new(adtBytes);
            MapFileSummary adtSummary = MapFileSummaryReader.Read(adtStream, adtArchiveVirtualPath);
            adtStream.Position = 0;
            placements = AdtPlacementReader.Read(adtStream, adtSummary);
        }
        else
        {
            throw new FileNotFoundException($"ADT file not found: {placementsPath}");
        }
        List<Pm4PlacementCollisionSummary> placementSummaries = [];

        foreach (AdtWorldModelPlacement wmoPlacement in placements.WorldModelPlacements)
        {
            Pm4PlacementCollisionSummary summary = BuildWmoCollisionSummary(wmoPlacement, archiveRoot, bootstrapOptions, warnings);
            placementSummaries.Add(summary);
        }

        foreach (AdtModelPlacement m2Placement in placements.ModelPlacements)
        {
            Pm4PlacementCollisionSummary summary = BuildM2CollisionSummary(m2Placement, archiveRoot, bootstrapOptions, warnings);
            placementSummaries.Add(summary);
        }

        List<Pm4CorrelationEntry> correlations = BuildCorrelations(placementSummaries, ck24Groups);

        return new Pm4CorrelateModelsResult(
            pm4Path, placementsPath, tileX, tileY,
            ck24Groups, placementSummaries, correlations, warnings);
    }

    private static List<Pm4Ck24GroupSummary> BuildCk24GroupSummaries(
        Pm4ResearchDocument document,
        IReadOnlyList<Vector3> meshVertices,
        IReadOnlyList<uint> meshIndices,
        int tileX, int tileY,
        bool isTileLocal)
    {
        List<Pm4Ck24GroupSummary> groups = [];

        foreach (IGrouping<uint, Pm4MsurEntry> ck24Group in document.KnownChunks.Msur
            .Where(static s => s.Ck24 != 0u && s.IndexCount >= 3)
            .GroupBy(static s => s.Ck24)
            .OrderByDescending(static g => g.Sum(static s => s.IndexCount)))
        {
            uint ck24 = ck24Group.Key;
            List<Pm4MsurEntry> surfaces = ck24Group.ToList();
            byte ck24Type = surfaces[0].Ck24Type;
            ushort ck24ObjectId = surfaces[0].Ck24ObjectId;
            int surfaceCount = surfaces.Count;
            int totalIndexCount = surfaces.Sum(static s => s.IndexCount);

            HashSet<int> vertexIndices = [];
            foreach (Pm4MsurEntry surface in surfaces)
            {
                int first = checked((int)surface.MsviFirstIndex);
                int end = Math.Min(first + surface.IndexCount, meshIndices.Count);
                for (int i = first; i < end; i++)
                {
                    int vi = checked((int)meshIndices[i]);
                    if ((uint)vi < (uint)meshVertices.Count)
                        vertexIndices.Add(vi);
                }
            }

            if (vertexIndices.Count == 0)
                continue;

            Vector3 pm4Min = new(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 pm4Max = new(float.MinValue, float.MinValue, float.MinValue);
            foreach (int vi in vertexIndices)
            {
                Vector3 v = meshVertices[vi];
                pm4Min = Vector3.Min(pm4Min, v);
                pm4Max = Vector3.Max(pm4Max, v);
            }

            Vector3 wowMin, wowMax;
            if (isTileLocal)
            {
                Vector3 swappedMin = new(pm4Min.Y, pm4Min.X, pm4Min.Z);
                Vector3 swappedMax = new(pm4Max.Y, pm4Max.X, pm4Max.Z);
                Vector3 rawMin = ConvertRawAdtToWowWorld(swappedMin, tileX, tileY);
                Vector3 rawMax = ConvertRawAdtToWowWorld(swappedMax, tileX, tileY);
                wowMin = Vector3.Min(rawMin, rawMax);
                wowMax = Vector3.Max(rawMin, rawMax);
            }
            else
            {
                float minWorldX = MapOrigin - pm4Max.X;
                float minWorldY = MapOrigin - pm4Max.Y;
                float maxWorldX = MapOrigin - pm4Min.X;
                float maxWorldY = MapOrigin - pm4Min.Y;
                wowMin = new Vector3(minWorldX, minWorldY, pm4Min.Z);
                wowMax = new Vector3(maxWorldX, maxWorldY, pm4Max.Z);
            }

            groups.Add(new Pm4Ck24GroupSummary(
                ck24, ck24Type, ck24ObjectId,
                surfaceCount, totalIndexCount, vertexIndices.Count,
                pm4Min, pm4Max, wowMin, wowMax));
        }

        return groups;
    }

    private static Pm4PlacementCollisionSummary BuildWmoCollisionSummary(
        AdtWorldModelPlacement placement,
        string archiveRoot,
        ArchiveCatalogBootstrapOptions bootstrapOptions,
        List<string> warnings)
    {
        int groupCount = 0;
        int totalVertices = 0;
        int totalFaces = 0;
        Vector3 localMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 localMax = new(float.MinValue, float.MinValue, float.MinValue);

        try
        {
            byte[] rootBytes = ArchiveVirtualFileReader.ReadVirtualFile(
                NormalizeVirtualPath(placement.ModelPath),
                [archiveRoot], bootstrapOptions);

            using MemoryStream rootStream = new(rootBytes, writable: false);
            WmoSummary summary = WmoSummaryReader.Read(rootStream, placement.ModelPath);
            groupCount = summary.ReportedGroupCount;

            Func<string, byte[]?> assetReader = virtualPath =>
            {
                try
                {
                    return ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], bootstrapOptions);
                }
                catch { return null; }
            };

            WmoRenderDocument renderDoc = WmoRenderDocumentReader.Read(
                new MemoryStream(rootBytes, writable: false), placement.ModelPath, assetReader);

            foreach (WmoEmbeddedGroupMeshDetail group in renderDoc.Groups)
            {
                totalVertices += group.Mesh.Vertices.Count;
                totalFaces += group.Mesh.FaceMaterials.Count;

                foreach (Vector3 v in group.Mesh.Vertices)
                {
                    localMin = Vector3.Min(localMin, v);
                    localMax = Vector3.Max(localMax, v);
                }
            }
        }
        catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException)
        {
            warnings.Add($"WMO read failed for '{placement.ModelPath}': {ex.Message}");
            localMin = placement.BoundsMin;
            localMax = placement.BoundsMax;
        }

        if (totalVertices == 0)
        {
            localMin = placement.BoundsMin;
            localMax = placement.BoundsMax;
        }

        Matrix4x4 transform = BuildWmoTransform(placement.Position, placement.Rotation);
        TransformBounds(localMin, localMax, transform, out Vector3 worldMin, out Vector3 worldMax);

        return new Pm4PlacementCollisionSummary(
            placement.UniqueId, placement.ModelPath, "wmo",
            placement.Position, placement.Rotation, 1f,
            groupCount, totalVertices, totalFaces,
            localMin, localMax, worldMin, worldMax);
    }

    private static Pm4PlacementCollisionSummary BuildM2CollisionSummary(
        AdtModelPlacement placement,
        string archiveRoot,
        ArchiveCatalogBootstrapOptions bootstrapOptions,
        List<string> warnings)
    {
        int groupCount = 0;
        int totalVertices = 0;
        int totalFaces = 0;
        Vector3 localMin = placement.Position - new Vector3(2f);
        Vector3 localMax = placement.Position + new Vector3(2f);

        try
        {
            byte[] bytes = ArchiveVirtualFileReader.ReadVirtualFile(
                NormalizeVirtualPath(placement.ModelPath),
                [archiveRoot], bootstrapOptions);

            using MemoryStream stream = new(bytes, writable: false);
            var summary = MdxSummaryReader.Read(stream, placement.ModelPath);
            if (summary.Collision is not null && summary.Collision.VertexCount > 0)
            {
                var c = summary.Collision;
                totalVertices = c.VertexCount;
                totalFaces = c.TriangleCount;
                localMin = c.BoundsMin ?? summary.BoundsMin ?? (placement.Position - new Vector3(2f));
                localMax = c.BoundsMax ?? summary.BoundsMax ?? (placement.Position + new Vector3(2f));
            }
            else if (summary.BoundsMin is not null && summary.BoundsMax is not null)
            {
                localMin = summary.BoundsMin.Value;
                localMax = summary.BoundsMax.Value;
            }
            else
            {
                localMin = placement.Position - new Vector3(2f);
                localMax = placement.Position + new Vector3(2f);
            }
        }
        catch (Exception ex) when (ex is FileNotFoundException or InvalidDataException or IOException)
        {
            warnings.Add($"M2 read failed for '{placement.ModelPath}': {ex.Message}");
        }

        Matrix4x4 transform = BuildM2Transform(placement.Position, placement.Rotation, placement.Scale);
        TransformBounds(localMin, localMax, transform, out Vector3 worldMin, out Vector3 worldMax);

        return new Pm4PlacementCollisionSummary(
            placement.UniqueId, placement.ModelPath, "m2",
            placement.Position, placement.Rotation, placement.Scale,
            groupCount, totalVertices, totalFaces,
            localMin, localMax, worldMin, worldMax);
    }

    private static List<Pm4CorrelationEntry> BuildCorrelations(
        IReadOnlyList<Pm4PlacementCollisionSummary> placements,
        IReadOnlyList<Pm4Ck24GroupSummary> ck24Groups)
    {
        List<Pm4CorrelationEntry> correlations = [];

        foreach (Pm4PlacementCollisionSummary placement in placements)
        {
            foreach (Pm4Ck24GroupSummary group in ck24Groups)
            {
                double wowOverlap = ComputeBoundsOverlap(
                    placement.WorldBoundsMin, placement.WorldBoundsMax,
                    group.WowBoundsMin, group.WowBoundsMax);

                double pm4Overlap = ComputeBoundsOverlap(
                    placement.WorldBoundsMin, placement.WorldBoundsMax,
                    group.Pm4BoundsMin, group.Pm4BoundsMax);

                Vector3 wowGroupCenter = (group.WowBoundsMin + group.WowBoundsMax) * 0.5f;
                Vector3 pm4GroupCenter = (group.Pm4BoundsMin + group.Pm4BoundsMax) * 0.5f;
                Vector3 placementCenter = (placement.WorldBoundsMin + placement.WorldBoundsMax) * 0.5f;

                double wowCenterDist = Vector3.Distance(placementCenter, wowGroupCenter);
                double pm4CenterDist = Vector3.Distance(placementCenter, pm4GroupCenter);

                if (wowOverlap > 0.001 || pm4Overlap > 0.001)
                {
                    correlations.Add(new Pm4CorrelationEntry(
                        placement.UniqueId, placement.ModelPath, placement.AssetKind,
                        group.Ck24, group.Ck24Type,
                        wowOverlap, pm4Overlap,
                        wowCenterDist, pm4CenterDist));
                }
            }
        }

        return correlations.OrderByDescending(static c => c.WowBoundsOverlap).ThenBy(static c => c.WowCenterDistance).ToList();
    }

    private static double ComputeBoundsOverlap(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB)
    {
        double overlapX = Math.Max(0, Math.Min(maxA.X, maxB.X) - Math.Max(minA.X, minB.X));
        double overlapY = Math.Max(0, Math.Min(maxA.Y, maxB.Y) - Math.Max(minA.Y, minB.Y));
        double overlapZ = Math.Max(0, Math.Min(maxA.Z, maxB.Z) - Math.Max(minA.Z, minB.Z));

        double volumeA = Math.Max(0, (maxA.X - minA.X)) * Math.Max(0, (maxA.Y - minA.Y)) * Math.Max(0, (maxA.Z - minA.Z));
        double volumeB = Math.Max(0, (maxB.X - minB.X)) * Math.Max(0, (maxB.Y - minB.Y)) * Math.Max(0, (maxB.Z - minB.Z));
        double intersection = overlapX * overlapY * overlapZ;

        if (volumeA <= 0 || volumeB <= 0)
            return 0;

        double union = volumeA + volumeB - intersection;
        return union > 0 ? intersection / union : 0;
    }

    private static Vector3 ConvertRawAdtToWowWorld(Vector3 rawAdt, int tileX, int tileY)
    {
        float localUp = rawAdt.Z;

        float rawAdtX = tileX * TileSize + rawAdt.X;
        float rawAdtY = tileY * TileSize + rawAdt.Y;

        return new Vector3(MapOrigin - rawAdtY, MapOrigin - rawAdtX, localUp);
    }

    private static Matrix4x4 BuildWmoTransform(Vector3 position, Vector3 rotationDegrees)
    {
        float rx = rotationDegrees.X * MathF.PI / 180f;
        float ry = rotationDegrees.Y * MathF.PI / 180f;
        float rz = rotationDegrees.Z * MathF.PI / 180f;

        return Matrix4x4.CreateRotationZ(MathF.PI)
            * Matrix4x4.CreateRotationX(rx)
            * Matrix4x4.CreateRotationY(ry)
            * Matrix4x4.CreateRotationZ(rz)
            * Matrix4x4.CreateTranslation(position);
    }

    private static Matrix4x4 BuildM2Transform(Vector3 position, Vector3 rotationDegrees, float scale)
    {
        float rx = -rotationDegrees.Y * MathF.PI / 180f;
        float ry = -rotationDegrees.X * MathF.PI / 180f;
        float rz = rotationDegrees.Z * MathF.PI / 180f;

        return Matrix4x4.CreateRotationZ(MathF.PI)
            * Matrix4x4.CreateScale(scale)
            * Matrix4x4.CreateRotationX(rx)
            * Matrix4x4.CreateRotationY(ry)
            * Matrix4x4.CreateRotationZ(rz)
            * Matrix4x4.CreateTranslation(position);
    }

    private static void TransformBounds(Vector3 localMin, Vector3 localMax, Matrix4x4 transform, out Vector3 worldMin, out Vector3 worldMax)
    {
        Span<Vector3> corners = stackalloc Vector3[8]
        {
            new(localMin.X, localMin.Y, localMin.Z),
            new(localMin.X, localMin.Y, localMax.Z),
            new(localMin.X, localMax.Y, localMin.Z),
            new(localMin.X, localMax.Y, localMax.Z),
            new(localMax.X, localMin.Y, localMin.Z),
            new(localMax.X, localMin.Y, localMax.Z),
            new(localMax.X, localMax.Y, localMin.Z),
            new(localMax.X, localMax.Y, localMax.Z),
        };

        worldMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        worldMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);
        for (int i = 0; i < corners.Length; i++)
        {
            Vector3 world = Vector3.Transform(corners[i], transform);
            worldMin = Vector3.Min(worldMin, world);
            worldMax = Vector3.Max(worldMax, world);
        }
    }

    public static List<WmoLocalBoundsEntry> ScanWmoLocalBounds(string archiveRoot, ArchiveCatalogBootstrapOptions bootstrapOptions)
    {
        ArchiveCatalogSession session = ArchiveCatalogSessionCache.GetOrCreate([archiveRoot], bootstrapOptions);
        IReadOnlyList<string> allFiles = session.ArchiveCatalog.GetAllKnownFiles();
        List<string> wmoPaths = allFiles
            .Where(static f => f.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase) && !f.Contains('_'))
            .OrderBy(static f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();

        Console.WriteLine($"Found {wmoPaths.Count} WMO root files in archive.");

        List<WmoLocalBoundsEntry> results = [];
        int read = 0;
        int failed = 0;

        foreach (string wmoPath in wmoPaths)
        {
            try
            {
                byte[] rootBytes = ArchiveVirtualFileReader.ReadVirtualFile(
                    wmoPath.Replace('\\', '/').TrimStart('/').ToLowerInvariant(),
                    [archiveRoot], bootstrapOptions);

                using MemoryStream rootStream = new(rootBytes, writable: false);
                WmoSummary summary = WmoSummaryReader.Read(rootStream, wmoPath);

                Vector3 bMin = summary.BoundsMin;
                Vector3 bMax = summary.BoundsMax;
                float dx = bMax.X - bMin.X;
                float dy = bMax.Y - bMin.Y;
                float dz = bMax.Z - bMin.Z;

                if (dx <= 0 || dy <= 0 || dz <= 0 || float.IsNaN(dx) || float.IsNaN(dy) || float.IsNaN(dz))
                {
                    failed++;
                    continue;
                }

                float[] dims = [dx, dy, dz];
                Array.Sort(dims);

                results.Add(new WmoLocalBoundsEntry(
                    wmoPath, summary.ReportedGroupCount,
                    bMin, bMax,
                    dims[0], dims[1], dims[2]));

                read++;
                if (read % 500 == 0)
                    Console.WriteLine($"  Scanned {read}/{wmoPaths.Count} WMOs...");
            }
            catch
            {
                failed++;
            }
        }

        Console.WriteLine($"Scanned {read} WMOs successfully, {failed} failed.");
        return results;
    }

    public static List<Pm4FingerprintGroup> BuildFingerprintGroups(List<Dictionary<string, JsonElement>> fingerprints)
    {
        Dictionary<string, Pm4FingerprintGroup> groupsByKey = [];

        foreach (Dictionary<string, JsonElement> fp in fingerprints)
        {
            int surfaces = fp["surfaces"].GetInt32();
            int indices = fp["indices"].GetInt32();
            int vertices = fp["vertices"].GetInt32();
            byte ck24Type = Convert.ToByte(fp["type"].GetString()!.TrimStart('0', 'x'), 16);
            uint ck24 = Convert.ToUInt32(fp["ck24"].GetString()!.TrimStart('0', 'x'), 16);
            int objectId = fp["objectId"].GetInt32();

            string sortedSize = fp["sortedSize"].GetString() ?? "";
            float dim0 = 0, dim1 = 0, dim2 = 0;
            if (!string.IsNullOrEmpty(sortedSize) && sortedSize.Contains('x'))
            {
                string[] parts = sortedSize.Split('x');
                if (parts.Length == 3)
                {
                    float.TryParse(parts[0], out dim0);
                    float.TryParse(parts[1], out dim1);
                    float.TryParse(parts[2], out dim2);
                }
            }

            string key = $"{surfaces}_{indices}_{vertices}_{ck24Type:X2}";

            if (!groupsByKey.TryGetValue(key, out Pm4FingerprintGroup? group))
            {
                group = new Pm4FingerprintGroup(key, surfaces, indices, vertices, ck24Type);
                group.Ck24Values.Add(ck24);
                group.ObjectIds.Add(objectId);
                group.MergedSortedDim0 = dim0;
                group.MergedSortedDim1 = dim1;
                group.MergedSortedDim2 = dim2;
                groupsByKey[key] = group;
            }
            else
            {
                group.Ck24Values.Add(ck24);
                group.ObjectIds.Add(objectId);

                if (dim0 > 0 && (group.MergedSortedDim0 == 0 || dim0 > group.MergedSortedDim0))
                {
                    group.MergedSortedDim0 = dim0;
                    group.MergedSortedDim1 = dim1;
                    group.MergedSortedDim2 = dim2;
                }
            }
        }

        return groupsByKey.Values
            .OrderByDescending(static g => g.ObjectIds.Count)
            .ThenByDescending(static g => g.Surfaces)
            .ToList();
    }

    public static List<Pm4IdentityMatch> MatchFingerprintsToWmos(
        List<Pm4FingerprintGroup> fingerprintGroups,
        List<WmoLocalBoundsEntry> wmoBounds,
        double minScore = 0.5)
    {
        List<Pm4IdentityMatch> matches = [];

        foreach (Pm4FingerprintGroup fpGroup in fingerprintGroups)
        {
            if (fpGroup.MergedSortedDim0 <= 0)
                continue;

            double bestScore = 0;
            WmoLocalBoundsEntry? bestWmo = null;

            foreach (WmoLocalBoundsEntry wmo in wmoBounds)
            {
                double score = ComputeDimensionSimilarity(
                    fpGroup.MergedSortedDim0, fpGroup.MergedSortedDim1, fpGroup.MergedSortedDim2,
                    wmo.SortedDim0, wmo.SortedDim1, wmo.SortedDim2);

                if (score > bestScore)
                {
                    bestScore = score;
                    bestWmo = wmo;
                }
            }

            if (bestWmo is not null && bestScore >= minScore)
            {
                double ratio = ComputeDimensionRatio(
                    fpGroup.MergedSortedDim0, fpGroup.MergedSortedDim1, fpGroup.MergedSortedDim2,
                    bestWmo.SortedDim0, bestWmo.SortedDim1, bestWmo.SortedDim2);

                matches.Add(new Pm4IdentityMatch(
                    fpGroup.Surfaces, fpGroup.Indices, fpGroup.Vertices,
                    fpGroup.Ck24Type, fpGroup.Fingerprint,
                    fpGroup.MergedSortedDim0, fpGroup.MergedSortedDim1, fpGroup.MergedSortedDim2,
                    bestWmo.WmoPath,
                    bestWmo.SortedDim0, bestWmo.SortedDim1, bestWmo.SortedDim2,
                    ratio, bestScore));
            }
        }

        return matches.OrderByDescending(static m => m.Score).ToList();
    }

    private static double ComputeDimensionSimilarity(float d0a, float d1a, float d2a, float d0b, float d1b, float d2b)
    {
        if (d0a <= 0 || d1a <= 0 || d2a <= 0 || d0b <= 0 || d1b <= 0 || d2b <= 0)
            return 0;

        double r0 = Math.Min(d0a, d0b) / (double)Math.Max(d0a, d0b);
        double r1 = Math.Min(d1a, d1b) / (double)Math.Max(d1a, d1b);
        double r2 = Math.Min(d2a, d2b) / (double)Math.Max(d2a, d2b);

        return (r0 + r1 + r2) / 3.0;
    }

    private static double ComputeDimensionRatio(float d0a, float d1a, float d2a, float d0b, float d1b, float d2b)
    {
        if (d0a <= 0 || d0b <= 0)
            return 0;

        return (d0a / d0b + d1a / d1b + d2a / d2b) / 3.0;
    }

    private static string NormalizeVirtualPath(string modelPath)
    {
        return modelPath.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();
    }

    public static Pm4Ck24GeometryExport ExportCk24GroupGeometry(
        Pm4ResearchDocument document,
        uint targetCk24)
    {
        IReadOnlyList<Vector3> msvt = document.KnownChunks.Msvt;
        IReadOnlyList<uint> msvi = document.KnownChunks.Msvi;
        List<Pm4MsurEntry> groupSurfaces = document.KnownChunks.Msur
            .Where(s => s.Ck24 == targetCk24 && s.IndexCount >= 3)
            .ToList();

        if (groupSurfaces.Count == 0)
            throw new InvalidOperationException($"No surfaces found for CK24=0x{targetCk24:X6}");

        byte ck24Type = groupSurfaces[0].Ck24Type;
        ushort ck24ObjectId = groupSurfaces[0].Ck24ObjectId;

        HashSet<int> vertexIndexSet = [];
        List<byte> indexCounts = [];
        foreach (Pm4MsurEntry surface in groupSurfaces)
        {
            indexCounts.Add(surface.IndexCount);
            int first = checked((int)surface.MsviFirstIndex);
            int end = Math.Min(first + surface.IndexCount, msvi.Count);
            for (int i = first; i < end; i++)
            {
                int vii = checked((int)msvi[i]);
                if ((uint)vii < (uint)msvt.Count)
                    vertexIndexSet.Add(vii);
            }
        }

        List<int> sortedVerts = vertexIndexSet.OrderBy(static i => i).ToList();
        Dictionary<int, uint> globalToLocal = new(sortedVerts.Count);
        List<Vector3> verts = new(sortedVerts.Count);
        for (int i = 0; i < sortedVerts.Count; i++)
        {
            globalToLocal[sortedVerts[i]] = (uint)i;
            verts.Add(msvt[sortedVerts[i]]);
        }

        List<uint> localCornerIndices = [];
        foreach (Pm4MsurEntry surface in groupSurfaces)
        {
            int first = checked((int)surface.MsviFirstIndex);
            int end = Math.Min(first + surface.IndexCount, msvi.Count);
            for (int i = first; i < end; i++)
            {
                uint globalIdx = msvi[i];
                if (globalToLocal.TryGetValue(checked((int)globalIdx), out uint localIdx))
                    localCornerIndices.Add(localIdx);
            }
        }

        Vector3 pm4Min = new(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 pm4Max = new(float.MinValue, float.MinValue, float.MinValue);
        foreach (Vector3 v in verts)
        {
            pm4Min = Vector3.Min(pm4Min, v);
            pm4Max = Vector3.Max(pm4Max, v);
        }

        return new Pm4Ck24GeometryExport(
            targetCk24, ck24Type, ck24ObjectId,
            groupSurfaces.Count, vertexIndexSet.Count,
            pm4Min, pm4Max, verts, localCornerIndices, indexCounts);
    }

    public static WmoMeshInPm4Space ReadWmoInPm4Space(
        AdtWorldModelPlacement placement,
        string archiveRoot,
        ArchiveCatalogBootstrapOptions bootstrapOptions,
        int tileX, int tileY)
    {
        byte[] rootBytes = ArchiveVirtualFileReader.ReadVirtualFile(
            NormalizeVirtualPath(placement.ModelPath),
            [archiveRoot], bootstrapOptions);

        using MemoryStream rootStream = new(rootBytes, writable: false);
        WmoSummary summary = WmoSummaryReader.Read(rootStream, placement.ModelPath);

        Func<string, byte[]?> assetReader = virtualPath =>
        {
            try { return ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], bootstrapOptions); }
            catch { return null; }
        };

        rootStream.Position = 0;
        WmoRenderDocument renderDoc = WmoRenderDocumentReader.Read(
            rootStream, placement.ModelPath, assetReader);

        float rx = placement.Rotation.X * MathF.PI / 180f;
        float ry = placement.Rotation.Y * MathF.PI / 180f;
        float rz = placement.Rotation.Z * MathF.PI / 180f;
        Matrix4x4 worldTransform = Matrix4x4.CreateRotationZ(MathF.PI)
            * Matrix4x4.CreateRotationX(rx)
            * Matrix4x4.CreateRotationY(ry)
            * Matrix4x4.CreateRotationZ(rz)
            * Matrix4x4.CreateTranslation(placement.Position);

        List<Vector3> allLocalVerts = [];
        List<Vector3> allWorldVerts = [];
        List<Vector3> allPm4Verts = [];
        List<ushort> allIndices = [];
        List<WmoGroupFaceMaterialDetail> allFaceMaterials = [];
        int indexOffset = 0;

        foreach (WmoEmbeddedGroupMeshDetail group in renderDoc.Groups)
        {
            foreach (Vector3 v in group.Mesh.Vertices)
            {
                allLocalVerts.Add(v);
                Vector3 world = Vector3.Transform(v, worldTransform);
                allWorldVerts.Add(world);
                float pm4X = MapOrigin - world.X;
                float pm4Y = MapOrigin - world.Y;
                float pm4Z = world.Z;
                allPm4Verts.Add(new Vector3(pm4X, pm4Y, pm4Z));
            }
            foreach (ushort idx in group.Mesh.Indices)
                allIndices.Add((ushort)(idx + indexOffset));
            indexOffset += group.Mesh.Vertices.Count;
            foreach (WmoGroupFaceMaterialDetail fm in group.Mesh.FaceMaterials)
                allFaceMaterials.Add(fm);
        }

        return new WmoMeshInPm4Space(
            placement.UniqueId, placement.ModelPath,
            placement.Position, placement.Rotation,
            renderDoc.Groups.Count,
            allLocalVerts, allWorldVerts, allPm4Verts,
            allIndices, allFaceMaterials);
    }
}
