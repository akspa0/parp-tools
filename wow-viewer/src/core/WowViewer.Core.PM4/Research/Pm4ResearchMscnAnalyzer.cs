using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public static class Pm4ResearchMscnAnalyzer
{
    private const float TileSize = 533.33333f;
    private const float MapCenter = 17066.66656f;
    private const float BoundsMargin = 2.0f;

    public static Pm4MscnRelationshipReport AnalyzeDirectory(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<Pm4ResearchDocument> files = Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .ToList();

        int filesWithMscn = 0;
        int filesWithTileCoordinates = 0;
        int totalMscnPointCount = 0;

        int mdosToMscnFits = 0;
        int mdosToMscnMisses = 0;
        int ck24GroupsWithMscn = 0;
        int ck24GroupsWithoutMscn = 0;
        int ck24GroupsWithBoth = 0;
        int ck24GroupsMscnOnly = 0;
        int ck24GroupsMeshOnly = 0;
        int ck24RawOverlapFits = 0;
        int ck24RawOverlapMisses = 0;
        int ck24SwappedOverlapFits = 0;
        int ck24SwappedOverlapMisses = 0;
        int mslkGroupLow16Fits = 0;
        int mslkGroupLow16Misses = 0;
        int mslkGroupLow24Fits = 0;
        int mslkGroupLow24Misses = 0;
        int groupsWithSharedMdosNodes = 0;

        int swappedWorldTileFitCount = 0;
        int rawWorldTileFitCount = 0;
        int ambiguousWorldTileFitCount = 0;
        int tileLocalLikeCount = 0;
        int neitherFitCount = 0;
        int filesSwappedDominant = 0;
        int filesRawDominant = 0;
        int filesTileLocalDominant = 0;
        int filesNoDominant = 0;

        Dictionary<string, int> ck24SurfaceCount = new(StringComparer.Ordinal);
        Dictionary<string, int> ck24DistinctMdosCount = new(StringComparer.Ordinal);
        Dictionary<string, int> ck24MeshVertexCount = new(StringComparer.Ordinal);
        Dictionary<string, int> ck24AlignmentMode = new(StringComparer.Ordinal);
        List<Pm4MscnClusterExample> clusterExamples = new();
        List<Pm4MscnClusterExample> invalidMdosExamples = new();

        foreach (Pm4ResearchDocument file in files)
        {
            IReadOnlyList<Vector3> mscn = file.KnownChunks.Mscn;
            IReadOnlyList<Pm4MsurEntry> msur = file.KnownChunks.Msur;
            IReadOnlyList<uint> msvi = file.KnownChunks.Msvi;
            IReadOnlyList<Vector3> msvt = file.KnownChunks.Msvt;

            if (mscn.Count > 0)
                filesWithMscn++;

            totalMscnPointCount += mscn.Count;

            bool hasTileCoordinates = Pm4CoordinateService.TryParseTileCoordinates(file.SourcePath ?? string.Empty, out int tileX, out int tileY);
            int? nullableTileX = hasTileCoordinates ? tileX : null;
            int? nullableTileY = hasTileCoordinates ? tileY : null;
            if (hasTileCoordinates)
                filesWithTileCoordinates++;

            int fileSwapped = 0;
            int fileRaw = 0;
            int fileTileLocal = 0;
            int fileNeither = 0;

            if (hasTileCoordinates)
            {
                foreach (Vector3 point in mscn)
                {
                    bool rawFits = IsWithinTileWorldBounds(point.X, point.Y, tileX, tileY);
                    bool swappedFits = IsWithinTileWorldBounds(point.Y, point.X, tileX, tileY);

                    if (swappedFits && rawFits)
                    {
                        ambiguousWorldTileFitCount++;
                    }
                    else if (swappedFits)
                    {
                        swappedWorldTileFitCount++;
                        fileSwapped++;
                    }
                    else if (rawFits)
                    {
                        rawWorldTileFitCount++;
                        fileRaw++;
                    }
                    else if (IsTileLocalLike(point))
                    {
                        tileLocalLikeCount++;
                        fileTileLocal++;
                    }
                    else
                    {
                        neitherFitCount++;
                        fileNeither++;
                    }
                }

                switch (ResolveDominantMode(fileSwapped, fileRaw, fileTileLocal, fileNeither))
                {
                    case "swapped-world":
                        filesSwappedDominant++;
                        break;
                    case "raw-world":
                        filesRawDominant++;
                        break;
                    case "tile-local":
                        filesTileLocalDominant++;
                        break;
                    default:
                        filesNoDominant++;
                        break;
                }
            }

            HashSet<uint> ck24Set = msur.Select(static surface => surface.Ck24).ToHashSet();
            HashSet<ushort> ck24ObjectIdSet = msur.Select(static surface => surface.Ck24ObjectId).ToHashSet();

            foreach (Pm4MslkEntry link in file.KnownChunks.Mslk)
            {
                if (link.GroupObjectId == 0)
                    continue;

                ushort low16 = (ushort)(link.GroupObjectId & 0xFFFF);
                uint low24 = link.GroupObjectId & 0x00FF_FFFF;

                if (ck24ObjectIdSet.Contains(low16))
                    mslkGroupLow16Fits++;
                else
                    mslkGroupLow16Misses++;

                if (ck24Set.Contains(low24))
                    mslkGroupLow24Fits++;
                else
                    mslkGroupLow24Misses++;
            }

            foreach (Pm4MsurEntry surface in msur)
            {
                if (surface.MdosIndex < mscn.Count)
                    mdosToMscnFits++;
                else
                    mdosToMscnMisses++;
            }

            foreach (IGrouping<uint, Pm4MsurEntry> group in msur.GroupBy(static surface => surface.Ck24))
            {
                List<Pm4MsurEntry> surfaces = group.ToList();
                HashSet<uint> distinctValidMdos = new();
                HashSet<uint> meshVertexIndices = new();
                int validMdosRefCount = 0;
                int invalidMdosRefCount = 0;

                foreach (Pm4MsurEntry surface in surfaces)
                {
                    if (surface.MdosIndex < mscn.Count)
                    {
                        validMdosRefCount++;
                        distinctValidMdos.Add(surface.MdosIndex);
                    }
                    else
                    {
                        invalidMdosRefCount++;
                    }

                    for (int index = 0; index < surface.IndexCount; index++)
                    {
                        uint msviIndex = surface.MsviFirstIndex + (uint)index;
                        if (msviIndex >= msvi.Count)
                            continue;

                        uint msvtIndex = msvi[(int)msviIndex];
                        if (msvtIndex < msvt.Count)
                            meshVertexIndices.Add(msvtIndex);
                    }
                }

                AddCount(ck24SurfaceCount, surfaces.Count.ToString());
                AddCount(ck24DistinctMdosCount, distinctValidMdos.Count.ToString());
                AddCount(ck24MeshVertexCount, meshVertexIndices.Count.ToString());

                bool hasMscn = distinctValidMdos.Count > 0;
                bool hasMesh = meshVertexIndices.Count > 0;
                if (hasMscn)
                    ck24GroupsWithMscn++;
                else
                    ck24GroupsWithoutMscn++;

                if (hasMscn && hasMesh)
                    ck24GroupsWithBoth++;
                else if (hasMscn)
                    ck24GroupsMscnOnly++;
                else if (hasMesh)
                    ck24GroupsMeshOnly++;

                if (validMdosRefCount > distinctValidMdos.Count && distinctValidMdos.Count > 0)
                    groupsWithSharedMdosNodes++;

                string alignmentMode = hasMscn && hasMesh
                    ? ClassifyAlignment(
                        BuildBounds(meshVertexIndices.Select(index => msvt[(int)index])),
                        BuildBounds(distinctValidMdos.Select(index => mscn[(int)index])),
                        BuildBounds(distinctValidMdos.Select(index => SwapXY(mscn[(int)index]))),
                        ref ck24RawOverlapFits,
                        ref ck24RawOverlapMisses,
                        ref ck24SwappedOverlapFits,
                        ref ck24SwappedOverlapMisses)
                    : hasMscn
                        ? "mscn-only"
                        : hasMesh
                            ? "mesh-only"
                            : "empty";

                AddCount(ck24AlignmentMode, alignmentMode);

                if (hasMscn)
                {
                    Pm4MsurEntry first = surfaces[0];
                    Pm4MscnClusterExample example = new(
                        file.SourcePath,
                        nullableTileX,
                        nullableTileY,
                        first.Ck24,
                        first.Ck24Type,
                        first.Ck24ObjectId,
                        surfaces.Count,
                        validMdosRefCount,
                        distinctValidMdos.Count,
                        invalidMdosRefCount,
                        meshVertexIndices.Count,
                        alignmentMode);

                    clusterExamples.Add(example);
                    if (invalidMdosRefCount > 0)
                        invalidMdosExamples.Add(example);
                }
            }
        }

        IReadOnlyList<Pm4RelationshipEdgeSummary> relationships =
        [
            BuildEdge("MSUR.MdosIndex -> MSCN", mdosToMscnFits, mdosToMscnMisses, "Surface records referencing MSCN scene-node entries.", "Inspect misses by CK24 family; these likely isolate placeholder or alternate-node semantics."),
            BuildEdge("CK24 group -> MSCN coverage", ck24GroupsWithMscn, ck24GroupsWithoutMscn, "CK24 groups with at least one valid MdosIndex-backed MSCN node.", "Break missing groups down by Ck24Type and surface count to see whether MSCN is intentionally absent for some object families."),
            BuildEdge("CK24 group -> combined mesh+MSCN geometry", ck24GroupsWithBoth, ck24GroupsMscnOnly + ck24GroupsMeshOnly, "Whether a CK24 family carries both mesh-side geometry and MSCN-side collision nodes.", "Use MSCN-only and mesh-only families as candidate missing-layer cases instead of assuming one geometry stream is sufficient."),
            BuildEdge("CK24 MSCN(raw) bounds -> MSVT bounds", ck24RawOverlapFits, ck24RawOverlapMisses, "AABB overlap between raw MSCN-backed bounds and mesh-backed bounds within the same CK24 group.", "If this stays weak while swapped overlap is strong, keep treating MSCN as an axis-swapped companion stream rather than raw mesh-space truth."),
            BuildEdge("CK24 MSCN(swapped XY) bounds -> MSVT bounds", ck24SwappedOverlapFits, ck24SwappedOverlapMisses, "AABB overlap after swapping MSCN X/Y before comparing to mesh-backed bounds.", "Use the better-fitting transform as the current MSCN-to-mesh working hypothesis, then validate it against ADT/object truth on trusted tiles."),
            BuildEdge("MSLK.GroupObjectId low16 -> CK24ObjectId", mslkGroupLow16Fits, mslkGroupLow16Misses, "Low 16 bits of link GroupObjectId compared against CK24 object ids present in the same file.", "If this is strong, treat low16 as an object-family or instance key candidate before inventing new GroupObjectId semantics."),
            BuildEdge("MSLK.GroupObjectId low24 -> CK24", mslkGroupLow24Fits, mslkGroupLow24Misses, "Low 24 bits of link GroupObjectId compared against full CK24 keys present in the same file.", "If this is weak while low16 is strong, GroupObjectId probably targets a narrower object-id layer than full CK24.")
        ];

        Pm4MscnCoordinateSummary coordinateSummary = new(
            totalMscnPointCount,
            swappedWorldTileFitCount,
            rawWorldTileFitCount,
            ambiguousWorldTileFitCount,
            tileLocalLikeCount,
            neitherFitCount,
            filesSwappedDominant,
            filesRawDominant,
            filesTileLocalDominant,
            filesNoDominant);

        IReadOnlyList<Pm4FieldDistribution> distributions =
        [
            BuildDistribution("CK24.SurfaceCount", ck24SurfaceCount, null, "How many MSUR surfaces typically contribute to one CK24 group."),
            BuildDistribution("CK24.DistinctMdosCount", ck24DistinctMdosCount, null, "Distinct MSCN scene-node references observed per CK24 group."),
            BuildDistribution("CK24.MeshVertexCount", ck24MeshVertexCount, null, "Distinct MSVT vertices reached through MSUR/MSVI for one CK24 group."),
            BuildDistribution("CK24.AlignmentMode", ck24AlignmentMode, null, "`swapped-only` is especially relevant if MSCN is an axis-swapped companion geometry stream.")
        ];

        IReadOnlyList<Pm4MscnClusterExample> topNonZeroClusters = clusterExamples
            .Where(static cluster => cluster.Ck24 != 0)
            .OrderByDescending(static cluster => cluster.DistinctMdosCount)
            .ThenByDescending(static cluster => cluster.SurfaceCount)
            .ThenBy(static cluster => cluster.SourcePath)
            .Take(24)
            .ToList();

        IReadOnlyList<Pm4MscnClusterExample> topZeroClusters = clusterExamples
            .Where(static cluster => cluster.Ck24 == 0)
            .OrderByDescending(static cluster => cluster.DistinctMdosCount)
            .ThenByDescending(static cluster => cluster.SurfaceCount)
            .ThenBy(static cluster => cluster.SourcePath)
            .Take(12)
            .ToList();

        IReadOnlyList<Pm4MscnClusterExample> topInvalidMdosClusters = invalidMdosExamples
            .OrderByDescending(static cluster => cluster.InvalidMdosRefCount)
            .ThenByDescending(static cluster => cluster.SurfaceCount)
            .ThenBy(static cluster => cluster.SourcePath)
            .Take(24)
            .ToList();

        IReadOnlyList<string> notes =
        [
            "This report keeps treating MSUR.MdosIndex as the main bridge into MSCN scene-node data.",
            "It is intended to test whether MSCN behaves like a missing ownership or collision layer, not to declare MSCN authoritative for final viewer reconstruction by itself.",
            $"CK24 groups reusing MSCN nodes: {groupsWithSharedMdosNodes}.",
            "If swapped XY overlap consistently beats raw overlap, MSCN remains more plausible as an axis-swapped companion stream than as raw mesh-space truth."
        ];

        return new Pm4MscnRelationshipReport(
            resolvedDirectory,
            files.Count,
            filesWithMscn,
            filesWithTileCoordinates,
            totalMscnPointCount,
            relationships,
            coordinateSummary,
            distributions,
            topNonZeroClusters,
            topZeroClusters,
            topInvalidMdosClusters,
            notes);
    }

    private static string ClassifyAlignment(
        Pm4Bounds3? meshBounds,
        Pm4Bounds3? rawMscnBounds,
        Pm4Bounds3? swappedMscnBounds,
        ref int rawFits,
        ref int rawMisses,
        ref int swappedFits,
        ref int swappedMisses)
    {
        bool rawOverlap = BoundsOverlap(meshBounds, rawMscnBounds);
        bool swappedOverlap = BoundsOverlap(meshBounds, swappedMscnBounds);

        if (rawOverlap)
            rawFits++;
        else
            rawMisses++;

        if (swappedOverlap)
            swappedFits++;
        else
            swappedMisses++;

        return rawOverlap switch
        {
            true when swappedOverlap => "both",
            true => "raw-only",
            false when swappedOverlap => "swapped-only",
            _ => "neither",
        };
    }

    private static Pm4RelationshipEdgeSummary BuildEdge(string edge, int fits, int misses, string evidence, string nextStep)
    {
        string status = fits > 0 && misses == 0
            ? "verified"
            : fits > 0 && misses > 0
                ? "partial"
                : fits == 0 && misses > 0
                    ? "unsupported"
                    : "unobserved";

        return new Pm4RelationshipEdgeSummary(edge, status, fits, misses, evidence, nextStep);
    }

    private static Pm4FieldDistribution BuildDistribution(string field, Dictionary<string, int> counts, string? range, string? notes)
    {
        return new Pm4FieldDistribution(
            field,
            counts.Values.Sum(),
            counts.Count,
            range,
            counts
                .OrderByDescending(static kv => kv.Value)
                .ThenBy(static kv => kv.Key)
                .Take(12)
                .Select(static kv => new Pm4ValueFrequency(kv.Key, kv.Value))
                .ToList(),
            notes);
    }

    private static void AddCount(Dictionary<string, int> counts, string key)
    {
        counts.TryGetValue(key, out int existing);
        counts[key] = existing + 1;
    }

    private static string ResolveDominantMode(int swapped, int raw, int tileLocal, int neither)
    {
        int max = Math.Max(Math.Max(swapped, raw), Math.Max(tileLocal, neither));
        int matchCount = 0;
        if (swapped == max)
            matchCount++;
        if (raw == max)
            matchCount++;
        if (tileLocal == max)
            matchCount++;
        if (neither == max)
            matchCount++;

        if (max == 0 || matchCount != 1)
            return "no-dominant";

        if (swapped == max)
            return "swapped-world";
        if (raw == max)
            return "raw-world";
        if (tileLocal == max)
            return "tile-local";

        return "neither";
    }

    private static bool IsTileLocalLike(Vector3 point)
    {
        return point.X >= 0 && point.X <= TileSize && point.Y >= 0 && point.Y <= TileSize;
    }

    private static bool IsWithinTileWorldBounds(float x, float y, int tileX, int tileY)
    {
        float minX = MapCenter - ((tileX + 1) * TileSize);
        float maxX = MapCenter - (tileX * TileSize);
        float minY = MapCenter - ((tileY + 1) * TileSize);
        float maxY = MapCenter - (tileY * TileSize);
        return x >= minX && x <= maxX && y >= minY && y <= maxY;
    }

    private static Pm4Bounds3? BuildBounds(IEnumerable<Vector3> points)
    {
        using IEnumerator<Vector3> enumerator = points.GetEnumerator();
        if (!enumerator.MoveNext())
            return null;

        Vector3 min = enumerator.Current;
        Vector3 max = enumerator.Current;
        while (enumerator.MoveNext())
        {
            min = Vector3.Min(min, enumerator.Current);
            max = Vector3.Max(max, enumerator.Current);
        }

        return new Pm4Bounds3(min, max);
    }

    private static bool BoundsOverlap(Pm4Bounds3? left, Pm4Bounds3? right)
    {
        if (left is null || right is null)
            return false;

        return left.Min.X <= right.Max.X + BoundsMargin && left.Max.X + BoundsMargin >= right.Min.X &&
               left.Min.Y <= right.Max.Y + BoundsMargin && left.Max.Y + BoundsMargin >= right.Min.Y &&
               left.Min.Z <= right.Max.Z + BoundsMargin && left.Max.Z + BoundsMargin >= right.Min.Z;
    }

    private static Vector3 SwapXY(Vector3 value)
    {
        return new Vector3(value.Y, value.X, value.Z);
    }
}