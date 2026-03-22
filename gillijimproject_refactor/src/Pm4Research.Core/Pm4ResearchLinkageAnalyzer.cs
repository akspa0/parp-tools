using System.Numerics;

namespace Pm4Research.Core;

public static class Pm4ResearchLinkageAnalyzer
{
    private static readonly string[] CandidateDomains =
    {
        "MSLK",
        "MSPI",
        "MSVI",
        "MSCN",
        "MPRL",
        "MSPV",
        "MSVT",
        "MPRR"
    };

    public static Pm4LinkageReport AnalyzeDirectory(string inputDirectory)
    {
        List<Pm4ResearchFile> files = Directory
            .EnumerateFiles(inputDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .ToList();

        int filesWithRefIndexMismatches = 0;
        int filesWithBadMdos = 0;
        int totalRefIndexMismatchCount = 0;
        int mdosToMscnFits = 0;
        int mdosToMscnMisses = 0;
        int mismatchLow16Fits = 0;
        int mismatchLow16Misses = 0;
        int mismatchLow24Fits = 0;
        int mismatchLow24Misses = 0;

        var mismatchTypeCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var mismatchSubtypeCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var badMdosTypeCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var badMdosSurfaceCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var objectIdReuseCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var objectIdTypeFanoutCounts = new Dictionary<string, int>(StringComparer.Ordinal);

        var distinctCk24 = new HashSet<uint>();
        var distinctCk24ObjectIds = new HashSet<ushort>();
        int objectIdGroupsAnalyzed = 0;
        int reusedObjectIdGroupCount = 0;
        int reusedAcrossTypeGroupCount = 0;

        List<Pm4BadMdosCluster> badMdosClusters = new();
        List<Pm4Ck24ObjectIdReuseCase> reuseCases = new();
        var mismatchFamilies = new Dictionary<string, MismatchFamilyAccumulator>(StringComparer.Ordinal);

        foreach (Pm4ResearchFile file in files)
        {
            IReadOnlyList<Pm4MsurEntry> msur = file.KnownChunks.Msur;
            IReadOnlyList<Pm4MslkEntry> mslk = file.KnownChunks.Mslk;
            IReadOnlyList<uint> msvi = file.KnownChunks.Msvi;
            IReadOnlyList<Vector3> msvt = file.KnownChunks.Msvt;
            int mscnCount = file.KnownChunks.Mscn.Count;
            int msurCount = msur.Count;

            (int? tileX, int? tileY) = TryParseTileCoordinates(file.SourcePath);

            HashSet<uint> fileCk24Set = msur.Select(static surface => surface.Ck24).ToHashSet();
            HashSet<ushort> fileCk24ObjectIds = msur.Select(static surface => surface.Ck24ObjectId).ToHashSet();
            foreach (uint value in fileCk24Set)
                distinctCk24.Add(value);
            foreach (ushort value in fileCk24ObjectIds)
                distinctCk24ObjectIds.Add(value);

            bool fileHasBadMdos = false;
            foreach (Pm4MsurEntry surface in msur)
            {
                if (surface.MdosIndex < mscnCount)
                    mdosToMscnFits++;
                else
                {
                    mdosToMscnMisses++;
                    fileHasBadMdos = true;
                }
            }

            if (fileHasBadMdos)
                filesWithBadMdos++;

            foreach (IGrouping<ushort, Pm4MsurEntry> objectIdGroup in msur.Where(static surface => surface.Ck24ObjectId != 0).GroupBy(static surface => surface.Ck24ObjectId))
            {
                List<Pm4MsurEntry> groupedSurfaces = objectIdGroup.ToList();
                List<uint> groupedCk24Values = groupedSurfaces.Select(static surface => surface.Ck24).Distinct().Order().ToList();
                int distinctTypeCount = groupedSurfaces.Select(static surface => surface.Ck24Type).Distinct().Count();

                objectIdGroupsAnalyzed++;
                AddCount(objectIdReuseCounts, groupedCk24Values.Count.ToString());
                AddCount(objectIdTypeFanoutCounts, distinctTypeCount.ToString());

                if (groupedCk24Values.Count > 1)
                    reusedObjectIdGroupCount++;
                if (distinctTypeCount > 1)
                    reusedAcrossTypeGroupCount++;

                if (groupedCk24Values.Count > 1 || distinctTypeCount > 1)
                {
                    var ck24Counts = groupedSurfaces
                        .GroupBy(static surface => surface.Ck24)
                        .OrderByDescending(static group => group.Count())
                        .ThenBy(static group => group.Key)
                        .Take(8)
                        .Select(static group => new Pm4ValueFrequency($"0x{group.Key:X6}", group.Count()))
                        .ToList();

                    reuseCases.Add(new Pm4Ck24ObjectIdReuseCase(
                        file.SourcePath,
                        tileX,
                        tileY,
                        objectIdGroup.Key,
                        groupedCk24Values.Count,
                        distinctTypeCount,
                        groupedSurfaces.Count,
                        ck24Counts));
                }
            }

            foreach (IGrouping<uint, Pm4MsurEntry> ck24Group in msur.GroupBy(static surface => surface.Ck24))
            {
                List<Pm4MsurEntry> surfaces = ck24Group.ToList();
                int validMdosCount = 0;
                int invalidMdosCount = 0;
                HashSet<uint> distinctValid = new();
                HashSet<uint> distinctInvalid = new();
                HashSet<uint> meshVertexIndices = new();

                foreach (Pm4MsurEntry surface in surfaces)
                {
                    if (surface.MdosIndex < mscnCount)
                    {
                        validMdosCount++;
                        distinctValid.Add(surface.MdosIndex);
                    }
                    else
                    {
                        invalidMdosCount++;
                        distinctInvalid.Add(surface.MdosIndex);
                    }

                    for (int i = 0; i < surface.IndexCount; i++)
                    {
                        uint msviIndex = surface.MsviFirstIndex + (uint)i;
                        if (msviIndex >= msvi.Count)
                            continue;

                        uint msvtIndex = msvi[(int)msviIndex];
                        if (msvtIndex < msvt.Count)
                            meshVertexIndices.Add(msvtIndex);
                    }
                }

                if (invalidMdosCount > 0)
                {
                    Pm4MsurEntry first = surfaces[0];
                    AddCount(badMdosTypeCounts, $"0x{first.Ck24Type:X2}");
                    AddCount(badMdosSurfaceCounts, surfaces.Count.ToString());

                    badMdosClusters.Add(new Pm4BadMdosCluster(
                        file.SourcePath,
                        tileX,
                        tileY,
                        first.Ck24,
                        first.Ck24Type,
                        first.Ck24ObjectId,
                        surfaces.Count,
                        invalidMdosCount,
                        validMdosCount,
                        distinctInvalid.Count,
                        distinctValid.Count,
                        meshVertexIndices.Count));
                }
            }

            bool fileHasMscn = mscnCount > 0;
            bool fileHasMismatch = false;

            foreach (Pm4MslkEntry link in mslk)
            {
                if (link.RefIndex < msurCount)
                    continue;

                fileHasMismatch = true;
                totalRefIndexMismatchCount++;
                AddCount(mismatchTypeCounts, $"0x{link.TypeFlags:X2}");
                AddCount(mismatchSubtypeCounts, link.Subtype.ToString());

                ushort low16 = (ushort)(link.GroupObjectId & 0xFFFF);
                uint low24 = link.GroupObjectId & 0x00FF_FFFF;

                if (fileCk24ObjectIds.Contains(low16))
                    mismatchLow16Fits++;
                else
                    mismatchLow16Misses++;

                if (fileCk24Set.Contains(low24))
                    mismatchLow24Fits++;
                else
                    mismatchLow24Misses++;

                string familyKey = BuildFamilyKey(link);
                if (!mismatchFamilies.TryGetValue(familyKey, out MismatchFamilyAccumulator? family))
                {
                    family = new MismatchFamilyAccumulator(familyKey);
                    mismatchFamilies.Add(familyKey, family);
                }

                family.EntryCount++;
                family.FilePaths.Add(file.SourcePath ?? string.Empty);
                family.GroupObjectIds.Add(link.GroupObjectId);
                family.Low16ObjectIds.Add(low16);
                family.RefIndices.Add(link.RefIndex);
                if (fileCk24ObjectIds.Contains(low16))
                    family.MatchingCk24ObjectIdEntryCount++;
                if (fileCk24Set.Contains(low24))
                    family.MatchingFullCk24EntryCount++;
                if (fileHasMscn)
                    family.EntriesInFilesWithMscn++;
                if (fileHasBadMdos)
                    family.EntriesInFilesWithBadMdos++;
                AddCount(family.Low16Counts, low16.ToString());
                AddDomainFits(family.DomainCounts, link.RefIndex, file);
            }

            if (fileHasMismatch)
                filesWithRefIndexMismatches++;
        }

        List<Pm4RelationshipEdgeSummary> relationships =
        [
            BuildEdge("MSUR.MdosIndex -> MSCN", mdosToMscnFits, mdosToMscnMisses, "Surface records referencing MSCN scene-node entries across the raw corpus.", "Break the misses down by CK24 type/object layer and compare them with placeholder or mismatch-heavy tiles."),
            BuildEdge("RefIndex mismatch low16(GroupObjectId) -> CK24ObjectId", mismatchLow16Fits, mismatchLow16Misses, "Bad MSLK.RefIndex entries whose GroupObjectId low16 matches a CK24ObjectId in the same file.", "If this clusters by LinkId/TypeFlags, low16 may be a member or hierarchy id instead of a full object key."),
            BuildEdge("RefIndex mismatch low24(GroupObjectId) -> CK24", mismatchLow24Fits, mismatchLow24Misses, "Bad MSLK.RefIndex entries whose GroupObjectId low24 matches a full CK24 key in the same file.", "Current weakness here argues against treating GroupObjectId as a direct full-CK24 identity field."),
            BuildEdge("Non-zero CK24ObjectId reused across full CK24s", reusedObjectIdGroupCount, Math.Max(0, objectIdGroupsAnalyzed - reusedObjectIdGroupCount), "File-local CK24 object-id groups whose low16 id maps to more than one full CK24 value.", "If reuse is common, treat the UI object-id as a sub-identifier or hierarchy-member key, not a globally unique PM4 object id."),
            BuildEdge("Non-zero CK24ObjectId reused across multiple CK24 types", reusedAcrossTypeGroupCount, Math.Max(0, objectIdGroupsAnalyzed - reusedAcrossTypeGroupCount), "File-local CK24 object-id groups that span multiple CK24 type bytes.", "Cross-type reuse is a stronger hint that low16 belongs to a broader linkage layer than one isolated rendered object."),
        ];

        Pm4LinkageIdentitySummary identitySummary = new(
            distinctCk24.Count,
            distinctCk24ObjectIds.Count,
            objectIdGroupsAnalyzed,
            reusedObjectIdGroupCount,
            reusedAcrossTypeGroupCount,
            reuseCases
                .OrderByDescending(static item => item.DistinctCk24Count)
                .ThenByDescending(static item => item.DistinctTypeCount)
                .ThenByDescending(static item => item.SurfaceCount)
                .ThenBy(static item => item.SourcePath)
                .Take(24)
                .ToList());

        List<Pm4FieldDistribution> distributions =
        [
            BuildDistribution("RefIndexMismatch.TypeFlags", mismatchTypeCounts, null, "Mismatch-heavy MSLK link type flags."),
            BuildDistribution("RefIndexMismatch.Subtype", mismatchSubtypeCounts, null, "Mismatch-heavy MSLK link subtype values."),
            BuildDistribution("BadMdos.Ck24Type", badMdosTypeCounts, null, "CK24 type bytes for groups carrying invalid MSCN node references."),
            BuildDistribution("BadMdos.SurfaceCount", badMdosSurfaceCounts, null, "Surface counts for CK24 groups with invalid MSCN node references."),
            BuildDistribution("Ck24ObjectId.ReuseCountPerFile", objectIdReuseCounts, null, "How many distinct full CK24 values share one non-zero low16 object id inside a file."),
            BuildDistribution("Ck24ObjectId.TypeFanoutPerFile", objectIdTypeFanoutCounts, null, "How many distinct CK24 type bytes share one non-zero low16 object id inside a file."),
        ];

        List<Pm4LinkageMismatchFamily> topMismatchFamilies = mismatchFamilies.Values
            .OrderByDescending(static family => family.EntryCount)
            .ThenBy(static family => family.FamilyKey)
            .Take(32)
            .Select(static family => family.ToReport())
            .ToList();

        List<Pm4BadMdosCluster> topBadMdosClusters = badMdosClusters
            .OrderByDescending(static cluster => cluster.InvalidMdosCount)
            .ThenByDescending(static cluster => cluster.SurfaceCount)
            .ThenBy(static cluster => cluster.SourcePath)
            .Take(32)
            .ToList();

        List<string> notes =
        [
            "The UI Ck24ObjectId is not an independent decoded field; it is the low 16 bits of MSUR.PackedParams-derived CK24.",
            "This report treats that low-16 object-id layer as a candidate hierarchy/member identifier and measures reuse against full CK24 values and type bytes.",
            "High reuse of one low16 value across multiple full CK24 values is evidence against treating Ck24ObjectId as a globally unique object key by itself.",
        ];

        return new Pm4LinkageReport(
            inputDirectory,
            files.Count,
            filesWithRefIndexMismatches,
            filesWithBadMdos,
            totalRefIndexMismatchCount,
            relationships,
            identitySummary,
            distributions,
            topMismatchFamilies,
            topBadMdosClusters,
            notes);
    }

    private static void AddDomainFits(Dictionary<string, int> domainCounts, ushort refIndex, Pm4ResearchFile file)
    {
        if (refIndex < file.KnownChunks.Mslk.Count)
            AddCount(domainCounts, "MSLK");
        if (refIndex < file.KnownChunks.Mspi.Count)
            AddCount(domainCounts, "MSPI");
        if (refIndex < file.KnownChunks.Msvi.Count)
            AddCount(domainCounts, "MSVI");
        if (refIndex < file.KnownChunks.Mscn.Count)
            AddCount(domainCounts, "MSCN");
        if (refIndex < file.KnownChunks.Mprl.Count)
            AddCount(domainCounts, "MPRL");
        if (refIndex < file.KnownChunks.Mspv.Count)
            AddCount(domainCounts, "MSPV");
        if (refIndex < file.KnownChunks.Msvt.Count)
            AddCount(domainCounts, "MSVT");
        if (refIndex < file.KnownChunks.Mprr.Count)
            AddCount(domainCounts, "MPRR");
    }

    private static string BuildFamilyKey(Pm4MslkEntry link)
    {
        string linkKey = TryDecodeTileLink(link.LinkId, out string? tileKey)
            ? $"tile={tileKey}"
            : $"link=0x{link.LinkId:X8}";
        return $"{linkKey}|flags=0x{link.TypeFlags:X2}|subtype={link.Subtype}";
    }

    private static bool TryDecodeTileLink(uint linkId, out string tileKey)
    {
        ushort high = (ushort)(linkId >> 16);
        if (high != 0xFFFF)
        {
            tileKey = string.Empty;
            return false;
        }

        ushort low = (ushort)(linkId & 0xFFFF);
        byte tileY = (byte)(low >> 8);
        byte tileX = (byte)(low & 0xFF);
        tileKey = $"{tileX}_{tileY}";
        return true;
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

    private static (int? TileX, int? TileY) TryParseTileCoordinates(string? sourcePath)
    {
        if (string.IsNullOrWhiteSpace(sourcePath))
            return (null, null);

        string fileName = Path.GetFileNameWithoutExtension(sourcePath);
        string[] parts = fileName.Split('_', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length < 2)
            return (null, null);

        if (!int.TryParse(parts[^2], out int tileX) || !int.TryParse(parts[^1], out int tileY))
            return (null, null);

        return (tileX, tileY);
    }

    private sealed class MismatchFamilyAccumulator
    {
        public MismatchFamilyAccumulator(string familyKey)
        {
            FamilyKey = familyKey;
        }

        public string FamilyKey { get; }
        public int EntryCount { get; set; }
        public HashSet<string> FilePaths { get; } = new(StringComparer.Ordinal);
        public HashSet<uint> GroupObjectIds { get; } = new();
        public HashSet<ushort> Low16ObjectIds { get; } = new();
        public HashSet<ushort> RefIndices { get; } = new();
        public int MatchingCk24ObjectIdEntryCount { get; set; }
        public int MatchingFullCk24EntryCount { get; set; }
        public int EntriesInFilesWithMscn { get; set; }
        public int EntriesInFilesWithBadMdos { get; set; }
        public Dictionary<string, int> DomainCounts { get; } = CandidateDomains.ToDictionary(static domain => domain, static _ => 0, StringComparer.Ordinal);
        public Dictionary<string, int> Low16Counts { get; } = new(StringComparer.Ordinal);

        public Pm4LinkageMismatchFamily ToReport()
        {
            return new Pm4LinkageMismatchFamily(
                FamilyKey,
                FilePaths.Count,
                EntryCount,
                GroupObjectIds.Count,
                Low16ObjectIds.Count,
                RefIndices.Count,
                MatchingCk24ObjectIdEntryCount,
                MatchingFullCk24EntryCount,
                EntriesInFilesWithMscn,
                EntriesInFilesWithBadMdos,
                DomainCounts
                    .OrderByDescending(static kv => kv.Value)
                    .ThenBy(static kv => kv.Key)
                    .Where(static kv => kv.Value > 0)
                    .Take(8)
                    .Select(static kv => new Pm4ValueFrequency(kv.Key, kv.Value))
                    .ToList(),
                Low16Counts
                    .OrderByDescending(static kv => kv.Value)
                    .ThenBy(static kv => kv.Key)
                    .Take(8)
                    .Select(static kv => new Pm4ValueFrequency(kv.Key, kv.Value))
                    .ToList());
        }
    }
}