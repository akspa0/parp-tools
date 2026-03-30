using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public static class Pm4ResearchUnknownsAnalyzer
{
    private static readonly string[] MslkMismatchCandidateDomains =
    [
        "MSLK",
        "MSPI",
        "MSVI",
        "MSCN",
        "MPRL",
        "MSPV",
        "MSVT",
        "MPRR"
    ];

    public static Pm4UnknownsReport AnalyzeDirectory(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<Pm4ResearchDocument> files = Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .ToList();

        Pm4CorpusAuditReport chunkPopulation = Pm4ResearchAuditAnalyzer.AnalyzeDirectory(resolvedDirectory);

        int nonEmptyFileCount = files.Count(static file => file.KnownChunks.Msur.Count > 0 || file.KnownChunks.Mslk.Count > 0 || file.KnownChunks.Mprl.Count > 0);

        int msviToMsvtFits = 0;
        int msviToMsvtMisses = 0;
        int mspiToMspvFits = 0;
        int mspiToMspvMisses = 0;
        int msurToMsviFits = 0;
        int msurToMsviMisses = 0;
        int mslkToMsurFits = 0;
        int mslkToMsurMisses = 0;
        int mslkToMprlFits = 0;
        int mslkToMprlMisses = 0;
        int mslkMspiWindowFits = 0;
        int mslkMspiWindowMisses = 0;
        int mdsfToMsurFits = 0;
        int mdsfToMsurMisses = 0;
        int mdsfToMdosFits = 0;
        int mdsfToMdosMisses = 0;
        int mdosToMdbhFits = 0;
        int mdosToMdbhMisses = 0;
        int mprrToMprlFits = 0;
        int mprrToMprlMisses = 0;
        int mprrToMsvtFits = 0;
        int mprrToMsvtMisses = 0;
        int groupObjectToMprlFits = 0;
        int groupObjectToMprlMisses = 0;

        int mspiIndicesOnly = 0;
        int mspiTrianglesOnly = 0;
        int mspiBoth = 0;
        int mspiNeither = 0;
        int mspiActiveLinks = 0;

        int linkIdTotal = 0;
        int linkIdSentinelTile = 0;
        int linkIdZero = 0;
        int linkIdOther = 0;

        Dictionary<string, int> mismatchDomainFits = MslkMismatchCandidateDomains.ToDictionary(static name => name, static _ => 0, StringComparer.Ordinal);

        Dictionary<string, int> typeFlagsCounts = new(StringComparer.Ordinal);
        Dictionary<string, int> subtypeCounts = new(StringComparer.Ordinal);
        Dictionary<string, int> systemFlagCounts = new(StringComparer.Ordinal);
        Dictionary<string, int> msurIndexCountCounts = new(StringComparer.Ordinal);
        Dictionary<string, int> msurAttributeMaskCounts = new(StringComparer.Ordinal);
        Dictionary<string, int> msurGroupKeyCounts = new(StringComparer.Ordinal);
        Dictionary<string, int> mshdField00Counts = new(StringComparer.Ordinal);
        Dictionary<string, int> mshdField04Counts = new(StringComparer.Ordinal);
        Dictionary<string, int> mshdField08Counts = new(StringComparer.Ordinal);
        Dictionary<string, int> mprlUnk02Counts = new(StringComparer.Ordinal);
        Dictionary<string, int> mprlUnk06Counts = new(StringComparer.Ordinal);
        Dictionary<string, int> mprlUnk14Counts = new(StringComparer.Ordinal);
        Dictionary<string, int> mprlUnk16Counts = new(StringComparer.Ordinal);
        Dictionary<string, int> mprrValue2Counts = new(StringComparer.Ordinal);
        Dictionary<string, int> decodedTileCounts = new(StringComparer.Ordinal);
        Dictionary<string, int> otherLinkIdCounts = new(StringComparer.Ordinal);

        short? mprlUnk14Min = null;
        short? mprlUnk14Max = null;

        foreach (Pm4ResearchDocument file in files)
        {
            if (file.KnownChunks.Mshd is not null)
            {
                AddCount(mshdField00Counts, file.KnownChunks.Mshd.Field00.ToString());
                AddCount(mshdField04Counts, file.KnownChunks.Mshd.Field04.ToString());
                AddCount(mshdField08Counts, file.KnownChunks.Mshd.Field08.ToString());
            }

            IReadOnlySet<ushort> mprlKeySet = file.KnownChunks.Mprl.Select(static entry => entry.Unk04).ToHashSet();
            int msurCount = file.KnownChunks.Msur.Count;
            int mprlCount = file.KnownChunks.Mprl.Count;
            int mspiCount = file.KnownChunks.Mspi.Count;
            int msvtCount = file.KnownChunks.Msvt.Count;
            int mscnCount = file.KnownChunks.Mscn.Count;
            int mslkCount = file.KnownChunks.Mslk.Count;
            int mspvCount = file.KnownChunks.Mspv.Count;
            int mprrCount = file.KnownChunks.Mprr.Count;

            foreach (uint value in file.KnownChunks.Msvi)
            {
                if (value < msvtCount)
                    msviToMsvtFits++;
                else
                    msviToMsvtMisses++;
            }

            foreach (uint value in file.KnownChunks.Mspi)
            {
                if (value < mspvCount)
                    mspiToMspvFits++;
                else
                    mspiToMspvMisses++;
            }

            foreach (Pm4MsurEntry surface in file.KnownChunks.Msur)
            {
                AddCount(msurIndexCountCounts, surface.IndexCount.ToString());
                AddCount(msurAttributeMaskCounts, $"0x{surface.AttributeMask:X2}");
                AddCount(msurGroupKeyCounts, surface.GroupKey.ToString());

                if (surface.IndexCount == 0 || ((long)surface.MsviFirstIndex + surface.IndexCount) <= file.KnownChunks.Msvi.Count)
                    msurToMsviFits++;
                else
                    msurToMsviMisses++;
            }

            foreach (Pm4MslkEntry link in file.KnownChunks.Mslk)
            {
                AddCount(typeFlagsCounts, $"0x{link.TypeFlags:X2}");
                AddCount(subtypeCounts, link.Subtype.ToString());
                AddCount(systemFlagCounts, $"0x{link.SystemFlag:X4}");

                if (link.RefIndex < msurCount)
                    mslkToMsurFits++;
                else
                {
                    mslkToMsurMisses++;
                    if (link.RefIndex < mslkCount)
                        mismatchDomainFits["MSLK"]++;
                    if (link.RefIndex < mspiCount)
                        mismatchDomainFits["MSPI"]++;
                    if (link.RefIndex < file.KnownChunks.Msvi.Count)
                        mismatchDomainFits["MSVI"]++;
                    if (link.RefIndex < mscnCount)
                        mismatchDomainFits["MSCN"]++;
                    if (link.RefIndex < mprlCount)
                        mismatchDomainFits["MPRL"]++;
                    if (link.RefIndex < mspvCount)
                        mismatchDomainFits["MSPV"]++;
                    if (link.RefIndex < msvtCount)
                        mismatchDomainFits["MSVT"]++;
                    if (link.RefIndex < mprrCount)
                        mismatchDomainFits["MPRR"]++;
                }

                if (link.RefIndex < mprlCount)
                    mslkToMprlFits++;
                else
                    mslkToMprlMisses++;

                if (link.GroupObjectId != 0 && link.GroupObjectId <= ushort.MaxValue && mprlKeySet.Contains((ushort)link.GroupObjectId))
                    groupObjectToMprlFits++;
                else if (link.GroupObjectId != 0)
                    groupObjectToMprlMisses++;

                if (link.MspiIndexCount > 0)
                {
                    mspiActiveLinks++;
                    bool indicesMode = link.MspiFirstIndex >= 0 && ((long)link.MspiFirstIndex + link.MspiIndexCount) <= mspiCount;
                    long trianglesEnd = ((long)link.MspiFirstIndex * 3) + (link.MspiIndexCount * 3L);
                    bool trianglesMode = link.MspiFirstIndex >= 0 && trianglesEnd <= mspiCount;

                    if (indicesMode)
                        mslkMspiWindowFits++;
                    else
                        mslkMspiWindowMisses++;

                    if (indicesMode && trianglesMode)
                        mspiBoth++;
                    else if (indicesMode)
                        mspiIndicesOnly++;
                    else if (trianglesMode)
                        mspiTrianglesOnly++;
                    else
                        mspiNeither++;
                }

                linkIdTotal++;
                if (link.LinkId == 0)
                {
                    linkIdZero++;
                }
                else if (TryDecodeTileLink(link.LinkId, out string? tileKey))
                {
                    linkIdSentinelTile++;
                    AddCount(decodedTileCounts, tileKey);
                }
                else
                {
                    linkIdOther++;
                    AddCount(otherLinkIdCounts, $"0x{link.LinkId:X8}");
                }
            }

            foreach (Pm4MprlEntry entry in file.KnownChunks.Mprl)
            {
                AddCount(mprlUnk02Counts, entry.Unk02.ToString());
                AddCount(mprlUnk06Counts, $"0x{entry.Unk06:X4}");
                AddCount(mprlUnk14Counts, entry.Unk14.ToString());
                AddCount(mprlUnk16Counts, $"0x{entry.Unk16:X4}");

                mprlUnk14Min = !mprlUnk14Min.HasValue || entry.Unk14 < mprlUnk14Min ? entry.Unk14 : mprlUnk14Min;
                mprlUnk14Max = !mprlUnk14Max.HasValue || entry.Unk14 > mprlUnk14Max ? entry.Unk14 : mprlUnk14Max;
            }

            foreach (Pm4MprrEntry entry in file.KnownChunks.Mprr)
            {
                AddCount(mprrValue2Counts, $"0x{entry.Value2:X4}");
                if (entry.IsSentinel)
                    continue;

                if (entry.Value1 < mprlCount)
                    mprrToMprlFits++;
                else
                    mprrToMprlMisses++;

                if (entry.Value1 < msvtCount)
                    mprrToMsvtFits++;
                else
                    mprrToMsvtMisses++;
            }

            foreach (Pm4MdsfEntry entry in file.KnownChunks.Mdsf)
            {
                if (entry.MsurIndex < msurCount)
                    mdsfToMsurFits++;
                else
                    mdsfToMsurMisses++;

                if (entry.MdosIndex < file.KnownChunks.Mdos.Count)
                    mdsfToMdosFits++;
                else
                    mdsfToMdosMisses++;
            }

            uint buildingCount = file.KnownChunks.Mdbh?.DestructibleBuildingCount ?? 0;
            foreach (Pm4MdosEntry entry in file.KnownChunks.Mdos)
            {
                if (buildingCount > 0 && entry.DestructibleBuildingIndex < buildingCount)
                    mdosToMdbhFits++;
                else if (buildingCount > 0 && (entry.DestructibleBuildingIndex != 0 || entry.DestructionState != 0))
                    mdosToMdbhMisses++;
            }
        }

        IReadOnlyList<Pm4RelationshipEdgeSummary> relationships =
        [
            BuildEdge("MSUR.Msvi window -> MSVI", msurToMsviFits, msurToMsviMisses, "Surface index spans into the mesh-index stream.", "If misses appear, inspect specific surface records and their CK24 group context."),
            BuildEdge("MSVI -> MSVT", msviToMsvtFits, msviToMsvtMisses, "Mesh-index stream into mesh vertices.", "Any miss is a hard decode/layout problem."),
            BuildEdge("MSLK.Mspi window -> MSPI", mslkMspiWindowFits, mslkMspiWindowMisses, "Link records into path-index stream.", "Compare count semantics with indices-mode vs triangles-mode ambiguity buckets."),
            BuildEdge("MSPI -> MSPV", mspiToMspvFits, mspiToMspvMisses, "Path-index stream into path vertices.", "Any miss suggests chunk-layout drift or broken link traversal."),
            BuildEdge("MSLK.RefIndex -> MSUR", mslkToMsurFits, mslkToMsurMisses, $"Mismatch candidate fits: {string.Join(", ", MslkMismatchCandidateDomains.Select(domain => $"{domain}={mismatchDomainFits[domain]}"))}", "Cluster bad values by LinkId and TypeFlags to isolate alternate RefIndex semantics."),
            BuildEdge("MSLK.RefIndex -> MPRL", mslkToMprlFits, mslkToMprlMisses, "Direct position-ref fit without considering other domains.", "Use this only as a direct-range signal, not as a proof of final semantics."),
            BuildEdge("MSLK.GroupObjectId -> MPRL.Unk04", groupObjectToMprlFits, groupObjectToMprlMisses, "Entry-level overlap between link-group ids and MPRL keys.", "Check whether misses cluster by TypeFlags or by tiles lacking linked MPRL payload."),
            BuildEdge("MDSF.MsurIndex -> MSUR", mdsfToMsurFits, mdsfToMsurMisses, "Surface ownership inside destructible mapping.", "If misses appear outside 00_00, inspect placeholder payload handling."),
            BuildEdge("MDSF.MdosIndex -> MDOS", mdsfToMdosFits, mdsfToMdosMisses, "Destructible mapping into MDOS entries.", "Use 00_00 as the primary populated reference tile for these chunks."),
            BuildEdge("MDOS.buildingIndex -> MDBH", mdosToMdbhFits, mdosToMdbhMisses, "Destructible state records into MDBH count.", "Determine whether buildingIndex is really an MDBH slot, MDBI index, hash, or another keyed identifier."),
            BuildEdge("MPRR.Value1 -> MPRL", mprrToMprlFits, mprrToMprlMisses, "Non-sentinel MPRR values against position refs.", "Check if Value1 instead behaves more like a geometry or scene-node reference on mismatch-heavy tiles."),
            BuildEdge("MPRR.Value1 -> MSVT", mprrToMsvtFits, mprrToMsvtMisses, "Non-sentinel MPRR values against mesh vertices.", "Compare this with MPRL fit to see whether MPRR is geometry-biased or mixed-mode.")
        ];

        IReadOnlyList<Pm4FieldDistribution> fieldDistributions =
        [
            BuildDistribution("MSHD.Field00", mshdField00Counts, null, null),
            BuildDistribution("MSHD.Field04", mshdField04Counts, null, null),
            BuildDistribution("MSHD.Field08", mshdField08Counts, null, null),
            BuildDistribution("MSLK.TypeFlags", typeFlagsCounts, null, "Object/link type classification remains open."),
            BuildDistribution("MSLK.Subtype", subtypeCounts, null, "Often looks floor- or layer-like, but semantics are not closed."),
            BuildDistribution("MSLK.SystemFlag", systemFlagCounts, null, "0x8000 dominates and likely acts as a constant flag or padding field."),
            BuildDistribution("MSUR.IndexCount", msurIndexCountCounts, null, "Surface fan/loop size distribution."),
            BuildDistribution("MSUR.AttributeMask", msurAttributeMaskCounts, null, "Bit meanings remain open."),
            BuildDistribution("MSUR.GroupKey", msurGroupKeyCounts, null, "Grouping/diagnostic semantics still open."),
            BuildDistribution("MPRL.Unk02", mprlUnk02Counts, null, "Often -1, but still not semantically closed."),
            BuildDistribution("MPRL.Unk06", mprlUnk06Counts, null, "Often 0x8000, likely a constant flag/status field."),
            BuildDistribution("MPRL.Unk14", mprlUnk14Counts, mprlUnk14Min.HasValue && mprlUnk14Max.HasValue ? $"{mprlUnk14Min}..{mprlUnk14Max}" : null, "Looks floor- or level-like in current evidence."),
            BuildDistribution("MPRL.Unk16", mprlUnk16Counts, null, "May encode entry class such as normal vs terminator."),
            BuildDistribution("MPRR.Value2", mprrValue2Counts, null, "Secondary MPRR field still open.")
        ];

        Pm4LinkIdPatternSummary linkIdPatterns = new(
            linkIdTotal,
            linkIdSentinelTile,
            linkIdZero,
            linkIdOther,
            ToTopFrequencies(decodedTileCounts),
            ToTopFrequencies(otherLinkIdCounts));

        Pm4MspiInterpretationSummary mspiInterpretation = new(
            mspiActiveLinks,
            mspiIndicesOnly,
            mspiTrianglesOnly,
            mspiBoth,
            mspiNeither);

        IReadOnlyList<Pm4UnknownFinding> unknowns =
        [
            new("MSLK.RefIndex semantics", "open", $"{mslkToMsurMisses} entries do not fit MSUR; mismatch domains strongest in MSPI/MSVI/MSCN/MSLK and weak in MPRL ({mismatchDomainFits["MPRL"]} fits).", "Group mismatches by LinkId, TypeFlags, and repeated RefIndex bands to isolate alternate target domains."),
            new("MSLK.MspiIndexCount interpretation", (mspiTrianglesOnly > 0 && mspiIndicesOnly > 0) || mspiBoth > 0 ? "open" : "partial", $"active={mspiActiveLinks}, indicesOnly={mspiIndicesOnly}, trianglesOnly={mspiTrianglesOnly}, both={mspiBoth}, neither={mspiNeither}", "Use mismatch clusters and old rollback traces to determine whether count is indices, triangles, or mode-dependent."),
            new("MSLK.TypeFlags and Subtype meaning", "open", $"TypeFlags distinct={typeFlagsCounts.Count}, Subtype distinct={subtypeCounts.Count}", "Correlate TypeFlags/Subtype with MPRL class, CK24 size, and LinkId families."),
            new("MPRL.Unk14 / Unk16 semantics", "open", $"Unk14 range={mprlUnk14Min}..{mprlUnk14Max}, Unk16 distinct={mprlUnk16Counts.Count}", "Compare per-object distributions against external ADT/object floor truth on trusted tiles."),
            new("MSHD header field meaning", "open", $"Field00 distinct={mshdField00Counts.Count}, Field04 distinct={mshdField04Counts.Count}, Field08 distinct={mshdField08Counts.Count}", "Check correlations with tile coordinates, chunk counts, and non-empty geometry/link payload."),
            new("MPRR field semantics", "open", $"Value1 fits: MPRL={mprrToMprlFits}, MSVT={mprrToMsvtFits}; Value2 distinct={mprrValue2Counts.Count}", "Cluster non-sentinel Value1/Value2 pairs by tile and compare with MSLK/scene-node usage."),
            new("Destructible payload integration", mdosToMdbhMisses > 0 ? "partial" : "open", $"MDBH/MDOS/MDSF are populated on one tile; MDOS->MDBH has fits={mdosToMdbhFits}, misses={mdosToMdbhMisses}", "Use development_00_00 and external Wintergrasp truth to determine whether MDOS.buildingIndex is a direct slot, indirection, or hashed id."),
            new("LinkId extended meaning", linkIdOther > 0 ? "partial" : "verified", $"sentinel tile links={linkIdSentinelTile}, zero={linkIdZero}, other={linkIdOther}", "Decode the non-0xFFFF patterns and compare them with mismatch-heavy LinkId clusters."),
            new("Coordinate/frame ownership", "open", "Corpus stats can validate link structure, but final PM4 local/world frame ownership still requires PM4<->ADT/object correlation on trusted tiles.", "Use the verified raw relationships here as constraints during PM4/ADT correlation instead of leading with viewer transforms.")
        ];

        IReadOnlyList<string> notes =
        [
            "00_00 remains the best destructible/reference tile but is not representative of the general MSLK.RefIndex mismatch population.",
            "Old rollback tooling previously split MSLK.RefIndex into MPRL-vs-MSVT buckets; the current corpus evidence keeps MPRL as a weak mismatch target.",
            "This report is decode-confidence evidence only; it does not prove final viewer reconstruction semantics by itself."
        ];

        return new Pm4UnknownsReport(
            resolvedDirectory,
            files.Count,
            nonEmptyFileCount,
            chunkPopulation.ChunkAudits,
            relationships,
            fieldDistributions,
            linkIdPatterns,
            mspiInterpretation,
            unknowns,
            notes);
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
        return new Pm4FieldDistribution(field, counts.Values.Sum(), counts.Count, range, ToTopFrequencies(counts), notes);
    }

    private static IReadOnlyList<Pm4ValueFrequency> ToTopFrequencies(Dictionary<string, int> counts)
    {
        return counts
            .OrderByDescending(static kv => kv.Value)
            .ThenBy(static kv => kv.Key)
            .Take(12)
            .Select(static kv => new Pm4ValueFrequency(kv.Key, kv.Value))
            .ToList();
    }

    private static void AddCount(Dictionary<string, int> counts, string key)
    {
        counts.TryGetValue(key, out int existing);
        counts[key] = existing + 1;
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
}