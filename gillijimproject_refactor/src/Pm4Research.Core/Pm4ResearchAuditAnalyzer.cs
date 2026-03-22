namespace Pm4Research.Core;

public static class Pm4ResearchAuditAnalyzer
{
    private static readonly string[] MslkRefIndexCandidateDomains =
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

    private static readonly IReadOnlyDictionary<string, int> ChunkStrides = new Dictionary<string, int>(StringComparer.Ordinal)
    {
        ["MSLK"] = 20,
        ["MSPV"] = 12,
        ["MSPI"] = 4,
        ["MSVT"] = 12,
        ["MSVI"] = 4,
        ["MSUR"] = 32,
        ["MSCN"] = 12,
        ["MPRL"] = 24,
        ["MPRR"] = 4,
        ["MDBH"] = 4,
        ["MDBI"] = 4,
        ["MDOS"] = 8,
        ["MDSF"] = 8,
    };

    private static readonly HashSet<string> KnownTypedSignatures = new(StringComparer.Ordinal)
    {
        "MVER",
        "MSHD",
        "MSLK",
        "MSPV",
        "MSPI",
        "MSVT",
        "MSVI",
        "MSUR",
        "MSCN",
        "MPRL",
        "MPRR",
        "MDBH",
        "MDBI",
        "MDBF",
        "MDOS",
        "MDSF"
    };

    public static Pm4DecodeAuditReport Analyze(Pm4ResearchFile file)
    {
        IReadOnlyList<Pm4ChunkDecodeAudit> chunkAudits = file.Chunks
            .GroupBy(static chunk => chunk.Signature)
            .OrderBy(static group => group.Key)
            .Select(group => BuildChunkAudit(file, group.Key, group.ToList()))
            .ToList();

        IReadOnlyList<Pm4ReferenceAudit> referenceAudits =
        [
            BuildMsviToMsvtAudit(file),
            BuildMspiToMspvAudit(file),
            BuildMsurToMsviAudit(file),
            BuildMslkRefIndexToMsurAudit(file),
            BuildMslkMspiWindowAudit(file),
            BuildMdsfToMsurAudit(file),
            BuildMdsfToMdosAudit(file),
            BuildMdosToMdbhAudit(file),
        ];

        return new Pm4DecodeAuditReport(
            file.SourcePath,
            file.Version,
            file.Chunks.Count,
            file.Chunks.Count(static chunk => KnownTypedSignatures.Contains(chunk.Signature)),
            file.Chunks.Count(static chunk => !KnownTypedSignatures.Contains(chunk.Signature)),
            file.Diagnostics.Any(static diagnostic => diagnostic.Contains("Trailing ", StringComparison.OrdinalIgnoreCase)),
            file.Diagnostics.Any(static diagnostic => diagnostic.Contains("overruns file", StringComparison.OrdinalIgnoreCase)),
            chunkAudits,
            referenceAudits,
            file.Chunks
                .Where(static chunk => !KnownTypedSignatures.Contains(chunk.Signature))
                .Select(static chunk => chunk.Signature)
                .Distinct(StringComparer.Ordinal)
                .OrderBy(static signature => signature)
                .ToList(),
            file.Diagnostics);
    }

    public static Pm4CorpusAuditReport AnalyzeDirectory(string inputDirectory)
    {
        List<Pm4DecodeAuditReport> reports = Directory
            .EnumerateFiles(inputDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .Select(Analyze)
            .ToList();

        IReadOnlyList<Pm4CorpusChunkAudit> chunkAudits = reports
            .SelectMany(static report => report.ChunkAudits)
            .GroupBy(static audit => audit.Signature)
            .Select(group => new Pm4CorpusChunkAudit(
                group.Key,
                group.Count(),
                group.Count(static audit => audit.HasMeaningfulData),
                group.Sum(static audit => audit.ChunkCount),
                group.Aggregate(0UL, static (sum, audit) => sum + audit.TotalBytes),
                group.Min(static audit => audit.MinChunkSize),
                group.Max(static audit => audit.MaxChunkSize),
                group.Select(static audit => audit.ExampleSizes).SelectMany(static sizes => sizes).Distinct().Count(),
                group.Sum(static audit => audit.EntryCount),
                group.Count(static audit => audit.StrideRemainderCount > 0),
                group.Count() > (reports.Count / 2),
                group.Select(static audit => audit.ExampleSizes).SelectMany(static sizes => sizes).Distinct().Order().Take(8).ToList()))
            .OrderByDescending(static audit => audit.IsCommon)
            .ThenByDescending(static audit => audit.FileCount)
            .ThenBy(static audit => audit.Signature)
            .ToList();

        IReadOnlyList<Pm4CorpusReferenceAudit> referenceAudits = reports
            .SelectMany(static report => report.ReferenceAudits)
            .GroupBy(static audit => audit.Name)
            .OrderBy(static group => group.Key)
            .Select(static group => new Pm4CorpusReferenceAudit(
                group.Key,
                group.Sum(static audit => audit.TotalCount),
                group.Sum(static audit => audit.InvalidCount),
                group.SelectMany(static audit => audit.Examples).Distinct().Take(8).ToList()))
            .ToList();

        IReadOnlyList<string> topDiagnostics = reports
            .SelectMany(static report => report.Diagnostics)
            .GroupBy(static diagnostic => diagnostic)
            .OrderByDescending(static group => group.Count())
            .ThenBy(static group => group.Key)
            .Take(20)
            .Select(static group => $"{group.Count()}x {group.Key}")
            .ToList();

        return new Pm4CorpusAuditReport(
            inputDirectory,
            reports.Count,
            reports.Count(static report => report.Diagnostics.Count > 0),
            reports.Count(static report => report.UnknownChunkSignatures.Count > 0),
            chunkAudits,
            referenceAudits,
            reports.SelectMany(static report => report.UnknownChunkSignatures).Distinct().Order().ToList(),
            topDiagnostics);
    }

    public static Pm4MslkRefIndexFileAudit AnalyzeMslkRefIndexFile(Pm4ResearchFile file)
    {
        List<Pm4MslkRefIndexMismatch> mismatches = new();
        int msurCount = file.KnownChunks.Msur.Count;
        for (int i = 0; i < file.KnownChunks.Mslk.Count; i++)
        {
            Pm4MslkEntry entry = file.KnownChunks.Mslk[i];
            if (entry.RefIndex < msurCount)
                continue;

            mismatches.Add(new Pm4MslkRefIndexMismatch(
                i,
                entry.RefIndex,
                entry.GroupObjectId,
                entry.LinkId,
                entry.TypeFlags,
                entry.Subtype,
                entry.MspiFirstIndex,
                entry.MspiIndexCount,
                BuildMslkRefIndexDomainFits(file, entry.RefIndex)));
        }

        (int? tileX, int? tileY) = TryParseTileCoordinates(file.SourcePath);
        return new Pm4MslkRefIndexFileAudit(
            file.SourcePath,
            tileX,
            tileY,
            file.Version,
            file.KnownChunks.Mslk.Count,
            msurCount,
            mismatches.Count,
            mismatches);
    }

    public static Pm4MslkRefIndexCorpusAudit AnalyzeMslkRefIndexDirectory(string inputDirectory)
    {
        List<Pm4MslkRefIndexFileAudit> fileAudits = Directory
            .EnumerateFiles(inputDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .Select(AnalyzeMslkRefIndexFile)
            .ToList();

        IReadOnlyList<Pm4MslkRefIndexDomainSummary> domainSummaries = MslkRefIndexCandidateDomains
            .Select(domain => new Pm4MslkRefIndexDomainSummary(
                domain,
                fileAudits.Sum(file => file.Mismatches.Count(mismatch => mismatch.DomainFits.Any(fit => fit.Domain == domain && fit.Fits))),
                fileAudits.Sum(file => file.Mismatches.Count(mismatch => mismatch.DomainFits.Any(fit => fit.Domain == domain && !fit.Fits)))))
            .ToList();

        IReadOnlyList<Pm4MslkRefIndexFileAudit> topFiles = fileAudits
            .Where(static file => file.InvalidRefIndexCount > 0)
            .OrderByDescending(static file => file.InvalidRefIndexCount)
            .ThenBy(static file => file.SourcePath)
            .Take(32)
            .Select(static file => file with { Mismatches = file.Mismatches.Take(32).ToList() })
            .ToList();

        return new Pm4MslkRefIndexCorpusAudit(
            inputDirectory,
            fileAudits.Count,
            fileAudits.Count(static file => file.InvalidRefIndexCount > 0),
            fileAudits.Sum(static file => file.InvalidRefIndexCount),
            domainSummaries,
            topFiles);
    }

    private static Pm4ChunkDecodeAudit BuildChunkAudit(Pm4ResearchFile file, string signature, IReadOnlyList<Pm4ChunkRecord> chunks)
    {
        ulong totalBytes = chunks.Aggregate(0UL, static (sum, chunk) => sum + chunk.Size);
        uint minChunkSize = chunks.Min(static chunk => chunk.Size);
        uint maxChunkSize = chunks.Max(static chunk => chunk.Size);
        IReadOnlyList<uint> exampleSizes = chunks.Select(static chunk => chunk.Size).Distinct().Order().Take(8).ToList();
        int strideRemainderCount = CountStrideRemainders(signature, chunks);

        return new Pm4ChunkDecodeAudit(
            signature,
            chunks.Count,
            totalBytes,
            minChunkSize,
            maxChunkSize,
            chunks.Select(static chunk => chunk.Size).Distinct().Count(),
            GetEntryCount(file, signature),
            HasMeaningfulData(file, signature),
            strideRemainderCount,
            exampleSizes);
    }

    private static int CountStrideRemainders(string signature, IReadOnlyList<Pm4ChunkRecord> chunks)
    {
        if (!ChunkStrides.TryGetValue(signature, out int stride))
            return 0;

        if (signature == "MDBH")
            return chunks.Count(static chunk => chunk.Size != 4);

        return chunks.Count(chunk => chunk.Size % stride != 0);
    }

    private static int GetEntryCount(Pm4ResearchFile file, string signature)
    {
        return signature switch
        {
            "MVER" => file.Version != 0 ? 1 : 0,
            "MSHD" => file.KnownChunks.Mshd is null ? 0 : 1,
            "MSLK" => file.KnownChunks.Mslk.Count,
            "MSPV" => file.KnownChunks.Mspv.Count,
            "MSPI" => file.KnownChunks.Mspi.Count,
            "MSVT" => file.KnownChunks.Msvt.Count,
            "MSVI" => file.KnownChunks.Msvi.Count,
            "MSUR" => file.KnownChunks.Msur.Count,
            "MSCN" => file.KnownChunks.Mscn.Count,
            "MPRL" => file.KnownChunks.Mprl.Count,
            "MPRR" => file.KnownChunks.Mprr.Count,
            "MDBH" => file.KnownChunks.Mdbh is null ? 0 : 1,
            "MDBI" => file.KnownChunks.Mdbi.Count,
            "MDBF" => file.KnownChunks.Mdbf.Count,
            "MDOS" => file.KnownChunks.Mdos.Count,
            "MDSF" => file.KnownChunks.Mdsf.Count,
            _ => 0,
        };
    }

    private static bool HasMeaningfulData(Pm4ResearchFile file, string signature)
    {
        return signature switch
        {
            "MVER" => file.Version != 0,
            "MSHD" => file.KnownChunks.Mshd is not null,
            "MSLK" => file.KnownChunks.Mslk.Count > 0,
            "MSPV" => file.KnownChunks.Mspv.Count > 0,
            "MSPI" => file.KnownChunks.Mspi.Count > 0,
            "MSVT" => file.KnownChunks.Msvt.Count > 0,
            "MSVI" => file.KnownChunks.Msvi.Count > 0,
            "MSUR" => file.KnownChunks.Msur.Count > 0,
            "MSCN" => file.KnownChunks.Mscn.Count > 0,
            "MPRL" => file.KnownChunks.Mprl.Count > 0,
            "MPRR" => file.KnownChunks.Mprr.Count > 0,
            "MDBH" => (file.KnownChunks.Mdbh?.DestructibleBuildingCount ?? 0) > 0,
            "MDBI" => file.KnownChunks.Mdbi.Count > 0,
            "MDBF" => file.KnownChunks.Mdbf.Any(static entry => !string.IsNullOrWhiteSpace(entry.Filename)),
            "MDOS" => file.KnownChunks.Mdos.Any(static entry => entry.DestructibleBuildingIndex != 0 || entry.DestructionState != 0),
            "MDSF" => file.KnownChunks.Mdsf.Count > 0,
            _ => false,
        };
    }

    private static IReadOnlyList<Pm4MslkRefIndexDomainFit> BuildMslkRefIndexDomainFits(Pm4ResearchFile file, ushort refIndex)
    {
        return
        [
            new("MSLK", file.KnownChunks.Mslk.Count, refIndex < file.KnownChunks.Mslk.Count),
            new("MSPI", file.KnownChunks.Mspi.Count, refIndex < file.KnownChunks.Mspi.Count),
            new("MSVI", file.KnownChunks.Msvi.Count, refIndex < file.KnownChunks.Msvi.Count),
            new("MSCN", file.KnownChunks.Mscn.Count, refIndex < file.KnownChunks.Mscn.Count),
            new("MPRL", file.KnownChunks.Mprl.Count, refIndex < file.KnownChunks.Mprl.Count),
            new("MSPV", file.KnownChunks.Mspv.Count, refIndex < file.KnownChunks.Mspv.Count),
            new("MSVT", file.KnownChunks.Msvt.Count, refIndex < file.KnownChunks.Msvt.Count),
            new("MPRR", file.KnownChunks.Mprr.Count, refIndex < file.KnownChunks.Mprr.Count),
        ];
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

    private static Pm4ReferenceAudit BuildMsviToMsvtAudit(Pm4ResearchFile file)
    {
        int vertexCount = file.KnownChunks.Msvt.Count;
        return BuildValueAudit(
            "MSVI->MSVT",
            file.KnownChunks.Msvi,
            value => value < vertexCount,
            value => $"MSVI value {value} >= MSVT count {vertexCount}");
    }

    private static Pm4ReferenceAudit BuildMspiToMspvAudit(Pm4ResearchFile file)
    {
        int vertexCount = file.KnownChunks.Mspv.Count;
        return BuildValueAudit(
            "MSPI->MSPV",
            file.KnownChunks.Mspi,
            value => value < vertexCount,
            value => $"MSPI value {value} >= MSPV count {vertexCount}");
    }

    private static Pm4ReferenceAudit BuildMsurToMsviAudit(Pm4ResearchFile file)
    {
        int indexCount = file.KnownChunks.Msvi.Count;
        return BuildRecordAudit(
            "MSUR->MSVI window",
            file.KnownChunks.Msur,
            static surface => (long)surface.MsviFirstIndex + surface.IndexCount,
            surface => surface.IndexCount == 0 || ((long)surface.MsviFirstIndex + surface.IndexCount) <= indexCount,
            surface => $"MSUR first={surface.MsviFirstIndex} count={surface.IndexCount} exceeds MSVI count {indexCount}");
    }

    private static Pm4ReferenceAudit BuildMslkRefIndexToMsurAudit(Pm4ResearchFile file)
    {
        int surfaceCount = file.KnownChunks.Msur.Count;
        return BuildRecordAudit(
            "MSLK.RefIndex->MSUR",
            file.KnownChunks.Mslk,
            static link => link.RefIndex,
            link => link.RefIndex < surfaceCount,
            link => $"MSLK refIndex {link.RefIndex} >= MSUR count {surfaceCount}");
    }

    private static Pm4ReferenceAudit BuildMslkMspiWindowAudit(Pm4ResearchFile file)
    {
        int indexCount = file.KnownChunks.Mspi.Count;
        IReadOnlyList<Pm4MslkEntry> activeLinks = file.KnownChunks.Mslk.Where(static link => link.MspiIndexCount > 0).ToList();
        return BuildRecordAudit(
            "MSLK.MSPI window",
            activeLinks,
            static link => link.MspiFirstIndex,
            link => link.MspiFirstIndex >= 0 && ((long)link.MspiFirstIndex + link.MspiIndexCount) <= indexCount,
            link => $"MSLK first={link.MspiFirstIndex} count={link.MspiIndexCount} exceeds MSPI count {indexCount}");
    }

    private static Pm4ReferenceAudit BuildMdsfToMsurAudit(Pm4ResearchFile file)
    {
        int surfaceCount = file.KnownChunks.Msur.Count;
        return BuildRecordAudit(
            "MDSF.MSUR->MSUR",
            file.KnownChunks.Mdsf,
            static entry => entry.MsurIndex,
            entry => entry.MsurIndex < surfaceCount,
            entry => $"MDSF msurIndex {entry.MsurIndex} >= MSUR count {surfaceCount}");
    }

    private static Pm4ReferenceAudit BuildMdsfToMdosAudit(Pm4ResearchFile file)
    {
        int mdosCount = file.KnownChunks.Mdos.Count;
        return BuildRecordAudit(
            "MDSF.MDOS->MDOS",
            file.KnownChunks.Mdsf,
            static entry => entry.MdosIndex,
            entry => entry.MdosIndex < mdosCount,
            entry => $"MDSF mdosIndex {entry.MdosIndex} >= MDOS count {mdosCount}");
    }

    private static Pm4ReferenceAudit BuildMdosToMdbhAudit(Pm4ResearchFile file)
    {
        uint buildingCount = file.KnownChunks.Mdbh?.DestructibleBuildingCount ?? 0;
        if (buildingCount == 0)
            return new Pm4ReferenceAudit("MDOS.buildingIndex->MDBH", 0, 0, 0, Array.Empty<string>());

        return BuildRecordAudit(
            "MDOS.buildingIndex->MDBH",
            file.KnownChunks.Mdos,
            static entry => entry.DestructibleBuildingIndex,
            entry => entry.DestructibleBuildingIndex < buildingCount,
            entry => $"MDOS buildingIndex {entry.DestructibleBuildingIndex} >= MDBH count {buildingCount}");
    }

    private static Pm4ReferenceAudit BuildValueAudit(
        string name,
        IReadOnlyList<uint> values,
        Func<uint, bool> predicate,
        Func<uint, string> describeFailure)
    {
        int validCount = 0;
        int invalidCount = 0;
        List<string> examples = new();
        for (int i = 0; i < values.Count; i++)
        {
            if (predicate(values[i]))
            {
                validCount++;
                continue;
            }

            invalidCount++;
            if (examples.Count < 8)
                examples.Add(describeFailure(values[i]));
        }

        return new Pm4ReferenceAudit(name, values.Count, validCount, invalidCount, examples);
    }

    private static Pm4ReferenceAudit BuildRecordAudit<T>(
        string name,
        IReadOnlyList<T> values,
        Func<T, long> keySelector,
        Func<T, bool> predicate,
        Func<T, string> describeFailure)
    {
        int validCount = 0;
        int invalidCount = 0;
        List<string> examples = new();
        for (int i = 0; i < values.Count; i++)
        {
            T value = values[i];
            if (predicate(value))
            {
                validCount++;
                continue;
            }

            invalidCount++;
            if (examples.Count < 8)
                examples.Add(describeFailure(value));
        }

        return new Pm4ReferenceAudit(name, values.Count, validCount, invalidCount, examples);
    }
}