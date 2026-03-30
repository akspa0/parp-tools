using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public static class Pm4ResearchAuditAnalyzer
{
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

    public static Pm4DecodeAuditReport Analyze(Pm4ResearchDocument document)
    {
        IReadOnlyList<Pm4ChunkDecodeAudit> chunkAudits = document.Chunks
            .GroupBy(static chunk => chunk.Signature)
            .OrderBy(static group => group.Key)
            .Select(group => BuildChunkAudit(document, group.Key, group.ToList()))
            .ToList();

        IReadOnlyList<Pm4ReferenceAudit> referenceAudits =
        [
            BuildMsviToMsvtAudit(document),
            BuildMspiToMspvAudit(document),
            BuildMsurToMsviAudit(document),
            BuildMslkRefIndexToMsurAudit(document),
            BuildMslkMspiWindowAudit(document),
            BuildMdsfToMsurAudit(document),
            BuildMdsfToMdosAudit(document),
            BuildMdosToMdbhAudit(document)
        ];

        return new Pm4DecodeAuditReport(
            document.SourcePath,
            document.Version,
            document.Chunks.Count,
            document.Chunks.Count(static chunk => KnownTypedSignatures.Contains(chunk.Signature)),
            document.Chunks.Count(static chunk => !KnownTypedSignatures.Contains(chunk.Signature)),
            document.Diagnostics.Any(static diagnostic => diagnostic.Contains("Trailing ", StringComparison.OrdinalIgnoreCase)),
            document.Diagnostics.Any(static diagnostic => diagnostic.Contains("overruns file", StringComparison.OrdinalIgnoreCase)),
            chunkAudits,
            referenceAudits,
            document.Chunks
                .Where(static chunk => !KnownTypedSignatures.Contains(chunk.Signature))
                .Select(static chunk => chunk.Signature)
                .Distinct(StringComparer.Ordinal)
                .OrderBy(static signature => signature)
                .ToList(),
            document.Diagnostics);
    }

    public static Pm4CorpusAuditReport AnalyzeDirectory(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);
        List<Pm4DecodeAuditReport> reports = Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
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
            resolvedDirectory,
            reports.Count,
            reports.Count(static report => report.Diagnostics.Count > 0),
            reports.Count(static report => report.UnknownChunkSignatures.Count > 0),
            chunkAudits,
            referenceAudits,
            reports.SelectMany(static report => report.UnknownChunkSignatures).Distinct().Order().ToList(),
            topDiagnostics);
    }

    private static Pm4ChunkDecodeAudit BuildChunkAudit(Pm4ResearchDocument document, string signature, IReadOnlyList<Pm4ChunkRecord> chunks)
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
            GetEntryCount(document, signature),
            HasMeaningfulData(document, signature),
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

    private static int GetEntryCount(Pm4ResearchDocument document, string signature)
    {
        return signature switch
        {
            "MVER" => document.Version != 0 ? 1 : 0,
            "MSHD" => document.KnownChunks.Mshd is null ? 0 : 1,
            "MSLK" => document.KnownChunks.Mslk.Count,
            "MSPV" => document.KnownChunks.Mspv.Count,
            "MSPI" => document.KnownChunks.Mspi.Count,
            "MSVT" => document.KnownChunks.Msvt.Count,
            "MSVI" => document.KnownChunks.Msvi.Count,
            "MSUR" => document.KnownChunks.Msur.Count,
            "MSCN" => document.KnownChunks.Mscn.Count,
            "MPRL" => document.KnownChunks.Mprl.Count,
            "MPRR" => document.KnownChunks.Mprr.Count,
            "MDBH" => document.KnownChunks.Mdbh is null ? 0 : 1,
            "MDBI" => document.KnownChunks.Mdbi.Count,
            "MDBF" => document.KnownChunks.Mdbf.Count,
            "MDOS" => document.KnownChunks.Mdos.Count,
            "MDSF" => document.KnownChunks.Mdsf.Count,
            _ => 0,
        };
    }

    private static bool HasMeaningfulData(Pm4ResearchDocument document, string signature)
    {
        return signature switch
        {
            "MVER" => document.Version != 0,
            "MSHD" => document.KnownChunks.Mshd is not null,
            "MSLK" => document.KnownChunks.Mslk.Count > 0,
            "MSPV" => document.KnownChunks.Mspv.Count > 0,
            "MSPI" => document.KnownChunks.Mspi.Count > 0,
            "MSVT" => document.KnownChunks.Msvt.Count > 0,
            "MSVI" => document.KnownChunks.Msvi.Count > 0,
            "MSUR" => document.KnownChunks.Msur.Count > 0,
            "MSCN" => document.KnownChunks.Mscn.Count > 0,
            "MPRL" => document.KnownChunks.Mprl.Count > 0,
            "MPRR" => document.KnownChunks.Mprr.Count > 0,
            "MDBH" => (document.KnownChunks.Mdbh?.DestructibleBuildingCount ?? 0) > 0,
            "MDBI" => document.KnownChunks.Mdbi.Count > 0,
            "MDBF" => document.KnownChunks.Mdbf.Any(static entry => !string.IsNullOrWhiteSpace(entry.Filename)),
            "MDOS" => document.KnownChunks.Mdos.Any(static entry => entry.DestructibleBuildingIndex != 0 || entry.DestructionState != 0),
            "MDSF" => document.KnownChunks.Mdsf.Count > 0,
            _ => false,
        };
    }

    private static Pm4ReferenceAudit BuildMsviToMsvtAudit(Pm4ResearchDocument document)
    {
        int vertexCount = document.KnownChunks.Msvt.Count;
        return BuildValueAudit(
            "MSVI->MSVT",
            document.KnownChunks.Msvi,
            value => value < vertexCount,
            value => $"MSVI value {value} >= MSVT count {vertexCount}");
    }

    private static Pm4ReferenceAudit BuildMspiToMspvAudit(Pm4ResearchDocument document)
    {
        int vertexCount = document.KnownChunks.Mspv.Count;
        return BuildValueAudit(
            "MSPI->MSPV",
            document.KnownChunks.Mspi,
            value => value < vertexCount,
            value => $"MSPI value {value} >= MSPV count {vertexCount}");
    }

    private static Pm4ReferenceAudit BuildMsurToMsviAudit(Pm4ResearchDocument document)
    {
        int indexCount = document.KnownChunks.Msvi.Count;
        return BuildRecordAudit(
            "MSUR->MSVI window",
            document.KnownChunks.Msur,
            surface => surface.IndexCount == 0 || ((long)surface.MsviFirstIndex + surface.IndexCount) <= indexCount,
            surface => $"MSUR first={surface.MsviFirstIndex} count={surface.IndexCount} exceeds MSVI count {indexCount}");
    }

    private static Pm4ReferenceAudit BuildMslkRefIndexToMsurAudit(Pm4ResearchDocument document)
    {
        int surfaceCount = document.KnownChunks.Msur.Count;
        return BuildRecordAudit(
            "MSLK.RefIndex->MSUR",
            document.KnownChunks.Mslk,
            link => link.RefIndex < surfaceCount,
            link => $"MSLK refIndex {link.RefIndex} >= MSUR count {surfaceCount}");
    }

    private static Pm4ReferenceAudit BuildMslkMspiWindowAudit(Pm4ResearchDocument document)
    {
        int indexCount = document.KnownChunks.Mspi.Count;
        IReadOnlyList<Pm4MslkEntry> activeLinks = document.KnownChunks.Mslk.Where(static link => link.MspiIndexCount > 0).ToList();
        return BuildRecordAudit(
            "MSLK.MSPI window",
            activeLinks,
            link => link.MspiFirstIndex >= 0 && ((long)link.MspiFirstIndex + link.MspiIndexCount) <= indexCount,
            link => $"MSLK first={link.MspiFirstIndex} count={link.MspiIndexCount} exceeds MSPI count {indexCount}");
    }

    private static Pm4ReferenceAudit BuildMdsfToMsurAudit(Pm4ResearchDocument document)
    {
        int surfaceCount = document.KnownChunks.Msur.Count;
        return BuildRecordAudit(
            "MDSF.MSUR->MSUR",
            document.KnownChunks.Mdsf,
            entry => entry.MsurIndex < surfaceCount,
            entry => $"MDSF msurIndex {entry.MsurIndex} >= MSUR count {surfaceCount}");
    }

    private static Pm4ReferenceAudit BuildMdsfToMdosAudit(Pm4ResearchDocument document)
    {
        int mdosCount = document.KnownChunks.Mdos.Count;
        return BuildRecordAudit(
            "MDSF.MDOS->MDOS",
            document.KnownChunks.Mdsf,
            entry => entry.MdosIndex < mdosCount,
            entry => $"MDSF mdosIndex {entry.MdosIndex} >= MDOS count {mdosCount}");
    }

    private static Pm4ReferenceAudit BuildMdosToMdbhAudit(Pm4ResearchDocument document)
    {
        uint buildingCount = document.KnownChunks.Mdbh?.DestructibleBuildingCount ?? 0;
        if (buildingCount == 0)
            return new Pm4ReferenceAudit("MDOS.buildingIndex->MDBH", 0, 0, 0, Array.Empty<string>());

        return BuildRecordAudit(
            "MDOS.buildingIndex->MDBH",
            document.KnownChunks.Mdos,
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