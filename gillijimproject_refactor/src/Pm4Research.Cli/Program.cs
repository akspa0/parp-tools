using System.Text.Json;
using Pm4Research.Core;

namespace Pm4Research.Cli;

public static class Program
{
    private sealed record CompactHypothesis(
        string Family,
        int FamilyObjectIndex,
        uint Ck24,
        byte Ck24Type,
        ushort Ck24ObjectId,
        int SurfaceCount,
        int TotalIndexCount,
        int MdosCount,
        int GroupKeyCount,
        int MslkGroupObjectIdCount,
        int MslkRefIndexCount,
        Pm4Bounds3? Bounds,
        Pm4MprlFootprintSummary MprlFootprint);

    private sealed record CompactFamilySummary(
        string Family,
        int ObjectCount,
        int MaxSurfaceCount,
        int MaxIndexCount,
        int TotalLinkedMprlCount,
        int TotalLinkedInBoundsCount);

    private sealed record CompactHypothesisReport(
        string? SourcePath,
        int? TileX,
        int? TileY,
        uint Version,
        int Ck24GroupCount,
        int TotalHypothesisCount,
        IReadOnlyList<CompactFamilySummary> Families,
        IReadOnlyList<CompactHypothesis> Objects,
        IReadOnlyList<string> Diagnostics);

    public static int Main(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            ShowUsage();
            return 0;
        }

        string command = args[0].ToLowerInvariant();
        string[] tail = args.Skip(1).ToArray();

        try
        {
            return command switch
            {
                "inspect" => RunInspect(tail),
                "inspect-audit" => RunInspectAudit(tail),
                "inspect-mslk-refindex" => RunInspectMslkRefIndex(tail),
                "export-json" => RunExportJson(tail),
                "scan-dir" => RunScanDirectory(tail),
                "inspect-hypotheses" => RunInspectHypotheses(tail),
                "export-hypotheses" => RunExportHypotheses(tail),
                "scan-hypotheses" => RunScanHypotheses(tail),
                "scan-hypotheses-ndjson" => RunScanHypothesesNdjson(tail),
                "scan-audit" => RunScanAudit(tail),
                "scan-mslk-refindex" => RunScanMslkRefIndex(tail),
                "scan-msur-geometry" => RunScanMsurGeometry(tail),
                "scan-mslk-refindex-classifier" => RunScanMslkRefIndexClassifier(tail),
                "scan-structure-confidence" => RunScanStructureConfidence(tail),
                "scan-linkage" => RunScanLinkage(tail),
                "scan-mscn" => RunScanMscn(tail),
                "scan-unknowns" => RunScanUnknowns(tail),
                _ => RunInspect(args)
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunInspect(string[] args)
    {
        string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        if (string.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("Error: input PM4 file is required.");
            return 1;
        }

        Pm4ResearchFile file = Pm4ResearchReader.ReadFile(input);
        Pm4AnalysisReport report = Pm4ResearchAnalyzer.Analyze(file);

        PrintReport(report);
        return 0;
    }

    private static int RunExportJson(string[] args)
    {
        string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("Error: input PM4 file is required.");
            return 1;
        }

        Pm4AnalysisReport report = Pm4ResearchAnalyzer.Analyze(Pm4ResearchReader.ReadFile(input));
        string json = JsonSerializer.Serialize(report, JsonOptions);

        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, json);
        }
        else
        {
            Console.WriteLine(json);
        }

        return 0;
    }

    private static int RunInspectAudit(string[] args)
    {
        string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("Error: input PM4 file is required.");
            return 1;
        }

        Pm4DecodeAuditReport report = Pm4ResearchAuditAnalyzer.Analyze(Pm4ResearchReader.ReadFile(input));
        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, JsonSerializer.Serialize(report, JsonOptions));
        }
        else
        {
            PrintAuditReport(report);
        }

        return 0;
    }

    private static int RunScanDirectory(string[] args)
    {
        string? inputDir = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
        {
            Console.Error.WriteLine("Error: input directory is required and must exist.");
            return 1;
        }

        List<Pm4AnalysisReport> reports = Directory
            .EnumerateFiles(inputDir, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .Select(Pm4ResearchAnalyzer.Analyze)
            .ToList();

        string json = JsonSerializer.Serialize(reports, JsonOptions);
        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, json);
        }
        else
        {
            Console.WriteLine(json);
        }

        Console.Error.WriteLine($"Scanned {reports.Count} PM4 files from '{inputDir}'.");
        return 0;
    }

    private static int RunInspectMslkRefIndex(string[] args)
    {
        string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("Error: input PM4 file is required.");
            return 1;
        }

        Pm4MslkRefIndexFileAudit report = Pm4ResearchAuditAnalyzer.AnalyzeMslkRefIndexFile(Pm4ResearchReader.ReadFile(input));
        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, JsonSerializer.Serialize(report, JsonOptions));
        }
        else
        {
            PrintMslkRefIndexFileAudit(report);
        }

        return 0;
    }

    private static int RunInspectHypotheses(string[] args)
    {
        string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        if (string.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("Error: input PM4 file is required.");
            return 1;
        }

        Pm4TileObjectHypothesisReport report = Pm4ResearchObjectHypothesisGenerator.Analyze(Pm4ResearchReader.ReadFile(input));
        PrintHypothesisReport(report);
        return 0;
    }

    private static int RunExportHypotheses(string[] args)
    {
        string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("Error: input PM4 file is required.");
            return 1;
        }

        Pm4TileObjectHypothesisReport report = Pm4ResearchObjectHypothesisGenerator.Analyze(Pm4ResearchReader.ReadFile(input));
        string json = JsonSerializer.Serialize(report, JsonOptions);

        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, json);
        }
        else
        {
            Console.WriteLine(json);
        }

        return 0;
    }

    private static int RunScanHypotheses(string[] args)
    {
        string? inputDir = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
        {
            Console.Error.WriteLine("Error: input directory is required and must exist.");
            return 1;
        }

        IEnumerable<string> files = Directory
            .EnumerateFiles(inputDir, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName);

        int reportCount = 0;
        int totalHypotheses = 0;

        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            using FileStream stream = File.Create(output);
            using Utf8JsonWriter writer = new(stream, new JsonWriterOptions { Indented = true });
            writer.WriteStartArray();

            foreach (string filePath in files)
            {
                Pm4TileObjectHypothesisReport report = Pm4ResearchObjectHypothesisGenerator.Analyze(Pm4ResearchReader.ReadFile(filePath));
                JsonSerializer.Serialize(writer, ToCompactReport(report), JsonOptions);
                reportCount++;
                totalHypotheses += report.TotalHypothesisCount;
            }

            writer.WriteEndArray();
            writer.Flush();
        }
        else
        {
            List<Pm4TileObjectHypothesisReport> reports = files
                .Select(Pm4ResearchReader.ReadFile)
                .Select(Pm4ResearchObjectHypothesisGenerator.Analyze)
                .ToList();
            string json = JsonSerializer.Serialize(reports.Select(ToCompactReport).ToList(), JsonOptions);
            Console.WriteLine(json);
            reportCount = reports.Count;
            totalHypotheses = reports.Sum(static report => report.TotalHypothesisCount);
        }

        Console.Error.WriteLine($"Scanned {reportCount} PM4 files into {totalHypotheses} object hypotheses from '{inputDir}'.");
        return 0;
    }

    private static int RunScanHypothesesNdjson(string[] args)
    {
        string? inputDir = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
        {
            Console.Error.WriteLine("Error: input directory is required and must exist.");
            return 1;
        }

        IEnumerable<string> files = Directory
            .EnumerateFiles(inputDir, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName);

        int reportCount = 0;
        int totalHypotheses = 0;

        TextWriter writer;
        bool ownsWriter = false;
        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            writer = new StreamWriter(File.Create(output));
            ownsWriter = true;
        }
        else
        {
            writer = Console.Out;
        }

        try
        {
            foreach (string filePath in files)
            {
                Pm4TileObjectHypothesisReport report = Pm4ResearchObjectHypothesisGenerator.Analyze(Pm4ResearchReader.ReadFile(filePath));
                writer.WriteLine(JsonSerializer.Serialize(ToCompactReport(report), CompactJsonOptions));
                reportCount++;
                totalHypotheses += report.TotalHypothesisCount;
            }

            writer.Flush();
        }
        finally
        {
            if (ownsWriter)
                writer.Dispose();
        }

        Console.Error.WriteLine($"Streamed {reportCount} PM4 files into {totalHypotheses} object hypotheses from '{inputDir}'.");
        return 0;
    }

    private static int RunScanAudit(string[] args)
    {
        string? inputDir = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
        {
            Console.Error.WriteLine("Error: input directory is required and must exist.");
            return 1;
        }

        Pm4CorpusAuditReport report = Pm4ResearchAuditAnalyzer.AnalyzeDirectory(inputDir);
        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, JsonSerializer.Serialize(report, JsonOptions));
        }
        else
        {
            PrintCorpusAuditReport(report);
        }

        return 0;
    }

    private static int RunScanMslkRefIndex(string[] args)
    {
        string? inputDir = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
        {
            Console.Error.WriteLine("Error: input directory is required and must exist.");
            return 1;
        }

        Pm4MslkRefIndexCorpusAudit report = Pm4ResearchAuditAnalyzer.AnalyzeMslkRefIndexDirectory(inputDir);
        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, JsonSerializer.Serialize(report, JsonOptions));
        }
        else
        {
            PrintMslkRefIndexCorpusAudit(report);
        }

        return 0;
    }

    private static int RunScanUnknowns(string[] args)
    {
        string? inputDir = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
        {
            Console.Error.WriteLine("Error: input directory is required and must exist.");
            return 1;
        }

        Pm4UnknownsReport report = Pm4ResearchUnknownsAnalyzer.AnalyzeDirectory(inputDir);
        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, JsonSerializer.Serialize(report, JsonOptions));
        }
        else
        {
            PrintUnknownsReport(report);
        }

        return 0;
    }

    private static int RunScanMsurGeometry(string[] args)
    {
        string? inputDir = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
        {
            Console.Error.WriteLine("Error: input directory is required and must exist.");
            return 1;
        }

        Pm4MsurGeometryReport report = Pm4ResearchMsurGeometryAnalyzer.AnalyzeDirectory(inputDir);
        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, JsonSerializer.Serialize(report, JsonOptions));
        }
        else
        {
            PrintMsurGeometryReport(report);
        }

        return 0;
    }

    private static int RunScanMslkRefIndexClassifier(string[] args)
    {
        string? inputDir = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
        {
            Console.Error.WriteLine("Error: input directory is required and must exist.");
            return 1;
        }

        Pm4RefIndexClassifierReport report = Pm4ResearchMslkRefIndexClassifier.AnalyzeDirectory(inputDir);
        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, JsonSerializer.Serialize(report, JsonOptions));
        }
        else
        {
            PrintMslkRefIndexClassifierReport(report);
        }

        return 0;
    }

    private static int RunScanMscn(string[] args)
    {
        string? inputDir = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
        {
            Console.Error.WriteLine("Error: input directory is required and must exist.");
            return 1;
        }

        Pm4MscnRelationshipReport report = Pm4ResearchMscnAnalyzer.AnalyzeDirectory(inputDir);
        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, JsonSerializer.Serialize(report, JsonOptions));
        }
        else
        {
            PrintMscnRelationshipReport(report);
        }

        return 0;
    }

    private static int RunScanStructureConfidence(string[] args)
    {
        string? inputDir = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
        {
            Console.Error.WriteLine("Error: input directory is required and must exist.");
            return 1;
        }

        Pm4StructureConfidenceReport report = Pm4ResearchStructureConfidenceAnalyzer.AnalyzeDirectory(inputDir);
        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, JsonSerializer.Serialize(report, JsonOptions));
        }
        else
        {
            PrintStructureConfidenceReport(report);
        }

        return 0;
    }

    private static int RunScanLinkage(string[] args)
    {
        string? inputDir = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(inputDir) || !Directory.Exists(inputDir))
        {
            Console.Error.WriteLine("Error: input directory is required and must exist.");
            return 1;
        }

        Pm4LinkageReport report = Pm4ResearchLinkageAnalyzer.AnalyzeDirectory(inputDir);
        if (!string.IsNullOrWhiteSpace(output))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            File.WriteAllText(output, JsonSerializer.Serialize(report, JsonOptions));
        }
        else
        {
            PrintLinkageReport(report);
        }

        return 0;
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    private static readonly JsonSerializerOptions CompactJsonOptions = new();

    private static string? GetOption(string[] args, params string[] names)
    {
        for (int i = 0; i < args.Length; i++)
        {
            if (!names.Contains(args[i], StringComparer.OrdinalIgnoreCase))
                continue;
            if (i + 1 < args.Length)
                return args[i + 1];
        }

        return null;
    }

    private static void PrintReport(Pm4AnalysisReport report)
    {
        Console.WriteLine("PM4 Research Report");
        Console.WriteLine("===================");
        Console.WriteLine($"Source:   {report.SourcePath}");
        Console.WriteLine($"Version:  {report.Version}");
        Console.WriteLine($"Chunks:   {report.ChunkOrder.Count}");
        Console.WriteLine();

        Console.WriteLine("Chunk Order");
        foreach (Pm4ChunkSummary chunk in report.ChunkOrder)
            Console.WriteLine($"  {chunk.Signature} size={chunk.Size}");
        Console.WriteLine();

        PrintVectorSummary(report.Mspv);
        PrintVectorSummary(report.Msvt);
        PrintVectorSummary(report.Mscn);
        PrintVectorSummary(report.MprlPositions);

        Console.WriteLine("MPRL Summary");
        Console.WriteLine($"  total={report.Mprl.TotalCount} normal={report.Mprl.NormalCount} terminator={report.Mprl.TerminatorCount}");
        Console.WriteLine($"  floors={report.Mprl.FloorMin}..{report.Mprl.FloorMax} rawRotationDeg={report.Mprl.RotationMinDegrees:F3}..{report.Mprl.RotationMaxDegrees:F3}");
        Console.WriteLine();

        Console.WriteLine("Top CK24 Groups");
        foreach (Pm4Ck24Summary summary in report.TopCk24Groups.Take(12))
        {
            Console.WriteLine($"  ck24=0x{summary.Ck24:X6} type=0x{summary.Ck24Type:X2} obj={summary.Ck24ObjectId} surfaces={summary.SurfaceCount} indices={summary.TotalIndexCount} avgH={summary.AverageHeight:F3} mdos={summary.DistinctMdosCount}");
        }
        Console.WriteLine();

        if (report.UnrecognizedChunks.Count > 0)
        {
            Console.WriteLine("Unrecognized Chunks");
            foreach (string chunk in report.UnrecognizedChunks)
                Console.WriteLine($"  {chunk}");
            Console.WriteLine();
        }

        if (report.Diagnostics.Count > 0)
        {
            Console.WriteLine("Diagnostics");
            foreach (string diagnostic in report.Diagnostics)
                Console.WriteLine($"  {diagnostic}");
        }
    }

    private static void PrintVectorSummary(Pm4VectorSetSummary summary)
    {
        Console.WriteLine(summary.Name + " Summary");
        Console.WriteLine($"  count={summary.Count}");
        if (summary.Bounds != null)
        {
            Console.WriteLine($"  min=({summary.Bounds.Min.X:F3}, {summary.Bounds.Min.Y:F3}, {summary.Bounds.Min.Z:F3})");
            Console.WriteLine($"  max=({summary.Bounds.Max.X:F3}, {summary.Bounds.Max.Y:F3}, {summary.Bounds.Max.Z:F3})");
            Console.WriteLine($"  span=({summary.Bounds.Span.X:F3}, {summary.Bounds.Span.Y:F3}, {summary.Bounds.Span.Z:F3})");
        }

        if (summary.Centroid.HasValue)
            Console.WriteLine($"  centroid=({summary.Centroid.Value.X:F3}, {summary.Centroid.Value.Y:F3}, {summary.Centroid.Value.Z:F3})");

        foreach (Pm4QuadrantSummary quadrant in summary.Quadrants)
        {
            Console.WriteLine($"  {quadrant.Plane} mid=({quadrant.MidA:F3}, {quadrant.MidB:F3}) ll={quadrant.LowLow} lh={quadrant.LowHigh} hl={quadrant.HighLow} hh={quadrant.HighHigh}");
        }

        Console.WriteLine();
    }

    private static void PrintAuditReport(Pm4DecodeAuditReport report)
    {
        Console.WriteLine("PM4 Decode Audit Report");
        Console.WriteLine("=======================");
        Console.WriteLine($"Source:            {report.SourcePath}");
        Console.WriteLine($"Version:           {report.Version}");
        Console.WriteLine($"Chunks:            {report.ChunkCount}");
        Console.WriteLine($"Recognized chunks: {report.RecognizedChunkCount}");
        Console.WriteLine($"Unknown chunks:    {report.UnknownChunkCount}");
        Console.WriteLine($"Trailing bytes:    {report.HasTrailingBytesDiagnostic}");
        Console.WriteLine($"Overrun warning:   {report.HasOverrunDiagnostic}");
        Console.WriteLine();

        Console.WriteLine("Chunk Audit");
        foreach (Pm4ChunkDecodeAudit chunk in report.ChunkAudits)
        {
            Console.WriteLine($"  {chunk.Signature} chunks={chunk.ChunkCount} entries={chunk.EntryCount} hasData={chunk.HasMeaningfulData} totalBytes={chunk.TotalBytes} sizes={string.Join(",", chunk.ExampleSizes)} strideRemainders={chunk.StrideRemainderCount}");
        }

        Console.WriteLine();
        Console.WriteLine("Reference Audit");
        foreach (Pm4ReferenceAudit audit in report.ReferenceAudits)
        {
            Console.WriteLine($"  {audit.Name} total={audit.TotalCount} valid={audit.ValidCount} invalid={audit.InvalidCount}");
            foreach (string example in audit.Examples.Take(3))
                Console.WriteLine($"    {example}");
        }

        if (report.UnknownChunkSignatures.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Unknown Chunk Signatures");
            foreach (string signature in report.UnknownChunkSignatures)
                Console.WriteLine($"  {signature}");
        }

        if (report.Diagnostics.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Diagnostics");
            foreach (string diagnostic in report.Diagnostics)
                Console.WriteLine($"  {diagnostic}");
        }
    }

    private static void PrintCorpusAuditReport(Pm4CorpusAuditReport report)
    {
        Console.WriteLine("PM4 Corpus Decode Audit");
        Console.WriteLine("=======================");
        Console.WriteLine($"Input:                  {report.InputDirectory}");
        Console.WriteLine($"Files:                  {report.FileCount}");
        Console.WriteLine($"Files with diagnostics: {report.FilesWithDiagnostics}");
        Console.WriteLine($"Files with unknowns:    {report.FilesWithUnknownChunks}");
        Console.WriteLine();

        Console.WriteLine("Common Chunk Audit");
        foreach (Pm4CorpusChunkAudit chunk in report.ChunkAudits.Where(static chunk => chunk.IsCommon))
        {
            Console.WriteLine($"  {chunk.Signature} files={chunk.FileCount} dataFiles={chunk.DataFileCount} entries={chunk.TotalEntryCount} chunks={chunk.TotalChunkCount} bytes={chunk.TotalBytes} sizes={string.Join(",", chunk.ExampleSizes)} remainderFiles={chunk.FilesWithStrideRemainders}");
        }

        IReadOnlyList<Pm4CorpusChunkAudit> rareChunks = report.ChunkAudits.Where(static chunk => !chunk.IsCommon).ToList();
        if (rareChunks.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Rare Chunk Audit");
            foreach (Pm4CorpusChunkAudit chunk in rareChunks)
            {
                Console.WriteLine($"  {chunk.Signature} files={chunk.FileCount} dataFiles={chunk.DataFileCount} entries={chunk.TotalEntryCount} chunks={chunk.TotalChunkCount} bytes={chunk.TotalBytes} sizes={string.Join(",", chunk.ExampleSizes)} remainderFiles={chunk.FilesWithStrideRemainders}");
            }
        }

        Console.WriteLine();
        Console.WriteLine("Reference Audit");
        foreach (Pm4CorpusReferenceAudit audit in report.ReferenceAudits)
        {
            Console.WriteLine($"  {audit.Name} total={audit.TotalCount} invalid={audit.InvalidCount}");
            foreach (string example in audit.ExampleFailures.Take(3))
                Console.WriteLine($"    {example}");
        }

        if (report.UnknownChunkSignatures.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Unknown Chunk Signatures");
            foreach (string signature in report.UnknownChunkSignatures)
                Console.WriteLine($"  {signature}");
        }

        if (report.TopDiagnostics.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Top Diagnostics");
            foreach (string diagnostic in report.TopDiagnostics)
                Console.WriteLine($"  {diagnostic}");
        }
    }

    private static void PrintMslkRefIndexFileAudit(Pm4MslkRefIndexFileAudit report)
    {
        Console.WriteLine("PM4 MSLK RefIndex Audit");
        Console.WriteLine("=======================");
        Console.WriteLine($"Source:           {report.SourcePath}");
        Console.WriteLine($"Tile:             {report.TileX}_{report.TileY}");
        Console.WriteLine($"Version:          {report.Version}");
        Console.WriteLine($"MSLK count:       {report.MslkCount}");
        Console.WriteLine($"MSUR count:       {report.MsurCount}");
        Console.WriteLine($"Invalid refIndex: {report.InvalidRefIndexCount}");

        if (report.Mismatches.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Example Mismatches");
            foreach (Pm4MslkRefIndexMismatch mismatch in report.Mismatches.Take(16))
            {
                string fits = string.Join(", ", mismatch.DomainFits.Where(static fit => fit.Fits).Select(static fit => fit.Domain));
                Console.WriteLine($"  mslk[{mismatch.MslkIndex}] ref={mismatch.RefIndex} group={mismatch.GroupObjectId} link={mismatch.LinkId} flags=0x{mismatch.TypeFlags:X2} subtype={mismatch.Subtype} mspi=({mismatch.MspiFirstIndex},{mismatch.MspiIndexCount}) fits=[{fits}]");
            }
        }
    }

    private static void PrintMslkRefIndexCorpusAudit(Pm4MslkRefIndexCorpusAudit report)
    {
        Console.WriteLine("PM4 MSLK RefIndex Corpus Audit");
        Console.WriteLine("==============================");
        Console.WriteLine($"Input:                 {report.InputDirectory}");
        Console.WriteLine($"Files:                 {report.FileCount}");
        Console.WriteLine($"Files with mismatches: {report.FilesWithMismatches}");
        Console.WriteLine($"Total mismatches:      {report.TotalMismatchCount}");

        Console.WriteLine();
        Console.WriteLine("Candidate Domains");
        foreach (Pm4MslkRefIndexDomainSummary summary in report.DomainSummaries)
        {
            Console.WriteLine($"  {summary.Domain} fits={summary.MatchingMismatchCount} nonFits={summary.NonMatchingMismatchCount}");
        }

        if (report.TopFiles.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Top Mismatch Tiles");
            foreach (Pm4MslkRefIndexFileAudit file in report.TopFiles)
            {
                Console.WriteLine($"  tile={file.TileX}_{file.TileY} invalid={file.InvalidRefIndexCount} mslk={file.MslkCount} msur={file.MsurCount} source={file.SourcePath}");
                foreach (Pm4MslkRefIndexMismatch mismatch in file.Mismatches.Take(4))
                {
                    string fits = string.Join(", ", mismatch.DomainFits.Where(static fit => fit.Fits).Select(static fit => fit.Domain));
                    Console.WriteLine($"    mslk[{mismatch.MslkIndex}] ref={mismatch.RefIndex} group={mismatch.GroupObjectId} link={mismatch.LinkId} fits=[{fits}]");
                }
            }
        }
    }

    private static void PrintUnknownsReport(Pm4UnknownsReport report)
    {
        Console.WriteLine("PM4 Unknowns Report");
        Console.WriteLine("===================");
        Console.WriteLine($"Input:           {report.InputDirectory}");
        Console.WriteLine($"Files:           {report.FileCount}");
        Console.WriteLine($"Non-empty files: {report.NonEmptyFileCount}");

        Console.WriteLine();
        Console.WriteLine("Chunk Population");
        foreach (Pm4CorpusChunkAudit chunk in report.ChunkPopulation)
        {
            Console.WriteLine($"  {chunk.Signature} files={chunk.FileCount} dataFiles={chunk.DataFileCount} entries={chunk.TotalEntryCount} sizes={string.Join(",", chunk.ExampleSizes)}");
        }

        Console.WriteLine();
        Console.WriteLine("Relationships");
        foreach (Pm4RelationshipEdgeSummary relation in report.Relationships)
        {
            Console.WriteLine($"  [{relation.Status}] {relation.Edge}: fits={relation.Fits} misses={relation.Misses}");
            Console.WriteLine($"    evidence: {relation.Evidence}");
            Console.WriteLine($"    next:     {relation.NextStep}");
        }

        Console.WriteLine();
        Console.WriteLine("MSPI Count Interpretation");
        Console.WriteLine($"  active={report.MspiInterpretation.ActiveLinkCount} indicesOnly={report.MspiInterpretation.IndicesModeOnlyCount} trianglesOnly={report.MspiInterpretation.TrianglesModeOnlyCount} both={report.MspiInterpretation.BothModesCount} neither={report.MspiInterpretation.NeitherModeCount}");

        Console.WriteLine();
        Console.WriteLine("LinkId Patterns");
        Console.WriteLine($"  total={report.LinkIdPatterns.TotalCount} sentinelTile={report.LinkIdPatterns.SentinelTileLinkCount} zero={report.LinkIdPatterns.ZeroCount} other={report.LinkIdPatterns.OtherCount}");
        Console.WriteLine("  top decoded tiles:");
        foreach (Pm4ValueFrequency value in report.LinkIdPatterns.TopDecodedTiles.Take(8))
            Console.WriteLine($"    {value.Value} -> {value.Count}");
        Console.WriteLine("  top other values:");
        foreach (Pm4ValueFrequency value in report.LinkIdPatterns.TopOtherValues.Take(8))
            Console.WriteLine($"    {value.Value} -> {value.Count}");

        Console.WriteLine();
        Console.WriteLine("Field Distributions");
        foreach (Pm4FieldDistribution distribution in report.FieldDistributions)
        {
            Console.WriteLine($"  {distribution.Field}: total={distribution.TotalCount} distinct={distribution.DistinctCount}{(string.IsNullOrWhiteSpace(distribution.Range) ? string.Empty : $" range={distribution.Range}")}");
            foreach (Pm4ValueFrequency value in distribution.TopValues.Take(6))
                Console.WriteLine($"    {value.Value} -> {value.Count}");
            if (!string.IsNullOrWhiteSpace(distribution.Notes))
                Console.WriteLine($"    note: {distribution.Notes}");
        }

        Console.WriteLine();
        Console.WriteLine("Unknowns");
        foreach (Pm4UnknownFinding finding in report.Unknowns)
        {
            Console.WriteLine($"  [{finding.Status}] {finding.Name}");
            Console.WriteLine($"    evidence: {finding.Evidence}");
            Console.WriteLine($"    next:     {finding.NextStep}");
        }

        if (report.Notes.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Notes");
            foreach (string note in report.Notes)
                Console.WriteLine($"  {note}");
        }
    }

    private static void PrintMsurGeometryReport(Pm4MsurGeometryReport report)
    {
        Console.WriteLine("PM4 MSUR Geometry Report");
        Console.WriteLine("========================");
        Console.WriteLine($"Input:                   {report.InputDirectory}");
        Console.WriteLine($"Files:                   {report.FileCount}");
        Console.WriteLine($"Analyzed surfaces:       {report.AnalyzedSurfaceCount}");
        Console.WriteLine($"Degenerate surfaces:     {report.DegenerateSurfaceCount}");
        Console.WriteLine($"Unit-like stored normal: {report.UnitLikeStoredNormalCount}");
        Console.WriteLine($"Strong alignment:        {report.StrongAlignmentCount}");
        Console.WriteLine($"Moderate alignment:      {report.ModerateAlignmentCount}");
        Console.WriteLine($"Weak alignment:          {report.WeakAlignmentCount}");
        Console.WriteLine($"Positive signed dot:     {report.PositiveAlignmentCount}");
        Console.WriteLine($"Negative signed dot:     {report.NegativeAlignmentCount}");
        Console.WriteLine($"Avg normal magnitude:    {report.AverageStoredNormalMagnitude:F4}");
        Console.WriteLine($"Avg |dot|:               {report.AverageAbsoluteDot:F4}");

        Console.WriteLine();
        Console.WriteLine("Height Candidates");
        foreach (Pm4HeightCandidateSummary candidate in report.HeightCandidates)
        {
            Console.WriteLine($"  {candidate.Candidate} meanAbsErr={candidate.MeanAbsoluteError:F4} fits<=0.1={candidate.FitsWithinPointOne} fits<=1={candidate.FitsWithinOne} fits<=4={candidate.FitsWithinFour}");
        }

        Console.WriteLine();
        Console.WriteLine("Distributions");
        foreach (Pm4FieldDistribution distribution in report.Distributions)
        {
            Console.WriteLine($"  {distribution.Field}: total={distribution.TotalCount} distinct={distribution.DistinctCount}");
            foreach (Pm4ValueFrequency value in distribution.TopValues.Take(6))
                Console.WriteLine($"    {value.Value} -> {value.Count}");
            if (!string.IsNullOrWhiteSpace(distribution.Notes))
                Console.WriteLine($"    note: {distribution.Notes}");
        }

        if (report.BestAlignedExamples.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Best-Aligned Examples");
            foreach (Pm4MsurGeometryExample example in report.BestAlignedExamples.Take(12))
            {
                Console.WriteLine($"  tile={example.TileX}_{example.TileY} surface={example.SurfaceIndex} ck24=0x{example.Ck24:X6} type=0x{example.Ck24Type:X2} obj={example.Ck24ObjectId} idxCount={example.IndexCount} |n|={example.StoredNormalMagnitude:F4} dot={example.SignedDot:F4} |dot|={example.AbsoluteDot:F4} h={example.Height:F4} geomPlane={example.GeometricPlaneDistance:F4} storedPlane={example.StoredPlaneDistance:F4}");
            }
        }

        if (report.WorstAlignedExamples.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Worst-Aligned Examples");
            foreach (Pm4MsurGeometryExample example in report.WorstAlignedExamples.Take(12))
            {
                Console.WriteLine($"  tile={example.TileX}_{example.TileY} surface={example.SurfaceIndex} ck24=0x{example.Ck24:X6} type=0x{example.Ck24Type:X2} obj={example.Ck24ObjectId} idxCount={example.IndexCount} |n|={example.StoredNormalMagnitude:F4} dot={example.SignedDot:F4} |dot|={example.AbsoluteDot:F4} h={example.Height:F4} geomPlane={example.GeometricPlaneDistance:F4} storedPlane={example.StoredPlaneDistance:F4}");
            }
        }

        if (report.Notes.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Notes");
            foreach (string note in report.Notes)
                Console.WriteLine($"  {note}");
        }
    }

    private static void PrintMslkRefIndexClassifierReport(Pm4RefIndexClassifierReport report)
    {
        Console.WriteLine("PM4 MSLK RefIndex Classifier");
        Console.WriteLine("============================");
        Console.WriteLine($"Input:                  {report.InputDirectory}");
        Console.WriteLine($"Files:                  {report.FileCount}");
        Console.WriteLine($"Files with mismatches:  {report.FilesWithMismatches}");
        Console.WriteLine($"Total mismatches:       {report.TotalMismatchCount}");
        Console.WriteLine($"Resolved families:      {report.Summary.ResolvedFamilyCount}");
        Console.WriteLine($"Ambiguous families:     {report.Summary.AmbiguousFamilyCount}");
        Console.WriteLine($"Resolved mismatch rows: {report.Summary.ResolvedEntryCount}");

        Console.WriteLine();
        Console.WriteLine("Domain Baselines");
        foreach (Pm4RefIndexDomainBaseline baseline in report.DomainBaselines)
        {
            Console.WriteLine($"  {baseline.Domain} fits={baseline.FitCount} coverage={baseline.Coverage:P1}");
        }

        Console.WriteLine();
        Console.WriteLine("Classification Summary");
        foreach (Pm4ValueFrequency classification in report.Summary.ClassificationCounts)
            Console.WriteLine($"  {classification.Value} -> {classification.Count}");

        if (report.TopFamilies.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Top Families");
            foreach (Pm4RefIndexClassifiedFamily family in report.TopFamilies.Take(20))
            {
                string scores = string.Join(", ", family.DomainScores.Select(static score => $"{score.Domain}:{score.Coverage:P0} Δ={score.CoverageDelta:+0.00;-0.00;0.00}"));
                string refs = string.Join(", ", family.TopRefIndices.Select(static value => $"{value.Value}={value.Count}"));
                Console.WriteLine($"  {family.FamilyKey} class={family.Classification} confidence={family.Confidence} files={family.FileCount} entries={family.EntryCount} refRange={family.MinRefIndex}..{family.MaxRefIndex}");
                Console.WriteLine($"    scores: {scores}");
                Console.WriteLine($"    refs:   {refs}");
            }
        }

        if (report.Notes.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Notes");
            foreach (string note in report.Notes)
                Console.WriteLine($"  {note}");
        }
    }

    private static void PrintMscnRelationshipReport(Pm4MscnRelationshipReport report)
    {
        Console.WriteLine("PM4 MSCN Relationship Report");
        Console.WriteLine("============================");
        Console.WriteLine($"Input:                 {report.InputDirectory}");
        Console.WriteLine($"Files:                 {report.FileCount}");
        Console.WriteLine($"Files with MSCN:       {report.FilesWithMscn}");
        Console.WriteLine($"Files with tile coords:{report.FilesWithTileCoordinates}");
        Console.WriteLine($"Total MSCN points:     {report.TotalMscnPointCount}");

        Console.WriteLine();
        Console.WriteLine("Relationships");
        foreach (Pm4RelationshipEdgeSummary relation in report.Relationships)
        {
            Console.WriteLine($"  [{relation.Status}] {relation.Edge}: fits={relation.Fits} misses={relation.Misses}");
            Console.WriteLine($"    evidence: {relation.Evidence}");
            Console.WriteLine($"    next:     {relation.NextStep}");
        }

        Console.WriteLine();
        Console.WriteLine("Coordinate Space");
        Console.WriteLine($"  points: swappedWorld={report.CoordinateSpace.SwappedWorldTileFitCount} rawWorld={report.CoordinateSpace.RawWorldTileFitCount} ambiguousWorld={report.CoordinateSpace.AmbiguousWorldTileFitCount} tileLocal={report.CoordinateSpace.TileLocalLikeCount} neither={report.CoordinateSpace.NeitherFitCount}");
        Console.WriteLine($"  files: swappedDominant={report.CoordinateSpace.FilesSwappedDominant} rawDominant={report.CoordinateSpace.FilesRawDominant} tileLocalDominant={report.CoordinateSpace.FilesTileLocalDominant} noDominant={report.CoordinateSpace.FilesNoDominant}");

        Console.WriteLine();
        Console.WriteLine("Cluster Distributions");
        foreach (Pm4FieldDistribution distribution in report.ClusterDistributions)
        {
            Console.WriteLine($"  {distribution.Field}: total={distribution.TotalCount} distinct={distribution.DistinctCount}");
            foreach (Pm4ValueFrequency value in distribution.TopValues.Take(6))
                Console.WriteLine($"    {value.Value} -> {value.Count}");
            if (!string.IsNullOrWhiteSpace(distribution.Notes))
                Console.WriteLine($"    note: {distribution.Notes}");
        }

        if (report.TopNonZeroClusters.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Top Non-Zero CK24 MSCN Clusters");
            foreach (Pm4MscnClusterExample cluster in report.TopNonZeroClusters)
            {
                Console.WriteLine($"  tile={cluster.TileX}_{cluster.TileY} ck24=0x{cluster.Ck24:X6} type=0x{cluster.Ck24Type:X2} obj={cluster.Ck24ObjectId} surfaces={cluster.SurfaceCount} mdosRefs={cluster.ValidMdosRefCount} distinctMdos={cluster.DistinctMdosCount} invalidMdos={cluster.InvalidMdosRefCount} meshVerts={cluster.MeshVertexCount} align={cluster.AlignmentMode}");
            }
        }

        if (report.TopZeroClusters.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Top CK24=0 MSCN Clusters");
            foreach (Pm4MscnClusterExample cluster in report.TopZeroClusters)
            {
                Console.WriteLine($"  tile={cluster.TileX}_{cluster.TileY} ck24=0x{cluster.Ck24:X6} type=0x{cluster.Ck24Type:X2} obj={cluster.Ck24ObjectId} surfaces={cluster.SurfaceCount} mdosRefs={cluster.ValidMdosRefCount} distinctMdos={cluster.DistinctMdosCount} invalidMdos={cluster.InvalidMdosRefCount} meshVerts={cluster.MeshVertexCount} align={cluster.AlignmentMode}");
            }
        }

        if (report.TopInvalidMdosClusters.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Top Invalid-Mdos CK24 Clusters");
            foreach (Pm4MscnClusterExample cluster in report.TopInvalidMdosClusters)
            {
                Console.WriteLine($"  tile={cluster.TileX}_{cluster.TileY} ck24=0x{cluster.Ck24:X6} type=0x{cluster.Ck24Type:X2} obj={cluster.Ck24ObjectId} surfaces={cluster.SurfaceCount} mdosRefs={cluster.ValidMdosRefCount} distinctMdos={cluster.DistinctMdosCount} invalidMdos={cluster.InvalidMdosRefCount} meshVerts={cluster.MeshVertexCount} align={cluster.AlignmentMode}");
            }
        }

        if (report.Notes.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Notes");
            foreach (string note in report.Notes)
                Console.WriteLine($"  {note}");
        }
    }

    private static void PrintLinkageReport(Pm4LinkageReport report)
    {
        Console.WriteLine("PM4 Linkage Report");
        Console.WriteLine("==================");
        Console.WriteLine($"Input:                     {report.InputDirectory}");
        Console.WriteLine($"Files:                     {report.FileCount}");
        Console.WriteLine($"Files with ref mismatches: {report.FilesWithRefIndexMismatches}");
        Console.WriteLine($"Files with bad MdosIndex:  {report.FilesWithBadMdos}");
        Console.WriteLine($"Total ref mismatches:      {report.TotalRefIndexMismatchCount}");

        Console.WriteLine();
        Console.WriteLine("Relationships");
        foreach (Pm4RelationshipEdgeSummary relation in report.Relationships)
        {
            Console.WriteLine($"  [{relation.Status}] {relation.Edge}: fits={relation.Fits} misses={relation.Misses}");
            Console.WriteLine($"    evidence: {relation.Evidence}");
            Console.WriteLine($"    next:     {relation.NextStep}");
        }

        Console.WriteLine();
        Console.WriteLine("Identity Summary");
        Console.WriteLine($"  distinctCK24={report.IdentitySummary.DistinctCk24Count} distinctObjectId={report.IdentitySummary.DistinctCk24ObjectIdCount} analyzedObjectIdGroups={report.IdentitySummary.ObjectIdGroupsAnalyzed}");
        Console.WriteLine($"  reusedObjectIdGroups={report.IdentitySummary.ReusedObjectIdGroupCount} reusedAcrossTypes={report.IdentitySummary.ReusedAcrossTypeGroupCount}");

        Console.WriteLine();
        Console.WriteLine("Distributions");
        foreach (Pm4FieldDistribution distribution in report.Distributions)
        {
            Console.WriteLine($"  {distribution.Field}: total={distribution.TotalCount} distinct={distribution.DistinctCount}");
            foreach (Pm4ValueFrequency value in distribution.TopValues.Take(6))
                Console.WriteLine($"    {value.Value} -> {value.Count}");
            if (!string.IsNullOrWhiteSpace(distribution.Notes))
                Console.WriteLine($"    note: {distribution.Notes}");
        }

        if (report.IdentitySummary.TopReuseCases.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Top CK24 ObjectId Reuse Cases");
            foreach (Pm4Ck24ObjectIdReuseCase reuse in report.IdentitySummary.TopReuseCases)
            {
                string topCk24 = string.Join(", ", reuse.TopCk24Values.Select(static value => $"{value.Value}->{value.Count}"));
                Console.WriteLine($"  tile={reuse.TileX}_{reuse.TileY} obj={reuse.Ck24ObjectId} ck24Count={reuse.DistinctCk24Count} typeCount={reuse.DistinctTypeCount} surfaces={reuse.SurfaceCount} top=[{topCk24}]");
            }
        }

        if (report.TopMismatchFamilies.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Top RefIndex Mismatch Families");
            foreach (Pm4LinkageMismatchFamily family in report.TopMismatchFamilies.Take(16))
            {
                string domains = string.Join(", ", family.CandidateDomains.Select(static value => $"{value.Value}={value.Count}"));
                string low16 = string.Join(", ", family.TopLow16ObjectIds.Select(static value => $"{value.Value}={value.Count}"));
                Console.WriteLine($"  {family.FamilyKey} entries={family.EntryCount} files={family.FileCount} groupIds={family.DistinctGroupObjectIdCount} low16Ids={family.DistinctLow16ObjectIdCount} refs={family.DistinctRefIndexCount} low16Matches={family.MatchingCk24ObjectIdEntryCount} low24Matches={family.MatchingFullCk24EntryCount} withMscn={family.EntriesInFilesWithMscn} withBadMdos={family.EntriesInFilesWithBadMdos}");
                Console.WriteLine($"    domains: {domains}");
                Console.WriteLine($"    low16:   {low16}");
            }
        }

        if (report.TopBadMdosClusters.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Top Bad-Mdos CK24 Clusters");
            foreach (Pm4BadMdosCluster cluster in report.TopBadMdosClusters.Take(16))
            {
                Console.WriteLine($"  tile={cluster.TileX}_{cluster.TileY} ck24=0x{cluster.Ck24:X6} type=0x{cluster.Ck24Type:X2} obj={cluster.Ck24ObjectId} surfaces={cluster.SurfaceCount} invalidMdos={cluster.InvalidMdosCount} validMdos={cluster.ValidMdosCount} distinctInvalid={cluster.DistinctInvalidMdosCount} distinctValid={cluster.DistinctValidMdosCount} meshVerts={cluster.MeshVertexCount}");
            }
        }

        if (report.Notes.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Notes");
            foreach (string note in report.Notes)
                Console.WriteLine($"  {note}");
        }
    }

    private static void PrintStructureConfidenceReport(Pm4StructureConfidenceReport report)
    {
        Console.WriteLine("PM4 Structure Confidence Report");
        Console.WriteLine("===============================");
        Console.WriteLine($"Input:                 {report.InputDirectory}");
        Console.WriteLine($"Files:                 {report.FileCount}");
        Console.WriteLine($"High-layout chunks:    {report.Summary.HighLayoutChunkCount}");
        Console.WriteLine($"Medium-layout chunks:  {report.Summary.MediumLayoutChunkCount}");
        Console.WriteLine($"Low-layout chunks:     {report.Summary.LowLayoutChunkCount}");
        Console.WriteLine($"High-semantic fields:  {report.Summary.HighSemanticFieldCount}");
        Console.WriteLine($"Medium-semantic fields:{report.Summary.MediumSemanticFieldCount}");
        Console.WriteLine($"Low-semantic fields:   {report.Summary.LowSemanticFieldCount}");
        Console.WriteLine($"Very-low semantic:     {report.Summary.VeryLowSemanticFieldCount}");
        Console.WriteLine($"Source conflicts:      {report.Summary.ConflictCount}");

        Console.WriteLine();
        Console.WriteLine("Chunk Confidence");
        foreach (Pm4ChunkStructureConfidence chunk in report.ChunkConfidence)
        {
            Console.WriteLine($"  {chunk.Signature} stride={chunk.Stride} files={chunk.FileCount} dataFiles={chunk.DataFileCount} entries={chunk.TotalEntryCount} remainders={chunk.FilesWithStrideRemainders} layout={chunk.LayoutConfidence} semantics={chunk.SemanticConfidence} risk={chunk.HallucinationRisk}");
            Console.WriteLine($"    evidence: {chunk.Evidence}");
            Console.WriteLine($"    next:     {chunk.NextStep}");
        }

        Console.WriteLine();
        Console.WriteLine("Field Confidence");
        foreach (Pm4FieldConfidenceFinding field in report.FieldConfidence)
        {
            Console.WriteLine($"  {field.Field} class={field.Classification} layout={field.LayoutConfidence} semantics={field.SemanticConfidence} risk={field.HallucinationRisk}");
            Console.WriteLine($"    evidence: {field.Evidence}");
            Console.WriteLine($"    next:     {field.NextStep}");
        }

        Console.WriteLine();
        Console.WriteLine("Conflicts");
        foreach (Pm4StructureConflict conflict in report.Conflicts)
        {
            Console.WriteLine($"  {conflict.Field}");
            Console.WriteLine($"    legacy:   {conflict.LegacyClaim}");
            Console.WriteLine($"    current:  {conflict.CurrentDecode}");
            Console.WriteLine($"    conflict: {conflict.Conflict}");
            Console.WriteLine($"    evidence: {conflict.Evidence}");
            Console.WriteLine($"    next:     {conflict.NextStep}");
        }

        if (report.Notes.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Notes");
            foreach (string note in report.Notes)
                Console.WriteLine($"  {note}");
        }
    }

    private static void ShowUsage()
    {
        Console.WriteLine("Pm4Research CLI");
        Console.WriteLine("Usage:");
        Console.WriteLine("  pm4research inspect --input <file.pm4>");
        Console.WriteLine("  pm4research inspect-audit --input <file.pm4> [--output report.json]");
        Console.WriteLine("  pm4research inspect-mslk-refindex --input <file.pm4> [--output report.json]");
        Console.WriteLine("  pm4research export-json --input <file.pm4> [--output report.json]");
        Console.WriteLine("  pm4research scan-dir --input <dir> [--output reports.json]");
        Console.WriteLine("  pm4research inspect-hypotheses --input <file.pm4>");
        Console.WriteLine("  pm4research export-hypotheses --input <file.pm4> [--output report.json]");
        Console.WriteLine("  pm4research scan-hypotheses --input <dir> [--output reports.json]");
        Console.WriteLine("  pm4research scan-hypotheses-ndjson --input <dir> [--output reports.ndjson]");
        Console.WriteLine("  pm4research scan-audit --input <dir> [--output report.json]");
        Console.WriteLine("  pm4research scan-mslk-refindex --input <dir> [--output report.json]");
        Console.WriteLine("  pm4research scan-msur-geometry --input <dir> [--output report.json]");
        Console.WriteLine("  pm4research scan-mslk-refindex-classifier --input <dir> [--output report.json]");
        Console.WriteLine("  pm4research scan-structure-confidence --input <dir> [--output report.json]");
        Console.WriteLine("  pm4research scan-linkage --input <dir> [--output report.json]");
        Console.WriteLine("  pm4research scan-mscn --input <dir> [--output report.json]");
        Console.WriteLine("  pm4research scan-unknowns --input <dir> [--output report.json]");
    }

    private static CompactHypothesisReport ToCompactReport(Pm4TileObjectHypothesisReport report)
    {
        List<CompactFamilySummary> families = report.Objects
            .GroupBy(static obj => obj.Family)
            .OrderBy(static group => group.Key)
            .Select(static group => new CompactFamilySummary(
                group.Key,
                group.Count(),
                group.Max(static obj => obj.SurfaceCount),
                group.Max(static obj => obj.TotalIndexCount),
                group.Sum(static obj => obj.MprlFootprint.LinkedRefCount),
                group.Sum(static obj => obj.MprlFootprint.LinkedInBoundsCount)))
            .ToList();

        List<CompactHypothesis> objects = report.Objects
            .Select(static obj => new CompactHypothesis(
                obj.Family,
                obj.FamilyObjectIndex,
                obj.Ck24,
                obj.Ck24Type,
                obj.Ck24ObjectId,
                obj.SurfaceCount,
                obj.TotalIndexCount,
                obj.MdosIndices.Count,
                obj.GroupKeys.Count,
                obj.MslkGroupObjectIds.Count,
                obj.MslkRefIndices.Count,
                obj.Bounds,
                obj.MprlFootprint))
            .ToList();

        return new CompactHypothesisReport(
            report.SourcePath,
            report.TileX,
            report.TileY,
            report.Version,
            report.Ck24GroupCount,
            report.TotalHypothesisCount,
            families,
            objects,
            report.Diagnostics);
    }

    private static void PrintHypothesisReport(Pm4TileObjectHypothesisReport report)
    {
        Console.WriteLine("PM4 Object Hypothesis Report");
        Console.WriteLine("===========================");
        Console.WriteLine($"Source:   {report.SourcePath}");
        Console.WriteLine($"Tile:     {report.TileX}_{report.TileY}");
        Console.WriteLine($"Version:  {report.Version}");
        Console.WriteLine($"CK24s:    {report.Ck24GroupCount}");
        Console.WriteLine($"Objects:  {report.TotalHypothesisCount}");
        Console.WriteLine();

        Console.WriteLine("Family Counts");
        foreach (IGrouping<string, Pm4ObjectHypothesis> family in report.Objects.GroupBy(static obj => obj.Family).OrderBy(static group => group.Key))
        {
            int maxSurfaceCount = family.Max(static obj => obj.SurfaceCount);
            int maxIndexCount = family.Max(static obj => obj.TotalIndexCount);
            Console.WriteLine($"  {family.Key} objects={family.Count()} maxSurfaces={maxSurfaceCount} maxIndices={maxIndexCount}");
        }

        Console.WriteLine();
        Console.WriteLine("Largest Hypotheses");
        foreach (Pm4ObjectHypothesis obj in report.Objects
            .OrderByDescending(static obj => obj.SurfaceCount)
            .ThenByDescending(static obj => obj.TotalIndexCount)
            .Take(16))
        {
            Console.WriteLine($"  {obj.Family} ck24=0x{obj.Ck24:X6} idx={obj.FamilyObjectIndex} surfaces={obj.SurfaceCount} indices={obj.TotalIndexCount} mdos={obj.MdosIndices.Count} links={obj.MslkGroupObjectIds.Count} mprlLinked={obj.MprlFootprint.LinkedRefCount} mprlIn={obj.MprlFootprint.LinkedInBoundsCount}/{obj.MprlFootprint.TileInBoundsCount}");
        }

        if (report.Diagnostics.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("Diagnostics");
            foreach (string diagnostic in report.Diagnostics)
                Console.WriteLine($"  {diagnostic}");
        }
    }
}