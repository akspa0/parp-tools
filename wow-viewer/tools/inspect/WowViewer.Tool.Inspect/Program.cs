using System.Text.Json;
using WowViewer.Core.PM4;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Runtime;

if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
{
	ShowUsage();
	return;
}

string area = args[0].ToLowerInvariant();
string[] tail = args.Skip(1).ToArray();

switch (area)
{
	case "pm4":
		RunPm4(tail);
		break;
	default:
		Console.Error.WriteLine($"Unknown inspect area '{area}'.");
		ShowUsage();
		Environment.ExitCode = 1;
		break;
}

static void RunPm4(string[] args)
{
	if (args.Length == 0)
	{
		ShowPm4Usage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "inspect":
			RunPm4Inspect(tail);
			break;
		case "audit":
			RunPm4Audit(tail);
			break;
		case "audit-directory":
			RunPm4AuditDirectory(tail);
			break;
		case "export-json":
			RunPm4ExportJson(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown pm4 command '{command}'.");
			ShowPm4Usage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunPm4Inspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4AnalysisReport report = Pm4ResearchAnalyzer.Analyze(Pm4ResearchReader.ReadFile(input));
	PrintPm4Report(report);
}

static void RunPm4Audit(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4DecodeAuditReport report = Pm4ResearchAuditAnalyzer.Analyze(Pm4ResearchReader.ReadFile(input));
	PrintPm4AuditReport(report);
}

static void RunPm4AuditDirectory(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4CorpusAuditReport report = Pm4ResearchAuditAnalyzer.AnalyzeDirectory(input);
	PrintPm4CorpusAuditReport(report);
}

static void RunPm4ExportJson(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4AnalysisReport report = Pm4ResearchAnalyzer.Analyze(Pm4ResearchReader.ReadFile(input));
	string json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, json);
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	Console.WriteLine(json);
}

static string? GetOption(string[] args, string longName, string shortName)
{
	for (int index = 0; index < args.Length - 1; index++)
	{
		if (string.Equals(args[index], longName, StringComparison.OrdinalIgnoreCase)
			|| string.Equals(args[index], shortName, StringComparison.OrdinalIgnoreCase))
		{
			return args[index + 1];
		}
	}

	return null;
}

static void PrintPm4Report(Pm4AnalysisReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 report");
	Console.WriteLine($"PM4 reference: {Pm4Boundary.RuntimeReference}");
	Console.WriteLine($"Runtime boundaries: {RuntimeBoundaries.All.Length}");
	Console.WriteLine($"Input: {report.SourcePath ?? "<memory>"}");
	Console.WriteLine($"Version: {report.Version}");
	Console.WriteLine($"Chunks: {report.ChunkOrder.Count}");
	Console.WriteLine($"Unknown chunks: {(report.UnknownChunks.Count == 0 ? "none" : string.Join(", ", report.UnknownChunks))}");
	Console.WriteLine();
	PrintVectorSet(report.Msvt);
	PrintVectorSet(report.Mscn);
	PrintVectorSet(report.MprlPositions);
	Console.WriteLine();
	Console.WriteLine($"MPRL total={report.Mprl.TotalCount}, normal={report.Mprl.NormalCount}, terminator={report.Mprl.TerminatorCount}");
	Console.WriteLine($"MPRL floor range={report.Mprl.FloorMin?.ToString() ?? "n/a"}..{report.Mprl.FloorMax?.ToString() ?? "n/a"}");
	Console.WriteLine($"MPRL rotation range={report.Mprl.RotationMinDegrees?.ToString("F2") ?? "n/a"}..{report.Mprl.RotationMaxDegrees?.ToString("F2") ?? "n/a"}");
	Console.WriteLine();
	Console.WriteLine("Top CK24 groups:");
	if (report.TopCk24Groups.Count == 0)
	{
		Console.WriteLine("  none");
	}
	else
	{
		foreach (Pm4Ck24Summary summary in report.TopCk24Groups.Take(10))
		{
			Console.WriteLine($"  ck24={summary.Ck24} type={summary.Ck24Type} object={summary.Ck24ObjectId} surfaces={summary.SurfaceCount} indices={summary.TotalIndexCount} mdos={summary.DistinctMdosCount}");
		}
	}

	if (report.ResearchNotes.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Research notes:");
		foreach (string note in report.ResearchNotes)
			Console.WriteLine($"  {note}");
	}

	if (report.Diagnostics.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Diagnostics:");
		foreach (string diagnostic in report.Diagnostics.Take(20))
			Console.WriteLine($"  {diagnostic}");
	}
}

static void PrintVectorSet(Pm4VectorSetSummary summary)
{
	Console.WriteLine($"{summary.Name}: count={summary.Count}");
	if (summary.Bounds is null || summary.Centroid is null)
		return;

	Console.WriteLine($"  bounds min={FormatVector(summary.Bounds.Min)} max={FormatVector(summary.Bounds.Max)}");
	Console.WriteLine($"  centroid={FormatVector(summary.Centroid.Value)}");
}

static string FormatVector(System.Numerics.Vector3 value)
{
	return $"({value.X:F2}, {value.Y:F2}, {value.Z:F2})";
}

static void PrintPm4AuditReport(Pm4DecodeAuditReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 decode audit");
	Console.WriteLine($"Input: {report.SourcePath ?? "<memory>"}");
	Console.WriteLine($"Version: {report.Version}");
	Console.WriteLine($"Chunks: {report.ChunkCount}, recognized={report.RecognizedChunkCount}, unknown={report.UnknownChunkCount}");
	Console.WriteLine($"Trailing-bytes diagnostic: {(report.HasTrailingBytesDiagnostic ? "yes" : "no")}");
	Console.WriteLine($"Overrun diagnostic: {(report.HasOverrunDiagnostic ? "yes" : "no")}");
	Console.WriteLine();
	Console.WriteLine("Reference audits:");
	foreach (Pm4ReferenceAudit audit in report.ReferenceAudits)
	{
		Console.WriteLine($"  {audit.Name}: total={audit.TotalCount} valid={audit.ValidCount} invalid={audit.InvalidCount}");
		foreach (string example in audit.Examples.Take(3))
			Console.WriteLine($"    {example}");
	}

	Console.WriteLine();
	Console.WriteLine("Chunk audits:");
	foreach (Pm4ChunkDecodeAudit audit in report.ChunkAudits.Take(12))
	{
		Console.WriteLine($"  {audit.Signature}: chunks={audit.ChunkCount} entries={audit.EntryCount} bytes={audit.TotalBytes} strideRemainders={audit.StrideRemainderCount}");
	}

	if (report.UnknownChunkSignatures.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine($"Unknown signatures: {string.Join(", ", report.UnknownChunkSignatures)}");
	}

	if (report.Diagnostics.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Diagnostics:");
		foreach (string diagnostic in report.Diagnostics.Take(20))
			Console.WriteLine($"  {diagnostic}");
	}
}

static void PrintPm4CorpusAuditReport(Pm4CorpusAuditReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 corpus audit");
	Console.WriteLine($"Input directory: {report.InputDirectory}");
	Console.WriteLine($"Files: {report.FileCount}");
	Console.WriteLine($"Files with diagnostics: {report.FilesWithDiagnostics}");
	Console.WriteLine($"Files with unknown chunks: {report.FilesWithUnknownChunks}");
	Console.WriteLine();
	Console.WriteLine("Chunk audits:");
	foreach (Pm4CorpusChunkAudit audit in report.ChunkAudits.Take(12))
	{
		Console.WriteLine($"  {audit.Signature}: files={audit.FileCount} totalChunks={audit.TotalChunkCount} totalEntries={audit.TotalEntryCount} strideFiles={audit.FilesWithStrideRemainders}");
	}

	Console.WriteLine();
	Console.WriteLine("Reference audits:");
	foreach (Pm4CorpusReferenceAudit audit in report.ReferenceAudits)
	{
		Console.WriteLine($"  {audit.Name}: total={audit.TotalCount} invalid={audit.InvalidCount}");
		foreach (string example in audit.ExampleFailures.Take(3))
			Console.WriteLine($"    {example}");
	}

	if (report.UnknownChunkSignatures.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine($"Unknown signatures: {string.Join(", ", report.UnknownChunkSignatures)}");
	}

	if (report.Diagnostics.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Top diagnostics:");
		foreach (string diagnostic in report.Diagnostics.Take(20))
			Console.WriteLine($"  {diagnostic}");
	}
}

static void ShowUsage()
{
	Console.WriteLine("WowViewer.Tool.Inspect");
	Console.WriteLine("Usage:");
	Console.WriteLine("  wowviewer-inspect pm4 inspect --input <file.pm4>");
	Console.WriteLine("  wowviewer-inspect pm4 audit --input <file.pm4>");
	Console.WriteLine("  wowviewer-inspect pm4 audit-directory --input <directory>");
	Console.WriteLine("  wowviewer-inspect pm4 export-json --input <file.pm4> [--output <report.json>]");
}

static void ShowPm4Usage()
{
	Console.WriteLine("PM4 commands:");
	Console.WriteLine("  pm4 inspect --input <file.pm4>");
	Console.WriteLine("  pm4 audit --input <file.pm4>");
	Console.WriteLine("  pm4 audit-directory --input <directory>");
	Console.WriteLine("  pm4 export-json --input <file.pm4> [--output <report.json>]");
}
