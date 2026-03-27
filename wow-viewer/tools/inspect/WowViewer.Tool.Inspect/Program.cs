using System.Text.Json;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Runtime;
using WowViewer.Core.Wmo;

if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
{
	ShowUsage();
	return;
}

string area = args[0].ToLowerInvariant();
string[] tail = args.Skip(1).ToArray();

switch (area)
{
	case "map":
		RunMap(tail);
		break;
	case "pm4":
		RunPm4(tail);
		break;
	case "wmo":
		RunWmo(tail);
		break;
	default:
		Console.Error.WriteLine($"Unknown inspect area '{area}'.");
		ShowUsage();
		Environment.ExitCode = 1;
		break;
}

static void RunMap(string[] args)
{
	if (args.Length == 0)
	{
		ShowMapUsage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "inspect":
			RunMapInspect(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown map command '{command}'.");
			ShowMapUsage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunMapInspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input map file is required.");
		Environment.ExitCode = 1;
		return;
	}

	MapFileSummary summary = MapFileSummaryReader.Read(input);
	PrintMapSummary(summary);
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
			case "linkage":
				RunPm4Linkage(tail);
				break;
			case "mscn":
				RunPm4Mscn(tail);
				break;
			case "unknowns":
				RunPm4Unknowns(tail);
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

static void RunWmo(string[] args)
{
	if (args.Length == 0)
	{
		ShowWmoUsage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "inspect":
			RunWmoInspect(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown wmo command '{command}'.");
			ShowWmoUsage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunWmoInspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input WMO file is required.");
		Environment.ExitCode = 1;
		return;
	}

	WowFileDetection detection = WowFileDetector.Detect(input);
	if (detection.Kind == WowFileKind.Wmo)
	{
		WmoSummary summary = WmoSummaryReader.Read(input);
		PrintWmoSummary(summary);
		return;
	}

	if (detection.Kind == WowFileKind.WmoGroup)
	{
		WmoGroupSummary summary = WmoGroupSummaryReader.Read(input);
		PrintWmoGroupSummary(summary);
		if (summary.HasLiquid)
		{
			WmoGroupLiquidSummary liquidSummary = WmoGroupLiquidSummaryReader.Read(input);
			PrintWmoGroupLiquidSummary(liquidSummary);
		}
		return;
	}

	Console.Error.WriteLine($"Error: expected WMO root or group file, but detected {detection.Kind}.");
	Environment.ExitCode = 1;
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

static void RunPm4Linkage(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4LinkageReport report = Pm4ResearchLinkageAnalyzer.AnalyzeDirectory(input);
	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	PrintPm4LinkageReport(report);
}

static void RunPm4Mscn(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4MscnRelationshipReport report = Pm4ResearchMscnAnalyzer.AnalyzeDirectory(input);
	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	PrintPm4MscnReport(report);
}

static void RunPm4Unknowns(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 directory is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4UnknownsReport report = Pm4ResearchUnknownsAnalyzer.AnalyzeDirectory(input);
	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	PrintPm4UnknownsReport(report);
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
	Console.WriteLine($"PM4 canonical owner: {Pm4Boundary.CanonicalOwner}");
	Console.WriteLine($"PM4 legacy reference: {Pm4Boundary.LegacyReference}");
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

static void PrintPm4LinkageReport(Pm4LinkageReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 linkage report");
	Console.WriteLine($"Input directory: {report.InputDirectory}");
	Console.WriteLine($"Files: {report.FileCount}");
	Console.WriteLine($"Files with ref-index mismatches: {report.FilesWithRefIndexMismatches}");
	Console.WriteLine($"Files with bad MDOS refs: {report.FilesWithBadMdos}");
	Console.WriteLine($"Total ref-index mismatches: {report.TotalRefIndexMismatchCount}");
	Console.WriteLine();
	Console.WriteLine("Relationships:");
	foreach (Pm4RelationshipEdgeSummary relationship in report.Relationships)
	{
		Console.WriteLine($"  {relationship.Edge}: status={relationship.Status} fits={relationship.Fits} misses={relationship.Misses}");
	}

	Console.WriteLine();
	Console.WriteLine($"Identity summary: ck24={report.IdentitySummary.DistinctCk24Count} low16={report.IdentitySummary.DistinctCk24ObjectIdCount} groups={report.IdentitySummary.ObjectIdGroupsAnalyzed} reused={report.IdentitySummary.ReusedObjectIdGroupCount} crossType={report.IdentitySummary.ReusedAcrossTypeGroupCount}");

	Console.WriteLine();
	Console.WriteLine("Top mismatch families:");
	foreach (Pm4LinkageMismatchFamily family in report.TopMismatchFamilies.Take(8))
	{
		Console.WriteLine($"  {family.FamilyKey}: files={family.FileCount} entries={family.EntryCount} low16Matches={family.MatchingCk24ObjectIdEntryCount} low24Matches={family.MatchingFullCk24EntryCount}");
	}

	if (report.Notes.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Notes:");
		foreach (string note in report.Notes)
			Console.WriteLine($"  {note}");
	}
}

static void PrintPm4MscnReport(Pm4MscnRelationshipReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 MSCN report");
	Console.WriteLine($"Input directory: {report.InputDirectory}");
	Console.WriteLine($"Files: {report.FileCount}");
	Console.WriteLine($"Files with MSCN: {report.FilesWithMscn}");
	Console.WriteLine($"Files with tile coordinates: {report.FilesWithTileCoordinates}");
	Console.WriteLine($"Total MSCN points: {report.TotalMscnPointCount}");
	Console.WriteLine();
	Console.WriteLine("Relationships:");
	foreach (Pm4RelationshipEdgeSummary relationship in report.Relationships)
	{
		Console.WriteLine($"  {relationship.Edge}: status={relationship.Status} fits={relationship.Fits} misses={relationship.Misses}");
	}

	Console.WriteLine();
	Console.WriteLine($"Coordinate space: swappedWorld={report.CoordinateSpace.SwappedWorldTileFitCount} rawWorld={report.CoordinateSpace.RawWorldTileFitCount} ambiguousWorld={report.CoordinateSpace.AmbiguousWorldTileFitCount} tileLocal={report.CoordinateSpace.TileLocalLikeCount} neither={report.CoordinateSpace.NeitherFitCount}");
	Console.WriteLine($"Dominant files: swapped={report.CoordinateSpace.FilesSwappedDominant} raw={report.CoordinateSpace.FilesRawDominant} tileLocal={report.CoordinateSpace.FilesTileLocalDominant} noDominant={report.CoordinateSpace.FilesNoDominant}");

	Console.WriteLine();
	Console.WriteLine("Cluster distributions:");
	foreach (Pm4FieldDistribution distribution in report.ClusterDistributions)
	{
		Console.WriteLine($"  {distribution.Field}: total={distribution.TotalCount} distinct={distribution.DistinctCount}");
		foreach (Pm4ValueFrequency value in distribution.TopValues.Take(4))
			Console.WriteLine($"    {value.Value} -> {value.Count}");
	}

	if (report.TopInvalidMdosClusters.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Top invalid-MDOS clusters:");
		foreach (Pm4MscnClusterExample cluster in report.TopInvalidMdosClusters.Take(8))
		{
			Console.WriteLine($"  tile={cluster.TileX}_{cluster.TileY} ck24=0x{cluster.Ck24:X6} type=0x{cluster.Ck24Type:X2} obj={cluster.Ck24ObjectId} invalidMdos={cluster.InvalidMdosRefCount} distinctMdos={cluster.DistinctMdosCount} align={cluster.AlignmentMode}");
		}
	}

	if (report.Notes.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Notes:");
		foreach (string note in report.Notes)
			Console.WriteLine($"  {note}");
	}
}

static void PrintPm4UnknownsReport(Pm4UnknownsReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 unknowns report");
	Console.WriteLine($"Input directory: {report.InputDirectory}");
	Console.WriteLine($"Files: {report.FileCount}");
	Console.WriteLine($"Non-empty files: {report.NonEmptyFileCount}");
	Console.WriteLine();
	Console.WriteLine("Relationships:");
	foreach (Pm4RelationshipEdgeSummary relationship in report.Relationships)
	{
		Console.WriteLine($"  {relationship.Edge}: status={relationship.Status} fits={relationship.Fits} misses={relationship.Misses}");
	}

	Console.WriteLine();
	Console.WriteLine($"MSPI interpretation: active={report.MspiInterpretation.ActiveLinkCount} indicesOnly={report.MspiInterpretation.IndicesModeOnlyCount} trianglesOnly={report.MspiInterpretation.TrianglesModeOnlyCount} both={report.MspiInterpretation.BothModesCount} neither={report.MspiInterpretation.NeitherModeCount}");
	Console.WriteLine($"LinkId patterns: total={report.LinkIdPatterns.TotalCount} sentinelTile={report.LinkIdPatterns.SentinelTileLinkCount} zero={report.LinkIdPatterns.ZeroCount} other={report.LinkIdPatterns.OtherCount}");

	Console.WriteLine();
	Console.WriteLine("Unknowns:");
	foreach (Pm4UnknownFinding finding in report.Unknowns)
	{
		Console.WriteLine($"  [{finding.Status}] {finding.Name}");
		Console.WriteLine($"    {finding.Evidence}");
	}

	if (report.Notes.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Notes:");
		foreach (string note in report.Notes)
			Console.WriteLine($"  {note}");
	}
}

static void PrintMapSummary(MapFileSummary summary)
{
	Console.WriteLine("WowViewer.Tool.Inspect map report");
	Console.WriteLine($"Input: {summary.SourcePath}");
	Console.WriteLine($"Kind: {summary.Kind}");
	Console.WriteLine($"Version: {summary.Version?.ToString() ?? "n/a"}");
	if (summary.Kind == MapFileKind.Wdt)
	{
		using FileStream stream = File.OpenRead(summary.SourcePath);
		WdtSummary wdtSummary = WdtSummaryReader.Read(stream, summary);
		Console.WriteLine($"WDT semantics: wmoBased={wdtSummary.IsWmoBased} tiles={wdtSummary.TilesWithData}/{wdtSummary.TotalTiles} mainCellBytes={wdtSummary.MainCellSizeBytes} doodadNames={wdtSummary.DoodadNameCount} wmoNames={wdtSummary.WorldModelNameCount} doodadPlacements={wdtSummary.DoodadPlacementCount} wmoPlacements={wdtSummary.WorldModelPlacementCount}");
	}
	else if (summary.Kind is MapFileKind.Adt or MapFileKind.AdtTex or MapFileKind.AdtObj)
	{
		using FileStream stream = File.OpenRead(summary.SourcePath);
		AdtSummary adtSummary = AdtSummaryReader.Read(stream, summary);
		Console.WriteLine($"ADT semantics: kind={adtSummary.Kind} terrainChunks={adtSummary.TerrainChunkCount} textures={adtSummary.TextureNameCount} doodadNames={adtSummary.ModelNameCount} wmoNames={adtSummary.WorldModelNameCount} doodadPlacements={adtSummary.ModelPlacementCount} wmoPlacements={adtSummary.WorldModelPlacementCount} hasMfbo={adtSummary.HasFlightBounds} hasMh2o={adtSummary.HasWater} hasMamp={adtSummary.HasTextureParams} hasMtxf={adtSummary.HasTextureFlags}");
		AdtMcnkSummary mcnkSummary = AdtMcnkSummaryReader.Read(stream, summary);
		Console.WriteLine($"ADT MCNK semantics: mcnk={mcnkSummary.McnkCount} zero={mcnkSummary.ZeroLengthMcnkCount} headerLike={mcnkSummary.HeaderLikeMcnkCount} distinctIndex={mcnkSummary.DistinctIndexCount} duplicateIndex={mcnkSummary.DuplicateIndexCount} areaIds={mcnkSummary.DistinctAreaIdCount} holes={mcnkSummary.ChunksWithHoles} liquidFlags={mcnkSummary.ChunksWithLiquidFlags} mccvFlags={mcnkSummary.ChunksWithMccvFlag} mcvt={mcnkSummary.ChunksWithMcvt} mcnr={mcnkSummary.ChunksWithMcnr} mcly={mcnkSummary.ChunksWithMcly} mcal={mcnkSummary.ChunksWithMcal} mcsh={mcnkSummary.ChunksWithMcsh} mccv={mcnkSummary.ChunksWithMccv} mclq={mcnkSummary.ChunksWithMclq} mcrd={mcnkSummary.ChunksWithMcrd} mcrw={mcnkSummary.ChunksWithMcrw} totalLayers={mcnkSummary.TotalLayerCount} maxLayers={mcnkSummary.MaxLayerCount} multiLayerChunks={mcnkSummary.ChunksWithMultipleLayers}");
	}
	Console.WriteLine($"Top-level chunks: {summary.ChunkCount}");
	string chunkOrder = string.Join(", ", summary.Chunks.Take(16).Select(chunk => chunk.Id.ToString()));
	if (summary.Chunks.Count > 16)
		chunkOrder = $"{chunkOrder}, ... ({summary.Chunks.Count - 16} more)";

	Console.WriteLine($"Chunk order: {chunkOrder}");
	Console.WriteLine();
	Console.WriteLine("Chunk counts:");
	foreach (IGrouping<string, MapChunkLocation> group in summary.Chunks.GroupBy(chunk => chunk.Id.ToString()).OrderBy(group => group.Key))
	{
		Console.WriteLine($"  {group.Key}: count={group.Count()} bytes={group.Sum(chunk => (long)chunk.Size)}");
	}

	Console.WriteLine();
	Console.WriteLine("First top-level chunks:");
	foreach (MapChunkLocation chunk in summary.Chunks.Take(12))
	{
		Console.WriteLine($"  {chunk.Id}: size={chunk.Size} header={chunk.HeaderOffset} data={chunk.DataOffset}");
	}

	if (summary.Chunks.Count > 12)
		Console.WriteLine($"  ... {summary.Chunks.Count - 12} more chunks");
}

static void PrintWmoSummary(WmoSummary summary)
{
	Console.WriteLine("WowViewer.Tool.Inspect WMO report");
	Console.WriteLine($"Input: {summary.SourcePath}");
	Console.WriteLine($"Version: {summary.Version?.ToString() ?? "n/a"}");
	Console.WriteLine($"WMO semantics: materials={summary.MaterialEntryCount}/{summary.ReportedMaterialCount} groups={summary.GroupInfoCount}/{summary.ReportedGroupCount} portals={summary.ReportedPortalCount} lights={summary.ReportedLightCount} textures={summary.TextureNameCount} doodadNames={summary.DoodadNameTableCount}/{summary.ReportedDoodadNameCount} doodadPlacements={summary.DoodadPlacementEntryCount}/{summary.ReportedDoodadPlacementCount} doodadSets={summary.DoodadSetEntryCount}/{summary.ReportedDoodadSetCount} flags=0x{summary.Flags:X8}");
	Console.WriteLine($"Bounds: min={FormatVector(summary.BoundsMin)} max={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoGroupSummary(WmoGroupSummary summary)
{
	Console.WriteLine("WowViewer.Tool.Inspect WMO group report");
	Console.WriteLine($"Input: {summary.SourcePath}");
	Console.WriteLine($"Version: {summary.Version?.ToString() ?? "n/a"}");
	Console.WriteLine($"Header: bytes={summary.HeaderSizeBytes} nameOff={summary.NameOffset} descOff={summary.DescriptiveNameOffset} flags=0x{summary.Flags:X8} portals={summary.PortalCount}@{summary.PortalStart} liquid={summary.GroupLiquid}");
	Console.WriteLine($"Geometry: faces={summary.FaceMaterialCount} vertices={summary.VertexCount} indices={summary.IndexCount} normals={summary.NormalCount} primaryUv={summary.PrimaryUvCount} extraUvSets={summary.AdditionalUvSetCount} batches={summary.BatchCount}/{summary.DeclaredBatchCount} vertexColors={summary.VertexColorCount} doodadRefs={summary.DoodadRefCount} hasLiquid={summary.HasLiquid}");
	Console.WriteLine($"Bounds: min={FormatVector(summary.BoundsMin)} max={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoGroupLiquidSummary(WmoGroupLiquidSummary summary)
{
	Console.WriteLine($"MLIQ: payloadBytes={summary.PayloadSizeBytes} verts={summary.XVertexCount}x{summary.YVertexCount} tiles={summary.XTileCount}x{summary.YTileCount} corner={FormatVector(summary.Corner)} materialId={summary.MaterialId} heights={summary.HeightCount} range=[{summary.MinHeight:F2}, {summary.MaxHeight:F2}] visibleTiles={summary.VisibleTileCount}/{summary.TileCount} tileFlags={summary.TileFlagByteCount} liquidType={summary.LiquidType}");
}

static void ShowUsage()
{
	Console.WriteLine("WowViewer.Tool.Inspect");
	Console.WriteLine("Usage:");
	Console.WriteLine("  wowviewer-inspect map inspect --input <file.wdt|file.adt>");
	Console.WriteLine("  wowviewer-inspect wmo inspect --input <file.wmo>");
	Console.WriteLine("  wowviewer-inspect pm4 inspect --input <file.pm4>");
	Console.WriteLine("  wowviewer-inspect pm4 linkage --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 mscn --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 unknowns --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 audit --input <file.pm4>");
	Console.WriteLine("  wowviewer-inspect pm4 audit-directory --input <directory>");
	Console.WriteLine("  wowviewer-inspect pm4 export-json --input <file.pm4> [--output <report.json>]");
}

static void ShowWmoUsage()
{
	Console.WriteLine("WMO commands:");
	Console.WriteLine("  wmo inspect --input <file.wmo>");
}

static void ShowMapUsage()
{
	Console.WriteLine("Map commands:");
	Console.WriteLine("  map inspect --input <file.wdt|file.adt>");
}

static void ShowPm4Usage()
{
	Console.WriteLine("PM4 commands:");
	Console.WriteLine("  pm4 inspect --input <file.pm4>");
	Console.WriteLine("  pm4 linkage --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 mscn --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 unknowns --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 audit --input <file.pm4>");
	Console.WriteLine("  pm4 audit-directory --input <directory>");
	Console.WriteLine("  pm4 export-json --input <file.pm4> [--output <report.json>]");
}
