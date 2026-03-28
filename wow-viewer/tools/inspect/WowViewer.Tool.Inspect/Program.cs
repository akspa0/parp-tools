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

	byte[]? archivedBytes = null;
	Stream OpenInputStream()
	{
		if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
			return File.OpenRead(input);

		archivedBytes ??= AlphaArchiveReader.ReadWithMpqFallback(input)
			?? throw new FileNotFoundException($"Could not read inspect input '{input}' directly or from a companion MPQ archive.", input);
		return new MemoryStream(archivedBytes, writable: false);
	}

	T ReadInput<T>(Func<Stream, string, T> reader)
	{
		using Stream stream = OpenInputStream();
		return reader(stream, input);
	}

	WowFileDetection detection;
	using (Stream detectionStream = OpenInputStream())
		detection = WowFileDetector.Detect(detectionStream, input);

	if (detection.Kind == WowFileKind.Wmo)
	{
		WmoSummary summary = ReadInput(WmoSummaryReader.Read);
		PrintWmoSummary(summary);
		if (summary.DoodadSetEntryCount > 0 && summary.DoodadPlacementEntryCount > 0)
		{
			WmoDoodadSetRangeSummary doodadSetRangeSummary = ReadInput(WmoDoodadSetRangeSummaryReader.Read);
			PrintWmoDoodadSetRangeSummary(doodadSetRangeSummary);
		}
		if (summary.GroupInfoCount > 0)
		{
			try
			{
				WmoGroupNameReferenceSummary groupNameReferenceSummary = ReadInput(WmoGroupNameReferenceSummaryReader.Read);
				PrintWmoGroupNameReferenceSummary(groupNameReferenceSummary);
			}
			catch (InvalidDataException)
			{
			}
		}
		if (summary.DoodadPlacementEntryCount > 0 && summary.DoodadNameTableCount > 0)
		{
			WmoDoodadNameReferenceSummary doodadNameReferenceSummary = ReadInput(WmoDoodadNameReferenceSummaryReader.Read);
			PrintWmoDoodadNameReferenceSummary(doodadNameReferenceSummary);
		}
		if (summary.ReportedLightCount > 0)
		{
			try
			{
				WmoLightSummary lightSummary = ReadInput(WmoLightSummaryReader.Read);
				PrintWmoLightSummary(lightSummary);
			}
			catch (InvalidDataException)
			{
			}
		}
		try
		{
			WmoFogSummary fogSummary = ReadInput(WmoFogSummaryReader.Read);
			PrintWmoFogSummary(fogSummary);
		}
		catch (InvalidDataException)
		{
		}
		try
		{
			WmoOpaqueChunkSummary mcvpSummary = ReadInput((stream, sourcePath) => WmoOpaqueChunkSummaryReader.Read(stream, sourcePath, WmoChunkIds.Mcvp));
			PrintWmoOpaqueChunkSummary(mcvpSummary);
		}
		catch (InvalidDataException)
		{
		}
		if (summary.ReportedPortalCount > 0)
		{
			try
			{
				WmoPortalVertexSummary portalVertexSummary = ReadInput(WmoPortalVertexSummaryReader.Read);
				PrintWmoPortalVertexSummary(portalVertexSummary);
				WmoPortalInfoSummary portalInfoSummary = ReadInput(WmoPortalInfoSummaryReader.Read);
				PrintWmoPortalInfoSummary(portalInfoSummary);
				WmoPortalRefSummary portalRefSummary = ReadInput(WmoPortalRefSummaryReader.Read);
				PrintWmoPortalRefSummary(portalRefSummary);
				WmoPortalVertexRangeSummary portalVertexRangeSummary = ReadInput(WmoPortalVertexRangeSummaryReader.Read);
				PrintWmoPortalVertexRangeSummary(portalVertexRangeSummary);
				WmoPortalRefRangeSummary portalRefRangeSummary = ReadInput(WmoPortalRefRangeSummaryReader.Read);
				PrintWmoPortalRefRangeSummary(portalRefRangeSummary);
				if (summary.GroupInfoCount > 0)
				{
					WmoPortalGroupRangeSummary portalGroupRangeSummary = ReadInput(WmoPortalGroupRangeSummaryReader.Read);
					PrintWmoPortalGroupRangeSummary(portalGroupRangeSummary);
				}
			}
			catch (InvalidDataException)
			{
			}
		}
		if (summary.MaterialEntryCount > 0 || summary.GroupInfoCount > 0 || summary.DoodadSetEntryCount > 0 || summary.DoodadPlacementEntryCount > 0 || summary.ReportedPortalCount > 0 || summary.ReportedLightCount > 0)
		{
			try
			{
				WmoVisibleVertexSummary visibleVertexSummary = ReadInput(WmoVisibleVertexSummaryReader.Read);
				PrintWmoVisibleVertexSummary(visibleVertexSummary);
			}
			catch (InvalidDataException)
			{
			}
			try
			{
				WmoVisibleBlockSummary visibleBlockSummary = ReadInput(WmoVisibleBlockSummaryReader.Read);
				PrintWmoVisibleBlockSummary(visibleBlockSummary);
			}
			catch (InvalidDataException)
			{
			}
			try
			{
				WmoVisibleBlockReferenceSummary visibleBlockReferenceSummary = ReadInput(WmoVisibleBlockReferenceSummaryReader.Read);
				PrintWmoVisibleBlockReferenceSummary(visibleBlockReferenceSummary);
			}
			catch (InvalidDataException)
			{
			}
		}
		try
		{
			WmoSkyboxSummary skyboxSummary = ReadInput(WmoSkyboxSummaryReader.Read);
			PrintWmoSkyboxSummary(skyboxSummary);
		}
		catch (InvalidDataException)
		{
		}
		try
		{
			WmoGroupNameTableSummary groupNameSummary = ReadInput(WmoGroupNameTableSummaryReader.Read);
			PrintWmoGroupNameTableSummary(groupNameSummary);
		}
		catch (InvalidDataException)
		{
		}
		if (summary.DoodadSetEntryCount > 0)
		{
			WmoDoodadSetSummary doodadSetSummary = ReadInput(WmoDoodadSetSummaryReader.Read);
			PrintWmoDoodadSetSummary(doodadSetSummary);
		}
		if (summary.DoodadPlacementEntryCount > 0)
		{
			WmoDoodadPlacementSummary doodadPlacementSummary = ReadInput(WmoDoodadPlacementSummaryReader.Read);
			PrintWmoDoodadPlacementSummary(doodadPlacementSummary);
		}
		if (summary.DoodadNameTableCount > 0)
		{
			WmoDoodadNameTableSummary doodadNameSummary = ReadInput(WmoDoodadNameTableSummaryReader.Read);
			PrintWmoDoodadNameTableSummary(doodadNameSummary);
		}
		if (summary.TextureNameCount > 0)
		{
			WmoTextureTableSummary textureSummary = ReadInput(WmoTextureTableSummaryReader.Read);
			PrintWmoTextureTableSummary(textureSummary);
		}
		if (summary.MaterialEntryCount > 0)
		{
			WmoMaterialSummary materialSummary = ReadInput(WmoMaterialSummaryReader.Read);
			PrintWmoMaterialSummary(materialSummary);
		}
		if (summary.GroupInfoCount > 0)
		{
			WmoGroupInfoSummary groupInfoSummary = ReadInput(WmoGroupInfoSummaryReader.Read);
			PrintWmoGroupInfoSummary(groupInfoSummary);
		}
		try
		{
			WmoEmbeddedGroupSummary embeddedGroupSummary = ReadInput(WmoEmbeddedGroupSummaryReader.Read);
			PrintWmoEmbeddedGroupSummary(embeddedGroupSummary);
		}
		catch (InvalidDataException)
		{
		}
		try
		{
			WmoEmbeddedGroupLinkageSummary embeddedGroupLinkageSummary = ReadInput(WmoEmbeddedGroupLinkageSummaryReader.Read);
			PrintWmoEmbeddedGroupLinkageSummary(embeddedGroupLinkageSummary);
		}
		catch (InvalidDataException)
		{
		}
		try
		{
			IReadOnlyList<WmoEmbeddedGroupDetail> embeddedGroupDetails = ReadInput(WmoEmbeddedGroupDetailReader.Read);
			PrintWmoEmbeddedGroupDetails(embeddedGroupDetails);
		}
		catch (InvalidDataException)
		{
		}
		return;
	}

	if (detection.Kind == WowFileKind.WmoGroup)
	{
		WmoGroupSummary summary = ReadInput(WmoGroupSummaryReader.Read);
		PrintWmoGroupSummary(summary);
		if (summary.NormalCount > 0)
		{
			WmoGroupNormalSummary normalSummary = ReadInput(WmoGroupNormalSummaryReader.Read);
			PrintWmoGroupNormalSummary(normalSummary);
		}
		if (summary.VertexCount > 0)
		{
			WmoGroupVertexSummary vertexSummary = ReadInput(WmoGroupVertexSummaryReader.Read);
			PrintWmoGroupVertexSummary(vertexSummary);
		}
		if (summary.IndexCount > 0)
		{
			WmoGroupIndexSummary indexSummary = ReadInput(WmoGroupIndexSummaryReader.Read);
			PrintWmoGroupIndexSummary(indexSummary);
		}
		if (summary.DoodadRefCount > 0)
		{
			WmoGroupDoodadRefSummary doodadRefSummary = ReadInput(WmoGroupDoodadRefSummaryReader.Read);
			PrintWmoGroupDoodadRefSummary(doodadRefSummary);
		}
		if (summary.LightRefCount > 0)
		{
			WmoGroupLightRefSummary lightRefSummary = ReadInput(WmoGroupLightRefSummaryReader.Read);
			PrintWmoGroupLightRefSummary(lightRefSummary);
		}
		if (summary.VertexColorCount > 0)
		{
			WmoGroupVertexColorSummary colorSummary = ReadInput(WmoGroupVertexColorSummaryReader.Read);
			PrintWmoGroupVertexColorSummary(colorSummary);
		}
		if (summary.PrimaryUvCount > 0)
		{
			WmoGroupUvSummary uvSummary = ReadInput(WmoGroupUvSummaryReader.Read);
			PrintWmoGroupUvSummary(uvSummary);
		}
		if (summary.FaceMaterialCount > 0)
		{
			WmoGroupFaceMaterialSummary faceSummary = ReadInput(WmoGroupFaceMaterialSummaryReader.Read);
			PrintWmoGroupFaceMaterialSummary(faceSummary);
		}
		if (summary.BatchCount > 0)
		{
			WmoGroupBatchSummary batchSummary = ReadInput(WmoGroupBatchSummaryReader.Read);
			PrintWmoGroupBatchSummary(batchSummary);
		}
		if (summary.BspNodeCount > 0)
		{
			WmoGroupBspNodeSummary bspNodeSummary = ReadInput(WmoGroupBspNodeSummaryReader.Read);
			PrintWmoGroupBspNodeSummary(bspNodeSummary);
		}
		if (summary.BspFaceRefCount > 0)
		{
			WmoGroupBspFaceSummary bspFaceSummary = ReadInput(WmoGroupBspFaceSummaryReader.Read);
			PrintWmoGroupBspFaceSummary(bspFaceSummary);
		}
		if (summary.BspNodeCount > 0 && summary.BspFaceRefCount > 0)
		{
			WmoGroupBspFaceRangeSummary bspFaceRangeSummary = ReadInput(WmoGroupBspFaceRangeSummaryReader.Read);
			PrintWmoGroupBspFaceRangeSummary(bspFaceRangeSummary);
		}
		if (summary.HasLiquid)
		{
			WmoGroupLiquidSummary liquidSummary = ReadInput(WmoGroupLiquidSummaryReader.Read);
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

static void PrintWmoGroupInfoSummary(WmoGroupInfoSummary summary)
{
	Console.WriteLine($"MOGI: payloadBytes={summary.PayloadSizeBytes} entryBytes={summary.EntrySizeBytes} entries={summary.EntryCount} distinctFlags={summary.DistinctFlagCount} nonZeroFlags={summary.NonZeroFlagCount} nameOffsetRange={summary.MinNameOffset}-{summary.MaxNameOffset} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoEmbeddedGroupSummary(WmoEmbeddedGroupSummary summary)
{
	Console.WriteLine($"MOGP(root): groups={summary.GroupCount} headerBytes={summary.MinHeaderSizeBytes}-{summary.MaxHeaderSizeBytes} groupsWithPortals={summary.GroupsWithPortals} groupsWithLiquid={summary.GroupsWithLiquid} faces={summary.TotalFaceMaterialCount} vertices={summary.TotalVertexCount} indices={summary.TotalIndexCount} normals={summary.TotalNormalCount} batches={summary.TotalBatchCount} doodadRefs={summary.TotalDoodadRefCount} lightRefs={summary.TotalLightRefCount} bspNodes={summary.TotalBspNodeCount} bspFaceRefs={summary.TotalBspFaceRefCount} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoEmbeddedGroupLinkageSummary(WmoEmbeddedGroupLinkageSummary summary)
{
	Console.WriteLine($"MOGI->MOGP(root): infos={summary.GroupInfoCount} groups={summary.EmbeddedGroupCount} coveredPairs={summary.CoveredPairCount} missingGroups={summary.MissingEmbeddedGroupCount} extraGroups={summary.ExtraEmbeddedGroupCount} flagMatches={summary.FlagMatchCount} boundsMatches={summary.BoundsMatchCount} maxBoundsDelta={summary.MaxBoundsDelta:F3}");
}

static void PrintWmoEmbeddedGroupDetails(IReadOnlyList<WmoEmbeddedGroupDetail> details)
{
	foreach (WmoEmbeddedGroupDetail detail in details)
	{
		PrintWmoEmbeddedGroupDetail(detail);
	}
}

static void PrintWmoEmbeddedGroupDetail(WmoEmbeddedGroupDetail detail)
{
	WmoGroupSummary summary = detail.GroupSummary;
	Console.WriteLine($"MOGP(root)[{detail.GroupIndex}]: offset={detail.GroupHeaderOffset} flags=0x{summary.Flags:X8} portals={summary.PortalCount}@{summary.PortalStart} faces={summary.FaceMaterialCount} vertices={summary.VertexCount} indices={summary.IndexCount} normals={summary.NormalCount} batches={summary.BatchCount}/{summary.DeclaredBatchCount} doodadRefs={summary.DoodadRefCount} lightRefs={summary.LightRefCount} bspNodes={summary.BspNodeCount} bspFaceRefs={summary.BspFaceRefCount} hasLiquid={summary.HasLiquid} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");

	if (detail.NormalSummary is not null)
		Console.WriteLine($"MONR(root)[{detail.GroupIndex}]: payloadBytes={detail.NormalSummary.PayloadSizeBytes} normals={detail.NormalSummary.NormalCount} rangeX=[{detail.NormalSummary.MinX:F3}, {detail.NormalSummary.MaxX:F3}] rangeY=[{detail.NormalSummary.MinY:F3}, {detail.NormalSummary.MaxY:F3}] rangeZ=[{detail.NormalSummary.MinZ:F3}, {detail.NormalSummary.MaxZ:F3}] lengthRange=[{detail.NormalSummary.MinLength:F3}, {detail.NormalSummary.MaxLength:F3}] avgLength={detail.NormalSummary.AverageLength:F3} nearUnit={detail.NormalSummary.NearUnitCount}");

	if (detail.VertexSummary is not null)
		Console.WriteLine($"MOVT(root)[{detail.GroupIndex}]: payloadBytes={detail.VertexSummary.PayloadSizeBytes} vertices={detail.VertexSummary.VertexCount} boundsMin={FormatVector(detail.VertexSummary.BoundsMin)} boundsMax={FormatVector(detail.VertexSummary.BoundsMax)}");

	if (detail.IndexSummary is not null)
		Console.WriteLine($"{detail.IndexSummary.ChunkId}(root)[{detail.GroupIndex}]: payloadBytes={detail.IndexSummary.PayloadSizeBytes} indices={detail.IndexSummary.IndexCount} triangles={detail.IndexSummary.TriangleCount} distinctIndices={detail.IndexSummary.DistinctIndexCount} indexRange={detail.IndexSummary.MinIndex}-{detail.IndexSummary.MaxIndex} degenerateTriangles={detail.IndexSummary.DegenerateTriangleCount}");

	if (detail.DoodadRefSummary is not null)
		Console.WriteLine($"MODR(root)[{detail.GroupIndex}]: payloadBytes={detail.DoodadRefSummary.PayloadSizeBytes} refs={detail.DoodadRefSummary.RefCount} distinctRefs={detail.DoodadRefSummary.DistinctRefCount} refRange={detail.DoodadRefSummary.MinRef}-{detail.DoodadRefSummary.MaxRef} duplicateRefs={detail.DoodadRefSummary.DuplicateRefCount}");

	if (detail.LightRefSummary is not null)
		Console.WriteLine($"MOLR(root)[{detail.GroupIndex}]: payloadBytes={detail.LightRefSummary.PayloadSizeBytes} refs={detail.LightRefSummary.RefCount} distinctRefs={detail.LightRefSummary.DistinctRefCount} refRange={detail.LightRefSummary.MinRef}-{detail.LightRefSummary.MaxRef} duplicateRefs={detail.LightRefSummary.DuplicateRefCount}");

	if (detail.VertexColorSummary is not null)
		Console.WriteLine($"MOCV(root)[{detail.GroupIndex}]: payloadBytes={detail.VertexColorSummary.PrimaryPayloadSizeBytes} primaryColors={detail.VertexColorSummary.PrimaryColorCount} rangeR=[{detail.VertexColorSummary.MinRed}, {detail.VertexColorSummary.MaxRed}] rangeG=[{detail.VertexColorSummary.MinGreen}, {detail.VertexColorSummary.MaxGreen}] rangeB=[{detail.VertexColorSummary.MinBlue}, {detail.VertexColorSummary.MaxBlue}] rangeA=[{detail.VertexColorSummary.MinAlpha}, {detail.VertexColorSummary.MaxAlpha}] avgA={detail.VertexColorSummary.AverageAlpha} extraColorSets={detail.VertexColorSummary.AdditionalColorSetCount} totalExtraColors={detail.VertexColorSummary.TotalAdditionalColorCount} maxExtraColors={detail.VertexColorSummary.MaxAdditionalColorCount}");

	if (detail.UvSummary is not null)
		Console.WriteLine($"MOTV(root)[{detail.GroupIndex}]: payloadBytes={detail.UvSummary.PrimaryPayloadSizeBytes} primaryUv={detail.UvSummary.PrimaryUvCount} rangeU=[{detail.UvSummary.MinU:F3}, {detail.UvSummary.MaxU:F3}] rangeV=[{detail.UvSummary.MinV:F3}, {detail.UvSummary.MaxV:F3}] extraUvSets={detail.UvSummary.AdditionalUvSetCount} totalExtraUv={detail.UvSummary.TotalAdditionalUvCount} maxExtraUv={detail.UvSummary.MaxAdditionalUvCount}");

	if (detail.FaceMaterialSummary is not null)
		Console.WriteLine($"MOPY(root)[{detail.GroupIndex}]: payloadBytes={detail.FaceMaterialSummary.PayloadSizeBytes} entryBytes={detail.FaceMaterialSummary.EntrySizeBytes} faces={detail.FaceMaterialSummary.FaceCount} distinctMaterials={detail.FaceMaterialSummary.DistinctMaterialIdCount} highestMaterialId={detail.FaceMaterialSummary.HighestMaterialId} hiddenFaces={detail.FaceMaterialSummary.HiddenFaceCount} flaggedFaces={detail.FaceMaterialSummary.FlaggedFaceCount}");

	if (detail.BatchSummary is not null)
		Console.WriteLine($"MOBA(root)[{detail.GroupIndex}]: payloadBytes={detail.BatchSummary.PayloadSizeBytes} entries={detail.BatchSummary.EntryCount} hasMaterialIds={detail.BatchSummary.HasMaterialIds} distinctMaterials={detail.BatchSummary.DistinctMaterialIdCount} highestMaterialId={detail.BatchSummary.HighestMaterialId} totalIndexCount={detail.BatchSummary.TotalIndexCount} firstIndexRange={detail.BatchSummary.MinFirstIndex}-{detail.BatchSummary.MaxFirstIndex} maxIndexEnd={detail.BatchSummary.MaxIndexEnd} flaggedBatches={detail.BatchSummary.FlaggedBatchCount}");

	if (detail.BspNodeSummary is not null)
		Console.WriteLine($"MOBN(root)[{detail.GroupIndex}]: payloadBytes={detail.BspNodeSummary.PayloadSizeBytes} nodes={detail.BspNodeSummary.NodeCount} leafNodes={detail.BspNodeSummary.LeafNodeCount} branchNodes={detail.BspNodeSummary.BranchNodeCount} childRefs={detail.BspNodeSummary.ChildReferenceCount} noChildRefs={detail.BspNodeSummary.NoChildReferenceCount} outOfRangeChildRefs={detail.BspNodeSummary.OutOfRangeChildReferenceCount} faceCountRange={detail.BspNodeSummary.MinFaceCount}-{detail.BspNodeSummary.MaxFaceCount} faceStartRange={detail.BspNodeSummary.MinFaceStart}-{detail.BspNodeSummary.MaxFaceStart} maxFaceEnd={detail.BspNodeSummary.MaxFaceEnd} planeDistRange=[{detail.BspNodeSummary.MinPlaneDistance:F3}, {detail.BspNodeSummary.MaxPlaneDistance:F3}]");

	if (detail.BspFaceSummary is not null)
		Console.WriteLine($"MOBR(root)[{detail.GroupIndex}]: payloadBytes={detail.BspFaceSummary.PayloadSizeBytes} refs={detail.BspFaceSummary.RefCount} distinctRefs={detail.BspFaceSummary.DistinctFaceRefCount} refRange={detail.BspFaceSummary.MinFaceRef}-{detail.BspFaceSummary.MaxFaceRef} duplicateRefs={detail.BspFaceSummary.DuplicateFaceRefCount}");

	if (detail.BspFaceRangeSummary is not null)
		Console.WriteLine($"MOBN->MOBR(root)[{detail.GroupIndex}]: nodes={detail.BspFaceRangeSummary.NodeCount} faceRefs={detail.BspFaceRangeSummary.FaceRefCount} zeroFaceNodes={detail.BspFaceRangeSummary.ZeroFaceNodeCount} coveredNodes={detail.BspFaceRangeSummary.CoveredNodeCount} outOfRangeNodes={detail.BspFaceRangeSummary.OutOfRangeNodeCount} maxFaceEnd={detail.BspFaceRangeSummary.MaxFaceEnd}");

	if (detail.LiquidSummary is not null)
		Console.WriteLine($"MLIQ(root)[{detail.GroupIndex}]: payloadBytes={detail.LiquidSummary.PayloadSizeBytes} verts={detail.LiquidSummary.XVertexCount}x{detail.LiquidSummary.YVertexCount} tiles={detail.LiquidSummary.XTileCount}x{detail.LiquidSummary.YTileCount} corner={FormatVector(detail.LiquidSummary.Corner)} materialId={detail.LiquidSummary.MaterialId} heights={detail.LiquidSummary.HeightCount} range=[{detail.LiquidSummary.MinHeight:F2}, {detail.LiquidSummary.MaxHeight:F2}] visibleTiles={detail.LiquidSummary.VisibleTileCount}/{detail.LiquidSummary.TileCount} tileFlags={detail.LiquidSummary.TileFlagByteCount} liquidType={detail.LiquidSummary.LiquidType}");
}

static void PrintWmoMaterialSummary(WmoMaterialSummary summary)
{
	Console.WriteLine($"MOMT: payloadBytes={summary.PayloadSizeBytes} entryBytes={summary.EntrySizeBytes} entries={summary.EntryCount} distinctShaders={summary.DistinctShaderCount} distinctBlendModes={summary.DistinctBlendModeCount} nonZeroFlags={summary.NonZeroFlagCount} maxTex1Ofs={summary.MaxTexture1Offset} maxTex2Ofs={summary.MaxTexture2Offset} maxTex3Ofs={summary.MaxTexture3Offset}");
}

static void PrintWmoTextureTableSummary(WmoTextureTableSummary summary)
{
	Console.WriteLine($"MOTX: payloadBytes={summary.PayloadSizeBytes} textures={summary.TextureCount} longestEntry={summary.LongestEntryLength} maxOffset={summary.MaxOffset} extensions={summary.DistinctExtensionCount} blpEntries={summary.BlpEntryCount}");
}

static void PrintWmoDoodadNameTableSummary(WmoDoodadNameTableSummary summary)
{
	Console.WriteLine($"MODN: payloadBytes={summary.PayloadSizeBytes} names={summary.NameCount} longestEntry={summary.LongestEntryLength} maxOffset={summary.MaxOffset} extensions={summary.DistinctExtensionCount} mdxEntries={summary.MdxEntryCount} m2Entries={summary.M2EntryCount}");
}

static void PrintWmoDoodadSetSummary(WmoDoodadSetSummary summary)
{
	Console.WriteLine($"MODS: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} nonEmptySets={summary.NonEmptySetCount} longestName={summary.LongestNameLength} totalDoodadRefs={summary.TotalDoodadRefs} maxStartIndex={summary.MaxStartIndex} maxRangeEnd={summary.MaxRangeEnd}");
}

static void PrintWmoDoodadPlacementSummary(WmoDoodadPlacementSummary summary)
{
	Console.WriteLine($"MODD: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} distinctNameIndices={summary.DistinctNameIndexCount} maxNameIndex={summary.MaxNameIndex} scaleRange=[{summary.MinScale:F3}, {summary.MaxScale:F3}] alphaRange=[{summary.MinAlpha}, {summary.MaxAlpha}] boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoGroupNameTableSummary(WmoGroupNameTableSummary summary)
{
	Console.WriteLine($"MOGN: payloadBytes={summary.PayloadSizeBytes} names={summary.NameCount} longestEntry={summary.LongestEntryLength} maxOffset={summary.MaxOffset}");
}

static void PrintWmoSkyboxSummary(WmoSkyboxSummary summary)
{
	Console.WriteLine($"MOSB: payloadBytes={summary.PayloadSizeBytes} skybox={summary.SkyboxName}");
}

static void PrintWmoPortalVertexSummary(WmoPortalVertexSummary summary)
{
	Console.WriteLine($"MOPV: payloadBytes={summary.PayloadSizeBytes} vertices={summary.VertexCount} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoPortalInfoSummary(WmoPortalInfoSummary summary)
{
	Console.WriteLine($"MOPT: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} maxStartVertex={summary.MaxStartVertex} maxVertexCount={summary.MaxVertexCount} planeDRange=[{summary.MinPlaneD:F3}, {summary.MaxPlaneD:F3}]");
}

static void PrintWmoPortalRefSummary(WmoPortalRefSummary summary)
{
	Console.WriteLine($"MOPR: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} distinctPortals={summary.DistinctPortalIndexCount} maxGroupIndex={summary.MaxGroupIndex} sides(+/-/0)={summary.PositiveSideCount}/{summary.NegativeSideCount}/{summary.NeutralSideCount}");
}

static void PrintWmoPortalVertexRangeSummary(WmoPortalVertexRangeSummary summary)
{
	Console.WriteLine($"MOPT->MOPV: portals={summary.EntryCount} vertices={summary.VertexCount} zeroVertexPortals={summary.ZeroVertexPortalCount} coveredPortals={summary.CoveredPortalCount} outOfRangePortals={summary.OutOfRangePortalCount} maxVertexEnd={summary.MaxVertexEnd}");
}

static void PrintWmoPortalRefRangeSummary(WmoPortalRefRangeSummary summary)
{
	Console.WriteLine($"MOPR->MOPT: refs={summary.RefCount} portals={summary.PortalCount} coveredRefs={summary.CoveredRefCount} outOfRangeRefs={summary.OutOfRangeRefCount} distinctPortalRefs={summary.DistinctPortalRefCount} maxPortalIndex={summary.MaxPortalIndex}");
}

static void PrintWmoPortalGroupRangeSummary(WmoPortalGroupRangeSummary summary)
{
	Console.WriteLine($"MOPR->MOGI: refs={summary.RefCount} groups={summary.GroupCount} coveredRefs={summary.CoveredRefCount} outOfRangeRefs={summary.OutOfRangeRefCount} distinctGroupRefs={summary.DistinctGroupRefCount} maxGroupIndex={summary.MaxGroupIndex}");
}

static void PrintWmoVisibleVertexSummary(WmoVisibleVertexSummary summary)
{
	Console.WriteLine($"MOVV: payloadBytes={summary.PayloadSizeBytes} vertices={summary.VertexCount} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoVisibleBlockSummary(WmoVisibleBlockSummary summary)
{
	Console.WriteLine($"MOVB: payloadBytes={summary.PayloadSizeBytes} blocks={summary.BlockCount} vertexRefs={summary.TotalVertexRefs} blockSizeRange={summary.MinVerticesPerBlock}-{summary.MaxVerticesPerBlock} firstVertexRange={summary.MinFirstVertex}-{summary.MaxFirstVertex} maxVertexEnd={summary.MaxVertexEnd}");
}

static void PrintWmoVisibleBlockReferenceSummary(WmoVisibleBlockReferenceSummary summary)
{
	Console.WriteLine($"MOVB->MOVV: blocks={summary.BlockCount} vertices={summary.VisibleVertexCount} zeroVertexBlocks={summary.ZeroVertexBlockCount} coveredBlocks={summary.CoveredBlockCount} outOfRangeBlocks={summary.OutOfRangeBlockCount} maxVertexEnd={summary.MaxVertexEnd}");
}

static void PrintWmoLightSummary(WmoLightSummary summary)
{
	Console.WriteLine($"MOLT: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} distinctTypes={summary.DistinctTypeCount} attenuated={summary.AttenuatedCount} intensityRange=[{summary.MinIntensity:F3}, {summary.MaxIntensity:F3}] maxAttenEnd={summary.MaxAttenEnd:F3} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoFogSummary(WmoFogSummary summary)
{
	Console.WriteLine($"MFOG: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} nonZeroFlags={summary.NonZeroFlagCount} minSmallRadius={summary.MinSmallRadius:F3} maxLargeRadius={summary.MaxLargeRadius:F3} maxFogEnd={summary.MaxFogEnd:F3} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoOpaqueChunkSummary(WmoOpaqueChunkSummary summary)
{
	Console.WriteLine($"{summary.ChunkId}: payloadBytes={summary.PayloadSizeBytes}");
}

static void PrintWmoDoodadSetRangeSummary(WmoDoodadSetRangeSummary summary)
{
	Console.WriteLine($"MODS->MODD: sets={summary.EntryCount} placements={summary.PlacementCount} emptySets={summary.EmptySetCount} coveredSets={summary.FullyCoveredSetCount} outOfRangeSets={summary.OutOfRangeSetCount} maxRangeEnd={summary.MaxRangeEnd}");
}

static void PrintWmoGroupNameReferenceSummary(WmoGroupNameReferenceSummary summary)
{
	Console.WriteLine($"MOGI->MOGN: entries={summary.EntryCount} resolvedNames={summary.ResolvedNameCount} unresolvedNames={summary.UnresolvedNameCount} distinctResolvedNames={summary.DistinctResolvedNameCount} maxNameLength={summary.MaxResolvedNameLength}");
}

static void PrintWmoDoodadNameReferenceSummary(WmoDoodadNameReferenceSummary summary)
{
	Console.WriteLine($"MODD->MODN: entries={summary.EntryCount} resolvedNames={summary.ResolvedNameCount} unresolvedNames={summary.UnresolvedNameCount} distinctResolvedNames={summary.DistinctResolvedNameCount} maxNameLength={summary.MaxResolvedNameLength}");
}

static void PrintWmoGroupSummary(WmoGroupSummary summary)
{
	Console.WriteLine("WowViewer.Tool.Inspect WMO group report");
	Console.WriteLine($"Input: {summary.SourcePath}");
	Console.WriteLine($"Version: {summary.Version?.ToString() ?? "n/a"}");
	Console.WriteLine($"Header: bytes={summary.HeaderSizeBytes} nameOff={summary.NameOffset} descOff={summary.DescriptiveNameOffset} flags=0x{summary.Flags:X8} portals={summary.PortalCount}@{summary.PortalStart} liquid={summary.GroupLiquid}");
	Console.WriteLine($"Geometry: faces={summary.FaceMaterialCount} vertices={summary.VertexCount} indices={summary.IndexCount} normals={summary.NormalCount} primaryUv={summary.PrimaryUvCount} extraUvSets={summary.AdditionalUvSetCount} batches={summary.BatchCount}/{summary.DeclaredBatchCount} vertexColors={summary.VertexColorCount} doodadRefs={summary.DoodadRefCount} lightRefs={summary.LightRefCount} bspNodes={summary.BspNodeCount} bspFaceRefs={summary.BspFaceRefCount} hasLiquid={summary.HasLiquid}");
	Console.WriteLine($"Bounds: min={FormatVector(summary.BoundsMin)} max={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoGroupLiquidSummary(WmoGroupLiquidSummary summary)
{
	Console.WriteLine($"MLIQ: payloadBytes={summary.PayloadSizeBytes} verts={summary.XVertexCount}x{summary.YVertexCount} tiles={summary.XTileCount}x{summary.YTileCount} corner={FormatVector(summary.Corner)} materialId={summary.MaterialId} heights={summary.HeightCount} range=[{summary.MinHeight:F2}, {summary.MaxHeight:F2}] visibleTiles={summary.VisibleTileCount}/{summary.TileCount} tileFlags={summary.TileFlagByteCount} liquidType={summary.LiquidType}");
}

static void PrintWmoGroupBatchSummary(WmoGroupBatchSummary summary)
{
	Console.WriteLine($"MOBA: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} hasMaterialIds={summary.HasMaterialIds} distinctMaterials={summary.DistinctMaterialIdCount} highestMaterialId={summary.HighestMaterialId} totalIndexCount={summary.TotalIndexCount} firstIndexRange={summary.MinFirstIndex}-{summary.MaxFirstIndex} maxIndexEnd={summary.MaxIndexEnd} flaggedBatches={summary.FlaggedBatchCount}");
}

static void PrintWmoGroupFaceMaterialSummary(WmoGroupFaceMaterialSummary summary)
{
	Console.WriteLine($"MOPY: payloadBytes={summary.PayloadSizeBytes} entryBytes={summary.EntrySizeBytes} faces={summary.FaceCount} distinctMaterials={summary.DistinctMaterialIdCount} highestMaterialId={summary.HighestMaterialId} hiddenFaces={summary.HiddenFaceCount} flaggedFaces={summary.FlaggedFaceCount}");
}

static void PrintWmoGroupUvSummary(WmoGroupUvSummary summary)
{
	Console.WriteLine($"MOTV: payloadBytes={summary.PrimaryPayloadSizeBytes} primaryUv={summary.PrimaryUvCount} rangeU=[{summary.MinU:F3}, {summary.MaxU:F3}] rangeV=[{summary.MinV:F3}, {summary.MaxV:F3}] extraUvSets={summary.AdditionalUvSetCount} totalExtraUv={summary.TotalAdditionalUvCount} maxExtraUv={summary.MaxAdditionalUvCount}");
}

static void PrintWmoGroupVertexColorSummary(WmoGroupVertexColorSummary summary)
{
	Console.WriteLine($"MOCV: payloadBytes={summary.PrimaryPayloadSizeBytes} primaryColors={summary.PrimaryColorCount} rangeR=[{summary.MinRed}, {summary.MaxRed}] rangeG=[{summary.MinGreen}, {summary.MaxGreen}] rangeB=[{summary.MinBlue}, {summary.MaxBlue}] rangeA=[{summary.MinAlpha}, {summary.MaxAlpha}] avgA={summary.AverageAlpha} extraColorSets={summary.AdditionalColorSetCount} totalExtraColors={summary.TotalAdditionalColorCount} maxExtraColors={summary.MaxAdditionalColorCount}");
}

static void PrintWmoGroupDoodadRefSummary(WmoGroupDoodadRefSummary summary)
{
	Console.WriteLine($"MODR: payloadBytes={summary.PayloadSizeBytes} refs={summary.RefCount} distinctRefs={summary.DistinctRefCount} refRange={summary.MinRef}-{summary.MaxRef} duplicateRefs={summary.DuplicateRefCount}");
}

static void PrintWmoGroupLightRefSummary(WmoGroupLightRefSummary summary)
{
	Console.WriteLine($"MOLR: payloadBytes={summary.PayloadSizeBytes} refs={summary.RefCount} distinctRefs={summary.DistinctRefCount} refRange={summary.MinRef}-{summary.MaxRef} duplicateRefs={summary.DuplicateRefCount}");
}

static void PrintWmoGroupIndexSummary(WmoGroupIndexSummary summary)
{
	Console.WriteLine($"{summary.ChunkId}: payloadBytes={summary.PayloadSizeBytes} indices={summary.IndexCount} triangles={summary.TriangleCount} distinctIndices={summary.DistinctIndexCount} indexRange={summary.MinIndex}-{summary.MaxIndex} degenerateTriangles={summary.DegenerateTriangleCount}");
}

static void PrintWmoGroupBspNodeSummary(WmoGroupBspNodeSummary summary)
{
	Console.WriteLine($"MOBN: payloadBytes={summary.PayloadSizeBytes} nodes={summary.NodeCount} leafNodes={summary.LeafNodeCount} branchNodes={summary.BranchNodeCount} childRefs={summary.ChildReferenceCount} noChildRefs={summary.NoChildReferenceCount} outOfRangeChildRefs={summary.OutOfRangeChildReferenceCount} faceCountRange={summary.MinFaceCount}-{summary.MaxFaceCount} faceStartRange={summary.MinFaceStart}-{summary.MaxFaceStart} maxFaceEnd={summary.MaxFaceEnd} planeDistRange=[{summary.MinPlaneDistance:F3}, {summary.MaxPlaneDistance:F3}]");
}

static void PrintWmoGroupBspFaceSummary(WmoGroupBspFaceSummary summary)
{
	Console.WriteLine($"MOBR: payloadBytes={summary.PayloadSizeBytes} refs={summary.RefCount} distinctRefs={summary.DistinctFaceRefCount} refRange={summary.MinFaceRef}-{summary.MaxFaceRef} duplicateRefs={summary.DuplicateFaceRefCount}");
}

static void PrintWmoGroupBspFaceRangeSummary(WmoGroupBspFaceRangeSummary summary)
{
	Console.WriteLine($"MOBN->MOBR: nodes={summary.NodeCount} faceRefs={summary.FaceRefCount} zeroFaceNodes={summary.ZeroFaceNodeCount} coveredNodes={summary.CoveredNodeCount} outOfRangeNodes={summary.OutOfRangeNodeCount} maxFaceEnd={summary.MaxFaceEnd}");
}

static void PrintWmoGroupVertexSummary(WmoGroupVertexSummary summary)
{
	Console.WriteLine($"MOVT: payloadBytes={summary.PayloadSizeBytes} vertices={summary.VertexCount} boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
}

static void PrintWmoGroupNormalSummary(WmoGroupNormalSummary summary)
{
	Console.WriteLine($"MONR: payloadBytes={summary.PayloadSizeBytes} normals={summary.NormalCount} rangeX=[{summary.MinX:F3}, {summary.MaxX:F3}] rangeY=[{summary.MinY:F3}, {summary.MaxY:F3}] rangeZ=[{summary.MinZ:F3}, {summary.MaxZ:F3}] lengthRange=[{summary.MinLength:F3}, {summary.MaxLength:F3}] avgLength={summary.AverageLength:F3} nearUnit={summary.NearUnitCount}");
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
