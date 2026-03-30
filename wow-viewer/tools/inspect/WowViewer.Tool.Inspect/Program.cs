using System.Numerics;
using System.Text.Json;
using WowViewer.Core.Blp;
using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Blp;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Mdx;
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
	case "blp":
		RunBlp(tail);
		break;
	case "mdx":
		RunMdx(tail);
		break;
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

static void RunBlp(string[] args)
{
	if (args.Length == 0)
	{
		ShowBlpUsage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "inspect":
			RunBlpInspect(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown blp command '{command}'.");
			ShowBlpUsage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunBlpInspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? virtualPath = GetOption(args, "--virtual-path", "-v");
	string? listfilePath = GetOption(args, "--listfile", "-l") ?? TryFindDefaultListfilePath();
	if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
		virtualPath = input;

	if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
	{
		Console.Error.WriteLine("Error: provide --input <file.blp> or --archive-root <dir> with --virtual-path <path/to/file.blp>.");
		Environment.ExitCode = 1;
		return;
	}

	byte[]? archivedBytes = null;
	string sourceLabel = !string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath)
		? virtualPath
		: input!;
	Stream OpenInputStream()
	{
		if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
		{
			archivedBytes ??= ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], listfilePath);
			return new MemoryStream(archivedBytes, writable: false);
		}

		if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
			return File.OpenRead(input);

		archivedBytes ??= AlphaArchiveReader.ReadWithMpqFallback(input!)
			?? throw new FileNotFoundException($"Could not read inspect input '{input}' directly or from a companion MPQ archive.", input);
		return new MemoryStream(archivedBytes, writable: false);
	}

	BlpSummary summary;
	using (Stream stream = OpenInputStream())
		summary = BlpSummaryReader.Read(stream, sourceLabel);

	PrintBlpSummary(summary);
}

static void RunMdx(string[] args)
{
	if (args.Length == 0)
	{
		ShowMdxUsage();
		Environment.ExitCode = 1;
		return;
	}

	string command = args[0].ToLowerInvariant();
	string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "export-json":
			RunMdxExportJson(tail);
			break;
		case "chunk-carriers":
			RunMdxChunkCarriers(tail);
			break;
		case "inspect":
			RunMdxInspect(tail);
			break;
		default:
			Console.Error.WriteLine($"Unknown mdx command '{command}'.");
			ShowMdxUsage();
			Environment.ExitCode = 1;
			break;
	}
}

static void RunMdxInspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? virtualPath = GetOption(args, "--virtual-path", "-v");
	string? listfilePath = GetOption(args, "--listfile", "-l") ?? TryFindDefaultListfilePath();
	if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
		virtualPath = input;

	if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
	{
		Console.Error.WriteLine("Error: provide --input <file.mdx> or --archive-root <dir> with --virtual-path <path/to/file.mdx>.");
		Environment.ExitCode = 1;
		return;
	}

	byte[]? archivedBytes = null;
	string sourceLabel = !string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath)
		? virtualPath
		: input!;
	Stream OpenInputStream()
	{
		if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
		{
			archivedBytes ??= ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], listfilePath);
			return new MemoryStream(archivedBytes, writable: false);
		}

		if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
			return File.OpenRead(input);

		archivedBytes ??= AlphaArchiveReader.ReadWithMpqFallback(input!)
			?? throw new FileNotFoundException($"Could not read inspect input '{input}' directly or from a companion MPQ archive.", input);
		return new MemoryStream(archivedBytes, writable: false);
	}

	MdxSummary summary;
	using (Stream stream = OpenInputStream())
		summary = MdxSummaryReader.Read(stream, sourceLabel);

	PrintMdxSummary(summary);
}

static void RunMdxExportJson(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? virtualPath = GetOption(args, "--virtual-path", "-v");
	string? listfilePath = GetOption(args, "--listfile", "-l") ?? TryFindDefaultListfilePath();
	string? output = GetOption(args, "--output", "-o");
	bool includeGeometry = HasOption(args, "--include-geometry");
	bool includeCollision = HasOption(args, "--include-collision");
	bool includeHitTest = HasOption(args, "--include-hit-test");
	bool includeTextureAnimations = HasOption(args, "--include-texture-animations");
	if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
		virtualPath = input;

	if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
	{
		Console.Error.WriteLine("Error: provide --input <file.mdx> or --archive-root <dir> with --virtual-path <path/to/file.mdx>.");
		Environment.ExitCode = 1;
		return;
	}

	byte[]? archivedBytes = null;
	string sourceLabel = !string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath)
		? virtualPath
		: input!;
	Stream OpenInputStream()
	{
		if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
		{
			archivedBytes ??= ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], listfilePath);
			return new MemoryStream(archivedBytes, writable: false);
		}

		if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
			return File.OpenRead(input);

		archivedBytes ??= AlphaArchiveReader.ReadWithMpqFallback(input!)
			?? throw new FileNotFoundException($"Could not read inspect input '{input}' directly or from a companion MPQ archive.", input);
		return new MemoryStream(archivedBytes, writable: false);
	}

	MdxSummary summary;
	using (Stream stream = OpenInputStream())
		summary = MdxSummaryReader.Read(stream, sourceLabel);

	MdxGeometryFile? geometry = null;
	if (includeGeometry)
	{
		using Stream stream = OpenInputStream();
		geometry = MdxGeometryReader.Read(stream, sourceLabel);
	}

	MdxCollisionFile? collision = null;
	if (includeCollision)
	{
		using Stream stream = OpenInputStream();
		collision = MdxCollisionReader.Read(stream, sourceLabel);
	}

	MdxHitTestFile? hitTest = null;
	if (includeHitTest)
	{
		using Stream stream = OpenInputStream();
		hitTest = MdxHitTestReader.Read(stream, sourceLabel);
	}

	MdxTextureAnimationFile? textureAnimations = null;
	if (includeTextureAnimations)
	{
		using Stream stream = OpenInputStream();
		textureAnimations = MdxTextureAnimationReader.Read(stream, sourceLabel);
	}

	Dictionary<string, object?> payload = new(StringComparer.Ordinal)
	{
		["summary"] = summary,
	};

	if (geometry is not null)
		payload["geometry"] = geometry;

	if (collision is not null)
		payload["collision"] = collision;

	if (hitTest is not null)
		payload["hitTest"] = hitTest;

	if (textureAnimations is not null)
		payload["textureAnimations"] = textureAnimations;

	string json = JsonSerializer.Serialize(payload, new JsonSerializerOptions
	{
		WriteIndented = true,
		IncludeFields = true,
	});
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

static void RunMdxChunkCarriers(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? listfilePath = GetOption(args, "--listfile", "-l") ?? TryFindDefaultListfilePath();
	string? pathFilter = GetOption(args, "--path-filter", "-p");
	string? chunkText = GetOption(args, "--chunks", "-c") ?? GetOption(args, "--chunk", "-c");
	string? limitText = GetOption(args, "--limit", "-n");

	if (string.IsNullOrWhiteSpace(chunkText))
	{
		Console.Error.WriteLine("Error: provide --chunks <FOURCC[,FOURCC...]>.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(input) && string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: provide --input <file|directory> or --archive-root <game|data dir>.");
		Environment.ExitCode = 1;
		return;
	}

	if (!string.IsNullOrWhiteSpace(input) && !string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: choose either --input <file|directory> or --archive-root <game|data dir>, not both.");
		Environment.ExitCode = 1;
		return;
	}

	if (!string.IsNullOrWhiteSpace(limitText) && (!int.TryParse(limitText, out int parsedLimit) || parsedLimit <= 0))
	{
		Console.Error.WriteLine($"Error: invalid --limit value '{limitText}'.");
		Environment.ExitCode = 1;
		return;
	}

	int? limit = string.IsNullOrWhiteSpace(limitText) ? null : int.Parse(limitText);
	IReadOnlyList<FourCC> targetChunks;
	try
	{
		targetChunks = ParseMdxChunkIds(chunkText);
	}
	catch (ArgumentException ex)
	{
		Console.Error.WriteLine($"Error: {ex.Message}");
		Environment.ExitCode = 1;
		return;
	}

	List<string> parseFailures = [];
	List<string> readMisses = [];
	int scanned = 0;
	int matched = 0;

	Console.WriteLine($"MDX chunk carrier scan: chunks={string.Join(',', targetChunks.Select(static chunk => chunk.ToString()))} source={(archiveRoot ?? input)!}");

	if (!string.IsNullOrWhiteSpace(archiveRoot))
	{
		using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
		ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, [archiveRoot], listfilePath);

		IEnumerable<string> candidates = archiveCatalog
			.GetAllKnownFiles()
			.Where(static path => path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase));

		if (!string.IsNullOrWhiteSpace(pathFilter))
			candidates = candidates.Where(path => path.Contains(pathFilter, StringComparison.OrdinalIgnoreCase));

		foreach (string path in candidates.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase))
		{
			if (limit.HasValue && scanned >= limit.Value)
				break;

			scanned++;
			byte[]? bytes = archiveCatalog.ReadFile(path);
			if (bytes is null)
			{
				TrackScanIssue(readMisses, $"{path}: archive read returned no bytes");
				continue;
			}

			using MemoryStream stream = new(bytes, writable: false);
			matched += PrintMdxCarrierMatch(path, stream, targetChunks, parseFailures);
		}
	}
	else
	{
		IEnumerable<string> candidates = EnumerateMdxInputPaths(input!);
		if (!string.IsNullOrWhiteSpace(pathFilter))
			candidates = candidates.Where(path => path.Contains(pathFilter, StringComparison.OrdinalIgnoreCase));

		foreach (string path in candidates.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase))
		{
			if (limit.HasValue && scanned >= limit.Value)
				break;

			scanned++;
			using FileStream stream = File.OpenRead(path);
			matched += PrintMdxCarrierMatch(path, stream, targetChunks, parseFailures);
		}
	}

	Console.WriteLine($"Scanned={scanned} matched={matched} readMisses={readMisses.Count} parseFailures={parseFailures.Count}");
	if (matched == 0)
		Console.WriteLine("No matching carriers found.");

	if (readMisses.Count > 0)
	{
		Console.WriteLine("Read misses:");
		foreach (string miss in readMisses)
			Console.WriteLine($"  {miss}");
	}

	if (parseFailures.Count > 0)
	{
		Console.WriteLine("Parse failures:");
		foreach (string failure in parseFailures)
			Console.WriteLine($"  {failure}");
	}
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
	bool dumpTexChunks = HasOption(args, "--dump-tex-chunks");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input map file is required.");
		Environment.ExitCode = 1;
		return;
	}

	MapFileSummary summary = MapFileSummaryReader.Read(input);
	PrintMapSummary(summary, dumpTexChunks);
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
		case "match":
			RunPm4Match(tail);
			break;
			case "hierarchy":
				RunPm4Hierarchy(tail);
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

static void RunPm4Match(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
	string? placements = GetOption(args, "--placements", "-p") ?? GetOption(args, "--adt-obj", "-a");
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? listfilePath = GetOption(args, "--listfile", "-l") ?? TryFindDefaultListfilePath();
	string? output = GetOption(args, "--output", "-o");
	string? objectOutputDirectory = GetOption(args, "--object-output-dir", "-d");
	string? maxMatchesText = GetOption(args, "--max-matches", "-n");
	string? searchRangeText = GetOption(args, "--search-range", "-s");
	int maxMatches = 8;
	float searchRange = 128f;
	if (!string.IsNullOrWhiteSpace(maxMatchesText) && (!int.TryParse(maxMatchesText, out maxMatches) || maxMatches <= 0))
	{
		Console.Error.WriteLine("Error: --max-matches must be a positive integer.");
		Environment.ExitCode = 1;
		return;
	}
	if (!string.IsNullOrWhiteSpace(searchRangeText) && (!float.TryParse(searchRangeText, out searchRange) || searchRange <= 0f))
	{
		Console.Error.WriteLine("Error: --search-range must be a positive number.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(archiveRoot))
	{
		Console.Error.WriteLine("Error: --archive-root is required for pm4 match so WMO/M2 assets can be read from game archives.");
		Environment.ExitCode = 1;
		return;
	}

	if (string.IsNullOrWhiteSpace(placements))
	{
		if (!Pm4CoordinateService.TryParseTileCoordinates(input, out int tileX, out int tileY))
		{
			Console.Error.WriteLine("Error: could not derive tile coordinates from the PM4 filename; provide --placements <tile_obj0.adt> explicitly.");
			Environment.ExitCode = 1;
			return;
		}

		string fileName = Path.GetFileNameWithoutExtension(input);
		int lastUnderscore = fileName.LastIndexOf('_');
		int previousUnderscore = lastUnderscore > 0 ? fileName.LastIndexOf('_', lastUnderscore - 1) : -1;
		string mapName = previousUnderscore > 0 ? fileName[..previousUnderscore] : fileName;
		placements = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(input)) ?? string.Empty, $"{mapName}_{tileX}_{tileY}_obj0.adt");
	}

	if (!File.Exists(placements))
	{
		Console.Error.WriteLine($"Error: placement source '{placements}' does not exist.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4MatchResult result = Pm4MatchSupport.Run(input, placements, archiveRoot, listfilePath, maxMatches, searchRange);
	bool wroteArtifact = false;
	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, Pm4MatchSupport.ToJson(result));
		Console.WriteLine($"Wrote {outputPath}");
		wroteArtifact = true;
	}

	if (!string.IsNullOrWhiteSpace(objectOutputDirectory))
	{
		IReadOnlyList<string> writtenPaths = Pm4MatchSupport.WriteObjectArtifacts(result, objectOutputDirectory);
		Console.WriteLine($"Wrote {writtenPaths.Count} PM4 match artifact files under {Path.GetFullPath(objectOutputDirectory)}");
		wroteArtifact = true;
	}

	if (wroteArtifact)
		return;

	Pm4MatchSupport.Print(result);
}

static void RunWmoInspect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? archiveRoot = GetOption(args, "--archive-root", "-r");
	string? virtualPath = GetOption(args, "--virtual-path", "-v");
	string? listfilePath = GetOption(args, "--listfile", "-l") ?? TryFindDefaultListfilePath();
	bool dumpLights = HasOption(args, "--dump-lights");
	if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
		virtualPath = input;

	if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
	{
		Console.Error.WriteLine("Error: provide --input <file.wmo|file.wmo.MPQ> or --archive-root <dir> with --virtual-path <world/...wmo>.");
		Environment.ExitCode = 1;
		return;
	}

	byte[]? archivedBytes = null;
	string sourceLabel = !string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath)
		? virtualPath
		: input!;
	Stream OpenInputStream()
	{
		if (!string.IsNullOrWhiteSpace(archiveRoot) && !string.IsNullOrWhiteSpace(virtualPath))
		{
			archivedBytes ??= ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], listfilePath);
			return new MemoryStream(archivedBytes, writable: false);
		}

		if (File.Exists(input) && !input.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
			return File.OpenRead(input);

		archivedBytes ??= AlphaArchiveReader.ReadWithMpqFallback(input!)
			?? throw new FileNotFoundException($"Could not read inspect input '{input}' directly or from a companion MPQ archive.", input);
		return new MemoryStream(archivedBytes, writable: false);
	}

	T ReadInput<T>(Func<Stream, string, T> reader)
	{
		using Stream stream = OpenInputStream();
		return reader(stream, sourceLabel);
	}

	WowFileDetection detection;
	using (Stream detectionStream = OpenInputStream())
		detection = WowFileDetector.Detect(detectionStream, sourceLabel);

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
				if (dumpLights)
				{
					IReadOnlyList<WmoLightDetail> lightDetails = ReadInput(WmoLightDetailReader.Read);
					PrintWmoLightDetails(lightDetails);
				}
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

static void RunPm4Hierarchy(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4TileObjectHypothesisReport report = Pm4ResearchHierarchyAnalyzer.Analyze(Pm4ResearchReader.ReadFile(input));
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

	PrintPm4HierarchyReport(report);
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
	string? ck24Text = GetOption(args, "--ck24", "-k");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input PM4 file is required.");
		Environment.ExitCode = 1;
		return;
	}

	Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(input);
	object report;
	if (!string.IsNullOrWhiteSpace(ck24Text))
	{
		if (!TryParseUInt32Flexible(ck24Text, out uint ck24))
		{
			Console.Error.WriteLine($"Error: invalid --ck24 value '{ck24Text}'. Use decimal or 0x-prefixed hex.");
			Environment.ExitCode = 1;
			return;
		}

		report = Pm4Ck24ForensicsAnalyzer.Analyze(document, ck24);
	}
	else
	{
		report = Pm4ResearchAnalyzer.Analyze(document);
	}

	string json = JsonSerializer.Serialize(report, new JsonSerializerOptions
	{
		WriteIndented = true,
		IncludeFields = true,
	});

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

static bool TryParseUInt32Flexible(string value, out uint parsed)
{
	if (value.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
		return uint.TryParse(value[2..], System.Globalization.NumberStyles.HexNumber, System.Globalization.CultureInfo.InvariantCulture, out parsed);

	return uint.TryParse(value, out parsed);
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

static string? GetFirstPositionalArgument(string[] args)
{
	for (int index = 0; index < args.Length; index++)
	{
		string current = args[index];
		if (current.StartsWith('-'))
		{
			index++;
			continue;
		}

		return current;
	}

	return null;
}

static IReadOnlyList<FourCC> ParseMdxChunkIds(string chunkText)
{
	string[] tokens = chunkText
		.Split([',', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

	if (tokens.Length == 0)
		throw new ArgumentException("--chunks requires at least one four-character chunk id.");

	List<FourCC> chunks = [];
	HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
	foreach (string token in tokens)
	{
		if (token.Length != 4)
			throw new ArgumentException($"Chunk id '{token}' must be exactly 4 ASCII characters.");

		if (!token.All(static ch => ch <= 0x7F && !char.IsWhiteSpace(ch)))
			throw new ArgumentException($"Chunk id '{token}' must contain only non-whitespace ASCII characters.");

		if (!seen.Add(token))
			continue;

		chunks.Add(FourCC.FromString(token.ToUpperInvariant()));
	}

	return chunks;
}

static IEnumerable<string> EnumerateMdxInputPaths(string input)
{
	if (File.Exists(input))
	{
		if (!input.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
			throw new FileNotFoundException($"Input file '{input}' is not an .mdx file.", input);

		yield return Path.GetFullPath(input);
		yield break;
	}

	if (!Directory.Exists(input))
		throw new DirectoryNotFoundException($"Could not find input path '{input}'.");

	foreach (string path in Directory.EnumerateFiles(input, "*.mdx", SearchOption.AllDirectories))
		yield return path;
}

static int PrintMdxCarrierMatch(string sourcePath, Stream stream, IReadOnlyList<FourCC> targetChunks, List<string> parseFailures)
{
	try
	{
		MdxSummary summary = MdxSummaryReader.Read(stream, sourcePath);
		List<string> matchedChunks = targetChunks
			.Where(target => summary.Chunks.Any(chunk => chunk.Id == target))
			.Select(static chunk => chunk.ToString())
			.ToList();

		if (matchedChunks.Count == 0)
			return 0;

		Console.WriteLine($"CARRIER: path={sourcePath} matchedChunks={string.Join(',', matchedChunks)} chunkCount={summary.ChunkCount} knownChunks={summary.KnownChunkCount} unknownChunks={summary.UnknownChunkCount}");
		return 1;
	}
	catch (Exception ex) when (ex is InvalidDataException or IOException)
	{
		TrackParseFailure(parseFailures, $"{sourcePath}: {ex.Message}");
		return 0;
	}
}

static void TrackParseFailure(List<string> parseFailures, string message)
{
	TrackScanIssue(parseFailures, message);
}

static void TrackScanIssue(List<string> issues, string message)
{
	if (issues.Count < 10)
		issues.Add(message);
}

static bool HasOption(string[] args, string name)
{
	return args.Any(arg => string.Equals(arg, name, StringComparison.OrdinalIgnoreCase));
}

static string? TryFindDefaultListfilePath()
{
	DirectoryInfo? current = new(AppContext.BaseDirectory);
	while (current is not null)
	{
		if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
		{
			string candidate = Path.Combine(current.FullName, "libs", "wowdev", "wow-listfile", "listfile.txt");
			return File.Exists(candidate) ? candidate : null;
		}

		current = current.Parent;
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
	if (report.Terminology.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Terminology:");
		foreach (Pm4TerminologyEntry entry in report.Terminology)
		{
			string alias = string.IsNullOrWhiteSpace(entry.LocalAlias) ? "" : $" -> local alias {entry.LocalAlias}";
			Console.WriteLine($"  {entry.RawField}{alias} ({entry.Confidence})");
			Console.WriteLine($"    {entry.Notes}");
		}
	}
	Console.WriteLine();
	Console.WriteLine("Top MSUR._0x1c-derived key24 groups:");
	if (report.TopCk24Groups.Count == 0)
	{
		Console.WriteLine("  none");
	}
	else
	{
		foreach (Pm4Ck24Summary summary in report.TopCk24Groups.Take(10))
		{
			Console.WriteLine($"  key24=0x{summary.Ck24:X6} type=0x{summary.Ck24Type:X2} low16={summary.Ck24ObjectId} surfaces={summary.SurfaceCount} indices={summary.TotalIndexCount} mdos={summary.DistinctMdosCount}");
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

static string FormatQuaternion(System.Numerics.Quaternion value)
{
	return $"({value.X:F3}, {value.Y:F3}, {value.Z:F3}, {value.W:F3})";
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

static void PrintPm4HierarchyReport(Pm4TileObjectHypothesisReport report)
{
	Console.WriteLine("WowViewer.Tool.Inspect PM4 hierarchy report");
	Console.WriteLine($"Input: {report.SourcePath ?? "<memory>"}");
	Console.WriteLine($"Version: {report.Version}");
	Console.WriteLine($"Tile: {(report.TileX.HasValue && report.TileY.HasValue ? $"{report.TileX}_{report.TileY}" : "n/a")}");
	Console.WriteLine($"Distinct CK24 groups: {report.Ck24GroupCount}");
	Console.WriteLine($"Hypothesis objects: {report.TotalHypothesisCount}");
	Console.WriteLine();
	Console.WriteLine("Top hierarchy candidates:");
	foreach (Pm4ObjectHypothesis hypothesis in report.Objects.Take(12))
	{
		Pm4ForensicsPlacementComparison placement = hypothesis.PlacementComparison;
		string headingText = placement.MprlHeadingMeanDegrees.HasValue
			? $" heading={placement.MprlHeadingMeanDegrees.Value:F2} delta={placement.HeadingDeltaDegrees?.ToString("F2") ?? "n/a"}"
			: string.Empty;
		Console.WriteLine($"  {hypothesis.Family}#{hypothesis.FamilyObjectIndex}: ck24=0x{hypothesis.Ck24:X6} surfaces={hypothesis.SurfaceCount} indices={hypothesis.TotalIndexCount} linkGroups={hypothesis.MslkGroupObjectIds.Count} dominantGroup=0x{hypothesis.DominantLinkGroupObjectId:X} linkedMPRL={hypothesis.MprlFootprint.LinkedRefCount}/{hypothesis.MprlFootprint.LinkedInBoundsCount} mode={placement.CoordinateMode} planar=(swap={placement.PlanarTransform.SwapPlanarAxes}, invertU={placement.PlanarTransform.InvertU}, invertV={placement.PlanarTransform.InvertV}) frameYaw={placement.FrameYawDegrees:F2}{headingText}");
	}

	if (report.Notes.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Notes:");
		foreach (string note in report.Notes)
			Console.WriteLine($"  {note}");
	}

	if (report.Diagnostics.Count > 0)
	{
		Console.WriteLine();
		Console.WriteLine("Diagnostics:");
		foreach (string diagnostic in report.Diagnostics.Take(12))
			Console.WriteLine($"  {diagnostic}");
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

static void PrintMapSummary(MapFileSummary summary, bool dumpTexChunks)
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
	else if (summary.Kind is MapFileKind.Adt or MapFileKind.AdtTex or MapFileKind.AdtObj or MapFileKind.AdtLod)
	{
		AdtTileFamily family = AdtTileFamilyResolver.Resolve(summary.SourcePath);
		Console.WriteLine($"ADT family: root={(family.HasRoot ? "present" : "missing")} tex0={(family.HasTex0 ? "present" : "missing")} obj0={(family.HasObj0 ? "present" : "missing")} lod={(family.HasLod ? "present" : "missing")} textureSource={FormatMapFileKind(family.TextureSourceKind)} placementSource={FormatMapFileKind(family.PlacementSourceKind)}");
	}

	if (summary.Kind is MapFileKind.Adt or MapFileKind.AdtTex or MapFileKind.AdtObj)
	{
		using FileStream stream = File.OpenRead(summary.SourcePath);
		AdtSummary adtSummary = AdtSummaryReader.Read(stream, summary);
		Console.WriteLine($"ADT semantics: kind={adtSummary.Kind} terrainChunks={adtSummary.TerrainChunkCount} textures={adtSummary.TextureNameCount} doodadNames={adtSummary.ModelNameCount} wmoNames={adtSummary.WorldModelNameCount} doodadPlacements={adtSummary.ModelPlacementCount} wmoPlacements={adtSummary.WorldModelPlacementCount} hasMfbo={adtSummary.HasFlightBounds} hasMh2o={adtSummary.HasWater} hasMamp={adtSummary.HasTextureParams} hasMtxf={adtSummary.HasTextureFlags}");
		AdtMcnkSummary mcnkSummary = AdtMcnkSummaryReader.Read(stream, summary);
		Console.WriteLine($"ADT MCNK semantics: mcnk={mcnkSummary.McnkCount} zero={mcnkSummary.ZeroLengthMcnkCount} headerLike={mcnkSummary.HeaderLikeMcnkCount} distinctIndex={mcnkSummary.DistinctIndexCount} duplicateIndex={mcnkSummary.DuplicateIndexCount} areaIds={mcnkSummary.DistinctAreaIdCount} holes={mcnkSummary.ChunksWithHoles} liquidFlags={mcnkSummary.ChunksWithLiquidFlags} mccvFlags={mcnkSummary.ChunksWithMccvFlag} mcvt={mcnkSummary.ChunksWithMcvt} mcnr={mcnkSummary.ChunksWithMcnr} mcly={mcnkSummary.ChunksWithMcly} mcal={mcnkSummary.ChunksWithMcal} mcsh={mcnkSummary.ChunksWithMcsh} mccv={mcnkSummary.ChunksWithMccv} mclq={mcnkSummary.ChunksWithMclq} mcrd={mcnkSummary.ChunksWithMcrd} mcrw={mcnkSummary.ChunksWithMcrw} totalLayers={mcnkSummary.TotalLayerCount} maxLayers={mcnkSummary.MaxLayerCount} multiLayerChunks={mcnkSummary.ChunksWithMultipleLayers}");
		if (summary.Kind is MapFileKind.Adt or MapFileKind.AdtTex)
		{
			AdtMcalSummary mcalSummary = AdtMcalSummaryReader.Read(stream, summary);
			Console.WriteLine($"ADT MCAL semantics: profile={mcalSummary.DecodeProfile} mcnkWithLayers={mcalSummary.McnkWithLayerTableCount} overlayLayers={mcalSummary.OverlayLayerCount} decodedLayers={mcalSummary.DecodedLayerCount} missingPayloadLayers={mcalSummary.MissingPayloadLayerCount} decodeFailures={mcalSummary.DecodeFailureCount} compressed={mcalSummary.CompressedLayerCount} bigAlpha={mcalSummary.BigAlphaLayerCount} bigAlphaFixed={mcalSummary.BigAlphaFixedLayerCount} packed4={mcalSummary.PackedLayerCount}");
			if (dumpTexChunks && summary.Kind is MapFileKind.Adt or MapFileKind.AdtTex)
			{
				AdtTextureFile textureFile = AdtTextureReader.Read(stream, summary);
				PrintAdtTextureFile(textureFile);
			}
		}
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

static string FormatMapFileKind(MapFileKind? kind)
{
	return kind?.ToString() ?? "n/a";
}

static void PrintAdtTextureFile(AdtTextureFile textureFile)
{
	Console.WriteLine($"ADT texture detail: kind={textureFile.Kind} profile={textureFile.DecodeProfile} textures={textureFile.TextureNames.Count} chunks={textureFile.Chunks.Count}");
	foreach (AdtTextureChunk chunk in textureFile.Chunks)
	{
		if (chunk.Layers.Count == 0)
			continue;

		Console.WriteLine($"MCNK(texture)[{chunk.ChunkIndex}]: xy=({chunk.ChunkX},{chunk.ChunkY}) layers={chunk.Layers.Count} alphaBytes={chunk.AlphaPayloadBytes} doNotFixAlphaMap={chunk.DoNotFixAlphaMap} decodedLayers={chunk.DecodedLayerCount}");
		foreach (AdtTextureChunkLayer layer in chunk.Layers)
		{
			string texturePath = string.IsNullOrWhiteSpace(layer.TexturePath) ? "n/a" : layer.TexturePath;
			string alphaSummary = layer.DecodedAlpha is null
				? "alpha=n/a"
				: $"alpha={layer.DecodedAlpha.Encoding} bytes={layer.DecodedAlpha.SourceBytesConsumed}";
			Console.WriteLine($"MCNK(texture)[{chunk.ChunkIndex}].LAYER[{layer.Index}]: textureId={layer.TextureId} texture={texturePath} flags=0x{layer.Flags:X8} alphaOffset={layer.AlphaOffset} effectId={layer.EffectId} {alphaSummary}");
		}
	}
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

static void PrintWmoLightDetails(IReadOnlyList<WmoLightDetail> details)
{
	foreach (WmoLightDetail detail in details)
	{
		PrintWmoLightDetail(detail);
	}
}

static void PrintWmoLightDetail(WmoLightDetail detail)
{
	string headerFlagsText = detail.HeaderFlagsWord is ushort headerFlagsWord
		? $"0x{headerFlagsWord:X4}"
		: "n/a";
	string rotationText = detail.Rotation is System.Numerics.Quaternion rotation
		? FormatQuaternion(rotation)
		: "n/a";
	string rotationLengthText = detail.RotationLength is float rotationLength
		? rotationLength.ToString("F3")
		: "n/a";

	Console.WriteLine($"MOLT[{detail.LightIndex}]: offset={detail.PayloadOffset} entryBytes={detail.EntrySizeBytes} type={detail.LightType} attenuated={detail.UsesAttenuation} headerFlagsWord={headerFlagsText} color=0x{detail.ColorBgra:X8} position={FormatVector(detail.Position)} intensity={detail.Intensity:F3} attenStart={detail.AttenStart:F3} attenEnd={detail.AttenEnd:F3} rotation={rotationText} rotationLen={rotationLengthText}");
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
	Console.WriteLine($"MOLT: payloadBytes={summary.PayloadSizeBytes} entries={summary.EntryCount} distinctTypes={summary.DistinctTypeCount} attenuated={summary.AttenuatedCount} intensityRange=[{summary.MinIntensity:F3}, {summary.MaxIntensity:F3}] attenStartRange=[{summary.MinAttenStart:F3}, {summary.MaxAttenStart:F3}] maxAttenEnd={summary.MaxAttenEnd:F3} headerFlagsWordRange=[0x{summary.MinHeaderFlagsWord:X4}, 0x{summary.MaxHeaderFlagsWord:X4}] headerFlagsWordDistinct={summary.DistinctHeaderFlagsWordCount} headerFlagsWordNonZero={summary.NonZeroHeaderFlagsWordCount} rotationEntries={summary.RotationEntryCount} nonIdentityRotations={summary.NonIdentityRotationCount} rotationLenRange=[{summary.MinRotationLength:F3}, {summary.MaxRotationLength:F3}] boundsMin={FormatVector(summary.BoundsMin)} boundsMax={FormatVector(summary.BoundsMax)}");
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

static void PrintBlpSummary(BlpSummary summary)
{
	Console.WriteLine($"BLP: format={summary.Signature} version={summary.Version?.ToString() ?? "n/a"} compression={summary.Compression} alphaBits={summary.AlphaDepthBits} pixelFormat={summary.PixelFormat} mipType={summary.MipMapTypeRaw} size={summary.Width}x{summary.Height} headerBytes={summary.HeaderSizeBytes} paletteBytes={summary.PaletteSizeBytes} jpegHeaderBytes={summary.JpegHeaderSizeBytes} mips={summary.MipMaps.Count} inBoundsMips={summary.InBoundsMipLevelCount} outOfBoundsMips={summary.OutOfBoundsMipLevelCount} maxMipEnd={summary.MaxMipEndOffset}");
	foreach (BlpMipMapEntry mipMap in summary.MipMaps)
		PrintBlpMipMap(mipMap);
}

static void PrintBlpMipMap(BlpMipMapEntry mipMap)
{
	Console.WriteLine($"MIP[{mipMap.Level}]: size={mipMap.Width}x{mipMap.Height} offset={mipMap.Offset} bytes={mipMap.SizeBytes} inBounds={mipMap.IsInBounds}");
}

static void PrintMdxSummary(MdxSummary summary)
{
	string modelName = string.IsNullOrWhiteSpace(summary.ModelName) ? "n/a" : summary.ModelName;
	string blendTime = summary.BlendTime?.ToString() ?? "n/a";
	string boundsMin = summary.BoundsMin is Vector3 min ? $"({min.X:F3}, {min.Y:F3}, {min.Z:F3})" : "n/a";
	string boundsMax = summary.BoundsMax is Vector3 max ? $"({max.X:F3}, {max.Y:F3}, {max.Z:F3})" : "n/a";
	string collisionVertices = summary.Collision?.VertexCount.ToString() ?? "0";
	string collisionTriangles = summary.Collision?.TriangleCount.ToString() ?? "0";
	Console.WriteLine($"MDX: signature={summary.Signature} version={summary.Version?.ToString() ?? "n/a"} model={modelName} blendTime={blendTime} chunks={summary.ChunkCount} knownChunks={summary.KnownChunkCount} unknownChunks={summary.UnknownChunkCount} globalSequences={summary.GlobalSequenceCount} sequences={summary.SequenceCount} geosets={summary.GeosetCount} geosetAnimations={summary.GeosetAnimationCount} bones={summary.BoneCount} lights={summary.LightCount} helpers={summary.HelperCount} attachments={summary.AttachmentCount} particleEmitters2={summary.ParticleEmitter2Count} ribbons={summary.RibbonCount} cameras={summary.CameraCount} events={summary.EventCount} hitTestShapes={summary.HitTestShapeCount} collisionVertices={collisionVertices} collisionTriangles={collisionTriangles} pivotPoints={summary.PivotPointCount} textures={summary.TextureCount} replaceableTextures={summary.ReplaceableTextureCount} materials={summary.MaterialCount} materialLayers={summary.MaterialLayerCount} boundsMin={boundsMin} boundsMax={boundsMax}");
	for (int index = 0; index < summary.Chunks.Count; index++)
		PrintMdxChunkSummary(index, summary.Chunks[index]);
	for (int index = 0; index < summary.GlobalSequences.Count; index++)
		PrintMdxGlobalSequenceSummary(summary.GlobalSequences[index]);
	for (int index = 0; index < summary.Sequences.Count; index++)
		PrintMdxSequenceSummary(summary.Sequences[index]);
	for (int index = 0; index < summary.Geosets.Count; index++)
		PrintMdxGeosetSummary(summary.Geosets[index]);
	for (int index = 0; index < summary.GeosetAnimations.Count; index++)
		PrintMdxGeosetAnimationSummary(summary.GeosetAnimations[index]);
	for (int index = 0; index < summary.Bones.Count; index++)
		PrintMdxBoneSummary(summary.Bones[index]);
	for (int index = 0; index < summary.Lights.Count; index++)
		PrintMdxLightSummary(summary.Lights[index]);
	for (int index = 0; index < summary.Helpers.Count; index++)
		PrintMdxHelperSummary(summary.Helpers[index]);
	for (int index = 0; index < summary.Attachments.Count; index++)
		PrintMdxAttachmentSummary(summary.Attachments[index]);
	for (int index = 0; index < summary.ParticleEmitters2.Count; index++)
		PrintMdxParticleEmitter2Summary(summary.ParticleEmitters2[index]);
	for (int index = 0; index < summary.Ribbons.Count; index++)
		PrintMdxRibbonEmitterSummary(summary.Ribbons[index]);
	for (int index = 0; index < summary.Cameras.Count; index++)
		PrintMdxCameraSummary(summary.Cameras[index]);
	for (int index = 0; index < summary.Events.Count; index++)
		PrintMdxEventSummary(summary.Events[index]);
	for (int index = 0; index < summary.HitTestShapes.Count; index++)
		PrintMdxHitTestShapeSummary(summary.HitTestShapes[index]);
	if (summary.Collision is not null)
		PrintMdxCollisionSummary(summary.Collision);
	for (int index = 0; index < summary.PivotPoints.Count; index++)
		PrintMdxPivotPointSummary(summary.PivotPoints[index]);
	for (int index = 0; index < summary.Textures.Count; index++)
		PrintMdxTextureSummary(summary.Textures[index]);
	for (int index = 0; index < summary.Materials.Count; index++)
		PrintMdxMaterialSummary(summary.Materials[index]);
}

static void PrintMdxChunkSummary(int index, MdxChunkSummary chunk)
{
	Console.WriteLine($"CHUNK[{index}]: id={chunk.Id} payloadBytes={chunk.PayloadSizeBytes} headerOffset={chunk.HeaderOffset} dataOffset={chunk.DataOffset} known={chunk.IsKnownChunk}");
}

static void PrintMdxTextureSummary(MdxTextureSummary texture)
{
	string path = string.IsNullOrWhiteSpace(texture.Path) ? "n/a" : texture.Path;
	Console.WriteLine($"TEXS[{texture.Index}]: replaceableId={texture.ReplaceableId} flags=0x{texture.Flags:X8} path={path}");
}

static void PrintMdxGlobalSequenceSummary(MdxGlobalSequenceSummary globalSequence)
{
	Console.WriteLine($"GLBS[{globalSequence.Index}]: duration={globalSequence.Duration}");
}

static void PrintMdxSequenceSummary(MdxSequenceSummary sequence)
{
	string name = string.IsNullOrWhiteSpace(sequence.Name) ? "n/a" : sequence.Name;
	string blendTime = sequence.BlendTime?.ToString() ?? "n/a";
	string boundsMin = sequence.BoundsMin is Vector3 min ? $"({min.X:F3}, {min.Y:F3}, {min.Z:F3})" : "n/a";
	string boundsMax = sequence.BoundsMax is Vector3 max ? $"({max.X:F3}, {max.Y:F3}, {max.Z:F3})" : "n/a";
	string boundsRadius = sequence.BoundsRadius?.ToString("F3") ?? "n/a";
	Console.WriteLine($"SEQS[{sequence.Index}]: name={name} time=[{sequence.StartTime}, {sequence.EndTime}] duration={sequence.Duration} moveSpeed={sequence.MoveSpeed:F3} flags=0x{sequence.Flags:X8} frequency={sequence.Frequency:F3} replay=[{sequence.ReplayStart}, {sequence.ReplayEnd}] blendTime={blendTime} boundsMin={boundsMin} boundsMax={boundsMax} boundsRadius={boundsRadius}");
}

static void PrintMdxGeosetSummary(MdxGeosetSummary geoset)
{
	string boundsMin = geoset.BoundsMin is Vector3 min ? $"({min.X:F3}, {min.Y:F3}, {min.Z:F3})" : "n/a";
	string boundsMax = geoset.BoundsMax is Vector3 max ? $"({max.X:F3}, {max.Y:F3}, {max.Z:F3})" : "n/a";
	string boundsRadius = geoset.BoundsRadius?.ToString("F3") ?? "n/a";
	Console.WriteLine($"GEOS[{geoset.Index}]: vertices={geoset.VertexCount} normals={geoset.NormalCount} uvSets={geoset.UvSetCount} primaryUvs={geoset.PrimaryUvCount} primitiveTypes={geoset.PrimitiveTypeCount} faceGroups={geoset.FaceGroupCount} indices={geoset.IndexCount} triangles={geoset.TriangleCount} vertexGroups={geoset.VertexGroupCount} matrixGroups={geoset.MatrixGroupCount} matrixIndices={geoset.MatrixIndexCount} boneIndices={geoset.BoneIndexCount} boneWeights={geoset.BoneWeightCount} materialId={geoset.MaterialId} selectionGroup={geoset.SelectionGroup} flags=0x{geoset.Flags:X8} animExtents={geoset.AnimationExtentCount} boundsMin={boundsMin} boundsMax={boundsMax} boundsRadius={boundsRadius}");
}

static void PrintMdxGeosetAnimationSummary(MdxGeosetAnimationSummary geosetAnimation)
{
	Vector3 staticColor = geosetAnimation.StaticColor;
	string geosetId = geosetAnimation.GeosetId == uint.MaxValue ? "none(0xFFFFFFFF)" : geosetAnimation.GeosetId.ToString();
	Console.WriteLine($"GEOA[{geosetAnimation.Index}]: geosetId={geosetId} staticAlpha={geosetAnimation.StaticAlpha:F3} staticColor=({staticColor.X:F3}, {staticColor.Y:F3}, {staticColor.Z:F3}) flags=0x{geosetAnimation.Flags:X8} usesStaticColor={geosetAnimation.UsesStaticColor} alphaTrack={FormatMdxGeosetAnimationTrack(geosetAnimation.AlphaTrack)} colorTrack={FormatMdxGeosetAnimationTrack(geosetAnimation.ColorTrack)}");
}

static void PrintMdxBoneSummary(MdxBoneSummary bone)
{
	string parentId = bone.HasParent ? bone.ParentId.ToString() : "none(-1)";
	string geosetId = bone.UsesGeoset ? bone.GeosetId.ToString() : "none(0xFFFFFFFF)";
	string geosetAnimationId = bone.UsesGeosetAnimation ? bone.GeosetAnimationId.ToString() : "none(0xFFFFFFFF)";
	Console.WriteLine($"BONE[{bone.Index}]: name={bone.Name} objectId={bone.ObjectId} parentId={parentId} flags=0x{bone.Flags:X8} geosetId={geosetId} geosetAnimId={geosetAnimationId} translationTrack={FormatMdxNodeTrack(bone.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(bone.RotationTrack)} scalingTrack={FormatMdxNodeTrack(bone.ScalingTrack)}");
}

static void PrintMdxLightSummary(MdxLightSummary light)
{
	string parentId = light.HasParent ? light.ParentId.ToString() : "none(-1)";
	Vector3 staticColor = light.StaticColor;
	Vector3 staticAmbientColor = light.StaticAmbientColor;
	Console.WriteLine($"LITE[{light.Index}]: name={light.Name} objectId={light.ObjectId} parentId={parentId} flags=0x{light.Flags:X8} type={FormatMdxLightType(light.LightType)} staticAttenStart={light.StaticAttenuationStart:F3} staticAttenEnd={light.StaticAttenuationEnd:F3} staticColor=({staticColor.X:F3}, {staticColor.Y:F3}, {staticColor.Z:F3}) staticIntensity={light.StaticIntensity:F3} staticAmbientColor=({staticAmbientColor.X:F3}, {staticAmbientColor.Y:F3}, {staticAmbientColor.Z:F3}) staticAmbientIntensity={light.StaticAmbientIntensity:F3} translationTrack={FormatMdxNodeTrack(light.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(light.RotationTrack)} scalingTrack={FormatMdxNodeTrack(light.ScalingTrack)} attenuationStartTrack={FormatMdxTrack(light.AttenuationStartTrack)} attenuationEndTrack={FormatMdxTrack(light.AttenuationEndTrack)} colorTrack={FormatMdxTrack(light.ColorTrack)} intensityTrack={FormatMdxTrack(light.IntensityTrack)} ambientColorTrack={FormatMdxTrack(light.AmbientColorTrack)} ambientIntensityTrack={FormatMdxTrack(light.AmbientIntensityTrack)} visibilityTrack={FormatMdxVisibilityTrack(light.VisibilityTrack)}");
}

static void PrintMdxHelperSummary(MdxHelperSummary helper)
{
	string parentId = helper.HasParent ? helper.ParentId.ToString() : "none(-1)";
	Console.WriteLine($"HELP[{helper.Index}]: name={helper.Name} objectId={helper.ObjectId} parentId={parentId} flags=0x{helper.Flags:X8} translationTrack={FormatMdxNodeTrack(helper.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(helper.RotationTrack)} scalingTrack={FormatMdxNodeTrack(helper.ScalingTrack)}");
}

static void PrintMdxAttachmentSummary(MdxAttachmentSummary attachment)
{
	string parentId = attachment.HasParent ? attachment.ParentId.ToString() : "none(-1)";
	string path = string.IsNullOrWhiteSpace(attachment.Path) ? "n/a" : attachment.Path;
	Console.WriteLine($"ATCH[{attachment.Index}]: name={attachment.Name} objectId={attachment.ObjectId} parentId={parentId} flags=0x{attachment.Flags:X8} attachmentId={attachment.AttachmentId} path={path} translationTrack={FormatMdxNodeTrack(attachment.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(attachment.RotationTrack)} scalingTrack={FormatMdxNodeTrack(attachment.ScalingTrack)} visibilityTrack={FormatMdxVisibilityTrack(attachment.VisibilityTrack)}");
}

static void PrintMdxParticleEmitter2Summary(MdxParticleEmitter2Summary particleEmitter)
{
	string parentId = particleEmitter.HasParent ? particleEmitter.ParentId.ToString() : "none(-1)";
	string geometryModel = string.IsNullOrWhiteSpace(particleEmitter.GeometryModel) ? "n/a" : particleEmitter.GeometryModel;
	string recursionModel = string.IsNullOrWhiteSpace(particleEmitter.RecursionModel) ? "n/a" : particleEmitter.RecursionModel;
	Vector3 startColor = particleEmitter.StartColor;
	Vector3 middleColor = particleEmitter.MiddleColor;
	Vector3 endColor = particleEmitter.EndColor;
	Console.WriteLine($"PRE2[{particleEmitter.Index}]: name={particleEmitter.Name} objectId={particleEmitter.ObjectId} parentId={parentId} flags=0x{particleEmitter.Flags:X8} emitterType={particleEmitter.EmitterType} staticSpeed={particleEmitter.StaticSpeed:F3} staticVariation={particleEmitter.StaticVariation:F3} staticLatitude={particleEmitter.StaticLatitude:F3} staticLongitude={particleEmitter.StaticLongitude:F3} staticGravity={particleEmitter.StaticGravity:F3} staticZSource={particleEmitter.StaticZSource:F3} staticLife={particleEmitter.StaticLife:F3} staticEmissionRate={particleEmitter.StaticEmissionRate:F3} staticLength={particleEmitter.StaticLength:F3} staticWidth={particleEmitter.StaticWidth:F3} rows={particleEmitter.Rows} cols={particleEmitter.Columns} particleType={particleEmitter.ParticleType} tailLength={particleEmitter.TailLength:F3} middleTime={particleEmitter.MiddleTime:F3} startColor=({startColor.X:F3}, {startColor.Y:F3}, {startColor.Z:F3}) middleColor=({middleColor.X:F3}, {middleColor.Y:F3}, {middleColor.Z:F3}) endColor=({endColor.X:F3}, {endColor.Y:F3}, {endColor.Z:F3}) alphas=[{particleEmitter.StartAlpha},{particleEmitter.MiddleAlpha},{particleEmitter.EndAlpha}] scales=[{particleEmitter.StartScale:F3},{particleEmitter.MiddleScale:F3},{particleEmitter.EndScale:F3}] blendMode={particleEmitter.BlendMode} textureId={particleEmitter.TextureId} priorityPlane={particleEmitter.PriorityPlane} replaceableId={particleEmitter.ReplaceableId} geometryModel={geometryModel} recursionModel={recursionModel} splineCount={particleEmitter.SplineCount} squirts={particleEmitter.Squirts} translationTrack={FormatMdxNodeTrack(particleEmitter.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(particleEmitter.RotationTrack)} scalingTrack={FormatMdxNodeTrack(particleEmitter.ScalingTrack)} visibilityTrack={FormatMdxVisibilityTrack(particleEmitter.VisibilityTrack)} speedTrack={FormatMdxTrack(particleEmitter.SpeedTrack)} variationTrack={FormatMdxTrack(particleEmitter.VariationTrack)} latitudeTrack={FormatMdxTrack(particleEmitter.LatitudeTrack)} longitudeTrack={FormatMdxTrack(particleEmitter.LongitudeTrack)} gravityTrack={FormatMdxTrack(particleEmitter.GravityTrack)} lifeTrack={FormatMdxTrack(particleEmitter.LifeTrack)} emissionRateTrack={FormatMdxTrack(particleEmitter.EmissionRateTrack)} widthTrack={FormatMdxTrack(particleEmitter.WidthTrack)} lengthTrack={FormatMdxTrack(particleEmitter.LengthTrack)} zSourceTrack={FormatMdxTrack(particleEmitter.ZSourceTrack)}");
}

static void PrintMdxRibbonEmitterSummary(MdxRibbonEmitterSummary ribbon)
{
	string parentId = ribbon.HasParent ? ribbon.ParentId.ToString() : "none(-1)";
	Vector3 staticColor = ribbon.StaticColor;
	Console.WriteLine($"RIBB[{ribbon.Index}]: name={ribbon.Name} objectId={ribbon.ObjectId} parentId={parentId} flags=0x{ribbon.Flags:X8} staticHeightAbove={ribbon.StaticHeightAbove:F3} staticHeightBelow={ribbon.StaticHeightBelow:F3} staticAlpha={ribbon.StaticAlpha:F3} staticColor=({staticColor.X:F3}, {staticColor.Y:F3}, {staticColor.Z:F3}) edgeLifetime={ribbon.EdgeLifetime:F3} staticTextureSlot={ribbon.StaticTextureSlot} edgesPerSecond={ribbon.EdgesPerSecond} textureRows={ribbon.TextureRows} textureCols={ribbon.TextureColumns} materialId={ribbon.MaterialId} gravity={ribbon.Gravity:F3} translationTrack={FormatMdxNodeTrack(ribbon.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(ribbon.RotationTrack)} scalingTrack={FormatMdxNodeTrack(ribbon.ScalingTrack)} heightAboveTrack={FormatMdxTrack(ribbon.HeightAboveTrack)} heightBelowTrack={FormatMdxTrack(ribbon.HeightBelowTrack)} alphaTrack={FormatMdxTrack(ribbon.AlphaTrack)} colorTrack={FormatMdxTrack(ribbon.ColorTrack)} textureSlotTrack={FormatMdxTrack(ribbon.TextureSlotTrack)} visibilityTrack={FormatMdxVisibilityTrack(ribbon.VisibilityTrack)}");
}

static void PrintMdxCameraSummary(MdxCameraSummary camera)
{
	Vector3 pivotPoint = camera.PivotPoint;
	Vector3 targetPivotPoint = camera.TargetPivotPoint;
	Console.WriteLine($"CAMS[{camera.Index}]: name={camera.Name} pivot=({pivotPoint.X:F3}, {pivotPoint.Y:F3}, {pivotPoint.Z:F3}) fieldOfView={camera.FieldOfView:F6} farClip={camera.FarClip:F6} nearClip={camera.NearClip:F6} targetPivot=({targetPivotPoint.X:F3}, {targetPivotPoint.Y:F3}, {targetPivotPoint.Z:F3}) positionTrack={FormatMdxTrack(camera.PositionTrack)} rollTrack={FormatMdxTrack(camera.RollTrack)} visibilityTrack={FormatMdxVisibilityTrack(camera.VisibilityTrack)} targetPositionTrack={FormatMdxTrack(camera.TargetPositionTrack)}");
}

static void PrintMdxEventSummary(MdxEventSummary evnt)
{
	string parentId = evnt.HasParent ? evnt.ParentId.ToString() : "none(-1)";
	Console.WriteLine($"EVTS[{evnt.Index}]: name={evnt.Name} objectId={evnt.ObjectId} parentId={parentId} flags=0x{evnt.Flags:X8} translationTrack={FormatMdxNodeTrack(evnt.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(evnt.RotationTrack)} scalingTrack={FormatMdxNodeTrack(evnt.ScalingTrack)} eventTrack={FormatMdxEventTrack(evnt.EventTrack)}");
}

static void PrintMdxHitTestShapeSummary(MdxHitTestShapeSummary shape)
{
	string parentId = shape.HasParent ? shape.ParentId.ToString() : "none(-1)";
	Console.WriteLine($"HTST[{shape.Index}]: name={shape.Name} objectId={shape.ObjectId} parentId={parentId} flags=0x{shape.Flags:X8} shapeType={FormatMdxGeometryShapeType(shape.ShapeType)} shape={FormatMdxHitTestShapeGeometry(shape)} translationTrack={FormatMdxNodeTrack(shape.TranslationTrack)} rotationTrack={FormatMdxNodeTrack(shape.RotationTrack)} scalingTrack={FormatMdxNodeTrack(shape.ScalingTrack)}");
}

static void PrintMdxCollisionSummary(MdxCollisionSummary collision)
{
	string boundsMin = collision.BoundsMin is Vector3 min ? $"({min.X:F3}, {min.Y:F3}, {min.Z:F3})" : "n/a";
	string boundsMax = collision.BoundsMax is Vector3 max ? $"({max.X:F3}, {max.Y:F3}, {max.Z:F3})" : "n/a";
	Console.WriteLine($"CLID: vertices={collision.VertexCount} triIndices={collision.TriangleIndexCount} triangles={collision.TriangleCount} facetNormals={collision.FacetNormalCount} maxIndex={collision.MaxTriangleIndex} boundsMin={boundsMin} boundsMax={boundsMax}");
}

static void PrintMdxPivotPointSummary(MdxPivotPointSummary pivotPoint)
{
	Vector3 position = pivotPoint.Position;
	Console.WriteLine($"PIVT[{pivotPoint.Index}]: position=({position.X:F3}, {position.Y:F3}, {position.Z:F3})");
}

static void PrintMdxMaterialSummary(MdxMaterialSummary material)
{
	Console.WriteLine($"MTLS[{material.Index}]: priorityPlane={material.PriorityPlane} layers={material.LayerCount}");
	for (int layerIndex = 0; layerIndex < material.Layers.Count; layerIndex++)
		PrintMdxMaterialLayerSummary(material.Index, material.Layers[layerIndex]);
}

static void PrintMdxMaterialLayerSummary(int materialIndex, MdxMaterialLayerSummary layer)
{
	Console.WriteLine($"MTLS[{materialIndex}].LAYER[{layer.Index}]: blendMode={FormatMdxBlendMode(layer.BlendMode)} flags=0x{layer.Flags:X8} textureId={layer.TextureId} transformId={layer.TransformId} coordId={layer.CoordId} staticAlpha={layer.StaticAlpha:F3}");
}

static string FormatMdxBlendMode(uint blendMode)
{
	return blendMode switch
	{
		0 => "Load(0)",
		1 => "Transparent(1)",
		2 => "Blend(2)",
		3 => "Add(3)",
		4 => "AddAlpha(4)",
		5 => "Modulate(5)",
		6 => "Modulate2X(6)",
		_ => blendMode.ToString()
	};
}

static string FormatMdxGeosetAnimationTrack(MdxGeosetAnimationTrackSummary? track)
{
	if (track is null)
		return "none";

	string timeRange = track.FirstKeyTime is int firstKeyTime && track.LastKeyTime is int lastKeyTime
		? $"[{firstKeyTime}, {lastKeyTime}]"
		: "n/a";

	return $"{track.Tag}(keys={track.KeyCount} interpolation={FormatMdxInterpolation(track.InterpolationType)} globalSeqId={track.GlobalSequenceId} time={timeRange})";
}

static string FormatMdxNodeTrack(MdxNodeTrackSummary? track)
{
	if (track is null)
		return "none";

	string timeRange = track.FirstKeyTime is int firstKeyTime && track.LastKeyTime is int lastKeyTime
		? $"[{firstKeyTime}, {lastKeyTime}]"
		: "n/a";

	return $"{track.Tag}(keys={track.KeyCount} interpolation={FormatMdxInterpolation(track.InterpolationType)} globalSeqId={track.GlobalSequenceId} time={timeRange})";
}

static string FormatMdxTrack(MdxTrackSummary? track)
{
	if (track is null)
		return "none";

	string timeRange = track.FirstKeyTime is int firstKeyTime && track.LastKeyTime is int lastKeyTime
		? $"[{firstKeyTime}, {lastKeyTime}]"
		: "n/a";

	return $"{track.Tag}(keys={track.KeyCount} interpolation={FormatMdxInterpolation(track.InterpolationType)} globalSeqId={track.GlobalSequenceId} time={timeRange})";
}

static string FormatMdxVisibilityTrack(MdxVisibilityTrackSummary? track)
{
	if (track is null)
		return "none";

	string timeRange = track.FirstKeyTime is int firstKeyTime && track.LastKeyTime is int lastKeyTime
		? $"[{firstKeyTime}, {lastKeyTime}]"
		: "n/a";

	return $"{track.Tag}(keys={track.KeyCount} interpolation={FormatMdxInterpolation(track.InterpolationType)} globalSeqId={track.GlobalSequenceId} time={timeRange})";
}

static string FormatMdxEventTrack(MdxEventTrackSummary? track)
{
	if (track is null)
		return "none";

	string timeRange = track.FirstKeyTime is int firstKeyTime && track.LastKeyTime is int lastKeyTime
		? $"[{firstKeyTime}, {lastKeyTime}]"
		: "n/a";

	return $"{track.Tag}(keys={track.KeyCount} globalSeqId={track.GlobalSequenceId} time={timeRange})";
}

static string FormatMdxGeometryShapeType(MdxGeometryShapeType shapeType)
{
	return shapeType switch
	{
		MdxGeometryShapeType.Box => "Box(0)",
		MdxGeometryShapeType.Cylinder => "Cylinder(1)",
		MdxGeometryShapeType.Sphere => "Sphere(2)",
		MdxGeometryShapeType.Plane => "Plane(3)",
		_ => ((byte)shapeType).ToString(),
	};
}

static string FormatMdxLightType(MdxLightType lightType)
{
	return lightType switch
	{
		MdxLightType.Omni => "Omni(0)",
		MdxLightType.Direct => "Direct(1)",
		MdxLightType.Ambient => "Ambient(2)",
		_ => ((uint)lightType).ToString(),
	};
}

static string FormatMdxHitTestShapeGeometry(MdxHitTestShapeSummary shape)
{
	return shape.ShapeType switch
	{
		MdxGeometryShapeType.Box when shape.Minimum is Vector3 minimum && shape.Maximum is Vector3 maximum
			=> $"boxMin=({minimum.X:F3}, {minimum.Y:F3}, {minimum.Z:F3}) boxMax=({maximum.X:F3}, {maximum.Y:F3}, {maximum.Z:F3})",
		MdxGeometryShapeType.Cylinder when shape.BasePoint is Vector3 basePoint && shape.Height is float height && shape.Radius is float radius
			=> $"base=({basePoint.X:F3}, {basePoint.Y:F3}, {basePoint.Z:F3}) height={height:F6} radius={radius:F6}",
		MdxGeometryShapeType.Sphere when shape.Center is Vector3 center && shape.Radius is float sphereRadius
			=> $"center=({center.X:F3}, {center.Y:F3}, {center.Z:F3}) radius={sphereRadius:F6}",
		MdxGeometryShapeType.Plane when shape.Length is float length && shape.Width is float width
			=> $"length={length:F6} width={width:F6}",
		_ => "n/a",
	};
}

static string FormatMdxInterpolation(uint interpolationType)
{
	return interpolationType switch
	{
		0 => "None(0)",
		1 => "Linear(1)",
		2 => "Hermite(2)",
		3 => "Bezier(3)",
		4 => "Bezier2(4)",
		_ => interpolationType.ToString()
	};
}

static void ShowUsage()
{
	Console.WriteLine("WowViewer.Tool.Inspect");
	Console.WriteLine("Usage:");
	Console.WriteLine("  wowviewer-inspect blp inspect --input <file.blp>");
	Console.WriteLine("  wowviewer-inspect blp inspect --archive-root <game|data dir> --virtual-path <path/to/file.blp> [--listfile <listfile.txt>]");
	Console.WriteLine("  wowviewer-inspect mdx inspect --input <file.mdx>");
	Console.WriteLine("  wowviewer-inspect mdx inspect --archive-root <game|data dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>]");
	Console.WriteLine("  wowviewer-inspect mdx export-json --input <file.mdx> [--output <report.json>] [--include-geometry] [--include-collision] [--include-hit-test] [--include-texture-animations]");
	Console.WriteLine("  wowviewer-inspect mdx export-json --archive-root <game|data dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>] [--output <report.json>] [--include-geometry] [--include-collision] [--include-hit-test] [--include-texture-animations]");
	Console.WriteLine("  wowviewer-inspect mdx chunk-carriers --chunks <FOURCC[,FOURCC...]> --input <file|directory> [--path-filter <text>] [--limit <n>]");
	Console.WriteLine("  wowviewer-inspect mdx chunk-carriers --chunks <FOURCC[,FOURCC...]> --archive-root <game|data dir> [--listfile <listfile.txt>] [--path-filter <text>] [--limit <n>]");
	Console.WriteLine("  wowviewer-inspect map inspect --input <file.wdt|file.adt>");
	Console.WriteLine("  wowviewer-inspect wmo inspect --input <file.wmo> [--dump-lights]");
	Console.WriteLine("  wowviewer-inspect wmo inspect --archive-root <game|data dir> --virtual-path <world/...wmo> [--listfile <listfile.txt>] [--dump-lights]");
	Console.WriteLine("  wowviewer-inspect pm4 inspect --input <file.pm4>");
	Console.WriteLine("  wowviewer-inspect pm4 linkage --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 mscn --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 unknowns --input <directory> [--output <report.json>]");
	Console.WriteLine("  wowviewer-inspect pm4 audit --input <file.pm4>");
	Console.WriteLine("  wowviewer-inspect pm4 audit-directory --input <directory>");
	Console.WriteLine("  wowviewer-inspect pm4 export-json --input <file.pm4> [--output <report.json>] [--ck24 <decimal|0xHEX>]");
}

static void ShowBlpUsage()
{
	Console.WriteLine("BLP commands:");
	Console.WriteLine("  blp inspect --input <file.blp>");
	Console.WriteLine("  blp inspect --archive-root <game|data dir> --virtual-path <path/to/file.blp> [--listfile <listfile.txt>]");
}

static void ShowMdxUsage()
{
	Console.WriteLine("MDX commands:");
	Console.WriteLine("  mdx inspect --input <file.mdx>");
	Console.WriteLine("  mdx inspect --archive-root <game|data dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>]");
	Console.WriteLine("  mdx export-json --input <file.mdx> [--output <report.json>] [--include-geometry] [--include-collision] [--include-hit-test] [--include-texture-animations]");
	Console.WriteLine("  mdx export-json --archive-root <game|data dir> --virtual-path <path/to/file.mdx> [--listfile <listfile.txt>] [--output <report.json>] [--include-geometry] [--include-collision] [--include-hit-test] [--include-texture-animations]");
	Console.WriteLine("  mdx chunk-carriers --chunks <FOURCC[,FOURCC...]> --input <file|directory> [--path-filter <text>] [--limit <n>]");
	Console.WriteLine("  mdx chunk-carriers --chunks <FOURCC[,FOURCC...]> --archive-root <game|data dir> [--listfile <listfile.txt>] [--path-filter <text>] [--limit <n>]");
}

static void ShowWmoUsage()
{
	Console.WriteLine("WMO commands:");
	Console.WriteLine("  wmo inspect --input <file.wmo> [--dump-lights]");
	Console.WriteLine("  wmo inspect --archive-root <game|data dir> --virtual-path <world/...wmo> [--listfile <listfile.txt>] [--dump-lights]");
}

static void ShowMapUsage()
{
	Console.WriteLine("Map commands:");
	Console.WriteLine("  map inspect --input <file.wdt|file.adt> [--dump-tex-chunks]");
}

static void ShowPm4Usage()
{
	Console.WriteLine("PM4 commands:");
	Console.WriteLine("  pm4 inspect --input <file.pm4>");
	Console.WriteLine("  pm4 match --input <file.pm4> --archive-root <game|data dir> [--placements <tile_obj0.adt>] [--listfile <listfile.txt>] [--max-matches <n>] [--search-range <units>] [--output <report.json>] [--object-output-dir <directory>]");
	Console.WriteLine("  pm4 hierarchy --input <file.pm4> [--output <report.json>]");
	Console.WriteLine("  pm4 linkage --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 mscn --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 unknowns --input <directory> [--output <report.json>]");
	Console.WriteLine("  pm4 audit --input <file.pm4>");
	Console.WriteLine("  pm4 audit-directory --input <directory>");
	Console.WriteLine("  pm4 export-json --input <file.pm4> [--output <report.json>] [--ck24 <decimal|0xHEX>]");
}
