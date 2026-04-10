using System.Text.Json;
using System.Text.Json.Serialization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WowViewer.Core.Files;
using WowViewer.Core.IO;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4;
using WowViewer.Core.Wmo;
using WowViewer.Tools.Shared;

if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
{
	ShowUsage();
	return;
}

string command = args[0].ToLowerInvariant();
string[] tail = args.Skip(1).ToArray();

switch (command)
{
	case "detect":
		RunDetect(tail);
		break;
	case "ml-corpus":
		RunMlCorpus(tail);
		break;
	case "ml-synth-no-liquid":
		RunMlSynthNoLiquid(tail);
		break;
	case "export-tex-json":
		RunExportTexJson(tail);
		break;
	default:
		Console.Error.WriteLine($"Unknown converter command '{command}'.");
		ShowUsage();
		Environment.ExitCode = 1;
		break;
}

static void RunDetect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input file is required.");
		Environment.ExitCode = 1;
		return;
	}

	WowFileDetection detection = WowFileDetector.Detect(input);
	Console.WriteLine("WowViewer.Tool.Converter detect report");
	Console.WriteLine($"Input: {detection.SourcePath}");
	Console.WriteLine($"Kind: {detection.Kind}");
	Console.WriteLine($"Version: {detection.Version?.ToString() ?? "n/a"}");
	if (detection.Kind is WowFileKind.Wmo or WowFileKind.WmoGroup)
	{
		WmoLiquidCoordinateFamily family = WmoLiquidLayoutResolver.ResolveCoordinateFamily(detection.Version);
		int baselineQuarterTurns = WmoLiquidLayoutResolver.GetBaselineRotationQuarterTurns(detection.Version);
		Console.WriteLine($"WMO liquid family: {family}");
		Console.WriteLine($"WMO MLIQ baseline rotation: {baselineQuarterTurns * 90}°");
	}
	Console.WriteLine($"Owns families: {string.Join(", ", IoBoundaries.OwnedFamilies)}");
	Console.WriteLine($"PM4 source-of-truth: canonical={Pm4Boundary.CanonicalOwner}, seed={Pm4Boundary.LibrarySeed}, legacy={Pm4Boundary.LegacyReference}");
	Console.WriteLine($"Planned hosts: {ToolHosts.Planned.Length}");
}

static void RunExportTexJson(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input root ADT or _tex0.adt file is required.");
		Environment.ExitCode = 1;
		return;
	}

	WowFileDetection detection = WowFileDetector.Detect(input);
	if (detection.Kind is not (WowFileKind.Adt or WowFileKind.AdtTex))
	{
		Console.Error.WriteLine($"Error: export-tex-json requires a root ADT or _tex0.adt input, but detected {detection.Kind}.");
		Environment.ExitCode = 1;
		return;
	}

	string json = JsonSerializer.Serialize(
		AdtTextureReader.Read(input),
		CreateJsonOptions());

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

static void RunMlCorpus(string[] args)
{
	string? configPath = GetOption(args, "--config", "-c");
	if (string.IsNullOrWhiteSpace(configPath))
	{
		Console.Error.WriteLine("Error: --config <ml-corpus.json> is required.");
		Environment.ExitCode = 1;
		return;
	}

	string fullConfigPath = Path.GetFullPath(configPath);
	if (!File.Exists(fullConfigPath))
	{
		Console.Error.WriteLine($"Error: config file not found: {fullConfigPath}");
		Environment.ExitCode = 1;
		return;
	}

	MlCorpusConfig? config = JsonSerializer.Deserialize<MlCorpusConfig>(
		File.ReadAllText(fullConfigPath),
		new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
	if (config is null)
	{
		Console.Error.WriteLine("Error: failed to parse ml-corpus config.");
		Environment.ExitCode = 1;
		return;
	}

	if (config.Clients.Count == 0)
	{
		Console.Error.WriteLine("Error: config requires at least one client entry.");
		Environment.ExitCode = 1;
		return;
	}

	string archiveRoot = ResolveOptionalRoot(
		GetOption(args, "--archive-root", "-a"),
		config.ArchiveRoot,
		fullConfigPath);
	string outputRoot = ResolveOptionalRoot(
		GetOption(args, "--output-root", "-o"),
		config.DefaultOutputRoot,
		fullConfigPath,
		Path.Combine(Environment.CurrentDirectory, "output", "ml-corpus"));

	bool dryRun = HasFlag(args, "--dry-run");

	Console.WriteLine("WowViewer.Tool.Converter ml-corpus report");
	Console.WriteLine($"Config: {fullConfigPath}");
	Console.WriteLine($"ArchiveRoot: {(string.IsNullOrWhiteSpace(archiveRoot) ? "(none)" : archiveRoot)}");
	Console.WriteLine($"OutputRoot: {outputRoot}");
	Console.WriteLine($"DryRun: {dryRun}");

	int mapsProcessed = 0;
	int tilesProcessed = 0;
	foreach (MlCorpusClientConfig client in config.Clients)
	{
		if (string.IsNullOrWhiteSpace(client.ClientPath))
			continue;

		string clientId = ResolveClientId(client);
		if (string.IsNullOrWhiteSpace(clientId))
			continue;

		string clientRoot = ResolveDataPath(client.ClientPath, archiveRoot, fullConfigPath);
		if (!Directory.Exists(clientRoot))
		{
			Console.Error.WriteLine($"Warning: skipping missing client root: {clientRoot}");
			continue;
		}

		IArchiveCatalog? archiveCatalog = null;

		IReadOnlyList<MlCorpusMapConfig> maps = ResolveMapsForClient(config, client);
		foreach (MlCorpusMapConfig map in maps)
		{
			if (string.IsNullOrWhiteSpace(map.MapName))
				continue;

			string resolvedMapPath = string.IsNullOrWhiteSpace(map.MapPath)
				? Path.Combine("Data", "World", "Maps", map.MapName)
				: map.MapPath;
			string mapPath = ResolveDataPath(resolvedMapPath, clientRoot, fullConfigPath);
			bool hasFilesystemMap = Directory.Exists(mapPath);
			if (!hasFilesystemMap)
			{
				archiveCatalog ??= CreateArchiveCatalog(clientRoot);
				if (!ArchiveMapExists(archiveCatalog, clientRoot, map.MapName, mapPath))
				{
					Console.Error.WriteLine($"Warning: skipping missing map source: {mapPath}");
					continue;
				}
			}

			string mapOutputRoot = Path.Combine(outputRoot, clientId, map.MapName);
			if (!dryRun)
				Directory.CreateDirectory(mapOutputRoot);

			MlCorpusMapReport report = BuildMapReport(clientId, map.MapName, clientRoot, mapPath, mapOutputRoot, dryRun, archiveCatalog);
			if (!dryRun)
			{
				string reportPath = Path.Combine(mapOutputRoot, "ml_corpus_map_report.json");
				File.WriteAllText(reportPath, JsonSerializer.Serialize(report, CreateJsonOptions()));
			}

			mapsProcessed++;
			tilesProcessed += report.TileCount;
			Console.WriteLine($"Processed map {clientId}/{map.MapName}: {report.TileCount} tiles");
		}

		archiveCatalog?.Dispose();
	}

	Console.WriteLine($"ml-corpus complete: maps={mapsProcessed} tiles={tilesProcessed}");
}

static MlCorpusMapReport BuildMapReport(string clientId, string mapName, string clientRoot, string mapPath, string mapOutputRoot, bool dryRun, IArchiveCatalog? archiveCatalog)
{
	if (Directory.Exists(mapPath))
		return BuildMapReportFromDirectory(clientId, mapName, mapPath, mapOutputRoot, dryRun);

	if (archiveCatalog is null)
		throw new FileNotFoundException($"Could not locate map data for {mapName}.", mapPath);

	return BuildMapReportFromArchive(clientId, mapName, clientRoot, mapPath, archiveCatalog, mapOutputRoot, dryRun);
}

static MlCorpusMapReport BuildMapReportFromDirectory(string clientId, string mapName, string mapPath, string mapOutputRoot, bool dryRun)
{
	MlCorpusIssueTracker issueTracker = new();
	List<string> adtFiles = Directory
		.EnumerateFiles(mapPath, $"{mapName}_*.adt", SearchOption.TopDirectoryOnly)
		.Where(static path => !path.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
			&& !path.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase)
			&& !path.EndsWith("_lod.adt", StringComparison.OrdinalIgnoreCase))
		.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase)
		.ToList();

	List<MlCorpusTileReport> tiles = new(adtFiles.Count);
	string tilesDir = Path.Combine(mapOutputRoot, "tiles");
	if (!dryRun)
		Directory.CreateDirectory(tilesDir);

	foreach (string adtPath in adtFiles)
	{
		string tileStem = Path.GetFileNameWithoutExtension(adtPath);
		AdtTileFamily family = AdtTileFamilyResolver.Resolve(adtPath);
		AdtSummary summary;
		try
		{
			summary = AdtSummaryReader.Read(family.RootPath);
		}
		catch (Exception ex)
		{
			issueTracker.Record("tile-summary", adtPath, ex);
			continue;
		}

		string textureSourcePath = family.TextureSourcePath ?? family.RootPath;
		AdtTextureFile textureFile = TryReadTextureFile(
			issueTracker,
			"textures",
			textureSourcePath,
			family.TextureSourceKind ?? MapFileKind.Adt,
			() => AdtTextureReader.Read(textureSourcePath));
		AdtPlacementCatalog? placements = string.IsNullOrWhiteSpace(family.PlacementSourcePath)
			? null
			: TryReadPlacementCatalog(
				issueTracker,
				"placements",
				family.PlacementSourcePath,
				() => AdtPlacementReader.Read(family.PlacementSourcePath));

		MlCorpusTileReport tile = CreateTileReport(
			tileStem,
			adtPath,
			family.HasObj0,
			summary,
			textureFile,
			placements);

		tiles.Add(tile);

		if (!dryRun)
		{
			string tilePath = Path.Combine(tilesDir, $"{tileStem}.json");
			File.WriteAllText(tilePath, JsonSerializer.Serialize(tile, CreateJsonOptions()));
		}
	}

	MlCorpusMapReport report = new(
		ClientId: clientId,
		MapName: mapName,
		MapPath: mapPath,
		TileCount: tiles.Count,
		GeneratedUtc: DateTime.UtcNow,
		Tiles: tiles);

	issueTracker.Print($"{clientId}/{mapName}");
	return report;
}

static MlCorpusMapReport BuildMapReportFromArchive(string clientId, string mapName, string clientRoot, string mapPath, IArchiveCatalog archiveCatalog, string mapOutputRoot, bool dryRun)
{
	MlCorpusIssueTracker issueTracker = new();
	string mapVirtualRoot = ResolveMapVirtualRoot(clientRoot, mapName, mapPath);
	string wdtVirtualPath = BuildMapWdtVirtualPath(mapVirtualRoot, mapName);
	byte[] wdtBytes = archiveCatalog.ReadFile(wdtVirtualPath)
		?? throw new FileNotFoundException($"Could not read archive-backed WDT '{wdtVirtualPath}'.", wdtVirtualPath);

	IReadOnlyList<WdtTileCoordinate> tileCoordinates = ReadArchiveWdtTiles(wdtBytes, wdtVirtualPath)
		.OrderBy(static tile => tile.TileY)
		.ThenBy(static tile => tile.TileX)
		.ToArray();

	List<MlCorpusTileReport> tiles = new(tileCoordinates.Count);
	string tilesDir = Path.Combine(mapOutputRoot, "tiles");
	if (!dryRun)
		Directory.CreateDirectory(tilesDir);

	foreach (WdtTileCoordinate tileCoordinate in tileCoordinates)
	{
		string tileStem = $"{mapName}_{tileCoordinate.TileX}_{tileCoordinate.TileY}";
		string rootVirtualPath = $"{mapVirtualRoot}\\{tileStem}.adt";
		string texVirtualPath = $"{mapVirtualRoot}\\{tileStem}_tex0.adt";
		string objVirtualPath = $"{mapVirtualRoot}\\{tileStem}_obj0.adt";

		byte[] rootBytes = archiveCatalog.ReadFile(rootVirtualPath) ?? [];
		if (rootBytes.Length == 0)
			continue;

		AdtSummary summary;
		try
		{
			summary = ReadArchiveAdtSummary(rootBytes, rootVirtualPath);
		}
		catch (Exception ex)
		{
			issueTracker.Record("tile-summary", rootVirtualPath, ex);
			continue;
		}

		AdtTextureFile textureFile = TryReadArchiveTextureFile(issueTracker, archiveCatalog, rootBytes, rootVirtualPath, texVirtualPath);
		AdtPlacementCatalog? placements = TryReadArchivePlacementCatalog(issueTracker, archiveCatalog, rootBytes, rootVirtualPath, objVirtualPath);

		MlCorpusTileReport tile = CreateTileReport(
			tileStem,
			rootVirtualPath,
			placements is not null && archiveCatalog.FileExists(objVirtualPath),
			summary,
			textureFile,
			placements);

		tiles.Add(tile);

		if (!dryRun)
		{
			string tilePath = Path.Combine(tilesDir, $"{tileStem}.json");
			File.WriteAllText(tilePath, JsonSerializer.Serialize(tile, CreateJsonOptions()));
		}
	}

	MlCorpusMapReport report = new(
		ClientId: clientId,
		MapName: mapName,
		MapPath: wdtVirtualPath,
		TileCount: tiles.Count,
		GeneratedUtc: DateTime.UtcNow,
		Tiles: tiles);

	issueTracker.Print($"{clientId}/{mapName}");
	return report;
}

static MlCorpusTileReport CreateTileReport(string tileStem, string adtPath, bool hasObj0, AdtSummary summary, AdtTextureFile textureFile, AdtPlacementCatalog? placements)
{
	HashSet<string> textures = new(StringComparer.OrdinalIgnoreCase);
	foreach (string textureName in textureFile.TextureNames)
	{
		if (!string.IsNullOrWhiteSpace(textureName))
			textures.Add(textureName);
	}

	int modelPlacementCount = placements?.ModelPlacements.Count ?? summary.ModelPlacementCount;
	int worldModelPlacementCount = placements?.WorldModelPlacements.Count ?? summary.WorldModelPlacementCount;

	return new MlCorpusTileReport(
		Tile: tileStem,
		AdtPath: adtPath,
		HasObj0: hasObj0,
		TerrainChunks: summary.TerrainChunkCount,
		HasWater: summary.HasWater,
		TextureNameCount: textures.Count,
		ModelPlacementCount: modelPlacementCount,
		WorldModelPlacementCount: worldModelPlacementCount,
		Textures: textures.OrderBy(static name => name, StringComparer.OrdinalIgnoreCase).ToArray());
}

static IArchiveCatalog CreateArchiveCatalog(string clientRoot)
{
	IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
	ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, [clientRoot], (ArchiveCatalogBootstrapOptions?)null);
	return archiveCatalog;
}

static bool ArchiveMapExists(IArchiveCatalog archiveCatalog, string clientRoot, string mapName, string mapPath)
{
	string mapVirtualRoot = ResolveMapVirtualRoot(clientRoot, mapName, mapPath);
	string wdtVirtualPath = BuildMapWdtVirtualPath(mapVirtualRoot, mapName);
	return archiveCatalog.FileExists(wdtVirtualPath);
}

static string ResolveMapVirtualRoot(string clientRoot, string mapName, string mapPath)
{
	string normalizedMapPath = mapPath;
	if (normalizedMapPath.EndsWith(".wdt", StringComparison.OrdinalIgnoreCase))
		normalizedMapPath = Path.GetDirectoryName(normalizedMapPath) ?? normalizedMapPath;

	string relativePath = Path.GetRelativePath(clientRoot, normalizedMapPath);
	if (!relativePath.StartsWith("..", StringComparison.OrdinalIgnoreCase))
	{
		if (relativePath.StartsWith("Data\\", StringComparison.OrdinalIgnoreCase))
			relativePath = relativePath[5..];

		return relativePath.Replace('/', '\\').TrimStart('\\');
	}

	return Path.Combine("World", "Maps", mapName).Replace('/', '\\');
}

static string BuildMapWdtVirtualPath(string mapVirtualRoot, string mapName)
{
	if (mapVirtualRoot.EndsWith(".wdt", StringComparison.OrdinalIgnoreCase))
		return mapVirtualRoot;

	return $"{mapVirtualRoot}\\{mapName}.wdt";
}

static IReadOnlyList<WdtTileCoordinate> ReadArchiveWdtTiles(byte[] wdtBytes, string wdtVirtualPath)
{
	using MemoryStream stream = new(wdtBytes, writable: false);
	MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, wdtVirtualPath);
	return WdtTileIndexReader.ReadOccupiedTiles(stream, fileSummary);
}

static AdtSummary ReadArchiveAdtSummary(byte[] adtBytes, string adtVirtualPath)
{
	using MemoryStream stream = new(adtBytes, writable: false);
	MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, adtVirtualPath);
	return AdtSummaryReader.Read(stream, fileSummary);
}

static AdtTextureFile ReadArchiveTextureFile(IArchiveCatalog archiveCatalog, byte[] rootBytes, string rootVirtualPath, string texVirtualPath)
{
	byte[]? texBytes = archiveCatalog.ReadFile(texVirtualPath);
	if (texBytes is { Length: > 0 })
	{
		using MemoryStream texStream = new(texBytes, writable: false);
		MapFileSummary texSummary = MapFileSummaryReader.Read(texStream, texVirtualPath);
		return AdtTextureReader.Read(texStream, texSummary);
	}

	using MemoryStream rootStream = new(rootBytes, writable: false);
	MapFileSummary rootSummary = MapFileSummaryReader.Read(rootStream, rootVirtualPath);
	return AdtTextureReader.Read(rootStream, rootSummary);
}

static AdtTextureFile TryReadArchiveTextureFile(MlCorpusIssueTracker issueTracker, IArchiveCatalog archiveCatalog, byte[] rootBytes, string rootVirtualPath, string texVirtualPath)
{
	try
	{
		return ReadArchiveTextureFile(archiveCatalog, rootBytes, rootVirtualPath, texVirtualPath);
	}
	catch (Exception ex)
	{
		string failingSource = archiveCatalog.FileExists(texVirtualPath) ? texVirtualPath : rootVirtualPath;
		issueTracker.Record("textures", failingSource, ex);
		return CreateEmptyTextureFile(failingSource, archiveCatalog.FileExists(texVirtualPath) ? MapFileKind.AdtTex : MapFileKind.Adt);
	}
}

static AdtPlacementCatalog? TryReadArchivePlacementCatalog(MlCorpusIssueTracker issueTracker, IArchiveCatalog archiveCatalog, byte[] rootBytes, string rootVirtualPath, string objVirtualPath)
{
	try
	{
		byte[]? objBytes = archiveCatalog.ReadFile(objVirtualPath);
		if (objBytes is { Length: > 0 })
		{
			using MemoryStream objStream = new(objBytes, writable: false);
			MapFileSummary objSummary = MapFileSummaryReader.Read(objStream, objVirtualPath);
			return AdtPlacementReader.Read(objStream, objSummary);
		}

		using MemoryStream rootStream = new(rootBytes, writable: false);
		MapFileSummary rootSummary = MapFileSummaryReader.Read(rootStream, rootVirtualPath);
		return AdtPlacementReader.Read(rootStream, rootSummary);
	}
	catch (Exception ex)
	{
		string failingSource = archiveCatalog.FileExists(objVirtualPath) ? objVirtualPath : rootVirtualPath;
		issueTracker.Record("placements", failingSource, ex);
		return null;
	}
}

static AdtTextureFile TryReadTextureFile(MlCorpusIssueTracker issueTracker, string category, string sourcePath, MapFileKind kind, Func<AdtTextureFile> reader)
{
	try
	{
		return reader();
	}
	catch (Exception ex)
	{
		issueTracker.Record(category, sourcePath, ex);
		return CreateEmptyTextureFile(sourcePath, kind);
	}
}

static AdtPlacementCatalog? TryReadPlacementCatalog(MlCorpusIssueTracker issueTracker, string category, string sourcePath, Func<AdtPlacementCatalog> reader)
{
	try
	{
		return reader();
	}
	catch (Exception ex)
	{
		issueTracker.Record(category, sourcePath, ex);
		return null;
	}
}

static AdtTextureFile CreateEmptyTextureFile(string sourcePath, MapFileKind kind)
{
	AdtMcalDecodeProfile decodeProfile = kind == MapFileKind.AdtTex
		? AdtMcalDecodeProfile.Cataclysm400
		: AdtMcalDecodeProfile.LichKingStrict;
	return new AdtTextureFile(sourcePath, kind, decodeProfile, Array.Empty<string>(), Array.Empty<AdtTextureChunk>());
}

static void RunMlSynthNoLiquid(string[] args)
{
	string? input = GetOption(args, "--input", "-i");
	string? mask = GetOption(args, "--mask", "-m");
	string? output = GetOption(args, "--output", "-o");
	string? inputDir = GetOption(args, "--input-dir", "-I");
	string? maskDir = GetOption(args, "--mask-dir", "-M");
	string? outputDir = GetOption(args, "--output-dir", "-O");

	if (!string.IsNullOrWhiteSpace(input) || !string.IsNullOrWhiteSpace(mask) || !string.IsNullOrWhiteSpace(output))
	{
		if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(mask) || string.IsNullOrWhiteSpace(output))
		{
			Console.Error.WriteLine("Error: single mode requires --input, --mask, and --output.");
			Environment.ExitCode = 1;
			return;
		}

		byte[] png = SynthesizeNoLiquidMinimap(Path.GetFullPath(input), Path.GetFullPath(mask));
		if (png.Length == 0)
		{
			Console.Error.WriteLine("Error: synthesis failed.");
			Environment.ExitCode = 1;
			return;
		}

		string outPath = Path.GetFullPath(output);
		Directory.CreateDirectory(Path.GetDirectoryName(outPath) ?? Environment.CurrentDirectory);
		File.WriteAllBytes(outPath, png);
		Console.WriteLine($"Wrote {outPath}");
		return;
	}

	if (string.IsNullOrWhiteSpace(inputDir) || string.IsNullOrWhiteSpace(maskDir) || string.IsNullOrWhiteSpace(outputDir))
	{
		Console.Error.WriteLine("Error: provide either single mode (--input/--mask/--output) or batch mode (--input-dir/--mask-dir/--output-dir).");
		Environment.ExitCode = 1;
		return;
	}

	string fullInputDir = Path.GetFullPath(inputDir);
	string fullMaskDir = Path.GetFullPath(maskDir);
	string fullOutputDir = Path.GetFullPath(outputDir);
	Directory.CreateDirectory(fullOutputDir);

	int generated = 0;
	foreach (string maskPath in Directory.EnumerateFiles(fullMaskDir, "*_liq_mask.png", SearchOption.TopDirectoryOnly))
	{
		string maskFile = Path.GetFileName(maskPath);
		string tileStem = maskFile[..^"_liq_mask.png".Length];
		string minimapPath = Path.Combine(fullInputDir, $"{tileStem}.png");
		if (!File.Exists(minimapPath))
			continue;

		byte[] png = SynthesizeNoLiquidMinimap(minimapPath, maskPath);
		if (png.Length == 0)
			continue;

		string outPath = Path.Combine(fullOutputDir, $"{tileStem}_no_liquid.png");
		File.WriteAllBytes(outPath, png);
		generated++;
	}

	Console.WriteLine($"Generated {generated} no-liquid minimaps in {fullOutputDir}");
}

static byte[] SynthesizeNoLiquidMinimap(string sourceMinimapPath, string liquidMaskPath)
{
	try
	{
		using Image<Rgba32> srcImage = Image.Load<Rgba32>(sourceMinimapPath);
		int width = srcImage.Width;
		int height = srcImage.Height;

		using Image<L8> maskImage = Image.Load<L8>(liquidMaskPath);
		if (maskImage.Width != width || maskImage.Height != height)
			maskImage.Mutate(ctx => ctx.Resize(width, height));

		Rgba32[] pixels = new Rgba32[width * height];
		bool[] resolved = new bool[width * height];

		srcImage.ProcessPixelRows(accessor =>
		{
			for (int y = 0; y < height; y++)
			{
				Span<Rgba32> row = accessor.GetRowSpan(y);
				for (int x = 0; x < width; x++)
					pixels[(y * width) + x] = row[x];
			}
		});

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				L8 m = maskImage[x, y];
				resolved[(y * width) + x] = m.PackedValue < 128;
			}
		}

		long rSum = 0;
		long gSum = 0;
		long bSum = 0;
		long count = 0;
		for (int i = 0; i < pixels.Length; i++)
		{
			if (!resolved[i])
				continue;

			rSum += pixels[i].R;
			gSum += pixels[i].G;
			bSum += pixels[i].B;
			count++;
		}

		Rgba32 fallback = count > 0
			? new Rgba32((byte)(rSum / count), (byte)(gSum / count), (byte)(bSum / count), 255)
			: new Rgba32(100, 120, 80, 255);

		bool[] pending = new bool[pixels.Length];
		for (int i = 0; i < pending.Length; i++)
			pending[i] = !resolved[i];

		int[] dx = { -1, 1, 0, 0 };
		int[] dy = { 0, 0, -1, 1 };

		for (int pass = 0; pass < 64; pass++)
		{
			bool anyResolved = false;
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					int idx = (y * width) + x;
					if (!pending[idx])
						continue;

					long r = 0;
					long g = 0;
					long b = 0;
					int neighbours = 0;
					for (int d = 0; d < 4; d++)
					{
						int nx = x + dx[d];
						int ny = y + dy[d];
						if (nx < 0 || nx >= width || ny < 0 || ny >= height)
							continue;

						int ni = (ny * width) + nx;
						if (!resolved[ni])
							continue;

						r += pixels[ni].R;
						g += pixels[ni].G;
						b += pixels[ni].B;
						neighbours++;
					}

					if (neighbours > 0)
					{
						pixels[idx] = new Rgba32((byte)(r / neighbours), (byte)(g / neighbours), (byte)(b / neighbours), 255);
						resolved[idx] = true;
						pending[idx] = false;
						anyResolved = true;
					}
				}
			}

			if (!anyResolved)
				break;
		}

		for (int i = 0; i < pixels.Length; i++)
		{
			if (pending[i])
				pixels[i] = fallback;
		}

		using Image<Rgba32> result = new(width, height);
		result.ProcessPixelRows(accessor =>
		{
			for (int y = 0; y < height; y++)
			{
				Span<Rgba32> row = accessor.GetRowSpan(y);
				pixels.AsSpan(y * width, width).CopyTo(row);
			}
		});

		using MemoryStream ms = new();
		result.SaveAsPng(ms);
		return ms.ToArray();
	}
	catch
	{
		return [];
	}
}

static bool HasFlag(IEnumerable<string> args, string name)
{
	return args.Any(arg => string.Equals(arg, name, StringComparison.OrdinalIgnoreCase));
}

static string ResolveClientId(MlCorpusClientConfig client)
{
	if (!string.IsNullOrWhiteSpace(client.ClientId))
		return client.ClientId;
	if (!string.IsNullOrWhiteSpace(client.Version))
		return client.Version;

	string pathTail = Path.GetFileName(client.ClientPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
	return string.IsNullOrWhiteSpace(pathTail) ? "client" : pathTail;
}

static IReadOnlyList<MlCorpusMapConfig> ResolveMapsForClient(MlCorpusConfig config, MlCorpusClientConfig client)
{
	if (config.Maps.Count > 0)
		return config.Maps;

	if (client.Maps.Count == 0)
		return [];

	return client.Maps
		.Where(static map => !string.IsNullOrWhiteSpace(map))
		.Select(static map => new MlCorpusMapConfig { MapName = map })
		.ToList();
}

static string ResolveDataPath(string path, string root, string configPath)
{
	if (Path.IsPathRooted(path))
		return Path.GetFullPath(path);

	if (!string.IsNullOrWhiteSpace(root))
		return Path.GetFullPath(Path.Combine(root, path));

	string configDir = Path.GetDirectoryName(configPath) ?? Environment.CurrentDirectory;
	return Path.GetFullPath(Path.Combine(configDir, path));
}

static string ResolveOptionalRoot(string? overrideRoot, string? configRoot, string configPath, string? fallback = null)
{
	if (!string.IsNullOrWhiteSpace(overrideRoot))
		return ResolveDataPath(overrideRoot, string.Empty, configPath);

	if (!string.IsNullOrWhiteSpace(configRoot))
		return ResolveDataPath(configRoot, string.Empty, configPath);

	if (!string.IsNullOrWhiteSpace(fallback))
		return Path.GetFullPath(fallback);

	return string.Empty;
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

static JsonSerializerOptions CreateJsonOptions()
{
	JsonSerializerOptions options = new()
	{
		WriteIndented = true,
	};
	options.Converters.Add(new JsonStringEnumConverter());
	return options;
}

static void ShowUsage()
{
	Console.WriteLine("WowViewer.Tool.Converter");
	Console.WriteLine("Usage:");
	Console.WriteLine("  wowviewer-converter detect --input <file>");
	Console.WriteLine("  wowviewer-converter export-tex-json --input <file.adt|file_tex0.adt> [--output <report.json>]");
	Console.WriteLine("  wowviewer-converter ml-corpus --config <ml-corpus.json> [--archive-root <path>] [--output-root <path>] [--dry-run]");
	Console.WriteLine("  wowviewer-converter ml-synth-no-liquid --input <minimap.png> --mask <liquid-mask.png> --output <no-liquid.png>");
	Console.WriteLine("  wowviewer-converter ml-synth-no-liquid --input-dir <images> --mask-dir <masks> --output-dir <images>");
}

file sealed class MlCorpusConfig
{
	[JsonPropertyName("archive_root")]
	public string? ArchiveRoot { get; set; }

	[JsonPropertyName("default_output_root")]
	public string? DefaultOutputRoot { get; set; }

	[JsonPropertyName("clients")]
	public List<MlCorpusClientConfig> Clients { get; set; } = [];

	[JsonPropertyName("maps")]
	public List<MlCorpusMapConfig> Maps { get; set; } = [];
}

file sealed class MlCorpusClientConfig
{
	[JsonPropertyName("client_id")]
	public string ClientId { get; set; } = string.Empty;

	[JsonPropertyName("version")]
	public string Version { get; set; } = string.Empty;

	[JsonPropertyName("client_path")]
	public string ClientPath { get; set; } = string.Empty;

	[JsonPropertyName("maps")]
	public List<string> Maps { get; set; } = [];
}

file sealed class MlCorpusMapConfig
{
	[JsonPropertyName("map_name")]
	public string MapName { get; set; } = string.Empty;

	[JsonPropertyName("map_path")]
	public string MapPath { get; set; } = string.Empty;
}

file sealed record MlCorpusMapReport(
	[property: JsonPropertyName("client_id")] string ClientId,
	[property: JsonPropertyName("map_name")] string MapName,
	[property: JsonPropertyName("map_path")] string MapPath,
	[property: JsonPropertyName("tile_count")] int TileCount,
	[property: JsonPropertyName("generated_utc")] DateTime GeneratedUtc,
	[property: JsonPropertyName("tiles")] List<MlCorpusTileReport> Tiles);

file sealed record MlCorpusTileReport(
	[property: JsonPropertyName("tile")] string Tile,
	[property: JsonPropertyName("adt_path")] string AdtPath,
	[property: JsonPropertyName("has_obj0")] bool HasObj0,
	[property: JsonPropertyName("terrain_chunks")] int TerrainChunks,
	[property: JsonPropertyName("has_water")] bool HasWater,
	[property: JsonPropertyName("texture_name_count")] int TextureNameCount,
	[property: JsonPropertyName("model_placements")] int ModelPlacementCount,
	[property: JsonPropertyName("world_model_placements")] int WorldModelPlacementCount,
	[property: JsonPropertyName("textures")] string[] Textures);

file sealed class MlCorpusIssueTracker
{
	private readonly Dictionary<string, MlCorpusIssueSummary> _issues = new(StringComparer.Ordinal);

	public void Record(string category, string samplePath, Exception exception)
	{
		string key = $"{category}|{exception.GetType().Name}|{exception.Message}";
		if (_issues.TryGetValue(key, out MlCorpusIssueSummary? existing))
		{
			existing.Count++;
			return;
		}

		_issues[key] = new MlCorpusIssueSummary(category, exception.GetType().Name, exception.Message, samplePath);
	}

	public void Print(string mapLabel)
	{
		foreach (MlCorpusIssueSummary issue in _issues.Values.OrderByDescending(static issue => issue.Count).ThenBy(static issue => issue.Category, StringComparer.Ordinal))
		{
			Console.Error.WriteLine($"Warning: {mapLabel} {issue.Category} failures={issue.Count}; sample={issue.SamplePath}; {issue.ExceptionType}: {issue.Message}");
		}
	}
}

file sealed class MlCorpusIssueSummary
{
	public MlCorpusIssueSummary(string category, string exceptionType, string message, string samplePath)
	{
		Category = category;
		ExceptionType = exceptionType;
		Message = message;
		SamplePath = samplePath;
		Count = 1;
	}

	public string Category { get; }

	public string ExceptionType { get; }

	public string Message { get; }

	public string SamplePath { get; }

	public int Count { get; set; }
}
