using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WowViewer.Core.Datasets;
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
		case "dataset-scan":
			RunDatasetScan(tail);
			break;
		case "detect":
			RunDetect(tail);
			break;
		case "ml-corpus":
			RunMlCorpus(tail);
			break;
	case "ml-audit-signals":
		RunMlAuditSignals(tail);
		break;
	case "ml-harvest-brushes":
		MlBrushImprintHarvester.Run(tail);
		break;
	case "ml-generate-controls":
		MlSyntheticControlGenerator.Run(tail);
		break;
	case "ml-repair-normalmaps":
		MlRepairNormalmapsCommand.Run(tail);
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

static void RunDatasetScan(string[] args)
{
	string? clientRootOption = GetOption(args, "--client-root", "-c");
	string? mapName = GetOption(args, "--map", "-m");
	string? buildLabel = GetOption(args, "--build", "-b");
	string? outputOption = GetOption(args, "--output", "-o");
	int? limit = GetIntOption(args, "--limit", "-n");

	if (string.IsNullOrWhiteSpace(clientRootOption) || string.IsNullOrWhiteSpace(mapName))
	{
		Console.Error.WriteLine("Error: dataset-scan requires --client-root <path> and --map <name>.");
		Environment.ExitCode = 1;
		return;
	}

	string clientRoot = Path.GetFullPath(clientRootOption);
	string normalizedMapName = mapName.Trim();
	string normalizedBuildLabel = string.IsNullOrWhiteSpace(buildLabel)
		? Path.GetFileName(clientRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
		: buildLabel.Trim();

	string mapPath = Path.Combine(clientRoot, "Data", "World", "Maps", normalizedMapName);
	List<TerrainTrainingSampleDescriptor> entries;
	if (Directory.Exists(mapPath))
	{
		entries = BuildDatasetScanEntriesFromDirectory(clientRoot, normalizedBuildLabel, normalizedMapName, mapPath, limit);
	}
	else
	{
		using IArchiveCatalog archiveCatalog = CreateArchiveCatalog(clientRoot);
		if (!ArchiveMapExists(archiveCatalog, clientRoot, normalizedMapName, mapPath))
		{
			Console.Error.WriteLine($"Error: map '{normalizedMapName}' was not found under filesystem or archive root '{clientRoot}'.");
			Environment.ExitCode = 1;
			return;
		}

		entries = BuildDatasetScanEntriesFromArchive(clientRoot, normalizedBuildLabel, normalizedMapName, mapPath, archiveCatalog, limit);
	}

	TerrainTrainingSampleManifest manifest = new(
		schemaVersion: "terrain-training-scan.v2",
		createdAtUtc: DateTimeOffset.UtcNow,
		sourceManifestKind: "scan",
		entries: entries);

	Console.WriteLine("WowViewer.Tool.Converter dataset-scan report");
	Console.WriteLine($"ClientRoot: {clientRoot}");
	Console.WriteLine($"Build: {normalizedBuildLabel}");
	Console.WriteLine($"Map: {normalizedMapName}");
	Console.WriteLine($"Samples: {entries.Count}");

	string json = JsonSerializer.Serialize(manifest, CreateJsonOptions());
	if (!string.IsNullOrWhiteSpace(outputOption))
	{
		string outputPath = Path.GetFullPath(outputOption);
		string? outputDirectory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(outputDirectory))
			Directory.CreateDirectory(outputDirectory);

		File.WriteAllText(outputPath, json);
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	Console.WriteLine(json);
}

static List<TerrainTrainingSampleDescriptor> BuildDatasetScanEntriesFromDirectory(string clientRoot, string buildLabel, string mapName, string mapPath, int? limit)
{
	List<string> adtFiles = Directory
		.EnumerateFiles(mapPath, $"{mapName}_*.adt", SearchOption.TopDirectoryOnly)
		.Where(static path => !path.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
			&& !path.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase)
			&& !path.EndsWith("_lod.adt", StringComparison.OrdinalIgnoreCase))
		.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase)
		.ToList();

	if (limit is > 0)
		adtFiles = adtFiles.Take(limit.Value).ToList();

	string? wdlPath = ResolveOptionalFilesystemCompanion(Path.Combine(mapPath, $"{mapName}.wdl"));
	List<TerrainTrainingSampleDescriptor> entries = new(adtFiles.Count);
	foreach (string adtPath in adtFiles)
	{
		string tileStem = Path.GetFileNameWithoutExtension(adtPath);
		if (!TryParseTileCoordinates(tileStem, out int tileX, out int tileY))
			continue;

		string? objAdtPath = ResolveOptionalFilesystemCompanion(Path.Combine(mapPath, $"{tileStem}_obj0.adt"));
		string? texAdtPath = ResolveOptionalFilesystemCompanion(Path.Combine(mapPath, $"{tileStem}_tex0.adt"));
		string? lodAdtPath = ResolveOptionalFilesystemCompanion(Path.Combine(mapPath, $"{tileStem}_lod.adt"));
		AdtSummary summary = AdtSummaryReader.Read(adtPath);

		entries.Add(CreateDatasetScanEntry(
			sampleId: $"{buildLabel}:{tileStem}",
			sourceKind: TerrainTrainingSampleSourceKind.ClientRoot,
			buildLabel: buildLabel,
			mapName: mapName,
			tileX: tileX,
			tileY: tileY,
			sourceRoot: clientRoot,
			rootAdtPath: adtPath,
			objAdtPath: objAdtPath,
			texAdtPath: texAdtPath,
			lodAdtPath: lodAdtPath,
			wdlPath: wdlPath,
			summary: summary,
			hasMinimap: false,
			hasTerrainOnlyMinimap: false,
			hasNoLiquidMinimap: false,
			hasNoObjectMinimap: false,
			hasNoMccvMinimap: false,
			hasNormalMap: false,
			hasLiquidMask: summary.HasWater,
			hasLiquidHeight: summary.HasWater,
			hasObjectMask: false,
			hasPm4Mask: false,
			hasHoleMask: false,
			hasAreaIdMap: false,
			hasChunkFlagsMap: false,
			hasAlphaLayers: texAdtPath is not null,
			hasTextureMetadata: texAdtPath is not null || summary.TextureNameCount > 0));
	}

	return entries;
}

static List<TerrainTrainingSampleDescriptor> BuildDatasetScanEntriesFromArchive(string clientRoot, string buildLabel, string mapName, string mapPath, IArchiveCatalog archiveCatalog, int? limit)
{
	string mapVirtualRoot = ResolveMapVirtualRoot(clientRoot, mapName, mapPath);
	string wdtVirtualPath = BuildMapWdtVirtualPath(mapVirtualRoot, mapName);
	byte[] wdtBytes = archiveCatalog.ReadFile(wdtVirtualPath)
		?? throw new FileNotFoundException($"Could not read archive-backed WDT '{wdtVirtualPath}'.", wdtVirtualPath);

	List<WdtTileCoordinate> tileCoordinates = ReadArchiveWdtTiles(wdtBytes, wdtVirtualPath)
		.OrderBy(static tile => tile.TileY)
		.ThenBy(static tile => tile.TileX)
		.ToList();
	if (limit is > 0)
		tileCoordinates = tileCoordinates.Take(limit.Value).ToList();

	string wdlVirtualPath = $"{mapVirtualRoot}\\{mapName}.wdl";
	bool hasWdl = archiveCatalog.FileExists(wdlVirtualPath);
	List<TerrainTrainingSampleDescriptor> entries = new(tileCoordinates.Count);
	foreach (WdtTileCoordinate tileCoordinate in tileCoordinates)
	{
		string tileStem = $"{mapName}_{tileCoordinate.TileX}_{tileCoordinate.TileY}";
		string rootVirtualPath = $"{mapVirtualRoot}\\{tileStem}.adt";
		byte[] rootBytes = archiveCatalog.ReadFile(rootVirtualPath) ?? [];
		if (rootBytes.Length == 0)
			continue;

		AdtSummary summary = ReadArchiveAdtSummary(rootBytes, rootVirtualPath);
		string objVirtualPath = $"{mapVirtualRoot}\\{tileStem}_obj0.adt";
		string texVirtualPath = $"{mapVirtualRoot}\\{tileStem}_tex0.adt";
		string lodVirtualPath = $"{mapVirtualRoot}\\{tileStem}_lod.adt";

		entries.Add(CreateDatasetScanEntry(
			sampleId: $"{buildLabel}:{tileStem}",
			sourceKind: TerrainTrainingSampleSourceKind.MountedArchive,
			buildLabel: buildLabel,
			mapName: mapName,
			tileX: tileCoordinate.TileX,
			tileY: tileCoordinate.TileY,
			sourceRoot: clientRoot,
			rootAdtPath: rootVirtualPath,
			objAdtPath: archiveCatalog.FileExists(objVirtualPath) ? objVirtualPath : null,
			texAdtPath: archiveCatalog.FileExists(texVirtualPath) ? texVirtualPath : null,
			lodAdtPath: archiveCatalog.FileExists(lodVirtualPath) ? lodVirtualPath : null,
			wdlPath: hasWdl ? wdlVirtualPath : null,
			summary: summary,
			hasMinimap: false,
			hasTerrainOnlyMinimap: false,
			hasNoLiquidMinimap: false,
			hasNoObjectMinimap: false,
			hasNoMccvMinimap: false,
			hasNormalMap: false,
			hasLiquidMask: summary.HasWater,
			hasLiquidHeight: summary.HasWater,
			hasObjectMask: false,
			hasPm4Mask: false,
			hasHoleMask: false,
			hasAreaIdMap: false,
			hasChunkFlagsMap: false,
			hasAlphaLayers: archiveCatalog.FileExists(texVirtualPath),
			hasTextureMetadata: archiveCatalog.FileExists(texVirtualPath) || summary.TextureNameCount > 0));
	}

	return entries;
}

static TerrainTrainingSampleDescriptor CreateDatasetScanEntry(
	string sampleId,
	TerrainTrainingSampleSourceKind sourceKind,
	string buildLabel,
	string mapName,
	int tileX,
	int tileY,
	string sourceRoot,
	string rootAdtPath,
	string? objAdtPath,
	string? texAdtPath,
	string? lodAdtPath,
	string? wdlPath,
	AdtSummary summary,
	bool hasMinimap,
	bool hasTerrainOnlyMinimap,
	bool hasNoLiquidMinimap,
	bool hasNoObjectMinimap,
	bool hasNoMccvMinimap,
	bool hasNormalMap,
	bool hasLiquidMask,
	bool hasLiquidHeight,
	bool hasObjectMask,
	bool hasPm4Mask,
	bool hasHoleMask,
	bool hasAreaIdMap,
	bool hasChunkFlagsMap,
	bool hasAlphaLayers,
	bool hasTextureMetadata)
{
	return new TerrainTrainingSampleDescriptor(
		sampleId: sampleId,
		sourceKind: sourceKind,
		buildLabel: buildLabel,
		mapName: mapName,
		tileX: tileX,
		tileY: tileY,
		sourceRoot: sourceRoot,
		rootAdtPath: rootAdtPath)
	{
		ObjAdtPath = objAdtPath,
		TexAdtPath = texAdtPath,
		LodAdtPath = lodAdtPath,
		WdlPath = wdlPath,
		Signals = new TerrainTrainingSignalAvailability
		{
			HasRootAdt = true,
			HasObjAdt = objAdtPath is not null,
			HasTexAdt = texAdtPath is not null,
			HasWdl = wdlPath is not null,
			HasMinimap = hasMinimap,
			HasTerrainOnlyMinimap = hasTerrainOnlyMinimap,
			HasNoLiquidMinimap = hasNoLiquidMinimap,
			HasNoObjectMinimap = hasNoObjectMinimap,
			HasNoMccvMinimap = hasNoMccvMinimap,
			HasNormalMap = hasNormalMap,
			HasLiquidMask = hasLiquidMask,
			HasLiquidHeight = hasLiquidHeight,
			HasObjectMask = hasObjectMask,
			HasBrushMask = false,
			HasPm4Mask = hasPm4Mask,
			HasHoleMask = hasHoleMask,
			HasAreaIdMap = hasAreaIdMap,
			HasChunkFlagsMap = hasChunkFlagsMap,
			HasAlphaLayers = hasAlphaLayers,
			HasTextureMetadata = hasTextureMetadata,
		},
		Metrics = new TerrainTrainingSampleMetrics
		{
			LiquidCoverage = summary.HasWater ? 1.0f : 0.0f,
			BrushCoverage = 0.0f,
			TextureLayerCount = summary.TextureNameCount,
		},
	};
}

static bool TryParseTileCoordinates(string tileStem, out int tileX, out int tileY)
{
	tileX = 0;
	tileY = 0;
	int lastUnderscore = tileStem.LastIndexOf('_');
	if (lastUnderscore <= 0 || lastUnderscore >= tileStem.Length - 1)
		return false;

	int secondLastUnderscore = tileStem.LastIndexOf('_', lastUnderscore - 1);
	if (secondLastUnderscore <= 0 || secondLastUnderscore >= lastUnderscore - 1)
		return false;

	return int.TryParse(tileStem.AsSpan(secondLastUnderscore + 1, lastUnderscore - secondLastUnderscore - 1), out tileX)
		&& int.TryParse(tileStem.AsSpan(lastUnderscore + 1), out tileY);
}

static string? ResolveOptionalFilesystemCompanion(string path)
{
	return File.Exists(path) ? path : null;
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

static void RunMlAuditSignals(string[] args)
{
	string? datasetRootOption = GetOption(args, "--dataset-root", "-d")
		?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	if (string.IsNullOrWhiteSpace(datasetRootOption))
	{
		Console.Error.WriteLine("Error: --dataset-root <path> is required.");
		Environment.ExitCode = 1;
		return;
	}

	string datasetRoot = Path.GetFullPath(datasetRootOption);
	string datasetDirectory = Path.Combine(datasetRoot, "dataset");
	if (!Directory.Exists(datasetDirectory))
	{
		Console.Error.WriteLine($"Error: dataset directory not found: {datasetDirectory}");
		Environment.ExitCode = 1;
		return;
	}

	string? outputPathOption = GetOption(args, "--output", "-o");
	int? limit = GetIntOption(args, "--limit", "-n");

	List<string> datasetFiles = Directory
		.EnumerateFiles(datasetDirectory, "*.json", SearchOption.TopDirectoryOnly)
		.Where(static path => !string.Equals(Path.GetFileName(path), "texture_database.json", StringComparison.OrdinalIgnoreCase))
		.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase)
		.ToList();

	if (limit is > 0)
		datasetFiles = datasetFiles.Take(limit.Value).ToList();

	if (datasetFiles.Count == 0)
	{
		Console.Error.WriteLine($"Error: no dataset tile JSON files found in {datasetDirectory}");
		Environment.ExitCode = 1;
		return;
	}

	Console.WriteLine("WowViewer.Tool.Converter ml-audit-signals report");
	Console.WriteLine($"DatasetRoot: {datasetRoot}");
	Console.WriteLine($"DatasetDirectory: {datasetDirectory}");
	Console.WriteLine($"TileJsonCount: {datasetFiles.Count}");

	MlCorpusIssueTracker issueTracker = new();
	List<MlSignalAuditTileDraft> drafts = new(datasetFiles.Count);
	foreach (string datasetFile in datasetFiles)
	{
		try
		{
			drafts.Add(CreateMlSignalAuditTileDraft(datasetRoot, datasetFile));
		}
		catch (Exception ex)
		{
			issueTracker.Record("ml-audit-signals", datasetFile, ex);
		}
	}

	List<MlSignalAuditGroupSummary> dedupeGroups = BuildMlSignalAuditGroupSummaries(
		drafts,
		static draft => draft.DedupeSignature,
		"dedupe");
	Dictionary<string, string> dedupeGroupIds = dedupeGroups.ToDictionary(static group => group.Signature, static group => group.GroupId, StringComparer.Ordinal);

	List<MlSignalAuditGroupSummary> conceptGroups = BuildMlSignalAuditGroupSummaries(
		drafts,
		static draft => draft.ConceptSignature,
		"concept");
	Dictionary<string, string> conceptGroupIds = conceptGroups.ToDictionary(static group => group.Signature, static group => group.GroupId, StringComparer.Ordinal);

	foreach (IGrouping<string, MlSignalAuditTileDraft> group in drafts
		.GroupBy(static draft => draft.DedupeSignature, StringComparer.Ordinal)
		.OrderByDescending(static group => group.Count())
		.ThenBy(static group => group.Key, StringComparer.Ordinal))
	{
		string groupId = dedupeGroupIds[group.Key];
		foreach (MlSignalAuditTileDraft draft in group.OrderBy(static draft => draft.TileName, StringComparer.OrdinalIgnoreCase))
			draft.DedupeGroupId = groupId;

		MlSignalAuditTileDraft canonical = group
			.OrderBy(static draft => draft.TileName, StringComparer.OrdinalIgnoreCase)
			.First();
		canonical.RetentionRecommendation = "canonical";

		foreach (MlSignalAuditTileDraft duplicate in group)
		{
			if (!ReferenceEquals(duplicate, canonical))
				duplicate.RetentionRecommendation = group.Count() > 1 ? "review-duplicate" : "canonical";
		}
	}

	foreach (MlSignalAuditTileDraft draft in drafts)
		draft.ConceptClusterId = conceptGroupIds[draft.ConceptSignature];

	List<MlSignalAuditTileReport> tiles = drafts
		.OrderBy(static draft => draft.MapName, StringComparer.OrdinalIgnoreCase)
		.ThenBy(static draft => draft.TileName, StringComparer.OrdinalIgnoreCase)
		.Select(static draft => draft.ToReport())
		.ToList();

	MlSignalAuditCoverage coverage = new(
		TilesProcessed: tiles.Count,
		TilesWithSourceMinimap: tiles.Count(static tile => tile.SourceMinimapExists),
		TilesWithLocalHeightmap: tiles.Count(static tile => tile.HeightmapLocalExists),
		TilesWithGlobalHeightmap: tiles.Count(static tile => tile.HeightmapGlobalExists),
		TilesWithAlphaAtlas: tiles.Count(static tile => tile.AlphaAtlasExists),
		TilesWithAnyAlphaMask: tiles.Count(static tile => tile.AlphaMaskCount > 0),
		TilesWithObjects: tiles.Count(static tile => tile.ObjectCount > 0),
		TilesWithLiquidMask: tiles.Count(static tile => tile.LiquidMaskExists),
		TilesWithNoLiquidMinimap: tiles.Count(static tile => tile.NoLiquidMinimapExists),
		TilesWithDeclaredLiquidLayers: tiles.Count(static tile => tile.LiquidLayerCount > 0),
		VisibleSurfaceLiquidTiles: tiles.Count(static tile => string.Equals(tile.LiquidSemanticClass, "visible-surface", StringComparison.Ordinal)),
		BelowTerrainLikelyLiquidTiles: tiles.Count(static tile => string.Equals(tile.LiquidSemanticClass, "below-terrain-likely", StringComparison.Ordinal)),
		UncertainLiquidTiles: tiles.Count(static tile => string.Equals(tile.LiquidSemanticClass, "uncertain", StringComparison.Ordinal)),
		NoLiquidTiles: tiles.Count(static tile => string.Equals(tile.LiquidSemanticClass, "none", StringComparison.Ordinal)),
		ConceptClusterCount: conceptGroups.Count,
		DedupeGroupCount: dedupeGroups.Count,
		DuplicateTileCount: dedupeGroups.Where(static group => group.TileCount > 1).Sum(static group => group.TileCount),
		RetainedCanonicalTileCount: tiles.Count(static tile => string.Equals(tile.RetentionRecommendation, "canonical", StringComparison.Ordinal)),
		ReviewDuplicateTileCount: tiles.Count(static tile => string.Equals(tile.RetentionRecommendation, "review-duplicate", StringComparison.Ordinal)));

	MlSignalAuditReport report = new(
		SchemaVersion: "wowviewer-ml-signal-audit.v1",
		GeneratedUtc: DateTime.UtcNow,
		DatasetRoot: datasetRoot,
		DatasetDirectory: datasetDirectory,
		TileCount: tiles.Count,
		Coverage: coverage,
		DedupeGroups: dedupeGroups,
		ConceptClusters: conceptGroups,
		Tiles: tiles);

	issueTracker.Print(Path.GetFileName(datasetRoot));
	Console.WriteLine($"Audit complete: tiles={coverage.TilesProcessed} concept_clusters={coverage.ConceptClusterCount} dedupe_groups={coverage.DedupeGroupCount} visible_surface_liquid={coverage.VisibleSurfaceLiquidTiles} below_terrain_likely={coverage.BelowTerrainLikelyLiquidTiles} uncertain_liquid={coverage.UncertainLiquidTiles}");

	string json = JsonSerializer.Serialize(report, CreateJsonOptions());
	if (!string.IsNullOrWhiteSpace(outputPathOption))
	{
		string outputPath = Path.GetFullPath(outputPathOption);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, json);
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	Console.WriteLine(json);
}

static MlSignalAuditTileDraft CreateMlSignalAuditTileDraft(string datasetRoot, string datasetFile)
{
	MlAuditDatasetSample sample = JsonSerializer.Deserialize<MlAuditDatasetSample>(
		File.ReadAllText(datasetFile),
		new JsonSerializerOptions
		{
			PropertyNameCaseInsensitive = true,
			NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
		}) ?? throw new InvalidDataException($"Failed to parse dataset tile JSON '{datasetFile}'.");

	if (sample.TerrainData is null)
		throw new InvalidDataException($"Dataset tile JSON '{datasetFile}' is missing terrain_data.");

	string tileName = string.IsNullOrWhiteSpace(sample.TerrainData.AdtTile)
		? Path.GetFileNameWithoutExtension(datasetFile)
		: sample.TerrainData.AdtTile;
	string mapName = ExtractAuditMapName(tileName);

	string? sourceMinimapPath = ResolveDatasetPath(datasetRoot, sample.ImagePath);
	string? heightmapLocalPath = ResolveDatasetPath(datasetRoot, sample.TerrainData.HeightmapLocalPath ?? sample.TerrainData.HeightmapPath);
	string? heightmapGlobalPath = ResolveDatasetPath(datasetRoot, sample.TerrainData.HeightmapGlobalPath);
	string? alphaAtlasPath = ResolveDatasetPath(datasetRoot, sample.TerrainData.AlphaAtlasPath);
	string? liquidMaskPath = ResolveDatasetPath(datasetRoot, sample.TerrainData.LiquidMaskPath);
	string? noLiquidMinimapPath = ResolveDatasetPath(datasetRoot, sample.TerrainData.NoLiquidMinimapPath);

	List<string> textures = sample.TerrainData.Textures
		.Where(static texture => !string.IsNullOrWhiteSpace(texture))
		.Select(static texture => texture.Trim())
		.Distinct(StringComparer.OrdinalIgnoreCase)
		.OrderBy(static texture => texture, StringComparer.OrdinalIgnoreCase)
		.ToList();

	MlSignalAuditImageSignature? sourceSignature = TryBuildAuditImageSignature(sourceMinimapPath);
	MlSignalAuditImageSignature? alphaAtlasSignature = TryBuildAuditImageSignature(alphaAtlasPath);
	MlSignalAuditLiquidAssessment liquidAssessment = ClassifyLiquidSemantics(
		sourceMinimapPath,
		noLiquidMinimapPath,
		liquidMaskPath,
		sample.TerrainData.Liquids);

	string textureSignature = ComputeStableHash(string.Join("|", textures.Select(static texture => texture.ToLowerInvariant())));
	string dedupeSignature = ComputeStableHash(string.Join("|", new[]
	{
		sourceSignature?.Sha256 ?? "none",
		alphaAtlasSignature?.Sha256 ?? "none",
		textureSignature,
		sample.TerrainData.ChunkLayers?.Length.ToString() ?? "0",
		(sample.TerrainData.Objects?.Length ?? 0).ToString(),
		liquidAssessment.LiquidSemanticClass,
	}));
	string conceptSignature = ComputeStableHash(string.Join("|", new[]
	{
		mapName.ToLowerInvariant(),
		sourceSignature?.AverageHash64 ?? "none",
		alphaAtlasSignature?.AverageHash64 ?? "none",
		textureSignature,
		BucketizeCount(sample.TerrainData.Objects?.Length ?? 0),
		BucketizeCount(sample.TerrainData.ChunkLayers?.Length ?? 0),
		liquidAssessment.LiquidSemanticClass,
	}));

	return new MlSignalAuditTileDraft
	{
		TileName = tileName,
		MapName = mapName,
		TileJsonPath = RelativizeAuditPath(datasetRoot, datasetFile),
		SourceMinimapExists = sourceMinimapPath is not null,
		HeightmapLocalExists = heightmapLocalPath is not null,
		HeightmapGlobalExists = heightmapGlobalPath is not null,
		AlphaAtlasExists = alphaAtlasPath is not null,
		AlphaMaskCount = sample.TerrainData.AlphaMasks?.Count(static path => !string.IsNullOrWhiteSpace(path)) ?? 0,
		TextureCount = textures.Count,
		TextureSignature = textureSignature,
		ObjectCount = sample.TerrainData.Objects?.Length ?? 0,
		ChunkLayerCount = sample.TerrainData.ChunkLayers?.Length ?? 0,
		LiquidLayerCount = sample.TerrainData.Liquids?.Length ?? 0,
		LiquidMaskExists = liquidMaskPath is not null,
		NoLiquidMinimapExists = noLiquidMinimapPath is not null,
		LiquidSemanticClass = liquidAssessment.LiquidSemanticClass,
		LiquidSemanticReason = liquidAssessment.Reason,
		LiquidMaskPixelCount = liquidAssessment.MaskedPixelCount,
		LiquidMeanRgbDelta = liquidAssessment.MeanRgbDelta,
		DedupeSignature = dedupeSignature,
		ConceptSignature = conceptSignature,
		RetentionRecommendation = "canonical",
		DedupeGroupId = $"dedupe-{dedupeSignature[..12].ToLowerInvariant()}",
		ConceptClusterId = $"concept-{conceptSignature[..12].ToLowerInvariant()}"
	};
}

static List<MlSignalAuditGroupSummary> BuildMlSignalAuditGroupSummaries(
	IEnumerable<MlSignalAuditTileDraft> drafts,
	Func<MlSignalAuditTileDraft, string> signatureSelector,
	string prefix)
{
	return drafts
		.GroupBy(signatureSelector, StringComparer.Ordinal)
		.OrderByDescending(static group => group.Count())
		.ThenBy(static group => group.Key, StringComparer.Ordinal)
		.Select(group =>
		{
			List<string> members = group
				.Select(static draft => draft.TileName)
				.OrderBy(static tile => tile, StringComparer.OrdinalIgnoreCase)
				.ToList();

			return new MlSignalAuditGroupSummary(
				GroupId: $"{prefix}-{group.Key[..12].ToLowerInvariant()}",
				Signature: group.Key,
				TileCount: members.Count,
				RepresentativeTile: members[0],
				Tiles: members);
		})
		.ToList();
}

static MlSignalAuditLiquidAssessment ClassifyLiquidSemantics(string? sourceMinimapPath, string? noLiquidMinimapPath, string? liquidMaskPath, MlAuditDatasetLiquid[]? liquids)
{
	int liquidLayerCount = liquids?.Length ?? 0;
	if (string.IsNullOrWhiteSpace(liquidMaskPath) && liquidLayerCount == 0)
		return new MlSignalAuditLiquidAssessment("none", "no liquid mask or liquid layer payload present", 0, null);

	if (string.IsNullOrWhiteSpace(liquidMaskPath))
		return new MlSignalAuditLiquidAssessment("uncertain", "liquid layers exist without a stitched liquid mask", 0, null);

	if (string.IsNullOrWhiteSpace(sourceMinimapPath) || string.IsNullOrWhiteSpace(noLiquidMinimapPath))
		return new MlSignalAuditLiquidAssessment("uncertain", "liquid semantic audit requires source minimap and no-liquid minimap", 0, null);

	MlSignalAuditMaskDiff? diff = TryAnalyzeMaskedDifference(sourceMinimapPath, noLiquidMinimapPath, liquidMaskPath);
	if (diff is null)
		return new MlSignalAuditLiquidAssessment("uncertain", "failed to compare source minimap against no-liquid minimap under liquid mask", 0, null);

	if (diff.MaskedPixelCount <= 0)
		return new MlSignalAuditLiquidAssessment("uncertain", "liquid mask contained no positive coverage", 0, null);

	if (diff.MeanRgbDelta < 8.0)
		return new MlSignalAuditLiquidAssessment("below-terrain-likely", "masked liquid region changes too little when liquids are inpainted out", diff.MaskedPixelCount, diff.MeanRgbDelta);

	return new MlSignalAuditLiquidAssessment("visible-surface", "masked liquid region changes materially when liquids are inpainted out", diff.MaskedPixelCount, diff.MeanRgbDelta);
}

static MlSignalAuditMaskDiff? TryAnalyzeMaskedDifference(string sourceMinimapPath, string noLiquidMinimapPath, string liquidMaskPath)
{
	try
	{
		using Image<Rgba32> source = Image.Load<Rgba32>(sourceMinimapPath);
		using Image<Rgba32> noLiquid = Image.Load<Rgba32>(noLiquidMinimapPath);
		using Image<L8> mask = Image.Load<L8>(liquidMaskPath);

		if (noLiquid.Width != source.Width || noLiquid.Height != source.Height)
			noLiquid.Mutate(ctx => ctx.Resize(source.Width, source.Height));

		if (mask.Width != source.Width || mask.Height != source.Height)
			mask.Mutate(ctx => ctx.Resize(source.Width, source.Height));

		long maskedPixelCount = 0;
		double totalDelta = 0;
		for (int y = 0; y < source.Height; y++)
		{
			for (int x = 0; x < source.Width; x++)
			{
				if (mask[x, y].PackedValue < 128)
					continue;

				Rgba32 sourcePixel = source[x, y];
				Rgba32 noLiquidPixel = noLiquid[x, y];
				totalDelta += (Math.Abs(sourcePixel.R - noLiquidPixel.R)
					+ Math.Abs(sourcePixel.G - noLiquidPixel.G)
					+ Math.Abs(sourcePixel.B - noLiquidPixel.B)) / 3.0;
				maskedPixelCount++;
			}
		}

		double meanRgbDelta = maskedPixelCount > 0 ? totalDelta / maskedPixelCount : 0;
		return new MlSignalAuditMaskDiff((int)maskedPixelCount, meanRgbDelta);
	}
	catch
	{
		return null;
	}
}

static MlSignalAuditImageSignature? TryBuildAuditImageSignature(string? resolvedPath)
{
	if (string.IsNullOrWhiteSpace(resolvedPath) || !File.Exists(resolvedPath))
		return null;

	try
	{
		byte[] bytes = File.ReadAllBytes(resolvedPath);
		string sha256Hex = Convert.ToHexString(SHA256.HashData(bytes));

		using Image<Rgba32> image = Image.Load<Rgba32>(resolvedPath);
		using Image<Rgba32> reduced = image.Clone(ctx => ctx.Resize(8, 8).Grayscale());
		byte[] values = new byte[64];
		int index = 0;
		int total = 0;
		reduced.ProcessPixelRows(accessor =>
		{
			for (int y = 0; y < accessor.Height; y++)
			{
				Span<Rgba32> row = accessor.GetRowSpan(y);
				for (int x = 0; x < row.Length; x++)
				{
					byte value = row[x].R;
					values[index++] = value;
					total += value;
				}
			}
		});

		int average = total / values.Length;
		ulong hash = 0;
		for (int i = 0; i < values.Length; i++)
		{
			if (values[i] >= average)
				hash |= 1UL << i;
		}

		return new MlSignalAuditImageSignature(sha256Hex, hash.ToString("X16"));
	}
	catch
	{
		return null;
	}
}

static string? ResolveDatasetPath(string datasetRoot, string? path)
{
	if (string.IsNullOrWhiteSpace(path))
		return null;

	string normalizedPath = path.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
	string candidate = Path.IsPathRooted(normalizedPath)
		? normalizedPath
		: Path.Combine(datasetRoot, normalizedPath);
	return File.Exists(candidate) ? candidate : null;
}

static string RelativizeAuditPath(string root, string path)
{
	try
	{
		string relative = Path.GetRelativePath(root, path).Replace('\\', '/');
		return relative.StartsWith("../", StringComparison.Ordinal) ? Path.GetFullPath(path) : relative;
	}
	catch
	{
		return Path.GetFullPath(path);
	}
}

static string ExtractAuditMapName(string tileName)
{
	int lastSeparator = tileName.LastIndexOf('_');
	if (lastSeparator <= 0)
		return tileName;

	int secondLastSeparator = tileName.LastIndexOf('_', lastSeparator - 1);
	if (secondLastSeparator <= 0)
		return tileName;

	return tileName[..secondLastSeparator];
}

static string ComputeStableHash(string value)
{
	byte[] bytes = System.Text.Encoding.UTF8.GetBytes(value);
	return Convert.ToHexString(SHA256.HashData(bytes));
}

static string BucketizeCount(int count)
{
	return count switch
	{
		<= 0 => "0",
		1 => "1",
		<= 4 => "2-4",
		<= 16 => "5-16",
		<= 64 => "17-64",
		_ => "65+",
	};
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

static int? GetIntOption(string[] args, string longName, string shortName)
{
	string? value = GetOption(args, longName, shortName);
	if (string.IsNullOrWhiteSpace(value))
		return null;

	return int.TryParse(value, out int parsed) ? parsed : null;
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
	Console.WriteLine("  wowviewer-converter dataset-scan --client-root <path> --map <name> [--build <label>] [--output <manifest.json>] [--limit <count>]");
	Console.WriteLine("  wowviewer-converter detect --input <file>");
	Console.WriteLine("  wowviewer-converter export-tex-json --input <file.adt|file_tex0.adt> [--output <report.json>]");
	Console.WriteLine("  wowviewer-converter ml-corpus --config <ml-corpus.json> [--archive-root <path>] [--output-root <path>] [--dry-run]");
	Console.WriteLine("  wowviewer-converter ml-audit-signals --dataset-root <path> [--output <report.json>] [--limit <count>]");
	Console.WriteLine("  wowviewer-converter ml-harvest-brushes --dataset-root <path> [--output-dir <dir>] [--limit <count>] [--write-previews]");
	Console.WriteLine("  wowviewer-converter ml-generate-controls [--dataset-root <path>] [--map-name <name>]");
	Console.WriteLine("  wowviewer-converter ml-repair-normalmaps --dataset-root <path> [--report <report.json>] [--limit <count>] [--rewrite-existing] [--rewrite-when-local-differs <mae>] [--only-liquid-tiles] [--dry-run]");
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

file sealed class MlAuditDatasetSample
{
	[JsonPropertyName("image")]
	public string? ImagePath { get; set; }

	[JsonPropertyName("terrain_data")]
	public MlAuditDatasetTerrainData? TerrainData { get; set; }
}

file sealed class MlAuditDatasetTerrainData
{
	[JsonPropertyName("adt_tile")]
	public string? AdtTile { get; set; }

	[JsonPropertyName("heightmap")]
	public string? HeightmapPath { get; set; }

	[JsonPropertyName("heightmap_local")]
	public string? HeightmapLocalPath { get; set; }

	[JsonPropertyName("heightmap_global")]
	public string? HeightmapGlobalPath { get; set; }

	[JsonPropertyName("alpha_masks")]
	public string[]? AlphaMasks { get; set; }

	[JsonPropertyName("alpha_atlas")]
	public string? AlphaAtlasPath { get; set; }

	[JsonPropertyName("liquid_mask")]
	public string? LiquidMaskPath { get; set; }

	[JsonPropertyName("no_liquid_minimap")]
	public string? NoLiquidMinimapPath { get; set; }

	[JsonPropertyName("chunk_layers")]
	public object[]? ChunkLayers { get; set; }

	[JsonPropertyName("liquids")]
	public MlAuditDatasetLiquid[]? Liquids { get; set; }

	[JsonPropertyName("objects")]
	public object[]? Objects { get; set; }

	[JsonPropertyName("textures")]
	public List<string> Textures { get; set; } = [];
}

file sealed class MlAuditDatasetLiquid
{
	[JsonPropertyName("idx")]
	public int ChunkIndex { get; set; }
}

file sealed class MlSignalAuditTileDraft
{
	public string TileName { get; set; } = string.Empty;
	public string MapName { get; set; } = string.Empty;
	public string TileJsonPath { get; set; } = string.Empty;
	public bool SourceMinimapExists { get; set; }
	public bool HeightmapLocalExists { get; set; }
	public bool HeightmapGlobalExists { get; set; }
	public bool AlphaAtlasExists { get; set; }
	public int AlphaMaskCount { get; set; }
	public int TextureCount { get; set; }
	public string TextureSignature { get; set; } = string.Empty;
	public int ObjectCount { get; set; }
	public int ChunkLayerCount { get; set; }
	public int LiquidLayerCount { get; set; }
	public bool LiquidMaskExists { get; set; }
	public bool NoLiquidMinimapExists { get; set; }
	public string LiquidSemanticClass { get; set; } = "uncertain";
	public string LiquidSemanticReason { get; set; } = string.Empty;
	public int LiquidMaskPixelCount { get; set; }
	public double? LiquidMeanRgbDelta { get; set; }
	public string DedupeSignature { get; set; } = string.Empty;
	public string ConceptSignature { get; set; } = string.Empty;
	public string DedupeGroupId { get; set; } = string.Empty;
	public string ConceptClusterId { get; set; } = string.Empty;
	public string RetentionRecommendation { get; set; } = "canonical";

	public MlSignalAuditTileReport ToReport()
	{
		return new MlSignalAuditTileReport(
			TileName: TileName,
			MapName: MapName,
			TileJsonPath: TileJsonPath,
			ConceptClusterId: ConceptClusterId,
			DedupeGroupId: DedupeGroupId,
			RetentionRecommendation: RetentionRecommendation,
			SourceMinimapExists: SourceMinimapExists,
			HeightmapLocalExists: HeightmapLocalExists,
			HeightmapGlobalExists: HeightmapGlobalExists,
			AlphaAtlasExists: AlphaAtlasExists,
			AlphaMaskCount: AlphaMaskCount,
			TextureCount: TextureCount,
			TextureSignature: TextureSignature,
			ObjectCount: ObjectCount,
			ChunkLayerCount: ChunkLayerCount,
			LiquidLayerCount: LiquidLayerCount,
			LiquidMaskExists: LiquidMaskExists,
			NoLiquidMinimapExists: NoLiquidMinimapExists,
			LiquidSemanticClass: LiquidSemanticClass,
			LiquidSemanticReason: LiquidSemanticReason,
			LiquidMaskPixelCount: LiquidMaskPixelCount,
			LiquidMeanRgbDelta: LiquidMeanRgbDelta,
			DedupeSignature: DedupeSignature,
			ConceptSignature: ConceptSignature);
	}
}

file sealed record MlSignalAuditReport(
	[property: JsonPropertyName("schema_version")] string SchemaVersion,
	[property: JsonPropertyName("generated_utc")] DateTime GeneratedUtc,
	[property: JsonPropertyName("dataset_root")] string DatasetRoot,
	[property: JsonPropertyName("dataset_directory")] string DatasetDirectory,
	[property: JsonPropertyName("tile_count")] int TileCount,
	[property: JsonPropertyName("coverage")] MlSignalAuditCoverage Coverage,
	[property: JsonPropertyName("dedupe_groups")] List<MlSignalAuditGroupSummary> DedupeGroups,
	[property: JsonPropertyName("concept_clusters")] List<MlSignalAuditGroupSummary> ConceptClusters,
	[property: JsonPropertyName("tiles")] List<MlSignalAuditTileReport> Tiles);

file sealed record MlSignalAuditCoverage(
	[property: JsonPropertyName("tiles_processed")] int TilesProcessed,
	[property: JsonPropertyName("tiles_with_source_minimap")] int TilesWithSourceMinimap,
	[property: JsonPropertyName("tiles_with_local_heightmap")] int TilesWithLocalHeightmap,
	[property: JsonPropertyName("tiles_with_global_heightmap")] int TilesWithGlobalHeightmap,
	[property: JsonPropertyName("tiles_with_alpha_atlas")] int TilesWithAlphaAtlas,
	[property: JsonPropertyName("tiles_with_any_alpha_mask")] int TilesWithAnyAlphaMask,
	[property: JsonPropertyName("tiles_with_objects")] int TilesWithObjects,
	[property: JsonPropertyName("tiles_with_liquid_mask")] int TilesWithLiquidMask,
	[property: JsonPropertyName("tiles_with_no_liquid_minimap")] int TilesWithNoLiquidMinimap,
	[property: JsonPropertyName("tiles_with_declared_liquid_layers")] int TilesWithDeclaredLiquidLayers,
	[property: JsonPropertyName("visible_surface_liquid_tiles")] int VisibleSurfaceLiquidTiles,
	[property: JsonPropertyName("below_terrain_likely_liquid_tiles")] int BelowTerrainLikelyLiquidTiles,
	[property: JsonPropertyName("uncertain_liquid_tiles")] int UncertainLiquidTiles,
	[property: JsonPropertyName("no_liquid_tiles")] int NoLiquidTiles,
	[property: JsonPropertyName("concept_cluster_count")] int ConceptClusterCount,
	[property: JsonPropertyName("dedupe_group_count")] int DedupeGroupCount,
	[property: JsonPropertyName("duplicate_tile_count")] int DuplicateTileCount,
	[property: JsonPropertyName("retained_canonical_tile_count")] int RetainedCanonicalTileCount,
	[property: JsonPropertyName("review_duplicate_tile_count")] int ReviewDuplicateTileCount);

file sealed record MlSignalAuditGroupSummary(
	[property: JsonPropertyName("group_id")] string GroupId,
	[property: JsonPropertyName("signature")] string Signature,
	[property: JsonPropertyName("tile_count")] int TileCount,
	[property: JsonPropertyName("representative_tile")] string RepresentativeTile,
	[property: JsonPropertyName("tiles")] List<string> Tiles);

file sealed record MlSignalAuditTileReport(
	[property: JsonPropertyName("tile_name")] string TileName,
	[property: JsonPropertyName("map_name")] string MapName,
	[property: JsonPropertyName("tile_json_path")] string TileJsonPath,
	[property: JsonPropertyName("concept_cluster_id")] string ConceptClusterId,
	[property: JsonPropertyName("dedupe_group_id")] string DedupeGroupId,
	[property: JsonPropertyName("retention_recommendation")] string RetentionRecommendation,
	[property: JsonPropertyName("source_minimap_exists")] bool SourceMinimapExists,
	[property: JsonPropertyName("heightmap_local_exists")] bool HeightmapLocalExists,
	[property: JsonPropertyName("heightmap_global_exists")] bool HeightmapGlobalExists,
	[property: JsonPropertyName("alpha_atlas_exists")] bool AlphaAtlasExists,
	[property: JsonPropertyName("alpha_mask_count")] int AlphaMaskCount,
	[property: JsonPropertyName("texture_count")] int TextureCount,
	[property: JsonPropertyName("texture_signature")] string TextureSignature,
	[property: JsonPropertyName("object_count")] int ObjectCount,
	[property: JsonPropertyName("chunk_layer_count")] int ChunkLayerCount,
	[property: JsonPropertyName("liquid_layer_count")] int LiquidLayerCount,
	[property: JsonPropertyName("liquid_mask_exists")] bool LiquidMaskExists,
	[property: JsonPropertyName("no_liquid_minimap_exists")] bool NoLiquidMinimapExists,
	[property: JsonPropertyName("liquid_semantic_class")] string LiquidSemanticClass,
	[property: JsonPropertyName("liquid_semantic_reason")] string LiquidSemanticReason,
	[property: JsonPropertyName("liquid_mask_pixel_count")] int LiquidMaskPixelCount,
	[property: JsonPropertyName("liquid_mean_rgb_delta")] double? LiquidMeanRgbDelta,
	[property: JsonPropertyName("dedupe_signature")] string DedupeSignature,
	[property: JsonPropertyName("concept_signature")] string ConceptSignature);

file sealed record MlSignalAuditLiquidAssessment(string LiquidSemanticClass, string Reason, int MaskedPixelCount, double? MeanRgbDelta);

file sealed record MlSignalAuditMaskDiff(int MaskedPixelCount, double MeanRgbDelta);

file sealed record MlSignalAuditImageSignature(string Sha256, string AverageHash64);
