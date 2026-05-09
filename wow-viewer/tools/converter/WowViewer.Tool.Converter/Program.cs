using System.Security.Cryptography;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Numerics;
using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WowViewer.App;
using WowViewer.Core.Datasets;
using WowViewer.Core.Files;
using WowViewer.Core.IO;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4;
using WowViewer.Core.Runtime.World.Liquid;
using WowViewer.Core.Runtime.World.Terrain;
using WowViewer.Core.Wmo;
using WowViewer.Core.Runtime.World.Wdl;
using WowViewer.Tools.Shared;
using WowViewer.Tool.Converter;

const int NativeTileSize = 257;
const int NativeMinimapSize = 256;
const float HeightGlobalMin = -1000f;
const float HeightGlobalMax = 3000f;
const float WorldTileSize = 533.33333f;
const float WorldMapOrigin = 32f * WorldTileSize;
const string V9TensorCacheManifestFile = "v9_tensor_cache_manifest.json";
const byte DefaultNormalR = 128;
const byte DefaultNormalG = 128;
const byte DefaultNormalB = 255;

if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
{
	ShowUsage();
	return;
}

string command = args[0].ToLowerInvariant();
string[] tail = args.Skip(1).ToArray();

	switch (command)
	{
		case "dataset-list-maps":
			RunDatasetListMaps(tail);
			break;
		case "dataset-scan":
			RunDatasetScan(tail);
			break;
		case "dataset-merge":
			RunDatasetMerge(tail);
			break;
		case "dataset-split-pm4":
			RunDatasetSplitPm4(tail);
			break;
		case "dataset-audit":
			RunDatasetAudit(tail);
			break;
		case "dataset-curate":
			RunDatasetCurate(tail);
			break;
		case "dataset-build-cache":
			RunDatasetBuildCache(tail);
			break;
		case "extract-map":
			RunExtractMap(tail);
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
	case "terrain-patch-adt":
		TerrainPatchAdtCommand.Run(tail);
		break;
	case "export-tex-json":
		RunExportTexJson(tail);
		break;
	case "extract-v10-tensors":
		RunExtractV10Tensors(tail);
		break;
		case "dataset-build-v10-stage1":
			RunDatasetBuildV10Stage1(tail);
			break;
	case "mine-v10-brushes":
		V10BrushMiningCommand.Run(tail);
		break;
	case "mine-v10-mcly":
		V10MclyDictionaryCommand.Run(tail);
		break;
	case "label-v10-mcly":
		V10MclyLabelManifestCommand.Run(tail);
		break;
	case "mine-v10-mcal-compositions":
		V10McalCompositionCommand.Run(tail);
		break;
	case "mine-v10-mcal-brushes":
		V10McalBrushDictionaryCommand.Run(tail);
		break;
	case "mine-v10-height-profiles":
		V10HeightProfileCommand.Run(tail);
		break;
	case "mine-v10-prefab-cells":
		V10PrefabCellCommand.Run(tail);
		break;
	case "convert-alpha-to-lk":
		AlphaToLkCommand.Run(tail);
		break;
	case "convert-lk-to-alpha":
		LkToAlphaCommand.Run(tail);
		break;
	case "validate-roundtrip":
		ValidateRoundTripCommand.Run(tail);
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

static void RunExtractV10Tensors(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	string? minimapRoot = GetOption(args, "--minimap-root", "-m");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: --input <root.adt> is required.");
		Environment.ExitCode = 1;
		return;
	}

	string? textureSource = GetOption(args, "--texture-source", "-t");
	if (string.IsNullOrWhiteSpace(textureSource))
	{
		// Auto-resolve _tex0.adt sibling
		string basePath = Path.Combine(Path.GetDirectoryName(input)!, Path.GetFileNameWithoutExtension(input));
		textureSource = basePath + "_tex0.adt";
		if (!File.Exists(textureSource))
			textureSource = null;
	}

	string outputPath = !string.IsNullOrWhiteSpace(output)
		? output
		: Path.Combine(Path.GetDirectoryName(input)!, Path.GetFileNameWithoutExtension(input) + "_v10.npz");
	string placementOutputPath = Path.Combine(
		Path.GetDirectoryName(outputPath) ?? Path.GetDirectoryName(input)!,
		Path.GetFileNameWithoutExtension(outputPath) + "_placements.json");

	try
	{
		V10TensorExtractionResult result = ExtractAndWriteV10TensorPack(input, textureSource, minimapRoot, outputPath, placementOutputPath, requireMinimap: false);
		if (!string.IsNullOrWhiteSpace(result.MinimapSourcePath))
			Console.WriteLine($"  Minimap source: {result.MinimapSourcePath}");
		Console.WriteLine($"Extracted v10 tensors: {outputPath}");
		Console.WriteLine($"  Placement sidecar: {placementOutputPath}");
		Console.WriteLine($"  Signals: {string.Join(", ", result.Pack.AvailableSignals)}");
	}
	catch (Exception ex)
	{
		Console.Error.WriteLine($"Error extracting v10 tensors from {input}: {ex.Message}");
		Environment.ExitCode = 1;
	}
}

static void RunDatasetBuildV10Stage1(string[] args)
{
	string? inputDirOption = GetOption(args, "--input-dir", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? outputDirOption = GetOption(args, "--output-dir", "-o");
	string? minimapRootOption = GetOption(args, "--minimap-root", "-m");
	string? manifestOption = GetOption(args, "--manifest", "-f");
	int limit = GetIntOption(args, "--limit", "-n") ?? int.MaxValue;
	bool overwrite = HasFlag(args, "--overwrite");

	if (string.IsNullOrWhiteSpace(inputDirOption) || string.IsNullOrWhiteSpace(outputDirOption) || string.IsNullOrWhiteSpace(minimapRootOption))
	{
		Console.Error.WriteLine("Error: dataset-build-v10-stage1 requires --input-dir <adt-dir>, --output-dir <dir>, and --minimap-root <dir>.");
		Environment.ExitCode = 1;
		return;
	}

	string inputDir = Path.GetFullPath(inputDirOption);
	string outputDir = Path.GetFullPath(outputDirOption);
	string minimapRoot = Path.GetFullPath(minimapRootOption);
	string manifestPath = Path.GetFullPath(string.IsNullOrWhiteSpace(manifestOption)
		? Path.Combine(outputDir, "v10_stage1_manifest.json")
		: manifestOption);

	if (!Directory.Exists(inputDir))
	{
		Console.Error.WriteLine($"Error: input directory not found: {inputDir}");
		Environment.ExitCode = 1;
		return;
	}

	if (!Directory.Exists(minimapRoot))
	{
		Console.Error.WriteLine($"Error: minimap root not found: {minimapRoot}");
		Environment.ExitCode = 1;
		return;
	}

	Directory.CreateDirectory(outputDir);
	string? manifestDirectory = Path.GetDirectoryName(manifestPath);
	if (!string.IsNullOrWhiteSpace(manifestDirectory))
		Directory.CreateDirectory(manifestDirectory);

	List<V10Stage1ManifestEntry> entries = [];
	List<V10Stage1ManifestSkip> skipped = [];
	int scanned = 0;

	foreach (string adtPath in EnumerateRootAdtFiles(inputDir))
	{
		if (entries.Count >= limit)
			break;

		scanned++;
		string tileStem = Path.GetFileNameWithoutExtension(adtPath);
		string outputPath = Path.Combine(outputDir, tileStem + "_v10.npz");
		string placementOutputPath = Path.Combine(outputDir, tileStem + "_v10_placements.json");

		if (!overwrite && File.Exists(outputPath))
		{
			skipped.Add(new V10Stage1ManifestSkip(tileStem, adtPath, "output_exists"));
			continue;
		}

		string basePath = Path.Combine(Path.GetDirectoryName(adtPath)!, tileStem);
		string textureSource = basePath + "_tex0.adt";
		string? resolvedTextureSource = File.Exists(textureSource) ? textureSource : null;

		try
		{
			V10TensorExtractionResult result = ExtractAndWriteV10TensorPack(adtPath, resolvedTextureSource, minimapRoot, outputPath, placementOutputPath, requireMinimap: true);
			entries.Add(new V10Stage1ManifestEntry(
				tileStem,
				adtPath,
				outputPath,
				File.Exists(placementOutputPath) ? placementOutputPath : null,
				result.MinimapSourcePath ?? string.Empty,
				result.Pack.MinimapSourceTag,
				result.Pack.AvailableSignals.OrderBy(static signal => signal, StringComparer.OrdinalIgnoreCase).ToArray()));
			Console.WriteLine($"Stage1 shard: {tileStem} -> {outputPath}");
		}
		catch (Exception ex)
		{
			skipped.Add(new V10Stage1ManifestSkip(tileStem, adtPath, ex.Message));
			Console.Error.WriteLine($"Skipping {tileStem}: {ex.Message}");
		}
	}

	// ── Second pass: PM4-only placeholder tiles ───────────────────────────
	HashSet<string> coveredTileNames = entries
		.Select(static entry => entry.TileName.ToLowerInvariant())
		.ToHashSet(StringComparer.OrdinalIgnoreCase);

	int placeholderWritten = 0;
	int placeholderSkipped = 0;
	const string buildKey = "4.0.0.11927";

	foreach ((int tileX, int tileY) in DiscoverPm4TileCoords(inputDir))
	{
		if (entries.Count + placeholderWritten >= limit)
			break;

		string tileName = $"development_{tileX}_{tileY}";
		if (coveredTileNames.Contains(tileName))
			continue;

		string outputPath = Path.Combine(outputDir, $"{tileName}_v10.npz");
		if (!overwrite && File.Exists(outputPath))
		{
			placeholderSkipped++;
			continue;
		}

		byte[,,]? minimapRgb = null;
		string? minimapSourcePath = null;
		if (TryLoadMinimapForTile(minimapRoot, "development", tileX, tileY, out byte[,,]? loadedMinimap, out string? loadedPath))
		{
			minimapRgb = loadedMinimap;
			minimapSourcePath = loadedPath;
		}

		TerrainTileTensorPack pack = AdtTensorPackBuilder.BuildPlaceholder(
			inputDir, "development", tileX, tileY, minimapRgb, buildKey);

		NpzTileSerializer.Serialize(pack, outputPath);

		entries.Add(new V10Stage1ManifestEntry(
			tileName,
			string.Empty, // no ADT source
			outputPath,
			null, // no placement sidecar
			minimapSourcePath ?? string.Empty,
			pack.MinimapSourceTag,
			pack.AvailableSignals.OrderBy(static signal => signal, StringComparer.OrdinalIgnoreCase).ToArray()));

		placeholderWritten++;
		Console.WriteLine($"Stage1 placeholder: {tileName} -> {outputPath}");
	}

	V10Stage1Manifest manifest = new(
		SchemaVersion: "v10-stage1-manifest.v1",
		CreatedAtUtc: DateTimeOffset.UtcNow,
		InputRoot: inputDir,
		OutputRoot: outputDir,
		MinimapRoot: minimapRoot,
		ScannedTileCount: scanned,
		WrittenTileCount: entries.Count,
		PlaceholderTileCount: placeholderWritten,
		Entries: entries,
		Skipped: skipped);

	File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, CreateJsonOptions()));

	Console.WriteLine("WowViewer.Tool.Converter dataset-build-v10-stage1 report");
	Console.WriteLine($"InputDir: {inputDir}");
	Console.WriteLine($"OutputDir: {outputDir}");
	Console.WriteLine($"Manifest: {manifestPath}");
	Console.WriteLine($"Scanned: {scanned}");
	Console.WriteLine($"Written: {entries.Count}");
	Console.WriteLine($"  ADT-backed: {entries.Count - placeholderWritten}");
	Console.WriteLine($"  Placeholder: {placeholderWritten}");
	Console.WriteLine($"Skipped: {skipped.Count + placeholderSkipped}");
}

static V10TensorExtractionResult ExtractAndWriteV10TensorPack(
	string inputAdtPath,
	string? textureSource,
	string? minimapRoot,
	string outputPath,
	string placementOutputPath,
	bool requireMinimap)
{
	TerrainTileTensorPack pack = AdtTensorPackBuilder.Build(inputAdtPath, textureSource);
	string? minimapSourcePath = null;
	if (!string.IsNullOrWhiteSpace(minimapRoot) && TryLoadV10MinimapRgb(inputAdtPath, minimapRoot, out byte[,,]? minimapRgb, out string? resolvedMinimapPath))
	{
		pack.MinimapRgb256 = minimapRgb;
		pack.MinimapSourceTag = "raw";
		HashSet<string> availableSignals = new(pack.AvailableSignals, StringComparer.OrdinalIgnoreCase)
		{
			"minimap_rgb_256"
		};
		pack.AvailableSignals = availableSignals;
		minimapSourcePath = resolvedMinimapPath;
	}

	if (requireMinimap && pack.MinimapRgb256 is null)
		throw new InvalidOperationException("missing minimap_rgb_256 for this tile under the provided minimap root");

	NpzTileSerializer.Serialize(pack, outputPath);
	WriteV10PlacementSidecar(inputAdtPath, placementOutputPath);
	return new V10TensorExtractionResult(pack, minimapSourcePath);
}

static IEnumerable<string> EnumerateRootAdtFiles(string inputDir)
{
	return Directory.EnumerateFiles(inputDir, "*.adt", SearchOption.TopDirectoryOnly)
		.Where(static path => !path.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
			&& !path.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase)
			&& !path.EndsWith("_lod.adt", StringComparison.OrdinalIgnoreCase))
		.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase);
}

static IEnumerable<(int TileX, int TileY)> DiscoverPm4TileCoords(string mapDirectory)
{
	HashSet<(int, int)> seen = [];
	string[] pm4Files;
	try
	{
		pm4Files = Directory.GetFiles(mapDirectory, "*.pm4", SearchOption.TopDirectoryOnly);
	}
	catch
	{
		yield break;
	}

	foreach (string pm4Path in pm4Files)
	{
		string fileName = Path.GetFileNameWithoutExtension(pm4Path);
		if (!TryParseTileCoordinates(fileName, out int tileX, out int tileY))
			continue;

		if (seen.Add((tileX, tileY)))
			yield return (tileX, tileY);
	}
}

static bool TryLoadMinimapForTile(string minimapRoot, string mapName, int tileX, int tileY, out byte[,,]? minimapRgb, out string? sourcePath)
{
	minimapRgb = null;
	sourcePath = null;

	string tileStem = $"{mapName}_{tileX}_{tileY}";

	foreach (string directCandidate in EnumerateLooseMinimapCandidates(tileStem))
	{
		string directPath = Path.Combine(minimapRoot, directCandidate);
		if (!File.Exists(directPath))
			continue;

		byte[]? directRgb = DecodeFilesystemMinimap(directPath);
		if (directRgb is not { Length: > 0 })
			continue;

		minimapRgb = ReshapeRgb256(directRgb);
		sourcePath = directPath;
		return true;
	}

	foreach (string candidate in EnumerateMinimapCandidates(mapName, tileX, tileY))
	{
		string? resolvedPath = ResolveFilesystemMinimapPath(minimapRoot, candidate);
		if (resolvedPath is null)
			continue;

		byte[]? rgb = DecodeFilesystemMinimap(resolvedPath);
		if (rgb is not { Length: > 0 })
			continue;

		minimapRgb = ReshapeRgb256(rgb);
		sourcePath = resolvedPath;
		return true;
	}

	return false;
}

static bool TryLoadV10MinimapRgb(string inputAdtPath, string minimapRoot, out byte[,,]? minimapRgb, out string? sourcePath)
{
	minimapRgb = null;
	sourcePath = null;

	string tileStem = Path.GetFileNameWithoutExtension(inputAdtPath);
	if (!TryParseTileCoordinates(tileStem, out int tileX, out int tileY))
		return false;

	string mapName = ExtractMapNameFromTileStem(tileStem);
	if (string.IsNullOrWhiteSpace(mapName))
		return false;

	foreach (string directCandidate in EnumerateLooseMinimapCandidates(tileStem))
	{
		string directPath = Path.Combine(minimapRoot, directCandidate);
		if (!File.Exists(directPath))
			continue;

		byte[]? directRgb = DecodeFilesystemMinimap(directPath);
		if (directRgb is not { Length: > 0 })
			continue;

		minimapRgb = ReshapeRgb256(directRgb);
		sourcePath = directPath;
		return true;
	}

	foreach (string candidate in EnumerateMinimapCandidates(mapName, tileX, tileY))
	{
		string? resolvedPath = ResolveFilesystemMinimapPath(minimapRoot, candidate);
		if (resolvedPath is null)
			continue;

		byte[]? rgb = DecodeFilesystemMinimap(resolvedPath);
		if (rgb is not { Length: > 0 })
			continue;

		minimapRgb = ReshapeRgb256(rgb);
		sourcePath = resolvedPath;
		return true;
	}

	return false;
}

static IEnumerable<string> EnumerateLooseMinimapCandidates(string tileStem)
{
	yield return $"{tileStem}.png";
	yield return Path.Combine("images", $"{tileStem}.png");
	yield return Path.Combine("reference_minimaps", $"{tileStem}_reference_minimap.png");
}

static string ExtractMapNameFromTileStem(string tileStem)
{
	int lastUnderscore = tileStem.LastIndexOf('_');
	if (lastUnderscore <= 0)
		return tileStem;

	int secondLastUnderscore = tileStem.LastIndexOf('_', lastUnderscore - 1);
	if (secondLastUnderscore <= 0)
		return tileStem;

	return tileStem[..secondLastUnderscore];
}

static byte[,,] ReshapeRgb256(byte[] rgb)
{
	if (rgb.Length != NativeMinimapSize * NativeMinimapSize * 3)
		throw new InvalidDataException($"Expected {NativeMinimapSize}x{NativeMinimapSize} RGB minimap bytes but found {rgb.Length} bytes.");

	byte[,,] result = new byte[NativeMinimapSize, NativeMinimapSize, 3];
	int index = 0;
	for (int y = 0; y < NativeMinimapSize; y++)
	{
		for (int x = 0; x < NativeMinimapSize; x++)
		{
			result[y, x, 0] = rgb[index++];
			result[y, x, 1] = rgb[index++];
			result[y, x, 2] = rgb[index++];
		}
	}

	return result;
}

static void WriteV10PlacementSidecar(string inputAdtPath, string outputPath)
{
	AdtTileFamily family = AdtTileFamilyResolver.Resolve(inputAdtPath);
	string? placementSourcePath = family.PlacementSourcePath;
	if (string.IsNullOrWhiteSpace(placementSourcePath) || !File.Exists(placementSourcePath))
		return;

	AdtPlacementCatalog placements = AdtPlacementReader.Read(placementSourcePath);
	string? directory = Path.GetDirectoryName(outputPath);
	if (!string.IsNullOrWhiteSpace(directory))
		Directory.CreateDirectory(directory);

	var payload = new
	{
		source_adt_path = Path.GetFullPath(inputAdtPath),
		placement_source_path = Path.GetFullPath(placementSourcePath),
		mddf = placements.ModelPlacements.Select(static placement => new
		{
			model_path = placement.ModelPath,
			unique_id = placement.UniqueId,
			position = new { x = placement.Position.X, y = placement.Position.Y, z = placement.Position.Z },
			rotation = new { x = placement.Rotation.X, y = placement.Rotation.Y, z = placement.Rotation.Z },
			scale = placement.Scale,
		}).ToArray(),
		modf = placements.WorldModelPlacements.Select(static placement => new
		{
			model_path = placement.ModelPath,
			unique_id = placement.UniqueId,
			position = new { x = placement.Position.X, y = placement.Position.Y, z = placement.Position.Z },
			rotation = new { x = placement.Rotation.X, y = placement.Rotation.Y, z = placement.Rotation.Z },
			bounds_min = new { x = placement.BoundsMin.X, y = placement.BoundsMin.Y, z = placement.BoundsMin.Z },
			bounds_max = new { x = placement.BoundsMax.X, y = placement.BoundsMax.Y, z = placement.BoundsMax.Z },
			flags = placement.Flags,
		}).ToArray(),
	};

	File.WriteAllText(outputPath, JsonSerializer.Serialize(payload, CreateJsonOptions()));
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
	List<TerrainTrainingSampleDescriptor> entries = [];
	bool hasLooseFilesystemMap = FilesystemMapExists(mapPath, normalizedMapName);
	if (hasLooseFilesystemMap)
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

static void RunDatasetListMaps(string[] args)
{
	string? clientRootOption = GetOption(args, "--client-root", "-c");
	string? outputOption = GetOption(args, "--output", "-o");

	if (string.IsNullOrWhiteSpace(clientRootOption))
	{
		Console.Error.WriteLine("Error: dataset-list-maps requires --client-root <path>.");
		Environment.ExitCode = 1;
		return;
	}

	string clientRoot = Path.GetFullPath(clientRootOption);
	List<MapDirectoryEntry> maps = DiscoverDatasetMaps(clientRoot);

	Console.WriteLine("WowViewer.Tool.Converter dataset-list-maps report");
	Console.WriteLine($"ClientRoot: {clientRoot}");
	Console.WriteLine($"Maps: {maps.Count}");

	string json = JsonSerializer.Serialize(maps, CreateJsonOptions());
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

	foreach (MapDirectoryEntry map in maps)
		Console.WriteLine(map.Directory);
}

static void RunDatasetMerge(string[] args)
{
	List<string> inputPaths = [];
	string? outputOption = null;

	for (int index = 0; index < args.Length; index++)
	{
		string arg = args[index];
		if (string.Equals(arg, "--input", StringComparison.OrdinalIgnoreCase)
			|| string.Equals(arg, "-i", StringComparison.OrdinalIgnoreCase))
		{
			if (index + 1 >= args.Length)
			{
				Console.Error.WriteLine("Error: dataset-merge requires a path after --input.");
				Environment.ExitCode = 1;
				return;
			}

			inputPaths.Add(Path.GetFullPath(args[++index]));
			continue;
		}

		if (string.Equals(arg, "--output", StringComparison.OrdinalIgnoreCase)
			|| string.Equals(arg, "-o", StringComparison.OrdinalIgnoreCase))
		{
			if (index + 1 >= args.Length)
			{
				Console.Error.WriteLine("Error: dataset-merge requires a path after --output.");
				Environment.ExitCode = 1;
				return;
			}

			outputOption = Path.GetFullPath(args[++index]);
			continue;
		}

		if (!arg.StartsWith('-'))
			inputPaths.Add(Path.GetFullPath(arg));
	}

	if (inputPaths.Count == 0)
	{
		Console.Error.WriteLine("Error: dataset-merge requires one or more input manifests via --input <path> or positional paths.");
		Environment.ExitCode = 1;
		return;
	}

	List<TerrainTrainingSampleManifest> manifests = inputPaths.Select(ReadTerrainTrainingManifest).ToList();
	TerrainTrainingSampleManifest firstManifest = manifests[0];
	foreach (TerrainTrainingSampleManifest manifest in manifests.Skip(1))
	{
		if (!string.Equals(manifest.SourceManifestKind, firstManifest.SourceManifestKind, StringComparison.OrdinalIgnoreCase))
		{
			Console.Error.WriteLine($"Error: dataset-merge requires all manifests to have the same SourceManifestKind. Found '{firstManifest.SourceManifestKind}' and '{manifest.SourceManifestKind}'.");
			Environment.ExitCode = 1;
			return;
		}

		if (!string.Equals(manifest.SchemaVersion, firstManifest.SchemaVersion, StringComparison.OrdinalIgnoreCase))
		{
			Console.Error.WriteLine($"Error: dataset-merge requires all manifests to have the same SchemaVersion. Found '{firstManifest.SchemaVersion}' and '{manifest.SchemaVersion}'.");
			Environment.ExitCode = 1;
			return;
		}
	}

	Dictionary<string, TerrainTrainingSampleDescriptor> mergedBySampleId = new(StringComparer.Ordinal);
	foreach (TerrainTrainingSampleManifest manifest in manifests)
	{
		foreach (TerrainTrainingSampleDescriptor entry in manifest.Entries)
		{
			if (!mergedBySampleId.TryAdd(entry.SampleId, entry))
			{
				Console.Error.WriteLine($"Error: dataset-merge found duplicate SampleId '{entry.SampleId}'. Inputs must be non-overlapping.");
				Environment.ExitCode = 1;
				return;
			}
		}
	}

	List<TerrainTrainingSampleDescriptor> mergedEntries = mergedBySampleId.Values
		.OrderBy(static entry => entry.BuildLabel, StringComparer.OrdinalIgnoreCase)
		.ThenBy(static entry => entry.MapName, StringComparer.OrdinalIgnoreCase)
		.ThenBy(static entry => entry.TileY)
		.ThenBy(static entry => entry.TileX)
		.ThenBy(static entry => entry.SampleId, StringComparer.Ordinal)
		.ToList();

	TerrainTrainingSampleManifest mergedManifest = new(
		schemaVersion: firstManifest.SchemaVersion,
		createdAtUtc: DateTimeOffset.UtcNow,
		sourceManifestKind: firstManifest.SourceManifestKind,
		entries: mergedEntries);

	Console.WriteLine("WowViewer.Tool.Converter dataset-merge report");
	Console.WriteLine($"SourceManifestKind: {mergedManifest.SourceManifestKind}");
	Console.WriteLine($"SchemaVersion: {mergedManifest.SchemaVersion}");
	Console.WriteLine($"Inputs: {inputPaths.Count}");
	Console.WriteLine($"MergedSamples: {mergedEntries.Count}");

	string json = JsonSerializer.Serialize(mergedManifest, CreateJsonOptions());
	if (!string.IsNullOrWhiteSpace(outputOption))
	{
		string? outputDirectory = Path.GetDirectoryName(outputOption);
		if (!string.IsNullOrWhiteSpace(outputDirectory))
			Directory.CreateDirectory(outputDirectory);

		File.WriteAllText(outputOption, json);
		Console.WriteLine($"Wrote {outputOption}");
		return;
	}

	Console.WriteLine(json);
}

static void RunDatasetSplitPm4(string[] args)
{
	string? directManifestOption = GetOption(args, "--direct-manifest", "-d");
	string? developmentManifestOption = GetOption(args, "--development-manifest", "-i");
	string? outputDirOption = GetOption(args, "--output-dir", "-o");
	string pm4Flag = GetOption(args, "--pm4-flag", "-p") ?? "has_pm4_mask_257";

	if (string.IsNullOrWhiteSpace(directManifestOption)
		|| string.IsNullOrWhiteSpace(developmentManifestOption)
		|| string.IsNullOrWhiteSpace(outputDirOption))
	{
		Console.Error.WriteLine("Error: dataset-split-pm4 requires --direct-manifest <cache.json>, --development-manifest <cache.json>, and --output-dir <dir>.");
		Environment.ExitCode = 1;
		return;
	}

	string directManifestPath = Path.GetFullPath(directManifestOption);
	string developmentManifestPath = Path.GetFullPath(developmentManifestOption);
	string outputDir = Path.GetFullPath(outputDirOption);

	if (!File.Exists(directManifestPath))
	{
		Console.Error.WriteLine($"Error: direct manifest not found: {directManifestPath}");
		Environment.ExitCode = 1;
		return;
	}

	if (!File.Exists(developmentManifestPath))
	{
		Console.Error.WriteLine($"Error: development manifest not found: {developmentManifestPath}");
		Environment.ExitCode = 1;
		return;
	}

	V9TensorCacheManifestData directManifest = ReadV9TensorCacheManifest(directManifestPath);
	V9TensorCacheManifestData developmentManifest = ReadV9TensorCacheManifest(developmentManifestPath);

	if (!string.Equals(directManifest.SchemaVersion, developmentManifest.SchemaVersion, StringComparison.OrdinalIgnoreCase))
	{
		Console.Error.WriteLine($"Error: dataset-split-pm4 requires matching schema versions. Found '{directManifest.SchemaVersion}' and '{developmentManifest.SchemaVersion}'.");
		Environment.ExitCode = 1;
		return;
	}

	List<JsonElement> pm4Entries = [];
	List<JsonElement> nonPm4Entries = [];
	foreach (JsonElement entry in developmentManifest.Entries)
	{
		if (TryGetBooleanProperty(entry, pm4Flag, out bool hasPm4) && hasPm4)
			pm4Entries.Add(entry);
		else
			nonPm4Entries.Add(entry);
	}

	List<JsonElement> mergedEntries = new(directManifest.Entries.Count + pm4Entries.Count);
	HashSet<string> seenKeys = new(StringComparer.Ordinal);
	foreach (JsonElement entry in directManifest.Entries)
	{
		string entryKey = BuildV9ManifestEntryKey(entry);
		if (!seenKeys.Add(entryKey))
		{
			Console.Error.WriteLine($"Error: duplicate direct manifest entry key encountered: {entryKey}");
			Environment.ExitCode = 1;
			return;
		}

		mergedEntries.Add(entry);
	}

	foreach (JsonElement entry in pm4Entries)
	{
		string entryKey = BuildV9ManifestEntryKey(entry);
		if (!seenKeys.Add(entryKey))
		{
			Console.Error.WriteLine($"Error: dataset-split-pm4 found duplicate merged entry key: {entryKey}");
			Environment.ExitCode = 1;
			return;
		}

		mergedEntries.Add(entry);
	}

	Directory.CreateDirectory(outputDir);

	string pm4SubsetPath = Path.Combine(outputDir, "v9_development_pm4_training_manifest.json");
	string holdoutPath = Path.Combine(outputDir, "v9_development_non_pm4_holdout_manifest.json");
	string mergedPath = Path.Combine(outputDir, "v9_direct_plus_development_pm4_training_manifest.json");

	object pm4SubsetManifest = new
	{
		schema_version = developmentManifest.SchemaVersion,
		created_at_utc = DateTimeOffset.UtcNow,
		source_cache_manifest = developmentManifestPath.Replace('\\', '/'),
		split_name = "development_pm4_training_subset",
		processed = pm4Entries.Count,
		skipped = 0,
		entries = pm4Entries,
	};

	object holdoutManifest = new
	{
		schema_version = developmentManifest.SchemaVersion,
		created_at_utc = DateTimeOffset.UtcNow,
		source_cache_manifest = developmentManifestPath.Replace('\\', '/'),
		split_name = "development_non_pm4_holdout",
		processed = nonPm4Entries.Count,
		skipped = 0,
		entries = nonPm4Entries,
	};

	object mergedManifest = new
	{
		schema_version = directManifest.SchemaVersion,
		created_at_utc = DateTimeOffset.UtcNow,
		source_cache_manifests = new[]
		{
			directManifestPath.Replace('\\', '/'),
			developmentManifestPath.Replace('\\', '/'),
		},
		split_name = "direct_archive_plus_development_pm4_training",
		processed = mergedEntries.Count,
		skipped = 0,
		entries = mergedEntries,
	};

	File.WriteAllText(pm4SubsetPath, JsonSerializer.Serialize(pm4SubsetManifest, CreateJsonOptions()));
	File.WriteAllText(holdoutPath, JsonSerializer.Serialize(holdoutManifest, CreateJsonOptions()));
	File.WriteAllText(mergedPath, JsonSerializer.Serialize(mergedManifest, CreateJsonOptions()));

	Console.WriteLine("WowViewer.Tool.Converter dataset-split-pm4 report");
	Console.WriteLine($"DirectManifest: {directManifestPath}");
	Console.WriteLine($"DevelopmentManifest: {developmentManifestPath}");
	Console.WriteLine($"Pm4Flag: {pm4Flag}");
	Console.WriteLine($"Pm4TrainingEntries: {pm4Entries.Count}");
	Console.WriteLine($"NonPm4HoldoutEntries: {nonPm4Entries.Count}");
	Console.WriteLine($"MergedTrainingEntries: {mergedEntries.Count}");
	Console.WriteLine($"Wrote {pm4SubsetPath}");
	Console.WriteLine($"Wrote {holdoutPath}");
	Console.WriteLine($"Wrote {mergedPath}");
}

static void RunDatasetAudit(string[] args)
{
	string? inputOption = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? outputOption = GetOption(args, "--output", "-o");
	int? limit = GetIntOption(args, "--limit", "-n");

	if (string.IsNullOrWhiteSpace(inputOption))
	{
		Console.Error.WriteLine("Error: dataset-audit requires --input <manifest.json>.");
		Environment.ExitCode = 1;
		return;
	}

	string inputPath = Path.GetFullPath(inputOption);
	TerrainTrainingSampleManifest sourceManifest = ReadTerrainTrainingManifest(inputPath);
	if (!string.Equals(sourceManifest.SourceManifestKind, "scan", StringComparison.OrdinalIgnoreCase))
	{
		Console.Error.WriteLine($"Error: dataset-audit requires a direct dataset-scan manifest, but '{inputPath}' has SourceManifestKind='{sourceManifest.SourceManifestKind}'.");
		Environment.ExitCode = 1;
		return;
	}

	List<TerrainTrainingSampleDescriptor> sourceEntries = sourceManifest.Entries.ToList();
	if (limit is > 0)
		sourceEntries = sourceEntries.Take(limit.Value).ToList();

	Dictionary<string, IArchiveCatalog> archiveCatalogs = new(StringComparer.OrdinalIgnoreCase);
	Dictionary<string, WdlSummary?> wdlCache = new(StringComparer.OrdinalIgnoreCase);
	Dictionary<string, Md5TranslateIndex?> minimapMd5Cache = new(StringComparer.OrdinalIgnoreCase);
	List<TerrainTrainingSampleDescriptor> auditedEntries = new(sourceEntries.Count);
	int liquidSampleCount = 0;
	int holeSampleCount = 0;
	int wdlDeltaSampleCount = 0;

	try
	{
		foreach (TerrainTrainingSampleDescriptor entry in sourceEntries)
		{
			TerrainTrainingSampleDescriptor auditedEntry = AuditDatasetEntry(entry, archiveCatalogs, wdlCache, minimapMd5Cache);
			auditedEntries.Add(auditedEntry);
			if (auditedEntry.Metrics.LiquidCoverage > 0f)
				liquidSampleCount++;
			if (auditedEntry.Metrics.HoleCoverage > 0f)
				holeSampleCount++;
			if (auditedEntry.Metrics.MaxAbsWdlDelta > 0f)
				wdlDeltaSampleCount++;
		}
	}
	finally
	{
		foreach (IArchiveCatalog archiveCatalog in archiveCatalogs.Values)
			archiveCatalog.Dispose();
	}

	TerrainTrainingSampleManifest auditedManifest = new(
		schemaVersion: "terrain-training-audit.v1",
		createdAtUtc: DateTimeOffset.UtcNow,
		sourceManifestKind: "audit",
		entries: auditedEntries);

	Console.WriteLine("WowViewer.Tool.Converter dataset-audit report");
	Console.WriteLine($"Input: {inputPath}");
	Console.WriteLine($"Samples: {auditedEntries.Count}");
	Console.WriteLine($"LiquidSamples: {liquidSampleCount}");
	Console.WriteLine($"HoleSamples: {holeSampleCount}");
	Console.WriteLine($"WdlDeltaSamples: {wdlDeltaSampleCount}");

	string json = JsonSerializer.Serialize(auditedManifest, CreateJsonOptions());
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

static void RunDatasetCurate(string[] args)
{
	string? inputOption = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? outputOption = GetOption(args, "--output", "-o");
	string? reportOption = GetOption(args, "--report", "-r");
	int? limit = GetIntOption(args, "--limit", "-n");
	int maxPerGroup = GetIntOption(args, "--max-per-group", "-g") ?? int.MaxValue;
	float minHeightRange = GetFloatOption(args, "--min-height-range", "-h") ?? 32f;
	float minMinimapVariance = GetFloatOption(args, "--min-minimap-variance", "-v") ?? 1e-5f;
	float minMinimapGradient = GetFloatOption(args, "--min-minimap-gradient", "-m") ?? 2e-3f;
	float maxMeanWdlDelta = GetFloatOption(args, "--max-mean-wdl-delta", "-w") ?? 256f;
	float maxAbsWdlDelta = GetFloatOption(args, "--max-abs-wdl-delta", "-a") ?? 1024f;
	bool requireWdl = ResolveBooleanOption(args, "--require-wdl", "--no-require-wdl", defaultValue: true);
	bool requireMinimap = ResolveBooleanOption(args, "--require-minimap", "--no-require-minimap", defaultValue: false);

	if (string.IsNullOrWhiteSpace(inputOption) || string.IsNullOrWhiteSpace(outputOption))
	{
		Console.Error.WriteLine("Error: dataset-curate requires --input <audit.json> and --output <curated.json>.");
		Environment.ExitCode = 1;
		return;
	}

	string inputPath = Path.GetFullPath(inputOption);
	TerrainTrainingSampleManifest sourceManifest = ReadTerrainTrainingManifest(inputPath);
	if (!string.Equals(sourceManifest.SourceManifestKind, "audit", StringComparison.OrdinalIgnoreCase))
	{
		Console.Error.WriteLine($"Error: dataset-curate requires a dataset-audit manifest, but '{inputPath}' has SourceManifestKind='{sourceManifest.SourceManifestKind}'.");
		Environment.ExitCode = 1;
		return;
	}

	List<DatasetCurateEvaluation> evaluations = sourceManifest.Entries
		.Select(entry => EvaluateCurateEntry(
			entry,
			requireWdl,
			requireMinimap,
			minHeightRange,
			minMinimapVariance,
			minMinimapGradient,
			maxMeanWdlDelta,
			maxAbsWdlDelta))
		.ToList();

	List<DatasetCurateEvaluation> acceptedPool = evaluations
		.Where(static evaluation => evaluation.Accepted)
		.ToList();
	List<TerrainTrainingSampleDescriptor> curatedEntries = SelectCuratedEntries(acceptedPool, limit, maxPerGroup);

	TerrainTrainingSampleManifest curatedManifest = new(
		schemaVersion: "terrain-training-curate.v1",
		createdAtUtc: DateTimeOffset.UtcNow,
		sourceManifestKind: "curate",
		entries: curatedEntries);

	Console.WriteLine("WowViewer.Tool.Converter dataset-curate report");
	Console.WriteLine($"Input: {inputPath}");
	Console.WriteLine($"AcceptedPool: {acceptedPool.Count}");
	Console.WriteLine($"Curated: {curatedEntries.Count}");
	Console.WriteLine($"Rejected: {evaluations.Count - acceptedPool.Count}");

	string outputPath = Path.GetFullPath(outputOption);
	string? outputDirectory = Path.GetDirectoryName(outputPath);
	if (!string.IsNullOrWhiteSpace(outputDirectory))
		Directory.CreateDirectory(outputDirectory);

	File.WriteAllText(outputPath, JsonSerializer.Serialize(curatedManifest, CreateJsonOptions()));
	Console.WriteLine($"Wrote {outputPath}");

	if (!string.IsNullOrWhiteSpace(reportOption))
	{
		string reportPath = Path.GetFullPath(reportOption);
		string? reportDirectory = Path.GetDirectoryName(reportPath);
		if (!string.IsNullOrWhiteSpace(reportDirectory))
			Directory.CreateDirectory(reportDirectory);

		object reportPayload = new
		{
			schema_version = "terrain-training-curate-report.v1",
			created_at_utc = DateTimeOffset.UtcNow,
			source_manifest = inputPath,
			accepted = curatedEntries.Count,
			accepted_pool = acceptedPool.Count,
			rejected = evaluations.Count - acceptedPool.Count,
			config = new
			{
				require_wdl = requireWdl,
				require_minimap = requireMinimap,
				min_height_range = minHeightRange,
				min_minimap_variance = minMinimapVariance,
				min_minimap_gradient = minMinimapGradient,
				max_mean_wdl_delta = maxMeanWdlDelta,
				max_abs_wdl_delta = maxAbsWdlDelta,
				limit,
				max_per_group = maxPerGroup,
			},
			items = evaluations.Select(evaluation => new
			{
				evaluation.Entry.SampleId,
				evaluation.Entry.BuildLabel,
				evaluation.Entry.MapName,
				evaluation.Entry.TileName,
				evaluation.Accepted,
				evaluation.RejectionReason,
				evaluation.QualityScore,
				evaluation.Entry.Metrics.HeightRange,
				evaluation.Entry.Metrics.LiquidCoverage,
				evaluation.Entry.Metrics.HoleCoverage,
				evaluation.Entry.Metrics.MinimapVariance,
				evaluation.Entry.Metrics.MinimapGradient,
				evaluation.Entry.Metrics.MeanWdlDelta,
				evaluation.Entry.Metrics.MaxAbsWdlDelta,
				evaluation.Entry.Signals.HasWdl,
				evaluation.Entry.Signals.HasMinimap,
			}),
		};

		File.WriteAllText(reportPath, JsonSerializer.Serialize(reportPayload, CreateJsonOptions()));
		Console.WriteLine($"Wrote {reportPath}");
	}
}

static void RunDatasetBuildCache(string[] args)
{
	string? inputOption = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? outputDirOption = GetOption(args, "--output-dir", "-o");
	int? limit = GetIntOption(args, "--limit", "-n");
	bool overwrite = HasFlag(args, "--overwrite");
	bool includeMinimap = ResolveBooleanOption(args, "--include-minimap", "--no-include-minimap", defaultValue: true);
	bool writeDebugJson = ResolveBooleanOption(args, "--write-debug-json", "--no-write-debug-json", defaultValue: true);

	if (string.IsNullOrWhiteSpace(inputOption) || string.IsNullOrWhiteSpace(outputDirOption))
	{
		Console.Error.WriteLine("Error: dataset-build-cache requires --input <audit-or-curate.json> and --output-dir <dir>.");
		Environment.ExitCode = 1;
		return;
	}

	string inputPath = Path.GetFullPath(inputOption);
	string outputDir = Path.GetFullPath(outputDirOption);
	TerrainTrainingSampleManifest sourceManifest = ReadTerrainTrainingManifest(inputPath);
	if (!string.Equals(sourceManifest.SourceManifestKind, "audit", StringComparison.OrdinalIgnoreCase)
		&& !string.Equals(sourceManifest.SourceManifestKind, "curate", StringComparison.OrdinalIgnoreCase))
	{
		Console.Error.WriteLine($"Error: dataset-build-cache requires a dataset-audit or dataset-curate manifest, but '{inputPath}' has SourceManifestKind='{sourceManifest.SourceManifestKind}'.");
		Environment.ExitCode = 1;
		return;
	}

	Directory.CreateDirectory(outputDir);
	Dictionary<string, IArchiveCatalog> archiveCatalogs = new(StringComparer.OrdinalIgnoreCase);
	Dictionary<string, WdlSummary?> wdlCache = new(StringComparer.OrdinalIgnoreCase);
	Dictionary<string, Md5TranslateIndex?> minimapMd5Cache = new(StringComparer.OrdinalIgnoreCase);
	List<Dictionary<string, object?>> manifestEntries = [];
	int processed = 0;
	int skipped = 0;

	try
	{
		IEnumerable<TerrainTrainingSampleDescriptor> sourceEntries = sourceManifest.Entries;
		if (limit is > 0)
			sourceEntries = sourceEntries.Take(limit.Value);

		foreach (TerrainTrainingSampleDescriptor entry in sourceEntries)
		{
			DirectCacheBuildResult? built = BuildDirectCacheEntry(
				entry,
				archiveCatalogs,
				wdlCache,
				minimapMd5Cache,
				outputDir,
				inputPath,
				includeMinimap,
				writeDebugJson,
				overwrite);
			if (built is null)
			{
				skipped++;
				continue;
			}

			manifestEntries.Add(new Dictionary<string, object?>
			{
				["dataset_root"] = entry.SourceRoot,
				["dataset_key"] = built.DatasetKey,
				["tile_name"] = entry.TileName,
				["shard_path"] = built.ShardPath,
				["source_json"] = built.DebugJsonPath,
				["height_min"] = built.HeightMin,
				["height_max"] = built.HeightMax,
				["has_wdl_17"] = built.HasWdl17,
				["has_minimap_rgb_256"] = built.HasMinimap,
				["has_normal_rgb_256"] = built.HasNativeNormalMap,
				["liquid_coverage"] = built.LiquidCoverage,
				["object_coverage"] = built.ObjectCoverage,
				["brush_coverage"] = built.BrushCoverage,
				["hole_coverage"] = built.HoleCoverage,
				["minimap_variance"] = built.MinimapVariance,
				["minimap_gradient"] = built.MinimapGradient,
				["detail_energy"] = built.DetailEnergy,
				["array_names"] = built.ArrayNames,
				["minimap_source"] = built.MinimapSource,
			});
			processed++;
		}
	}
	finally
	{
		foreach (IArchiveCatalog archiveCatalog in archiveCatalogs.Values)
			archiveCatalog.Dispose();
	}

	object manifestPayload = new
	{
		schema_version = "v9-native-tensor-cache.v2",
		created_at_utc = DateTimeOffset.UtcNow,
		output_dir = outputDir,
		source_manifest = inputPath,
		source_manifest_kind = sourceManifest.SourceManifestKind,
		processed,
		skipped,
		supported_native_sizes = new[] { 257, 129, 65, 33, 17 },
		entries = manifestEntries,
	};

	string manifestPath = Path.Combine(outputDir, V9TensorCacheManifestFile);
	File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifestPayload, CreateJsonOptions()));

	Console.WriteLine("WowViewer.Tool.Converter dataset-build-cache report");
	Console.WriteLine($"Input: {inputPath}");
	Console.WriteLine($"Processed: {processed}");
	Console.WriteLine($"Skipped: {skipped}");
	Console.WriteLine($"Wrote {manifestPath}");
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
			mapDirectory: mapName,
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
	string mapDirectory = GetMapDirectoryFromMapVirtualRoot(mapVirtualRoot, mapName);
	Dictionary<string, IArchiveCatalog> archiveCatalogs = new(StringComparer.OrdinalIgnoreCase)
	{
		[clientRoot] = archiveCatalog,
	};
	Dictionary<string, WdlSummary?> wdlCache = new(StringComparer.OrdinalIgnoreCase);
	Dictionary<string, Md5TranslateIndex?> minimapMd5Cache = new(StringComparer.OrdinalIgnoreCase);
	string wdtVirtualPath = BuildMapWdtVirtualPath(mapVirtualRoot, mapName);
	byte[] wdtBytes = archiveCatalog.ReadFile(wdtVirtualPath)
		?? throw new FileNotFoundException($"Could not read archive-backed WDT '{wdtVirtualPath}'.", wdtVirtualPath);

	List<WdtTileCoordinate> allTileCoordinates = ReadArchiveWdtTiles(wdtBytes, wdtVirtualPath)
		.OrderBy(static tile => tile.TileY)
		.ThenBy(static tile => tile.TileX)
		.ToList();

	List<WdtTileCoordinate> tileCoordinates = allTileCoordinates;
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
		string objVirtualPath = $"{mapVirtualRoot}\\{tileStem}_obj0.adt";
		string texVirtualPath = $"{mapVirtualRoot}\\{tileStem}_tex0.adt";
		string lodVirtualPath = $"{mapVirtualRoot}\\{tileStem}_lod.adt";
		string rootPathForEntry = rootVirtualPath;
		string? objPathForEntry = archiveCatalog.FileExists(objVirtualPath) ? objVirtualPath : null;
		string? texPathForEntry = archiveCatalog.FileExists(texVirtualPath) ? texVirtualPath : null;
		string? lodPathForEntry = archiveCatalog.FileExists(lodVirtualPath) ? lodVirtualPath : null;
		AdtSummary summary;
		AlphaEmbeddedAdtTileData? alphaTile = null;

		if (rootBytes.Length > 0)
		{
			summary = ReadArchiveAdtSummary(rootBytes, rootVirtualPath);
		}
		else if (AlphaEmbeddedAdtReader.TryReadTile(clientRoot, mapDirectory, tileCoordinate.TileX, tileCoordinate.TileY, archiveCatalog, out alphaTile))
		{
			summary = BuildAlphaEmbeddedAdtSummary(alphaTile!);
			rootPathForEntry = alphaTile!.SourcePath;
			objPathForEntry = null;
			texPathForEntry = null;
			lodPathForEntry = null;
		}
		else
		{
			continue;
		}

		TerrainTrainingSampleDescriptor entry = CreateDatasetScanEntry(
			sampleId: $"{buildLabel}:{tileStem}",
			sourceKind: TerrainTrainingSampleSourceKind.MountedArchive,
			buildLabel: buildLabel,
			mapName: mapName,
			mapDirectory: mapDirectory,
			tileX: tileCoordinate.TileX,
			tileY: tileCoordinate.TileY,
			sourceRoot: clientRoot,
			rootAdtPath: rootPathForEntry,
			objAdtPath: objPathForEntry,
			texAdtPath: texPathForEntry,
			lodAdtPath: lodPathForEntry,
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
			hasAlphaLayers: texPathForEntry is not null,
			hasTextureMetadata: texPathForEntry is not null || summary.TextureNameCount > 0);

		if (alphaTile is not null)
			entry = AuditDatasetEntry(entry, archiveCatalogs, wdlCache, minimapMd5Cache);

		entries.Add(entry);
	}

	return entries;
}

static void RunExtractMap(string[] args)
{
	string? clientRootOption = GetOption(args, "--client-root", "-c") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? mapName = GetOption(args, "--map", "-m");
	string? outputDirOption = GetOption(args, "--output-dir", "-o");
	int? limit = GetIntOption(args, "--limit", "-n");

	if (string.IsNullOrWhiteSpace(clientRootOption) || string.IsNullOrWhiteSpace(mapName) || string.IsNullOrWhiteSpace(outputDirOption))
	{
		Console.Error.WriteLine("Error: extract-map requires --client-root <path>, --map <name>, and --output-dir <dir>.");
		Environment.ExitCode = 1;
		return;
	}

	string clientRoot = Path.GetFullPath(clientRootOption);
	string normalizedMapName = mapName.Trim();
	string outputDir = Path.GetFullPath(outputDirOption);
	Directory.CreateDirectory(outputDir);

	using IArchiveCatalog archiveCatalog = CreateArchiveCatalog(clientRoot);
	string mapPath = Path.Combine(clientRoot, "Data", "World", "Maps", normalizedMapName);
	if (!ArchiveMapExists(archiveCatalog, clientRoot, normalizedMapName, mapPath))
	{
		Console.Error.WriteLine($"Error: map '{normalizedMapName}' not found in client '{clientRoot}'.");
		Environment.ExitCode = 1;
		return;
	}

	string mapVirtualRoot = ResolveMapVirtualRoot(clientRoot, normalizedMapName, mapPath);
	string wdtVirtualPath = BuildMapWdtVirtualPath(mapVirtualRoot, normalizedMapName);
	byte[] wdtBytes = archiveCatalog.ReadFile(wdtVirtualPath)
		?? throw new FileNotFoundException($"Could not read WDT '{wdtVirtualPath}'.", wdtVirtualPath);

	List<WdtTileCoordinate> tileCoordinates = ReadArchiveWdtTiles(wdtBytes, wdtVirtualPath)
		.OrderBy(static tile => tile.TileY)
		.ThenBy(static tile => tile.TileX)
		.ToList();

	if (limit is > 0)
		tileCoordinates = tileCoordinates.Take(limit.Value).ToList();

	int extracted = 0;
	int skipped = 0;

	foreach (WdtTileCoordinate tileCoordinate in tileCoordinates)
	{
		string tileStem = $"{normalizedMapName}_{tileCoordinate.TileX}_{tileCoordinate.TileY}";
		string rootVirtualPath = $"{mapVirtualRoot}\\{tileStem}.adt";
		string objVirtualPath = $"{mapVirtualRoot}\\{tileStem}_obj0.adt";
		string texVirtualPath = $"{mapVirtualRoot}\\{tileStem}_tex0.adt";

		byte[] rootBytes = archiveCatalog.ReadFile(rootVirtualPath) ?? [];
		if (rootBytes.Length == 0)
		{
			skipped++;
			continue;
		}

		string rootOutputPath = Path.Combine(outputDir, $"{tileStem}.adt");
		File.WriteAllBytes(rootOutputPath, rootBytes);
		extracted++;

		byte[] objBytes = archiveCatalog.ReadFile(objVirtualPath) ?? [];
		if (objBytes.Length > 0)
		{
			File.WriteAllBytes(Path.Combine(outputDir, $"{tileStem}_obj0.adt"), objBytes);
		}

		byte[] texBytes = archiveCatalog.ReadFile(texVirtualPath) ?? [];
		if (texBytes.Length > 0)
		{
			File.WriteAllBytes(Path.Combine(outputDir, $"{tileStem}_tex0.adt"), texBytes);
		}

		Console.WriteLine($"Extracted: {tileStem}");
	}

	Console.WriteLine("WowViewer.Tool.Converter extract-map report");
	Console.WriteLine($"ClientRoot: {clientRoot}");
	Console.WriteLine($"Map: {normalizedMapName}");
	Console.WriteLine($"OutputDir: {outputDir}");
	Console.WriteLine($"Extracted: {extracted}");
	Console.WriteLine($"Skipped: {skipped}");
}

static TerrainTrainingSampleDescriptor CreateDatasetScanEntry(
	string sampleId,
	TerrainTrainingSampleSourceKind sourceKind,
	string buildLabel,
	string mapName,
	string? mapDirectory,
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
		MapDirectory = string.IsNullOrWhiteSpace(mapDirectory) ? null : mapDirectory,
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

static TerrainTrainingSampleDescriptor AuditDatasetEntry(
	TerrainTrainingSampleDescriptor entry,
	Dictionary<string, IArchiveCatalog> archiveCatalogs,
	Dictionary<string, WdlSummary?> wdlCache,
	Dictionary<string, Md5TranslateIndex?> minimapMd5Cache)
{
	(bool useArchive, WorldTerrainTileData terrain, AdtLiquidFile liquid, AdtMcnkSummary mcnkSummary) = ReadDatasetAuditSource(entry, archiveCatalogs);
	WdlSummary? wdlSummary = TryGetCachedWdlSummary(entry, useArchive, archiveCatalogs, wdlCache);
	(int companionTileX, int companionTileY) = ResolveCompanionTileCoordinates(entry);
	(bool hasWdlTile, float meanWdlDelta, float maxAbsWdlDelta) = ComputeWdlDeltaMetrics(terrain, wdlSummary, companionTileX, companionTileY);
	(float liquidCoverage, bool hasLiquidHeights) = ComputeLiquidMetrics(liquid, mcnkSummary);
	byte[]? minimapRgb256 = TryLoadMinimapRgb(entry, archiveCatalogs, minimapMd5Cache, out _);
	float minimapVariance = minimapRgb256 is not null ? ComputeRgbVariance(minimapRgb256) : 0f;
	float minimapGradient = minimapRgb256 is not null ? ComputeAverageGradientMagnitude(minimapRgb256, NativeMinimapSize, NativeMinimapSize) : 0f;
	float holeCoverage = ComputeHoleCoverage(terrain);
	bool hasLiquidSignal = liquidCoverage > 0f || mcnkSummary.ChunksWithMclq > 0;
	WorldTerrainHeightmapData? heightmap = terrain.Heightmap;

	return new TerrainTrainingSampleDescriptor(
		sampleId: entry.SampleId,
		sourceKind: entry.SourceKind,
		buildLabel: entry.BuildLabel,
		mapName: entry.MapName,
		tileX: entry.TileX,
		tileY: entry.TileY,
		sourceRoot: entry.SourceRoot,
		rootAdtPath: entry.RootAdtPath)
	{
		ObjAdtPath = entry.ObjAdtPath,
		TexAdtPath = entry.TexAdtPath,
		LodAdtPath = entry.LodAdtPath,
		WdlPath = entry.WdlPath,
		MapDirectory = entry.MapDirectory,
		LooseOverlayRoot = entry.LooseOverlayRoot,
		CompatibilityTileJsonPath = entry.CompatibilityTileJsonPath,
		Signals = new TerrainTrainingSignalAvailability
		{
			HasRootAdt = true,
			HasObjAdt = entry.ObjAdtPath is not null,
			HasTexAdt = entry.TexAdtPath is not null,
			HasWdl = hasWdlTile,
			HasMinimap = minimapRgb256 is not null,
			HasTerrainOnlyMinimap = entry.Signals.HasTerrainOnlyMinimap,
			HasNoLiquidMinimap = entry.Signals.HasNoLiquidMinimap,
			HasNoObjectMinimap = entry.Signals.HasNoObjectMinimap,
			HasNoMccvMinimap = entry.Signals.HasNoMccvMinimap,
			HasNormalMap = entry.Signals.HasNormalMap,
			HasLiquidMask = hasLiquidSignal,
			HasLiquidHeight = hasLiquidHeights || mcnkSummary.ChunksWithMclq > 0,
			HasObjectMask = entry.Signals.HasObjectMask,
			HasBrushMask = entry.Signals.HasBrushMask,
			HasPm4Mask = entry.Signals.HasPm4Mask,
			HasHoleMask = holeCoverage > 0f,
			HasAreaIdMap = terrain.DistinctAreaIdCount > 0,
			HasChunkFlagsMap = terrain.ChunkCount > 0,
			HasAlphaLayers = entry.Signals.HasAlphaLayers,
			HasTextureMetadata = entry.Signals.HasTextureMetadata,
		},
		Metrics = new TerrainTrainingSampleMetrics
		{
			HeightMin = heightmap?.MinHeight ?? 0f,
			HeightMax = heightmap?.MaxHeight ?? 0f,
			HeightRange = heightmap is not null ? heightmap.MaxHeight - heightmap.MinHeight : 0f,
			LiquidCoverage = liquidCoverage,
			ObjectCoverage = entry.Metrics.ObjectCoverage,
			BrushCoverage = entry.Metrics.BrushCoverage,
			Pm4Coverage = entry.Metrics.Pm4Coverage,
			HoleCoverage = holeCoverage,
			MinimapVariance = minimapVariance,
			MinimapGradient = minimapGradient,
			MeanWdlDelta = meanWdlDelta,
			MaxAbsWdlDelta = maxAbsWdlDelta,
			TextureLayerCount = Math.Max(entry.Metrics.TextureLayerCount, mcnkSummary.MaxLayerCount),
		},
	};
}

static (bool UseArchive, WorldTerrainTileData Terrain, AdtLiquidFile Liquid, AdtMcnkSummary McnkSummary) ReadDatasetAuditSource(
	TerrainTrainingSampleDescriptor entry,
	Dictionary<string, IArchiveCatalog> archiveCatalogs)
{
	bool useArchive = entry.SourceKind == TerrainTrainingSampleSourceKind.MountedArchive;
	if (useArchive)
	{
		IArchiveCatalog archiveCatalog = GetOrCreateArchiveCatalog(entry.SourceRoot, archiveCatalogs);
		if (TryReadAlphaEmbeddedTile(entry, archiveCatalog, out AlphaEmbeddedAdtTileData? alphaTile))
		{
			return (
				UseArchive: true,
				Terrain: alphaTile!.TerrainTileData,
				Liquid: BuildAlphaEmbeddedLiquidFile(alphaTile),
				McnkSummary: BuildAlphaEmbeddedMcnkSummary(alphaTile));
		}

		byte[] rootBytes = archiveCatalog.ReadFile(entry.RootAdtPath)
			?? throw new FileNotFoundException($"Could not read archive-backed root ADT '{entry.RootAdtPath}' for '{entry.SampleId}'.", entry.RootAdtPath);
		using MemoryStream stream = new(rootBytes, writable: false);
		MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, entry.RootAdtPath);
		return (
			UseArchive: true,
			Terrain: WorldTerrainTileBuilder.Read(stream, fileSummary),
			Liquid: AdtLiquidReader.Read(stream, fileSummary),
			McnkSummary: AdtMcnkSummaryReader.Read(stream, fileSummary));
	}

	using FileStream fileStream = File.OpenRead(entry.RootAdtPath);
	MapFileSummary filesystemSummary = MapFileSummaryReader.Read(fileStream, Path.GetFullPath(entry.RootAdtPath));
	return (
		UseArchive: false,
		Terrain: WorldTerrainTileBuilder.Read(fileStream, filesystemSummary),
		Liquid: AdtLiquidReader.Read(fileStream, filesystemSummary),
		McnkSummary: AdtMcnkSummaryReader.Read(fileStream, filesystemSummary));
}

static TerrainTrainingSampleManifest ReadTerrainTrainingManifest(string inputPath)
{
	string json = File.ReadAllText(inputPath);
	TerrainTrainingSampleManifest? manifest = JsonSerializer.Deserialize<TerrainTrainingSampleManifest>(json, CreateJsonOptions());
	if (manifest is null)
		throw new InvalidDataException($"Could not deserialize terrain-training manifest '{inputPath}'.");

	return manifest;
}

static IArchiveCatalog GetOrCreateArchiveCatalog(string clientRoot, Dictionary<string, IArchiveCatalog> archiveCatalogs)
{
	if (archiveCatalogs.TryGetValue(clientRoot, out IArchiveCatalog? existing))
		return existing;

	IArchiveCatalog archiveCatalog = CreateArchiveCatalog(clientRoot);
	archiveCatalogs[clientRoot] = archiveCatalog;
	return archiveCatalog;
}

static WdlSummary? TryGetCachedWdlSummary(
	TerrainTrainingSampleDescriptor entry,
	bool useArchive,
	Dictionary<string, IArchiveCatalog> archiveCatalogs,
	Dictionary<string, WdlSummary?> wdlCache)
{
	if (string.IsNullOrWhiteSpace(entry.WdlPath))
		return null;

	string cacheKey = $"{entry.SourceKind}|{entry.SourceRoot}|{entry.WdlPath}";
	if (wdlCache.TryGetValue(cacheKey, out WdlSummary? cached))
		return cached;

	WdlSummary? resolved = null;
	if (useArchive)
	{
		IArchiveCatalog archiveCatalog = GetOrCreateArchiveCatalog(entry.SourceRoot, archiveCatalogs);
		byte[]? wdlBytes = archiveCatalog.ReadFile(entry.WdlPath);
		if (wdlBytes is { Length: > 0 })
		{
			using MemoryStream stream = new(wdlBytes, writable: false);
			resolved = WdlSummaryReader.Read(stream, entry.WdlPath);
		}
	}
	else if (File.Exists(entry.WdlPath))
	{
		resolved = WdlSummaryReader.Read(entry.WdlPath);
	}

	wdlCache[cacheKey] = resolved;
	return resolved;
}

static (bool HasDelta, float MeanWdlDelta, float MaxAbsWdlDelta) ComputeWdlDeltaMetrics(
	WorldTerrainTileData terrain,
	WdlSummary? summary,
	int tileX,
	int tileY)
{
	WorldTerrainHeightmapData? heightmap = terrain.Heightmap;
	if (heightmap is null || summary is null || !summary.TryGetTile(tileX, tileY, out WdlTileSummary? tile) || tile is null)
		return (false, 0f, 0f);

	float[] height17 = DownsampleHeightGrid(heightmap.Heights.ToArray(), NativeTileSize, 17);
	WdlAlignment? alignment = TryBuildAlignedWdl17(height17, summary, tileX, tileY);
	if (alignment is null)
		return (false, 0f, 0f);

	return (true, alignment.MeanAbsoluteDelta, alignment.MaxAbsoluteDelta);
}

static (float LiquidCoverage, bool HasLiquidHeights) ComputeLiquidMetrics(AdtLiquidFile liquidFile, AdtMcnkSummary mcnkSummary)
{
	float mh2oCoverage = ComputeMh2oCoverage(liquidFile);
	float mclqFallbackCoverage = mcnkSummary.McnkCount > 0
		? (float)mcnkSummary.ChunksWithMclq / mcnkSummary.McnkCount
		: 0f;
	bool hasLiquidHeights = liquidFile.Chunks.Any(static chunk => chunk.Layers.Any(static layer => layer.Heights is { Length: > 0 }))
		|| mcnkSummary.ChunksWithMclq > 0;
	return (Math.Clamp(MathF.Max(mh2oCoverage, mclqFallbackCoverage), 0f, 1f), hasLiquidHeights);
}

static float ComputeMh2oCoverage(AdtLiquidFile liquidFile)
{
	if (liquidFile.Chunks.Count == 0)
		return 0f;

	int visibleTileCount = 0;
	foreach (AdtLiquidChunk chunk in liquidFile.Chunks)
	{
		bool[] occupied = new bool[64];
		foreach (AdtLiquidLayer layer in chunk.Layers)
		{
			for (int localY = 0; localY < layer.Height; localY++)
			{
				for (int localX = 0; localX < layer.Width; localX++)
				{
					if (!layer.TileExists(localX, localY))
						continue;

					int chunkX = layer.XOffset + localX;
					int chunkY = layer.YOffset + localY;
					if ((uint)chunkX >= 8 || (uint)chunkY >= 8)
						continue;

					occupied[(chunkY * 8) + chunkX] = true;
				}
			}
		}

		for (int index = 0; index < occupied.Length; index++)
		{
			if (occupied[index])
				visibleTileCount++;
		}
	}

	return (float)visibleTileCount / (liquidFile.Chunks.Count * 64);
}

static float ComputeHoleCoverage(WorldTerrainTileData terrain)
{
	if (terrain.ChunkCount == 0)
		return 0f;

	int totalHoleBits = 0;
	foreach (WorldTerrainChunkData chunk in terrain.Chunks)
		totalHoleBits += BitOperations.PopCount((uint)chunk.HoleMask);

	return (float)totalHoleBits / (terrain.ChunkCount * 16);
}

static DatasetCurateEvaluation EvaluateCurateEntry(
	TerrainTrainingSampleDescriptor entry,
	bool requireWdl,
	bool requireMinimap,
	float minHeightRange,
	float minMinimapVariance,
	float minMinimapGradient,
	float maxMeanWdlDelta,
	float maxAbsWdlDelta)
{
	string? rejectionReason = null;
	if (entry.Metrics.HeightRange < minHeightRange)
		rejectionReason = "height_range";
	else if (requireWdl && !entry.Signals.HasWdl)
		rejectionReason = "missing_wdl";
	else if (requireMinimap && !entry.Signals.HasMinimap)
		rejectionReason = "missing_minimap";
	else if (entry.Signals.HasMinimap && entry.Metrics.MinimapVariance < minMinimapVariance)
		rejectionReason = "low_minimap_variance";
	else if (entry.Signals.HasMinimap && entry.Metrics.MinimapGradient < minMinimapGradient)
		rejectionReason = "low_minimap_gradient";
	else if (entry.Signals.HasWdl && entry.Metrics.MeanWdlDelta > maxMeanWdlDelta)
		rejectionReason = "high_mean_wdl_delta";
	else if (entry.Signals.HasWdl && entry.Metrics.MaxAbsWdlDelta > maxAbsWdlDelta)
		rejectionReason = "high_abs_wdl_delta";

	float qualityScore = -1f;
	if (rejectionReason is null)
	{
		float heightScore = MathF.Min(entry.Metrics.HeightRange / 64f, 4f);
		float minimapScore = entry.Signals.HasMinimap
			? MathF.Min(entry.Metrics.MinimapGradient / 0.02f, 3f) + MathF.Min(entry.Metrics.MinimapVariance / 0.01f, 3f)
			: 0f;
		float wdlPenalty = entry.Signals.HasWdl
			? MathF.Min(entry.Metrics.MeanWdlDelta / 128f, 2f) + MathF.Min(entry.Metrics.MaxAbsWdlDelta / 512f, 2f)
			: 0f;
		float holePenalty = MathF.Min(entry.Metrics.HoleCoverage * 2f, 1f);
		qualityScore = heightScore + minimapScore - wdlPenalty - holePenalty;
	}

	return new DatasetCurateEvaluation(entry, rejectionReason is null, rejectionReason, qualityScore);
}

static List<TerrainTrainingSampleDescriptor> SelectCuratedEntries(
	IReadOnlyList<DatasetCurateEvaluation> acceptedPool,
	int? limit,
	int maxPerGroup)
{
	Dictionary<string, Queue<DatasetCurateEvaluation>> grouped = acceptedPool
		.GroupBy(static evaluation => $"{evaluation.Entry.BuildLabel}::{evaluation.Entry.MapName}", StringComparer.OrdinalIgnoreCase)
		.ToDictionary(
			static group => group.Key,
			static group => new Queue<DatasetCurateEvaluation>(group
				.OrderByDescending(static evaluation => evaluation.QualityScore)
				.ThenBy(static evaluation => evaluation.Entry.TileY)
				.ThenBy(static evaluation => evaluation.Entry.TileX)),
			StringComparer.OrdinalIgnoreCase);

	Dictionary<string, int> acceptedPerGroup = new(StringComparer.OrdinalIgnoreCase);
	List<TerrainTrainingSampleDescriptor> selected = [];
	List<string> groupOrder = grouped
		.OrderByDescending(static pair => pair.Value.Count > 0 ? pair.Value.Peek().QualityScore : float.MinValue)
		.Select(static pair => pair.Key)
		.ToList();

	while (groupOrder.Count > 0 && (!limit.HasValue || selected.Count < limit.Value))
	{
		bool anyAdded = false;
		for (int index = 0; index < groupOrder.Count && (!limit.HasValue || selected.Count < limit.Value); index++)
		{
			string groupKey = groupOrder[index];
			Queue<DatasetCurateEvaluation> queue = grouped[groupKey];
			int currentCount = acceptedPerGroup.TryGetValue(groupKey, out int existingCount) ? existingCount : 0;
			if (currentCount >= maxPerGroup)
				continue;
			if (queue.Count == 0)
				continue;

			selected.Add(queue.Dequeue().Entry);
			acceptedPerGroup[groupKey] = currentCount + 1;
			anyAdded = true;
		}

		groupOrder = groupOrder
			.Where(groupKey => grouped[groupKey].Count > 0 && (acceptedPerGroup.TryGetValue(groupKey, out int count) ? count : 0) < maxPerGroup)
			.ToList();

		if (!anyAdded)
			break;
	}

	return selected;
}

static DirectCacheBuildResult? BuildDirectCacheEntry(
	TerrainTrainingSampleDescriptor entry,
	Dictionary<string, IArchiveCatalog> archiveCatalogs,
	Dictionary<string, WdlSummary?> wdlCache,
	Dictionary<string, Md5TranslateIndex?> minimapMd5Cache,
	string outputDir,
	string sourceManifestPath,
	bool includeMinimap,
	bool writeDebugJson,
	bool overwrite)
{
	(bool useArchive, WorldTerrainTileData terrain, AdtLiquidFile liquid, AdtMcnkSummary mcnkSummary) = ReadDatasetAuditSource(entry, archiveCatalogs);
	WorldTerrainHeightmapData? heightmap = terrain.Heightmap;
	if (heightmap is null)
		return null;

	string datasetKey = BuildDirectDatasetKey(entry);
	string shardDirectory = Path.Combine(outputDir, "shards", datasetKey);
	Directory.CreateDirectory(shardDirectory);
	string shardPath = Path.Combine(shardDirectory, $"{entry.TileName}.npz");
	if (File.Exists(shardPath) && !overwrite)
	{
		string debugExistingPath = Path.Combine(outputDir, "debug", datasetKey, $"{entry.TileName}.json");
		return new DirectCacheBuildResult(
			DatasetKey: datasetKey,
			ShardPath: shardPath,
			DebugJsonPath: File.Exists(debugExistingPath) ? debugExistingPath : sourceManifestPath,
			HeightMin: heightmap.MinHeight,
			HeightMax: heightmap.MaxHeight,
			LiquidCoverage: entry.Metrics.LiquidCoverage,
			ObjectCoverage: entry.Metrics.ObjectCoverage,
			BrushCoverage: entry.Metrics.BrushCoverage,
			HoleCoverage: entry.Metrics.HoleCoverage,
			MinimapVariance: entry.Metrics.MinimapVariance,
			MinimapGradient: entry.Metrics.MinimapGradient,
			DetailEnergy: 0f,
			HasWdl17: false,
			HasMinimap: entry.Signals.HasMinimap,
			HasNativeNormalMap: false,
			MinimapSource: "existing-shard",
			ArrayNames: Array.Empty<string>());
	}

	float[] chunkHeights = BuildChunkHeightsTensor(terrain);
	float[] height257 = heightmap.Heights.ToArray();
	float[] height129 = DownsampleHeightGrid(height257, NativeTileSize, 129);
	float[] height65 = DownsampleHeightGrid(height257, NativeTileSize, 65);
	float[] height33 = DownsampleHeightGrid(height257, NativeTileSize, 33);
	float[] height17 = DownsampleHeightGrid(height257, NativeTileSize, 17);
	byte[] holeMask16 = BuildHoleMask16x16(terrain);
	byte[] normalRgb256 = CreateSolidRgbImage(NativeMinimapSize, NativeMinimapSize, DefaultNormalR, DefaultNormalG, DefaultNormalB);
	float[] heightHints = BuildHeightHints(heightmap.MinHeight, heightmap.MaxHeight);
	(byte[] liquidMask257, float[] liquidHeight257, bool hasNativeLiquidHeights, string liquidMode) = RasterizeLiquidSignals(terrain, liquid, mcnkSummary, height257);
	WdlSummary? wdlSummary = TryGetCachedWdlSummary(entry, useArchive, archiveCatalogs, wdlCache);
	(int companionTileX, int companionTileY) = ResolveCompanionTileCoordinates(entry);
	WdlAlignment? wdlAlignment = TryBuildAlignedWdl17(height17, wdlSummary, companionTileX, companionTileY);
	bool includeVerifiedWdl = IsVerifiedWdlAlignment(wdlAlignment);
	float[]? wdl17 = includeVerifiedWdl ? wdlAlignment!.AlignedHeights17 : null;
	string? minimapSourceName = null;
	byte[]? minimapRgb256 = includeMinimap ? TryLoadMinimapRgb(entry, archiveCatalogs, minimapMd5Cache, out minimapSourceName) : null;
	string minimapSource = includeMinimap && minimapRgb256 is not null ? minimapSourceName! : "missing";
	byte[] objectMask257 = BuildObjectMask257(entry, archiveCatalogs);
	byte[] brushMask257 = new byte[NativeTileSize * NativeTileSize];
	float liquidCoverage = ComputeBinaryCoverage(liquidMask257);
	float objectCoverage = ComputeBinaryCoverage(objectMask257);
	float brushCoverage = ComputeBinaryCoverage(brushMask257);
	float holeCoverage = ComputeBinaryCoverage(holeMask16);
	float minimapVariance = minimapRgb256 is not null ? ComputeRgbVariance(minimapRgb256) : 0f;
	float minimapGradient = minimapRgb256 is not null ? ComputeAverageGradientMagnitude(minimapRgb256, NativeMinimapSize, NativeMinimapSize) : 0f;
	float detailEnergy = ComputeDetailEnergy(height257, height65);

	// MCAL/MCLY: extract per-chunk texture layers and assemble alpha compositing
	(float[] mcalAlphaPack256, byte[] mclyLayerMask, int[] mclyTextureIds) = BuildMcalMclySignals(terrain);
	bool hasMcly = mclyTextureIds.Any(static id => id >= 0);

	List<(string Name, NpyArray Array)> payload =
	[
		("chunk_heights_256x145", NpyArray.FromFloat32(chunkHeights, 256, 145)),
		("height_257", NpyArray.FromFloat32(height257, NativeTileSize, NativeTileSize)),
		("height_129", NpyArray.FromFloat32(height129, 129, 129)),
		("height_65", NpyArray.FromFloat32(height65, 65, 65)),
		("height_33", NpyArray.FromFloat32(height33, 33, 33)),
		("height_17", NpyArray.FromFloat32(height17, 17, 17)),
		("hole_mask_16x16", NpyArray.FromUInt8(holeMask16, 16, 16)),
		("normal_rgb_256", NpyArray.FromUInt8(normalRgb256, NativeMinimapSize, NativeMinimapSize, 3)),
		("height_hints_v7", NpyArray.FromFloat32(heightHints, 2)),
		("liquid_mask_257", NpyArray.FromUInt8(liquidMask257, NativeTileSize, NativeTileSize)),
		("liquid_height_257", NpyArray.FromFloat32(liquidHeight257, NativeTileSize, NativeTileSize)),
		("object_mask_257", NpyArray.FromUInt8(objectMask257, NativeTileSize, NativeTileSize)),
		("brush_mask_257", NpyArray.FromUInt8(brushMask257, NativeTileSize, NativeTileSize)),
	];
	if (hasMcly)
	{
		payload.Add(("mcal_alpha_pack_256", NpyArray.FromFloat32(mcalAlphaPack256, 256, 256, 4)));
		payload.Add(("mcly_layer_mask", NpyArray.FromUInt8(mclyLayerMask, 16, 16, 4)));
		payload.Add(("mcly_texture_ids", NpyArray.FromInt32(mclyTextureIds, 16, 16, 4)));
	}
	if (wdl17 is not null)
	{
		payload.Add(("wdl_17", NpyArray.FromFloat32(wdl17, 17, 17)));
		payload.Add(("wdl_delta_17", NpyArray.FromFloat32(SubtractGrids(height17, wdl17), 17, 17)));
	}
	if (minimapRgb256 is not null)
		payload.Add(("minimap_rgb_256", NpyArray.FromUInt8(minimapRgb256, NativeMinimapSize, NativeMinimapSize, 3)));

	WriteNpz(shardPath, payload);

	string debugJsonPath = sourceManifestPath;
	if (writeDebugJson)
	{
		string debugDirectory = Path.Combine(outputDir, "debug", datasetKey);
		Directory.CreateDirectory(debugDirectory);
		debugJsonPath = Path.Combine(debugDirectory, $"{entry.TileName}.json");
		object debugPayload = new
		{
			schema_version = "terrain-training-cache-debug.v1",
			created_at_utc = DateTimeOffset.UtcNow,
			source_manifest = sourceManifestPath,
			sample = new
			{
				entry.SampleId,
				entry.BuildLabel,
				entry.MapName,
				entry.TileX,
				entry.TileY,
				entry.SourceRoot,
				entry.RootAdtPath,
				entry.ObjAdtPath,
				entry.TexAdtPath,
				entry.WdlPath,
				SourceKind = entry.SourceKind.ToString(),
			},
			terrain = new
			{
				terrain.ChunkCount,
				terrain.DistinctAreaIdCount,
				terrain.LiquidFlagChunkCount,
				HeightMin = heightmap.MinHeight,
				HeightMax = heightmap.MaxHeight,
				HeightRange = heightmap.MaxHeight - heightmap.MinHeight,
				AuthoritativeSamples = heightmap.AuthoritativeSampleCount,
			},
			liquid = new
			{
				liquid.Chunks.Count,
				LayerCount = liquid.Chunks.Sum(static chunk => chunk.Layers.Count),
				liquidMode,
				has_native_liquid_heights = hasNativeLiquidHeights,
				liquid_coverage = liquidCoverage,
				mcnkSummary.ChunksWithMclq,
				mcnkSummary.ChunksWithLiquidFlags,
			},
			wdl = new
			{
				has_wdl_17 = wdl17 is not null,
				alignment_verified = includeVerifiedWdl,
				alignment_offset = wdlAlignment?.VerticalOffset ?? 0f,
				mean_wdl_delta = wdlAlignment?.MeanAbsoluteDelta ?? 0f,
				max_abs_wdl_delta = wdlAlignment?.MaxAbsoluteDelta ?? 0f,
			},
			minimap = new
			{
				has_minimap = minimapRgb256 is not null,
				minimap_source = minimapSource,
				minimap_variance = minimapVariance,
				minimap_gradient = minimapGradient,
			},
			objects = new
			{
				coverage = objectCoverage,
				mode = objectCoverage > 0f ? "placement-centroids" : "none",
			},
			brush = new
			{
				coverage = brushCoverage,
				mode = "none",
			},
			arrays = payload.Select(item => new { name = item.Name, shape = item.Array.Shape, dtype = item.Array.Descriptor }),
		};

		File.WriteAllText(debugJsonPath, JsonSerializer.Serialize(debugPayload, CreateJsonOptions()));
	}

	return new DirectCacheBuildResult(
		DatasetKey: datasetKey,
		ShardPath: shardPath,
		DebugJsonPath: debugJsonPath,
		HeightMin: heightmap.MinHeight,
		HeightMax: heightmap.MaxHeight,
		LiquidCoverage: liquidCoverage,
		ObjectCoverage: objectCoverage,
		BrushCoverage: brushCoverage,
		HoleCoverage: holeCoverage,
		MinimapVariance: minimapVariance,
		MinimapGradient: minimapGradient,
		DetailEnergy: detailEnergy,
		HasWdl17: wdl17 is not null,
		HasMinimap: minimapRgb256 is not null,
		HasNativeNormalMap: false,
		MinimapSource: minimapSource,
		ArrayNames: payload.Select(static item => item.Name).OrderBy(static name => name, StringComparer.Ordinal).ToArray());
}

static (float[] McalAlphaPack256, byte[] MclyLayerMask, int[] MclyTextureIds) BuildMcalMclySignals(
	WorldTerrainTileData terrain)
{
	const int TileChunks = 16;
	const int ChunkAlphaSize = 64;
	const int OutputAlphaSize = 256;
	const int MaxLayers = 4;
	const int DownsampleFactor = ChunkAlphaSize / (OutputAlphaSize / TileChunks);

	float[] alphaPack = new float[OutputAlphaSize * OutputAlphaSize * MaxLayers];
	int[] textureIds = new int[TileChunks * TileChunks * MaxLayers];
	byte[] layerMask = new byte[TileChunks * TileChunks * MaxLayers];

	for (int i = 0; i < textureIds.Length; i++)
		textureIds[i] = -1;

	foreach (WorldTerrainChunkData chunk in terrain.Chunks)
	{
		int cx = chunk.IndexX;
		int cy = chunk.IndexY;
		if ((uint)cx >= TileChunks || (uint)cy >= TileChunks)
			continue;

		int chunkBaseIdx = (cy * TileChunks + cx) * MaxLayers;

		for (int layerIdx = 0; layerIdx < chunk.TextureLayers.Count && layerIdx < MaxLayers; layerIdx++)
		{
			AdtTextureChunkLayer layer = chunk.TextureLayers[layerIdx];
			textureIds[chunkBaseIdx + layerIdx] = (int)layer.TextureId;
			layerMask[chunkBaseIdx + layerIdx] = 1;

			if (layer.DecodedAlpha?.AlphaMap is not { Length: ChunkAlphaSize * ChunkAlphaSize })
				continue;

			byte[] alphaMap = layer.DecodedAlpha.AlphaMap;

			int outOriginX = cx * (OutputAlphaSize / TileChunks);
			int outOriginY = cy * (OutputAlphaSize / TileChunks);

			for (int ly = 0; ly < OutputAlphaSize / TileChunks; ly++)
			{
				for (int lx = 0; lx < OutputAlphaSize / TileChunks; lx++)
				{
					float sum = 0;
					for (int sy = 0; sy < DownsampleFactor; sy++)
					{
						int srcY = ly * DownsampleFactor + sy;
						for (int sx = 0; sx < DownsampleFactor; sx++)
						{
							int srcX = lx * DownsampleFactor + sx;
							sum += alphaMap[srcY * ChunkAlphaSize + srcX];
						}
					}
					float avg = sum / (DownsampleFactor * DownsampleFactor);

					int outX = outOriginX + lx;
					int outY = outOriginY + ly;
					int flatIdx = (outY * OutputAlphaSize + outX) * MaxLayers + layerIdx;
					alphaPack[flatIdx] = avg / 255f;
				}
			}
		}
	}

	return (alphaPack, layerMask, textureIds);
}

static string BuildDirectDatasetKey(TerrainTrainingSampleDescriptor entry)
{
	return $"{SanitizeDatasetKeySegment(entry.BuildLabel)}__{SanitizeDatasetKeySegment(entry.MapName)}";
}

static string SanitizeDatasetKeySegment(string value)
{
	Span<char> buffer = stackalloc char[value.Length];
	int written = 0;
	foreach (char character in value)
	{
		buffer[written++] = char.IsLetterOrDigit(character) ? char.ToLowerInvariant(character) : '_';
	}

	string sanitized = new string(buffer[..written]).Trim('_');
	return string.IsNullOrWhiteSpace(sanitized) ? "dataset" : sanitized;
}

static float[] BuildChunkHeightsTensor(WorldTerrainTileData terrain)
{
	float[] tensor = new float[256 * 145];
	foreach (WorldTerrainChunkData chunk in terrain.Chunks)
	{
		if (chunk.Heights is null || !chunk.HasHeights)
			continue;

		int chunkIndex = (chunk.IndexY * 16) + chunk.IndexX;
		if ((uint)chunkIndex >= 256)
			continue;

		Buffer.BlockCopy(chunk.Heights, 0, tensor, chunkIndex * 145 * sizeof(float), 145 * sizeof(float));
	}

	return tensor;
}

static byte[] BuildHoleMask16x16(WorldTerrainTileData terrain)
{
	byte[] mask = new byte[16 * 16];
	foreach (WorldTerrainChunkData chunk in terrain.Chunks)
	{
		int index = (chunk.IndexY * 16) + chunk.IndexX;
		if ((uint)index >= mask.Length)
			continue;

		mask[index] = chunk.HoleMask != 0 ? (byte)1 : (byte)0;
	}

	return mask;
}

static float[] BuildHeightHints(float heightMin, float heightMax)
{
	float globalRange = MathF.Max(HeightGlobalMax - HeightGlobalMin, 1e-6f);
	return
	[
		Math.Clamp((heightMin - HeightGlobalMin) / globalRange, 0f, 1f),
		Math.Clamp((heightMax - HeightGlobalMin) / globalRange, 0f, 1f),
	];
}

static float[] DownsampleHeightGrid(float[] source, int sourceSize, int targetSize)
{
	int step = (sourceSize - 1) / (targetSize - 1);
	float[] result = new float[targetSize * targetSize];
	for (int y = 0; y < targetSize; y++)
	{
		for (int x = 0; x < targetSize; x++)
			result[(y * targetSize) + x] = source[(y * step * sourceSize) + (x * step)];
	}

	return result;
}

static float[] ResizeFloatGridBilinear(float[] source, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
{
	float[] result = new float[targetWidth * targetHeight];
	for (int y = 0; y < targetHeight; y++)
	{
		float sourceY = targetHeight == 1 ? 0f : y * (sourceHeight - 1f) / (targetHeight - 1f);
		int y0 = Math.Clamp((int)MathF.Floor(sourceY), 0, sourceHeight - 1);
		int y1 = Math.Clamp(y0 + 1, 0, sourceHeight - 1);
		float fy = sourceY - y0;
		for (int x = 0; x < targetWidth; x++)
		{
			float sourceX = targetWidth == 1 ? 0f : x * (sourceWidth - 1f) / (targetWidth - 1f);
			int x0 = Math.Clamp((int)MathF.Floor(sourceX), 0, sourceWidth - 1);
			int x1 = Math.Clamp(x0 + 1, 0, sourceWidth - 1);
			float fx = sourceX - x0;

			float top = Lerp(source[(y0 * sourceWidth) + x0], source[(y0 * sourceWidth) + x1], fx);
			float bottom = Lerp(source[(y1 * sourceWidth) + x0], source[(y1 * sourceWidth) + x1], fx);
			result[(y * targetWidth) + x] = Lerp(top, bottom, fy);
		}
	}

	return result;
}

static float[] SubtractGrids(float[] left, float[] right)
{
	float[] result = new float[left.Length];
	for (int index = 0; index < left.Length; index++)
		result[index] = left[index] - right[index];

	return result;
}

static float Lerp(float a, float b, float t) => a + ((b - a) * t);

static float ComputeDetailEnergy(float[] height257, float[] height65)
{
	float[] upsampled = ResizeFloatGridBilinear(height65, 65, 65, NativeTileSize, NativeTileSize);
	float sum = 0f;
	for (int index = 0; index < height257.Length; index++)
		sum += MathF.Abs(height257[index] - upsampled[index]);

	return sum / height257.Length;
}

static byte[] CreateSolidRgbImage(int width, int height, byte r, byte g, byte b)
{
	byte[] rgb = new byte[width * height * 3];
	for (int index = 0; index < rgb.Length; index += 3)
	{
		rgb[index] = r;
		rgb[index + 1] = g;
		rgb[index + 2] = b;
	}

	return rgb;
}

static (byte[] Mask, float[] Heights, bool HasNativeHeights, string Mode) RasterizeLiquidSignals(
	WorldTerrainTileData terrain,
	AdtLiquidFile liquid,
	AdtMcnkSummary mcnkSummary,
	float[] terrainHeight257)
{
	byte[] mask = new byte[NativeTileSize * NativeTileSize];
	float[] heights = new float[NativeTileSize * NativeTileSize];
	bool[] resolved = new bool[NativeTileSize * NativeTileSize];
	bool hasNativeHeights = false;

	foreach (AdtLiquidChunk chunk in liquid.Chunks)
	{
		int chunkX = chunk.ChunkIndex % 16;
		int chunkY = chunk.ChunkIndex / 16;
		foreach (AdtLiquidLayer layer in chunk.Layers)
		{
			for (int localY = 0; localY < layer.Height; localY++)
			{
				for (int localX = 0; localX < layer.Width; localX++)
				{
					if (!layer.TileExists(localX, localY))
						continue;

					int gridBaseX = (chunkX * 16) + ((layer.XOffset + localX) * 2);
					int gridBaseY = (chunkY * 16) + ((layer.YOffset + localY) * 2);
					for (int dy = 0; dy <= 2; dy++)
					{
						for (int dx = 0; dx <= 2; dx++)
						{
							int x = gridBaseX + dx;
							int y = gridBaseY + dy;
							if ((uint)x >= NativeTileSize || (uint)y >= NativeTileSize)
								continue;

							int index = (y * NativeTileSize) + x;
							mask[index] = 1;
							float height = layer.MinHeight;
							if (TrySampleLiquidHeight(layer, localX, localY, dx * 0.5f, dy * 0.5f, out float sampledHeight))
							{
								height = sampledHeight;
								hasNativeHeights = true;
							}

							if (!resolved[index] || height > heights[index])
							{
								heights[index] = height;
								resolved[index] = true;
							}
						}
					}
				}
			}
		}
	}

	if (!mask.Any(static value => value != 0) && (terrain.LiquidFlagChunkCount > 0 || mcnkSummary.ChunksWithMclq > 0))
	{
		PaintChunkLiquidFallback(mask, heights, resolved, terrain, terrainHeight257);
		return (mask, heights, false, "chunk-liquid-flag-fallback");
	}

	for (int index = 0; index < heights.Length; index++)
	{
		if (mask[index] == 0)
			heights[index] = 0f;
	}

	return (mask, heights, hasNativeHeights, hasNativeHeights ? "mh2o-native" : "mh2o-flat");
}

static void PaintChunkLiquidFallback(byte[] mask, float[] heights, bool[] resolved, WorldTerrainTileData terrain, float[] terrainHeight257)
{
	foreach (WorldTerrainChunkData chunk in terrain.Chunks)
	{
		if (!chunk.HasLiquidFlags)
			continue;

		int baseX = chunk.IndexX * 16;
		int baseY = chunk.IndexY * 16;
		for (int localY = 0; localY <= 16; localY++)
		{
			for (int localX = 0; localX <= 16; localX++)
			{
				int x = baseX + localX;
				int y = baseY + localY;
				if ((uint)x >= NativeTileSize || (uint)y >= NativeTileSize)
					continue;

				int index = (y * NativeTileSize) + x;
				mask[index] = 1;
				if (!resolved[index])
					heights[index] = terrainHeight257[index];
			}
		}
	}
}

static bool TrySampleLiquidHeight(AdtLiquidLayer layer, int tileX, int tileY, float offsetX, float offsetY, out float height)
{
	height = layer.MinHeight;
	if (layer.Heights is null)
		return false;

	int vertexWidth = layer.Width + 1;
	int vertexHeight = layer.Height + 1;
	if (layer.Heights.Length < vertexWidth * vertexHeight)
		return false;

	float sourceX = tileX + Math.Clamp(offsetX, 0f, 1f);
	float sourceY = tileY + Math.Clamp(offsetY, 0f, 1f);
	int x0 = Math.Clamp((int)MathF.Floor(sourceX), 0, vertexWidth - 1);
	int y0 = Math.Clamp((int)MathF.Floor(sourceY), 0, vertexHeight - 1);
	int x1 = Math.Clamp(x0 + 1, 0, vertexWidth - 1);
	int y1 = Math.Clamp(y0 + 1, 0, vertexHeight - 1);
	float fx = sourceX - x0;
	float fy = sourceY - y0;

	float top = Lerp(layer.Heights[(y0 * vertexWidth) + x0], layer.Heights[(y0 * vertexWidth) + x1], fx);
	float bottom = Lerp(layer.Heights[(y1 * vertexWidth) + x0], layer.Heights[(y1 * vertexWidth) + x1], fx);
	height = Lerp(top, bottom, fy);
	return true;
}

static float[]? TryBuildWdl17(WdlSummary? summary, int tileX, int tileY)
{
	if (summary is null || !summary.TryGetTile(tileX, tileY, out WdlTileSummary? tile) || tile is null)
		return null;

	float[] heights = new float[WdlTileSummary.OuterHeightCount];
	for (int index = 0; index < heights.Length; index++)
		heights[index] = tile.OuterHeights[index];

	return heights;
}

static WdlAlignment? TryBuildAlignedWdl17(float[] terrainHeight17, WdlSummary? summary, int tileX, int tileY)
{
	float[]? rawWdl17 = TryBuildWdl17(summary, tileX, tileY);
	if (rawWdl17 is null || terrainHeight17.Length != rawWdl17.Length)
		return null;

	float[] deltas = new float[terrainHeight17.Length];
	for (int index = 0; index < deltas.Length; index++)
		deltas[index] = terrainHeight17[index] - rawWdl17[index];

	float verticalOffset = ComputeMedian(deltas);
	float[] alignedHeights17 = new float[rawWdl17.Length];
	float absoluteDeltaSum = 0f;
	float maxAbsoluteDelta = 0f;
	for (int index = 0; index < alignedHeights17.Length; index++)
	{
		float aligned = rawWdl17[index] + verticalOffset;
		alignedHeights17[index] = aligned;
		float absoluteDelta = MathF.Abs(terrainHeight17[index] - aligned);
		absoluteDeltaSum += absoluteDelta;
		if (absoluteDelta > maxAbsoluteDelta)
			maxAbsoluteDelta = absoluteDelta;
	}

	return new WdlAlignment(
		AlignedHeights17: alignedHeights17,
		VerticalOffset: verticalOffset,
		MeanAbsoluteDelta: absoluteDeltaSum / alignedHeights17.Length,
		MaxAbsoluteDelta: maxAbsoluteDelta);
}

static bool IsVerifiedWdlAlignment(WdlAlignment? alignment)
{
	return alignment is not null
		&& alignment.MeanAbsoluteDelta <= 256f
		&& alignment.MaxAbsoluteDelta <= 1024f;
}

static float ComputeMedian(float[] values)
{
	ArgumentNullException.ThrowIfNull(values);
	if (values.Length == 0)
		return 0f;

	float[] copy = values.ToArray();
	Array.Sort(copy);
	int middle = copy.Length / 2;
	if ((copy.Length & 1) == 1)
		return copy[middle];

	return (copy[middle - 1] + copy[middle]) * 0.5f;
}

static byte[]? TryLoadMinimapRgb(
	TerrainTrainingSampleDescriptor entry,
	Dictionary<string, IArchiveCatalog> archiveCatalogs,
	Dictionary<string, Md5TranslateIndex?> minimapMd5Cache,
	out string? sourceName)
{
	string mapDirectory = ResolveMinimapMapDirectory(entry);
	(int companionTileX, int companionTileY) = ResolveCompanionTileCoordinates(entry);
	IArchiveCatalog? archiveCatalog = null;
	if (entry.SourceKind == TerrainTrainingSampleSourceKind.MountedArchive)
		archiveCatalog = GetOrCreateArchiveCatalog(entry.SourceRoot, archiveCatalogs);

	foreach (string candidate in EnumerateMinimapCandidates(mapDirectory, companionTileX, companionTileY))
	{
		if (entry.SourceKind == TerrainTrainingSampleSourceKind.MountedArchive)
		{
			byte[]? candidateBytes = archiveCatalog!.ReadFile(candidate);
			if (candidateBytes is { Length: > 0 })
			{
				sourceName = candidate;
				return DecodeArchiveBackedMinimap(candidateBytes, candidate);
			}
		}
		else
		{
			string? path = ResolveFilesystemMinimapPath(entry.SourceRoot, candidate);
			if (path is not null)
			{
				sourceName = path;
				return DecodeFilesystemMinimap(path);
			}
		}
	}

	if (archiveCatalog is not null)
	{
		Md5TranslateIndex? translateIndex = TryGetCachedMinimapMd5Index(entry, archiveCatalogs, minimapMd5Cache);
		if (translateIndex is not null)
		{
			foreach (string candidate in EnumerateMinimapCandidates(mapDirectory, companionTileX, companionTileY))
			{
				string lookupKey = translateIndex.Normalize(candidate);
				foreach (string translatedPath in translateIndex.GetHashCandidates(lookupKey))
				{
					if (string.IsNullOrWhiteSpace(translatedPath))
						continue;

					string archivePath = translatedPath.Replace('/', '\\');
					byte[]? candidateBytes = archiveCatalog.ReadFile(archivePath);
					if (candidateBytes is not { Length: > 0 })
						continue;

					sourceName = archivePath;
					return DecodeArchiveBackedMinimap(candidateBytes, archivePath);
				}
			}
		}
	}

	sourceName = null;
	return null;
}

static Md5TranslateIndex? TryGetCachedMinimapMd5Index(
	TerrainTrainingSampleDescriptor entry,
	Dictionary<string, IArchiveCatalog> archiveCatalogs,
	Dictionary<string, Md5TranslateIndex?> minimapMd5Cache)
{
	if (entry.SourceKind != TerrainTrainingSampleSourceKind.MountedArchive)
		return null;

	string mapDirectory = ResolveMinimapMapDirectory(entry);
	string cacheKey = $"{entry.SourceRoot}|{mapDirectory}";
	if (minimapMd5Cache.TryGetValue(cacheKey, out Md5TranslateIndex? cachedIndex))
		return cachedIndex;

	IArchiveCatalog archiveCatalog = GetOrCreateArchiveCatalog(entry.SourceRoot, archiveCatalogs);
	string spacedMap = InsertSpaceBeforeCapitals(mapDirectory);
	List<string> extraCandidates =
	[
		$"World\\Maps\\{mapDirectory}\\md5translate.trs",
		$"World\\Maps\\{mapDirectory}\\md5translate.txt",
	];
	if (!string.Equals(spacedMap, mapDirectory, StringComparison.Ordinal))
	{
		extraCandidates.Add($"World\\Maps\\{spacedMap}\\md5translate.trs");
		extraCandidates.Add($"World\\Maps\\{spacedMap}\\md5translate.txt");
	}

	Md5TranslateResolver.TryLoad(
		searchPaths: BuildLegacySearchRoots(entry.SourceRoot),
		archiveFileExists: archiveCatalog.FileExists,
		archiveReadFile: archiveCatalog.ReadFile,
		index: out Md5TranslateIndex? resolvedIndex,
		extraCandidates: extraCandidates);

	minimapMd5Cache[cacheKey] = resolvedIndex;
	return resolvedIndex;
}

static string ResolveMinimapMapDirectory(TerrainTrainingSampleDescriptor entry)
{
	return string.IsNullOrWhiteSpace(entry.MapDirectory)
		? entry.MapName
		: entry.MapDirectory;
}

static IEnumerable<string> EnumerateMinimapCandidates(string mapName, int tileX, int tileY)
{
	string x2 = tileX.ToString("D2");
	string y2 = tileY.ToString("D2");
	string trsName = $"map{tileX}_{y2}.blp";
	string paddedName = $"map{x2}_{y2}.blp";
	string spacedMap = InsertSpaceBeforeCapitals(mapName);
	HashSet<string> yielded = new(StringComparer.OrdinalIgnoreCase);

	IEnumerable<string> EmitMapFolderCandidates(string folderName)
	{
		yield return $"{folderName}\\{trsName}";
		yield return $"{folderName}\\{paddedName}";
		yield return $"textures\\minimap\\{folderName}\\{trsName}";
		yield return $"textures\\minimap\\{folderName}\\{folderName}_{x2}_{y2}.blp";
		yield return $"textures\\minimap\\{folderName}\\{paddedName}";
		yield return $"World\\Minimaps\\{folderName}\\{trsName}";
		yield return $"World\\Minimaps\\{folderName}\\map{tileX}_{tileY}.blp";
		yield return $"World\\Minimaps\\{folderName}\\{paddedName}";
	}

	foreach (string candidate in EmitMapFolderCandidates(mapName))
	{
		if (yielded.Add(candidate))
			yield return candidate;
	}

	string lowerFolderCandidate = MinimapService.GetMinimapTilePath(mapName, tileX, tileY).Replace('/', '\\');
	if (yielded.Add(lowerFolderCandidate))
		yield return lowerFolderCandidate;

	string lowerFolderTrs = $"textures\\minimap\\{mapName.ToLowerInvariant()}\\{trsName}";
	if (yielded.Add(lowerFolderTrs))
		yield return lowerFolderTrs;

	if (!string.Equals(spacedMap, mapName, StringComparison.Ordinal))
	{
		foreach (string candidate in EmitMapFolderCandidates(spacedMap))
		{
			if (yielded.Add(candidate))
				yield return candidate;
		}

		string spacedLowerFolderTrs = $"textures\\minimap\\{spacedMap.ToLowerInvariant()}\\{trsName}";
		if (yielded.Add(spacedLowerFolderTrs))
			yield return spacedLowerFolderTrs;

		string spacedLowerFolderPadded = $"textures\\minimap\\{spacedMap.ToLowerInvariant()}\\{paddedName}";
		if (yielded.Add(spacedLowerFolderPadded))
			yield return spacedLowerFolderPadded;
	}

	string mapTileName = $"{mapName}_{x2}_{y2}.blp";
	string mapTileNameRaw = $"{mapName}_{tileX}_{tileY}.blp";
	string mapTilePng = $"{mapName}_{tileX}_{tileY}.png";
	string mapTilePaddedPng = $"{mapName}_{x2}_{y2}.png";
	foreach (string candidate in new[]
	{
		$"Textures\\Minimap\\{mapTileName}",
		$"Textures\\Minimap\\{mapTileNameRaw}",
		$"World\\Textures\\Minimap\\{mapTilePng}",
		$"World\\Textures\\Minimap\\{mapTilePaddedPng}",
		$"Textures\\Minimap\\{mapTilePng}",
		$"Textures\\Minimap\\{mapTilePaddedPng}",
	})
	{
		if (yielded.Add(candidate))
			yield return candidate;
	}
}

static string InsertSpaceBeforeCapitals(string input)
{
	if (string.IsNullOrWhiteSpace(input))
		return input;

	StringBuilder builder = new(input.Length + 4);
	for (int index = 0; index < input.Length; index++)
	{
		char character = input[index];
		if (index > 0 && char.IsUpper(character) && !char.IsWhiteSpace(input[index - 1]))
			builder.Append(' ');
		builder.Append(character);
	}

	return builder.ToString();
}

static string? ResolveFilesystemMinimapPath(string clientRoot, string relativePath)
{
	string normalized = relativePath.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
	string underData = Path.Combine(clientRoot, "Data", normalized);
	if (File.Exists(underData))
		return underData;

	string underDataMpq = underData + ".MPQ";
	if (File.Exists(underDataMpq))
		return underDataMpq;

	string underRoot = Path.Combine(clientRoot, normalized);
	if (File.Exists(underRoot))
		return underRoot;

	string underRootMpq = underRoot + ".MPQ";
	return File.Exists(underRootMpq) ? underRootMpq : null;
}

static byte[]? DecodeArchiveBackedMinimap(byte[] bytes, string sourcePath)
{
	if (sourcePath.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
		return DecodeRgbImage(bytes);

	using MemoryStream stream = new(bytes, writable: false);
	using BlpFile blp = new(stream);
	using System.Drawing.Bitmap bitmap = blp.GetBitmap(0);
	return NormalizeBitmap(bitmap, NativeMinimapSize, NativeMinimapSize);
}

static byte[]? DecodeFilesystemMinimap(string path)
{
	if (path.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
		return DecodeRgbImage(File.ReadAllBytes(path));

	using FileStream stream = File.OpenRead(path);
	using BlpFile blp = new(stream);
	using System.Drawing.Bitmap bitmap = blp.GetBitmap(0);
	return NormalizeBitmap(bitmap, NativeMinimapSize, NativeMinimapSize);
}

static byte[]? DecodeRgbImage(byte[] bytes)
{
	using Image<Rgba32> image = Image.Load<Rgba32>(bytes);
	if (image.Width != NativeMinimapSize || image.Height != NativeMinimapSize)
		image.Mutate(context => context.Resize(NativeMinimapSize, NativeMinimapSize));

	return FlattenImageRgb(image);
}

static byte[] NormalizeRgbPixels(byte[] rgbaPixels, int width, int height, int targetWidth, int targetHeight)
{
	using Image<Rgba32> image = Image.LoadPixelData<Rgba32>(rgbaPixels, width, height);
	if (image.Width != targetWidth || image.Height != targetHeight)
		image.Mutate(context => context.Resize(targetWidth, targetHeight));

	return FlattenImageRgb(image);
}

static byte[] NormalizeBitmap(System.Drawing.Bitmap bitmap, int targetWidth, int targetHeight)
{
	using MemoryStream pngStream = new();
	bitmap.Save(pngStream, System.Drawing.Imaging.ImageFormat.Png);
	return DecodeRgbImage(pngStream.ToArray())!;
}

static byte[] FlattenImageRgb(Image<Rgba32> image)
{
	byte[] rgb = new byte[image.Width * image.Height * 3];
	image.ProcessPixelRows(accessor =>
	{
		for (int y = 0; y < image.Height; y++)
		{
			Span<Rgba32> row = accessor.GetRowSpan(y);
			int offset = y * image.Width * 3;
			for (int x = 0; x < image.Width; x++)
			{
				rgb[offset++] = row[x].R;
				rgb[offset++] = row[x].G;
				rgb[offset++] = row[x].B;
			}
		}
	});

	return rgb;
}

static float ComputeBinaryCoverage(byte[] values)
{
	if (values.Length == 0)
		return 0f;

	int nonZero = 0;
	for (int index = 0; index < values.Length; index++)
	{
		if (values[index] != 0)
			nonZero++;
	}

	return (float)nonZero / values.Length;
}

static float ComputeRgbVariance(byte[] rgb)
{
	if (rgb.Length == 0)
		return 0f;

	double sum = 0d;
	double sumSquares = 0d;
	int sampleCount = rgb.Length / 3;
	for (int index = 0; index < rgb.Length; index += 3)
	{
		double gray = (rgb[index] + rgb[index + 1] + rgb[index + 2]) / (3d * 255d);
		sum += gray;
		sumSquares += gray * gray;
	}

	double mean = sum / sampleCount;
	return (float)Math.Max(0d, (sumSquares / sampleCount) - (mean * mean));
}

static float ComputeAverageGradientMagnitude(byte[] rgb, int width, int height)
{
	if (width <= 1 || height <= 1)
		return 0f;

	float[] gray = new float[width * height];
	for (int index = 0, pixel = 0; index < gray.Length; index++, pixel += 3)
		gray[index] = (rgb[pixel] + rgb[pixel + 1] + rgb[pixel + 2]) / (3f * 255f);

	double sum = 0d;
	int samples = 0;
	for (int y = 0; y < height - 1; y++)
	{
		for (int x = 0; x < width - 1; x++)
		{
			float dx = gray[(y * width) + (x + 1)] - gray[(y * width) + x];
			float dy = gray[((y + 1) * width) + x] - gray[(y * width) + x];
			sum += Math.Sqrt((dx * dx) + (dy * dy));
			samples++;
		}
	}

	return samples == 0 ? 0f : (float)(sum / samples);
}

static float ComputeMeanAbsoluteDifference(float[] left, float[] right)
{
	double sum = 0d;
	for (int index = 0; index < left.Length; index++)
		sum += Math.Abs(left[index] - right[index]);

	return (float)(sum / left.Length);
}

static float ComputeMaxAbsoluteDifference(float[] left, float[] right)
{
	float max = 0f;
	for (int index = 0; index < left.Length; index++)
		max = MathF.Max(max, MathF.Abs(left[index] - right[index]));

	return max;
}

static byte[] BuildObjectMask257(TerrainTrainingSampleDescriptor entry, Dictionary<string, IArchiveCatalog> archiveCatalogs)
{
	byte[] mask = new byte[NativeTileSize * NativeTileSize];
	AdtPlacementCatalog? placements = TryReadPlacementCatalogForBuildCache(entry, archiveCatalogs);
	if (placements is null)
		return mask;

	foreach (AdtModelPlacement placement in placements.ModelPlacements)
		PaintPlacementCentroid(mask, placement.Position, entry.TileX, entry.TileY, radiusPixels: 2);
	foreach (AdtWorldModelPlacement placement in placements.WorldModelPlacements)
		PaintPlacementCentroid(mask, placement.Position, entry.TileX, entry.TileY, radiusPixels: 3);

	return mask;
}

static AdtPlacementCatalog? TryReadPlacementCatalogForBuildCache(TerrainTrainingSampleDescriptor entry, Dictionary<string, IArchiveCatalog> archiveCatalogs)
{
	string? placementPath = entry.ObjAdtPath ?? entry.RootAdtPath;
	if (string.IsNullOrWhiteSpace(placementPath))
		return null;

	if (entry.SourceKind == TerrainTrainingSampleSourceKind.MountedArchive)
	{
		IArchiveCatalog archiveCatalog = GetOrCreateArchiveCatalog(entry.SourceRoot, archiveCatalogs);
		if (TryReadAlphaEmbeddedTile(entry, archiveCatalog, out AlphaEmbeddedAdtTileData? alphaTile))
			return alphaTile!.PlacementCatalog;

		byte[] bytes = archiveCatalog.ReadFile(placementPath) ?? [];
		if (bytes.Length == 0)
			return null;

		using MemoryStream stream = new(bytes, writable: false);
		MapFileSummary summary = MapFileSummaryReader.Read(stream, placementPath);
		return AdtPlacementReader.Read(stream, summary);
	}

	if (!File.Exists(placementPath))
		return null;

	using FileStream fileStream = File.OpenRead(placementPath);
	MapFileSummary filesystemSummary = MapFileSummaryReader.Read(fileStream, Path.GetFullPath(placementPath));
	return AdtPlacementReader.Read(fileStream, filesystemSummary);
}

static void PaintPlacementCentroid(byte[] mask, Vector3 position, int tileX, int tileY, int radiusPixels)
{
	if (!TryProjectPlacementToTilePixel(position, tileX, tileY, out int centerX, out int centerY))
		return;

	int radiusSquared = radiusPixels * radiusPixels;
	for (int dy = -radiusPixels; dy <= radiusPixels; dy++)
	{
		for (int dx = -radiusPixels; dx <= radiusPixels; dx++)
		{
			if ((dx * dx) + (dy * dy) > radiusSquared)
				continue;

			int x = centerX + dx;
			int y = centerY + dy;
			if ((uint)x >= NativeTileSize || (uint)y >= NativeTileSize)
				continue;

			mask[(y * NativeTileSize) + x] = 1;
		}
	}
}

static bool TryProjectPlacementToTilePixel(Vector3 position, int tileX, int tileY, out int pixelX, out int pixelY)
{
	pixelX = 0;
	pixelY = 0;
	(float U, float V)[] candidates =
	[
		((position.X / WorldTileSize) - tileX, (position.Z / WorldTileSize) - tileY),
		(((WorldMapOrigin - position.Z) / WorldTileSize) - tileX, ((WorldMapOrigin - position.X) / WorldTileSize) - tileY),
		((position.X / WorldTileSize) - tileX, (position.Y / WorldTileSize) - tileY),
		(((WorldMapOrigin - position.Y) / WorldTileSize) - tileX, ((WorldMapOrigin - position.X) / WorldTileSize) - tileY),
	];

	float bestScore = float.MinValue;
	(float U, float V) best = default;
	bool found = false;
	foreach ((float U, float V) candidate in candidates)
	{
		if (candidate.U < -0.25f || candidate.U > 1.25f || candidate.V < -0.25f || candidate.V > 1.25f)
			continue;

		float distanceToCenter = MathF.Abs(candidate.U - 0.5f) + MathF.Abs(candidate.V - 0.5f);
		float score = -distanceToCenter;
		if (score > bestScore)
		{
			bestScore = score;
			best = candidate;
			found = true;
		}
	}

	if (!found)
		return false;

	pixelX = Math.Clamp((int)MathF.Round(best.U * (NativeTileSize - 1)), 0, NativeTileSize - 1);
	pixelY = Math.Clamp((int)MathF.Round(best.V * (NativeTileSize - 1)), 0, NativeTileSize - 1);
	return true;
}

static void WriteNpz(string path, IReadOnlyList<(string Name, NpyArray Array)> payload)
{
	using FileStream stream = File.Create(path);
	using ZipArchive archive = new(stream, ZipArchiveMode.Create, leaveOpen: false);
	foreach ((string name, NpyArray array) in payload)
	{
		ZipArchiveEntry entry = archive.CreateEntry($"{name}.npy", CompressionLevel.Optimal);
		using Stream entryStream = entry.Open();
		WriteNpy(entryStream, array);
	}
}

static void WriteNpy(Stream stream, NpyArray array)
{
	byte[] magic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 0x01, 0x00];
	stream.Write(magic);

	string shapeText = array.Shape.Length == 1
		? $"({array.Shape[0]},)"
		: $"({string.Join(", ", array.Shape)})";
	string headerText = $"{{'descr': '{array.Descriptor}', 'fortran_order': False, 'shape': {shapeText}, }}";
	int preambleLength = magic.Length + 2;
	int paddedLength = headerText.Length + 1;
	while ((paddedLength + preambleLength) % 16 != 0)
		paddedLength++;
	string finalHeader = headerText.PadRight(paddedLength - 1, ' ') + '\n';
	byte[] headerBytes = System.Text.Encoding.ASCII.GetBytes(finalHeader);
	ushort headerLength = checked((ushort)headerBytes.Length);
	stream.WriteByte((byte)(headerLength & 0xFF));
	stream.WriteByte((byte)(headerLength >> 8));
	stream.Write(headerBytes);
	stream.Write(array.Data);
}

static bool ResolveBooleanOption(string[] args, string enableName, string disableName, bool defaultValue)
{
	if (HasFlag(args, enableName))
		return true;
	if (HasFlag(args, disableName))
		return false;
	return defaultValue;
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

static bool FilesystemMapExists(string mapPath, string mapName)
{
	return Directory.Exists(mapPath)
		&& Directory.EnumerateFiles(mapPath, $"{mapName}_*.adt", SearchOption.TopDirectoryOnly)
			.Any(static path => !path.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
				&& !path.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase)
				&& !path.EndsWith("_lod.adt", StringComparison.OrdinalIgnoreCase));
}

static List<MapDirectoryEntry> DiscoverDatasetMaps(string clientRoot)
{
	using IArchiveCatalog archiveCatalog = CreateArchiveCatalog(clientRoot);
	MapDirectoryLookup lookup = new();
	lookup.Load(BuildLegacySearchRoots(clientRoot), archiveCatalog);

	List<MapDirectoryEntry> discovered = [];
	foreach (MapDirectoryEntry entry in lookup.Entries.OrderBy(static entry => entry.Directory, StringComparer.OrdinalIgnoreCase))
	{
		string mapPath = Path.Combine(clientRoot, "Data", "World", "Maps", entry.Directory);
		if (FilesystemMapExists(mapPath, entry.Directory) || ArchiveMapExists(archiveCatalog, clientRoot, entry.Directory, mapPath))
			discovered.Add(entry);
	}

	return discovered;
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
	IArchiveCatalog archiveCatalog = new NativeMpqServiceFactory().Create();
	ArchiveCatalogBootstrapper.Bootstrap(
		archiveCatalog,
		BuildLegacySearchRoots(clientRoot),
		new ArchiveCatalogBootstrapOptions(ExternalListfilePath: ResolveLegacyListfilePath()));

	return archiveCatalog;
}

static IReadOnlyList<string> BuildLegacySearchRoots(string clientRoot)
{
	List<string> roots = [];
	string dataRoot = Path.Combine(clientRoot, "Data");
	if (Directory.Exists(dataRoot))
		roots.Add(dataRoot);

	if (!string.Equals(clientRoot, dataRoot, StringComparison.OrdinalIgnoreCase))
		roots.Add(clientRoot);

	return roots.Count > 0 ? roots : [clientRoot];
}

static string? ResolveLegacyListfilePath()
{
	string[] candidates =
	[
		Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MdxViewer", "community-listfile-withcapitals.csv"),
		Path.Combine(AppContext.BaseDirectory, "community-listfile-withcapitals.csv"),
		"community-listfile-withcapitals.csv",
		"listfile.csv",
	];

	foreach (string candidate in candidates)
	{
		if (File.Exists(candidate))
			return candidate;
	}

	return null;
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

static string GetMapDirectoryFromMapVirtualRoot(string mapVirtualRoot, string mapName)
{
	string trimmed = mapVirtualRoot.TrimEnd('\\', '/');
	if (trimmed.EndsWith(".wdt", StringComparison.OrdinalIgnoreCase))
		return Path.GetFileNameWithoutExtension(trimmed) ?? mapName;

	return Path.GetFileName(trimmed) ?? mapName;
}

static bool TryReadAlphaEmbeddedTile(TerrainTrainingSampleDescriptor entry, IArchiveCatalog archiveCatalog, out AlphaEmbeddedAdtTileData? alphaTile)
{
	alphaTile = null;
	if (!TryParseAlphaEmbeddedTileSourcePath(entry.RootAdtPath, out string mapDirectory, out int tileX, out int tileY))
		return false;

	return AlphaEmbeddedAdtReader.TryReadTile(entry.SourceRoot, mapDirectory, tileX, tileY, archiveCatalog, out alphaTile);
}

static bool TryParseAlphaEmbeddedTileSourcePath(string sourcePath, out string mapDirectory, out int tileX, out int tileY)
{
	mapDirectory = string.Empty;
	tileX = 0;
	tileY = 0;

	const string marker = "#alpha-tile(";
	int markerIndex = sourcePath.LastIndexOf(marker, StringComparison.OrdinalIgnoreCase);
	if (markerIndex < 0 || !sourcePath.EndsWith(')'))
		return false;

	string prefix = sourcePath[..markerIndex].TrimEnd('\\', '/');
	string coords = sourcePath[(markerIndex + marker.Length)..^1];
	string[] parts = coords.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
	if (parts.Length != 2
		|| !int.TryParse(parts[0], out tileX)
		|| !int.TryParse(parts[1], out tileY))
	{
		return false;
	}

	string? directoryPath = Path.GetDirectoryName(prefix.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar));
	mapDirectory = string.IsNullOrWhiteSpace(directoryPath) ? string.Empty : Path.GetFileName(directoryPath) ?? string.Empty;
	return !string.IsNullOrWhiteSpace(mapDirectory);
}

static (int TileX, int TileY) ResolveCompanionTileCoordinates(TerrainTrainingSampleDescriptor entry)
{
	if (TryParseAlphaEmbeddedTileSourcePath(entry.RootAdtPath, out _, out _, out _))
		return (entry.TileY, entry.TileX);

	return (entry.TileX, entry.TileY);
}

static AdtSummary BuildAlphaEmbeddedAdtSummary(AlphaEmbeddedAdtTileData alphaTile)
{
	return new AdtSummary(
		alphaTile.SourcePath,
		MapFileKind.Adt,
		terrainChunkCount: alphaTile.TerrainTileData.ChunkCount,
		textureNameCount: 0,
		modelNameCount: alphaTile.PlacementCatalog.ModelNames.Count,
		worldModelNameCount: alphaTile.PlacementCatalog.WorldModelNames.Count,
		modelPlacementCount: alphaTile.PlacementCatalog.ModelPlacements.Count,
		worldModelPlacementCount: alphaTile.PlacementCatalog.WorldModelPlacements.Count,
		hasFlightBounds: false,
		hasWater: alphaTile.LiquidTileData.ActiveChunkCount > 0,
		hasTextureParams: false,
		hasTextureFlags: false);
}

static AdtLiquidFile BuildAlphaEmbeddedLiquidFile(AlphaEmbeddedAdtTileData alphaTile)
{
	IReadOnlyList<AdtLiquidChunk> chunks = alphaTile.LiquidTileData.Chunks
		.Select(static chunk => new AdtLiquidChunk(
			chunk.ChunkIndex,
			chunk.FishableMask,
			chunk.DeepMask,
			chunk.Layers.Select(static layer => new AdtLiquidLayer(
				layer.LiquidTypeId,
				layer.BasicType,
				layer.VertexFormat,
				layer.MinHeight,
				layer.MaxHeight,
				layer.XOffset,
				layer.YOffset,
				layer.Width,
				layer.Height,
				existsBitmap: Array.Empty<byte>(),
				heights: null,
				depths: null,
				uvs: null)).ToArray()))
		.ToArray();

	return new AdtLiquidFile(alphaTile.SourcePath, MapFileKind.Adt, chunks);
}

static AdtMcnkSummary BuildAlphaEmbeddedMcnkSummary(AlphaEmbeddedAdtTileData alphaTile)
{
	WorldTerrainTileData terrain = alphaTile.TerrainTileData;
	int totalLayerCount = terrain.Chunks.Sum(static chunk => chunk.LayerCount);
	int maxLayerCount = terrain.Chunks.Count > 0 ? terrain.Chunks.Max(static chunk => chunk.LayerCount) : 0;
	int chunksWithMcly = terrain.Chunks.Count(static chunk => chunk.LayerCount > 0);
	int chunksWithMultipleLayers = terrain.Chunks.Count(static chunk => chunk.LayerCount > 1);
	int chunksWithMclq = alphaTile.LiquidTileData.ActiveChunkCount;

	return new AdtMcnkSummary(
		alphaTile.SourcePath,
		MapFileKind.Adt,
		mcnkCount: terrain.ChunkCount,
		zeroLengthMcnkCount: 0,
		headerLikeMcnkCount: terrain.ChunkCount,
		distinctIndexCount: terrain.ChunkCount,
		duplicateIndexCount: 0,
		distinctAreaIdCount: terrain.DistinctAreaIdCount,
		chunksWithHoles: terrain.HoleChunkCount,
		chunksWithLiquidFlags: terrain.LiquidFlagChunkCount,
		chunksWithMccvFlag: 0,
		chunksWithMcvt: terrain.ChunksWithHeights,
		chunksWithMcnr: 0,
		chunksWithMcly: chunksWithMcly,
		chunksWithMcal: 0,
		chunksWithMcsh: 0,
		chunksWithMcse: 0,
		chunksWithMccv: 0,
		chunksWithMclq: chunksWithMclq,
		chunksWithMcrd: 0,
		chunksWithMcrw: 0,
		totalMcsePayloadBytes: 0,
		totalLayerCount: totalLayerCount,
		maxLayerCount: maxLayerCount,
		chunksWithMultipleLayers: chunksWithMultipleLayers,
		mccvFlagWithoutPayloadCount: 0,
		liquidFlagWithoutPayloadCount: Math.Max(0, terrain.LiquidFlagChunkCount - chunksWithMclq));
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
	return new AdtTextureFile(sourcePath, kind, decodeProfile, Array.Empty<string>(), Array.Empty<AdtTextureChunk>(), null);
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

static float? GetFloatOption(string[] args, string longName, string shortName)
{
	string? value = GetOption(args, longName, shortName);
	if (string.IsNullOrWhiteSpace(value))
		return null;

	return float.TryParse(value, out float parsed) ? parsed : null;
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

static V9TensorCacheManifestData ReadV9TensorCacheManifest(string path)
{
	using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
	JsonElement root = document.RootElement;

	if (!root.TryGetProperty("schema_version", out JsonElement schemaVersionElement)
		|| schemaVersionElement.ValueKind != JsonValueKind.String)
	{
		throw new InvalidDataException($"Manifest '{path}' is missing string property 'schema_version'.");
	}

	if (!root.TryGetProperty("entries", out JsonElement entriesElement)
		|| entriesElement.ValueKind != JsonValueKind.Array)
	{
		throw new InvalidDataException($"Manifest '{path}' is missing array property 'entries'.");
	}

	List<JsonElement> entries = new(entriesElement.GetArrayLength());
	foreach (JsonElement entry in entriesElement.EnumerateArray())
		entries.Add(entry.Clone());

	return new V9TensorCacheManifestData(schemaVersionElement.GetString() ?? string.Empty, entries);
}

static bool TryGetBooleanProperty(JsonElement element, string propertyName, out bool value)
{
	if (element.TryGetProperty(propertyName, out JsonElement property))
	{
		if (property.ValueKind == JsonValueKind.True)
		{
			value = true;
			return true;
		}

		if (property.ValueKind == JsonValueKind.False)
		{
			value = false;
			return true;
		}
	}

	value = false;
	return false;
}

static string BuildV9ManifestEntryKey(JsonElement entry)
{
	string datasetKey = entry.TryGetProperty("dataset_key", out JsonElement datasetKeyElement) && datasetKeyElement.ValueKind == JsonValueKind.String
		? datasetKeyElement.GetString() ?? string.Empty
		: string.Empty;
	string tileName = entry.TryGetProperty("tile_name", out JsonElement tileNameElement) && tileNameElement.ValueKind == JsonValueKind.String
		? tileNameElement.GetString() ?? string.Empty
		: string.Empty;
	string shardPath = entry.TryGetProperty("shard_path", out JsonElement shardPathElement) && shardPathElement.ValueKind == JsonValueKind.String
		? shardPathElement.GetString() ?? string.Empty
		: string.Empty;
	return string.Join("|", datasetKey, tileName, shardPath);
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
	Console.WriteLine("  wowviewer-converter dataset-list-maps --client-root <path> [--output <maps.json>]");
	Console.WriteLine("  wowviewer-converter dataset-scan --client-root <path> --map <name> [--build <label>] [--output <manifest.json>] [--limit <count>]");
	Console.WriteLine("  wowviewer-converter dataset-merge --input <manifest.json> [--input <manifest.json> ...] [--output <merged.json>] [manifest.json ...]");
	Console.WriteLine("  wowviewer-converter dataset-split-pm4 --direct-manifest <cache.json> --development-manifest <cache.json> --output-dir <dir> [--pm4-flag <field>]");
	Console.WriteLine("  wowviewer-converter dataset-audit --input <scan.json> [--output <audit.json>] [--limit <count>]");
	Console.WriteLine("  wowviewer-converter dataset-curate --input <audit.json> --output <curated.json> [--report <curation-report.json>] [--limit <count>] [--max-per-group <count>] [--require-wdl|--no-require-wdl] [--require-minimap|--no-require-minimap]");
	Console.WriteLine("  wowviewer-converter dataset-build-cache --input <audit-or-curate.json> --output-dir <dir> [--limit <count>] [--overwrite] [--include-minimap|--no-include-minimap] [--write-debug-json|--no-write-debug-json]");
	Console.WriteLine("  wowviewer-converter extract-v10-tensors --input <root.adt> [--output <npz>] [--texture-source <tex0.adt>] [--minimap-root <dir>]  (also writes matching *_placements.json when placement data exists)");
	Console.WriteLine("  wowviewer-converter dataset-build-v10-stage1 --input-dir <adt-dir> --output-dir <dir> --minimap-root <dir> [--manifest <manifest.json>] [--limit <count>] [--overwrite]");
	Console.WriteLine("  wowviewer-converter mine-v10-brushes --input-dir <npz-dir> --output-dir <dir> [--placement-dir <dir>] [--anchor-mode objects|terrain|hybrid] [--context-radius <n>] [--dictionary-size <n>] [--min-occurrences <n>] [--terrain-samples-per-tile <n>] [--seed <n>]");
	Console.WriteLine("  wowviewer-converter mine-v10-mcly --input-dir <npz-dir> --output-dir <dir> [--min-occurrences <n>] [--example-limit <n>] [--include-empty]");
	Console.WriteLine("  wowviewer-converter label-v10-mcly --input <stage1-manifest|npz-dir|npz> --dictionary <mclay_dictionary.json> --output <label-manifest.json> [--min-retained-chunks <n>]");
	Console.WriteLine("  wowviewer-converter mine-v10-mcal-compositions --input-dir <npz-dir> --output-dir <dir> [--dictionary-size <n>] [--min-occurrences <n>] [--min-active-layers <n>] [--min-layer-std <v>] [--min-gradient <v>] [--example-limit <n>]");
	Console.WriteLine("  wowviewer-converter mine-v10-mcal-brushes --input-dir <npz-dir> --output-dir <dir> [--dictionary-size <n>] [--min-occurrences <n>] [--min-layer-std <v>] [--min-gradient <v>] [--min-range <v>] [--max-samples <n>] [--seed <n>]");
	Console.WriteLine("  wowviewer-converter mine-v10-height-profiles --input-dir <npz-dir> --output-dir <dir> [--dictionary-size <n>] [--min-occurrences <n>] [--profile-size <n>] [--max-iterations <n>] [--example-limit <n>] [--seed <n>]");
	Console.WriteLine("  wowviewer-converter mine-v10-prefab-cells --input-dir <npz-dir> --output-dir <dir> [--cell-width <n>] [--cell-height <n>] [--dictionary-size <n>] [--min-occurrences <n>] [--example-limit <n>] [--height-quant <v>] [--no-mcly]");
	Console.WriteLine("  wowviewer-converter detect --input <file>");
	Console.WriteLine("  wowviewer-converter export-tex-json --input <file.adt|file_tex0.adt> [--output <report.json>]");
	Console.WriteLine("  wowviewer-converter terrain-patch-adt --input-adt-dir <dir> --inference-dir <dir> --output-dir <dir> [--no-copy-family] [--no-export-guide-textures] [--no-export-texture-supervision] [--export-glb] [--center-mesh] [--tile-world-size <size>] [--height-offset <value>]");
	Console.WriteLine("  wowviewer-converter ml-corpus --config <ml-corpus.json> [--archive-root <path>] [--output-root <path>] [--dry-run]");
	Console.WriteLine("  wowviewer-converter ml-audit-signals --dataset-root <path> [--output <report.json>] [--limit <count>]");
	Console.WriteLine("  wowviewer-converter ml-harvest-brushes --dataset-root <path> [--output-dir <dir>] [--limit <count>] [--write-previews]");
	Console.WriteLine("  wowviewer-converter ml-generate-controls [--dataset-root <path>] [--map-name <name>]");
	Console.WriteLine("  wowviewer-converter ml-repair-normalmaps --dataset-root <path> [--report <report.json>] [--limit <count>] [--rewrite-existing] [--rewrite-when-local-differs <mae>] [--only-liquid-tiles] [--dry-run]");
	Console.WriteLine("  wowviewer-converter ml-synth-no-liquid --input <minimap.png> --mask <liquid-mask.png> --output <no-liquid.png>");
	Console.WriteLine("  wowviewer-converter ml-synth-no-liquid --input-dir <images> --mask-dir <masks> --output-dir <images>");
	Console.WriteLine("  wowviewer-converter convert-alpha-to-lk --input <Azeroth.wdt> --output <output-dir> [--verbose|-v]");
	Console.WriteLine("  wowviewer-converter convert-lk-to-alpha --input <dir> --output <output.wdt> [--verbose|-v]");
	Console.WriteLine("  wowviewer-converter convert-lk-to-alpha --client-root <dir> --map <name> --output <output.wdt> [--limit <n>] [--verbose|-v]");
}

file sealed record V10TensorExtractionResult(
	TerrainTileTensorPack Pack,
	string? MinimapSourcePath);

file sealed record V10Stage1Manifest(
	string SchemaVersion,
	DateTimeOffset CreatedAtUtc,
	string InputRoot,
	string OutputRoot,
	string MinimapRoot,
	int ScannedTileCount,
	int WrittenTileCount,
	int PlaceholderTileCount,
	IReadOnlyList<V10Stage1ManifestEntry> Entries,
	IReadOnlyList<V10Stage1ManifestSkip> Skipped);

file sealed record V10Stage1ManifestEntry(
	string TileName,
	string SourceAdtPath,
	string ShardPath,
	string? PlacementPath,
	string MinimapSourcePath,
	string MinimapSourceTag,
	IReadOnlyList<string> AvailableSignals);

file sealed record V10Stage1ManifestSkip(
	string TileName,
	string SourceAdtPath,
	string Reason);

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

file sealed record V9TensorCacheManifestData(
	string SchemaVersion,
	List<JsonElement> Entries);

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

file sealed record DatasetCurateEvaluation(
	TerrainTrainingSampleDescriptor Entry,
	bool Accepted,
	string? RejectionReason,
	float QualityScore);

file sealed record DirectCacheBuildResult(
	string DatasetKey,
	string ShardPath,
	string DebugJsonPath,
	float HeightMin,
	float HeightMax,
	float LiquidCoverage,
	float ObjectCoverage,
	float BrushCoverage,
	float HoleCoverage,
	float MinimapVariance,
	float MinimapGradient,
	float DetailEnergy,
	bool HasWdl17,
	bool HasMinimap,
	bool HasNativeNormalMap,
	string MinimapSource,
	IReadOnlyList<string> ArrayNames);

file sealed record WdlAlignment(
	float[] AlignedHeights17,
	float VerticalOffset,
	float MeanAbsoluteDelta,
	float MaxAbsoluteDelta);

file sealed record NpyArray(string Descriptor, int[] Shape, byte[] Data)
{
	public static NpyArray FromFloat32(float[] values, params int[] shape)
	{
		byte[] bytes = new byte[values.Length * sizeof(float)];
		Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
		return new NpyArray("<f4", shape, bytes);
	}

	public static NpyArray FromInt32(int[] values, params int[] shape)
	{
		byte[] bytes = new byte[values.Length * sizeof(int)];
		Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
		return new NpyArray("<i4", shape, bytes);
	}

	public static NpyArray FromUInt8(byte[] values, params int[] shape)
	{
		return new NpyArray("|u1", shape, values);
	}
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
