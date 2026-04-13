using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

internal static class MlBrushImprintHarvester
{
	private const int TilePatchGridSize = 256;
	private const int TileVertexGridSize = 257;
	private const int ChunkCountPerRow = 16;
	private const int PatchCountPerChunk = 16;
	private const string ManifestFileName = "brush_imprint_manifest.json";
	private const string GroupDirectoryName = "groups";
	private const string TileMaskDirectoryName = "tile_masks";

	public static void Run(string[] args)
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

		int? limit = GetIntOption(args, "--limit", "-n");
		bool writePreviews = HasFlag(args, "--write-previews");
		string outputDirectory = Path.GetFullPath(
			GetOption(args, "--output-dir", "-o")
			?? Path.Combine(datasetRoot, "brush_imprints"));

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

		Directory.CreateDirectory(outputDirectory);
		string groupsDirectory = Path.Combine(outputDirectory, "groups");
		Directory.CreateDirectory(groupsDirectory);
		string tileMasksDirectory = Path.Combine(outputDirectory, TileMaskDirectoryName);
		Directory.CreateDirectory(tileMasksDirectory);
		string previewsDirectory = Path.Combine(outputDirectory, "previews");
		if (writePreviews)
			Directory.CreateDirectory(previewsDirectory);

		Console.WriteLine("WowViewer.Tool.Converter ml-harvest-brushes report");
		Console.WriteLine($"DatasetRoot: {datasetRoot}");
		Console.WriteLine($"DatasetDirectory: {datasetDirectory}");
		Console.WriteLine($"OutputDirectory: {outputDirectory}");
		Console.WriteLine($"TileJsonCount: {datasetFiles.Count}");
		Console.WriteLine($"WritePreviews: {writePreviews}");

		MlBrushIssueTracker issueTracker = new();
		List<MlBrushTileSummary> tileSummaries = new(datasetFiles.Count);
		List<string> groupFiles = [];
		int tilesSkippedMissingHeightmap = 0;
		int groupsWritten = 0;
		int patchesWritten = 0;

		foreach (string datasetFile in datasetFiles)
		{
			try
			{
				MlBrushDatasetSample sample = JsonSerializer.Deserialize<MlBrushDatasetSample>(
					File.ReadAllText(datasetFile),
					new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
					?? throw new InvalidDataException($"Failed to parse dataset JSON '{datasetFile}'.");

				if (sample.TerrainData is null)
					throw new InvalidDataException($"Dataset tile '{datasetFile}' is missing terrain_data.");

				string tileName = string.IsNullOrWhiteSpace(sample.TerrainData.AdtTile)
					? Path.GetFileNameWithoutExtension(datasetFile)
					: sample.TerrainData.AdtTile;
				string mapName = ExtractMapName(tileName);

				string? heightmapPath = ResolveDatasetPath(datasetRoot, sample.TerrainData.HeightmapGlobalPath ?? sample.TerrainData.HeightmapPath);
				if (heightmapPath is null)
				{
					tilesSkippedMissingHeightmap++;
					tileSummaries.Add(new MlBrushTileSummary(
						TileName: tileName,
						MapName: mapName,
						BrushMaskPath: null,
						PatchCandidates: 0,
						GroupsWritten: 0,
						SkippedReason: "missing-heightmap-global"));
					continue;
				}

				float[] heightmap = LoadHeightmapL16(heightmapPath);
				string[] chunkTextureSignatures = BuildChunkTextureSignatures(sample.TerrainData.ChunkLayers);
				MlBrushPatchCell[] patchCells = BuildPatchCells(heightmap, chunkTextureSignatures);
				MlBrushPatchCell[] activeCells = SelectActivePatchCells(patchCells);

				List<MlBrushGroupCandidate> groups = BuildGroups(tileName, mapName, heightmap, activeCells);
				string? brushMaskRelativePath = null;
				if (groups.Count > 0)
				{
					string tileMaskPath = Path.Combine(tileMasksDirectory, tileName + "_brush_mask.png");
					WriteTileGroupMask(groups, tileMaskPath);
					brushMaskRelativePath = Path.GetRelativePath(outputDirectory, tileMaskPath).Replace('\\', '/');
				}

				for (int index = 0; index < groups.Count; index++)
				{
					MlBrushGroupCandidate group = groups[index];
					string groupId = $"{tileName}_g{(index + 1).ToString("D4", CultureInfo.InvariantCulture)}";
					MlBrushGroupReport report = group.ToReport(groupId, datasetRoot, sample, heightmapPath);
					string groupPath = Path.Combine(groupsDirectory, groupId + ".json");
					File.WriteAllText(groupPath, JsonSerializer.Serialize(report, CreateBrushJsonOptions()));
					groupFiles.Add(Path.GetRelativePath(outputDirectory, groupPath).Replace('\\', '/'));
					groupsWritten++;
					patchesWritten += report.PatchCount;

					if (writePreviews)
					{
						string previewPath = Path.Combine(previewsDirectory, groupId + "_mask.png");
						WritePreviewMask(report, previewPath);
					}
				}

				tileSummaries.Add(new MlBrushTileSummary(
					TileName: tileName,
					MapName: mapName,
					BrushMaskPath: brushMaskRelativePath,
					PatchCandidates: activeCells.Length,
					GroupsWritten: groups.Count,
					SkippedReason: null));
			}
			catch (Exception ex)
			{
				issueTracker.Record(datasetFile, ex);
			}
		}

		MlBrushHarvestManifest manifest = new(
			SchemaVersion: "wowviewer-ml-brush-imprint.v1",
			GeneratedUtc: DateTime.UtcNow,
			DatasetRoot: datasetRoot,
			OutputDirectory: outputDirectory,
			TilesSeen: datasetFiles.Count,
			TilesProcessed: tileSummaries.Count(static tile => tile.SkippedReason is null),
			TilesSkippedMissingHeightmap: tilesSkippedMissingHeightmap,
			GroupsWritten: groupsWritten,
			PatchesWritten: patchesWritten,
			GroupFiles: groupFiles,
			Tiles: tileSummaries);

		string manifestPath = Path.Combine(outputDirectory, "brush_imprint_manifest.json");
		File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, CreateBrushJsonOptions()));

		issueTracker.Print();
		Console.WriteLine($"Brush harvest complete: processed_tiles={manifest.TilesProcessed} skipped_missing_heightmap={tilesSkippedMissingHeightmap} groups={groupsWritten} patches={patchesWritten}");
		Console.WriteLine($"Wrote {manifestPath}");
	}

	private static float[] LoadHeightmapL16(string path)
	{
		using Image<L16> image = Image.Load<L16>(path);
		if (image.Width != TileVertexGridSize || image.Height != TileVertexGridSize)
			image.Mutate(ctx => ctx.Resize(TileVertexGridSize, TileVertexGridSize, KnownResamplers.Lanczos3));

		float[] values = new float[TileVertexGridSize * TileVertexGridSize];
		for (int y = 0; y < image.Height; y++)
		{
			for (int x = 0; x < image.Width; x++)
				values[(y * image.Width) + x] = image[x, y].PackedValue / 65535f;
		}

		return values;
	}

	private static string[] BuildChunkTextureSignatures(MlBrushChunkLayers[]? chunks)
	{
		string[] signatures = Enumerable.Repeat("none", 256).ToArray();
		if (chunks is null)
			return signatures;

		foreach (MlBrushChunkLayers chunk in chunks)
		{
			if (chunk.ChunkIndex < 0 || chunk.ChunkIndex >= signatures.Length)
				continue;

			string signature = string.Join("|",
				chunk.Layers
					.Where(static layer => !string.IsNullOrWhiteSpace(layer.TexturePath))
					.Select(static layer => layer.TexturePath!.Trim().ToLowerInvariant()));

			signatures[chunk.ChunkIndex] = string.IsNullOrWhiteSpace(signature) ? "none" : signature;
		}

		return signatures;
	}

	private static MlBrushPatchCell[] BuildPatchCells(float[] heightmap, string[] chunkTextureSignatures)
	{
		MlBrushPatchCell[] patches = new MlBrushPatchCell[TilePatchGridSize * TilePatchGridSize];
		int index = 0;
		for (int patchY = 0; patchY < TilePatchGridSize; patchY++)
		{
			for (int patchX = 0; patchX < TilePatchGridSize; patchX++)
			{
				float h00 = heightmap[(patchY * TileVertexGridSize) + patchX];
				float h10 = heightmap[(patchY * TileVertexGridSize) + patchX + 1];
				float h01 = heightmap[((patchY + 1) * TileVertexGridSize) + patchX];
				float h11 = heightmap[((patchY + 1) * TileVertexGridSize) + patchX + 1];

				float min = MathF.Min(MathF.Min(h00, h10), MathF.Min(h01, h11));
				float max = MathF.Max(MathF.Max(h00, h10), MathF.Max(h01, h11));
				float relief = max - min;
				float dx = ((h10 + h11) - (h00 + h01)) * 0.5f;
				float dy = ((h01 + h11) - (h00 + h10)) * 0.5f;
				float slope = MathF.Sqrt((dx * dx) + (dy * dy));
				float diagonal = MathF.Abs((h00 + h11) - (h10 + h01));
				float score = relief + (slope * 0.5f) + (diagonal * 0.25f);

				int chunkX = patchX / PatchCountPerChunk;
				int chunkY = patchY / PatchCountPerChunk;
				int chunkIndex = (chunkY * ChunkCountPerRow) + chunkX;
				patches[index++] = new MlBrushPatchCell(
					PatchX: patchX,
					PatchY: patchY,
					ChunkIndex: chunkIndex,
					LocalPatchX: patchX % PatchCountPerChunk,
					LocalPatchY: patchY % PatchCountPerChunk,
					TextureSignature: chunkTextureSignatures[chunkIndex],
					MinHeight: min,
					MaxHeight: max,
					Relief: relief,
					Slope: slope,
					Diagonal: diagonal,
					Score: score);
			}
		}

		return patches;
	}

	private static MlBrushPatchCell[] SelectActivePatchCells(MlBrushPatchCell[] patchCells)
	{
		float[] positiveScores = patchCells
			.Select(static patch => patch.Score)
			.Where(static score => score > 0f)
			.OrderBy(static score => score)
			.ToArray();

		if (positiveScores.Length == 0)
			return [];

		int percentileIndex = (int)MathF.Floor((positiveScores.Length - 1) * 0.85f);
		float threshold = positiveScores[Math.Clamp(percentileIndex, 0, positiveScores.Length - 1)];
		threshold = MathF.Max(threshold, 0.0005f);

		return patchCells
			.Where(patch => patch.Score >= threshold)
			.ToArray();
	}

	private static List<MlBrushGroupCandidate> BuildGroups(string tileName, string mapName, float[] heightmap, MlBrushPatchCell[] activeCells)
	{
		Dictionary<(int X, int Y), MlBrushPatchCell> lookup = activeCells.ToDictionary(static patch => (patch.PatchX, patch.PatchY));
		HashSet<(int X, int Y)> visited = [];
		List<MlBrushGroupCandidate> groups = [];
		(int DX, int DY)[] neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1)];

		foreach (MlBrushPatchCell seed in activeCells.OrderByDescending(static patch => patch.Score))
		{
			if (!visited.Add((seed.PatchX, seed.PatchY)))
				continue;

			Queue<MlBrushPatchCell> queue = new();
			queue.Enqueue(seed);
			List<MlBrushPatchCell> groupPatches = [];

			while (queue.Count > 0)
			{
				MlBrushPatchCell current = queue.Dequeue();
				groupPatches.Add(current);

				foreach ((int dx, int dy) in neighbours)
				{
					(int nx, int ny) = (current.PatchX + dx, current.PatchY + dy);
					if (nx < 0 || ny < 0 || nx >= TilePatchGridSize || ny >= TilePatchGridSize)
						continue;

					if (!lookup.TryGetValue((nx, ny), out MlBrushPatchCell next))
						continue;

					if (visited.Add((nx, ny)))
						queue.Enqueue(next);
				}
			}

			if (groupPatches.Count < 8)
				continue;

			groups.Add(MlBrushGroupCandidate.Create(tileName, mapName, heightmap, groupPatches));
		}

		return groups;
	}

	private static void WritePreviewMask(MlBrushGroupReport group, string previewPath)
	{
		using Image<L8> image = new(group.PatchWidth, group.PatchHeight);
		foreach (MlBrushPatchPoint patch in group.Patches)
			image[patch.X, patch.Y] = new L8(255);
		image.SaveAsPng(previewPath);
	}

	private static void WriteTileGroupMask(IReadOnlyList<MlBrushGroupCandidate> groups, string outputPath)
	{
		using Image<L8> image = new(TilePatchGridSize, TilePatchGridSize);
		foreach (MlBrushGroupCandidate group in groups)
		{
			foreach (MlBrushPatchCell patch in group.Patches)
				image[patch.PatchX, patch.PatchY] = new L8(255);
		}
		image.SaveAsPng(outputPath);
	}

	private static JsonSerializerOptions CreateBrushJsonOptions()
	{
		JsonSerializerOptions options = new()
		{
			WriteIndented = true,
			DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
		};
		options.Converters.Add(new JsonStringEnumConverter());
		return options;
	}

	private static string? ResolveDatasetPath(string datasetRoot, string? relativePath)
	{
		if (string.IsNullOrWhiteSpace(relativePath))
			return null;

		string normalized = relativePath.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
		string candidate = Path.IsPathRooted(normalized)
			? normalized
			: Path.Combine(datasetRoot, normalized);
		return File.Exists(candidate) ? candidate : null;
	}

	private static string ExtractMapName(string tileName)
	{
		int lastSeparator = tileName.LastIndexOf('_');
		if (lastSeparator <= 0)
			return tileName;

		int secondLastSeparator = tileName.LastIndexOf('_', lastSeparator - 1);
		if (secondLastSeparator <= 0)
			return tileName;

		return tileName[..secondLastSeparator];
	}

	private static bool HasFlag(IEnumerable<string> args, string name)
	{
		return args.Any(arg => string.Equals(arg, name, StringComparison.OrdinalIgnoreCase));
	}

	private static string? GetOption(string[] args, string longName, string shortName)
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

	private static int? GetIntOption(string[] args, string longName, string shortName)
	{
		string? value = GetOption(args, longName, shortName);
		if (string.IsNullOrWhiteSpace(value))
			return null;

		return int.TryParse(value, out int parsed) ? parsed : null;
	}
}

internal sealed class MlBrushDatasetSample
{
	[JsonPropertyName("image")]
	public string? ImagePath { get; set; }

	[JsonPropertyName("terrain_data")]
	public MlBrushTerrainData? TerrainData { get; set; }
}

internal sealed class MlBrushTerrainData
{
	[JsonPropertyName("adt_tile")]
	public string? AdtTile { get; set; }

	[JsonPropertyName("heightmap")]
	public string? HeightmapPath { get; set; }

	[JsonPropertyName("heightmap_global")]
	public string? HeightmapGlobalPath { get; set; }

	[JsonPropertyName("chunk_layers")]
	public MlBrushChunkLayers[]? ChunkLayers { get; set; }
}

internal sealed class MlBrushChunkLayers
{
	[JsonPropertyName("idx")]
	public int ChunkIndex { get; set; }

	[JsonPropertyName("layers")]
	public MlBrushTextureLayer[] Layers { get; set; } = [];
}

internal sealed class MlBrushTextureLayer
{
	[JsonPropertyName("texture_path")]
	public string? TexturePath { get; set; }
}

internal sealed record MlBrushPatchCell(
	int PatchX,
	int PatchY,
	int ChunkIndex,
	int LocalPatchX,
	int LocalPatchY,
	string TextureSignature,
	float MinHeight,
	float MaxHeight,
	float Relief,
	float Slope,
	float Diagonal,
	float Score);

internal sealed class MlBrushGroupCandidate
{
	private MlBrushGroupCandidate(
		string tileName,
		string mapName,
		int minPatchX,
		int minPatchY,
		int maxPatchX,
		int maxPatchY,
		List<MlBrushPatchCell> patches,
		float[] normalizedHeightGrid,
		int heightGridWidth,
		int heightGridHeight,
		List<string> textureSignatures)
	{
		TileName = tileName;
		MapName = mapName;
		MinPatchX = minPatchX;
		MinPatchY = minPatchY;
		MaxPatchX = maxPatchX;
		MaxPatchY = maxPatchY;
		Patches = patches;
		NormalizedHeightGrid = normalizedHeightGrid;
		HeightGridWidth = heightGridWidth;
		HeightGridHeight = heightGridHeight;
		TextureSignatures = textureSignatures;
	}

	public string TileName { get; }
	public string MapName { get; }
	public int MinPatchX { get; }
	public int MinPatchY { get; }
	public int MaxPatchX { get; }
	public int MaxPatchY { get; }
	public List<MlBrushPatchCell> Patches { get; }
	public float[] NormalizedHeightGrid { get; }
	public int HeightGridWidth { get; }
	public int HeightGridHeight { get; }
	public List<string> TextureSignatures { get; }

	public static MlBrushGroupCandidate Create(string tileName, string mapName, float[] heightmap, List<MlBrushPatchCell> patches)
	{
		int minPatchX = patches.Min(static patch => patch.PatchX);
		int minPatchY = patches.Min(static patch => patch.PatchY);
		int maxPatchX = patches.Max(static patch => patch.PatchX);
		int maxPatchY = patches.Max(static patch => patch.PatchY);

		int heightGridWidth = (maxPatchX - minPatchX) + 2;
		int heightGridHeight = (maxPatchY - minPatchY) + 2;
		float[] rawHeights = new float[heightGridWidth * heightGridHeight];
		float min = float.MaxValue;
		float max = float.MinValue;
		for (int y = minPatchY; y <= maxPatchY + 1; y++)
		{
			for (int x = minPatchX; x <= maxPatchX + 1; x++)
			{
				float height = heightmap[(y * 257) + x];
				rawHeights[((y - minPatchY) * heightGridWidth) + (x - minPatchX)] = height;
				if (height < min)
					min = height;
				if (height > max)
					max = height;
			}
		}

		float range = MathF.Max(max - min, 1e-6f);
		float[] normalized = rawHeights
			.Select(value => MathF.Round((value - min) / range, 6))
			.ToArray();

		List<string> textureSignatures = patches
			.Select(static patch => patch.TextureSignature)
			.Where(static signature => !string.Equals(signature, "none", StringComparison.Ordinal))
			.Distinct(StringComparer.Ordinal)
			.OrderBy(static signature => signature, StringComparer.Ordinal)
			.ToList();

		return new MlBrushGroupCandidate(tileName, mapName, minPatchX, minPatchY, maxPatchX, maxPatchY, patches, normalized, heightGridWidth, heightGridHeight, textureSignatures);
	}

	public MlBrushGroupReport ToReport(string groupId, string datasetRoot, MlBrushDatasetSample sample, string heightmapPath)
	{
		float meanScore = Patches.Count == 0 ? 0f : Patches.Average(static patch => patch.Score);
		float maxScore = Patches.Count == 0 ? 0f : Patches.Max(static patch => patch.Score);
		int minChunkIndex = Patches.Min(static patch => patch.ChunkIndex);
		int maxChunkIndex = Patches.Max(static patch => patch.ChunkIndex);

		List<MlBrushPatchPoint> patchPoints = Patches
			.OrderBy(static patch => patch.PatchY)
			.ThenBy(static patch => patch.PatchX)
			.Select(patch => new MlBrushPatchPoint(
				X: patch.PatchX - MinPatchX,
				Y: patch.PatchY - MinPatchY,
				ChunkIndex: patch.ChunkIndex,
				LocalPatchX: patch.LocalPatchX,
				LocalPatchY: patch.LocalPatchY,
				Score: MathF.Round(patch.Score, 6),
				Relief: MathF.Round(patch.Relief, 6),
				Slope: MathF.Round(patch.Slope, 6)))
			.ToList();

		return new MlBrushGroupReport(
			SchemaVersion: "wowviewer-ml-brush-group.v1",
			GroupId: groupId,
			DatasetRoot: datasetRoot,
			TileName: TileName,
			MapName: MapName,
			SourceImagePath: sample.ImagePath,
			HeightmapGlobalPath: Path.GetRelativePath(datasetRoot, heightmapPath).Replace('\\', '/'),
			PatchMinX: MinPatchX,
			PatchMinY: MinPatchY,
			PatchMaxX: MaxPatchX,
			PatchMaxY: MaxPatchY,
			PatchWidth: (MaxPatchX - MinPatchX) + 1,
			PatchHeight: (MaxPatchY - MinPatchY) + 1,
			PatchCount: patchPoints.Count,
			ChunkMinIndex: minChunkIndex,
			ChunkMaxIndex: maxChunkIndex,
			MeanScore: MathF.Round(meanScore, 6),
			MaxScore: MathF.Round(maxScore, 6),
			TextureSignatures: TextureSignatures,
			HeightGridWidth: HeightGridWidth,
			HeightGridHeight: HeightGridHeight,
			NormalizedHeightGrid: NormalizedHeightGrid,
			Patches: patchPoints);
	}
}

internal sealed record MlBrushPatchPoint(
	[property: JsonPropertyName("x")] int X,
	[property: JsonPropertyName("y")] int Y,
	[property: JsonPropertyName("chunk_index")] int ChunkIndex,
	[property: JsonPropertyName("local_patch_x")] int LocalPatchX,
	[property: JsonPropertyName("local_patch_y")] int LocalPatchY,
	[property: JsonPropertyName("score")] float Score,
	[property: JsonPropertyName("relief")] float Relief,
	[property: JsonPropertyName("slope")] float Slope);

internal sealed record MlBrushGroupReport(
	[property: JsonPropertyName("schema_version")] string SchemaVersion,
	[property: JsonPropertyName("group_id")] string GroupId,
	[property: JsonPropertyName("dataset_root")] string DatasetRoot,
	[property: JsonPropertyName("tile_name")] string TileName,
	[property: JsonPropertyName("map_name")] string MapName,
	[property: JsonPropertyName("source_image_path")] string? SourceImagePath,
	[property: JsonPropertyName("heightmap_global_path")] string HeightmapGlobalPath,
	[property: JsonPropertyName("patch_min_x")] int PatchMinX,
	[property: JsonPropertyName("patch_min_y")] int PatchMinY,
	[property: JsonPropertyName("patch_max_x")] int PatchMaxX,
	[property: JsonPropertyName("patch_max_y")] int PatchMaxY,
	[property: JsonPropertyName("patch_width")] int PatchWidth,
	[property: JsonPropertyName("patch_height")] int PatchHeight,
	[property: JsonPropertyName("patch_count")] int PatchCount,
	[property: JsonPropertyName("chunk_min_index")] int ChunkMinIndex,
	[property: JsonPropertyName("chunk_max_index")] int ChunkMaxIndex,
	[property: JsonPropertyName("mean_score")] float MeanScore,
	[property: JsonPropertyName("max_score")] float MaxScore,
	[property: JsonPropertyName("texture_signatures")] List<string> TextureSignatures,
	[property: JsonPropertyName("height_grid_width")] int HeightGridWidth,
	[property: JsonPropertyName("height_grid_height")] int HeightGridHeight,
	[property: JsonPropertyName("normalized_height_grid")] float[] NormalizedHeightGrid,
	[property: JsonPropertyName("patches")] List<MlBrushPatchPoint> Patches);

internal sealed record MlBrushTileSummary(
	[property: JsonPropertyName("tile_name")] string TileName,
	[property: JsonPropertyName("map_name")] string MapName,
	[property: JsonPropertyName("brush_mask_path")] string? BrushMaskPath,
	[property: JsonPropertyName("patch_candidates")] int PatchCandidates,
	[property: JsonPropertyName("groups_written")] int GroupsWritten,
	[property: JsonPropertyName("skipped_reason")] string? SkippedReason);

internal sealed record MlBrushHarvestManifest(
	[property: JsonPropertyName("schema_version")] string SchemaVersion,
	[property: JsonPropertyName("generated_utc")] DateTime GeneratedUtc,
	[property: JsonPropertyName("dataset_root")] string DatasetRoot,
	[property: JsonPropertyName("output_directory")] string OutputDirectory,
	[property: JsonPropertyName("tiles_seen")] int TilesSeen,
	[property: JsonPropertyName("tiles_processed")] int TilesProcessed,
	[property: JsonPropertyName("tiles_skipped_missing_heightmap")] int TilesSkippedMissingHeightmap,
	[property: JsonPropertyName("groups_written")] int GroupsWritten,
	[property: JsonPropertyName("patches_written")] int PatchesWritten,
	[property: JsonPropertyName("group_files")] List<string> GroupFiles,
	[property: JsonPropertyName("tiles")] List<MlBrushTileSummary> Tiles);

internal sealed class MlBrushIssueTracker
{
	private readonly Dictionary<string, MlBrushIssueSummary> _issues = new(StringComparer.Ordinal);

	public void Record(string samplePath, Exception exception)
	{
		string key = exception.GetType().Name + "|" + exception.Message;
		if (_issues.TryGetValue(key, out MlBrushIssueSummary? existing))
		{
			existing.Count++;
			return;
		}

		_issues[key] = new MlBrushIssueSummary(samplePath, exception.GetType().Name, exception.Message);
	}

	public void Print()
	{
		foreach (MlBrushIssueSummary issue in _issues.Values.OrderByDescending(static issue => issue.Count))
			Console.Error.WriteLine($"Warning: brush harvest failures={issue.Count}; sample={issue.SamplePath}; {issue.ExceptionType}: {issue.Message}");
	}
}

internal sealed class MlBrushIssueSummary
{
	public MlBrushIssueSummary(string samplePath, string exceptionType, string message)
	{
		SamplePath = samplePath;
		ExceptionType = exceptionType;
		Message = message;
		Count = 1;
	}

	public string SamplePath { get; }
	public string ExceptionType { get; }
	public string Message { get; }
	public int Count { get; set; }
}
