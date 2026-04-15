using System.Globalization;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

internal static class MlRepairNormalmapsCommand
{
	private const int DefaultNormalmapSize = 256;
	private const float TileSize = 533.33333f;

	public static void Run(string[] args)
	{
		string? datasetRootArg = GetOption(args, "--dataset-root", "-d");
		if (string.IsNullOrWhiteSpace(datasetRootArg))
		{
			Console.Error.WriteLine("Error: --dataset-root <path> is required.");
			Environment.ExitCode = 1;
			return;
		}

		string datasetRoot = Path.GetFullPath(datasetRootArg);
		string datasetDirectory = Path.Combine(datasetRoot, "dataset");
		if (!Directory.Exists(datasetDirectory))
		{
			Console.Error.WriteLine($"Error: dataset directory not found: {datasetDirectory}");
			Environment.ExitCode = 1;
			return;
		}

		string reportPath = Path.GetFullPath(GetOption(args, "--report", "-o") ?? Path.Combine(datasetRoot, "normalmap_repair_report.json"));
		int? limit = ParseOptionalInt(GetOption(args, "--limit", "-l"));
		bool rewriteExisting = HasFlag(args, "--rewrite-existing");
		bool onlyLiquidTiles = HasFlag(args, "--only-liquid-tiles");
		bool dryRun = HasFlag(args, "--dry-run");
		double rewriteWhenLocalDiffers = ParseOptionalDouble(GetOption(args, "--rewrite-when-local-differs", "-m")) ?? 0d;

		string[] jsonFiles = Directory.GetFiles(datasetDirectory, "*.json", SearchOption.TopDirectoryOnly)
			.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase)
			.ToArray();

		if (limit is int sampleLimit && sampleLimit >= 0)
			jsonFiles = jsonFiles.Take(sampleLimit).ToArray();

		List<MlNormalmapRepairTileReport> tileReports = new(jsonFiles.Length);
		int repairedCount = 0;
		int skippedLiquidFilter = 0;
		int skippedMissingHeightmap = 0;

		Console.WriteLine("WowViewer.Tool.Converter ml-repair-normalmaps report");
		Console.WriteLine($"DatasetRoot: {datasetRoot}");
		Console.WriteLine($"DatasetDirectory: {datasetDirectory}");
		Console.WriteLine($"TileJsonCount: {jsonFiles.Length}");
		Console.WriteLine($"RewriteExisting: {rewriteExisting}");
		Console.WriteLine($"RewriteWhenLocalDiffers: {rewriteWhenLocalDiffers.ToString("0.###", CultureInfo.InvariantCulture)}");
		Console.WriteLine($"OnlyLiquidTiles: {onlyLiquidTiles}");
		Console.WriteLine($"DryRun: {dryRun}");

		foreach (string jsonPath in jsonFiles)
		{
			JsonNode? rootNode = JsonNode.Parse(File.ReadAllText(jsonPath));
			JsonObject? rootObject = rootNode as JsonObject;
			JsonObject? terrainData = rootObject?["terrain_data"] as JsonObject;
			if (rootObject is null || terrainData is null)
				continue;

			string tileName = terrainData["adt_tile"]?.GetValue<string>() ?? Path.GetFileNameWithoutExtension(jsonPath);
			string? normalmapRelativePath = ReadOptionalString(terrainData, "normalmap");
			string? localHeightmapRelativePath = ReadOptionalString(terrainData, "heightmap_local") ?? ReadOptionalString(terrainData, "heightmap");
			string? globalHeightmapRelativePath = ReadOptionalString(terrainData, "heightmap_global") ?? ReadOptionalString(terrainData, "heightmap");
			string? liquidMaskRelativePath = ReadOptionalString(terrainData, "liquid_mask");

			string? normalmapAbsolutePath = ResolveDatasetPath(datasetRoot, normalmapRelativePath);
			string? localHeightmapAbsolutePath = ResolveDatasetPath(datasetRoot, localHeightmapRelativePath);
			string? globalHeightmapAbsolutePath = ResolveDatasetPath(datasetRoot, globalHeightmapRelativePath);
			string? liquidMaskAbsolutePath = ResolveDatasetPath(datasetRoot, liquidMaskRelativePath);

			bool liquidMaskExists = liquidMaskAbsolutePath is not null && File.Exists(liquidMaskAbsolutePath);
			if (onlyLiquidTiles && !liquidMaskExists)
			{
				skippedLiquidFilter++;
				continue;
			}

			bool normalmapReferenceExists = !string.IsNullOrWhiteSpace(normalmapRelativePath);
			bool normalmapFileExists = normalmapAbsolutePath is not null && File.Exists(normalmapAbsolutePath);

			HeightmapSurface? localSurface = TryLoadHeightmapSurface(localHeightmapAbsolutePath, terrainData, preferLocalRange: true);
			HeightmapSurface? globalSurface = TryLoadHeightmapSurface(globalHeightmapAbsolutePath, terrainData, preferLocalRange: false);

			if (localSurface is null && globalSurface is null)
			{
				skippedMissingHeightmap++;
				tileReports.Add(new MlNormalmapRepairTileReport(
					TileName: tileName,
					NormalmapPath: normalmapRelativePath,
					NormalmapReferenceExists: normalmapReferenceExists,
					NormalmapFileExists: normalmapFileExists,
					LiquidMaskExists: liquidMaskExists,
					UsedHeightmap: null,
					RepairReason: "missing_heightmap",
					Repaired: false,
					OutputNormalmapPath: null,
					LocalGlobalMeanAbsoluteDelta: null,
					LocalGlobalMaxAbsoluteDelta: null));
				continue;
			}

			double? localGlobalMae = null;
			double? localGlobalMax = null;
			if (localSurface is not null && globalSurface is not null)
				(localGlobalMae, localGlobalMax) = CompareHeightmaps(localSurface, globalSurface);

			bool needsRepair = !normalmapReferenceExists || !normalmapFileExists;
			string repairReason = !normalmapReferenceExists
				? "missing_normalmap_reference"
				: (!normalmapFileExists ? "missing_normalmap_file" : "unchanged");

			if (!needsRepair && rewriteExisting)
			{
				needsRepair = true;
				repairReason = "rewrite_existing";
			}
			else if (!needsRepair && rewriteWhenLocalDiffers > 0d && localGlobalMae.HasValue && localGlobalMae.Value >= rewriteWhenLocalDiffers && localSurface is not null)
			{
				needsRepair = true;
				repairReason = "rewrite_local_global_divergence";
			}

			HeightmapSurface sourceSurface = localSurface ?? globalSurface!;
			string usedHeightmap = localSurface is not null ? "heightmap_local" : "heightmap_global";

			string outputRelativePath = !string.IsNullOrWhiteSpace(normalmapRelativePath)
				? NormalizeRelativePath(normalmapRelativePath)
				: $"images/{tileName}_normal.png";
			string outputAbsolutePath = Path.Combine(datasetRoot, outputRelativePath.Replace('/', Path.DirectorySeparatorChar));

			if (needsRepair)
			{
				repairedCount++;
				if (!dryRun)
				{
					Directory.CreateDirectory(Path.GetDirectoryName(outputAbsolutePath)!);
					using Image<Rgba32> normalmapImage = CreateNormalmapFromHeightmap(sourceSurface);
					normalmapImage.SaveAsPng(outputAbsolutePath);

					terrainData["normalmap"] = outputRelativePath;
					terrainData["normalmap_generated_from"] = usedHeightmap;
					terrainData["normalmap_generated_reason"] = repairReason;
					File.WriteAllText(jsonPath, rootObject.ToJsonString(CreateJsonOptions()));
				}
			}

			tileReports.Add(new MlNormalmapRepairTileReport(
				TileName: tileName,
				NormalmapPath: normalmapRelativePath,
				NormalmapReferenceExists: normalmapReferenceExists,
				NormalmapFileExists: normalmapFileExists,
				LiquidMaskExists: liquidMaskExists,
				UsedHeightmap: needsRepair ? usedHeightmap : null,
				RepairReason: repairReason,
				Repaired: needsRepair,
				OutputNormalmapPath: needsRepair ? outputRelativePath : null,
				LocalGlobalMeanAbsoluteDelta: localGlobalMae,
				LocalGlobalMaxAbsoluteDelta: localGlobalMax));
		}

		MlNormalmapRepairReport report = new(
			SchemaVersion: "wowviewer-ml-normalmap-repair.v1",
			GeneratedUtc: DateTime.UtcNow,
			DatasetRoot: datasetRoot,
			DatasetDirectory: datasetDirectory,
			TileJsonCount: jsonFiles.Length,
			TilesRepaired: repairedCount,
			TilesSkippedLiquidFilter: skippedLiquidFilter,
			TilesSkippedMissingHeightmap: skippedMissingHeightmap,
			DryRun: dryRun,
			RewriteExisting: rewriteExisting,
			RewriteWhenLocalDiffers: rewriteWhenLocalDiffers,
			OnlyLiquidTiles: onlyLiquidTiles,
			Tiles: tileReports);

		Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
		File.WriteAllText(reportPath, JsonSerializer.Serialize(report, CreateJsonOptions()));

		Console.WriteLine($"Normalmap repair complete: repaired={repairedCount} skipped_missing_heightmap={skippedMissingHeightmap} skipped_liquid_filter={skippedLiquidFilter}");
		Console.WriteLine($"Wrote {reportPath}");
	}

	private static HeightmapSurface? TryLoadHeightmapSurface(string? path, JsonObject terrainData, bool preferLocalRange)
	{
		if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
			return null;

		float minHeight = preferLocalRange
			? ReadOptionalSingle(terrainData, "height_min") ?? 0f
			: ReadOptionalSingle(terrainData, "height_global_min") ?? (ReadOptionalSingle(terrainData, "height_min") ?? 0f);
		float maxHeight = preferLocalRange
			? ReadOptionalSingle(terrainData, "height_max") ?? 1f
			: ReadOptionalSingle(terrainData, "height_global_max") ?? (ReadOptionalSingle(terrainData, "height_max") ?? 1f);

		if (maxHeight - minHeight <= 1e-6f)
			maxHeight = minHeight + 1f;

		using Image<L16> image = Image.Load<L16>(path);
		float[] heights = new float[image.Width * image.Height];
		image.ProcessPixelRows(accessor =>
		{
			for (int y = 0; y < image.Height; y++)
			{
				Span<L16> row = accessor.GetRowSpan(y);
				int baseIndex = y * image.Width;
				for (int x = 0; x < image.Width; x++)
				{
					float normalized = row[x].PackedValue / 65535f;
					heights[baseIndex + x] = minHeight + (normalized * (maxHeight - minHeight));
				}
			}
		});

		return new HeightmapSurface(path, image.Width, image.Height, heights);
	}

	private static Image<Rgba32> CreateNormalmapFromHeightmap(HeightmapSurface surface)
	{
		Vector3[] normals = new Vector3[surface.Heights.Length];
		float horizontalSpacing = TileSize / Math.Max(surface.Width - 1, 1);

		for (int y = 0; y < surface.Height; y++)
		{
			for (int x = 0; x < surface.Width; x++)
			{
				float left = surface.Heights[(y * surface.Width) + Math.Max(x - 1, 0)];
				float right = surface.Heights[(y * surface.Width) + Math.Min(x + 1, surface.Width - 1)];
				float up = surface.Heights[(Math.Max(y - 1, 0) * surface.Width) + x];
				float down = surface.Heights[(Math.Min(y + 1, surface.Height - 1) * surface.Width) + x];

				Vector3 tangentX = new(2f * horizontalSpacing, 0f, right - left);
				Vector3 tangentY = new(0f, 2f * horizontalSpacing, down - up);
				Vector3 normal = Vector3.Normalize(Vector3.Cross(tangentY, tangentX));
				if (!float.IsFinite(normal.X) || !float.IsFinite(normal.Y) || !float.IsFinite(normal.Z))
					normal = new Vector3(0f, 0f, 1f);

				normals[(y * surface.Width) + x] = normal;
			}
		}

		Image<Rgba32> image = new(surface.Width, surface.Height);
		image.ProcessPixelRows(accessor =>
		{
			for (int y = 0; y < surface.Height; y++)
			{
				Span<Rgba32> row = accessor.GetRowSpan(y);
				int baseIndex = y * surface.Width;
				for (int x = 0; x < surface.Width; x++)
				{
					Vector3 normal = normals[baseIndex + x];
					row[x] = new Rgba32(
						(byte)Math.Clamp((int)MathF.Round((normal.X * 0.5f + 0.5f) * 255f), 0, 255),
						(byte)Math.Clamp((int)MathF.Round((normal.Y * 0.5f + 0.5f) * 255f), 0, 255),
						(byte)Math.Clamp((int)MathF.Round((normal.Z * 0.5f + 0.5f) * 255f), 0, 255),
						255);
				}
			}
		});

		if (surface.Width != DefaultNormalmapSize || surface.Height != DefaultNormalmapSize)
			image.Mutate(ctx => ctx.Resize(DefaultNormalmapSize, DefaultNormalmapSize, KnownResamplers.Bicubic));

		return image;
	}

	private static (double MeanAbsoluteDelta, double MaxAbsoluteDelta) CompareHeightmaps(HeightmapSurface localSurface, HeightmapSurface globalSurface)
	{
		int width = Math.Min(localSurface.Width, globalSurface.Width);
		int height = Math.Min(localSurface.Height, globalSurface.Height);
		double total = 0d;
		double max = 0d;
		int count = 0;

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				double local = localSurface.Heights[(y * localSurface.Width) + x];
				double global = globalSurface.Heights[(y * globalSurface.Width) + x];
				double delta = Math.Abs(local - global);
				total += delta;
				max = Math.Max(max, delta);
				count++;
			}
		}

		return count == 0 ? (0d, 0d) : (total / count, max);
	}

	private static string? ResolveDatasetPath(string datasetRoot, string? relativePath)
	{
		if (string.IsNullOrWhiteSpace(relativePath))
			return null;

		return Path.GetFullPath(Path.Combine(datasetRoot, relativePath.Replace('/', Path.DirectorySeparatorChar)));
	}

	private static string NormalizeRelativePath(string relativePath)
	{
		return relativePath.Replace('\\', '/');
	}

	private static string? ReadOptionalString(JsonObject jsonObject, string key)
	{
		return jsonObject[key]?.GetValue<string>();
	}

	private static float? ReadOptionalSingle(JsonObject jsonObject, string key)
	{
		JsonNode? node = jsonObject[key];
		if (node is null)
			return null;

		if (node is not JsonValue value)
			return null;

		if (value.TryGetValue<float>(out float singleValue))
			return singleValue;

		if (value.TryGetValue<double>(out double doubleValue))
			return (float)doubleValue;

		if (float.TryParse(node.ToJsonString().Trim('"'), NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed))
			return parsed;

		return null;
	}

	private static int? ParseOptionalInt(string? value)
	{
		return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) ? parsed : null;
	}

	private static double? ParseOptionalDouble(string? value)
	{
		return double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) ? parsed : null;
	}

	private static bool HasFlag(string[] args, string flag)
	{
		return args.Any(arg => string.Equals(arg, flag, StringComparison.OrdinalIgnoreCase));
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

	private static JsonSerializerOptions CreateJsonOptions()
	{
		return new JsonSerializerOptions
		{
			WriteIndented = true,
			DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
		};
	}

	private sealed record HeightmapSurface(string Path, int Width, int Height, float[] Heights);
}

internal sealed record MlNormalmapRepairReport(
	[property: JsonPropertyName("schema_version")] string SchemaVersion,
	[property: JsonPropertyName("generated_utc")] DateTime GeneratedUtc,
	[property: JsonPropertyName("dataset_root")] string DatasetRoot,
	[property: JsonPropertyName("dataset_directory")] string DatasetDirectory,
	[property: JsonPropertyName("tile_json_count")] int TileJsonCount,
	[property: JsonPropertyName("tiles_repaired")] int TilesRepaired,
	[property: JsonPropertyName("tiles_skipped_liquid_filter")] int TilesSkippedLiquidFilter,
	[property: JsonPropertyName("tiles_skipped_missing_heightmap")] int TilesSkippedMissingHeightmap,
	[property: JsonPropertyName("dry_run")] bool DryRun,
	[property: JsonPropertyName("rewrite_existing")] bool RewriteExisting,
	[property: JsonPropertyName("rewrite_when_local_differs")] double RewriteWhenLocalDiffers,
	[property: JsonPropertyName("only_liquid_tiles")] bool OnlyLiquidTiles,
	[property: JsonPropertyName("tiles")] List<MlNormalmapRepairTileReport> Tiles);

internal sealed record MlNormalmapRepairTileReport(
	[property: JsonPropertyName("tile_name")] string TileName,
	[property: JsonPropertyName("normalmap_path")] string? NormalmapPath,
	[property: JsonPropertyName("normalmap_reference_exists")] bool NormalmapReferenceExists,
	[property: JsonPropertyName("normalmap_file_exists")] bool NormalmapFileExists,
	[property: JsonPropertyName("liquid_mask_exists")] bool LiquidMaskExists,
	[property: JsonPropertyName("used_heightmap")] string? UsedHeightmap,
	[property: JsonPropertyName("repair_reason")] string RepairReason,
	[property: JsonPropertyName("repaired")] bool Repaired,
	[property: JsonPropertyName("output_normalmap_path")] string? OutputNormalmapPath,
	[property: JsonPropertyName("local_global_mean_absolute_delta")] double? LocalGlobalMeanAbsoluteDelta,
	[property: JsonPropertyName("local_global_max_absolute_delta")] double? LocalGlobalMaxAbsoluteDelta);