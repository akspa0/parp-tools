using System.Globalization;
using System.Security.Cryptography;
using System.Text;
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
	private const string StitchedDirectoryName = "stitched";
	private const string ArchetypeDirectoryName = "archetypes";
	private const string ArchetypeManifestFileName = "brush_archetype_manifest.json";
	internal const float DefaultFractalCandidateThreshold = 0.035f;
	private const int MaxFractalLevels = 4;

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
		string stitchedDirectory = Path.Combine(outputDirectory, StitchedDirectoryName);
		Directory.CreateDirectory(stitchedDirectory);
		string archetypesDirectory = Path.Combine(outputDirectory, ArchetypeDirectoryName);
		Directory.CreateDirectory(archetypesDirectory);
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
		List<string> stitchedFiles = [];
		Dictionary<string, MlBrushArchetypeAccumulator> archetypeAccumulators = new(StringComparer.Ordinal);
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
						FractalDetailPath: null,
						FractalCandidateMaskPath: null,
						LayerStackDepthPath: null,
						FractalStackProxyPath: null,
						FractalStackCandidateMaskPath: null,
						FractalMeanScore: 0f,
						FractalMaxScore: 0f,
						FractalStackMeanScore: 0f,
						FractalStackMaxScore: 0f,
						LayerStackMaxDepth: 0f,
						PatchCandidates: 0,
						GroupsWritten: 0,
						SkippedReason: "missing-heightmap-global"));
					continue;
				}

				float[] heightmap = LoadHeightmapL16(heightmapPath);
				float[] patchHeightGrid = BuildPatchHeightGrid(heightmap);
				float[] layerStackDepthGrid = BuildLayerStackDepthGrid(sample.TerrainData.ChunkLayers);
				MlBrushFractalArtifacts fractalArtifacts = BuildFractalArtifacts(
					patchHeightGrid,
					layerStackDepthGrid,
					TilePatchGridSize,
					TilePatchGridSize);
				string[] chunkTextureSignatures = BuildChunkTextureSignatures(sample.TerrainData.ChunkLayers);
				MlBrushPatchCell[] patchCells = BuildPatchCells(heightmap, chunkTextureSignatures);
				MlBrushPatchCell[] activeCells = SelectActivePatchCells(patchCells);

				List<MlBrushGroupCandidate> groups = BuildGroups(tileName, mapName, heightmap, activeCells);
				string? brushMaskRelativePath = null;
				string? fractalDetailRelativePath = null;
				string? fractalCandidateRelativePath = null;
				string? layerStackDepthRelativePath = null;
				string? fractalStackProxyRelativePath = null;
				string? fractalStackCandidateRelativePath = null;
				if (groups.Count > 0)
				{
					string tileMaskPath = Path.Combine(tileMasksDirectory, tileName + "_brush_mask.png");
					WriteTileGroupMask(groups, tileMaskPath);
					brushMaskRelativePath = Path.GetRelativePath(outputDirectory, tileMaskPath).Replace('\\', '/');
				}

				if (fractalArtifacts.HeatmapPngBytes.Length > 0)
				{
					string fractalDetailPath = Path.Combine(tileMasksDirectory, tileName + "_fractal_detail.png");
					File.WriteAllBytes(fractalDetailPath, fractalArtifacts.HeatmapPngBytes);
					fractalDetailRelativePath = Path.GetRelativePath(outputDirectory, fractalDetailPath).Replace('\\', '/');
				}

				if (fractalArtifacts.CandidateMaskPngBytes.Length > 0)
				{
					string fractalCandidatePath = Path.Combine(tileMasksDirectory, tileName + "_fractal_candidate_mask.png");
					File.WriteAllBytes(fractalCandidatePath, fractalArtifacts.CandidateMaskPngBytes);
					fractalCandidateRelativePath = Path.GetRelativePath(outputDirectory, fractalCandidatePath).Replace('\\', '/');
				}

				if (fractalArtifacts.LayerStackDepthPngBytes.Length > 0)
				{
					string layerStackDepthPath = Path.Combine(tileMasksDirectory, tileName + "_layer_stack_depth.png");
					File.WriteAllBytes(layerStackDepthPath, fractalArtifacts.LayerStackDepthPngBytes);
					layerStackDepthRelativePath = Path.GetRelativePath(outputDirectory, layerStackDepthPath).Replace('\\', '/');
				}

				if (fractalArtifacts.StackedProxyPngBytes.Length > 0)
				{
					string fractalStackProxyPath = Path.Combine(tileMasksDirectory, tileName + "_fractal_stack_proxy.png");
					File.WriteAllBytes(fractalStackProxyPath, fractalArtifacts.StackedProxyPngBytes);
					fractalStackProxyRelativePath = Path.GetRelativePath(outputDirectory, fractalStackProxyPath).Replace('\\', '/');
				}

				if (fractalArtifacts.StackedCandidateMaskPngBytes.Length > 0)
				{
					string fractalStackCandidatePath = Path.Combine(tileMasksDirectory, tileName + "_fractal_stack_candidate_mask.png");
					File.WriteAllBytes(fractalStackCandidatePath, fractalArtifacts.StackedCandidateMaskPngBytes);
					fractalStackCandidateRelativePath = Path.GetRelativePath(outputDirectory, fractalStackCandidatePath).Replace('\\', '/');
				}

				for (int index = 0; index < groups.Count; index++)
				{
					MlBrushGroupCandidate group = groups[index];
					string groupId = $"{tileName}_g{(index + 1).ToString("D4", CultureInfo.InvariantCulture)}";
					MlBrushArchetypeDescriptor archetype = group.DescribeArchetype();
					MlBrushGroupReport report = group.ToReport(groupId, datasetRoot, sample, heightmapPath, archetype);
					string groupPath = Path.Combine(groupsDirectory, groupId + ".json");
					File.WriteAllText(groupPath, JsonSerializer.Serialize(report, CreateBrushJsonOptions()));
					string relativeGroupPath = Path.GetRelativePath(outputDirectory, groupPath).Replace('\\', '/');
					groupFiles.Add(relativeGroupPath);
					if (!archetypeAccumulators.TryGetValue(archetype.ArchetypeId, out MlBrushArchetypeAccumulator? accumulator) || accumulator is null)
					{
						accumulator = new MlBrushArchetypeAccumulator(archetype);
						archetypeAccumulators.Add(archetype.ArchetypeId, accumulator);
					}
					accumulator.Add(report, relativeGroupPath);
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
					FractalDetailPath: fractalDetailRelativePath,
					FractalCandidateMaskPath: fractalCandidateRelativePath,
					LayerStackDepthPath: layerStackDepthRelativePath,
					FractalStackProxyPath: fractalStackProxyRelativePath,
					FractalStackCandidateMaskPath: fractalStackCandidateRelativePath,
					FractalMeanScore: fractalArtifacts.MeanScore,
					FractalMaxScore: fractalArtifacts.MaxScore,
					FractalStackMeanScore: fractalArtifacts.StackedMeanScore,
					FractalStackMaxScore: fractalArtifacts.StackedMaxScore,
					LayerStackMaxDepth: fractalArtifacts.MaxLayerStackDepth,
					PatchCandidates: activeCells.Length,
					GroupsWritten: groups.Count,
					SkippedReason: null));
			}
			catch (Exception ex)
			{
				issueTracker.Record(datasetFile, ex);
			}
		}

		stitchedFiles.AddRange(WriteStitchedTileLayers(tileSummaries, outputDirectory, stitchedDirectory));

		List<string> archetypeFiles = [];
		List<MlBrushArchetypeSummary> archetypeSummaries = [];
		foreach (MlBrushArchetypeAccumulator accumulator in archetypeAccumulators.Values
			.OrderByDescending(static entry => entry.GroupCount)
			.ThenBy(static entry => entry.ArchetypeId, StringComparer.Ordinal))
		{
			MlBrushArchetypeSummary summary = accumulator.ToSummary();
			string archetypePath = Path.Combine(archetypesDirectory, summary.ArchetypeId + ".json");
			File.WriteAllText(archetypePath, JsonSerializer.Serialize(summary, CreateBrushJsonOptions()));
			archetypeFiles.Add(Path.GetRelativePath(outputDirectory, archetypePath).Replace('\\', '/'));
			archetypeSummaries.Add(summary);
		}

		MlBrushArchetypeManifest archetypeManifest = new(
			SchemaVersion: "wowviewer-ml-brush-archetype.v1",
			GeneratedUtc: DateTime.UtcNow,
			DatasetRoot: datasetRoot,
			OutputDirectory: outputDirectory,
			ArchetypeCount: archetypeSummaries.Count,
			ArchetypeFiles: archetypeFiles,
			Archetypes: archetypeSummaries);

		string archetypeManifestPath = Path.Combine(outputDirectory, ArchetypeManifestFileName);
		File.WriteAllText(archetypeManifestPath, JsonSerializer.Serialize(archetypeManifest, CreateBrushJsonOptions()));

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
			ArchetypeCount: archetypeSummaries.Count,
			ArchetypeManifestPath: Path.GetFileName(archetypeManifestPath),
			StitchedFiles: stitchedFiles,
			GroupFiles: groupFiles,
			Tiles: tileSummaries);

		string manifestPath = Path.Combine(outputDirectory, "brush_imprint_manifest.json");
		File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, CreateBrushJsonOptions()));

		issueTracker.Print();
		Console.WriteLine($"Brush harvest complete: processed_tiles={manifest.TilesProcessed} skipped_missing_heightmap={tilesSkippedMissingHeightmap} groups={groupsWritten} patches={patchesWritten} archetypes={archetypeSummaries.Count}");
		Console.WriteLine($"Wrote {manifestPath}");
		Console.WriteLine($"Wrote {archetypeManifestPath}");
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

	private static float[] BuildPatchHeightGrid(float[] heightmap)
	{
		float[] patchHeights = new float[TilePatchGridSize * TilePatchGridSize];
		for (int patchY = 0; patchY < TilePatchGridSize; patchY++)
		{
			for (int patchX = 0; patchX < TilePatchGridSize; patchX++)
			{
				float h00 = heightmap[(patchY * TileVertexGridSize) + patchX];
				float h10 = heightmap[(patchY * TileVertexGridSize) + patchX + 1];
				float h01 = heightmap[((patchY + 1) * TileVertexGridSize) + patchX];
				float h11 = heightmap[((patchY + 1) * TileVertexGridSize) + patchX + 1];
				patchHeights[(patchY * TilePatchGridSize) + patchX] = (h00 + h10 + h01 + h11) * 0.25f;
			}
		}

		return patchHeights;
	}

	private static float[] BuildLayerStackDepthGrid(MlBrushChunkLayers[]? chunks)
	{
		float[] patchDepths = new float[TilePatchGridSize * TilePatchGridSize];
		if (chunks is null || chunks.Length == 0)
			return patchDepths;

		float[] chunkDepths = new float[ChunkCountPerRow * ChunkCountPerRow];
		foreach (MlBrushChunkLayers chunk in chunks)
		{
			if (chunk.ChunkIndex < 0 || chunk.ChunkIndex >= chunkDepths.Length)
				continue;

			chunkDepths[chunk.ChunkIndex] = Math.Max(0, (chunk.Layers?.Length ?? 0) - 1);
		}

		for (int patchY = 0; patchY < TilePatchGridSize; patchY++)
		{
			for (int patchX = 0; patchX < TilePatchGridSize; patchX++)
			{
				int chunkX = patchX / PatchCountPerChunk;
				int chunkY = patchY / PatchCountPerChunk;
				int chunkIndex = (chunkY * ChunkCountPerRow) + chunkX;
				patchDepths[(patchY * TilePatchGridSize) + patchX] = chunkDepths[chunkIndex];
			}
		}

		return patchDepths;
	}

	private static MlBrushFractalArtifacts BuildFractalArtifacts(float[] patchHeightGrid, float[] layerStackDepthGrid, int width, int height)
	{
		float[] scores = ComputeFractalDetailHeatmap(patchHeightGrid, width, height);
		if (scores.Length == 0)
			return new MlBrushFractalArtifacts(
				Array.Empty<byte>(),
				Array.Empty<byte>(),
				Array.Empty<byte>(),
				Array.Empty<byte>(),
				Array.Empty<byte>(),
				0f,
				0f,
				0f,
				0f,
				0f);

		float maxScore = 0f;
		double sumScore = 0d;
		for (int index = 0; index < scores.Length; index++)
		{
			float score = scores[index];
			if (score > maxScore)
				maxScore = score;
			sumScore += score;
		}

		float meanScore = scores.Length == 0 ? 0f : (float)(sumScore / scores.Length);
		float maxLayerDepth = 0f;
		for (int index = 0; index < layerStackDepthGrid.Length; index++)
		{
			if (layerStackDepthGrid[index] > maxLayerDepth)
				maxLayerDepth = layerStackDepthGrid[index];
		}

		float[] stackedScores = BuildStackedFractalProxy(scores, layerStackDepthGrid, maxLayerDepth);
		float stackedMaxScore = 0f;
		double stackedSumScore = 0d;
		for (int index = 0; index < stackedScores.Length; index++)
		{
			float score = stackedScores[index];
			if (score > stackedMaxScore)
				stackedMaxScore = score;
			stackedSumScore += score;
		}
		float stackedMeanScore = stackedScores.Length == 0 ? 0f : (float)(stackedSumScore / stackedScores.Length);

		byte[] heatmapPng = RenderFractalHeatmap(scores, width, height, maxScore);
		byte[] candidateMaskPng = RenderFractalCandidateMask(scores, width, height, DefaultFractalCandidateThreshold);
		byte[] layerStackDepthPng = RenderFractalHeatmap(layerStackDepthGrid, width, height, MathF.Max(maxLayerDepth, 1f));
		byte[] stackedProxyPng = RenderFractalHeatmap(stackedScores, width, height, MathF.Max(stackedMaxScore, 1e-6f));
		byte[] stackedCandidateMaskPng = RenderStackedFractalCandidateMask(stackedScores, layerStackDepthGrid, width, height, DefaultFractalCandidateThreshold);
		return new MlBrushFractalArtifacts(
			HeatmapPngBytes: heatmapPng,
			CandidateMaskPngBytes: candidateMaskPng,
			LayerStackDepthPngBytes: layerStackDepthPng,
			StackedProxyPngBytes: stackedProxyPng,
			StackedCandidateMaskPngBytes: stackedCandidateMaskPng,
			MeanScore: MathF.Round(meanScore, 6),
			MaxScore: MathF.Round(maxScore, 6),
			StackedMeanScore: MathF.Round(stackedMeanScore, 6),
			StackedMaxScore: MathF.Round(stackedMaxScore, 6),
			MaxLayerStackDepth: MathF.Round(maxLayerDepth, 6));
	}

	private static float[] BuildStackedFractalProxy(float[] fractalScores, float[] layerStackDepthGrid, float maxLayerDepth)
	{
		float[] combined = new float[fractalScores.Length];
		float layerNormalizer = MathF.Max(maxLayerDepth, 1f);
		for (int index = 0; index < combined.Length; index++)
		{
			float layerWeight = Math.Clamp(layerStackDepthGrid[index] / layerNormalizer, 0f, 1f);
			float baseScore = fractalScores[index];
			combined[index] = baseScore + (baseScore * layerWeight * 2f) + (DefaultFractalCandidateThreshold * 0.75f * layerWeight);
		}

		return combined;
	}

	private static float[] ComputeFractalDetailHeatmap(float[] values, int width, int height)
	{
		if (values.Length == 0 || width <= 1 || height <= 1)
			return [];

		float[] baseScores = new float[width * height];
		float[] current = values.ToArray();
		int currentWidth = width;
		int currentHeight = height;
		float totalWeight = 0f;

		for (int level = 0; level < MaxFractalLevels; level++)
		{
			float[]? downsampled = BlockAverage(current, currentWidth, currentHeight, out int downWidth, out int downHeight);
			if (downsampled == null)
				break;

			float[] approximated = UpsampleNearest(downsampled, downWidth, downHeight, currentWidth, currentHeight);
			float[] residual = new float[current.Length];
			for (int index = 0; index < current.Length; index++)
				residual[index] = MathF.Abs(current[index] - approximated[index]);

			float[] residualAtBase = currentWidth == width && currentHeight == height
				? residual
				: UpsampleNearest(residual, currentWidth, currentHeight, width, height);

			float weight = level + 1;
			for (int index = 0; index < baseScores.Length; index++)
				baseScores[index] += residualAtBase[index] * weight;

			totalWeight += weight;
			current = downsampled;
			currentWidth = downWidth;
			currentHeight = downHeight;
		}

		if (totalWeight <= 0f)
			return baseScores;

		for (int index = 0; index < baseScores.Length; index++)
			baseScores[index] /= totalWeight;

		return baseScores;
	}

	internal static float ComputeFractalDetailScore(float[] values, int width, int height)
	{
		if (values.Length == 0 || width <= 1 || height <= 1)
			return 0f;

		float[] current = values.ToArray();
		int currentWidth = width;
		int currentHeight = height;
		List<float> residuals = [];

		for (int level = 0; level < MaxFractalLevels; level++)
		{
			float[]? downsampled = BlockAverage(current, currentWidth, currentHeight, out int downWidth, out int downHeight);
			if (downsampled == null)
				break;

			float[] approximated = UpsampleNearest(downsampled, downWidth, downHeight, currentWidth, currentHeight);
			double residualMean = 0d;
			for (int index = 0; index < current.Length; index++)
				residualMean += MathF.Abs(current[index] - approximated[index]);

			residuals.Add((float)(residualMean / current.Length));
			current = downsampled;
			currentWidth = downWidth;
			currentHeight = downHeight;
		}

		if (residuals.Count == 0)
			return 0f;

		float weighted = 0f;
		float normalizer = 0f;
		for (int index = 0; index < residuals.Count; index++)
		{
			float weight = index + 1;
			weighted += residuals[index] * weight;
			normalizer += weight;
		}

		return MathF.Round(weighted / MathF.Max(normalizer, 1f), 6);
	}

	private static float[]? BlockAverage(float[] values, int width, int height, out int pooledWidth, out int pooledHeight)
	{
		pooledWidth = width / 2;
		pooledHeight = height / 2;
		if (pooledWidth < 2 || pooledHeight < 2)
			return null;

		float[] pooled = new float[pooledWidth * pooledHeight];
		for (int y = 0; y < pooledHeight; y++)
		{
			for (int x = 0; x < pooledWidth; x++)
			{
				int sourceX = x * 2;
				int sourceY = y * 2;
				float sum = values[(sourceY * width) + sourceX]
					+ values[(sourceY * width) + sourceX + 1]
					+ values[((sourceY + 1) * width) + sourceX]
					+ values[((sourceY + 1) * width) + sourceX + 1];
				pooled[(y * pooledWidth) + x] = sum * 0.25f;
			}
		}

		return pooled;
	}

	private static float[] UpsampleNearest(float[] values, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
	{
		float[] upsampled = new float[targetWidth * targetHeight];
		for (int y = 0; y < targetHeight; y++)
		{
			int sourceY = Math.Min(sourceHeight - 1, y * sourceHeight / Math.Max(targetHeight, 1));
			for (int x = 0; x < targetWidth; x++)
			{
				int sourceX = Math.Min(sourceWidth - 1, x * sourceWidth / Math.Max(targetWidth, 1));
				upsampled[(y * targetWidth) + x] = values[(sourceY * sourceWidth) + sourceX];
			}
		}

		return upsampled;
	}

	private static byte[] RenderFractalHeatmap(float[] scores, int width, int height, float maxScore)
	{
		using Image<L8> image = new(width, height);
		float normalizer = MathF.Max(maxScore, 1e-6f);
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				float normalized = Math.Clamp(scores[(y * width) + x] / normalizer, 0f, 1f);
				image[x, y] = new L8((byte)Math.Clamp(MathF.Round(normalized * 255f), 0f, 255f));
			}
		}

		using MemoryStream stream = new();
		image.SaveAsPng(stream);
		return stream.ToArray();
	}

	private static byte[] RenderFractalCandidateMask(float[] scores, int width, int height, float threshold)
	{
		using Image<L8> image = new(width, height);
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
				image[x, y] = new L8(scores[(y * width) + x] >= threshold ? (byte)255 : (byte)0);
		}

		using MemoryStream stream = new();
		image.SaveAsPng(stream);
		return stream.ToArray();
	}

	private static byte[] RenderStackedFractalCandidateMask(float[] scores, float[] layerStackDepthGrid, int width, int height, float threshold)
	{
		using Image<L8> image = new(width, height);
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int index = (y * width) + x;
				bool active = layerStackDepthGrid[index] > 0f && scores[index] >= threshold;
				image[x, y] = new L8(active ? (byte)255 : (byte)0);
			}
		}

		using MemoryStream stream = new();
		image.SaveAsPng(stream);
		return stream.ToArray();
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

					if (!lookup.TryGetValue((nx, ny), out MlBrushPatchCell? next) || next is null)
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

	private static List<string> WriteStitchedTileLayers(IReadOnlyList<MlBrushTileSummary> tileSummaries, string outputDirectory, string stitchedDirectory)
	{
		List<string> stitchedFiles = [];
		foreach (IGrouping<string, MlBrushTileSummary> mapGroup in tileSummaries
			.Where(static tile => string.IsNullOrWhiteSpace(tile.SkippedReason))
			.GroupBy(static tile => tile.MapName, StringComparer.OrdinalIgnoreCase))
		{
			TryWriteStitchedLayer(mapGroup, outputDirectory, stitchedDirectory, static tile => tile.BrushMaskPath, mapGroup.Key + "_full_brush_mask.png", stitchedFiles);
			TryWriteStitchedLayer(mapGroup, outputDirectory, stitchedDirectory, static tile => tile.FractalDetailPath, mapGroup.Key + "_full_fractal_detail.png", stitchedFiles);
			TryWriteStitchedLayer(mapGroup, outputDirectory, stitchedDirectory, static tile => tile.FractalCandidateMaskPath, mapGroup.Key + "_full_fractal_candidate_mask.png", stitchedFiles);
			TryWriteStitchedLayer(mapGroup, outputDirectory, stitchedDirectory, static tile => tile.LayerStackDepthPath, mapGroup.Key + "_full_layer_stack_depth.png", stitchedFiles);
			TryWriteStitchedLayer(mapGroup, outputDirectory, stitchedDirectory, static tile => tile.FractalStackProxyPath, mapGroup.Key + "_full_fractal_stack_proxy.png", stitchedFiles);
			TryWriteStitchedLayer(mapGroup, outputDirectory, stitchedDirectory, static tile => tile.FractalStackCandidateMaskPath, mapGroup.Key + "_full_fractal_stack_candidate_mask.png", stitchedFiles);
		}

		return stitchedFiles;
	}

	private static void TryWriteStitchedLayer(
		IEnumerable<MlBrushTileSummary> tiles,
		string outputDirectory,
		string stitchedDirectory,
		Func<MlBrushTileSummary, string?> pathSelector,
		string outputFileName,
		List<string> stitchedFiles)
	{
		List<(int TileX, int TileY, string Path)> resolvedTiles = [];
		foreach (MlBrushTileSummary tile in tiles)
		{
			string? relativePath = pathSelector(tile);
			if (string.IsNullOrWhiteSpace(relativePath) || !TryParseTileCoordinates(tile.TileName, out int tileX, out int tileY))
				continue;

			string absolutePath = Path.Combine(outputDirectory, relativePath.Replace('/', Path.DirectorySeparatorChar));
			if (!File.Exists(absolutePath))
				continue;

			resolvedTiles.Add((tileX, tileY, absolutePath));
		}

		if (resolvedTiles.Count == 0)
			return;

		int minX = resolvedTiles.Min(static tile => tile.TileX);
		int minY = resolvedTiles.Min(static tile => tile.TileY);
		int maxX = resolvedTiles.Max(static tile => tile.TileX);
		int maxY = resolvedTiles.Max(static tile => tile.TileY);

		using Image<L8> firstTile = Image.Load<L8>(resolvedTiles[0].Path);
		int tileWidth = firstTile.Width;
		int tileHeight = firstTile.Height;
		using Image<L8> stitched = new((maxX - minX + 1) * tileWidth, (maxY - minY + 1) * tileHeight);

		foreach ((int tileX, int tileY, string path) in resolvedTiles)
		{
			using Image<L8> tileImage = Image.Load<L8>(path);
			if (tileImage.Width != tileWidth || tileImage.Height != tileHeight)
				tileImage.Mutate(ctx => ctx.Resize(tileWidth, tileHeight));

			int destX = (tileX - minX) * tileWidth;
			int destY = (tileY - minY) * tileHeight;
			stitched.Mutate(ctx => ctx.DrawImage(tileImage, new Point(destX, destY), 1f));
		}

		string outputPath = Path.Combine(stitchedDirectory, outputFileName);
		stitched.SaveAsPng(outputPath);
		stitchedFiles.Add(Path.GetRelativePath(outputDirectory, outputPath).Replace('\\', '/'));
	}

	private static bool TryParseTileCoordinates(string tileName, out int tileX, out int tileY)
	{
		tileX = 0;
		tileY = 0;
		if (string.IsNullOrWhiteSpace(tileName))
			return false;

		string[] parts = tileName.Split('_', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
		if (parts.Length < 3)
			return false;

		return int.TryParse(parts[^2], NumberStyles.Integer, CultureInfo.InvariantCulture, out tileX)
			&& int.TryParse(parts[^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out tileY);
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

	public MlBrushArchetypeDescriptor DescribeArchetype()
	{
		int patchWidth = (MaxPatchX - MinPatchX) + 1;
		int patchHeight = (MaxPatchY - MinPatchY) + 1;
		float fillRatio = patchWidth <= 0 || patchHeight <= 0
			? 0f
			: MathF.Round(Patches.Count / (float)(patchWidth * patchHeight), 6);
		float meanRelief = Patches.Count == 0 ? 0f : MathF.Round(Patches.Average(static patch => patch.Relief), 6);
		float meanSlope = Patches.Count == 0 ? 0f : MathF.Round(Patches.Average(static patch => patch.Slope), 6);
		string textureFamilyKey = BuildTextureFamilyKey(TextureSignatures);
		string shapeFingerprint = BuildShapeFingerprint(NormalizedHeightGrid, HeightGridWidth, HeightGridHeight);
		string label = BuildArchetypeLabel(patchWidth, patchHeight, fillRatio, meanRelief, meanSlope);
		string archetypeKey = string.Create(CultureInfo.InvariantCulture, $"{label}|{Bucket(patchWidth, 8)}x{Bucket(patchHeight, 8)}|fill:{Bucket(fillRatio, 0.1f)}|relief:{Bucket(meanRelief, 0.01f)}|slope:{Bucket(meanSlope, 0.01f)}|tex:{textureFamilyKey}|shape:{shapeFingerprint}");
		string archetypeId = "br_" + Convert.ToHexString(SHA1.HashData(Encoding.UTF8.GetBytes(archetypeKey)))[..10].ToLowerInvariant();

		return new MlBrushArchetypeDescriptor(
			ArchetypeId: archetypeId,
			ArchetypeKey: archetypeKey,
			ArchetypeLabel: label,
			ShapeFingerprint: shapeFingerprint,
			TextureFamilyKey: textureFamilyKey,
			PatchWidth: patchWidth,
			PatchHeight: patchHeight,
			PatchCount: Patches.Count,
			FillRatio: fillRatio,
			MeanRelief: meanRelief,
			MeanSlope: meanSlope);
	}

	public MlBrushGroupReport ToReport(string groupId, string datasetRoot, MlBrushDatasetSample sample, string heightmapPath, MlBrushArchetypeDescriptor archetype)
	{
		float meanScore = Patches.Count == 0 ? 0f : Patches.Average(static patch => patch.Score);
		float maxScore = Patches.Count == 0 ? 0f : Patches.Max(static patch => patch.Score);
		float meanRelief = Patches.Count == 0 ? 0f : Patches.Average(static patch => patch.Relief);
		float meanSlope = Patches.Count == 0 ? 0f : Patches.Average(static patch => patch.Slope);
		float fractalDetailScore = MlBrushImprintHarvester.ComputeFractalDetailScore(NormalizedHeightGrid, HeightGridWidth, HeightGridHeight);
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
			ArchetypeId: archetype.ArchetypeId,
			ArchetypeKey: archetype.ArchetypeKey,
			ArchetypeLabel: archetype.ArchetypeLabel,
			ShapeFingerprint: archetype.ShapeFingerprint,
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
			FillRatio: archetype.FillRatio,
			MeanScore: MathF.Round(meanScore, 6),
			MeanRelief: MathF.Round(meanRelief, 6),
			MeanSlope: MathF.Round(meanSlope, 6),
			FractalDetailScore: fractalDetailScore,
			FractalCandidate: fractalDetailScore >= MlBrushImprintHarvester.DefaultFractalCandidateThreshold,
			MaxScore: MathF.Round(maxScore, 6),
			TextureSignatures: TextureSignatures,
			HeightGridWidth: HeightGridWidth,
			HeightGridHeight: HeightGridHeight,
			NormalizedHeightGrid: NormalizedHeightGrid,
			Patches: patchPoints);
	}

	private static string BuildTextureFamilyKey(IEnumerable<string> signatures)
	{
		List<string> families = signatures
			.SelectMany(static signature => signature.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
			.Select(static signature => Path.GetFileNameWithoutExtension(signature).ToLowerInvariant())
			.Where(static signature => !string.IsNullOrWhiteSpace(signature))
			.Distinct(StringComparer.Ordinal)
			.OrderBy(static signature => signature, StringComparer.Ordinal)
			.Take(3)
			.ToList();

		return families.Count == 0 ? "none" : string.Join("+", families);
	}

	private static string BuildShapeFingerprint(float[] normalizedHeightGrid, int width, int height)
	{
		if (normalizedHeightGrid.Length == 0 || width <= 0 || height <= 0)
			return "0000000000000000";

		StringBuilder builder = new(capacity: 16);
		for (int cellY = 0; cellY < 4; cellY++)
		{
			for (int cellX = 0; cellX < 4; cellX++)
			{
				int startX = (cellX * width) / 4;
				int endX = ((cellX + 1) * width) / 4;
				int startY = (cellY * height) / 4;
				int endY = ((cellY + 1) * height) / 4;

				float sum = 0f;
				int count = 0;
				for (int y = startY; y < endY; y++)
				{
					for (int x = startX; x < endX; x++)
					{
						sum += normalizedHeightGrid[(y * width) + x];
						count++;
					}
				}

				float average = count == 0 ? 0f : sum / count;
				int bucket = Math.Clamp((int)MathF.Round(average * 15f), 0, 15);
				builder.Append(bucket.ToString("x", CultureInfo.InvariantCulture));
			}
		}

		return builder.ToString();
	}

	private static string BuildArchetypeLabel(int patchWidth, int patchHeight, float fillRatio, float meanRelief, float meanSlope)
	{
		float aspectRatio = patchWidth >= patchHeight
			? patchWidth / (float)Math.Max(1, patchHeight)
			: patchHeight / (float)Math.Max(1, patchWidth);

		if (fillRatio < 0.35f)
			return "trace";
		if (meanSlope >= 0.08f && aspectRatio >= 2.5f)
			return "ridge";
		if (meanRelief >= 0.05f && fillRatio >= 0.55f)
			return "mound";
		if (meanRelief < 0.02f && meanSlope < 0.02f)
			return "shelf";
		if (aspectRatio >= 2.0f)
			return "band";
		return "terrace";
	}

	private static string Bucket(float value, float step)
	{
		int bucket = step <= 0f ? 0 : (int)MathF.Round(value / step);
		return bucket.ToString(CultureInfo.InvariantCulture);
	}

	private static string Bucket(int value, int step)
	{
		int bucket = step <= 0 ? value : (int)MathF.Round(value / (float)step);
		return bucket.ToString(CultureInfo.InvariantCulture);
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
	[property: JsonPropertyName("archetype_id")] string ArchetypeId,
	[property: JsonPropertyName("archetype_key")] string ArchetypeKey,
	[property: JsonPropertyName("archetype_label")] string ArchetypeLabel,
	[property: JsonPropertyName("shape_fingerprint")] string ShapeFingerprint,
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
	[property: JsonPropertyName("fill_ratio")] float FillRatio,
	[property: JsonPropertyName("mean_score")] float MeanScore,
	[property: JsonPropertyName("mean_relief")] float MeanRelief,
	[property: JsonPropertyName("mean_slope")] float MeanSlope,
	[property: JsonPropertyName("fractal_detail_score")] float FractalDetailScore,
	[property: JsonPropertyName("fractal_candidate")] bool FractalCandidate,
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
	[property: JsonPropertyName("fractal_detail_path")] string? FractalDetailPath,
	[property: JsonPropertyName("fractal_candidate_mask_path")] string? FractalCandidateMaskPath,
	[property: JsonPropertyName("layer_stack_depth_path")] string? LayerStackDepthPath,
	[property: JsonPropertyName("fractal_stack_proxy_path")] string? FractalStackProxyPath,
	[property: JsonPropertyName("fractal_stack_candidate_mask_path")] string? FractalStackCandidateMaskPath,
	[property: JsonPropertyName("fractal_mean_score")] float FractalMeanScore,
	[property: JsonPropertyName("fractal_max_score")] float FractalMaxScore,
	[property: JsonPropertyName("fractal_stack_mean_score")] float FractalStackMeanScore,
	[property: JsonPropertyName("fractal_stack_max_score")] float FractalStackMaxScore,
	[property: JsonPropertyName("layer_stack_max_depth")] float LayerStackMaxDepth,
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
	[property: JsonPropertyName("archetype_count")] int ArchetypeCount,
	[property: JsonPropertyName("archetype_manifest_path")] string ArchetypeManifestPath,
	[property: JsonPropertyName("stitched_files")] List<string> StitchedFiles,
	[property: JsonPropertyName("group_files")] List<string> GroupFiles,
	[property: JsonPropertyName("tiles")] List<MlBrushTileSummary> Tiles);

internal sealed record MlBrushFractalArtifacts(
	byte[] HeatmapPngBytes,
	byte[] CandidateMaskPngBytes,
	byte[] LayerStackDepthPngBytes,
	byte[] StackedProxyPngBytes,
	byte[] StackedCandidateMaskPngBytes,
	float MeanScore,
	float MaxScore,
	float StackedMeanScore,
	float StackedMaxScore,
	float MaxLayerStackDepth);

internal sealed record MlBrushArchetypeDescriptor(
	string ArchetypeId,
	string ArchetypeKey,
	string ArchetypeLabel,
	string ShapeFingerprint,
	string TextureFamilyKey,
	int PatchWidth,
	int PatchHeight,
	int PatchCount,
	float FillRatio,
	float MeanRelief,
	float MeanSlope);

internal sealed record MlBrushArchetypeSummary(
	[property: JsonPropertyName("schema_version")] string SchemaVersion,
	[property: JsonPropertyName("archetype_id")] string ArchetypeId,
	[property: JsonPropertyName("archetype_key")] string ArchetypeKey,
	[property: JsonPropertyName("archetype_label")] string ArchetypeLabel,
	[property: JsonPropertyName("shape_fingerprint")] string ShapeFingerprint,
	[property: JsonPropertyName("texture_family_key")] string TextureFamilyKey,
	[property: JsonPropertyName("group_count")] int GroupCount,
	[property: JsonPropertyName("patch_count")] int PatchCount,
	[property: JsonPropertyName("average_patch_width")] float AveragePatchWidth,
	[property: JsonPropertyName("average_patch_height")] float AveragePatchHeight,
	[property: JsonPropertyName("average_fill_ratio")] float AverageFillRatio,
	[property: JsonPropertyName("average_mean_score")] float AverageMeanScore,
	[property: JsonPropertyName("average_mean_relief")] float AverageMeanRelief,
	[property: JsonPropertyName("average_mean_slope")] float AverageMeanSlope,
	[property: JsonPropertyName("texture_signatures")] List<string> TextureSignatures,
	[property: JsonPropertyName("representative_group_id")] string RepresentativeGroupId,
	[property: JsonPropertyName("representative_group_file")] string RepresentativeGroupFile,
	[property: JsonPropertyName("group_ids")] List<string> GroupIds);

internal sealed record MlBrushArchetypeManifest(
	[property: JsonPropertyName("schema_version")] string SchemaVersion,
	[property: JsonPropertyName("generated_utc")] DateTime GeneratedUtc,
	[property: JsonPropertyName("dataset_root")] string DatasetRoot,
	[property: JsonPropertyName("output_directory")] string OutputDirectory,
	[property: JsonPropertyName("archetype_count")] int ArchetypeCount,
	[property: JsonPropertyName("archetype_files")] List<string> ArchetypeFiles,
	[property: JsonPropertyName("archetypes")] List<MlBrushArchetypeSummary> Archetypes);

internal sealed class MlBrushArchetypeAccumulator
{
	private readonly HashSet<string> _textureSignatures = new(StringComparer.Ordinal);
	private readonly List<string> _groupIds = [];
	private string? _representativeGroupId;
	private string? _representativeGroupFile;
	private int _representativePatchCount = -1;
	private float _representativeScore = float.MinValue;
	private float _averagePatchWidthSum;
	private float _averagePatchHeightSum;
	private float _averageFillRatioSum;
	private float _averageMeanScoreSum;
	private float _averageMeanReliefSum;
	private float _averageMeanSlopeSum;

	public MlBrushArchetypeAccumulator(MlBrushArchetypeDescriptor descriptor)
	{
		ArchetypeId = descriptor.ArchetypeId;
		ArchetypeKey = descriptor.ArchetypeKey;
		ArchetypeLabel = descriptor.ArchetypeLabel;
		ShapeFingerprint = descriptor.ShapeFingerprint;
		TextureFamilyKey = descriptor.TextureFamilyKey;
	}

	public string ArchetypeId { get; }
	public string ArchetypeKey { get; }
	public string ArchetypeLabel { get; }
	public string ShapeFingerprint { get; }
	public string TextureFamilyKey { get; }
	public int GroupCount { get; private set; }

	public void Add(MlBrushGroupReport report, string relativeGroupPath)
	{
		GroupCount++;
		PatchCount += report.PatchCount;
		_averagePatchWidthSum += report.PatchWidth;
		_averagePatchHeightSum += report.PatchHeight;
		_averageFillRatioSum += report.FillRatio;
		_averageMeanScoreSum += report.MeanScore;
		_averageMeanReliefSum += report.MeanRelief;
		_averageMeanSlopeSum += report.MeanSlope;
		_groupIds.Add(report.GroupId);
		foreach (string signature in report.TextureSignatures)
			_textureSignatures.Add(signature);

		if (report.PatchCount > _representativePatchCount || (report.PatchCount == _representativePatchCount && report.MeanScore > _representativeScore))
		{
			_representativePatchCount = report.PatchCount;
			_representativeScore = report.MeanScore;
			_representativeGroupId = report.GroupId;
			_representativeGroupFile = relativeGroupPath;
		}
	}

	public int PatchCount { get; private set; }

	public MlBrushArchetypeSummary ToSummary()
	{
		float divisor = Math.Max(GroupCount, 1);
		return new MlBrushArchetypeSummary(
			SchemaVersion: "wowviewer-ml-brush-archetype-summary.v1",
			ArchetypeId: ArchetypeId,
			ArchetypeKey: ArchetypeKey,
			ArchetypeLabel: ArchetypeLabel,
			ShapeFingerprint: ShapeFingerprint,
			TextureFamilyKey: TextureFamilyKey,
			GroupCount: GroupCount,
			PatchCount: PatchCount,
			AveragePatchWidth: MathF.Round(_averagePatchWidthSum / divisor, 6),
			AveragePatchHeight: MathF.Round(_averagePatchHeightSum / divisor, 6),
			AverageFillRatio: MathF.Round(_averageFillRatioSum / divisor, 6),
			AverageMeanScore: MathF.Round(_averageMeanScoreSum / divisor, 6),
			AverageMeanRelief: MathF.Round(_averageMeanReliefSum / divisor, 6),
			AverageMeanSlope: MathF.Round(_averageMeanSlopeSum / divisor, 6),
			TextureSignatures: _textureSignatures.OrderBy(static value => value, StringComparer.Ordinal).ToList(),
			RepresentativeGroupId: _representativeGroupId ?? string.Empty,
			RepresentativeGroupFile: _representativeGroupFile ?? string.Empty,
			GroupIds: _groupIds.OrderBy(static value => value, StringComparer.Ordinal).ToList());
	}
}

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
