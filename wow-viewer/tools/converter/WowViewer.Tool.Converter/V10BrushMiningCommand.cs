using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using WowViewer.Core.Files;

namespace WowViewer.Tool.Converter;

internal static class V10BrushMiningCommand
{
	private const string CanonicalDictionaryBaseName = "brush_dictionary";
	private const string LegacyDictionaryBaseName = "object_anchored_brush_dictionary";
	private static readonly byte[] NpyMagic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];
	private static readonly Regex TileCoordRegex = new(@"_(\d+)_(\d+)(?:_|$)", RegexOptions.Compiled);
	private static readonly string[] ObjectTypes = ["tree", "rock", "building", "structure", "detail", "wmo", "other", "terrain"];
	private static readonly string[] AnchorTypes = ["object", "terrain"];

	public static void Run(string[] args)
	{
		try
		{
			BrushMiningOptions options = ParseOptions(args);
			List<string> npzFiles = Directory
				.EnumerateFiles(options.InputDirectory, "*.npz", SearchOption.AllDirectories)
				.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase)
				.ToList();

			if (npzFiles.Count == 0)
			{
				Console.Error.WriteLine($"Error: no .npz files found in {options.InputDirectory}");
				Environment.ExitCode = 1;
				return;
			}

			Console.WriteLine($"Found {npzFiles.Count} .npz files");

			Random rng = new(options.Seed);
			List<BrushInstance> instances = [];
			foreach (string npzPath in npzFiles)
			{
				if (options.AnchorMode is BrushAnchorMode.Objects or BrushAnchorMode.Hybrid)
					instances.AddRange(ExtractObjectAnchoredInstances(npzPath, options));

				if (options.AnchorMode is BrushAnchorMode.Terrain or BrushAnchorMode.Hybrid)
					instances.AddRange(ExtractTerrainAnchoredInstances(npzPath, options));

				int maxInstances = options.DictionarySize * 100;
				if (instances.Count > maxInstances)
				{
					Shuffle(instances, rng);
					instances.RemoveRange(maxInstances, instances.Count - maxInstances);
				}
			}

			if (instances.Count < options.DictionarySize)
			{
				Console.Error.WriteLine($"ERROR: Only {instances.Count} instances found, need at least {options.DictionarySize}");
				Environment.ExitCode = 1;
				return;
			}

			Console.WriteLine($"Total brush instances after extraction: {instances.Count}");
			BrushMiningResult result = ClusterBrushPatterns(instances, options, rng);
			SaveDictionary(result, options.OutputDirectory);
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"Error: {ex.Message}");
			Environment.ExitCode = 1;
		}
	}

	private static BrushMiningOptions ParseOptions(string[] args)
	{
		string? inputDir = GetOption(args, "--input-dir", "-i");
		string? outputDir = GetOption(args, "--output-dir", "-o");
		if (string.IsNullOrWhiteSpace(inputDir))
			throw new InvalidOperationException("--input-dir <npz-dir> is required.");
		if (string.IsNullOrWhiteSpace(outputDir))
			throw new InvalidOperationException("--output-dir <dir> is required.");

		string inputDirectory = Path.GetFullPath(inputDir);
		if (!Directory.Exists(inputDirectory))
			throw new DirectoryNotFoundException($"Input directory '{inputDirectory}' does not exist.");

		BrushAnchorMode anchorMode = ParseAnchorMode(GetOption(args, "--anchor-mode", "-a") ?? "hybrid");
		string? placementDir = GetOption(args, "--placement-dir", "-p");
		if (anchorMode is not BrushAnchorMode.Terrain && string.IsNullOrWhiteSpace(placementDir))
			placementDir = inputDirectory;

		string? placementDirectory = string.IsNullOrWhiteSpace(placementDir) ? null : Path.GetFullPath(placementDir);
		if (!string.IsNullOrWhiteSpace(placementDirectory) && !Directory.Exists(placementDirectory))
			throw new DirectoryNotFoundException($"Placement directory '{placementDirectory}' does not exist.");

		return new BrushMiningOptions(
			InputDirectory: inputDirectory,
			PlacementDirectory: placementDirectory,
			OutputDirectory: Path.GetFullPath(outputDir),
			AnchorMode: anchorMode,
			ContextRadius: GetIntOption(args, "--context-radius", "-r") ?? 64,
			DictionarySize: GetIntOption(args, "--dictionary-size", "-d") ?? 128,
			MinOccurrences: GetIntOption(args, "--min-occurrences", "-m") ?? 3,
			TerrainSamplesPerTile: GetIntOption(args, "--terrain-samples-per-tile", "-t") ?? 128,
			Seed: GetIntOption(args, "--seed", "-S") ?? 42);
	}

	private static BrushAnchorMode ParseAnchorMode(string value)
	{
		return value.ToLowerInvariant() switch
		{
			"objects" => BrushAnchorMode.Objects,
			"terrain" => BrushAnchorMode.Terrain,
			_ => BrushAnchorMode.Hybrid,
		};
	}

	private static List<BrushInstance> ExtractObjectAnchoredInstances(string npzPath, BrushMiningOptions options)
	{
		TileTensorData? tile = TryLoadTileTensor(npzPath);
		if (tile is null)
			return [];

		if (!TryParseTileCoords(tile.TileName, out int tileX, out int tileY))
			return [];

		List<PlacementRecord> placements = LoadPlacements(options.PlacementDirectory, tile.TileName);
		if (placements.Count == 0)
			return [];

		List<BrushInstance> instances = [];
		foreach (PlacementRecord placement in placements)
		{
			if (!TryWorldToTileUv(placement.WorldX, placement.WorldY, tileX, tileY, out float u, out float v))
				continue;

			BrushContext? context = ExtractAlphaContextAtUv(tile, u, v, options.ContextRadius);
			if (context is null)
				continue;

			instances.Add(new BrushInstance(
				TileName: tile.TileName,
				AssetPath: placement.AssetPath,
				ObjectType: AssetPathTaxonomy.ClassifyObjectType(placement.AssetPath),
				AnchorType: "object",
				TerrainSignature: ClassifyTerrainSignature(context.TerrainStats),
				TerrainStats: context.TerrainStats,
				AlphaContext: context.Patch,
				PatchHeight: context.PatchHeight,
				PatchWidth: context.PatchWidth,
				DominantLayer: context.DominantLayer,
				LayerStats: context.LayerStats,
				HeightAtPoint: context.HeightAtPoint,
				ObjectScale: placement.Scale,
				SampleScore: ComputeAlphaEnergy(context.Patch, context.PatchHeight, context.PatchWidth)));
		}

		return instances;
	}

	private static List<BrushInstance> ExtractTerrainAnchoredInstances(string npzPath, BrushMiningOptions options)
	{
		if (options.TerrainSamplesPerTile <= 0)
			return [];

		TileTensorData? tile = TryLoadTileTensor(npzPath);
		if (tile?.Heightmap is null)
			return [];

		int stride = Math.Max(options.ContextRadius * 2, 32);
		List<(double Score, BrushInstance Instance)> candidates = [];
		for (int py = options.ContextRadius; py < tile.Alpha.Height - options.ContextRadius; py += stride)
		{
			for (int px = options.ContextRadius; px < tile.Alpha.Width - options.ContextRadius; px += stride)
			{
				float u = px / (float)(tile.Alpha.Width - 1);
				float v = py / (float)(tile.Alpha.Height - 1);
				BrushContext? context = ExtractAlphaContextAtUv(tile, u, v, options.ContextRadius);
				if (context is null)
					continue;

				double alphaEnergy = ComputeAlphaEnergy(context.Patch, context.PatchHeight, context.PatchWidth);
				double terrainEnergy = context.TerrainStats.Relief + context.TerrainStats.SlopeMean + context.TerrainStats.CurvatureAbsMean + context.TerrainStats.Roughness;
				double layerContrast = ComputeLayerContrast(context.LayerStats);
				double score = alphaEnergy + layerContrast + (0.1d * terrainEnergy);
				if (score <= 0.05d)
					continue;

				BrushInstance instance = new(
					TileName: tile.TileName,
					AssetPath: "__terrain__",
					ObjectType: "terrain",
					AnchorType: "terrain",
					TerrainSignature: ClassifyTerrainSignature(context.TerrainStats),
					TerrainStats: context.TerrainStats,
					AlphaContext: context.Patch,
					PatchHeight: context.PatchHeight,
					PatchWidth: context.PatchWidth,
					DominantLayer: context.DominantLayer,
					LayerStats: context.LayerStats,
					HeightAtPoint: context.HeightAtPoint,
					ObjectScale: 1f,
					SampleScore: (float)score);
				candidates.Add((score, instance));
			}
		}

		return candidates
			.OrderByDescending(static entry => entry.Score)
			.Take(options.TerrainSamplesPerTile)
			.Select(static entry => entry.Instance)
			.ToList();
	}

	private static TileTensorData? TryLoadTileTensor(string npzPath)
	{
		using FileStream stream = File.OpenRead(npzPath);
		using ZipArchive archive = new(stream, ZipArchiveMode.Read, leaveOpen: false);

		if (!TryReadFloatTensor3(archive, "mcal_alpha_pack_256", out FloatTensor3 alpha))
			return null;

		FloatTensor2? heightmap = TryReadFloatTensor2(archive, "height_257", out FloatTensor2 height)
			? height
			: null;

		return new TileTensorData(Path.GetFileNameWithoutExtension(npzPath), alpha, heightmap);
	}

	private static bool TryReadFloatTensor2(ZipArchive archive, string entryBaseName, out FloatTensor2 tensor)
	{
		tensor = default;
		if (!TryReadNpyEntry(archive, entryBaseName, out NpyPayload payload))
			return false;
		if (payload.Shape.Length != 2)
			return false;
		if (payload.Descr is not "<f4" and not "<f8")
			return false;

		float[] values = ReadFloatValues(payload);
		tensor = new FloatTensor2(payload.Shape[0], payload.Shape[1], values);
		return true;
	}

	private static bool TryReadFloatTensor3(ZipArchive archive, string entryBaseName, out FloatTensor3 tensor)
	{
		tensor = default;
		if (!TryReadNpyEntry(archive, entryBaseName, out NpyPayload payload))
			return false;
		if (payload.Shape.Length != 3)
			return false;
		if (payload.Descr is not "<f4" and not "<f8")
			return false;

		float[] values = ReadFloatValues(payload);
		tensor = new FloatTensor3(payload.Shape[0], payload.Shape[1], payload.Shape[2], values);
		return true;
	}

	private static float[] ReadFloatValues(NpyPayload payload)
	{
		int count = payload.Shape.Aggregate(1, static (accumulator, dimension) => accumulator * dimension);
		float[] values = new float[count];
		if (payload.Descr == "<f4")
		{
			for (int index = 0; index < count; index++)
				values[index] = BinaryPrimitives.ReadSingleLittleEndian(payload.Data.AsSpan(index * sizeof(float), sizeof(float)));
			return values;
		}

		for (int index = 0; index < count; index++)
			values[index] = (float)BinaryPrimitives.ReadDoubleLittleEndian(payload.Data.AsSpan(index * sizeof(double), sizeof(double)));
		return values;
	}

	private static bool TryReadNpyEntry(ZipArchive archive, string entryBaseName, out NpyPayload payload)
	{
		payload = default;
		ZipArchiveEntry? entry = archive.GetEntry(entryBaseName + ".npy");
		if (entry is null)
			return false;

		using Stream entryStream = entry.Open();
		using MemoryStream buffer = new();
		entryStream.CopyTo(buffer);
		byte[] bytes = buffer.ToArray();
		if (bytes.Length < 10 || !(bytes.AsSpan(0, 6).SequenceEqual(NpyMagic) || bytes.AsSpan(0, 6).SequenceEqual("?NUMPY"u8)))
			throw new InvalidDataException($"Archive entry '{entry.FullName}' is not a supported NumPy payload.");

		byte major = bytes[6];
		int headerLength;
		int headerOffset;
		switch (major)
		{
			case 1:
				headerLength = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(8, 2));
				headerOffset = 10;
				break;
			case 2:
			case 3:
				headerLength = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(8, 4)));
				headerOffset = 12;
				break;
			default:
				throw new InvalidDataException($"Unsupported NumPy header version {major}.");
		}

		string header = Encoding.ASCII.GetString(bytes, headerOffset, headerLength).Trim();
		string descr = ReadHeaderValue(header, "descr");
		bool fortranOrder = string.Equals(ReadHeaderValue(header, "fortran_order"), "True", StringComparison.OrdinalIgnoreCase);
		if (fortranOrder)
			throw new InvalidDataException("Fortran-order NumPy arrays are not supported.");

		int[] shape = ReadShape(header);
		int dataOffset = headerOffset + headerLength;
		int payloadLength = bytes.Length - dataOffset;
		byte[] data = new byte[payloadLength];
		Buffer.BlockCopy(bytes, dataOffset, data, 0, payloadLength);
		payload = new NpyPayload(descr, shape, data);
		return true;
	}

	private static string ReadHeaderValue(string header, string name)
	{
		string singleQuotePrefix = $"'{name}': ";
		int start = header.IndexOf(singleQuotePrefix, StringComparison.Ordinal);
		if (start >= 0)
		{
			start += singleQuotePrefix.Length;
			if (header[start] == '\'')
			{
				int quotedEnd = header.IndexOf('\'', start + 1);
				return header[(start + 1)..quotedEnd];
			}

			int valueEnd = header.IndexOfAny([',', '}'], start);
			return header[start..valueEnd].Trim();
		}

		throw new InvalidDataException($"NumPy header value '{name}' is missing.");
	}

	private static int[] ReadShape(string header)
	{
		int start = header.IndexOf("'shape':", StringComparison.Ordinal);
		if (start < 0)
			throw new InvalidDataException("NumPy shape is missing from the header.");

		start = header.IndexOf('(', start);
		int end = header.IndexOf(')', start);
		if (start < 0 || end < 0)
			throw new InvalidDataException("NumPy shape tuple is malformed.");

		return header[(start + 1)..end]
			.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
			.Select(static value => int.Parse(value, CultureInfo.InvariantCulture))
			.ToArray();
	}

	private static bool TryParseTileCoords(string tileName, out int tileX, out int tileY)
	{
		Match match = TileCoordRegex.Match(tileName);
		if (match.Success)
		{
			tileX = int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
			tileY = int.Parse(match.Groups[2].Value, CultureInfo.InvariantCulture);
			return true;
		}

		tileX = 0;
		tileY = 0;
		return false;
	}

	private static bool TryWorldToTileUv(float worldX, float worldY, int tileX, int tileY, out float u, out float v)
	{
		(float U, float V)[] candidates =
		[
			((worldX / 533.33333f) - tileX, (worldY / 533.33333f) - tileY),
			(((17066.66656f - worldY) / 533.33333f) - tileX, ((17066.66656f - worldX) / 533.33333f) - tileY),
		];

		float bestScore = float.NegativeInfinity;
		u = float.NaN;
		v = float.NaN;
		foreach ((float candidateU, float candidateV) in candidates)
		{
			if (candidateU < -0.25f || candidateU > 1.25f || candidateV < -0.25f || candidateV > 1.25f)
				continue;

			float score = -(MathF.Abs(candidateU - 0.5f) + MathF.Abs(candidateV - 0.5f));
			if (score > bestScore)
			{
				bestScore = score;
				u = candidateU;
				v = candidateV;
			}
		}

		return !float.IsNaN(u) && !float.IsNaN(v);
	}

	private static BrushContext? ExtractAlphaContextAtUv(TileTensorData tile, float u, float v, int radius)
	{
		if (float.IsNaN(u) || float.IsNaN(v))
			return null;

		int px = (int)(u * (tile.Alpha.Width - 1));
		int py = (int)(v * (tile.Alpha.Height - 1));
		if ((uint)px >= (uint)tile.Alpha.Width || (uint)py >= (uint)tile.Alpha.Height)
			return null;

		int x0 = Math.Max(0, px - radius);
		int x1 = Math.Min(tile.Alpha.Width, px + radius);
		int y0 = Math.Max(0, py - radius);
		int y1 = Math.Min(tile.Alpha.Height, py + radius);
		int patchWidth = x1 - x0;
		int patchHeight = y1 - y0;
		if (patchWidth < radius * 2 || patchHeight < radius * 2)
			return null;

		float[] patch = new float[patchWidth * patchHeight * tile.Alpha.Channels];
		for (int y = 0; y < patchHeight; y++)
		{
			for (int x = 0; x < patchWidth; x++)
			{
				for (int channel = 0; channel < tile.Alpha.Channels; channel++)
				{
					patch[((y * patchWidth) + x) * tile.Alpha.Channels + channel] = tile.Alpha[y0 + y, x0 + x, channel];
				}
			}
		}

		(float? heightAtPoint, TerrainStats terrainStats) = ComputeTerrainStats(tile.Heightmap, u, v, tile.Alpha.Width, tile.Alpha.Height, radius);
		LayerStats[] layerStats = new LayerStats[tile.Alpha.Channels];
		for (int channel = 0; channel < tile.Alpha.Channels; channel++)
			layerStats[channel] = ComputeLayerStats(patch, patchHeight, patchWidth, tile.Alpha.Channels, channel);

		int dominantLayer = 0;
		float dominantValue = float.MinValue;
		for (int channel = 0; channel < layerStats.Length; channel++)
		{
			if (layerStats[channel].Mean > dominantValue)
			{
				dominantValue = layerStats[channel].Mean;
				dominantLayer = channel;
			}
		}

		return new BrushContext(patch, patchHeight, patchWidth, dominantLayer, layerStats, heightAtPoint, terrainStats);
	}

	private static (float? HeightAtPoint, TerrainStats Stats) ComputeTerrainStats(FloatTensor2? heightmap, float u, float v, int alphaWidth, int alphaHeight, int radius)
	{
		if (heightmap is null)
			return (null, TerrainStats.Zero);

		int hx = Math.Clamp((int)MathF.Round(u * (heightmap.Value.Width - 1)), 0, heightmap.Value.Width - 1);
		int hy = Math.Clamp((int)MathF.Round(v * (heightmap.Value.Height - 1)), 0, heightmap.Value.Height - 1);
		int scaleX = Math.Max(1, (int)MathF.Round(radius * ((heightmap.Value.Width - 1f) / Math.Max(1, alphaWidth - 1))));
		int scaleY = Math.Max(1, (int)MathF.Round(radius * ((heightmap.Value.Height - 1f) / Math.Max(1, alphaHeight - 1))));

		int x0 = Math.Max(0, hx - scaleX);
		int x1 = Math.Min(heightmap.Value.Width - 1, hx + scaleX);
		int y0 = Math.Max(0, hy - scaleY);
		int y1 = Math.Min(heightmap.Value.Height - 1, hy + scaleY);

		int width = x1 - x0 + 1;
		int height = y1 - y0 + 1;
		if (width <= 0 || height <= 0)
			return (null, TerrainStats.Zero);

		float min = float.MaxValue;
		float max = float.MinValue;
		double sum = 0d;
		double sumSquares = 0d;
		double slopeSum = 0d;
		double slopeSquares = 0d;
		double curvatureSum = 0d;
		double curvatureAbsSum = 0d;
		int sampleCount = 0;

		for (int y = y0; y <= y1; y++)
		{
			for (int x = x0; x <= x1; x++)
			{
				float value = heightmap.Value[y, x];
				min = Math.Min(min, value);
				max = Math.Max(max, value);
				sum += value;
				sumSquares += value * value;

				float left = heightmap.Value[y, Math.Max(x0, x - 1)];
				float right = heightmap.Value[y, Math.Min(x1, x + 1)];
				float up = heightmap.Value[Math.Max(y0, y - 1), x];
				float down = heightmap.Value[Math.Min(y1, y + 1), x];

				float dx = (right - left) * 0.5f;
				float dy = (down - up) * 0.5f;
				double slope = Math.Sqrt((dx * dx) + (dy * dy));
				slopeSum += slope;
				slopeSquares += slope * slope;

				double curvature = (right - (2f * value) + left) + (down - (2f * value) + up);
				curvatureSum += curvature;
				curvatureAbsSum += Math.Abs(curvature);
				sampleCount++;
			}
		}

		double mean = sum / sampleCount;
		double variance = Math.Max(0d, (sumSquares / sampleCount) - (mean * mean));
		double slopeMean = slopeSum / sampleCount;
		double slopeVariance = Math.Max(0d, (slopeSquares / sampleCount) - (slopeMean * slopeMean));
		return (heightmap.Value[hy, hx], new TerrainStats(
			Relief: max - min,
			SlopeMean: (float)slopeMean,
			SlopeStd: (float)Math.Sqrt(slopeVariance),
			CurvatureMean: (float)(curvatureSum / sampleCount),
			CurvatureAbsMean: (float)(curvatureAbsSum / sampleCount),
			Roughness: (float)Math.Sqrt(variance)));
	}

	private static LayerStats ComputeLayerStats(float[] patch, int patchHeight, int patchWidth, int channels, int channel)
	{
		double sum = 0d;
		double sumSquares = 0d;
		float max = float.MinValue;
		float[] values = new float[patchHeight * patchWidth];
		for (int y = 0; y < patchHeight; y++)
		{
			for (int x = 0; x < patchWidth; x++)
			{
				float value = patch[((y * patchWidth) + x) * channels + channel];
				values[(y * patchWidth) + x] = value;
				sum += value;
				sumSquares += value * value;
				max = Math.Max(max, value);
			}
		}

		int count = patchHeight * patchWidth;
		double mean = sum / count;
		double variance = Math.Max(0d, (sumSquares / count) - (mean * mean));
		return new LayerStats((float)mean, (float)Math.Sqrt(variance), max, ComputeEntropy(values));
	}

	private static float ComputeEntropy(float[] values)
	{
		const int bins = 16;
		int[] histogram = new int[bins];
		foreach (float value in values)
		{
			int bin = Math.Clamp((int)(value * bins), 0, bins - 1);
			histogram[bin]++;
		}

		double total = values.Length;
		double entropy = 0d;
		foreach (int count in histogram)
		{
			if (count == 0)
				continue;
			double probability = count / total;
			entropy -= probability * Math.Log2(probability + 1e-8d);
		}
		return (float)entropy;
	}

	private static float ComputeAlphaEnergy(float[] patch, int patchHeight, int patchWidth)
	{
		float[] blended = BlendPatch(patch, patchHeight, patchWidth, 4);
		double sum = 0d;
		double sumSquares = 0d;
		foreach (float value in blended)
		{
			sum += value;
			sumSquares += value * value;
		}
		double mean = sum / blended.Length;
		double variance = Math.Max(0d, (sumSquares / blended.Length) - (mean * mean));
		return (float)(Math.Sqrt(variance) + ComputeEntropy(blended));
	}

	private static double ComputeLayerContrast(LayerStats[] stats)
	{
		double mean = stats.Average(static stat => stat.Mean);
		double variance = stats.Select(stat => Math.Pow(stat.Mean - mean, 2d)).Average();
		return Math.Sqrt(variance);
	}

	private static string ClassifyTerrainSignature(TerrainStats stats)
	{
		if (stats.Relief < 0.75f && stats.SlopeMean < 0.08f)
			return "flat";
		if (stats.CurvatureMean > 0.05f && stats.SlopeMean > 0.12f)
			return "ridge";
		if (stats.CurvatureMean < -0.05f && stats.SlopeMean > 0.12f)
			return "basin";
		if (stats.SlopeMean > 0.22f)
			return "slope";
		if (stats.Roughness > 1.0f)
			return "rough";
		return "undulating";
	}

	private static BrushMiningResult ClusterBrushPatterns(List<BrushInstance> instances, BrushMiningOptions options, Random rng)
	{
		int dictionarySize = options.DictionarySize;
		if (instances.Count < dictionarySize)
		{
			Console.WriteLine($"Warning: only {instances.Count} instances, reducing dictionary size");
			dictionarySize = Math.Max(1, instances.Count / 2);
		}

		Console.WriteLine($"Computing features for {instances.Count} brush instances...");
		float[][] features = instances.Select(ComputeBrushFeatureVector).ToArray();
		(float[] FeatureMean, float[] FeatureStd, float[][] Normalized) normalized = NormalizeFeatures(features);

		Console.WriteLine($"Clustering into {dictionarySize} 3D brush patterns...");
		float[][] centroids = KMeansPlusPlus(normalized.Normalized, dictionarySize, rng);
		int[] labels = LloydIterations(normalized.Normalized, centroids, rng, maxIterations: 100);

		List<BrushPattern> dictionary = [];
		for (int clusterIndex = 0; clusterIndex < dictionarySize; clusterIndex++)
		{
			List<BrushInstance> clusterInstances = [];
			for (int index = 0; index < labels.Length; index++)
			{
				if (labels[index] == clusterIndex)
					clusterInstances.Add(instances[index]);
			}

			if (clusterInstances.Count < options.MinOccurrences)
				continue;

			int patchHeight = clusterInstances.Min(static instance => instance.PatchHeight);
			int patchWidth = clusterInstances.Min(static instance => instance.PatchWidth);
			float[] stamp = AveragePatch(clusterInstances, patchHeight, patchWidth);

			Dictionary<string, int> objectTypeDistribution = CountBy(clusterInstances.Select(static instance => instance.ObjectType));
			Dictionary<string, int> anchorTypeDistribution = CountBy(clusterInstances.Select(static instance => instance.AnchorType));
			Dictionary<string, int> terrainSignatureDistribution = CountBy(clusterInstances.Select(static instance => instance.TerrainSignature));
			Dictionary<string, int> assetCounts = CountBy(clusterInstances.Where(static instance => !string.Equals(instance.AssetPath, "__terrain__", StringComparison.Ordinal)).Select(static instance => instance.AssetPath));
			Dictionary<string, int> assetCategoryCounts = CountBy(clusterInstances
				.Where(static instance => !string.Equals(instance.AssetPath, "__terrain__", StringComparison.Ordinal))
				.Select(static instance => AssetPathTaxonomy.Describe(instance.AssetPath).CategoryKey));

			List<(string AssetPath, int Count)> topAssets = assetCounts
				.OrderByDescending(static entry => entry.Value)
				.ThenBy(static entry => entry.Key, StringComparer.Ordinal)
				.Take(10)
				.Select(static entry => (entry.Key, entry.Value))
				.ToList();

			List<(string CategoryPath, int Count)> topAssetCategories = assetCategoryCounts
				.OrderByDescending(static entry => entry.Value)
				.ThenBy(static entry => entry.Key, StringComparer.Ordinal)
				.Take(10)
				.Select(static entry => (entry.Key, entry.Value))
				.ToList();

			float[] layerMeans = new float[4];
			for (int layer = 0; layer < layerMeans.Length; layer++)
				layerMeans[layer] = clusterInstances.Average(instance => instance.LayerStats[layer].Mean);

			float[] heightValues = clusterInstances.Where(static instance => instance.HeightAtPoint.HasValue).Select(static instance => instance.HeightAtPoint!.Value).ToArray();
			double heightMean = heightValues.Length == 0 ? 0d : heightValues.Average(static value => value);
			double heightVariance = heightValues.Length == 0 ? 0d : heightValues.Select(value => Math.Pow(value - heightMean, 2d)).Average();

			dictionary.Add(new BrushPattern(
				PatternId: clusterIndex,
				Stamp: stamp,
				StampHeight: patchHeight,
				StampWidth: patchWidth,
				ClusterSize: clusterInstances.Count,
				DominantAnchorType: anchorTypeDistribution.OrderByDescending(static entry => entry.Value).First().Key,
				DominantObjectType: objectTypeDistribution.OrderByDescending(static entry => entry.Value).First().Key,
				AnchorTypeDistribution: anchorTypeDistribution,
				ObjectTypeDistribution: objectTypeDistribution,
				TerrainSignatureDistribution: terrainSignatureDistribution,
				TopAssets: topAssets,
				TopAssetCategories: topAssetCategories,
				HeightMean: (float)heightMean,
				HeightStd: (float)Math.Sqrt(heightVariance),
				MeanLayerPresence: layerMeans));
		}

		return new BrushMiningResult(instances.Count, dictionary, normalized.FeatureMean, normalized.FeatureStd);
	}

	private static float[] ComputeBrushFeatureVector(BrushInstance instance)
	{
		float[] blended = BlendPatch(instance.AlphaContext, instance.PatchHeight, instance.PatchWidth, 4);
		ComputeMeanStd(blended, out float mean, out float std);
		float entropy = ComputeEntropy(blended);
		(float lowFrequency, float highFrequency) = ComputeFrequencyFeatures(blended, instance.PatchHeight, instance.PatchWidth);

		float[] layerPresence = instance.LayerStats.Select(static stat => stat.Mean).ToArray();
		float[] objectTypeOneHot = OneHot(ObjectTypes, instance.ObjectType);
		float[] anchorTypeOneHot = OneHot(AnchorTypes, instance.AnchorType);
		TerrainStats terrain = instance.TerrainStats;

		return
		[
			mean,
			std,
			entropy,
			lowFrequency,
			highFrequency,
			instance.HeightAtPoint ?? 0f,
			instance.ObjectScale,
			terrain.Relief,
			terrain.SlopeMean,
			terrain.SlopeStd,
			terrain.CurvatureMean,
			terrain.CurvatureAbsMean,
			terrain.Roughness,
			instance.SampleScore,
			.. layerPresence,
			.. objectTypeOneHot,
			.. anchorTypeOneHot,
		];
	}

	private static float[] BlendPatch(float[] patch, int height, int width, int channels)
	{
		float[] blended = new float[height * width];
		for (int index = 0; index < blended.Length; index++)
		{
			float max = float.MinValue;
			int baseIndex = index * channels;
			for (int channel = 0; channel < channels; channel++)
				max = Math.Max(max, patch[baseIndex + channel]);
			blended[index] = max;
		}
		return blended;
	}

	private static void ComputeMeanStd(float[] values, out float mean, out float std)
	{
		double sum = 0d;
		double sumSquares = 0d;
		foreach (float value in values)
		{
			sum += value;
			sumSquares += value * value;
		}
		double average = sum / values.Length;
		double variance = Math.Max(0d, (sumSquares / values.Length) - (average * average));
		mean = (float)average;
		std = (float)Math.Sqrt(variance);
	}

	private static (float LowFrequency, float HighFrequency) ComputeFrequencyFeatures(float[] blended, int height, int width)
	{
		int blockSize = Math.Max(1, Math.Min(height, width) / 4);
		List<float> blockMeans = [];
		for (int y = 0; y < height; y += blockSize)
		{
			for (int x = 0; x < width; x += blockSize)
			{
				double sum = 0d;
				int count = 0;
				for (int by = y; by < Math.Min(height, y + blockSize); by++)
				{
					for (int bx = x; bx < Math.Min(width, x + blockSize); bx++)
					{
						sum += blended[(by * width) + bx];
						count++;
					}
				}
				blockMeans.Add((float)(sum / count));
			}
		}

		ComputeMeanStd(blockMeans.ToArray(), out _, out float lowFrequency);

		double edgeSum = 0d;
		int edgeCount = 0;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				float center = blended[(y * width) + x];
				if (x + 1 < width)
				{
					edgeSum += Math.Abs(center - blended[(y * width) + x + 1]);
					edgeCount++;
				}
				if (y + 1 < height)
				{
					edgeSum += Math.Abs(center - blended[((y + 1) * width) + x]);
					edgeCount++;
				}
			}
		}

		return (lowFrequency, edgeCount == 0 ? 0f : (float)(edgeSum / edgeCount));
	}

	private static float[] OneHot(string[] vocabulary, string value)
	{
		float[] result = new float[vocabulary.Length];
		for (int index = 0; index < vocabulary.Length; index++)
		{
			if (string.Equals(vocabulary[index], value, StringComparison.Ordinal))
			{
				result[index] = 1f;
				break;
			}
		}
		return result;
	}

	private static (float[] FeatureMean, float[] FeatureStd, float[][] Normalized) NormalizeFeatures(float[][] features)
	{
		int dimension = features[0].Length;
		float[] mean = new float[dimension];
		float[] std = new float[dimension];
		for (int dimensionIndex = 0; dimensionIndex < dimension; dimensionIndex++)
		{
			double sum = 0d;
			double sumSquares = 0d;
			for (int sampleIndex = 0; sampleIndex < features.Length; sampleIndex++)
			{
				float value = features[sampleIndex][dimensionIndex];
				sum += value;
				sumSquares += value * value;
			}
			double average = sum / features.Length;
			double variance = Math.Max(0d, (sumSquares / features.Length) - (average * average));
			mean[dimensionIndex] = (float)average;
			std[dimensionIndex] = (float)Math.Sqrt(variance) + 1e-8f;
		}

		float[][] normalized = new float[features.Length][];
		for (int sampleIndex = 0; sampleIndex < features.Length; sampleIndex++)
		{
			normalized[sampleIndex] = new float[dimension];
			for (int dimensionIndex = 0; dimensionIndex < dimension; dimensionIndex++)
				normalized[sampleIndex][dimensionIndex] = (features[sampleIndex][dimensionIndex] - mean[dimensionIndex]) / std[dimensionIndex];
		}

		return (mean, std, normalized);
	}

	private static float[][] KMeansPlusPlus(float[][] data, int k, Random rng)
	{
		int dimension = data[0].Length;
		float[][] centroids = new float[k][];
		centroids[0] = (float[])data[rng.Next(data.Length)].Clone();
		for (int centroidIndex = 1; centroidIndex < k; centroidIndex++)
		{
			double[] distances = new double[data.Length];
			double distanceSum = 0d;
			for (int sampleIndex = 0; sampleIndex < data.Length; sampleIndex++)
			{
				double bestDistance = double.MaxValue;
				for (int existingIndex = 0; existingIndex < centroidIndex; existingIndex++)
					bestDistance = Math.Min(bestDistance, SquaredDistance(data[sampleIndex], centroids[existingIndex]));
				distances[sampleIndex] = bestDistance;
				distanceSum += bestDistance;
			}

			if (distanceSum <= 1e-8d)
			{
				centroids[centroidIndex] = (float[])data[rng.Next(data.Length)].Clone();
				continue;
			}

			double target = rng.NextDouble() * distanceSum;
			double running = 0d;
			int chosenIndex = data.Length - 1;
			for (int sampleIndex = 0; sampleIndex < data.Length; sampleIndex++)
			{
				running += distances[sampleIndex];
				if (running >= target)
				{
					chosenIndex = sampleIndex;
					break;
				}
			}

			centroids[centroidIndex] = new float[dimension];
			Array.Copy(data[chosenIndex], centroids[centroidIndex], dimension);
		}

		return centroids;
	}

	private static int[] LloydIterations(float[][] data, float[][] centroids, Random rng, int maxIterations)
	{
		int[] labels = new int[data.Length];
		for (int iteration = 0; iteration < maxIterations; iteration++)
		{
			bool changed = false;
			for (int sampleIndex = 0; sampleIndex < data.Length; sampleIndex++)
			{
				double bestDistance = double.MaxValue;
				int bestIndex = 0;
				for (int centroidIndex = 0; centroidIndex < centroids.Length; centroidIndex++)
				{
					double distance = SquaredDistance(data[sampleIndex], centroids[centroidIndex]);
					if (distance < bestDistance)
					{
						bestDistance = distance;
						bestIndex = centroidIndex;
					}
				}

				if (iteration == 0 || labels[sampleIndex] != bestIndex)
				{
					labels[sampleIndex] = bestIndex;
					changed = true;
				}
			}

			float[][] nextCentroids = new float[centroids.Length][];
			int[] counts = new int[centroids.Length];
			for (int centroidIndex = 0; centroidIndex < centroids.Length; centroidIndex++)
				nextCentroids[centroidIndex] = new float[centroids[centroidIndex].Length];

			for (int sampleIndex = 0; sampleIndex < data.Length; sampleIndex++)
			{
				int label = labels[sampleIndex];
				counts[label]++;
				for (int dimensionIndex = 0; dimensionIndex < data[sampleIndex].Length; dimensionIndex++)
					nextCentroids[label][dimensionIndex] += data[sampleIndex][dimensionIndex];
			}

			for (int centroidIndex = 0; centroidIndex < nextCentroids.Length; centroidIndex++)
			{
				if (counts[centroidIndex] == 0)
				{
					nextCentroids[centroidIndex] = (float[])data[rng.Next(data.Length)].Clone();
					continue;
				}

				for (int dimensionIndex = 0; dimensionIndex < nextCentroids[centroidIndex].Length; dimensionIndex++)
					nextCentroids[centroidIndex][dimensionIndex] /= counts[centroidIndex];
			}

			centroids = nextCentroids;
			if (!changed)
				break;
		}

		return labels;
	}

	private static double SquaredDistance(float[] left, float[] right)
	{
		double distance = 0d;
		for (int index = 0; index < left.Length; index++)
		{
			double delta = left[index] - right[index];
			distance += delta * delta;
		}
		return distance;
	}

	private static float[] AveragePatch(List<BrushInstance> instances, int patchHeight, int patchWidth)
	{
		float[] result = new float[patchHeight * patchWidth * 4];
		foreach (BrushInstance instance in instances)
		{
			for (int y = 0; y < patchHeight; y++)
			{
				for (int x = 0; x < patchWidth; x++)
				{
					for (int channel = 0; channel < 4; channel++)
					{
						result[((y * patchWidth) + x) * 4 + channel] += instance.AlphaContext[((y * instance.PatchWidth) + x) * 4 + channel];
					}
				}
			}
		}

		for (int index = 0; index < result.Length; index++)
			result[index] /= instances.Count;
		return result;
	}

	private static Dictionary<string, int> CountBy(IEnumerable<string> values)
	{
		Dictionary<string, int> counts = new(StringComparer.Ordinal);
		foreach (string value in values)
		{
			counts.TryGetValue(value, out int count);
			counts[value] = count + 1;
		}
		return counts;
	}

	private static void SaveDictionary(BrushMiningResult result, string outputDirectory)
	{
		Directory.CreateDirectory(outputDirectory);
		float[] stamps = FlattenStamps(result.Dictionary, out int patternCount, out int patchHeight, out int patchWidth);
		int[] patternIds = result.Dictionary.Select(static pattern => pattern.PatternId).ToArray();
		int[] clusterSizes = result.Dictionary.Select(static pattern => pattern.ClusterSize).ToArray();

		WriteDictionaryNpz(Path.Combine(outputDirectory, CanonicalDictionaryBaseName + ".npz"), stamps, patternCount, patchHeight, patchWidth, patternIds, clusterSizes, result.FeatureMean, result.FeatureStd);
		WriteDictionaryNpz(Path.Combine(outputDirectory, LegacyDictionaryBaseName + ".npz"), stamps, patternCount, patchHeight, patchWidth, patternIds, clusterSizes, result.FeatureMean, result.FeatureStd);

		object jsonPayload = new
		{
			total_instances = result.TotalInstances,
			dictionary_size = result.Dictionary.Count,
			patterns = result.Dictionary.Select(static pattern => new
			{
				pattern_id = pattern.PatternId,
				cluster_size = pattern.ClusterSize,
				dominant_anchor_type = pattern.DominantAnchorType,
				dominant_object_type = pattern.DominantObjectType,
				anchor_type_distribution = pattern.AnchorTypeDistribution,
				object_type_distribution = pattern.ObjectTypeDistribution,
				terrain_signature_distribution = pattern.TerrainSignatureDistribution,
				top_assets = pattern.TopAssets.Select(static entry => new object[] { entry.AssetPath, entry.Count }).ToArray(),
				top_asset_categories = pattern.TopAssetCategories.Select(static entry => new object[] { entry.CategoryPath, entry.Count }).ToArray(),
				height_mean = pattern.HeightMean,
				height_std = pattern.HeightStd,
				mean_layer_presence = pattern.MeanLayerPresence,
			}),
		};

		string json = JsonSerializer.Serialize(jsonPayload, new JsonSerializerOptions { WriteIndented = true });
		File.WriteAllText(Path.Combine(outputDirectory, CanonicalDictionaryBaseName + ".json"), json);
		File.WriteAllText(Path.Combine(outputDirectory, LegacyDictionaryBaseName + ".json"), json);

		Console.WriteLine($"Saved {result.Dictionary.Count} anchor-aware 3D brush patterns to {outputDirectory}");
	}

	private static float[] FlattenStamps(List<BrushPattern> dictionary, out int patternCount, out int patchHeight, out int patchWidth)
	{
		patternCount = dictionary.Count;
		patchHeight = dictionary.Count == 0 ? 0 : dictionary[0].StampHeight;
		patchWidth = dictionary.Count == 0 ? 0 : dictionary[0].StampWidth;
		float[] result = new float[patternCount * patchHeight * patchWidth * 4];
		for (int patternIndex = 0; patternIndex < dictionary.Count; patternIndex++)
			Array.Copy(dictionary[patternIndex].Stamp, 0, result, patternIndex * patchHeight * patchWidth * 4, dictionary[patternIndex].Stamp.Length);
		return result;
	}

	private static void WriteDictionaryNpz(string path, float[] stamps, int patternCount, int patchHeight, int patchWidth, int[] patternIds, int[] clusterSizes, float[] featureMean, float[] featureStd)
	{
		using FileStream stream = File.Create(path);
		using ZipArchive archive = new(stream, ZipArchiveMode.Create, leaveOpen: false);
		WriteNpyEntry(archive, "stamps", "<f4", [patternCount, patchHeight, patchWidth, 4], ToBytes(stamps));
		WriteNpyEntry(archive, "pattern_ids", "<i4", [patternIds.Length], ToBytes(patternIds));
		WriteNpyEntry(archive, "cluster_sizes", "<i4", [clusterSizes.Length], ToBytes(clusterSizes));
		WriteNpyEntry(archive, "feature_mean", "<f4", [featureMean.Length], ToBytes(featureMean));
		WriteNpyEntry(archive, "feature_std", "<f4", [featureStd.Length], ToBytes(featureStd));
	}

	private static void WriteNpyEntry(ZipArchive archive, string name, string descr, int[] shape, byte[] payload)
	{
		ZipArchiveEntry entry = archive.CreateEntry(name + ".npy", CompressionLevel.Fastest);
		using Stream stream = entry.Open();
		byte[] header = BuildNpyHeader(descr, shape);
		stream.Write(header, 0, header.Length);
		stream.Write(payload, 0, payload.Length);
	}

	private static byte[] BuildNpyHeader(string descr, int[] shape)
	{
		string shapeText = string.Join(", ", shape.Select(static value => value.ToString(CultureInfo.InvariantCulture)));
		if (shape.Length == 1)
			shapeText += ",";
		string headerText = $"{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shapeText}),}}";
		int prefixLength = NpyMagic.Length + 2 + 2;
		int totalSize = prefixLength + headerText.Length + 1;
		int padding = (64 - (totalSize % 64)) % 64;
		headerText += new string(' ', padding) + '\n';
		byte[] headerBytes = Encoding.ASCII.GetBytes(headerText);
		byte[] result = new byte[prefixLength + headerBytes.Length];
		NpyMagic.CopyTo(result.AsSpan(0, NpyMagic.Length));
		result[6] = 1;
		result[7] = 0;
		BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(8, 2), (ushort)headerBytes.Length);
		Buffer.BlockCopy(headerBytes, 0, result, 10, headerBytes.Length);
		return result;
	}

	private static byte[] ToBytes(float[] values)
	{
		byte[] result = new byte[values.Length * sizeof(float)];
		for (int index = 0; index < values.Length; index++)
			BinaryPrimitives.WriteSingleLittleEndian(result.AsSpan(index * sizeof(float), sizeof(float)), values[index]);
		return result;
	}

	private static byte[] ToBytes(int[] values)
	{
		byte[] result = new byte[values.Length * sizeof(int)];
		for (int index = 0; index < values.Length; index++)
			BinaryPrimitives.WriteInt32LittleEndian(result.AsSpan(index * sizeof(int), sizeof(int)), values[index]);
		return result;
	}

	private static List<PlacementRecord> LoadPlacements(string? placementDirectory, string tileName)
	{
		if (string.IsNullOrWhiteSpace(placementDirectory))
			return [];

		string[] candidates =
		[
			Path.Combine(placementDirectory, tileName + "_placements.json"),
			Path.Combine(placementDirectory, tileName + ".json"),
			Path.Combine(placementDirectory, tileName, "placements.json"),
		];

		string? path = candidates.FirstOrDefault(File.Exists);
		if (path is null)
			return [];

		using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
		List<PlacementRecord> result = [];
		if (document.RootElement.ValueKind == JsonValueKind.Object)
		{
			ReadPlacements(document.RootElement, "mddf", result);
			ReadPlacements(document.RootElement, "modf", result);
		}
		return result;
	}

	private static void ReadPlacements(JsonElement root, string propertyName, List<PlacementRecord> result)
	{
		if (!root.TryGetProperty(propertyName, out JsonElement array) || array.ValueKind != JsonValueKind.Array)
			return;

		foreach (JsonElement item in array.EnumerateArray())
		{
			string assetPath = item.TryGetProperty("model_path", out JsonElement modelPath) && modelPath.ValueKind == JsonValueKind.String
				? modelPath.GetString() ?? string.Empty
				: string.Empty;
			if (string.IsNullOrWhiteSpace(assetPath))
				continue;

			if (!item.TryGetProperty("position", out JsonElement position) || position.ValueKind != JsonValueKind.Object)
				continue;

			result.Add(new PlacementRecord(
				AssetPath: assetPath,
				WorldX: ReadFloat(position, "x"),
				WorldY: ReadFloat(position, "y"),
				WorldZ: ReadFloat(position, "z"),
				Scale: item.TryGetProperty("scale", out JsonElement scale) && scale.TryGetSingle(out float parsedScale) ? parsedScale : 1f));
		}
	}

	private static float ReadFloat(JsonElement element, string propertyName)
	{
		return element.TryGetProperty(propertyName, out JsonElement property) && property.TryGetSingle(out float value)
			? value
			: 0f;
	}

	private static void Shuffle<T>(List<T> values, Random rng)
	{
		for (int index = values.Count - 1; index > 0; index--)
		{
			int swapIndex = rng.Next(index + 1);
			(values[index], values[swapIndex]) = (values[swapIndex], values[index]);
		}
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
		return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) ? parsed : null;
	}

	private readonly record struct BrushMiningOptions(
		string InputDirectory,
		string? PlacementDirectory,
		string OutputDirectory,
		BrushAnchorMode AnchorMode,
		int ContextRadius,
		int DictionarySize,
		int MinOccurrences,
		int TerrainSamplesPerTile,
		int Seed);

	private enum BrushAnchorMode
	{
		Objects,
		Terrain,
		Hybrid,
	}

	private readonly record struct PlacementRecord(string AssetPath, float WorldX, float WorldY, float WorldZ, float Scale);
	private readonly record struct NpyPayload(string Descr, int[] Shape, byte[] Data);
	private readonly record struct FloatTensor2(int Height, int Width, float[] Values)
	{
		public float this[int y, int x] => Values[(y * Width) + x];
	}
	private readonly record struct FloatTensor3(int Height, int Width, int Channels, float[] Values)
	{
		public float this[int y, int x, int channel] => Values[((y * Width) + x) * Channels + channel];
	}
	private sealed record TileTensorData(string TileName, FloatTensor3 Alpha, FloatTensor2? Heightmap);
	private readonly record struct TerrainStats(float Relief, float SlopeMean, float SlopeStd, float CurvatureMean, float CurvatureAbsMean, float Roughness)
	{
		public static TerrainStats Zero => new(0f, 0f, 0f, 0f, 0f, 0f);
	}
	private readonly record struct LayerStats(float Mean, float Std, float Max, float Entropy);
	private sealed record BrushContext(float[] Patch, int PatchHeight, int PatchWidth, int DominantLayer, LayerStats[] LayerStats, float? HeightAtPoint, TerrainStats TerrainStats);
	private sealed record BrushInstance(
		string TileName,
		string AssetPath,
		string ObjectType,
		string AnchorType,
		string TerrainSignature,
		TerrainStats TerrainStats,
		float[] AlphaContext,
		int PatchHeight,
		int PatchWidth,
		int DominantLayer,
		LayerStats[] LayerStats,
		float? HeightAtPoint,
		float ObjectScale,
		float SampleScore);
	private sealed record BrushPattern(
		int PatternId,
		float[] Stamp,
		int StampHeight,
		int StampWidth,
		int ClusterSize,
		string DominantAnchorType,
		string DominantObjectType,
		Dictionary<string, int> AnchorTypeDistribution,
		Dictionary<string, int> ObjectTypeDistribution,
		Dictionary<string, int> TerrainSignatureDistribution,
		List<(string AssetPath, int Count)> TopAssets,
		List<(string CategoryPath, int Count)> TopAssetCategories,
		float HeightMean,
		float HeightStd,
		float[] MeanLayerPresence);
	private sealed record BrushMiningResult(int TotalInstances, List<BrushPattern> Dictionary, float[] FeatureMean, float[] FeatureStd);
}