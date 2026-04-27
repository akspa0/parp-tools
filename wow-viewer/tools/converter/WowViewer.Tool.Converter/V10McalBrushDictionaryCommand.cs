using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace WowViewer.Tool.Converter;

internal static class V10McalBrushDictionaryCommand
{
	private const int ChunkSize = 64;
	private const int FeaturePatchSize = 16;
	private static readonly byte[] NpyMagic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

	public static void Run(string[] args)
	{
		try
		{
			BrushDictionaryOptions options = ParseOptions(args);
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

			List<BrushStrokeSample> samples = [];
			List<SkippedShard> skipped = [];
			int tilesRead = 0;
			int patchesRead = 0;
			int rejectedUniform = 0;

			foreach (string npzPath in npzFiles)
			{
				string tileName = NormalizeTileName(Path.GetFileNameWithoutExtension(npzPath));
				if (!TryLoadAlpha(npzPath, out FloatTensor3 alpha, out string? skipReason))
				{
					skipped.Add(new SkippedShard(tileName, npzPath, skipReason ?? "missing_mcal_alpha_pack_256"));
					continue;
				}

				tilesRead++;
				foreach (BrushStrokeSample sample in ExtractSamples(npzPath, tileName, alpha, options, ref patchesRead, ref rejectedUniform))
				{
					samples.Add(sample);
					if (samples.Count >= options.MaxSamples)
						break;
				}

				if (samples.Count >= options.MaxSamples)
					break;
			}

			if (samples.Count == 0)
			{
				Console.Error.WriteLine("Error: no non-uniform MCAL brush stroke samples were readable.");
				Environment.ExitCode = 1;
				return;
			}

			int clusterCount = Math.Min(options.DictionarySize, samples.Count);
			Random rng = new(options.Seed);
			BrushDictionaryResult result = ClusterBrushes(samples, clusterCount, options, rng);
			SaveDictionary(options, npzFiles.Count, tilesRead, patchesRead, rejectedUniform, samples, result, skipped);
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"Error: {ex.Message}");
			Environment.ExitCode = 1;
		}
	}

	private static BrushDictionaryOptions ParseOptions(string[] args)
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

		int dictionarySize = Math.Max(1, GetIntOption(args, "--dictionary-size", "-d") ?? 64);
		return new BrushDictionaryOptions(
			InputDirectory: inputDirectory,
			OutputDirectory: Path.GetFullPath(outputDir),
			DictionarySize: dictionarySize,
			MinOccurrences: Math.Max(1, GetIntOption(args, "--min-occurrences", "-m") ?? 2),
			MinLayerStd: Math.Max(0f, GetFloatOption(args, "--min-layer-std", "-s") ?? 0.025f),
			MinGradient: Math.Max(0f, GetFloatOption(args, "--min-gradient", "-g") ?? 0.01f),
			MinRange: Math.Max(0f, GetFloatOption(args, "--min-range", "-r") ?? 0.08f),
			MaxIterations: Math.Max(1, GetIntOption(args, "--max-iterations", "-n") ?? 60),
			ExampleLimit: Math.Max(1, GetIntOption(args, "--example-limit", "-e") ?? 8),
			MaxSamples: Math.Max(dictionarySize, GetIntOption(args, "--max-samples", "-x") ?? dictionarySize * 2048),
			Seed: GetIntOption(args, "--seed", "-S") ?? 1337);
	}

	private static List<BrushStrokeSample> ExtractSamples(
		string npzPath,
		string tileName,
		FloatTensor3 alpha,
		BrushDictionaryOptions options,
		ref int patchesRead,
		ref int rejectedUniform)
	{
		List<BrushStrokeSample> samples = [];
		int chunkRows = alpha.Height / ChunkSize;
		int chunkColumns = alpha.Width / ChunkSize;
		int layerCount = Math.Min(alpha.Channels, 4);

		for (int chunkY = 0; chunkY < chunkRows; chunkY++)
		{
			for (int chunkX = 0; chunkX < chunkColumns; chunkX++)
			{
				for (int layer = 0; layer < layerCount; layer++)
				{
					patchesRead++;
					float[] patch = ExtractLayerPatch(alpha, chunkX, chunkY, layer);
					PatchStats stats = ComputePatchStats(patch, ChunkSize, ChunkSize);
					if (!IsUsableBrushStroke(stats, options))
					{
						rejectedUniform++;
						continue;
					}

					float[] features = BuildFeatureVector(patch, stats);
					samples.Add(new BrushStrokeSample(npzPath, tileName, chunkX, chunkY, layer, patch, features, stats));
				}
			}
		}

		return samples;
	}

	private static bool IsUsableBrushStroke(PatchStats stats, BrushDictionaryOptions options)
	{
		if (stats.Range < options.MinRange)
			return false;
		return stats.Std >= options.MinLayerStd || stats.GradientMean >= options.MinGradient;
	}

	private static float[] ExtractLayerPatch(FloatTensor3 alpha, int chunkX, int chunkY, int layer)
	{
		float[] patch = new float[ChunkSize * ChunkSize];
		int x0 = chunkX * ChunkSize;
		int y0 = chunkY * ChunkSize;
		for (int y = 0; y < ChunkSize; y++)
		{
			for (int x = 0; x < ChunkSize; x++)
				patch[(y * ChunkSize) + x] = alpha[y0 + y, x0 + x, layer];
		}

		return patch;
	}

	private static PatchStats ComputePatchStats(float[] patch, int height, int width)
	{
		float min = float.MaxValue;
		float max = float.MinValue;
		double sum = 0d;
		double sumSquares = 0d;
		int occupied = 0;
		foreach (float value in patch)
		{
			min = Math.Min(min, value);
			max = Math.Max(max, value);
			sum += value;
			sumSquares += value * value;
			if (value > 0.05f)
				occupied++;
		}

		float mean = (float)(sum / patch.Length);
		float std = (float)Math.Sqrt(Math.Max(0d, (sumSquares / patch.Length) - (mean * mean)));
		float range = max - min;
		float gradient = ComputeGradientMean(patch, height, width);
		float radial = ComputeRadialSymmetryScore(patch, height, width, mean, range);
		float square = ComputeSquareEdgeScore(patch, height, width, min, range);
		float hardness = range <= 1e-6f ? 0f : Math.Clamp((gradient / range) * 12f, 0f, 1f);
		float diameter = EstimateDiameterPixels(patch, min + (range * 0.5f));
		return new PatchStats(mean, std, min, max, range, occupied / (float)patch.Length, gradient, hardness, radial, square, diameter);
	}

	private static float ComputeGradientMean(float[] patch, int height, int width)
	{
		double sum = 0d;
		int count = 0;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				float value = patch[(y * width) + x];
				if (x + 1 < width)
				{
					sum += Math.Abs(value - patch[(y * width) + x + 1]);
					count++;
				}
				if (y + 1 < height)
				{
					sum += Math.Abs(value - patch[((y + 1) * width) + x]);
					count++;
				}
			}
		}

		return count == 0 ? 0f : (float)(sum / count);
	}

	private static float ComputeRadialSymmetryScore(float[] patch, int height, int width, float mean, float range)
	{
		if (range <= 1e-6f)
			return 0f;

		double weightSum = 0d;
		double cx = 0d;
		double cy = 0d;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				double weight = Math.Abs(patch[(y * width) + x] - mean);
				weightSum += weight;
				cx += x * weight;
				cy += y * weight;
			}
		}

		if (weightSum <= 1e-8d)
			return 0f;

		cx /= weightSum;
		cy /= weightSum;
		const int bins = 16;
		double[] sums = new double[bins];
		int[] counts = new int[bins];
		double maxRadius = Math.Sqrt((width * width) + (height * height)) * 0.5d;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				double radius = Math.Sqrt(((x - cx) * (x - cx)) + ((y - cy) * (y - cy)));
				int bin = Math.Clamp((int)Math.Floor((radius / maxRadius) * bins), 0, bins - 1);
				sums[bin] += patch[(y * width) + x];
				counts[bin]++;
			}
		}

		double residual = 0d;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				double radius = Math.Sqrt(((x - cx) * (x - cx)) + ((y - cy) * (y - cy)));
				int bin = Math.Clamp((int)Math.Floor((radius / maxRadius) * bins), 0, bins - 1);
				double expected = counts[bin] == 0 ? mean : sums[bin] / counts[bin];
				residual += Math.Abs(patch[(y * width) + x] - expected);
			}
		}

		double normalized = residual / (patch.Length * range);
		return Math.Clamp((float)(1d - normalized), 0f, 1f);
	}

	private static float ComputeSquareEdgeScore(float[] patch, int height, int width, float min, float range)
	{
		if (range <= 1e-6f)
			return 0f;

		float threshold = min + (range * 0.5f);
		int minX = width;
		int minY = height;
		int maxX = -1;
		int maxY = -1;
		int count = 0;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				if (patch[(y * width) + x] < threshold)
					continue;

				minX = Math.Min(minX, x);
				minY = Math.Min(minY, y);
				maxX = Math.Max(maxX, x);
				maxY = Math.Max(maxY, y);
				count++;
			}
		}

		if (count == 0)
			return 0f;

		int boxWidth = maxX - minX + 1;
		int boxHeight = maxY - minY + 1;
		float fill = count / (float)(boxWidth * boxHeight);
		float aspect = Math.Min(boxWidth, boxHeight) / (float)Math.Max(boxWidth, boxHeight);
		return Math.Clamp(fill * aspect, 0f, 1f);
	}

	private static float EstimateDiameterPixels(float[] patch, float threshold)
	{
		int area = patch.Count(value => value >= threshold);
		if (area == 0)
			return 0f;

		return 2f * MathF.Sqrt(area / MathF.PI);
	}

	private static float[] BuildFeatureVector(float[] patch, PatchStats stats)
	{
		float[] downsampled = DownsamplePatch(patch, ChunkSize, ChunkSize, FeaturePatchSize);
		float[] features = new float[downsampled.Length + 10];
		Array.Copy(downsampled, features, downsampled.Length);
		int offset = downsampled.Length;
		features[offset++] = stats.Mean;
		features[offset++] = stats.Std;
		features[offset++] = stats.Range;
		features[offset++] = stats.Coverage;
		features[offset++] = stats.GradientMean;
		features[offset++] = stats.EdgeHardness;
		features[offset++] = stats.RadialSymmetry;
		features[offset++] = stats.SquareEdgeScore;
		features[offset++] = stats.EstimatedDiameterPixels / ChunkSize;
		features[offset] = stats.Max;
		return features;
	}

	private static float[] DownsamplePatch(float[] patch, int height, int width, int targetSize)
	{
		float[] result = new float[targetSize * targetSize];
		float scaleX = width / (float)targetSize;
		float scaleY = height / (float)targetSize;
		for (int ty = 0; ty < targetSize; ty++)
		{
			for (int tx = 0; tx < targetSize; tx++)
			{
				int x0 = (int)MathF.Floor(tx * scaleX);
				int x1 = Math.Min(width, (int)MathF.Ceiling((tx + 1) * scaleX));
				int y0 = (int)MathF.Floor(ty * scaleY);
				int y1 = Math.Min(height, (int)MathF.Ceiling((ty + 1) * scaleY));
				double sum = 0d;
				int count = 0;
				for (int y = y0; y < y1; y++)
				{
					for (int x = x0; x < x1; x++)
					{
						sum += patch[(y * width) + x];
						count++;
					}
				}

				result[(ty * targetSize) + tx] = count == 0 ? 0f : (float)(sum / count);
			}
		}

		return result;
	}

	private static BrushDictionaryResult ClusterBrushes(List<BrushStrokeSample> samples, int clusterCount, BrushDictionaryOptions options, Random rng)
	{
		float[][] rawFeatures = samples.Select(static sample => sample.Features).ToArray();
		(float[] featureMean, float[] featureStd, float[][] normalizedFeatures) = NormalizeFeatures(rawFeatures);
		float[][] centroids = KMeansPlusPlus(normalizedFeatures, clusterCount, rng);
		int[] labels = LloydIterations(normalizedFeatures, centroids, rng, options.MaxIterations);

		List<BrushCluster> clusters = [];
		for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++)
		{
			List<int> memberIndices = [];
			for (int sampleIndex = 0; sampleIndex < labels.Length; sampleIndex++)
			{
				if (labels[sampleIndex] == clusterIndex)
					memberIndices.Add(sampleIndex);
			}

			if (memberIndices.Count < options.MinOccurrences)
				continue;

			clusters.Add(BuildCluster(clusters.Count, clusterIndex, memberIndices, samples, options.ExampleLimit));
		}

		return new BrushDictionaryResult(clusters, labels, featureMean, featureStd);
	}

	private static BrushCluster BuildCluster(int brushId, int rawClusterId, List<int> memberIndices, List<BrushStrokeSample> samples, int exampleLimit)
	{
		float[] stamp = new float[ChunkSize * ChunkSize];
		float meanSum = 0f;
		float stdSum = 0f;
		float gradientSum = 0f;
		float hardnessSum = 0f;
		float radialSum = 0f;
		float squareSum = 0f;
		float diameterSum = 0f;
		Dictionary<int, int> layerCounts = [];
		HashSet<string> tileNames = new(StringComparer.OrdinalIgnoreCase);

		foreach (int index in memberIndices)
		{
			BrushStrokeSample sample = samples[index];
			for (int valueIndex = 0; valueIndex < stamp.Length; valueIndex++)
				stamp[valueIndex] += sample.Patch[valueIndex];

			meanSum += sample.Stats.Mean;
			stdSum += sample.Stats.Std;
			gradientSum += sample.Stats.GradientMean;
			hardnessSum += sample.Stats.EdgeHardness;
			radialSum += sample.Stats.RadialSymmetry;
			squareSum += sample.Stats.SquareEdgeScore;
			diameterSum += sample.Stats.EstimatedDiameterPixels;
			tileNames.Add(sample.TileName);
			layerCounts.TryGetValue(sample.Layer, out int layerCount);
			layerCounts[sample.Layer] = layerCount + 1;
		}

		for (int valueIndex = 0; valueIndex < stamp.Length; valueIndex++)
			stamp[valueIndex] /= memberIndices.Count;

		PatchStats stampStats = ComputePatchStats(stamp, ChunkSize, ChunkSize);
		string shapeClass = ClassifyShape(stampStats);
		string signature = $"{shapeClass}|mean={Quantize(stampStats.Mean, 0.05f)}|std={Quantize(stampStats.Std, 0.025f)}|gradient={Quantize(stampStats.GradientMean, 0.01f)}|radial={Quantize(stampStats.RadialSymmetry, 0.1f)}|square={Quantize(stampStats.SquareEdgeScore, 0.1f)}";
		string hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(signature)))[..16].ToLowerInvariant();

		List<BrushExample> examples = memberIndices
			.Select(index => samples[index])
			.OrderBy(sample => SquaredDistance(sample.Patch, stamp))
			.Take(exampleLimit)
			.Select(sample => new BrushExample(sample.TileName, sample.Path, sample.ChunkX, sample.ChunkY, sample.Layer, sample.Stats))
			.ToList();

		int count = memberIndices.Count;
		BrushSummaryStats meanStats = new(
			Mean: meanSum / count,
			Std: stdSum / count,
			GradientMean: gradientSum / count,
			EdgeHardness: hardnessSum / count,
			RadialSymmetry: radialSum / count,
			SquareEdgeScore: squareSum / count,
			EstimatedDiameterPixels: diameterSum / count);

		return new BrushCluster(
			BrushId: brushId,
			RawClusterId: rawClusterId,
			BrushHash: hash,
			ShapeClass: shapeClass,
			Frequency: count,
			TileCount: tileNames.Count,
			LayerDistribution: layerCounts
				.OrderByDescending(static entry => entry.Value)
				.ThenBy(static entry => entry.Key)
				.Select(static entry => new int[] { entry.Key, entry.Value })
				.ToArray(),
			MeanStats: meanStats,
			StampStats: stampStats,
			Stamp: stamp,
			Examples: examples);
	}

	private static string ClassifyShape(PatchStats stats)
	{
		if (stats.RadialSymmetry >= 0.74f && stats.SquareEdgeScore < 0.62f)
			return stats.EdgeHardness >= 0.42f ? "hard_circular" : "soft_circular";
		if (stats.SquareEdgeScore >= 0.64f)
			return stats.EdgeHardness >= 0.42f ? "hard_square" : "soft_square";
		if (stats.EdgeHardness >= 0.48f)
			return "hard_edge";
		if (stats.EdgeHardness <= 0.22f)
			return "soft_edge";
		return "irregular";
	}

	private static (float[] Mean, float[] Std, float[][] Normalized) NormalizeFeatures(float[][] features)
	{
		int dimension = features[0].Length;
		float[] mean = new float[dimension];
		float[] std = new float[dimension];
		for (int dimensionIndex = 0; dimensionIndex < dimension; dimensionIndex++)
		{
			double sum = 0d;
			double sumSquares = 0d;
			foreach (float[] sample in features)
			{
				sum += sample[dimensionIndex];
				sumSquares += sample[dimensionIndex] * sample[dimensionIndex];
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
		Array.Fill(labels, -1);
		for (int iteration = 0; iteration < maxIterations; iteration++)
		{
			bool changed = false;
			for (int sampleIndex = 0; sampleIndex < data.Length; sampleIndex++)
			{
				int bestIndex = FindNearestCentroid(data[sampleIndex], centroids);
				if (labels[sampleIndex] != bestIndex)
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

	private static int FindNearestCentroid(float[] sample, float[][] centroids)
	{
		double bestDistance = double.MaxValue;
		int bestIndex = 0;
		for (int centroidIndex = 0; centroidIndex < centroids.Length; centroidIndex++)
		{
			double distance = SquaredDistance(sample, centroids[centroidIndex]);
			if (distance < bestDistance)
			{
				bestDistance = distance;
				bestIndex = centroidIndex;
			}
		}

		return bestIndex;
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

	private static bool TryLoadAlpha(string npzPath, out FloatTensor3 alpha, out string? skipReason)
	{
		alpha = default;
		skipReason = null;
		using FileStream stream = File.OpenRead(npzPath);
		using ZipArchive archive = new(stream, ZipArchiveMode.Read, leaveOpen: false);

		if (!TryReadFloatTensor3(archive, "mcal_alpha_pack_256", out alpha))
		{
			skipReason = "missing_mcal_alpha_pack_256";
			return false;
		}

		if (alpha.Height < ChunkSize || alpha.Width < ChunkSize || alpha.Channels < 1)
		{
			skipReason = $"unsupported_mcal_shape_{alpha.Height}x{alpha.Width}x{alpha.Channels}";
			return false;
		}

		return true;
	}

	private static bool TryReadFloatTensor3(ZipArchive archive, string entryBaseName, out FloatTensor3 tensor)
	{
		tensor = default;
		if (!TryReadNpyEntry(archive, entryBaseName, out NpyPayload payload) || payload.Shape.Length != 3)
			return false;
		if (payload.Descr is not "<f4" and not "<f8")
			return false;

		tensor = new FloatTensor3(payload.Shape[0], payload.Shape[1], payload.Shape[2], ReadFloatValues(payload));
		return true;
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
		if (bytes.Length < 10
			|| !(bytes.AsSpan(0, NpyMagic.Length).SequenceEqual(NpyMagic)
				|| bytes.AsSpan(0, NpyMagic.Length).SequenceEqual("?NUMPY"u8)))
		{
			throw new InvalidDataException($"Archive entry '{entry.FullName}' is not a supported NumPy payload.");
		}

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

		if (headerOffset + headerLength > bytes.Length)
			throw new InvalidDataException($"Archive entry '{entry.FullName}' has a truncated NumPy header.");

		string header = Encoding.ASCII.GetString(bytes, headerOffset, headerLength).Trim();
		string descr = ReadHeaderValue(header, "descr");
		bool fortranOrder = string.Equals(ReadHeaderValue(header, "fortran_order"), "True", StringComparison.OrdinalIgnoreCase);
		if (fortranOrder)
			throw new InvalidDataException("Fortran-order NumPy arrays are not supported.");

		int[] shape = ReadShape(header);
		int dataOffset = headerOffset + headerLength;
		byte[] data = new byte[bytes.Length - dataOffset];
		Buffer.BlockCopy(bytes, dataOffset, data, 0, data.Length);
		payload = new NpyPayload(descr, shape, data);
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

	private static void SaveDictionary(
		BrushDictionaryOptions options,
		int discoveredShardCount,
		int tilesRead,
		int patchesRead,
		int rejectedUniform,
		List<BrushStrokeSample> samples,
		BrushDictionaryResult result,
		List<SkippedShard> skipped)
	{
		Directory.CreateDirectory(options.OutputDirectory);
		string jsonPath = Path.Combine(options.OutputDirectory, "mcal_brush_dictionary.json");
		string npzPath = Path.Combine(options.OutputDirectory, "mcal_brush_dictionary.npz");

		var payload = new
		{
			schema_version = "v10-mcal-brush-dictionary.v1",
			generated_utc = DateTimeOffset.UtcNow,
			input_dir = options.InputDirectory,
			discovered_shard_count = discoveredShardCount,
			tiles_read = tilesRead,
			patches_read = patchesRead,
			rejected_uniform_patch_count = rejectedUniform,
			candidate_brush_count = samples.Count,
			retained_brush_count = result.Clusters.Count,
			dictionary_size = options.DictionarySize,
			min_occurrences = options.MinOccurrences,
			min_layer_std = options.MinLayerStd,
			min_gradient = options.MinGradient,
			min_range = options.MinRange,
			dictionary = result.Clusters.Select(static cluster => new
			{
				brush_id = cluster.BrushId,
				raw_cluster_label = cluster.RawClusterId,
				brush_hash = cluster.BrushHash,
				shape_class = cluster.ShapeClass,
				frequency = cluster.Frequency,
				tile_count = cluster.TileCount,
				layer_distribution = cluster.LayerDistribution,
				mean_stats = cluster.MeanStats,
				stamp_stats = cluster.StampStats,
				examples = cluster.Examples,
			}),
			labels = samples.Select((sample, index) => new
			{
				tile_name = sample.TileName,
				path = sample.Path,
				chunk_x = sample.ChunkX,
				chunk_y = sample.ChunkY,
				layer = sample.Layer,
				raw_cluster_label = result.Labels[index],
				retained_brush_id = result.Clusters.FirstOrDefault(cluster => cluster.RawClusterId == result.Labels[index])?.BrushId,
			}),
			skipped_shards = skipped.Select(static shard => new
			{
				tile_name = shard.TileName,
				path = shard.Path,
				reason = shard.Reason,
			}),
		};

		File.WriteAllText(jsonPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
		WriteDictionaryNpz(npzPath, result);

		Console.WriteLine("WowViewer.Tool.Converter mine-v10-mcal-brushes report");
		Console.WriteLine($"InputDir: {options.InputDirectory}");
		Console.WriteLine($"OutputDir: {options.OutputDirectory}");
		Console.WriteLine($"Shards: {discoveredShardCount}");
		Console.WriteLine($"TilesRead: {tilesRead}");
		Console.WriteLine($"PatchesRead: {patchesRead}");
		Console.WriteLine($"RejectedUniformPatches: {rejectedUniform}");
		Console.WriteLine($"CandidateBrushes: {samples.Count}");
		Console.WriteLine($"RetainedBrushes: {result.Clusters.Count}");
		Console.WriteLine($"Dictionary: {jsonPath}");
	}

	private static void WriteDictionaryNpz(string path, BrushDictionaryResult result)
	{
		int count = result.Clusters.Count;
		float[] stamps = new float[count * ChunkSize * ChunkSize];
		int[] brushIds = new int[count];
		int[] frequencies = new int[count];
		float[] shapeFeatures = new float[count * 7];

		for (int index = 0; index < count; index++)
		{
			BrushCluster cluster = result.Clusters[index];
			Array.Copy(cluster.Stamp, 0, stamps, index * ChunkSize * ChunkSize, cluster.Stamp.Length);
			brushIds[index] = cluster.BrushId;
			frequencies[index] = cluster.Frequency;
			shapeFeatures[(index * 7) + 0] = cluster.MeanStats.Mean;
			shapeFeatures[(index * 7) + 1] = cluster.MeanStats.Std;
			shapeFeatures[(index * 7) + 2] = cluster.MeanStats.GradientMean;
			shapeFeatures[(index * 7) + 3] = cluster.MeanStats.EdgeHardness;
			shapeFeatures[(index * 7) + 4] = cluster.MeanStats.RadialSymmetry;
			shapeFeatures[(index * 7) + 5] = cluster.MeanStats.SquareEdgeScore;
			shapeFeatures[(index * 7) + 6] = cluster.MeanStats.EstimatedDiameterPixels;
		}

		using FileStream stream = File.Create(path);
		using ZipArchive archive = new(stream, ZipArchiveMode.Create, leaveOpen: false);
		WriteNpyEntry(archive, "stamps", "<f4", [count, ChunkSize, ChunkSize], ToBytes(stamps));
		WriteNpyEntry(archive, "brush_ids", "<i4", [brushIds.Length], ToBytes(brushIds));
		WriteNpyEntry(archive, "frequencies", "<i4", [frequencies.Length], ToBytes(frequencies));
		WriteNpyEntry(archive, "shape_features", "<f4", [count, 7], ToBytes(shapeFeatures));
		WriteNpyEntry(archive, "feature_mean", "<f4", [result.FeatureMean.Length], ToBytes(result.FeatureMean));
		WriteNpyEntry(archive, "feature_std", "<f4", [result.FeatureStd.Length], ToBytes(result.FeatureStd));
	}

	private static void WriteNpyEntry(ZipArchive archive, string name, string descr, int[] shape, byte[] data)
	{
		ZipArchiveEntry entry = archive.CreateEntry(name + ".npy", CompressionLevel.Fastest);
		using Stream stream = entry.Open();
		byte[] header = BuildNpyHeader(descr, shape);
		stream.Write(header, 0, header.Length);
		stream.Write(data, 0, data.Length);
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

	private static string Quantize(float value, float step)
	{
		float quantized = MathF.Round(value / step) * step;
		return quantized.ToString("0.###", CultureInfo.InvariantCulture);
	}

	private static string NormalizeTileName(string fileStem)
	{
		return fileStem.EndsWith("_v10", StringComparison.OrdinalIgnoreCase)
			? fileStem[..^4]
			: fileStem;
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

	private static float? GetFloatOption(string[] args, string longName, string shortName)
	{
		string? value = GetOption(args, longName, shortName);
		return float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) ? parsed : null;
	}

	private readonly record struct BrushDictionaryOptions(
		string InputDirectory,
		string OutputDirectory,
		int DictionarySize,
		int MinOccurrences,
		float MinLayerStd,
		float MinGradient,
		float MinRange,
		int MaxIterations,
		int ExampleLimit,
		int MaxSamples,
		int Seed);

	private readonly record struct NpyPayload(string Descr, int[] Shape, byte[] Data);

	private readonly record struct FloatTensor3(int Height, int Width, int Channels, float[] Values)
	{
		public float this[int y, int x, int channel] => Values[((y * Width) + x) * Channels + channel];
	}

	private readonly record struct PatchStats(
		float Mean,
		float Std,
		float Min,
		float Max,
		float Range,
		float Coverage,
		float GradientMean,
		float EdgeHardness,
		float RadialSymmetry,
		float SquareEdgeScore,
		float EstimatedDiameterPixels);

	private readonly record struct BrushSummaryStats(
		float Mean,
		float Std,
		float GradientMean,
		float EdgeHardness,
		float RadialSymmetry,
		float SquareEdgeScore,
		float EstimatedDiameterPixels);

	private sealed record BrushStrokeSample(
		string Path,
		string TileName,
		int ChunkX,
		int ChunkY,
		int Layer,
		float[] Patch,
		float[] Features,
		PatchStats Stats);

	private sealed record BrushExample(
		string TileName,
		string Path,
		int ChunkX,
		int ChunkY,
		int Layer,
		PatchStats Stats);

	private sealed record BrushCluster(
		int BrushId,
		int RawClusterId,
		string BrushHash,
		string ShapeClass,
		int Frequency,
		int TileCount,
		int[][] LayerDistribution,
		BrushSummaryStats MeanStats,
		PatchStats StampStats,
		float[] Stamp,
		List<BrushExample> Examples);

	private sealed record BrushDictionaryResult(List<BrushCluster> Clusters, int[] Labels, float[] FeatureMean, float[] FeatureStd);

	private sealed record SkippedShard(string TileName, string Path, string Reason);
}
