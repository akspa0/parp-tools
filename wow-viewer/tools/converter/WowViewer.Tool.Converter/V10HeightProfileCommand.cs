using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace WowViewer.Tool.Converter;

internal static class V10HeightProfileCommand
{
	private static readonly byte[] NpyMagic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

	public static void Run(string[] args)
	{
		try
		{
			HeightProfileOptions options = ParseOptions(args);
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

			List<HeightProfileSample> samples = [];
			List<SkippedShard> skipped = [];
			foreach (string npzPath in npzFiles)
			{
				string tileName = NormalizeTileName(Path.GetFileNameWithoutExtension(npzPath));
				if (!TryLoadHeightProfile(npzPath, tileName, options.ProfileSize, out HeightProfileSample sample, out string? skipReason))
				{
					skipped.Add(new SkippedShard(tileName, npzPath, skipReason ?? "missing_height_257"));
					continue;
				}

				samples.Add(sample);
			}

			if (samples.Count == 0)
			{
				Console.Error.WriteLine("Error: no shards with height_257 were readable.");
				Environment.ExitCode = 1;
				return;
			}

			int requestedClusterCount = Math.Min(options.DictionarySize, samples.Count);
			Random rng = new(options.Seed);
			HeightProfileMiningResult result = ClusterHeightProfiles(samples, requestedClusterCount, options, rng);
			SaveDictionary(options, npzFiles.Count, samples, result, skipped);
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"Error: {ex.Message}");
			Environment.ExitCode = 1;
		}
	}

	private static HeightProfileOptions ParseOptions(string[] args)
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

		return new HeightProfileOptions(
			InputDirectory: inputDirectory,
			OutputDirectory: Path.GetFullPath(outputDir),
			DictionarySize: Math.Max(1, GetIntOption(args, "--dictionary-size", "-d") ?? 16),
			MinOccurrences: Math.Max(1, GetIntOption(args, "--min-occurrences", "-m") ?? 1),
			ProfileSize: Math.Clamp(GetIntOption(args, "--profile-size", "-p") ?? 17, 5, 65),
			MaxIterations: Math.Max(1, GetIntOption(args, "--max-iterations", "-n") ?? 40),
			ExampleLimit: Math.Max(1, GetIntOption(args, "--example-limit", "-e") ?? 8),
			Seed: GetIntOption(args, "--seed", "-s") ?? 1337);
	}

	private static bool TryLoadHeightProfile(string npzPath, string tileName, int profileSize, out HeightProfileSample sample, out string? skipReason)
	{
		sample = null!;
		skipReason = null;
		using FileStream stream = File.OpenRead(npzPath);
		using ZipArchive archive = new(stream, ZipArchiveMode.Read, leaveOpen: false);

		if (!TryReadFloatTensor2(archive, "height_257", out FloatTensor2 height))
		{
			skipReason = "missing_height_257";
			return false;
		}

		if (height.Height < 2 || height.Width < 2)
		{
			skipReason = $"unsupported_height_shape_{height.Height}x{height.Width}";
			return false;
		}

		float[] downsampled = Downsample(height, profileSize);
		HeightStats stats = ComputeHeightStats(downsampled);
		float[] normalized = NormalizeLocalProfile(downsampled, stats);
		float[] features = BuildFeatureVector(normalized, stats);
		sample = new HeightProfileSample(tileName, npzPath, downsampled, normalized, features, stats);
		return true;
	}

	private static HeightProfileMiningResult ClusterHeightProfiles(List<HeightProfileSample> samples, int clusterCount, HeightProfileOptions options, Random rng)
	{
		float[][] rawFeatures = samples.Select(static sample => sample.Features).ToArray();
		(float[] featureMean, float[] featureStd, float[][] normalizedFeatures) = NormalizeFeatures(rawFeatures);
		float[][] centroids = KMeansPlusPlus(normalizedFeatures, clusterCount, rng);
		int[] labels = LloydIterations(normalizedFeatures, centroids, rng, options.MaxIterations);

		List<HeightProfileCluster> clusters = [];
		for (int clusterId = 0; clusterId < clusterCount; clusterId++)
		{
			List<int> memberIndices = labels
				.Select((label, index) => (label, index))
				.Where(entry => entry.label == clusterId)
				.Select(static entry => entry.index)
				.ToList();
			if (memberIndices.Count < options.MinOccurrences)
				continue;

			clusters.Add(BuildCluster(clusters.Count, clusterId, memberIndices, samples, options.ExampleLimit));
		}

		return new HeightProfileMiningResult(clusters, labels, featureMean, featureStd);
	}

	private static HeightProfileCluster BuildCluster(int profileId, int rawClusterId, List<int> memberIndices, List<HeightProfileSample> samples, int exampleLimit)
	{
		int profileSize = (int)Math.Sqrt(samples[memberIndices[0]].NormalizedProfile.Length);
		float[] normalizedCentroid = new float[profileSize * profileSize];
		float[] absoluteCentroid = new float[profileSize * profileSize];
		float reliefSum = 0f;
		float slopeSum = 0f;
		float roughnessSum = 0f;
		float meanElevationSum = 0f;

		foreach (int index in memberIndices)
		{
			HeightProfileSample sample = samples[index];
			for (int valueIndex = 0; valueIndex < normalizedCentroid.Length; valueIndex++)
			{
				normalizedCentroid[valueIndex] += sample.NormalizedProfile[valueIndex];
				absoluteCentroid[valueIndex] += sample.DownsampledHeight[valueIndex];
			}

			reliefSum += sample.Stats.Relief;
			slopeSum += sample.Stats.SlopeMean;
			roughnessSum += sample.Stats.Roughness;
			meanElevationSum += sample.Stats.MeanElevation;
		}

		for (int valueIndex = 0; valueIndex < normalizedCentroid.Length; valueIndex++)
		{
			normalizedCentroid[valueIndex] /= memberIndices.Count;
			absoluteCentroid[valueIndex] /= memberIndices.Count;
		}

		List<HeightProfileExample> examples = memberIndices
			.Select(index => samples[index])
			.OrderBy(sample => SquaredDistance(sample.NormalizedProfile, normalizedCentroid))
			.Take(exampleLimit)
			.Select(sample => new HeightProfileExample(sample.TileName, sample.Path, ClassifyHeightArchetype(sample.Stats), sample.Stats))
			.ToList();

		HeightStats meanStats = new(
			MeanElevation: meanElevationSum / memberIndices.Count,
			Relief: reliefSum / memberIndices.Count,
			SlopeMean: slopeSum / memberIndices.Count,
			Roughness: roughnessSum / memberIndices.Count);
		string archetype = ClassifyHeightArchetype(meanStats);
		string signature = $"{archetype}|relief={Quantize(meanStats.Relief, 25f)}|slope={Quantize(meanStats.SlopeMean, 1f)}|rough={Quantize(meanStats.Roughness, 10f)}";
		string hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(signature)))[..16].ToLowerInvariant();

		return new HeightProfileCluster(profileId, rawClusterId, hash, archetype, memberIndices.Count, meanStats, normalizedCentroid, absoluteCentroid, examples);
	}

	private static string ClassifyHeightArchetype(HeightStats stats)
	{
		if (stats.Relief < 3f && stats.SlopeMean < 0.35f)
			return "flat_plain";
		if (stats.Relief < 20f && stats.SlopeMean < 1.2f)
			return "rolling_plain";
		if (stats.Relief >= 80f && stats.SlopeMean >= 4f)
			return "mountain_or_cliff";
		if (stats.Relief >= 35f && stats.SlopeMean >= 2f)
			return "ridge_or_valley";
		if (stats.Roughness >= 20f)
			return "rough_highland";
		return "mixed_terrain";
	}

	private static float[] Downsample(FloatTensor2 source, int targetSize)
	{
		float[] result = new float[targetSize * targetSize];
		float scaleX = (source.Width - 1f) / (targetSize - 1);
		float scaleY = (source.Height - 1f) / (targetSize - 1);
		for (int y = 0; y < targetSize; y++)
		{
			for (int x = 0; x < targetSize; x++)
			{
				float sourceX = x * scaleX;
				float sourceY = y * scaleY;
				int ix = Math.Clamp((int)sourceX, 0, source.Width - 2);
				int iy = Math.Clamp((int)sourceY, 0, source.Height - 2);
				float fx = sourceX - ix;
				float fy = sourceY - iy;

				float v00 = source[iy, ix];
				float v10 = source[iy, ix + 1];
				float v01 = source[iy + 1, ix];
				float v11 = source[iy + 1, ix + 1];
				float top = v00 + ((v10 - v00) * fx);
				float bottom = v01 + ((v11 - v01) * fx);
				result[(y * targetSize) + x] = top + ((bottom - top) * fy);
			}
		}

		return result;
	}

	private static HeightStats ComputeHeightStats(float[] downsampled)
	{
		float min = float.MaxValue;
		float max = float.MinValue;
		double sum = 0d;
		double sumSquares = 0d;
		foreach (float value in downsampled)
		{
			min = Math.Min(min, value);
			max = Math.Max(max, value);
			sum += value;
			sumSquares += value * value;
		}

		int size = (int)MathF.Sqrt(downsampled.Length);
		double slopeSum = 0d;
		int slopeCount = 0;
		for (int y = 1; y < size - 1; y++)
		{
			for (int x = 1; x < size - 1; x++)
			{
				float dx = (downsampled[(y * size) + x + 1] - downsampled[(y * size) + x - 1]) * 0.5f;
				float dy = (downsampled[((y + 1) * size) + x] - downsampled[((y - 1) * size) + x]) * 0.5f;
				slopeSum += Math.Sqrt((dx * dx) + (dy * dy));
				slopeCount++;
			}
		}

		float mean = (float)(sum / downsampled.Length);
		float roughness = (float)Math.Sqrt(Math.Max(0d, (sumSquares / downsampled.Length) - (mean * mean)));
		return new HeightStats(mean, max - min, slopeCount == 0 ? 0f : (float)(slopeSum / slopeCount), roughness);
	}

	private static float[] NormalizeLocalProfile(float[] values, HeightStats stats)
	{
		float scale = Math.Max(stats.Relief, 1f);
		float[] normalized = new float[values.Length];
		for (int index = 0; index < values.Length; index++)
			normalized[index] = (values[index] - stats.MeanElevation) / scale;
		return normalized;
	}

	private static float[] BuildFeatureVector(float[] normalizedProfile, HeightStats stats)
	{
		float[] features = new float[normalizedProfile.Length + 3];
		Array.Copy(normalizedProfile, features, normalizedProfile.Length);
		features[^3] = MathF.Log(1f + Math.Max(0f, stats.Relief));
		features[^2] = MathF.Log(1f + Math.Max(0f, stats.SlopeMean));
		features[^1] = MathF.Log(1f + Math.Max(0f, stats.Roughness));
		return features;
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

	private static bool TryReadFloatTensor2(ZipArchive archive, string entryBaseName, out FloatTensor2 tensor)
	{
		tensor = default;
		if (!TryReadNpyEntry(archive, entryBaseName, out NpyPayload payload) || payload.Shape.Length != 2)
			return false;
		if (payload.Descr is not "<f4" and not "<f8")
			return false;

		tensor = new FloatTensor2(payload.Shape[0], payload.Shape[1], ReadFloatValues(payload));
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

	private static void SaveDictionary(HeightProfileOptions options, int discoveredShardCount, List<HeightProfileSample> samples, HeightProfileMiningResult result, List<SkippedShard> skipped)
	{
		Directory.CreateDirectory(options.OutputDirectory);
		string jsonPath = Path.Combine(options.OutputDirectory, "height_profile_dictionary.json");
		string npzPath = Path.Combine(options.OutputDirectory, "height_profile_dictionary.npz");

		var payload = new
		{
			schema_version = "v10-height-profile-dictionary.v1",
			generated_utc = DateTimeOffset.UtcNow,
			input_dir = options.InputDirectory,
			discovered_shard_count = discoveredShardCount,
			tiles_read = samples.Count,
			retained_profile_count = result.Clusters.Count,
			dictionary_size = options.DictionarySize,
			min_occurrences = options.MinOccurrences,
			profile_size = options.ProfileSize,
			dictionary = result.Clusters.Select(static cluster => new
			{
				profile_id = cluster.ProfileId,
				raw_cluster_label = cluster.RawClusterId,
				profile_hash = cluster.ProfileHash,
				archetype = cluster.Archetype,
				cluster_size = cluster.ClusterSize,
				mean_stats = cluster.MeanStats,
				examples = cluster.Examples,
			}),
			labels = samples.Select((sample, index) => new
			{
				tile_name = sample.TileName,
				path = sample.Path,
				raw_cluster_label = result.Labels[index],
				retained_profile_id = result.Clusters.FirstOrDefault(cluster => cluster.RawClusterId == result.Labels[index])?.ProfileId,
				stats = sample.Stats,
			}),
			skipped_shards = skipped.Select(static shard => new
			{
				tile_name = shard.TileName,
				path = shard.Path,
				reason = shard.Reason,
			}),
		};

		File.WriteAllText(jsonPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
		WriteDictionaryNpz(npzPath, result, options.ProfileSize);

		Console.WriteLine("WowViewer.Tool.Converter mine-v10-height-profiles report");
		Console.WriteLine($"InputDir: {options.InputDirectory}");
		Console.WriteLine($"OutputDir: {options.OutputDirectory}");
		Console.WriteLine($"Shards: {discoveredShardCount}");
		Console.WriteLine($"TilesRead: {samples.Count}");
		Console.WriteLine($"RetainedProfiles: {result.Clusters.Count}");
		Console.WriteLine($"Dictionary: {jsonPath}");
	}

	private static void WriteDictionaryNpz(string path, HeightProfileMiningResult result, int profileSize)
	{
		int count = result.Clusters.Count;
		float[] normalizedCentroids = new float[count * profileSize * profileSize];
		float[] absoluteCentroids = new float[count * profileSize * profileSize];
		int[] profileIds = new int[count];
		int[] clusterSizes = new int[count];
		float[] summary = new float[count * 4];

		for (int index = 0; index < result.Clusters.Count; index++)
		{
			HeightProfileCluster cluster = result.Clusters[index];
			profileIds[index] = cluster.ProfileId;
			clusterSizes[index] = cluster.ClusterSize;
			Array.Copy(cluster.NormalizedCentroid, 0, normalizedCentroids, index * profileSize * profileSize, cluster.NormalizedCentroid.Length);
			Array.Copy(cluster.AbsoluteCentroid, 0, absoluteCentroids, index * profileSize * profileSize, cluster.AbsoluteCentroid.Length);
			summary[(index * 4) + 0] = cluster.MeanStats.MeanElevation;
			summary[(index * 4) + 1] = cluster.MeanStats.Relief;
			summary[(index * 4) + 2] = cluster.MeanStats.SlopeMean;
			summary[(index * 4) + 3] = cluster.MeanStats.Roughness;
		}

		using FileStream stream = File.Create(path);
		using ZipArchive archive = new(stream, ZipArchiveMode.Create, leaveOpen: false);
		WriteNpyEntry(archive, "normalized_centroids", "<f4", [count, profileSize, profileSize], ToBytes(normalizedCentroids));
		WriteNpyEntry(archive, "absolute_centroids", "<f4", [count, profileSize, profileSize], ToBytes(absoluteCentroids));
		WriteNpyEntry(archive, "profile_ids", "<i4", [profileIds.Length], ToBytes(profileIds));
		WriteNpyEntry(archive, "cluster_sizes", "<i4", [clusterSizes.Length], ToBytes(clusterSizes));
		WriteNpyEntry(archive, "summary_features", "<f4", [count, 4], ToBytes(summary));
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

	private readonly record struct HeightProfileOptions(
		string InputDirectory,
		string OutputDirectory,
		int DictionarySize,
		int MinOccurrences,
		int ProfileSize,
		int MaxIterations,
		int ExampleLimit,
		int Seed);

	private readonly record struct NpyPayload(string Descr, int[] Shape, byte[] Data);

	private readonly record struct FloatTensor2(int Height, int Width, float[] Values)
	{
		public float this[int y, int x] => Values[(y * Width) + x];
	}

	private readonly record struct HeightStats(float MeanElevation, float Relief, float SlopeMean, float Roughness);
	private sealed record HeightProfileSample(string TileName, string Path, float[] DownsampledHeight, float[] NormalizedProfile, float[] Features, HeightStats Stats);
	private sealed record HeightProfileExample(string TileName, string Path, string Archetype, HeightStats Stats);
	private sealed record HeightProfileCluster(
		int ProfileId,
		int RawClusterId,
		string ProfileHash,
		string Archetype,
		int ClusterSize,
		HeightStats MeanStats,
		float[] NormalizedCentroid,
		float[] AbsoluteCentroid,
		List<HeightProfileExample> Examples);

	private sealed record HeightProfileMiningResult(List<HeightProfileCluster> Clusters, int[] Labels, float[] FeatureMean, float[] FeatureStd);
	private sealed record SkippedShard(string TileName, string Path, string Reason);
}
