using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace WowViewer.Tool.Converter;

internal static class V10McalCompositionCommand
{
	private const int ChunkSize = 64;
	private const int LayerCount = 4;
	private static readonly byte[] NpyMagic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

	public static void Run(string[] args)
	{
		try
		{
			CompositionMiningOptions options = ParseOptions(args);
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

			Dictionary<string, CompositionAccumulator> compositions = new(StringComparer.Ordinal);
			List<SkippedShard> skipped = [];
			int tilesRead = 0;
			int chunksRead = 0;
			int candidateCount = 0;

			foreach (string npzPath in npzFiles)
			{
				string tileName = NormalizeTileName(Path.GetFileNameWithoutExtension(npzPath));
				if (!TryLoadTile(npzPath, out TileCompositionData tile, out string? skipReason))
				{
					skipped.Add(new SkippedShard(tileName, npzPath, skipReason ?? "missing_mcal_alpha_pack_256"));
					continue;
				}

				tilesRead++;
				for (int chunkY = 0; chunkY < tile.Alpha.Height / ChunkSize; chunkY++)
				{
					for (int chunkX = 0; chunkX < tile.Alpha.Width / ChunkSize; chunkX++)
					{
						chunksRead++;
						CompositionSample? sample = ExtractChunkComposition(tile, tileName, chunkX, chunkY, options);
						if (sample is null)
							continue;

						candidateCount++;
						if (!compositions.TryGetValue(sample.SignatureKey, out CompositionAccumulator? accumulator))
						{
							accumulator = new CompositionAccumulator(sample.SignatureKey, sample.CompositionHash);
							compositions.Add(sample.SignatureKey, accumulator);
						}

						accumulator.Add(sample, options.ExampleLimit);
					}
				}
			}

			List<CompositionAccumulator> retained = compositions.Values
				.Where(accumulator => accumulator.Frequency >= options.MinOccurrences)
				.OrderByDescending(static accumulator => accumulator.Frequency)
				.ThenBy(static accumulator => accumulator.SignatureKey, StringComparer.Ordinal)
				.Take(options.DictionarySize)
				.ToList();

			SaveDictionary(options, npzFiles.Count, tilesRead, chunksRead, candidateCount, compositions.Count, retained, skipped);
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"Error: {ex.Message}");
			Environment.ExitCode = 1;
		}
	}

	private static CompositionMiningOptions ParseOptions(string[] args)
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

		return new CompositionMiningOptions(
			InputDirectory: inputDirectory,
			OutputDirectory: Path.GetFullPath(outputDir),
			MinOccurrences: Math.Max(1, GetIntOption(args, "--min-occurrences", "-m") ?? 2),
			DictionarySize: Math.Max(1, GetIntOption(args, "--dictionary-size", "-d") ?? 128),
			ExampleLimit: Math.Max(1, GetIntOption(args, "--example-limit", "-e") ?? 8),
			MinActiveLayers: Math.Clamp(GetIntOption(args, "--min-active-layers", "-a") ?? 2, 1, LayerCount),
			MinLayerStd: Math.Max(0f, GetFloatOption(args, "--min-layer-std", "-s") ?? 0.03f),
			MinGradient: Math.Max(0f, GetFloatOption(args, "--min-gradient", "-g") ?? 0.015f));
	}

	private static CompositionSample? ExtractChunkComposition(TileCompositionData tile, string tileName, int chunkX, int chunkY, CompositionMiningOptions options)
	{
		int x0 = chunkX * ChunkSize;
		int y0 = chunkY * ChunkSize;
		LayerCompositionStats[] layerStats = new LayerCompositionStats[LayerCount];
		float[] patch = new float[ChunkSize * ChunkSize * LayerCount];
		for (int layer = 0; layer < LayerCount; layer++)
			layerStats[layer] = ComputeLayerStats(tile.Alpha, x0, y0, layer, patch);

		int[] activeLayers = Enumerable.Range(0, LayerCount)
			.Where(layer => layerStats[layer].Std >= options.MinLayerStd || layerStats[layer].GradientMean >= options.MinGradient)
			.OrderByDescending(layer => layerStats[layer].Coverage)
			.ThenByDescending(layer => layerStats[layer].GradientMean)
			.ToArray();
		if (activeLayers.Length < options.MinActiveLayers)
			return null;

		float totalGradient = layerStats.Sum(static stats => stats.GradientMean);
		if (totalGradient < options.MinGradient)
			return null;

		int dominantLayer = activeLayers[0];
		int secondaryLayer = activeLayers.Length > 1 ? activeLayers[1] : -1;
		int tertiaryLayer = activeLayers.Length > 2 ? activeLayers[2] : -1;
		HeightCompositionStats heightStats = tile.Height is null
			? HeightCompositionStats.Empty
			: ComputeHeightStats(tile.Height.Value, chunkX, chunkY);

		string signatureKey = BuildSignatureKey(layerStats, activeLayers, heightStats);
		string compositionHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(signatureKey)))[..16].ToLowerInvariant();

		return new CompositionSample(
			TileName: tileName,
			ChunkX: chunkX,
			ChunkY: chunkY,
			SignatureKey: signatureKey,
			CompositionHash: compositionHash,
			ActiveLayers: activeLayers,
			DominantLayer: dominantLayer,
			SecondaryLayer: secondaryLayer,
			TertiaryLayer: tertiaryLayer,
			TotalGradient: totalGradient,
			HeightStats: heightStats,
			LayerStats: layerStats,
			Patch: patch);
	}

	private static LayerCompositionStats ComputeLayerStats(FloatTensor3 alpha, int x0, int y0, int layer, float[] patch)
	{
		double sum = 0d;
		double sumSquares = 0d;
		double gradientSum = 0d;
		int gradientCount = 0;
		int occupied = 0;
		double[] quadrantSums = new double[4];

		for (int y = 0; y < ChunkSize; y++)
		{
			for (int x = 0; x < ChunkSize; x++)
			{
				float value = alpha[y0 + y, x0 + x, layer];
				patch[((y * ChunkSize) + x) * LayerCount + layer] = value;
				sum += value;
				sumSquares += value * value;
				if (value > 0.05f)
					occupied++;

				int quadrant = (y >= ChunkSize / 2 ? 2 : 0) + (x >= ChunkSize / 2 ? 1 : 0);
				quadrantSums[quadrant] += value;

				if (x + 1 < ChunkSize)
				{
					gradientSum += Math.Abs(value - alpha[y0 + y, x0 + x + 1, layer]);
					gradientCount++;
				}
				if (y + 1 < ChunkSize)
				{
					gradientSum += Math.Abs(value - alpha[y0 + y + 1, x0 + x, layer]);
					gradientCount++;
				}
			}
		}

		int sampleCount = ChunkSize * ChunkSize;
		float mean = (float)(sum / sampleCount);
		float std = (float)Math.Sqrt(Math.Max(0d, (sumSquares / sampleCount) - (mean * mean)));
		float gradient = gradientCount == 0 ? 0f : (float)(gradientSum / gradientCount);
		float coverage = occupied / (float)sampleCount;
		float quadrantImbalance = ComputeQuadrantImbalance(quadrantSums);
		return new LayerCompositionStats(mean, std, coverage, gradient, quadrantImbalance);
	}

	private static float ComputeQuadrantImbalance(double[] quadrantSums)
	{
		double total = quadrantSums.Sum();
		if (total <= 1e-8d)
			return 0f;

		double expected = total / quadrantSums.Length;
		double maxDelta = quadrantSums.Max(value => Math.Abs(value - expected));
		return (float)(maxDelta / total);
	}

	private static HeightCompositionStats ComputeHeightStats(FloatTensor2 height, int chunkX, int chunkY)
	{
		int x0 = chunkX * 16;
		int y0 = chunkY * 16;
		int x1 = Math.Min(height.Width - 1, x0 + 16);
		int y1 = Math.Min(height.Height - 1, y0 + 16);
		float min = float.MaxValue;
		float max = float.MinValue;
		double sum = 0d;
		double sumSquares = 0d;
		double slopeSum = 0d;
		int count = 0;

		for (int y = y0; y <= y1; y++)
		{
			for (int x = x0; x <= x1; x++)
			{
				float value = height[y, x];
				min = Math.Min(min, value);
				max = Math.Max(max, value);
				sum += value;
				sumSquares += value * value;

				float left = height[y, Math.Max(x0, x - 1)];
				float right = height[y, Math.Min(x1, x + 1)];
				float up = height[Math.Max(y0, y - 1), x];
				float down = height[Math.Min(y1, y + 1), x];
				float dx = (right - left) * 0.5f;
				float dy = (down - up) * 0.5f;
				slopeSum += Math.Sqrt((dx * dx) + (dy * dy));
				count++;
			}
		}

		float mean = (float)(sum / count);
		float roughness = (float)Math.Sqrt(Math.Max(0d, (sumSquares / count) - (mean * mean)));
		return new HeightCompositionStats(max - min, (float)(slopeSum / count), roughness);
	}

	private static string BuildSignatureKey(LayerCompositionStats[] layerStats, int[] activeLayers, HeightCompositionStats heightStats)
	{
		string active = string.Join("+", activeLayers.Take(3));
		string coverage = string.Join(",", activeLayers.Take(3).Select(layer => Quantize(layerStats[layer].Coverage, 0.1f)));
		string gradients = string.Join(",", activeLayers.Take(3).Select(layer => Quantize(layerStats[layer].GradientMean, 0.02f)));
		string balance = string.Join(",", activeLayers.Take(3).Select(layer => Quantize(layerStats[layer].QuadrantImbalance, 0.1f)));
		string terrain = $"relief={Quantize(heightStats.Relief, 10f)};slope={Quantize(heightStats.SlopeMean, 1f)};rough={Quantize(heightStats.Roughness, 5f)}";
		return $"layers={active};coverage={coverage};gradient={gradients};balance={balance};{terrain}";
	}

	private static string Quantize(float value, float step)
	{
		if (step <= 0f)
			return value.ToString("0.###", CultureInfo.InvariantCulture);

		float quantized = MathF.Round(value / step) * step;
		return quantized.ToString("0.###", CultureInfo.InvariantCulture);
	}

	private static bool TryLoadTile(string npzPath, out TileCompositionData tile, out string? skipReason)
	{
		tile = default;
		skipReason = null;
		using FileStream stream = File.OpenRead(npzPath);
		using ZipArchive archive = new(stream, ZipArchiveMode.Read, leaveOpen: false);

		if (!TryReadFloatTensor3(archive, "mcal_alpha_pack_256", out FloatTensor3 alpha))
		{
			skipReason = "missing_mcal_alpha_pack_256";
			return false;
		}

		if (alpha.Height < ChunkSize || alpha.Width < ChunkSize || alpha.Channels < LayerCount)
		{
			skipReason = $"unsupported_mcal_shape_{alpha.Height}x{alpha.Width}x{alpha.Channels}";
			return false;
		}

		FloatTensor2? height = TryReadFloatTensor2(archive, "height_257", out FloatTensor2 loadedHeight)
			? loadedHeight
			: null;

		tile = new TileCompositionData(alpha, height);
		return true;
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
		if (bytes.Length < 10 || !bytes.AsSpan(0, NpyMagic.Length).SequenceEqual(NpyMagic))
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
		CompositionMiningOptions options,
		int discoveredShardCount,
		int tilesRead,
		int chunksRead,
		int candidateCount,
		int rawCompositionCount,
		List<CompositionAccumulator> retained,
		List<SkippedShard> skipped)
	{
		Directory.CreateDirectory(options.OutputDirectory);
		string jsonPath = Path.Combine(options.OutputDirectory, "mcal_composition_dictionary.json");
		string npzPath = Path.Combine(options.OutputDirectory, "mcal_composition_dictionary.npz");

		var payload = new
		{
			schema_version = "v10-mcal-composition-dictionary.v1",
			generated_utc = DateTimeOffset.UtcNow,
			input_dir = options.InputDirectory,
			discovered_shard_count = discoveredShardCount,
			tiles_read = tilesRead,
			chunks_read = chunksRead,
			candidate_composition_count = candidateCount,
			raw_composition_count = rawCompositionCount,
			retained_composition_count = retained.Count,
			min_occurrences = options.MinOccurrences,
			min_active_layers = options.MinActiveLayers,
			min_layer_std = options.MinLayerStd,
			min_gradient = options.MinGradient,
			dictionary = retained.Select(static accumulator => new
			{
				composition_id = accumulator.CompositionId,
				composition_hash = accumulator.CompositionHash,
				signature_key = accumulator.SignatureKey,
				frequency = accumulator.Frequency,
				tile_count = accumulator.TileNames.Count,
				dominant_layers = accumulator.DominantLayerDistribution.OrderByDescending(static entry => entry.Value).Select(static entry => new object[] { entry.Key, entry.Value }),
				active_layer_distribution = accumulator.ActiveLayerDistribution.OrderByDescending(static entry => entry.Value).Select(static entry => new object[] { entry.Key, entry.Value }),
				mean_total_gradient = accumulator.MeanTotalGradient,
				mean_height_relief = accumulator.MeanHeightRelief,
				mean_height_slope = accumulator.MeanHeightSlope,
				example_chunks = accumulator.Examples.Select(static example => new
				{
					tile_name = example.TileName,
					chunk_x = example.ChunkX,
					chunk_y = example.ChunkY,
					active_layers = example.ActiveLayers,
					dominant_layer = example.DominantLayer,
					secondary_layer = example.SecondaryLayer,
					tertiary_layer = example.TertiaryLayer,
				}),
			}),
			skipped_shards = skipped.Select(static shard => new
			{
				tile_name = shard.TileName,
				path = shard.Path,
				reason = shard.Reason,
			}),
		};

		File.WriteAllText(jsonPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
		WriteDictionaryNpz(npzPath, retained);

		Console.WriteLine("WowViewer.Tool.Converter mine-v10-mcal-compositions report");
		Console.WriteLine($"InputDir: {options.InputDirectory}");
		Console.WriteLine($"OutputDir: {options.OutputDirectory}");
		Console.WriteLine($"Shards: {discoveredShardCount}");
		Console.WriteLine($"TilesRead: {tilesRead}");
		Console.WriteLine($"ChunksRead: {chunksRead}");
		Console.WriteLine($"CandidateCompositions: {candidateCount}");
		Console.WriteLine($"RawCompositions: {rawCompositionCount}");
		Console.WriteLine($"RetainedCompositions: {retained.Count}");
		Console.WriteLine($"Dictionary: {jsonPath}");
	}

	private static void WriteDictionaryNpz(string path, List<CompositionAccumulator> retained)
	{
		int count = retained.Count;
		float[] centroids = new float[count * ChunkSize * ChunkSize * LayerCount];
		int[] compositionIds = new int[count];
		int[] frequencies = new int[count];
		float[] summary = new float[count * 3];

		for (int index = 0; index < retained.Count; index++)
		{
			CompositionAccumulator accumulator = retained[index];
			compositionIds[index] = accumulator.CompositionId;
			frequencies[index] = accumulator.Frequency;
			float[] centroid = accumulator.BuildCentroid();
			Array.Copy(centroid, 0, centroids, index * ChunkSize * ChunkSize * LayerCount, centroid.Length);
			summary[(index * 3) + 0] = accumulator.MeanTotalGradient;
			summary[(index * 3) + 1] = accumulator.MeanHeightRelief;
			summary[(index * 3) + 2] = accumulator.MeanHeightSlope;
		}

		using FileStream stream = File.Create(path);
		using ZipArchive archive = new(stream, ZipArchiveMode.Create, leaveOpen: false);
		WriteNpyEntry(archive, "centroids", "<f4", [count, ChunkSize, ChunkSize, LayerCount], ToBytes(centroids));
		WriteNpyEntry(archive, "composition_ids", "<i4", [compositionIds.Length], ToBytes(compositionIds));
		WriteNpyEntry(archive, "frequencies", "<i4", [frequencies.Length], ToBytes(frequencies));
		WriteNpyEntry(archive, "summary_features", "<f4", [count, 3], ToBytes(summary));
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

	private readonly record struct CompositionMiningOptions(
		string InputDirectory,
		string OutputDirectory,
		int MinOccurrences,
		int DictionarySize,
		int ExampleLimit,
		int MinActiveLayers,
		float MinLayerStd,
		float MinGradient);

	private readonly record struct NpyPayload(string Descr, int[] Shape, byte[] Data);

	private readonly record struct FloatTensor2(int Height, int Width, float[] Values)
	{
		public float this[int y, int x] => Values[(y * Width) + x];
	}

	private readonly record struct FloatTensor3(int Height, int Width, int Channels, float[] Values)
	{
		public float this[int y, int x, int channel] => Values[((y * Width) + x) * Channels + channel];
	}

	private readonly record struct TileCompositionData(FloatTensor3 Alpha, FloatTensor2? Height);
	private readonly record struct LayerCompositionStats(float Mean, float Std, float Coverage, float GradientMean, float QuadrantImbalance);
	private readonly record struct HeightCompositionStats(float Relief, float SlopeMean, float Roughness)
	{
		public static HeightCompositionStats Empty => new(0f, 0f, 0f);
	}

	private sealed record CompositionSample(
		string TileName,
		int ChunkX,
		int ChunkY,
		string SignatureKey,
		string CompositionHash,
		int[] ActiveLayers,
		int DominantLayer,
		int SecondaryLayer,
		int TertiaryLayer,
		float TotalGradient,
		HeightCompositionStats HeightStats,
		LayerCompositionStats[] LayerStats,
		float[] Patch);

	private sealed record CompositionExample(
		string TileName,
		int ChunkX,
		int ChunkY,
		int[] ActiveLayers,
		int DominantLayer,
		int SecondaryLayer,
		int TertiaryLayer);

	private sealed record SkippedShard(string TileName, string Path, string Reason);

	private sealed class CompositionAccumulator
	{
		private readonly HashSet<string> _tileNames = new(StringComparer.OrdinalIgnoreCase);
		private readonly Dictionary<int, int> _dominantLayerDistribution = [];
		private readonly Dictionary<string, int> _activeLayerDistribution = new(StringComparer.Ordinal);
		private readonly float[] _patchSum = new float[ChunkSize * ChunkSize * LayerCount];
		private float _totalGradientSum;
		private float _heightReliefSum;
		private float _heightSlopeSum;

		public CompositionAccumulator(string signatureKey, string compositionHash)
		{
			SignatureKey = signatureKey;
			CompositionHash = compositionHash;
			CompositionId = BinaryPrimitives.ReadInt32LittleEndian(SHA256.HashData(Encoding.UTF8.GetBytes(signatureKey)).AsSpan(0, 4)) & int.MaxValue;
		}

		public string SignatureKey { get; }
		public string CompositionHash { get; }
		public int CompositionId { get; }
		public int Frequency { get; private set; }
		public IReadOnlySet<string> TileNames => _tileNames;
		public IReadOnlyDictionary<int, int> DominantLayerDistribution => _dominantLayerDistribution;
		public IReadOnlyDictionary<string, int> ActiveLayerDistribution => _activeLayerDistribution;
		public List<CompositionExample> Examples { get; } = [];
		public float MeanTotalGradient => Frequency == 0 ? 0f : _totalGradientSum / Frequency;
		public float MeanHeightRelief => Frequency == 0 ? 0f : _heightReliefSum / Frequency;
		public float MeanHeightSlope => Frequency == 0 ? 0f : _heightSlopeSum / Frequency;

		public void Add(CompositionSample sample, int exampleLimit)
		{
			Frequency++;
			_tileNames.Add(sample.TileName);
			_totalGradientSum += sample.TotalGradient;
			_heightReliefSum += sample.HeightStats.Relief;
			_heightSlopeSum += sample.HeightStats.SlopeMean;

			_dominantLayerDistribution.TryGetValue(sample.DominantLayer, out int dominantCount);
			_dominantLayerDistribution[sample.DominantLayer] = dominantCount + 1;

			string activeKey = string.Join("+", sample.ActiveLayers);
			_activeLayerDistribution.TryGetValue(activeKey, out int activeCount);
			_activeLayerDistribution[activeKey] = activeCount + 1;

			for (int index = 0; index < _patchSum.Length; index++)
				_patchSum[index] += sample.Patch[index];

			if (Examples.Count < exampleLimit)
			{
				Examples.Add(new CompositionExample(
					sample.TileName,
					sample.ChunkX,
					sample.ChunkY,
					[.. sample.ActiveLayers],
					sample.DominantLayer,
					sample.SecondaryLayer,
					sample.TertiaryLayer));
			}
		}

		public float[] BuildCentroid()
		{
			float[] centroid = new float[_patchSum.Length];
			if (Frequency == 0)
				return centroid;

			for (int index = 0; index < centroid.Length; index++)
				centroid[index] = _patchSum[index] / Frequency;
			return centroid;
		}
	}
}
