using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace WowViewer.Tool.Converter;

internal static class V10MclyLabelManifestCommand
{
	private const int ChunkGridSize = 16;
	private const int LayerCount = 4;
	private const int IgnoreIndex = -100;
	private static readonly byte[] NpyMagic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

	public static void Run(string[] args)
	{
		try
		{
			LabelManifestOptions options = ParseOptions(args);
			List<string> npzPaths = ResolveNpzPaths(options.InputPath);
			if (npzPaths.Count == 0)
			{
				Console.Error.WriteLine($"Error: no .npz shards found from {options.InputPath}");
				Environment.ExitCode = 1;
				return;
			}

			IReadOnlyList<DictionaryEntry> dictionary = LoadDictionary(options.DictionaryPath);
			Dictionary<string, DictionaryEntry> dictionaryByKey = dictionary
				.Where(static entry => !string.IsNullOrWhiteSpace(entry.CombinationKey))
				.ToDictionary(static entry => entry.CombinationKey, static entry => entry, StringComparer.Ordinal);

			List<LabeledTileEntry> entries = [];
			List<SkippedShard> skipped = [];
			Dictionary<int, LabelUsageAccumulator> labelUsage = [];
			int retainedChunkCount = 0;
			int discoveredWithMclyCount = 0;

			foreach (string npzPath in npzPaths)
			{
				string tileName = NormalizeTileName(Path.GetFileNameWithoutExtension(npzPath));
				if (!TryLabelShard(npzPath, dictionaryByKey, out LabeledShardLabels labels, out string? skipReason))
				{
					skipped.Add(new SkippedShard(tileName, npzPath, skipReason ?? "missing_mcly_texture_ids"));
					continue;
				}

				discoveredWithMclyCount++;
				if (labels.RetainedChunkCount < options.MinRetainedChunks)
				{
					skipped.Add(new SkippedShard(tileName, npzPath, $"retained_chunks_below_minimum:{labels.RetainedChunkCount}"));
					continue;
				}

				retainedChunkCount += labels.RetainedChunkCount;
				foreach ((int labelIndex, int count) in labels.LabelCounts)
				{
					if (!labelUsage.TryGetValue(labelIndex, out LabelUsageAccumulator? accumulator))
					{
						DictionaryEntry dictionaryEntry = dictionary[labelIndex];
						accumulator = new LabelUsageAccumulator(dictionaryEntry);
						labelUsage[labelIndex] = accumulator;
					}

					accumulator.AddTile(tileName, count);
				}

				DictionaryEntry dominantEntry = dictionary[labels.DominantLabelIndex];
				entries.Add(new LabeledTileEntry(
					TileName: labels.TileName ?? tileName,
					ShardPath: npzPath,
					RetainedChunkCount: labels.RetainedChunkCount,
					IgnoredChunkCount: (ChunkGridSize * ChunkGridSize) - labels.RetainedChunkCount,
					DominantDictionaryLabelIndex: labels.DominantLabelIndex,
					DominantCombinationHash: dominantEntry.CombinationHash,
					DominantCombinationKey: dominantEntry.CombinationKey,
					DominantChunkCount: labels.DominantChunkCount,
					DominantFraction: labels.DominantChunkCount / (double)(ChunkGridSize * ChunkGridSize),
					LabelGrid16: labels.LabelGrid));
			}

			SaveManifest(options, npzPaths.Count, discoveredWithMclyCount, entries, skipped, retainedChunkCount, labelUsage);
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"Error: {ex.Message}");
			Environment.ExitCode = 1;
		}
	}

	private static LabelManifestOptions ParseOptions(string[] args)
	{
		string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
		string? dictionary = GetOption(args, "--dictionary", "-d");
		string? output = GetOption(args, "--output", "-o");
		if (string.IsNullOrWhiteSpace(input))
			throw new InvalidOperationException("--input <stage1-manifest|npz-dir|npz> is required.");
		if (string.IsNullOrWhiteSpace(dictionary))
			throw new InvalidOperationException("--dictionary <mclay_dictionary.json> is required.");
		if (string.IsNullOrWhiteSpace(output))
			throw new InvalidOperationException("--output <label-manifest.json> is required.");

		string inputPath = Path.GetFullPath(input);
		if (!File.Exists(inputPath) && !Directory.Exists(inputPath))
			throw new FileNotFoundException($"Input '{inputPath}' does not exist.", inputPath);

		string dictionaryPath = Path.GetFullPath(dictionary);
		if (!File.Exists(dictionaryPath))
			throw new FileNotFoundException($"Dictionary '{dictionaryPath}' does not exist.", dictionaryPath);

		return new LabelManifestOptions(
			InputPath: inputPath,
			DictionaryPath: dictionaryPath,
			OutputPath: Path.GetFullPath(output),
			MinRetainedChunks: Math.Max(1, GetIntOption(args, "--min-retained-chunks", "-m") ?? 1));
	}

	private static List<string> ResolveNpzPaths(string inputPath)
	{
		if (Directory.Exists(inputPath))
		{
			return Directory
				.EnumerateFiles(inputPath, "*.npz", SearchOption.AllDirectories)
				.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase)
				.Select(Path.GetFullPath)
				.ToList();
		}

		if (Path.GetExtension(inputPath).Equals(".npz", StringComparison.OrdinalIgnoreCase))
			return [Path.GetFullPath(inputPath)];

		using JsonDocument document = JsonDocument.Parse(File.ReadAllText(inputPath));
		List<string> collected = [];
		CollectNpzPaths(document.RootElement, Path.GetDirectoryName(inputPath) ?? Environment.CurrentDirectory, collected);
		return collected
			.Where(File.Exists)
			.Select(Path.GetFullPath)
			.Distinct(StringComparer.OrdinalIgnoreCase)
			.OrderBy(static path => path, StringComparer.OrdinalIgnoreCase)
			.ToList();
	}

	private static void CollectNpzPaths(JsonElement element, string baseDirectory, List<string> collected)
	{
		switch (element.ValueKind)
		{
			case JsonValueKind.Object:
				foreach (JsonProperty property in element.EnumerateObject())
					CollectNpzPaths(property.Value, baseDirectory, collected);
				break;
			case JsonValueKind.Array:
				foreach (JsonElement item in element.EnumerateArray())
					CollectNpzPaths(item, baseDirectory, collected);
				break;
			case JsonValueKind.String:
				string? value = element.GetString();
				if (!string.IsNullOrWhiteSpace(value) && value.EndsWith(".npz", StringComparison.OrdinalIgnoreCase))
					collected.Add(Path.IsPathRooted(value) ? value : Path.GetFullPath(Path.Combine(baseDirectory, value)));
				break;
		}
	}

	private static IReadOnlyList<DictionaryEntry> LoadDictionary(string dictionaryPath)
	{
		using JsonDocument document = JsonDocument.Parse(File.ReadAllText(dictionaryPath));
		if (!document.RootElement.TryGetProperty("dictionary", out JsonElement dictionaryElement)
			|| dictionaryElement.ValueKind != JsonValueKind.Array)
		{
			throw new InvalidDataException($"Dictionary '{dictionaryPath}' does not contain a dictionary array.");
		}

		List<DictionaryEntry> entries = [];
		int index = 0;
		foreach (JsonElement entryElement in dictionaryElement.EnumerateArray())
		{
			string combinationKey = GetString(entryElement, "combination_key");
			string[] textureNames = GetStringArray(entryElement, "texture_names")
				.Select(NormalizeTexturePath)
				.Take(LayerCount)
				.ToArray();
			if (string.IsNullOrWhiteSpace(combinationKey) && textureNames.Length > 0)
				combinationKey = string.Join("|", textureNames.Concat(Enumerable.Repeat(string.Empty, LayerCount)).Take(LayerCount));

			entries.Add(new DictionaryEntry(
				LabelIndex: index,
				CombinationHash: GetString(entryElement, "combination_hash"),
				CombinationKey: combinationKey,
				TextureNames: textureNames,
				Frequency: GetInt(entryElement, "frequency"),
				InferredBiomeTag: GetString(entryElement, "inferred_biome_tag", "unknown")));
			index++;
		}

		if (entries.Count == 0)
			throw new InvalidDataException($"Dictionary '{dictionaryPath}' contains no entries.");

		return entries;
	}

	private static bool TryLabelShard(
		string npzPath,
		Dictionary<string, DictionaryEntry> dictionaryByKey,
		out LabeledShardLabels labels,
		out string? skipReason)
	{
		labels = LabeledShardLabels.Empty;
		skipReason = null;

		using FileStream stream = File.OpenRead(npzPath);
		using ZipArchive archive = new(stream, ZipArchiveMode.Read, leaveOpen: false);
		IReadOnlyList<string> textureNames = ReadMetadataTextureNames(archive, out string? tileName);
		if (!TryReadMclyTextureIds(archive, out IntTensor3 textureIds, out skipReason))
			return false;

		if (textureIds.Height != ChunkGridSize || textureIds.Width != ChunkGridSize)
		{
			skipReason = $"unsupported_mcly_texture_ids_shape:{textureIds.Height}x{textureIds.Width}x{textureIds.Channels}";
			return false;
		}

		int[][] labelGrid = new int[ChunkGridSize][];
		Dictionary<int, int> labelCounts = [];
		int retainedChunkCount = 0;
		for (int y = 0; y < ChunkGridSize; y++)
		{
			labelGrid[y] = new int[ChunkGridSize];
			for (int x = 0; x < ChunkGridSize; x++)
			{
				string key = BuildCombinationKey(textureIds, textureNames, x, y);
				if (!dictionaryByKey.TryGetValue(key, out DictionaryEntry? entry))
				{
					labelGrid[y][x] = IgnoreIndex;
					continue;
				}

				labelGrid[y][x] = entry.LabelIndex;
				retainedChunkCount++;
				labelCounts.TryGetValue(entry.LabelIndex, out int count);
				labelCounts[entry.LabelIndex] = count + 1;
			}
		}

		if (retainedChunkCount == 0)
		{
			skipReason = "no_retained_dictionary_labels";
			return false;
		}

		KeyValuePair<int, int> dominant = labelCounts
			.OrderByDescending(static entry => entry.Value)
			.ThenBy(static entry => entry.Key)
			.First();
		labels = new LabeledShardLabels(tileName, labelGrid, retainedChunkCount, dominant.Key, dominant.Value, labelCounts);
		return true;
	}

	private static IReadOnlyList<string> ReadMetadataTextureNames(ZipArchive archive, out string? tileName)
	{
		tileName = null;
		ZipArchiveEntry? entry = archive.GetEntry("metadata.json");
		if (entry is null)
			return Array.Empty<string>();

		using Stream stream = entry.Open();
		using JsonDocument document = JsonDocument.Parse(stream);
		if (document.RootElement.TryGetProperty("tile_name", out JsonElement tileElement)
			&& tileElement.ValueKind == JsonValueKind.String)
		{
			tileName = tileElement.GetString();
		}

		if (!document.RootElement.TryGetProperty("mcly_texture_names", out JsonElement namesElement)
			|| namesElement.ValueKind != JsonValueKind.Array)
		{
			return Array.Empty<string>();
		}

		List<string> names = new(namesElement.GetArrayLength());
		foreach (JsonElement nameElement in namesElement.EnumerateArray())
		{
			if (nameElement.ValueKind == JsonValueKind.String)
				names.Add(NormalizeTexturePath(nameElement.GetString() ?? string.Empty));
		}

		return names;
	}

	private static bool TryReadMclyTextureIds(ZipArchive archive, out IntTensor3 textureIds, out string? skipReason)
	{
		textureIds = default;
		skipReason = null;
		if (!TryReadNpyEntry(archive, "mcly_texture_ids", out NpyPayload payload))
		{
			skipReason = "missing_mcly_texture_ids";
			return false;
		}

		if (payload.Shape.Length != 3)
			throw new InvalidDataException($"mcly_texture_ids must be rank 3, but was rank {payload.Shape.Length}.");
		if (payload.Descr is not "<i4" and not "|i4" and not "<u4")
			throw new InvalidDataException($"mcly_texture_ids has unsupported dtype '{payload.Descr}'.");

		int count = payload.Shape.Aggregate(1, static (accumulator, dimension) => accumulator * dimension);
		if (payload.Data.Length < count * sizeof(int))
			throw new InvalidDataException("mcly_texture_ids is truncated.");

		int[] values = new int[count];
		for (int index = 0; index < count; index++)
			values[index] = BinaryPrimitives.ReadInt32LittleEndian(payload.Data.AsSpan(index * sizeof(int), sizeof(int)));

		textureIds = new IntTensor3(payload.Shape[0], payload.Shape[1], payload.Shape[2], values);
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

	private static string BuildCombinationKey(IntTensor3 textureIds, IReadOnlyList<string> textureNames, int x, int y)
	{
		string[] resolved = new string[LayerCount];
		for (int layer = 0; layer < LayerCount; layer++)
		{
			int textureId = layer < textureIds.Channels ? textureIds[y, x, layer] : -1;
			resolved[layer] = textureId >= 0 && textureId < textureNames.Count
				? NormalizeTexturePath(textureNames[textureId])
				: textureId >= 0 ? "#" + textureId.ToString(CultureInfo.InvariantCulture) : string.Empty;
		}

		return string.Join("|", resolved);
	}

	private static void SaveManifest(
		LabelManifestOptions options,
		int discoveredShardCount,
		int discoveredWithMclyCount,
		List<LabeledTileEntry> entries,
		List<SkippedShard> skipped,
		int retainedChunkCount,
		Dictionary<int, LabelUsageAccumulator> labelUsage)
	{
		string? directory = Path.GetDirectoryName(options.OutputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		var payload = new
		{
			schema_version = "v10-mcly-label-manifest.v1",
			generated_utc = DateTimeOffset.UtcNow,
			input = options.InputPath,
			dictionary = options.DictionaryPath,
			discovered_shard_count = discoveredShardCount,
			shards_with_mcly_texture_ids = discoveredWithMclyCount,
			labeled_sample_count = entries.Count,
			retained_chunk_count = retainedChunkCount,
			active_label_count = labelUsage.Count,
			min_retained_chunks = options.MinRetainedChunks,
			ignore_index = IgnoreIndex,
			labels = labelUsage.Values
				.OrderByDescending(static usage => usage.ChunkCount)
				.ThenBy(static usage => usage.Entry.LabelIndex)
				.Select(static usage => new
				{
					dictionary_label_index = usage.Entry.LabelIndex,
					combination_hash = usage.Entry.CombinationHash,
					combination_key = usage.Entry.CombinationKey,
					texture_names = usage.Entry.TextureNames,
					inferred_biome_tag = usage.Entry.InferredBiomeTag,
					dictionary_frequency = usage.Entry.Frequency,
					chunk_count = usage.ChunkCount,
					tile_count = usage.TileNames.Count,
					example_tiles = usage.TileNames.OrderBy(static tile => tile, StringComparer.OrdinalIgnoreCase).Take(8),
				}),
			entries = entries,
			skipped_shards = skipped,
		};

		File.WriteAllText(options.OutputPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));

		Console.WriteLine("WowViewer.Tool.Converter label-v10-mcly report");
		Console.WriteLine($"Input: {options.InputPath}");
		Console.WriteLine($"Dictionary: {options.DictionaryPath}");
		Console.WriteLine($"Output: {options.OutputPath}");
		Console.WriteLine($"Shards: {discoveredShardCount}");
		Console.WriteLine($"ShardsWithMcly: {discoveredWithMclyCount}");
		Console.WriteLine($"LabeledSamples: {entries.Count}");
		Console.WriteLine($"RetainedChunks: {retainedChunkCount}");
		Console.WriteLine($"ActiveLabels: {labelUsage.Count}");
		Console.WriteLine($"Skipped: {skipped.Count}");
	}

	private static string NormalizeTileName(string fileStem)
	{
		return fileStem.EndsWith("_v10", StringComparison.OrdinalIgnoreCase)
			? fileStem[..^4]
			: fileStem;
	}

	private static string NormalizeTexturePath(string texturePath)
	{
		return texturePath.Replace('\\', '/').Trim();
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

	private static string GetString(JsonElement element, string name, string fallback = "")
	{
		return element.TryGetProperty(name, out JsonElement property) && property.ValueKind == JsonValueKind.String
			? property.GetString() ?? fallback
			: fallback;
	}

	private static string[] GetStringArray(JsonElement element, string name)
	{
		if (!element.TryGetProperty(name, out JsonElement property) || property.ValueKind != JsonValueKind.Array)
			return [];

		List<string> values = [];
		foreach (JsonElement item in property.EnumerateArray())
		{
			if (item.ValueKind == JsonValueKind.String)
				values.Add(item.GetString() ?? string.Empty);
		}

		return [.. values];
	}

	private static int GetInt(JsonElement element, string name)
	{
		return element.TryGetProperty(name, out JsonElement property) && property.TryGetInt32(out int value)
			? value
			: 0;
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

	private readonly record struct LabelManifestOptions(
		string InputPath,
		string DictionaryPath,
		string OutputPath,
		int MinRetainedChunks);

	private readonly record struct NpyPayload(string Descr, int[] Shape, byte[] Data);

	private readonly record struct IntTensor3(int Height, int Width, int Channels, int[] Values)
	{
		public int this[int y, int x, int channel] => Values[((y * Width) + x) * Channels + channel];
	}

	private sealed record DictionaryEntry(
		int LabelIndex,
		string CombinationHash,
		string CombinationKey,
		IReadOnlyList<string> TextureNames,
		int Frequency,
		string InferredBiomeTag);

	private sealed record LabeledShardLabels(
		string? TileName,
		int[][] LabelGrid,
		int RetainedChunkCount,
		int DominantLabelIndex,
		int DominantChunkCount,
		IReadOnlyDictionary<int, int> LabelCounts)
	{
		public static LabeledShardLabels Empty { get; } = new(null, [], 0, IgnoreIndex, 0, new Dictionary<int, int>());
	}

	private sealed record LabeledTileEntry(
		string TileName,
		string ShardPath,
		int RetainedChunkCount,
		int IgnoredChunkCount,
		int DominantDictionaryLabelIndex,
		string DominantCombinationHash,
		string DominantCombinationKey,
		int DominantChunkCount,
		double DominantFraction,
		int[][] LabelGrid16);

	private sealed record SkippedShard(string TileName, string Path, string Reason);

	private sealed class LabelUsageAccumulator(DictionaryEntry entry)
	{
		public DictionaryEntry Entry { get; } = entry;
		public int ChunkCount { get; private set; }
		public HashSet<string> TileNames { get; } = new(StringComparer.OrdinalIgnoreCase);

		public void AddTile(string tileName, int chunkCount)
		{
			ChunkCount += chunkCount;
			TileNames.Add(tileName);
		}
	}
}
