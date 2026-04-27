using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace WowViewer.Tool.Converter;

internal static class V10MclyDictionaryCommand
{
	private static readonly byte[] NpyMagic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

	public static void Run(string[] args)
	{
		try
		{
			MclyDictionaryOptions options = ParseOptions(args);
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

			Dictionary<MclyCombinationKey, MclyCombinationAccumulator> combinations = [];
			List<MclySkippedTile> skipped = [];
			int tilesRead = 0;
			int chunksRead = 0;

			foreach (string npzPath in npzFiles)
			{
				string tileName = NormalizeTileName(Path.GetFileNameWithoutExtension(npzPath));
				if (!TryLoadMclyTextureIds(npzPath, out IntTensor3 textureIds, out IReadOnlyList<string> textureNames, out string? skipReason))
				{
					skipped.Add(new MclySkippedTile(tileName, npzPath, skipReason ?? "missing_mcly_texture_ids"));
					continue;
				}

				tilesRead++;
				for (int chunkY = 0; chunkY < textureIds.Height; chunkY++)
				{
					for (int chunkX = 0; chunkX < textureIds.Width; chunkX++)
					{
						int[] ids =
						[
							textureIds[chunkY, chunkX, 0],
							textureIds.Channels > 1 ? textureIds[chunkY, chunkX, 1] : -1,
							textureIds.Channels > 2 ? textureIds[chunkY, chunkX, 2] : -1,
							textureIds.Channels > 3 ? textureIds[chunkY, chunkX, 3] : -1,
						];

						if (!options.IncludeEmpty && ids.All(static id => id < 0))
							continue;

						chunksRead++;
						string[] resolvedTextureNames = ResolveTextureNames(ids, textureNames);
						MclyCombinationKey key = MclyCombinationKey.Create(ids, resolvedTextureNames);
						if (!combinations.TryGetValue(key, out MclyCombinationAccumulator? accumulator))
						{
							accumulator = new MclyCombinationAccumulator(key);
							combinations[key] = accumulator;
						}

						accumulator.Add(tileName, chunkX, chunkY, ids, resolvedTextureNames, options.ExampleLimit);
					}
				}
			}

			List<MclyCombinationAccumulator> retained = combinations.Values
				.Where(accumulator => accumulator.Frequency >= options.MinOccurrences)
				.OrderByDescending(static accumulator => accumulator.Frequency)
				.ThenBy(static accumulator => accumulator.Key.ToStableText(), StringComparer.Ordinal)
				.ToList();

			SaveDictionary(options, npzFiles.Count, tilesRead, chunksRead, combinations.Count, retained, skipped);
		}
		catch (Exception ex)
		{
			Console.Error.WriteLine($"Error: {ex.Message}");
			Environment.ExitCode = 1;
		}
	}

	private static MclyDictionaryOptions ParseOptions(string[] args)
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

		return new MclyDictionaryOptions(
			InputDirectory: inputDirectory,
			OutputDirectory: Path.GetFullPath(outputDir),
			MinOccurrences: Math.Max(1, GetIntOption(args, "--min-occurrences", "-m") ?? 1),
			ExampleLimit: Math.Max(1, GetIntOption(args, "--example-limit", "-e") ?? 8),
			IncludeEmpty: HasFlag(args, "--include-empty"));
	}

	private static bool TryLoadMclyTextureIds(string npzPath, out IntTensor3 tensor, out IReadOnlyList<string> textureNames, out string? skipReason)
	{
		tensor = default;
		textureNames = Array.Empty<string>();
		skipReason = null;
		using FileStream stream = File.OpenRead(npzPath);
		using ZipArchive archive = new(stream, ZipArchiveMode.Read, leaveOpen: false);
		textureNames = ReadMetadataTextureNames(archive);

		if (!TryReadNpyEntry(archive, "mcly_texture_ids", out NpyPayload payload))
		{
			skipReason = "missing_mcly_texture_ids";
			return false;
		}

		if (payload.Shape.Length != 3)
			throw new InvalidDataException($"mcly_texture_ids in '{npzPath}' must be rank 3, but was rank {payload.Shape.Length}.");
		if (payload.Descr is not "<i4" and not "|i4" and not "<u4")
			throw new InvalidDataException($"mcly_texture_ids in '{npzPath}' has unsupported dtype '{payload.Descr}'.");

		int count = payload.Shape.Aggregate(1, static (accumulator, dimension) => accumulator * dimension);
		if (payload.Data.Length < count * sizeof(int))
			throw new InvalidDataException($"mcly_texture_ids in '{npzPath}' is truncated.");

		int[] values = new int[count];
		for (int index = 0; index < count; index++)
			values[index] = BinaryPrimitives.ReadInt32LittleEndian(payload.Data.AsSpan(index * sizeof(int), sizeof(int)));

		tensor = new IntTensor3(payload.Shape[0], payload.Shape[1], payload.Shape[2], values);
		return true;
	}

	private static IReadOnlyList<string> ReadMetadataTextureNames(ZipArchive archive)
	{
		ZipArchiveEntry? entry = archive.GetEntry("metadata.json");
		if (entry is null)
			return Array.Empty<string>();

		using Stream stream = entry.Open();
		using JsonDocument document = JsonDocument.Parse(stream);
		if (!document.RootElement.TryGetProperty("mcly_texture_names", out JsonElement namesElement)
			|| namesElement.ValueKind != JsonValueKind.Array)
		{
			return Array.Empty<string>();
		}

		List<string> names = new(namesElement.GetArrayLength());
		foreach (JsonElement nameElement in namesElement.EnumerateArray())
		{
			if (nameElement.ValueKind == JsonValueKind.String)
				names.Add(nameElement.GetString() ?? string.Empty);
		}

		return names;
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
		MclyDictionaryOptions options,
		int discoveredShardCount,
		int tilesRead,
		int chunksRead,
		int rawCombinationCount,
		List<MclyCombinationAccumulator> retained,
		List<MclySkippedTile> skipped)
	{
		Directory.CreateDirectory(options.OutputDirectory);
		var payload = new
		{
			schema_version = "v10-mcly-dictionary.v1",
			generated_utc = DateTimeOffset.UtcNow,
			input_dir = options.InputDirectory,
			discovered_shard_count = discoveredShardCount,
			tiles_read = tilesRead,
			chunks_read = chunksRead,
			raw_combination_count = rawCombinationCount,
			retained_combination_count = retained.Count,
			min_occurrences = options.MinOccurrences,
			dictionary = retained.Select(static accumulator =>
			{
				BiomeInference biome = InferBiomeTag(accumulator.Key.TextureNames);
				return new
				{
					combination_hash = accumulator.Key.Hash,
					combination_key = accumulator.Key.ToStableText(),
					texture_ids = accumulator.MostCommonTextureIds,
					texture_names = accumulator.Key.TextureNames,
					frequency = accumulator.Frequency,
					tile_count = accumulator.TileNames.Count,
					id_tuple_distribution = accumulator.IdTupleDistribution
						.OrderByDescending(static entry => entry.Value)
						.ThenBy(static entry => entry.Key, StringComparer.Ordinal)
						.Select(static entry => new object[] { entry.Key, entry.Value }),
					example_chunks = accumulator.Examples.Select(static example => new
					{
						tile_name = example.TileName,
						chunk_x = example.ChunkX,
						chunk_y = example.ChunkY,
						texture_ids = example.TextureIds,
					}),
					inferred_biome_tag = biome.Tag,
					inference_reason = biome.Reason,
				};
			}),
			skipped_tiles = skipped.Select(static tile => new
			{
				tile_name = tile.TileName,
				path = tile.Path,
				reason = tile.Reason,
			}),
		};

		string json = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
		string planPath = Path.Combine(options.OutputDirectory, "mclay_dictionary.json");
		string chunkPath = Path.Combine(options.OutputDirectory, "mcly_dictionary.json");
		File.WriteAllText(planPath, json);
		File.WriteAllText(chunkPath, json);

		Console.WriteLine("WowViewer.Tool.Converter mine-v10-mcly report");
		Console.WriteLine($"InputDir: {options.InputDirectory}");
		Console.WriteLine($"OutputDir: {options.OutputDirectory}");
		Console.WriteLine($"Shards: {discoveredShardCount}");
		Console.WriteLine($"TilesRead: {tilesRead}");
		Console.WriteLine($"ChunksRead: {chunksRead}");
		Console.WriteLine($"RawCombinations: {rawCombinationCount}");
		Console.WriteLine($"RetainedCombinations: {retained.Count}");
		Console.WriteLine($"Dictionary: {planPath}");
	}

	private static string NormalizeTileName(string fileStem)
	{
		return fileStem.EndsWith("_v10", StringComparison.OrdinalIgnoreCase)
			? fileStem[..^4]
			: fileStem;
	}

	private static string[] ResolveTextureNames(int[] textureIds, IReadOnlyList<string> textureNames)
	{
		string[] resolved = new string[4];
		for (int index = 0; index < resolved.Length; index++)
		{
			int textureId = index < textureIds.Length ? textureIds[index] : -1;
			resolved[index] = textureId >= 0 && textureId < textureNames.Count
				? NormalizeTexturePath(textureNames[textureId])
				: string.Empty;
		}

		return resolved;
	}

	private static string NormalizeTexturePath(string texturePath)
	{
		return texturePath.Replace('\\', '/').Trim();
	}

	private static BiomeInference InferBiomeTag(IReadOnlyList<string> textureNames)
	{
		Dictionary<string, int> scores = new(StringComparer.OrdinalIgnoreCase);
		foreach (string textureName in textureNames)
		{
			if (string.IsNullOrWhiteSpace(textureName))
				continue;

			string normalized = textureName.Replace('\\', '/').ToLowerInvariant();
			AddScore(scores, normalized, "snow", ["snow", "ice", "frost", "glacier", "winter"]);
			AddScore(scores, normalized, "desert", ["sand", "desert", "tanaris", "uldum", "dune"]);
			AddScore(scores, normalized, "volcanic", ["lava", "magma", "volcan", "fire"]);
			AddScore(scores, normalized, "swamp", ["swamp", "marsh", "bog"]);
			AddScore(scores, normalized, "built", ["brick", "cobble", "city", "wood", "roadstone", "wall"]);
			AddScore(scores, normalized, "dirt_path", ["dirt", "mud", "path", "road", "soil"]);
			AddScore(scores, normalized, "grassland", ["grass", "forest", "jungle", "moss", "leaf", "leaves"]);
			AddScore(scores, normalized, "rocky", ["rock", "stone", "cliff", "mountain", "boulder"]);
		}

		if (scores.Count == 0)
			return new BiomeInference("unknown", "no recognized texture-name biome tokens");

		KeyValuePair<string, int> best = scores
			.OrderByDescending(static entry => entry.Value)
			.ThenBy(static entry => entry.Key, StringComparer.Ordinal)
			.First();
		return new BiomeInference(best.Key, $"matched {best.Value} texture-name token(s)");
	}

	private static void AddScore(Dictionary<string, int> scores, string normalizedTextureName, string tag, string[] tokens)
	{
		foreach (string token in tokens)
		{
			if (!normalizedTextureName.Contains(token, StringComparison.Ordinal))
				continue;

			scores.TryGetValue(tag, out int score);
			scores[tag] = score + 1;
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

	private static bool HasFlag(string[] args, string name)
	{
		return args.Any(arg => string.Equals(arg, name, StringComparison.OrdinalIgnoreCase));
	}

	private readonly record struct MclyDictionaryOptions(
		string InputDirectory,
		string OutputDirectory,
		int MinOccurrences,
		int ExampleLimit,
		bool IncludeEmpty);

	private readonly record struct NpyPayload(string Descr, int[] Shape, byte[] Data);

	private readonly record struct IntTensor3(int Height, int Width, int Channels, int[] Values)
	{
		public int this[int y, int x, int channel] => Values[((y * Width) + x) * Channels + channel];
	}

	private sealed record MclyCombinationKey(string Layer0, string Layer1, string Layer2, string Layer3)
	{
		public static MclyCombinationKey Create(int[] textureIds, string[] textureNames)
		{
			string[] tokens = new string[4];
			for (int index = 0; index < tokens.Length; index++)
			{
				string textureName = index < textureNames.Length ? textureNames[index] : string.Empty;
				int textureId = index < textureIds.Length ? textureIds[index] : -1;
				tokens[index] = !string.IsNullOrWhiteSpace(textureName)
					? textureName
					: textureId >= 0 ? "#" + textureId.ToString(CultureInfo.InvariantCulture) : string.Empty;
			}

			return new MclyCombinationKey(tokens[0], tokens[1], tokens[2], tokens[3]);
		}

		public string[] TextureNames => [Layer0, Layer1, Layer2, Layer3];

		public string Hash => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(ToStableText())))[..16].ToLowerInvariant();

		public string ToStableText()
		{
			return string.Join("|", TextureNames);
		}
	}

	private sealed class MclyCombinationAccumulator(MclyCombinationKey key)
	{
		private readonly HashSet<string> _tileNames = new(StringComparer.OrdinalIgnoreCase);
		private readonly Dictionary<string, int> _idTupleDistribution = new(StringComparer.Ordinal);

		public MclyCombinationKey Key { get; } = key;

		public int Frequency { get; private set; }

		public IReadOnlySet<string> TileNames => _tileNames;

		public List<MclyExampleChunk> Examples { get; } = [];

		public IReadOnlyDictionary<string, int> IdTupleDistribution => _idTupleDistribution;

		public int[] MostCommonTextureIds
		{
			get
			{
				string idTuple = _idTupleDistribution
					.OrderByDescending(static entry => entry.Value)
					.ThenBy(static entry => entry.Key, StringComparer.Ordinal)
					.FirstOrDefault().Key ?? "-1|-1|-1|-1";
				return idTuple
					.Split('|', StringSplitOptions.TrimEntries)
					.Select(static value => int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) ? parsed : -1)
					.ToArray();
			}
		}

		public void Add(string tileName, int chunkX, int chunkY, int[] textureIds, string[] textureNames, int exampleLimit)
		{
			Frequency++;
			_tileNames.Add(tileName);
			string idTuple = string.Join("|", textureIds.Select(static id => id.ToString(CultureInfo.InvariantCulture)));
			_idTupleDistribution.TryGetValue(idTuple, out int idCount);
			_idTupleDistribution[idTuple] = idCount + 1;
			if (Examples.Count < exampleLimit)
				Examples.Add(new MclyExampleChunk(tileName, chunkX, chunkY, [.. textureIds]));
		}
	}

	private readonly record struct BiomeInference(string Tag, string Reason);

	private sealed record MclyExampleChunk(string TileName, int ChunkX, int ChunkY, int[] TextureIds);

	private sealed record MclySkippedTile(string TileName, string Path, string Reason);
}
