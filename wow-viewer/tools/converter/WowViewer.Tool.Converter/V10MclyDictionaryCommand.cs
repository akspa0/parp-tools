using System.Buffers.Binary;
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
				if (!TryLoadMclyTextureIds(npzPath, out IntTensor3 textureIds, out string? skipReason))
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
						MclyCombinationKey key = new(ids[0], ids[1], ids[2], ids[3]);
						if (!combinations.TryGetValue(key, out MclyCombinationAccumulator? accumulator))
						{
							accumulator = new MclyCombinationAccumulator(key);
							combinations[key] = accumulator;
						}

						accumulator.Add(tileName, chunkX, chunkY, options.ExampleLimit);
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

	private static bool TryLoadMclyTextureIds(string npzPath, out IntTensor3 tensor, out string? skipReason)
	{
		tensor = default;
		skipReason = null;
		using FileStream stream = File.OpenRead(npzPath);
		using ZipArchive archive = new(stream, ZipArchiveMode.Read, leaveOpen: false);

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
			dictionary = retained.Select(static accumulator => new
			{
				combination_hash = accumulator.Key.ToStableText(),
				texture_ids = accumulator.Key.ToArray(),
				frequency = accumulator.Frequency,
				tile_count = accumulator.TileNames.Count,
				example_chunks = accumulator.Examples.Select(static example => new
				{
					tile_name = example.TileName,
					chunk_x = example.ChunkX,
					chunk_y = example.ChunkY,
				}),
				inferred_biome_tag = "unknown",
				inference_reason = "texture-name lookup is not present in the current v10 NPZ contract",
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

	private readonly record struct MclyCombinationKey(int Layer0, int Layer1, int Layer2, int Layer3)
	{
		public int[] ToArray() => [Layer0, Layer1, Layer2, Layer3];

		public string ToStableText()
		{
			return string.Join("_", ToArray().Select(static value => value.ToString(CultureInfo.InvariantCulture)));
		}
	}

	private sealed class MclyCombinationAccumulator(MclyCombinationKey key)
	{
		private readonly HashSet<string> _tileNames = new(StringComparer.OrdinalIgnoreCase);

		public MclyCombinationKey Key { get; } = key;

		public int Frequency { get; private set; }

		public IReadOnlySet<string> TileNames => _tileNames;

		public List<MclyExampleChunk> Examples { get; } = [];

		public void Add(string tileName, int chunkX, int chunkY, int exampleLimit)
		{
			Frequency++;
			_tileNames.Add(tileName);
			if (Examples.Count < exampleLimit)
				Examples.Add(new MclyExampleChunk(tileName, chunkX, chunkY));
		}
	}

	private sealed record MclyExampleChunk(string TileName, int ChunkX, int ChunkY);

	private sealed record MclySkippedTile(string TileName, string Path, string Reason);
}
