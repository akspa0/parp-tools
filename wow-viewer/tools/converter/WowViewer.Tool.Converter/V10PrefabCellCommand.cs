using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace WowViewer.Tool.Converter;

internal static class V10PrefabCellCommand
{
    private const int ChunksPerTile = 16;
    private const int AlphaPerChunk = 64;
    private const int HeightPerChunk = 16;
    private const int LayerCount = 4;
    private static readonly byte[] NpyMagic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

    public static void Run(string[] args)
    {
        try
        {
            PrefabCellOptions options = ParseOptions(args);
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

            bool anyRetained = false;
            foreach ((int cellWidth, int cellHeight) in options.CellSizes)
            {
                Dictionary<string, CellAccumulator> cells = new(StringComparer.Ordinal);
                List<SkippedShard> skipped = [];
                int tilesRead = 0;
                int cellsEnumerated = 0;

                foreach (string npzPath in npzFiles)
                {
                    string tileName = NormalizeTileName(Path.GetFileNameWithoutExtension(npzPath));
                    if (!TryLoadTile(npzPath, out TileData tile, out string? skipReason))
                    {
                        skipped.Add(new SkippedShard(tileName, npzPath, skipReason ?? "missing_required_signals"));
                        continue;
                    }

                    tilesRead++;
                    int maxCellX = ChunksPerTile - cellWidth + 1;
                    int maxCellY = ChunksPerTile - cellHeight + 1;
                    if (maxCellX <= 0 || maxCellY <= 0)
                        continue;

                    for (int cellY = 0; cellY < maxCellY; cellY++)
                    {
                        for (int cellX = 0; cellX < maxCellX; cellX++)
                        {
                            cellsEnumerated++;
                            CellFingerprint fingerprint = ComputeFingerprint(tile, cellX, cellY, cellWidth, cellHeight, options);
                            string key = fingerprint.StrictHash;

                            if (!cells.TryGetValue(key, out CellAccumulator? accumulator))
                            {
                                accumulator = new CellAccumulator(key, fingerprint.RelaxedHash, cellWidth, cellHeight);
                                cells.Add(key, accumulator);
                            }

                            accumulator.Add(new CellInstance(tileName, npzPath, cellX, cellY, fingerprint));
                        }
                    }
                }

                List<CellAccumulator> retained = cells.Values
                    .Where(accumulator => accumulator.Frequency >= options.MinOccurrences)
                    .OrderByDescending(static accumulator => accumulator.Frequency)
                    .ThenBy(static accumulator => accumulator.StrictHash, StringComparer.Ordinal)
                    .Take(options.DictionarySize)
                    .ToList();

                if (retained.Count > 0)
                    anyRetained = true;

                string sizeOutputDir = Path.Combine(options.OutputDirectory, $"{cellWidth}x{cellHeight}");
                SaveDictionary(sizeOutputDir, options, npzFiles.Count, tilesRead, cellsEnumerated, cells.Count, retained, skipped, cellWidth, cellHeight);
            }

            if (!anyRetained)
            {
                Console.WriteLine("Warning: no prefab cells were retained for any size.");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            Environment.ExitCode = 1;
        }
    }

    private static PrefabCellOptions ParseOptions(string[] args)
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

        List<(int Width, int Height)> cellSizes = ParseCellSizes(args);
        if (cellSizes.Count == 0)
            cellSizes = [(4, 4)];

        return new PrefabCellOptions(
            InputDirectory: inputDirectory,
            OutputDirectory: Path.GetFullPath(outputDir),
            CellSizes: cellSizes,
            MinOccurrences: Math.Max(1, GetIntOption(args, "--min-occurrences", "-m") ?? 2),
            DictionarySize: Math.Max(1, GetIntOption(args, "--dictionary-size", "-d") ?? 64),
            ExampleLimit: Math.Max(1, GetIntOption(args, "--example-limit", "-e") ?? 8),
            HeightQuantizationStep: Math.Max(0.1f, GetFloatOption(args, "--height-quant", "-q") ?? 1.0f),
            UseMclyWhenAvailable: !HasFlag(args, "--no-mcly"));
    }

    private static List<(int Width, int Height)> ParseCellSizes(string[] args)
    {
        string? sizesOption = GetOption(args, "--cell-sizes", "-s");
        if (!string.IsNullOrWhiteSpace(sizesOption))
        {
            List<(int, int)> result = [];
            foreach (string part in sizesOption.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
            {
                string[] dims = part.Split('x', StringSplitOptions.TrimEntries);
                if (dims.Length == 2
                    && int.TryParse(dims[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int w)
                    && int.TryParse(dims[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int h))
                {
                    w = Math.Clamp(w, 1, ChunksPerTile);
                    h = Math.Clamp(h, 1, ChunksPerTile);
                    result.Add((w, h));
                }
            }
            return result;
        }

        int? cellWidth = GetIntOption(args, "--cell-width", "-w");
        int? cellHeight = GetIntOption(args, "--cell-height", "-h");
        if (cellWidth.HasValue || cellHeight.HasValue)
        {
            int w = Math.Clamp(cellWidth ?? 4, 1, ChunksPerTile);
            int h = Math.Clamp(cellHeight ?? 4, 1, ChunksPerTile);
            return [(w, h)];
        }

        return [];
    }

    private static CellFingerprint ComputeFingerprint(TileData tile, int cellX, int cellY, int cellWidth, int cellHeight, PrefabCellOptions options)
    {
        StringBuilder strictBuilder = new();
        StringBuilder relaxedBuilder = new();
        List<string> heightCategories = [];
        List<int> alphaLayers = [];
        float[] layerCoverageSums = new float[LayerCount];
        int chunkCount = 0;

        for (int dy = 0; dy < cellHeight; dy++)
        {
            for (int dx = 0; dx < cellWidth; dx++)
            {
                int chunkX = cellX + dx;
                int chunkY = cellY + dy;
                chunkCount++;

                // Hole bit
                byte holeBit = tile.HoleMask[chunkY, chunkX];
                strictBuilder.Append('H').Append(holeBit);
                relaxedBuilder.Append('H').Append(holeBit);

                // MCLY texture tuple
                if (tile.MclyTextureIds is not null)
                {
                    int[] tuple = tile.MclyTextureIds.Value[chunkY, chunkX];
                    string tupleKey = tuple.Length == 0 ? "none" : string.Join("+", tuple);
                    strictBuilder.Append('T').Append(tupleKey);
                    relaxedBuilder.Append('T').Append(tupleKey);
                }
                else
                {
                    strictBuilder.Append("T?");
                    relaxedBuilder.Append("T?");
                }

                // Height stats
                ChunkHeightStats heightStats = ComputeChunkHeightStats(tile.Height, chunkX, chunkY);
                float strictMean = Quantize(heightStats.Mean, options.HeightQuantizationStep);
                float strictRange = Quantize(heightStats.Range, options.HeightQuantizationStep * 2);
                strictBuilder.Append("Hm").Append(strictMean.ToString("0.#", CultureInfo.InvariantCulture))
                    .Append("Hr").Append(strictRange.ToString("0.#", CultureInfo.InvariantCulture));

                string relaxedHeightCategory = ClassifyHeightCategory(heightStats);
                relaxedBuilder.Append("Hc").Append(relaxedHeightCategory);
                heightCategories.Add(relaxedHeightCategory);

                // Alpha signature - per-layer analysis
                ChunkAlphaStats alphaStats = ComputeChunkAlphaStats(tile.Alpha, chunkX, chunkY);
                strictBuilder.Append("Ad").Append(alphaStats.DominantLayer)
                    .Append("Ac").Append(Quantize(alphaStats.DominantCoverage, 0.25f).ToString("0.##", CultureInfo.InvariantCulture))
                    .Append("Al").Append(alphaStats.ActiveLayerCount);

                // Relaxed per-layer quantized coverage signature
                relaxedBuilder.Append("Al").Append(alphaStats.ActiveLayerCount);
                for (int layer = 0; layer < LayerCount; layer++)
                {
                    float q = Quantize(alphaStats.LayerCoverages[layer], 0.25f);
                    relaxedBuilder.Append('L').Append(layer).Append('q').Append(q.ToString("0.##", CultureInfo.InvariantCulture));
                    layerCoverageSums[layer] += alphaStats.LayerCoverages[layer];
                }
                alphaLayers.Add(alphaStats.DominantLayer);
            }
        }

        string strictHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(strictBuilder.ToString())))[..16].ToLowerInvariant();
        string relaxedHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(relaxedBuilder.ToString())))[..16].ToLowerInvariant();

        // Determine dominant height category and alpha layer across the cell
        string dominantHeightCategory = heightCategories.Count > 0
            ? heightCategories.GroupBy(static c => c).OrderByDescending(static g => g.Count()).First().Key
            : "unknown";
        int dominantAlphaLayer = alphaLayers.Count > 0
            ? alphaLayers.GroupBy(static l => l).OrderByDescending(static g => g.Count()).First().Key
            : 0;

        // Build cell-level per-layer average coverage signature
        StringBuilder layerSig = new();
        for (int layer = 0; layer < LayerCount; layer++)
        {
            float avg = chunkCount > 0 ? layerCoverageSums[layer] / chunkCount : 0f;
            float q = Quantize(avg, 0.25f);
            layerSig.Append('L').Append(layer).Append('q').Append(q.ToString("0.##", CultureInfo.InvariantCulture));
        }

        return new CellFingerprint(strictHash, relaxedHash, dominantHeightCategory, dominantAlphaLayer, layerSig.ToString());
    }

    private static ChunkHeightStats ComputeChunkHeightStats(FloatTensor2 height, int chunkX, int chunkY)
    {
        int x0 = chunkX * HeightPerChunk;
        int y0 = chunkY * HeightPerChunk;
        int x1 = Math.Min(height.Width - 1, x0 + HeightPerChunk);
        int y1 = Math.Min(height.Height - 1, y0 + HeightPerChunk);

        float min = float.MaxValue;
        float max = float.MinValue;
        double sum = 0d;
        int count = 0;

        for (int y = y0; y <= y1; y++)
        {
            for (int x = x0; x <= x1; x++)
            {
                float value = height[y, x];
                min = Math.Min(min, value);
                max = Math.Max(max, value);
                sum += value;
                count++;
            }
        }

        return new ChunkHeightStats((float)(sum / count), max - min);
    }

    private static string ClassifyHeightCategory(ChunkHeightStats stats)
    {
        if (stats.Range < 2f)
            return "flat";
        if (stats.Range < 10f)
            return "gentle";
        if (stats.Range < 30f)
            return "moderate";
        return "steep";
    }

    private static ChunkAlphaStats ComputeChunkAlphaStats(FloatTensor3 alpha, int chunkX, int chunkY)
    {
        int x0 = chunkX * AlphaPerChunk;
        int y0 = chunkY * AlphaPerChunk;
        double[] layerMeans = new double[LayerCount];
        int sampleCount = AlphaPerChunk * AlphaPerChunk;

        for (int y = 0; y < AlphaPerChunk; y++)
        {
            for (int x = 0; x < AlphaPerChunk; x++)
            {
                for (int layer = 0; layer < LayerCount; layer++)
                {
                    layerMeans[layer] += alpha[y0 + y, x0 + x, layer];
                }
            }
        }

        int dominantLayer = 0;
        double dominantMean = 0d;
        int activeLayerCount = 0;
        float[] layerCoverages = new float[LayerCount];
        for (int layer = 0; layer < LayerCount; layer++)
        {
            double mean = layerMeans[layer] / sampleCount;
            layerCoverages[layer] = (float)(mean / 255f);
            if (mean > dominantMean)
            {
                dominantMean = mean;
                dominantLayer = layer;
            }
            if (mean > 0.05d)
                activeLayerCount++;
        }

        return new ChunkAlphaStats(dominantLayer, (float)(dominantMean / 255f), activeLayerCount, layerCoverages);
    }

    private static float Quantize(float value, float step)
    {
        if (step <= 0f)
            return value;
        return MathF.Round(value / step) * step;
    }

    private static bool TryLoadTile(string npzPath, out TileData tile, out string? skipReason)
    {
        tile = default;
        skipReason = null;
        using FileStream stream = File.OpenRead(npzPath);
        using ZipArchive archive = new(stream, ZipArchiveMode.Read, leaveOpen: false);

        if (!TryReadFloatTensor2(archive, "height_257", out FloatTensor2 height))
        {
            skipReason = "missing_height_257";
            return false;
        }

        if (!TryReadFloatTensor3(archive, "mcal_alpha_pack_256", out FloatTensor3 alpha))
        {
            skipReason = "missing_mcal_alpha_pack_256";
            return false;
        }

        if (!TryReadUInt8Tensor2(archive, "hole_mask_16", out UInt8Tensor2 holeMask))
        {
            skipReason = "missing_hole_mask_16";
            return false;
        }

        Int32Tensor2? mclyTextureIds = TryReadInt32Tensor2(archive, "mcly_texture_ids", out Int32Tensor2 loadedMcly)
            ? loadedMcly
            : null;

        tile = new TileData(height, alpha, holeMask, mclyTextureIds);
        return true;
    }

    private static void SaveDictionary(
        string sizeOutputDir,
        PrefabCellOptions options,
        int discoveredShardCount,
        int tilesRead,
        int cellsEnumerated,
        int rawCellCount,
        List<CellAccumulator> retained,
        List<SkippedShard> skipped,
        int cellWidth,
        int cellHeight)
    {
        Directory.CreateDirectory(sizeOutputDir);
        string jsonPath = Path.Combine(sizeOutputDir, "prefab_cell_dictionary.json");
        string npzPath = Path.Combine(sizeOutputDir, "prefab_cell_dictionary.npz");

        var payload = new
        {
            schema_version = "v10-prefab-cell-dictionary.v2",
            generated_utc = DateTimeOffset.UtcNow,
            input_dir = options.InputDirectory,
            cell_width_chunks = cellWidth,
            cell_height_chunks = cellHeight,
            discovered_shard_count = discoveredShardCount,
            tiles_read = tilesRead,
            cells_enumerated = cellsEnumerated,
            raw_fingerprint_count = rawCellCount,
            retained_cell_count = retained.Count,
            min_occurrences = options.MinOccurrences,
            dictionary = retained.Select(static accumulator => new
            {
                cell_id = accumulator.CellId,
                strict_hash = accumulator.StrictHash,
                relaxed_hash = accumulator.RelaxedHash,
                frequency = accumulator.Frequency,
                tile_count = accumulator.TileNames.Count,
                cell_width_chunks = accumulator.CellWidth,
                cell_height_chunks = accumulator.CellHeight,
                dominant_height_category = accumulator.HeightCategoryDistribution.Count > 0
                    ? accumulator.HeightCategoryDistribution.OrderByDescending(static entry => entry.Value).First().Key
                    : "unknown",
                height_category_distribution = accumulator.HeightCategoryDistribution.OrderByDescending(static entry => entry.Value).Select(static entry => new object[] { entry.Key, entry.Value }),
                dominant_alpha_layers = accumulator.AlphaLayerDistribution.OrderByDescending(static entry => entry.Value).Take(4).Select(static entry => new object[] { entry.Key, entry.Value }),
                alpha_layer_signatures = accumulator.AlphaLayerSignatureDistribution.OrderByDescending(static entry => entry.Value).Select(static entry => new object[] { entry.Key, entry.Value }),
                examples = accumulator.Examples.Select(static example => new
                {
                    tile_name = example.TileName,
                    chunk_x = example.ChunkX,
                    chunk_y = example.ChunkY,
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
        WriteDictionaryNpz(npzPath, retained, cellWidth, cellHeight);

        Console.WriteLine("WowViewer.Tool.Converter mine-v10-prefab-cells report");
        Console.WriteLine($"InputDir: {options.InputDirectory}");
        Console.WriteLine($"OutputDir: {sizeOutputDir}");
        Console.WriteLine($"CellSize: {cellWidth}x{cellHeight} chunks");
        Console.WriteLine($"Shards: {discoveredShardCount}");
        Console.WriteLine($"TilesRead: {tilesRead}");
        Console.WriteLine($"CellsEnumerated: {cellsEnumerated}");
        Console.WriteLine($"RawFingerprints: {rawCellCount}");
        Console.WriteLine($"RetainedCells: {retained.Count}");
        Console.WriteLine($"Dictionary: {jsonPath}");
    }

    private static void WriteDictionaryNpz(string path, List<CellAccumulator> retained, int cellWidth, int cellHeight)
    {
        int count = retained.Count;
        int heightVerticesX = cellWidth * HeightPerChunk + 1;
        int heightVerticesY = cellHeight * HeightPerChunk + 1;
        int alphaPixelsX = cellWidth * AlphaPerChunk;
        int alphaPixelsY = cellHeight * AlphaPerChunk;

        float[] cellHeights = new float[count * heightVerticesY * heightVerticesX];
        float[] cellAlphas = new float[count * alphaPixelsY * alphaPixelsX * LayerCount];
        byte[] cellHoles = new byte[count * cellHeight * cellWidth];
        int[] cellIds = new int[count];
        int[] cellFrequencies = new int[count];
        int[] cellDimensions = new int[count * 2];

        for (int index = 0; index < retained.Count; index++)
        {
            CellAccumulator accumulator = retained[index];
            cellIds[index] = accumulator.CellId;
            cellFrequencies[index] = accumulator.Frequency;
            cellDimensions[index * 2 + 0] = accumulator.CellWidth;
            cellDimensions[index * 2 + 1] = accumulator.CellHeight;

            float[] avgHeight = accumulator.AverageHeight(heightVerticesY, heightVerticesX);
            float[] avgAlpha = accumulator.AverageAlpha(alphaPixelsY, alphaPixelsX);
            byte[] avgHoles = accumulator.AverageHoles();

            Array.Copy(avgHeight, 0, cellHeights, index * heightVerticesY * heightVerticesX, avgHeight.Length);
            Array.Copy(avgAlpha, 0, cellAlphas, index * alphaPixelsY * alphaPixelsX * LayerCount, avgAlpha.Length);
            Array.Copy(avgHoles, 0, cellHoles, index * cellHeight * cellWidth, avgHoles.Length);
        }

        using FileStream stream = File.Create(path);
        using ZipArchive archive = new(stream, ZipArchiveMode.Create, leaveOpen: false);
        WriteNpyEntry(archive, "cell_heights", "<f4", [count, heightVerticesY, heightVerticesX], ToBytes(cellHeights));
        WriteNpyEntry(archive, "cell_alphas", "<f4", [count, alphaPixelsY, alphaPixelsX, LayerCount], ToBytes(cellAlphas));
        WriteNpyEntry(archive, "cell_holes", "|u1", [count, cellHeight, cellWidth], cellHoles);
        WriteNpyEntry(archive, "cell_ids", "<i4", [count], ToBytes(cellIds));
        WriteNpyEntry(archive, "cell_frequencies", "<i4", [count], ToBytes(cellFrequencies));
        WriteNpyEntry(archive, "cell_dimensions", "<i4", [count, 2], ToBytes(cellDimensions));
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

    private static bool TryReadUInt8Tensor2(ZipArchive archive, string entryBaseName, out UInt8Tensor2 tensor)
    {
        tensor = default;
        if (!TryReadNpyEntry(archive, entryBaseName, out NpyPayload payload) || payload.Shape.Length != 2)
            return false;
        if (payload.Descr is not "|u1" and not "<u1" and not "u1" and not "|b1" and not "<b1" and not "b1")
            return false;

        tensor = new UInt8Tensor2(payload.Shape[0], payload.Shape[1], payload.Data);
        return true;
    }

    private static bool TryReadInt32Tensor2(ZipArchive archive, string entryBaseName, out Int32Tensor2 tensor)
    {
        tensor = default;
        if (!TryReadNpyEntry(archive, entryBaseName, out NpyPayload payload) || payload.Shape.Length != 2)
            return false;
        if (payload.Descr is not "<i4" and not "i4")
            return false;

        int count = payload.Shape[0] * payload.Shape[1];
        int[][] rows = new int[payload.Shape[0]][];
        for (int y = 0; y < payload.Shape[0]; y++)
        {
            rows[y] = new int[payload.Shape[1]];
            for (int x = 0; x < payload.Shape[1]; x++)
            {
                rows[y][x] = BinaryPrimitives.ReadInt32LittleEndian(payload.Data.AsSpan((y * payload.Shape[1] + x) * sizeof(int), sizeof(int)));
            }
        }
        tensor = new Int32Tensor2(payload.Shape[0], payload.Shape[1], rows);
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

    private static bool HasFlag(string[] args, string flag)
    {
        return args.Any(arg => string.Equals(arg, flag, StringComparison.OrdinalIgnoreCase));
    }

    private readonly record struct PrefabCellOptions(
        string InputDirectory,
        string OutputDirectory,
        List<(int Width, int Height)> CellSizes,
        int MinOccurrences,
        int DictionarySize,
        int ExampleLimit,
        float HeightQuantizationStep,
        bool UseMclyWhenAvailable);

    private readonly record struct NpyPayload(string Descr, int[] Shape, byte[] Data);

    private readonly record struct FloatTensor2(int Height, int Width, float[] Values)
    {
        public float this[int y, int x] => Values[(y * Width) + x];
    }

    private readonly record struct FloatTensor3(int Height, int Width, int Channels, float[] Values)
    {
        public float this[int y, int x, int channel] => Values[((y * Width) + x) * Channels + channel];
    }

    private readonly record struct UInt8Tensor2(int Height, int Width, byte[] Values)
    {
        public byte this[int y, int x] => Values[(y * Width) + x];
    }

    private readonly record struct Int32Tensor2(int Height, int Width, int[][] Rows)
    {
        public int[] this[int y, int x] => Rows[y][x] == -1 ? Array.Empty<int>() : [Rows[y][x]];
    }

    private readonly record struct TileData(FloatTensor2 Height, FloatTensor3 Alpha, UInt8Tensor2 HoleMask, Int32Tensor2? MclyTextureIds);
    private readonly record struct ChunkHeightStats(float Mean, float Range);
    private readonly record struct ChunkAlphaStats(int DominantLayer, float DominantCoverage, int ActiveLayerCount, float[] LayerCoverages);
    private readonly record struct CellFingerprint(string StrictHash, string RelaxedHash, string DominantHeightCategory, int DominantAlphaLayer, string AlphaLayerSignature);
    private readonly record struct CellInstance(string TileName, string Path, int ChunkX, int ChunkY, CellFingerprint Fingerprint);
    private readonly record struct SkippedShard(string TileName, string Path, string Reason);
    private readonly record struct CellExample(string TileName, int ChunkX, int ChunkY);

    private sealed class CellAccumulator
    {
        private readonly HashSet<string> _tileNames = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, int> _heightCategoryDistribution = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _alphaLayerDistribution = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _alphaLayerSignatureDistribution = new(StringComparer.Ordinal);
        private readonly List<CellInstance> _instances = [];
        private readonly int _cellWidth;
        private readonly int _cellHeight;

        public CellAccumulator(string strictHash, string relaxedHash, int cellWidth, int cellHeight)
        {
            StrictHash = strictHash;
            RelaxedHash = relaxedHash;
            _cellWidth = cellWidth;
            _cellHeight = cellHeight;
            CellId = BinaryPrimitives.ReadInt32LittleEndian(SHA256.HashData(Encoding.UTF8.GetBytes(strictHash)).AsSpan(0, 4)) & int.MaxValue;
        }

        public string StrictHash { get; }
        public string RelaxedHash { get; }
        public int CellId { get; }
        public int Frequency { get; private set; }
        public IReadOnlySet<string> TileNames => _tileNames;
        public IReadOnlyDictionary<string, int> HeightCategoryDistribution => _heightCategoryDistribution;
        public IReadOnlyDictionary<string, int> AlphaLayerDistribution => _alphaLayerDistribution;
        public IReadOnlyDictionary<string, int> AlphaLayerSignatureDistribution => _alphaLayerSignatureDistribution;
        public List<CellExample> Examples { get; } = [];
        public int CellWidth => _cellWidth;
        public int CellHeight => _cellHeight;

        public void Add(CellInstance instance)
        {
            Frequency++;
            _tileNames.Add(instance.TileName);
            _instances.Add(instance);

            string heightCategory = instance.Fingerprint.DominantHeightCategory;
            _heightCategoryDistribution.TryGetValue(heightCategory, out int heightCount);
            _heightCategoryDistribution[heightCategory] = heightCount + 1;

            string alphaLayerKey = instance.Fingerprint.DominantAlphaLayer.ToString(CultureInfo.InvariantCulture);
            _alphaLayerDistribution.TryGetValue(alphaLayerKey, out int alphaCount);
            _alphaLayerDistribution[alphaLayerKey] = alphaCount + 1;

            string layerSig = instance.Fingerprint.AlphaLayerSignature;
            _alphaLayerSignatureDistribution.TryGetValue(layerSig, out int sigCount);
            _alphaLayerSignatureDistribution[layerSig] = sigCount + 1;

            if (Examples.Count < 8)
            {
                Examples.Add(new CellExample(instance.TileName, instance.ChunkX, instance.ChunkY));
            }
        }

        public float[] AverageHeight(int targetHeight, int targetWidth)
        {
            float[] result = new float[targetHeight * targetWidth];
            int count = 0;
            foreach (CellInstance instance in _instances)
            {
                if (!TryLoadTileHeight(instance, result, targetHeight, targetWidth))
                    continue;
                count++;
            }
            if (count > 0)
            {
                for (int i = 0; i < result.Length; i++)
                    result[i] /= count;
            }
            return result;
        }

        public float[] AverageAlpha(int targetHeight, int targetWidth)
        {
            float[] result = new float[targetHeight * targetWidth * LayerCount];
            int count = 0;
            foreach (CellInstance instance in _instances)
            {
                if (!TryLoadTileAlpha(instance, result, targetHeight, targetWidth))
                    continue;
                count++;
            }
            if (count > 0)
            {
                for (int i = 0; i < result.Length; i++)
                    result[i] /= count;
            }
            return result;
        }

        public byte[] AverageHoles()
        {
            byte[] result = new byte[_cellHeight * _cellWidth];
            int count = 0;
            foreach (CellInstance instance in _instances)
            {
                if (!TryLoadTileHoles(instance, result))
                    continue;
                count++;
            }
            if (count > 0)
            {
                for (int i = 0; i < result.Length; i++)
                    result[i] = (byte)(result[i] / count);
            }
            return result;
        }

        private static bool TryLoadTileHeight(CellInstance instance, float[] accumulator, int targetHeight, int targetWidth)
        {
            try
            {
                using FileStream stream = File.OpenRead(instance.Path);
                using ZipArchive archive = new(stream, ZipArchiveMode.Read, leaveOpen: false);
                if (!TryReadFloatTensor2(archive, "height_257", out FloatTensor2 height))
                    return false;

                int x0 = instance.ChunkX * HeightPerChunk;
                int y0 = instance.ChunkY * HeightPerChunk;
                int srcH = Math.Min(height.Height - y0, targetHeight);
                int srcW = Math.Min(height.Width - x0, targetWidth);

                for (int y = 0; y < srcH; y++)
                {
                    for (int x = 0; x < srcW; x++)
                    {
                        accumulator[y * targetWidth + x] += height[y0 + y, x0 + x];
                    }
                }
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static bool TryLoadTileAlpha(CellInstance instance, float[] accumulator, int targetHeight, int targetWidth)
        {
            try
            {
                using FileStream stream = File.OpenRead(instance.Path);
                using ZipArchive archive = new(stream, ZipArchiveMode.Read, leaveOpen: false);
                if (!TryReadFloatTensor3(archive, "mcal_alpha_pack_256", out FloatTensor3 alpha))
                    return false;

                int x0 = instance.ChunkX * AlphaPerChunk;
                int y0 = instance.ChunkY * AlphaPerChunk;
                int srcH = Math.Min(alpha.Height - y0, targetHeight);
                int srcW = Math.Min(alpha.Width - x0, targetWidth);

                for (int y = 0; y < srcH; y++)
                {
                    for (int x = 0; x < srcW; x++)
                    {
                        for (int layer = 0; layer < LayerCount; layer++)
                        {
                            accumulator[(y * targetWidth + x) * LayerCount + layer] += alpha[y0 + y, x0 + x, layer];
                        }
                    }
                }
                return true;
            }
            catch
            {
                return false;
            }
        }

        private bool TryLoadTileHoles(CellInstance instance, byte[] accumulator)
        {
            try
            {
                using FileStream stream = File.OpenRead(instance.Path);
                using ZipArchive archive = new(stream, ZipArchiveMode.Read, leaveOpen: false);
                if (!TryReadUInt8Tensor2(archive, "hole_mask_16", out UInt8Tensor2 holeMask))
                    return false;

                for (int dy = 0; dy < _cellHeight; dy++)
                {
                    for (int dx = 0; dx < _cellWidth; dx++)
                    {
                        accumulator[dy * _cellWidth + dx] += holeMask[instance.ChunkY + dy, instance.ChunkX + dx];
                    }
                }
                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}
