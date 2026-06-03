using System.Text.Json;
using WowViewer.Core.Maps;

namespace WoWViewer.Terrain;

/// <summary>
/// Loads a Zarr v3 tile dataset produced by the harvester's
/// <c>WowViewer.Tool.Harvest harvest-stream</c> → <c>build_v16_dataset.py</c>
/// pipeline. The expected layout is documented in
/// <c>wow-viewer/data-harvester/src/harvester/zarr_io.py</c>:
///
/// <code>
///   &lt;build&gt;.zarr/
///     zarr.json                 (group metadata, v3)
///     height_257/zarr.json      (per-array metadata)
///     height_257/c/0/0  ...     (chunks, Blosc+Zstd+bitshuffle compressed)
///     liquid_basic_type_257/    ← canonical resolved liquid type (spec 041)
///     ...
/// </code>
///
/// Spec 041 §T-09 is the implementation slice for the harvester-side
/// emission; this loader is spec 041 §T-10 (viewer-side consumption).
/// The C# harvester already emits <c>liquid_basic_type_257</c> in the
/// ARRY stream (RawArraySerializer) and the Python decoder auto-recognises
/// the new key in <c>zarr_io._CHUNK_PRESETS</c>; the C# consumer is the
/// remaining gap.
/// </summary>
public sealed class ZarrTileDatasetLoader
{
    private readonly string _datasetRoot;

    public string DatasetRoot { get; }
    public string MapName { get; private set; } = string.Empty;
    public List<(int tileX, int tileY)> TileCoords { get; } = new();

    public ZarrTileDatasetLoader(string datasetRoot)
    {
        _datasetRoot = datasetRoot ?? throw new ArgumentNullException(nameof(datasetRoot));
        DatasetRoot = _datasetRoot;
    }

    /// <summary>
    /// Validates the folder is a Zarr v3 store and discovers the build
    /// metadata. Throws if the store is malformed. The full chunk-decoding
    /// pipeline lands in the spec 041 T-10 implementation slice.
    /// </summary>
    public ZarrStoreSummary Open()
    {
        if (!Directory.Exists(_datasetRoot))
            throw new DirectoryNotFoundException($"Zarr dataset root not found: {_datasetRoot}");

        string zarrJsonPath = Path.Combine(_datasetRoot, "zarr.json");
        if (!File.Exists(zarrJsonPath))
        {
            string[] subdirsWithZarrJson = Directory
                .EnumerateDirectories(_datasetRoot, "*.zarr", SearchOption.TopDirectoryOnly)
                .Where(static d => File.Exists(Path.Combine(d, "zarr.json")))
                .ToArray();

            if (subdirsWithZarrJson.Length == 0)
            {
                throw new InvalidDataException(
                    $"No zarr.json found at '{zarrJsonPath}' and no '*.zarr/' subdirectory with zarr.json exists. " +
                    "The Zarr tile dataset must be either a v3 LocalStore directory with zarr.json at the root " +
                    "or a parent folder containing one or more '<build>.zarr/' subdirectories.");
            }

            if (subdirsWithZarrJson.Length > 1)
            {
                Console.Error.WriteLine(
                    $"[Zarr] '{_datasetRoot}' contains {subdirsWithZarrJson.Length} builds: " +
                    $"{string.Join(", ", subdirsWithZarrJson.Select(Path.GetFileName))}. " +
                    "Defaulting to the first; the viewer will gain a build-picker in a follow-up slice.");
            }

            return OpenStoreAt(subdirsWithZarrJson[0]);
        }

        return OpenStoreAt(_datasetRoot);
    }

    private ZarrStoreSummary OpenStoreAt(string storeRoot)
    {
        string zarrJsonPath = Path.Combine(storeRoot, "zarr.json");
        using FileStream fs = File.OpenRead(zarrJsonPath);
        using JsonDocument doc = JsonDocument.Parse(fs, options: default);

        JsonElement root = doc.RootElement;
        string zarrFormat = root.TryGetProperty("zarr_format", out JsonElement fmt) ? fmt.ToString() : "<unknown>";
        string storeKind = storeRoot.EndsWith(".zarr", StringComparison.OrdinalIgnoreCase)
            ? Path.GetFileName(storeRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
            : Path.GetFileName(storeRoot);

        MapName = storeKind;

        List<string> arrays = Directory
            .EnumerateDirectories(storeRoot)
            .Where(static d => File.Exists(Path.Combine(d, "zarr.json")))
            .Select(static d => Path.GetFileName(d)!)
            .OrderBy(static s => s, StringComparer.Ordinal)
            .ToList();

        bool hasLiquidBasicType = arrays.Contains("liquid_basic_type_257", StringComparer.Ordinal);
        bool hasMh2oTypeMask = arrays.Contains("mh2o_type_mask", StringComparer.Ordinal);

        if (!hasLiquidBasicType)
        {
            Console.Error.WriteLine(
                $"[Zarr] '{storeRoot}' is missing the canonical 'liquid_basic_type_257' array (spec 041). " +
                "The harvester will populate it on the next run; rebuild tiles with WowViewer.Tool.Harvest harvest-stream " +
                "and re-run the dataset build to get the resolved liquid type field.");
        }

        return new ZarrStoreSummary(
            StoreRoot: storeRoot,
            ZarrFormat: zarrFormat,
            MapName: storeKind,
            Arrays: arrays,
            HasLiquidBasicType: hasLiquidBasicType,
            HasMh2oTypeMask: hasMh2oTypeMask);
    }

    /// <summary>
    /// Loads a single tile from the Zarr store. NOT YET IMPLEMENTED — the
    /// Blosc+Zstd+bitshuffle chunk decoder is the spec 041 T-10 follow-up.
    /// </summary>
    public TerrainTileTensorPack LoadTile(int tileX, int tileY)
    {
        throw new NotImplementedException(
            "ZarrTileDatasetLoader.LoadTile is the spec 041 T-10 implementation slice. " +
            "The C# harvester already emits 'liquid_basic_type_257' in the ARRY stream " +
            "and the Python decoder stores it in '<build>.zarr/liquid_basic_type_257/'. " +
            "The remaining work is the Blosc+Zstd+bitshuffle chunk decoder and the " +
            "TerrainTileTensorPack rehydration. See wow-viewer/specs/041-mh2o-mclq-liquid-type-determination-fix/spec.md.");
    }
}

/// <summary>Summary of a Zarr v3 store discovered by <see cref="ZarrTileDatasetLoader"/>.</summary>
public sealed record ZarrStoreSummary(
    string StoreRoot,
    string ZarrFormat,
    string MapName,
    IReadOnlyList<string> Arrays,
    bool HasLiquidBasicType,
    bool HasMh2oTypeMask);
