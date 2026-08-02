using System.Collections.Concurrent;
using System.Numerics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using StreamProfile = WowViewer.Core.IO.Maps.RawArraySerializer.StreamProfile;
using WowViewer.Core.Blp;
using WowViewer.Core.Curation;
using WowViewer.Core.IO.Blp;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Lit;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Maps;
using WowViewer.Core.Renderer.Headless;
using WowViewer.Core.Renderer.ObjectCapture;
using WowViewer.Core.Renderer.Terrain;
using WowViewer.Core.Runtime.World.Terrain;
using WowViewer.Core.Terrain;

namespace WowViewer.Tools.Harvest;

static class Program
{
    private static Dictionary<string, string>? _md5Lookup;
    private static readonly ConcurrentDictionary<string, Lazy<WlLooseFileEntry[]>> _wlLooseFileCache = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ConditionalWeakTable<NativeMpqService, KnownTerrainTexturePaths> _knownTerrainTexturePaths = new();
    private static readonly int DefaultHarvestTileWorkers = Math.Max(1, Math.Min(8, Environment.ProcessorCount));
    private sealed record HarvestMapDiscoveryResult(
        string Map,
        string DisplayName,
        bool Include,
        string Reason,
        bool IsAlpha,
        bool IsWmoBased,
        bool HasWorldModelAsset,
        int WorldModelNameCount,
        int TilesWithData,
        bool HasReadableTile,
        bool HasUsableTile,
        int? ProbeTileX,
        int? ProbeTileY);

    private sealed class WlLooseFileEntry
    {
        public required string Path { get; init; }
        public required WlFile File { get; init; }
    }

    private readonly record struct HarvestTileJob(int Order, int TileX, int TileY);
    private readonly record struct HarvestTileResult(int Order, byte[]? Blob, bool HadError, string? ErrorMessage);
    private sealed record SyntheticMinimapLightingProfile(
        TerrainMinimapLighting Lighting,
        string Source,
        string EvidenceState,
        string ProfileRevision,
        string? LitSourcePath,
        uint? LitVersion,
        string? LightName,
        int? LightIndex,
        string DirectionEvidenceState,
        string McshEvidenceState,
        string? Diagnostic);
    private sealed record SyntheticMinimapTileResult(
        int TileX,
        int TileY,
        string Status,
        string? OutputPath,
        int TextureCount,
        string? Detail,
        IReadOnlyList<TerrainTextureFallbackResolution>? TextureFallbacks = null,
        string? LiquidOutputPath = null,
        int LiquidPixelCount = 0,
        SyntheticMinimapLightingProfile? Lighting = null,
        string? AuthoredOutputPath = null,
        string? ComparisonOutputPath = null);
    private sealed record SyntheticMinimapManifest(
        string Format,
        string ClientRoot,
        string? BuildVersion,
        string MapName,
        string TimeOfDay,
        float TimeOfDayHours,
        float NormalizedGameTime,
        int TileResolution,
        string RenderMode,
        float TextureRepeatsPerChunk,
        bool PerTileRequested,
        bool WholeMapRequested,
        bool WmoOverlay,
        SyntheticMinimapLightingProfile Lighting,
        string LiquidRenderProfile,
        TerrainMinimapStitchResult? WholeMap,
        TerrainMinimapStitchResult? LiquidWholeMap,
        IReadOnlyList<SyntheticMinimapTileResult> Tiles);

    static int Main(string[] args)
    {
        Environment.ExitCode = 0;
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            ShowUsage();
            return 0;
        }

        string command = args[0].ToLowerInvariant();
        string[] tail = args.Skip(1).ToArray();

        switch (command)
        {
            case "harvest-tile":
                RunHarvestTile(tail);
                break;
            case "harvest-map":
                RunHarvestMap(tail);
                break;
            case "synthetic-minimap":
                RunSyntheticMinimap(tail);
                break;
            case "sun-diagnostic":
                RunSunDiagnostic(tail);
                break;
            case "inspect-minimap-blp":
                RunInspectMinimapBlp(tail);
                break;
            case "extract-unified":
                RunExtractUnified(tail);
                break;
            case "harvest-map-mpq":
                RunHarvestMapMpq(tail);
                break;
            case "harvest-stream":
                RunHarvestStream(tail);
                break;
            case "discover-maps":
                RunDiscoverMaps(tail);
                break;
            case "extract-holes":
                RunExtractHoles(tail);
                break;
            case "extract-tilesets":
                RunExtractTilesets(tail);
                break;
            case "split-minimap-image":
                RunSplitMinimapImage(tail);
                break;
            case "relief-to-map":
                RunReliefToMap(tail);
                break;
            case "dump-texture-names":
                RunDumpTextureNames(tail);
                break;
            case "capture-objects":
                RunCaptureObjects(tail);
                break;
            case "curate":
                RunCurate(tail);
                break;
            default:
                Console.Error.WriteLine($"Unknown command '{command}'.");
                ShowUsage();
                return 1;
        }

        return Environment.ExitCode;
    }

    static void ShowUsage()
    {
        Console.WriteLine("""
            WowViewer.Tool.Harvest — V14 dataset generation tool

            Usage: WowViewer.Tool.Harvest <command> [options]

            Commands:
              harvest-tile      Extract NPZ shard from a single ADT tile (disk path)
              harvest-map       Batch-extract all tiles from a map directory (disk path)
              extract-unified   Extract NPZ shard from a tile inside MPQ archives
                                (reads tileset BLP + ADT + WDL from MPQ, outputs NPZ shard)
              harvest-stream    Stream raw tile blobs from a map to stdout
              discover-maps     List terrain-trainable maps from a staged client using
                                WDT summary + tile probe checks
              extract-holes     Dump raw per-chunk MCNK hole bitmasks (uint16, 4x4
                                hole groups) for every terrain tile of the given maps
                                to JSON (era-aware: alpha WDT + LK/split ADT)
              extract-tilesets  Decode the listed tileset BLPs from this client's
                                MPQs to PNGs + a manifest JSON (era-specific pixels)
              synthetic-minimap Compose paired terrain-only and _liquid minimaps directly from
                                client tiles, with optional per-tile and whole-map PNG outputs.
                                Use --time-hours (HHmm or HH:mm) to set the sun position;
                                defaults to 12:00 (noon, full-bright). Terrain renders with Lambert
                                hillshading; analytic cast shadows are an ADDITION the client never
                                had and default OFF for the Alpha era (--cast-shadows enables them,
                                --no-cast-shadows forces them off). Add
                                --authored-reference to emit real-client side-by-side comparisons,
                                and --match-time (requires --authored-reference) to infer each
                                tile's best-matching time-of-day from the authored minimap's shading
                                and render at it. Normal RGB omits the client's baked MCSH map
                                (a separate signal); --bake-mcsh is an exceptional-history preview.
              sun-diagnostic    Light a synthetic hill with the current sun model at 06:00-18:00 and
                                write compass-marked PNGs. No client needed. Use this to tell a
                                wrong lighting model apart from wrongly-oriented terrain data.
                                Accepts --era / --sun-model / --sun-azimuth / --ambient /
                                --cast-shadow-strength / --shadow-softness / --resolution.
              split-minimap-image
                                Slice one standalone composite map image (e.g. a full-continent
                                scan/render with no client/MPQ backing) into a fixed-size tile grid
                                for downstream model inference. Near-uniform (blank) tiles are
                                skipped by default.
              capture-objects   Spec 118: render every M2/MDX/WMO object in the client (or a
                                supplied asset list) top-down, headless, and stream one
                                (image_rgb, mask) record per object to stdout in the same
                                ARRY/ENDS harvest-stream wire format harvest-stream uses --
                                consumed directly by harvester.raw_reader.read_tile_blob on
                                the Python side. No GUI, no manual per-object click; one
                                command produces the whole object-mask library's raw signal.
              relief-to-map     Turn a folder of predicted relief16 PNGs (from
                                v50_infer_geometry_detailer.py / v50_infer_direct_geometry.py) into
                                a real, loadable synthetic map: one blank-textured ADT per tile with
                                the predicted heightmap patched into MCVT/MCNR (BlankAdtFactory +
                                AdtTerrainWriter), plus one shared WDT and WDL covering every tile.
                                Output lands in <output-dir>/World/Maps/<map-name>/, the same layout
                                WowViewer.Tool.Inspect's "map generate-blank" uses.

            Global options:
              --build, -b       Client build version (e.g. "4.3.4.15595") for
                               version-aware ADT profile selection. Auto-detected
                               from input path if not specified.
              --client-root     WoW client root directory (for extract-unified)
              --map, -m         Map name (e.g. "Azeroth") for archive-backed commands

            synthetic-minimap uses one achromatic global light at --time-hours (default 12:00),
              shaded in linear space; map LIT and Light DBC profiles belong to the viewer and are
              never applied to minimap generation.
            synthetic-minimap notable options: --build <exact-build>, --per-tile, --whole-map,
              --tile-x/--tile-y <0..63> for one occupied tile, --tile-list "x,y;x,y;..." for a bounded set,
              --authored-reference to require and emit the real client minimap plus a side-by-side comparison,
              --match-time (with --authored-reference) to render each tile at its inferred hour,
              --limit <emitted terrain/_liquid PNG pairs>,
              --detail (mip-filtered real BLP texels for 1024+ SR targets),
              --no-cast-shadows (hillshading only, no ridge-cast shadows), and --bake-mcsh
              (diagnostic preview only).
            synthetic-minimap contrast knobs (sweep these against --authored-reference):
              --ambient <v | r,g,b>       sky term; the floor every shadowed pixel lands on.
                                          Lower = deeper shadows. A tinted value gives shadows a
                                          different hue from lit ground. Default 0.12.
              --cast-shadow-strength <0..1>  how dark a fully cast-shadowed pixel gets. Default 0.70.
              --shadow-softness <world-units>  penumbra width of the shadow ramp. Lower = crisper,
                                          narrower shadow edges in crevices. Default 4.
              --max-shadow-length <world-units>  clamp how far a shadow is thrown; 0 = uncapped
                                          (default, and physically correct). The traced sun is LOW,
                                          so shadows run 1.3-2.75x the occluder height. If you need
                                          this to look right, try --no-cast-shadows first: the
                                          client shaded with Lambert + baked MCSH and never
                                          ray-traced terrain shadows at all.
              --light-gain <v>            overrides the gain that is otherwise DERIVED from
                                          --ambient to hold overall brightness steady. Only needed
                                          if you want to rebrighten/darken the whole image.
              --liquid-palette <name>     prealpha053 (default, 0.5.3 cyan-teal) | viewer (legacy).
              --water-color <r,g,b[,a]>   override water/ocean tint in 0..1 against the palette.
            synthetic-minimap era gating: Blizzard changed minimap generation across builds, so the
              era profile (resolved from the build, or forced with --era) supplies every
              era-sensitive default before the flags above override it.
              --era <name>                alpha (0.5.3) | beta1 (0.6.0) | release (1.0.0+).
              --sun-model <fixed|sweep>   fixed bearing all day, or an east-to-west sweep. Only the
                                          1.0.0 era's model is traced from a client; Alpha's is
                                          assumed and unverified.
              --sun-azimuth <degrees>     source bearing from +X (North) toward +Y (West).
                                          45 = north-west (traced), 270 = east, 90 = west.
              --score                     (needs --authored-reference) measure agreement with the
                                          authored minimap -- mean brightness, CONTRAST, luma
                                          correlation, MAE and per-channel ratios -- and re-render
                                          the pre-cast-shadow/pre-linear-shading configuration on the
                                          same tiles so the report says whether the current settings
                                          are better, worse or the same. Writes
                                          authored-comparison.csv.
              --light-overlay             emit a per-tile *_lights.png plotting LIT light positions
                                          with their colour swatch and influence dome. Diagnostic
                                          only; kept out of the terrain RGB corpus.
              --measure-sun               (needs --authored-reference) score each authored tile
                                          across bearings AND hours and report which bearing best
                                          explains the real minimap's shading. This is how to settle
                                          whether Alpha's sun is fixed or sweeps.
            harvest-stream --stream-profile v22 emits full terrain texture/model sidecars plus
              conservative minimap_lighting provenance when the client data permits it.
            split-minimap-image required: --input <image> --output-dir <dir>.
              Optional: --tile-size <px> (default 256, must match the target model's input
              contract), --prefix <name> (default: input file stem), --white-threshold /
              --black-threshold <mean 0-255> (default 250 / 5, skip near-blank tiles),
              --keep-blank (disable blank-tile skipping).
            relief-to-map required: --manifest <inference_manifest.json> --output-dir <dir>.
              Optional: --map-name <name> (default "relief_predicted"), --texture <blp path>
              (default tileset\ocean\westfallseafloor.blp, placeholder only -- Spec 114 texture-
              family reconstruction is a separate unbuilt stage), --height-scale <float> (default
              100.0, visualization exaggeration only), --no-center (disable per-tile mean-
              centering), --verify (read each written ADT back and compare heights to what was
              written; reports a max-abs-diff failure per tile if the round trip doesn't match),
              --no-fill-gaps (disable gap-filling; default fills every missing tile inside the
              inferred tiles' bounding rectangle with a flat/blank ADT -- e.g. tiles
              split-minimap-image skipped as blank source input -- so the map is a complete,
              hole-free quilt across its own extent instead of leaving real absent-geometry gaps),
              --bake-mccv (bake each inferred tile's source minimap RGB into per-vertex MCCV color,
              tinting the placeholder texture so the terrain visually resembles the source minimap
              in-viewer -- a quick-look guide for sharing model output, NOT real texture-layer
              reconstruction; unaffected gap-fill tiles carry no MCCV since they have no source
              image; sets the WDT's MPHD AdtHasMccv flag).
              NOTE: relief is per-tile RELATIVE height (contract v112.1) -- there is no cross-tile
              absolute continuity, so adjacent INFERRED tiles will still show a height step at
              their shared border; this is the model's documented contract, not a bug, and
              --no-fill-gaps/gap-filling does not and cannot change it.
            """);
    }

    static void RunDiscoverMaps(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        if (string.IsNullOrWhiteSpace(clientRoot))
        {
            Console.Error.WriteLine("Error: --client-root <dir> is required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        LoadMd5Translate(clientRoot, catalog);

        MapDirectoryLookup lookup = new();
        lookup.Load(BuildClientSearchRoots(clientRoot), catalog);
        if (!lookup.IsLoaded)
        {
            Console.Error.WriteLine("Error: Map.dbc could not be loaded from the staged client.");
            Environment.ExitCode = 1;
            return;
        }

        List<HarvestMapDiscoveryResult> results = [];
        foreach (MapDirectoryEntry entry in lookup.Entries.OrderBy(static entry => entry.Directory, StringComparer.OrdinalIgnoreCase))
        {
            results.Add(DiscoverMap(catalog, entry));
        }

        string json = System.Text.Json.JsonSerializer.Serialize(
            results,
            new System.Text.Json.JsonSerializerOptions
            {
                PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase,
                WriteIndented = true
            });
        Console.WriteLine(json);
    }

    static string? TryFindDefaultListfilePath()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            string candidate = Path.Combine(current.FullName, "libs", "wowdev", "wow-listfile", "listfile.txt");
            if (File.Exists(candidate))
                return candidate;

            current = current.Parent;
        }

        return null;
    }

    static void TryLoadSupplementalListfile(NativeMpqService catalog)
    {
        string? listfilePath = TryFindDefaultListfilePath();
        if (string.IsNullOrWhiteSpace(listfilePath) || !File.Exists(listfilePath))
            return;

        try
        {
            catalog.LoadListfile(listfilePath);
            Console.Error.WriteLine($"  Loaded supplemental listfile: {listfilePath}");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Warning: failed to load supplemental listfile '{listfilePath}': {ex.Message}");
        }
    }

    static string[] ReadSupplementalListfileEntriesForMap(string mapName)
    {
        string? listfilePath = TryFindDefaultListfilePath();
        if (string.IsNullOrWhiteSpace(listfilePath) || !File.Exists(listfilePath))
            return [];

        string mapPrefix = $"world\\maps\\{mapName}\\".ToLowerInvariant();
        return File.ReadLines(listfilePath)
            .Select(line => line.Trim())
            .Where(line => line.Length > 0)
            .Select(line => line.Replace('/', '\\'))
            .Where(path =>
                path.StartsWith(mapPrefix, StringComparison.OrdinalIgnoreCase) &&
                (path.EndsWith(".wlw", StringComparison.OrdinalIgnoreCase) ||
                 path.EndsWith(".wlm", StringComparison.OrdinalIgnoreCase) ||
                 path.EndsWith(".wlq", StringComparison.OrdinalIgnoreCase) ||
                 path.EndsWith(".wll", StringComparison.OrdinalIgnoreCase)))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    static void RunHarvestTile(string[] args)
    {
        string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(a => !a.StartsWith('-'));
        string? output = GetOption(args, "--output", "-o");
        string? textureSource = GetOption(args, "--texture-source", "-t");
        string? minimapRoot = GetOption(args, "--minimap-root", "-m");
        string? buildVersion = GetOption(args, "--build", "-b");
        int? tileXOpt = GetIntOption(args, "--tile-x", "-x");
        int? tileYOpt = GetIntOption(args, "--tile-y", "-y");

        if (string.IsNullOrWhiteSpace(input))
        {
            Console.Error.WriteLine("Error: --input <root.adt or alpha.wdt> is required.");
            Environment.ExitCode = 1;
            return;
        }

        if (string.IsNullOrWhiteSpace(output))
        {
            string dir = Path.GetDirectoryName(input) ?? ".";
            string stem = Path.GetFileNameWithoutExtension(input);
            output = Path.Combine(dir, $"{stem}_harvest.npz");
        }

        try
        {
            TerrainTileTensorPack pack;

            if (Path.GetExtension(input).Equals(".wdt", StringComparison.OrdinalIgnoreCase)
                && TryDetectAlphaWdt(input, out bool isAlpha, out bool isWmoBased))
            {
                if (isWmoBased)
                {
                    Console.Error.WriteLine($"Error: {input} is a WMO-based Alpha map (no terrain tiles). "
                        + "WMO-based Alpha maps are not yet supported for tensor pack export.");
                    Environment.ExitCode = 1;
                    return;
                }

                if (!isAlpha)
                {
                    Console.Error.WriteLine($"Error: {input} is a Retail WDT file. "
                        + "Use a root ADT file (e.g., mapname_XX_YY.adt) instead.");
                    Environment.ExitCode = 1;
                    return;
                }

                if (!tileXOpt.HasValue || !tileYOpt.HasValue)
                {
                    Console.Error.WriteLine("Error: Alpha WDT tiles require --tile-x <0-63> and --tile-y <0-63>.");
                    Environment.ExitCode = 1;
                    return;
                }

                int tileX = tileXOpt.Value;
                int tileY = tileYOpt.Value;
                if ((uint)tileX >= 64 || (uint)tileY >= 64)
                {
                    Console.Error.WriteLine("Error: tile coordinates must be 0-63.");
                    Environment.ExitCode = 1;
                    return;
                }

                if (!AlphaWdtReader.TryReadTile(input, tileX, tileY, out AlphaTileData? tileData))
                {
                    Console.Error.WriteLine($"Error: tile ({tileX},{tileY}) is not present in {input} "
                        + "(or the tile data could not be parsed).");
                    Environment.ExitCode = 1;
                    return;
                }

                pack = AlphaTensorPackBuilder.Build(tileData, tileX, tileY);
            }
            else
            {
                pack = AdtTensorPackBuilder.Build(input, textureSource, buildVersion);
            }

            if (!string.IsNullOrWhiteSpace(minimapRoot))
            {
                if (TryLoadMinimap(input, minimapRoot, out byte[,,]? minimap))
                {
                    pack.MinimapRgb256 = minimap;
                    pack.MinimapSourceTag = "raw";
                    pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals, StringComparer.OrdinalIgnoreCase)
                    {
                        "minimap_rgb_256"
                    };
                }
            }

            NpzTileSerializer.Serialize(pack, output);
            Console.WriteLine($"Harvested: {output}");
            Console.WriteLine($"Signals: {string.Join(", ", pack.AvailableSignals)}");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error harvesting {input}: {ex.Message}");
            Environment.ExitCode = 1;
        }
    }

    private static bool TryDetectAlphaWdt(string wdtPath, out bool isAlpha, out bool isWmoBased)
    {
        isAlpha = false;
        isWmoBased = false;
        try
        {
            byte[] bytes = File.ReadAllBytes(wdtPath);
            isAlpha = AlphaWdtReader.IsAlphaWdt(bytes);

            if (isAlpha && bytes.Length >= 16)
            {
                int mphdDataOffset = 8 + 4;
                if (mphdDataOffset + 8 <= bytes.Length)
                {
                    int mdnmOffset = BitConverter.ToInt32(bytes, mphdDataOffset + 4);
                    int monmOffset = BitConverter.ToInt32(bytes, mphdDataOffset + 12);
                    if (mdnmOffset == 2 || monmOffset == 2)
                        isWmoBased = true;
                }
            }
        }
        catch
        {
        }
        return true;
    }

    static void RunHarvestMap(string[] args)
    {
        string? inputDir = GetOption(args, "--input-dir", "-i") ?? args.FirstOrDefault(a => !a.StartsWith('-'));
        string? outputDir = GetOption(args, "--output-dir", "-o");
        string? minimapRoot = GetOption(args, "--minimap-root", "-m");
        string? buildVersion = GetOption(args, "--build", "-b");
        int limit = GetIntOption(args, "--limit", "-n") ?? int.MaxValue;
        bool overwrite = HasFlag(args, "--overwrite");

        if (string.IsNullOrWhiteSpace(inputDir) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Error: --input-dir <dir> and --output-dir <dir> are required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(inputDir))
        {
            Console.Error.WriteLine($"Error: input directory not found: {inputDir}");
            Environment.ExitCode = 1;
            return;
        }

        Directory.CreateDirectory(outputDir);

        int processed = 0;
        int skipped = 0;

        foreach (string adtPath in Directory.EnumerateFiles(inputDir, "*.adt")
            .Where(p => !p.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
                && !p.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase)
                && !p.EndsWith("_lod.adt", StringComparison.OrdinalIgnoreCase))
            .Take(limit))
        {
            string stem = Path.GetFileNameWithoutExtension(adtPath);
            string outputPath = Path.Combine(outputDir, $"{stem}_harvest.npz");

            if (!overwrite && File.Exists(outputPath))
            {
                skipped++;
                continue;
            }

            string candidateTextureSource = Path.Combine(inputDir, $"{stem}_tex0.adt");
            string? textureSource = File.Exists(candidateTextureSource) ? candidateTextureSource : null;

            try
            {
                TerrainTileTensorPack pack = AdtTensorPackBuilder.Build(adtPath, textureSource, buildVersion);

                if (!string.IsNullOrWhiteSpace(minimapRoot))
                {
                    if (TryLoadMinimap(adtPath, minimapRoot, out byte[,,]? minimap))
                    {
                        pack.MinimapRgb256 = minimap;
                        pack.MinimapSourceTag = "raw";
                        pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals, StringComparer.OrdinalIgnoreCase)
                        {
                            "minimap_rgb_256"
                        };
                    }
                }

                NpzTileSerializer.Serialize(pack, outputPath);
                processed++;
                Console.WriteLine($"Harvested: {stem}");
            }
            catch (Exception ex)
            {
                skipped++;
                Console.Error.WriteLine($"Skipped {stem}: {ex.Message}");
            }
        }

        Console.WriteLine($"Done. Processed={processed} Skipped={skipped}");
    }

    private sealed record SplitMinimapTileResult(int TileX, int TileY, string Status, string? OutputPath, double MeanValue);
    private sealed record SplitMinimapManifest(
        string Format,
        string InputPath,
        string InputSha256,
        int SourceWidth,
        int SourceHeight,
        int TileSize,
        int TilesX,
        int TilesY,
        double WhiteThreshold,
        double BlackThreshold,
        bool KeepBlank,
        int Written,
        int SkippedBlank,
        IReadOnlyList<SplitMinimapTileResult> Tiles);

    static void RunSplitMinimapImage(string[] args)
    {
        string? input = GetOption(args, "--input", "-i");
        string? outputDir = GetOption(args, "--output-dir", "-o");
        int tileSize = GetIntOption(args, "--tile-size", "-s") ?? 256;
        string? prefix = GetOption(args, "--prefix", "-p");
        double whiteThreshold = GetFloatOption(args, "--white-threshold", "") ?? 250.0;
        double blackThreshold = GetFloatOption(args, "--black-threshold", "") ?? 5.0;
        bool keepBlank = HasFlag(args, "--keep-blank");

        if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Error: --input <image> and --output-dir <dir> are required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!File.Exists(input))
        {
            Console.Error.WriteLine($"Error: input image not found: {input}");
            Environment.ExitCode = 1;
            return;
        }

        if (tileSize is < 1 or > 8192)
        {
            Console.Error.WriteLine("Error: --tile-size must be within 1..8192.");
            Environment.ExitCode = 1;
            return;
        }

        prefix = string.IsNullOrWhiteSpace(prefix) ? Path.GetFileNameWithoutExtension(input) : prefix;
        Directory.CreateDirectory(outputDir);

        string inputSha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(input)));

        using Image<Rgb24> source = Image.Load<Rgb24>(input);
        int width = source.Width;
        int height = source.Height;
        int tilesX = width / tileSize;
        int tilesY = height / tileSize;
        Console.WriteLine($"Loaded {input}: {width}x{height} -> grid {tilesX}x{tilesY} at {tileSize}px/tile" +
            (keepBlank ? string.Empty : $" (skipping tiles with mean < {blackThreshold} or > {whiteThreshold})"));

        if (tilesX == 0 || tilesY == 0)
        {
            Console.Error.WriteLine($"Error: image is smaller than one {tileSize}px tile.");
            Environment.ExitCode = 1;
            return;
        }

        var results = new List<SplitMinimapTileResult>(tilesX * tilesY);
        int written = 0;
        int skippedBlank = 0;

        for (int ty = 0; ty < tilesY; ty++)
        {
            for (int tx = 0; tx < tilesX; tx++)
            {
                var rect = new Rectangle(tx * tileSize, ty * tileSize, tileSize, tileSize);
                using Image<Rgb24> tile = source.Clone(context => context.Crop(rect));

                double mean = MeanPixelValue(tile);
                if (!keepBlank && (mean < blackThreshold || mean > whiteThreshold))
                {
                    skippedBlank++;
                    results.Add(new SplitMinimapTileResult(tx, ty, "skipped_blank", null, mean));
                    continue;
                }

                string outputPath = Path.Combine(outputDir, $"{prefix}_{tx:D2}_{ty:D2}.png");
                tile.SaveAsPng(outputPath);
                written++;
                results.Add(new SplitMinimapTileResult(tx, ty, "written", outputPath, mean));
            }

            if (ty % 10 == 0 || ty == tilesY - 1)
                Console.WriteLine($"  Row {ty}/{tilesY - 1} - written {written}, skipped blank {skippedBlank}");
        }

        var manifest = new SplitMinimapManifest(
            "split-minimap-image-v1",
            Path.GetFullPath(input),
            inputSha256,
            width,
            height,
            tileSize,
            tilesX,
            tilesY,
            whiteThreshold,
            blackThreshold,
            keepBlank,
            written,
            skippedBlank,
            results);
        string manifestPath = Path.Combine(outputDir, "split-manifest.json");
        File.WriteAllText(
            manifestPath,
            System.Text.Json.JsonSerializer.Serialize(
                manifest,
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine($"Done. Written={written} SkippedBlank={skippedBlank} -> {outputDir}");
        Console.WriteLine($"Manifest: {manifestPath}");

        if (written == 0)
            Environment.ExitCode = 1;
    }

    private static double MeanPixelValue(Image<Rgb24> tile)
    {
        long sum = 0;
        long count = 0;
        tile.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> row = accessor.GetRowSpan(y);
                foreach (Rgb24 pixel in row)
                {
                    sum += pixel.R + pixel.G + pixel.B;
                    count += 3;
                }
            }
        });
        return count == 0 ? 0.0 : sum / (double)count;
    }

    private sealed record TextureNameTileRecord(int TileX, int TileY, IReadOnlyList<string> TextureNames);
    private sealed record TextureNameDump(
        string Format,
        string ClientRoot,
        string? Build,
        string Map,
        int TileCount,
        IReadOnlyList<TextureNameTileRecord> Tiles);

    /// <summary>
    /// Spec 115: dump each occupied tile's ORDERED MTEX texture-name table.
    ///
    /// The v50 Zarr store keeps <c>mcly_texture_ids</c> (a per-tile local index into this table) but
    /// does not persist any index-to-name mapping: the global tileset list lives only in the
    /// transient build-time enrichment stream. This command recovers the names directly from the
    /// client so terrain-feature labels can be derived without rebuilding any store. Ordering is
    /// load-bearing -- position in TextureNames IS the value stored in mcly_texture_ids.
    /// </summary>
    static void RunDumpTextureNames(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapName = GetOption(args, "--map", "-m");
        string? outputPath = GetOption(args, "--output", "-o");
        string? requestedBuild = GetOption(args, "--build", "-b");
        int maxTiles = GetIntOption(args, "--limit", "-n") ?? int.MaxValue;

        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(mapName) || string.IsNullOrWhiteSpace(outputPath))
        {
            Console.Error.WriteLine("Error: --client-root, --map, and --output are required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);
        string? buildVersion = requestedBuild ?? DetectBuildVersionFromClientRoot(clientRoot);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
        byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
        if (wdtBytes is null || wdtBytes.Length == 0)
        {
            Console.Error.WriteLine($"Error: could not read WDT '{wdtVirtual}' from client.");
            Environment.ExitCode = 1;
            return;
        }

        IReadOnlyList<WdtTileCoordinate> occupiedTiles;
        using (var summaryStream = new MemoryStream(wdtBytes, writable: false))
        {
            MapFileSummary summary = MapFileSummaryReader.Read(summaryStream, wdtVirtual);
            occupiedTiles = WdtTileIndexReader.ReadOccupiedTiles(summaryStream, summary);
        }

        bool isAlpha = AlphaWdtReader.IsAlphaWdt(wdtBytes);
        var records = new List<TextureNameTileRecord>();
        int skipped = 0;

        foreach (WdtTileCoordinate tile in occupiedTiles
                     .OrderBy(static t => t.TileY).ThenBy(static t => t.TileX)
                     .Take(maxTiles))
        {
            try
            {
                TerrainTileTensorPack? pack = TryBuildSyntheticMinimapPack(
                    catalog, clientRoot, mapName, wdtBytes, isAlpha, tile.TileX, tile.TileY, buildVersion);
                if (pack is null || pack.MclyTextureNames.Count == 0)
                {
                    skipped++;
                    continue;
                }

                records.Add(new TextureNameTileRecord(tile.TileX, tile.TileY, pack.MclyTextureNames.ToArray()));
            }
            catch (Exception ex)
            {
                skipped++;
                Console.Error.WriteLine($"  tile {tile.TileX},{tile.TileY} skipped: {ex.GetType().Name}: {ex.Message}");
            }
        }

        if (records.Count == 0)
        {
            Console.Error.WriteLine($"Error: no tile produced an MTEX texture-name table for map '{mapName}'.");
            Environment.ExitCode = 1;
            return;
        }

        var dump = new TextureNameDump(
            "terrain-texture-name-dump-v1", clientRoot, buildVersion, mapName, records.Count, records);

        string? outputDirectory = Path.GetDirectoryName(Path.GetFullPath(outputPath));
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);
        File.WriteAllText(
            outputPath,
            JsonSerializer.Serialize(dump, new JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine($"Wrote texture-name dump: {outputPath}");
        Console.WriteLine($"  map={mapName} build={buildVersion ?? "(unknown)"} tiles={records.Count} skipped={skipped}");
    }

    private static readonly System.Text.RegularExpressions.Regex WmoGroupFileSuffix =
        new(@"_\d{3}\.wmo$", System.Text.RegularExpressions.RegexOptions.IgnoreCase | System.Text.RegularExpressions.RegexOptions.Compiled);

    /// <summary>
    /// Spec 118: render every M2/MDX/WMO in the client (or a supplied asset list) top-down,
    /// headless, via WowViewer.Core.Renderer.ObjectCapture (no viewer, no ImGui, no GUI click
    /// per object), and stream one (image_rgb, mask) record per object to stdout using the same
    /// ARRY/ENDS wire format harvest-stream already uses -- harvester.raw_reader.read_tile_blob
    /// decodes this format generically with zero Python-side changes.
    /// </summary>
    static void RunCaptureObjects(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? assetListPath = GetOption(args, "--asset-list", "");
        int resolution = GetIntOption(args, "--resolution", "-r") ?? 256;
        int maxObjects = GetIntOption(args, "--limit", "-n") ?? int.MaxValue;

        if (string.IsNullOrWhiteSpace(clientRoot))
        {
            Console.Error.WriteLine("Error: --client-root is required.");
            Environment.ExitCode = 1;
            return;
        }
        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);
        string? buildVersion = DetectBuildVersionFromClientRoot(clientRoot);

        // Redirect Console.Out to stderr so stdout is pure binary (matches RunHarvestStream).
        Console.SetOut(Console.Error);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        List<string> assetPaths;
        if (!string.IsNullOrWhiteSpace(assetListPath) && File.Exists(assetListPath))
        {
            assetPaths = JsonSerializer.Deserialize<List<string>>(File.ReadAllText(assetListPath)) ?? [];
            Console.Error.WriteLine($"Loaded {assetPaths.Count} asset paths from {assetListPath}");
        }
        else
        {
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            assetPaths = [];
            foreach (string raw in catalog.ListFiles("*"))
            {
                string ext = Path.GetExtension(raw).ToLowerInvariant();
                if (ext is not (".m2" or ".mdx" or ".wmo"))
                    continue;
                string normalized = raw.Replace('/', '\\');
                if (ext == ".wmo" && WmoGroupFileSuffix.IsMatch(normalized))
                    continue; // WMO group files are not standalone objects.
                if (seen.Add(normalized))
                    assetPaths.Add(normalized);
            }
            assetPaths.Sort(StringComparer.OrdinalIgnoreCase);
            Console.Error.WriteLine($"Enumerated {assetPaths.Count} object paths from client listfile.");
        }

        using var headless = new HeadlessContext(resolution, resolution);
        using var capture = new ObjectCaptureRenderer(headless.GL, catalog);

        var stdout = Console.OpenStandardOutput();
        int captured = 0, skipped = 0, errors = 0;
        string? firstError = null;

        foreach (string assetPath in assetPaths)
        {
            if (captured + skipped + errors >= maxObjects)
                break;

            string ext = Path.GetExtension(assetPath).ToLowerInvariant();
            try
            {
                ObjectCaptureResult? result = ext == ".wmo"
                    ? CaptureWmoObject(catalog, capture, assetPath, resolution)
                    : CaptureMdxObject(catalog, capture, assetPath, resolution);

                if (result is null)
                {
                    skipped++;
                    continue;
                }

                byte[] innerBlob = BuildObjectRecordBlob(assetPath, ext, buildVersion, result);
                WriteHarvestStreamBlob(stdout, innerBlob);
                captured++;
                if (captured % 100 == 0)
                    Console.Error.WriteLine($"  captured {captured} (skipped {skipped}, errors {errors})...");
            }
            catch (Exception ex)
            {
                errors++;
                firstError ??= $"{assetPath}: {ex.Message}";
            }
        }

        byte[] endMarker = new byte[8];
        System.Text.Encoding.ASCII.GetBytes("ENDS").CopyTo(endMarker, 0);
        stdout.Write(endMarker, 0, 8);
        stdout.Flush();

        Console.Error.WriteLine($"Captured {captured} objects, {skipped} skipped, {errors} errors");
        if (firstError is not null)
            Console.Error.WriteLine($"First capture-objects error: {firstError}");
        if (captured == 0)
            Environment.ExitCode = 1;
    }

    private static ObjectCaptureResult? CaptureWmoObject(
        NativeMpqService catalog, ObjectCaptureRenderer capture, string assetPath, int resolution)
    {
        WmoV14ToV17Converter.WmoV14Data? wmo = WmoFullLoader.Load(catalog, assetPath);
        if (wmo is null || wmo.Groups.Count == 0)
            return null;
        return capture.CaptureWmo(wmo, resolution);
    }

    private static ObjectCaptureResult? CaptureMdxObject(
        NativeMpqService catalog, ObjectCaptureRenderer capture, string assetPath, int resolution)
    {
        byte[]? data = catalog.ReadFile(assetPath);
        if (data is null || data.Length == 0)
            return null;

        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);
        MdxFile mdx = MdxFile.Load(br);
        if (mdx.Geosets.Count == 0)
            return null;
        return capture.CaptureMdx(mdx, resolution);
    }

    /// <summary>
    /// Build one object record's inner ARRY blob: metadata (asset path, type, build, normalized
    /// path) + image_rgb (H,W,3) + mask (H,W), both derived from the capture's RGBA passes. The
    /// mask pass clears to black and the shader writes flat white for every covered fragment
    /// (alpha is always 255 from the FBO's opaque clear), so RGB-channel intensity -- not alpha
    /// -- is the coverage signal.
    /// </summary>
    private static byte[] BuildObjectRecordBlob(string assetPath, string extension, string? buildVersion, ObjectCaptureResult result)
    {
        int w = result.Width, h = result.Height;
        var imageRgb = new byte[h, w, 3];
        var mask = new byte[h, w];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int i = (y * w + x) * 4;
                imageRgb[y, x, 0] = result.ImageRgba[i + 0];
                imageRgb[y, x, 1] = result.ImageRgba[i + 1];
                imageRgb[y, x, 2] = result.ImageRgba[i + 2];

                byte mr = result.MaskRgba[i + 0], mg = result.MaskRgba[i + 1], mb = result.MaskRgba[i + 2];
                mask[y, x] = (byte)(Math.Max(mr, Math.Max(mg, mb)) > 0 ? 255 : 0);
            }
        }

        string normalized = assetPath.Replace('\\', '/').ToLowerInvariant();
        while (normalized.Contains("//"))
            normalized = normalized.Replace("//", "/");

        var metadata = new Dictionary<string, object?>
        {
            ["asset_path"] = assetPath,
            ["normalized_asset_path"] = normalized,
            ["asset_type"] = extension.TrimStart('.'),
            ["build"] = buildVersion,
            ["capture_mode"] = "orthographic_topdown",
            ["resolution"] = result.Width,
            ["bounds_min"] = new[] { result.BoundsMin.X, result.BoundsMin.Y, result.BoundsMin.Z },
            ["bounds_max"] = new[] { result.BoundsMax.X, result.BoundsMax.Y, result.BoundsMax.Z },
        };
        var arrays = new (string Name, Array Array)[]
        {
            ("image_rgb", imageRgb),
            ("mask", mask),
        };

        using var ms = new MemoryStream();
        RawArraySerializer.SerializeGeneric(metadata, arrays, ms);
        return ms.ToArray();
    }

    private static readonly System.Text.RegularExpressions.Regex ReliefTileNameRegex =
        new(@"^(?<prefix>.+)_(?<tx>\d+)_(?<ty>\d+)$", System.Text.RegularExpressions.RegexOptions.Compiled);
    private const int ReliefGridSize = 257;

    static void RunReliefToMap(string[] args)
    {
        string? manifestPath = GetOption(args, "--manifest", "");
        string? outputDir = GetOption(args, "--output-dir", "-o");
        string mapName = GetOption(args, "--map-name", "-m") ?? "relief_predicted";
        string texture = GetOption(args, "--texture", "-t") ?? "tileset\\ocean\\westfallseafloor.blp";
        float heightScale = GetFloatOption(args, "--height-scale", "") ?? 100.0f;
        bool noCenter = HasFlag(args, "--no-center");
        bool verify = HasFlag(args, "--verify");
        bool fillGaps = !HasFlag(args, "--no-fill-gaps");
        bool bakeMccv = HasFlag(args, "--bake-mccv");

        if (string.IsNullOrWhiteSpace(manifestPath) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Error: --manifest <inference_manifest.json> and --output-dir <dir> are required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!File.Exists(manifestPath))
        {
            Console.Error.WriteLine($"Error: manifest not found: {manifestPath}");
            Environment.ExitCode = 1;
            return;
        }

        Console.WriteLine(
            "NOTE: Spec 114 relief is PER-TILE relative height (contract v112.1) -- there is no " +
            "cross-tile absolute continuity. Adjacent tiles will show a height step at their " +
            "shared border; this reflects the model's relative-only contract, not a bug here.");

        using JsonDocument manifest = JsonDocument.Parse(File.ReadAllText(manifestPath));
        if (!manifest.RootElement.TryGetProperty("tiles", out JsonElement tilesElement) || tilesElement.ValueKind != JsonValueKind.Array)
        {
            Console.Error.WriteLine($"Error: manifest {manifestPath} has no 'tiles' array.");
            Environment.ExitCode = 1;
            return;
        }

        string manifestDirectory = Path.GetDirectoryName(Path.GetFullPath(manifestPath)) ?? ".";
        string mapDir = Path.Combine(outputDir, "World", "Maps", mapName);
        Directory.CreateDirectory(mapDir);

        var allTiles = new HashSet<(int TileX, int TileY)>();
        var wdlTiles = new List<WdlHeightTile>();
        int written = 0;
        var unplaced = new List<string>();

        foreach (JsonElement entry in tilesElement.EnumerateArray())
        {
            if (!entry.TryGetProperty("output", out JsonElement outputElement) || outputElement.ValueKind != JsonValueKind.String)
            {
                string inputHint = entry.TryGetProperty("input", out JsonElement inputHint2) ? inputHint2.GetString() ?? "<unknown>" : "<unknown>";
                unplaced.Add(inputHint);
                continue;
            }

            // The manifest records paths relative to the Python script's original working
            // directory, not relative to the manifest file itself, so a naive Combine can double
            // up path segments. Relief PNGs are always written flat next to the manifest
            // (v50_infer_geometry_detailer.py / v50_infer_direct_geometry.py), so resolve by
            // filename against the manifest's own directory instead of trusting the raw path.
            string reliefPath = Path.Combine(manifestDirectory, Path.GetFileName(outputElement.GetString()!));

            string inputPath = entry.TryGetProperty("input", out JsonElement inputElement) ? inputElement.GetString() ?? reliefPath : reliefPath;
            string stem = Path.GetFileNameWithoutExtension(inputPath);
            System.Text.RegularExpressions.Match match = ReliefTileNameRegex.Match(stem);
            if (!match.Success)
            {
                unplaced.Add(stem);
                continue;
            }

            int tileX = int.Parse(match.Groups["tx"].Value, CultureInfo.InvariantCulture);
            int tileY = int.Parse(match.Groups["ty"].Value, CultureInfo.InvariantCulture);
            if (tileX is < 0 or >= 64 || tileY is < 0 or >= 64)
            {
                unplaced.Add(stem);
                continue;
            }

            if (!File.Exists(reliefPath))
            {
                Console.Error.WriteLine($"  {stem}: relief PNG not found: {reliefPath}");
                unplaced.Add(stem);
                continue;
            }

            float[,] height2D = LoadReliefHeightmap(reliefPath, heightScale, noCenter);

            LkAdtData blank = BlankAdtFactory.CreateBlank(mapName, tileX, tileY, texture);
            float[] heightFlat = new float[ReliefGridSize * ReliefGridSize];
            for (int y = 0; y < ReliefGridSize; y++)
                for (int x = 0; x < ReliefGridSize; x++)
                    heightFlat[(y * ReliefGridSize) + x] = height2D[y, x];

            byte[] patchedBytes;
            if (bakeMccv)
            {
                if (!File.Exists(inputPath))
                {
                    Console.Error.WriteLine($"  {stem}: --bake-mccv requires the source minimap tile, not found: {inputPath}");
                    unplaced.Add(stem);
                    continue;
                }

                byte[,,] sourceRgb = LoadSourceRgb256(inputPath);
                var chunks = new LkMcnkData[blank.Chunks.Count];
                for (int i = 0; i < blank.Chunks.Count; i++)
                {
                    LkMcnkData template = blank.Chunks[i];
                    chunks[i] = BuildInferredChunk(template, heightFlat, sourceRgb);
                }

                var populated = new LkAdtData
                {
                    MapName = blank.MapName,
                    TileX = blank.TileX,
                    TileY = blank.TileY,
                    TextureNames = blank.TextureNames,
                    ModelNames = blank.ModelNames,
                    WorldModelNames = blank.WorldModelNames,
                    ModelPlacements = blank.ModelPlacements,
                    WorldModelPlacements = blank.WorldModelPlacements,
                    Chunks = chunks,
                    MhdrFlags = blank.MhdrFlags,
                    MfboFlightBounds = blank.MfboFlightBounds,
                };
                patchedBytes = LkAdtWriter.Build(populated);
            }
            else
            {
                byte[] blankBytes = LkAdtWriter.Build(blank);
                patchedBytes = AdtTerrainWriter.ApplyHeightmap(blankBytes, "synthetic.adt", heightFlat);
            }

            string adtPath = Path.Combine(mapDir, $"{mapName}_{tileX}_{tileY}.adt");
            File.WriteAllBytes(adtPath, patchedBytes);

            if (verify)
            {
                WorldTerrainTileData readBack = WorldTerrainTileBuilder.Read(adtPath);
                float[]? readHeights = readBack.Heightmap?.Heights;
                if (readHeights is null || readHeights.Length != heightFlat.Length)
                {
                    Console.Error.WriteLine($"  VERIFY FAIL {mapName}_{tileX}_{tileY}: heightmap missing or wrong length on readback.");
                }
                else
                {
                    // MCVT only stores the quincunx checkerboard subset of the 257x257 grid (145
                    // samples/chunk: 9x9 outer + 8x8 diagonally-offset inner); the remaining grid
                    // positions are interpolated by any reader, never stored verbatim. Comparing
                    // the full dense grid would flag that expected interpolation loss as a false
                    // "write" bug, so only the positions actually written to MCVT are checked here.
                    float maxAbsDiff = 0f;
                    for (int cy = 0; cy < 16; cy++)
                    {
                        for (int cx = 0; cx < 16; cx++)
                        {
                            for (int sampleIndex = 0; sampleIndex < 145; sampleIndex++)
                            {
                                VerifyResolveStoredSamplePosition(cx, cy, sampleIndex, out int sampleX, out int sampleY);
                                int flatIndex = (sampleY * ReliefGridSize) + sampleX;
                                maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(readHeights[flatIndex] - heightFlat[flatIndex]));
                            }
                        }
                    }
                    if (maxAbsDiff > 0.01f)
                        Console.Error.WriteLine($"  VERIFY FAIL {mapName}_{tileX}_{tileY}: max abs height diff at stored MCVT samples = {maxAbsDiff:F4}");
                }
            }

            allTiles.Add((tileX, tileY));
            wdlTiles.Add(WdlWriter.ExtractTileHeightsFromAlpha(height2D, tileX, tileY));
            written++;
        }

        if (allTiles.Count == 0)
        {
            Console.Error.WriteLine("Error: no tiles were written; nothing to build a WDT/WDL from.");
            Environment.ExitCode = 1;
            return;
        }

        int gapsFilled = 0;
        if (fillGaps)
        {
            // Inference input tiles are commonly sparse -- e.g. split-minimap-image skips
            // near-uniform blank source tiles, so most of a source image's tile grid never gets a
            // relief prediction at all. Left as-is, the WDT only declares the inferred tiles,
            // leaving real holes (no ADT, no geometry) at every skipped coordinate inside the
            // covered area -- not just a height mismatch at a shared border, an actual absence.
            // Fill every missing tile inside the bounding rectangle of what WAS inferred with a
            // flat/blank ADT (BlankAdtFactory's already-established "white plate = valid empty
            // terrain" convention), so the map is a complete, gap-free quilt across its own extent.
            int minTileX = allTiles.Min(static t => t.TileX);
            int maxTileX = allTiles.Max(static t => t.TileX);
            int minTileY = allTiles.Min(static t => t.TileY);
            int maxTileY = allTiles.Max(static t => t.TileY);

            for (int ty = minTileY; ty <= maxTileY; ty++)
            {
                for (int tx = minTileX; tx <= maxTileX; tx++)
                {
                    if (allTiles.Contains((tx, ty)))
                        continue;

                    LkAdtData blankGapTile = BlankAdtFactory.CreateBlank(mapName, tx, ty, texture);
                    byte[] blankGapBytes = LkAdtWriter.Build(blankGapTile);
                    string gapAdtPath = Path.Combine(mapDir, $"{mapName}_{tx}_{ty}.adt");
                    File.WriteAllBytes(gapAdtPath, blankGapBytes);

                    allTiles.Add((tx, ty));
                    wdlTiles.Add(BlankAdtFactory.CreateBlankWdlTile(tx, ty));
                    gapsFilled++;
                }
            }
        }

        string wdtPath = Path.Combine(mapDir, $"{mapName}.wdt");
        LkWdtWriter.Write(wdtPath, allTiles, new LkWdtWriteOptions { HasMccv = bakeMccv });

        string wdlPath = Path.Combine(mapDir, $"{mapName}.wdl");
        WdlWriter.Write(wdlPath, wdlTiles);

        Console.WriteLine($"Wrote {written} inferred ADT(s)" +
            (fillGaps ? $" + {gapsFilled} flat gap-fill ADT(s)" : string.Empty) +
            $", 1 WDT, 1 WDL to {mapDir}");
        Console.WriteLine($"  WDT: {wdtPath}");
        Console.WriteLine($"  WDL: {wdlPath}");
        if (unplaced.Count > 0)
        {
            Console.WriteLine($"Skipped {unplaced.Count} entr(y/ies) with no relief output or unparseable '..._<tx>_<ty>' filename " +
                $"(first 5): {string.Join(", ", unplaced.Take(5))}");
        }
    }

    // Ported from AdtTerrainWriter's private ResolveTileSampleCoordinates/GetVertexPosition so
    // --verify checks the exact same quincunx sample positions AdtTerrainWriter writes to MCVT.
    private static void VerifyResolveStoredSamplePosition(int chunkX, int chunkY, int sampleIndex, out int sampleX, out int sampleY)
    {
        int remaining = sampleIndex;
        int row = 0, col = 0;
        bool isInner = false;
        for (int currentRow = 0; currentRow < 17; currentRow++)
        {
            int rowSize = (currentRow % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = currentRow;
                col = remaining;
                isInner = (currentRow % 2) != 0;
                break;
            }
            remaining -= rowSize;
        }

        int localX = isInner ? (col * 2) + 1 : col * 2;
        int localY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;
        sampleX = (chunkX * 16) + localX;
        sampleY = (chunkY * 16) + localY;
    }

    private static byte[,,] LoadSourceRgb256(string path)
    {
        using Image<Rgb24> image = Image.Load<Rgb24>(path);
        var rgb = new byte[image.Height, image.Width, 3];
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<Rgb24> row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    rgb[y, x, 0] = row[x].R;
                    rgb[y, x, 1] = row[x].G;
                    rgb[y, x, 2] = row[x].B;
                }
            }
        });
        return rgb;
    }

    /// <summary>
    /// Builds one MCNK's height/normal/MCCV arrays from a dense 257x257 relative-relief grid and,
    /// optionally, the source minimap tile used to infer it. Constructing fresh LkMcnkData (rather
    /// than byte-patching an existing serialized ADT, as AdtTerrainWriter.ApplyHeightmap does) is
    /// required here because MCCV is a whole new subchunk with no existing byte range to patch into.
    /// </summary>
    private static LkMcnkData BuildInferredChunk(LkMcnkData template, float[] heightFlat257, byte[,,]? sourceRgb256)
    {
        var heights = new float[145];
        var normals = new byte[448];
        byte[]? mccv = sourceRgb256 is null ? null : new byte[580];
        int sourceHeight = sourceRgb256?.GetLength(0) ?? 0;
        int sourceWidth = sourceRgb256?.GetLength(1) ?? 0;

        for (int sampleIndex = 0; sampleIndex < 145; sampleIndex++)
        {
            VerifyResolveStoredSamplePosition(template.IndexX, template.IndexY, sampleIndex, out int sampleX, out int sampleY);
            heights[sampleIndex] = heightFlat257[(sampleY * ReliefGridSize) + sampleX];

            Vector3 normal = AdtTerrainMath.ComputeNormal(heightFlat257, sampleX, sampleY);
            int normalOffset = sampleIndex * 3;
            normals[normalOffset + 0] = EncodeNormalComponent(normal.X);
            normals[normalOffset + 1] = EncodeNormalComponent(normal.Z);
            normals[normalOffset + 2] = EncodeNormalComponent(normal.Y);

            if (mccv is not null && sourceRgb256 is not null)
            {
                int imageX = Math.Clamp(sampleX, 0, sourceWidth - 1);
                int imageY = Math.Clamp(sampleY, 0, sourceHeight - 1);
                int mccvOffset = sampleIndex * 4;
                // TerrainRenderer's shader computes tintColor = clamp(vertexColor.rgb * 2, 0, 2), so
                // halving the source byte here recovers it after that x2 unscaling. tintStrength =
                // clamp(vertexColor.a * 2 - 1, 0, 1): alpha must be 255 for full tint strength, or
                // the shader blends toward "no tint" instead of showing this color at all.
                mccv[mccvOffset + 0] = (byte)(sourceRgb256[imageY, imageX, 0] / 2);
                mccv[mccvOffset + 1] = (byte)(sourceRgb256[imageY, imageX, 1] / 2);
                mccv[mccvOffset + 2] = (byte)(sourceRgb256[imageY, imageX, 2] / 2);
                mccv[mccvOffset + 3] = 255;
            }
        }

        return new LkMcnkData
        {
            IndexX = template.IndexX,
            IndexY = template.IndexY,
            Flags = template.Flags,
            AreaId = template.AreaId,
            NLayers = template.NLayers,
            HoleMask = template.HoleMask,
            BaseHeight = template.BaseHeight,
            Heights = heights,
            Normals = normals,
            ShadowMap = template.ShadowMap,
            AlphaMapData = template.AlphaMapData,
            AlphaMapSize = template.AlphaMapSize,
            Layers = template.Layers,
            DoodadRefs = template.DoodadRefs,
            WorldModelRefs = template.WorldModelRefs,
            LiquidData = template.LiquidData,
            MccvColors = mccv,
            MclvLighting = template.MclvLighting,
            PosX = template.PosX,
            PosY = template.PosY,
            PosZ = template.PosZ,
        };
    }

    private static byte EncodeNormalComponent(float value) =>
        unchecked((byte)(sbyte)MathF.Round(Math.Clamp(value, -1f, 1f) * 127f));

    private static float[,] LoadReliefHeightmap(string reliefPath, float heightScale, bool noCenter)
    {
        using Image<L16> image = Image.Load<L16>(reliefPath);
        if (image.Width != ReliefGridSize || image.Height != ReliefGridSize)
            throw new InvalidDataException($"{reliefPath} must be {ReliefGridSize}x{ReliefGridSize}, got {image.Width}x{image.Height}.");

        var height = new float[ReliefGridSize, ReliefGridSize];
        double sum = 0;
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < accessor.Height; y++)
            {
                Span<L16> row = accessor.GetRowSpan(y);
                for (int x = 0; x < row.Length; x++)
                {
                    float normalized = row[x].PackedValue / 65535.0f;
                    height[y, x] = normalized;
                    sum += normalized;
                }
            }
        });

        if (!noCenter)
        {
            float mean = (float)(sum / (ReliefGridSize * ReliefGridSize));
            for (int y = 0; y < ReliefGridSize; y++)
                for (int x = 0; x < ReliefGridSize; x++)
                    height[y, x] -= mean;
        }

        for (int y = 0; y < ReliefGridSize; y++)
            for (int x = 0; x < ReliefGridSize; x++)
                height[y, x] *= heightScale;

        return height;
    }

    static void RunSyntheticMinimap(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapName = GetOption(args, "--map", "-m");
        string? outputDirectory = GetOption(args, "--output-dir", "-o");
        string? requestedBuildVersion = GetOption(args, "--build", "-b");
        int resolution = GetIntOption(args, "--resolution", "-r") ?? TerrainMinimapCompositor.DefaultResolution;
        int maxTiles = GetIntOption(args, "--limit", "-n") ?? int.MaxValue;
        // Spec 112 T007 diagnostic: bound the tile-composition parallelism (1 = fully sequential).
        // minimap_rgb_1024 coverage trails minimap_rgb on real builds; an A/B run of the same map
        // at --synthesis-workers 1 vs default isolates whether in-process parallel decode is the
        // loss mechanism before any blind "fix".
        int synthesisWorkers = GetIntOption(args, "--synthesis-workers", "-w") ?? -1;
        if (synthesisWorkers is 0 or < -1)
        {
            Console.Error.WriteLine("Error: --synthesis-workers must be -1 (unbounded, default) or a positive worker count.");
            Environment.ExitCode = 1;
            return;
        }
        int? requestedTileX = GetIntOption(args, "--tile-x", "-x");
        int? requestedTileY = GetIntOption(args, "--tile-y", "-y");
        string? requestedTileList = GetOption(args, "--tile-list", "");
        string? requestedTimeOfDay = GetOption(args, "--time-hours", "-t");
        TimeOfDayClock timeOfDay;
        if (requestedTimeOfDay is null)
        {
            timeOfDay = new TimeOfDayClock(12, 0);
        }
        else if (!TimeOfDayClock.TryParse(requestedTimeOfDay, out timeOfDay))
        {
            Console.Error.WriteLine(
                "Error: --time-hours must be a time within one day: HHmm (1215), HH:mm (12:15), or decimal hours (12.25).");
            Environment.ExitCode = 1;
            return;
        }

        float timeOfDayHours = timeOfDay.Hours;
        bool emitPerTile = HasFlag(args, "--per-tile");
        bool emitWholeMap = HasFlag(args, "--whole-map");
        bool bakeMcsh = HasFlag(args, "--bake-mcsh");
        bool includeWmos = HasFlag(args, "--include-wmos");
        bool emitAuthoredReference = HasFlag(args, "--authored-reference");
        // Cast shadows are on by default: Lambert alone cannot darken flat ground behind a ridge,
        // which is why synthesized tiles read as shadowless next to authored minimaps. The opt-out
        // exists for A/B comparison and for callers that need the pre-v0.5.3 hillshade-only look.
        // Cast shadows are an ADDITION, not a reconstruction: the client shaded terrain with Lambert
        // plus baked MCSH and never ray-traced. The era profile decides the default (off for Alpha);
        // --cast-shadows forces them on and --no-cast-shadows forces them off.
        bool? castShadowsOverride = HasFlag(args, "--cast-shadows") ? true
            : HasFlag(args, "--no-cast-shadows") ? false
            : null;
        // --match-time: infer each tile's best-matching time-of-day from the authored minimap
        // (MinimapShadingMatch hour sweep) and render the synthesized tile at THAT hour, instead
        // of one global --time-hours. Requires --authored-reference.
        bool matchTime = HasFlag(args, "--match-time");
        if (matchTime && !emitAuthoredReference)
        {
            Console.Error.WriteLine("Error: --match-time requires --authored-reference (it infers the hour from the authored minimap).");
            Environment.ExitCode = 1;
            return;
        }
        // --measure-sun: score each authored tile across sun BEARINGS as well as hours, so real
        // client data can settle whether this era's sun is fixed or sweeps east-to-west.
        bool measureSun = HasFlag(args, "--measure-sun");
        if (measureSun && !emitAuthoredReference)
        {
            Console.Error.WriteLine("Error: --measure-sun requires --authored-reference (it scores against the authored minimap).");
            Environment.ExitCode = 1;
            return;
        }
        AzimuthSweepAccumulator? azimuthSweep = measureSun ? new AzimuthSweepAccumulator() : null;
        // --light-overlay: emit a third PNG per tile plotting LIT light positions with their colour
        // swatches and influence domes. Diagnostic only -- deliberately a separate file so authored
        // light annotations can never contaminate the terrain RGB corpus.
        bool lightOverlay = HasFlag(args, "--light-overlay");
        // --score: measure agreement with the authored minimap instead of leaving it to the eye,
        // and render the pre-cast-shadow/pre-linear-shading configuration alongside so the report
        // states whether the current settings are actually an improvement.
        bool scoreAgainstAuthored = HasFlag(args, "--score");
        if (scoreAgainstAuthored && !emitAuthoredReference)
        {
            Console.Error.WriteLine("Error: --score requires --authored-reference (there is nothing to score against otherwise).");
            Environment.ExitCode = 1;
            return;
        }
        SyntheticMinimapScorecard? scorecard = scoreAgainstAuthored ? new SyntheticMinimapScorecard() : null;
        const string BaselineVariantName = "legacy";
        const string CurrentVariantName = "current";
        // --dxt1-parity: emit a DXT1-compressed parity companion per tile alongside the pristine
        // render, so authored and synthetic tiles compare on equal terms (FR-015). The pristine
        // render remains the primary output; the parity companion is a parity-only companion.
        bool dxt1Parity = HasFlag(args, "--dxt1-parity");
        // --encoding-survey: report the per-build/map distribution of encodings found, so the DXT1
        // assumption is verified rather than inherited (FR-013).
        bool encodingSurvey = HasFlag(args, "--encoding-survey");
        // --lighting-baseline: survey authored tiles for a shared lighting baseline and account for
        // it when scoring (FR-016).
        bool lightingBaseline = HasFlag(args, "--lighting-baseline");
        if (lightingBaseline && !emitAuthoredReference)
        {
            Console.Error.WriteLine("Error: --lighting-baseline requires --authored-reference (it surveys the authored minimap).");
            Environment.ExitCode = 1;
            return;
        }
        // --textureless-residuals: emit per-tile shading-only images (no objects, no textures) plus a
        // stitched whole-map output — the clean terrain-shadow residual signal (FR-022). The albedo
        // is replaced with neutral white so only the Lambert hillshade + cast shadows remain.
        bool texturelessResiduals = HasFlag(args, "--textureless-residuals");
        if (!emitPerTile && !emitWholeMap)
        {
            emitPerTile = true;
            emitWholeMap = true;
        }

        if (requestedTileX.HasValue != requestedTileY.HasValue
            || requestedTileX is < 0 or > 63
            || requestedTileY is < 0 or > 63)
        {
            Console.Error.WriteLine("Error: --tile-x and --tile-y must be supplied together within 0..63.");
            Environment.ExitCode = 1;
            return;
        }

        if (requestedTileX.HasValue && !string.IsNullOrWhiteSpace(requestedTileList))
        {
            Console.Error.WriteLine("Error: choose either --tile-x/--tile-y or --tile-list, not both.");
            Environment.ExitCode = 1;
            return;
        }

        if (GetOption(args, "--dbd-dir", "") is not null)
        {
            Console.Error.WriteLine(
                "Error: --dbd-dir does not apply to synthetic-minimap; Light DBC profiles are viewer-only.");
            Environment.ExitCode = 1;
            return;
        }

        if (!TryParseSyntheticMinimapTileList(requestedTileList, out HashSet<(int TileX, int TileY)> listedTiles, out string? tileListError))
        {
            Console.Error.WriteLine($"Error: {tileListError}");
            Environment.ExitCode = 1;
            return;
        }

        if (string.IsNullOrWhiteSpace(clientRoot)
            || string.IsNullOrWhiteSpace(mapName)
            || string.IsNullOrWhiteSpace(outputDirectory))
        {
            Console.Error.WriteLine("Error: --client-root, --map, and --output-dir are required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        if (resolution is < 1 or > 4096)
        {
            Console.Error.WriteLine("Error: --resolution must be within 1..4096.");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);
        string? buildVersion = requestedBuildVersion ?? DetectBuildVersionFromClientRoot(clientRoot);

        // Parsed only once the build is known: the era profile it seeds is chosen by build, since
        // minimap generation differs between Alpha (0.5.3), Beta 1 (0.6.0) and release (1.0.0).
        SyntheticMinimapTuning? tuning = TryParseSyntheticMinimapTuning(args, buildVersion, out string? tuningError);
        if (tuningError is not null)
        {
            Console.Error.WriteLine($"Error: {tuningError}");
            Environment.ExitCode = 1;
            return;
        }

        bool castShadows = castShadowsOverride ?? tuning!.Era.CastShadowsEnabled;

        Console.WriteLine($"Minimap era profile: {tuning!.Era.EraLabel} ({tuning.Era.Name})");
        if (!tuning.ExactEraMatch)
        {
            Console.WriteLine(
                $"  WARNING: build '{buildVersion ?? "unknown"}' did not match a known era; " +
                $"falling back to {MinimapEraProfile.Default.EraLabel} rules. Pass --era to be explicit.");
        }
        Console.WriteLine(
            $"  Sun: {tuning.Era.AzimuthModel} at {tuning.Era.NoonAzimuthDegrees:0}deg " +
            $"({tuning.Era.AzimuthProvenance})");
        if (tuning.Era.HasUnverifiedSolarModel)
        {
            Console.WriteLine(
                "  NOTE: this era's solar model is carried over from a different build's trace and " +
                "has never been verified against its own authored minimaps.");
        }

        Directory.CreateDirectory(outputDirectory);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        IReadOnlyList<MinimapLightMarker> lightMarkers = lightOverlay
            ? LoadMinimapLightMarkers(catalog, mapName, timeOfDayHours / 24f)
            : [];

        string wdtPath = $"World\\Maps\\{mapName}\\{mapName}.wdt";
        byte[]? wdtBytes = catalog.ReadFile(wdtPath);
        if (wdtBytes is null || wdtBytes.Length == 0)
        {
            Console.Error.WriteLine($"Error: could not read WDT '{wdtPath}' from client.");
            Environment.ExitCode = 1;
            return;
        }

        IReadOnlyList<WdtTileCoordinate> occupiedTiles;
        using (var summaryStream = new MemoryStream(wdtBytes, writable: false))
        {
            MapFileSummary summary = MapFileSummaryReader.Read(summaryStream, wdtPath);
            occupiedTiles = WdtTileIndexReader.ReadOccupiedTiles(summaryStream, summary);
        }

        if (requestedTileX.HasValue)
        {
            occupiedTiles = occupiedTiles
                .Where(tile => tile.TileX == requestedTileX.Value && tile.TileY == requestedTileY!.Value)
                .ToArray();
        }
        else if (listedTiles.Count > 0)
        {
            HashSet<(int TileX, int TileY)> occupiedSet = occupiedTiles
                .Select(static tile => (tile.TileX, tile.TileY))
                .ToHashSet();
            (int TileX, int TileY)[] missing = listedTiles
                .Where(tile => !occupiedSet.Contains(tile))
                .OrderBy(static tile => tile.TileY)
                .ThenBy(static tile => tile.TileX)
                .ToArray();
            if (missing.Length > 0)
            {
                Console.Error.WriteLine(
                    $"Error: map '{mapName}' does not mark requested tile(s) occupied: " +
                    string.Join(", ", missing.Select(static tile => $"{tile.TileX},{tile.TileY}")));
                Environment.ExitCode = 1;
                return;
            }

            occupiedTiles = occupiedTiles
                .Where(tile => listedTiles.Contains((tile.TileX, tile.TileY)))
                .ToArray();
        }

        if (occupiedTiles.Count == 0)
        {
            Console.Error.WriteLine($"Error: map '{mapName}' has no occupied terrain tiles to synthesize.");
            Environment.ExitCode = 1;
            return;
        }

        bool isAlpha = AlphaWdtReader.IsAlphaWdt(wdtBytes);
        float gameTime = timeOfDayHours / 24f;
        SyntheticMinimapLightingProfile lighting = ResolveSyntheticMinimapLighting(gameTime, castShadows, tuning);
        if (bakeMcsh)
            lighting = WithBakedMcsh(lighting);
        Console.WriteLine($"Synthetic minimap lighting: {lighting.Source} ({lighting.EvidenceState})");
        Console.WriteLine(bakeMcsh
            ? "  MCSH: explicitly baked into RGB preview (exceptional historical-minimap mode)."
            : "  MCSH: omitted from normal minimap RGB (real minimaps do not bake MCSH).");
        Console.WriteLine(castShadows
            ? "  Cast shadows: analytic sun-driven ray march over this tile's MCVT heightfield " +
              $"(strength {lighting.Lighting.CastShadowStrength:0.00}, softness " +
              $"{lighting.Lighting.CastShadowSoftness:0.0}wu); shadows do not cross tile seams."
            : "  Cast shadows: OFF (era default for this build; the client never ray-traced terrain "
              + "shadows). Lambert hillshading only. Use --cast-shadows to enable.");
        Vector3 ambientColor = lighting.Lighting.AmbientColor;
        Console.WriteLine(
            $"  Contrast: ambient {ambientColor.X:0.000},{ambientColor.Y:0.000},{ambientColor.Z:0.000}" +
            $"; light gain {lighting.Lighting.LinearLightGain:0.000}" +
            (tuning!.LightGainOverride is null ? " (derived from ambient)" : " (--light-gain override)") +
            ". Tune with --ambient / --cast-shadow-strength; the derived gain holds overall " +
            "brightness steady as they change.");
        MinimapLiquidStyle water = tuning.LiquidPalette.Water;
        Console.WriteLine(
            $"  Liquids: paired _liquid PNGs use {tuning.LiquidPalette.RenderProfile}; " +
            $"water {water.Red:0.00},{water.Green:0.00},{water.Blue:0.00} @ opacity {water.Opacity:0.00}; " +
            "the terrain baseline remains liquid-free. Tune with --liquid-palette / --water-color.");
        if (!string.IsNullOrWhiteSpace(lighting.Diagnostic))
            Console.WriteLine($"  Lighting note: {lighting.Diagnostic}");

        // Spec 113: --detail selects real-texel sampling (the SR HR target); default stays the
        // material-average view every existing consumer expects.
        bool detailTexels = HasFlag(args, "--detail");
        if (detailTexels)
            Console.WriteLine($"Render mode: detail (real texel sampling, {TerrainMinimapCompositionOptions.TextureRepeatsPerChunk} repeat(s)/chunk)");
        var results = new ConcurrentBag<SyntheticMinimapTileResult>();
        var emittedTiles = new ConcurrentDictionary<(int TileX, int TileY), string>();
        var emittedLiquidTiles = new ConcurrentDictionary<(int TileX, int TileY), string>();
        var emittedResidualTiles = new ConcurrentDictionary<(int TileX, int TileY), string>();
        // Collectors for the encoding survey (raw authored BLP bytes) and the lighting-baseline
        // survey (authored RGB tiles), populated during the parallel pass and reported after it.
        var surveyedBlpBytes = new ConcurrentBag<byte[]>();
        var surveyedAuthoredTiles = new ConcurrentBag<byte[,,]>();
        string tilesDirectory = emitPerTile
            ? Path.Combine(outputDirectory, "tiles")
            : Path.Combine(outputDirectory, ".stitch-cache", $"{mapName}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tilesDirectory);

        var sortedTiles = occupiedTiles.OrderBy(tile => tile.TileY).ThenBy(tile => tile.TileX).ToArray();
        var tilesToProcess = sortedTiles.Take(maxTiles).ToArray();

        var parallelOptions = new System.Threading.Tasks.ParallelOptions
        {
            MaxDegreeOfParallelism = synthesisWorkers
        };
        System.Threading.Tasks.Parallel.ForEach(tilesToProcess, parallelOptions, tile =>
        {
            string stage = "decoding terrain";
            try
            {
                TerrainTileTensorPack? pack = TryBuildSyntheticMinimapPack(
                    catalog,
                    clientRoot,
                    mapName,
                    wdtBytes,
                    isAlpha,
                    tile.TileX,
                    tile.TileY,
                    buildVersion);
                if (pack is null)
                {
                    results.Add(new SyntheticMinimapTileResult(tile.TileX, tile.TileY, "skipped", null, 0, "tile data could not be decoded"));
                    return;
                }

                byte[,,]? authoredReference = null;
                if (emitAuthoredReference)
                {
                    stage = "decoding authored client minimap";
                    authoredReference = TryLoadMinimapFromMpq(catalog, mapName, tile.TileX, tile.TileY);
                    if (authoredReference is not null && HasVisibleRgb(authoredReference))
                    {
                        surveyedAuthoredTiles.Add(authoredReference);
                        byte[]? blpBytes = TryLoadMinimapBlpBytes(catalog, mapName, tile.TileX, tile.TileY);
                        if (blpBytes is not null)
                            surveyedBlpBytes.Add(blpBytes);
                    }

                    if (authoredReference is null || !HasVisibleRgb(authoredReference))
                    {
                        results.Add(new SyntheticMinimapTileResult(
                            tile.TileX,
                            tile.TileY,
                            "skipped",
                            null,
                            0,
                            "--authored-reference was requested but no visible real client minimap decoded"));
                        Console.Error.WriteLine(
                            $"Synthetic minimap tile {tile.TileX:D2},{tile.TileY:D2} skipped: " +
                            "no visible authored client minimap decoded for comparison.");
                        return;
                    }
                }

                if (azimuthSweep is not null && authoredReference is not null)
                {
                    stage = "sweeping solar azimuth against the authored minimap";
                    Dictionary<int, byte[,,]> sweepTextures = LoadSyntheticMinimapTextures(catalog, pack);
                    MinimapShadingMatch.AzimuthSweepResult sweep = MinimapShadingMatch.SweepSolarAzimuth(
                        pack, sweepTextures, authoredReference);
                    if (sweep.Evaluable)
                        azimuthSweep.Add(tile.TileX, tile.TileY, sweep);
                }

                SyntheticMinimapLightingProfile tileLighting = lighting;
                if (matchTime && authoredReference is not null)
                {
                    // --match-time: infer this tile's best-matching time-of-day from the authored
                    // minimap's shading (MinimapShadingMatch sweeps every hour through the
                    // production compositor and correlates luma patterns), then render at that
                    // hour so the synthesized tile matches the authored sun elevation.
                    stage = "inferring per-tile time-of-day from authored minimap";
                    float? inferredHours = TryInferAuthoredMinimapHour(
                        pack, catalog, mapName, authoredReference, buildVersion);
                    if (inferredHours is { } inferred && float.IsFinite(inferred))
                    {
                        tileLighting = ResolveSyntheticMinimapLighting(inferred / 24f, castShadows, tuning);
                        // Re-resolving the profile for the inferred hour rebuilds it from scratch,
                        // which would silently drop an explicit --bake-mcsh for match-time tiles.
                        if (bakeMcsh)
                            tileLighting = WithBakedMcsh(tileLighting);
                        Console.WriteLine(
                            $"  Tile {tile.TileX:D2},{tile.TileY:D2}: inferred time-of-day {inferred:0.00}h " +
                            $"(was {timeOfDayHours:0.00}h) from authored minimap shading");
                    }
                    else
                    {
                        Console.WriteLine(
                            $"  Tile {tile.TileX:D2},{tile.TileY:D2}: shading match not evaluable; " +
                            $"keeping global {timeOfDayHours:0.00}h");
                    }
                }
                var compositionOptions = new TerrainMinimapCompositionOptions(
                    resolution,
                    tileLighting.Lighting,
                    DetailTexels: detailTexels);

                stage = "decoding terrain textures";
                Dictionary<int, byte[,,]> textures = LoadSyntheticMinimapTextures(catalog, pack);
                // A tile with no MTEX names is untextured, not broken: it still has valid MCVT
                // heights + MCNR normals, and the compositor renders it with a neutral white base
                // under the same lighting/cast-shadow path. Only skip when textures were declared
                // but none could be decoded.
                bool untextured = !pack.MclyTextureNames.Any(static n => !string.IsNullOrWhiteSpace(n));
                if (textures.Count == 0 && !untextured)
                {
                    results.Add(new SyntheticMinimapTileResult(tile.TileX, tile.TileY, "skipped", null, 0, "no referenced BLP texture could be decoded"));
                    return;
                }

                string tilePath = Path.Combine(tilesDirectory, $"{mapName}_{tile.TileX:D2}_{tile.TileY:D2}_synthesized.png");
                string liquidTilePath = Path.Combine(tilesDirectory, $"{mapName}_{tile.TileX:D2}_{tile.TileY:D2}_synthesized_liquid.png");
                string? authoredTilePath = emitAuthoredReference
                    ? Path.Combine(tilesDirectory, $"{mapName}_{tile.TileX:D2}_{tile.TileY:D2}_authored.png")
                    : null;
                string? comparisonTilePath = emitAuthoredReference
                    ? Path.Combine(tilesDirectory, $"{mapName}_{tile.TileX:D2}_{tile.TileY:D2}_authored_vs_synthesized.png")
                    : null;
                stage = "compositing terrain minimap";
                using Image<Rgba32> image = TerrainMinimapCompositor.Compose(pack, textures, compositionOptions);
                if (!HasVisibleRgb(image))
                    throw new InvalidDataException(
                        "composed terrain RGB is entirely black; the tile is not valid visual proof");
                stage = "compositing liquid minimap";
                using Image<Rgba32> liquidImage = TerrainMinimapLiquidCompositor.Compose(
                    image, pack, out int liquidPixelCount, tuning.LiquidPalette);
                stage = "writing terrain PNG";
                image.SaveAsPng(tilePath);
                stage = "writing liquid PNG";
                liquidImage.SaveAsPng(liquidTilePath);
                if (dxt1Parity)
                {
                    // Emit a DXT1-compressed parity companion so authored and synthetic tiles compare
                    // on equal terms (FR-015). The pristine liquid render remains the primary output.
                    stage = "writing DXT1 parity companion";
                    using Image<Rgba32> parityImage = Dxt1TileCodec.EncodeDecode(liquidImage);
                    parityImage.SaveAsPng(Path.Combine(
                        tilesDirectory,
                        $"{mapName}_{tile.TileX:D2}_{tile.TileY:D2}_dxt1.png"));
                }
                if (texturelessResiduals)
                {
                    // Emit a textureless terrain-shadow residual: compose with a neutral white albedo
                    // so only the Lambert hillshade + cast shadows remain (no objects, no textures).
                    // This is the clean training signal for residuals-based reconstruction (FR-022).
                    stage = "compositing textureless residual";
                    Dictionary<int, byte[,,]> neutralTextures = CreateNeutralWhiteTextures(textures);
                    using Image<Rgba32> residualImage = TerrainMinimapCompositor.Compose(
                        pack, neutralTextures, compositionOptions);
                    residualImage.SaveAsPng(Path.Combine(
                        tilesDirectory,
                        $"{mapName}_{tile.TileX:D2}_{tile.TileY:D2}_residual.png"));
                    emittedResidualTiles[(tile.TileX, tile.TileY)] = Path.Combine(
                        tilesDirectory,
                        $"{mapName}_{tile.TileX:D2}_{tile.TileY:D2}_residual.png");
                }
                if (lightOverlay)
                {
                    stage = "compositing LIT light overlay";
                    using Image<Rgba32> overlayImage = TerrainMinimapLightOverlayCompositor.Compose(
                        liquidImage, tile.TileX, tile.TileY, lightMarkers, out int visibleLights);
                    overlayImage.SaveAsPng(Path.Combine(
                        tilesDirectory,
                        $"{mapName}_{tile.TileX:D2}_{tile.TileY:D2}_lights.png"));
                    if (visibleLights > 0)
                        Console.WriteLine($"  Tile {tile.TileX:D2},{tile.TileY:D2}: {visibleLights} LIT light(s) reach this tile");
                }
                if (scorecard is not null && authoredReference is not null)
                {
                    stage = "scoring against the authored minimap";
                    scorecard.Add(
                        tile.TileX,
                        tile.TileY,
                        CurrentVariantName,
                        MinimapComparisonMetrics.Compare(authoredReference, liquidImage));

                    // Re-render with the configuration that predates cast shadows, linear-space
                    // shading and the era liquid palette, so the report compares like with like on
                    // the same tile rather than against a remembered number.
                    var legacyOptions = new TerrainMinimapCompositionOptions(
                        resolution,
                        TerrainMinimapLighting.CreateWhiteTopEdge(gameTime) with { ToneMapped = true },
                        DetailTexels: detailTexels);
                    using Image<Rgba32> legacyTerrain = TerrainMinimapCompositor.Compose(pack, textures, legacyOptions);
                    using Image<Rgba32> legacyLiquid = TerrainMinimapLiquidCompositor.Compose(
                        legacyTerrain, pack, out _, MinimapLiquidPalette.ViewerFlatV1);
                    scorecard.Add(
                        tile.TileX,
                        tile.TileY,
                        BaselineVariantName,
                        MinimapComparisonMetrics.Compare(authoredReference, legacyLiquid));
                }

                if (authoredReference is not null && authoredTilePath is not null && comparisonTilePath is not null)
                {
                    stage = "writing authored client reference";
                    using Image<Rgba32> authoredImage = CreateRgbImage(authoredReference);
                    authoredImage.SaveAsPng(authoredTilePath);
                    stage = "writing authored-versus-synthetic comparison";
                    // Compare against the LIQUID-bearing synthesized tile: authored minimaps
                    // include water, so comparing the terrain-only (no-liquid) render against
                    // them is not a fair A/B.
                    using Image<Rgba32> comparisonImage = CreateAuthoredSyntheticComparison(authoredImage, liquidImage);
                    comparisonImage.SaveAsPng(comparisonTilePath);
                }
                emittedTiles[(tile.TileX, tile.TileY)] = tilePath;
                emittedLiquidTiles[(tile.TileX, tile.TileY)] = liquidTilePath;
                results.Add(new SyntheticMinimapTileResult(
                    tile.TileX,
                    tile.TileY,
                    emitPerTile ? "written" : "stitched-only",
                    emitPerTile ? tilePath : null,
                    textures.Count,
                    emitPerTile ? null : "Temporary tile used for the whole-map output only.",
                    pack.MinimapTextureFallbacks.Values
                        .OrderBy(static fallback => fallback.TextureId)
                        .ToArray(),
                    emitPerTile ? liquidTilePath : null,
                    liquidPixelCount,
                    tileLighting,
                    emitPerTile ? authoredTilePath : null,
                    emitPerTile ? comparisonTilePath : null));
                Console.WriteLine(
                    $"Synthetic minimap tile: {tile.TileX:D2},{tile.TileY:D2} -> {tilePath} + {liquidTilePath}" +
                    (comparisonTilePath is null ? string.Empty : $" + {authoredTilePath} + {comparisonTilePath}") +
                    $" ({liquidPixelCount} liquid pixels)");
            }
            catch (Exception ex)
            {
                string detail = $"{stage}: {DescribeSyntheticMinimapFailure(ex)}";
                results.Add(new SyntheticMinimapTileResult(tile.TileX, tile.TileY, "failed", null, 0, detail));
                Console.Error.WriteLine($"Synthetic minimap tile {tile.TileX:D2},{tile.TileY:D2} failed: {detail}");
            }
        });

        // ── WMO OVERLAY PASS (sequential, headless GL) ──────────────────────
        // After the parallel terrain pass, composite placed WMO geometry onto the
        // terrain minimap tiles using the same solar lighting direction.
        if (includeWmos && results.Any(static r => r.Status is "written" or "stitched-only"))
        {
            Console.WriteLine("WMO overlay pass: rendering placed WMO geometry onto minimap tiles...");
            Vector3 wmoLightDir = Vector3.Normalize(TerrainSolarDirection.Evaluate(gameTime));

            // 0.5.3 geometry uses DirectX winding (clockwise). OpenGL defaults to CCW, which
            // inverts effective Z normals. Negate lightDir.Z to compensate.
            // 0.6.0+ geometry uses OpenGL winding (CCW), so no negation needed.
            if (ClientBuildKey.TryParse(buildVersion, out ClientBuildKey buildKey)
                && buildKey.Major == 0 && buildKey.Minor == 5)
            {
                wmoLightDir = new Vector3(wmoLightDir.X, wmoLightDir.Y, -wmoLightDir.Z);
                Console.WriteLine($"  WMO light direction: 0.5.x build detected, negated Z for DirectX winding");
            }
            else
            {
                Console.WriteLine($"  WMO light direction: raw solar direction (0.6.0+ or unknown build)");
            }

            try
            {
                using var headless = new HeadlessContext(resolution, resolution);
                using var capture = new ObjectCaptureRenderer(headless.GL, catalog);
                capture.SetLightDirection(wmoLightDir);

                const float tileWorldSize = 533.33333f;
                const float mapOrigin = 32f * tileWorldSize;

                foreach (SyntheticMinimapTileResult tileResult in results)
                {
                    if (tileResult.Status is not "written" and not "stitched-only")
                        continue;

                    int tileX = tileResult.TileX;
                    int tileY = tileResult.TileY;

                    // Rebuild the pack to get placement data and terrain height.
                    TerrainTileTensorPack? pack = TryBuildSyntheticMinimapPack(
                        catalog, clientRoot, mapName, wdtBytes, isAlpha, tileX, tileY, buildVersion);
                    if (pack?.PlacementModfData is null || pack.PlacementModfCount == 0)
                        continue;

                    float terrainMaxZ = ComputeTerrainMaxHeight(pack);
                    float tileCenterX = mapOrigin - (tileY + 0.5f) * tileWorldSize;
                    float tileCenterY = mapOrigin - (tileX + 0.5f) * tileWorldSize;
                    float tileSpan = tileWorldSize * 0.5f;

                    capture.CaptureTileWmos(
                        pack.PlacementModfData,
                        pack.PlacementModfNames,
                        tileCenterX, tileCenterY, tileSpan,
                        resolution, terrainMaxZ,
                        out byte[] wmoImageRgba,
                        out byte[] wmoMaskRgba);

                    // Composite: terrain * (1 - mask) + wmo * mask
                    string tilePath = tileResult.OutputPath
                        ?? Path.Combine(tilesDirectory, $"{mapName}_{tileX:D2}_{tileY:D2}_synthesized.png");
                    CompositeWmoOverlay(tilePath, wmoImageRgba, wmoMaskRgba, resolution);
                    Console.WriteLine($"  WMO overlay applied to tile {tileX:D2},{tileY:D2}");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"WMO overlay pass failed: {ex.Message}");
                Console.Error.WriteLine("  Continuing with terrain-only output.");
            }
        }

        TerrainMinimapStitchResult? stitched = null;
        TerrainMinimapStitchResult? liquidStitched = null;
        TerrainMinimapStitchResult? residualStitched = null;
        Exception? stitchFailure = null;
        if (emitWholeMap && emittedTiles.Count > 0)
        {
            string stitchedPath = Path.Combine(outputDirectory, "stitched", $"{mapName}_synthesized_minimap.png");
            string liquidStitchedPath = Path.Combine(outputDirectory, "stitched", $"{mapName}_synthesized_minimap_liquid.png");
            string residualStitchedPath = Path.Combine(outputDirectory, "stitched", $"{mapName}_textureless_residual.png");
            try
            {
                stitched = TerrainMinimapStitcher.Stitch(emittedTiles, stitchedPath, resolution);
                liquidStitched = TerrainMinimapStitcher.Stitch(emittedLiquidTiles, liquidStitchedPath, resolution);
                Console.WriteLine($"Synthetic minimap map: {stitchedPath} ({stitched.Width}x{stitched.Height}, tiles {stitched.MinTileX:D2},{stitched.MinTileY:D2} -> {stitched.MaxTileX:D2},{stitched.MaxTileY:D2})");
                Console.WriteLine($"Synthetic minimap liquid map: {liquidStitchedPath} ({liquidStitched.Width}x{liquidStitched.Height})");
                if (texturelessResiduals && emittedResidualTiles.Count > 0)
                {
                    residualStitched = TerrainMinimapStitcher.Stitch(emittedResidualTiles, residualStitchedPath, resolution);
                    Console.WriteLine($"Textureless residual map: {residualStitchedPath} ({residualStitched.Width}x{residualStitched.Height})");
                }
            }
            catch (Exception ex)
            {
                stitchFailure = ex;
                Console.Error.WriteLine($"Synthetic minimap stitching failed: {ex.Message}");
            }
        }

        if (!emitPerTile && stitchFailure is null && stitched is not null && liquidStitched is not null)
            Directory.Delete(tilesDirectory, recursive: true);

        var manifest = new SyntheticMinimapManifest(
            "terrain-minimap-synthesis-v6",
            clientRoot,
            buildVersion,
            mapName,
            timeOfDay.ToString(),
            timeOfDayHours,
            gameTime,
            resolution,
            detailTexels ? "detail" : "material_average",
            TerrainMinimapCompositionOptions.TextureRepeatsPerChunk,
            emitPerTile,
            emitWholeMap,
            includeWmos,
            lighting,
            // Record the palette actually used, not the default -- a corpus rendered with a tuned
            // water colour must state which water it has.
            tuning.LiquidPalette.RenderProfile,
            stitched,
            liquidStitched,
            results.OrderBy(static r => r.TileY).ThenBy(static r => r.TileX).ToArray());

        string manifestPath = Path.Combine(outputDirectory, "synthesis-manifest.json");
        File.WriteAllText(
            manifestPath,
            System.Text.Json.JsonSerializer.Serialize(
                manifest,
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true, IncludeFields = true }));
        Console.WriteLine($"Synthetic minimap manifest: {manifestPath}");
        scorecard?.Report(outputDirectory, BaselineVariantName, CurrentVariantName);
        azimuthSweep?.Report(tuning.Era);
        if (encodingSurvey && surveyedBlpBytes.Count > 0)
        {
            EncodingSurveyResult survey = MinimapEncodingSurvey.Survey(
                buildVersion, mapName, surveyedBlpBytes, tuning.ExactEraMatch);
            Console.WriteLine();
            Console.WriteLine($"=== Encoding survey: {mapName} ({buildVersion}) ===");
            Console.WriteLine($"  Build recognised: {survey.BuildRecognised}");
            foreach ((string label, int count) in survey.EncodingDistribution.OrderByDescending(static kv => kv.Value))
                Console.WriteLine($"  {label,-48} {count}");
        }

        if (lightingBaseline && surveyedAuthoredTiles.Count > 0)
        {
            LightingBaselineResult baseline = MinimapLightingBaseline.Survey(
                mapName, buildVersion, surveyedAuthoredTiles, tuning.ExactEraMatch);
            Console.WriteLine();
            Console.WriteLine($"=== Lighting baseline: {mapName} ({buildVersion}) ===");
            Console.WriteLine($"  Build recognised: {baseline.BuildRecognised}");
            Console.WriteLine($"  Mean luma: {baseline.MeanLuma:0.000}  cross-tile std: {baseline.StdLuma:0.000}");
            Console.WriteLine($"  Shared lighting baseline: {(baseline.BaselinePresent ? "PRESENT" : "absent")}");
            if (baseline.BaselinePresent)
                Console.WriteLine("  Comparison will account for the shared baseline rather than treating per-tile differences as codec damage.");
        }
        Console.WriteLine($"Synthetic minimap summary: written={results.Count(result => result.Status is "written" or "stitched-only")}, skipped={results.Count(result => result.Status == "skipped")}, failed={results.Count(result => result.Status == "failed")}");

        bool authoredProofIncomplete = emitAuthoredReference
            && results.Any(static result => result.Status is not ("written" or "stitched-only"));
        if (emittedTiles.Count == 0 || (emitWholeMap && stitchFailure is not null) || authoredProofIncomplete)
            Environment.ExitCode = 1;
    }

    private static string DescribeSyntheticMinimapFailure(Exception exception)
    {
        var parts = new List<string>();
        for (Exception? current = exception; current is not null; current = current.InnerException)
        {
            string detail = $"{current.GetType().Name}: {current.Message}";
            string? frame = current.StackTrace?
                .Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .FirstOrDefault(static line => line.Contains("WowViewer", StringComparison.Ordinal));
            if (!string.IsNullOrWhiteSpace(frame))
                detail += $" [{frame}]";
            parts.Add(detail);
        }

        return string.Join(" <- ", parts);
    }

    private static bool TryParseSyntheticMinimapTileList(
        string? value,
        out HashSet<(int TileX, int TileY)> tiles,
        out string? error)
    {
        tiles = [];
        error = null;
        if (string.IsNullOrWhiteSpace(value))
            return true;

        foreach (string entry in value.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string[] parts = entry.Split(',', StringSplitOptions.TrimEntries);
            if (parts.Length != 2
                || !int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out int tileX)
                || !int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int tileY)
                || tileX is < 0 or > 63
                || tileY is < 0 or > 63)
            {
                error = $"--tile-list entry '{entry}' must be x,y with both coordinates in 0..63.";
                return false;
            }

            tiles.Add((tileX, tileY));
        }

        if (tiles.Count == 0)
        {
            error = "--tile-list did not contain any x,y coordinates.";
            return false;
        }

        return true;
    }

    private static Image<Rgba32> CreateRgbImage(byte[,,] rgb)
    {
        int height = rgb.GetLength(0);
        int width = rgb.GetLength(1);
        if (height <= 0 || width <= 0 || rgb.GetLength(2) < 3)
            throw new InvalidDataException("RGB reference must have shape [height,width,>=3].");

        var image = new Image<Rgba32>(width, height);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                image[x, y] = new Rgba32(rgb[y, x, 0], rgb[y, x, 1], rgb[y, x, 2], 255);
        return image;
    }

    private static bool HasVisibleRgb(byte[,,] rgb)
    {
        int height = rgb.GetLength(0);
        int width = rgb.GetLength(1);
        int channels = rgb.GetLength(2);
        if (height <= 0 || width <= 0 || channels < 3)
            return false;

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                if (rgb[y, x, 0] > 2 || rgb[y, x, 1] > 2 || rgb[y, x, 2] > 2)
                    return true;
        return false;
    }

    private static bool HasVisibleRgb(Image<Rgba32> image)
    {
        for (int y = 0; y < image.Height; y++)
            for (int x = 0; x < image.Width; x++)
            {
                Rgba32 pixel = image[x, y];
                if (pixel.R > 2 || pixel.G > 2 || pixel.B > 2)
                    return true;
            }
        return false;
    }

    private static Image<Rgba32> CreateAuthoredSyntheticComparison(
        Image<Rgba32> authored,
        Image<Rgba32> synthesized)
    {
        var comparison = new Image<Rgba32>(synthesized.Width * 2, synthesized.Height);
        for (int y = 0; y < synthesized.Height; y++)
        {
            int authoredY = Math.Min(authored.Height - 1, y * authored.Height / synthesized.Height);
            for (int x = 0; x < synthesized.Width; x++)
            {
                int authoredX = Math.Min(authored.Width - 1, x * authored.Width / synthesized.Width);
                comparison[x, y] = authored[authoredX, authoredY];
                comparison[x + synthesized.Width, y] = synthesized[x, y];
            }
        }

        return comparison;
    }

    private sealed class KnownTerrainTexturePaths
    {
        public KnownTerrainTexturePaths(NativeMpqService catalog)
        {
            Paths = catalog.GetAllKnownFiles()
                .Where(static path => path.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
                .ToArray();
        }

        public IReadOnlyList<string> Paths { get; }

        private readonly ConcurrentDictionary<string, IReadOnlyList<TerrainTextureFallbackCandidate>> _relatedCandidates = new(StringComparer.OrdinalIgnoreCase);
        private readonly ConcurrentDictionary<string, IReadOnlyList<TerrainTextureFallbackCandidate>> _catalogLastResortCandidates = new(StringComparer.OrdinalIgnoreCase);
        private readonly ConcurrentDictionary<string, byte[,,]> _decodedRgbTextures = new(StringComparer.OrdinalIgnoreCase);

        public IReadOnlyList<TerrainTextureFallbackCandidate> GetRelatedCandidates(string requestedPath)
        {
            return _relatedCandidates.GetOrAdd(
                requestedPath,
                requested => TerrainTextureFallbackPolicy.GetRelatedDiffuseRgbProxyCandidates(requested, Paths));
        }

        public IReadOnlyList<TerrainTextureFallbackCandidate> GetCatalogLastResortCandidates(string requestedPath)
        {
            return _catalogLastResortCandidates.GetOrAdd(
                requestedPath,
                requested => TerrainTextureFallbackPolicy.GetCatalogRgbLastResortCandidates(requested, Paths));
        }

        public void RememberDecodedTexture(string path, byte[,,] pixels)
        {
            _decodedRgbTextures.TryAdd(path, pixels);
        }

        public bool TryGetDecodedTexture(string path, out byte[,,]? pixels) =>
            _decodedRgbTextures.TryGetValue(path, out pixels);

        public bool TryGetAnyDecodedTexture(out string resolvedPath, out byte[,,]? pixels)
        {
            KeyValuePair<string, byte[,,]> candidate = _decodedRgbTextures
                .OrderBy(static pair => pair.Key, StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault();
            if (string.IsNullOrWhiteSpace(candidate.Key))
            {
                resolvedPath = string.Empty;
                pixels = null;
                return false;
            }

            resolvedPath = candidate.Key;
            pixels = candidate.Value;
            return true;
        }
    }

    static void RunHarvestMapMpq(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapName = GetOption(args, "--map", "-m");
        string? outputDir = GetOption(args, "--output-dir", "-o");
        int? limit = GetIntOption(args, "--limit", "-n");
        string? tileListRaw = GetOption(args, "--tile-list", "");
        bool force = HasFlag(args, "--force");
        int maxTiles = limit ?? int.MaxValue;

        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(mapName) || string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Error: --client-root, --map, and --output-dir are required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!TryParseSyntheticMinimapTileList(tileListRaw, out HashSet<(int TileX, int TileY)> requestedTiles, out string? tileListError))
        {
            Console.Error.WriteLine($"Error: {tileListError}");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);

        string? buildVersion = DetectBuildVersionFromClientRoot(clientRoot);
        Console.WriteLine($"Build version: {buildVersion ?? "(unknown)"}");

        Directory.CreateDirectory(outputDir);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
        byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
        if (wdtBytes is null)
        {
            Console.Error.WriteLine($"Error: Could not read WDT '{wdtVirtual}' from client.");
            Environment.ExitCode = 1;
            return;
        }

        int extracted = 0, skipped = 0, errors = 0;
        var sw = System.Diagnostics.Stopwatch.StartNew();

        IEnumerable<(int TileX, int TileY)> tilesToVisit = requestedTiles.Count > 0
            ? requestedTiles.OrderBy(static t => t.TileY).ThenBy(static t => t.TileX)
            : Enumerable.Range(0, 64).SelectMany(tx => Enumerable.Range(0, 64).Select(ty => (TileX: tx, TileY: ty)));

        foreach ((int tx, int ty) in tilesToVisit)
        {
            if (extracted >= maxTiles) break;

            string outputPath = Path.Combine(outputDir, $"{mapName}_{tx}_{ty}_harvest.npz");
            if (!force && File.Exists(outputPath)) { skipped++; continue; }

            try
            {
                var oldErr = Console.Error;
                Console.SetError(TextWriter.Null);
                try
                {
                    if (RunExtractTileFromMpq(catalog, clientRoot, mapName, wdtBytes, tx, ty, outputPath, exportPlacements: false, buildVersion: buildVersion))
                        extracted++;
                }
                finally { Console.SetError(oldErr); }
            }
            catch { }
        }

        sw.Stop();
        Console.WriteLine($"Done. Extracted={extracted} Skipped={skipped} Errors={errors} in {sw.Elapsed.TotalSeconds:F0}s ({sw.Elapsed.TotalSeconds / Math.Max(1, extracted):F1}s/tile)");
    }

    static void RunHarvestStream(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapName = GetOption(args, "--map", "-m");
        int? limit = GetIntOption(args, "--limit", "-n");
        int maxTiles = limit ?? int.MaxValue;
        int tileWorkers = Math.Max(1, GetIntOption(args, "--tile-workers", "--tile-workers") ?? DefaultHarvestTileWorkers);
        string streamProfileRaw = (GetOption(args, "--stream-profile", "") ?? "v16").Trim();
        StreamProfile streamProfile = ResolveStreamProfile(streamProfileRaw);

        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(mapName))
        {
            Console.Error.WriteLine("Error: --client-root and --map are required for harvest-stream.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);

        // Redirect Console.Out to stderr so stdout is pure binary
        var originalOut = Console.Out;
        Console.SetOut(Console.Error);

        string? buildVersion = DetectBuildVersionFromClientRoot(clientRoot);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
        byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
        if (wdtBytes is null)
        {
            Console.Error.WriteLine($"Error: Could not read WDT '{wdtVirtual}' from client.");
            Environment.ExitCode = 1;
            return;
        }

        int extracted = 0;
        int errors = 0;
        string? firstError = null;
        var stdout = Console.OpenStandardOutput();
        bool isAlpha = AlphaWdtReader.IsAlphaWdt(wdtBytes);

        if (tileWorkers <= 1)
        {
            for (int tx = 0; tx < 64; tx++)
            {
                for (int ty = 0; ty < 64; ty++)
                {
                    if (extracted >= maxTiles) break;

                    HarvestTileResult result = HarvestStreamTileWorker(
                        catalog,
                        clientRoot,
                        mapName,
                        wdtBytes,
                        isAlpha,
                        buildVersion,
                        streamProfile,
                        new HarvestTileJob(extracted + errors, tx, ty));
                    if (result.HadError)
                    {
                        errors++;
                        firstError ??= result.ErrorMessage;
                        continue;
                    }
                    if (result.Blob is null)
                        continue;

                    WriteHarvestStreamBlob(stdout, result.Blob);
                    extracted++;
                }
                if (extracted >= maxTiles) break;
            }
        }
        else
        {
            Console.Error.WriteLine($"  harvest-stream tile_workers={tileWorkers} profile={streamProfileRaw}");
            List<HarvestTileJob> jobs = new(64 * 64);
            int order = 0;
            for (int tx = 0; tx < 64; tx++)
                for (int ty = 0; ty < 64; ty++)
                    jobs.Add(new HarvestTileJob(order++, tx, ty));

            int prefetch = Math.Max(tileWorkers * 2, tileWorkers);
            Dictionary<int, Task<HarvestTileResult>> inflight = [];
            int nextLaunch = 0;

            void LaunchUntilWindowFull()
            {
                while (nextLaunch < jobs.Count && inflight.Count < prefetch)
                {
                    HarvestTileJob job = jobs[nextLaunch++];
                    inflight[job.Order] = Task.Run(() =>
                        HarvestStreamTileWorker(catalog, clientRoot, mapName, wdtBytes, isAlpha, buildVersion, streamProfile, job));
                }
            }

            LaunchUntilWindowFull();
            for (int nextOrder = 0; nextOrder < jobs.Count; nextOrder++)
            {
                HarvestTileResult result = inflight[nextOrder].GetAwaiter().GetResult();
                inflight.Remove(nextOrder);
                LaunchUntilWindowFull();

                if (result.HadError)
                {
                    errors++;
                    firstError ??= result.ErrorMessage;
                    continue;
                }
                if (result.Blob is null)
                    continue;

                WriteHarvestStreamBlob(stdout, result.Blob);
                extracted++;
                if (extracted >= maxTiles)
                    break;
            }

            if (inflight.Count > 0)
                Task.WaitAll(inflight.Values.ToArray());
        }

        // Write end marker: 4 bytes "ENDS" + 4 zero bytes
        byte[] endMarker = new byte[8];
        System.Text.Encoding.ASCII.GetBytes("ENDS").CopyTo(endMarker, 0);
        stdout.Write(endMarker, 0, 8);
        stdout.Flush();

        Console.Error.WriteLine($"Streamed {extracted} tiles, {errors} errors");
        if (firstError is not null)
            Console.Error.WriteLine($"First harvest-stream tile error: {firstError}");
        if (extracted == 0)
            Environment.ExitCode = 1;
    }

    private static HarvestTileResult HarvestStreamTileWorker(
        NativeMpqService catalog,
        string clientRoot,
        string mapName,
        byte[] wdtBytes,
        bool isAlpha,
        string? buildVersion,
        StreamProfile streamProfile,
        HarvestTileJob job)
    {
        try
        {
            return new HarvestTileResult(
                job.Order,
                TryBuildHarvestStreamBlob(catalog, clientRoot, mapName, wdtBytes, isAlpha, buildVersion, streamProfile, job.TileX, job.TileY),
                false,
                null);
        }
        catch (Exception ex)
        {
            return new HarvestTileResult(job.Order, null, true, $"tile ({job.TileX},{job.TileY}): {ex.Message}");
        }
    }

    private static byte[]? TryBuildHarvestStreamBlob(
        NativeMpqService catalog,
        string clientRoot,
        string mapName,
        byte[] wdtBytes,
        bool isAlpha,
        string? buildVersion,
        StreamProfile streamProfile,
        int tileX,
        int tileY)
    {
        bool needsTerrainTexturePayloads = streamProfile is StreamProfile.Full or StreamProfile.V22;
        TerrainTileTensorPack? pack = BuildEnrichedTensorPackForTile(
            catalog, clientRoot, mapName, wdtBytes, isAlpha, buildVersion, tileX, tileY,
            fullTextureDecode: needsTerrainTexturePayloads);
        if (pack is null)
            return null;

        using var ms = new MemoryStream();
        RawArraySerializer.Serialize(pack, ms, streamProfile);
        return ms.ToArray();
    }

    /// <summary>
    /// Builds a fully-enriched <see cref="TerrainTileTensorPack"/> for one tile: base ADT/alpha-WDT
    /// decode, WL loose-liquid enrichment, authored minimap load, and (when
    /// <paramref name="fullTextureDecode"/>) minimap lighting/shading-match analysis and
    /// name-aligned texture pixel attachment. Factored out of <see cref="TryBuildHarvestStreamBlob"/>
    /// so <c>curate</c> (which classifies an already-existing store's tiles by re-deriving the same
    /// pack the harvest-stream path already builds, rather than reading Zarr array data C# has no
    /// reader for) shares one definition of "how to build a tile's pack" instead of a second,
    /// divergent one.
    /// </summary>
    private static TerrainTileTensorPack? BuildEnrichedTensorPackForTile(
        NativeMpqService catalog,
        string clientRoot,
        string mapName,
        byte[] wdtBytes,
        bool isAlpha,
        string? buildVersion,
        int tileX,
        int tileY,
        bool fullTextureDecode)
    {
        TerrainTileTensorPack? pack = null;
        if (isAlpha)
        {
            string alphaWdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
            if (!AlphaWdtReader.TryReadTile(wdtBytes, tileX, tileY, alphaWdtVirtual, out AlphaTileData? tileData) || tileData is null)
                return null;
            pack = AlphaTensorPackBuilder.Build(tileData, tileX, tileY);
        }
        else
        {
            pack = BuildPackFromArchiveAdt(catalog, mapName, tileX, tileY, buildVersion);
        }

        if (pack is null)
            return null;

        if (pack.UnifiedLiquidMask is null)
            TryAddWlLiquidFromArchiveFiles(catalog, clientRoot, mapName, tileX, tileY, pack);

        if (pack.MinimapRgb256 is null)
        {
            byte[,,]? minimapRgb = TryLoadMinimapFromMpq(catalog, mapName, tileX, tileY);
            if (minimapRgb is not null)
            {
                pack.MinimapRgb256 = minimapRgb;
                pack.MinimapSourceTag = "mpq_blp";
                pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals) { "minimap_rgb" };
            }
        }

        if (fullTextureDecode)
            AnalyzeAuthoredMinimapLighting(catalog, mapName, pack, buildVersion);
        else if (pack.MinimapRgb256 is not null)
            SetMinimapLightingProvenance(pack, MinimapLightingProvenance.NotEvaluated("analysis_requires_full_texture_decode"));

        if (fullTextureDecode)
            AttachNameAlignedTexturePixels(catalog, pack);

        return pack;
    }

    /// <summary>
    /// Spec 122: canonical dataset curation. Classifies every tile already listed in an existing
    /// v50 store's <c>index.parquet</c> -- difficulty/coverage/lighting buckets and
    /// height-normal-mismatch/non-finite/has-flag findings (<see cref="WowViewer.Core.Curation"/>)
    /// -- by re-deriving each tile's <see cref="TerrainTileTensorPack"/> the same way
    /// <c>harvest-stream</c> does (this codebase has no C# Zarr reader, so the store's own already-
    /// decoded arrays are never opened; only its row identity is read). Dry-run by default (no
    /// <c>--write</c>): prints the plan and writes nothing. The store's own signal arrays and
    /// <c>index.parquet</c> are never modified (FR-014).
    /// </summary>
    static void RunCurate(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? storePath = GetOption(args, "--store", "-s");
        string? mapFilter = GetOption(args, "--map", "-m");
        bool write = HasFlag(args, "--write");

        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(storePath))
        {
            Console.Error.WriteLine("Error: --client-root and --store are required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(storePath))
        {
            Console.Error.WriteLine($"Error: store not found: {storePath}");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);
        string? buildVersion = DetectBuildVersionFromClientRoot(clientRoot);
        Console.WriteLine($"Build version: {buildVersion ?? "(unknown)"}");

        IReadOnlyList<StoreIndexRow> allRows;
        try
        {
            allRows = StoreIndexReader.Read(storePath);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: failed to read store index: {ex.Message}");
            Environment.ExitCode = 1;
            return;
        }

        List<StoreIndexRow> rows = string.IsNullOrWhiteSpace(mapFilter)
            ? allRows.ToList()
            : allRows.Where(r => r.Map.Equals(mapFilter, StringComparison.OrdinalIgnoreCase)).ToList();

        Console.WriteLine($"Store: {storePath}");
        Console.WriteLine($"Planned tile count: {rows.Count}" + (mapFilter is not null ? $" (map={mapFilter})" : $" (all {allRows.Select(r => r.Map).Distinct().Count()} map(s))"));
        Console.WriteLine($"Checks planned: {string.Join(", ", TileCurator.KnownChecks)}");
        Console.WriteLine("Per-tile evaluability (which checks actually ran vs. were not_evaluable for a given tile) is recorded in the written findings, not predicted here.");

        if (!write)
        {
            Console.WriteLine($"Output would be written under: {Path.Combine(storePath, "curation", "<curation_run_id>")}");
            Console.WriteLine("Dry run complete. Pass --write to actually classify and persist.");
            return;
        }

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        var records = new List<TileCurationRecord>();
        var findings = new List<MismatchFinding>();
        var failures = new List<string>();
        var sw = System.Diagnostics.Stopwatch.StartNew();

        string runId = CurationRunRecord.GenerateRunId(buildVersion ?? "unknown", storePath, DateTimeOffset.UtcNow);

        foreach (IGrouping<string, StoreIndexRow> mapGroup in rows.GroupBy(r => r.Map))
        {
            string mapName = mapGroup.Key;
            string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
            byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
            if (wdtBytes is null)
            {
                failures.Add($"map '{mapName}': could not read WDT '{wdtVirtual}' from client");
                continue;
            }
            bool isAlpha = AlphaWdtReader.IsAlphaWdt(wdtBytes);

            foreach (StoreIndexRow row in mapGroup)
            {
                TerrainTileTensorPack? pack;
                try
                {
                    pack = BuildEnrichedTensorPackForTile(
                        catalog, clientRoot, mapName, wdtBytes, isAlpha, buildVersion,
                        row.TileX, row.TileY, fullTextureDecode: true);
                }
                catch (Exception ex)
                {
                    failures.Add($"tile ({mapName},{row.TileX},{row.TileY}): {ex.Message}");
                    continue;
                }

                if (pack is null)
                {
                    failures.Add($"tile ({mapName},{row.TileX},{row.TileY}): could not be re-extracted from the client");
                    continue;
                }

                (TileCurationRecord record, IReadOnlyList<MismatchFinding> tileFindings) =
                    TileCurator.Classify(pack, row.Build, row.Map, row.TileX, row.TileY, row.TileId, runId);
                records.Add(record);
                findings.AddRange(tileFindings);
            }
        }

        sw.Stop();
        Console.WriteLine($"Classified {records.Count}/{rows.Count} tiles in {sw.Elapsed.TotalSeconds:F0}s.");

        if (failures.Count > 0)
        {
            Console.Error.WriteLine($"Error: {failures.Count} tile(s) could not be classified -- refusing to write a partial manifest (FR-008 full coverage is a hard gate):");
            foreach (string failure in failures.Take(20))
                Console.Error.WriteLine($"  - {failure}");
            if (failures.Count > 20)
                Console.Error.WriteLine($"  ... and {failures.Count - 20} more");
            Environment.ExitCode = 1;
            return;
        }

        var bucketCounts = new Dictionary<string, IReadOnlyDictionary<string, int>>
        {
            ["difficulty_bucket"] = TallyBucket(records.Select(r => r.DifficultyBucket)),
            ["coverage_bucket"] = TallyBucket(records.Select(r => r.CoverageBucket)),
            ["lighting_bucket"] = TallyBucket(records.Select(r => r.LightingBucket)),
        };
        var findingCounts = findings
            .GroupBy(f => f.Category)
            .ToDictionary(g => g.Key, g => g.Count());

        var runRecord = new CurationRunRecord
        {
            CurationRunId = runId,
            StorePath = storePath,
            BuildFingerprint = buildVersion ?? "unknown",
            ChecksRun = TileCurator.KnownChecks,
            TileCount = records.Count,
            BucketCounts = bucketCounts,
            FindingCounts = findingCounts,
            ToolVersion = typeof(Program).Assembly.GetName().Version?.ToString() ?? "0.0.0",
            CreatedAt = DateTimeOffset.UtcNow,
        };

        string runDir = CurationManifestWriter.Write(storePath, runRecord, records, findings);
        Console.WriteLine($"Wrote curation manifest for {records.Count} tiles to: {runDir}");
        Console.WriteLine($"Findings: {findings.Count} across {findingCounts.Count} categor{(findingCounts.Count == 1 ? "y" : "ies")}.");
    }

    private static Dictionary<string, int> TallyBucket(IEnumerable<string> values)
    {
        var counts = new Dictionary<string, int>();
        foreach (string v in values)
            counts[v] = counts.GetValueOrDefault(v) + 1;
        return counts;
    }

    private static StreamProfile ResolveStreamProfile(string value)
    {
        if (value.Equals("full", StringComparison.OrdinalIgnoreCase))
            return StreamProfile.Full;
        if (value.Equals("v22", StringComparison.OrdinalIgnoreCase))
            return StreamProfile.V22;
        return StreamProfile.V16;
    }

    private static void WriteHarvestStreamBlob(Stream stdout, byte[] blob)
    {
        byte[] header = new byte[8];
        System.Text.Encoding.ASCII.GetBytes("ARRY").CopyTo(header, 0);
        BitConverter.TryWriteBytes(header.AsSpan(4, 4), blob.Length);
        stdout.Write(header, 0, 8);
        stdout.Write(blob, 0, blob.Length);
    }

    static void RunExtractUnified(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapName = GetOption(args, "--map", "-m");
        string? output = GetOption(args, "--output", "-o");
        int? tileX = GetIntOption(args, "--tile-x", "-x");
        int? tileY = GetIntOption(args, "--tile-y", "-y");
        bool exportPlacements = HasFlag(args, "--export-placements");
        string? syntheticMinimap = GetOption(args, "--synthetic-minimap", "-s");
        bool dumpHex = HasFlag(args, "--dump-hex");

        if (string.IsNullOrWhiteSpace(clientRoot))
        {
            Console.Error.WriteLine("Error: --client-root <dir> is required.");
            Environment.ExitCode = 1;
            return;
        }

        if (!Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine($"Error: client root not found: {clientRoot}");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);

        string? buildVersion = DetectBuildVersionFromClientRoot(clientRoot);

        if (string.IsNullOrWhiteSpace(mapName))
        {
            Console.Error.WriteLine("Error: --map <name> is required.");
            Environment.ExitCode = 1;
            return;
        }

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
        byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
        if (wdtBytes is null)
        {
            Console.Error.WriteLine($"Error: Could not read WDT '{wdtVirtual}' from client.");
            Environment.ExitCode = 1;
            return;
        }

        if (tileX.HasValue && tileY.HasValue)
        {
            if (!RunExtractTileFromMpq(catalog, clientRoot, mapName, wdtBytes, tileX.Value, tileY.Value, output, exportPlacements, syntheticMinimap, buildVersion: buildVersion))
                Environment.ExitCode = 1;
        }
        else
        {
            using var ms = new MemoryStream(wdtBytes);
            MapFileSummary fileSummary = MapFileSummaryReader.Read(ms, wdtVirtual);
            var wdt = WdtSummaryReader.Read(ms, fileSummary);
            Console.WriteLine($"WDT: {mapName}");
            Console.WriteLine($"  IsWmoBased: {wdt.IsWmoBased}");
            Console.WriteLine($"  Tiles with data: {wdt.TilesWithData}/{wdt.TotalTiles}");
            Console.WriteLine();
            Console.WriteLine("Use --tile-x <0-63> --tile-y <0-63> to harvest a specific tile.");
        }
    }

    static void RunExtractHoles(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapsOption = GetOption(args, "--maps", "-m");
        string? output = GetOption(args, "--output", "-o");

        if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine("Error: --client-root <dir> is required and must exist.");
            Environment.ExitCode = 1;
            return;
        }
        if (string.IsNullOrWhiteSpace(mapsOption))
        {
            Console.Error.WriteLine("Error: --maps <comma-separated map names> is required.");
            Environment.ExitCode = 1;
            return;
        }
        if (string.IsNullOrWhiteSpace(output))
        {
            Console.Error.WriteLine("Error: --output <json path> is required.");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);
        string? buildVersion = DetectBuildVersionFromClientRoot(clientRoot);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        string[] maps = mapsOption.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var mapsOut = new Dictionary<string, List<Dictionary<string, object>>>();
        int totalTiles = 0;
        int totalHoled = 0;

        foreach (string mapName in maps)
        {
            string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
            byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
            if (wdtBytes is null)
            {
                Console.Error.WriteLine($"Warning: WDT not readable for map '{mapName}', skipping.");
                continue;
            }

            var tiles = new List<Dictionary<string, object>>();
            bool isAlpha = AlphaWdtReader.IsAlphaWdt(wdtBytes);

            for (int tileY = 0; tileY < 64; tileY++)
            {
                for (int tileX = 0; tileX < 64; tileX++)
                {
                    ushort[,]? holesYx = null;

                    if (isAlpha)
                    {
                        if (!AlphaWdtReader.TryReadTile(wdtBytes, tileX, tileY, wdtVirtual, out AlphaTileData? tileData) || tileData?.HoleFullMasks is null)
                            continue;
                        // Alpha reader stores [cx, cy]; normalize to [y, x].
                        holesYx = new ushort[16, 16];
                        for (int cy = 0; cy < 16; cy++)
                            for (int cx = 0; cx < 16; cx++)
                                holesYx[cy, cx] = tileData.HoleFullMasks[cx, cy];
                    }
                    else
                    {
                        string adtVirtual = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}.adt";
                        byte[]? adtBytes = catalog.ReadFile(adtVirtual);
                        if (adtBytes is null)
                            continue;
                        using var ms = new MemoryStream(adtBytes);
                        holesYx = AdtTensorPackBuilder.ReadHoleBitmasks(ms, adtVirtual);
                        if (holesYx is null)
                            continue;
                    }

                    int[] flat = new int[256];
                    bool anyHole = false;
                    for (int cy = 0; cy < 16; cy++)
                    {
                        for (int cx = 0; cx < 16; cx++)
                        {
                            flat[cy * 16 + cx] = holesYx[cy, cx];
                            anyHole |= holesYx[cy, cx] != 0;
                        }
                    }

                    tiles.Add(new Dictionary<string, object>
                    {
                        ["x"] = tileX,
                        ["y"] = tileY,
                        ["holes"] = flat,
                    });
                    totalTiles++;
                    if (anyHole)
                        totalHoled++;
                }
            }

            mapsOut[mapName] = tiles;
            Console.WriteLine($"extract-holes: {mapName}: {tiles.Count} tiles");
        }

        var payload = new Dictionary<string, object>
        {
            ["build_version"] = buildVersion ?? "",
            ["client_root"] = clientRoot,
            ["hole_field"] = "mcnk_holes_uint16_row_major_yx",
            ["maps"] = mapsOut,
        };

        string json = System.Text.Json.JsonSerializer.Serialize(payload);
        File.WriteAllText(output, json);
        Console.WriteLine($"extract-holes: wrote {totalTiles} tiles ({totalHoled} with holes) -> {output}");
    }

    static void RunExtractTilesets(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? pathsFile = GetOption(args, "--paths-file", "-p");
        string? outputDir = GetOption(args, "--output-dir", "-o");

        if (string.IsNullOrWhiteSpace(clientRoot) || !Directory.Exists(clientRoot))
        {
            Console.Error.WriteLine("Error: --client-root <dir> is required and must exist.");
            Environment.ExitCode = 1;
            return;
        }
        if (string.IsNullOrWhiteSpace(pathsFile) || !File.Exists(pathsFile))
        {
            Console.Error.WriteLine("Error: --paths-file <txt, one BLP virtual path per line> is required.");
            Environment.ExitCode = 1;
            return;
        }
        if (string.IsNullOrWhiteSpace(outputDir))
        {
            Console.Error.WriteLine("Error: --output-dir <dir> is required.");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);
        Directory.CreateDirectory(outputDir);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        string[] paths = File.ReadAllLines(pathsFile)
            .Select(l => l.Trim())
            .Where(l => l.Length > 0)
            .ToArray();

        var entries = new List<Dictionary<string, object>>();
        int failed = 0;
        int index = 0;
        foreach (string virtualPath in paths)
        {
            // Dataset stores carry normalized names (forward slashes, .png);
            // MPQ lookup wants the raw BLP virtual path. Try as-given first,
            // then the BLP-converted form. The manifest keeps the original
            // string so downstream joins stay exact.
            string converted = virtualPath.Replace('/', '\\');
            if (converted.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                converted = converted[..^4] + ".blp";

            byte[,,]? rgb = LoadTextureFromMpq(catalog, virtualPath)
                ?? (converted != virtualPath ? LoadTextureFromMpq(catalog, converted) : null);
            if (rgb is null)
            {
                failed++;
                continue;
            }

            int h = rgb.GetLength(0);
            int w = rgb.GetLength(1);
            string fileName = $"t{index:D4}.png";
            using (var image = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(w, h))
            {
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        image[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(rgb[y, x, 0], rgb[y, x, 1], rgb[y, x, 2], 255);
                using var fs = File.Create(Path.Combine(outputDir, fileName));
                image.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
            }

            entries.Add(new Dictionary<string, object>
            {
                ["path"] = virtualPath,
                ["file"] = fileName,
                ["width"] = w,
                ["height"] = h,
            });
            index++;
        }

        var payload = new Dictionary<string, object>
        {
            ["build_version"] = DetectBuildVersionFromClientRoot(clientRoot) ?? "",
            ["client_root"] = clientRoot,
            ["tilesets"] = entries,
        };
        string manifestPath = Path.Combine(outputDir, "manifest.json");
        File.WriteAllText(manifestPath, System.Text.Json.JsonSerializer.Serialize(payload));
        Console.WriteLine($"extract-tilesets: {entries.Count} decoded, {failed} failed -> {manifestPath}");
    }

    static bool RunExtractTileFromMpq(NativeMpqService catalog, string clientRoot, string mapName, byte[] wdtBytes, int tileX, int tileY, string? outputPath, bool exportPlacements, string? syntheticMinimapPath = null, string? buildVersion = null)
    {
        TerrainTileTensorPack pack;
        AdtPlacementCatalog? placementCatalog = null;

if (AlphaWdtReader.IsAlphaWdt(wdtBytes))
            {
                string alphaWdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
                if (!AlphaWdtReader.TryReadTile(wdtBytes, tileX, tileY, alphaWdtVirtual, out AlphaTileData? tileData) || tileData is null)
                {
                    Console.Error.WriteLine($"Error: Alpha tile ({tileX},{tileY}) not present in WDT.");
                    return false;
                }
                pack = AlphaTensorPackBuilder.Build(tileData, tileX, tileY);
                if (exportPlacements)
                    placementCatalog = tileData.ToPlacementCatalog();
            }
            else
            {
                pack = BuildPackFromArchiveAdt(catalog, mapName, tileX, tileY, buildVersion);
                if (pack is null)
                {
                    string adtVirtual = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}.adt";
                    Console.Error.WriteLine($"Error: Could not read ADT '{adtVirtual}' from client.");
                    return false;
                }
            }

            // Try loose WL* liquid fallback for tiles with no MH2O/MCLQ coverage.
            if (pack.UnifiedLiquidMask is null)
                TryAddWlLiquidFromArchiveFiles(catalog, clientRoot, mapName, tileX, tileY, pack);

        if (pack.MinimapRgb256 is null)
        {
            byte[,,]? minimapRgb = TryLoadMinimapFromMpq(catalog, mapName, tileX, tileY);
            if (minimapRgb is not null)
            {
                pack.MinimapRgb256 = minimapRgb;
                pack.MinimapSourceTag = "mpq_blp";
                pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals) { "minimap_rgb" };
            }
        }

        AnalyzeAuthoredMinimapLighting(catalog, mapName, pack, buildVersion);

        // MTEX table indices are semantic. Do not shift later texture payloads into an earlier
        // missing slot: either retain a fully name-aligned payload table or record incompleteness.
        AttachNameAlignedTexturePixels(catalog, pack);

        if (string.IsNullOrWhiteSpace(outputPath))
            outputPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), $"{mapName}_{tileX}_{tileY}_v14.npz");

        if (outputPath == "-")
        {
            // Write raw binary to stdout as length-prefixed blob
            using var ms = new MemoryStream();
            RawArraySerializer.Serialize(pack, ms);
            byte[] blob = ms.ToArray();
            var stdout = Console.OpenStandardOutput();
            // 8-byte header: magic "ARRY" + 4-byte little-endian length
            stdout.Write(System.Text.Encoding.ASCII.GetBytes("ARRY"), 0, 4);
            byte[] lenBytes = BitConverter.GetBytes(blob.Length);
            stdout.Write(lenBytes, 0, 4);
            stdout.Write(blob, 0, blob.Length);
            stdout.Flush();
        }
        else
        {
            NpzTileSerializer.Serialize(pack, outputPath);
            Console.WriteLine($"Harvested: {outputPath}");
            Console.WriteLine($"Signals: {string.Join(", ", pack.AvailableSignals)}");

            if (pack.MinimapRgb256 != null)
                Console.WriteLine($"  minimap: 256x256 RGB from MPQ");
        }

        if (exportPlacements && placementCatalog is not null)
        {
            string placementPath = Path.ChangeExtension(outputPath, ".placement.json");
            string json = System.Text.Json.JsonSerializer.Serialize(placementCatalog, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(placementPath, json);
            Console.WriteLine($"Placements: {placementPath}");
        }

        if (!string.IsNullOrWhiteSpace(syntheticMinimapPath))
            GenerateSyntheticMinimap(catalog, pack, tileX, tileY, syntheticMinimapPath);

        return true;
    }

    private static HarvestMapDiscoveryResult DiscoverMap(NativeMpqService catalog, MapDirectoryEntry entry)
    {
        string mapName = entry.Directory;
        string wdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
        byte[]? wdtBytes = catalog.ReadFile(wdtVirtual);
        if (wdtBytes is null || wdtBytes.Length == 0)
        {
            return new HarvestMapDiscoveryResult(
                Map: mapName,
                DisplayName: entry.Name,
                Include: false,
                Reason: "wdt_missing",
                IsAlpha: false,
                IsWmoBased: false,
                HasWorldModelAsset: false,
                WorldModelNameCount: 0,
                TilesWithData: 0,
                HasReadableTile: false,
                HasUsableTile: false,
                ProbeTileX: null,
                ProbeTileY: null);
        }

        using MemoryStream ms = new(wdtBytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(ms, wdtVirtual);
        WdtSummary summary = WdtSummaryReader.Read(ms, fileSummary);
        IReadOnlyList<WdtTileCoordinate> occupiedTiles = WdtTileIndexReader.ReadOccupiedTiles(ms, fileSummary);
        bool isAlpha = AlphaWdtReader.IsAlphaWdt(wdtBytes);
        bool hasWorldModelAsset = summary.WorldModelNameCount > 0;

        int? probeTileX = null;
        int? probeTileY = null;
        bool hasReadableTile = false;
        bool hasUsableTile = false;

        if (summary.TilesWithData > 0)
        {
            foreach (WdtTileCoordinate tile in occupiedTiles)
            {
                ProbeTileState state = ProbeMapTile(catalog, mapName, tile, wdtBytes, isAlpha);
                if (!state.HasReadableTile)
                    continue;

                hasReadableTile = true;
                if (probeTileX is null || probeTileY is null)
                {
                    probeTileX = tile.TileX;
                    probeTileY = tile.TileY;
                }

                if (!state.HasUsableTile)
                    continue;

                hasUsableTile = true;
                probeTileX = tile.TileX;
                probeTileY = tile.TileY;
                break;
            }
        }

        string reason;
        bool include;
        if (summary.TilesWithData <= 0 && hasWorldModelAsset)
        {
            include = false;
            reason = "wmo_only";
        }
        else if (summary.TilesWithData <= 0)
        {
            include = false;
            reason = "no_tiles";
        }
        else if (!hasReadableTile)
        {
            include = false;
            reason = "no_readable_tile";
        }
        else if (!hasUsableTile)
        {
            include = false;
            reason = "no_v16_usable_tile";
        }
        else if (hasWorldModelAsset)
        {
            include = true;
            reason = "terrain_plus_wmo";
        }
        else
        {
            include = true;
            reason = "terrain";
        }

        return new HarvestMapDiscoveryResult(
            Map: mapName,
            DisplayName: entry.Name,
            Include: include,
            Reason: reason,
            IsAlpha: isAlpha,
            IsWmoBased: summary.IsWmoBased,
            HasWorldModelAsset: hasWorldModelAsset,
            WorldModelNameCount: summary.WorldModelNameCount,
            TilesWithData: summary.TilesWithData,
            HasReadableTile: hasReadableTile,
            HasUsableTile: hasUsableTile,
            ProbeTileX: probeTileX,
            ProbeTileY: probeTileY);
    }

    private readonly record struct ProbeTileState(bool HasReadableTile, bool HasUsableTile);

    private static ProbeTileState ProbeMapTile(
        NativeMpqService catalog,
        string mapName,
        WdtTileCoordinate tile,
        byte[] wdtBytes,
        bool isAlpha)
    {
        if (isAlpha)
        {
            string alphaWdtVirtual = $"World\\Maps\\{mapName}\\{mapName}.wdt";
            if (!AlphaWdtReader.TryReadTile(wdtBytes, tile.TileX, tile.TileY, alphaWdtVirtual, out AlphaTileData? tileData) || tileData is null)
                return new ProbeTileState(false, false);

            TerrainTileTensorPack pack = AlphaTensorPackBuilder.Build(tileData, tile.TileX, tile.TileY);
            if (pack.MinimapRgb256 is null)
                pack.MinimapRgb256 = TryLoadMinimapFromMpq(catalog, mapName, tile.TileX, tile.TileY);

            bool hasUsableTile = pack.Height257 is not null && pack.MinimapRgb256 is not null;
            return new ProbeTileState(true, hasUsableTile);
        }

        TerrainTileTensorPack? archivePack = BuildPackFromArchiveAdt(catalog, mapName, tile.TileX, tile.TileY, buildVersion: null);
        if (archivePack is null)
            return new ProbeTileState(false, false);

        if (archivePack.MinimapRgb256 is null)
            archivePack.MinimapRgb256 = TryLoadMinimapFromMpq(catalog, mapName, tile.TileX, tile.TileY);

        bool usable = archivePack.Height257 is not null && archivePack.MinimapRgb256 is not null;
        return new ProbeTileState(true, usable);
    }

    private static IReadOnlyList<string> BuildClientSearchRoots(string clientRoot)
    {
        List<string> roots = [];
        string dataRoot = Path.Combine(clientRoot, "Data");
        if (Directory.Exists(dataRoot))
            roots.Add(dataRoot);

        if (!string.Equals(clientRoot, dataRoot, StringComparison.OrdinalIgnoreCase))
            roots.Add(clientRoot);

        return roots.Count > 0 ? roots : [clientRoot];
    }

    private static TerrainTileTensorPack? BuildPackFromArchiveAdt(
        NativeMpqService catalog,
        string mapName,
        int tileX,
        int tileY,
        string? buildVersion)
    {
        string adtVirtual = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}.adt";
        byte[]? adtBytes = catalog.ReadFile(adtVirtual);
        if (adtBytes is null)
            return null;

        string tex0Virtual = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}_tex0.adt";
        string obj0Virtual = $"World\\Maps\\{mapName}\\{mapName}_{tileX}_{tileY}_obj0.adt";
        byte[]? tex0Bytes = catalog.ReadFile(tex0Virtual);
        byte[]? obj0Bytes = catalog.ReadFile(obj0Virtual);

        return AdtTensorPackBuilder.BuildFromBytes(
            adtVirtual,
            adtBytes,
            tex0Bytes,
            obj0Bytes,
            buildVersion,
            tex0Bytes is not null ? tex0Virtual : null,
            obj0Bytes is not null ? obj0Virtual : null,
            catalog.ReadFile);
    }

    private static TerrainTileTensorPack? TryBuildSyntheticMinimapPack(
        NativeMpqService catalog,
        string clientRoot,
        string mapName,
        byte[] wdtBytes,
        bool isAlpha,
        int tileX,
        int tileY,
        string? buildVersion)
    {
        TerrainTileTensorPack? pack;
        if (!isAlpha)
        {
            pack = BuildPackFromArchiveAdt(catalog, mapName, tileX, tileY, buildVersion);
        }
        else
        {
            string wdtPath = $"World\\Maps\\{mapName}\\{mapName}.wdt";
            pack = AlphaWdtReader.TryReadTile(wdtBytes, tileX, tileY, wdtPath, out AlphaTileData? tile)
            && tile is not null
                ? AlphaTensorPackBuilder.Build(tile, tileX, tileY)
                : null;
        }

        if (pack is not null && pack.UnifiedLiquidMask is null)
            TryAddWlLiquidFromArchiveFiles(catalog, clientRoot, mapName, tileX, tileY, pack);

        return pack;
    }

    private static Dictionary<int, byte[,,]> LoadSyntheticMinimapTextures(NativeMpqService catalog, TerrainTileTensorPack pack)
    {
        if (!pack.MclyTextureNames.Any(static textureName => !string.IsNullOrWhiteSpace(textureName)))
            return [];

        int[,,]? textureIds = pack.MclyTextureIds;
        var usedIds = new HashSet<int>();
        if (textureIds is not null)
        {
            for (int y = 0; y < textureIds.GetLength(0); y++)
                for (int x = 0; x < textureIds.GetLength(1); x++)
                    for (int layer = 0; layer < Math.Min(4, textureIds.GetLength(2)); layer++)
                        if (textureIds[y, x, layer] >= 0)
                            usedIds.Add(textureIds[y, x, layer]);
        }

        // A named material grid can still contain stale IDs. Keep ID zero in the recovery path so
        // those references receive a deterministic provenance-labeled RGB proxy. Truly empty
        // MTEX tables returned above are white empty terrain, never catalog-coloured terrain.
        if (usedIds.Count == 0)
            usedIds.Add(0);

        var textures = new Dictionary<int, byte[,,]>();
        foreach (int textureId in usedIds.OrderBy(value => value))
        {
            string texturePath = textureId >= 0 && textureId < pack.MclyTextureNames.Count
                && !string.IsNullOrWhiteSpace(pack.MclyTextureNames[textureId])
                ? pack.MclyTextureNames[textureId]
                : $"<missing MTEX entry {textureId} in {pack.TileName}>";
            byte[,,]? pixels = LoadTerrainTextureRgbProxy(
                catalog,
                texturePath,
                out string resolvedPath,
                out string? resolutionKind);
            if (pixels is not null)
            {
                textures[textureId] = pixels;
                if (resolutionKind is not null)
                {
                    RecordTerrainTextureFallback(pack, textureId, texturePath, resolvedPath, resolutionKind);
                    Console.Error.WriteLine(
                        $"Warning: terrain texture [{textureId}] {texturePath} could not decode; " +
                        $"using RGB proxy {resolvedPath} ({resolutionKind}).");
                }
            }
            else
                Console.Error.WriteLine($"Warning: could not decode terrain texture [{textureId}] {texturePath}.");
        }

        return textures;
    }

    /// <summary>
    /// Build a neutral-white albedo texture set matching the keys of the real texture set, so the
    /// compositor renders only the terrain shadow (Lambert hillshade + cast shadows) with no albedo
    /// texture and no objects — the textureless residual signal (FR-022).
    /// </summary>
    private static Dictionary<int, byte[,,]> CreateNeutralWhiteTextures(IReadOnlyDictionary<int, byte[,,]> realTextures)
    {
        var neutral = new Dictionary<int, byte[,,]>();
        foreach ((int textureId, byte[,,] real) in realTextures)
        {
            int height = real.GetLength(0);
            int width = real.GetLength(1);
            var white = new byte[height, width, 3];
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    white[y, x, 0] = 255;
                    white[y, x, 1] = 255;
                    white[y, x, 2] = 255;
                }

            neutral[textureId] = white;
        }

        return neutral;
    }

    private static byte[,,]? LoadTerrainTextureRgbProxy(
        NativeMpqService catalog,
        string requestedPath,
        out string resolvedPath,
        out string? resolutionKind)
    {
        resolvedPath = requestedPath;
        resolutionKind = null;

        KnownTerrainTexturePaths knownPaths = _knownTerrainTexturePaths
            .GetValue(catalog, static source => new KnownTerrainTexturePaths(source));

        byte[,,]? pixels = LoadTextureFromMpq(catalog, requestedPath);
        if (pixels is not null)
        {
            knownPaths.RememberDecodedTexture(requestedPath, pixels);
            return pixels;
        }

        string? companionPath = TerrainTextureFallbackPolicy.GetSpecularCompanionPath(requestedPath);
        if (companionPath is not null)
        {
            pixels = LoadTextureFromMpq(catalog, companionPath);
            if (pixels is not null)
            {
                knownPaths.RememberDecodedTexture(companionPath, pixels);
                resolvedPath = companionPath;
                resolutionKind = TerrainTextureFallbackPolicy.SpecularCompanionRgbProxy;
                return pixels;
            }
        }

        foreach (TerrainTextureFallbackCandidate candidate in knownPaths.GetRelatedCandidates(requestedPath))
        {
            pixels = LoadTextureFromMpq(catalog, candidate.ResolvedPath);
            if (pixels is null)
                continue;

            knownPaths.RememberDecodedTexture(candidate.ResolvedPath, pixels);
            resolvedPath = candidate.ResolvedPath;
            resolutionKind = candidate.ResolutionKind;
            return pixels;
        }

        foreach (TerrainTextureFallbackCandidate candidate in knownPaths.GetCatalogLastResortCandidates(requestedPath))
        {
            if (knownPaths.TryGetDecodedTexture(candidate.ResolvedPath, out pixels) && pixels is not null)
            {
                resolvedPath = candidate.ResolvedPath;
                resolutionKind = candidate.ResolutionKind;
                return pixels;
            }

            pixels = LoadTextureFromMpq(catalog, candidate.ResolvedPath);
            if (pixels is null)
                continue;

            knownPaths.RememberDecodedTexture(candidate.ResolvedPath, pixels);
            resolvedPath = candidate.ResolvedPath;
            resolutionKind = candidate.ResolutionKind;
            return pixels;
        }

        // Catalog path discovery is incomplete on some early clients. Once this export has decoded
        // any terrain BLP, retain that verified RGB material as the final recorded fallback rather
        // than discarding a geometrically readable tile because its stale MTEX name was absent from
        // the listfile.
        if (knownPaths.TryGetAnyDecodedTexture(out string cachedPath, out pixels) && pixels is not null)
        {
            resolvedPath = cachedPath;
            resolutionKind = TerrainTextureFallbackPolicy.CatalogRgbLastResortProxy;
            return pixels;
        }

        return null;
    }

    private static void RecordTerrainTextureFallback(
        TerrainTileTensorPack pack,
        int textureId,
        string requestedPath,
        string resolvedPath,
        string resolutionKind)
    {
        var fallbacks = new Dictionary<int, TerrainTextureFallbackResolution>(pack.MinimapTextureFallbacks)
        {
            [textureId] = new TerrainTextureFallbackResolution(
                textureId,
                requestedPath,
                resolvedPath,
                resolutionKind),
        };
        pack.MinimapTextureFallbacks = fallbacks;
        string signal = resolutionKind switch
        {
            TerrainTextureFallbackPolicy.SpecularCompanionRgbProxy => "mtex_specular_companion_rgb_proxy",
            TerrainTextureFallbackPolicy.RelatedDiffuseRgbProxy => "mtex_related_diffuse_rgb_proxy",
            TerrainTextureFallbackPolicy.CatalogRgbLastResortProxy => "mtex_catalog_rgb_last_resort_proxy",
            _ => "mtex_rgb_proxy",
        };
        pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals, StringComparer.OrdinalIgnoreCase)
        {
            signal,
        };
    }

    private static void AnalyzeAuthoredMinimapLighting(
        NativeMpqService catalog,
        string mapName,
        TerrainTileTensorPack pack,
        string? buildVersion = null)
    {
        if (pack.MinimapRgb256 is null)
        {
            SetMinimapLightingProvenance(pack, MinimapLightingProvenance.NotEvaluated("no_authored_minimap_rgb"));
            return;
        }

        Dictionary<int, byte[,,]> textures = LoadSyntheticMinimapTextures(catalog, pack);
        if (textures.Count == 0)
        {
            SetMinimapLightingProvenance(pack, MinimapLightingProvenance.NotEvaluated("no_decoded_terrain_texture_baseline"));
            return;
        }

        foreach (int textureId in EnumerateReferencedTextureIds(pack))
        {
            if (!textures.ContainsKey(textureId))
            {
                SetMinimapLightingProvenance(pack, MinimapLightingProvenance.NotEvaluated("incomplete_terrain_texture_baseline"));
                return;
            }
        }

        try
        {
            using Image<Rgba32> baseline = TerrainMinimapCompositor.Compose(
                pack,
                textures,
                new TerrainMinimapCompositionOptions(
                    TerrainMinimapCompositor.DefaultResolution,
                    TerrainMinimapLighting.Neutral));
            byte[,,] baselineRgb = CopyRgb(baseline);
            IReadOnlyList<MinimapLightingTimeCandidate> candidates = LoadMinimapLightingTimeCandidates(catalog, mapName);
            MinimapLightingProvenance provenance = MinimapLightingProvenance.Infer(
                pack.MinimapRgb256,
                baselineRgb,
                pack.McshShadowMask256,
                candidates);

            // Spec 111: geometric shading-direction inference, additive to the tint-ratio result
            // above. MinimapShadingMatch.Evaluate gates on the exact 0.5.3.3368 build fingerprint
            // internally and renders zero candidates for any other build, so chaining it here adds
            // no measurable cost to Full/V22 exports of other client builds.
            provenance = MinimapShadingMatch.Evaluate(
                provenance,
                pack,
                textures,
                pack.MinimapRgb256,
                buildVersion ?? string.Empty);

            SetMinimapLightingProvenance(pack, provenance);
        }
        catch (Exception ex)
        {
            SetMinimapLightingProvenance(
                pack,
                MinimapLightingProvenance.NotEvaluated($"analysis_failed:{ex.GetType().Name}"));
        }
    }

    /// <summary>
    /// Reports how a build's authored minimap BLPs are actually encoded, and whether their pixels
    /// show the fingerprints of palette quantisation and rescaling.
    ///
    /// WHY: if the authored minimaps are palettized (256 colours) and dithered, then their colours
    /// are quantisation artefacts, not an art direction to match -- and any attempt to colour-match a
    /// full-colour render against them is chasing noise. The "spray-can" texture is the visible
    /// signature of error-diffusion dithering, and it would also explain fine speckle that a smooth
    /// synthesized render can never reproduce. All of this is decidable from the files themselves.
    /// </summary>
    static void RunInspectMinimapBlp(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapName = GetOption(args, "--map", "-m");
        int limit = GetIntOption(args, "--limit", "-n") ?? 12;
        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(mapName))
        {
            Console.Error.WriteLine("Error: --client-root and --map are required.");
            Environment.ExitCode = 1;
            return;
        }

        clientRoot = ResolveGameClientRoot(clientRoot);
        string? buildVersion = GetOption(args, "--build", "-b") ?? DetectBuildVersionFromClientRoot(clientRoot);

        using var catalog = new NativeMpqService();
        catalog.LoadArchives([clientRoot]);
        TryLoadSupplementalListfile(catalog);
        LoadMd5Translate(clientRoot, catalog);

        Console.WriteLine($"Authored minimap BLP encoding report — build {buildVersion ?? "unknown"}, map {mapName}");
        Console.WriteLine();

        int inspected = 0;
        var uniqueColorCounts = new List<int>();
        var ditherScores = new List<double>();
        var blockScores = new List<double>();
        var formats = new Dictionary<string, int>(StringComparer.Ordinal);
        var topMipSizes = new Dictionary<string, int>(StringComparer.Ordinal);

        for (int tileY = 0; tileY < 64 && inspected < limit; tileY++)
        {
            for (int tileX = 0; tileX < 64 && inspected < limit; tileX++)
            {
                byte[]? bytes = TryReadMinimapBlpBytes(catalog, mapName, tileX, tileY);
                if (bytes is null || bytes.Length == 0)
                    continue;

                BlpSummary summary;
                try
                {
                    using var stream = new MemoryStream(bytes, writable: false);
                    summary = BlpSummaryReader.Read(stream, $"{mapName}_{tileX}_{tileY}.blp");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  {tileX:D2},{tileY:D2}: header unreadable ({ex.Message})");
                    continue;
                }

                string formatKey = $"{summary.Format}/{summary.Compression}/alpha{summary.AlphaDepthBits}/{summary.PixelFormat}";
                formats[formatKey] = formats.GetValueOrDefault(formatKey) + 1;
                string sizeKey = $"{summary.Width}x{summary.Height}";
                topMipSizes[sizeKey] = topMipSizes.GetValueOrDefault(sizeKey) + 1;

                byte[,,]? rgb = TryLoadMinimapFromMpq(catalog, mapName, tileX, tileY);
                int uniqueColors = -1;
                double dither = double.NaN;
                double blockiness = double.NaN;
                bool degenerate = false;
                if (rgb is not null)
                {
                    uniqueColors = CountUniqueColors(rgb);
                    dither = MeasureHighFrequencyNoise(rgb);
                    blockiness = MeasureBlockEdgeRatio(rgb);
                    // A single-colour tile is an unrendered/empty tile, not evidence about encoding.
                    // Averaging it in drags every aggregate toward zero and can flip the verdict.
                    degenerate = uniqueColors <= 2;
                    if (!degenerate)
                    {
                        uniqueColorCounts.Add(uniqueColors);
                        ditherScores.Add(dither);
                        blockScores.Add(blockiness);
                    }
                }

                Console.WriteLine(
                    $"  {tileX:D2},{tileY:D2}  {summary.Format,-5} {summary.Compression,-12} " +
                    $"alpha{summary.AlphaDepthBits,-2} {summary.Width}x{summary.Height} " +
                    $"mips={summary.InBoundsMipLevelCount,-2} palette={(summary.PaletteSizeBytes > 0 ? "YES" : "no"),-3} " +
                    $"colors={(uniqueColors < 0 ? "?" : uniqueColors.ToString()),-6} hf-noise={dither:0.000} " +
                    $"block4x4={blockiness:0.00}{(degenerate ? "  [empty tile, excluded]" : string.Empty)}");
                inspected++;
            }
        }

        if (inspected == 0)
        {
            Console.Error.WriteLine("No authored minimap BLP could be read for this map.");
            Environment.ExitCode = 1;
            return;
        }

        Console.WriteLine();
        Console.WriteLine("Encodings seen:");
        foreach ((string key, int count) in formats.OrderByDescending(static kv => kv.Value))
            Console.WriteLine($"  {count,4} x  {key}");
        Console.WriteLine("Top-mip dimensions:");
        foreach ((string key, int count) in topMipSizes.OrderByDescending(static kv => kv.Value))
            Console.WriteLine($"  {count,4} x  {key}");

        if (uniqueColorCounts.Count > 0)
        {
            int median = uniqueColorCounts.OrderBy(static c => c).ElementAt(uniqueColorCounts.Count / 2);
            Console.WriteLine();
            Console.WriteLine($"Unique colours per tile: min {uniqueColorCounts.Min()}, median {median}, max {uniqueColorCounts.Max()}");
            Console.WriteLine(
                uniqueColorCounts.Max() <= 256
                    ? "  <= 256 everywhere: the decoded image IS palette-quantised. Its colours are\n" +
                      "  quantisation output, so exact colour-matching a full-colour render is chasing artefacts."
                    : "  > 256 colours present: the decoded image is NOT limited to a 256-entry palette.");
            Console.WriteLine();
            Console.WriteLine($"High-frequency noise (deviation from 3x3 median): {ditherScores.Average():0.000}");
            Console.WriteLine(
                ditherScores.Average() > 0.35
                    ? "  High: consistent with per-pixel dithering."
                    : "  Moderate/low: not per-pixel dithering. Note this detector is the wrong instrument\n" +
                      "  for BLOCK codecs -- DXT artefacts sit on 4-pixel boundaries, not on single pixels.");

            Console.WriteLine();
            Console.WriteLine($"4x4 block-edge ratio (discontinuity on block boundaries vs inside blocks): {blockScores.Average():0.00}");
            Console.WriteLine(
                blockScores.Average() > 1.15
                    ? "  >1: pixel differences concentrate on 4-pixel boundaries -- the signature of DXT block\n" +
                      "  compression. THIS is the 'spray-can' texture, and no smooth render reproduces it."
                    : "  ~1: no block-aligned structure detected in this sample.");
        }

        Console.WriteLine();
        Console.WriteLine("Reminder: a top mip smaller than 256x256 means the client's own minimaps were");
        Console.WriteLine("upscaled for display, and our 256+ renders are strictly higher fidelity than the target.");
    }

    private static byte[]? TryReadMinimapBlpBytes(NativeMpqService catalog, string mapName, int tileX, int tileY)
    {
        foreach (string candidate in new[]
        {
            $"World\\Minimaps\\{mapName}\\map{tileX:D2}_{tileY:D2}.blp",
            $"textures\\Minimap\\{mapName}\\map{tileX:D2}_{tileY:D2}.blp",
            $"World\\Minimaps\\{mapName}\\{mapName}_{tileX:D2}_{tileY:D2}.blp",
        })
        {
            byte[]? bytes = catalog.ReadFile(candidate);
            if (bytes is { Length: > 0 })
                return bytes;
        }

        return null;
    }

    private static int CountUniqueColors(byte[,,] rgb)
    {
        var seen = new HashSet<int>();
        for (int y = 0; y < rgb.GetLength(0); y++)
        {
            for (int x = 0; x < rgb.GetLength(1); x++)
                seen.Add((rgb[y, x, 0] << 16) | (rgb[y, x, 1] << 8) | rgb[y, x, 2]);
        }

        return seen.Count;
    }

    /// <summary>
    /// Ratio of mean horizontal pixel difference ACROSS 4-pixel-aligned column boundaries to the mean
    /// difference at non-aligned columns. DXT compresses in 4x4 blocks with independent endpoints, so
    /// its error concentrates exactly on those boundaries: a ratio above ~1.15 is the fingerprint.
    ///
    /// This is the right detector for a block codec. A 3x3-median test measures per-PIXEL noise and
    /// largely misses block structure, which is why it can report "low noise" on an image that is
    /// visibly blocky.
    /// </summary>
    private static double MeasureBlockEdgeRatio(byte[,,] rgb)
    {
        int height = rgb.GetLength(0);
        int width = rgb.GetLength(1);
        if (height < 8 || width < 8)
            return double.NaN;

        double alignedSum = 0.0, interiorSum = 0.0;
        int alignedCount = 0, interiorCount = 0;
        for (int y = 0; y < height; y++)
        {
            for (int x = 1; x < width; x++)
            {
                int delta = Math.Abs(rgb[y, x, 0] - rgb[y, x - 1, 0])
                    + Math.Abs(rgb[y, x, 1] - rgb[y, x - 1, 1])
                    + Math.Abs(rgb[y, x, 2] - rgb[y, x - 1, 2]);
                if (x % 4 == 0)
                {
                    alignedSum += delta;
                    alignedCount++;
                }
                else
                {
                    interiorSum += delta;
                    interiorCount++;
                }
            }
        }

        if (alignedCount == 0 || interiorCount == 0)
            return double.NaN;

        double interiorMean = interiorSum / interiorCount;
        return interiorMean <= 1e-9 ? double.NaN : (alignedSum / alignedCount) / interiorMean;
    }

    /// <summary>
    /// Fraction of pixels whose luma differs from their 3x3 neighbourhood median by more than a
    /// small threshold. Dithering scatters isolated off-median pixels everywhere, so it scores high;
    /// a smoothly shaded render scores low even when it has plenty of large-scale detail.
    /// </summary>
    private static double MeasureHighFrequencyNoise(byte[,,] rgb)
    {
        int height = rgb.GetLength(0);
        int width = rgb.GetLength(1);
        if (height < 3 || width < 3)
            return double.NaN;

        int deviating = 0;
        int total = 0;
        var window = new int[9];
        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                int index = 0;
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int ny = y + dy;
                        int nx = x + dx;
                        window[index++] = (rgb[ny, nx, 0] * 2126 + rgb[ny, nx, 1] * 7152 + rgb[ny, nx, 2] * 722) / 10000;
                    }
                }

                Array.Sort(window);
                int median = window[4];
                int centre = (rgb[y, x, 0] * 2126 + rgb[y, x, 1] * 7152 + rgb[y, x, 2] * 722) / 10000;
                if (Math.Abs(centre - median) > 3)
                    deviating++;
                total++;
            }
        }

        return total == 0 ? double.NaN : deviating / (double)total;
    }

    /// <summary>
    /// Renders a synthetic dome-shaped hill under the current sun model at a series of hours, with a
    /// compass burned into each frame.
    ///
    /// WHY: when a real tile "looks wrong", two very different causes produce the same complaint --
    /// the lighting model pointing the wrong way, or real terrain data being oriented differently
    /// than assumed. A known hill separates them. There is exactly one correct answer here, it needs
    /// no client, and it takes seconds: if these frames light the correct flank, the sun model is
    /// right and any remaining problem is in how real tiles are oriented or shaded.
    /// </summary>
    static void RunSunDiagnostic(string[] args)
    {
        string outputDirectory = GetOption(args, "--output-dir", "-o") ?? "sun-diagnostic";
        int resolution = GetIntOption(args, "--resolution", "-r") ?? 256;
        resolution = Math.Clamp(resolution, 64, 1024);

        SyntheticMinimapTuning? tuning = TryParseSyntheticMinimapTuning(args, buildVersion: null, out string? tuningError);
        if (tuningError is not null)
        {
            Console.Error.WriteLine($"Error: {tuningError}");
            Environment.ExitCode = 1;
            return;
        }

        Directory.CreateDirectory(outputDirectory);
        MinimapEraProfile era = tuning!.Era;

        Console.WriteLine("Sun diagnostic: a synthetic hill lit by the current sun model.");
        Console.WriteLine($"  Era:   {era.EraLabel} ({era.Name})");
        Console.WriteLine($"  Model: {era.AzimuthModel} at {era.NoonAzimuthDegrees:0} deg ({era.AzimuthProvenance})");
        Console.WriteLine("  Raster orientation: NORTH is up, EAST is right (row increases south, column increases east).");
        Console.WriteLine("  Expected under a north-west sun: the hill's UPPER-LEFT flank is lit,");
        Console.WriteLine("  the LOWER-RIGHT flank is shaded, and shadows are thrown DOWN-RIGHT.");
        Console.WriteLine();

        TerrainTileTensorPack pack = BuildDiagnosticHillPack();
        var textures = new Dictionary<int, byte[,,]> { [0] = new byte[1, 1, 3] { { { 190, 190, 190 } } } };

        foreach (int hour in new[] { 6, 8, 10, 12, 14, 16, 18 })
        {
            float gameTime = hour / 24f;
            TerrainMinimapLighting lighting = TerrainMinimapLighting.CreateShadedTerrain(
                gameTime,
                tuning.AmbientColor,
                tuning.CastShadowStrength,
                tuning.CastShadowSoftness) with
            {
                LightDirection = era.ResolveLightDirection(gameTime),
            };

            using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
                pack, textures, new TerrainMinimapCompositionOptions(resolution, lighting));
            DrawCompass(image);

            string path = Path.Combine(outputDirectory, $"sun-{hour:D2}00.png");
            image.SaveAsPng(path);

            Vector3 direction = lighting.LightDirection;
            float azimuth = (MathF.Atan2(direction.Y, direction.X) * 180f / MathF.PI + 360f) % 360f;
            float elevation = MathF.Asin(Math.Clamp(direction.Z, -1f, 1f)) * 180f / MathF.PI;
            Console.WriteLine(
                $"  {hour:D2}:00  azimuth {azimuth,5:0.0} deg ({DescribeBearing(azimuth)})  " +
                $"elevation {elevation,4:0.0} deg  -> {Path.GetFileName(path)}");
        }

        Console.WriteLine();
        Console.WriteLine($"Wrote 7 frames to {Path.GetFullPath(outputDirectory)}");
        Console.WriteLine("If the lit flank moves between frames, the bearing is sweeping. If it stays put");
        Console.WriteLine("but sits on the wrong side, the bearing is inverted. If these frames are correct");
        Console.WriteLine("and real tiles are not, the fault is in the real tiles' orientation, not the sun.");
    }

    /// <summary>Names a bearing in world axes, where +X is North and +Y is West.</summary>
    private static string DescribeBearing(float azimuthDegrees) => azimuthDegrees switch
    {
        < 22.5f or >= 337.5f => "North",
        < 67.5f => "North-West",
        < 112.5f => "West",
        < 157.5f => "South-West",
        < 202.5f => "South",
        < 247.5f => "South-East",
        < 292.5f => "East",
        _ => "North-East",
    };

    /// <summary>A smooth central hill plus MCNR-equivalent normals derived from its own heights.</summary>
    private static TerrainTileTensorPack BuildDiagnosticHillPack()
    {
        const int Grid = 257;
        const float Centre = (Grid - 1) / 2f;
        const float Sigma = Grid / 6f;

        var height = new float[Grid, Grid];
        var flat = new float[Grid * Grid];
        for (int row = 0; row < Grid; row++)
        {
            for (int column = 0; column < Grid; column++)
            {
                float dr = (row - Centre) / Sigma;
                float dc = (column - Centre) / Sigma;
                float value = 400f * MathF.Exp(-0.5f * ((dr * dr) + (dc * dc)));
                height[row, column] = value;
                flat[(row * Grid) + column] = value;
            }
        }

        var normals = new float[Grid, Grid, 3];
        var mask = new bool[Grid, Grid];
        for (int row = 0; row < Grid; row++)
        {
            for (int column = 0; column < Grid; column++)
            {
                Vector3 normal = AdtTerrainMath.ComputeNormal(flat, column, row);
                normals[row, column, 0] = normal.X;
                normals[row, column, 1] = normal.Y;
                normals[row, column, 2] = normal.Z;
                mask[row, column] = true;
            }
        }

        var textureIds = new int[1, 1, 4];
        textureIds[0, 0, 0] = 0;
        textureIds[0, 0, 1] = -1;
        textureIds[0, 0, 2] = -1;
        textureIds[0, 0, 3] = -1;

        return new TerrainTileTensorPack
        {
            TileX = 32,
            TileY = 32,
            MclyTextureIds = textureIds,
            MclyTextureNames = ["diagnostic.blp"],
            McnrNormalXyz = normals,
            McnrMask257 = mask,
            Height257 = height,
        };
    }

    /// <summary>Burns N/E/S/W ticks into the frame so the reader never has to assume an orientation.</summary>
    private static void DrawCompass(Image<Rgba32> image)
    {
        int size = image.Width;
        int margin = Math.Max(4, size / 32);
        var north = new Rgba32(255, 64, 64, 255);
        var other = new Rgba32(255, 255, 255, 255);

        // North marker is longer and red; the remaining three are plain ticks.
        DrawBar(image, size / 2, margin, 3, margin * 3, north);
        DrawBar(image, size / 2, size - margin - (margin * 2), 3, margin * 2, other);
        DrawBar(image, margin, size / 2, margin * 2, 3, other);
        DrawBar(image, size - margin - (margin * 2), size / 2, margin * 2, 3, other);
    }

    private static void DrawBar(Image<Rgba32> image, int x, int y, int width, int height, Rgba32 color)
    {
        for (int dy = 0; dy < height; dy++)
        {
            for (int dx = 0; dx < width; dx++)
            {
                int px = x + dx;
                int py = y + dy;
                if ((uint)px < (uint)image.Width && (uint)py < (uint)image.Height)
                    image[px, py] = color;
            }
        }
    }

    /// <summary>
    /// Scores synthesized tiles against their authored counterparts, for one or more named variants,
    /// and reports whether a change actually improved agreement.
    ///
    /// Reports brightness, contrast and structure separately and never collapses them before the
    /// per-metric lines are printed: a change can improve one and wreck another, and averaging first
    /// would hide exactly the failure mode that produced the flat-render bug.
    /// </summary>
    private sealed class SyntheticMinimapScorecard
    {
        private readonly List<(int TileX, int TileY, string Variant, MinimapComparisonMetrics Metrics)> _rows = [];

        public void Add(int tileX, int tileY, string variant, MinimapComparisonMetrics metrics)
        {
            if (metrics.PixelCount <= 0)
                return;

            lock (_rows)
                _rows.Add((tileX, tileY, variant, metrics));
        }

        public void Report(string outputDirectory, string baselineVariant, string currentVariant)
        {
            lock (_rows)
            {
                if (_rows.Count == 0)
                {
                    Console.WriteLine();
                    Console.WriteLine("=== Authored comparison: no tile produced a scorable comparison. ===");
                    return;
                }

                string csvPath = Path.Combine(outputDirectory, "authored-comparison.csv");
                var csv = new List<string> { MinimapComparisonMetrics.CsvHeader };
                foreach ((int tileX, int tileY, string variant, MinimapComparisonMetrics metrics) in
                    _rows.OrderBy(static r => r.TileY).ThenBy(static r => r.TileX).ThenBy(static r => r.Variant))
                {
                    csv.Add(metrics.ToCsvRow(tileX, tileY, variant));
                }

                File.WriteAllLines(csvPath, csv);

                Console.WriteLine();
                Console.WriteLine("=== Authored comparison ===");
                Console.WriteLine($"Per-tile rows: {csvPath}");
                Console.WriteLine();
                Console.WriteLine("Target for every ratio is 1.000; correlation and score target 1.000; MAE targets 0.");
                Console.WriteLine();
                Console.WriteLine($"{"variant",-12} {"tiles",5} {"mean",8} {"contrast",9} {"corr",8} {"MAE",8} {"score",8}");

                foreach (string variant in _rows.Select(static r => r.Variant).Distinct())
                {
                    List<MinimapComparisonMetrics> metrics = _rows
                        .Where(r => r.Variant == variant)
                        .Select(static r => r.Metrics)
                        .ToList();

                    Console.WriteLine(
                        $"{variant,-12} {metrics.Count,5} " +
                        $"{metrics.Average(static m => m.MeanRatio),8:0.000} " +
                        $"{metrics.Average(static m => m.ContrastRatio),9:0.000} " +
                        $"{metrics.Average(static m => m.LumaCorrelation),8:0.000} " +
                        $"{metrics.Average(static m => m.MeanAbsoluteError),8:0.000} " +
                        $"{metrics.Average(static m => m.Score),8:0.000}");
                }

                ReportDelta(baselineVariant, currentVariant);
            }
        }

        private void ReportDelta(string baselineVariant, string currentVariant)
        {
            List<MinimapComparisonMetrics> baseline = _rows.Where(r => r.Variant == baselineVariant).Select(static r => r.Metrics).ToList();
            List<MinimapComparisonMetrics> current = _rows.Where(r => r.Variant == currentVariant).Select(static r => r.Metrics).ToList();
            if (baseline.Count == 0 || current.Count == 0)
                return;

            Console.WriteLine();
            Console.WriteLine($"Change from '{baselineVariant}' to '{currentVariant}':");
            ReportMetricDelta("mean ratio", baseline.Average(static m => m.MeanRatio), current.Average(static m => m.MeanRatio), 1f);
            ReportMetricDelta("contrast ratio", baseline.Average(static m => m.ContrastRatio), current.Average(static m => m.ContrastRatio), 1f);
            ReportMetricDelta("correlation", baseline.Average(static m => m.LumaCorrelation), current.Average(static m => m.LumaCorrelation), 1f);
            ReportMetricDelta("MAE", baseline.Average(static m => m.MeanAbsoluteError), current.Average(static m => m.MeanAbsoluteError), 0f);
            ReportMetricDelta("score", baseline.Average(static m => m.Score), current.Average(static m => m.Score), 1f);

            int improved = 0;
            for (int index = 0; index < Math.Min(baseline.Count, current.Count); index++)
            {
                if (current[index].Score > baseline[index].Score)
                    improved++;
            }

            Console.WriteLine();
            Console.WriteLine($"Tiles improved by score: {improved}/{Math.Min(baseline.Count, current.Count)}");
        }

        /// <summary>
        /// Prints a metric with its distance from target, so "better" means measurably closer to the
        /// target rather than merely larger or smaller.
        /// </summary>
        private static void ReportMetricDelta(string label, float baseline, float current, float target)
        {
            float baselineError = MathF.Abs(baseline - target);
            float currentError = MathF.Abs(current - target);
            string verdict = currentError < baselineError - 1e-4f
                ? "BETTER"
                : currentError > baselineError + 1e-4f ? "WORSE" : "same";
            Console.WriteLine(
                $"  {label,-16} {baseline,8:0.000} -> {current,8:0.000}   " +
                $"(|err| {baselineError:0.000} -> {currentError:0.000})  {verdict}");
        }
    }

    /// <summary>
    /// Accumulates per-tile azimuth-sweep results so the run can state whether this era's sun holds
    /// a fixed bearing or sweeps, rather than each tile reporting in isolation.
    /// </summary>
    private sealed class AzimuthSweepAccumulator
    {
        private readonly List<(int TileX, int TileY, float Azimuth, float Hour, float Score, float Margin)> _results = [];

        public void Add(int tileX, int tileY, MinimapShadingMatch.AzimuthSweepResult result)
        {
            lock (_results)
            {
                _results.Add((tileX, tileY, result.BestAzimuthDegrees, result.BestHour, result.BestScore01, result.AzimuthMargin));
            }
        }

        public void Report(MinimapEraProfile era)
        {
            lock (_results)
            {
                Console.WriteLine();
                Console.WriteLine($"=== Solar azimuth sweep: {era.EraLabel} ===");
                if (_results.Count == 0)
                {
                    Console.WriteLine("  No tile produced an evaluable match (flat terrain or missing normals).");
                    return;
                }

                foreach ((int tileX, int tileY, float azimuth, float hour, float score, float margin) in
                    _results.OrderBy(static r => r.TileY).ThenBy(static r => r.TileX))
                {
                    Console.WriteLine(
                        $"  Tile {tileX:D2},{tileY:D2}: best azimuth {azimuth,5:0}deg  hour {hour,5:0.0}  " +
                        $"score {score:0.000}  azimuth-margin {margin:0.000}");
                }

                // Circular mean, because bearings wrap: a naive average of 350 and 10 gives 180.
                double sumSin = 0.0, sumCos = 0.0;
                foreach ((_, _, float azimuth, _, _, _) in _results)
                {
                    double radians = azimuth * Math.PI / 180.0;
                    sumSin += Math.Sin(radians);
                    sumCos += Math.Cos(radians);
                }

                double meanAzimuth = (Math.Atan2(sumSin, sumCos) * 180.0 / Math.PI + 360.0) % 360.0;
                // Circular resultant length: 1.0 means every tile agreed on one bearing, near 0
                // means they are scattered around the compass.
                double concentration = Math.Sqrt((sumSin * sumSin) + (sumCos * sumCos)) / _results.Count;
                double meanMargin = _results.Average(static r => r.Margin);

                Console.WriteLine();
                Console.WriteLine($"  Tiles evaluated:      {_results.Count}");
                Console.WriteLine($"  Circular mean bearing: {meanAzimuth:0.0} deg");
                Console.WriteLine($"  Agreement (0..1):      {concentration:0.000}");
                Console.WriteLine($"  Mean azimuth-margin:   {meanMargin:0.000}");
                Console.WriteLine(
                    $"  Traced 1.0.0 bearing:  {TerrainSolarDirection.TracedSourceAzimuthDegrees:0} deg " +
                    $"(separation {MinimapShadingMatch.AngularSeparationDegrees((float)meanAzimuth, TerrainSolarDirection.TracedSourceAzimuthDegrees):0.0} deg)");
                Console.WriteLine();
                Console.WriteLine("  How to read this: high agreement means the tiles concur on one bearing. Whether");
                Console.WriteLine("  that means the SUN is fixed depends on the tiles spanning different capture times;");
                Console.WriteLine("  tiles all captured at one time of day would agree even under a sweeping sun.");
                Console.WriteLine("  A low mean azimuth-margin means the scoring could not pin a bearing at all, and");
                Console.WriteLine("  the mean above should not be trusted regardless of how tight the agreement looks.");
            }
        }
    }

    /// <summary>
    /// Runs the Spec 111 shading-match hour sweep against the authored minimap and returns the
    /// best-matching time of day (hours) for this tile, or null when the match is not evaluable
    /// (wrong build fingerprint, missing normals/textures, or too-flat terrain). Used by
    /// <c>synthetic-minimap --match-time</c> to render each tile at its inferred sun elevation.
    /// </summary>
    private static float? TryInferAuthoredMinimapHour(
        TerrainTileTensorPack pack,
        NativeMpqService catalog,
        string mapName,
        byte[,,] authoredMinimapRgb,
        string? buildVersion)
    {
        if (pack.McnrNormalXyz is null)
            return null;

        Dictionary<int, byte[,,]> textures = LoadSyntheticMinimapTextures(catalog, pack);
        bool untextured = !pack.MclyTextureNames.Any(static n => !string.IsNullOrWhiteSpace(n));
        if (textures.Count == 0 && !untextured)
            return null;

        try
        {
            MinimapLightingProvenance provenance = MinimapShadingMatch.Evaluate(
                MinimapLightingProvenance.NotEvaluated("synthetic-minimap --match-time"),
                pack,
                textures,
                authoredMinimapRgb,
                buildVersion ?? string.Empty);
            return provenance.ShadingMatchStatus is "matched" or "low_confidence_ambiguous"
                ? provenance.ShadingMatchedTimeOfDayHours
                : null;
        }
        catch
        {
            return null;
        }
    }

    private static void AttachNameAlignedTexturePixels(NativeMpqService catalog, TerrainTileTensorPack pack)
    {
        if (pack.MclyTextureNames.Count == 0)
            return;

        var pixels = new List<byte[,,]>(pack.MclyTextureNames.Count);
        for (int textureId = 0; textureId < pack.MclyTextureNames.Count; textureId++)
        {
            string textureName = pack.MclyTextureNames[textureId];
            byte[,,]? texture = LoadTerrainTextureRgbProxy(
                catalog,
                textureName,
                out string resolvedPath,
                out string? resolutionKind);
            if (texture is null)
            {
                pack.MclyTexturePixels = null;
                pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals, StringComparer.OrdinalIgnoreCase)
                {
                    "mcly_texture_pixels_incomplete",
                };
                return;
            }

            if (resolutionKind is not null)
                RecordTerrainTextureFallback(pack, textureId, textureName, resolvedPath, resolutionKind);

            pixels.Add(texture);
        }

        pack.MclyTexturePixels = pixels;
        pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals, StringComparer.OrdinalIgnoreCase)
        {
            "mcly_texture_pixels",
        };
    }

    private static IEnumerable<int> EnumerateReferencedTextureIds(TerrainTileTensorPack pack)
    {
        int[,,]? textureIds = pack.MclyTextureIds;
        if (textureIds is null)
            yield break;

        var result = new HashSet<int>();
        for (int y = 0; y < textureIds.GetLength(0); y++)
            for (int x = 0; x < textureIds.GetLength(1); x++)
                for (int layer = 0; layer < Math.Min(4, textureIds.GetLength(2)); layer++)
                    if (textureIds[y, x, layer] >= 0)
                        result.Add(textureIds[y, x, layer]);

        foreach (int textureId in result)
            yield return textureId;
    }

    private static byte[,,] CopyRgb(Image<Rgba32> image)
    {
        var result = new byte[image.Height, image.Width, 3];
        for (int y = 0; y < image.Height; y++)
        {
            for (int x = 0; x < image.Width; x++)
            {
                Rgba32 pixel = image[x, y];
                result[y, x, 0] = pixel.R;
                result[y, x, 1] = pixel.G;
                result[y, x, 2] = pixel.B;
            }
        }

        return result;
    }

    private static IReadOnlyList<MinimapLightingTimeCandidate> LoadMinimapLightingTimeCandidates(
        NativeMpqService catalog,
        string mapName)
    {
        foreach (string candidatePath in EnumerateMapLitPaths(mapName))
        {
            byte[]? bytes = catalog.ReadFile(candidatePath);
            if (bytes is null || bytes.Length == 0)
                continue;

            try
            {
                using var stream = new MemoryStream(bytes, writable: false);
                LitFileProfile profile = LitProfileReader.Read(stream, candidatePath);
                var candidates = new List<MinimapLightingTimeCandidate>(24);
                for (int hour = 0; hour < 24; hour++)
                {
                    TerrainLightingSample sample = LitTerrainDayNightProfile.EvaluateGlobalClear(profile, hour / 24f).Lighting;
                    candidates.Add(new MinimapLightingTimeCandidate(
                        hour,
                        sample.DirectionalColor.X + sample.AmbientColor.X,
                        sample.DirectionalColor.Y + sample.AmbientColor.Y,
                        sample.DirectionalColor.Z + sample.AmbientColor.Z,
                        $"LitGlobalClear:{candidatePath}"));
                }

                return candidates;
            }
            catch
            {
                // A failed LIT decode only means time cannot be bucketed from this file; the
                // tint/shadow analysis remains useful without a candidate palette.
            }
        }

        return [];
    }

    private static IEnumerable<string> EnumerateMapLitPaths(string mapName) =>
    [
        $"World\\{mapName}\\lights.lit",
        $"World\\Maps\\{mapName}\\lights.lit",
        $"World\\{mapName}\\areatest.lit",
        $"World\\Maps\\{mapName}\\areatest.lit",
        $"World\\{mapName}\\light.lit",
        $"World\\Maps\\{mapName}\\light.lit",
    ];

    /// <summary>
    /// Loads the map's LIT lights as world-space overlay markers, or an empty list when the map has
    /// no readable LIT file. Colour comes from each light's clear-weather track evaluated at the
    /// render's time of day, so the swatch shows what that light actually contributes at that hour.
    /// </summary>
    private static IReadOnlyList<MinimapLightMarker> LoadMinimapLightMarkers(
        NativeMpqService catalog,
        string mapName,
        float gameTime)
    {
        foreach (string litPath in EnumerateMapLitPaths(mapName))
        {
            byte[]? bytes = catalog.ReadFile(litPath);
            if (bytes is null || bytes.Length == 0)
                continue;

            try
            {
                using var stream = new MemoryStream(bytes, writable: false);
                LitFileProfile profile = LitProfileReader.Read(stream, litPath);
                var markers = new List<MinimapLightMarker>();
                foreach (LitLightProfile light in profile.Lights)
                {
                    // The default entry (-1/-1/-1) is the map-wide fallback, not a placed light; it
                    // has no meaningful position to plot.
                    if (light.Header is not { } header || header.IsDefault)
                        continue;
                    if (header.WorldOuterRadius <= 0f)
                        continue;

                    markers.Add(new MinimapLightMarker(
                        header.WorldPosition,
                        header.WorldRadius,
                        header.WorldOuterRadius,
                        ResolveLightSwatchColor(light, gameTime),
                        header.Name));
                }

                Console.WriteLine($"LIT lights: {markers.Count} placed light(s) from {litPath}");
                return markers;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"LIT lights: {litPath} could not be decoded ({ex.Message}).");
            }
        }

        Console.WriteLine($"LIT lights: no readable LIT file found for map '{mapName}'; overlay will be empty.");
        return [];
    }

    /// <summary>
    /// Colour for a light's overlay swatch: its clear-weather track at the requested time of day.
    /// Falls back to white so a light with no evaluable colour still plots its position rather than
    /// vanishing from the diagnostic.
    /// </summary>
    private static Vector3 ResolveLightSwatchColor(LitLightProfile light, float gameTime)
    {
        LitLightGroupProfile? clear = light.Groups.FirstOrDefault(static g => g.Kind == LitLightGroupKind.Clear)
            ?? light.Groups.FirstOrDefault();
        if (clear is null || clear.Tracks.Count == 0)
            return Vector3.One;

        float litTime = Math.Clamp(gameTime, 0f, 1f) * LitProfileReader.TimeUnitsPerDay;
        foreach (LitColorTrack track in clear.Tracks)
        {
            if (track.TryEvaluate(litTime, out Vector3 color) && color.LengthSquared() > 1e-6f)
                return Vector3.Clamp(color, Vector3.Zero, Vector3.One);
        }

        return Vector3.One;
    }

    private static void SetMinimapLightingProvenance(
        TerrainTileTensorPack pack,
        MinimapLightingProvenance provenance)
    {
        pack.MinimapLightingProvenance = provenance;
        var signals = new HashSet<string>(pack.AvailableSignals, StringComparer.OrdinalIgnoreCase)
        {
            "minimap_lighting_provenance_v1"
        };
        if (!string.Equals(provenance.ShadingMatchStatus, "not_evaluated", StringComparison.Ordinal))
            signals.Add("minimap_shading_match_v1");
        pack.AvailableSignals = signals;
    }

    /// <summary>
    /// Computes the maximum terrain height on a tile from its Height257 array.
    /// Used by the WMO overlay pass to skip WMOs that are entirely below ground.
    /// </summary>
    private static float ComputeTerrainMaxHeight(TerrainTileTensorPack pack)
    {
        float[,]? height257 = pack.Height257;
        if (height257 is null)
            return float.MaxValue;

        float maxZ = float.NegativeInfinity;
        for (int y = 0; y < height257.GetLength(0); y++)
            for (int x = 0; x < height257.GetLength(1); x++)
                if (float.IsFinite(height257[y, x]) && height257[y, x] > maxZ)
                    maxZ = height257[y, x];
        return float.IsFinite(maxZ) ? maxZ : float.MaxValue;
    }

    /// <summary>
    /// Composites a WMO overlay image onto an existing terrain-only minimap PNG.
    /// Uses the WMO mask to blend: terrain * (1 - mask) + wmo * mask.
    /// </summary>
    private static void CompositeWmoOverlay(string tilePath, byte[] wmoImageRgba, byte[] wmoMaskRgba, int resolution)
    {
        using Image<Rgba32> terrain = Image.Load<Rgba32>(tilePath);
        if (terrain.Width != resolution || terrain.Height != resolution)
            return;

        terrain.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < resolution; y++)
            {
                Span<Rgba32> row = accessor.GetRowSpan(y);
                for (int x = 0; x < resolution; x++)
                {
                    int idx = (y * resolution + x) * 4;
                    // Mask is white where a WMO fragment is visible.
                    if (wmoMaskRgba[idx] > 128)
                    {
                        row[x].R = wmoImageRgba[idx + 0];
                        row[x].G = wmoImageRgba[idx + 1];
                        row[x].B = wmoImageRgba[idx + 2];
                    }
                }
            }
        });

        terrain.SaveAsPng(tilePath);
    }

    /// <summary>
    /// The contrast controls for synthetic-minimap export, surfaced as CLI flags so they can be
    /// swept against an authored comparison without a rebuild. Ambient and cast strength set how
    /// deep shadows read; overall brightness is held steady across both by the derived light gain.
    /// </summary>
    private sealed record SyntheticMinimapTuning(
        Vector3 AmbientColor,
        float CastShadowStrength,
        float CastShadowSoftness,
        float MaxCastShadowLength,
        float? LightGainOverride,
        MinimapLiquidPalette LiquidPalette,
        MinimapEraProfile Era,
        bool ExactEraMatch)
    {
        /// <summary>
        /// Era supplies every era-sensitive default; explicit CLI flags then override individual
        /// values on top. Blizzard changed minimap generation between Alpha (0.5.3), Beta 1 (0.6.0)
        /// and release (1.0.0), so a build's era -- not a global constant -- is the right source.
        /// </summary>
        public static SyntheticMinimapTuning FromEra(MinimapEraProfile era, bool exactEraMatch) => new(
            era.AmbientColor,
            era.CastShadowStrength,
            era.CastShadowSoftness,
            TerrainCastShadowMap.UncappedShadowLength,
            null,
            era.Liquid,
            era,
            exactEraMatch);

        public static SyntheticMinimapTuning Default { get; } =
            FromEra(MinimapEraProfile.Default, exactEraMatch: false);
    }

    private static SyntheticMinimapLightingProfile ResolveSyntheticMinimapLighting(
        float gameTime,
        bool castShadows = true,
        SyntheticMinimapTuning? tuningOverride = null)
    {
        SyntheticMinimapTuning tuning = tuningOverride ?? SyntheticMinimapTuning.Default;
        string timeLabel = TimeOfDayClock.FromHours(gameTime * 24f).CompactText;
        // Lambert hillshading PLUS analytic sun-driven cast shadows, at the sun direction from
        // --time-hours, shaded in linear space and encoded to sRGB on output. MCSH stays out of
        // normal RGB (real authored minimaps do not bake it); --bake-mcsh is the explicit opt-in
        // and is a separate signal from the cast-shadow pass.
        TerrainMinimapLighting lighting = TerrainMinimapLighting.CreateShadedTerrain(
            gameTime,
            tuning.AmbientColor,
            tuning.CastShadowStrength,
            tuning.CastShadowSoftness,
            tuning.MaxCastShadowLength) with
        {
            // Bearing comes from the era profile, not the hardcoded traced constant: whether the
            // sun holds a fixed bearing or sweeps east-to-west is era-specific and, for Alpha,
            // still unverified.
            LightDirection = tuning.Era.ResolveLightDirection(gameTime),
            ApplyMcshToMinimap = false,
            ApplyCastShadows = castShadows
        };
        if (tuning.LightGainOverride is { } explicitGain)
            lighting = lighting with { LinearLightGain = explicitGain };
        string castLabel = castShadows ? "+cast-shadows-v1" : string.Empty;
        return new SyntheticMinimapLightingProfile(
            lighting,
            $"ShadedTerrain@{timeLabel}",
            $"synthetic_minimap_solar_{timeLabel}_global_white",
            $"terrain-minimap-solar-white-lambert-v3+renderer-space-mcnr-v2{castLabel}" +
            $"+linear-shading-v1+{tuning.Era.RenderProfile}",
            null,
            null,
            null,
            null,
            $"solar_{timeLabel}_shared_solar_direction_not_lit_or_dbc_data",
            "mcsh_omitted_from_normal_minimap_rgb",
            "Synthetic minimaps use the shared solar direction at the requested time with Lambert " +
            (castShadows ? "hillshading plus analytic single-tile cast shadows" : "hillshading only") +
            ", shaded in linear space; MCSH is omitted from normal RGB and map LIT / Light DBC " +
            "profiles are excluded.");
    }

    private static SyntheticMinimapLightingProfile WithBakedMcsh(SyntheticMinimapLightingProfile profile) =>
        profile with
        {
            Lighting = profile.Lighting with
            {
                ApplyMcshToMinimap = true,
                McshShadowStrength = TerrainLightingMath.DefaultAuthoredMcshShadowStrength
            },
            McshEvidenceState = $"{profile.McshEvidenceState}; explicit_baked_mcsh_preview"
        };

    static void GenerateSyntheticMinimap(NativeMpqService catalog, TerrainTileTensorPack pack, int tileX, int tileY, string outputPath)
    {
        try
        {
            Dictionary<int, byte[,,]> textures = LoadSyntheticMinimapTextures(catalog, pack);
            if (textures.Count == 0)
                throw new InvalidDataException("No referenced BLP texture could be decoded.");

            var compositionOptions = new TerrainMinimapCompositionOptions(
                TerrainMinimapCompositor.DefaultResolution,
                TerrainMinimapLighting.CreateNoonWhiteGlobal());
            using Image<Rgba32> image = TerrainMinimapCompositor.Compose(pack, textures, compositionOptions);
            using Image<Rgba32> liquidImage = TerrainMinimapLiquidCompositor.Compose(image, pack, out int liquidPixelCount);
            string? directory = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrWhiteSpace(directory))
                Directory.CreateDirectory(directory);
            image.SaveAsPng(outputPath);
            string liquidOutputPath = Path.Combine(
                directory ?? string.Empty,
                $"{Path.GetFileNameWithoutExtension(outputPath)}_liquid{Path.GetExtension(outputPath)}");
            liquidImage.SaveAsPng(liquidOutputPath);
            Console.WriteLine($"Synthetic minimap: {outputPath} + {liquidOutputPath} ({liquidPixelCount} liquid pixels)");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: synthetic minimap generation failed: {ex.Message}");
        }
    }

    static WlLooseFileEntry[] GetArchiveWlFiles(NativeMpqService catalog, string clientRoot, string mapName)
    {
        string cacheKey = $"{Path.GetFullPath(clientRoot)}|{mapName}";
        Lazy<WlLooseFileEntry[]> lazy = _wlLooseFileCache.GetOrAdd(
            cacheKey,
            _ => new Lazy<WlLooseFileEntry[]>(
                () => LoadArchiveWlFiles(catalog, clientRoot, mapName),
                LazyThreadSafetyMode.ExecutionAndPublication));
        return lazy.Value;
    }

    static WlLooseFileEntry[] LoadArchiveWlFiles(NativeMpqService catalog, string clientRoot, string mapName)
    {
        string mapPrefix = $"World\\Maps\\{mapName}\\";
        string[] paths = catalog.GetAllKnownFiles()
            .Select(path => path.Replace('/', '\\'))
            .Where(path =>
                path.StartsWith(mapPrefix, StringComparison.OrdinalIgnoreCase) &&
                (path.EndsWith(".wlw", StringComparison.OrdinalIgnoreCase) ||
                 path.EndsWith(".wlm", StringComparison.OrdinalIgnoreCase) ||
                 path.EndsWith(".wlq", StringComparison.OrdinalIgnoreCase) ||
                 path.EndsWith(".wll", StringComparison.OrdinalIgnoreCase)))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (paths.Length == 0)
            paths = ReadSupplementalListfileEntriesForMap(mapName);

        if (paths.Length == 0)
        {
            Console.WriteLine($"  [WL] no WL* files found in loaded archives for {mapName}");
            return [];
        }

        var loaded = new List<WlLooseFileEntry>(paths.Length);
        foreach (string path in paths)
        {
            try
            {
                byte[]? data = catalog.ReadFile(path);
                if (data is null || data.Length == 0)
                    continue;

                using var ms = new MemoryStream(data, writable: false);
                loaded.Add(new WlLooseFileEntry
                {
                    Path = path,
                    File = WlFileReader.Read(ms, path)
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [WL] failed to read {Path.GetFileName(path)}: {ex.Message}");
            }
        }

        Console.WriteLine($"  [WL] discovered {loaded.Count}/{paths.Length} WL* files in loaded archives for {mapName}");
        return loaded.ToArray();
    }

    static void TryAddWlLiquidFromArchiveFiles(NativeMpqService catalog, string clientRoot, string mapName, int tileX, int tileY, TerrainTileTensorPack pack)
    {
        WlLooseFileEntry[] wlFiles = GetArchiveWlFiles(catalog, clientRoot, mapName);
        if (wlFiles.Length == 0)
            return;

        if (WlLiquidRasterizer.TryRasterize(
                wlFiles.Select(static entry => entry.File),
                tileX,
                tileY,
                out float[,]? mask,
                out float[,]? heights,
                out byte[,]? basicTypes))
        {
            if (mask is null || heights is null || basicTypes is null
                || WlLiquidRasterizer.KeepOnlyAboveTerrain(mask, heights, pack.Height257, basicTypes) == 0)
                return;

            pack.WlLiquidMask = mask;
            pack.WlLiquidHeight = heights;
            pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals)
            {
                "wl_liquid_mask",
                "wl_liquid_height",
                WlLiquidRasterizer.SurfaceRasterizationSignal,
                WlLiquidRasterizer.AboveTerrainSignal,
                WlLiquidRasterizer.BasicTypeSignal
            };

            pack.LiquidBasicType257 = LiquidBasicTypePackBuilder.OverlayWlFallbackTypes(
                pack.LiquidBasicType257,
                mask,
                basicTypes,
                pack.Mh2oSurfaceHeight,
                pack.Mh2oPresenceMask,
                pack.MclqSurfaceHeight,
                pack.MclqPresenceMask);

            // Rebuild unified liquid: MH2O > MCLQ > WL*
            if (pack.UnifiedLiquidMask is null && pack.Mh2oSurfaceHeight is null && pack.MclqSurfaceHeight is null)
            {
                pack.UnifiedLiquidMask = mask;
                pack.UnifiedLiquidHeight = heights;
                pack.AvailableSignals = new HashSet<string>(pack.AvailableSignals) { "unified_liquid_mask", "unified_liquid_height" };
            }
        }
    }

    static bool TryLoadMinimap(string adtPath, string minimapRoot, out byte[,,]? minimap)
    {
        minimap = null;
        string stem = Path.GetFileNameWithoutExtension(adtPath);

        string[] candidates =
        [
            Path.Combine(minimapRoot, $"{stem}.png"),
            Path.Combine(minimapRoot, "images", $"{stem}.png"),
            Path.Combine(minimapRoot, "reference_minimaps", $"{stem}_reference_minimap.png"),
        ];

        foreach (string candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                try
                {
                    using var img = SixLabors.ImageSharp.Image.Load<Rgba32>(candidate);
                    if (img.Width == 256 && img.Height == 256)
                    {
                        minimap = new byte[256, 256, 3];
                        for (int y = 0; y < 256; y++)
                        {
                            for (int x = 0; x < 256; x++)
                            {
                                var px = img[x, y];
                                minimap[y, x, 0] = px.R;
                                minimap[y, x, 1] = px.G;
                                minimap[y, x, 2] = px.B;
                            }
                        }
                        return true;
                    }
                }
                catch { }
            }
        }

        return false;
    }

    static void LoadMd5Translate(string clientRoot, NativeMpqService catalog)
    {
        if (Md5TranslateResolver.TryLoad(
            new[] { clientRoot },
            path => catalog.FileExists(path),
            path => catalog.ReadFile(path),
            out var md5Index))
        {
            _md5Lookup = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var kv in md5Index.PlainToHash)
                _md5Lookup[kv.Key] = kv.Value;
            Console.Error.WriteLine($"  Loaded {_md5Lookup.Count} md5translate entries");
        }
    }

    static byte[,,]? TryLoadMinimapFromMpq(NativeMpqService catalog, string mapName, int tileX, int tileY)
    {
        string mapLower = mapName.ToLowerInvariant();
        string x2 = tileX.ToString("00");
        string y2 = tileY.ToString("00");
        string canonicalPath = $"textures/minimap/{mapLower}/map{x2}_{y2}.blp";

        // MD5 translate lookup
        if (_md5Lookup is not null && _md5Lookup.TryGetValue(canonicalPath, out string? md5Name))
        {
            byte[]? blpBytes = catalog.ReadFile(md5Name);
            if (blpBytes is not null && blpBytes.Length >= 8)
            {
                try
                {
                    byte[,,]? rgb = DecodeBlpToRgb(blpBytes);
                    if (rgb is not null) return rgb;
                }
                catch { }
            }
        }

        // Fallback candidates matching WoWViewer EnumerateTileCandidates
        string[] candidates =
        [
            canonicalPath.Replace('/', '\\'),                                        // textures\minimap\azeroth\map32_32.blp
            $"textures\\minimap\\{mapLower}\\map{x2}_{y2}.blp",                    // textures\minimap\azeroth\map32_32.blp
            $"textures\\Minimap\\{mapLower}\\map{x2}_{y2}.blp",                    // textures\Minimap\azeroth\map32_32.blp
            $"world\\minimaps\\{mapLower}\\map{x2}_{y2}.blp",                      // world\minimaps\azeroth\map32_32.blp
            $"world\\Minimaps\\{mapName}\\map{x2}_{y2}.blp",                       // world\Minimaps\Azeroth\map32_32.blp
            $"{mapLower}\\map{x2}_{y2}.blp",                                        // azeroth\map32_32.blp
        ];

        foreach (string candidate in candidates)
        {
            byte[]? blpBytes = catalog.ReadFile(candidate);
            if (blpBytes is null || blpBytes.Length < 8) continue;

            try
            {
                byte[,,]? rgb = DecodeBlpToRgb(blpBytes);
                if (rgb is not null) return rgb;
            }
            catch { }
        }

        return null;
    }

    /// <summary>
    /// Load the raw authored minimap BLP bytes for a tile, for the encoding survey (FR-013). Mirrors
    /// the candidate paths of <see cref="TryLoadMinimapFromMpq"/> but returns the raw bytes.
    /// </summary>
    static byte[]? TryLoadMinimapBlpBytes(NativeMpqService catalog, string mapName, int tileX, int tileY)
    {
        string mapLower = mapName.ToLowerInvariant();
        string x2 = tileX.ToString("00");
        string y2 = tileY.ToString("00");
        string canonicalPath = $"textures/minimap/{mapLower}/map{x2}_{y2}.blp";

        if (_md5Lookup is not null && _md5Lookup.TryGetValue(canonicalPath, out string? md5Name))
        {
            byte[]? blpBytes = catalog.ReadFile(md5Name);
            if (blpBytes is not null && blpBytes.Length >= 8)
                return blpBytes;
        }

        string[] candidates =
        [
            canonicalPath.Replace('/', '\\'),
            $"textures\\minimap\\{mapLower}\\map{x2}_{y2}.blp",
            $"textures\\Minimap\\{mapLower}\\map{x2}_{y2}.blp",
            $"world\\minimaps\\{mapLower}\\map{x2}_{y2}.blp",
            $"world\\Minimaps\\{mapName}\\map{x2}_{y2}.blp",
            $"{mapLower}\\map{x2}_{y2}.blp",
        ];

        foreach (string candidate in candidates)
        {
            byte[]? blpBytes = catalog.ReadFile(candidate);
            if (blpBytes is not null && blpBytes.Length >= 8)
                return blpBytes;
        }

        return null;
    }

    static byte[,,]? LoadTextureFromMpq(NativeMpqService catalog, string virtualPath)
    {
        byte[]? blpBytes = catalog.ReadFile(virtualPath);
        if (blpBytes is null || blpBytes.Length < 8)
            return null;
        try { return DecodeBlpToRgbNative(blpBytes); }
        catch { return null; }
    }

    static byte[,,]? DecodeBlpToRgbNative(byte[] blpBytes)
    {
        using var ms = new MemoryStream(blpBytes, writable: false);
        using var blp = new SereniaBLPLib.BlpFile(ms);
        using Image<Rgba32>? image = blp.GetImage(0);
        if (image == null) return null;

        int w = image.Width;
        int h = image.Height;
        if (w < 1 || h < 1) return null;

        var rgb = new byte[h, w, 3];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                Rgba32 px = image[x, y];
                rgb[y, x, 0] = px.R;
                rgb[y, x, 1] = px.G;
                rgb[y, x, 2] = px.B;
            }
        }
        return rgb;
    }

    static byte[,,]? DecodeBlpToRgb(byte[] blpBytes)
    {
        using var ms = new MemoryStream(blpBytes, writable: false);
        using var blp = new SereniaBLPLib.BlpFile(ms);
        using Image<Rgba32>? image = blp.GetImage(0);
        if (image == null) return null;

        int w = image.Width;
        int h = image.Height;
        if (w < 1 || h < 1) return null;

        var rgb = new byte[256, 256, 3];
        float scaleX = (float)(w - 1) / 255f;
        float scaleY = (float)(h - 1) / 255f;

        for (int y = 0; y < 256; y++)
        {
            for (int x = 0; x < 256; x++)
            {
                int sx = Math.Clamp((int)(x * scaleX + 0.5f), 0, w - 1);
                int sy = Math.Clamp((int)(y * scaleY + 0.5f), 0, h - 1);
                Rgba32 px = image[sx, sy];
                rgb[y, x, 0] = px.R;
                rgb[y, x, 1] = px.G;
                rgb[y, x, 2] = px.B;
            }
        }

        return rgb;
    }

    static string? DetectBuildVersionFromClientRoot(string clientRoot)
    {
        string dirName = Path.GetFileName(clientRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        if (string.IsNullOrWhiteSpace(dirName))
            return null;

        // If the last directory is a generic game root (not a build key), walk up.
        if (dirName.Equals("World of Warcraft", StringComparison.OrdinalIgnoreCase)
            || dirName.Equals("Data", StringComparison.OrdinalIgnoreCase))
        {
            string? parent = Path.GetDirectoryName(clientRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            if (string.IsNullOrWhiteSpace(parent))
                return null;
            dirName = Path.GetFileName(parent);
        }

        string normalizedPath = clientRoot.Replace('_', '.');
        System.Text.RegularExpressions.MatchCollection matches =
            System.Text.RegularExpressions.Regex.Matches(
                normalizedPath,
                @"(?<!\d)(\d+\.\d+\.\d+\.\d+)(?!\d)");
        for (int index = matches.Count - 1; index >= 0; index--)
        {
            string candidate = matches[index].Groups[1].Value;
            if (ClientBuildKey.TryParse(candidate, out _))
                return candidate;
        }

        string versionString = dirName.Replace('_', '.');
        return ClientBuildKey.TryParse(versionString, out _) ? versionString : null;
    }

    static string ResolveGameClientRoot(string clientRoot)
    {
        string normalizedRoot = clientRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string dataDir = Path.Combine(normalizedRoot, "Data");
        if (Directory.Exists(dataDir))
            return normalizedRoot;

        string nestedGameRoot = Path.Combine(normalizedRoot, "World of Warcraft");
        if (Directory.Exists(Path.Combine(nestedGameRoot, "Data")))
            return nestedGameRoot;

        return normalizedRoot;
    }

    /// <summary>
    /// Parses the synthetic-minimap contrast knobs. <c>--ambient</c> takes either one value or an
    /// <c>r,g,b</c> triple: shadowed ground is lit by ambient alone, so a tinted ambient is what
    /// makes shadows a different HUE from lit ground instead of merely a darker copy of it.
    /// </summary>
    private static SyntheticMinimapTuning? TryParseSyntheticMinimapTuning(
        string[] args,
        string? buildVersion,
        out string? error)
    {
        error = null;

        // Era first: it seeds every era-sensitive default, and the explicit flags below then
        // override individual values on top of it.
        MinimapEraProfile era = MinimapEraProfile.ResolveForBuild(buildVersion, out bool exactEraMatch);
        string? eraName = GetOption(args, "--era", "");
        if (!string.IsNullOrWhiteSpace(eraName))
        {
            MinimapEraProfile? explicitEra = MinimapEraProfile.TryResolveByName(eraName);
            if (explicitEra is null)
            {
                error = $"--era '{eraName}' is not recognised. Available: {MinimapEraProfile.AvailableNames}.";
                return null;
            }

            era = explicitEra;
            exactEraMatch = true;
        }

        SyntheticMinimapTuning tuning = SyntheticMinimapTuning.FromEra(era, exactEraMatch);

        string? azimuthModel = GetOption(args, "--sun-model", "");
        if (!string.IsNullOrWhiteSpace(azimuthModel))
        {
            SolarAzimuthModel? model = azimuthModel.Trim().ToLowerInvariant() switch
            {
                "fixed" => SolarAzimuthModel.Fixed,
                "sweep" or "eastwest" or "east-west" => SolarAzimuthModel.EastToWestSweep,
                _ => null,
            };
            if (model is null)
            {
                error = $"--sun-model expects 'fixed' or 'sweep', got '{azimuthModel}'.";
                return null;
            }

            tuning = tuning with { Era = tuning.Era with { AzimuthModel = model.Value } };
        }

        float? azimuth = GetFloatOption(args, "--sun-azimuth", "");
        if (azimuth is { } azimuthDegrees)
        {
            if (!float.IsFinite(azimuthDegrees))
            {
                error = $"--sun-azimuth expects degrees, got {azimuthDegrees}.";
                return null;
            }

            tuning = tuning with { Era = tuning.Era with { NoonAzimuthDegrees = azimuthDegrees } };
        }

        string? ambient = GetOption(args, "--ambient", "");
        if (!string.IsNullOrWhiteSpace(ambient))
        {
            string[] parts = ambient.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            var channels = new float[parts.Length];
            for (int index = 0; index < parts.Length; index++)
            {
                if (!float.TryParse(parts[index], NumberStyles.Float, CultureInfo.InvariantCulture, out channels[index])
                    || channels[index] < 0f
                    || channels[index] > 4f)
                {
                    error = $"--ambient expects one value or 'r,g,b' in 0..4, got '{ambient}'.";
                    return null;
                }
            }

            tuning = parts.Length switch
            {
                1 => tuning with { AmbientColor = new Vector3(channels[0]) },
                3 => tuning with { AmbientColor = new Vector3(channels[0], channels[1], channels[2]) },
                _ => tuning
            };
            if (parts.Length is not (1 or 3))
            {
                error = $"--ambient expects one value or 'r,g,b', got '{ambient}'.";
                return null;
            }
        }

        float? castStrength = GetFloatOption(args, "--cast-shadow-strength", "");
        if (castStrength is { } strength)
        {
            if (strength is < 0f or > 1f)
            {
                error = $"--cast-shadow-strength expects 0..1, got {strength}.";
                return null;
            }

            tuning = tuning with { CastShadowStrength = strength };
        }

        float? softness = GetFloatOption(args, "--shadow-softness", "");
        if (softness is { } softnessValue)
        {
            if (softnessValue is <= 0f or > 64f)
            {
                error = $"--shadow-softness expects world units in (0, 64], got {softnessValue}.";
                return null;
            }

            tuning = tuning with { CastShadowSoftness = softnessValue };
        }

        float? maxShadowLength = GetFloatOption(args, "--max-shadow-length", "");
        if (maxShadowLength is { } maxLength)
        {
            if (maxLength < 0f || maxLength > 4096f)
            {
                error = $"--max-shadow-length expects world units in 0..4096 (0 = uncapped), got {maxLength}.";
                return null;
            }

            tuning = tuning with { MaxCastShadowLength = maxLength };
        }

        float? gain = GetFloatOption(args, "--light-gain", "");
        if (gain is { } explicitGain)
        {
            if (explicitGain <= 0f || explicitGain > 8f)
            {
                error = $"--light-gain expects a positive value up to 8, got {explicitGain}.";
                return null;
            }

            tuning = tuning with { LightGainOverride = explicitGain };
        }

        string? paletteName = GetOption(args, "--liquid-palette", "");
        if (!string.IsNullOrWhiteSpace(paletteName))
        {
            MinimapLiquidPalette? palette = MinimapLiquidPalette.TryResolve(paletteName);
            if (palette is null)
            {
                error = $"--liquid-palette '{paletteName}' is not recognised. Available: {MinimapLiquidPalette.AvailableNames}.";
                return null;
            }

            tuning = tuning with { LiquidPalette = palette };
        }

        string? waterColor = GetOption(args, "--water-color", "");
        if (!string.IsNullOrWhiteSpace(waterColor))
        {
            string[] parts = waterColor.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length is not (3 or 4))
            {
                error = $"--water-color expects 'r,g,b' or 'r,g,b,opacity' in 0..1, got '{waterColor}'.";
                return null;
            }

            var channels = new float[parts.Length];
            for (int index = 0; index < parts.Length; index++)
            {
                if (!float.TryParse(parts[index], NumberStyles.Float, CultureInfo.InvariantCulture, out channels[index])
                    || channels[index] < 0f
                    || channels[index] > 1f)
                {
                    error = $"--water-color expects values in 0..1, got '{waterColor}'.";
                    return null;
                }
            }

            tuning = tuning with
            {
                LiquidPalette = tuning.LiquidPalette.WithWaterColor(
                    channels[0],
                    channels[1],
                    channels[2],
                    parts.Length == 4 ? channels[3] : null)
            };
        }

        return tuning;
    }

    static string? GetOption(string[] args, string name, string shortName)
    {
        int idx = Array.IndexOf(args, name);
        if (idx >= 0 && idx + 1 < args.Length) return args[idx + 1];
        if (!string.IsNullOrEmpty(shortName))
        {
            idx = Array.IndexOf(args, shortName);
            if (idx >= 0 && idx + 1 < args.Length) return args[idx + 1];
        }
        return null;
    }

    static int? GetIntOption(string[] args, string name, string shortName)
    {
        string? val = GetOption(args, name, shortName);
        return int.TryParse(val, out int v) ? v : null;
    }

    static float? GetFloatOption(string[] args, string name, string shortName)
    {
        string? value = GetOption(args, name, shortName);
        return float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed)
            ? parsed
            : null;
    }

    static bool HasFlag(string[] args, string flag)
    {
        return args.Contains(flag, StringComparer.OrdinalIgnoreCase);
    }
}
