using System.Numerics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Lit;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Renderer.Headless;
using WowViewer.Core.Renderer.Scene;
using WowViewer.Core.Renderer.Terrain;
using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.Tools.Capture;

internal static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "--help" or "-h")
        {
            Console.WriteLine("""
                WowViewer.Tool.Capture — headless terrain tile renderer

                Usage:
                  render    --client-root <dir> --tile-name <name> --output <path>
                            [--variant primary|no-objects|objects-only|no-liquids]
                            [--resolution <int>] [--game-time <0..1>]
                            [--lighting-source authored|lit]
                            [--lit-file <path> | --lit-virtual-path <archive-path>]
                            [--lit-client-root <staged-client-dir>]
                            [--lighting-metadata-output <path>]

                Examples:
                  render --client-root "I:\parp\parp-tools\output\tmp\wowarchive-clients\0_5_3_3368" --tile-name Azeroth_30_48 --output output.png
                """);
            return 0;
        }

        return args[0] switch
        {
            "render" => RunRender(args[1..]),
            var x => throw new InvalidOperationException($"Unknown command: {x}")
        };
    }

    private static int RunRender(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? tileName = GetOption(args, "--tile-name", "-t");
        string? outputPath = GetOption(args, "--output", "-o");
        int resolution = GetIntOption(args, "--resolution", "-r") ?? 512;
        float gameTime = GetFloatOption(args, "--game-time") ?? AuthoredTerrainDayNightProfile.DefaultGameTime;
        string? litFile = GetOption(args, "--lit-file");
        string? litVirtualPath = GetOption(args, "--lit-virtual-path");
        string? litClientRoot = GetOption(args, "--lit-client-root");
        string lightingSource = GetOption(args, "--lighting-source")
            ?? (!string.IsNullOrWhiteSpace(litFile) || !string.IsNullOrWhiteSpace(litVirtualPath) ? "lit" : "authored");
        lightingSource = lightingSource.Trim().ToLowerInvariant();
        string? metadataOutput = GetOption(args, "--lighting-metadata-output");
        // Occlusion-aware object masks come from the real renderer, not a rasterizer: render
        // 'objects-only' (objects composited alone) or diff 'primary' against 'no-objects'.
        string variantName = (GetOption(args, "--variant") ?? "primary").Trim().ToLowerInvariant();

        if (string.IsNullOrWhiteSpace(clientRoot) || string.IsNullOrWhiteSpace(tileName) || string.IsNullOrWhiteSpace(outputPath))
        {
            Console.Error.WriteLine("Error: --client-root, --tile-name, and --output are required.");
            return 1;
        }
        if (!float.IsFinite(gameTime) || gameTime < 0f || gameTime >= 1f)
        {
            Console.Error.WriteLine("Error: --game-time must be a normalized fraction in [0, 1).");
            return 1;
        }
        if (lightingSource is not ("authored" or "lit"))
        {
            Console.Error.WriteLine("Error: --lighting-source must be authored or lit.");
            return 1;
        }
        if (variantName is not ("primary" or "no-objects" or "objects-only" or "no-liquids"))
        {
            Console.Error.WriteLine("Error: --variant must be primary, no-objects, objects-only, or no-liquids.");
            return 1;
        }
        if (!string.IsNullOrWhiteSpace(litFile) && !string.IsNullOrWhiteSpace(litVirtualPath))
        {
            Console.Error.WriteLine("Error: choose only one of --lit-file or --lit-virtual-path.");
            return 1;
        }
        bool hasLitOptions = !string.IsNullOrWhiteSpace(litFile)
            || !string.IsNullOrWhiteSpace(litVirtualPath)
            || !string.IsNullOrWhiteSpace(litClientRoot);
        if (lightingSource == "authored" && hasLitOptions)
        {
            Console.Error.WriteLine("Error: LIT source options require --lighting-source lit.");
            return 1;
        }
        if (!string.IsNullOrWhiteSpace(litFile) && !string.IsNullOrWhiteSpace(litClientRoot))
        {
            Console.Error.WriteLine("Error: --lit-file is a complete local source; do not combine it with --lit-client-root.");
            return 1;
        }
        if (!TryParseTileName(tileName, out string mapName, out int tileX, out int tileY))
        {
            Console.Error.WriteLine("Error: --tile-name must end in _<tile-x>_<tile-y>.");
            return 1;
        }

        Console.WriteLine($"Loading tile: {tileName}");
        Console.WriteLine($"Client root: {clientRoot}");
        Console.WriteLine($"Output: {outputPath}");
        Console.WriteLine($"Resolution: {resolution}");
        Console.WriteLine($"Game time: {gameTime:F6} (normalized day fraction)");
        Console.WriteLine($"Lighting source: {lightingSource}");

        string adtPath = $"World\\Maps\\{mapName}\\{tileName}.adt";

        IArchiveCatalogFactory factory = new NativeMpqServiceFactory();
        using IArchiveCatalog terrainCatalog = factory.Create();
        terrainCatalog.LoadArchives([clientRoot]);
        using IArchiveCatalog? isolatedLitCatalog = lightingSource == "lit"
            && string.IsNullOrWhiteSpace(litFile)
            && !string.IsNullOrWhiteSpace(litClientRoot)
                ? CreateArchiveCatalog(factory, litClientRoot)
                : null;
        IArchiveCatalog lightingCatalog = isolatedLitCatalog ?? terrainCatalog;
        string lightingArchiveRoot = Path.GetFullPath(litClientRoot ?? clientRoot);

        if (!terrainCatalog.FileExists(adtPath))
        {
            Console.Error.WriteLine($"ADT file not found: {adtPath}");
            return 2;
        }

        byte[]? adtData = terrainCatalog.ReadFile(adtPath);
        if (adtData == null)
        {
            Console.Error.WriteLine($"Failed to read ADT: {adtPath}");
            return 2;
        }

        var fileSummary = MapFileSummaryReader.Read(new MemoryStream(adtData), adtPath);
        var tileData = WorldTerrainTileBuilder.Read(new MemoryStream(adtData), fileSummary);

        if (tileData.ChunkCount == 0)
        {
            Console.Error.WriteLine("Tile has no chunks");
            return 2;
        }

        Console.WriteLine($"Chunks: {tileData.ChunkCount}, Layers: {tileData.TotalTextureLayerCount}");

        using var context = new HeadlessContext(resolution, resolution);
        using var surface = new RenderSurface(context.GL, resolution, resolution);
        var camera = new SceneCamera();
        using var renderer = new SceneRenderer(context.GL);

        float minHeight = tileData.Heightmap?.MinHeight ?? 0f;
        float maxHeight = tileData.Heightmap?.MaxHeight ?? 0f;
        TerrainCaptureCamera captureCamera = TerrainCaptureView.CreateTopDown(tileX, tileY, minHeight, maxHeight);
        camera.SetViewProjection(captureCamera.View, captureCamera.Projection, captureCamera.Position);

        LightingSelection lightingSelection;
        try
        {
            lightingSelection = ResolveLightingSelection(
                lightingSource,
                gameTime,
                mapName,
                litFile,
                litVirtualPath,
                lightingCatalog,
                lightingArchiveRoot);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine($"Lighting source failed closed: {ex.Message}");
            return 2;
        }

        TerrainLightingSample lighting = lightingSelection.Lighting;
        renderer.Terrain.LightDirection = lighting.LightDirection;
        renderer.Terrain.DirectionalLightColor = lighting.DirectionalColor * lighting.DirectionalIntensity;
        renderer.Terrain.AmbientLightColor = lighting.AmbientColor * lighting.AmbientIntensity;
        renderer.Terrain.ShadowStrength = lighting.McshShadowStrength;
        renderer.Terrain.ShowShadowMap = true;
        renderer.Terrain.FogColor = lighting.FogColor;

        surface.Clear(lighting.FogColor.X, lighting.FogColor.Y, lighting.FogColor.Z, 1f);

        RenderVariant renderVariant = variantName switch
        {
            "no-objects" => RenderVariant.NoObjects,
            "objects-only" => RenderVariant.ObjectsOnly,
            "no-liquids" => RenderVariant.NoLiquids,
            _ => RenderVariant.Primary,
        };
        renderer.RenderTile(camera, tileData, renderVariant);

        var capture = new FrameCapture(context, resolution, resolution);
        byte[] rgba = capture.CaptureRgba();
        PngWriter.WritePng(outputPath, rgba, resolution, resolution);

        string metadataPath = metadataOutput ?? $"{outputPath}.lighting.json";
        WriteLightingMetadata(
            metadataPath,
            clientRoot,
            adtPath,
            adtData,
            outputPath,
            tileName,
            mapName,
            tileX,
            tileY,
            resolution,
            minHeight,
            maxHeight,
            captureCamera,
            lightingSelection,
            renderer.Terrain.FogStart,
            renderer.Terrain.FogEnd);

        Console.WriteLine($"Saved: {outputPath}");
        Console.WriteLine($"Lighting evidence: {metadataPath}");
        return 0;
    }

    private static IArchiveCatalog CreateArchiveCatalog(
        IArchiveCatalogFactory factory,
        string archiveRoot)
    {
        IArchiveCatalog catalog = factory.Create();
        try
        {
            catalog.LoadArchives([archiveRoot]);
            return catalog;
        }
        catch
        {
            catalog.Dispose();
            throw;
        }
    }

    private static bool TryParseTileName(string tileName, out string mapName, out int tileX, out int tileY)
    {
        mapName = string.Empty;
        tileX = 0;
        tileY = 0;
        int ySeparator = tileName.LastIndexOf('_');
        int xSeparator = ySeparator > 0 ? tileName.LastIndexOf('_', ySeparator - 1) : -1;
        if (xSeparator <= 0 || ySeparator <= xSeparator + 1)
            return false;
        if (!int.TryParse(tileName.AsSpan(xSeparator + 1, ySeparator - xSeparator - 1), NumberStyles.Integer, CultureInfo.InvariantCulture, out tileX) ||
            !int.TryParse(tileName.AsSpan(ySeparator + 1), NumberStyles.Integer, CultureInfo.InvariantCulture, out tileY))
            return false;
        mapName = tileName[..xSeparator];
        return mapName.Length > 0;
    }

    private static void WriteLightingMetadata(
        string metadataPath,
        string clientRoot,
        string adtPath,
        byte[] adtData,
        string outputPath,
        string tileName,
        string mapName,
        int tileX,
        int tileY,
        int resolution,
        float minHeight,
        float maxHeight,
        TerrainCaptureCamera camera,
        LightingSelection lightingSelection,
        float rendererFogStart,
        float rendererFogEnd)
    {
        TerrainLightingSample lighting = lightingSelection.Lighting;
        static float[] Vector(Vector3 value) => [value.X, value.Y, value.Z];
        var metadata = new
        {
            schema = "wowviewer-terrain-capture-lighting-v2",
            renderer_contract = lighting.LightingModel,
            lighting_source_kind = lightingSelection.SourceKind,
            lighting_profile_revision = lighting.ProfileRevision,
            lighting_evidence_state = lighting.EvidenceState,
            source_precedence_note = "Explicit source selection; LIT, Light-star DBC, and authored values are never implicitly mixed",
            tile = new { name = tileName, map = mapName, x = tileX, y = tileY },
            input = new
            {
                client_root = Path.GetFullPath(clientRoot),
                adt_path = adtPath.Replace('\\', '/'),
                adt_sha256 = Convert.ToHexString(SHA256.HashData(adtData)).ToLowerInvariant(),
                rights_class = "provenance_unverified_capture_source",
                legal_status = "not_a_legal_determination",
            },
            output = new
            {
                png_path = Path.GetFullPath(outputPath),
                png_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(outputPath))).ToLowerInvariant(),
                width = resolution,
                height = resolution,
                material_source = "generated_checker_v1",
            },
            camera = new
            {
                mode = "top_down_orthographic_one_adt_tile_v1",
                position = Vector(camera.Position),
                far_plane = camera.FarPlane,
                terrain_min_height = minHeight,
                terrain_max_height = maxHeight,
                image_axis_contract = "right=adt_tile_x_positive;down=adt_tile_y_positive",
            },
            lighting_source = new
            {
                identifier = lightingSelection.SourceIdentifier,
                root = lightingSelection.SourceRoot,
                sha256 = lightingSelection.SourceSha256,
                lit_version = lightingSelection.LitVersion,
                lit_light_index = lightingSelection.LitLightIndex,
                lit_light_name = lightingSelection.LitLightName,
                lit_group_index = lightingSelection.LitGroupIndex,
                lit_time = lightingSelection.LitTime,
                contributing_track_ids = lightingSelection.ContributingTrackIds,
                direction_evidence_state = lightingSelection.DirectionEvidenceState,
                mcsh_evidence_state = lightingSelection.McshEvidenceState,
                fog_distance_evidence_state = "renderer_defaults_not_recovered_from_selected_profile",
            },
            lighting = new
            {
                game_time = lighting.GameTime,
                light_direction = Vector(lighting.LightDirection),
                directional_color = Vector(lighting.DirectionalColor),
                directional_intensity = lighting.DirectionalIntensity,
                ambient_color = Vector(lighting.AmbientColor),
                ambient_intensity = lighting.AmbientIntensity,
                fog_color = Vector(lighting.FogColor),
                fog_start = rendererFogStart,
                fog_end = rendererFogEnd,
                mcsh_shadow_strength = lighting.McshShadowStrength,
                mcsh_policy = "preserve_present_64x64_per_chunk;absent_is_unshadowed",
            },
        };
        string? directory = Path.GetDirectoryName(Path.GetFullPath(metadataPath));
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        File.WriteAllText(metadataPath, JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static LightingSelection ResolveLightingSelection(
        string sourceKind,
        float gameTime,
        string mapName,
        string? litFile,
        string? litVirtualPath,
        IArchiveCatalog catalog,
        string lightingArchiveRoot)
    {
        if (sourceKind == "authored")
        {
            TerrainLightingSample authored = AuthoredTerrainDayNightProfile.Evaluate(gameTime);
            return new LightingSelection(
                authored,
                "authored_fallback",
                authored.ProfileRevision,
                null,
                null,
                null,
                null,
                null,
                null,
                null,
                [],
                "authored_solar_direction",
                "authored_mcsh_strength_not_client_exact");
        }

        byte[] litBytes;
        string sourceIdentifier;
        string? sourceRoot;
        if (!string.IsNullOrWhiteSpace(litFile))
        {
            sourceIdentifier = Path.GetFullPath(litFile);
            sourceRoot = null;
            litBytes = File.ReadAllBytes(sourceIdentifier);
        }
        else
        {
            string selectedVirtualPath = litVirtualPath
                ?? ResolveDefaultLitVirtualPath(catalog, mapName)
                ?? throw new FileNotFoundException($"No LIT file found for map '{mapName}'. Pass --lit-file or --lit-virtual-path explicitly.");
            sourceIdentifier = selectedVirtualPath.Replace('\\', '/');
            sourceRoot = lightingArchiveRoot;
            litBytes = catalog.ReadFile(selectedVirtualPath)
                ?? throw new FileNotFoundException($"Failed to read LIT archive path '{selectedVirtualPath}'.");
        }

        LitFileProfile profile = LitProfileReader.Read(new MemoryStream(litBytes, writable: false), sourceIdentifier);
        LitTerrainLightingEvaluation evaluation = LitTerrainDayNightProfile.EvaluateGlobalClear(profile, gameTime);
        return new LightingSelection(
            evaluation.Lighting,
            "client_lit_global_clear",
            sourceIdentifier,
            sourceRoot,
            Sha256(litBytes),
            $"0x{evaluation.LitVersion:X8}",
            evaluation.LightIndex,
            evaluation.LightName,
            evaluation.GroupIndex,
            evaluation.LitTimeOfDay,
            evaluation.ContributingTrackIds,
            evaluation.DirectionEvidenceState,
            evaluation.McshEvidenceState);
    }

    private static string? ResolveDefaultLitVirtualPath(IArchiveCatalog catalog, string mapName)
    {
        string[] candidates =
        [
            $"World\\{mapName}\\lights.lit",
            $"World\\Maps\\{mapName}\\lights.lit",
            $"World\\{mapName}\\areatest.lit",
            $"World\\Maps\\{mapName}\\areatest.lit",
            $"World\\{mapName}\\light.lit",
            $"World\\Maps\\{mapName}\\light.lit",
        ];
        return candidates.FirstOrDefault(catalog.FileExists);
    }

    private static string Sha256(byte[] data)
    {
        return Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();
    }

    private sealed record LightingSelection(
        TerrainLightingSample Lighting,
        string SourceKind,
        string SourceIdentifier,
        string? SourceRoot,
        string? SourceSha256,
        string? LitVersion,
        int? LitLightIndex,
        string? LitLightName,
        int? LitGroupIndex,
        float? LitTime,
        IReadOnlyList<int> ContributingTrackIds,
        string DirectionEvidenceState,
        string McshEvidenceState);

    private static string? GetOption(string[] args, params string[] names)
    {
        for (int i = 0; i < args.Length; i++)
        {
            if (names.Contains(args[i], StringComparer.OrdinalIgnoreCase) && i + 1 < args.Length)
                return args[i + 1];
        }
        return null;
    }

    private static int? GetIntOption(string[] args, params string[] names)
    {
        string? value = GetOption(args, names);
        return int.TryParse(value, out int parsed) ? parsed : null;
    }

    private static float? GetFloatOption(string[] args, params string[] names)
    {
        string? value = GetOption(args, names);
        return float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed)
            ? parsed
            : null;
    }
}
