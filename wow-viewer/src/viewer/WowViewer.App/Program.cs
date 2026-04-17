using System.Text.Json;
using WowViewer.Core;
using WowViewer.Core.PM4;
using WowViewer.Core.Runtime;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Liquid;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Terrain;
using WowViewer.Core.Runtime.World.Wdl;

namespace WowViewer.App;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Contains("--help", StringComparer.OrdinalIgnoreCase) || args.Contains("-h", StringComparer.OrdinalIgnoreCase))
        {
            ShowUsage();
            return 0;
        }

        if (args.Length == 0)
        {
            using WowViewerDesktopApp app = new();
            app.Run();
            return 0;
        }

        string command = args[0].ToLowerInvariant();
        string[] tail = args.Skip(1).ToArray();

        try
        {
            switch (command)
            {
                case "viewer":
                case "app":
                    using (WowViewerDesktopApp app = new(ParseViewerSession(tail)))
                        app.Run();
                    return 0;

                case "m2-frame":
                    return RunM2Frame(tail);

                case "m2-gpu-frame":
                    return RunM2GpuFrame(tail);

                case "world-bootstrap":
                    return RunWorldBootstrap(tail);

                case "world-frame":
                    return RunWorldFrame(tail);

                default:
                    using (WowViewerDesktopApp app = new(ParseViewerSession(args)))
                        app.Run();
                    return 0;
            }
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static int RunM2Frame(string[] args)
    {
        M2PreviewLoadRequest request = ParseRequiredM2Request(args, defaultVisualSize: 256);
        M2PreviewLoadResult result = M2PreviewLoader.Load(request);
        M2RuntimeGoldenFrame frame = result.FrameResult.GoldenFrame;

        Console.WriteLine($"WowViewer.App m2-frame: model={frame.CanonicalModelPath} version=0x{frame.ModelVersion:X} sequence={frame.RequestedSequenceIndex}->{frame.ResolvedSequenceIndex} timeMs={frame.TimeMs} bones={frame.BoneCount} skinnedVertices={frame.SkinnedVertexCount} visiblePasses={frame.VisiblePassCount} batches={frame.Batches.Count} hash={frame.RuntimeHash}");
        foreach (M2RuntimeGoldenBatch batch in frame.Batches)
            Console.WriteLine($"WowViewer.App m2-frame batch[{batch.BatchIndex}]: family={batch.Family} handler={batch.Handler} direct={batch.Direct} entries={batch.EntryCount} effect={batch.EffectKey} vertices={batch.VertexCount} indices={batch.IndexCount}");

        M2RenderFrame renderFrame = result.FrameResult.RenderFrame;
        Console.WriteLine($"WowViewer.App m2-frame render-frame: commands={renderFrame.CommandCount} backendVertices={renderFrame.BackendVertexCount} backendIndices={renderFrame.BackendIndexCount} submittedVertices={renderFrame.SubmittedVertexCount} submittedIndices={renderFrame.SubmittedIndexCount} hash={renderFrame.FrameHash}");
        Console.WriteLine($"WowViewer.App m2-frame effects: particles={result.FrameResult.EffectRuntimeState.Particles.Count} visibleParticles={result.FrameResult.EffectRuntimeState.VisibleParticleEmitterCount} ribbons={result.FrameResult.EffectRuntimeState.Ribbons.Count} visibleRibbons={result.FrameResult.EffectRuntimeState.VisibleRibbonEmitterCount}");
        Console.WriteLine($"WowViewer.App m2-frame visual: size={result.FrameResult.VisualSnapshot.Width}x{result.FrameResult.VisualSnapshot.Height} litPixels={result.FrameResult.VisualSnapshot.LitPixelCount} hash={result.FrameResult.VisualSnapshot.VisualHash}");

        string? goldenOutput = GetOption(args, "--golden-output");
        string? renderFrameOutput = GetOption(args, "--render-frame-output");
        string? visualOutput = GetOption(args, "--visual-output");
        if (!string.IsNullOrWhiteSpace(goldenOutput))
            WriteJson(goldenOutput, frame);
        if (!string.IsNullOrWhiteSpace(renderFrameOutput))
            WriteJson(renderFrameOutput, renderFrame);
        if (!string.IsNullOrWhiteSpace(visualOutput))
            WriteBmp(visualOutput, result.FrameResult.VisualSnapshot);

        return 0;
    }

    private static int RunM2GpuFrame(string[] args)
    {
        M2PreviewLoadRequest request = ParseRequiredM2Request(args, defaultVisualSize: 512);
        string? output = GetOption(args, "--output", "-o");
        if (string.IsNullOrWhiteSpace(output))
            throw new ArgumentException("Provide --output <file.bmp> for m2-gpu-frame.");

        string outputPath = Path.GetFullPath(output);
        M2GpuPreviewCaptureRunner.Capture(request, outputPath);
        Console.WriteLine($"Wrote {outputPath}");
        return 0;
    }

    private static int RunWorldBootstrap(string[] args)
    {
        WowViewerWorldSessionOpenRequest request = ParseRequiredWorldRequest(args);
        WowViewerWorldSessionBootstrapResult result = WowViewerWorldSessionBootstrapper.Open(request);

        Console.WriteLine($"WowViewer.App world-bootstrap: root={result.ClientRoot} build={FormatOptionalValue(result.BuildLabel)} map={result.RequestedMapInput}->{result.ResolvedMapDirectory} source={(result.LoadedFromArchive ? "archive" : "loose")} lookupLoaded={result.UsedMapDirectoryLookup} resolvedViaDbc={result.ResolvedViaDbc} loadMs={result.LoadDuration.TotalMilliseconds:F1}");
        Console.WriteLine($"WowViewer.App world-bootstrap wdt: path={result.WdtSourcePath} kind={result.FileSummary.Kind} version={result.FileSummary.Version?.ToString() ?? "n/a"} chunks={result.FileSummary.ChunkCount}");
        Console.WriteLine($"WowViewer.App world-bootstrap semantics: wmoBased={result.WdtSummary.IsWmoBased} tiles={result.WdtSummary.TilesWithData}/{result.WdtSummary.TotalTiles} mainCellBytes={result.WdtSummary.MainCellSizeBytes} doodadNames={result.WdtSummary.DoodadNameCount} wmoNames={result.WdtSummary.WorldModelNameCount} doodadPlacements={result.WdtSummary.DoodadPlacementCount} wmoPlacements={result.WdtSummary.WorldModelPlacementCount}");
        if (result.WdtSummary.MainFlags is not null)
        {
            Console.WriteLine($"WowViewer.App world-bootstrap main-flags: any={result.WdtSummary.MainFlags.CellsWithAnyFlags} hasAdt={result.WdtSummary.MainFlags.CellsWithHasAdt} allWater={result.WdtSummary.MainFlags.CellsWithAllWater} loaded={result.WdtSummary.MainFlags.CellsWithLoaded} unknown={result.WdtSummary.MainFlags.CellsWithUnknownFlags} asyncIds={result.WdtSummary.MainFlags.CellsWithAsyncId} distinct={FormatWdtMainFlags(result.WdtSummary.MainFlags)}");
        }

        string sampleTiles = result.OccupiedTiles.Count == 0
            ? "none"
            : string.Join(", ", result.OccupiedTiles.Take(12).Select(static tile => $"({tile.TileX},{tile.TileY})"));
        Console.WriteLine($"WowViewer.App world-bootstrap occupied-tiles: count={result.OccupiedTiles.Count} sample={sampleTiles}");
        return 0;
    }

    private static int RunWorldFrame(string[] args)
    {
        string? terrainPreviewOutput = GetOption(args, "--terrain-preview-output");
        WowViewerWorldRuntimeFrameRequest request = ParseRequiredWorldFrameRequest(args);
        WowViewerWorldRuntimeFrameResult result = WowViewerWorldRuntimeBridge.Build(request);

        Console.WriteLine($"WowViewer.App world-frame: map={result.Session.RequestedMapInput}->{result.Session.ResolvedMapDirectory} tile=({result.SelectedTileX},{result.SelectedTileY}) placementSource={result.PlacementSourcePath} objectPhase={result.ObjectPhaseExecuted} totalMs={result.Stats.TotalCpuMs:F2}");
        Console.WriteLine($"WowViewer.App world-frame options: wmo={result.PassOptions.WmosVisible} mdx={result.PassOptions.DoodadsVisible} sky={result.PassOptions.SkyVisible} wdl={result.PassOptions.WdlVisible} terrain={result.PassOptions.TerrainVisible} liquid={result.PassOptions.LiquidVisible} overlay={result.PassOptions.OverlayVisible}");
        Console.WriteLine($"WowViewer.App world-frame terrain: source={result.TileStageSummary.SourcePath} wdlTiles={result.Stats.WdlVisibleTileCount}/{result.TileStageSummary.WdlVisibleTileCount} terrainChunks={result.Stats.TerrainChunksRendered}/{result.TileStageSummary.TerrainChunkCount} holes={result.TileStageSummary.TerrainHoleChunkCount} liquidChunks={result.Stats.Liquid.VisibleCount}/{result.TileStageSummary.LiquidChunkCount} liquidLayers={result.TileStageSummary.LiquidLayerCount} liquidVisibleTiles={result.Stats.Liquid.SubmittedCount}/{result.TileStageSummary.VisibleLiquidTileCount} hasWater={result.TileStageSummary.HasWater}");
        Console.WriteLine($"WowViewer.App world-frame wdl-service: found={result.WdlTileData.SourceFound} hasData={result.WdlTileData.HasData} source={result.WdlTileData.SourcePath} version={FormatOptionalUInt(result.WdlTileData.Version)} range={FormatHeightRange(result.WdlTileData)} center={FormatOptionalHeight(result.WdlTileData.CenterHeight)} corners={FormatWdlCorners(result.WdlTileData)} samples={result.WdlTileData.OuterHeightCount}+{result.WdlTileData.InnerHeightCount}");
        Console.WriteLine($"WowViewer.App world-frame terrain-service: sourceChunks={result.TerrainTileData.ChunkCount} holeChunks={result.TerrainTileData.HoleChunkCount} liquidFlagChunks={result.TerrainTileData.LiquidFlagChunkCount} areas={result.TerrainTileData.DistinctAreaIdCount} sample={FormatTerrainChunkSample(result.TerrainTileData)}");
        Console.WriteLine($"WowViewer.App world-frame terrain-visual: size={result.TerrainVisualSnapshot.Width}x{result.TerrainVisualSnapshot.Height} sampled={result.TerrainVisualSnapshot.SampledPixelCount} range={FormatTerrainHeightRange(result.TerrainTileData)} center={FormatTerrainCenter(result.TerrainTileData)} corners={FormatTerrainCorners(result.TerrainTileData)} hash={result.TerrainVisualSnapshot.VisualHash}");
        Console.WriteLine($"WowViewer.App world-frame liquid-service: sourceChunks={result.LiquidTileData.ActiveChunkCount} sourceLayers={result.LiquidTileData.LayerCount} sourceVisibleTiles={result.LiquidTileData.VisibleTileCount} types={FormatLiquidTypeCounts(result.LiquidTileData)} sample={FormatLiquidChunkSample(result.LiquidTileData)}");
        Console.WriteLine($"WowViewer.App world-frame placements: wmo={result.WmoInstances.Count} readyWmo={result.ReadyWmoCount} mdx={result.MdxInstances.Count} readyMdx={result.ReadyMdxCount} pending={result.PendingAssetKeys.Count}");
        Console.WriteLine($"WowViewer.App world-frame visibility: visibleWmo={result.Visibility.VisibleWmos.Count} culledWmo={result.CulledWmoCount} visibleMdx={result.Visibility.VisibleMdx.Count} culledMdx={result.CulledMdxCount} taxi={result.Visibility.VisibleTaxiMdxCount}");
        Console.WriteLine($"WowViewer.App world-frame passes: updatedMdx={result.Stats.MdxAnimation.SubmittedCount} wmoOpaque={result.Stats.WmoSubmission.SubmittedCount} mdxOpaque={result.Stats.MdxOpaqueSubmission.SubmittedCount} mdxTransparent={result.Stats.MdxTransparentSubmission.SubmittedCount} opaqueRoutes={result.PassFrame.OpaqueVisibleMdxRoutes.Count} transparentRoutes={result.PassFrame.TransparentVisibleMdxRoutes.Count}");
        Console.WriteLine($"WowViewer.App world-frame hint: {result.OptimizationHint}");
        if (result.PendingAssetKeys.Count > 0)
            Console.WriteLine($"WowViewer.App world-frame pending-sample: {string.Join(", ", result.PendingAssetKeys.Take(8))}");

        if (!string.IsNullOrWhiteSpace(terrainPreviewOutput))
            WriteBmp(terrainPreviewOutput, result.TerrainVisualSnapshot);

        return 0;
    }

    private static string FormatLiquidTypeCounts(WorldLiquidTileData liquidTileData)
    {
        if (liquidTileData.Chunks.Count == 0)
            return "none";

        return string.Join(", ",
            liquidTileData.Chunks
                .SelectMany(static chunk => chunk.Layers)
                .GroupBy(static layer => layer.BasicType)
                .OrderBy(static group => group.Key)
                .Select(static group => $"{group.Key}:{group.Count()}"));
    }

    private static string FormatLiquidChunkSample(WorldLiquidTileData liquidTileData)
    {
        if (liquidTileData.Chunks.Count == 0)
            return "none";

        return string.Join(", ", liquidTileData.Chunks.Take(4).Select(static chunk =>
            $"({chunk.ChunkX},{chunk.ChunkY}) layers={chunk.Layers.Count} visible={chunk.VisibleTileCount} type={chunk.Layers[0].BasicType}"));
    }

    private static string FormatTerrainChunkSample(WorldTerrainTileData terrainTileData)
    {
        if (terrainTileData.Chunks.Count == 0)
            return "none";

        return string.Join(", ", terrainTileData.Chunks.Take(4).Select(static chunk =>
            $"({chunk.IndexX},{chunk.IndexY}) area={chunk.AreaId} holes={chunk.HasHoles} liquidFlags={chunk.HasLiquidFlags}"));
    }

    private static string FormatTerrainHeightRange(WorldTerrainTileData terrainTileData)
    {
        WorldTerrainHeightmapData? heightmap = terrainTileData.Heightmap;
        if (heightmap is null)
            return "n/a";

        return $"{heightmap.MinHeight:F2}..{heightmap.MaxHeight:F2}";
    }

    private static string FormatTerrainCenter(WorldTerrainTileData terrainTileData)
    {
        WorldTerrainHeightmapData? heightmap = terrainTileData.Heightmap;
        return heightmap is null ? "n/a" : $"{heightmap.CenterHeight:F2}";
    }

    private static string FormatTerrainCorners(WorldTerrainTileData terrainTileData)
    {
        WorldTerrainHeightmapData? heightmap = terrainTileData.Heightmap;
        if (heightmap is null)
            return "n/a";

        return $"nw={heightmap.NorthWestHeight:F2} ne={heightmap.NorthEastHeight:F2} sw={heightmap.SouthWestHeight:F2} se={heightmap.SouthEastHeight:F2}";
    }

    private static string FormatHeightRange(WorldWdlTileData wdlTileData)
    {
        if (!wdlTileData.MinHeight.HasValue || !wdlTileData.MaxHeight.HasValue)
            return "n/a";

        return $"{wdlTileData.MinHeight.Value}..{wdlTileData.MaxHeight.Value}";
    }

    private static string FormatWdlCorners(WorldWdlTileData wdlTileData)
    {
        if (!wdlTileData.HasData)
            return "n/a";

        return $"nw={FormatOptionalHeight(wdlTileData.NorthWestHeight)} ne={FormatOptionalHeight(wdlTileData.NorthEastHeight)} sw={FormatOptionalHeight(wdlTileData.SouthWestHeight)} se={FormatOptionalHeight(wdlTileData.SouthEastHeight)}";
    }

    private static string FormatOptionalHeight(short? value)
    {
        return value.HasValue ? value.Value.ToString() : "n/a";
    }

    private static string FormatOptionalUInt(uint? value)
    {
        return value.HasValue ? value.Value.ToString() : "n/a";
    }

    private static WowViewerSession? ParseViewerSession(string[] args)
    {
        string? workspaceText = GetOption(args, "--workspace", "-w");
        string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
        string? archiveRoot = GetOption(args, "--archive-root", "-r");
        string? virtualPath = GetOption(args, "--virtual-path", "-v");
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapInput = GetOption(args, "--map", "-m");
        string? tileXText = GetOption(args, "--tile-x");
        string? tileYText = GetOption(args, "--tile-y");
        string? buildLabel = GetOption(args, "--build-label", "-b");
        string? profileIndexText = GetOption(args, "--profile-index", "-p");
        string? sequenceIndexText = GetOption(args, "--sequence-index", "-s");
        string? timeMsText = GetOption(args, "--time-ms", "-t");
        string? visualSizeText = GetOption(args, "--visual-size");
        WowViewerWorkspaceMode workspaceMode = ParseWorkspaceMode(workspaceText);
        bool hasExplicitWorkspace = !string.IsNullOrWhiteSpace(workspaceText);
        bool hasM2Bootstrap = !string.IsNullOrWhiteSpace(input) || !string.IsNullOrWhiteSpace(archiveRoot) || !string.IsNullOrWhiteSpace(virtualPath);
        bool hasWorldBootstrap = !string.IsNullOrWhiteSpace(clientRoot) || !string.IsNullOrWhiteSpace(mapInput);

        if (!hasExplicitWorkspace && !hasM2Bootstrap && !hasWorldBootstrap)
            return null;

        int.TryParse(profileIndexText, out int profileIndex);
        int.TryParse(sequenceIndexText, out int sequenceIndex);
        int.TryParse(timeMsText, out int timeMs);
        int.TryParse(visualSizeText, out int visualSize);
        if (visualSize < 16)
            visualSize = 384;

        WowViewerSession session = WowViewerSession.CreateDefault();
        session.WorkspaceMode = workspaceMode;

        if (workspaceMode == WowViewerWorkspaceMode.WorldSession)
        {
            session.World.ClientRoot = clientRoot ?? string.Empty;
            session.World.MapInput = mapInput ?? input ?? string.Empty;
            session.World.BuildLabel = buildLabel ?? string.Empty;
            if (int.TryParse(tileXText, out int tileX))
                session.World.TileX = tileX;
            if (int.TryParse(tileYText, out int tileY))
                session.World.TileY = tileY;
            session.World.ShowWmos = !HasFlag(args, "--hide-wmos");
            session.World.ShowDoodads = !HasFlag(args, "--hide-doodads");
            session.World.ShowSky = !HasFlag(args, "--hide-sky");
            session.World.ShowWdl = !HasFlag(args, "--hide-wdl");
            session.World.ShowTerrain = !HasFlag(args, "--hide-terrain");
            session.World.ShowLiquid = !HasFlag(args, "--hide-liquid");
            session.World.ShowOverlay = !HasFlag(args, "--hide-overlay");
        }
        else
        {
            session.Source.Kind = string.IsNullOrWhiteSpace(archiveRoot)
                ? WowViewerAssetSourceKind.LocalFile
                : WowViewerAssetSourceKind.ArchiveVirtualPath;
            session.Source.InputPath = string.IsNullOrWhiteSpace(archiveRoot) ? input ?? string.Empty : string.Empty;
            session.Source.ArchiveRoot = archiveRoot ?? string.Empty;
            session.Source.VirtualPath = string.IsNullOrWhiteSpace(archiveRoot) ? string.Empty : (virtualPath ?? input ?? string.Empty);
            session.Source.BuildLabel = buildLabel ?? string.Empty;
        }

        session.ProfileIndex = profileIndex;
        session.SequenceIndex = Math.Max(0, sequenceIndex);
        session.TimeMs = Math.Max(0, timeMs);
        session.VisualSize = visualSize;
        session.Normalize();
        return session;
    }

    private static WowViewerWorldSessionOpenRequest ParseRequiredWorldRequest(string[] args)
    {
        string? clientRoot = GetOption(args, "--client-root", "-c");
        string? mapInput = GetOption(args, "--map", "-m") ?? GetFirstPositionalArgument(args);
        string? buildLabel = GetOption(args, "--build-label", "-b");

        if (string.IsNullOrWhiteSpace(clientRoot))
            throw new ArgumentException("Provide --client-root <dir> for world-bootstrap.");

        if (string.IsNullOrWhiteSpace(mapInput))
            throw new ArgumentException("Provide --map <directory|id|name> for world-bootstrap.");

        return new WowViewerWorldSessionOpenRequest(clientRoot, mapInput, buildLabel ?? string.Empty);
    }

    private static WowViewerWorldRuntimeFrameRequest ParseRequiredWorldFrameRequest(string[] args)
    {
        WowViewerWorldSessionOpenRequest request = ParseRequiredWorldRequest(args);
        string? tileXText = GetOption(args, "--tile-x");
        string? tileYText = GetOption(args, "--tile-y");

        int tileX = -1;
        int tileY = -1;
        if (!string.IsNullOrWhiteSpace(tileXText) && (!int.TryParse(tileXText, out tileX) || tileX < 0 || tileX > 63))
            throw new ArgumentOutOfRangeException(nameof(tileXText), "--tile-x must be in the range 0..63.");

        if (!string.IsNullOrWhiteSpace(tileYText) && (!int.TryParse(tileYText, out tileY) || tileY < 0 || tileY > 63))
            throw new ArgumentOutOfRangeException(nameof(tileYText), "--tile-y must be in the range 0..63.");

        WorldFramePassOptions passOptions = new(
            objectsVisible: !HasFlag(args, "--hide-wmos") || !HasFlag(args, "--hide-doodads"),
            wmosVisible: !HasFlag(args, "--hide-wmos"),
            doodadsVisible: !HasFlag(args, "--hide-doodads"),
            skyVisible: !HasFlag(args, "--hide-sky"),
            wdlVisible: !HasFlag(args, "--hide-wdl"),
            terrainVisible: !HasFlag(args, "--hide-terrain"),
            liquidVisible: !HasFlag(args, "--hide-liquid"),
            overlayVisible: !HasFlag(args, "--hide-overlay"));

        return new WowViewerWorldRuntimeFrameRequest(request.ClientRoot, request.MapInput, request.BuildLabel, tileX, tileY, passOptions);
    }

    private static M2PreviewLoadRequest ParseRequiredM2Request(string[] args, int defaultVisualSize)
    {
        string? input = GetOption(args, "--input", "-i") ?? GetFirstPositionalArgument(args);
        string? archiveRoot = GetOption(args, "--archive-root", "-r");
        string? virtualPath = GetOption(args, "--virtual-path", "-v");
        string? buildLabel = GetOption(args, "--build-label", "-b");
        string? profileIndexText = GetOption(args, "--profile-index", "-p");
        string? sequenceIndexText = GetOption(args, "--sequence-index", "-s");
        string? timeMsText = GetOption(args, "--time-ms", "-t");
        string? visualSizeText = GetOption(args, "--visual-size");

        if (!string.IsNullOrWhiteSpace(archiveRoot) && string.IsNullOrWhiteSpace(virtualPath))
            virtualPath = input;

        if (string.IsNullOrWhiteSpace(input) && (string.IsNullOrWhiteSpace(archiveRoot) || string.IsNullOrWhiteSpace(virtualPath)))
            throw new ArgumentException("Provide --input <file.m2|file.mdx|file.mdl> or --archive-root <dir> with --virtual-path <path/to/file.m2|file.mdx|file.mdl>.");

        int profileIndex = 0;
        if (!string.IsNullOrWhiteSpace(profileIndexText)
            && (!int.TryParse(profileIndexText, out profileIndex) || profileIndex < 0 || profileIndex > 99))
        {
            throw new ArgumentOutOfRangeException(nameof(profileIndex), "--profile-index must be an integer in the range 0..99.");
        }

        if (string.IsNullOrWhiteSpace(sequenceIndexText) || !int.TryParse(sequenceIndexText, out int sequenceIndex) || sequenceIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(sequenceIndexText), "--sequence-index must be a non-negative integer.");

        int timeMs = 0;
        if (!string.IsNullOrWhiteSpace(timeMsText) && !int.TryParse(timeMsText, out timeMs))
            throw new ArgumentOutOfRangeException(nameof(timeMsText), "--time-ms must be an integer.");

        int visualSize = defaultVisualSize;
        if (!string.IsNullOrWhiteSpace(visualSizeText))
        {
            if (!int.TryParse(visualSizeText, out visualSize) || visualSize < 16)
                throw new ArgumentOutOfRangeException(nameof(visualSizeText), "--visual-size must be an integer greater than or equal to 16.");
        }

        return new M2PreviewLoadRequest
        {
            InputPath = string.IsNullOrWhiteSpace(archiveRoot) ? input : null,
            ArchiveRoot = archiveRoot,
            VirtualPath = string.IsNullOrWhiteSpace(archiveRoot) ? null : (virtualPath ?? input),
            BuildLabel = buildLabel,
            ProfileIndex = profileIndex,
            SequenceIndex = sequenceIndex,
            TimeMs = timeMs,
            VisualWidth = visualSize,
            VisualHeight = visualSize,
        };
    }

    private static void WriteJson<T>(string output, T payload)
    {
        string outputPath = Path.GetFullPath(output);
        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        File.WriteAllText(outputPath, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true, IncludeFields = true }));
        Console.WriteLine($"Wrote {outputPath}");
    }

    private static void WriteBmp(string output, M2SoftwareVisualSnapshot snapshot)
    {
        string outputPath = Path.GetFullPath(output);
        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        using FileStream stream = File.Create(outputPath);
        M2SoftwareVisualSnapshotBuilder.WriteBmp(stream, snapshot);
        Console.WriteLine($"Wrote {outputPath}");
    }

    private static void WriteBmp(string output, WorldTerrainVisualSnapshot snapshot)
    {
        string outputPath = Path.GetFullPath(output);
        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        using FileStream stream = File.Create(outputPath);
        WorldTerrainVisualSnapshotBuilder.WriteBmp(stream, snapshot);
        Console.WriteLine($"Wrote {outputPath}");
    }

    private static string? GetOption(string[] args, params string[] names)
    {
        for (int index = 0; index < args.Length; index++)
        {
            if (!names.Contains(args[index], StringComparer.OrdinalIgnoreCase))
                continue;

            if (index + 1 >= args.Length)
                return null;

            return args[index + 1];
        }

        return null;
    }

    private static string? GetFirstPositionalArgument(string[] args)
    {
        for (int index = 0; index < args.Length; index++)
        {
            string arg = args[index];
            if (!arg.StartsWith("-", StringComparison.Ordinal))
                return arg;

            index++;
        }

        return null;
    }

    private static bool HasFlag(string[] args, params string[] names)
    {
        return args.Any(arg => names.Contains(arg, StringComparer.OrdinalIgnoreCase));
    }

    private static void ShowUsage()
    {
        Console.WriteLine("WowViewer.App");
        Console.WriteLine($"Solution: {ProjectIdentity.SolutionName} {ProjectIdentity.PlannedVersion}");
        Console.WriteLine($"PM4 canonical owner: {Pm4Boundary.CanonicalOwner}");
        Console.WriteLine($"PM4 legacy reference: {Pm4Boundary.LegacyReference}");
        Console.WriteLine($"Runtime service areas: {RuntimeBoundaries.All.Length}");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  wowviewer-app");
        Console.WriteLine("  wowviewer-app viewer [--workspace m2|wmo|mdx|world] [--archive-root <game|data dir> --virtual-path <path/to/file> | --input <file> | --client-root <game dir> --map <directory|id|name>] [--build-label <label>] [--profile-index <n>] [--sequence-index <n>] [--time-ms <ms>] [--visual-size <px>] [--hide-wmos] [--hide-doodads] [--hide-sky] [--hide-wdl] [--hide-terrain] [--hide-liquid] [--hide-overlay]");
        Console.WriteLine("  wowviewer-app m2-frame --archive-root <game|data dir> --virtual-path <path/to/file.m2> --sequence-index <n> [--build-label <label>] [--time-ms <ms>] [--profile-index <n>] [--golden-output <json>] [--render-frame-output <json>] [--visual-output <bmp>]");
        Console.WriteLine("  wowviewer-app m2-frame --input <file.m2> --sequence-index <n> [--build-label <label>] [--time-ms <ms>] [--profile-index <n>] [--golden-output <json>] [--render-frame-output <json>] [--visual-output <bmp>]");
        Console.WriteLine("  wowviewer-app m2-gpu-frame --archive-root <game|data dir> --virtual-path <path/to/file.m2> --sequence-index <n> --output <file.bmp> [--build-label <label>] [--time-ms <ms>] [--profile-index <n>] [--visual-size <px>]");
        Console.WriteLine("  wowviewer-app m2-gpu-frame --input <file.m2> --sequence-index <n> --output <file.bmp> [--build-label <label>] [--time-ms <ms>] [--profile-index <n>] [--visual-size <px>]");
        Console.WriteLine("  wowviewer-app world-bootstrap --client-root <game dir> --map <directory|id|name> [--build-label <label>]");
        Console.WriteLine("  wowviewer-app world-frame --client-root <game dir> --map <directory|id|name> [--tile-x <0..63> --tile-y <0..63>] [--build-label <label>] [--hide-wmos] [--hide-doodads] [--hide-sky] [--hide-wdl] [--hide-terrain] [--hide-liquid] [--hide-overlay] [--terrain-preview-output <file.bmp>]");
    }

    private static WowViewerWorkspaceMode ParseWorkspaceMode(string? workspaceText)
    {
        return workspaceText?.Trim().ToLowerInvariant() switch
        {
            null or "" or "m2" => WowViewerWorkspaceMode.StandaloneM2,
            "wmo" => WowViewerWorkspaceMode.StandaloneWmo,
            "mdx" => WowViewerWorkspaceMode.StandaloneMdx,
            "world" => WowViewerWorkspaceMode.WorldSession,
            _ => throw new ArgumentException("--workspace must be one of: m2, wmo, mdx, world."),
        };
    }

    private static string FormatOptionalValue(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? "n/a" : value;
    }

    private static string FormatWdtMainFlags(WowViewer.Core.Maps.WdtMainFlagsSummary summary)
    {
        return summary.DistinctNonZeroValues.Count == 0
            ? "none"
            : string.Join(",", summary.DistinctNonZeroValues.Select(static value => $"0x{value.Value:x}:{value.TileCount}"));
    }
}
